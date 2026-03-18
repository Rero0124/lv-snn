//! Lock-free 이벤트 드리븐 발화 엔진
//!
//! 틱 기반 순차 처리 대신, 신호가 큐를 통해 자연스럽게 퍼지는 구조.
//! 각 뉴런은 독립적으로 신호를 수신하고, 시냅스별 확률적 발화를 수행.
//! CPU 코어를 최대한 활용하며, lock-free 큐로 오버헤드 최소화.

use crate::neuron::{NeuronId, OutgoingSynapse, SIGMOID_TEMPERATURE, TOP_K_FIRES};
use crate::region::RegionType;
use crate::synapse::SynapseId;

use crossbeam::deque::{Injector, Stealer, Worker};
use crossbeam::queue::SegQueue;
use rand::RngExt;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// ── 상수 ──

const MAX_TOTAL_SIGNALS: usize = 50_000;  // 최대 처리 신호 수 (무한 루프 방지)
const MAX_HOPS: u32 = 50;                  // 최대 전파 깊이
const DECAY_PER_HOP: f64 = 0.5;           // 홉당 감쇠율

/// 발화 엔진에 로드할 뉴런 데이터 (read-only)
pub struct EngineNeuron {
    pub outgoing: Vec<EngineSynapse>,
    pub threshold: f64,
    pub region: RegionType,
}

/// 시냅스 데이터 (read-only)
pub struct EngineSynapse {
    pub id: SynapseId,
    pub weight: f64,
    pub modifier: f64,
    pub target_idx: usize,
    pub token: Option<String>,
}

/// 큐에 넣는 신호
struct Signal {
    target_idx: usize,
    value: f64,
    synapse_id: SynapseId,
    source_idx: usize,
    token: Option<String>,
    hop: u32,
}

/// 발화된 시냅스 기록
pub struct FiredSynapse {
    pub synapse_id: SynapseId,
    pub source_idx: usize,
    pub target_idx: usize,
    pub forward: f64,
    pub token: Option<String>,
    pub hop: u32,
}

/// 병렬 발화 결과
pub struct FireEngineResult {
    pub fired_synapses: Vec<FiredSynapse>,
    pub output_tokens: Vec<(String, SynapseId, u32)>,  // (token, synapse_id, hop)
    pub total_signals: usize,
    pub max_hop_reached: u32,
}

/// Lock-free 발화 엔진
pub struct FireEngine {
    neurons: Vec<EngineNeuron>,
    // 뉴런별 활성화값 (AtomicU64로 f64 bit-cast)
    activations: Vec<AtomicU64>,
    // 뉴런별 처리 플래그 (한 번에 하나의 스레드만 처리)
    processing: Vec<AtomicBool>,
    // Output 구역 뉴런 인덱스
    output_indices: HashSet<usize>,
    // ID ↔ index 매핑
    id_to_idx: HashMap<NeuronId, usize>,
    idx_to_id: Vec<NeuronId>,
}

impl FireEngine {
    /// Network에서 데이터를 추출해서 엔진 생성
    pub fn new(
        neurons_data: Vec<(NeuronId, EngineNeuron)>,
        output_neuron_ids: &[NeuronId],
    ) -> Self {
        let mut id_to_idx = HashMap::new();
        let mut idx_to_id = Vec::new();
        let mut neurons = Vec::new();
        let mut activations = Vec::new();
        let mut processing = Vec::new();

        for (i, (nid, neuron)) in neurons_data.into_iter().enumerate() {
            id_to_idx.insert(nid.clone(), i);
            idx_to_id.push(nid);
            neurons.push(neuron);
            activations.push(AtomicU64::new(0f64.to_bits()));
            processing.push(AtomicBool::new(false));
        }

        let output_indices: HashSet<usize> = output_neuron_ids
            .iter()
            .filter_map(|nid| id_to_idx.get(nid).copied())
            .collect();

        Self {
            neurons,
            activations,
            processing,
            output_indices,
            id_to_idx,
            idx_to_id,
        }
    }

    /// 뉴런 ID → index
    pub fn neuron_idx(&self, id: &str) -> Option<usize> {
        self.id_to_idx.get(id).copied()
    }

    /// index → 뉴런 ID
    pub fn neuron_id(&self, idx: usize) -> &NeuronId {
        &self.idx_to_id[idx]
    }

    /// 활성화값 atomic read
    fn load_activation(&self, idx: usize) -> f64 {
        f64::from_bits(self.activations[idx].load(Ordering::Relaxed))
    }

    /// 활성화값 atomic add (CAS loop)
    fn add_activation(&self, idx: usize, value: f64) -> f64 {
        loop {
            let old_bits = self.activations[idx].load(Ordering::Relaxed);
            let old_val = f64::from_bits(old_bits);
            let new_val = (old_val + value).min(1.0);  // MAX_ACTIVATION
            let new_bits = new_val.to_bits();
            if self.activations[idx]
                .compare_exchange_weak(old_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return new_val;
            }
        }
    }

    /// 활성화값 초기화
    fn reset_activations(&self) {
        for a in &self.activations {
            a.store(0f64.to_bits(), Ordering::Relaxed);
        }
    }

    /// 병렬 발화 실행
    pub fn fire(
        &self,
        input_signals: Vec<(usize, f64)>,  // (neuron_idx, activation_value)
        emitted_tokens: &HashSet<String>,
    ) -> FireEngineResult {
        self.reset_activations();

        // 결과 수집용 lock-free 큐
        let fired_queue: SegQueue<FiredSynapse> = SegQueue::new();
        let output_queue: SegQueue<(String, SynapseId, u32)> = SegQueue::new();
        let emitted_set: SegQueue<String> = SegQueue::new();

        // 글로벌 작업 큐
        let injector: Injector<Signal> = Injector::new();
        let total_signals = AtomicUsize::new(0);
        let pending = AtomicUsize::new(0);

        // 입력 신호 주입
        for (idx, value) in &input_signals {
            self.add_activation(*idx, *value);
        }

        // 초기 발화 대상: 활성화된 입력 뉴런
        let mut initial_indices: HashSet<usize> = HashSet::new();
        for (idx, _) in &input_signals {
            initial_indices.insert(*idx);
        }
        for idx in &initial_indices {
            injector.push(Signal {
                target_idx: *idx,
                value: 0.0,  // 이미 activation에 반영됨
                synapse_id: String::new(),
                source_idx: *idx,
                token: None,
                hop: 0,
            });
            pending.fetch_add(1, Ordering::Relaxed);
        }

        // 워커 스레드 수 = CPU 코어 수
        let num_workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // 워커별 로컬 큐 + stealer 생성
        let workers: Vec<Worker<Signal>> = (0..num_workers)
            .map(|_| Worker::new_fifo())
            .collect();
        let stealers: Vec<Stealer<Signal>> = workers.iter().map(|w| w.stealer()).collect();

        // emitted_tokens를 Arc로 공유
        let emitted_tokens = Arc::new(emitted_tokens.clone());

        thread::scope(|s| {
            for worker in workers {
                let injector = &injector;
                let stealers = &stealers;
                let fired_queue = &fired_queue;
                let output_queue = &output_queue;
                let total_signals = &total_signals;
                let pending = &pending;
                let emitted_tokens = &emitted_tokens;

                s.spawn(move || {
                    let mut rng = rand::rng();
                    let mut idle_spins = 0u32;

                    loop {
                        // 신호 가져오기: 로컬 → 글로벌 → 훔치기
                        let signal = worker.pop().or_else(|| {
                            loop {
                                match injector.steal_batch_and_pop(&worker) {
                                    crossbeam::deque::Steal::Success(s) => return Some(s),
                                    crossbeam::deque::Steal::Empty => return None,
                                    crossbeam::deque::Steal::Retry => continue,
                                }
                            }
                        }).or_else(|| {
                            for stealer in stealers {
                                loop {
                                    match stealer.steal() {
                                        crossbeam::deque::Steal::Success(s) => return Some(s),
                                        crossbeam::deque::Steal::Empty => break,
                                        crossbeam::deque::Steal::Retry => continue,
                                    }
                                }
                            }
                            None
                        });

                        let signal = match signal {
                            Some(s) => {
                                idle_spins = 0;
                                s
                            }
                            None => {
                                // 모든 큐가 비어있고 pending이 0이면 종료
                                if pending.load(Ordering::Relaxed) == 0 {
                                    idle_spins += 1;
                                    if idle_spins > 100 {
                                        break;
                                    }
                                    std::hint::spin_loop();
                                    continue;
                                }
                                std::hint::spin_loop();
                                continue;
                            }
                        };

                        let total = total_signals.fetch_add(1, Ordering::Relaxed);
                        if total > MAX_TOTAL_SIGNALS || signal.hop >= MAX_HOPS {
                            pending.fetch_sub(1, Ordering::Relaxed);
                            continue;
                        }

                        let idx = signal.target_idx;

                        // 신호값을 뉴런에 더하기 (초기 신호는 이미 반영됨)
                        if signal.value > 0.0 {
                            self.add_activation(idx, signal.value);
                        }

                        // 이 뉴런 처리 시도 (lock-free claim)
                        // 이미 다른 스레드가 처리 중이면 skip
                        // (나중에 들어온 신호분은 activation에 이미 반영되어 있으므로
                        //  현재 처리 중인 스레드가 자연스럽게 반영)
                        if !self.processing[idx].compare_exchange(
                            false, true, Ordering::Acquire, Ordering::Relaxed
                        ).is_ok() {
                            pending.fetch_sub(1, Ordering::Relaxed);
                            continue;
                        }

                        let activation = self.load_activation(idx);
                        if activation <= 0.0 {
                            self.processing[idx].store(false, Ordering::Release);
                            pending.fetch_sub(1, Ordering::Relaxed);
                            continue;
                        }

                        let neuron = &self.neurons[idx];
                        let mut fires: Vec<(usize, &EngineSynapse, f64)> = Vec::new();

                        // 시냅스별 확률적 발화
                        for os in &neuron.outgoing {
                            let mut forward = activation * os.weight + os.modifier;
                            if forward <= 0.0 {
                                continue;
                            }

                            let fire_prob = stepped_fire_prob(forward);
                            if rng.random::<f64>() > fire_prob {
                                continue;
                            }

                            // 이미 출력된 토큰 감쇠
                            if let Some(ref tok) = os.token {
                                if emitted_tokens.contains(tok) {
                                    forward *= 0.5;
                                }
                            }

                            fires.push((os.target_idx, os, forward));
                        }

                        // TOP_K 제한
                        if fires.len() > TOP_K_FIRES {
                            fires.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                            fires.truncate(TOP_K_FIRES);
                        }

                        // 발화 결과를 큐에 넣기
                        for (target_idx, os, forward) in fires {
                            let decayed = forward * DECAY_PER_HOP;

                            // 발화 기록
                            fired_queue.push(FiredSynapse {
                                synapse_id: os.id.clone(),
                                source_idx: idx,
                                target_idx,
                                forward,
                                token: os.token.clone(),
                                hop: signal.hop,
                            });

                            // Output 뉴런으로 향하는 토큰 수집
                            if let Some(ref tok) = os.token {
                                if self.output_indices.contains(&target_idx) {
                                    output_queue.push((tok.clone(), os.id.clone(), signal.hop));
                                }
                            }

                            // 다음 신호 전파
                            if decayed > 0.01 {
                                pending.fetch_add(1, Ordering::Relaxed);
                                injector.push(Signal {
                                    target_idx,
                                    value: decayed,
                                    synapse_id: os.id.clone(),
                                    source_idx: idx,
                                    token: os.token.clone(),
                                    hop: signal.hop + 1,
                                });
                            }
                        }

                        // 처리 완료, 다른 스레드가 처리 가능하게
                        self.processing[idx].store(false, Ordering::Release);
                        pending.fetch_sub(1, Ordering::Relaxed);
                    }
                });
            }
        });

        // 결과 수집
        let mut fired_synapses = Vec::new();
        while let Some(f) = fired_queue.pop() {
            fired_synapses.push(f);
        }

        let mut output_tokens = Vec::new();
        let mut seen_tokens = HashSet::new();
        while let Some((tok, sid, hop)) = output_queue.pop() {
            if seen_tokens.insert(tok.clone()) {
                output_tokens.push((tok, sid, hop));
            }
        }

        let max_hop = fired_synapses.iter().map(|f| f.hop).max().unwrap_or(0);

        FireEngineResult {
            total_signals: total_signals.load(Ordering::Relaxed),
            max_hop_reached: max_hop,
            fired_synapses,
            output_tokens,
        }
    }
}

/// 계단식 비선형 발화 확률 (neuron.rs와 동일)
fn stepped_fire_prob(forward: f64) -> f64 {
    let steps: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.2, 0.001, 0.03),
        (0.2, 0.4, 0.03, 0.10),
        (0.4, 0.65, 0.10, 0.40),
        (0.65, 0.75, 0.40, 0.65),
        (0.75, 1.0, 0.65, 0.98),
    ];

    if forward >= 1.0 {
        return 0.96;
    }

    for &(lo, hi, base, next) in steps {
        if forward < hi {
            let t = (forward - lo) / (hi - lo);
            let s = 1.0 / (1.0 + (-(t * 6.0 - 3.0)).exp());
            return base + (next - base) * s;
        }
    }
    0.96
}
