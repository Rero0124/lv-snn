use crate::neuron::{Neuron, NeuronId};
use crate::region::RegionType;
use crate::tokenizer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── 상수 ──

const MAX_TICKS: u64 = 30;
const INITIAL_WEIGHT: f64 = 0.4;
const MIN_WEIGHT: f64 = 0.01;
const PRUNE_INTERVAL: u64 = 50;
const EMOTION_COUNT: usize = 2000;
const REASON_COUNT: usize = 2000;

// ── 발화 기록 ──

pub struct DeliveryTrace {
    pub from: NeuronId,
    pub to: NeuronId,
    pub from_potential: f64,
    pub to_before: f64,
    pub weight: f64,
    pub delivered: f64,
    pub tick: u64,
}

pub struct FireRecord {
    pub id: u64,
    pub output: String,
    pub output_tokens: Vec<(String, f64)>,
    pub spiked_neurons: Vec<(NeuronId, u64)>,
    pub traces: Vec<DeliveryTrace>,
}

// ── 네트워크 ──

pub struct Network {
    /// 모든 뉴런 (Vec index = NeuronId)
    neurons: Vec<Neuron>,
    /// 토큰 → seq
    vocab: HashMap<String, u32>,
    /// seq → 토큰
    reverse_vocab: Vec<String>,
    /// seq → 입력 뉴런 인덱스
    input_idx: Vec<usize>,
    /// seq → 출력 뉴런 인덱스
    output_idx: Vec<usize>,
    /// 감정 뉴런 범위 [start, end)
    emotion_range: (usize, usize),
    /// 이성 뉴런 범위 [start, end)
    reason_range: (usize, usize),

    fire_records: Vec<FireRecord>,
    next_fire_id: u64,
    pub debug: bool,
    pub threshold: f64,
    sprout_cooldown: HashMap<NeuronId, u64>,
    global_tick: u64,
}

impl Network {
    pub fn new() -> Self {
        let mut neurons: Vec<Neuron> = Vec::new();

        // 감정 뉴런 (20% 억제성)
        let emotion_start = neurons.len();
        let inhibitory_boundary = (EMOTION_COUNT as f64 * 0.8) as usize;
        let cols = 50usize;
        for i in 0..EMOTION_COUNT {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32;
            let id = neurons.len() as NeuronId;
            if i >= inhibitory_boundary {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let emotion_end = neurons.len();

        // 이성 뉴런 (20% 억제성)
        let reason_start = neurons.len();
        let inhibitory_boundary = (REASON_COUNT as f64 * 0.8) as usize;
        for i in 0..REASON_COUNT {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32 + 100.0;
            let id = neurons.len() as NeuronId;
            if i >= inhibitory_boundary {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let reason_end = neurons.len();

        let mut net = Network {
            neurons,
            vocab: HashMap::new(),
            reverse_vocab: Vec::new(),
            input_idx: Vec::new(),
            output_idx: Vec::new(),
            emotion_range: (emotion_start, emotion_end),
            reason_range: (reason_start, reason_end),
            fire_records: Vec::new(),
            next_fire_id: 1,
            debug: false,
            threshold: 0.5,
            sprout_cooldown: HashMap::new(),
            global_tick: 0,
        };

        // 초기 자모 등록
        let initial_tokens = tokenizer::all_tokens();
        for tok in &initial_tokens {
            net.register_token(tok);
        }

        // 초기 랜덤 시냅스
        net.seed_random_synapses(30);

        let total_synapses: usize = net.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [초기화] 어휘 {}개, 뉴런 {}개 (감정 {}, 이성 {}, 입출력 {}), 시냅스 {}개",
            net.vocab.len(), net.neurons.len(),
            EMOTION_COUNT, REASON_COUNT, net.input_idx.len() * 2, total_synapses);
        net
    }

    /// 토큰 등록: 입력 뉴런 + 출력 뉴런 생성
    fn register_token(&mut self, token: &str) -> u32 {
        if let Some(&seq) = self.vocab.get(token) {
            return seq;
        }

        let seq = self.reverse_vocab.len() as u32;
        self.vocab.insert(token.to_string(), seq);
        self.reverse_vocab.push(token.to_string());

        // 입력 뉴런
        let x = (seq % 20) as f32;
        let y_in = -10.0 + (seq / 20) as f32;
        let input_id = self.neurons.len() as NeuronId;
        self.neurons.push(Neuron::new(input_id, x, y_in));
        self.input_idx.push(input_id as usize);

        // 출력 뉴런
        let y_out = 200.0 + (seq / 20) as f32;
        let output_id = self.neurons.len() as NeuronId;
        self.neurons.push(Neuron::new(output_id, x, y_out));
        self.output_idx.push(output_id as usize);

        seq
    }

    fn get_seq(&mut self, token: &str) -> u32 {
        self.register_token(token)
    }

    #[inline]
    fn input_neuron(&self, seq: u32) -> usize { self.input_idx[seq as usize] }
    #[inline]
    fn output_neuron(&self, seq: u32) -> usize { self.output_idx[seq as usize] }

    fn neuron_region(&self, idx: usize) -> RegionType {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        if idx >= es && idx < ee { RegionType::Emotion }
        else if idx >= rs && idx < re { RegionType::Reason }
        else {
            // 입력/출력: input_idx에 있으면 Input, output_idx에 있으면 Output
            if self.output_idx.contains(&idx) { RegionType::Output }
            else { RegionType::Input }
        }
    }

    fn is_output_neuron(&self, idx: usize) -> bool {
        self.output_idx.contains(&idx)
    }

    /// 거리 기반 랜덤 시냅스 생성
    fn seed_random_synapses(&mut self, per_neuron: usize) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let middle: Vec<usize> = (es..ee).chain(rs..re).collect();
        let middle_info: Vec<(usize, f32, f32)> = middle.iter()
            .map(|&i| (i, self.neurons[i].x, self.neurons[i].y)).collect();

        fn pick_nearby(rng: &mut rand::rngs::ThreadRng, sx: f32, sy: f32, candidates: &[(usize, f32, f32)], exclude: usize) -> Option<usize> {
            let weights: Vec<f64> = candidates.iter()
                .map(|&(ci, cx, cy)| {
                    if ci == exclude { return 0.0; }
                    let d = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt() as f64 + 1.0;
                    1.0 / d
                }).collect();
            let total: f64 = weights.iter().sum();
            if total <= 0.0 { return None; }
            let mut r = rng.random_range(0.0..total);
            for (i, &w) in weights.iter().enumerate() {
                r -= w;
                if r <= 0.0 { return Some(candidates[i].0); }
            }
            Some(candidates.last()?.0)
        }

        // 입력 → 감정/이성
        let input_indices: Vec<usize> = self.input_idx.clone();
        for &idx in &input_indices {
            let (x, y) = (self.neurons[idx].x, self.neurons[idx].y);
            for _ in 0..per_neuron {
                if let Some(target) = pick_nearby(&mut rng, x, y, &middle_info, idx) {
                    let tid = self.neurons[target].id;
                    self.neurons[idx].add_synapse(tid, INITIAL_WEIGHT);
                }
            }
        }

        // 감정/이성 내부 + 상호 연결 + 일부 → 출력
        let output_info: Vec<(usize, f32, f32)> = self.output_idx.iter()
            .map(|&i| (i, self.neurons[i].x, self.neurons[i].y)).collect();
        let middle_copy = middle_info.clone();
        for &(idx, x, y) in &middle_copy {
            for _ in 0..per_neuron {
                // 80% 내부/상호, 20% → 출력
                if rng.random_bool(0.8) {
                    if let Some(target) = pick_nearby(&mut rng, x, y, &middle_info, idx) {
                        let tid = self.neurons[target].id;
                        self.neurons[idx].add_synapse(tid, INITIAL_WEIGHT);
                    }
                } else {
                    let target = &output_info[rng.random_range(0..output_info.len())];
                    let tid = self.neurons[target.0].id;
                    self.neurons[idx].add_synapse(tid, INITIAL_WEIGHT);
                }
            }
        }
    }

    /// 발화
    pub fn fire(&mut self, text: &str) -> u64 {
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;
        let fire_start = std::time::Instant::now();

        let jamo_list = tokenizer::decompose_to_jamo(text);
        let seqs: Vec<u32> = jamo_list.iter().map(|j| self.get_seq(j)).collect();

        // 모든 뉴런 potential 리셋
        for n in &mut self.neurons {
            n.potential = 0.0;
        }

        // 입력 뉴런 활성화
        for &seq in &seqs {
            let idx = self.input_neuron(seq);
            self.neurons[idx].potential = 1.0;
        }

        let mut all_spikes: Vec<(NeuronId, u64)> = Vec::new();
        let mut output_spikes: Vec<(NeuronId, u64)> = Vec::new();
        let mut traces: Vec<DeliveryTrace> = Vec::new();
        let collect_traces = self.debug;
        let threshold = self.threshold;

        for tick in 0..MAX_TICKS {
            self.global_tick += 1;

            // 병렬 발화 판정
            let fired: Vec<(NeuronId, f64, Vec<(NeuronId, f64)>)> = self.neurons
                .par_iter_mut()
                .filter_map(|n| {
                    let pot = n.potential;
                    n.try_fire(tick, threshold).map(|deliveries| (n.id, pot, deliveries))
                })
                .collect();

            if fired.is_empty() {
                break;
            }

            for (nid, from_potential, deliveries) in fired {
                all_spikes.push((nid, tick));
                if self.is_output_neuron(nid as usize) {
                    output_spikes.push((nid, tick));
                }
                for (target, weight) in deliveries {
                    let tidx = target as usize;
                    if tidx < self.neurons.len() {
                        if collect_traces {
                            let to_before = self.neurons[tidx].potential;
                            self.neurons[tidx].receive(weight);
                            traces.push(DeliveryTrace {
                                from: nid, to: target, from_potential,
                                to_before, weight, delivered: weight, tick,
                            });
                        } else {
                            self.neurons[tidx].receive(weight);
                        }
                    }
                }
            }

            if output_spikes.len() >= 50 {
                break;
            }
        }

        // STDP
        self.apply_stdp(&all_spikes);

        // 축삭발아
        self.sprout(&all_spikes);

        // 출력: 발화한 출력 뉴런 → seq → 토큰
        let mut output_tokens: Vec<(String, f64)> = Vec::new();
        let mut seen_seqs = std::collections::HashSet::new();
        for &(nid, _) in &output_spikes {
            let idx = nid as usize;
            // output_idx에서 seq 찾기
            if let Some(seq) = self.output_idx.iter().position(|&oi| oi == idx) {
                if seen_seqs.insert(seq) {
                    if let Some(token) = self.reverse_vocab.get(seq) {
                        output_tokens.push((token.clone(), 1.0));
                    }
                }
            }
        }

        let token_strs: Vec<String> = output_tokens.iter().map(|(t, _)| t.clone()).collect();
        let output = tokenizer::recompose_tokens(&token_strs);

        let elapsed = fire_start.elapsed();
        eprint!(
            "  [fire #{fire_id}] {:.1}ms, spikes={}, output_spikes={}\n",
            elapsed.as_secs_f64() * 1000.0,
            all_spikes.len(),
            output_spikes.len(),
        );

        self.fire_records.push(FireRecord {
            id: fire_id, output: output.clone(), output_tokens,
            traces, spiked_neurons: all_spikes,
        });
        if self.fire_records.len() > 100 {
            self.fire_records.remove(0);
        }

        if fire_id % PRUNE_INTERVAL == 0 {
            let removed = self.prune(MIN_WEIGHT);
            if removed > 0 {
                eprintln!("  [prune] {removed}개 시냅스 제거");
            }
        }

        // 임계값 점진 상승: 0.5 → 1.0 (10만 fire마다 0.001씩)
        if self.threshold < 1.0 {
            self.threshold = (0.5 + (fire_id / 100_000) as f64 * 0.001).min(1.0);
        }

        fire_id
    }

    fn apply_stdp(&mut self, spikes: &[(NeuronId, u64)]) {
        let spike_map: HashMap<NeuronId, u64> = spikes.iter().cloned().collect();

        for &(pre_id, pre_tick) in spikes {
            let pidx = pre_id as usize;
            if pidx >= self.neurons.len() { continue; }
            let targets: Vec<(usize, NeuronId)> = self.neurons[pidx]
                .synapses.iter().enumerate()
                .map(|(i, s)| (i, s.target)).collect();

            for (syn_idx, target) in targets {
                if let Some(&post_tick) = spike_map.get(&target) {
                    let dt = post_tick as f64 - pre_tick as f64;
                    let w = self.neurons[pidx].synapses[syn_idx].weight;
                    let dw = if dt > 0.0 {
                        0.01 * (1.0 - w) * (-dt.abs() / 20.0).exp()
                    } else if dt < 0.0 {
                        -0.012 * w * (-dt.abs() / 20.0).exp()
                    } else {
                        0.0
                    };
                    if dw != 0.0 {
                        let w = &mut self.neurons[pidx].synapses[syn_idx].weight;
                        *w = (*w + dw).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    fn sprout(&mut self, spikes: &[(NeuronId, u64)]) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        const SPROUT_WEIGHT: f64 = 0.1;
        const SPROUT_RADIUS: f32 = 5.0;
        const MAX_SPROUT_PER_NEURON: usize = 1;
        const SPROUT_COOLDOWN_TICKS: u64 = 500;
        const SPROUT_PROBABILITY: f64 = 0.05;
        const MAX_SYNAPSES_PER_NEURON: usize = 30;

        let current_tick = self.global_tick;

        let spiked_info: Vec<(NeuronId, f32, f32)> = spikes.iter()
            .filter_map(|&(nid, _)| {
                let idx = nid as usize;
                if idx < self.neurons.len() {
                    Some((nid, self.neurons[idx].x, self.neurons[idx].y))
                } else { None }
            }).collect();

        let mut new_synapses: Vec<(usize, NeuronId)> = Vec::new();

        for &(nid, x, y) in &spiked_info {
            let idx = nid as usize;
            let (es, ee) = self.emotion_range;
            let (rs, re) = self.reason_range;
            let is_input = self.input_idx.contains(&idx);
            let is_output = self.is_output_neuron(idx);
            if !rng.random_bool(SPROUT_PROBABILITY) { continue; }
            if let Some(&last) = self.sprout_cooldown.get(&nid) {
                if current_tick - last < SPROUT_COOLDOWN_TICKS { continue; }
            }
            if self.neurons[idx].synapses.len() >= MAX_SYNAPSES_PER_NEURON { continue; }

            let existing: std::collections::HashSet<NeuronId> =
                self.neurons[idx].synapses.iter().map(|s| s.target).collect();

            let mut candidates: Vec<(NeuronId, f32)> = spiked_info.iter()
                .filter_map(|&(oid, ox, oy)| {
                    if oid == nid || existing.contains(&oid) { return None; }
                    let oidx = oid as usize;
                    let target_is_input = self.input_idx.contains(&oidx);
                    let target_is_output = self.output_idx.contains(&oidx);
                    // 입력↔입력, 출력↔출력, 입력↔출력 차단
                    if is_input && (target_is_input || target_is_output) { return None; }
                    if is_output && (target_is_output || target_is_input) { return None; }
                    let dist = ((x - ox).powi(2) + (y - oy).powi(2)).sqrt();
                    if dist <= SPROUT_RADIUS { Some((oid, dist)) } else { None }
                }).collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut added = 0;
            for (target, _) in candidates {
                if added >= MAX_SPROUT_PER_NEURON { break; }
                new_synapses.push((idx, target));
                added += 1;
            }
            if added > 0 {
                self.sprout_cooldown.insert(nid, current_tick);
            }
        }

        for (pre_idx, post_id) in new_synapses {
            self.neurons[pre_idx].add_synapse(post_id, SPROUT_WEIGHT);
        }
    }


    pub fn feedback(&mut self, fire_id: u64, positive: bool, strength: f64) {
        let record = match self.fire_records.iter().find(|r| r.id == fire_id) {
            Some(r) => r, None => return,
        };
        let spiked: Vec<(NeuronId, u64)> = record.spiked_neurons.clone();

        for &(nid, _) in &spiked {
            let idx = nid as usize;
            if idx >= self.neurons.len() { continue; }
            for syn in &mut self.neurons[idx].synapses {
                let dw = if positive {
                    0.01 * strength * (1.0 - syn.weight)
                } else {
                    -0.015 * strength * syn.weight
                };
                syn.weight = (syn.weight + dw).clamp(0.0, 1.0);
            }
        }
    }

    fn prune(&mut self, min_weight: f64) -> usize {
        let mut removed = 0;
        for neuron in &mut self.neurons {
            let before = neuron.synapses.len();
            neuron.synapses.retain(|s| s.weight >= min_weight);
            removed += before - neuron.synapses.len();
        }
        removed
    }

    // ── 상태 조회 ──

    pub fn synapse_count(&self) -> usize {
        self.neurons.iter().map(|n| n.synapses.len()).sum()
    }
    pub fn fire_count(&self) -> u64 { self.next_fire_id - 1 }
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    pub fn get_last_output(&self) -> String {
        self.fire_records.last().map(|r| r.output.clone()).unwrap_or_default()
    }
    pub fn get_last_record(&self) -> Option<&FireRecord> {
        self.fire_records.last()
    }

    pub fn nid_label(&self, nid: NeuronId) -> String {
        let idx = nid as usize;
        let region = self.neuron_region(idx);
        let name = match region {
            RegionType::Input => "입력",
            RegionType::Emotion => "감정",
            RegionType::Reason => "이성",
            RegionType::Output => "출력",
        };
        format!("{name} #{nid}")
    }

    pub fn get_status(&self) -> serde_json::Value {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let emotion_s: usize = (es..ee).map(|i| self.neurons[i].synapses.len()).sum();
        let reason_s: usize = (rs..re).map(|i| self.neurons[i].synapses.len()).sum();
        let input_s: usize = self.input_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();
        let output_s: usize = self.output_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();

        serde_json::json!({
            "neurons": self.neurons.len(),
            "synapses": self.synapse_count(),
            "fire_count": self.fire_count(),
            "vocab_size": self.vocab_size(),
            "threshold": format!("{:.3}", self.threshold),
            "regions": {
                "입력": { "neurons": self.input_idx.len(), "synapses": input_s },
                "감정": { "neurons": EMOTION_COUNT, "synapses": emotion_s },
                "이성": { "neurons": REASON_COUNT, "synapses": reason_s },
                "출력": { "neurons": self.output_idx.len(), "synapses": output_s },
            },
        })
    }

    pub fn print_summary(&self) {
        println!("  어휘: {}개", self.vocab_size());
        println!("  뉴런: {}개, 시냅스: {}개", self.neurons.len(), self.synapse_count());
    }

    // ── 저장/불러오기 ──

    pub fn save(&self, path: &Path) {
        let snap = Snapshot {
            neurons: self.neurons.clone(),
            vocab: self.vocab.clone(),
            reverse_vocab: self.reverse_vocab.clone(),
            input_idx: self.input_idx.clone(),
            output_idx: self.output_idx.clone(),
            emotion_range: self.emotion_range,
            reason_range: self.reason_range,
            next_fire_id: self.next_fire_id,
            threshold: self.threshold,
        };
        let data = serde_json::to_vec(&snap).expect("직렬화 실패");
        std::fs::write(path, &data).expect("저장 실패");
        eprintln!("  [저장] {} ({:.1}MB, 뉴런 {}개, 시냅스 {}개)",
            path.display(), data.len() as f64 / 1048576.0,
            self.neurons.len(), self.synapse_count());
    }

    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let snap: Snapshot = serde_json::from_slice(&data).ok()?;
        let synapse_count: usize = snap.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [불러오기] {} (뉴런 {}개, 시냅스 {}개, 어휘 {}개)",
            path.display(), snap.neurons.len(), synapse_count, snap.vocab.len());
        Some(Network {
            neurons: snap.neurons,
            vocab: snap.vocab,
            reverse_vocab: snap.reverse_vocab,
            input_idx: snap.input_idx,
            output_idx: snap.output_idx,
            emotion_range: snap.emotion_range,
            reason_range: snap.reason_range,
            fire_records: Vec::new(),
            next_fire_id: snap.next_fire_id,
            debug: false,
            threshold: snap.threshold,
            sprout_cooldown: HashMap::new(),
            global_tick: 0,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    neurons: Vec<Neuron>,
    vocab: HashMap<String, u32>,
    reverse_vocab: Vec<String>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    emotion_range: (usize, usize),
    reason_range: (usize, usize),
    next_fire_id: u64,
    #[serde(default = "default_threshold")]
    threshold: f64,
}
fn default_threshold() -> f64 { 0.5 }
