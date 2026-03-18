use crate::neuron::NeuronId;
use crossbeam::channel::{Receiver, Sender};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 패턴 추출 시 서브시퀀스 길이
const PATTERN_LEN: usize = 3;
/// co-firing 판정: 같은 fire 내에서 이 틱 이내에 둘 다 활성화되면 동시 발화
const COFIRE_TICK_WINDOW: u64 = 3;
/// co-firing 쌍이 시냅스로 변환되는 최소 빈도
const COFIRE_MIN_FREQ: u64 = 2;
/// co-firing 통합 시 최대 반환 쌍 수
const COFIRE_MAX_PAIRS: usize = 20;
/// consolidation 주기 (초)
const CONSOLIDATION_SECS: u64 = 30;
/// 패턴 최소 빈도
const MIN_PATTERN_FREQ: u64 = 3;

// ─── 패턴 해시 ───

type PatternHash = u64;

fn hash_pattern(neurons: &[NeuronId]) -> PatternHash {
    let mut hasher = FxHasher::default();
    for nid in neurons {
        nid.hash(&mut hasher);
    }
    hasher.finish()
}

// ─── 채널 타입 ───

/// Fire 스레드 → 해마 스레드
pub enum HippoInput {
    ActivationData {
        neurons_activated: Vec<NeuronId>,
        neuron_fire_ticks: Vec<(NeuronId, u64)>,
    },
    /// 상태 내보내기 요청 (저장 시)
    ExportState(crossbeam::channel::Sender<HippocampusState>),
    Shutdown,
}

/// 해마 스레드 → Fire 스레드
pub struct ConsolidationResult {
    pub patterns: Vec<(Vec<NeuronId>, u64)>,
    pub cofire_pairs: Vec<((NeuronId, NeuronId), u64)>,
}

// ─── 직렬화 ───

#[derive(Serialize, Deserialize)]
pub struct HippocampusState {
    pub pattern_counts: Vec<(Vec<NeuronId>, u64)>,
    pub cofire_counts: Vec<((NeuronId, NeuronId), u64)>,
    pub fire_count: usize,
}

// ─── 해마 스레드 공유 카운터 ───

pub struct HippoStats {
    pub fire_count: AtomicUsize,
    pub pattern_count: AtomicU64,
}

impl HippoStats {
    pub fn new() -> Self {
        Self {
            fire_count: AtomicUsize::new(0),
            pattern_count: AtomicU64::new(0),
        }
    }
}

// ─── 해마 본체 ───

pub struct Hippocampus {
    /// 패턴 해시 → (원본 패턴, 빈도)
    pattern_counts: HashMap<PatternHash, (Vec<NeuronId>, u64)>,
    /// co-firing 쌍 → 빈도
    cofire_counts: HashMap<(NeuronId, NeuronId), u64>,
    fire_count: usize,
    min_frequency: u64,
}

impl Hippocampus {
    pub fn new(min_frequency: u64) -> Self {
        Self {
            pattern_counts: HashMap::new(),
            cofire_counts: HashMap::new(),
            fire_count: 0,
            min_frequency,
        }
    }

    /// 발화 경로 기록: 뉴런 시퀀스에서 길이 3 서브패턴을 해시 키로 저장
    pub fn record(&mut self, neurons: &[NeuronId]) {
        self.fire_count += 1;

        if neurons.len() >= PATTERN_LEN {
            for window in neurons.windows(PATTERN_LEN) {
                let h = hash_pattern(window);
                let entry = self.pattern_counts.entry(h)
                    .or_insert_with(|| (window.to_vec(), 0));
                entry.1 += 1;
            }
        }
    }

    /// co-firing 기록: 틱별 그룹핑으로 O(틱수 × k²)
    pub fn record_cofiring(&mut self, fired_neurons: &[(NeuronId, u64)]) {
        if fired_neurons.is_empty() {
            return;
        }

        let mut by_tick: HashMap<u64, Vec<&NeuronId>> = HashMap::new();
        for (nid, tick) in fired_neurons {
            by_tick.entry(*tick).or_default().push(nid);
        }

        let mut ticks: Vec<u64> = by_tick.keys().cloned().collect();
        ticks.sort();

        for (ti, &tick_a) in ticks.iter().enumerate() {
            let neurons_a = &by_tick[&tick_a];
            // 같은 틱 내 쌍
            for i in 0..neurons_a.len() {
                for j in (i + 1)..neurons_a.len() {
                    let (a, b) = (neurons_a[i], neurons_a[j]);
                    if a == b { continue; }
                    let key = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                    *self.cofire_counts.entry(key).or_insert(0) += 1;
                }
            }
            // 인접 틱 간 쌍
            for &tick_b in &ticks[ti + 1..] {
                if tick_b - tick_a > COFIRE_TICK_WINDOW {
                    break;
                }
                let neurons_b = &by_tick[&tick_b];
                let limit_a = neurons_a.len().min(20);
                let limit_b = neurons_b.len().min(20);
                for a in &neurons_a[..limit_a] {
                    for b in &neurons_b[..limit_b] {
                        if a == b { continue; }
                        let key = if a < b { ((*a).clone(), (*b).clone()) } else { ((*b).clone(), (*a).clone()) };
                        *self.cofire_counts.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    /// 강제 통합: 빈도 높은 패턴 반환 (타이밍은 외부에서 관리)
    pub fn force_consolidate(&mut self) -> Vec<(Vec<NeuronId>, u64)> {
        let mut patterns: Vec<(Vec<NeuronId>, u64)> = self
            .pattern_counts
            .values()
            .filter(|(_, count)| *count >= self.min_frequency)
            .map(|(p, c)| (p.clone(), *c))
            .collect();

        patterns.sort_by(|a, b| b.1.cmp(&a.1));

        // 반환된 패턴의 해시 제거
        for (pattern, _) in &patterns {
            let h = hash_pattern(pattern);
            self.pattern_counts.remove(&h);
        }

        // 나머지 감쇠
        for (_, count) in self.pattern_counts.values_mut() {
            *count /= 2;
        }
        self.pattern_counts.retain(|_, (_, c)| *c > 0);

        patterns
    }

    /// 강제 co-firing 통합
    pub fn force_consolidate_cofiring(&mut self) -> Vec<((NeuronId, NeuronId), u64)> {
        let mut pairs: Vec<((NeuronId, NeuronId), u64)> = self
            .cofire_counts
            .iter()
            .filter(|(_, count)| **count >= COFIRE_MIN_FREQ)
            .map(|(p, &c)| (p.clone(), c))
            .collect();

        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(COFIRE_MAX_PAIRS);

        for (pair, _) in &pairs {
            self.cofire_counts.remove(pair);
        }

        for count in self.cofire_counts.values_mut() {
            *count = (*count * 3) / 4;
        }
        self.cofire_counts.retain(|_, c| *c > 0);

        pairs
    }

    pub fn fire_count(&self) -> usize {
        self.fire_count
    }

    pub fn pattern_count(&self) -> usize {
        self.pattern_counts.len()
    }

    pub fn export_state(&self) -> HippocampusState {
        HippocampusState {
            pattern_counts: self.pattern_counts.values()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            cofire_counts: self.cofire_counts.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            fire_count: self.fire_count,
        }
    }

    pub fn import_state(&mut self, state: HippocampusState) {
        self.pattern_counts = state.pattern_counts.into_iter()
            .map(|(pattern, count)| {
                let h = hash_pattern(&pattern);
                (h, (pattern, count))
            })
            .collect();
        self.cofire_counts = state.cofire_counts.into_iter().collect();
        self.fire_count = state.fire_count;
    }
}

// ─── 해마 스레드 메인 루프 ───

pub fn hippo_thread_main(
    hippo_rx: Receiver<HippoInput>,
    consolidation_tx: Sender<ConsolidationResult>,
    stats: Arc<HippoStats>,
    initial_state: Option<HippocampusState>,
    shutdown: Arc<AtomicBool>,
) {
    let mut hippocampus = Hippocampus::new(MIN_PATTERN_FREQ);
    if let Some(state) = initial_state {
        hippocampus.import_state(state);
        eprintln!("  [해마 스레드] 상태 복원: 패턴 {}개, fire_count {}",
            hippocampus.pattern_count(), hippocampus.fire_count());
    }

    // 초기 stats 동기화
    stats.fire_count.store(hippocampus.fire_count(), Ordering::Relaxed);
    stats.pattern_count.store(hippocampus.pattern_count() as u64, Ordering::Relaxed);

    let interval = Duration::from_secs(CONSOLIDATION_SECS);
    let mut last_consolidation = Instant::now();

    eprintln!("  [해마 스레드] 시작 (통합 주기: {}초)", CONSOLIDATION_SECS);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        match hippo_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(HippoInput::ActivationData { neurons_activated, neuron_fire_ticks }) => {
                hippocampus.record(&neurons_activated);
                hippocampus.record_cofiring(&neuron_fire_ticks);
                stats.fire_count.store(hippocampus.fire_count(), Ordering::Relaxed);
                stats.pattern_count.store(hippocampus.pattern_count() as u64, Ordering::Relaxed);
            }
            Ok(HippoInput::ExportState(reply)) => {
                let state = hippocampus.export_state();
                let _ = reply.send(state);
            }
            Ok(HippoInput::Shutdown) => {
                // 최종 통합 후 종료
                let patterns = hippocampus.force_consolidate();
                let cofire_pairs = hippocampus.force_consolidate_cofiring();
                if !patterns.is_empty() || !cofire_pairs.is_empty() {
                    let _ = consolidation_tx.send(ConsolidationResult { patterns, cofire_pairs });
                }
                eprintln!("  [해마 스레드] 종료");
                break;
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                eprintln!("  [해마 스레드] 채널 끊김, 종료");
                break;
            }
        }

        // 시간 기반 통합 (수면 모델)
        if last_consolidation.elapsed() >= interval {
            let patterns = hippocampus.force_consolidate();
            let cofire_pairs = hippocampus.force_consolidate_cofiring();

            if !patterns.is_empty() || !cofire_pairs.is_empty() {
                eprintln!(
                    "  [해마 통합] 패턴 {}개, co-fire 쌍 {}개 (fire #{}, 잔여 패턴 {}개)",
                    patterns.len(),
                    cofire_pairs.len(),
                    hippocampus.fire_count(),
                    hippocampus.pattern_count(),
                );
                let _ = consolidation_tx.try_send(ConsolidationResult { patterns, cofire_pairs });
            }

            stats.pattern_count.store(hippocampus.pattern_count() as u64, Ordering::Relaxed);
            last_consolidation = Instant::now();
        }
    }
}
