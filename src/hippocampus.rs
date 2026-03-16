use crate::neuron::NeuronId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 패턴 추출 시 서브시퀀스 길이
const PATTERN_LEN: usize = 3;
/// co-firing 판정: 같은 fire 내에서 이 틱 이내에 둘 다 활성화되면 동시 발화
const COFIRE_TICK_WINDOW: u64 = 3;
/// co-firing 쌍이 시냅스로 변환되는 최소 빈도
const COFIRE_MIN_FREQ: u64 = 2;
/// co-firing 통합 시 최대 반환 쌍 수
const COFIRE_MAX_PAIRS: usize = 20;

/// 해마 상태 직렬화용
#[derive(Serialize, Deserialize)]
pub struct HippocampusState {
    pub pattern_counts: Vec<(Vec<NeuronId>, u64)>,
    pub cofire_counts: Vec<((NeuronId, NeuronId), u64)>,
    pub fire_count: usize,
}

/// 해마: 발화 경로를 뉴런 단위로 추적하고 자주 보이는 패턴을 기억 구역에 전달
/// + co-firing 감지: 비슷한 타이밍에 발화한 뉴런 쌍을 추적하여 시냅스 생성 제안
pub struct Hippocampus {
    /// 뉴런 경로 패턴 → 빈도
    pattern_counts: HashMap<Vec<NeuronId>, u64>,
    /// co-firing 쌍 → 빈도 (뉴런A, 뉴런B가 비슷한 시간에 발화한 횟수)
    cofire_counts: HashMap<(NeuronId, NeuronId), u64>,
    fire_count: usize,
    consolidation_interval: usize,
    min_frequency: u64,
}

impl Hippocampus {
    pub fn new(consolidation_interval: usize, min_frequency: u64) -> Self {
        Self {
            pattern_counts: HashMap::new(),
            cofire_counts: HashMap::new(),
            fire_count: 0,
            consolidation_interval,
            min_frequency,
        }
    }

    /// 발화 경로 기록: 뉴런 시퀀스에서 길이 3 서브패턴 추출
    pub fn record(&mut self, neurons: &[NeuronId]) {
        self.fire_count += 1;

        if neurons.len() >= PATTERN_LEN {
            for window in neurons.windows(PATTERN_LEN) {
                *self.pattern_counts.entry(window.to_vec()).or_insert(0) += 1;
            }
        }
    }

    /// co-firing 기록: (뉴런ID, 발화 틱) 목록에서 가까운 틱에 발화한 뉴런 쌍 추적
    pub fn record_cofiring(&mut self, fired_neurons: &[(NeuronId, u64)]) {
        // 틱 순서로 정렬
        let mut sorted = fired_neurons.to_vec();
        sorted.sort_by_key(|(_, tick)| *tick);

        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                let (ref a, tick_a) = sorted[i];
                let (ref b, tick_b) = sorted[j];
                if tick_b - tick_a > COFIRE_TICK_WINDOW {
                    break; // 이후는 더 멀리 있으니 skip
                }
                if a == b {
                    continue;
                }
                // 정렬된 키 (a < b) 로 저장하여 중복 방지
                let key = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                *self.cofire_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// 통합 시점이면 빈도 높은 패턴 반환, 전달된 패턴은 해마에서 제거
    pub fn maybe_consolidate(&mut self) -> Vec<(Vec<NeuronId>, u64)> {
        if self.fire_count % self.consolidation_interval != 0 {
            return Vec::new();
        }

        let mut patterns: Vec<(Vec<NeuronId>, u64)> = self
            .pattern_counts
            .iter()
            .filter(|(_, count)| **count >= self.min_frequency)
            .map(|(p, &c)| (p.clone(), c))
            .collect();

        patterns.sort_by(|a, b| b.1.cmp(&a.1));

        // 기억 구역에 전달된 패턴은 해마에서 제거 (중복 보유 방지)
        for (pattern, _) in &patterns {
            self.pattern_counts.remove(pattern);
        }

        // 나머지 카운터 감쇠 + 0 제거
        for count in self.pattern_counts.values_mut() {
            *count /= 2;
        }
        self.pattern_counts.retain(|_, c| *c > 0);

        patterns
    }

    /// co-firing 통합: 빈도 높은 뉴런 쌍 반환 → 네트워크가 시냅스 생성/강화
    pub fn consolidate_cofiring(&mut self) -> Vec<((NeuronId, NeuronId), u64)> {
        if self.fire_count % self.consolidation_interval != 0 {
            return Vec::new();
        }

        let mut pairs: Vec<((NeuronId, NeuronId), u64)> = self
            .cofire_counts
            .iter()
            .filter(|(_, count)| **count >= COFIRE_MIN_FREQ)
            .map(|(p, &c)| (p.clone(), c))
            .collect();

        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(COFIRE_MAX_PAIRS);

        // 반환된 쌍은 제거
        for (pair, _) in &pairs {
            self.cofire_counts.remove(pair);
        }

        // 나머지 감쇠
        for count in self.cofire_counts.values_mut() {
            *count = (*count * 3) / 4; // 75% 유지
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

    pub fn cofire_pair_count(&self) -> usize {
        self.cofire_counts.len()
    }

    pub fn export_state(&self) -> HippocampusState {
        HippocampusState {
            pattern_counts: self.pattern_counts.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            cofire_counts: self.cofire_counts.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            fire_count: self.fire_count,
        }
    }

    pub fn import_state(&mut self, state: HippocampusState) {
        self.pattern_counts = state.pattern_counts.into_iter().collect();
        self.cofire_counts = state.cofire_counts.into_iter().collect();
        self.fire_count = state.fire_count;
    }
}
