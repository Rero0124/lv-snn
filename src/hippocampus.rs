use crate::neuron::NeuronId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 패턴 추출 시 서브시퀀스 길이
const PATTERN_LEN: usize = 3;

/// 해마 상태 직렬화용
#[derive(Serialize, Deserialize)]
pub struct HippocampusState {
    pub pattern_counts: Vec<(Vec<NeuronId>, u64)>,
    pub fire_count: usize,
}

/// 해마: 발화 경로를 뉴런 단위로 추적하고 자주 보이는 패턴을 기억 구역에 전달
pub struct Hippocampus {
    /// 뉴런 경로 패턴 → 빈도
    pattern_counts: HashMap<Vec<NeuronId>, u64>,
    fire_count: usize,
    consolidation_interval: usize,
    min_frequency: u64,
}

impl Hippocampus {
    pub fn new(consolidation_interval: usize, min_frequency: u64) -> Self {
        Self {
            pattern_counts: HashMap::new(),
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

    pub fn fire_count(&self) -> usize {
        self.fire_count
    }

    pub fn pattern_count(&self) -> usize {
        self.pattern_counts.len()
    }

    pub fn export_state(&self) -> HippocampusState {
        HippocampusState {
            pattern_counts: self.pattern_counts.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            fire_count: self.fire_count,
        }
    }

    pub fn import_state(&mut self, state: HippocampusState) {
        self.pattern_counts = state.pattern_counts.into_iter().collect();
        self.fire_count = state.fire_count;
    }
}
