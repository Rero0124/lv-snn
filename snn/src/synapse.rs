use crate::config;
use crate::neuron::NeuronId;
use serde::{Deserialize, Serialize};

/// 시냅스: 목적지 뉴런 + 가중치
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub target: NeuronId,
    pub weight: f64,
    /// 초기 시냅스 여부 — true면 가지치기 면제
    #[serde(default)]
    pub seed: bool,
    /// 시냅스 피로도 (1.0 = 정상, 0.0 = 완전 고갈)
    #[serde(default = "default_fatigue")]
    pub fatigue: f64,
    /// 마지막 사용 tick (자체 prune 판정용)
    #[serde(default)]
    pub last_used_tick: u64,
    /// 전달 빈도 (자주 쓰일수록 약화)
    #[serde(default)]
    pub recent_rate: f64,
}

fn default_fatigue() -> f64 { config::FATIGUE_INIT }

impl Synapse {
    pub fn new(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: false, fatigue: config::FATIGUE_INIT, last_used_tick: 0, recent_rate: 0.0 }
    }

    pub fn new_seed(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: true, fatigue: config::FATIGUE_INIT, last_used_tick: 0, recent_rate: 0.0 }
    }

    /// 발화 시 피로(fatigue) 누적. recent_rate 는 전달 빈도 추적용으로만 갱신
    /// (현재 전달력에는 미반영 — 쿨다운 페널티/할인 제거됨).
    #[inline]
    pub fn fire_fatigue(&mut self) {
        self.fatigue *= config::FATIGUE_DECAY;
        self.recent_rate = (self.recent_rate + config::RECENT_RATE_STEP).min(config::RECENT_RATE_MAX);
    }

    /// 매 틱 피로 회복 + rate 감쇠
    #[inline]
    pub fn recover(&mut self) {
        self.fatigue = (self.fatigue + config::FATIGUE_RECOVER).min(config::FATIGUE_MAX);
        self.recent_rate *= config::RECENT_RATE_DECAY;
    }

    /// 실효 가중치 (weight × fatigue)
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.fatigue
    }
}
