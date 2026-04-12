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
    /// LTP 누적 카운터: 짧은 시간 내 반복 활성화 횟수
    #[serde(default)]
    pub ltp_trace: f64,
}

fn default_fatigue() -> f64 { 1.0 }

impl Synapse {
    pub fn new(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: false, fatigue: 1.0, ltp_trace: 0.0 }
    }

    pub fn new_seed(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: true, fatigue: 1.0, ltp_trace: 0.0 }
    }

    /// 발화 시 피로 적용
    #[inline]
    pub fn fire_fatigue(&mut self) {
        self.fatigue *= 0.80;
    }

    /// 매 틱 피로 회복 + LTP trace 감쇠
    #[inline]
    pub fn recover(&mut self) {
        self.fatigue = (self.fatigue + 0.01).min(1.0);
        self.ltp_trace *= 0.95; // 자연 감쇠
    }

    /// LTP 누적: 활성화될 때마다 trace 증가, 누적된 만큼 추가 강화 반환
    #[inline]
    pub fn accumulate_ltp(&mut self) -> f64 {
        self.ltp_trace += 1.0;
        // trace가 2 이상이면 반복 자극 → 추가 LTP
        if self.ltp_trace >= 2.0 {
            (self.ltp_trace - 1.0) * 0.005
        } else {
            0.0
        }
    }

    /// 실효 가중치 (weight × fatigue)
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.fatigue
    }
}
