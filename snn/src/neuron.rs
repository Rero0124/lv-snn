use crate::synapse::Synapse;
use serde::{Deserialize, Serialize};

pub type NeuronId = u32;

/// 뉴런: potential + 시냅스 리스트 + 2D 위치
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,
    /// 현재 막전위 (>= 1.0이면 발화)
    pub potential: f64,
    /// 마지막 스파이크 틱
    pub last_spike_tick: Option<u64>,
    /// 나가는 시냅스 목록
    pub synapses: Vec<Synapse>,
    /// 2D 그리드 위치
    pub x: f32,
    pub y: f32,
    /// 억제성 뉴런 여부 (true면 시냅스 weight가 음수로 전달)
    pub inhibitory: bool,
}

impl Neuron {
    pub fn new(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id,
            potential: 0.0,
            last_spike_tick: None,
            synapses: Vec::new(),
            x,
            y,
            inhibitory: false,
        }
    }

    pub fn new_inhibitory(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id,
            potential: 0.0,
            last_spike_tick: None,
            synapses: Vec::new(),
            x,
            y,
            inhibitory: true,
        }
    }

    #[inline]
    pub fn receive(&mut self, amount: f64) {
        self.potential += amount;
    }

    /// 발화 판정: potential >= threshold이면 발화, 또는 자발적 발화 (0.001 확률)
    #[inline]
    pub fn try_fire(&mut self, tick: u64, threshold: f64) -> Option<Vec<(NeuronId, f64)>> {
        let should_fire = self.potential >= threshold
            || (self.potential > 0.0 && rand::random_bool(0.001));

        if should_fire {
            self.potential = 0.0;
            self.last_spike_tick = Some(tick);
            let sign = if self.inhibitory { -1.0 } else { 1.0 };
            let deliveries: Vec<_> = self.synapses.iter()
                .map(|s| (s.target, s.weight * sign))
                .collect();
            Some(deliveries)
        } else {
            None
        }
    }

    pub fn add_synapse(&mut self, target: NeuronId, weight: f64) {
        self.synapses.push(Synapse::new(target, weight));
    }

    /// 두 뉴런 간 거리
    pub fn distance_to(&self, other: &Neuron) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
