use crate::synapse::Synapse;
use serde::{Deserialize, Serialize};

pub type NeuronId = u32;

/// 불응기 상수
const ABSOLUTE_REFRACTORY: u64 = 4;
const ABSOLUTE_REFRACTORY_INH: u64 = 2;   // 억제성: 빠른 재발화로 지속 억제
const RELATIVE_REFRACTORY: u64 = 10;
const RELATIVE_REFRACTORY_INH: u64 = 4;   // 억제성: 짧은 상대 불응기

/// 뉴런: potential + 시냅스 리스트 + 2D 위치
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,
    pub potential: f64,
    pub last_spike_tick: Option<u64>,
    pub synapses: Vec<Synapse>,
    pub x: f32,
    pub y: f32,
    pub inhibitory: bool,
    /// 불응기 스킵 (입출력 뉴런용)
    #[serde(default)]
    pub skip_refractory: bool,
    /// 감응도 (항상성 가소성)
    #[serde(default = "default_excitability")]
    pub excitability: f64,
    /// 최근 1000틱 발화 횟수
    #[serde(default)]
    pub fire_count_window: u32,
    /// 막전위 감쇠율 (기본 0.7, 기억/해마는 0.85)
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f64,
    /// 뉴런별 임계값 오프셋 (기본 1.0, map 영역은 1.7 = 0.5*1.7=0.85)
    #[serde(default = "default_threshold_scale")]
    pub threshold_scale: f64,
    /// 뉴런별 자가 임계값 (기본 0.5, 발화 시 +0.001, 미발화 시 -0.001, 범위 0.5~1.0)
    #[serde(default = "default_self_threshold")]
    pub self_threshold: f64,
    /// 자신에게로 들어오는 시냅스 역참조 (pre_neuron_idx, syn_idx)
    #[serde(default)]
    pub incoming: Vec<(usize, usize)>,
}

fn default_decay_rate() -> f64 { 0.7 }
fn default_threshold_scale() -> f64 { 1.0 }
fn default_self_threshold() -> f64 { 0.5 }

fn default_excitability() -> f64 { 1.0 }

impl Neuron {
    pub fn new(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: false, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.5, incoming: Vec::new(),
        }
    }

    pub fn new_inhibitory(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: true, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.5, incoming: Vec::new(),
        }
    }

    /// 기억/해마용: 느린 감쇠
    pub fn new_slow_decay(id: NeuronId, x: f32, y: f32, inhibitory: bool) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.85, threshold_scale: 1.0, self_threshold: 0.5, incoming: Vec::new(),
        }
    }

    pub fn new_io(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: false, skip_refractory: true,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.3, incoming: Vec::new(),
        }
    }

    #[inline]
    pub fn decay(&mut self) {
        self.potential *= self.decay_rate;
        for s in &mut self.synapses {
            s.recover();
        }
    }

    #[inline]
    pub fn receive(&mut self, amount: f64) {
        self.potential += amount;
    }

    #[inline]
    pub fn try_fire(&mut self, tick: u64, threshold: f64, noise_range: f64, stimulated: bool) -> Option<Vec<(NeuronId, f64)>> {
        // 불응기 체크 (입출력 뉴런은 스킵)
        if !self.skip_refractory {
            if let Some(last) = self.last_spike_tick {
                let elapsed = tick.saturating_sub(last);
                let abs_ref = if self.inhibitory { ABSOLUTE_REFRACTORY_INH } else { ABSOLUTE_REFRACTORY };
                let rel_ref = if self.inhibitory { RELATIVE_REFRACTORY_INH } else { RELATIVE_REFRACTORY };
                if elapsed < abs_ref {
                    return None;
                }
                if elapsed < rel_ref {
                    let noise = if noise_range > 0.0 {
                        (rand::random::<f64>() * 2.0 - 1.0) * noise_range
                    } else { 0.0 };
                    if self.potential + noise < (self.self_threshold * self.threshold_scale / self.excitability) * 2.0 {
                        return None;
                    }
                }
            }
        }
        let _ = threshold; // 전역 threshold 미사용 (뉴런별 self_threshold로 대체)

        let noise = if noise_range > 0.0 {
            (rand::random::<f64>() * 2.0 - 1.0) * noise_range
        } else { 0.0 };
        let effective = self.potential + noise;
        let adj_threshold = self.self_threshold * self.threshold_scale / self.excitability;
        let should_fire = effective >= adj_threshold
            || (!self.skip_refractory && rand::random_bool(if stimulated { 0.000001 } else { 0.00001 }));

        // 신호 받은 상태인지 (potential > 0: 입력이 있었다는 뜻)
        let received_signal = self.potential > 0.01;

        if should_fire {
            self.potential = 0.0;
            self.last_spike_tick = Some(tick);
            self.fire_count_window += 1;
            // 발화 시: self_threshold +0.001 (최대 1.0)
            self.self_threshold = (self.self_threshold + 0.001).min(1.0);
            let sign = if self.inhibitory { -1.0 } else { 1.0 };
            let deliveries: Vec<_> = self.synapses.iter_mut()
                .map(|s| {
                    let w = s.effective_weight() * sign;
                    s.fire_fatigue();
                    s.last_used_tick = tick;
                    (s.target, w)
                })
                .collect();
            Some(deliveries)
        } else {
            // 미발화 + 신호 받음: self_threshold -0.001 (최소 0.5)
            if received_signal {
                // 입출력 뉴런은 최소 0.3, 일반 뉴런은 최소 0.5
                let min_thr = if self.skip_refractory { 0.3 } else { 0.5 };
                self.self_threshold = (self.self_threshold - 0.001).max(min_thr);
            }
            None
        }
    }

    /// 항상성 가소성: 1000틱마다 호출, 발화율에 따라 감응도 조절
    /// 항상성 가소성: 1만틱마다 호출, 발화율에 따라 감응도 조절
    pub fn homeostasis(&mut self) {
        let fire_rate = self.fire_count_window as f64 / 10_000.0;
        if fire_rate > 0.05 {
            self.excitability *= 0.95;
        } else if fire_rate < 0.01 {
            self.excitability *= 1.05;
        }
        self.excitability = self.excitability.clamp(0.5, 1.5);
        self.fire_count_window = 0;
    }

    pub fn add_synapse(&mut self, target: NeuronId, weight: f64) {
        self.synapses.push(Synapse::new(target, weight));
    }

    pub fn add_seed_synapse(&mut self, target: NeuronId, weight: f64) {
        self.synapses.push(Synapse::new_seed(target, weight));
    }

    pub fn distance_to(&self, other: &Neuron) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
