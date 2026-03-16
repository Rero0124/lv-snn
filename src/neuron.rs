use crate::synapse::{PathMemory, SynapseId, SynapseStore};
use rand::RngExt;
use serde::{Deserialize, Serialize};

pub type NeuronId = String;

pub const DEFAULT_THRESHOLD: f64 = 0.25;
pub const MAX_ACTIVATION: f64 = 1.0;
pub const DB_WEIGHT_DISCOUNT: f64 = 0.8;

// 확률적 뉴런 상수
pub const DIVERGE_RATIO: f64 = 1.5;        // activation > threshold × 이 값 → 발산 (신호 증폭)
pub const PASS_RATIO: f64 = 0.85;          // STDP/출력 수집용 (발화 판정은 시그모이드)
pub const DIVERGE_BOOST: f64 = 1.3;        // 발산 시 신호 증폭 비율
pub const TOP_K_FIRES: usize = 10;         // 뉴런당 최대 발화 시냅스 수
pub const SIGMOID_TEMPERATURE: f64 = 0.15; // 시그모이드 온도 (낮을수록 급격한 전환)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,
    pub outgoing: Vec<SynapseId>,
    #[serde(skip, default)]
    pub activation: f64,
    pub threshold: f64,
    /// STDP: 마지막 발화(spike) 틱
    #[serde(skip, default)]
    pub last_spike_tick: Option<u64>,
}

impl Neuron {
    pub fn new(id: NeuronId) -> Self {
        Self {
            id,
            outgoing: Vec::new(),
            activation: 0.0,
            threshold: DEFAULT_THRESHOLD,
            last_spike_tick: None,
        }
    }

    pub fn reset(&mut self) {
        self.activation = 0.0;
        self.last_spike_tick = None;
    }

    pub fn receive(&mut self, value: f64) {
        self.activation = (self.activation + value).min(MAX_ACTIVATION);
    }

    /// 확률적 뉴런 발화 (시그모이드 기반):
    /// - 발화 확률: p = 1 / (1 + exp(-(activation - threshold) / temperature))
    /// - activation >> threshold → p ≈ 1 (거의 확실히 발화)
    /// - activation << threshold → p ≈ 0 (거의 발화 안 함)
    /// - activation ≈ threshold → p ≈ 0.5 (50% 확률)
    /// - 발산: activation > threshold × DIVERGE_RATIO → 신호 증폭
    pub fn compute_fires(
        &self,
        store: &SynapseStore,
        emitted_tokens: &std::collections::HashSet<String>,
    ) -> Vec<(SynapseId, NeuronId, f64, Option<String>)> {
        let mut fires = Vec::new();

        // 시그모이드 발화 확률 계산
        let fire_prob = sigmoid(self.activation - self.threshold, SIGMOID_TEMPERATURE);

        // 확률적 판정: 난수로 발화 여부 결정
        let mut rng = rand::rng();
        if rng.random::<f64>() > fire_prob {
            return fires; // 발화하지 않음
        }

        let diverging = self.activation > self.threshold * DIVERGE_RATIO;

        for sid in &self.outgoing {
            let cached = store.is_cached(sid);
            if let Some(syn) = store.get(sid) {
                let discount = if cached { 1.0 } else { DB_WEIGHT_DISCOUNT };
                let mut forward = self.activation * syn.weight * discount;

                // 이미 출력된 토큰이면 가중치 50% 감소
                if let Some(ref tok) = syn.token {
                    if emitted_tokens.contains(tok) {
                        forward *= 0.5;
                    }
                }

                // 발산 구간: 신호 증폭
                if diverging {
                    forward *= DIVERGE_BOOST;
                }

                // 시냅스별 확률적 전달: forward 값에 비례한 확률
                let syn_prob = sigmoid(forward - self.threshold * 0.5, SIGMOID_TEMPERATURE);
                if rng.random::<f64>() > syn_prob {
                    continue; // 이 시냅스는 전달하지 않음
                }

                fires.push((
                    sid.clone(),
                    syn.post_neuron.clone(),
                    forward,
                    syn.token.clone(),
                ));
            }
        }

        // 상위 TOP_K 시냅스만 발화 (신호 집중)
        if fires.len() > TOP_K_FIRES {
            fires.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            fires.truncate(TOP_K_FIRES);
        }

        fires
    }

    /// 시냅스 생성 (확률적 발화와 무관, 구조 변경 없음)
    pub fn create_synapse(
        &mut self,
        store: &SynapseStore,
        target: NeuronId,
        weight: f64,
        token: Option<String>,
        memory: Option<PathMemory>,
    ) -> SynapseId {
        let sid = store.create(self.id.clone(), target, weight, token, memory);
        self.outgoing.push(sid.clone());
        sid
    }
}

/// 시그모이드 함수: 1 / (1 + exp(-x / temperature))
fn sigmoid(x: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-x / temperature).exp())
}
