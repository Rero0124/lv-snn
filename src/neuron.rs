use crate::synapse::{PathMemory, SynapseId, SynapseStore};
use crate::tokenizer::TokenType;
use serde::{Deserialize, Serialize};

pub type NeuronId = String;

pub const DEFAULT_THRESHOLD: f64 = 0.5;
pub const MAX_ACTIVATION: f64 = 3.0;
pub const DB_WEIGHT_DISCOUNT: f64 = 0.5;

// 3-state 뉴런 상수
pub const DIVERGE_RATIO: f64 = 1.5; // activation > threshold × 이 값 → 발산
pub const PASS_RATIO: f64 = 0.85;   // activation >= threshold × 이 값 → 전달 (높을수록 엄격)
pub const DIVERGE_BOOST: f64 = 1.3; // 발산 시 신호 증폭 비율
pub const TOP_K_FIRES: usize = 10;  // 뉴런당 최대 발화 시냅스 수 (상위 K개만)

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

    /// 3-state 뉴런 발화:
    /// - 소멸: activation < threshold × PASS_RATIO → 신호 전파 없음
    /// - 전달: activation >= threshold × PASS_RATIO → 신호 그대로 전달
    /// - 발산: activation > threshold × DIVERGE_RATIO → 신호 증폭 전달
    pub fn compute_fires(
        &self,
        store: &SynapseStore,
        emitted_tokens: &std::collections::HashSet<String>,
    ) -> Vec<(SynapseId, NeuronId, f64, Option<String>, Option<TokenType>)> {
        let mut fires = Vec::new();

        // 소멸 구간: 임계값보다 한참 낮으면 전파 없음
        if self.activation < self.threshold * PASS_RATIO {
            return fires;
        }

        let diverging = self.activation > self.threshold * DIVERGE_RATIO;
        let min_forward = self.threshold * PASS_RATIO; // 이 값 미만의 forward는 전달 안 함

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

                // 전달 구간에서도 약한 시냅스는 걸러냄
                if forward < min_forward {
                    continue;
                }

                fires.push((
                    sid.clone(),
                    syn.post_neuron.clone(),
                    forward,
                    syn.token.clone(),
                    syn.token_type.clone(),
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

    /// 시냅스 생성
    pub fn create_synapse(
        &mut self,
        store: &SynapseStore,
        target: NeuronId,
        weight: f64,
        token: Option<String>,
        token_type: Option<TokenType>,
        memory: Option<PathMemory>,
    ) -> SynapseId {
        let sid = store.create(self.id.clone(), target, weight, token, token_type, memory);
        self.outgoing.push(sid.clone());
        sid
    }
}
