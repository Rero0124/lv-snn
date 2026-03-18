use crate::synapse::{PathMemory, SynapseId, SynapseStore};
use rand::RngExt;
use serde::{Deserialize, Serialize};

pub type NeuronId = String;

pub const DEFAULT_THRESHOLD: f64 = 0.4;
pub const MAX_ACTIVATION: f64 = 1.0;
pub const DB_WEIGHT_DISCOUNT: f64 = 0.8;

// 확률적 뉴런 상수
pub const DIVERGE_RATIO: f64 = 1.5;        // activation > threshold × 이 값 → 발산 (신호 증폭)
pub const PASS_RATIO: f64 = 0.85;          // STDP/출력 수집용 (발화 판정은 시그모이드)
pub const DIVERGE_BOOST: f64 = 1.3;        // 발산 시 신호 증폭 비율
pub const TOP_K_FIRES: usize = 10;         // 뉴런당 최대 발화 시냅스 수
pub const SIGMOID_TEMPERATURE: f64 = 0.15; // 시그모이드 온도 (낮을수록 급격한 전환)

/// 뉴런이 들고 있는 시냅스 정보 (발화 시 DB 조회 불필요)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutgoingSynapse {
    pub id: SynapseId,
    pub weight: f64,
    pub modifier: f64,
    #[serde(default)]
    pub post_neuron: NeuronId,
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,
    /// 하위 호환용 (기존 DB 로드 시)
    #[serde(default)]
    pub outgoing: Vec<SynapseId>,
    /// 시냅스 요약 정보 (weight/modifier 캐시)
    #[serde(default)]
    pub outgoing_cache: Vec<OutgoingSynapse>,
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
            outgoing_cache: Vec::new(),
            activation: 0.0,
            threshold: DEFAULT_THRESHOLD,
            last_spike_tick: None,
        }
    }

    /// 기존 outgoing(ID만)에서 outgoing_cache로 마이그레이션
    pub fn migrate_outgoing(&mut self, store: &SynapseStore) {
        if !self.outgoing_cache.is_empty() || self.outgoing.is_empty() {
            return;
        }
        for sid in &self.outgoing {
            if let Some(syn) = store.get(sid) {
                self.outgoing_cache.push(OutgoingSynapse {
                    id: sid.clone(),
                    weight: syn.weight,
                    modifier: syn.modifier,
                    post_neuron: syn.post_neuron.clone(),
                    token: syn.token.clone(),
                });
            }
        }
        self.outgoing.clear();
    }

    /// outgoing_cache에서 post_neuron이 비어있는 항목을 store에서 채움
    pub fn fill_missing_post_neurons(&mut self, store: &SynapseStore) {
        for os in &mut self.outgoing_cache {
            if os.post_neuron.is_empty() {
                if let Some(syn) = store.get(&os.id) {
                    os.post_neuron = syn.post_neuron.clone();
                    os.token = syn.token.clone();
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.activation = 0.0;
        self.last_spike_tick = None;
    }

    pub fn receive(&mut self, value: f64) {
        self.activation = (self.activation + value).min(MAX_ACTIVATION);
    }

    /// 시냅스별 확률적 발화: 뉴런은 발화하지 않음
    /// forward = activation × weight + modifier → 시그모이드 확률로 발화 여부 결정
    pub fn compute_fires(
        &self,
        emitted_tokens: &std::collections::HashSet<String>,
    ) -> Vec<(SynapseId, NeuronId, f64, Option<String>)> {
        let mut fires = Vec::new();

        if self.activation <= 0.0 {
            return fires;
        }

        let mut rng = rand::rng();

        // 각 시냅스별로 독립적으로 발화 판정
        for os in self.outgoing_cache.iter() {
            let mut forward = self.activation * os.weight + os.modifier;
            if forward <= 0.0 {
                continue;
            }
            // 계단식 비선형 확률: 구간별 베이스 + 구간 내 시그모이드 보간
            let fire_prob = stepped_fire_prob(forward);
            if rng.random::<f64>() > fire_prob {
                continue;
            }
            if let Some(ref tok) = os.token {
                if emitted_tokens.contains(tok) {
                    forward *= 0.5;
                }
            }
            fires.push((
                os.id.clone(),
                os.post_neuron.clone(),
                forward,
                os.token.clone(),
            ));
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
        let sid = store.create(self.id.clone(), target.clone(), weight, token.clone(), memory);
        self.outgoing_cache.push(OutgoingSynapse {
            id: sid.clone(),
            weight,
            modifier: 0.0,
            post_neuron: target,
            token,
        });
        sid
    }
}

/// 시그모이드 함수: 1 / (1 + exp(-x / temperature))
fn sigmoid(x: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-x / temperature).exp())
}

/// 계단식 비선형 발화 확률
/// 구간별 베이스 확률 + 구간 내 시그모이드 보간으로 자연스러운 전환
fn stepped_fire_prob(forward: f64) -> f64 {
    // (구간 시작, 구간 끝, 베이스 확률, 다음 확률)
    let steps: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.2, 0.001, 0.03),
        (0.2, 0.4, 0.03, 0.10),
        (0.4, 0.65, 0.10, 0.40),
        (0.65, 0.75, 0.40, 0.65),
        (0.75, 1.0, 0.65, 0.98),
    ];

    if forward >= 1.0 {
        return 0.96;
    }

    for &(lo, hi, base, next) in steps {
        if forward < hi {
            // 구간 내 위치 (0~1)
            let t = (forward - lo) / (hi - lo);
            // 시그모이드 보간 (S자 곡선으로 부드럽게)
            let s = 1.0 / (1.0 + (-(t * 6.0 - 3.0)).exp());
            return base + (next - base) * s;
        }
    }
    0.96
}
