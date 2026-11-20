use crate::synapse::Synapse;
use serde::{Deserialize, Serialize};

pub type NeuronId = u32;

/// 불응기 상수
const ABSOLUTE_REFRACTORY: u64 = 4;
const ABSOLUTE_REFRACTORY_INH: u64 = 2;   // 억제성: 빠른 재발화로 지속 억제
const RELATIVE_REFRACTORY: u64 = 10;
const RELATIVE_REFRACTORY_INH: u64 = 4;   // 억제성: 짧은 상대 불응기

/// 세포 종류 (포유류 대뇌피질 분류 기준). 종류마다 발화 특성과 연결 규칙이 다르다.
///   Pyramidal : 흥분성 추체세포. 규칙 발화 + 발화할수록 느려짐(적응). 다른 영역으로 멀리 투사
///   Pv        : 억제성, 고속 발화(불응기 짧음, 적응 없음). 국소. 흥분성 뉴런과 Pv 끼리 억제
///   Sst       : 억제성, 적응 있음. 국소. 흥분성·Pv·Vip 를 억제하고 Sst 끼리는 거의 억제하지 않음
///   Vip       : 억제성. 국소. 주로 Sst 를 억제 → 흥분성 뉴런의 억제를 풀어 줌(탈억제)
///   Modulatory: 조절 뉴런(도파민 등). 전위를 올리지 않고 학습만 조절
///   Value: 가치 뉴런(흥분성). 보상을 예고하는 자극에 반응해 도파민 뉴런을 흥분시킨다 (예고 자극이 2차 보상이 되는 경로)
///   Predictor: 보상 예측 뉴런(억제성). 도파민 뉴런의 보상 입력을 상쇄한다. 입력 시냅스는 (최근 활성화 × 도파민 변화)로 학습
///   Legacy    : 종류 구분 없는 기존 뉴런 (흥분/억제 플래그만)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CellType { #[default] Legacy, Pyramidal, Pv, Sst, Vip, Modulatory, Predictor, Value }

/// 자기 목표를 잴 때 세는 발화 간격의 수: 첫 발화에서 (OWN_COUNT + 1)째 발화까지의 틱으로 발화율을 잰다
pub const OWN_COUNT: u32 = 16;

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
    /// 불응기 배수 (기본 1.0, 기억/해마 2.0)
    #[serde(default = "default_refractory_scale")]
    pub refractory_scale: u64,
    /// 최근 발화율 (틱당 발화 확률의 지수이동평균, 시간 상수 약 100틱). 발화율 기준 BCM 에서 사용
    #[serde(default)]
    pub rate_fast: f64,
    /// 장기 평균 발화율 (틱당 발화 확률의 지수이동평균, 시간 상수 약 5000틱). 교과서 BCM 에서 사용
    #[serde(default)]
    pub rate_avg: f64,
    /// 항상성의 뉴런별 목표 발화율 (자기 평소 수준). 음수 = 아직 정해지지 않음
    #[serde(default = "default_set_point")]
    pub set_point: f64,
    /// 자기에게 들어오는 시냅스 가중치의 중앙값 / 최댓값 (plasticity::refresh_local_scales 가 기록).
    /// 학습 변화량과 soft bound 를 뉴런이 알 수 있는 값으로 정하는 데 쓴다
    #[serde(default)]
    pub w_ref_med: f64,
    #[serde(default)]
    pub w_ref_max: f64,
    /// 수상돌기 구획 사용 (추체세포의 두 구획 모델). false 면 기존의 점 뉴런
    #[serde(default)]
    pub has_dendrite: bool,
    /// 수상돌기 구획의 전위. 세포체보다 느리게 사라진다 (틱당 ×0.85)
    #[serde(default)]
    pub dend_potential: f64,
    /// 직전 발화가 수상돌기 고원 전위와 겹친 버스트였는가
    #[serde(default)]
    pub last_burst: bool,
    /// 세포 종류 (기본 Legacy)
    #[serde(default)]
    pub cell_type: CellType,
    /// 종류별 불응기 (틱). 0 이면 기존 규칙(흥분 4/10, 억제 2/4)을 따른다
    #[serde(default)]
    pub abs_refractory: u64,
    #[serde(default)]
    pub rel_refractory: u64,
    /// 발화 빈도 적응: 발화할 때마다 임계값에 adapt_inc 만큼 더해지고(adapt), 매 틱 adapt_decay 배로 줄어든다
    #[serde(default)]
    pub adapt: f64,
    #[serde(default)]
    pub adapt_inc: f64,
    #[serde(default = "default_out_gain")]
    pub adapt_decay: f64,
    /// 기존의 느린 임계값 적응(발화 시 +0.001, 미발화 시 −0.001) 사용 여부. Pv 는 적응이 거의 없으므로 끈다
    #[serde(default = "default_true")]
    pub slow_adapt: bool,
    /// 조절 뉴런(도파민 등): 발화해도 받는 쪽의 전위를 올리지 않고 조절 수준(mod_level)만 올린다
    #[serde(default)]
    pub modulatory: bool,
    /// 이 뉴런에 도착한 조절 신호의 수준. 수백 ms 에 걸쳐 사라진다 (틱당 ×0.98)
    #[serde(default)]
    pub mod_level: f64,
    /// 이 뉴런이 평소에 받는 조절 신호의 기준선. 조절 신호를 받기 시작한 뒤 처음 MOD_BASE_TICKS 틱의 평균으로 정해지고 그 뒤에는 고정된다.
    /// 학습은 조절 수준이 이 기준선에서 얼마나 벗어났는지(위 = 양, 아래 = 음)에 따라 일어난다
    #[serde(default)]
    pub mod_base: f64,
    /// 조절 신호를 받기 시작한 뒤 지난 틱 수 (기준선 평균을 내는 데 사용)
    #[serde(default)]
    pub mod_seen: u32,
    /// 스스로 내는 일정한 전류 (매 틱 전위에 더해짐). 0 보다 크면 입력이 없어도 규칙적으로 발화한다 (도파민 뉴런의 기준선 발화)
    #[serde(default)]
    pub tonic: f64,
    /// 스스로 발화하는 뉴런의 수상돌기 기여를 0 아래로 내리지 않는다 (시험용 --pred-floor: 예측 억제가 보상 입력을 깎기만 하고 평소 발화는 누르지 못한다).
    /// 수상돌기에 쌓이는 값 자체는 음수가 될 수 있다. 저장하지 않는다(불러올 때 옵션으로 다시 정한다). 초파리는 켜지 않는다
    #[serde(skip)]
    pub dend_floor: bool,
    /// 자발 발화 확률 배수 (기본 1.0, 0 이면 자발 발화 없음)
    #[serde(default = "default_out_gain")]
    pub spont_scale: f64,
    /// 가지치기 수명의 시간 단위와 기준 가중치 (prune_tau 가 0 이면 가지치기 안 함). 수명 = prune_tau × ln(가중치 ÷ prune_min), 최소 2만 틱
    #[serde(default)]
    pub prune_tau: f64,
    #[serde(default)]
    pub prune_min: f64,
    /// 가지치기 기한을 다시 세는 조건의 바닥 (보내는 뉴런의 값): 도파민으로 강화 쪽이 굳었을 때, 굳은 뒤의 가중치가 이 값 이상이어야 기한을 다시 센다.
    ///   0 이면(기본, 이 칸이 없는 저장 파일도) 조건이 없는 것과 같다 — 가중치는 0 이상이라 늘 참이다
    #[serde(default)]
    pub reset_min: f64,
    /// 자기 값으로 정하는 목표 (받기 문의 목표를 뉴런 각자가 잰다). own_on 이 거짓이면(기본, 이 칸들이 없는 저장 파일도) 아무것도 하지 않는다.
    ///   켜져 있고 아직 목표가 없으면(own_target == 0) 자기 발화를 센다: 첫 발화의 틱을 own_t0 에 적고, 열일곱째 발화에서
    ///   목표 = 16 ÷ (지금 − own_t0) 로 굳힌다 (틱당 발화율). 자기 자발 발화 확률(0.00001 × spont_scale) 아래로는 내려가지 않고, 위 한계는 없다.
    ///   한 번 굳으면 바뀌지 않는다. 공통 박자 없이 자기 발화와 그때 읽은 틱만 쓴다
    #[serde(default)]
    pub own_on: bool,
    #[serde(default)]
    pub own_target: f64,
    #[serde(default)]
    pub own_n: u32,
    #[serde(default)]
    pub own_t0: u64,
    /// 감응도 항상성이 굳힌 자기 목표를 본다 (기본 거짓, 이 칸이 없는 저장 파일도 거짓 = homeostasis_own 이 지금과 같다).
    ///   참이고 자기 목표가 굳어 있으면(own_target > 0) 평소 수준(set_point) 대신 own_target 과 견주고, set_point 는 고치지 않는다
    #[serde(default)]
    pub own_homeo: bool,
    /// 조절 수준이 틱마다 줄어드는 배수 (기본 0.98). 가치 뉴런은 더 빨리 줄어들게 해서(0.9) 도파민의 급증을 짧게 읽는다
    #[serde(default = "default_mod_decay")]
    pub mod_decay: f64,
    /// 몸의 상태 뉴런(배고픔, 목마름)의 결핍 정도 (0 = 채워짐, 1 = 가장 부족). 매 틱 need_rate 만큼 오르고,
    /// 억제 입력(포만 신호)을 받으면 그만큼 내려간다. 자체 전류가 결핍 정도에 비례해서, 부족할수록 자주 발화한다
    #[serde(default)]
    pub need: f64,
    /// 결핍이 틱마다 오르는 양 (0 보다 크면 몸의 상태 뉴런)
    #[serde(default)]
    pub need_rate: f64,
    /// 방금 가지치기를 했는지 (했으면 네트워크가 이 뉴런으로 들어오는 시냅스의 기한도 확인한다)
    #[serde(skip)]
    pub did_prune: bool,
    /// 방금의 가지치기에서 사라진 접점 수 (0 이 아니면 네트워크가 죽은 연결의 자리를 정리한다)
    #[serde(skip)]
    pub pruned_n: u32,
    /// 이 틱 전에는 자기 시냅스의 가지치기를 다시 하지 않는다 (마지막으로 정리한 틱 + 1000)
    #[serde(default)]
    pub next_prune: u64,
    /// 내보내는 시냅스 중 흔적(elig)이 남아 있는 것의 번호 (조절 신호 방식에서, 흔적을 넘길 때 전체 시냅스를 돌지 않기 위한 목록)
    #[serde(skip)]
    pub elig_idx: Vec<u32>,
    /// 자기에게 들어오는 시냅스 중 흔적이 남아 있는 것 (보내는 뉴런, 시냅스 번호). 조절 신호가 벗어난 뉴런이 적을 때는 이쪽 목록으로 처리한다
    #[serde(skip)]
    pub elig_in: Vec<(u32, u32)>,
    /// 교과서 BCM 의 LTD 배수와 그것을 계산한 틱 (입력 시냅스 전체를 훑어야 하는 값이라 BCM_CACHE_TICKS 틱 동안 다시 쓴다)
    /// 저장한다: 저장하지 않으면 불러온 직후 배수를 새로 계산하게 되어, 끊지 않은 실행(아직 앞에 계산한 값을 쓰는 중)과 약화의 양이 달라진다
    #[serde(default = "default_out_gain")]
    pub ltd_mult: f64,
    #[serde(default)]
    pub ltd_mult_tick: u64,
    /// 자신에게로 들어오는 시냅스 역참조 (pre_neuron_idx, syn_idx)
    #[serde(default)]
    pub incoming: Vec<(usize, usize)>,
    /// 결정적 난수의 씨앗 (0 = 쓰지 않음: 지금처럼 전역 난수기에서 뽑는다). 0 이 아니면 잡음·자발 발화·확률 전달의 난수를
    /// (씨앗, 쓰임새, 뉴런 번호, 시냅스 번호, 틱)에서 정한다: 도는 순서와 스레드 수, 다른 난수를 몇 번 뽑았는지와 무관하다
    #[serde(default)]
    pub rng_seed: u64,
}

/// 수상돌기 구획: 고원 전위가 일어나는 문턱, 그때 세포체에 주는 탈분극, 그 아래에서 새어 들어가는 비율, 버스트의 전달 배수
pub const DEND_THRESHOLD: f64 = 0.5;
pub const DEND_PLATEAU: f64 = 0.35;
pub const DEND_LEAK: f64 = 0.1;
pub const BURST_GAIN: f64 = 2.0;
/// 조절 신호의 기준선을 정하는 데 쓰는 틱 수 (받기 시작한 뒤 이만큼의 평균)
pub const MOD_BASE_TICKS: u32 = 2000;
pub const MOD_BASE_SKIP: u32 = 300;

/// Sprout 시 초기 가중치
pub const SPROUT_WEIGHT: f64 = 0.2;

fn default_decay_rate() -> f64 { 0.7 }
fn default_threshold_scale() -> f64 { 1.0 }
fn default_self_threshold() -> f64 { 0.5 }
fn default_refractory_scale() -> u64 { 1 }
fn default_out_gain() -> f64 { 1.0 }
fn default_true() -> bool { true }
fn default_mod_decay() -> f64 { 0.98 }
fn default_set_point() -> f64 { -1.0 }

fn default_excitability() -> f64 { 1.0 }

/// 열쇠에서 정해지는 난수 (64비트): (씨앗, 쓰임새, 뉴런 번호, 시냅스 번호, 틱)을 splitmix64 로 섞는다
#[inline]
pub fn keyed_u64(seed: u64, purpose: u64, neuron: u64, syn: u64, tick: u64) -> u64 {
    #[inline] fn mix(mut z: u64) -> u64 { z = z.wrapping_add(0x9E3779B97F4A7C15); z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
    let mut h = mix(seed ^ purpose.wrapping_mul(0xD6E8FEB86659FD93));
    h = mix(h ^ neuron);
    h = mix(h ^ syn.wrapping_mul(0xA24BAED4963EE407));
    mix(h ^ tick)
}
/// [0, 1) 의 실수 (위 53비트)
#[inline]
pub fn keyed_f64(seed: u64, purpose: u64, neuron: u64, syn: u64, tick: u64) -> f64 {
    (keyed_u64(seed, purpose, neuron, syn, tick) >> 11) as f64 / (1u64 << 53) as f64
}

impl Neuron {
    pub fn new(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: false, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.5, refractory_scale: 1, rate_fast: 0.0, rate_avg: 0.0, set_point: -1.0, w_ref_med: 0.0, w_ref_max: 0.0, has_dendrite: false, dend_potential: 0.0, last_burst: false, cell_type: CellType::Legacy, abs_refractory: 0, rel_refractory: 0, adapt: 0.0, adapt_inc: 0.0, adapt_decay: 1.0, slow_adapt: true, modulatory: false, mod_level: 0.0, mod_base: 0.0, mod_seen: 0, tonic: 0.0, dend_floor: false, next_prune: 0, prune_tau: 0.0, prune_min: 0.0, reset_min: 0.0, own_on: false, own_target: 0.0, own_n: 0, own_t0: 0, own_homeo: false, pruned_n: 0, did_prune: false, need: 0.0, need_rate: 0.0, mod_decay: 0.98, elig_idx: Vec::new(), elig_in: Vec::new(), ltd_mult: 1.0, ltd_mult_tick: 0, spont_scale: 1.0, incoming: Vec::new(), rng_seed: 0,
        }
    }

    pub fn new_inhibitory(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: true, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.5, refractory_scale: 1, rate_fast: 0.0, rate_avg: 0.0, set_point: -1.0, w_ref_med: 0.0, w_ref_max: 0.0, has_dendrite: false, dend_potential: 0.0, last_burst: false, cell_type: CellType::Legacy, abs_refractory: 0, rel_refractory: 0, adapt: 0.0, adapt_inc: 0.0, adapt_decay: 1.0, slow_adapt: true, modulatory: false, mod_level: 0.0, mod_base: 0.0, mod_seen: 0, tonic: 0.0, dend_floor: false, next_prune: 0, prune_tau: 0.0, prune_min: 0.0, reset_min: 0.0, own_on: false, own_target: 0.0, own_n: 0, own_t0: 0, own_homeo: false, pruned_n: 0, did_prune: false, need: 0.0, need_rate: 0.0, mod_decay: 0.98, elig_idx: Vec::new(), elig_in: Vec::new(), ltd_mult: 1.0, ltd_mult_tick: 0, spont_scale: 1.0, incoming: Vec::new(), rng_seed: 0,
        }
    }

    /// 기억/해마용: 느린 감쇠 + 2배 불응기
    pub fn new_slow_decay(id: NeuronId, x: f32, y: f32, inhibitory: bool) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory, skip_refractory: false,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.75, threshold_scale: 1.0, self_threshold: 0.5, refractory_scale: 2, rate_fast: 0.0, rate_avg: 0.0, set_point: -1.0, w_ref_med: 0.0, w_ref_max: 0.0, has_dendrite: false, dend_potential: 0.0, last_burst: false, cell_type: CellType::Legacy, abs_refractory: 0, rel_refractory: 0, adapt: 0.0, adapt_inc: 0.0, adapt_decay: 1.0, slow_adapt: true, modulatory: false, mod_level: 0.0, mod_base: 0.0, mod_seen: 0, tonic: 0.0, dend_floor: false, next_prune: 0, prune_tau: 0.0, prune_min: 0.0, reset_min: 0.0, own_on: false, own_target: 0.0, own_n: 0, own_t0: 0, own_homeo: false, pruned_n: 0, did_prune: false, need: 0.0, need_rate: 0.0, mod_decay: 0.98, elig_idx: Vec::new(), elig_in: Vec::new(), ltd_mult: 1.0, ltd_mult_tick: 0, spont_scale: 1.0, incoming: Vec::new(), rng_seed: 0,
        }
    }

    pub fn new_io(id: NeuronId, x: f32, y: f32) -> Self {
        Self {
            id, potential: 0.0, last_spike_tick: None,
            synapses: Vec::new(), x, y, inhibitory: false, skip_refractory: true,
            excitability: 1.0, fire_count_window: 0, decay_rate: 0.7, threshold_scale: 1.0, self_threshold: 0.5, refractory_scale: 1, rate_fast: 0.0, rate_avg: 0.0, set_point: -1.0, w_ref_med: 0.0, w_ref_max: 0.0, has_dendrite: false, dend_potential: 0.0, last_burst: false, cell_type: CellType::Legacy, abs_refractory: 0, rel_refractory: 0, adapt: 0.0, adapt_inc: 0.0, adapt_decay: 1.0, slow_adapt: true, modulatory: false, mod_level: 0.0, mod_base: 0.0, mod_seen: 0, tonic: 0.0, dend_floor: false, next_prune: 0, prune_tau: 0.0, prune_min: 0.0, reset_min: 0.0, own_on: false, own_target: 0.0, own_n: 0, own_t0: 0, own_homeo: false, pruned_n: 0, did_prune: false, need: 0.0, need_rate: 0.0, mod_decay: 0.98, elig_idx: Vec::new(), elig_in: Vec::new(), ltd_mult: 1.0, ltd_mult_tick: 0, spont_scale: 1.0, incoming: Vec::new(), rng_seed: 0,
        }
    }

    #[inline]
    pub fn decay(&mut self) {
        if self.need_rate > 0.0 {
            self.need = (self.need + self.need_rate).min(1.0);
            self.potential = self.potential * self.decay_rate + self.tonic * self.need;
        } else {
            self.potential = self.potential * self.decay_rate + self.tonic;
        }
        self.rate_fast *= 0.99;
        self.rate_avg *= 0.9998;
        self.mod_level *= self.mod_decay;
        // 기준선: 조절 신호를 받기 시작한 뒤 처음 MOD_BASE_TICKS 틱 동안의 평균으로 정하고, 그 뒤에는 바꾸지 않는다.
        // 계속 따라가게 하면 보상이 잦을 때는 기준선이 올라가 평소 상태가 "아래"로 읽히고(무관한 입력이 처벌받음),
        // 도파민이 자주 멈출 때는 내려가 평소 상태가 "위"로 읽힌다. 실제 수용체도 감도가 정해져 있어서
        // 평소 농도보다 높을 때와 낮을 때를 고정된 기준으로 구분한다
        // 처음 MOD_BASE_SKIP 틱은 평균에 넣지 않는다: 조절 수준이 0 에서 평소 수준까지 차오르는 구간이라, 넣으면 기준선이 실제보다 낮게 잡힌다
        if self.mod_seen < MOD_BASE_SKIP + MOD_BASE_TICKS && (self.mod_level > 0.0 || self.mod_seen > 0) {
            self.mod_seen += 1;
            if self.mod_seen > MOD_BASE_SKIP {
                self.mod_base += (self.mod_level - self.mod_base) / (self.mod_seen - MOD_BASE_SKIP) as f64;
            }
        }
        self.adapt *= self.adapt_decay;
        self.dend_potential *= 0.85;
        // 시냅스의 감쇠는 여기서 돌지 않는다: 시냅스가 쓰일 때 sync(now) 로 한꺼번에 적용된다
    }

    /// 수상돌기 구획으로 들어온 입력. 세포체 전위에는 직접 더해지지 않는다
    #[inline]
    pub fn receive_dend(&mut self, amount: f64) {
        self.dend_potential = (self.dend_potential + amount).max(-1.0);
    }

    /// 수상돌기가 세포체에 주는 기여.
    ///   고원 전위 아래: 거의 전달되지 않음 (수상돌기 입력만으로는 발화가 어렵다)
    ///   고원 전위(DEND_THRESHOLD) 이상: 세포체에 큰 탈분극을 준다 → 세포체 입력과 겹치면 발화가 크게 늘어난다
    #[inline]
    pub fn dendritic_drive(&self) -> f64 {
        if !self.has_dendrite { return 0.0; }
        // 스스로 발화하는 뉴런(도파민): 수상돌기의 (보상 입력 − 예측 억제)에 비례해 발화가 빨라지거나 느려진다.
        //   보상 > 예측: 연달아 발화.  보상 = 예측: 평소대로.  예측 > 보상(오지 않은 보상 포함): 발화가 멈춘다
        if self.tonic > 0.0 {
            let d = self.dend_potential.clamp(-1.0, 1.0) * 0.5;
            return if self.dend_floor { d.max(0.0) } else { d };
        }
        if self.dend_potential >= DEND_THRESHOLD { DEND_PLATEAU } else { (self.dend_potential * DEND_LEAK).max(0.0) }
    }

    /// 세포 종류 지정: 흥분/억제 구분, 불응기, 적응 정도를 종류에 맞게 설정한다.
    /// 수치의 근거 — 불응기: Pv 는 추체세포보다 최대 발화율이 몇 배 높다.
    /// 적응: 일정한 입력에서 추체세포는 발화율이 처음의 절반 안팎으로 떨어지고 Pv 는 거의 그대로다.
    pub fn set_type(&mut self, t: CellType) {
        self.cell_type = t;
        match t {
            CellType::Pyramidal => { self.inhibitory = false; self.abs_refractory = 4; self.rel_refractory = 10; self.adapt_inc = 0.15; self.adapt_decay = 0.97; }
            CellType::Pv => { self.inhibitory = true; self.abs_refractory = 1; self.rel_refractory = 2; self.adapt_inc = 0.0; self.adapt_decay = 1.0; self.slow_adapt = false; }
            CellType::Sst => { self.inhibitory = true; self.abs_refractory = 3; self.rel_refractory = 6; self.adapt_inc = 0.08; self.adapt_decay = 0.97; }
            CellType::Vip => { self.inhibitory = true; self.abs_refractory = 3; self.rel_refractory = 6; self.adapt_inc = 0.02; self.adapt_decay = 0.97; }
            CellType::Modulatory => { self.inhibitory = false; self.modulatory = true; }
            CellType::Value => { self.inhibitory = false; self.abs_refractory = 2; self.rel_refractory = 4; self.adapt_inc = 0.0; self.adapt_decay = 1.0; self.slow_adapt = false; }
            CellType::Predictor => { self.inhibitory = true; self.abs_refractory = 2; self.rel_refractory = 4; self.adapt_inc = 0.0; self.adapt_decay = 1.0; self.slow_adapt = false; }
            CellType::Legacy => {}
        }
    }

    /// 조절 신호 수신: 전위는 그대로 두고 조절 수준만 올린다
    #[inline]
    pub fn receive_mod(&mut self, amount: f64) {
        self.mod_level += amount;
    }

    #[inline]
    pub fn receive(&mut self, amount: f64) {
        // 몸의 상태 뉴런: 억제 입력은 전위가 아니라 결핍 정도를 낮춘다 (포만 신호)
        if self.need_rate > 0.0 && amount < 0.0 { self.need = (self.need + amount).max(0.0); return; }
        self.potential += amount;
        // 스스로 발화하는 뉴런은 억제를 받아도 전위가 0 아래로 내려가지 않는다 (억제의 역전 전위).
        // 그래서 억제가 끝나면 평소 발화 간격 한 번 만큼만 더 쉬고 다시 발화한다
        if self.tonic > 0.0 && self.potential < 0.0 { self.potential = 0.0; }
    }

    #[inline]
    pub fn try_fire(&mut self, tick: u64, threshold: f64, noise_range: f64, stimulated: bool) -> Option<Vec<(NeuronId, f64)>> {
        // 불응기 체크 (입출력 뉴런은 스킵)
        if !self.skip_refractory {
            if let Some(last) = self.last_spike_tick {
                let elapsed = tick.saturating_sub(last);
                let abs_ref = (if self.abs_refractory > 0 { self.abs_refractory } else if self.inhibitory { ABSOLUTE_REFRACTORY_INH } else { ABSOLUTE_REFRACTORY }) * self.refractory_scale;
                let rel_ref = (if self.rel_refractory > 0 { self.rel_refractory } else if self.inhibitory { RELATIVE_REFRACTORY_INH } else { RELATIVE_REFRACTORY }) * self.refractory_scale;
                if elapsed < abs_ref {
                    return None;
                }
                if elapsed < rel_ref {
                    let base = self.potential + self.dendritic_drive();
                    let bar = ((self.self_threshold + self.adapt) * self.threshold_scale / self.excitability) * 2.0;
                    // 잡음은 −폭 이상 +폭 미만이다. 가장 큰 잡음을 더해도 못 넘으면 난수를 뽑지 않는다 (판정은 어떤 잡음 값에서도 같다)
                    if base + noise_range < bar { return None; }
                    let noise = if noise_range > 0.0 {
                        let u = if self.rng_seed != 0 { keyed_f64(self.rng_seed, 1, self.id as u64, 0, tick) } else { rand::random::<f64>() };
                        (u * 2.0 - 1.0) * noise_range
                    } else { 0.0 };
                    if base + noise < bar {
                        return None;
                    }
                }
            }
        }
        let _ = threshold; // 전역 threshold 미사용 (뉴런별 self_threshold로 대체)

        let base = self.potential + self.dendritic_drive();
        let adj_threshold = (self.self_threshold + self.adapt) * self.threshold_scale / self.excitability;
        // 잡음은 −폭 이상 +폭 미만이다. 가장 큰 잡음을 더해도 문턱에 못 미치면 난수를 뽑지 않는다 (문턱으로는 발화하지 않는다. 자발 발화는 아래에서 그대로 본다)
        let over_threshold = if noise_range <= 0.0 { base >= adj_threshold } else if base + noise_range <= adj_threshold { false } else {
            let noise = if noise_range > 0.0 {
                let u = if self.rng_seed != 0 { keyed_f64(self.rng_seed, 2, self.id as u64, 0, tick) } else { rand::random::<f64>() };
                (u * 2.0 - 1.0) * noise_range
            } else { 0.0 };
            base + noise >= adj_threshold
        };
        let should_fire = over_threshold
            || (!self.skip_refractory && self.spont_scale > 0.0
                && { let p = (if stimulated { 0.000001 } else { 0.00001 }) * self.spont_scale;
                     if self.rng_seed != 0 { keyed_f64(self.rng_seed, 3, self.id as u64, 0, tick) < p } else { rand::random_bool(p) } });

        // 신호 받은 상태인지 (potential > 0: 입력이 있었다는 뜻)
        let received_signal = self.potential > 0.01;

        if should_fire {
            self.potential = 0.0;
            // 수상돌기 고원 전위와 겹친 발화는 버스트: 이번 발화의 전달량이 커진다
            // (스스로 발화하는 뉴런은 제외: 도파민 뉴런의 급증은 발화 횟수로 나타나고, 한 번의 전달량은 그대로다)
            self.last_burst = self.has_dendrite && self.tonic == 0.0 && self.dend_potential >= DEND_THRESHOLD;
            self.last_spike_tick = Some(tick);
            self.fire_count_window += 1;
            self.rate_fast += 0.01;
            self.adapt += self.adapt_inc;
            self.rate_avg += 0.0002;
            // 자기 목표를 재는 중이면 이 발화를 센다 (own_on 이 거짓이면 아무것도 하지 않는다)
            if self.own_on && self.own_target == 0.0 {
                if self.own_n == 0 { self.own_t0 = tick; }
                self.own_n += 1;
                if self.own_n >= OWN_COUNT + 1 {
                    let dt = tick.saturating_sub(self.own_t0).max(1) as f64;
                    self.own_target = (OWN_COUNT as f64 / dt).max(1e-5 * self.spont_scale);
                }
            }
            // 발화 시: self_threshold +0.001 (최대 1.0)
            if self.slow_adapt { self.self_threshold = (self.self_threshold + 0.001).min(1.0); }
            // 신호를 보내기 전에 먼저 자기 시냅스의 가지치기를 본다 (기한이 지난 연결은 이번 신호부터 전달하지 않는다)
            if self.prune_tau > 0.0 && tick >= self.next_prune { self.prune_own(tick); }
            let sign = if self.inhibitory { -1.0 } else { 1.0 };
            let burst = if self.last_burst { BURST_GAIN } else { 1.0 };
            let (rng_seed, my_id) = (self.rng_seed, self.id as u64);
            let deliveries: Vec<_> = self.synapses.iter_mut().enumerate()
                .map(|(k_syn, s)| {
                    s.sync(tick);
                    // 지연 시냅스: 지금은 전달하지 않고 흔적만 올린다 (전류는 매 틱 따로 전달된다)
                    if s.slow_rate > 0.0 {
                        s.slow1 += 1.0;
                        s.last_used_tick = tick;
                        if !s.slow_fast { return (s.target, 0.0); }
                    }
                    // 확률적 전달: 실패하면 이번에는 아무것도 보내지 않는다 (자원도 쓰지 않는다)
                    // 확률 전달의 난수: 결정적 난수면 열쇠에 대상이 아니라 시냅스 번호를 쓴다 (같은 대상으로 가는 칸이 여럿일 수 있다)
                    if s.p_rel > 0.0 && (if rng_seed != 0 { keyed_f64(rng_seed, 4, my_id, k_syn as u64, tick) } else { rand::random::<f64>() }) >= s.p_rel {
                        s.last_used_tick = tick;
                        return (s.target, 0.0);
                    }
                    let gain = if s.p_rel > 0.0 { 1.0 / s.p_rel } else { 1.0 };
                    let w = s.effective_weight() * sign * burst * gain;
                    s.fire_fatigue();
                    s.last_used_tick = tick;
                    (s.target, w)
                })
                .collect();
            Some(deliveries)
        } else {
            // 미발화 + 신호 받음: self_threshold -0.001 (최소 0.5)
            if received_signal && self.slow_adapt {
                self.self_threshold = (self.self_threshold - 0.001).max(0.5);
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

    /// 항상성 (뉴런별 목표): 지난 window 틱의 발화율을 자기 평소 발화율(set_point)과 비교해
    /// 벗어난 비율만큼만 감응도를 조정한다. 원래 조용한 뉴런은 목표도 0 이라 건드리지 않는다.
    ///   eta: 한 번에 조정하는 최대 비율 (예: 0.02), alpha: 목표가 최근 발화율을 따라가는 속도 (예: 0.1)
    pub fn homeostasis_own(&mut self, window: f64, eta: f64, alpha: f64) {
        let recent = self.fire_count_window as f64 / window;
        self.fire_count_window = 0;
        // 굳힌 자기 목표가 있고 그것을 보게 돼 있으면: 따라가는 평소 수준 대신 그 목표와 견준다 (오차의 자름, 빠르기, 한계는 아래와 같다). set_point 는 건드리지 않는다
        if self.own_homeo && self.own_target > 0.0 {
            let dev = ((recent - self.own_target) / (self.own_target + 1e-4)).clamp(-1.0, 1.0);
            self.excitability = (self.excitability * (1.0 - eta * dev)).clamp(0.5, 1.5);
            return;
        }
        if self.set_point < 0.0 {
            self.set_point = recent; // 첫 실행: 지금 수준을 평소 수준으로 삼는다
            return;
        }
        if self.set_point > 0.0 || recent > 0.0 {
            let dev = ((recent - self.set_point) / (self.set_point + 1e-4)).clamp(-1.0, 1.0);
            self.excitability = (self.excitability * (1.0 - eta * dev)).clamp(0.5, 1.5);
        }
        self.set_point += (recent - self.set_point) * alpha;
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

    /// 자기 시냅스의 가지치기. 기한(마지막으로 함께 활동한 틱 + 수명)이 지난 연결은 접점 하나를 잃는다.
    /// 접점이 0개가 된 연결은 죽은 것으로 표시만 한다(가중치 0, 접점 0) — 자리를 비우는 일은 받는 쪽 목록도 고쳐야 해서 네트워크가 한다
    pub fn prune_own(&mut self, now: u64) {
        let (tau, min_w) = (self.prune_tau, self.prune_min.max(1e-12));
        let mut removed = 0u32;
        for s in self.synapses.iter_mut() {
            if s.slow_rate > 0.0 || s.count == 0 { continue; }
            let life = (tau * (s.weight / min_w).max(1.0).ln()).max(20_000.0) as u64;
            if s.last_co.max(s.born) + life > now { continue; }
            removed += 1;
            if s.count > 1 {
                s.weight -= s.weight / s.count as f64;
                s.count -= 1;
                s.last_co = now; // 남은 접점은 지금부터 다시 기한을 센다
            } else {
                s.weight = 0.0;
                s.count = 0;
                s.elig = 0.0;
                s.w_short = 0.0; // 일시 성분도 지운다: 남겨 두면 죽은 연결이 이번 틱에 일시 성분만큼 한 번 더 전달한다
            }
        }
        self.pruned_n += removed;
        self.did_prune = true;
        self.next_prune = now + 1000;
    }

    /// 자기가 내보내는 시냅스 가중치의 중앙값 (새 시냅스의 초기 가중치로 쓴다. 시냅스가 없으면 SPROUT_WEIGHT)
    pub fn median_out_weight(&self) -> f64 {
        if self.synapses.is_empty() { return SPROUT_WEIGHT; }
        let mut w: Vec<f64> = self.synapses.iter().filter(|s| s.slow_rate == 0.0).map(|s| s.weight).collect();
        if w.is_empty() { return SPROUT_WEIGHT; }
        let mid = w.len() / 2;
        *w.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
    }

}
