//! 가소성 규칙 (STDP / iSTDP / R-STDP eligibility).
//! Network(포유류 구조)와 fly 크레이트(초파리 커넥톰)가 같은 규칙을 쓰도록 분리한 모듈.
//! crate::neuron 에만 의존한다.

use crate::neuron::{Neuron, NeuronId};
use std::collections::HashMap;

// ── 굳는 양의 관찰 (읽기만) ─────────────────────────────────────────────────────────
// 도파민으로 굳을 때(조절 수준이 기준선을 벗어난 받는 뉴런으로 흔적이 가중치로 넘어갈 때), 표시해 둔 받는 뉴런(예: 출력 뉴런)으로 가는
// 시냅스의 '굳은 양 ÷ 굳은 뒤의 가중치'와 가지치기 기한을 다시 받은 쌍을 모아 둔다. OBS_ON 이 꺼져 있으면(기본, 초파리도) 아무것도 하지 않고,
// 켜져 있어도 가중치·흔적·기한을 바꾸지 않는다 (동작에 닿지 않는다).
pub static OBS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// 관찰 묶음: 받는 뉴런 표시(뉴런 번호 → 참이면 센다), 보내는 뉴런 번호의 위 끝(그 아래만 센다),
/// 강화 쪽(l > 0)에서 굳은 양이 양수였던 사건의 '굳은 양 ÷ 가중치' 로그 눈금 히스토그램(10^-7 ~ 10, 한 자리에 열 칸), 사건 수들, 기한을 다시 받은 쌍
pub struct ConsolObs {
    pub mark: Vec<bool>,
    pub pre_end: usize,
    pub hist: [u64; 80],
    pub pos_events: u64,   // l > 0 이고 굳은 양 > 0
    pub nonpos_events: u64, // l > 0 이고 굳은 양 <= 0 (흔적이 0 이하)
    pub down_events: u64,  // l < 0 (조절 수준이 기준선 아래)
    pub over1: u64,        // 굳은 양 > 가중치의 1%
    pub max_ratio: f64,
    pub reset_events: u64,
    pub reset_pairs: std::collections::HashSet<(u32, u32)>,
    /// 크기의 합 (사건의 수가 아니라 양): 기준선 위에서 오른 양(굳은 양 > 0) / 기준선 위에서 내린 양(굳은 양 < 0) / 기준선 아래에서 내린 양(굳은 양 > 0 이 깎임) / 기준선 아래에서 오른 양
    pub sum_up: f64,
    pub sum_up_neg: f64,
    pub sum_down: f64,
    pub sum_down_neg: f64,
    /// 쌍마다의 곁 표 (보내는 뉴런, 받는 뉴런) → 쌓인 값. 구간을 넘어 이어진다 (obs_take 가 비우지 않는다)
    pub pairs: HashMap<(u32, u32), PairAcc>,
    /// 사건 도중에 '같은 쌍인데 생긴 틱이 달라진 것'(떨어졌다가 다시 생긴 쌍)을 보면 옛 것을 여기에 옮겨 둔다
    pub dropped: Vec<PairAcc>,
}
/// 쌍 하나의 쌓인 값 (관찰용): 생긴 틱, 마지막으로 본 틱과 그때의 가중치, 생긴 뒤(또는 기한을 마지막으로 받은 뒤) 강화 쪽으로 굳은 양의 합,
/// 그 합 ÷ 그때의 가중치가 가장 컸던 값, 기한을 다시 받은 횟수, 받는 뉴런
#[derive(Clone, Copy)]
pub struct PairAcc { pub born: u64, pub last_tick: u64, pub last_w: f64, pub acc: f64, pub peak: f64, pub resets: u32, pub post: u32 }
static OBS: std::sync::Mutex<Option<ConsolObs>> = std::sync::Mutex::new(None);
// 표시된 받는 뉴런 번호의 범위 [OBS_LO, OBS_HI): 잠그기 전에 이 범위 밖의 시냅스를 걸러 낸다 (빠르게)
static OBS_LO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static OBS_HI: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// 관찰을 켠다: mark[받는 뉴런 번호] 가 참인 시냅스만, 보내는 뉴런 번호가 pre_end 아래인 것만 센다
pub fn obs_start(mark: Vec<bool>, pre_end: usize) {
    let lo = mark.iter().position(|&m| m).unwrap_or(0); let hi = mark.iter().rposition(|&m| m).map_or(0, |i| i + 1);
    OBS_LO.store(lo, std::sync::atomic::Ordering::Relaxed); OBS_HI.store(hi, std::sync::atomic::Ordering::Relaxed);
    *OBS.lock().unwrap() = Some(ConsolObs { mark, pre_end, hist: [0; 80], pos_events: 0, nonpos_events: 0, down_events: 0, over1: 0, max_ratio: 0.0, reset_events: 0, reset_pairs: std::collections::HashSet::new(), sum_up: 0.0, sum_up_neg: 0.0, sum_down: 0.0, sum_down_neg: 0.0, pairs: HashMap::new(), dropped: Vec::new() });
    OBS_ON.store(true, std::sync::atomic::Ordering::Relaxed);
}
/// 지금까지 모은 것을 꺼내고 비운다 (관찰은 켜진 채로 둔다). 켜져 있지 않으면 None
pub fn obs_take() -> Option<ConsolObs> {
    let mut g = OBS.lock().unwrap();
    let o = g.as_mut()?;
    // 쌍마다의 곁 표는 구간을 넘어 이어져야 하므로 새 묶음으로 옮겨 두고, 꺼내는 쪽에는 그 사본을 준다
    let pairs = std::mem::take(&mut o.pairs);
    let fresh = ConsolObs { mark: o.mark.clone(), pre_end: o.pre_end, hist: [0; 80], pos_events: 0, nonpos_events: 0, down_events: 0, over1: 0, max_ratio: 0.0, reset_events: 0, reset_pairs: std::collections::HashSet::new(),
        sum_up: 0.0, sum_up_neg: 0.0, sum_down: 0.0, sum_down_neg: 0.0, pairs: pairs.clone(), dropped: Vec::new() };
    let mut out = std::mem::replace(o, fresh);
    out.pairs = pairs;
    // 사건 도중에 옮겨 둔 '떨어진 쌍'은 구간 끝의 obs_sync_pairs 가 가져가도록 남겨 둔다
    o.dropped = std::mem::take(&mut out.dropped);
    Some(out)
}
/// 구간 끝에: 지금 있는 쌍의 목록(보내는, 받는, 생긴 틱, 가중치)을 받아, 곁 표에서 없어진 쌍(또는 생긴 틱이 달라진 쌍)을 떨어진 것으로 돌려주고
/// 표를 지금의 쌍에 맞춘다(처음 보는 쌍은 쌓인 양 0 으로 넣는다). 돌려주는 것: (떨어진 쌍들, 남은 쌍들 — 가중치는 지금 값으로)
pub fn obs_sync_pairs(current: &[(u32, u32, u64, f64)], now: u64) -> (Vec<PairAcc>, Vec<PairAcc>) {
    let mut g = OBS.lock().unwrap();
    let Some(o) = g.as_mut() else { return (Vec::new(), Vec::new()) };
    let mut dropped = std::mem::take(&mut o.dropped);
    let live: HashMap<(u32, u32), (u64, f64)> = current.iter().map(|&(a, b, born, w)| ((a, b), (born, w))).collect();
    let keys: Vec<(u32, u32)> = o.pairs.keys().cloned().collect();
    for k in keys {
        let gone = match live.get(&k) { None => true, Some(&(born, _)) => born != o.pairs[&k].born };
        if gone { dropped.push(o.pairs.remove(&k).unwrap()); }
    }
    let mut alive = Vec::with_capacity(current.len());
    for &(a, b, born, w) in current {
        let e = o.pairs.entry((a, b)).or_insert(PairAcc { born, last_tick: now, last_w: w, acc: 0.0, peak: 0.0, resets: 0, post: b });
        e.last_tick = now; e.last_w = w;
        alive.push(*e);
    }
    (dropped, alive)
}
fn obs_consol(pre: usize, post: usize, l: f64, moved: f64, w: f64, reset: bool, born: u64, tick: u64) {
    if post < OBS_LO.load(std::sync::atomic::Ordering::Relaxed) || post >= OBS_HI.load(std::sync::atomic::Ordering::Relaxed) { return; }
    let mut g = OBS.lock().unwrap();
    let Some(o) = g.as_mut() else { return };
    if pre >= o.pre_end || !o.mark.get(post).copied().unwrap_or(false) { return; }
    // 쌍의 곁 표: 처음 보거나, 생긴 틱이 달라졌으면(떨어졌다가 다시 생긴 쌍) 새로 연다
    let key = (pre as u32, post as u32);
    let stale = match o.pairs.get(&key) { Some(e) => e.born != born, None => false };
    if stale { let old = o.pairs.remove(&key).unwrap(); o.dropped.push(old); }
    let e = o.pairs.entry(key).or_insert(PairAcc { born, last_tick: tick, last_w: w, acc: 0.0, peak: 0.0, resets: 0, post: post as u32 });
    e.last_tick = tick; e.last_w = w;
    if l < 0.0 { o.down_events += 1; if moved > 0.0 { o.sum_down += moved; } else { o.sum_down_neg += -moved; } return; }
    if moved <= 0.0 { o.nonpos_events += 1; o.sum_up_neg += -moved; return; }
    o.pos_events += 1;
    o.sum_up += moved;
    e.acc += moved;
    let pr = e.acc / w.max(1e-300);
    if pr > e.peak { e.peak = pr; }
    if reset { e.resets += 1; e.acc = 0.0; }
    let r = moved / w.max(1e-300);
    if moved > 0.01 * w { o.over1 += 1; }
    if r > o.max_ratio { o.max_ratio = r; }
    let b = ((r.log10() + 7.0) * 10.0).floor().clamp(0.0, 79.0) as usize;
    o.hist[b] += 1;
    if reset { o.reset_events += 1; o.reset_pairs.insert((pre as u32, post as u32)); }
}

/// 실험용 스위치. 기본값은 기존 동작과 동일.
#[derive(Debug, Clone)]
pub struct PlasticityOptions {
    /// 억제성 pre 시냅스에도 일반 STDP / R-STDP eligibility 를 적용 (iSTDP 는 항상 적용)
    pub stdp_on_inhibitory: bool,
    /// LTD 크기 배수 (일반 STDP 와 R-STDP eligibility 의 LTD 항)
    pub ltd_scale: f64,
    /// 비지도 STDP(흥분성 일반 STDP) 크기 배수. 0 이면 끔. iSTDP 와 R-STDP 에는 영향 없음
    pub stdp_scale: f64,
    /// iSTDP(억제성 시냅스 전용 규칙) 크기 배수. 0 이면 끔
    pub istdp_scale: f64,
    /// iSTDP 를 목표 발화율 방식으로 (None = 기존 규칙).
    /// 억제 pre 가 발화할 때 post 의 최근 발화 흔적(0~1)이 이 값보다 크면 억제 강화, 작으면 약화.
    /// 기존 규칙은 post 가 발화하기만 하면 억제를 계속 키워서 활성 경로를 꺼버린다.
    pub istdp_target: Option<f64>,
    /// 흥분성 STDP / R-STDP 의 가중치 의존 방식
    pub bound: Bound,
    /// post 발화 하나당 LTD 를 한 번만 적용 (그 뒤 첫 pre 발화에만).
    /// 기존에는 post 발화 뒤 50틱 안의 모든 pre 발화가 LTD 로 잡혀서, pre 가 계속 발화하는
    /// 지속 입력에서는 LTD 횟수가 LTP 의 5~10배가 된다.
    pub ltd_once_per_post: bool,
    /// BCM 메타가소성 사용 (기본 켬). post 의 누적 발화 수로 LTP/LTD 배수를 조절한다
    pub bcm: bool,
    /// 발화율 기준 BCM 의 목표 발화율 (틱당 발화 확률). Some 이면 누적 횟수 대신 최근 발화율을 쓴다.
    /// 기존 기준(누적 25회)은 실제로 쓰이는 경로의 뉴런이 자극 두세 번 만에 넘어서 LTD 만 남는다.
    pub bcm_rate_target: Option<f64>,
    /// 발화율 기준 BCM 의 목표를 고정값 대신 "발화 중인 뉴런들의 발화율 중앙값 × 이 배수"로 (예: 0.9)
    pub bcm_rate_relative: Option<f64>,
    /// 발화율 기준 BCM 의 목표를 뉴런별 지역 정보로: "자기에게 입력을 주는 뉴런들 중
    /// 발화 중인 것들의 발화율 중앙값 × 이 배수" (예: 0.9). 전역 통계를 쓰지 않는다
    pub bcm_local: Option<f64>,
    /// 교과서 BCM 의 목표 발화율 ρ0 (틱당 발화 확률). Some 이면 LTD × (장기 평균 ÷ ρ0)², LTP 는 그대로.
    /// 뉴런 자신의 장기 평균만 쓰므로 지역 정보이고, 제곱이라 강화가 쌓일수록 제동이 세진다
    pub bcm_classic: Option<f64>,
    /// 교과서 BCM 의 목표를 고정 상수 대신 뉴런별로: "자기 입력 뉴런들(활동 중인 것)의
    /// 장기 평균 발화율 중앙값 × 이 배수". LTD × (자기 장기 평균 ÷ 그 목표)²
    pub bcm_classic_local: Option<f64>,
    /// iSTDP(기존 규칙)의 가중치 의존 방식. Soft = 가우시안, Hard = 일정
    pub istdp_bound: Bound,
    /// iSTDP soft bound 의 가우시안 폭 (기존 0.4). 넓히면 낮은 가중치에서도 약화가 걸린다
    pub istdp_sigma: f64,
    /// iSTDP soft bound 의 중심값(0.2, 1.0)과 폭에 곱하는 배수 (기본 1.0).
    /// 가중치가 대부분 낮은 구간에 몰린 배선에서는 곡선을 그 구간으로 옮겨야 약화가 걸린다
    pub istdp_bound_scale: f64,
    /// 흥분성 STDP / R-STDP soft bound 의 중심값(0.2, 1.0)과 폭(0.4)에 곱하는 배수 (기본 1.0).
    /// iSTDP 의 istdp_bound_scale 과 한 쌍으로 같은 값을 쓴다
    pub bound_scale: f64,
    /// 변화량과 soft bound 를 전역 배수 대신 뉴런별 지역 값으로 (refresh_local_scales 필요).
    /// 켜면 stdp_scale / istdp_scale 은 1.0 으로 두고, bound_scale 은 값이 없는 뉴런의 대체값으로만 쓰인다
    pub local_scale: bool,
    /// 조절 신호(도파민 등)에 의한 가소성의 크기. 0 이면 끔
    pub mod_eta: f64,
    /// 시냅스 활동 → 조절 신호 순서일 때의 부호 (초파리 버섯체: −1 약화, 포유류 D1 경로: +1 강화)
    pub mod_forward_sign: f64,
    /// 조절 신호 → 시냅스 활동 순서일 때의 부호 (초파리 버섯체: +1 강화)
    pub mod_backward_sign: f64,
    /// 조절 신호가 STDP 를 조절하는 정도 (0 = 끔). 조절 수준 m 인 뉴런에서 LTP ×(1 + gain·m), LTD ÷(1 + gain·m)
    pub mod_stdp_gain: f64,
    /// 조절 신호가 있어야만 STDP 가 실제 가중치 변화로 이어지는 방식 (3요소 규칙). Some(gain) 이면 켬.
    ///   STDP·iSTDP 는 가중치를 바꾸지 않고 시냅스에 흔적(elig)만 남긴다.
    ///   받는 뉴런에 조절 신호가 있는 동안 매 틱 흔적의 (gain × 조절 수준) 비율(최대 전부)이 가중치로 넘어간다.
    ///   조절 신호가 없으면 gate_base 비율만 넘어가고 나머지 흔적은 사라진다
    pub mod_gate: Option<f64>,
    /// 조절 신호 방식에서, 조절 신호 없이도 평소에 가중치로 넘어가는 흔적의 비율 (STDP 사건마다). 0 이면 조절 신호가 있을 때만 변한다
    pub gate_base: f64,
    /// 조절 신호를 기준선 대비 변화로 읽는다 (mod_gate 와 함께 사용).
    ///   받는 뉴런의 조절 수준이 자기 기준선(mod_base)보다 높으면 쌓인 흔적이 그대로 가중치로,
    ///   낮으면(도파민 뉴런의 정지) 부호가 뒤집혀서 넘어간다. 기준선 근처(±MOD_DEADBAND)에서는 넘어가지 않는다.
    ///   이 방식에서는 조절 뉴런의 출력 시냅스와 조절 뉴런으로 들어오는 시냅스는 타고난 배선으로 보고 학습하지 않는다
    pub mod_baseline: bool,
    /// 보상 예측 뉴런(Predictor)의 입력 시냅스 학습 크기 (기준선 방식에서만, 0 = 끔).
    ///   매 틱 가중치 += pred_eta × (기준선 대비 변화) × (그 시냅스의 지연 흔적 slow2).
    ///   받는 뉴런이 발화했는지는 보지 않는다 — 처음에는 예측 뉴런이 전혀 발화하지 않기 때문 (초파리 버섯체의 KC→MBON 규칙과 같은 형태)
    pub pred_eta: f64,
    /// 일시적인 변화 방식 (mod_gate 와 함께 사용): STDP 의 변화는 시냅스의 일시적인 부분(w_short)에 바로 반영되어 전달에 쓰이고,
    /// 틱마다 이 배수로 줄어 0 으로 돌아간다. 오래가는 가중치는 조절 신호가 흔적을 넘길 때만 변한다.
    /// 이 방식에서는 gate_base(평소에도 조금씩 영구적으로 넘어가는 비율)를 쓰지 않는다
    pub short_decay: Option<f64>,
    /// 가치 뉴런 입력 시냅스의 학습 크기 (0 이면 pred_eta 와 같게)
    pub val_eta: f64,
}

/// 조절 수준이 기준선에서 이 비율 이상 벗어나야 신호로 본다 (기준선 발화의 흔들림을 신호로 읽지 않기 위한 값)
pub const MOD_DEADBAND: f64 = 0.25;
/// 교과서 BCM 의 LTD 배수를 다시 계산하는 간격 (틱)
pub const BCM_CACHE_TICKS: u64 = 100;
/// 예측·가치 뉴런의 입력 학습에서 무시하는 벗어남의 크기
pub const PRED_DEADBAND: f64 = 0.1;
pub const VALUE_UP: f64 = 1.0;
pub const VALUE_DOWN: f64 = 0.6;

/// STDP 의 변화량 적용: 기존 방식이면 가중치를 바로 바꾸고, 조절 신호 방식이면 흔적에만 더한다
#[inline]
fn apply_dw(syn: &mut crate::synapse::Synapse, dw: f64, gate_base: Option<f64>) -> bool {
    match gate_base {
        None => { syn.weight = (syn.weight + dw).clamp(0.0, 1.0); false }
        Some(base) => {
            // 반환값: 흔적이 0 이었다가 생겼는지 (그 뉴런의 "흔적이 있는 시냅스 목록"에 새로 올려야 하는지)
            let was_empty = syn.elig == 0.0;
            syn.elig += dw;
            // 일시적인 변화 방식: 지금 당장의 전달에는 반영하되(가중치 + 일시적 부분이 0~1 을 벗어나지 않게), 굳히지는 않는다
            if syn.short_decay > 0.0 {
                syn.w_short = (syn.weight + syn.w_short + dw).clamp(0.0, 1.0) - syn.weight;
                return syn.elig != 0.0 && (!syn.lp || !syn.lq);
            }
            // 평소에도 쌓인 흔적의 작은 비율이 가중치로 넘어간다 (넘어간 만큼 흔적은 줄어든다).
            // 쌓인 양에 비례하므로, 가끔 한 번 짝지어진 시냅스는 거의 변하지 않고 계속 반복된 시냅스만 조금씩 변한다
            if base > 0.0 {
                let moved = base * syn.elig;
                syn.weight = (syn.weight + moved).clamp(0.0, 1.0);
                syn.elig -= moved;
            }
            let _ = was_empty;
            syn.elig != 0.0 && (!syn.lp || !syn.lq)
        }
    }
}

/// iSTDP 의 가중치 의존 배수. center = 기존 중심값 (강화 0.2, 약화 1.0).
/// istdp_bound_scale 로 중심값과 폭을 같은 비율로 줄여 곡선을 가중치 분포에 맞춘다
#[inline]
fn istdp_factor(w: f64, center: f64, scale: f64, opt: &PlasticityOptions) -> f64 {
    match opt.istdp_bound {
        Bound::Soft => {
            let diff = w - center * scale;
            let sigma = opt.istdp_sigma * scale;
            (-(diff * diff) / (2.0 * sigma * sigma)).exp()
        }
        _ => 1.0,
    }
}

#[inline]
fn bcm_factor(fire_count: u32, target: f64) -> f64 {
    if target.is_infinite() { 1.0 } else { target / (fire_count as f64 + target) }
}

/// 가중치에 따라 LTP/LTD 크기가 어떻게 달라지는가
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// 기존: 가우시안 soft bound (LTP 는 w=0.2, LTD 는 w=1.0 근처에서 최대)
    Soft,
    /// 가중치와 무관하게 일정, 0 과 1 에서만 자름
    Hard,
    /// 현재 가중치에 비례 (w=0.2 에서 기존 최대값과 같은 크기)
    Proportional,
}

#[inline]
fn ltp_factor(w: f64, scale: f64, opt: &PlasticityOptions) -> f64 {
    match opt.bound {
        Bound::Soft => { let (d, sg) = (w - 0.2 * scale, 0.4 * scale); (-(d * d) / (2.0 * sg * sg)).exp() }
        Bound::Hard => 1.0,
        Bound::Proportional => w / 0.2,
    }
}

#[inline]
fn ltd_factor(w: f64, scale: f64, opt: &PlasticityOptions) -> f64 {
    match opt.bound {
        Bound::Soft => { let (d, sg) = (w - 1.0 * scale, 0.4 * scale); (-(d * d) / (2.0 * sg * sg)).exp() }
        Bound::Hard => 1.0,
        Bound::Proportional => w / 0.2,
    }
}

impl Default for PlasticityOptions {
    fn default() -> Self {
        Self { stdp_on_inhibitory: true, ltd_scale: 1.0, stdp_scale: 1.0, istdp_scale: 1.0, istdp_target: None, bound: Bound::Soft, ltd_once_per_post: false, bcm: true, bcm_rate_target: None, bcm_rate_relative: None, bcm_local: None, bcm_classic: None, bcm_classic_local: None, istdp_bound: Bound::Soft, istdp_sigma: 0.4, istdp_bound_scale: 1.0, bound_scale: 1.0, local_scale: false, mod_eta: 0.0, mod_forward_sign: -1.0, mod_backward_sign: 1.0, mod_stdp_gain: 0.0, mod_gate: None, gate_base: 0.0, mod_baseline: false, pred_eta: 0.0, short_decay: None, val_eta: 0.0 }
    }
}

pub struct EligibilityTrace {
    pub neuron_idx: usize,
    pub synapse_idx: usize,
    pub dw: f64,
}

/// 한 틱의 STDP + iSTDP. tick_spikes = 이번 틱에 발화한 뉴런.
/// 각 뉴런의 last_spike_tick 은 이번 틱 발화까지 반영된 상태여야 한다.
/// 조절 신호 방식의 흔적 넘기기: 조절 수준이 기준선에서 벗어난 뉴런의 입력 시냅스에서, 남아 있는 흔적을 가중치 변화로 바꾼다.
/// 발화와 무관하게 매 틱 돈다 (stdp_step 이 부른다)
fn mod_transfer(neurons: &mut [Neuron], global_tick: u64, opt: &PlasticityOptions) {
    // 조절 신호 방식: 조절 수준이 남아 있는 뉴런의 입력 시냅스에서, 남아 있는 흔적을 가중치 변화로 바꾼다
    if let Some(gain) = opt.mod_gate {
        // level = 흔적을 넘기는 세기와 부호. 기준선 방식이면 (조절 수준 ÷ 자기 기준선 − 1), 아니면 조절 수준 그대로
        let active: Vec<(usize, f64)> = neurons.iter().enumerate().filter_map(|(i, n)| {
            if opt.mod_baseline {
                if n.mod_base <= 1e-6 { return None; }
                let r = n.mod_level / n.mod_base - 1.0;
                if r.abs() < MOD_DEADBAND { None } else { Some((i, r)) }
            } else if n.mod_level > 0.001 { Some((i, n.mod_level)) } else { None }
        }).collect();
        if !active.is_empty() && active.len() * 20 < neurons.len() {
            // 조절 신호가 벗어난 뉴런이 적을 때(평소): 그 뉴런들이 자기에게 들어오는 시냅스 중 흔적이 있는 것만 본다
            for &(post, l) in &active {
                let mut list = std::mem::take(&mut neurons[post].elig_in);
                list.retain(|&(pre, k)| {
                    // 기한을 다시 세는 바닥은 보내는 뉴런의 값 (시냅스를 빌리기 전에 읽어 둔다. 0 이면 조건이 없는 것과 같다)
                    let floor = neurons.get(pre as usize).map_or(0.0, |n| n.reset_min);
                    let Some(s) = neurons.get_mut(pre as usize).and_then(|n| n.synapses.get_mut(k as usize)) else { return false };
                    if s.target as usize != post { return false; }
                    if s.elig == 0.0 { s.lq = false; return false; }
                    s.sync(global_tick);
                    let moved = (gain * l.abs()).min(1.0) * s.elig;
                    s.weight = (s.weight + moved * l.signum()).clamp(0.0, 1.0);
                    s.elig -= moved;
                    // 굳은 만큼은 더 이상 일시적인 부분이 아니다 (합쳐진 세기가 두 번 세어지지 않게)
                    if s.short_decay > 0.0 { s.w_short -= moved; }
                    // 도파민으로 강화 쪽이 굳었으면 이 시냅스는 쓰인 것이다: 가지치기 기한을 지금부터 다시 센다
                    let reset = l > 0.0 && moved > 0.01 * s.weight && s.weight >= floor; // 가중치의 1% 이상 굳었고, 굳은 뒤의 가중치가 바닥(reset_min) 이상일 때만
                    if reset { s.last_co = global_tick; }
                    if OBS_ON.load(std::sync::atomic::Ordering::Relaxed) { obs_consol(pre as usize, post, l, moved, s.weight, reset, s.born, global_tick); }
                    if s.elig.abs() < 1e-3 * s.weight.max(1e-9) { s.elig = 0.0; s.lq = false; false } else { true }
                });
                neurons[post].elig_in = list;
            }
        } else if !active.is_empty() {
            // 많을 때(보상·처벌이 온 동안): 보내는 뉴런 단위로 여러 스레드가 나눠 처리한다
            use rayon::prelude::*;
            let mut level = vec![0.0f64; neurons.len()];
            for &(post, l) in &active { level[post] = l; }
            neurons.par_iter_mut().for_each(|n| {
                // 흔적이 남아 있는 시냅스의 목록만 본다 (시냅스 전체를 훑지 않는다)
                if n.modulatory || n.elig_idx.is_empty() { return; }
                let mut list = std::mem::take(&mut n.elig_idx);
                let floor = n.reset_min;
                let pre_id = n.id as usize;
                list.retain(|&k| {
                    let Some(s) = n.synapses.get_mut(k as usize) else { return false };
                    if s.elig == 0.0 { s.lp = false; return false; }
                    s.sync(global_tick);
                    if let Some(&l) = level.get(s.target as usize) {
                        if l != 0.0 {
                            // 쌓여 있던 흔적을 가중치로 내보낸다. 내보낸 만큼 흔적은 비워지므로 변화 총량은 쌓인 흔적을 넘지 않는다
                            let moved = (gain * l.abs()).min(1.0) * s.elig;
                            s.weight = (s.weight + moved * l.signum()).clamp(0.0, 1.0);
                            s.elig -= moved;
                            if s.short_decay > 0.0 { s.w_short -= moved; }
                            // 도파민으로 강화 쪽이 굳었으면 이 시냅스는 쓰인 것이다: 가지치기 기한을 지금부터 다시 센다
                            let reset = l > 0.0 && moved > 0.01 * s.weight && s.weight >= floor; // 가중치의 1% 이상 굳었고, 굳은 뒤의 가중치가 바닥(reset_min) 이상일 때만
                            if reset { s.last_co = global_tick; }
                            if OBS_ON.load(std::sync::atomic::Ordering::Relaxed) { obs_consol(pre_id, s.target as usize, l, moved, s.weight, reset, s.born, global_tick); }
                        }
                    }
                    // 흔적이 가중치의 0.1% 아래로 줄면 없는 것으로 본다 (틱당 0.993배로 줄어서, 그냥 두면 수천 틱 동안 목록에 남는다)
                    if s.elig.abs() < 1e-3 * s.weight.max(1e-9) { s.elig = 0.0; s.lp = false; false } else { true }
                });
                n.elig_idx = list;
            });
        }
    }
}

pub fn stdp_step(neurons: &mut [Neuron], tick_spikes: &[NeuronId], global_tick: u64, opt: &PlasticityOptions) {
    // 발화가 없는 틱: 아래의 STDP·iSTDP 는 모두 이번 틱에 발화한 뉴런을 도는 루프라 할 일이 없다.
    // 뉴런 수 길이의 배열(BCM 배수, 마지막 발화 틱, 지역 배수 등)을 만들지 않고 흔적 넘기기만 하고 끝낸다
    if tick_spikes.is_empty() { mod_transfer(neurons, global_tick, opt); return; }
    // STDP: 발화 타이밍 기반 강화/약화 (LTD = LTP × 1.2)
    const STDP_A_PLUS_BASE: f64 = 0.003;
    const STDP_A_MINUS_BASE: f64 = 0.0036; // LTD:LTP = 1.2:1
    const STDP_TAU: f64 = 20.0;
    // BCM: post 의 누적 발화 수(fire_count_window)가 이 값보다 많을수록 LTP 는 줄고 LTD 는 는다.
    // bcm 옵션을 끄면 배수가 항상 1 (LTP ×1, LTD ×1)
    let BCM_TARGET_RATE: f64 = if opt.bcm { 25.0 } else { f64::INFINITY };
    // 뉴런별 BCM 배수 (post 기준). 1 에 가까우면 LTP 그대로, 0 에 가까우면 LTP 억제·LTD 강화
    //   발화율 기준(bcm_rate_target): 목표율 / (최근 발화율 + 목표율)
    //   기존(누적 횟수 기준):        25 / (누적 발화 수 + 25)
    // 상대 목표: 지금 발화 중인 뉴런들(최근 발화율 > 0.005)의 발화율 중앙값 × 배수.
    // 네트워크 전체 활동이 커지면 목표도 따라 올라간다.
    let relative_target: Option<f64> = opt.bcm_rate_relative.and_then(|factor| {
        let mut rates: Vec<f64> = neurons.iter().map(|n| n.rate_fast).filter(|&r| r > 0.005).collect();
        if rates.is_empty() { return None; }
        let mid = rates.len() / 2;
        let (_, median, _) = rates.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        Some(*median * factor)
    });
    let rate_target = relative_target.or(opt.bcm_rate_target);
    let mut bcm_vals: Vec<f64> = neurons.iter().map(|n| match rate_target {
        Some(t) => t / (n.rate_fast + t),
        None if opt.bcm_rate_relative.is_some() || opt.bcm_local.is_some() || opt.bcm_classic.is_some() || opt.bcm_classic_local.is_some() => 1.0,
        None => bcm_factor(n.fire_count_window, BCM_TARGET_RATE),
    }).collect();
    // 교과서 BCM: LTP 는 그대로 두고, LTD 에 (자기 장기 평균 발화율 ÷ 목표율)² 을 곱한다.
    // 평균이 목표를 넘으면 제동이 제곱으로 커지고, 밑돌면 LTD 가 거의 걸리지 않는다.
    let mut classic_ltd: Option<Vec<f64>> = opt.bcm_classic.map(|rho0| {
        bcm_vals.iter_mut().for_each(|v| *v = 1.0);
        neurons.iter().map(|n| { let x = n.rate_avg / rho0; (x * x).min(10.0) }).collect()
    });
    // 교과서 BCM 의 목표를 뉴런별로: 자기에게 붙은 시냅스들 중 활동 중인 것들
    // (활성화 빈도 장기 평균 > 0.001)의 중앙값 × 배수. 이번 틱에 관여하는 post 만 계산한다.
    if let Some(factor) = opt.bcm_classic_local {
        bcm_vals.iter_mut().for_each(|v| *v = 1.0);
        let mut mult = vec![1.0f64; neurons.len()];
        let mut needed: Vec<usize> = Vec::new();
        for &nid in tick_spikes {
            let i = nid as usize;
            if i >= neurons.len() { continue; }
            // 기준선 방식에서 조절 뉴런의 출력 시냅스는 학습하지 않으므로 그 상대의 배수는 필요 없다
            // (조절 뉴런은 수천 개에 닿고 계속 발화해서, 넣으면 매번 큰 목록을 만들게 된다)
            if opt.mod_baseline && neurons[i].modulatory { continue; }
            // 이 배수는 약화에만 쓰이고, 약화는 상대가 방금(50틱 이내) 먼저 발화했을 때만 일어난다 → 그런 상대만 계산한다
            needed.extend(neurons[i].synapses.iter().map(|s| s.target as usize)
                .filter(|&t| neurons.get(t).and_then(|n| n.last_spike_tick).is_some_and(|ts| ts < global_tick && global_tick - ts < 50)));
        }
        needed.sort_unstable();
        needed.dedup();
        let mut rates: Vec<f64> = Vec::new();
        for post in needed {
            if post >= neurons.len() { continue; }
            // 입력 시냅스 전체를 훑는 값이고 천천히 변하므로(시간 상수 약 5000틱), 뉴런마다 BCM_CACHE_TICKS 틱 동안은 지난 값을 쓴다
            if neurons[post].ltd_mult_tick != 0 && global_tick.saturating_sub(neurons[post].ltd_mult_tick) < BCM_CACHE_TICKS {
                mult[post] = neurons[post].ltd_mult;
                continue;
            }
            rates.clear();
            // 다른 뉴런의 내부 상태는 읽지 않는다. 자기에게 붙은 시냅스가 스스로 센 활성화 빈도만 본다
            rates.extend(neurons[post].incoming.iter()
                .filter_map(|&(pre, syn)| neurons.get(pre).and_then(|n| n.synapses.get(syn)).map(|s| s.use_avg_at(global_tick)))
                .filter(|&r| r > 0.001));
            // 활동 중인 입력이 없으면 LTD 배수 없음 (1.0)
            let m = if rates.is_empty() { 1.0 } else {
                let mid = rates.len() / 2;
                let (_, median, _) = rates.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
                let x = neurons[post].rate_avg / (*median * factor);
                (x * x).min(10.0)
            };
            mult[post] = m;
            neurons[post].ltd_mult = m;
            neurons[post].ltd_mult_tick = global_tick.max(1);
        }
        classic_ltd = Some(mult);
    }
    // 지역 목표: 뉴런이 알 수 있는 정보만 쓴다 — 자기에게 입력을 주는 뉴런들 중
    // 지금 발화 중인 것들(최근 발화율 > 0.005)의 발화율 중앙값 × 배수.
    // 이번 틱의 STDP 에 관여하는 post(발화한 뉴런과 그 시냅스 대상)만 계산한다.
    if let Some(factor) = opt.bcm_local {
        let mut needed: Vec<usize> = Vec::new();
        for &nid in tick_spikes {
            let i = nid as usize;
            if i >= neurons.len() { continue; }
            needed.push(i);
            needed.extend(neurons[i].synapses.iter().map(|s| s.target as usize));
        }
        needed.sort_unstable();
        needed.dedup();
        let mut rates: Vec<f64> = Vec::new();
        for post in needed {
            if post >= neurons.len() { continue; }
            rates.clear();
            rates.extend(neurons[post].incoming.iter()
                .filter_map(|&(pre, syn)| neurons.get(pre).and_then(|n| n.synapses.get(syn)).map(|s| s.recent_rate_at(global_tick) * 0.1))
                .filter(|&r| r > 0.005));
            if rates.is_empty() { continue; } // 발화 중인 입력이 없으면 배수 없음
            let mid = rates.len() / 2;
            let (_, median, _) = rates.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let target = *median * factor;
            bcm_vals[post] = target / (neurons[post].rate_fast + target);
        }
    }
    let gate_base: Option<f64> = opt.mod_gate.map(|_| opt.gate_base);
    // 조절 신호 방식의 흔적 넘기기 (발화와 무관하게 매 틱)
    mod_transfer(neurons, global_tick, opt);
    let spike_ticks: Vec<Option<u64>> = neurons.iter()
        .map(|n| n.last_spike_tick).collect();
    // 지역 배수: 시냅스가 붙어 있는 post 뉴런이 자기 입력 시냅스에서 알 수 있는 값만 쓴다.
    //   변화량 = 입력 가중치 중앙값 ÷ 0.2 (기존 설계에서 0.2 = 평범한 시냅스)
    //   soft bound 배수 = 가장 강한 입력 가중치 (기존 설계에서 1.0 = 가장 강한 시냅스)
    // 값은 refresh_local_scales 가 뉴런에 기록해 둔 것을 읽는다. 아직 없으면(0) 전역 배수를 쓴다.
    let local: Vec<(f64, f64)> = if opt.local_scale {
        neurons.iter().map(|n| (n.w_ref_med / 0.2, n.w_ref_max)).collect()
    } else { Vec::new() };
    let amp = |i: usize| -> f64 { match local.get(i) { Some(&(a, _)) if a > 0.0 => a, _ => 1.0 } };
    let bnd = |i: usize, global: f64| -> f64 { match local.get(i) { Some(&(_, b)) if b > 0.0 => b, _ => global } };
    // 조절 신호(도파민 등)가 STDP 자체를 바꾸는 효과: 조절 수준이 높은 뉴런의 입력 시냅스는
    // LTP 가 (1 + gain × 조절 수준)배로 커지고 LTD 는 그만큼 줄어든다 (포유류 실험: 도파민이 LTD 를 LTP 쪽으로 바꿈)
    let mod_levels: Vec<f64> = if opt.mod_stdp_gain > 0.0 { neurons.iter().map(|n| n.mod_level).collect() } else { Vec::new() };
    let mod_ltp = |i: usize| -> f64 { match mod_levels.get(i) { Some(&m) => 1.0 + opt.mod_stdp_gain * m, None => 1.0 } };
    // 기준선 방식: 조절 뉴런이 내보내는 시냅스와 조절 뉴런으로 들어오는 시냅스는 학습하지 않는다.
    // 예측 뉴런의 입력 시냅스는 STDP 가 아니라 위의 지연 흔적 규칙으로만 학습한다
    let fixed: Vec<bool> = if opt.mod_baseline { neurons.iter().map(|n| n.modulatory || n.cell_type == crate::neuron::CellType::Predictor || n.cell_type == crate::neuron::CellType::Value).collect() } else { Vec::new() };
    let is_fixed = |i: usize| -> bool { fixed.get(i).copied().unwrap_or(false) };

    // 이번 틱에 흔적이 새로 생긴 시냅스 (보내는 뉴런, 시냅스 번호). 끝에서 각 뉴런의 목록에 올린다
    let mut pending: Vec<(usize, u32)> = Vec::new();

    // Pass 1: pre가 지금 발화 → post의 과거 발화 확인 → dt ≤ 0 케이스 처리
    // BCM은 post 기준 (post 자주 발화 → LTP 어려움, LTD 쉬움)
    for &nid in tick_spikes {
        let pidx = nid as usize;
        if pidx >= neurons.len() { continue; }
        if !opt.stdp_on_inhibitory && neurons[pidx].inhibitory { continue; }
        if is_fixed(pidx) { continue; }
        for (k_syn, syn) in neurons[pidx].synapses.iter_mut().enumerate() {
            let tidx = syn.target as usize;
            if tidx >= spike_ticks.len() || is_fixed(tidx) { continue; }
            syn.sync(global_tick);

            if let Some(post_tick) = spike_ticks[tidx] {
                let dt = post_tick as f64 - global_tick as f64;
                let diff_ltp = syn.weight - 0.2;
                let diff_ltd = syn.weight - 1.0;
                let f_ltp = ltp_factor(syn.weight, bnd(tidx, opt.bound_scale), opt);
                let f_ltd = ltd_factor(syn.weight, bnd(tidx, opt.bound_scale), opt);
                // post 기준 BCM
                let post_bcm = bcm_vals[tidx];
                let stdp_a_plus = STDP_A_PLUS_BASE * post_bcm * mod_ltp(tidx);        // post 자주 발화 → LTP 어려움
                let ltd_mult = match &classic_ltd { Some(v) => v[tidx], None => 2.0 - post_bcm };
                let stdp_a_minus = STDP_A_MINUS_BASE * opt.ltd_scale * ltd_mult / mod_ltp(tidx); // post 자주 발화 → LTD 쉬움
                if dt == 0.0 {
                    // 동시 발화: LTP
                    let ltp_bonus = syn.accumulate_ltp();
                    // 조절 신호 방식에서는 STDP 만으로 "쓰였다"고 보지 않는다 (우연한 겹침도 STDP 를 일으킨다). 굳었을 때만 기한이 밀린다
                    if gate_base.is_none() { syn.last_co = global_tick; }
                    let dw = (stdp_a_plus + ltp_bonus) * f_ltp * opt.stdp_scale * amp(tidx);
                    if apply_dw(syn, dw, gate_base) { pending.push((pidx, k_syn as u32)); }
                } else if dt < 0.0 && dt > -50.0 {
                    // post가 먼저 발화: LTD
                    if opt.ltd_once_per_post {
                        if syn.ltd_mark == post_tick { continue; }
                        syn.ltd_mark = post_tick;
                    }
                    let dw = stdp_a_minus * (dt / STDP_TAU).exp() * f_ltd * opt.stdp_scale * amp(tidx);
                    if apply_dw(syn, -dw, gate_base) { pending.push((pidx, k_syn as u32)); }
                }
            }
        }
    }

    // Pass 2: 진짜 LTP — post가 지금 발화, pre가 과거 발화 (dt > 0)
    // 뉴런의 incoming 역참조로 해당 post의 incoming 시냅스만 순회
    // BCM은 post 기준
    for &post_id in tick_spikes {
        let post_idx = post_id as usize;
        if post_idx >= neurons.len() { continue; }
        if is_fixed(post_idx) { continue; }
        let post_bcm = bcm_vals[post_idx];
        let stdp_a_plus = STDP_A_PLUS_BASE * post_bcm * mod_ltp(post_idx);
        let pairs = neurons[post_idx].incoming.clone();
        for (pre_idx, syn_idx) in pairs {
            if is_fixed(pre_idx) { continue; }
            if !opt.stdp_on_inhibitory && neurons.get(pre_idx).is_some_and(|n| n.inhibitory) { continue; }
            let pre_tick = match spike_ticks.get(pre_idx).copied().flatten() {
                Some(t) if t < global_tick && global_tick - t < 50 => t,
                _ => continue,
            };
            if syn_idx >= neurons[pre_idx].synapses.len() { continue; }
            let dt = (global_tick - pre_tick) as f64;
            let syn = &mut neurons[pre_idx].synapses[syn_idx];
            syn.sync(global_tick);
            let diff_ltp = syn.weight - 0.2;
            let f_ltp = ltp_factor(syn.weight, bnd(post_idx, opt.bound_scale), opt);
            let base = stdp_a_plus * (-dt / STDP_TAU).exp();
            let ltp_bonus = syn.accumulate_ltp();
            if gate_base.is_none() { syn.last_co = global_tick; }
            let dw = (base + ltp_bonus) * f_ltp * opt.stdp_scale * amp(post_idx);
            if apply_dw(syn, dw, gate_base) { pending.push((pre_idx, syn_idx as u32)); }
        }
    }

    // iSTDP: spike-timing + Gaussian soft-bound
    // 억제 pre → post 발화 (dt > 0): 억제 부족 → LTP (억제 강화)
    // post 먼저 → pre 억제 (dt ≤ 0): 억제 타이밍 늦음 → LTD
    const ISTDP_A_PLUS: f64 = 0.012;
    const ISTDP_A_MINUS: f64 = 0.0144;
    const ISTDP_TAU: f64 = 20.0;

    // iSTDP Pass 1: 억제성 pre 발화 시, outgoing의 post 발화 확인 (dt ≤ 0: LTD)
    // BCM 반전: post 자주 발화 → iLTD 약화 (post 활동 억제해야 함)
    for &pre_id in tick_spikes {
        let pidx = pre_id as usize;
        if pidx >= neurons.len() || !neurons[pidx].inhibitory { continue; }
        if let Some(alpha) = opt.istdp_target {
            // 목표 발화율 방식: dw = η × (post 흔적 − α)
            let eta = ISTDP_A_PLUS * opt.istdp_scale;
            for syn in &mut neurons[pidx].synapses {
                let tidx = syn.target as usize;
                if tidx >= spike_ticks.len() { continue; }
                let trace = match spike_ticks[tidx] {
                    Some(t) if global_tick - t < 50 => (-((global_tick - t) as f64) / ISTDP_TAU).exp(),
                    _ => 0.0,
                };
                syn.weight = (syn.weight + eta * (trace - alpha)).clamp(0.0, 1.0);
            }
            continue;
        }
        for (k_syn, syn) in neurons[pidx].synapses.iter_mut().enumerate() {
            let tidx = syn.target as usize;
            if tidx >= spike_ticks.len() || is_fixed(tidx) { continue; }
            syn.sync(global_tick);
            if let Some(post_tick) = spike_ticks[tidx] {
                let dt = post_tick as f64 - global_tick as f64;
                let diff_ltd = syn.weight - 1.0;
                let f_ltd = istdp_factor(syn.weight, 1.0, bnd(tidx, opt.istdp_bound_scale), opt);
                // 반전 BCM: post 자주 → post_bcm ↓ → iLTD 배수 작아짐
                let post_bcm = bcm_vals[tidx];
                let istdp_a_minus = ISTDP_A_MINUS * post_bcm;
                if dt == 0.0 {
                    let dw = istdp_a_minus * 0.5 * f_ltd * opt.istdp_scale * amp(tidx);
                    if apply_dw(syn, -dw, gate_base) { pending.push((pidx, k_syn as u32)); }
                } else if dt < 0.0 && dt > -50.0 {
                    let dw = istdp_a_minus * (dt / ISTDP_TAU).exp() * f_ltd * opt.istdp_scale * amp(tidx);
                    if apply_dw(syn, -dw, gate_base) { pending.push((pidx, k_syn as u32)); }
                }
            }
        }
    }

    // iSTDP Pass 2: post 발화 시, incoming 억제 pre 발화 확인 (dt > 0: LTP)
    // BCM 반전: post 자주 발화 → iLTP 강화 (억제 필요)
    for &post_id in tick_spikes {
        let post_idx = post_id as usize;
        if post_idx >= neurons.len() { continue; }
        if is_fixed(post_idx) { continue; }
        let post_bcm = bcm_vals[post_idx];
        let istdp_a_plus = ISTDP_A_PLUS * (2.0 - post_bcm); // post 자주 → iLTP 강화
        let pairs = neurons[post_idx].incoming.clone();
        for (pre_idx, syn_idx) in pairs {
            if pre_idx >= neurons.len() { continue; }
            if !neurons[pre_idx].inhibitory { continue; }
            if syn_idx >= neurons[pre_idx].synapses.len() { continue; }
            let pre_tick = match spike_ticks.get(pre_idx).copied().flatten() {
                Some(t) if t < global_tick && global_tick - t < 50 => t,
                _ => continue,
            };
            let dt = (global_tick - pre_tick) as f64;
            let syn = &mut neurons[pre_idx].synapses[syn_idx];
            syn.sync(global_tick);
            if opt.istdp_target.is_some() {
                let dw = ISTDP_A_PLUS * opt.istdp_scale * (-dt / ISTDP_TAU).exp();
                if apply_dw(syn, dw, gate_base) { pending.push((pre_idx, syn_idx as u32)); }
                continue;
            }
            let diff_ltp = syn.weight - 0.2;
            let f_ltp = istdp_factor(syn.weight, 0.2, bnd(post_idx, opt.istdp_bound_scale), opt);
            let dw = istdp_a_plus * (-dt / ISTDP_TAU).exp() * f_ltp * opt.istdp_scale * amp(post_idx);
            if apply_dw(syn, dw, gate_base) { pending.push((pre_idx, syn_idx as u32)); }
        }
    }
    for (n, k) in pending {
        let Some(s) = neurons[n].synapses.get_mut(k as usize) else { continue };
        let (add_p, add_q, target) = (!s.lp, !s.lq, s.target as usize);
        s.lp = true;
        s.lq = true;
        if add_p { neurons[n].elig_idx.push(k); }
        if add_q { if let Some(t) = neurons.get_mut(target) { t.elig_in.push((n as u32, k)); } }
    }
}

/// 예측 뉴런의 입력(지연 시냅스) 학습. slow = 지연 시냅스의 위치 (보내는 뉴런, 시냅스 번호).
///   매 틱 가중치 += pred_eta × (받는 뉴런의 조절 수준 ÷ 기준선 − 1) × (이 시냅스의 지연 흔적)
///   → 보상이 온 시점에 켜져 있던 지연의 시냅스가 강해지고, 예측이 넘쳐 도파민이 내려간 시점의 시냅스는 약해진다.
/// 여기에는 ±MOD_DEADBAND 를 두지 않는다: 도파민의 급증은 짧고 크고 정지는 길고 얕아서, 작은 변화를 버리면
/// 정지 쪽만 버려져 예측이 한 방향으로만 자란다. 버리지 않으면 변화의 합은 도파민 발화 수가 평소보다 많았는지 적었는지에 비례한다
/// predictor_step 과 같은 규칙을 "발화했을 때만 알리기" 방식으로: 지연 시냅스 전체를 훑지 않고, 받는 뉴런(예측·가치)마다
/// 흔적이 살아 있는 입력의 목록만 본다. 조절 편차가 문턱을 넘지 않은 받는 뉴런은 건너뛴다.
///   recv: 받는 뉴런들, live[p]: recv[p] 의 살아 있는 입력 (보내는 뉴런, 시냅스 번호), listed: 목록에 올라 있는 시냅스의 열쇠 (보내는 뉴런 << 32 | 번호)
///   deltas: 예측 뉴런의 입력 가중치가 바뀐 만큼의 (p, Σ Δ가중치 × slow1, Σ Δ가중치 × slow2) — 부르는 쪽이 받는 쪽 합에 더한다
///   따라잡은 뒤 두 흔적이 모두 0.001 아래인 시냅스는 목록에서 뺀다 (다시 발화하기 전에는 흔적이 0.01 을 넘을 수 없다)
pub fn predictor_step_events(neurons: &mut [Neuron], recv: &[usize], live: &mut [Vec<(u32, u32)>], listed: &mut std::collections::HashSet<u64>,
                             now: u64, opt: &PlasticityOptions, deltas: &mut Vec<(usize, f64, f64)>) {
    if !opt.mod_baseline || opt.pred_eta == 0.0 { return; }
    for (p, &t) in recv.iter().enumerate() {
        if live[p].is_empty() { continue; }
        let r = match neurons.get(t) { Some(n) if n.mod_base > 1e-6 => n.mod_level / n.mod_base - 1.0, _ => continue };
        // 이 받는 뉴런의 입력은 모두 같은 종류다 (가치 뉴런의 입력은 slow_fast, 예측 뉴런의 입력은 아님): 첫 항목으로 문턱을 정한다
        let fast = { let (pre, j) = live[p][0]; neurons[pre as usize].synapses.get(j as usize).is_some_and(|s| s.slow_fast) };
        if fast { if r < VALUE_UP && r > -VALUE_DOWN { continue; } } else if r.abs() < PRED_DEADBAND { continue; }
        let (mut d1, mut d2) = (0.0, 0.0);
        let mut i = 0;
        while i < live[p].len() {
            let (pre, j) = live[p][i];
            let Some(s) = neurons[pre as usize].synapses.get_mut(j as usize) else { i += 1; continue };
            s.sync(now);
            let tr = if s.slow_fast { (s.slow2 - s.slow1).max(0.0) } else { s.slow2 };
            if tr > 0.01 {
                let eta = if s.slow_fast && opt.val_eta > 0.0 { opt.val_eta } else { opt.pred_eta };
                let before = s.weight;
                s.weight = (s.weight + eta * r * tr).clamp(0.0, 1.0);
                if !s.slow_fast { let dw = s.weight - before; d1 += dw * s.slow1; d2 += dw * s.slow2; }
            }
            if s.slow1 < 0.001 && s.slow2 < 0.001 {
                listed.remove(&(((pre as u64) << 32) | j as u64));
                live[p].swap_remove(i);
            } else { i += 1; }
        }
        if d1 != 0.0 || d2 != 0.0 { deltas.push((p, d1, d2)); }
    }
}

pub fn predictor_step(neurons: &mut [Neuron], slow: &[(usize, usize)], now: u64, opt: &PlasticityOptions) {
    if !opt.mod_baseline || opt.pred_eta == 0.0 { return; }
    for &(pre, j) in slow {
        // 흔적: 예측 뉴런의 입력은 지연 흔적 그대로. 가치 뉴런의 입력(전달은 바로 하는 시냅스)은 (늦은 흔적 − 이른 흔적)의 양수 부분만 쓴다
        //   → 방금 활성화된 직후(약 20틱)에는 0 이고 그 뒤에 올라온다. 가치 뉴런이 자극 직후 스스로 일으킨 도파민으로는
        //     자기 입력이 커지지 않고, 조금 뒤에 오는 보상으로만 커지게 하기 위한 것 (되먹임으로 보상 없이 불어나는 것을 막는다)
        let (t, trace, fast) = match neurons[pre].synapses.get_mut(j) {
            Some(s) => { s.sync(now); let tr = if s.slow_fast { (s.slow2 - s.slow1).max(0.0) } else { s.slow2 }; if tr > 0.01 { (s.target as usize, tr, s.slow_fast) } else { continue } }
            None => continue,
        };
        let r = match neurons.get(t) { Some(n) if n.mod_base > 1e-6 => n.mod_level / n.mod_base - 1.0, _ => continue };
        // 작은 벗어남은 양쪽 똑같이 버린다. 가치 뉴런은 도파민을 짧게 읽어서 평소의 흔들림이 크므로 더 넓게(±50%) 버린다:
        // 급증(+수백 %)과 완전한 정지(−100% 근처)만 남는다
        // 가치 뉴런이 읽는 도파민은 흔들림이 한쪽으로 치우쳐 있다 (측정: +50% 초과 6.5%, −50% 미만 0.02%).
        // 그래서 위쪽은 +100% 를 넘어야, 아래쪽은 −60% 아래여야 신호로 본다. 급증은 +수백 %, 완전한 정지는 −100% 근처라 둘 다 통과한다
        if fast { if r < VALUE_UP && r > -VALUE_DOWN { continue; } } else if r.abs() < PRED_DEADBAND { continue; }
        let s = &mut neurons[pre].synapses[j];
        let eta = if fast && opt.val_eta > 0.0 { opt.val_eta } else { opt.pred_eta };
        s.weight = (s.weight + eta * r * trace).clamp(0.0, 1.0);
    }
}

/// post 뉴런 기준의 (변화량 배수, soft bound 배수). local_scale 이 꺼져 있거나 값이 없으면 전역 값
#[inline]
fn local_scales(neurons: &[Neuron], post: usize, opt: &PlasticityOptions) -> (f64, f64) {
    if opt.local_scale {
        if let Some(n) = neurons.get(post) {
            if n.w_ref_med > 0.0 && n.w_ref_max > 0.0 {
                return (n.w_ref_med / 0.2, n.w_ref_max);
            }
        }
    }
    (1.0, opt.bound_scale)
}

/// R-STDP eligibility: 한 자극 동안의 발화 기록에서 시냅스별 dw 를 계산
pub fn compute_eligibility(neurons: &[Neuron], spikes: &[(NeuronId, u64)], opt: &PlasticityOptions) -> Vec<EligibilityTrace> {
    // 뉴런별 발화 틱 목록 (오름차순). 마지막 발화만 남기면 여러 번 발화한 뉴런
    // (입력 뉴런은 sustain 동안 매 틱 발화)의 인과 쌍이 사라지므로 전부 보관한다.
    let mut spike_map: HashMap<NeuronId, Vec<u64>> = HashMap::new();
    for &(nid, t) in spikes {
        spike_map.entry(nid).or_default().push(t);
    }
    for ticks in spike_map.values_mut() {
        ticks.sort_unstable();
    }
    // tick 직전의 가장 최근 발화 (온라인 STDP의 last_spike_tick과 같은 의미)
    fn last_before(ticks: &[u64], tick: u64) -> Option<u64> {
        let n = ticks.partition_point(|&t| t < tick);
        if n == 0 { None } else { Some(ticks[n - 1]) }
    }
    let mut eligibility = Vec::new();

    // Pass 1: pre 기준 — pre 발화 시점에서 post의 과거 발화 확인 (dt < 0: LTD)
    // Gaussian soft-bound: LTD peak w=1.0 (강한 시냅스는 약화가 쉬움)
    for &(pre_id, pre_tick) in spikes {
        let pidx = pre_id as usize;
        if pidx >= neurons.len() { continue; }
        if !opt.stdp_on_inhibitory && neurons[pidx].inhibitory { continue; }

        for (syn_idx, syn) in neurons[pidx].synapses.iter().enumerate() {
            if let Some(post_tick) = spike_map.get(&syn.target)
                .and_then(|ticks| last_before(ticks, pre_tick))
            {
                let dt = post_tick as f64 - pre_tick as f64;
                if dt < 0.0 {
                    let diff_ltd = syn.weight - 1.0;
                    let (amp, bnd) = local_scales(neurons, syn.target as usize, opt);
                    let f_ltd = ltd_factor(syn.weight, bnd, opt);
                    let dw = -0.01728 * opt.ltd_scale * (-dt.abs() / 20.0).exp() * f_ltd * amp; // LTD 1.2배 (1.2:1)
                    eligibility.push(EligibilityTrace {
                        neuron_idx: pidx, synapse_idx: syn_idx, dw,
                    });
                }
            }
        }
    }

    // Pass 2: post 기준 (incoming 역참조) — post 발화 시 pre의 과거 발화 확인 (dt > 0: LTP)
    // Gaussian soft-bound: LTP peak w=0.2 (약한 시냅스는 강화가 쉬움)
    for &(post_id, post_tick) in spikes {
        let post_idx = post_id as usize;
        if post_idx >= neurons.len() { continue; }
        for &(pre_idx, syn_idx) in &neurons[post_idx].incoming {
            if !opt.stdp_on_inhibitory && neurons.get(pre_idx).is_some_and(|n| n.inhibitory) { continue; }
            if let Some(pre_tick) = spike_map.get(&(pre_idx as NeuronId))
                .and_then(|ticks| last_before(ticks, post_tick))
            {
                let dt = post_tick as f64 - pre_tick as f64;
                if dt > 0.0 {
                    if syn_idx < neurons[pre_idx].synapses.len() {
                        let syn = &neurons[pre_idx].synapses[syn_idx];
                        let diff_ltp = syn.weight - 0.2;
                        let (amp, bnd) = local_scales(neurons, post_idx, opt);
                        let f_ltp = ltp_factor(syn.weight, bnd, opt);
                        let dw = 0.0144 * (-dt.abs() / 20.0).exp() * f_ltp * amp;
                        eligibility.push(EligibilityTrace {
                            neuron_idx: pre_idx, synapse_idx: syn_idx, dw,
                        });
                    }
                }
            }
        }
    }

    eligibility
}

/// eligibility × reward 를 가중치에 반영. factor(pre_idx, post_idx) 로 영역별 배수를 줄 수 있다.
pub fn apply_reward(
    neurons: &mut [Neuron],
    eligibility: &[EligibilityTrace],
    reward: f64,
    factor: impl Fn(usize, usize) -> f64,
) {
    for e in eligibility {
        let (nidx, sidx) = (e.neuron_idx, e.synapse_idx);
        if nidx >= neurons.len() { continue; }
        if sidx >= neurons[nidx].synapses.len() { continue; }
        let target = neurons[nidx].synapses[sidx].target as usize;
        let change = e.dw * reward * factor(nidx, target);
        let w = &mut neurons[nidx].synapses[sidx].weight;
        *w = (*w + change).clamp(0.0, 1.0);
    }
}

/// 각 뉴런의 incoming 역참조 (pre_idx, syn_idx) 재구축
pub fn rebuild_incoming(neurons: &mut [Neuron]) {
    for n in neurons.iter_mut() {
        n.incoming.clear();
    }
    for pre_idx in 0..neurons.len() {
        let synapses_len = neurons[pre_idx].synapses.len();
        for syn_idx in 0..synapses_len {
            let target = neurons[pre_idx].synapses[syn_idx].target as usize;
            if target < neurons.len() {
                neurons[target].incoming.push((pre_idx, syn_idx));
            }
        }
    }
}

/// 뉴런별 지역 배수 갱신: 각 뉴런에 "자기 입력 시냅스 가중치의 중앙값과 상위 1% 값"을 기록한다.
/// 가중치가 학습으로 바뀌므로 주기적으로(예: 항상성과 같은 주기) 다시 부른다.
pub fn refresh_local_scales(neurons: &mut [Neuron], strong_pct: f64) {
    let stats: Vec<(f64, f64)> = neurons.iter().map(|n| {
        let mut w: Vec<f64> = n.incoming.iter()
            .filter_map(|&(p, s)| neurons.get(p).and_then(|pre| pre.synapses.get(s)).map(|syn| syn.weight))
            .collect();
        if w.is_empty() { return (0.0, 0.0); }
        // "가장 강한 입력"은 최댓값 대신 자기 입력 분포의 strong_pct 지점(예: 0.99 = 상위 1%)을 쓴다. 입력이 수천 개인 뉴런은 최댓값이
        // 극단값 하나에 끌려가 soft bound 가 너무 넓어지고, 약한 시냅스에 강화만 걸리게 된다.
        // (입력이 100개 미만이면 최댓값과 같다)
        let hi = ((w.len() as f64 * strong_pct) as usize).min(w.len() - 1);
        let (_, strong, _) = w.select_nth_unstable_by(hi, |a, b| a.partial_cmp(b).unwrap());
        let strong = *strong;
        let mid = w.len() / 2;
        let (_, med, _) = w.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        (*med, strong)
    }).collect();
    for (n, (med, max)) in neurons.iter_mut().zip(stats) {
        n.w_ref_med = med;
        n.w_ref_max = max;
    }
}

// ── 조절 신호에 의한 가소성 (3요소 규칙) ──
// 조절 뉴런은 전위를 올리지 않는다. 대신 신호가 닿은 뉴런의 입력 시냅스를, 시냅스가 스스로 기록한
// 최근 활성화 흔적(recent_rate)에 따라 바꾼다. 변화는 현재 가중치에 비례한다.
//   시냅스 활동 → 조절 신호:  조절 신호가 도착했을 때 최근에 활성화됐던 시냅스가 변한다
//   조절 신호 → 시냅스 활동:  조절 수준이 남아 있는 동안 (한동안 조용하다가) 활성화된 시냅스가 변한다

/// 조절 신호가 post 에 도착했을 때 호출. amount = 도착한 신호의 크기
pub fn on_modulator(neurons: &mut [Neuron], post: usize, amount: f64, now: u64, opt: &PlasticityOptions) {
    if opt.mod_eta == 0.0 || post >= neurons.len() { return; }
    let pairs = neurons[post].incoming.clone();
    for (pre, syn) in pairs {
        if pre >= neurons.len() || neurons[pre].modulatory { continue; }
        let Some(s) = neurons[pre].synapses.get_mut(syn) else { continue };
        s.sync(now);
        if s.recent_rate <= 0.001 { continue; }
        let factor = 1.0 + opt.mod_eta * amount * s.recent_rate * opt.mod_forward_sign;
        s.weight = (s.weight * factor.max(0.0)).clamp(0.0, 1.0);
    }
}

/// 조절 수준이 남아 있는 post 로 일반 신호가 전달될 때 호출 (pre 의 syn 번째 시냅스).
/// 그 시냅스가 직전까지 조용했던 경우에만 적용한다 — 이미 활동 중이던 시냅스는 위의 규칙이 담당한다.
pub fn on_pre_under_modulator(neurons: &mut [Neuron], pre: usize, syn: usize, opt: &PlasticityOptions) {
    if opt.mod_eta == 0.0 || pre >= neurons.len() { return; }
    let Some(target) = neurons[pre].synapses.get(syn).map(|s| s.target as usize) else { return };
    let level = match neurons.get(target) { Some(n) => n.mod_level, None => return };
    if level <= 0.001 { return; }
    let s = &mut neurons[pre].synapses[syn];
    if s.recent_rate > 0.1001 { return; } // 방금의 발화로 0.1 이 더해진 상태 = 그 전에는 조용했음
    let factor = 1.0 + opt.mod_eta * level * opt.mod_backward_sign;
    s.weight = (s.weight * factor.max(0.0)).clamp(0.0, 1.0);
}
