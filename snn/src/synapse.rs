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
    /// 시냅스 피로도 (0.0 = 쉼, 1.0 = 완전 피로)
    /// 전달 시 effective_weight = weight × (1 - fatigue)
    #[serde(default)]
    pub fatigue: f64,
    /// LTP 누적 카운터: 짧은 시간 내 반복 활성화 횟수
    #[serde(default)]
    pub ltp_trace: f64,
    /// 마지막 사용 tick (자체 prune 판정용)
    #[serde(default)]
    pub last_used_tick: u64,
    /// 전달 빈도 (자주 쓰일수록 약화)
    #[serde(default)]
    pub recent_rate: f64,
    /// STDP 가 남긴 흔적 (조절 신호가 있어야만 가중치가 변하는 방식에서 사용).
    /// 양수면 강화 쪽, 음수면 약화 쪽. 약 1초(틱당 ×0.993)에 걸쳐 사라진다
    #[serde(default)]
    pub elig: f64,
    /// 이 시냅스가 붙는 위치: true 면 받는 뉴런의 수상돌기 구획, false 면 세포체
    #[serde(default)]
    pub on_dendrite: bool,
    /// 촉진형 시냅스: 연달아 활성화되면 전달이 커진다 (예: 흥분성 → Sst).
    /// 기본(false)은 연달아 활성화되면 전달이 줄어드는 피로형
    #[serde(default)]
    pub facilitating: bool,
    /// 촉진 정도 (0~1). 활성화마다 올라가고 매 틱 ×0.97 로 줄어든다
    #[serde(default)]
    pub facil: f64,
    /// 이 시냅스가 활성화된 빈도의 장기 평균 (틱당 확률, 시간 상수 약 5000틱).
    /// 시냅스가 스스로 세는 값이라, post 뉴런이 입력 쪽 활동을 알 때는 이것만 본다
    #[serde(default)]
    pub use_avg: f64,
    /// 이미 LTD 를 적용한 post 발화 틱 (post 발화 하나당 LTD 한 번만 적용하는 옵션용)
    #[serde(default)]
    pub ltd_mark: u64,
    /// 지연 시냅스 (0 = 일반 시냅스). 0 보다 크면 보내는 뉴런이 발화해도 바로 전달하지 않고,
    /// 약 1/slow_rate 틱 뒤에 가장 커지는 느린 전류(가중치 × slow2)를 매 틱 전달한다.
    /// 예측 뉴런이 "자극 뒤 얼마쯤 지났는지"를 구분하는 데 쓴다 (지연이 서로 다른 시냅스 묶음)
    #[serde(default)]
    pub slow_rate: f64,
    /// 지연 시냅스의 두 단계 흔적: 발화하면 slow1 이 오르고, slow2 가 slow1 을 천천히 따라간다
    #[serde(default)]
    pub slow1: f64,
    #[serde(default)]
    pub slow2: f64,
    /// 피로·흔적 등 시간이 지나면 줄어드는 값들이 마지막으로 계산된 틱.
    /// 매 틱 모든 시냅스를 돌지 않고, 시냅스가 쓰일 때 sync(now) 로 지난 틱 수만큼 한꺼번에 줄인다 (결과는 매 틱 줄이는 것과 같다)
    #[serde(default)]
    pub t: u64,
    /// 이 시냅스가 생긴 틱 (처음부터 있던 것은 0). 갓 생긴 시냅스를 바로 가지치기하지 않기 위한 값
    #[serde(default)]
    pub born: u64,
    /// 이 연결(한 상대와의 연결)을 이루는 시냅스 접점의 개수. 가중치는 연결 전체의 세기이고, 접점 하나의 세기는 가중치 ÷ count
    #[serde(default = "default_count")]
    pub count: u32,
    /// 보내는 쪽과 받는 쪽이 마지막으로 함께 활동한 틱 (STDP 가 강화를 적용한 틱). 가지치기 기한을 여기서부터 센다
    #[serde(default)]
    pub last_co: u64,
    /// 전달 확률 (0 = 항상 전달). 0 보다 크면 보내는 뉴런이 발화할 때마다 이 확률로만 전달하고,
    /// 전달될 때는 세기를 1/p_rel 배로 해서 평균 전달량은 그대로 둔다 (실제 시냅스의 확률적 방출)
    #[serde(default)]
    pub p_rel: f64,
    /// 지연 흔적(slow1/slow2)은 쌓되 전달은 일반 시냅스처럼 바로 하는 시냅스 (가치 뉴런의 입력).
    /// 전달은 빠르고, 학습에 쓰는 흔적만 늦게 올라온다
    #[serde(default)]
    pub slow_fast: bool,
    /// 일시적인 세기 변화 (STDP 가 만든 것). 전달에는 가중치와 합쳐져 쓰이고, 시간이 지나면 0 으로 돌아간다.
    /// 조절 신호(도파민)가 와야 오래가는 가중치(weight)가 변한다 — 이 값은 굳지 않으면 사라진다
    #[serde(default)]
    pub w_short: f64,
    /// 일시적인 변화가 틱마다 줄어드는 배수 (0 = 이 방식을 쓰지 않음)
    #[serde(default)]
    pub short_decay: f64,
    /// 쌓인 알짜 (--prune-reset-net 이 켜져 있을 때만 쌓는다. 꺼져 있으면 늘 0 이고 저장 파일에도 쓰지 않는다):
    /// 조절 신호가 흔적을 가중치로 넘길 때 가중치가 실제로 바뀐 양을 부호째 더한 값. 가지치기의 기한을 다시 받으면 0 으로 돌아간다. 줄이는 박자는 없다
    #[serde(default, skip_serializing_if = "net_acc_is_zero")]
    pub net_acc: f64,
    /// 이 시냅스가 "흔적이 있는 시냅스" 목록에 올라 있는지: 보내는 뉴런의 목록(lp), 받는 뉴런의 목록(lq)
    #[serde(skip)]
    pub lp: bool,
    #[serde(skip)]
    pub lq: bool,
}
fn default_count() -> u32 { 1 }
fn net_acc_is_zero(v: &f64) -> bool { *v == 0.0 }

impl Synapse {
    pub fn new(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: false, fatigue: 0.0, ltp_trace: 0.0, last_used_tick: 0, recent_rate: 0.0, elig: 0.0, on_dendrite: false, facilitating: false, facil: 0.0, use_avg: 0.0, ltd_mark: 0, slow_rate: 0.0, slow1: 0.0, slow2: 0.0, t: 0, born: 0, count: 1, last_co: 0, p_rel: 0.0, slow_fast: false, w_short: 0.0, short_decay: 0.0, lp: false, lq: false, net_acc: 0.0 }
    }

    pub fn new_seed(target: NeuronId, weight: f64) -> Self {
        Self { target, weight, seed: true, fatigue: 0.0, ltp_trace: 0.0, last_used_tick: 0, recent_rate: 0.0, elig: 0.0, on_dendrite: false, facilitating: false, facil: 0.0, use_avg: 0.0, ltd_mark: 0, slow_rate: 0.0, slow1: 0.0, slow2: 0.0, t: 0, born: 0, count: 1, last_co: 0, p_rel: 0.0, slow_fast: false, w_short: 0.0, short_decay: 0.0, lp: false, lq: false, net_acc: 0.0 }
    }

    /// 발화 시 남은 자원의 15%를 소모 (fatigue += 0.15 × (1 - fatigue))
    /// 선형 누적(+0.15)은 30틱에 1번보다 자주 발화하면 fatigue가 1.0에 붙어
    /// 전달이 완전히 끊긴다 (초파리 커넥톰 테스트: 지속 자극 시 하류 발화 소멸).
    #[inline]
    pub fn fire_fatigue(&mut self) {
        if self.facilitating {
            // 촉진형: 피로는 1/3 만 쌓이고, 대신 촉진이 쌓인다
            self.fatigue += 0.05 * (1.0 - self.fatigue);
            self.facil += 0.3 * (1.0 - self.facil);
        } else {
            self.fatigue += 0.15 * (1.0 - self.fatigue);
        }
        self.recent_rate = (self.recent_rate + 0.1).min(1.0);
        self.use_avg += 0.0002;
    }

    /// 지난 틱 수만큼의 감쇠를 한꺼번에 적용해 now 시점의 값으로 맞춘다. 이 시냅스의 감쇠 값을 읽거나 바꾸기 전에 부른다
    #[inline]
    pub fn sync(&mut self, now: u64) {
        if now <= self.t { return; }
        let n = (now - self.t).min(1_000_000) as usize;
        self.t = now;
        if self.w_short != 0.0 { self.w_short *= self.short_decay.powi(n as i32); if self.w_short.abs() < 1e-12 { self.w_short = 0.0; } }
        if n == 1 { self.recover(); return; }
        let p = pow_tables();
        self.fatigue *= p.get(0, n);
        self.facil *= p.get(1, n);
        self.elig *= p.get(2, n);
        self.ltp_trace *= p.get(0, n);
        self.recent_rate *= p.get(3, n);
        self.use_avg *= p.get(4, n);
        if self.slow_rate > 0.0 {
            // 두 단계 흔적을 n 틱 진행: slow2 ← a^n·slow2 + n·r·a^(n−1)·slow1,  slow1 ← a^n·slow1  (a = 1 − r)
            let a = 1.0 - self.slow_rate;
            let an1 = a.powi(n as i32 - 1);
            self.slow2 = an1 * a * self.slow2 + n as f64 * self.slow_rate * an1 * self.slow1;
            self.slow1 *= an1 * a;
        }
    }

    /// now 시점의 사용 빈도 장기 평균 / 최근 활성화 (값을 바꾸지 않고 읽기만)
    #[inline]
    pub fn use_avg_at(&self, now: u64) -> f64 {
        if now <= self.t || self.use_avg == 0.0 { self.use_avg } else { self.use_avg * pow_tables().get(4, (now - self.t).min(1_000_000) as usize) }
    }
    #[inline]
    pub fn recent_rate_at(&self, now: u64) -> f64 {
        if now <= self.t || self.recent_rate == 0.0 { self.recent_rate } else { self.recent_rate * pow_tables().get(3, (now - self.t).min(1_000_000) as usize) }
    }

    /// 한 틱의 감쇠: 피로 ×0.95 (지수 회복) + LTP trace 감쇠 + rate 감쇠
    #[inline]
    pub fn recover(&mut self) {
        self.fatigue *= 0.95;
        self.facil *= 0.97;
        self.elig *= 0.993;
        self.ltp_trace *= 0.95;
        self.recent_rate *= 0.99;
        self.use_avg *= 0.9998;
        if self.slow_rate > 0.0 {
            self.slow2 += self.slow_rate * (self.slow1 - self.slow2);
            self.slow1 *= 1.0 - self.slow_rate;
        }
    }

    /// LTP 누적: 활성화될 때마다 trace 증가, 누적된 만큼 추가 강화 반환
    #[inline]
    pub fn accumulate_ltp(&mut self) -> f64 {
        self.ltp_trace += 1.0;
        // trace가 2 이상이면 반복 자극 → 추가 LTP
        if self.ltp_trace >= 2.0 {
            (self.ltp_trace - 1.0) * 0.001
        } else {
            0.0
        }
    }

    /// 실효 가중치 (weight × (1 - fatigue))
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        (self.weight + self.w_short).max(0.0) * (1.0 - self.fatigue) * (1.0 + 2.0 * self.facil)
    }
}

/// 감쇠 배수의 거듭제곱 표 (0.95, 0.97, 0.993, 0.99, 0.9998 의 n 제곱). 쓰일 때마다 거듭제곱을 계산하지 않기 위한 것
pub struct PowTables { t: [Vec<f64>; 5] }
const POW_BASES: [f64; 5] = [0.95, 0.97, 0.993, 0.99, 0.9998];
const POW_TABLE_LEN: usize = 8192;
impl PowTables {
    #[inline]
    pub fn get(&self, which: usize, n: usize) -> f64 {
        if n < POW_TABLE_LEN { self.t[which][n] } else { POW_BASES[which].powi(n as i32) }
    }
}
pub fn pow_tables() -> &'static PowTables {
    static T: std::sync::OnceLock<PowTables> = std::sync::OnceLock::new();
    T.get_or_init(|| PowTables { t: POW_BASES.map(|b| { let mut v = Vec::with_capacity(POW_TABLE_LEN); let mut x = 1.0f64; for _ in 0..POW_TABLE_LEN { v.push(x); x *= b; } v }) })
}
