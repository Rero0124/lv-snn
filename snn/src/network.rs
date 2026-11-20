use crate::neuron::{CellType, Neuron, NeuronId};
use crate::plasticity::{self, PlasticityOptions};
use crate::region::RegionType;
use crate::tokenizer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── 상수 ──

const INITIAL_WEIGHT: f64 = 0.2;
const MIN_WEIGHT: f64 = 0.1;
const EMOTION_COUNT: usize = 5000;
const REASON_COUNT: usize = 5000;
const MEMORY_COUNT: usize = 2000;
const HIPPOCAMPUS_COUNT: usize = 500;
/// 자극 하나를 관찰하는 최대 틱 수 (fire() 무한 루프 방지)
const MAX_STIMULUS_TICKS: u64 = 2_000;

// ── 수면 파라미터 ──
const SLEEP_FIRE_INTERVAL: u64 = 30_000;
const SLEEP_TICK_INTERVAL: u64 = 500_000;
const SLEEP_DURATION_TICKS: u64 = 1_000;
const SLEEP_WEIGHT_SCALE: f64 = 0.92;

// ── 네트워크 구성 파라미터 ──

/// 배선/가중치 구성. Default = 기존 구성, connectome() = 실제 커넥톰 비율에 맞춘 구성.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NetParams {
    /// 은닉 뉴런당 나가는 시냅스 수 (기존 10)
    pub synapses_per_neuron: usize,
    /// 은닉 뉴런이 내보내는 시냅스의 초기 가중치에 곱하는 배율. 연결 수를 10배로 늘리면 0.1 근처가 같은 총 입력량
    pub hidden_gain: f64,
    /// 초기 가중치를 꼬리 긴 분포(로그정규)로: 약한 연결 다수 + 강한 연결 소수
    pub lognormal: bool,
    /// 감정/이성 영역 억제성 비율 (기존 0.15)
    pub inhibitory_ratio: f64,
    /// 입력 뉴런 하나가 대표하는 감각뉴런 수 (입력 시냅스 수 = synapses_per_neuron × input_pop)
    pub input_pop: usize,
    /// prune 기준 가중치 (기존 0.1)
    pub min_weight: f64,
    /// Main 뉴런의 시냅스 중 출력 뉴런으로 가는 비율 (기존 0.3). 출력 뉴런이 80개뿐이라
    /// 이 값이 크면 출력 하나가 수천 개 입력을 받는 초대형 허브가 된다
    pub output_frac: f64,
    /// 세포 종류 구분 사용: 흥분성은 추체세포, 억제성은 Pv 45% / Sst 40% / Vip 15% 로 나누고,
    /// 억제성 뉴런은 자기 영역 안에서만(국소) 종류별 규칙대로 연결한다
    pub typed: bool,
    /// 초파리 커넥톰에서 검증한 학습 구성을 쓴다 (fly/ 의 --validated 와 같은 규칙):
    ///   STDP·iSTDP 의 변화량과 soft bound 를 뉴런별 지역 값으로, LTD 는 post 발화당 한 번,
    ///   억제 시냅스에는 일반 STDP 를 겹쳐 걸지 않음, 교과서 BCM(목표 = 자기 시냅스 활성화 빈도 중앙값),
    ///   항상성은 뉴런별 목표, 노이즈는 자기 입력 가중치 중앙값의 1/10, 자발 발화 없음.
    ///   sprout / prune / 수면은 검증되지 않아 이 모드에서는 끈다
    pub validated: bool,
    /// 형태: 추체세포를 세포체 + 수상돌기 두 구획으로 (typed 필요).
    ///   다른 영역에서 오는 흥분성 입력과 Sst 의 억제는 수상돌기로, 같은 영역 입력과 Pv 의 억제는 세포체로
    pub morph: bool,
    /// 도파민 뉴런 사용: 기준선 발화를 하는 조절 뉴런 집단 + 보상/처벌 입력 뉴런을 만든다.
    ///   feedback 은 숫자를 곱하는 대신 보상(또는 처벌) 입력 뉴런을 발화시키고,
    ///   학습은 받는 뉴런의 조절 수준이 기준선에서 벗어난 만큼(위 = 강화 방향, 아래 = 반대 방향) 일어난다
    pub dopamine: bool,
    /// 도파민 방식에서, 조절 신호 없이도 평소에 가중치로 넘어가는 흔적의 비율 (초파리 벤치에서 2~10% 가 적정)
    pub gate_base: f64,
    /// 도파민 방식에서, 기준선 대비 변화 1 당 매 틱 가중치로 넘어가는 흔적의 비율
    pub da_gain: f64,
    /// 보상 예측 경로 (dopamine 필요): 은닉 → 예측 뉴런(억제성) → 도파민 뉴런.
    ///   보상이 반복되면 그 자극에 예측 뉴런이 반응하게 되고, 예측 뉴런은 도파민 뉴런으로 들어오는 보상 입력을 상쇄한다
    ///   → 이미 예측된 보상에는 도파민이 덜 나오고 학습도 줄어든다
    pub predict: bool,
    /// 예측 뉴런 입력 시냅스의 학습 크기
    pub pred_eta: f64,
    /// 1대1 방식(fire)에서 "반응이 끝났다"의 기준을 출력 뉴런으로만 본다.
    ///   false(기존): 네트워크 전체가 10틱 조용해야 끝.  true: 출력 뉴런이 10틱 조용하면 끝 — 내부 활동은 계속되고 있어도 된다
    pub end_on_output: bool,
    /// 자발 발화 배수 (은닉 뉴런). None = 기존 동작(기본 구성 1.0, 검증 구성 0).
    ///   뉴런 하나가 입력 없이 한 틱에 발화할 확률 = 0.00001 × 배수 (자극이 들어오는 동안은 그 1/10)
    ///   spont_fixed 가 있으면 이 배수는 쓰이지 않는다. spont_flat 은 여기에 곱해지는 종류별 배수를 끈다
    pub spont: Option<f64>,
    /// 자발 발화 확률을 입력 가중치와 무관하게 고정 (실험용). Some(P) 이면 은닉 뉴런이 한 틱에 스스로 발화할 확률 = P × 종류 배수.
    ///   주면 spont 의 배수와 검증 구성의 "입력 가중치 중앙값 ÷ 0.2" 는 쓰지 않는다 (접점이 늘거나 세기가 달라져도 씨앗이 그대로다)
    pub spont_fixed: Option<f64>,
    /// 종류별 자발 배수(Pv 10, Sst 5, Vip 3)를 끈다: 모든 은닉 뉴런이 추체와 같은 확률로 스스로 발화한다 (실험용).
    ///   spont 나 spont_fixed 가 없으면(자발 발화가 꺼져 있으면) 효과가 없다
    pub spont_flat: bool,
    /// 새 접점 세기 기준(unit_med)을 구할 때 조절(도파민) 뉴런에서 오는 입력을 뺀다 (실험용).
    ///   false(기존): 거름 조건이 받는 뉴런 자신을 봐서 은닉 뉴런에 대해서는 아무것도 거르지 않는다.
    ///   발달 과정(grow)이 꺼져 있으면 이 기준을 계산하지 않으므로 효과가 없다
    pub unit_med_fix: bool,
    /// 입력의 자리 배치: 입력 뉴런마다 각 영역(감정·이성·기억)에 정해진 자리가 있고, 그 주변(표준편차 TOPO_SIGMA)으로만 연결한다.
    ///   false(기존): 영역 전체에 무작위. true: 같은 입력을 받는 뉴런들이 서로 이웃이 된다
    pub topo: bool,
    /// 영역 내부 연결 중 가까운 뉴런(표준편차 LOCAL_SIGMA)으로 가는 비율. 0(기존) = 영역 전체에 무작위
    pub local_frac: f64,
    /// 은닉 영역 크기 배수 (1.0 = 감정 5000 / 이성 5000 / 기억 2000 / 해마 500). 도파민 뉴런 수도 같이 늘어난다
    pub scale: f64,
    /// 발달 과정 (Some(발화 한 번당 성장 기본 확률)). 성장과 가지치기를 켠다.
    ///   성장: 발화한 은닉 뉴런이 (기본 확률 × 의욕)개의 새 시냅스를 가까운 뉴런에 만든다. 의욕 = (target_syn ÷ 지금 시냅스 수)²
    ///   가지치기: 연결마다 기한이 예약된다. 기한 = 마지막으로 함께 활동한 틱 + 수명, 수명 = prune_tau × ln(가중치 ÷ min_weight).
    ///            기한까지 함께 활동하지 않으면 접점이 하나 사라지고(0개가 되면 연결이 사라진다), 가중치는 깎지 않는다
    pub grow: Option<f64>,
    /// 뉴런당 내보내는 시냅스 수의 목표 (발달 과정에서 성장 의욕의 기준)
    pub target_syn: usize,
    /// 가지치기 수명의 시간 단위 (틱). 가중치가 min_weight 의 e 배인 연결이 함께 활동하지 않고 버티는 시간
    pub prune_tau: f64,
    /// 성장할 때 가까운 뉴런을 고르는 범위 (격자 칸 단위 표준편차). 뉴런당 시냅스가 많을수록 넓어야 고를 상대가 남는다
    pub grow_sigma: f64,
    /// 새로 생기는 접점의 세기 = 받는 뉴런의 입력 가중치 중앙값(접점 하나 기준) × 이 값. 약하게 시작해서 쓰이면 강해진다
    pub new_weight: f64,
    /// 새 접점 세기의 퍼짐: 위 값의 (1 − new_spread) ~ (1 + new_spread) 배 사이에서 고르게 뽑는다
    pub new_spread: f64,
    /// 성장을 도파민에 맡기는 정도 (None = 항상 같은 속도로 성장). Some(b) 이면 성장 속도 = 기본 확률 × (b + 그 뉴런의 조절 수준이 기준선보다 높은 정도).
    ///   뜻밖의 보상이 많은 초기에는 많이 자라고, 보상이 예측되어 도파민이 줄면 성장도 줄어든다 → 가지치기가 따라잡아 접점이 내려온다
    pub grow_da: Option<f64>,
    /// 접점끼리의 경쟁 (0 = 없음). 받는 뉴런이 과하게 활동해서 스스로 감응도를 낮춘 상태면, 그 뉴런으로 들어오는 굳지 않은 접점의
    /// 수명이 짧아진다 (수명을 계산할 때 가중치에 감응도^compete 를 곱한다). 접점이 적은 초기에는 전부 버티고,
    /// 입력이 넘치면 받는 뉴런이 줄 수 있는 몫이 줄어 굳지 못한 것부터 사라진다. 기준은 개수가 아니라 그 뉴런 자신의 활동이다
    pub compete: f64,
    /// 일시적인 변화 방식 (dopamine 필요): STDP 의 변화가 이만큼의 틱에 걸쳐 사라진다 (시간 상수). 도파민이 와야 오래가는 변화가 된다.
    /// None 이면 기존 방식(평소에도 gate_base 비율이 영구적으로 넘어감)
    pub short_tau: Option<f64>,
    /// 가치 경로 (dopamine 필요): 은닉 → 가치 뉴런(흥분성) → 도파민 뉴런.
    ///   보상이 반복되면 그 직전에 활동한 뉴런들의 가치 뉴런 입력이 강해져서, 다음부터는 그 자극이 들어오는 순간 도파민이 나온다.
    ///   예고 자극이 스스로 도파민을 일으키므로, 그 자극을 예고하는 더 앞의 자극도 학습된다 (2차, 3차)
    pub value: bool,
    /// 가치 뉴런 입력 시냅스의 학습 크기 (0 이면 pred_eta 와 같게)
    pub val_eta: f64,
    /// 가치 뉴런 → 도파민 뉴런, 예측 뉴런 → 도파민 뉴런 시냅스의 세기.
    /// 예측 쪽이 가치 쪽보다 훨씬 세면, 가치가 조금만 생겨도 보상이 다 상쇄되어 가치가 더 자라지 못한다
    pub v2da: f64,
    /// None 이면 구조에 맞는 기본값: 예측 뉴런이 은닉에서 직접 입력을 받으면 0.5, 가치 뉴런에서 받으면 0.1
    pub p2da: Option<f64>,
    /// 1차 보상과 몸의 상태 (dopamine 필요): 단맛(먹이)·물 감각 입력과 배고픔·목마름 뉴런을 만든다.
    ///   감각 입력은 중계 뉴런을 거쳐 도파민 뉴런을 흥분시키는데, 중계 뉴런은 해당 결핍 뉴런의 입력이 함께 있어야 발화한다
    ///   → 배고플 때만 먹이가 보상이 된다. 중계 뉴런이 발화하면 포만 뉴런을 거쳐 결핍이 내려간다
    pub needs: bool,
    /// 확률적 전달: 은닉 뉴런이 내보내는 시냅스가 이 확률로만 전달한다 (None = 항상 전달).
    /// 평균 전달량은 같고, 같은 입력에도 시행마다 전달되는 조합이 달라진다
    pub release: Option<f64>,
    /// 배선 씨앗. Some(N) 이면 초기 배선(은닉·입출력 연결, 도파민·가치·예측·몸 배선, 초기 접점 기한 흩음)의 난수를 N 에서 뽑는다.
    ///   보장하는 것은 초기 배선까지다: 워밍업과 실행 중의 난수(잡음, 자발 발화, 확률 전달, 성장)에는 씨앗이 없다
    pub seed: Option<u64>,
    /// 처음부터 있던 접점의 기한을 흩는다 (발달 과정): 시계를 "워밍업 끝 + (0~1 난수) × 자기 수명" 으로 둔다.
    ///   false(기존): 모두 워밍업 끝에 맞춰져, 수명이 같은 접점들의 기한이 한꺼번에 온다
    pub scatter_co: bool,
    /// 활동 목표의 배수 K: 뉴런의 목표 발화율 = K × 자기 자발 발화 확률(0.00001 × spont_scale). 받기 문과 활동 의존 소멸이 쓴다.
    ///   자발 발화 확률이 0 인 뉴런에는 문도 소멸도 걸지 않는다
    pub act_target: Option<f64>,
    /// 받기 문 (발달 과정의 성장): 받는 뉴런의 발화율이 목표를 넘으면 그 뉴런으로 가는 새 접점 시도를 버린다 (다른 상대로 넘기지 않는다).
    ///   Fast = 빠른 발화율(rate_fast, 시정수 약 100틱), Slow = 느린 발화율(rate_avg, 약 5000틱), Both = 둘 중 하나라도
    pub grow_gate: Option<GateMode>,
    /// 받기 문을 차츰 줄이는 문으로 (지수 K). 기본 Some(4.0). None = 옛 닫힘/열림 문(--grow-gate-hard): 받는 뉴런의 발화율이 목표를 넘으면 시도를 모두 버린다.
    ///   Some(K): 모두 버리는 대신 확률 (목표 ÷ 발화율)^K 로 받는다 (목표 이하면 1). 발화율은 grow_gate 가 고른 것 — Fast = 빠른, Slow = 느린, Both = 둘 중 큰 쪽
    ///   (K 가 아주 크면 옛 문과 같아진다). grow_gate 가 없으면 효과가 없다.
    ///   K = 4 는 3(묶지 못함)과 6(입력을 잃는 추체가 덜 고쳐짐) 사이에서 대조로 고른 값이다. 빠른 발화율이 묶는 힘의 한 몫인 것은 뇌 근거가 아니라 계산의 장치다
    ///   (판정: lv-snn-runs/review-session/softgate-verdict.md).
    ///   이 칸이 없는 옛 저장 파일은 None 으로 읽는다(저장한 때의 동작 = 옛 문 그대로 이어진다)
    #[serde(default = "gate_soft_missing")]
    pub grow_gate_soft: Option<f64>,
    /// 성장이 출력 뉴런도 상대로 삼는다 (기본 꺼짐. 이 칸이 없는 저장 파일은 꺼짐으로 읽는다).
    ///   발화한 은닉 흥분 뉴런이 처음 배선에서 출력으로 보내던 영역(감정 주·이성 주: output_frac, 기억: output_frac × 2/3)에 있으면,
    ///   은닉으로 가는 시도에 얹어서(줄이지 않는다) 성장 빠르기 × 그 몫만큼 출력 뉴런 하나(거리 없이 고르게 무작위)에 새 접점을 시도한다.
    ///   받는 출력 뉴런에는 은닉과 같은 받기 문(grow_gate, grow_gate_soft)을 걸고, 목표는 추체의 목표(act_target × spont_fixed)를 쓴다 —
    ///   출력은 자발 발화 배수가 0 이라 자기 목표가 없다. 바깥에서 준 목표가 출력에도 쓰이는 것이다. 새 상수는 없다
    ///   (명세: lv-snn-runs/review-session/output-growth-spec.md)
    pub grow_out: bool,
    /// 가지치기 기한을 다시 세는 조건에 바닥을 둔다 (기본 켜짐. --prune-reset-floor-off 로 끈다. 이 칸이 없는 옛 저장 파일은 꺼짐으로 읽는다 —
    ///   바닥 없이 기른 망을 이어 돌릴 때 값이 바뀌지 않게. 불러온 망은 저장 파일의 설정을 따른다).
    ///   도파민으로 강화 쪽이 굳어도, 굳은 뒤의 가중치가 최소 가중치(min_weight)보다 작으면 기한을 다시 세지 않는다 —
    ///   바닥 아래로 내려간 연결이 아주 작은 굳음마다 기한을 새로 받아 끝없이 남는 것을 막는다. 그런 연결은 마지막으로 기한을 받은 뒤 2만 틱(가장 짧은 수명) 안에 떨어진다.
    ///   값은 은닉 뉴런의 reset_min 에 넣는다 (켜면 min_weight, 끄면 0). 새 상수는 없다 (명세: lv-snn-runs/review-session/prune-hole-spec.md)
    #[serde(default = "reset_floor_missing")]
    pub prune_reset_floor: bool,
    /// 뉴런마다 자기 값으로 정하는 목표 (기본 없음. 이 칸이 없는 저장 파일은 없음으로 읽는다).
    ///   All = 은닉 뉴런 모두, Pyr = 추체만(억제 뉴런은 act_target × 자기 자발 그대로). 켜진 뉴런(own_on)은 자기 발화 열일곱 번에 걸린 틱으로
    ///   목표를 재어 굳히고(neuron.rs 의 own_*), 받기 문·prune_act·scale_in·관찰 줄이 act_target_of 로 그 값을 읽는다.
    ///   재는 동안(own_target == 0) 그 뉴런으로 가는 새 접점 시도는 버린다. 감응도 항상성과 출력 뉴런은 그대로다
    ///   (명세: lv-snn-runs/review-session/self-target-spec.md)
    pub own_target: Option<OwnMode>,
    /// 자기 목표를 재는 동안에도 새 접점을 받는다 (기본 꺼짐 = 재는 동안 그 뉴런으로 가는 시도를 버린다. 이 칸이 없는 저장 파일은 꺼짐).
    ///   켜면 재는 중인 뉴런은 목표가 아직 0 이라 받기 문 없이 받는다. own_target 이 없으면 효과가 없다
    pub own_target_open: bool,
    /// (시험용) 입력 뉴런 하나가 가는 은닉 흥분 뉴런의 수 F: 주면 감정 주·이성 주·기억의 흥분 뉴런 가운데 F 개를 **되풀이 없이** 고른다.
    ///   없으면 지금 길(synapses_per_neuron × input_pop 번, 되풀이 허락). 처음 배선에만 쓰인다. --topo(발달 구성)에서는 자리 둘레로 고르는 법은 그대로 두고 수만 F 로 바꾼다(영역에 크기대로 나눔).
    ///   명세: lv-snn-runs/review-session/sparse-inhibit-spec.md ㉠. 바깥에서 준 값이다
    #[serde(default)]
    pub in_fan: Option<usize>,
    /// (시험용) 입력 시냅스의 처음 세기 w: 주면 모든 입력 시냅스가 이 값으로 시작한다(없으면 지금의 넓은 분포).
    ///   문턱(0.5)에 매인 값이다 — "글자 하나로는 아래, 둘이면 위"(명세 값 0.22). 문턱이 바뀌면 함께 바뀌어야 한다. 바깥에서 준 값이다
    #[serde(default)]
    pub in_weight: Option<f64>,
    /// 활동 의존 소멸의 지수 E (0 = 없음): 받는 뉴런의 느린 발화율이 목표보다 높으면, 받는 쪽 점검에서 그 뉴런으로 들어오는 접점의
    ///   수명에 (목표 ÷ 느린 발화율)^E 를 곱한다 (최소 수명 2만 틱을 적용한 뒤에 곱한다). 목표 이하면 기존 그대로
    pub prune_act: f64,
    /// 지연 시냅스를 "발화했을 때만 알리기"로 계산한다: 예측 뉴런마다 입력의 합(Σ 가중치 × 흔적)을 들고 매 틱 그 합만 진행시키고,
    ///   학습은 조절 편차가 문턱을 넘은 받는 뉴런의 "흔적이 살아 있는 입력"만 본다. 규칙은 같고, 더하는 순서와
    ///   아주 작은 흔적(1e-4 이하)을 버리던 것만 다르다. false(기존): 매 틱 지연 시냅스 전체를 두 번 훑는다
    pub slow_events: bool,
    /// slow_events 의 확인: 옛 방식(전체 훑기)의 값도 함께 계산해서 예측 뉴런이 받은 양과 학습된 가중치의 차이를 1000틱마다 찍는다
    pub slow_check: bool,
    /// 기준선을 잡는 준비 구간(2,300틱) 동안만 은닉 뉴런의 자발 발화를 끈다. 준비 뒤에 다시 건다.
    ///   예측 경로(predict)를 쓰는 망에서는 기본으로 켜진다 (no_warmup_quiet 로 끈다). 예측을 쓰지 않으면 주어졌을 때만.
    ///   꺼져 있으면: 자발 발화(spont, spont_fixed)가 주어져 있을 때 준비 구간도 자발 발화가 켜진 채로 돈다
    pub warmup_quiet: bool,
    /// 예측 경로를 쓰는 망에서도 준비 구간의 자발 발화를 끄지 않는다 (옛 동작)
    pub no_warmup_quiet: bool,
    /// 깨어남 확인 (동작은 바꾸지 않는다): "입력을 받았거나, 불응기가 끝났거나, 바깥 자극을 받은 뉴런만 판정하면 된다"는 규칙이 맞는지를
    ///   틱마다 세어 1000틱마다 찍는다. 깨어나지 않았는데 발화한 뉴런과 그 이유, 깨어나지 않았는데 전위가 달라진 뉴런을 센다
    pub wake_check: bool,
    /// 저장된 망을 불러온다 (Some(경로)): 망을 새로 만들지 않고 그 파일에서 읽는다. 설정은 저장된 것을 쓰고(명령줄의 망 옵션은 무시),
    ///   기준선을 잡는 준비 구간은 돌리지 않는다. 계산 방식 옵션(slow_events, slow_check, wake_check)만 명령줄의 것을 따른다
    #[serde(skip)]
    pub load_from: Option<String>,
    /// 1만 틱마다의 항상성에서 몸의 상태 뉴런(맛·필요·중계·포만)을 뺀다. 기본 true.
    ///   false(옛 동작, --homeo-need): 이 뉴런들에도 항상성을 건다. 그러면 처음 1만 틱에 조용했던 뉴런의 평소 발화율이 0 으로 잡혀,
    ///   그 뒤 발화할 때마다 감응도가 1만 틱에 2% 씩 깎이고 보상 한 번이 결핍을 줄이는 양이 시간이 갈수록 준다
    pub no_homeo_need: bool,
    /// 1만 틱마다의 항상성에서 조절 쪽 뉴런(도파민·가치·예측·보상·처벌 입력)을 뺀다. false(기존): 모든 뉴런에 건다
    pub no_homeo_da: bool,
    /// 시험용: 도파민 뉴런의 수상돌기 기여를 0 아래로 내리지 않는다 (예측 억제가 평소 발화를 누르지 못한다. 빠진 보상의 내려감도 없어진다 — 뇌와 다르다)
    pub pred_floor: bool,
    /// 입력의 세기를 고르는 규칙 (None = 끔). Some(η): 1만 틱마다 은닉의 추체마다 오차 = clamp(느린 발화율 ÷ 받기 문의 목표 − 1, ±1) 를 내어,
    ///   그 뉴런으로 들어오는 흥분 시냅스(은닉의 흥분 뉴런과 입력층에서 오는 것)의 굳은 가중치에 (1 − η × 오차)를 곱한다. 오차의 절댓값이 0.1 아래면 하지 않는다.
    ///   억제·조절·지연 시냅스와 일시 성분·흔적은 건드리지 않는다. act_target(목표)이 있어야 한다
    pub scale_in: Option<f64>,
    /// scale_in 에서 입력층에서 오는 시냅스를 뺀다 (은닉의 흥분 뉴런에서 오는 것만 곱한다)
    pub scale_in_hidden_only: bool,
    /// 1만 틱마다의 항상성(감응도)에서 은닉 뉴런을 뺀다 (대조용: 입력 세기 조절만으로 발화율이 묶이는지)
    pub no_homeo_hidden: bool,
    /// 결정적 난수의 씨앗 (None = 지금처럼 전역 난수기). Some(N) 이면 실행 중의 난수(잡음, 자발 발화, 확률 전달, 성장)를
    ///   (N, 쓰임새, 뉴런, 시냅스 번호, 틱)에서 정한다. 배선 씨앗(seed)이 없으면 배선도 N 으로 한다
    pub rng_seed: Option<u64>,
}

/// wake_check 의 상태와 세는 값 (관찰용)
#[derive(Default)]
struct WakeCheck {
    /// 뉴런별 표시: 1 = 전위 변화를 받음 / 바깥 자극, 4 = 불응기가 이 틱에 끝남, 2 = 조절 신호를 받음.
    /// 첫째 규칙 = 셋 중 하나, 둘째 규칙 = 1 또는 4 (조절 신호 제외), 셋째 규칙 = 실행 큐(queue)에 있음. now 는 이번 틱, next 는 다음 틱의 것
    now: Vec<u8>,
    next: Vec<u8>,
    /// 지난 틱 발화 판정 직후의 전위, 이번 틱 시작 때의 전위, 감쇠 직후의 전위, 판정 직전의 (전위 + 수상돌기 기여), 넘어야 하는 문턱(상대 불응기면 2배, 절대 불응기면 무한대)
    pot_end: Vec<f64>,
    pot_begin: Vec<f64>,
    pot_decay: Vec<f64>,
    base: Vec<f64>,
    bar: Vec<f64>,
    refr: Vec<bool>,
    started: bool,
    /// 구간과 누계: [뉴런·틱 수, 깨어난 수(조절 포함), 깨어난 수(조절 제외), 발화 수, 놓친 발화(조절 포함 규칙), 놓친 발화(조절 제외 규칙),
    ///  이유: 자발 발화, 잡음, 지속 입력(tonic)이 있는 뉴런, 그 밖, 그 밖 가운데 불응기 안, 놓친 전위 변화, 조절 신호만 받고(깨어나지 않고) 발화한 수,
    ///  그 발화의 이유: 자발 발화, 잡음, 지속 입력, 그 밖, 그 밖 가운데 불응기 안,
    ///  셋째 규칙: 큐에 있던 수, 놓친 발화, 그 이유(자발 발화, 잡음, 지속 입력, 그 밖, 그 밖 가운데 불응기 안)]
    /// 기준선을 잡는 준비 구간(2,300틱)은 idle_tick 으로 돌므로 여기에 들어오지 않는다
    acc: [u64; 25],
    total: [u64; 25],
    /// 셋째 규칙의 실행 큐: 깨어난 뉴런은 (전위 + 수상돌기 기여 + 잡음 폭 < 적응을 0 으로 놓은 문턱)이 될 때까지 남는다
    queue: Vec<bool>,
}

/// 자기 값으로 정하는 목표를 쓰는 뉴런: 은닉 모두 / 추체만
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OwnMode { All, Pyr }

/// 받기 문이 보는 발화율
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateMode { Fast, Slow, Both }

/// 발달 시험의 관찰값 (읽기만 한다)
pub struct DevObs {
    /// 은닉 뉴런이 내보내는 접점 수: 처음부터 있던 쌍(born = 0)의 접점, 그중 은닉→은닉(지연 시냅스 제외), 자란 쌍(born > 0)의 접점
    pub init_contacts: usize,
    pub init_hh: usize,
    pub grown_contacts: usize,
    /// 쌍(연결) 수: 처음부터 있던 것, 자란 것
    pub init_pairs: usize,
    pub grown_pairs: usize,
    /// 은닉→은닉 시냅스(지연 시냅스 제외) 가운데, 같은 (보내는 뉴런, 받는 뉴런) 쌍의 두 번째 이상인 것의 수와 그 시냅스 칸의 전체 수
    pub dup_syn: usize,
    pub hh_syn: usize,
    /// 은닉 뉴런마다: (종류 0 추체·1 Pv·2 Sst·3 Vip·4 그 밖, 영역 0~5, 빠른 발화율, 느린 발화율, 목표(틱당. 목표가 없으면 0),
    ///   은닉 흥분성 뉴런에서 들어오는 접점 수, 은닉 억제성 뉴런에서 들어오는 접점 수)
    pub per: Vec<(usize, usize, f64, f64, f64, usize, usize)>,
}

impl Default for NetParams {
    fn default() -> Self {
        Self { synapses_per_neuron: 10, hidden_gain: 1.0, lognormal: false, inhibitory_ratio: 0.15, input_pop: 1, min_weight: MIN_WEIGHT, output_frac: 0.3, typed: false, validated: false, morph: false, dopamine: false, gate_base: 0.02, da_gain: 0.05, predict: false, pred_eta: 5e-5, end_on_output: false, spont: None, spont_fixed: None, spont_flat: false, unit_med_fix: false, topo: false, local_frac: 0.0, scale: 1.0, grow: None, target_syn: 2000, prune_tau: 95_000.0, grow_sigma: 12.0, new_weight: 1.0, new_spread: 0.5, grow_da: None, compete: 0.0, short_tau: None, value: false, val_eta: 0.0, v2da: VALUE_TO_DA_WEIGHT, p2da: None, needs: false, release: None, seed: None, scatter_co: false, act_target: None, grow_gate: None, grow_gate_soft: Some(4.0), grow_out: false, in_fan: None, in_weight: None, prune_reset_floor: true, own_target: None, own_target_open: false, prune_act: 0.0, slow_events: true, slow_check: false, warmup_quiet: false, no_warmup_quiet: false, wake_check: false, load_from: None, no_homeo_need: true, no_homeo_da: false, pred_floor: false, scale_in: None, scale_in_hidden_only: false, no_homeo_hidden: false, rng_seed: None }
    }
}

impl NetParams {
    /// 커넥톰 비율 (초파리 성체 기준): 뉴런당 연결 ~100, 연결 세기 중앙값:평균 ≈ 2:3.6 의 꼬리 긴 분포,
    /// 가장 강한 입력 하나가 전체 입력의 10~30%, 억제성 뉴런 30%
    pub fn connectome() -> Self {
        Self { synapses_per_neuron: 100, hidden_gain: 0.1, lognormal: true, inhibitory_ratio: 0.30, input_pop: 1, min_weight: 0.002, output_frac: 0.10, typed: false, validated: false, morph: false, dopamine: false, gate_base: 0.02, da_gain: 0.05, predict: false, pred_eta: 5e-5, end_on_output: false, spont: None, spont_fixed: None, spont_flat: false, unit_med_fix: false, topo: false, local_frac: 0.0, scale: 1.0, grow: None, target_syn: 2000, prune_tau: 95_000.0, grow_sigma: 12.0, new_weight: 1.0, new_spread: 0.5, grow_da: None, compete: 0.0, short_tau: None, value: false, val_eta: 0.0, v2da: VALUE_TO_DA_WEIGHT, p2da: None, needs: false, release: None, seed: None, scatter_co: false, act_target: None, grow_gate: None, grow_gate_soft: Some(4.0), grow_out: false, in_fan: None, in_weight: None, prune_reset_floor: true, own_target: None, own_target_open: false, prune_act: 0.0, slow_events: true, slow_check: false, warmup_quiet: false, no_warmup_quiet: false, wake_check: false, load_from: None, no_homeo_need: true, no_homeo_da: false, pred_floor: false, scale_in: None, scale_in_hidden_only: false, no_homeo_hidden: false, rng_seed: None }
    }
}

// ── 발화 기록 ──

pub struct DeliveryTrace {
    pub from: NeuronId,
    pub to: NeuronId,
    pub from_potential: f64,
    pub to_before: f64,
    pub weight: f64,
    pub delivered: f64,
    pub tick: u64,
}

pub struct FireRecord {
    pub id: u64,
    pub output: String,
    pub output_tokens: Vec<(String, f64)>,
    pub spiked_neurons: Vec<(NeuronId, u64)>,
    pub traces: Vec<DeliveryTrace>,
}

pub struct ActiveStimulus {
    pub fire_id: u64,
    pub input_indices: Vec<usize>,
    pub remaining_sustain: u64,
    pub elapsed_ticks: u64,
    pub silent_ticks: u64,
    pub all_spikes: Vec<(NeuronId, u64)>,
    pub output_spikes: Vec<(NeuronId, u64)>,
    pub traces: Vec<DeliveryTrace>,
    pub collect_traces: bool,
}

// ── 네트워크 ──

pub struct Network {
    neurons: Vec<Neuron>,
    vocab: HashMap<String, u32>,
    reverse_vocab: Vec<String>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    emotion_range: (usize, usize),
    emotion_main_range: (usize, usize),
    emotion_map_range: (usize, usize),
    reason_range: (usize, usize),
    reason_main_range: (usize, usize),
    reason_map_range: (usize, usize),
    memory_range: (usize, usize),
    hippocampus_range: (usize, usize),

    pub fire_records: Vec<FireRecord>,
    next_fire_id: u64,
    pub debug: bool,
    pub active_stimuli: Vec<ActiveStimulus>,
    recent_spikes: Vec<(NeuronId, u64)>,
    pub threshold: f64,
    pub noise_range: f64,
    global_tick: u64,
    fires_since_sleep: u64,
    last_sleep_tick: u64,
    plasticity: PlasticityOptions,
    params: NetParams,
    /// 뉴런별 노이즈 크기 (검증된 구성에서만 사용. 비어 있으면 전역 noise_range)
    noise_amp: Vec<f64>,
    /// 도파민 뉴런 / 보상 입력 뉴런 / 처벌 입력 뉴런의 인덱스 (도파민 방식에서만 채워짐)
    da_idx: Vec<usize>,
    reward_idx: Vec<usize>,
    punish_idx: Vec<usize>,
    /// 보상 예측 뉴런의 인덱스 (predict 일 때만)
    pred_idx: Vec<usize>,
    /// 1차 보상 종류별 (0 = 먹이, 1 = 물): 감각 입력 뉴런들, 결핍 뉴런들
    taste_idx: Vec<Vec<usize>>,
    need_idx: Vec<Vec<usize>>,
    /// 도파민·보상·예측·가치·1차 보상 관련 뉴런이 차지하는 인덱스 범위의 끝 (반응으로 세지 않는 뉴런들)
    aux_end: usize,
    /// 가치 뉴런의 인덱스 (value 일 때만), 지금까지의 가치 뉴런 발화 수 (관찰용)
    val_idx: Vec<usize>,
    pub val_spikes: u64,
    /// 지금까지의 도파민 뉴런 발화 수 합계 (관찰용)
    pub da_spikes: u64,
    /// 지금까지의 예측 뉴런 발화 수 합계 (관찰용)
    pub pred_spikes: u64,
    /// 연속 모드의 입력: (켜 둘 입력 뉴런들, 남은 틱 수). 자극의 시작·끝을 판정하지 않고 정해진 틱 동안 켜 두기만 한다
    drives: Vec<(Vec<usize>, u64)>,
    /// 지연 시냅스의 위치 (보내는 뉴런, 시냅스 번호)
    slow_syn: Vec<(usize, usize)>,
    /// 활동 기반 발아로 만든 시냅스 수 (관찰용)
    pub sprouted: u64,
    /// 발달 과정의 가지치기로 사라진 시냅스 수 (관찰용)
    pub pruned: u64,
    /// 뉴런별: 자기 입력 연결들의 "접점 하나의 세기"(가중치 ÷ 접점 수) 중앙값. 새 접점의 세기를 정할 때 쓴다
    unit_med: Vec<f64>,
    /// 방금 틱에 발화한 뉴런 (도파민·보상·예측 뉴런 제외). 연속 모드에서 출력을 발화하는 대로 읽을 때 쓴다
    pub tick_ids: Vec<NeuronId>,
    /// 초기 배선의 합계 값 (보내는 뉴런, 받는 뉴런, 세기 등을 차례로 섞은 값). 같은 배선 씨앗이면 같아야 한다 (관찰용)
    pub wiring_sum: u64,
    /// 받는 쪽 종류별(추체, Pv, Sst, Vip, 그 밖) 누계: 생긴 접점 수, 받기 문에 걸려 버린 시도 수 (관찰용)
    pub grow_made: [u64; 5],
    pub gate_drop: [u64; 5],
    /// 받는 뉴런이 자기 목표를 재는 중이라 버린 새 접점 시도의 누계 (gate_drop 에도 들어간다)
    pub own_wait_drop: u64,
    /// 처음부터 있던 쌍(born = 0)에 더해진 접점 수 누계 (관찰용)
    pub added_on_initial: u64,
    /// 받는 쪽 점검에서 사라진 접점 중, 활동 의존 소멸로 수명이 줄지 않았다면 아직 남았을 접점 수 누계 (관찰용)
    pub pruned_short: u64,
    /// 사라진 접점이 속한 쌍의 나이(사라진 틱 − 쌍이 생긴 틱) 목록. 읽는 쪽(dev_test)이 비운다. 나이를 가릴 수 없던 수는 prune_age_unknown (관찰용)
    pub prune_ages: Vec<u64>,
    pub prune_age_unknown: u64,
    /// slow_events: 지연 시냅스를 받는 뉴런들(가치, 예측 순), 뉴런 번호 → 그 목록에서의 자리(없으면 u32::MAX),
    /// 받는 뉴런마다 [Σ 가중치 × slow1, Σ 가중치 × slow2, 지연 비율](가치 뉴런은 전달을 바로 하므로 쓰지 않는다: 지연 비율 0),
    /// 받는 뉴런마다 흔적이 살아 있는 입력 (보내는 뉴런, 시냅스 번호), 목록에 올라 있는 시냅스의 열쇠
    slow_recv: Vec<usize>,
    slow_pos: Vec<u32>,
    slow_sum: Vec<[f64; 3]>,
    slow_live: Vec<Vec<(u32, u32)>>,
    slow_listed: std::collections::HashSet<u64>,
    /// 저장 파일에서 읽은 slow_events 의 상태 (받는 뉴런별 합, 살아 있는 입력 목록). 불러온 직후 한 번 쓰고 비운다
    slow_saved: Option<(Vec<[f64; 3]>, Vec<Vec<(u32, u32)>>)>,
    /// slow_check: 옛 규칙으로 따로 굴린 지연 시냅스 가중치(slow_syn 과 같은 순서), 구간의 최대 차이들 (받은 양, 다시 맞출 때 합의 어긋남)
    slow_shadow: Vec<f64>,
    slow_init: Vec<f64>,
    slow_chk_recv: f64,
    slow_chk_drift: f64,
    wake: Option<Box<WakeCheck>>,
    /// scale_in 의 관찰값 (읽기만. 저장하지 않는다): 지난 보고 뒤로 [줄인 뉴런·번, 키운 뉴런·번, 아래 한계에 걸린 시냅스·번, 위 한계에 걸린 시냅스·번, 곱한 시냅스·번] 과 뉴런별 누적 배수
    pub scale_obs: [u64; 5],
    pub scale_cum: Vec<f64>,
    /// 발달 시험(dev_test)이 진행한 틱 수. 저장할 때 함께 적어, 불러온 뒤 그 틱부터 이어 간다
    pub dev_tick: u64,
    /// 발달 시험이 들고 있는 자극 상태 (저장 직전에 시험이 넣고, 불러온 뒤 시험이 꺼내 쓴다). 옛 저장 파일에는 없다
    pub dev_state: Option<DevState>,
}

// ── 도파민 구성 ──
/// 도파민 뉴런 수, 받는 뉴런 하나에 닿는 도파민 뉴런 수, 그 시냅스의 세기
const DA_COUNT: usize = 20;
const DA_FANIN: usize = 4;
const DA_WEIGHT: f64 = 0.25;
/// 도파민 뉴런이 스스로 내는 전류와 감쇠: 입력이 없으면 약 34틱에 한 번 발화한다
const DA_TONIC: f64 = 0.02;
const DA_DECAY: f64 = 0.98;
/// 보상 / 처벌 입력 뉴런 수와 도파민 뉴런으로 가는 시냅스 세기
const REWARD_COUNT: usize = 5;
const REWARD_WEIGHT: f64 = 0.2;
const PUNISH_WEIGHT: f64 = 0.06;
/// feedback 세기 1.0 일 때 보상(처벌) 입력을 켜 두는 틱 수, 그 뒤 조절 수준이 기준선으로 돌아오기를 기다리는 틱 수
pub const REWARD_TICKS: u64 = 20;
pub const REWARD_TAIL: u64 = 200;
/// 만든 직후 기준선(mod_base)이 자리 잡도록 돌리는 틱 수
const DA_WARMUP_TICKS: u64 = 2_300;
/// 예측 뉴런 수, 예측 뉴런 하나가 받는 은닉(흥분성) 입력 수와 초기 가중치, 예측 뉴런 → 도파민 뉴런 시냅스 세기
/// 입력 자리 배치의 퍼짐, 가까운 내부 연결의 퍼짐 (격자 칸 단위)
const TOPO_SIGMA: f64 = 3.0;
const LOCAL_SIGMA: f64 = 4.0;
/// 연결 하나(한 상대)의 세기가 접점이 쌓여서 올라갈 수 있는 한계. 문턱(0.5)의 절반이라, 접점이 아무리 쌓여도
/// 연결 하나만으로는 상대를 발화시키지 못한다. 한계에 가까울수록 새 접점이 보태는 양이 줄어든다
const CONN_CAP: f64 = 0.25;
/// 한 쌍(한 상대) 사이에 생길 수 있는 접점 수의 한계
const MAX_CONTACTS_PER_PAIR: u32 = 10;
const PRED_COUNT: usize = 40;
const PRED_FANIN: usize = 1500;
const PRED_INIT_WEIGHT: f64 = 0.001;
const PRED_TO_DA_WEIGHT: f64 = 0.5;
/// 가치 뉴런 수, 가치 뉴런 → 도파민 뉴런 시냅스 세기, 입력 시냅스의 학습 흔적이 가장 커지는 시점(틱)
/// 1차 보상 한 종류를 이루는 집단들의 크기, 결핍이 가득 차는 데 걸리는 틱 수, 시냅스 세기들
const NEED_GROUP: usize = 5;
const NEED_FILL_TICKS: f64 = 20_000.0;
const TASTE_TO_RELAY: f64 = 0.05;   // 감각 입력만으로는 중계 뉴런이 문턱에 못 미친다
const NEED_TO_RELAY: f64 = 0.2;      // 결핍 뉴런의 입력은 중계 뉴런의 수상돌기로 들어가, 충분히 쌓였을 때만 감각 입력이 문턱을 넘게 해 준다
const RELAY_TO_DA: f64 = 0.4;
const RELAY_TO_SATIETY: f64 = 0.3;
const SATIETY_TO_NEED: f64 = 0.01;  // 포만 뉴런 발화 한 번이 결핍을 낮추는 양
const VALUE_COUNT: usize = 40;
const VALUE_TO_DA_WEIGHT: f64 = 0.2;
/// 예측 뉴런이 가치 뉴런에서 입력을 받는 구조일 때의 예측 뉴런 → 도파민 세기 (가치가 충분히 자란 뒤에야 보상이 상쇄되도록 약하게)
const PRED_TO_DA_LINKED: f64 = 0.1;
const VALUE_TRACE_DELAY: f64 = 20.0;
/// 예측 뉴런 묶음의 지연 (틱). 묶음마다 입력 시냅스의 지연이 달라서, 자극 뒤 서로 다른 시점에 활성화된다
const PRED_DELAYS: [f64; 8] = [8.0, 12.0, 17.0, 26.0, 38.0, 55.0, 82.0, 120.0];

impl Network {
    pub fn new() -> Self {
        Self::with_params(NetParams::default())
    }

    pub fn with_params(params: NetParams) -> Self {
        // 저장된 망 불러오기: 새로 만들지 않는다 (준비 구간도 돌리지 않는다. 기준선은 뉴런에 저장돼 있다)
        if let Some(path) = params.load_from.clone() {
            let mut net = Network::load(Path::new(&path)).unwrap_or_else(|| { eprintln!("[오류] 저장된 망을 읽지 못했다: {}", path); std::process::exit(2); });
            net.params.slow_events = params.slow_events;
            net.params.slow_check = params.slow_check;
            net.params.wake_check = params.wake_check;
            net.params.pred_floor = params.pred_floor;
            net.apply_pred_floor();
            // 자기 목표: 저장 파일의 설정을 따른다. 옵션 없이 기른 망을 옵션을 켜서 부르는 것은 막는다 (이미 자란 망의 발화율이 목표로 굳는다)
            if params.own_target.is_some() && net.params.own_target.is_none() {
                eprintln!("[오류] --own-target 없이 기른 망({})을 --own-target 을 켜서 불러올 수 없다 (자란 뒤의 발화율이 목표로 굳는다)", path); std::process::exit(2);
            }
            // 기한을 다시 세는 바닥(prune_reset_floor)은 저장 파일의 설정 그대로다: 뉴런의 reset_min 이 저장돼 있고, 명령줄로 켜거나 끄지 않는다
            // (이 칸이 없는 옛 저장 파일은 꺼짐 — 바닥 없이 기른 망이 이어 돌 때 그대로 이어진다)
            // 지연 시냅스의 방식은 명령줄을 따른다. 새 방식이고 저장 파일에 그 상태(합과 목록)가 있으면 그대로 되살린다: 끊지 않은 실행과 글자 그대로 이어진다.
            // 없으면(옛 방식으로 저장한 파일, 옛 형식) 시냅스에서 다시 계산한다
            if net.params.slow_events { net.slow_pos.clear(); if !net.slow_restore() { net.slow_resync(); } net.slow_chk_drift = 0.0; }
            net.slow_saved = None;
            if net.params.wake_check { net.wake = Some(Box::new(WakeCheck::default())); }
            return net;
        }
        let inh_cut = 20 - (20.0 * params.inhibitory_ratio).round() as usize;
        let mut neurons: Vec<Neuron> = Vec::new();
        let cols = 50usize;
        // 영역 크기: 배수를 곱하고 격자 한 줄(50칸) 단위로 맞춘다
        let sized = |base: usize| -> usize { (((base as f64 * params.scale) / 50.0).round() as usize).max(1) * 50 };
        let (emotion_count, reason_count, memory_count, hippo_count) = (sized(EMOTION_COUNT), sized(REASON_COUNT), sized(MEMORY_COUNT), sized(HIPPOCAMPUS_COUNT));

        // 감정 뉴런: Main 80% + Map 20% (15% 억제성)
        let emotion_start = neurons.len();
        let emotion_main_count = emotion_count * 80 / 100;
        for i in 0..emotion_count {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32;
            let id = neurons.len() as NeuronId;
            if i % 20 >= inh_cut {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let emotion_end = neurons.len();
        let emotion_main_range = (emotion_start, emotion_start + emotion_main_count);
        let emotion_map_range = (emotion_start + emotion_main_count, emotion_end);
        // Map 영역 threshold_scale 설정 (0.85 / 0.50 = 1.7)
        for idx in emotion_map_range.0..emotion_map_range.1 {
            neurons[idx].threshold_scale = 1.7;
        }

        // 이성 뉴런: Main 80% + Map 20% (15% 억제성)
        let reason_start = neurons.len();
        let reason_main_count = reason_count * 80 / 100;
        for i in 0..reason_count {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32 + 100.0;
            let id = neurons.len() as NeuronId;
            if i % 20 >= inh_cut {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let reason_end = neurons.len();
        let reason_main_range = (reason_start, reason_start + reason_main_count);
        let reason_map_range = (reason_start + reason_main_count, reason_end);
        for idx in reason_map_range.0..reason_map_range.1 {
            neurons[idx].threshold_scale = 1.7;
        }

        // 기억 피질 뉴런 (30% 억제성, 느린 감쇠 0.85)
        let memory_start = neurons.len();
        for i in 0..memory_count {
            let x = (i % cols) as f32 + 50.0;
            let y = (i / cols) as f32 + 50.0;
            let id = neurons.len() as NeuronId;
            let inhibitory = i % 10 >= 7;
            neurons.push(Neuron::new_slow_decay(id, x, y, inhibitory));
        }
        let memory_end = neurons.len();

        // 해마 뉴런 (20% 억제성, 느린 감쇠 0.85)
        let hippo_start = neurons.len();
        for i in 0..hippo_count {
            let x = (i % 25) as f32 + 60.0;
            let y = (i / 25) as f32 + 75.0;
            let id = neurons.len() as NeuronId;
            let inhibitory = i % 10 >= 8;
            neurons.push(Neuron::new_slow_decay(id, x, y, inhibitory));
        }
        let hippo_end = neurons.len();

        let mut net = Network {
            neurons,
            vocab: HashMap::new(),
            reverse_vocab: Vec::new(),
            input_idx: Vec::new(),
            output_idx: Vec::new(),
            emotion_range: (emotion_start, emotion_end),
            emotion_main_range,
            emotion_map_range,
            reason_range: (reason_start, reason_end),
            reason_main_range,
            reason_map_range,
            memory_range: (memory_start, memory_end),
            hippocampus_range: (hippo_start, hippo_end),
            fire_records: Vec::new(),
            next_fire_id: 1,
            debug: false,
            active_stimuli: Vec::new(),
            recent_spikes: Vec::new(),
            threshold: 0.50,
            noise_range: 0.2,
            global_tick: 0,
            fires_since_sleep: 0,
            last_sleep_tick: 0,
            plasticity: PlasticityOptions::default(),
            params,
            noise_amp: Vec::new(),
            da_idx: Vec::new(),
            reward_idx: Vec::new(),
            punish_idx: Vec::new(),
            pred_idx: Vec::new(),
            val_idx: Vec::new(),
            val_spikes: 0,
            taste_idx: Vec::new(),
            need_idx: Vec::new(),
            aux_end: 0,
            da_spikes: 0,
            pred_spikes: 0,
            drives: Vec::new(),
            slow_syn: Vec::new(),
            sprouted: 0,
            pruned: 0,
            unit_med: Vec::new(),
            tick_ids: Vec::new(),
            wiring_sum: 0,
            grow_made: [0; 5],
            gate_drop: [0; 5],
            own_wait_drop: 0,
            added_on_initial: 0,
            pruned_short: 0,
            prune_ages: Vec::new(),
            prune_age_unknown: 0,
            slow_recv: Vec::new(),
            slow_pos: Vec::new(),
            slow_sum: Vec::new(),
            slow_live: Vec::new(),
            slow_listed: std::collections::HashSet::new(),
            slow_saved: None,
            slow_shadow: Vec::new(),
            slow_init: Vec::new(),
            slow_chk_recv: 0.0,
            slow_chk_drift: 0.0,
            wake: None,
            scale_obs: [0; 5],
            scale_cum: Vec::new(),
            dev_tick: 0,
            dev_state: None,
        };

        // 세포 종류 지정: 억제성 뉴런을 순서대로 Pv 9 : Sst 8 : Vip 3 (45% / 40% / 15%) 으로 나눈다
        if net.params.typed {
            let mut k = 0usize;
            for n in net.neurons.iter_mut() {
                if n.inhibitory {
                    let t = match k % 20 { 0..=8 => CellType::Pv, 9..=16 => CellType::Sst, _ => CellType::Vip };
                    k += 1;
                    n.set_type(t);
                } else {
                    n.set_type(CellType::Pyramidal);
                }
            }
        }

        // 초기 자모 등록
        let initial_tokens = tokenizer::all_tokens();
        for tok in &initial_tokens {
            net.register_token(tok);
        }

        // 초기 랜덤 시냅스
        net.seed_random_synapses(net.params.synapses_per_neuron);
        // 은닉 뉴런 출력 이득 (입출력 뉴런은 1.0 유지)
        let io: std::collections::HashSet<usize> = net.input_idx.iter().chain(net.output_idx.iter()).cloned().collect();
        let gain = net.params.hidden_gain;
        for (i, n) in net.neurons.iter_mut().enumerate() {
            // 배율은 뉴런이 아니라 시냅스 가중치에 직접 반영한다 (가중치는 시냅스에만 존재)
            if !io.contains(&i) { for s in &mut n.synapses { s.weight *= gain; } }
        }
        if net.params.dopamine { net.build_dopamine(); }
        // 결정적 난수: 뉴런마다 씨앗을 적는다 (준비 구간 전에. 0 은 "쓰지 않음"이라 0 이 되지 않게 섞는다)
        if let Some(seed) = net.params.rng_seed {
            let v = crate::neuron::keyed_u64(seed, 0, 0, 0, 0) | 1;
            for n in net.neurons.iter_mut() { n.rng_seed = v; }
        }
        net.wiring_sum = net.wiring_checksum();
        net.rebuild_incoming();
        net.rebuild_slow_index();
        if net.params.slow_events { net.slow_resync(); net.slow_chk_drift = 0.0; }
        let spont = net.params.spont;
        if net.params.validated { net.apply_validated(); }
        if net.params.dopamine {
            net.apply_dopamine_options();
            // 기준선을 잡는 준비 구간. warmup_quiet 일 때만 자발 발화 없이 돈다: 검증 구성에서 자발 발화(spont, spont_fixed)가 주어져 있으면
            // 위의 apply_validated 가 이미 자발 확률을 걸어 두었으므로, 옵션이 없으면 준비 구간도 자발 발화가 켜진 채로 돈다 (아래에서 준비 뒤에 한 번 더 건다)
            // 예측 경로를 쓰는 망은 기본으로 자발 발화 없이 기준선을 잡는다 (자발 발화가 켜진 채로 잡으면 예측 뉴런이 배경 활동으로 도파민을 누른 상태가 "평소"가 된다)
            if net.params.predict && !net.params.no_warmup_quiet { net.params.warmup_quiet = true; }
            if net.params.warmup_quiet {
                let hidden_end = net.hippocampus_range.1;
                for n in net.neurons[..hidden_end].iter_mut() { n.spont_scale = 0.0; }
            }
            for _ in 0..DA_WARMUP_TICKS { net.idle_tick(); }
        }
        if spont.is_some() || net.params.spont_fixed.is_some() { net.apply_spont(); }
        if let Some(p) = net.params.release {
            let hidden_end = net.hippocampus_range.1;
            for n in net.neurons[..hidden_end].iter_mut() {
                for s in n.synapses.iter_mut() { if s.slow_rate == 0.0 { s.p_rel = p; } }
            }
        }
        if net.params.grow.is_some() {
            // 처음부터 있던 연결의 기한을 지금부터 세고, 은닉 뉴런에 가지치기 설정을 넣는다
            let hidden_end = net.hippocampus_range.1;
            let now = net.global_tick;
            let (tau, min_w) = (net.params.prune_tau, net.params.min_weight);
            // scatter_co: 기한이 한꺼번에 오지 않게, 시계를 지금부터 자기 수명(prune_life 와 같은 식) 안의 무작위 시점으로 둔다 (난수는 배선 씨앗에서)
            let scatter = net.params.scatter_co;
            let mut srng = net.stage_rng(6);
            for n in net.neurons[..hidden_end].iter_mut() {
                for s in n.synapses.iter_mut() {
                    s.last_co = now;
                    if scatter {
                        use rand::prelude::*;
                        let life = (tau * (s.weight / min_w.max(1e-12)).max(1.0).ln()).max(20_000.0);
                        s.last_co = now + (srng.random::<f64>() * life) as u64;
                    }
                }
                n.next_prune = now + 1000;
                n.prune_tau = tau;
                n.prune_min = min_w;
                n.reset_min = if net.params.prune_reset_floor { min_w } else { 0.0 };
                // 자기 목표를 재기 시작한다 (준비 구간 뒤. 시계는 각자의 첫 발화에서 시작한다)
                n.own_on = match net.params.own_target { Some(OwnMode::All) => true, Some(OwnMode::Pyr) => n.cell_type == CellType::Pyramidal, None => false };
            }
        }

        if net.params.wake_check { net.wake = Some(Box::new(WakeCheck::default())); }
        net.apply_pred_floor();
        let total_synapses: usize = net.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [초기화] 어휘 {}개, 뉴런 {}개 (감정 {}, 이성 {}, 기억 {}, 해마 {}, 입출력 {}), 시냅스 {}개",
            net.vocab.len(), net.neurons.len(),
            emotion_count, reason_count, memory_count, hippo_count,
            net.input_idx.len() * 2, total_synapses);
        net
    }

    /// 초기 배선의 단계별 난수. 배선 씨앗이 있으면 (씨앗, 단계 번호)로 정해지고, 없으면 실행마다 다르다.
    /// 단계마다 따로 두는 것은 한 단계의 유무(예: 가치 경로)가 다른 단계의 배선을 바꾸지 않게 하기 위해서다
    fn stage_rng(&self, stage: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        match self.params.seed.or(self.params.rng_seed) {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s.wrapping_mul(1_000_003).wrapping_add(stage)),
            None => rand::rngs::StdRng::seed_from_u64(rand::random::<u64>()),
        }
    }

    /// 초기 배선의 합계 값: 모든 시냅스의 (보내는 뉴런, 받는 뉴런, 세기, 지연, 붙는 자리)와 뉴런의 스스로 내는 전류·처음 전위를 차례로 섞는다
    fn wiring_checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: u64| { h ^= v; h = h.wrapping_mul(0x100000001b3); };
        for (i, n) in self.neurons.iter().enumerate() {
            mix(i as u64); mix(n.tonic.to_bits()); mix(n.potential.to_bits());
            for s in &n.synapses {
                mix(s.target as u64); mix(s.weight.to_bits()); mix(s.slow_rate.to_bits()); mix(s.on_dendrite as u64);
            }
        }
        h
    }

    /// 뉴런의 활동 목표 (틱당 발화율): K × 자기 자발 발화 확률. 목표가 없거나 자발 발화가 0 이면 0
    fn act_target_of(&self, idx: usize) -> f64 {
        // 자기 목표를 쓰는 뉴런이면 그 값 (아직 재는 중이면 0)
        if self.params.own_target.is_some() && self.neurons[idx].own_on { return self.neurons[idx].own_target; }
        match self.params.act_target { Some(k) => k * 1e-5 * self.neurons[idx].spont_scale, None => 0.0 }
    }

    fn kind_index(t: CellType) -> usize {
        match t { CellType::Pyramidal => 0, CellType::Pv => 1, CellType::Sst => 2, CellType::Vip => 3, _ => 4 }
    }

    /// 은닉 뉴런 가운데 조절 수준이 기준선보다 25% 넘게 높은 수, 25% 넘게 낮은 수, 기준선이 잡힌 수 (관찰용)
    pub fn gate_open_counts(&self) -> (usize, usize, usize) {
        let (mut up, mut down, mut n) = (0, 0, 0);
        for x in &self.neurons[..self.hippocampus_range.1] {
            if x.mod_base <= 1e-6 { continue; }
            n += 1;
            let d = x.mod_level / x.mod_base - 1.0;
            if d >= plasticity::MOD_DEADBAND { up += 1; } else if d <= -plasticity::MOD_DEADBAND { down += 1; }
        }
        (up, down, n)
    }
    /// 도파민 뉴런의 수상돌기 값의 평균 (관찰용. 보상 입력이 올리고 예측 뉴런이 누른다)
    pub fn da_dend_mean(&self) -> f64 {
        if self.da_idx.is_empty() { return 0.0; }
        self.da_idx.iter().map(|&i| self.neurons[i].dend_potential).sum::<f64>() / self.da_idx.len() as f64
    }

    /// 발달 시험의 관찰값
    pub fn dev_observe(&self) -> DevObs {
        let hidden_end = self.hippocampus_range.1;
        let mut o = DevObs { init_contacts: 0, init_hh: 0, grown_contacts: 0, init_pairs: 0, grown_pairs: 0, dup_syn: 0, hh_syn: 0, per: Vec::with_capacity(hidden_end) };
        let mut inc = vec![(0usize, 0usize); hidden_end];
        let mut seen: Vec<NeuronId> = Vec::new();
        for n in &self.neurons[..hidden_end] {
            seen.clear();
            seen.extend(n.synapses.iter().filter(|s| s.slow_rate == 0.0 && (s.target as usize) < hidden_end).map(|s| s.target));
            seen.sort_unstable();
            o.hh_syn += seen.len();
            o.dup_syn += seen.windows(2).filter(|w| w[0] == w[1]).count();
            for s in &n.synapses {
                let c = s.count as usize;
                let hh = s.slow_rate == 0.0 && (s.target as usize) < hidden_end;
                if s.born == 0 { o.init_contacts += c; o.init_pairs += 1; if hh { o.init_hh += c; } } else { o.grown_contacts += c; o.grown_pairs += 1; }
                if hh { if n.inhibitory { inc[s.target as usize].1 += c; } else { inc[s.target as usize].0 += c; } }
            }
        }
        for (i, n) in self.neurons[..hidden_end].iter().enumerate() {
            o.per.push((Self::kind_index(n.cell_type), self.area_of(i), n.rate_fast, n.rate_avg, self.act_target_of(i), inc[i].0, inc[i].1));
        }
        o
    }

    /// 도파민 뉴런 집단과 보상/처벌 입력 뉴런을 만들고 배선한다.
    ///   도파민 뉴런: 스스로 일정하게 발화(기준선)하고, 은닉·출력 뉴런 전체에 조절 신호를 보낸다 (받는 뉴런마다 DA_FANIN 개)
    ///   보상 입력: 도파민 뉴런을 흥분시켜 연달아 발화하게 한다.  처벌 입력: 억제해서 발화를 멈추게 한다
    fn build_dopamine(&mut self) {
        use rand::prelude::*;
        // 단계마다 난수를 따로 둔다: 1 도파민 뉴런, 2 가치, 3 예측, 4 도파민 → 표적, 5 몸
        let mut rng = self.stage_rng(1);
        let mut targets: Vec<usize> = (0..self.hippocampus_range.1).collect();
        targets.extend(self.output_idx.iter().cloned());
        let da_count = ((DA_COUNT as f64 * self.params.scale).round() as usize).max(1);
        for i in 0..da_count {
            let id = self.neurons.len() as NeuronId;
            let mut n = Neuron::new(id, (i * 10) as f32, 400.0);
            n.set_type(CellType::Modulatory);
            n.decay_rate = DA_DECAY;
            // 뉴런마다 주기가 조금씩 다르고 시작 위상도 달라서 서로 같은 틱에 몰려 발화하지 않는다
            n.tonic = DA_TONIC * (0.85 + 0.3 * rng.random::<f64>());
            n.potential = 0.5 * rng.random::<f64>();
            n.abs_refractory = 3;
            n.rel_refractory = 3;
            n.slow_adapt = false;
            n.spont_scale = 0.0;
            // 보상 입력과 예측 억제는 수상돌기에서 만난다 (세포체의 기준선 발화는 건드리지 않는다)
            n.has_dendrite = true;
            self.da_idx.push(id as usize);
            self.neurons.push(n);
        }
        // 예측 뉴런: 은닉 흥분성 뉴런에서 넓게 입력을 받고(처음엔 아주 약함), 도파민 뉴런의 수상돌기를 억제한다.
        // 입력 시냅스는 지연 시냅스이고 지연이 묶음마다 다르다 → 각 묶음은 자극 뒤 정해진 시점 근처에서만 활성화된다.
        // 보상이 오는 시점에 맞는 묶음의 입력이 강해지므로, 예측은 "보상이 올 시점"에만 작용한다
        // 가치 뉴런: 은닉 흥분성 뉴런에서 넓게 입력을 받고(처음엔 아주 약함), 도파민 뉴런의 수상돌기를 흥분시킨다.
        // 입력은 바로 전달되지만 학습에 쓰는 흔적은 약 20틱 뒤에 가장 커진다. 그래서 자극 직후 가치 뉴런이 스스로 일으킨
        // 도파민으로는 거의 강해지지 않고, 조금 뒤에 오는 보상(또는 다음 예고 자극이 일으킨 도파민)으로 강해진다
        if self.params.value {
            let mut rng = self.stage_rng(2);
            let (ms, me) = self.memory_range;
            let sources: Vec<usize> = (self.emotion_range.0..self.reason_range.1).chain(ms..me)
                .filter(|&i| !self.neurons[i].inhibitory).collect();
            for i in 0..VALUE_COUNT {
                let id = self.neurons.len() as NeuronId;
                let mut n = Neuron::new(id, (i * 10) as f32, 800.0);
                n.set_type(CellType::Value);
                n.spont_scale = 0.0;
                n.mod_decay = 0.9;
                for &d in &self.da_idx {
                    n.add_seed_synapse(d as NeuronId, self.params.v2da);
                    n.synapses.last_mut().unwrap().on_dendrite = true;
                }
                self.val_idx.push(id as usize);
                self.neurons.push(n);
                let mut picked: std::collections::HashSet<usize> = std::collections::HashSet::new();
                while picked.len() < PRED_FANIN.min(sources.len()) {
                    picked.insert(sources[rng.random_range(0..sources.len())]);
                }
                // 집합의 순회 순서는 실행마다 다르므로 정렬해서 쓴다 (같은 씨앗이면 같은 배선이 되게)
                let mut picked: Vec<usize> = picked.into_iter().collect();
                picked.sort_unstable();
                for src in picked {
                    self.neurons[src].add_seed_synapse(id, PRED_INIT_WEIGHT);
                    let s = self.neurons[src].synapses.last_mut().unwrap();
                    s.slow_rate = 1.0 / VALUE_TRACE_DELAY;
                    s.slow_fast = true;
                }
            }
            targets.extend(self.val_idx.iter().cloned());
        }
        if self.params.predict {
            let mut rng = self.stage_rng(3);
            let (ms, me) = self.memory_range;
            // 가치 경로가 있으면 예측 뉴런은 은닉이 아니라 가치 뉴런에서 입력을 받는다: 예측(억제)은 가치(흥분)의 늦은 사본이 된다.
            // 그래야 가치가 먼저 생기고 그만큼만 보상이 상쇄되며, 둘이 같은 보상 신호를 두고 따로 경쟁하지 않는다
            let sources: Vec<usize> = if self.val_idx.is_empty() {
                (self.emotion_range.0..self.reason_range.1).chain(ms..me).filter(|&i| !self.neurons[i].inhibitory).collect()
            } else { self.val_idx.clone() };
            for i in 0..PRED_COUNT {
                let id = self.neurons.len() as NeuronId;
                let mut n = Neuron::new_inhibitory(id, (i * 10) as f32, 700.0);
                n.set_type(CellType::Predictor);
                n.spont_scale = 0.0;
                for &d in &self.da_idx {
                    n.add_seed_synapse(d as NeuronId, self.params.p2da.unwrap_or(if self.val_idx.is_empty() { PRED_TO_DA_WEIGHT } else { PRED_TO_DA_LINKED }));
                    n.synapses.last_mut().unwrap().on_dendrite = true;
                }
                self.pred_idx.push(id as usize);
                self.neurons.push(n);
                let mut picked: std::collections::HashSet<usize> = std::collections::HashSet::new();
                while picked.len() < PRED_FANIN.min(sources.len()) {
                    picked.insert(sources[rng.random_range(0..sources.len())]);
                }
                let delay = PRED_DELAYS[i % PRED_DELAYS.len()];
                let mut picked: Vec<usize> = picked.into_iter().collect();
                picked.sort_unstable();
                for src in picked {
                    self.neurons[src].add_seed_synapse(id, PRED_INIT_WEIGHT);
                    self.neurons[src].synapses.last_mut().unwrap().slow_rate = 1.0 / delay;
                }
            }
            targets.extend(self.pred_idx.iter().cloned());
        }
        // 가치·예측 뉴런은 도파민 뉴런 전부에서 신호를 받는다 (세기는 합이 같도록 나눔). 네 개에서만 받으면 평소 발화의 흔들림이
        // ±20% 가까이 되어 그것만으로 입력이 조금씩 커진다. 전부에서 받으면 흔들림이 평균으로 줄어든다
        let dense: Vec<usize> = self.val_idx.iter().chain(self.pred_idx.iter()).cloned().collect();
        for &t in &dense {
            let w = DA_WEIGHT * DA_FANIN as f64 / da_count as f64;
            for k in 0..da_count { let d = self.da_idx[k]; self.neurons[d].add_seed_synapse(t as NeuronId, w); }
        }
        targets.retain(|t| !dense.contains(t));
        let mut rng = self.stage_rng(4);
        for &t in &targets {
            let mut picked: Vec<usize> = Vec::with_capacity(DA_FANIN);
            while picked.len() < DA_FANIN.min(da_count) {
                let d = self.da_idx[rng.random_range(0..da_count)];
                if !picked.contains(&d) { picked.push(d); }
            }
            for d in picked { self.neurons[d].add_seed_synapse(t as NeuronId, DA_WEIGHT); }
        }
        for (k, inhibitory) in [(0usize, false), (1, true)] {
            for i in 0..REWARD_COUNT {
                let id = self.neurons.len() as NeuronId;
                let mut n = Neuron::new_io(id, (i * 10) as f32, 500.0 + 100.0 * k as f32);
                n.inhibitory = inhibitory;
                n.spont_scale = 0.0;
                let w = if inhibitory { PUNISH_WEIGHT } else { REWARD_WEIGHT };
                for &d in &self.da_idx {
                    n.add_seed_synapse(d as NeuronId, w);
                    // 보상 입력은 수상돌기로, 처벌 입력은 세포체로 (기준선 발화 자체를 멈춘다)
                    if !inhibitory { n.synapses.last_mut().unwrap().on_dendrite = true; }
                }
                if inhibitory { self.punish_idx.push(id as usize); } else { self.reward_idx.push(id as usize); }
                self.neurons.push(n);
            }
        }
        // 1차 보상과 몸의 상태: 종류마다 감각 입력 → 중계 → 도파민, 결핍 → 중계, 중계 → 포만 → 결핍
        if self.params.needs {
            let mut rng = self.stage_rng(5);
            for kind in 0..2usize {
                let y = 900.0 + 100.0 * kind as f32;
                let mut make = |net: &mut Network, n: Neuron| -> usize { let i = net.neurons.len(); net.neurons.push(n); i };
                let (mut taste, mut need, mut relay, mut sat) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for i in 0..NEED_GROUP {
                    let id = self.neurons.len() as NeuronId;
                    let mut n = Neuron::new_io(id, (i * 10) as f32, y);
                    n.spont_scale = 0.0;
                    taste.push(make(self, n));
                }
                for i in 0..NEED_GROUP {
                    let id = self.neurons.len() as NeuronId;
                    let mut n = Neuron::new(id, (i * 10) as f32, y + 20.0);
                    n.decay_rate = 0.9;
                    n.tonic = 0.1 * (0.9 + 0.2 * rng.random::<f64>());
                    n.need_rate = 1.0 / NEED_FILL_TICKS;
                    n.abs_refractory = 3; n.rel_refractory = 3;
                    n.slow_adapt = false; n.spont_scale = 0.0;
                    need.push(make(self, n));
                }
                for i in 0..NEED_GROUP {
                    let id = self.neurons.len() as NeuronId;
                    let mut n = Neuron::new(id, (i * 10) as f32, y + 40.0);
                    n.slow_adapt = false; n.spont_scale = 0.0;
                    n.has_dendrite = true;
                    relay.push(make(self, n));
                }
                for i in 0..NEED_GROUP {
                    let id = self.neurons.len() as NeuronId;
                    let mut n = Neuron::new_inhibitory(id, (i * 10) as f32, y + 60.0);
                    n.slow_adapt = false; n.spont_scale = 0.0;
                    sat.push(make(self, n));
                }
                for &r in &relay {
                    for &t in &taste { self.neurons[t].add_seed_synapse(r as NeuronId, TASTE_TO_RELAY); }
                    for &q in &need {
                        self.neurons[q].add_seed_synapse(r as NeuronId, NEED_TO_RELAY);
                        self.neurons[q].synapses.last_mut().unwrap().on_dendrite = true;
                    }
                    for &d in &self.da_idx.clone() {
                        self.neurons[r].add_seed_synapse(d as NeuronId, RELAY_TO_DA);
                        self.neurons[r].synapses.last_mut().unwrap().on_dendrite = true;
                    }
                    for &x in &sat { self.neurons[r].add_seed_synapse(x as NeuronId, RELAY_TO_SATIETY); }
                }
                for &x in &sat { for &q in &need { self.neurons[x].add_seed_synapse(q as NeuronId, SATIETY_TO_NEED); } }
                self.taste_idx.push(taste);
                self.need_idx.push(need);
            }
        }
        self.aux_end = self.neurons.len();
    }

    /// 1차 보상의 감각 입력을 ticks 틱 동안 켠다 (kind: 0 = 먹이, 1 = 물). 도파민이 나오는지는 그때의 결핍 정도가 정한다
    pub fn drive_primary(&mut self, kind: usize, ticks: u64) {
        if let Some(idx) = self.taste_idx.get(kind) { if ticks > 0 { self.drives.push((idx.clone(), ticks)); } }
    }
    /// 결핍 정도의 평균 (kind: 0 = 배고픔, 1 = 목마름)
    pub fn need_level(&self, kind: usize) -> f64 {
        match self.need_idx.get(kind) { Some(v) if !v.is_empty() => v.iter().map(|&i| self.neurons[i].need).sum::<f64>() / v.len() as f64, _ => 0.0 }
    }
    /// 결핍 정도를 직접 정한다 (시험용)
    pub fn set_need(&mut self, kind: usize, level: f64) {
        if let Some(v) = self.need_idx.get(kind) { for &i in &v.clone() { self.neurons[i].need = level; } }
    }

    /// 도파민 방식의 학습 설정: STDP·iSTDP 는 흔적만 남기고, 조절 수준이 기준선에서 벗어날 때 가중치로 넘어간다
    fn apply_dopamine_options(&mut self) {
        self.plasticity.mod_gate = Some(self.params.da_gain);
        self.plasticity.gate_base = self.params.gate_base;
        self.plasticity.mod_baseline = true;
        self.plasticity.pred_eta = if self.params.predict || self.params.value { self.params.pred_eta } else { 0.0 };
        self.plasticity.val_eta = self.params.val_eta;
        if let Some(tau) = self.params.short_tau {
            // 일시적인 변화 방식: 평소 비율은 쓰지 않고, 모든 은닉·출력으로 가는 시냅스에 일시적 부분의 감쇠 배수를 넣는다
            let d = (-1.0 / tau.max(1.0)).exp();
            self.plasticity.short_decay = Some(d);
            self.plasticity.gate_base = 0.0;
            let aux: Vec<bool> = (0..self.neurons.len()).map(|i| self.is_aux(i)).collect();
            for (i, n) in self.neurons.iter_mut().enumerate() {
                if aux[i] { continue; }
                for s in n.synapses.iter_mut() { if s.slow_rate == 0.0 { s.short_decay = d; } }
            }
        }
    }

    /// 지연 시냅스의 전류 전달: 매 틱 (가중치 × 지연 흔적)을 받는 뉴런에 더한다
    fn deliver_slow_inhibition(&mut self) {
        if self.params.slow_events {
            // 받는 쪽 합을 한 틱 진행: A2 ← a·A2 + r·A1, A1 ← a·A1 (시냅스별 점화식의 합과 같다), 그리고 A2 를 전달한다
            for v in self.slow_sum.iter_mut() {
                let r = v[2];
                if r == 0.0 { continue; }
                v[1] = (1.0 - r) * v[1] + r * v[0];
                v[0] *= 1.0 - r;
            }
            if self.params.slow_check {
                // 옛 방식의 값: 시냅스마다 지금 틱까지 따라잡은 흔적(사본으로 계산해 실제 상태는 건드리지 않는다)으로 가중치 × slow2 를 더한다 (slow2 > 1e-4 인 것만)
                let now = self.global_tick;
                let mut old = vec![0.0f64; self.slow_sum.len()];
                for &(pre, j) in &self.slow_syn {
                    let Some(s) = self.neurons[pre].synapses.get(j) else { continue };
                    if s.slow_fast { continue; }
                    let mut c = s.clone();
                    c.sync(now);
                    if c.slow2 > 1e-4 { if let Some(&p) = self.slow_pos.get(c.target as usize) { if p != u32::MAX { old[p as usize] += c.weight * c.slow2; } } }
                }
                for (p, v) in self.slow_sum.iter().enumerate() { if v[2] > 0.0 { self.slow_chk_recv = self.slow_chk_recv.max((v[1] - old[p]).abs()); } }
            }
            for p in 0..self.slow_sum.len() {
                let v = self.slow_sum[p];
                if v[2] > 0.0 && v[1] != 0.0 { let t = self.slow_recv[p]; self.neurons[t].receive(v[1]); }
            }
            return;
        }
        for k in 0..self.slow_syn.len() {
            let (pre, j) = self.slow_syn[k];
            let now = self.global_tick;
            let (t, amt) = match self.neurons[pre].synapses.get_mut(j) {
                Some(s) => { if s.slow_fast { continue; } s.sync(now); if s.slow2 > 1e-4 { (s.target as usize, s.weight * s.slow2) } else { continue } }
                None => continue,
            };
            if t < self.neurons.len() { self.neurons[t].receive(amt); }
        }
    }
    /// slow_events: 저장 파일에서 읽은 합과 목록을 그대로 되살린다. 되살렸으면 true (저장된 것이 없거나 받는 뉴런 수가 맞지 않으면 false: 부르는 쪽이 다시 계산한다)
    fn slow_restore(&mut self) -> bool {
        let Some((sum, live)) = self.slow_saved.take() else { return false };
        let recv: Vec<usize> = self.val_idx.iter().chain(self.pred_idx.iter()).cloned().collect();
        if sum.len() != recv.len() { return false; }
        self.slow_recv = recv;
        self.slow_pos = vec![u32::MAX; self.neurons.len()];
        for (p, &t) in self.slow_recv.iter().enumerate() { self.slow_pos[t] = p as u32; }
        self.slow_listed = live.iter().flatten().map(|&(pre, j)| ((pre as u64) << 32) | j as u64).collect();
        self.slow_sum = sum;
        self.slow_live = live;
        if self.params.slow_check {
            self.slow_shadow = self.slow_syn.iter().map(|&(pre, j)| self.neurons[pre].synapses.get(j).map_or(0.0, |s| s.weight)).collect();
            self.slow_init = self.slow_shadow.clone();
        }
        true
    }
    /// 지연 시냅스의 위치 목록을 다시 만든다 (전달할 때 전체 시냅스를 훑지 않기 위한 색인)
    /// slow_events: 모든 지연 시냅스를 지금 틱까지 따라잡고 받는 쪽 합을 처음부터 다시 계산한다 (만든 직후, 불러온 직후, 1만 틱마다. 부동소수 오차가 쌓이지 않게).
    /// 흔적이 남아 있는데 목록에 없는 시냅스는 목록에 올린다
    fn slow_resync(&mut self) {
        let now = self.global_tick;
        if self.slow_pos.len() != self.neurons.len() {
            self.slow_recv = self.val_idx.iter().chain(self.pred_idx.iter()).cloned().collect();
            self.slow_pos = vec![u32::MAX; self.neurons.len()];
            for (p, &t) in self.slow_recv.iter().enumerate() { self.slow_pos[t] = p as u32; }
            self.slow_sum = vec![[0.0; 3]; self.slow_recv.len()];
            self.slow_live = vec![Vec::new(); self.slow_recv.len()];
            self.slow_listed.clear();
            if self.params.slow_check {
                self.slow_shadow = self.slow_syn.iter().map(|&(pre, j)| self.neurons[pre].synapses.get(j).map_or(0.0, |s| s.weight)).collect();
                self.slow_init = self.slow_shadow.clone();
            }
        }
        let mut fresh = vec![[0.0f64; 2]; self.slow_recv.len()];
        // 받는 뉴런 하나로 들어오는 지연 시냅스는 지연 값이 모두 같아야 한다 (합의 점화식이 받는 뉴런마다 값 하나를 쓴다). 다르면 틀린 값을 내므로 멈춘다
        let mut rate_seen = vec![0.0f64; self.slow_recv.len()];
        for k in 0..self.slow_syn.len() {
            let (pre, j) = self.slow_syn[k];
            let Some(s) = self.neurons[pre].synapses.get_mut(j) else { continue };
            s.sync(now);
            let p = match self.slow_pos.get(s.target as usize) { Some(&p) if p != u32::MAX => p as usize, _ => continue };
            if !s.slow_fast {
                fresh[p][0] += s.weight * s.slow1;
                fresh[p][1] += s.weight * s.slow2;
                if rate_seen[p] != 0.0 && rate_seen[p] != s.slow_rate {
                    eprintln!("--slow-events 는 받는 뉴런 하나로 들어오는 지연 시냅스의 지연 값이 서로 다른 배선에서 쓸 수 없습니다 (받는 뉴런 {}: {} 와 {})", s.target, rate_seen[p], s.slow_rate);
                    std::process::exit(2);
                }
                rate_seen[p] = s.slow_rate;
                self.slow_sum[p][2] = s.slow_rate;
            }
            if (s.slow1 >= 0.001 || s.slow2 >= 0.001) && self.slow_listed.insert(((pre as u64) << 32) | j as u64) {
                self.slow_live[p].push((pre as u32, j as u32));
            }
        }
        for p in 0..fresh.len() {
            self.slow_chk_drift = self.slow_chk_drift.max((self.slow_sum[p][1] - fresh[p][1]).abs());
            self.slow_sum[p][0] = fresh[p][0];
            self.slow_sum[p][1] = fresh[p][1];
        }
    }

    /// slow_events: 보내는 뉴런이 발화했을 때 그 뉴런의 지연 시냅스(번호 k)를 받는 쪽에 알린다.
    /// 예측 뉴런의 입력이면 가중치를 합(Σ 가중치 × slow1)에 더하고(시냅스의 slow1 += 1 은 try_fire 가 이미 했다), 받는 뉴런의 살아 있는 입력 목록에 올린다
    fn slow_on_spike(&mut self, pre: usize, k: usize) {
        let (t, w, fast) = match self.neurons[pre].synapses.get(k) { Some(s) if s.slow_rate > 0.0 => (s.target as usize, s.weight, s.slow_fast), _ => return };
        let p = match self.slow_pos.get(t) { Some(&p) if p != u32::MAX => p as usize, _ => return };
        if !fast { self.slow_sum[p][0] += w; }
        if self.slow_listed.insert(((pre as u64) << 32) | k as u64) { self.slow_live[p].push((pre as u32, k as u32)); }
    }

    /// slow_check: 옛 규칙(predictor_step 과 같은 식, 지연 시냅스 전체)으로 따로 굴리는 가중치를 한 틱 진행한다. 실제 시냅스는 건드리지 않는다
    fn slow_shadow_step(&mut self) {
        let opt = &self.plasticity;
        if !opt.mod_baseline || opt.pred_eta == 0.0 { return; }
        let now = self.global_tick;
        for (k, &(pre, j)) in self.slow_syn.iter().enumerate() {
            let Some(s) = self.neurons[pre].synapses.get(j) else { continue };
            let mut c = s.clone();
            c.sync(now);
            let tr = if c.slow_fast { (c.slow2 - c.slow1).max(0.0) } else { c.slow2 };
            if tr <= 0.01 { continue; }
            let r = match self.neurons.get(c.target as usize) { Some(n) if n.mod_base > 1e-6 => n.mod_level / n.mod_base - 1.0, _ => continue };
            if c.slow_fast { if r < plasticity::VALUE_UP && r > -plasticity::VALUE_DOWN { continue; } } else if r.abs() < plasticity::PRED_DEADBAND { continue; }
            let eta = if c.slow_fast && opt.val_eta > 0.0 { opt.val_eta } else { opt.pred_eta };
            self.slow_shadow[k] = (self.slow_shadow[k] + eta * r * tr).clamp(0.0, 1.0);
        }
    }

    /// wake_check 의 틱 안 단계들. 0: 틱 시작(표시 넘기기, 놓친 전위 변화 세기), 1: 바깥 자극을 넣은 뒤, 2: 감쇠 뒤, 3: 지연 전달 뒤(판정 직전)
    fn wake_stage(&mut self, stage: u8) {
        let Some(mut w) = self.wake.take() else { return };
        let n = self.neurons.len();
        match stage {
            0 => {
                if w.next.len() != n { w.next = vec![0; n]; w.pot_end = vec![0.0; n]; w.queue = vec![false; n]; w.started = false; }
                w.now = std::mem::replace(&mut w.next, vec![0; n]);
                // 지난 틱의 발화 판정 뒤로 전위가 달라졌는데 전달을 받았다는 표시가 없으면 놓친 전위 변화다
                if w.started { for i in 0..n { if self.neurons[i].potential != w.pot_end[i] && w.now[i] & 1 == 0 { w.acc[11] += 1; } } }
                w.pot_begin = self.neurons.iter().map(|x| x.potential).collect();
            }
            1 => { for i in 0..n { if self.neurons[i].potential != w.pot_begin[i] { w.now[i] |= 1; } } }   // 이 틱에 바깥 자극을 받음
            2 => { w.pot_decay = self.neurons.iter().map(|x| x.potential).collect(); }
            _ => {
                let now = self.global_tick;
                w.base.clear(); w.bar.clear(); w.refr.clear();
                for (i, x) in self.neurons.iter().enumerate() {
                    if x.potential != w.pot_decay[i] { w.now[i] |= 1; }   // 지연 전달로 전위가 달라짐
                    let adj = (x.self_threshold + x.adapt) * x.threshold_scale / x.excitability;
                    let mut bar = adj;
                    let mut refr = false;
                    if !x.skip_refractory { if let Some(last) = x.last_spike_tick {
                        // 불응기 길이: neuron.rs 의 try_fire 와 같은 식 (기본값: 흥분성 4 / 10, 억제성 2 / 4)
                        let abs_ref = (if x.abs_refractory > 0 { x.abs_refractory } else if x.inhibitory { 2 } else { 4 }) * x.refractory_scale;
                        let rel_ref = (if x.rel_refractory > 0 { x.rel_refractory } else if x.inhibitory { 4 } else { 10 }) * x.refractory_scale;
                        let elapsed = now.saturating_sub(last);
                        if elapsed == rel_ref.max(abs_ref) { w.now[i] |= 4; }   // 불응기(절대 + 상대)가 이 틱에 끝남
                        if elapsed < abs_ref { bar = f64::INFINITY; refr = true; } else if elapsed < rel_ref { bar = adj * 2.0; refr = true; }
                    } }
                    w.base.push(x.potential + x.dendritic_drive());
                    w.bar.push(bar);
                    w.refr.push(refr);
                }
                // 셋째 규칙: 전위 변화나 바깥 자극을 받은 뉴런이 큐에 들어간다. 지속 입력이 있는 뉴런은 늘 큐에 있다.
                // 항상성이 감응도를 바꾼 다음 틱(1만 틱 갱신 뒤)에는 모두 한 번 판정한다. 불응기가 끝나는 틱에는 따로 깨우지 않는다
                let all = now % 10_000 == 1;
                for (i, x) in self.neurons.iter().enumerate() { if all || w.now[i] & 1 != 0 || x.tonic > 0.0 { w.queue[i] = true; } }
                w.acc[0] += n as u64;
                w.acc[1] += w.now.iter().filter(|&&f| f != 0).count() as u64;
                w.acc[2] += w.now.iter().filter(|&&f| f & 5 != 0).count() as u64;
                w.acc[18] += w.queue.iter().filter(|&&q| q).count() as u64;
            }
        }
        self.wake = Some(w);
    }

    /// wake_check: 발화 판정 뒤. 깨어나지 않았는데 발화한 뉴런을 세고 이유를 가른다
    ///   지속 입력이 있는 뉴런(tonic > 0: 도파민·필요 뉴런은 감쇠 단계에서 매 틱 전위가 스스로 오른다) / 그 밖(잡음 없이도 문턱을 넘었다: 깨우는 규칙이 놓친 것) / 잡음(잡음을 더하면 넘을 수 있었다. 자발 발화였을 수도 있다) / 자발 발화(잡음으로는 넘을 수 없었다)
    fn wake_after_fire(&mut self, fired: &[NeuronId]) {
        let Some(mut w) = self.wake.take() else { return };
        w.acc[3] += fired.len() as u64;
        for &id in fired {
            let i = id as usize;
            if w.now[i] == 0 { w.acc[4] += 1; }
            if w.now[i] & 5 != 0 && w.queue[i] { continue; }
            let x = &self.neurons[i];
            let amp = self.noise_amp.get(i).copied().unwrap_or(self.noise_range);
            // 이유: 0 자발 발화, 1 잡음, 2 지속 입력, 3 그 밖
            let why = if x.tonic > 0.0 { 2 } else if w.base[i] >= w.bar[i] { 3 } else if w.base[i] + amp > w.bar[i] { 1 } else if x.spont_scale > 0.0 { 0 } else { 3 };
            // 셋째 규칙: 큐에 없었는데 발화함
            if !w.queue[i] { w.acc[19] += 1; w.acc[20 + why] += 1; if why == 3 && w.refr[i] { w.acc[24] += 1; } }
            if w.now[i] & 5 != 0 { continue; }
            w.acc[5] += 1;
            w.acc[6 + why] += 1;
            if why == 3 && w.refr[i] { w.acc[10] += 1; }
            // 조절 신호만 받은 뉴런이 발화함: 발화 판정은 조절 수준을 읽지 않으므로, 이 발화의 이유도 위와 같이 가른다
            if w.now[i] == 2 { w.acc[12] += 1; w.acc[13 + why] += 1; if why == 3 && w.refr[i] { w.acc[17] += 1; } }
        }
        // 셋째 규칙: 판정을 마친 뉴런 가운데, 가장 큰 잡음을 더해도 적응 0 의 문턱에 못 미치는 것은 큐에서 뺀다 (새 입력 없이는 발화할 수 없다). 지속 입력이 있는 뉴런은 빼지 않는다
        for (i, x) in self.neurons.iter().enumerate() {
            if !w.queue[i] || x.tonic > 0.0 { continue; }
            let amp = self.noise_amp.get(i).copied().unwrap_or(self.noise_range);
            if x.potential + x.dendritic_drive() + amp < x.self_threshold * x.threshold_scale / x.excitability { w.queue[i] = false; }
        }
        w.pot_end = self.neurons.iter().map(|x| x.potential).collect();
        w.started = true;
        self.wake = Some(w);
    }

    /// wake_check: 1000틱마다 구간 값과 누계를 찍는다
    fn wake_report(&mut self) {
        let Some(w) = self.wake.as_mut() else { return };
        for k in 0..25 { w.total[k] += w.acc[k]; }
        let line = |a: &[u64; 25]| format!("깨어난 비율: 조절 포함 {:.1}%, 조절 제외 {:.1}% | 발화 {} | 깨어나지 않았는데 발화: 조절 포함 규칙 {}, 조절 제외 규칙 {} (자발 발화 {}, 잡음 {}, 지속 입력(tonic)이 있는 뉴런 {}, 그 밖 {} — 그중 불응기 안 {}) | 조절 신호만 받고 발화 {} (자발 발화 {}, 잡음 {}, 지속 입력 {}, 그 밖 {} — 그중 불응기 안 {}) | 놓친 전위 변화 {} | 셋째 규칙(큐에 남기): 큐에 있던 비율 {:.1}%, 큐에 없었는데 발화 {} (자발 발화 {}, 잡음 {}, 지속 입력 {}, 그 밖 {} — 그중 불응기 안 {})",
            a[1] as f64 / a[0].max(1) as f64 * 100.0, a[2] as f64 / a[0].max(1) as f64 * 100.0, a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[12], a[13], a[14], a[15], a[16], a[17], a[11], a[18] as f64 / a[0].max(1) as f64 * 100.0, a[19], a[20], a[21], a[22], a[23], a[24]);
        println!("[wake-check] 틱 {} | 구간 | {}", self.global_tick, line(&w.acc));
        println!("[wake-check] 틱 {} | 누계 | {}", self.global_tick, line(&w.total));
        w.acc = [0; 25];
    }

    /// slow_check: 1000틱마다 확인 줄을 찍는다
    fn slow_check_report(&mut self) {
        let (mut rel, mut abs) = (0.0f64, 0.0f64);
        let (mut changed, mut change_max) = (0usize, 0.0f64);
        for (k, &(pre, j)) in self.slow_syn.iter().enumerate() {
            let Some(s) = self.neurons[pre].synapses.get(j) else { continue };
            let c = (s.weight - self.slow_init[k]).abs();
            if c > 0.0 { changed += 1; change_max = change_max.max(c); }
            let d = (s.weight - self.slow_shadow[k]).abs();
            abs = abs.max(d);
            rel = rel.max(d / self.slow_shadow[k].abs().max(1e-12));
        }
        let listed: usize = self.slow_live.iter().map(|l| l.len()).sum();
        println!("[slow-check] 틱 {} | 지난 1000틱에 예측 뉴런이 받은 양의 최대 차이 {:.3e} | 지연 시냅스 가중치의 최대 차이: 상대 {:.3e}, 절대 {:.3e} | 다시 맞출 때 합의 최대 어긋남 {:.3e} | 살아 있는 입력 {}개 / 지연 시냅스 {}개 | 처음 값에서 바뀐 가중치 {}개 (가장 큰 변화 {:.3e})",
            self.global_tick, self.slow_chk_recv, rel, abs, self.slow_chk_drift, listed, self.slow_syn.len(), changed, change_max);
        self.slow_chk_recv = 0.0;
        self.slow_chk_drift = 0.0;
    }

    fn rebuild_slow_index(&mut self) {
        self.slow_syn.clear();
        for (pre, n) in self.neurons.iter().enumerate() {
            for (j, s) in n.synapses.iter().enumerate() {
                if s.slow_rate > 0.0 { self.slow_syn.push((pre, j)); }
            }
        }
    }
    #[inline]
    fn is_pred(&self, idx: usize) -> bool {
        match self.pred_idx.first() { Some(&s) => idx >= s && idx < s + self.pred_idx.len(), None => false }
    }

    /// 자극 없이 ticks 틱 진행 (학습 포함). 보상을 주지 않는 시행에서 보상 시행과 같은 시간을 흘려보낼 때 쓴다
    pub fn rest(&mut self, ticks: u64) {
        for _ in 0..ticks { self.tick(); }
    }

    /// 보상(positive) 또는 처벌 입력 뉴런을 strength × REWARD_TICKS 틱 동안 발화시키고, 그 뒤 REWARD_TAIL 틱을 더 진행한다.
    /// 반환: 입력을 켜 둔 동안의 도파민 뉴런 발화 수
    pub fn deliver_reward(&mut self, positive: bool, strength: f64) -> u64 {
        let ticks = ((REWARD_TICKS as f64) * strength).round().max(1.0) as u64;
        let before = self.da_spikes;
        self.drive_reward(positive, ticks);
        self.rest(ticks);
        let during = self.da_spikes - before;
        self.rest(REWARD_TAIL);
        during
    }

    // ── 연속 모드 ──
    // 자극의 끝을 판정하지 않는다. 시간은 step() 으로 한 틱씩 흐르고, 입력과 보상은 아무 때나 켤 수 있으며
    // (이전 반응이 남아 있어도 겹쳐서 들어간다), 출력은 발화하는 대로 읽는다.
    // fire() / feedback() 은 이 위에 "반응이 끝날 때까지 기다리기"를 얹은 시험용 방식이다.

    /// 글자 입력을 ticks 틱 동안 켠다 (fire 의 입력 유지 시간은 4틱)
    pub fn drive_text(&mut self, text: &str, ticks: u64) {
        let idx: Vec<usize> = tokenizer::decompose_to_jamo(text).iter()
            .map(|j| { let seq = self.get_seq(j); self.input_neuron(seq) }).collect();
        if ticks > 0 && !idx.is_empty() { self.drives.push((idx, ticks)); }
    }

    /// 보상(positive) 또는 처벌 입력을 ticks 틱 동안 켠다
    pub fn drive_reward(&mut self, positive: bool, ticks: u64) {
        let idx = if positive { self.reward_idx.clone() } else { self.punish_idx.clone() };
        if ticks > 0 && !idx.is_empty() { self.drives.push((idx, ticks)); }
    }

    /// 한 틱 진행. 반환: 이번 틱에 발화한 출력 뉴런의 글자
    pub fn step(&mut self) -> Vec<String> {
        self.tick();
        let mut out = Vec::new();
        for &nid in &self.tick_ids {
            if let Some(seq) = self.output_idx.iter().position(|&o| o == nid as usize) {
                if let Some(tok) = self.reverse_vocab.get(seq) { out.push(tok.clone()); }
            }
        }
        out
    }

    pub fn hidden_count(&self) -> usize { self.hippocampus_range.1 }
    pub fn params_ref(&self) -> &NetParams { &self.params }
    /// 관찰용 영역 번호: 0 감정_Main, 1 감정_Map, 2 이성_Main, 3 이성_Map, 4 기억, 5 해마, 6 출력, 7 그 밖(입력 등)
    pub fn area_of(&self, idx: usize) -> usize {
        let within = |r: (usize, usize)| idx >= r.0 && idx < r.1;
        if within(self.emotion_main_range) { 0 } else if within(self.emotion_map_range) { 1 }
        else if within(self.reason_main_range) { 2 } else if within(self.reason_map_range) { 3 }
        else if within(self.memory_range) { 4 } else if within(self.hippocampus_range) { 5 }
        else if self.output_idx.contains(&idx) { 6 } else { 7 }
    }
    /// 영역별 뉴런 수 (area_of 와 같은 순서, 출력까지)
    pub fn area_sizes(&self) -> [usize; 7] {
        let len = |r: (usize, usize)| r.1 - r.0;
        [len(self.emotion_main_range), len(self.emotion_map_range), len(self.reason_main_range), len(self.reason_map_range),
         len(self.memory_range), len(self.hippocampus_range), self.output_idx.len()]
    }
    /// 가치 뉴런들의 (조절 수준 ÷ 기준선 − 1) 값 목록 (관찰용)
    pub fn value_deviations(&self) -> Vec<f64> {
        self.val_idx.iter().map(|&i| &self.neurons[i]).filter(|n| n.mod_base > 1e-6).map(|n| n.mod_level / n.mod_base - 1.0).collect()
    }
    /// 뉴런 집단의 (조절 수준 ÷ 기준선 − 1) 값 목록 (관찰용). which: 0 은닉, 1 출력, 그 밖 가치. 조절 신호를 받지 않는 뉴런은 빠진다
    pub fn deviations_of(&self, which: usize) -> Vec<f64> {
        let dev = |i: usize| { let n = &self.neurons[i]; if n.mod_base > 1e-6 { Some(n.mod_level / n.mod_base - 1.0) } else { None } };
        match which {
            0 => (0..self.hippocampus_range.1).filter_map(dev).collect(),
            1 => self.output_idx.iter().filter_map(|&i| dev(i)).collect(),
            _ => self.val_idx.iter().filter_map(|&i| dev(i)).collect(),
        }
    }
    pub fn val_count(&self) -> usize { self.val_idx.len() }
    /// 은닉 뉴런마다 (종류: 0 추체, 1 Pv, 2 Sst, 3 Vip, 4 그 밖, x, y) (관찰용, 읽기만)
    pub fn probe_hidden_info(&self) -> Vec<(usize, f32, f32)> {
        self.neurons[..self.hippocampus_range.1].iter().map(|n| (match n.cell_type {
            CellType::Pyramidal => 0, CellType::Pv => 1, CellType::Sst => 2, CellType::Vip => 3, _ => 4 }, n.x, n.y)).collect()
    }
    /// 출력 뉴런마다, 그 출력으로 보내는 은닉 뉴런의 목록(접점 1 이상인 쌍) (관찰용, 읽기만)
    pub fn probe_out_sources(&self) -> Vec<Vec<usize>> {
        let pos: std::collections::HashMap<usize, usize> = self.output_idx.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let mut v = vec![Vec::new(); self.output_idx.len()];
        for (i, n) in self.neurons[..self.hippocampus_range.1].iter().enumerate() {
            for s in &n.synapses { if s.count == 0 { continue; } if let Some(&k) = pos.get(&(s.target as usize)) { v[k].push(i); } }
        }
        v
    }
    /// text 의 글자들의 입력 뉴런이 직접 닿는 은닉 뉴런의 목록(되풀이 없이) (관찰용. 이미 아는 글자만 넣을 것 — drive_text 와 같은 길로 글자를 찾는다)
    pub fn probe_input_targets(&mut self, text: &str) -> Vec<usize> {
        let h = self.hippocampus_range.1;
        let idx: Vec<usize> = tokenizer::decompose_to_jamo(text).iter().map(|j| { let seq = self.get_seq(j); self.input_neuron(seq) }).collect();
        let mut t: Vec<usize> = idx.iter().flat_map(|&i| self.neurons[i].synapses.iter().filter(|s| s.count > 0).map(|s| s.target as usize)).filter(|&x| x < h).collect();
        t.sort_unstable(); t.dedup(); t
    }
    /// 입력 뉴런이 내보내는 시냅스의 세기 (관찰용, 읽기만): (개수, 중앙, 90%, 최대, th 를 넘은 것의 몫)
    pub fn input_weight_stats(&self, th: f64) -> (usize, f64, f64, f64, f64) {
        let mut w: Vec<f64> = self.input_idx.iter().flat_map(|&i| self.neurons[i].synapses.iter().map(|s| s.weight)).collect();
        if w.is_empty() { return (0, 0.0, 0.0, 0.0, 0.0); }
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = w.len();
        (n, w[n / 2], w[(n * 9 / 10).min(n - 1)], w[n - 1], w.iter().filter(|&&x| x > th).count() as f64 / n as f64)
    }
    /// 지연 시냅스의 가중치 합과 개수 (관찰용): (가치 뉴런으로 들어오는 것, 예측 뉴런으로 들어오는 것)
    pub fn slow_input_weights(&self) -> ((f64, usize), (f64, usize)) {
        let (mut v, mut p) = ((0.0, 0usize), (0.0, 0usize));
        for &(pre, k) in &self.slow_syn {
            let Some(s) = self.neurons[pre].synapses.get(k) else { continue };
            if s.slow_fast { v.0 += s.weight; v.1 += 1; } else if self.is_pred(s.target as usize) { p.0 += s.weight; p.1 += 1; }
        }
        (v, p)
    }
    /// 가치 뉴런으로 들어오는 시냅스의 가중치 평균: (set 에 속한 뉴런에서 오는 것, 그 밖의 뉴런에서 오는 것)
    pub fn value_input_weights(&self, set: &[NeuronId]) -> (f64, f64) {
        let inset: std::collections::HashSet<usize> = set.iter().map(|&x| x as usize).collect();
        let (mut a, mut na, mut b, mut nb) = (0.0, 0usize, 0.0, 0usize);
        for &(pre, k) in &self.slow_syn {
            let Some(s) = self.neurons[pre].synapses.get(k) else { continue };
            if !s.slow_fast { continue; }
            if inset.contains(&pre) { a += s.weight; na += 1; } else { b += s.weight; nb += 1; }
        }
        (if na > 0 { a / na as f64 } else { 0.0 }, if nb > 0 { b / nb as f64 } else { 0.0 })
    }
    /// 출력 뉴런의 글자 번호 (출력 뉴런이 아니면 None)
    pub fn output_seq(&self, idx: usize) -> Option<usize> { self.output_idx.iter().position(|&o| o == idx) }
    pub fn output_count(&self) -> usize { self.output_idx.len() }
    pub fn token_of(&self, seq: usize) -> String { self.reverse_vocab.get(seq).cloned().unwrap_or_default() }
    pub fn is_input(&self, idx: usize) -> bool { self.input_idx.contains(&idx) }
    pub fn is_output(&self, idx: usize) -> bool { self.output_idx.contains(&idx) }

    /// deliver_reward 와 같지만, 진행하는 동안 조절 수준이 기준선에서 가장 크게 벗어난 값을 peak 에 기록한다 (관찰용)
    pub fn deliver_reward_observed(&mut self, positive: bool, strength: f64, peak: &mut f64) -> u64 {
        let ticks = ((REWARD_TICKS as f64) * strength).round().max(1.0) as u64;
        let idx = if positive { self.reward_idx.clone() } else { self.punish_idx.clone() };
        let before = self.da_spikes;
        let mut during = 0;
        for t in 0..ticks + REWARD_TAIL {
            if t < ticks {
                for &i in &idx { self.neurons[i].potential = 1.0; self.neurons[i].self_threshold = 0.5; self.neurons[i].excitability = 1.0; }
            }
            self.tick();
            if t + 1 == ticks { during = self.da_spikes - before; }
            let d = self.mod_deviation();
            if d.abs() > peak.abs() { *peak = d; }
        }
        during
    }

    /// 조절 신호를 받는 뉴런들의 (조절 수준 ÷ 기준선 − 1) 평균 — 0 이면 평소 상태
    pub fn mod_deviation(&self) -> f64 {
        let (mut sum, mut n) = (0.0, 0usize);
        for x in &self.neurons {
            if x.mod_base > 1e-6 { sum += x.mod_level / x.mod_base - 1.0; n += 1; }
        }
        if n > 0 { sum / n as f64 } else { 0.0 }
    }

    pub fn da_count(&self) -> usize { self.da_idx.len() }
    /// pred_floor: 도파민 뉴런에 "수상돌기 기여를 0 아래로 내리지 않는다" 표시를 켠다 (옵션이 꺼져 있으면 아무것도 하지 않는다)
    fn apply_pred_floor(&mut self) {
        if !self.params.pred_floor { return; }
        for &i in &self.da_idx.clone() { if let Some(n) = self.neurons.get_mut(i) { n.dend_floor = true; } }
    }
    /// 예측 뉴런 수 (관찰용)
    pub fn pred_count(&self) -> usize { self.pred_idx.len() }

    /// 스스로 발화하는 뉴런과 보상/처벌 입력: 자극에 대한 반응으로 세지 않는다
    #[inline]
    fn is_aux(&self, idx: usize) -> bool {
        match self.da_idx.first() { Some(&start) => idx >= start && idx < self.aux_end, None => false }
    }

    fn register_token(&mut self, token: &str) -> u32 {
        if let Some(&seq) = self.vocab.get(token) {
            return seq;
        }
        let seq = self.reverse_vocab.len() as u32;
        self.vocab.insert(token.to_string(), seq);
        self.reverse_vocab.push(token.to_string());

        let x = (seq % 20) as f32;
        let y_in = -10.0 + (seq / 20) as f32;
        let input_id = self.neurons.len() as NeuronId;
        self.neurons.push(Neuron::new_io(input_id, x, y_in));
        self.input_idx.push(input_id as usize);

        let y_out = 200.0 + (seq / 20) as f32;
        let output_id = self.neurons.len() as NeuronId;
        let mut out_neuron = Neuron::new_io(output_id, x, y_out);
        // 출력 뉴런 excitability 기본값(1.0) 유지
        self.neurons.push(out_neuron);
        self.output_idx.push(output_id as usize);

        seq
    }

    fn get_seq(&mut self, token: &str) -> u32 {
        self.register_token(token)
    }

    #[inline]
    fn input_neuron(&self, seq: u32) -> usize { self.input_idx[seq as usize] }
    #[inline]
    fn output_neuron(&self, seq: u32) -> usize { self.output_idx[seq as usize] }

    fn neuron_region(&self, idx: usize) -> RegionType {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;
        if idx >= es && idx < ee { RegionType::Emotion }
        else if idx >= rs && idx < re { RegionType::Reason }
        else if idx >= ms && idx < me { RegionType::Memory }
        else if idx >= hs && idx < he { RegionType::Hippocampus }
        else {
            if self.output_idx.contains(&idx) { RegionType::Output }
            else { RegionType::Input }
        }
    }

    fn is_output_neuron(&self, idx: usize) -> bool {
        self.output_idx.contains(&idx)
    }

    /// 구조 기반 랜덤 시냅스 생성
    fn seed_random_synapses(&mut self, per_neuron: usize) {
        use rand::prelude::*;
        let mut rng = self.stage_rng(0);

        let (em_s, em_e) = self.emotion_main_range;
        let (emap_s, emap_e) = self.emotion_map_range;
        let (rm_s, rm_e) = self.reason_main_range;
        let (rmap_s, rmap_e) = self.reason_map_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;

        fn to_info(neurons: &[Neuron], range: std::ops::Range<usize>) -> Vec<(usize, f32, f32)> {
            range.map(|i| (i, neurons[i].x, neurons[i].y)).collect()
        }

        let emo_main_info = to_info(&self.neurons, em_s..em_e);
        let emo_map_info = to_info(&self.neurons, emap_s..emap_e);
        let rea_main_info = to_info(&self.neurons, rm_s..rm_e);
        let rea_map_info = to_info(&self.neurons, rmap_s..rmap_e);
        let mem_info = to_info(&self.neurons, ms..me);
        let hippo_info = to_info(&self.neurons, hs..he);
        let output_info: Vec<(usize, f32, f32)> = self.output_idx.iter()
            .map(|&i| (i, self.neurons[i].x, self.neurons[i].y)).collect();

        // Main 영역 (입력이 연결되는 곳): 감정_Main + 이성_Main + 기억
        let main_areas: Vec<(usize, f32, f32)> = emo_main_info.iter()
            .chain(rea_main_info.iter())
            .chain(mem_info.iter())
            .cloned().collect();

        // Map 영역: 감정_Map + 이성_Map
        let map_areas: Vec<(usize, f32, f32)> = emo_map_info.iter()
            .chain(rea_map_info.iter())
            .cloned().collect();

        fn pick_random(rng: &mut rand::rngs::StdRng, candidates: &[(usize, f32, f32)], exclude: usize) -> Option<usize> {
            if candidates.len() <= 1 { return None; }
            for _ in 0..10 {
                let idx = rng.random_range(0..candidates.len());
                if candidates[idx].0 != exclude {
                    return Some(candidates[idx].0);
                }
            }
            None
        }

        // 격자 위에서 (col, row) 근처의 한 칸을 고른다 (정규분포, 범위를 벗어나면 다시). list 는 한 영역의 뉴런들이 격자 순서로 들어 있다
        fn pick_near(rng: &mut rand::rngs::StdRng, list: &[(usize, f32, f32)], cols: usize, col: f64, row: f64, sigma: f64) -> Option<usize> {
            let rows = list.len() / cols;
            for _ in 0..10 {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random();
                let r = (-2.0 * u1.ln()).sqrt() * sigma;
                let (c, w) = ((col + r * (std::f64::consts::TAU * u2).cos()).round(), (row + r * (std::f64::consts::TAU * u2).sin()).round());
                if c < 0.0 || w < 0.0 || c >= cols as f64 || w >= rows as f64 { continue; }
                return Some(list[w as usize * cols + c as usize].0);
            }
            None
        }
        // 영역 내부 연결의 상대: local_frac 의 확률로 가까운 뉴런, 아니면 영역 전체에서 무작위
        fn pick_internal(rng: &mut rand::rngs::StdRng, list: &[(usize, f32, f32)], idx: usize, local_frac: f64) -> Option<usize> {
            if local_frac > 0.0 && rng.random::<f64>() < local_frac && !list.is_empty() {
                let k = idx - list[0].0;
                for _ in 0..5 {
                    if let Some(t) = pick_near(rng, list, 50, (k % 50) as f64, (k / 50) as f64, LOCAL_SIGMA) { if t != idx { return Some(t); } }
                }
            }
            pick_random(rng, list, idx)
        }

        // 가중치 샘플러: 꼬리 긴 분포면 평균 mean 인 로그정규(중앙값 ≈ 0.43×평균, 상위 1% ≈ 9×평균), 최대 1.0
        let p = self.params.clone();
        let sample = move |rng: &mut rand::rngs::StdRng, mean: f64| -> f64 {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            (mean * 1.5 * (1.3 * z).exp() / 3.5).clamp(0.01, 1.0)
        };
        let lognormal = p.lognormal;
        let hid_w = move |rng: &mut rand::rngs::StdRng| if lognormal { sample(rng, INITIAL_WEIGHT) } else { INITIAL_WEIGHT };
        let io_w = move |rng: &mut rand::rngs::StdRng| if lognormal { sample(rng, 0.6) } else { 0.4 + rng.random::<f64>() * 0.4 };

        // 1. 입력 → 감정_Main, 이성_Main, 기억 (흥분성만, weight 0.4~0.8 랜덤)
        let main_excitatory: Vec<(usize, f32, f32)> = main_areas.iter()
            .filter(|&&(idx, _, _)| !self.neurons[idx].inhibitory)
            .cloned().collect();
        let input_indices: Vec<usize> = self.input_idx.clone();
        for &idx in &input_indices {
            if p.topo {
                // 입력마다 영역별로 자리 하나를 정하고 그 주변의 흥분성 뉴런에만 연결한다 (같은 뉴런에 두 번 연결하지 않음)
                let areas: [&Vec<(usize, f32, f32)>; 3] = [&emo_main_info, &rea_main_info, &mem_info];
                let total: usize = areas.iter().map(|a| a.len()).sum();
                let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
                for area in areas {
                    let rows = area.len() / 50;
                    let (col, row) = (rng.random::<f64>() * 50.0, rng.random::<f64>() * rows as f64);
                    // (시험용, --in-fan) 발달 구성에서도 글자 하나가 닿는 은닉 수를 바꾼다: 영역에 나누는 몫과 자리 둘레로 고르는 법은 그대로
                    let n = p.in_fan.unwrap_or(per_neuron * p.input_pop) * area.len() / total;
                    let mut made = 0;
                    for _ in 0..n * 6 {
                        if made >= n { break; }
                        let Some(t) = pick_near(&mut rng, area, 50, col, row, TOPO_SIGMA) else { continue };
                        if self.neurons[t].inhibitory || !used.insert(t) { continue; }
                        let w = match p.in_weight { Some(w) => w, None => io_w(&mut rng) };
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                        made += 1;
                    }
                }
                continue;
            }
            if let Some(fan) = p.in_fan {
                // (시험용, --in-fan) 되풀이 없이 fan 개: 후보 목록의 앞쪽을 한 칸씩 무작위로 바꿔 넣어 고른다
                let mut pool: Vec<usize> = main_excitatory.iter().map(|x| x.0).filter(|&t| t != idx).collect();
                let fan = fan.min(pool.len());
                for k in 0..fan {
                    let j = rng.random_range(k..pool.len());
                    pool.swap(k, j);
                    let tid = self.neurons[pool[k]].id;
                    let w = match p.in_weight { Some(w) => w, None => io_w(&mut rng) };
                    self.neurons[idx].add_seed_synapse(tid, w);
                }
                continue;
            }
            for _ in 0..per_neuron * p.input_pop {
                if let Some(target) = pick_random(&mut rng, &main_excitatory, idx) {
                    let tid = self.neurons[target].id;
                    let w = match p.in_weight { Some(w) => w, None => io_w(&mut rng) };
                    self.neurons[idx].add_seed_synapse(tid, w);
                }
            }
        }

        // 2. 감정_Main: 내부 + → 감정_Map + → 출력
        for &(idx, _, _) in &emo_main_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.8 - p.output_frac {
                    // 내부 연결
                    if let Some(t) = pick_internal(&mut rng, &emo_main_info, idx, p.local_frac) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else if r < 1.0 - p.output_frac {
                    // → 감정_Map
                    if let Some(t) = pick_random(&mut rng, &emo_map_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else {
                    // → 출력
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        {
                        let w_io = io_w(&mut rng);
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w_io);
                    }
                    }
                }
            }
        }

        // 3. 이성_Main: 내부 + → 이성_Map + → 출력
        for &(idx, _, _) in &rea_main_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.8 - p.output_frac {
                    if let Some(t) = pick_internal(&mut rng, &rea_main_info, idx, p.local_frac) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else if r < 1.0 - p.output_frac {
                    if let Some(t) = pick_random(&mut rng, &rea_map_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else {
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        {
                        let w_io = io_w(&mut rng);
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w_io);
                    }
                    }
                }
            }
        }

        // 4. 기억: → 출력 + → 해마
        for &(idx, _, _) in &mem_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.5 {
                    // 내부
                    if let Some(t) = pick_internal(&mut rng, &mem_info, idx, p.local_frac) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else if r < 0.5 + p.output_frac * 2.0 / 3.0 {
                    // → 출력
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        {
                        let w_io = io_w(&mut rng);
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w_io);
                    }
                    }
                } else {
                    // → 해마
                    if let Some(t) = pick_random(&mut rng, &hippo_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                }
            }
        }

        // 5. 감정_Map ↔ 기억, 해마
        let mem_hippo: Vec<(usize, f32, f32)> = mem_info.iter()
            .chain(hippo_info.iter()).cloned().collect();
        for &(idx, _, _) in &emo_map_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.4 {
                    // → 감정_Main (역방향)
                    if let Some(t) = pick_random(&mut rng, &emo_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else {
                    // → 기억 or 해마
                    if let Some(t) = pick_random(&mut rng, &mem_hippo, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                }
            }
        }

        // 6. 이성_Map ↔ 기억, 해마
        for &(idx, _, _) in &rea_map_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.4 {
                    if let Some(t) = pick_random(&mut rng, &rea_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                } else {
                    if let Some(t) = pick_random(&mut rng, &mem_hippo, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                    }
                }
            }
        }

        // 7. 해마 → Map 영역 + 기억 (역방향)
        let map_and_mem: Vec<(usize, f32, f32)> = map_areas.iter()
            .chain(mem_info.iter()).cloned().collect();
        for &(idx, _, _) in &hippo_info {
            for _ in 0..per_neuron {
                if let Some(t) = pick_random(&mut rng, &map_and_mem, idx) {
                    self.neurons[idx].add_seed_synapse(t as NeuronId, hid_w(&mut rng));
                }
            }
        }

        // 8. 세포 종류 구분: 억제성 뉴런은 멀리 투사하지 않고 자기 영역 안에서만 연결한다.
        //    Pv → 추체 80% / Pv 20%,  Sst → 추체 70% / Pv 20% / Vip 10% (Sst 끼리는 안 함),  Vip → Sst
        //    그리고 추체 → Sst 시냅스는 촉진형(연달아 활성화되면 강해짐)으로 표시한다.
        if p.typed {
            for list in [&emo_main_info, &emo_map_info, &rea_main_info, &rea_map_info, &mem_info, &hippo_info] {
                let of = |t: CellType| -> Vec<usize> {
                    list.iter().map(|x| x.0).filter(|&i| self.neurons[i].cell_type == t).collect()
                };
                let (pyr, pv, sst, vip) = (of(CellType::Pyramidal), of(CellType::Pv), of(CellType::Sst), of(CellType::Vip));
                let inhibitory: Vec<usize> = list.iter().map(|x| x.0).filter(|&i| self.neurons[i].inhibitory).collect();
                for idx in inhibitory {
                    self.neurons[idx].synapses.clear();
                    for _ in 0..per_neuron {
                        let r: f64 = rng.random();
                        let pool = match self.neurons[idx].cell_type {
                            CellType::Pv => if r < 0.8 { &pyr } else { &pv },
                            CellType::Sst => if r < 0.7 { &pyr } else if r < 0.9 { &pv } else { &vip },
                            _ => &sst,
                        };
                        if pool.is_empty() { continue; }
                        let t = pool[rng.random_range(0..pool.len())];
                        if t != idx {
                            let w = hid_w(&mut rng);
                            self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                        }
                    }
                }
            }
            // 9. 형태: 추체세포에 수상돌기 구획을 주고, 시냅스가 붙는 위치를 정한다
            if p.morph {
                // 뉴런이 속한 영역 번호 (입출력 뉴런은 영역 없음)
                let mut region = vec![usize::MAX; self.neurons.len()];
                for (r, list) in [&emo_main_info, &emo_map_info, &rea_main_info, &rea_map_info, &mem_info, &hippo_info].iter().enumerate() {
                    for x in list.iter() { region[x.0] = r; }
                }
                let is_pyr: Vec<bool> = self.neurons.iter().map(|n| n.cell_type == CellType::Pyramidal).collect();
                let kinds: Vec<CellType> = self.neurons.iter().map(|n| n.cell_type).collect();
                for n in self.neurons.iter_mut() {
                    if n.cell_type == CellType::Pyramidal { n.has_dendrite = true; }
                }
                for (pre, n) in self.neurons.iter_mut().enumerate() {
                    for s in &mut n.synapses {
                        let post = s.target as usize;
                        if !is_pyr.get(post).copied().unwrap_or(false) { continue; }
                        s.on_dendrite = match kinds[pre] {
                            CellType::Sst => true,                                  // Sst 는 수상돌기를 억제
                            CellType::Pyramidal => region[pre] != region[post],     // 다른 영역에서 온 흥분성 입력
                            _ => false,                                             // Pv 와 입력 뉴런은 세포체
                        };
                    }
                }
            }
            let is_sst: Vec<bool> = self.neurons.iter().map(|n| n.cell_type == CellType::Sst).collect();
            for n in self.neurons.iter_mut() {
                if n.cell_type != CellType::Pyramidal { continue; }
                for s in &mut n.synapses {
                    if is_sst.get(s.target as usize).copied().unwrap_or(false) { s.facilitating = true; }
                }
            }
        }
    }

    /// 자극 주입
    pub fn inject_stimulus(&mut self, text: &str) -> u64 {
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;
        let jamo_list = tokenizer::decompose_to_jamo(text);
        let seqs: Vec<u32> = jamo_list.iter().map(|j| self.get_seq(j)).collect();
        let input_indices: Vec<usize> = seqs.iter().map(|&s| self.input_neuron(s)).collect();
        for &idx in &input_indices {
            self.neurons[idx].potential = 1.0;
        }
        self.active_stimuli.push(ActiveStimulus {
            fire_id,
            input_indices,
            remaining_sustain: 4,
            elapsed_ticks: 0,
            silent_ticks: 0,
            all_spikes: Vec::new(),
            output_spikes: Vec::new(),
            traces: Vec::new(),
            collect_traces: self.debug,
        });
        fire_id
    }

    /// idle 틱
    pub fn idle_tick(&mut self) {
        self.global_tick += 1;
        let global_tick = self.global_tick;

        // 감쇠는 뉴런마다 독립이고 난수를 쓰지 않으므로 병렬로 돌려도 결과가 같다
        self.neurons.par_iter_mut().for_each(|n| n.decay());

        self.deliver_slow_inhibition();

        let threshold = self.threshold;
        let noise_range = self.noise_range;
        let noise_amp = &self.noise_amp;
        let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = self.neurons
            .par_iter_mut()
            .filter_map(|n| {
                n.try_fire(global_tick, threshold, noise_amp.get(n.id as usize).copied().unwrap_or(noise_range), false)
                    .map(|deliveries| (n.id, deliveries))
            })
            .collect();

        let mut tick_spikes: Vec<NeuronId> = Vec::new();
        let slow_events = self.params.slow_events;
        for (nid, deliveries) in &fired {
            tick_spikes.push(*nid);
            for (k, (target, weight)) in deliveries.iter().enumerate() {
                if slow_events && *weight == 0.0 { self.slow_on_spike(*nid as usize, k); }
                let tidx = *target as usize;
                if tidx < self.neurons.len() {
                    let dend = self.neurons[*nid as usize].synapses.get(k).is_some_and(|s| s.on_dendrite);
                    if self.neurons[*nid as usize].modulatory { self.neurons[tidx].receive_mod(weight.abs()); }
                    else if dend { self.neurons[tidx].receive_dend(*weight); } else { self.neurons[tidx].receive(*weight); }
                }
            }
        }

        for &nid in &tick_spikes {
            self.recent_spikes.push((nid, global_tick));
        }

        if global_tick % 10_000 == 0 {
            self.periodic_homeostasis();
            if self.params.slow_events { self.slow_resync(); }
        }
        if self.params.slow_check && global_tick % 1000 == 0 { self.slow_check_report(); }

        if !self.recent_spikes.is_empty() {
            let spikes: Vec<(NeuronId, u64)> = self.recent_spikes.drain(..).collect();
            let _ = spikes;
        }
    }

    pub fn tick(&mut self) -> Vec<u64> {
        self.global_tick += 1;
        let global_tick = self.global_tick;
        if self.wake.is_some() { self.wake_stage(0); }

        // 연속 모드의 입력: 남은 틱 동안 켜 둔다
        if !self.drives.is_empty() {
            for d in 0..self.drives.len() {
                for k in 0..self.drives[d].0.len() {
                    let idx = self.drives[d].0[k];
                    self.neurons[idx].potential = 1.0;
                    self.neurons[idx].self_threshold = 0.5;
                    self.neurons[idx].excitability = 1.0;
                }
                self.drives[d].1 -= 1;
            }
            self.drives.retain(|d| d.1 > 0);
        }
        self.tick_ids.clear();

        for stim in &mut self.active_stimuli {
            stim.elapsed_ticks += 1;
            if stim.remaining_sustain > 0 {
                for &idx in &stim.input_indices {
                    self.neurons[idx].potential = 1.0;
                    // 입력 뉴런은 감각기: 자가 임계 적응을 하면 자주 쓰는 입력일수록
                    // 발화 확률이 50%까지 떨어진다 (초파리 테스트에서 확인) → 기본값 고정
                    self.neurons[idx].self_threshold = 0.5;
                    // 항상성도 마찬가지: 자주 쓰는 입력은 발화율 5% 초과로 감응도가 계속 깎여
                    // 결국 자극이 임계값을 못 넘는다 (초파리 테스트 350회차에서 확인) → 1.0 고정
                    self.neurons[idx].excitability = 1.0;
                }
                stim.remaining_sustain -= 1;
            }
        }

        if self.wake.is_some() { self.wake_stage(1); }
        // 감쇠는 뉴런마다 독립이고 난수를 쓰지 않으므로 병렬로 돌려도 결과가 같다
        self.neurons.par_iter_mut().for_each(|n| n.decay());
        if self.wake.is_some() { self.wake_stage(2); }

        self.deliver_slow_inhibition();
        if self.wake.is_some() { self.wake_stage(3); }

        let threshold = self.threshold;
        let noise_range = self.noise_range;
        let noise_amp = &self.noise_amp;

        // 1대1 방식의 자극이 진행 중일 때만 "자극 중"으로 본다 (연속 모드와 휴지 구간은 평소 확률)
        let stimulated = !self.active_stimuli.is_empty();
        let fired: Vec<(NeuronId, f64, Vec<(NeuronId, f64)>)> = self.neurons
            .par_iter_mut()
            .filter_map(|n| {
                let pot = n.potential;
                n.try_fire(global_tick, threshold, noise_amp.get(n.id as usize).copied().unwrap_or(noise_range), stimulated)
                    .map(|deliveries| (n.id, pot, deliveries))
            })
            .collect();

        if self.wake.is_some() { let ids: Vec<NeuronId> = fired.iter().map(|f| f.0).collect(); self.wake_after_fire(&ids); }
        let mut tick_spikes: Vec<NeuronId> = Vec::new();
        let output_idx_snapshot = self.output_idx.clone();

        // 자극에 대한 반응으로 세는 발화 수 (스스로 발화하는 도파민 뉴런과 보상/처벌 입력은 제외)
        let mut response_spikes = 0usize;
        let mut output_spikes_now = 0usize;
        let slow_events = self.params.slow_events;
        // 전달 기록을 모으는 자극이 있는가 (틱마다 한 번만 본다)
        let collecting = self.active_stimuli.iter().any(|st| st.collect_traces);
        for (nid, from_potential, deliveries) in &fired {
            let nid = *nid;
            let from_potential = *from_potential;
            tick_spikes.push(nid);

            let is_output = output_idx_snapshot.contains(&(nid as usize));
            let aux = self.is_aux(nid as usize);
            if aux {
                if self.neurons[nid as usize].modulatory { self.da_spikes += 1; }
                if self.is_pred(nid as usize) { self.pred_spikes += 1; }
                if self.val_idx.first().is_some_and(|&v| nid as usize >= v && (nid as usize) < v + self.val_idx.len()) { self.val_spikes += 1; }
            } else {
                response_spikes += 1;
                if is_output { output_spikes_now += 1; }
                self.tick_ids.push(nid);
                for stim in &mut self.active_stimuli {
                    stim.all_spikes.push((nid, global_tick));
                    if is_output {
                        stim.output_spikes.push((nid, global_tick));
                    }
                }
            }

            for (k, (target, weight)) in deliveries.iter().enumerate() {
                let target = *target;
                let weight = *weight;
                if slow_events { self.slow_on_spike(nid as usize, k); }
                let tidx = target as usize;
                if tidx >= self.neurons.len() { continue; }
                // 값이 0 인 전달(확률 전달의 실패, 지연 시냅스)은 건너뛴다: 대상에 0 을 더할 뿐이라 결과가 같다.
                // 전달 기록을 모으는 자극이 있으면 건너뛰지 않는다. 조절 수준에 따른 가소성(mod_eta ≠ 0)은 값과 무관하게 시냅스를 고치므로 지금처럼 부른다
                if weight == 0.0 && !collecting {
                    if self.plasticity.mod_eta != 0.0 && !self.neurons[nid as usize].modulatory && self.neurons[tidx].mod_level > 0.001 {
                        plasticity::on_pre_under_modulator(&mut self.neurons, nid as usize, k, &self.plasticity);
                    }
                    continue;
                }
                // wake_check: 0 이 아닌 전달을 받은 뉴런은 다음 틱에 깨어난다 (조절 신호는 따로 표시)
                if weight != 0.0 { if let Some(w) = self.wake.as_mut() { w.next[tidx] |= if self.neurons[nid as usize].modulatory { 2 } else { 1 }; } }

                let to_before = self.neurons[tidx].potential;
                // 시냅스가 붙은 위치에 따라 세포체 또는 수상돌기로 전달
                let dend = self.neurons[nid as usize].synapses.get(k).is_some_and(|s| s.on_dendrite);
                if self.neurons[nid as usize].modulatory {
                    // 조절 뉴런: 전위는 올리지 않고 조절 수준만 올린다 + 조절 신호에 의한 가소성
                    self.neurons[tidx].receive_mod(weight.abs());
                    plasticity::on_modulator(&mut self.neurons, tidx, weight.abs(), global_tick, &self.plasticity);
                } else {
                    if dend { self.neurons[tidx].receive_dend(weight); } else { self.neurons[tidx].receive(weight); }
                    if self.neurons[tidx].mod_level > 0.001 {
                        plasticity::on_pre_under_modulator(&mut self.neurons, nid as usize, k, &self.plasticity);
                    }
                }

                for stim in &mut self.active_stimuli {
                    if stim.collect_traces {
                        stim.traces.push(DeliveryTrace {
                            from: nid, to: target, from_potential,
                            to_before, weight, delivered: weight,
                            tick: global_tick,
                        });
                    }
                }
            }
        }

        // STDP + iSTDP (plasticity.rs — 초파리 크레이트와 공유)
        plasticity::stdp_step(&mut self.neurons, &tick_spikes, global_tick, &self.plasticity);
        if self.params.slow_events {
            if self.params.slow_check { self.slow_shadow_step(); }
            let mut deltas: Vec<(usize, f64, f64)> = Vec::new();
            plasticity::predictor_step_events(&mut self.neurons, &self.slow_recv, &mut self.slow_live, &mut self.slow_listed, global_tick, &self.plasticity, &mut deltas);
            for (p, d1, d2) in deltas { self.slow_sum[p][0] += d1; self.slow_sum[p][1] += d2; }
        } else {
            plasticity::predictor_step(&mut self.neurons, &self.slow_syn, global_tick, &self.plasticity);
        }

        for &nid in &tick_spikes {
            if !self.is_aux(nid as usize) { self.recent_spikes.push((nid, global_tick)); }
        }

        let end_on_output = self.params.end_on_output;
        for stim in &mut self.active_stimuli {
            // 출력 기준: 출력이 한 번이라도 나온 뒤에는 출력 뉴런만 본다 (내부 활동이 이어져도 표현은 끝난 것).
            // 출력이 아직 없으면 기존처럼 전체 활동을 본다 — 출력이 나오기 전에 끊지 않기 위해서
            let has_spikes = if end_on_output && !stim.output_spikes.is_empty() { output_spikes_now > 0 } else { response_spikes > 0 };
            if has_spikes {
                stim.silent_ticks = 0;
            } else if stim.remaining_sustain == 0 {
                stim.silent_ticks += 1;
            }
        }

        let mut completed_ids: Vec<u64> = Vec::new();
        let mut completed_stimuli: Vec<ActiveStimulus> = Vec::new();
        let mut remaining: Vec<ActiveStimulus> = Vec::new();

        for stim in self.active_stimuli.drain(..) {
            // 재귀 연결로 활동이 스스로 유지되면 영원히 끝나지 않으므로 상한을 둔다
            if stim.silent_ticks >= 10 || stim.elapsed_ticks >= MAX_STIMULUS_TICKS {
                completed_ids.push(stim.fire_id);
                completed_stimuli.push(stim);
            } else {
                remaining.push(stim);
            }
        }
        self.active_stimuli = remaining;

        for stim in completed_stimuli {

            // output_spikes는 (nid, tick) 시간순으로 기록됨 → 발화 순서대로 토큰 출력
            let mut output_tokens: Vec<(String, f64)> = Vec::new();
            let mut sorted_spikes = stim.output_spikes.clone();
            sorted_spikes.sort_by_key(|&(_, t)| t);
            for &(nid, _) in &sorted_spikes {
                let idx = nid as usize;
                if let Some(seq) = self.output_idx.iter().position(|&oi| oi == idx) {
                    if let Some(token) = self.reverse_vocab.get(seq) {
                        output_tokens.push((token.clone(), 1.0));
                    }
                }
            }

            let token_strs: Vec<String> = output_tokens.iter().map(|(t, _)| t.clone()).collect();
            let output = tokenizer::recompose_tokens(&token_strs);

            eprintln!(
                "  [fire #{}] ticks={}, spikes={}, output_spikes={}",
                stim.fire_id, stim.elapsed_ticks,
                stim.all_spikes.len(), stim.output_spikes.len(),
            );

            self.fire_records.push(FireRecord {
                id: stim.fire_id,
                output,
                output_tokens,
                traces: stim.traces,
                spiked_neurons: stim.all_spikes,
            });
            if self.fire_records.len() > 100 {
                self.fire_records.remove(0);
            }
        }

        if global_tick % 10_000 == 0 {
            self.periodic_homeostasis();
            if self.params.slow_events { self.slow_resync(); }
        }
        if self.params.slow_check && global_tick % 1000 == 0 { self.slow_check_report(); }
        if self.wake.is_some() && global_tick % 1000 == 0 { self.wake_report(); }

        if !self.recent_spikes.is_empty() {
            let spikes: Vec<(NeuronId, u64)> = self.recent_spikes.drain(..).collect();
            if let Some(rate) = self.params.grow { self.run_prune(&spikes); self.grow_step(&spikes, rate); }
        }

        completed_ids
    }

    /// 수면
    pub fn enter_sleep(&mut self) {
        let enter_tick = self.global_tick;
        let fires = self.fires_since_sleep;

        // 1. 시냅스 fatigue 감소 + 균일 weight scaling (SHY 가설)
        const SLEEP_WEIGHT_DECAY: f64 = 0.95;
        let mut scaled = 0usize;
        for n in &mut self.neurons {
            for s in &mut n.synapses {
                s.fatigue *= SLEEP_WEIGHT_SCALE;
                s.weight *= SLEEP_WEIGHT_DECAY;
                scaled += 1;
            }
        }

        // 2. 전역 흥분 상태 리셋
        for n in &mut self.neurons {
            n.potential = 0.0;
            n.excitability = 1.0;
            n.fire_count_window = 0;
        }

        // 3. 해마 재생: 해마 뉴런을 주기적으로 자극 → 시냅스 통해 전파 + STDP 적용
        let (hs, he) = self.hippocampus_range;
        const REPLAY_ROUNDS: usize = 5;
        const REPLAY_TICKS: u64 = 10;
        const STDP_A_PLUS: f64 = 0.003;
        const STDP_A_MINUS: f64 = 0.0036;
        const STDP_TAU: f64 = 20.0;
        const BCM_TARGET_RATE: f64 = 25.0;
        for _ in 0..REPLAY_ROUNDS {
            for replay_tick in 0..REPLAY_TICKS {
                self.global_tick += 1;
                let global_tick = self.global_tick;
                let threshold = self.threshold;
                for n in &mut self.neurons {
                    n.decay();
                }
                // 해마 자극은 decay 이후에 주입 (decay 전에 0.5를 넣으면
                // 0.5×decay_rate < threshold 가 되어 아무도 발화하지 못함)
                if replay_tick == 0 {
                    for idx in hs..he {
                        if idx < self.neurons.len() {
                            self.neurons[idx].potential = threshold.max(1.0);
                        }
                    }
                }
                let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = self.neurons
                    .iter_mut()
                    .filter_map(|n| {
                        if n.potential >= threshold {
                            n.potential = 0.0;
                            n.last_spike_tick = Some(global_tick);
                            let sign = if n.inhibitory { -1.0 } else { 1.0 };
                            let deliveries: Vec<_> = n.synapses.iter()
                                .map(|s| (s.target, s.effective_weight() * sign))
                                .collect();
                            Some((n.id, deliveries))
                        } else { None }
                    })
                    .collect();
                for (_, deliveries) in &fired {
                    for &(target, weight) in deliveries {
                        let tidx = target as usize;
                        if tidx < self.neurons.len() {
                            self.neurons[tidx].receive(weight);
                        }
                    }
                }
                // 재생 STDP Pass 1: pre 기준 (동시 + LTD)
                let tick_spikes: Vec<NeuronId> = fired.iter().map(|(nid, _)| *nid).collect();
                let tick_spike_set: std::collections::HashSet<NeuronId> =
                    tick_spikes.iter().cloned().collect();
                let spike_ticks: Vec<Option<u64>> = self.neurons.iter().map(|n| n.last_spike_tick).collect();
                let fire_counts: Vec<u32> = self.neurons.iter().map(|n| n.fire_count_window).collect();
                for &nid in &tick_spikes {
                    let pidx = nid as usize;
                    if pidx >= self.neurons.len() { continue; }
                    for syn in &mut self.neurons[pidx].synapses {
                        let tidx = syn.target as usize;
                        if tidx >= spike_ticks.len() { continue; }
                        if let Some(post_tick) = spike_ticks[tidx] {
                            let dt = post_tick as f64 - global_tick as f64;
                            let diff_ltp = syn.weight - 0.2;
                            let diff_ltd = syn.weight - 1.0;
                            let f_ltp = (-(diff_ltp * diff_ltp) / (2.0 * 0.4 * 0.4)).exp();
                            let f_ltd = (-(diff_ltd * diff_ltd) / (2.0 * 0.4 * 0.4)).exp();
                            // BCM: post 자주 발화 → LTP 어려움, LTD 쉬움
                            let post_bcm = BCM_TARGET_RATE / (fire_counts[tidx] as f64 + BCM_TARGET_RATE);
                            let a_plus = STDP_A_PLUS * post_bcm;
                            let a_minus = STDP_A_MINUS * (2.0 - post_bcm);
                            if dt == 0.0 {
                                syn.weight = (syn.weight + a_plus * f_ltp).min(1.0);
                            } else if dt < 0.0 && dt > -50.0 {
                                let dw = a_minus * (dt / STDP_TAU).exp() * f_ltd;
                                syn.weight = (syn.weight - dw).max(0.0);
                            }
                        }
                    }
                }
                // 재생 STDP Pass 2: post 기준 (incoming 역참조로 진짜 LTP) + BCM
                for &post_id in &tick_spikes {
                    let post_idx = post_id as usize;
                    if post_idx >= self.neurons.len() { continue; }
                    let post_bcm = BCM_TARGET_RATE / (fire_counts[post_idx] as f64 + BCM_TARGET_RATE);
                    let a_plus = STDP_A_PLUS * post_bcm;
                    let pairs = self.neurons[post_idx].incoming.clone();
                    for (pre_idx, syn_idx) in pairs {
                        let pre_tick = match spike_ticks.get(pre_idx).copied().flatten() {
                            Some(t) if t < global_tick && global_tick - t < 50 => t,
                            _ => continue,
                        };
                        if syn_idx >= self.neurons[pre_idx].synapses.len() { continue; }
                        let dt = (global_tick - pre_tick) as f64;
                        let syn = &mut self.neurons[pre_idx].synapses[syn_idx];
                        let diff_ltp = syn.weight - 0.2;
                        let f_ltp = (-(diff_ltp * diff_ltp) / (2.0 * 0.4 * 0.4)).exp();
                        let dw = a_plus * (-dt / STDP_TAU).exp() * f_ltp;
                        syn.weight = (syn.weight + dw).min(1.0);
                    }
                }
            }
            for n in &mut self.neurons {
                n.potential = 0.0;
            }
        }

        // 4. 휴지기: 순수 감쇠 (fatigue 자연 회복 포함)
        for _ in 0..SLEEP_DURATION_TICKS {
            self.global_tick += 1;
            for n in &mut self.neurons {
                n.decay();
            }
        }

        self.fires_since_sleep = 0;
        self.last_sleep_tick = self.global_tick;
        self.recent_spikes.clear();

        eprintln!(
            "  [sleep] tick {} → {} (지속 {}틱, 누적 {}fire, {}시냅스, 해마재생 {}회)",
            enter_tick, self.global_tick, SLEEP_DURATION_TICKS,
            fires, scaled, REPLAY_ROUNDS,
        );
    }

    fn should_sleep(&self) -> bool {
        self.fires_since_sleep >= SLEEP_FIRE_INTERVAL
            || self.global_tick.saturating_sub(self.last_sleep_tick) >= SLEEP_TICK_INTERVAL
    }

    /// 발화 (동기 방식)
    pub fn fire(&mut self, text: &str) -> u64 {
        if !self.params.validated && self.should_sleep() {
            self.enter_sleep();
        }
        self.fires_since_sleep += 1;

        let fire_id = self.inject_stimulus(text);
        loop {
            let completed = self.tick();
            if completed.contains(&fire_id) { break; }
        }

        fire_id
    }

    /// 발달 과정의 성장 (params.grow): 발화한 은닉 뉴런이 자기 시냅스 수가 목표보다 적을수록 더 많은 새 시냅스를 만든다.
    ///   상대: 흥분성은 80% 가 같은 영역의 가까운 뉴런, 20% 가 은닉 전체에서 무작위. 억제성은 같은 영역의 가까운 뉴런만
    ///   가중치: 자기가 이미 가진 시냅스 하나를 무작위로 골라 그 값을 쓴다 (자기 시냅스들의 분포를 따른다)
    fn grow_step(&mut self, spikes: &[(NeuronId, u64)], rate: f64) {
        use rand::prelude::*;
        // 성장의 난수: 평소에는 전역 난수기 하나를 순서대로 쓴다(기존). 결정적 난수면 발화한 뉴런마다
        // (씨앗, 5, 뉴런, 0, 틱)에서 작은 난수기를 만들어 그 뉴런의 성장에만 쓴다
        let seed = self.params.rng_seed.map(|s| crate::neuron::keyed_u64(s, 0, 0, 0, 0) | 1);
        let mut rng = GrowDraw::Shared(rand::rng());
        let hidden_end = self.hippocampus_range.1;
        let now = self.global_tick;
        let target = self.params.target_syn as f64;
        let regions = [(self.emotion_range, 50usize), (self.reason_range, 50), (self.memory_range, 50), (self.hippocampus_range, 25)];
        let gated = self.params.grow_gate.is_some();
        // 재는 중인 받는 뉴런으로 가는 시도를 버리는가 (own_target_open 이면 버리지 않는다)
        let own_mode = self.params.own_target.is_some() && !self.params.own_target_open;
        // 출력으로 가는 성장 (grow_out): 출력의 목표는 추체의 목표. 목표가 없으면(act_target 이나 spont_fixed 가 없으면) 출력으로 자라지 않는다
        let out_goal = self.out_target();
        let out_on = self.params.grow_out && gated && out_goal > 0.0 && !self.output_idx.is_empty();
        for &(nid, _) in spikes {
            let pre = nid as usize;
            if pre >= hidden_end { continue; }
            if let Some(sd) = seed { rng = GrowDraw::Keyed(rand::rngs::StdRng::seed_from_u64(crate::neuron::keyed_u64(sd, 5, pre as u64, 0, now))); }
            // 뉴런이 세는 것은 연결 수가 아니라 자기 시냅스 접점 수의 합
            let n = self.neurons[pre].synapses.iter().map(|s| s.count as f64).sum::<f64>();
            // 목표가 0 이면 개수를 보지 않는다: 성장은 일정하고, 개수는 소멸과의 균형으로 정해진다
            let drive = if target > 0.0 { (target / (n + 1.0)).powi(2).min(100.0) } else { 1.0 };
            let da = match self.params.grow_da {
                // 평소의 흔들림(±25%)은 세지 않는다: 그것만으로도 계속 자라게 된다
                Some(b) => { let n = &self.neurons[pre]; b + if n.mod_base > 1e-6 { (n.mod_level / n.mod_base - 1.0 - plasticity::MOD_DEADBAND).max(0.0) } else { 0.0 } }
                None => 1.0,
            };
            let expect = rate * drive * da;
            let mut count = expect.floor() as usize;
            if rng.f() < expect - expect.floor() { count += 1; }
            let inhibitory = self.neurons[pre].inhibitory;
            if count > 0 {
                if let Some(&((start, end), cols)) = regions.iter().find(|r| pre >= r.0.0 && pre < r.0.1) {
                    let (k, rows) = (pre - start, (end - start) / cols);
                    let (col, row) = ((k % cols) as f64, (k / cols) as f64);
                    for _ in 0..count {
                        let post = if inhibitory || rng.f() < 0.8 {
                            let u1: f64 = rng.f().max(1e-12);
                            let u2: f64 = rng.f();
                            let r = (-2.0 * u1.ln()).sqrt() * self.params.grow_sigma;
                            let (c, w) = ((col + r * (std::f64::consts::TAU * u2).cos()).round(), (row + r * (std::f64::consts::TAU * u2).sin()).round());
                            if c < 0.0 || w < 0.0 || c >= cols as f64 || w >= rows as f64 { continue; }
                            start + w as usize * cols + c as usize
                        } else { rng.below(hidden_end) };
                        if post == pre { continue; }
                        // 받는 뉴런이 자기 목표를 재는 중이면 이 시도를 버린다 (목표 0 은 '문 없음'이 아니라 '닫힘'). 걸린 시도로 센다
                        if own_mode && self.neurons[post].own_on && self.neurons[post].own_target == 0.0 {
                            self.gate_drop[Self::kind_index(self.neurons[post].cell_type)] += 1; self.own_wait_drop += 1; continue;
                        }
                        // 받기 문의 목표: 받는 뉴런 자신의 값만 읽는다 (문이 없으면 0 = 걸지 않는다)
                        let goal = if gated { self.act_target_of(post) } else { 0.0 };
                        self.grow_attempt(pre, post, goal, &mut rng, now);
                    }
                }
            }
            // 출력으로 가는 시도: 은닉으로 가는 시도를 다 한 뒤에 얹는다. 억제 뉴런은 하지 않는다(성장에서 억제 뉴런은 가까운 칸으로만 간다).
            //   몫은 처음 배선에서 출력으로 가던 몫 그대로: 감정 주·이성 주 output_frac, 기억 output_frac × 2/3, 그 밖(지도·해마) 0
            if !out_on || inhibitory { continue; }
            let within = |r: (usize, usize)| pre >= r.0 && pre < r.1;
            let frac = if within(self.emotion_main_range) || within(self.reason_main_range) { self.params.output_frac }
                else if within(self.memory_range) { self.params.output_frac * 2.0 / 3.0 } else { 0.0 };
            if frac <= 0.0 { continue; }
            // 결정적 난수면 (씨앗, 7, 뉴런, 0, 틱)의 따로 된 난수기: 은닉 성장이 뽑는 난수의 차례를 건드리지 않는다. 씨앗이 없으면 전역 난수기를 이어 쓴다
            if let Some(sd) = seed { rng = GrowDraw::Keyed(rand::rngs::StdRng::seed_from_u64(crate::neuron::keyed_u64(sd, 7, pre as u64, 0, now))); }
            let expect = rate * da * frac;
            let mut count = expect.floor() as usize;
            if rng.f() < expect - expect.floor() { count += 1; }
            for _ in 0..count {
                let post = self.output_idx[rng.below(self.output_idx.len())];
                self.grow_attempt(pre, post, out_goal, &mut rng, now);
            }
        }
    }

    /// 출력 뉴런에 거는 받기 문의 목표 (틱당 발화율): 추체의 목표 = act_target × spont_fixed. 둘 중 하나라도 없으면 0
    fn out_target(&self) -> f64 {
        match (self.params.act_target, self.params.spont_fixed) { (Some(k), Some(p)) => k * p, _ => 0.0 }
    }

    /// 새 접점 시도 하나 (pre → post): 받기 문(goal > 0 일 때만) → 이미 이어진 쌍이면 접점 +1, 아니면 새 쌍. 은닉으로 가는 시도와 출력으로 가는 시도가 함께 쓴다
    fn grow_attempt(&mut self, pre: usize, post: usize, goal: f64, rng: &mut GrowDraw, now: u64) {
        let min_w = self.params.min_weight;
        // 받기 문: 받는 뉴런의 발화율이 목표를 넘으면 이 시도를 버린다 (다른 상대로 넘기지 않는다)
        let post_kind = Self::kind_index(self.neurons[post].cell_type);
        if let Some(mode) = self.params.grow_gate {
            if goal > 0.0 {
                let q = &self.neurons[post];
                let closed = match self.params.grow_gate_soft {
                    None => match mode {
                        GateMode::Fast => q.rate_fast > goal,
                        GateMode::Slow => q.rate_avg > goal,
                        GateMode::Both => q.rate_fast > goal || q.rate_avg > goal,
                    },
                    // 차츰 줄이는 문: 목표를 넘은 만큼 받을 확률이 준다. 난수는 이 뉴런의 성장 난수기에서 뽑는다 (목표를 넘었을 때만 하나)
                    Some(k) => {
                        let r = match mode { GateMode::Fast => q.rate_fast, GateMode::Slow => q.rate_avg, GateMode::Both => q.rate_fast.max(q.rate_avg) };
                        r > goal && rng.f() >= (goal / r).powf(k)
                    }
                };
                if closed { self.gate_drop[post_kind] += 1; return; }
            }
        }
        let jitter = 1.0 + self.params.new_spread * (rng.f() * 2.0 - 1.0);
        let (post_med, new_w) = (self.unit_med.get(post).copied().unwrap_or(0.0), self.params.new_weight * jitter);
        // 이미 연결된 상대면 접점이 하나 늘어난다: 접점 하나의 세기(가중치 ÷ 개수)만큼 가중치가 커진다
        if let Some(s) = self.neurons[pre].synapses.iter_mut().find(|s| s.target as usize == post && s.slow_rate == 0.0) {
            // 한 쌍 사이의 접점 수에는 한계가 있다 (축삭과 수상돌기가 만나는 자리가 한정됨). 많을수록 더 생기기 어렵고, 한계에 닿으면 생기지 않는다.
            // 세기가 한계(CONN_CAP)에 닿은 연결에도 더 생기지 않는다 — 세기는 그대로인데 개수만 늘어나는 일을 막는다
            if s.count >= MAX_CONTACTS_PER_PAIR || s.weight >= CONN_CAP * 0.98 { return; }
            if rng.f() < s.count as f64 / MAX_CONTACTS_PER_PAIR as f64 { return; }
            // 늘어나는 접점도 새 접점과 같은 세기로 시작한다
            let unit = if post_med > 0.0 { post_med * new_w } else { s.weight / s.count.max(1) as f64 };
            // 이미 강한 연결일수록 덜 보탠다 (한계에 닿으면 0)
            let room = (1.0 - s.weight / CONN_CAP).max(0.0);
            s.weight += unit.max(2.0 * min_w) * room;
            s.count += 1;
            let initial = s.born == 0;
            self.sprouted += 1;
            self.grow_made[post_kind] += 1;
            if initial { self.added_on_initial += 1; }
            return;
        }
        let len = self.neurons[pre].synapses.len();
        // 새 연결: 접점 하나짜리. 세기는 받는 뉴런의 입력에서 접점 하나의 세기 중앙값 × new_weight (보통보다 약하게 시작).
        // 받는 뉴런에 아직 그 값이 없으면 보내는 뉴런이 가진 연결들의 중앙값을 쓴다
        let um = self.unit_med.get(post).copied().unwrap_or(0.0);
        let base = if um > 0.0 { um } else { self.neurons[pre].median_out_weight() };
        // 너무 약하면 있으나 마나이고 곧 사라지므로 가지치기 기준의 2배를 바닥으로, 너무 강하지 않게 한계를 천장으로 둔다
        let w = (base * new_w).max(2.0 * min_w).min(CONN_CAP);
        let mut syn = crate::synapse::Synapse::new(post as NeuronId, w);
        syn.born = now;
        syn.t = now;
        // 형태 규칙: 추체세포가 다른 영역의 추체세포에 붙으면 수상돌기, Sst 는 추체세포의 수상돌기, 추체 → Sst 는 촉진형
        let (pt, qt) = (self.neurons[pre].cell_type, self.neurons[post].cell_type);
        if self.neurons[post].has_dendrite {
            syn.on_dendrite = pt == CellType::Sst || (pt == CellType::Pyramidal && self.area_of(pre) != self.area_of(post));
        }
        if pt == CellType::Pyramidal && qt == CellType::Sst { syn.facilitating = true; }
        syn.last_co = now;
        if let Some(d) = self.plasticity.short_decay { syn.short_decay = d; }
        if let Some(p) = self.params.release { syn.p_rel = p; }
        self.neurons[pre].synapses.push(syn);
        self.neurons[post].incoming.push((pre, len));
        self.sprouted += 1;
        self.grow_made[post_kind] += 1;
    }

    /// 자기 목표의 관찰값 (읽기만): 은닉 뉴런마다 (종류 0 추체 1 Pv 2 Sst 3 Vip 4 그 밖, own_on, own_target, 자발 발화 확률 0.00001 × spont_scale, own_n, own_t0)
    pub fn own_observe(&self) -> Vec<(usize, bool, f64, f64, u32, u64)> {
        self.neurons[..self.hippocampus_range.1].iter().map(|n| (Self::kind_index(n.cell_type), n.own_on, n.own_target, 1e-5 * n.spont_scale, n.own_n, n.own_t0)).collect()
    }

    /// 출력 쪽 관찰값 (발달 시험의 "출력" 줄, 읽기만): 출력 뉴런마다 (은닉에서 받는 쌍 수, 그 가운데 자란 쌍(born > 0) 수, 접점 수 합,
    /// 느린 발화율 rate_avg, 감응도 excitability, 내보내는 시냅스 칸 수), 그리고 출력에 거는 목표(없으면 0).
    /// 쌍 = 보내는 뉴런이 은닉이고 받는 뉴런이 출력이며 접점(count)이 1 이상인 시냅스 칸
    pub fn out_observe(&self) -> (Vec<(usize, usize, usize, f64, f64, usize)>, f64) {
        let hidden_end = self.hippocampus_range.1;
        let pos: std::collections::HashMap<usize, usize> = self.output_idx.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let mut v: Vec<(usize, usize, usize, f64, f64, usize)> = self.output_idx.iter().map(|&i| { let n = &self.neurons[i]; (0, 0, 0, n.rate_avg, n.excitability, n.synapses.len()) }).collect();
        for n in &self.neurons[..hidden_end] {
            for s in &n.synapses {
                if s.count == 0 { continue; }
                if let Some(&k) = pos.get(&(s.target as usize)) { v[k].0 += 1; if s.born > 0 { v[k].1 += 1; } v[k].2 += s.count as usize; }
            }
        }
        (v, self.out_target())
    }

    /// 연결이 함께 활동하지 않고 버티는 시간 (틱): prune_tau × ln(가중치 ÷ min_weight), 최소 2만 틱.
    /// 강한 연결일수록 오래 버틴다. 가중치를 실제로 깎지는 않는다
    fn prune_life(&self, weight: f64) -> u64 {
        let x = (weight / self.params.min_weight.max(1e-12)).max(1.0).ln();
        (self.params.prune_tau * x).max(20_000.0) as u64
    }

    /// 가지치기의 뒷정리. 가지치기 자체는 뉴런이 발화할 때 신호를 보내기 전에 스스로 한다(Neuron::prune_own).
    /// 여기서는 방금 가지치기를 한 뉴런에서 죽은 연결(접점 0개)의 자리를 비우고, 받는 쪽의 목록을 고친다
    fn run_prune(&mut self, spikes: &[(NeuronId, u64)]) {
        let hidden_end = self.hippocampus_range.1;
        let now = self.global_tick;
        for &(nid, _) in spikes {
            let post = nid as usize;
            if post >= hidden_end || !self.neurons[post].did_prune { continue; }
            self.neurons[post].did_prune = false;
            // (1) 자기가 내보내는 쪽: 가지치기는 이미 했고, 죽은 연결의 자리만 비운다.
            //     뒤에서부터 본다: 연결을 지우면 마지막 시냅스가 그 자리로 오는데, 마지막 것은 이미 본 뒤다
            if self.neurons[post].pruned_n > 0 {
                let total = self.neurons[post].pruned_n as u64;
                self.pruned += total;
                self.neurons[post].pruned_n = 0;
                // 관찰용: 사라진 접점이 속한 쌍의 나이. 쌍이 통째로 사라진 것은 바로 알 수 있고, 접점만 하나 준 쌍은
                // 방금 시계가 다시 맞춰진 것(last_co = 지금)으로 찾는다. 그 수가 맞지 않으면(같은 틱에 굳은 접점이 섞이면) 모름으로 센다
                let mut dead = 0u64;
                let mut reset: Vec<u64> = Vec::new();
                for s in &self.neurons[post].synapses {
                    if s.slow_rate > 0.0 { continue; }
                    if s.count == 0 { dead += 1; self.prune_ages.push(now.saturating_sub(s.born)); }
                    else if s.last_co == now { reset.push(now.saturating_sub(s.born)); }
                }
                let dec = total.saturating_sub(dead);
                if reset.len() as u64 == dec { self.prune_ages.extend(reset); } else { self.prune_age_unknown += dec; }
                let mut k = self.neurons[post].synapses.len();
                while k > 0 {
                    k -= 1;
                    let s = &self.neurons[post].synapses[k];
                    if s.count == 0 && s.slow_rate == 0.0 { self.remove_synapse(post, k); }
                }
            }
            // (2) 자기에게 들어오는 쪽: 보내는 뉴런이 조용해서 스스로 정리하지 못한 시냅스를, 받는 뉴런이 활동할 때 확인한다.
            //     (시냅스는 양쪽에 걸쳐 있으므로 어느 쪽이든 활동하는 쪽이 본다)
            //     받는 목록의 (보내는 뉴런, 번호)로 그 시냅스를 바로 읽는다. 같은 쌍에 시냅스가 여럿이면(초기 배선의 중복) 각각 한 번씩 점검받는다.
            //     목록을 복사해 돈다: 도중에 칸을 지우면 번호가 옮겨지므로, 읽은 시냅스가 자기를 향하는지 확인하고 아니면 건너뛴다(다음 회차에 본다)
            let sources: Vec<(usize, usize)> = self.neurons[post].incoming.iter().cloned().filter(|e| e.0 < hidden_end).collect();
            // 활동 의존 소멸: 받는 뉴런(지금 발화한 이 뉴런)의 느린 발화율이 자기 목표보다 높으면 들어오는 접점의 수명이 (목표 ÷ 느린 발화율)^E 배가 된다
            let shrink = if self.params.prune_act > 0.0 {
                let (goal, slow) = (self.act_target_of(post), self.neurons[post].rate_avg);
                if goal > 0.0 && slow > goal { (goal / slow).powf(self.params.prune_act) } else { 1.0 }
            } else { 1.0 };
            for (pre, k) in sources {
                match self.neurons[pre].synapses.get(k) { Some(s) if s.target as usize == post && s.slow_rate == 0.0 => {}, _ => continue }
                let (last, w, count, born) = { let s = &self.neurons[pre].synapses[k]; (s.last_co.max(s.born), s.weight, s.count, s.born) };
                // 경쟁: 받는 뉴런(지금 발화한 이 뉴런)의 감응도가 낮을수록 수명이 짧다
                let share = if self.params.compete > 0.0 { self.neurons[post].excitability.min(1.0).powf(self.params.compete) } else { 1.0 };
                // 최소 수명(2만 틱)을 적용한 뒤에 곱한다. 곱한 값에는 바닥이 없다
                let full = self.prune_life(w * share);
                let life = if shrink < 1.0 { (full as f64 * shrink) as u64 } else { full };
                if count == 0 || last + life > now { continue; }
                self.pruned += 1;
                if last + full > now { self.pruned_short += 1; }
                self.prune_ages.push(now.saturating_sub(born));
                if count > 1 {
                    let s = &mut self.neurons[pre].synapses[k];
                    s.weight -= s.weight / count as f64;
                    s.count -= 1;
                    s.last_co = now;
                } else {
                    self.remove_synapse(pre, k);
                }
            }
        }
    }

    /// 연결 하나를 없앤다. 마지막 시냅스를 빈자리로 옮기고, 번호를 기억하는 곳(받는 쪽의 역참조, 흔적 목록, 지연 시냅스 목록)을 고친다
    fn remove_synapse(&mut self, pre: usize, k: usize) {
        let last = self.neurons[pre].synapses.len() - 1;
        let target = self.neurons[pre].synapses[k].target as usize;
        if let Some(n) = self.neurons.get_mut(target) {
            if let Some(i) = n.incoming.iter().position(|&e| e == (pre, k)) { n.incoming.swap_remove(i); }
            n.elig_in.retain(|&e| e != (pre as u32, k as u32));
        }
        if k != last {
            let moved_target = self.neurons[pre].synapses[last].target as usize;
            let moved_slow = self.neurons[pre].synapses[last].slow_rate > 0.0;
            if let Some(n) = self.neurons.get_mut(moved_target) {
                if let Some(e) = n.incoming.iter_mut().find(|e| **e == (pre, last)) { *e = (pre, k); }
                for e in n.elig_in.iter_mut() { if *e == (pre as u32, last as u32) { *e = (pre as u32, k as u32); } }
            }
            if moved_slow {
                if let Some(e) = self.slow_syn.iter_mut().find(|e| **e == (pre, last)) { *e = (pre, k); }
                // slow_events: 받는 뉴런의 살아 있는 입력 목록도 새 번호로 고친다
                if self.slow_listed.remove(&(((pre as u64) << 32) | last as u64)) {
                    self.slow_listed.insert(((pre as u64) << 32) | k as u64);
                    if let Some(&p) = self.slow_pos.get(moved_target) { if p != u32::MAX {
                        if let Some(e) = self.slow_live[p as usize].iter_mut().find(|e| **e == (pre as u32, last as u32)) { *e = (pre as u32, k as u32); }
                    } }
                }
            }
        }
        self.neurons[pre].synapses.swap_remove(k);
        let n = &mut self.neurons[pre];
        n.elig_idx.retain(|&i| i as usize != k);
        for i in n.elig_idx.iter_mut() { if *i as usize == last { *i = k as u32; } }
    }

    /// 은닉 뉴런의 감응도 (평균, 최소, 최대) — 항상성이 얼마나 움직였는지 (관찰용)
    pub fn excitability_stats(&self) -> (f64, f64, f64) {
        let h = &self.neurons[..self.hippocampus_range.1];
        let (mut sum, mut lo, mut hi) = (0.0, f64::MAX, 0.0f64);
        for n in h { sum += n.excitability; lo = lo.min(n.excitability); hi = hi.max(n.excitability); }
        (sum / h.len() as f64, lo, hi)
    }

    /// 은닉 뉴런이 내보내는 시냅스 접점 수의 (평균, 최소, 최대), 그리고 연결(상대) 수의 평균
    pub fn out_degree_stats(&self) -> (f64, usize, usize, f64) {
        let h = &self.neurons[..self.hippocampus_range.1];
        let (mut sum, mut lo, mut hi, mut conns) = (0usize, usize::MAX, 0usize, 0usize);
        for n in h {
            let k: usize = n.synapses.iter().map(|s| s.count as usize).sum();
            sum += k; lo = lo.min(k); hi = hi.max(k); conns += n.synapses.len();
        }
        (sum as f64 / h.len() as f64, lo, hi, conns as f64 / h.len() as f64)
    }

    /// 1만 틱마다: 항상성 + (검증된 구성이면) 뉴런별 지역 값 갱신.
    /// 항상성은 은닉·입출력뿐 아니라 보조 뉴런(도파민·가치·예측·보상 입력, 맛·필요·중계·포만)에도 걸린다. 뉴런별 목표(homeostasis_own)는
    /// 첫 1만 틱의 발화율을 평소로 삼으므로, 그때 조용했던 뉴런은 평소가 0 으로 잡히고 그 뒤 발화하면 매번 감응도가 최대 비율(2%)로 깎인다
    /// ("원래 조용한 뉴런은 건드리지 않는다"는 계속 조용할 때만 맞다). no_homeo_need / no_homeo_da 는 그 무리를 항상성에서 뺀다
    fn periodic_homeostasis(&mut self) {
        // 보조 뉴런의 두 무리: [도파민 시작, 몸의 상태 시작) = 도파민·가치·예측·보상·처벌, [몸의 상태 시작, aux_end) = 맛·필요·중계·포만
        let da_start = self.da_idx.first().copied().unwrap_or(usize::MAX);
        let need_start = self.taste_idx.iter().flatten().copied().min().unwrap_or(self.aux_end);
        let (skip_need, skip_da, aux_end) = (self.params.no_homeo_need, self.params.no_homeo_da, self.aux_end);
        let validated = self.params.validated;
        let (skip_hidden, hidden_end) = (self.params.no_homeo_hidden, self.hippocampus_range.1);
        for (i, n) in self.neurons.iter_mut().enumerate() {
            let skip = (skip_need && i >= need_start && i < aux_end) || (skip_da && i >= da_start && i < need_start.min(aux_end)) || (skip_hidden && i < hidden_end);
            if skip { n.fire_count_window = 0; continue; }
            if validated { n.homeostasis_own(10_000.0, 0.02, 0.1); } else { n.homeostasis(); }
        }
        // 차례: 감응도 → 입력 세기 조절 → 지역 값(경계, 잡음 크기, 새 접점 세기 기준). 곱한 가중치로 지역 값이 잡히게 한다
        self.scale_inputs();
        if validated { self.refresh_local(); }
    }

    /// scale_in: 은닉의 추체마다 느린 발화율이 받기 문의 목표에서 벗어난 만큼, 그 뉴런으로 들어오는 흥분 시냅스의 굳은 가중치를 같은 비율로 조금 고친다.
    ///   오차 = clamp(느린 발화율 ÷ 목표 − 1, −1, +1), 배수 = 1 − η × 오차. |오차| < 0.1 이면 하지 않는다. 목표가 0 인 뉴런은 건너뛴다.
    ///   대상: 보내는 쪽이 은닉의 흥분 뉴런이거나 입력층(scale_in_hidden_only 면 은닉만)이고 지연 시냅스가 아닌 것. 억제·조절(도파민) 시냅스는 건드리지 않는다.
    ///   한계: 줄일 때는 최소 가중치의 2배 아래로 내리지 않고(이미 그 아래면 그대로), 키울 때는 연결 한계(CONN_CAP) 위로 올리지 않는다(이미 그 위면 그대로).
    ///   난수를 쓰지 않고 새 상태도 없다(가중치만 바뀐다). 곱하기 전에 그 시냅스를 지금 틱까지 따라잡는다
    fn scale_inputs(&mut self) {
        let Some(eta) = self.params.scale_in else { return };
        if eta == 0.0 { return; }
        let h = self.hippocampus_range.1.min(self.neurons.len());
        let now = self.global_tick;
        if self.scale_cum.len() != h { self.scale_cum = vec![1.0; h]; }
        let mut mult = vec![1.0f64; h];
        let mut any = false;
        for i in 0..h {
            if self.neurons[i].cell_type != CellType::Pyramidal { continue; }
            let goal = self.act_target_of(i);
            if goal <= 0.0 { continue; }
            let err = (self.neurons[i].rate_avg / goal - 1.0).clamp(-1.0, 1.0);
            if err.abs() < 0.1 { continue; }
            mult[i] = 1.0 - eta * err;
            self.scale_cum[i] *= mult[i];
            if err > 0.0 { self.scale_obs[0] += 1; } else { self.scale_obs[1] += 1; }
            any = true;
        }
        if !any { return; }
        let mut is_input = vec![false; self.neurons.len()];
        if !self.params.scale_in_hidden_only { for &i in &self.input_idx { if i < is_input.len() { is_input[i] = true; } } }
        let (lo, hi) = (2.0 * self.params.min_weight, CONN_CAP);
        let (mut at_lo, mut at_hi, mut done) = (0u64, 0u64, 0u64);
        for pre in 0..self.neurons.len() {
            let n = &mut self.neurons[pre];
            let from_hidden = pre < h && !n.inhibitory && !n.modulatory;
            if !from_hidden && !is_input[pre] { continue; }
            for s in n.synapses.iter_mut() {
                let t = s.target as usize;
                if t >= h || mult[t] == 1.0 || s.slow_rate != 0.0 { continue; }
                s.sync(now);
                let m = mult[t];
                let w = s.weight;
                let nw = if m < 1.0 { (w * m).max(lo.min(w)) } else { (w * m).min(hi.max(w)) };
                if m < 1.0 && nw > w * m { at_lo += 1; }
                if m > 1.0 && nw < w * m { at_hi += 1; }
                s.weight = nw;
                done += 1;
            }
        }
        self.scale_obs[2] += at_lo; self.scale_obs[3] += at_hi; self.scale_obs[4] += done;
    }

    /// scale_in 의 관찰값을 읽고 0 으로 되돌린다: (줄인 뉴런·번, 키운 뉴런·번, 아래 한계에 걸린 시냅스·번, 위 한계에 걸린 시냅스·번, 곱한 시냅스·번, 뉴런별 누적 배수의 10·50·90% 값 — 배수가 1 이 아닌 뉴런만)
    pub fn scale_obs_take(&mut self) -> ([u64; 5], [f64; 3], usize) {
        let o = self.scale_obs;
        self.scale_obs = [0; 5];
        let mut v: Vec<f64> = self.scale_cum.iter().cloned().filter(|&x| x != 1.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| if v.is_empty() { 1.0 } else { v[((v.len() - 1) as f64 * p).round() as usize] };
        let r = [q(0.1), q(0.5), q(0.9)];
        let n = v.len();
        for x in self.scale_cum.iter_mut() { *x = 1.0; }
        (o, r, n)
    }

    /// 시냅스 하나를 지금 틱까지 따라잡은 사본 (실제 상태는 건드리지 않는다). 상태 비교용
    fn synapse_view(&self, pre: usize, k: usize) -> crate::synapse::Synapse {
        let mut c = self.neurons[pre].synapses[k].clone();
        c.sync(self.global_tick);
        c
    }

    /// 망 상태의 합계 값 (관찰용, 상태는 건드리지 않는다): 모든 시냅스를 지금 틱까지 따라잡은 사본의
    /// (대상, 가중치, 흔적, 일시 성분, 접점 수, 가지치기 시계)를 차례로 섞은 값과, 가중치 합·흔적 절댓값 합·시냅스 수.
    /// 같은 씨앗의 두 실행이 어느 틱에서 처음 달라지는지 찾는 데 쓴다
    pub fn state_hash(&self) -> (u64, f64, f64, usize) {
        let mut h: u64 = 0xcbf29ce484222325;
        let (mut w, mut e, mut n) = (0.0f64, 0.0f64, 0usize);
        for pre in 0..self.neurons.len() {
            for k in 0..self.neurons[pre].synapses.len() {
                let s = self.synapse_view(pre, k);
                for v in [s.target as u64, s.weight.to_bits(), s.elig.to_bits(), s.w_short.to_bits(), s.count as u64, s.last_co] { h ^= v; h = h.wrapping_mul(0x100000001b3); }
                w += s.weight; e += s.elig.abs(); n += 1;
            }
        }
        (h, w, e, n)
    }

    /// 망 상태를 글로 적는다 (상태 비교용): 시냅스마다 "보내는 뉴런, 번호, 대상, 가중치, 흔적, 일시 성분, 접점 수, 가지치기 시계, 생긴 틱" 한 줄. 지금 틱까지 따라잡은 사본으로
    pub fn state_dump(&self, path: &Path) {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("상태 파일을 만들지 못했다"));
        writeln!(f, "# 전역 틱 {}", self.global_tick).ok();
        for pre in 0..self.neurons.len() {
            for k in 0..self.neurons[pre].synapses.len() {
                let s = self.synapse_view(pre, k);
                writeln!(f, "{} {} {} {:e} {:e} {:e} {} {} {}", pre, k, s.target, s.weight, s.elig, s.w_short, s.count, s.last_co, s.born).ok();
            }
        }
    }

    /// 보조 뉴런 무리별 평균 감응도 (관찰용): 맛, 필요, 중계, 포만, 도파민, 예측, 가치. 무리가 없으면 NaN
    pub fn aux_excitability(&self) -> [(&'static str, f64); 7] {
        let mean = |v: &[usize]| if v.is_empty() { f64::NAN } else { v.iter().map(|&i| self.neurons[i].excitability).sum::<f64>() / v.len() as f64 };
        let taste: Vec<usize> = self.taste_idx.iter().flatten().copied().collect();
        let need: Vec<usize> = self.need_idx.iter().flatten().copied().collect();
        // 종류마다 맛 → 필요 → 중계 → 포만이 NEED_GROUP 개씩 잇달아 만들어진다
        let (mut relay, mut sat) = (Vec::new(), Vec::new());
        for t in &self.taste_idx { if let Some(&first) = t.first() { relay.extend(first + 2 * NEED_GROUP..first + 3 * NEED_GROUP); sat.extend(first + 3 * NEED_GROUP..first + 4 * NEED_GROUP); } }
        [("맛", mean(&taste)), ("필요", mean(&need)), ("중계", mean(&relay)), ("포만", mean(&sat)), ("도파민", mean(&self.da_idx)), ("예측", mean(&self.pred_idx)), ("가치", mean(&self.val_idx))]
    }

    /// 뉴런별 지역 값(학습 변화량·soft bound·노이즈) 다시 계산
    fn refresh_local(&mut self) {
        plasticity::refresh_local_scales(&mut self.neurons, 0.95);
        self.noise_amp = self.neurons.iter().map(|n| n.w_ref_med * 0.1).collect();
        if self.params.grow.is_some() {
            let neurons = &self.neurons;
            // unit_med_fix: 조절 뉴런이 "보내는" 입력을 뺀다. 기존 조건은 받는 뉴런(자기 자신)이 조절 뉴런인지를 봐서 은닉 뉴런에 대해서는 아무것도 거르지 않는다
            let fix = self.params.unit_med_fix;
            self.unit_med = neurons.par_iter().map(|n| {
                let mut v: Vec<f64> = n.incoming.iter()
                    .filter_map(|&(pre, k)| neurons.get(pre).and_then(|p| p.synapses.get(k)).map(|s| (pre, s)))
                    .filter(|&(pre, s)| s.slow_rate == 0.0 && !(if fix { neurons[pre].modulatory } else { neurons[s.target as usize].modulatory }))
                    .map(|(_, s)| s.weight / s.count.max(1) as f64).collect();
                if v.is_empty() { return 0.0; }
                let mid = v.len() / 2;
                *v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
            }).collect();
        }
        if self.params.spont.is_some() || self.params.spont_fixed.is_some() { self.apply_spont(); }
    }

    /// 자발 발화 확률 설정 (은닉 뉴런).
    ///   검증 구성: 배수 × (자기 입력 가중치 중앙값 ÷ 0.2). 입력 하나하나가 약한 뉴런(입력이 많은 큰 회로의 뉴런)일수록
    ///   스스로 발화할 확률이 낮다 — 뉴런은 네트워크 크기를 모르지만 자기 입력 가중치는 안다. 노이즈 크기와 같은 기준이다
    ///   기존 구성: 배수 그대로
    ///   세포 종류 구분(typed)이 켜져 있으면 종류별 배수가 더 곱해진다: 추체 1, Pv 10, Sst 5, Vip 3
    ///   spont_fixed(Some(P)): 위 식 대신 확률 P 를 그대로 쓴다 (입력 가중치를 보지 않는다). 종류별 배수는 곱해진다
    ///   spont_flat: 종류별 배수를 곱하지 않는다
    fn apply_spont(&mut self) {
        // spont_fixed 가 있으면 그 확률을 그대로 쓴다 (확률 = 0.00001 × spont_scale 이므로 배수로 바꿔 넣는다. 입력 가중치는 보지 않는다)
        let (x, by_weight) = match (self.params.spont_fixed, self.params.spont) {
            (Some(p), _) => (p / 1e-5, false),
            (None, Some(x)) => (x, self.params.validated),
            (None, None) => return,
        };
        let flat = self.params.spont_flat;
        let hidden_end = self.hippocampus_range.1;
        for n in self.neurons[..hidden_end].iter_mut() {
            // 세포 종류별 배수 (깨어 있는 생쥐 피질의 평소 발화율 비: 추체세포는 드물게, 억제성은 자주. 대략적인 비율이다)
            let by_type = if flat { 1.0 } else { match n.cell_type {
                CellType::Pyramidal => 1.0,
                CellType::Pv => 10.0,
                CellType::Sst => 5.0,
                CellType::Vip => 3.0,
                _ => 1.0,
            } };
            // 발화 확률(0.00001 × spont_scale)이 1을 넘지 않게 자른다 (넘으면 random_bool 이 실행을 죽인다)
            n.spont_scale = (by_type * if by_weight { x * n.w_ref_med / 0.2 } else { x }).min(1e5);
        }
    }

    /// 은닉 뉴런 전체에서 한 틱에 기대되는 자발 발화 수 (관찰용), 그리고 입력 가중치 중앙값의 평균
    pub fn spont_expected(&self) -> (f64, f64) {
        let h = &self.neurons[..self.hippocampus_range.1];
        (h.iter().map(|n| 1e-5 * n.spont_scale).sum(), h.iter().map(|n| n.w_ref_med).sum::<f64>() / h.len() as f64)
    }

    /// 은닉 뉴런마다 (종류: 0 추체, 1 Pv, 2 Sst, 3 Vip, 4 그 밖), 느린 발화율 rate_avg(틱당 발화 수), 입력 가중치 중앙값 w_ref_med,
    /// 틱당 자발 발화 확률(try_fire 가 평소에 쓰는 값: 0.00001 × spont_scale). 읽기만 한다 (관찰용)
    pub fn hidden_observe(&self) -> Vec<(usize, f64, f64, f64)> {
        self.neurons[..self.hippocampus_range.1].iter().map(|n| {
            let k = match n.cell_type { CellType::Pyramidal => 0, CellType::Pv => 1, CellType::Sst => 2, CellType::Vip => 3, _ => 4 };
            (k, n.rate_avg, n.w_ref_med, 1e-5 * n.spont_scale)
        }).collect()
    }

    /// 새 접점 세기 기준(unit_med)의 은닉 뉴런 평균 (관찰용. 발달 과정이 꺼져 있으면 0)
    pub fn unit_med_mean(&self) -> f64 {
        let h = self.hippocampus_range.1.min(self.unit_med.len());
        if h == 0 { 0.0 } else { self.unit_med[..h].iter().sum::<f64>() / h as f64 }
    }

    /// 검증된 학습 구성 적용
    fn validated_plasticity() -> PlasticityOptions {
        PlasticityOptions {
            stdp_on_inhibitory: false,
            local_scale: true,
            ltd_once_per_post: true,
            bcm_classic_local: Some(1.0),
            ..PlasticityOptions::default()
        }
    }
    fn apply_validated(&mut self) {
        self.plasticity = Self::validated_plasticity();
        for n in &mut self.neurons { n.spont_scale = 0.0; }
        self.refresh_local();
    }

    /// 마지막 fire 에서 발화한 뉴런 집합 (입출력 뉴런 제외)
    pub fn last_active_hidden(&self) -> Vec<NeuronId> {
        let io: std::collections::HashSet<usize> = self.input_idx.iter().chain(self.output_idx.iter()).cloned().collect();
        let mut v: Vec<NeuronId> = match self.fire_records.last() {
            Some(r) => r.spiked_neurons.iter().map(|s| s.0).filter(|&n| !io.contains(&(n as usize))).collect(),
            None => Vec::new(),
        };
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 모든 시냅스 가중치 사본 (뉴런 순서 → 시냅스 순서)
    pub fn weights_snapshot(&self) -> Vec<Vec<f64>> {
        self.neurons.iter().map(|n| n.synapses.iter().map(|s| s.weight).collect()).collect()
    }

    /// pre 와 post 가 모두 set 에 속한 시냅스(도파민·보상 배선 제외)의 가중치 변화: (평균 변화율, 시냅스 수)
    pub fn weight_change_within(&self, set: &[NeuronId], before: &[Vec<f64>]) -> (f64, usize) {
        let inset: std::collections::HashSet<usize> = set.iter().map(|&x| x as usize).collect();
        let (mut b, mut a, mut n) = (0.0, 0.0, 0usize);
        for &pre in &inset {
            if self.is_aux(pre) { continue; }
            for (k, s) in self.neurons[pre].synapses.iter().enumerate() {
                if !inset.contains(&(s.target as usize)) { continue; }
                if let Some(&w0) = before.get(pre).and_then(|v| v.get(k)) { b += w0; a += s.weight; n += 1; }
            }
        }
        (if b > 0.0 { a / b - 1.0 } else { 0.0 }, n)
    }

    /// 마지막 fire 에서 발화한 뉴런 (은닉 + 출력, 입력 제외)
    pub fn last_active(&self) -> Vec<NeuronId> {
        let inp: std::collections::HashSet<usize> = self.input_idx.iter().cloned().collect();
        let mut v: Vec<NeuronId> = match self.fire_records.last() {
            Some(r) => r.spiked_neurons.iter().map(|s| s.0).filter(|&n| !inp.contains(&(n as usize))).collect(),
            None => Vec::new(),
        };
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 마지막 fire 의 (총 발화 수, 출력 발화 수, 발화한 서로 다른 뉴런 수)
    pub fn last_stats(&self) -> (usize, usize, usize) {
        match self.fire_records.last() {
            Some(r) => {
                let distinct: std::collections::HashSet<NeuronId> = r.spiked_neurons.iter().map(|s| s.0).collect();
                (r.spiked_neurons.len(), r.output_tokens.len(), distinct.len())
            }
            None => (0, 0, 0),
        }
    }

    /// 비지도 STDP 크기 배수 설정 (0 = 끔)
    pub fn set_stdp_scale(&mut self, scale: f64) {
        self.plasticity.stdp_scale = scale;
    }

    /// teach: 정답 출력 뉴런 사전 활성화 → fire
    /// 보상(positive) 또는 처벌을 준다: 보상(처벌) 입력 뉴런을 발화시킨다. 무엇이 얼마나 변할지는
    /// 그때 시냅스에 남아 있는 흔적과 각 뉴런이 받는 조절 신호가 정한다. 도파민 방식(--dopamine)에서만 동작한다
    /// (숫자를 곱하던 예전 방식의 보상은 없앴다)
    pub fn feedback(&mut self, _fire_id: u64, positive: bool, strength: f64) {
        if !self.params.dopamine {
            eprintln!("  [feedback] 도파민 방식이 꺼져 있어 보상을 줄 수 없습니다 (--dopamine 으로 실행)");
            return;
        }
        self.deliver_reward(positive, strength);
    }

    /// 정답 문자열이 주어진 경우: 그 fire 의 출력이 정답과 같으면 보상, 다르면 처벌
    pub fn feedback_by_target(&mut self, fire_id: u64, target: &str, strength: f64) {
        let correct = self.fire_records.iter().find(|r| r.id == fire_id).is_some_and(|r| r.output == target);
        self.feedback(fire_id, correct, strength);
    }

    pub fn teach(&mut self, input: &str, target: &str) -> u64 {
        let target_jamo = tokenizer::decompose_to_jamo(target);
        let target_seqs: Vec<u32> = target_jamo.iter().map(|j| self.get_seq(j)).collect();
        for &seq in &target_seqs {
            let idx = self.output_neuron(seq);
            self.neurons[idx].potential += 0.6;
        }
        self.fire(input)
    }

    /// 각 뉴런의 incoming 역참조 재구축
    fn rebuild_incoming(&mut self) {
        plasticity::rebuild_incoming(&mut self.neurons);
    }

    // ── 상태 조회 ──

    pub fn synapse_count(&self) -> usize {
        self.neurons.iter().map(|n| n.synapses.len()).sum()
    }
    pub fn fire_count(&self) -> u64 { self.next_fire_id - 1 }
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    pub fn get_last_output(&self) -> String {
        self.fire_records.last().map(|r| r.output.clone()).unwrap_or_default()
    }
    pub fn get_last_record(&self) -> Option<&FireRecord> {
        self.fire_records.last()
    }

    pub fn nid_label(&self, nid: NeuronId) -> String {
        let idx = nid as usize;
        let region = self.neuron_region(idx);
        let name = match region {
            RegionType::Input => "입력",
            RegionType::Emotion => "감정",
            RegionType::Reason => "이성",
            RegionType::Memory => "기억",
            RegionType::Hippocampus => "해마",
            RegionType::Output => "출력",
        };
        format!("{name} #{nid}")
    }

    pub fn get_status(&self) -> serde_json::Value {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;
        let emotion_s: usize = (es..ee).map(|i| self.neurons[i].synapses.len()).sum();
        let reason_s: usize = (rs..re).map(|i| self.neurons[i].synapses.len()).sum();
        let memory_s: usize = (ms..me).map(|i| self.neurons[i].synapses.len()).sum();
        let hippo_s: usize = (hs..he).map(|i| self.neurons[i].synapses.len()).sum();
        let input_s: usize = self.input_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();
        let output_s: usize = self.output_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();

        // weight 분포 계산
        let mut w_bins = [0usize; 10]; // 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        let mut w_sum = 0.0f64;
        let mut w_count = 0usize;
        for n in &self.neurons {
            for s in &n.synapses {
                let bin = ((s.weight * 10.0) as usize).min(9);
                w_bins[bin] += 1;
                w_sum += s.weight;
                w_count += 1;
            }
        }
        let w_avg = if w_count > 0 { w_sum / w_count as f64 } else { 0.0 };

        serde_json::json!({
            "neurons": self.neurons.len(),
            "synapses": self.synapse_count(),
            "fire_count": self.fire_count(),
            "vocab_size": self.vocab_size(),
            "threshold": format!("{:.3}", self.threshold),
            "regions": {
                "입력": { "neurons": self.input_idx.len(), "synapses": input_s },
                "감정": { "neurons": self.emotion_range.1 - self.emotion_range.0, "synapses": emotion_s },
                "이성": { "neurons": self.reason_range.1 - self.reason_range.0, "synapses": reason_s },
                "기억": { "neurons": self.memory_range.1 - self.memory_range.0, "synapses": memory_s },
                "해마": { "neurons": self.hippocampus_range.1 - self.hippocampus_range.0, "synapses": hippo_s },
                "출력": { "neurons": self.output_idx.len(), "synapses": output_s },
            },
            "weight_dist": {
                "avg": format!("{:.4}", w_avg),
                "0.0-0.1": w_bins[0],
                "0.1-0.2": w_bins[1],
                "0.2-0.3": w_bins[2],
                "0.3-0.4": w_bins[3],
                "0.4-0.5": w_bins[4],
                "0.5-0.6": w_bins[5],
                "0.6-0.7": w_bins[6],
                "0.7-0.8": w_bins[7],
                "0.8-0.9": w_bins[8],
                "0.9-1.0": w_bins[9],
            },
        })
    }

    pub fn print_summary(&self) {
        println!("  어휘: {}개", self.vocab_size());
        println!("  뉴런: {}개, 시냅스: {}개", self.neurons.len(), self.synapse_count());
    }

    // ── 서버용 헬퍼 ──

    pub fn get_seq_pub(&mut self, token: &str) -> u32 { self.get_seq(token) }
    pub fn output_neuron_pub(&self, seq: u32) -> usize { self.output_neuron(seq) }
    pub fn neurons_mut(&mut self) -> &mut Vec<Neuron> { &mut self.neurons }
    pub fn has_active_stimuli(&self) -> bool { !self.active_stimuli.is_empty() }

    // ── 저장/불러오기 ──

    pub fn save(&self, path: &Path) {
        let snap = Snapshot {
            neurons: self.neurons.clone(),
            vocab: self.vocab.clone(),
            reverse_vocab: self.reverse_vocab.clone(),
            input_idx: self.input_idx.clone(),
            output_idx: self.output_idx.clone(),
            emotion_range: self.emotion_range,
            emotion_main_range: self.emotion_main_range,
            emotion_map_range: self.emotion_map_range,
            reason_range: self.reason_range,
            reason_main_range: self.reason_main_range,
            reason_map_range: self.reason_map_range,
            memory_range: self.memory_range,
            hippocampus_range: self.hippocampus_range,
            next_fire_id: self.next_fire_id,
            threshold: self.threshold,
            noise_range: self.noise_range,
            global_tick: self.global_tick,
            last_sleep_tick: Some(self.last_sleep_tick),
            fires_since_sleep: self.fires_since_sleep,
            min_weight: self.params.min_weight,
            validated: self.params.validated,
            dopamine: self.params.dopamine,
            gate_base: self.params.gate_base,
            da_gain: self.params.da_gain,
            predict: self.params.predict,
            pred_eta: self.params.pred_eta,
            pred_idx: self.pred_idx.clone(),
            da_idx: self.da_idx.clone(),
            reward_idx: self.reward_idx.clone(),
            punish_idx: self.punish_idx.clone(),
            ext: Some(SnapExt {
                params: self.params.clone(),
                val_idx: self.val_idx.clone(), taste_idx: self.taste_idx.clone(), need_idx: self.need_idx.clone(), aux_end: self.aux_end,
                unit_med: self.unit_med.clone(), noise_amp: self.noise_amp.clone(),
                sprouted: self.sprouted, pruned: self.pruned, da_spikes: self.da_spikes, pred_spikes: self.pred_spikes, val_spikes: self.val_spikes,
                grow_made: self.grow_made, gate_drop: self.gate_drop, own_wait_drop: self.own_wait_drop, added_on_initial: self.added_on_initial, pruned_short: self.pruned_short,
                wiring_sum: self.wiring_sum, dev_tick: self.dev_tick,
                drives: self.drives.clone(), dev_state: self.dev_state.clone(),
                slow_sum: if self.params.slow_events { self.slow_sum.clone() } else { Vec::new() },
                slow_live: if self.params.slow_events { self.slow_live.clone() } else { Vec::new() },
            }),
        };
        let data = serde_json::to_vec(&snap).expect("직렬화 실패");
        // 쓰다가 끊겨도 앞의 파일이 깨지지 않게 임시 파일에 쓰고 이름을 바꾼다
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &data).expect("저장 실패");
        std::fs::rename(&tmp, path).expect("저장 실패");
        eprintln!("  [저장] {} ({:.1}MB, 뉴런 {}개, 시냅스 {}개)",
            path.display(), data.len() as f64 / 1048576.0,
            self.neurons.len(), self.synapse_count());
    }

    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let snap: Snapshot = serde_json::from_slice(&data).ok()?;
        let synapse_count: usize = snap.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [불러오기] {} (뉴런 {}개, 시냅스 {}개, 어휘 {}개)",
            path.display(), snap.neurons.len(), synapse_count, snap.vocab.len());
        // 틱 복원: 뉴런의 last_spike_tick은 저장 당시의 틱 기준이다.
        // global_tick을 0으로 되돌리면 경과 시간이 0으로 계산되어
        // 한 번이라도 발화한 뉴런이 절대 불응기에 갇힌다.
        // 구버전 스냅샷(global_tick 없음)은 가장 최근 발화 틱 이후로 맞춘다.
        let max_spike_tick = snap.neurons.iter()
            .filter_map(|n| n.last_spike_tick).max().unwrap_or(0);
        // 저장 파일에 전역 틱이 있으면 그대로 쓴다 (마지막 틱에 발화한 뉴런이 있어도 틱을 밀지 않는다: 밀면 불러온 망의 시계가 한 틱 어긋난다)
        let global_tick = if snap.global_tick > 0 { snap.global_tick } else { max_spike_tick + 1 };
        let last_sleep_tick = snap.last_sleep_tick.unwrap_or(global_tick).min(global_tick);
        let mut net = Network {
            neurons: snap.neurons,
            vocab: snap.vocab,
            reverse_vocab: snap.reverse_vocab,
            input_idx: snap.input_idx,
            output_idx: snap.output_idx,
            emotion_range: snap.emotion_range,
            emotion_main_range: snap.emotion_main_range,
            emotion_map_range: snap.emotion_map_range,
            reason_range: snap.reason_range,
            reason_main_range: snap.reason_main_range,
            reason_map_range: snap.reason_map_range,
            memory_range: snap.memory_range,
            hippocampus_range: snap.hippocampus_range,
            fire_records: Vec::new(),
            next_fire_id: snap.next_fire_id,
            debug: false,
            active_stimuli: Vec::new(),
            recent_spikes: Vec::new(),
            threshold: snap.threshold,
            noise_range: snap.noise_range,
            global_tick,
            fires_since_sleep: snap.fires_since_sleep,
            last_sleep_tick,
            plasticity: PlasticityOptions::default(),
            params: NetParams { min_weight: snap.min_weight, validated: snap.validated, dopamine: snap.dopamine,
                gate_base: snap.gate_base, da_gain: snap.da_gain, predict: snap.predict, pred_eta: snap.pred_eta, ..NetParams::default() },
            noise_amp: Vec::new(),
            da_idx: snap.da_idx,
            reward_idx: snap.reward_idx,
            punish_idx: snap.punish_idx,
            pred_idx: snap.pred_idx,
            val_idx: Vec::new(),
            val_spikes: 0,
            taste_idx: Vec::new(),
            need_idx: Vec::new(),
            aux_end: 0,
            da_spikes: 0,
            pred_spikes: 0,
            drives: Vec::new(),
            slow_syn: Vec::new(),
            sprouted: 0,
            pruned: 0,
            unit_med: Vec::new(),
            tick_ids: Vec::new(),
            wiring_sum: 0,
            grow_made: [0; 5],
            gate_drop: [0; 5],
            own_wait_drop: 0,
            added_on_initial: 0,
            pruned_short: 0,
            prune_ages: Vec::new(),
            prune_age_unknown: 0,
            slow_recv: Vec::new(),
            slow_pos: Vec::new(),
            slow_sum: Vec::new(),
            slow_live: Vec::new(),
            slow_listed: std::collections::HashSet::new(),
            slow_saved: None,
            slow_shadow: Vec::new(),
            slow_init: Vec::new(),
            slow_chk_recv: 0.0,
            slow_chk_drift: 0.0,
            wake: None,
            scale_obs: [0; 5],
            scale_cum: Vec::new(),
            dev_tick: 0,
            dev_state: None,
        };
        net.rebuild_incoming();
        net.rebuild_slow_index();
        // 시냅스의 감쇠 기준 틱을 지금으로 맞춘다 (구버전 스냅샷은 값이 0이라 그대로 두면 한꺼번에 사라진다)
        for n in net.neurons.iter_mut() { for s in n.synapses.iter_mut() { if s.t == 0 { s.t = global_tick; } } }
        match snap.ext {
            Some(x) => {
                // 설정 전체와, 뉴런에 들어 있지 않은 상태를 되돌린다. 자발 발화 확률·기준선·입력 가중치 중앙값은 뉴런에 저장돼 있어 다시 계산하지 않는다
                net.params = x.params;
                net.val_idx = x.val_idx; net.taste_idx = x.taste_idx; net.need_idx = x.need_idx; net.aux_end = x.aux_end;
                net.unit_med = x.unit_med; net.noise_amp = x.noise_amp;
                net.sprouted = x.sprouted; net.pruned = x.pruned; net.da_spikes = x.da_spikes; net.pred_spikes = x.pred_spikes; net.val_spikes = x.val_spikes;
                net.grow_made = x.grow_made; net.gate_drop = x.gate_drop; net.own_wait_drop = x.own_wait_drop; net.added_on_initial = x.added_on_initial; net.pruned_short = x.pruned_short;
                net.wiring_sum = x.wiring_sum; net.dev_tick = x.dev_tick;
                net.drives = x.drives; net.dev_state = x.dev_state;
                if !x.slow_sum.is_empty() && x.slow_sum.len() == x.slow_live.len() { net.slow_saved = Some((x.slow_sum, x.slow_live)); }
                if net.params.validated { net.plasticity = Self::validated_plasticity(); }
                if net.params.dopamine { net.apply_dopamine_options(); }
                // 흔적이 남아 있는 시냅스의 목록(저장하지 않는다)을 다시 만든다
                let mut ins: Vec<(usize, u32, u32)> = Vec::new();
                for (pre, n) in net.neurons.iter_mut().enumerate() {
                    n.elig_idx.clear(); n.elig_in.clear();
                    for (k, s) in n.synapses.iter_mut().enumerate() {
                        if s.elig != 0.0 { s.lp = true; s.lq = true; n.elig_idx.push(k as u32); ins.push((s.target as usize, pre as u32, k as u32)); }
                    }
                }
                for (t, pre, k) in ins { if let Some(n) = net.neurons.get_mut(t) { n.elig_in.push((pre, k)); } }
            }
            None => {
                // 구버전 저장 파일: 기존 동작 그대로
                if net.params.validated { net.apply_validated(); }
                if net.params.dopamine { net.apply_dopamine_options(); }
            }
        }
        if net.params.slow_events && net.slow_saved.is_none() { net.slow_resync(); net.slow_chk_drift = 0.0; }
        Some(net)
    }
}

fn default_emotion_main_range() -> (usize, usize) { (0, 4000) }
fn default_emotion_map_range() -> (usize, usize) { (4000, 5000) }
fn default_reason_main_range() -> (usize, usize) { (5000, 9000) }
fn default_reason_map_range() -> (usize, usize) { (9000, 10000) }
fn default_memory_range() -> (usize, usize) { (10000, 12000) }
fn default_hippo_range() -> (usize, usize) { (12000, 12500) }

#[derive(Serialize, Deserialize)]
struct Snapshot {
    neurons: Vec<Neuron>,
    vocab: HashMap<String, u32>,
    reverse_vocab: Vec<String>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    emotion_range: (usize, usize),
    #[serde(default = "default_emotion_main_range")]
    emotion_main_range: (usize, usize),
    #[serde(default = "default_emotion_map_range")]
    emotion_map_range: (usize, usize),
    reason_range: (usize, usize),
    #[serde(default = "default_reason_main_range")]
    reason_main_range: (usize, usize),
    #[serde(default = "default_reason_map_range")]
    reason_map_range: (usize, usize),
    #[serde(default = "default_memory_range")]
    memory_range: (usize, usize),
    #[serde(default = "default_hippo_range")]
    hippocampus_range: (usize, usize),
    next_fire_id: u64,
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default = "default_noise")]
    noise_range: f64,
    #[serde(default)]
    global_tick: u64,
    #[serde(default)]
    last_sleep_tick: Option<u64>,
    #[serde(default)]
    fires_since_sleep: u64,
    #[serde(default = "default_min_weight")]
    min_weight: f64,
    #[serde(default)]
    validated: bool,
    #[serde(default)]
    dopamine: bool,
    #[serde(default = "default_gate_base")]
    gate_base: f64,
    #[serde(default = "default_da_gain")]
    da_gain: f64,
    #[serde(default)]
    da_idx: Vec<usize>,
    #[serde(default)]
    reward_idx: Vec<usize>,
    #[serde(default)]
    punish_idx: Vec<usize>,
    #[serde(default)]
    predict: bool,
    #[serde(default = "default_pred_eta")]
    pred_eta: f64,
    #[serde(default)]
    pred_idx: Vec<usize>,
    /// 설정 전체와, 뉴런 구조체에 들어 있지 않은 망의 상태 (없으면 구버전 저장 파일)
    #[serde(default)]
    ext: Option<SnapExt>,
}

/// 저장 파일의 덧붙임: 설정 전체, 보조 뉴런 목록과 경계, 새 접점 세기 기준, 뉴런별 잡음 크기, 누적 수.
/// 저장하지 않는 것: 난수 상태(망 안의 난수는 --rng-seed 면 틱으로 정해지고, 발달 시험의 자극 난수는 dev_state 로 되살린다), 1대1 자극, 이번 틱의 발화 목록
#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct SnapExt {
    params: NetParams,
    val_idx: Vec<usize>,
    taste_idx: Vec<Vec<usize>>,
    need_idx: Vec<Vec<usize>>,
    aux_end: usize,
    unit_med: Vec<f64>,
    noise_amp: Vec<f64>,
    sprouted: u64,
    pruned: u64,
    da_spikes: u64,
    pred_spikes: u64,
    val_spikes: u64,
    grow_made: [u64; 5],
    gate_drop: [u64; 5],
    #[serde(default)]
    own_wait_drop: u64,
    added_on_initial: u64,
    pruned_short: u64,
    wiring_sum: u64,
    dev_tick: u64,
    /// 진행 중인 연속 입력 (입력 뉴런들, 남은 틱). 자극이 켜진 틱에 저장해도 이어 돌릴 때 같은 입력이 남은 틱만큼 들어간다
    drives: Vec<(Vec<usize>, u64)>,
    dev_state: Option<DevState>,
    /// slow_events 로 돌던 망의 받는 뉴런별 합(Σ 가중치 × slow1, Σ 가중치 × slow2, 지연 값)과 살아 있는 입력 목록(차례 그대로).
    /// 다시 계산하면 더하는 차례가 달라 끝자리가 달라지므로 저장한다. 옛 방식으로 돌던 망은 비어 있다
    slow_sum: Vec<[f64; 3]>,
    slow_live: Vec<Vec<(u32, u32)>>,
}

/// 발달 시험(dev_test)이 틱 사이에 들고 다니는 자극 상태. 저장 파일에 넣어 두면 불러온 실행이 끊지 않은 실행과 같은 자극 차례를 받는다.
/// 구간마다 0 으로 되돌리는 관찰용 누적값(자극 밖 틱 수 따위)은 넣지 않는다: 저장 틱이 보고 간격의 배수이면 그 값들은 0 이다
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
#[serde(default)]
pub struct DevState {
    /// 자극 난수를 지금까지 뽑은 차례: 자극 때마다 한 글자, '1' = 단어와 간격을 뽑음, '0' = 간격만 뽑음(보상 대기 중이라 단어를 넣지 않음).
    /// 불러올 때 같은 씨앗의 난수에서 이 차례대로 뽑아 버리면 같은 위치가 된다
    pub draws: String,
    pub next_stim: u64,
    /// 예정된 1차 보상 (틱, 종류)
    pub pending: Option<(u64, usize)>,
    pub stim_until: u64,
    /// 보상 구간의 도파민 발화를 재는 중 (끝 틱, 시작 때의 발화 수)
    pub da_mark: Option<(u64, u64)>,
    pub need_before: f64,
    /// 다음 보고 줄까지의 누적값 (저장 틱이 보고 간격의 배수가 아닐 때를 위해)
    pub spikes: u64, pub food_n: u64, pub water_n: u64, pub da_acc: u64, pub da_n: u64,
    pub need_pre_acc: f64, pub need_drop_acc: f64, pub need_drop_n: u64,
}
/// 성장의 난수기: 전역 난수기 하나를 차례로 쓰거나(씨앗 없음), 뉴런마다 열쇠로 만든 작은 난수기를 쓴다(결정적 난수)
enum GrowDraw { Shared(rand::rngs::ThreadRng), Keyed(rand::rngs::StdRng) }
impl GrowDraw {
    fn f(&mut self) -> f64 { use rand::prelude::*; match self { GrowDraw::Shared(r) => r.random::<f64>(), GrowDraw::Keyed(r) => r.random::<f64>() } }
    fn below(&mut self, n: usize) -> usize { use rand::prelude::*; match self { GrowDraw::Shared(r) => r.random_range(0..n), GrowDraw::Keyed(r) => r.random_range(0..n) } }
}
/// 저장 파일에 prune_reset_floor 가 없으면(이 칸이 생기기 전의 파일) 꺼짐
fn reset_floor_missing() -> bool { false }
/// 저장 파일에 grow_gate_soft 가 없으면(이 칸이 생기기 전의 파일) 옛 닫힘/열림 문
fn gate_soft_missing() -> Option<f64> { None }
fn default_pred_eta() -> f64 { 5e-5 }
fn default_gate_base() -> f64 { 0.02 }
fn default_da_gain() -> f64 { 0.05 }
fn default_min_weight() -> f64 { MIN_WEIGHT }
fn default_threshold() -> f64 { 0.50 }
fn default_noise() -> f64 { 0.2 }
