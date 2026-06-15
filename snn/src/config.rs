//! 중앙 설정: 네트워크 / 뉴런 / 시냅스 / 가소성(STDP·iSTDP) / 수면·재생 / 발아 /
//! 서버 등 **모든 튜닝 수치를 한 곳에서** 관리한다.
//!
//! 값을 바꾸려면 이 파일만 수정하면 된다 (컴파일 타임 상수이므로 런타임 오버헤드 없음).
//! 각 상수는 사용처(파일·함수)와 함께 의미를 주석으로 남겨 두었다.
//!
//! 주의: 구조적/정의적 상수(예: 한글 유니코드 범위, djb2 해시 시드,
//! weight 정규화 범위 0.0~1.0, rand→[-1,1] 매핑)는 "튜닝 대상 수치"가 아니므로
//! 의도적으로 각 모듈에 그대로 둔다.

// 일부 상수(BITMAP_*, FIRE_BONUS_* 등)는 아직 메인 경로에 연결되지 않은 모듈
// (bitmap, tokenizer::fire_bonus)에서만 참조되어 dead_code 경고가 날 수 있다.
// 중앙 레지스트리 성격상 전부 보존하므로 모듈 단위로 허용한다.
#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────
// 네트워크 구조 (network.rs :: Network::new / seed_random_synapses)
// ─────────────────────────────────────────────────────────────

/// 영역별 뉴런 수
pub const EMOTION_COUNT: usize = 5000;
pub const REASON_COUNT: usize = 5000;
pub const MEMORY_COUNT: usize = 2000;
pub const HIPPOCAMPUS_COUNT: usize = 500;

/// 2D 격자 배치 열 수
pub const GRID_COLS: usize = 50; // 감정 / 이성 / 기억
pub const HIPPO_COLS: usize = 25; // 해마

/// Main 영역 비율(%). 나머지(20%)는 Map 영역.
pub const MAIN_AREA_PCT: usize = 80;

/// 억제성 뉴런 판정: `i % INHIBITORY_MOD >= THRESHOLD`
pub const INHIBITORY_MOD: usize = 10;
pub const INHIBITORY_THRESHOLD: usize = 7; // 감정/이성/기억: 30% 억제성
pub const HIPPO_INHIBITORY_THRESHOLD: usize = 8; // 해마: 20% 억제성

/// Map 영역 임계값 스케일 (0.85 / 0.50 = 1.7)
pub const MAP_THRESHOLD_SCALE: f64 = 1.7;

/// 영역 배치 좌표 오프셋
pub const REASON_Y_OFFSET: f32 = 100.0;
pub const MEMORY_X_OFFSET: f32 = 50.0;
pub const MEMORY_Y_OFFSET: f32 = 50.0;
pub const HIPPO_X_OFFSET: f32 = 60.0;
pub const HIPPO_Y_OFFSET: f32 = 75.0;

/// 입출력 뉴런 배치 (register_token)
pub const IO_COLS: u32 = 20; // seq % IO_COLS
pub const INPUT_Y_BASE: f32 = -10.0;
pub const OUTPUT_Y_BASE: f32 = 200.0;

/// 초기 시드 시냅스: 뉴런당 생성 개수
pub const SEED_SYNAPSES_PER_NEURON: usize = 10;

// ─────────────────────────────────────────────────────────────
// 가중치 (network.rs)
// ─────────────────────────────────────────────────────────────

pub const INITIAL_WEIGHT: f64 = 0.2; // 기본 시냅스 weight / 발아 weight
pub const MIN_WEIGHT: f64 = 0.1; // prune 기준
pub const WEIGHT_MIN: f64 = 0.0; // weight 정규화 하한
pub const WEIGHT_MAX: f64 = 1.0; // weight 정규화 상한

/// 입력/출력 시드 weight: SEED_WEIGHT_MIN ~ (MIN + RANGE) = 0.4 ~ 0.8
pub const SEED_WEIGHT_MIN: f64 = 0.4;
pub const SEED_WEIGHT_RANGE: f64 = 0.4;

/// 시드 연결 분기 확률 (난수 r 과 비교)
pub const SEED_P_INTERNAL: f64 = 0.5; // r < 0.5 → 내부 연결
pub const SEED_P_MAP: f64 = 0.7; // r < 0.7 → Map (그 외 → 출력)
pub const SEED_P_MAP_BACK: f64 = 0.4; // Map→Main 역방향 r < 0.4

// ─────────────────────────────────────────────────────────────
// 뉴런 동역학 (neuron.rs)
// ─────────────────────────────────────────────────────────────

/// 불응기 (틱)
pub const ABSOLUTE_REFRACTORY: u64 = 4;
pub const ABSOLUTE_REFRACTORY_INH: u64 = 2; // 억제성: 빠른 재발화
pub const RELATIVE_REFRACTORY: u64 = 10;
pub const RELATIVE_REFRACTORY_INH: u64 = 4;
/// 상대 불응기 동안 강제 발화 임계 배수
pub const RELATIVE_REFRACTORY_FACTOR: f64 = 2.0;

/// 막전위 감쇠율
pub const DECAY_RATE: f64 = 0.7; // 기본
pub const DECAY_RATE_SLOW: f64 = 0.85; // 기억 / 해마
pub const THRESHOLD_SCALE: f64 = 1.0; // 기본 임계 스케일

/// 자가 임계값 (self_threshold)
pub const SELF_THRESHOLD_INIT: f64 = 0.5; // 일반 뉴런 기본값 = 하한
pub const SELF_THRESHOLD_IO: f64 = 0.3; // 입출력 뉴런 기본값 = 하한
pub const SELF_THRESHOLD_MAX: f64 = 1.0; // 상한
pub const SELF_THRESHOLD_STEP: f64 = 0.001; // 발화 시 +, 미발화 시 -

pub const EXCITABILITY_INIT: f64 = 1.0; // 감응도 기본값
pub const RECEIVED_SIGNAL_EPS: f64 = 0.01; // potential > eps → 신호 받음 판정

/// 자발 발화 확률
pub const SPONTANEOUS_FIRE_PROB_STIM: f64 = 0.000001; // 자극 중
pub const SPONTANEOUS_FIRE_PROB_IDLE: f64 = 0.00001; // 유휴 중

/// 항상성 가소성 (homeostasis)
pub const FIRE_RATE_WINDOW: f64 = 10_000.0; // 발화율 정규화 윈도우(틱)
pub const HOMEOSTASIS_RATE_HIGH: f64 = 0.05; // 과활성 기준
pub const HOMEOSTASIS_RATE_LOW: f64 = 0.01; // 저활성 기준
pub const HOMEOSTASIS_DOWN: f64 = 0.95; // 과활성 시 감응도 감소
pub const HOMEOSTASIS_UP: f64 = 1.05; // 저활성 시 감응도 증가
pub const EXCITABILITY_MIN: f64 = 0.5;
pub const EXCITABILITY_MAX: f64 = 1.5;

// ─────────────────────────────────────────────────────────────
// 시냅스 동역학 (synapse.rs)
// ─────────────────────────────────────────────────────────────

pub const FATIGUE_INIT: f64 = 1.0;
pub const FATIGUE_DECAY: f64 = 0.90; // 발화 시 fatigue 감소
pub const FATIGUE_RECOVER: f64 = 0.01; // 매 틱 fatigue 회복
pub const FATIGUE_MAX: f64 = 1.0;

pub const RECENT_RATE_STEP: f64 = 0.1; // 발화 시 전달 빈도 증가
pub const RECENT_RATE_MAX: f64 = 1.0;
pub const RECENT_RATE_DECAY: f64 = 0.99; // 매 틱 감쇠
pub const RATE_WEIGHT_PENALTY: f64 = 0.01; // weight -= penalty * rate

pub const LTP_TRACE_DECAY: f64 = 0.95; // 매 틱 trace 감쇠
pub const LTP_TRACE_THRESHOLD: f64 = 2.0; // 이 값 이상이면 추가 LTP
pub const LTP_TRACE_GAIN: f64 = 0.001; // (trace - 1) * gain

// ─────────────────────────────────────────────────────────────
// STDP / BCM (network.rs :: tick)
// ─────────────────────────────────────────────────────────────

pub const STDP_A_PLUS: f64 = 0.003; // LTP 기본량
pub const STDP_A_MINUS: f64 = 0.0036; // LTD 기본량 (LTD:LTP = 1.2:1)
pub const STDP_TAU: f64 = 20.0; // 시간 상수
pub const STDP_WINDOW: f64 = 50.0; // dt 유효 범위 (틱)
pub const BCM_TARGET_RATE: f64 = 25.0; // BCM 목표 발화율
pub const BCM_LTD_ANCHOR: f64 = 2.0; // BCM LTD 비대칭 앵커: LTD = A_minus × (anchor − bcm_scale)

/// weight 의존 가우시안 (학습 가능 weight 영역으로 끌어당김)
pub const STDP_WEIGHT_TARGET_LTP: f64 = 0.2; // LTP 가우시안 중심
pub const STDP_WEIGHT_TARGET_LTD: f64 = 1.0; // LTD 가우시안 중심
pub const STDP_WEIGHT_SIGMA: f64 = 0.4; // 가우시안 표준편차

// ─────────────────────────────────────────────────────────────
// iSTDP (억제 가소성, network.rs :: tick)
// ─────────────────────────────────────────────────────────────

pub const ISTDP_RATE: f64 = 0.04;
pub const ISTDP_TARGET_RATE: f64 = 10.0;
pub const ISTDP_DOWN_FACTOR: f64 = 0.5; // 목표 미만 시 감소 배수

// ─────────────────────────────────────────────────────────────
// 자극 / 틱 주기 (network.rs)
// ─────────────────────────────────────────────────────────────

pub const STIMULUS_POTENTIAL: f64 = 1.0; // 입력 주입 막전위
pub const STIMULUS_SUSTAIN: u64 = 4; // 입력 지속 틱
pub const STIMULUS_SILENCE_TICKS: u64 = 1; // 발화 종료 판정: 연속 무발화 틱 수
pub const SPIKE_COLLECT_WINDOW: u64 = 14; // 발화 수집 윈도우 (elapsed_ticks 이하)
pub const HOMEOSTASIS_INTERVAL: u64 = 10_000; // 항상성 호출 주기
pub const SPROUT_INTERVAL: u64 = 500; // 발아 호출 주기
pub const FIRE_RECORDS_MAX: usize = 100; // 보관할 발화 기록 수

// ─────────────────────────────────────────────────────────────
// 발아 (구조적 가소성, network.rs :: sprout)
// ─────────────────────────────────────────────────────────────

pub const SPROUT_RADIUS: f32 = 5.0; // 발아 반경
pub const MAX_SPROUT_PER_NEURON: usize = 1;
pub const SPROUT_COOLDOWN_TICKS: u64 = 500;
pub const SPROUT_PROBABILITY: f64 = 0.01;
pub const SPROUT_MIN_RATE: f64 = 0.01; // 발아 허용 발화율 하한
pub const SPROUT_MAX_RATE: f64 = 0.05; // 발아 허용 발화율 상한

// ─────────────────────────────────────────────────────────────
// 가지치기 (network.rs)
// ─────────────────────────────────────────────────────────────

pub const PRUNE_INTERVAL: u64 = 5_000; // fire_id 주기

// ─────────────────────────────────────────────────────────────
// 보상 / 적격 흔적 / teach (network.rs)
// ─────────────────────────────────────────────────────────────

pub const ELIGIBILITY_LTP: f64 = 0.0072; // dt > 0 강화
pub const ELIGIBILITY_LTD: f64 = 0.006; // dt < 0 약화
pub const ELIGIBILITY_TAU: f64 = 20.0;
pub const TEACH_POTENTIAL: f64 = 0.6; // teach 시 출력 뉴런 사전 활성

// ─────────────────────────────────────────────────────────────
// 전역 런타임: threshold / noise 램프 (network.rs)
// ─────────────────────────────────────────────────────────────

pub const THRESHOLD_INIT: f64 = 0.50;
pub const THRESHOLD_MAX: f64 = 1.0;
pub const NOISE_INIT: f64 = 0.2;
pub const NOISE_MIN: f64 = 0.1;
pub const RAMP_PERIOD: u64 = 10_000; // fire_id / RAMP_PERIOD
pub const RAMP_STEP: f64 = 0.001; // 주기당 가감량

// ─────────────────────────────────────────────────────────────
// 수면 / 해마 재생 (network.rs :: enter_sleep / should_sleep)
// ─────────────────────────────────────────────────────────────

pub const SLEEP_FIRE_INTERVAL: u64 = 30_000; // 누적 fire 수 기준
pub const SLEEP_TICK_INTERVAL: u64 = 500_000; // 경과 틱 기준
pub const SLEEP_DURATION_TICKS: u64 = 1_000; // 휴지기 길이
pub const SLEEP_WEIGHT_SCALE: f64 = 0.92; // fatigue scaling
pub const SLEEP_WEIGHT_DECAY: f64 = 0.95; // weight scaling (SHY 가설)
pub const REPLAY_ROUNDS: usize = 5; // 해마 재생 라운드
pub const REPLAY_TICKS: u64 = 10; // 라운드당 틱
pub const REPLAY_POTENTIAL: f64 = 0.5; // 해마 자극 막전위

// ─────────────────────────────────────────────────────────────
// 토크나이저 발화 보너스 (tokenizer.rs :: fire_bonus, 토큰 길이별)
// ─────────────────────────────────────────────────────────────

pub const FIRE_BONUS_1: f64 = 0.6; // 1글자 (자모)
pub const FIRE_BONUS_2: f64 = 0.8; // bigram
pub const FIRE_BONUS_3: f64 = 0.9; // trigram
pub const FIRE_BONUS_SHORT_WORD: f64 = 1.2; // 4~5글자 짧은 단어
pub const FIRE_BONUS_WORD: f64 = 1.0; // 6~10글자 일반 단어
pub const FIRE_BONUS_LONG: f64 = 0.7; // 그 이상

// ─────────────────────────────────────────────────────────────
// 서버 / IO (server.rs / main.rs)
// ─────────────────────────────────────────────────────────────

pub const DEFAULT_PORT: u16 = 3000;
pub const DEFAULT_SAVE_PATH: &str = "data/snn.json";
pub const AUTO_SAVE_SECS: u64 = 300; // 자동 저장 주기
pub const REQUEST_CHANNEL_CAP: usize = 256; // 요청 채널 용량
pub const JSON_LIMIT_BYTES: usize = 1_048_576; // JSON 본문 제한 (1MB)
pub const KEEP_ALIVE_SECS: u64 = 300;
pub const CLIENT_TIMEOUT_SECS: u64 = 300;
pub const STATUS_UPDATE_SECS: u64 = 1; // status 캐시 갱신 주기
pub const DEFAULT_FEEDBACK_STRENGTH: f64 = 1.0;

// ─────────────────────────────────────────────────────────────
// 비트맵 렌더링 (bitmap.rs)
// ─────────────────────────────────────────────────────────────

pub const BITMAP_GRID: usize = 32; // 32×32 격자
pub const BITMAP_RENDER_PX: f32 = 28.0; // 폰트 렌더 크기 (격자 여백 확보)
pub const BITMAP_PIXEL_THRESHOLD: u8 = 128; // 픽셀 활성 임계
