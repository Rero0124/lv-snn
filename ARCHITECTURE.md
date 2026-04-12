# LV-SNN v2 아키텍처

## 개요

뇌의 구역 구조를 모방한 스파이킹 신경망(SNN) 연구 프로젝트 v2.
연속 틱 모델에서 막전위(potential)가 감쇠/누적되며 임계값을 넘으면 발화하고,
R-STDP(reward-modulated STDP)로 경로를 학습한다.
생물학적으로 깨어있는 동안 누적되는 시냅스 총량을 주기적으로 수면을 통해
재정규화(SHY 가설 기반)한다.

## 파일 구조

```
snn/src/
├── main.rs       # 서버/대화형 모드 진입점 (--serve, --port, --data)
├── server.rs     # HTTP API + 단일 워커 스레드 기반 순차 처리
├── network.rs    # Network — 뉴런/시냅스/틱 루프/수면/R-STDP/축삭발아
├── neuron.rs     # Neuron — 막전위, 불응기, 발화 판정, 항상성 가소성
├── synapse.rs    # Synapse — weight, 피로도 fatigue, seed 플래그
├── region.rs     # RegionType (Input/Emotion/Reason/Output)
└── tokenizer.rs  # 한글 자모 분해/재조합

scripts/
├── ai_train.py     # Ollama 기반 자율 탐색 학습 (점수 0.0~1.0)
├── fast_train.py   # 빠른 학습
├── test_words.py   # teach + fire + feedback 학습 스크립트
└── test_jamo*.py   # 자모 단위 학습 테스트

data/
└── snn.json       # Network 스냅샷 (neurons/synapses/vocab/tick 상태)
```

## 뉴런 구조 (neuron.rs)

```rust
Neuron {
    id: NeuronId,                    // u32
    potential: f64,                  // 막전위 (매 틱 감쇠)
    last_spike_tick: Option<u64>,    // 불응기 판정용
    synapses: Vec<Synapse>,          // 나가는 시냅스
    x, y: f32,                       // 2D 위치 (축삭발아 거리 계산)
    inhibitory: bool,                // 억제성 뉴런이면 weight × -1
    skip_refractory: bool,           // 입출력 뉴런은 불응기 스킵
    excitability: f64,               // 항상성 가소성 (0.5~1.5, 기본 1.0)
    fire_count_window: u32,          // 최근 1만틱 발화 횟수
}
```

### 발화 판정 (try_fire)

1. **불응기 체크**
   - 절대 불응기(ABSOLUTE_REFRACTORY=4틱): 발화 불가
   - 상대 불응기(RELATIVE_REFRACTORY=10틱): 임계값 × 2 필요
2. **자발적 발화** — 자극 중 `0.000001`, idle 중 `0.00001`
3. **판정**: `potential + noise ≥ threshold / excitability`
4. **발화 시**: `potential = 0`, 시냅스 fatigue 적용, 신호 전달

### 항상성 가소성 (homeostasis)

1만 틱마다 호출:
- fire_rate > 5% → `excitability *= 0.95` (억제)
- fire_rate < 1% → `excitability *= 1.05` (촉진)
- 범위: `clamp(0.5, 1.5)`

## 시냅스 구조 (synapse.rs)

```rust
Synapse {
    target: NeuronId,
    weight: f64,          // 구조적 연결 강도
    seed: bool,            // 초기 시냅스 (가지치기 면제)
    fatigue: f64,          // 0.0~1.0 — 발화 시 × 0.80, 매 틱 +0.01 회복
}

effective_weight = weight × fatigue
```

## 네트워크 구조 (network.rs)

```
                    입력 뉴런 (80개)
                        │
                        ▼
         ┌──────────────────────────────┐
         │  감정(5000) ↔↔↔↔ 이성(5000)   │
         │  (30% 억제성)    (30% 억제성) │
         └──────────────┬───────────────┘
                        ▼
                    출력 뉴런 (80개)
```

- 입력/출력 뉴런: 어휘 1개당 1쌍 (자모 80개 기본)
- 감정/이성: 내부 시냅스 10개 + 상대 구역 10개 + 출력 10개 (초기 seed)
- 시냅스 초기화: 거리 기반 확률적 근거리 연결

## 틱 루프 (network.rs — tick / idle_tick)

```
매 틱:
  1. global_tick += 1
  2. 활성 자극의 입력 뉴런 sustain (remaining_sustain 동안 potential=1.0)
  3. 모든 뉴런 decay (potential *= 0.7)
  4. 병렬 발화 판정 (rayon par_iter_mut → try_fire)
  5. 신호 전달 (발화한 뉴런 → 시냅스 → 타깃 receive)
  6. recent_spikes 기록
  7. silent_ticks 업데이트
  8. 완료 판정 (silent_ticks ≥ 1 → fire 종료)
  9. 완료된 자극 처리 (eligibility 계산 + FireRecord 생성)
  10. 항상성 가소성 (global_tick % 10_000 == 0)
  11. 축삭발아 (global_tick % 100 == 0)
```

idle_tick은 자극 없을 때 동작 — decay/발화/전달/축삭발아/항상성 동일.

## 수면 메커니즘 (network.rs — enter_sleep)

SHY 가설(Synaptic Homeostasis Hypothesis) 기반: 깨어있는 동안 누적된
LTP를 재정규화하는 유일한 메커니즘.

### 트리거 (should_sleep)

```rust
fires_since_sleep >= SLEEP_FIRE_INTERVAL (30_000)
|| global_tick - last_sleep_tick >= SLEEP_TICK_INTERVAL (100_000)
```

### 동작

1. 모든 시냅스 `weight *= SLEEP_WEIGHT_SCALE (0.92)` — 균일 다운스케일
   (상대적 강약 보존)
2. 모든 뉴런 `potential=0`, `excitability=1.0`, `fire_count_window=0`
3. SLEEP_DURATION_TICKS(1_000) 동안 순수 감쇠만 (발화/학습/sprout 정지)
4. 카운터 리셋

## R-STDP (network.rs — compute_eligibility / feedback)

```
발화 시에는 적격성 흔적(eligibility)만 계산, 가중치 변경 없음.
feedback 호출 시 reward를 곱해서 실제 적용.

dt = t_post - t_pre
  dt > 0 (LTP): dw = +0.01 × exp(-|dt|/20)
  dt < 0 (LTD): dw = -0.01 × exp(-|dt|/20)

feedback(positive, strength):
  reward = positive ? strength : -strength
  new_weight = weight + dw × reward (clamp 0~1)
```

## 축삭발아 (network.rs — sprout)

100틱마다 최근 발화 뉴런 기준:
- SPROUT_RADIUS=5.0 내 후보 중 가까운 순
- SPROUT_PROBABILITY=0.5
- SPROUT_COOLDOWN_TICKS=500
- MAX_SPROUT_PER_NEURON=1
- 입력↔입력, 출력↔출력, 입력↔출력 연결 차단

## Threshold 점진 상승 (network.rs — fire)

1만 fire마다:
```
threshold = min(1.0, 0.72 + (fire_id / 10_000) × 0.001)
모든 weight -= 0.003  (threshold 상승분 보정, 2026-04-10 수정: 0.01 → 0.003)
noise_range = max(0.1, 0.2 - (fire_id / 10_000) × 0.001)
```

## 가지치기 (network.rs — prune)

PRUNE_INTERVAL(10_000) fire마다:
- `weight < MIN_WEIGHT(0.1)` 시냅스 제거 (seed 제외는 아직 구현 안 됨)

## HTTP API (server.rs)

```
POST /fire      {"text": "..."}                              → fire_id + output
POST /teach     {"input": "...", "target": "..."}            → fire + 출력 뉴런 pre-자극
POST /feedback  {"fire_id": N, "positive": bool, "strength"} → R-STDP 적용
POST /save
GET  /status                                                  → neurons/synapses/fire_count/threshold
```

- Network는 워커 스레드 단독 소유 (Mutex 없음)
- busy 플래그로 처리 중 상태 표시

## 핵심 상수 일람 (2026-04-12 현재)

### 뉴런 / 시냅스 (neuron.rs, synapse.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| `potential *= X` (decay) | **0.7** | 막전위 틱당 감쇠 계수 (원래 0.8, 2026-04-12: 0.7로 강화) |
| `fatigue *= X` (fire_fatigue) | **0.80** | 발화 시 피로도 감소 계수 (원래 0.77 → 0.85 → 0.80) |
| fatigue recover | +0.01/틱 | 매 틱 피로 회복 |
| ABSOLUTE_REFRACTORY | 4 | 절대 불응기 틱 수 |
| RELATIVE_REFRACTORY | 10 | 상대 불응기 틱 수 (임계값 × 2) |
| excitability 범위 | 0.5 ~ 1.5 | 항상성 clamp 범위 |
| homeostasis 주기 | 10_000틱 | |
| 자발 발화 (자극 중) | 0.000001 | |
| 자발 발화 (idle) | 0.00001 | |

### 네트워크 구조 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| EMOTION_COUNT | 5000 | 감정 뉴런 수 |
| REASON_COUNT | 5000 | 이성 뉴런 수 |
| 억제성 비율 | **30%** | `i % 10 >= 7` (원래 30 → 20 → 30 복귀) |
| INITIAL_WEIGHT | **0.2** | 초기 시냅스 가중치 |
| 중간→출력 seed weight | **1.0** | 출력 경로 seed 강도 (2026-04-12 추가) |
| MIN_WEIGHT | 0.1 | 가지치기 기준 |
| seed 시냅스 (입력→중간) | 10개 | 뉴런당 내부 연결 |
| seed 시냅스 (중간 내부) | 30개 | 80% 내부 + 20% 출력 (w=1.0) |
| initial threshold | **0.72** | 점진 상승, weight 감산 제거됨 |
| initial noise_range | 0.2 | |

### 수면 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| SLEEP_FIRE_INTERVAL | **60_000** | fire 기준 수면 트리거 (2026-04-12 갱신) |
| SLEEP_TICK_INTERVAL | **500_000** | tick 기준 트리거 (2026-04-12 갱신) |
| SLEEP_DURATION_TICKS | 1_000 | 수면 지속 틱 |
| SLEEP_WEIGHT_SCALE | 0.92 | 수면 시 fatigue 배율 (weight → fatigue 전환, 2026-04-12) |
| sleep LTD | **-0.002** | 수면 중 시냅스 LTD (2026-04-12 추가) |
| prune (수면 중) | **제거됨** | 2026-04-12 비활성화 |

### Threshold 점진 상승

| 상수 | 값 | 설명 |
|------|-----|------|
| threshold ramp | +0.001 / 10k fire | 최대 1.0 |
| ramp weight 감산 | **제거됨** | 2026-04-12 제거 (기존 -0.003) |
| noise ramp | -0.001 / 10k fire | 최소 0.1 |

### R-STDP (wake)

| 상수 | 값 | 설명 |
|------|-----|------|
| STDP LTP (A_plus) | **0.015** | 2026-04-12 추가 |
| STDP LTD (A_minus) | **0.006** | 2026-04-12 추가 |
| τ | 20.0 | 지수 감쇠 시간 상수 |
| R-STDP reward (positive) | +0.01 | feedback 시 |
| R-STDP reward (negative) | -0.01 | feedback 시 |

### 헤비안 / LTP 누적 / BCM (2026-04-12 추가)

| 상수 | 값 | 설명 |
|------|-----|------|
| 헤비안 동시발화 보정 | **+0.01** | 동시 발화 시 weight 강화 |
| LTP trace 감쇠 | **× 0.95** | 매 틱 trace 감쇠 |
| LTP trace bonus 기준 | **≥ 2** | trace 누적 시 추가 강화 |
| LTP trace bonus 크기 | **+0.005** | |
| BCM TARGET_RATE | **25** | BCM 메타가소성 목표 발화율 |

### teach

| 상수 | 값 | 설명 |
|------|-----|------|
| teach pre-자극 | **0.6** | 출력 뉴런 사전 자극 강도 (2026-04-12, 자동 feedback 제거) |

### 축삭발아 (sprout)

| 상수 | 값 | 설명 |
|------|-----|------|
| SPROUT_RADIUS | 5.0 | 탐색 반경 |
| SPROUT_PROBABILITY | **0.1** | 2026-04-12: 0.5 → 0.1 (25배 축소) |
| SPROUT_COOLDOWN_TICKS | **500** | 뉴런당 최소 간격 (2026-04-12) |
| MAX_SPROUT_PER_NEURON | 1 | 틱당 |
| sprout 주기 | **500틱마다** | |

### Fire 내부

| 상수 | 값 | 설명 |
|------|-----|------|
| input potential | 1.0 | 입력 뉴런 자극 강도 |
| input sustain | 4틱 | remaining_sustain |
| silent 완료 기준 | 1틱 | |
| PRUNE_INTERVAL | 10_000 fire | |

## 학습 흐름

```
1. /fire "안녕" → inject_stimulus → tick 루프 → 출력 토큰 조합
2. 평가 (스크립트) → /feedback fire_id positive strength
3. feedback이 eligibility × reward로 weight 업데이트
4. 내부 상태: threshold 점진 상승, fatigue/excitability/항상성,
   주기적 sleep으로 재정규화
```

## 수정 이력 (2026-04-10 이후)

| 날짜 | 항목 | 변경 | 비고 |
|------|------|------|------|
| 2026-04-10 | fatigue | 0.77 → 0.85 → 0.80 | 출력 부족 해결 시도 → 폭주 → 중간값 |
| 2026-04-10 | 억제 비율 | 30% → 20% → 30% | 폭주 확인 후 복귀 |
| 2026-04-10 | SLEEP_FIRE_INTERVAL | 500 → 10_000 → 30_000 | 수면 텀 확장 |
| 2026-04-10 | threshold ramp weight 감산 | 0.01 → 0.003 | 학습 파괴 방지 |
| 2026-04-10 | sleep 메커니즘 추가 | — | SHY 가설 기반 |
| 2026-04-12 | decay | 0.8 → 0.7 | 단어 입력 폭주 방지 (부작용: 출력 없음) |
| 2026-04-12 | STDP | 추가 (LTP 0.015, LTD 0.006) | wake STDP 도입 |
| 2026-04-12 | 헤비안 | 동시 발화 +0.01 | 경로 공동 강화 |
| 2026-04-12 | LTP 누적 | trace 기반 반복 활성 강화 | trace ×0.95, ≥2 시 bonus 0.005 |
| 2026-04-12 | BCM 메타가소성 | TARGET_RATE=25 | 과활성 억제 |
| 2026-04-12 | sleep | weight → fatigue 전환, LTD -0.002, prune 제거 | SHY 방식 개선 |
| 2026-04-12 | sleep 주기 | 60k fire / 500k tick | 수면 트리거 완화 |
| 2026-04-12 | threshold ramp | weight 감산 제거 | 기존 -0.003 비활성화 |
| 2026-04-12 | 중간→출력 seed | weight 1.0 | 출력 경로 초기 강도 확보 |
| 2026-04-12 | sprout | 500틱 / 0.1 확률 | 기존 대비 25배 축소 |
| 2026-04-12 | teach | pre-자극 0.6, 자동 feedback 제거 | 출력 사전 자극 강화 |

## 현재 문제

단어 입력(예: "안녕", "고마워") 학습 시:
- **출력률 10~22%**: 목표 70%+ 대비 크게 부족 (신호가 출력 뉴런에 도달하지 못함)
- **정답률 0%**: 경로 분화가 전혀 이루어지지 않음 (어떤 입력에도 동일 출력 또는 무응답)
- **폭주 문제는 해결됨**: decay 0.7 + fatigue 0.80 조합으로 안정화

STDP/헤비안/LTP 누적/BCM 등 다수 학습 메커니즘을 추가했으나 경로 분화 미달.
출력 경로 seed weight 1.0 적용에도 출력률 저조 → 입력 자극이 중간층을 거쳐
출력까지 도달하는 전파 경로 자체가 불충분한 것이 근본 원인으로 추정.

## 조정 후보 파라미터 (우선순위순)

현재 미해결 문제: **출력률 10~22%** (목표 70%+), **정답률 0%** (경로 분화 없음)

### 우선순위 높음 — 출력 전파 강화 (출력률 개선)

1. **decay** 0.7 → 0.72~0.74 ([neuron.rs:60](snn/src/neuron.rs#L60))
   - 현재 0.7은 신호가 중간층 통과 전 소멸하는 주원인으로 추정
   - 폭주 없이 전파 가능한 균형점 탐색 필요
2. **input potential** 1.0 → 1.5 ([network.rs:283](snn/src/network.rs#L283))
   - 입력 자극 강도 증가 → 중간층 도달 신호량 확보
3. **input sustain** 4틱 → 8틱 ([network.rs:292](snn/src/network.rs#L292))
   - 자극 지속 시간 확장 → 활동 확산 시간 확보
4. **threshold** 0.72 → 0.65 ([network.rs:146](snn/src/network.rs#L146))
   - 발화 임계 하향 → 약한 전파도 출력 발화 가능

### 우선순위 높음 — 경로 분화 (정답률 개선)

5. **R-STDP A_plus / A_minus** 0.01 → 0.02~0.03 ([network.rs:634-636](snn/src/network.rs#L634-L636))
   - 학습률 강화 → feedback 시 경로 분화 가속
6. **STDP LTP/LTD 비율** LTP 0.015 / LTD 0.006 → LTP 0.02 / LTD 0.005
   - LTP 강화, LTD 약화 → 활성 경로 강도 축적
7. **BCM TARGET_RATE** 25 → 15~20
   - 목표 발화율 낮춤 → 선택적 억제 완화, 분화 용이

### 우선순위 중간 — 활동 유지 / 구조

8. **INITIAL_WEIGHT** 0.2 → 0.3 ([network.rs:11](snn/src/network.rs#L11))
   - seed 경로 초기 강도 증가 (폭주 주의)
9. **noise_range** 0.2 → 0.3 ([network.rs:147](snn/src/network.rs#L147))
   - 확률적 발화 기회 증가
10. **seed synapses (입력→중간)** 10개 → 15~20개 ([network.rs:159](snn/src/network.rs#L159))
    - 초기 연결 밀도 증가 (네트워크 생성 시점만 적용)

### 우선순위 낮음 — 수면 / 발아

11. **SLEEP_WEIGHT_SCALE** 0.92 → 0.95 ([network.rs:25](snn/src/network.rs#L25))
    - 수면 중 덜 깎음 → 누적 학습 보존
12. **sleep LTD** -0.002 → -0.001
    - 수면 중 LTD 약화 → 학습된 경로 손실 감소
13. **SPROUT_RADIUS** 5.0 → 8.0 ([network.rs:654](snn/src/network.rs#L654))
    - 발아 범위 확대 → 출력 경로 신규 연결 가능성
14. **SPROUT_PROBABILITY** 0.1 → 0.2 ([network.rs:655](snn/src/network.rs#L655))
    - 발아 확률 증가

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 발화 판정 (par_iter_mut) |
| serde + serde_json | 직렬화 (스냅샷) |
| rand 0.10 | 확률적 선택 |
