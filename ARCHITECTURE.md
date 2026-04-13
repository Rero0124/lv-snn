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
├── test_jamo*.py   # 자모 단위 학습 테스트
└── scripts_util.py # 자모 겹침 평가 유틸리티

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

1. **불응기 체크** (억제/흥분 분리)
   - 흥분성: 절대 4틱, 상대 10틱
   - 억제성: 절대 2틱, 상대 4틱 (빠른 재발화로 지속 억제)
   - 상대 불응기 중: 임계값 × 2 필요
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
    fatigue: f64,          // 0.0~1.0 — 발화 시 × 0.90, 매 틱 +0.01 회복
    ltp_trace: f64,        // LTP 누적 카운터
}

effective_weight = weight × fatigue
```

## 네트워크 구조 (network.rs)

```
                    입력 뉴런 (80개)
                        │ (w=0.35)
                        ▼
         ┌──────────────────────────────┐
         │  감정(5000) ↔↔↔↔ 이성(5000)   │
         │  (30% 억제성)    (30% 억제성) │
         │         (w=0.2)              │
         └──────────────┬───────────────┘
                        │ (w=1.0)
                        ▼
                    출력 뉴런 (80개)
```

- 입력/출력 뉴런: 어휘 1개당 1쌍 (자모 80개 기본)
- 입력→중간: 10개 seed 시냅스 (w=0.35, 시상피질 비율)
- 중간 내부: 30개 seed (80% 내부/상호 w=0.2 + 20% 출력 w=1.0)
- 시냅스 초기화: 거리 기반 확률적 근거리 연결

## 틱 루프 (network.rs — tick / idle_tick)

```
매 틱:
  1. global_tick += 1
  2. 활성 자극의 입력 뉴런 sustain (remaining_sustain 동안 potential=1.0)
  3. 모든 뉴런 decay (potential *= 0.7)
  4. 병렬 발화 판정 (rayon par_iter_mut → try_fire)
  5. 신호 전달 (발화한 뉴런 → 시냅스 → 타깃 receive)
  6. STDP (동시 발화 LTP + 타이밍 기반 LTD)
  7. iSTDP (억제 시냅스 자동 조정)
  8. recent_spikes 기록
  9. silent_ticks 업데이트
  10. 완료 판정 (silent_ticks ≥ 1 → fire 종료)
  11. 완료된 자극 처리 (eligibility 계산 + FireRecord 생성)
  12. 항상성 가소성 (global_tick % 10_000 == 0)
  13. 축삭발아 (global_tick % 500 == 0, 발화율 조건부)
```

idle_tick은 자극 없을 때 동작 — decay/발화/전달/축삭발아/항상성 동일.

## 수면 메커니즘 (network.rs — enter_sleep)

SHY 가설(Synaptic Homeostasis Hypothesis) 기반: 깨어있는 동안 누적된
LTP를 재정규화하는 유일한 메커니즘.

### 트리거 (should_sleep)

```rust
fires_since_sleep >= SLEEP_FIRE_INTERVAL (60_000)
|| global_tick - last_sleep_tick >= SLEEP_TICK_INTERVAL (500_000)
```

### 동작

1. 모든 시냅스 `fatigue *= SLEEP_WEIGHT_SCALE (0.92)` + `weight -= SLEEP_LTD (0.001)`
2. 모든 뉴런 `potential=0`, `excitability=1.0`, `fire_count_window=0`
3. SLEEP_DURATION_TICKS(1_000) 동안 순수 감쇠만 (발화/학습/sprout 정지)
4. 카운터 리셋

## 학습 메커니즘

### STDP (tick 루프 내, 비지도)

```
동시 발화 (dt == 0):
  dw = +A_plus (0.004) × BCM_scale + LTP_trace_bonus
  → 동시 활성 경로 강화

LTD (dt < 0, post가 먼저 발화):
  dw = -A_minus (0.0048) × (2 - BCM_scale) × exp(dt/τ)
  → 비인과적 연결 약화
  → LTD = LTP × 1.2 (약화 우세)
```

### iSTDP (억제 시냅스 전용)

```
억제 뉴런 발화 시:
  타깃 fire_count > TARGET_RATE (10):
    weight += RATE (0.04)   → 과활성 타깃에 대한 억제 강화
  타깃 fire_count ≤ TARGET_RATE:
    weight -= RATE × 0.5    → 저활성 타깃에 대한 억제 약화
```

### R-STDP (feedback 시, 지도)

```
발화 시에는 적격성 흔적(eligibility)만 계산, 가중치 변경 없음.
feedback 호출 시 reward를 곱해서 실제 적용.

dt = t_post - t_pre
  dt > 0: dw = +0.006 × exp(-|dt|/20)
  dt < 0: dw = -0.006 × exp(-|dt|/20)

feedback(positive, strength):
  reward = positive ? strength : -strength
  new_weight = weight + dw × reward (clamp 0~1)
```

### BCM 메타가소성

```
bcm_scale = TARGET_RATE(25) / (fire_count + TARGET_RATE)
→ 과활성 뉴런: bcm_scale ↓ → LTP 약화, LTD 강화
→ 저활성 뉴런: bcm_scale ↑ → LTP 강화, LTD 약화
```

### LTP 누적 (synapse.rs)

```
발화 시 ltp_trace += 1.0, 매 틱 trace *= 0.95
trace ≥ 2: bonus = (trace - 1) × 0.001
→ 반복 활성 경로 추가 강화
```

## 축삭발아 (network.rs — sprout)

500틱마다 최근 발화 뉴런 기준:
- **발화율 조건**: fire_rate 1%~5% 범위 뉴런만 허용 (과활성 뉴런 발아 차단)
- SPROUT_RADIUS=5.0 내 후보 중 가까운 순
- SPROUT_PROBABILITY=0.1
- SPROUT_COOLDOWN_TICKS=500
- MAX_SPROUT_PER_NEURON=1
- 입력↔입력, 출력↔출력, 입력↔출력 연결 차단

## Threshold 점진 상승 (network.rs — fire)

1만 fire마다:
```
threshold = min(1.0, 0.72 + (fire_id / 10_000) × 0.001)
noise_range = max(0.1, 0.2 - (fire_id / 10_000) × 0.001)
```

## 가지치기 (network.rs — prune)

PRUNE_INTERVAL(5_000) fire마다:
- `weight < MIN_WEIGHT(0.15)` 시냅스 제거

## HTTP API (server.rs)

```
POST /fire      {"text": "..."}                              → fire_id + output
POST /teach     {"input": "...", "target": "..."}            → fire + 출력 뉴런 pre-자극
POST /feedback  {"fire_id": N, "positive": bool, "strength"} → R-STDP 적용
POST /save
GET  /status                                                  → neurons/synapses/fire_count/threshold/weight_dist
```

- Network는 워커 스레드 단독 소유 (Mutex 없음)
- busy 플래그로 처리 중 상태 표시

## 핵심 상수 일람 (2026-04-13 현재)

### 뉴런 (neuron.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| decay | **0.7** | 막전위 틱당 감쇠 |
| fatigue (발화 시) | **×0.90** | 시냅스 피로도 감소 |
| fatigue 회복 | +0.01/틱 | 매 틱 피로 회복 |
| ABSOLUTE_REFRACTORY | 4 | 흥분성 절대 불응기 |
| ABSOLUTE_REFRACTORY_INH | **2** | 억제성 절대 불응기 (빠른 재발화) |
| RELATIVE_REFRACTORY | 10 | 흥분성 상대 불응기 |
| RELATIVE_REFRACTORY_INH | **4** | 억제성 상대 불응기 |
| excitability 범위 | 0.5 ~ 1.5 | 항상성 clamp 범위 |
| homeostasis 주기 | 10_000틱 | |
| 자발 발화 (자극 중) | 0.000001 | |
| 자발 발화 (idle) | 0.00001 | |

### 네트워크 구조 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| EMOTION_COUNT | 5000 | 감정 뉴런 수 |
| REASON_COUNT | 5000 | 이성 뉴런 수 |
| 억제성 비율 | 30% | `i % 10 >= 7` |
| INITIAL_WEIGHT | 0.2 | 중간→중간 seed 가중치 |
| 입력→중간 seed weight | **0.35** | 시상피질 비율 (내피질의 1.75배) |
| 중간→출력 seed weight | 1.0 | 출력 경로 강도 |
| MIN_WEIGHT | **0.15** | 가지치기 기준 |
| PRUNE_INTERVAL | **5_000** | fire 기준 가지치기 주기 |
| initial threshold | 0.72 | 점진 상승 |
| initial noise_range | 0.2 | |

### STDP / iSTDP (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| STDP A_plus | **0.004** | 동시 발화 LTP |
| STDP A_minus | **0.0048** | LTD (= LTP × 1.2) |
| STDP τ | 20.0 | 지수 감쇠 시간 상수 |
| Hebbian | **제거** | STDP A_plus로 통합 |
| BCM TARGET_RATE | 25.0 | 메타가소성 목표 |
| iSTDP RATE | **0.04** | 억제 시냅스 조정률 |
| iSTDP TARGET_RATE | **10.0** | 과활성 판정 기준 |
| LTP trace 감쇠 | ×0.95/틱 | |
| LTP trace bonus | (trace-1)×0.001 | trace ≥ 2 시 |

### R-STDP

| 상수 | 값 | 설명 |
|------|-----|------|
| eligibility dw | **±0.006** | feedback 시 적용 |
| τ | 20.0 | |

### 수면 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| SLEEP_FIRE_INTERVAL | 60_000 | fire 기준 수면 트리거 |
| SLEEP_TICK_INTERVAL | 500_000 | tick 기준 트리거 |
| SLEEP_DURATION_TICKS | 1_000 | 수면 지속 틱 |
| SLEEP_WEIGHT_SCALE | 0.92 | fatigue 배율 |
| SLEEP_LTD | **-0.001** | 수면 중 weight LTD |

### 축삭발아 (sprout)

| 상수 | 값 | 설명 |
|------|-----|------|
| SPROUT_RADIUS | 5.0 | 탐색 반경 |
| SPROUT_PROBABILITY | 0.1 | 발아 확률 |
| SPROUT_COOLDOWN_TICKS | 500 | 뉴런당 최소 간격 |
| MAX_SPROUT_PER_NEURON | 1 | 틱당 |
| sprout 주기 | 500틱마다 | |
| **발화율 조건** | **1%~5%** | 과활성 뉴런 발아 차단 |

### Fire 내부

| 상수 | 값 | 설명 |
|------|-----|------|
| input potential | 1.0 | 입력 뉴런 자극 강도 |
| input sustain | 4틱 | remaining_sustain |
| silent 완료 기준 | 1틱 | |
| teach pre-자극 | +0.6 | 출력 뉴런 사전 자극 |

## 학습 흐름

```
1. /fire "안녕" → inject_stimulus → tick 루프 → 출력 토큰 조합
2. 평가 (스크립트) → /feedback fire_id positive strength
3. feedback이 eligibility × reward로 weight 업데이트
4. 내부 상태: threshold 점진 상승, fatigue/excitability/항상성,
   주기적 sleep으로 재정규화
5. STDP/iSTDP: 매 틱 비지도 학습 (경로 강화/약화)
```

## 수정 이력 (2026-04-13)

| 항목 | 변경 | 비고 |
|------|------|------|
| 입력→중간 seed weight | 0.2 → **0.35** | 시상피질 비율, 수학적 최소 0.309 기반 |
| fatigue | 0.80 → **0.90** | 과활성 방지 + 전파 균형 |
| STDP LTP dead code | **제거** | 원래 dt>0 분기가 실행 불가능 → 정리 |
| Hebbian | **제거** | STDP A_plus로 통합 (과활성 방지) |
| STDP A_plus/A_minus | 0.015/0.006 → **0.004/0.0048** | LTD:LTP = 1.2:1 |
| R-STDP | ±0.01 → **±0.006** | |
| LTP trace bonus | 0.005 → **0.001** | |
| iSTDP | **신규** | RATE 0.04, TARGET_RATE 10 |
| 억제뉴런 불응기 | 흥분과 동일 → **절대2/상대4** | 빠른 재발화로 지속 억제 |
| sprout 조건 | 무조건 → **발화율 1~5%만** | 과활성 뉴런 발아 차단 |
| MIN_WEIGHT | 0.1 → **0.15** | 공격적 가지치기 |
| PRUNE_INTERVAL | 10_000 → **5_000** | |
| SLEEP_LTD | -0.002 → **-0.001** | |
| status API | **weight_dist 추가** | 10구간 분포 + 평균 |

## 현재 상태

- **안정성**: fire 124k+ 달성, 시냅스 293~301k 안정 유지
- **출력률**: 낮음 (대부분 빈 출력 또는 1~2자)
- **정답률**: 0% (경로 분화 미달)
- **과제**: teach/feedback 전략 개선, LTD:LTP 비율 재조정

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 발화 판정 (par_iter_mut) |
| serde + serde_json | 직렬화 (스냅샷) |
| rand 0.10 | 확률적 선택 |
