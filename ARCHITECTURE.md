# LV-SNN v2 아키텍처

## 개요

뇌의 구역 구조를 모방한 스파이킹 신경망(SNN) 연구 프로젝트 v2.
연속 틱 모델에서 막전위(potential)가 감쇠/누적되며 임계값을 넘으면 발화하고,
R-STDP(reward-modulated STDP)로 경로를 학습한다.
생물학적으로 깨어있는 동안 누적되는 시냅스 총량을 주기적으로 수면을 통해
재정규화(SHY 가설 기반)하며, 수면 중 해마 재생으로 기억을 고정화한다.

## 파일 구조

```
snn/src/
├── main.rs       # 서버/대화형 모드 진입점 (--serve, --port, --data)
├── server.rs     # HTTP API + 단일 워커 스레드 기반 순차 처리
├── network.rs    # Network — 뉴런/시냅스/틱 루프/수면/R-STDP/축삭발아
├── neuron.rs     # Neuron — 막전위, 불응기, 발화 판정, 항상성 가소성
├── synapse.rs    # Synapse — weight, 피로도 fatigue, seed 플래그
├── region.rs     # RegionType (Input/Emotion/Reason/Memory/Hippocampus/Output)
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

## 네트워크 구조

```
                    입력 뉴런 (80개)
                   /      |      \
                  ↓       ↓       ↓
         감정_Main     이성_Main    기억
         (4000)       (4000)     (2000)
           ↕             ↕         ↕
         감정_Map      이성_Map     │
         (1000)       (1000)      │
           ↕     ↕     ↕          │
           └─────해마(500)────────┘
                   (입출력 미연결)
                  \      |      /
         감정_Main  이성_Main  기억
                   ↓      ↓      ↓
                    출력 뉴런 (80개)
```

### 영역 구조

- **입력/출력 뉴런**: 어휘 1개당 1쌍 (자모 80개 기본)
- **감정 (5000)**: Main 80% (4000) + Map 20% (1000)
- **이성 (5000)**: Main 80% (4000) + Map 20% (1000)
- **기억 피질 (2000)**: 분리 없음, 느린 감쇠 (decay 0.85)
- **해마 (500)**: 입출력 미연결, Map/기억에만 연결, 느린 감쇠 (decay 0.85)

### 연결 구조

- **입력 → 감정_Main, 이성_Main, 기억** (병렬, w=0.35)
- **감정_Main**: 내부 50% + → 감정_Map 20% + → 출력 30% (w=1.0)
- **이성_Main**: 내부 50% + → 이성_Map 20% + → 출력 30% (w=1.0)
- **기억**: 내부 50% + → 출력 20% + → 해마 30%
- **감정_Map ↔ 기억, 해마** (40% 역방향 Main + 60% 기억/해마)
- **이성_Map ↔ 기억, 해마** (40% 역방향 Main + 60% 기억/해마)
- **해마 → Map 영역, 기억** (역방향)

### Map 영역의 역할

- threshold_scale = 1.7 (실효 임계값 0.85) → Main보다 늦게 발화
- Main 영역의 활동을 요약해서 기억/해마로 전달하는 게이트 역할
- 기억/해마에서 돌아오는 정보를 Main으로 전달

## 뉴런 구조 (neuron.rs)

```rust
Neuron {
    id: NeuronId,
    potential: f64,                  // 막전위 (매 틱 감쇠)
    last_spike_tick: Option<u64>,    // 불응기 판정용
    synapses: Vec<Synapse>,          // 나가는 시냅스
    x, y: f32,                       // 2D 위치 (축삭발아 거리 계산)
    inhibitory: bool,                // 억제성 뉴런이면 weight × -1
    skip_refractory: bool,           // 입출력 뉴런은 불응기 스킵
    excitability: f64,               // 항상성 가소성 (0.5~1.5, 기본 1.0)
    fire_count_window: u32,          // 최근 1만틱 발화 횟수
    decay_rate: f64,                 // 막전위 감쇠율 (기본 0.7, 기억/해마 0.85)
    threshold_scale: f64,            // 임계값 배율 (기본 1.0, Map 1.7)
}
```

### 발화 판정 (try_fire)

1. **불응기 체크** (억제/흥분 분리)
   - 흥분성: 절대 4틱, 상대 10틱
   - 억제성: 절대 2틱, 상대 4틱 (빠른 재발화로 지속 억제)
   - 상대 불응기 중: 임계값 × 2 필요
2. **자발적 발화** — 자극 중 `0.000001`, idle 중 `0.00001`
3. **판정**: `potential + noise ≥ threshold × threshold_scale / excitability`
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
    seed: bool,            // 초기 시냅스 플래그
    fatigue: f64,          // 0.0~1.0 — 발화 시 × 0.90, 매 틱 +0.01 회복
    ltp_trace: f64,        // LTP 누적 카운터
}

effective_weight = weight × fatigue
```

## 틱 루프 (network.rs — tick / idle_tick)

```
매 틱:
  1. global_tick += 1
  2. 활성 자극의 입력 뉴런 sustain (remaining_sustain 동안 potential=1.0)
  3. 모든 뉴런 decay (potential *= decay_rate)
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

## 학습 메커니즘

### STDP (tick 루프 내, 비지도)

```
동시 발화 (dt == 0):
  dw = +A_plus (0.004) × BCM_scale + LTP_trace_bonus
  → 동시 활성 경로 강화

LTD (dt < 0, post가 먼저 발화):
  dw = -A_minus (0.0048) × (2 - BCM_scale) × exp(dt/τ)
  → 비인과적 연결 약화
  → LTD:LTP = 1.2:1 (약화 우세)
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
→ 과활성 뉴런: LTP ↓, LTD ↑
→ 저활성 뉴런: LTP ↑, LTD ↓
```

### LTP 누적 (synapse.rs)

```
발화 시 ltp_trace += 1.0, 매 틱 trace *= 0.95
trace ≥ 2: bonus = (trace - 1) × 0.001
```

## 수면 메커니즘 (network.rs — enter_sleep)

### 트리거

```rust
fires_since_sleep >= 30_000
|| global_tick - last_sleep_tick >= 500_000
```

### 동작

1. 모든 시냅스 `fatigue *= 0.92` + `weight -= 0.001`
2. 모든 뉴런 `potential=0`, `excitability=1.0`, `fire_count_window=0`
3. **해마 재생** (5회 × 10틱):
   - 해마 뉴런 전체를 0.5로 자극
   - 10틱 동안 발화+전달 (STDP 없이)
   - 시냅스를 통해 기억 피질/Map 영역으로 자연 전파
   - 라운드 후 potential 리셋
4. 휴지기: 1,000틱 순수 감쇠

## 축삭발아 (network.rs — sprout)

500틱마다 최근 발화 뉴런 기준:
- **발화율 조건**: fire_rate 1%~5% 범위 뉴런만 허용
- SPROUT_RADIUS=5.0 내 후보 중 가까운 순
- SPROUT_PROBABILITY=0.1
- SPROUT_COOLDOWN_TICKS=500
- MAX_SPROUT_PER_NEURON=1

## Threshold 점진 상승

```
1만 fire마다:
  threshold = min(1.0, 0.50 + (fire_id / 10_000) × 0.001)
  noise_range = max(0.1, 0.2 - (fire_id / 10_000) × 0.001)
```

## 가지치기

PRUNE_INTERVAL(5_000) fire마다:
- `weight < MIN_WEIGHT(0.1)` 시냅스 제거

## HTTP API (server.rs)

```
POST /fire      {"text": "..."}                              → fire_id + output
POST /teach     {"input": "...", "target": "..."}            → fire + 출력 뉴런 pre-자극(+0.6)
POST /feedback  {"fire_id": N, "positive": bool, "strength"} → R-STDP 적용
POST /save
GET  /status    → neurons/synapses/fire_count/threshold/weight_dist/regions
```

## 핵심 상수 일람 (2026-04-14 현재)

### 뉴런 (neuron.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| decay_rate (기본) | 0.7 | 막전위 틱당 감쇠 |
| decay_rate (기억/해마) | **0.85** | 느린 감쇠 |
| threshold_scale (기본) | 1.0 | |
| threshold_scale (Map) | **1.7** | 실효 임계값 0.85 |
| fatigue (발화 시) | **×0.90** | |
| fatigue 회복 | +0.01/틱 | |
| ABSOLUTE_REFRACTORY | 4 | 흥분성 |
| ABSOLUTE_REFRACTORY_INH | **2** | 억제성 (빠른 재발화) |
| RELATIVE_REFRACTORY | 10 | 흥분성 |
| RELATIVE_REFRACTORY_INH | **4** | 억제성 |
| excitability 범위 | 0.5~1.5 | |
| homeostasis 주기 | 10_000틱 | |
| 자발 발화 (자극/idle) | 0.000001 / 0.00001 | |

### 네트워크 구조 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| EMOTION_COUNT | 5000 | Main 4000 + Map 1000 |
| REASON_COUNT | 5000 | Main 4000 + Map 1000 |
| MEMORY_COUNT | **2000** | 기억 피질 |
| HIPPOCAMPUS_COUNT | **500** | 해마 |
| 억제 비율 (감정/이성/기억) | 30% | |
| 억제 비율 (해마) | 20% | |
| INITIAL_WEIGHT | 0.2 | 중간→중간 |
| 입력→중간 seed | **0.35** | |
| 중간→출력 seed | 1.0 | |
| seed 수 (입력→중간) | 10개 | |
| seed 수 (중간 내부) | **10개** | |
| MIN_WEIGHT | 0.1 | |
| PRUNE_INTERVAL | **5_000** | |
| initial threshold | **0.50** | |
| initial noise_range | 0.2 | |
| input potential | 1.0 | |
| input sustain | 4틱 | |
| teach pre-자극 | +0.6 | |

### STDP / iSTDP (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| STDP A_plus | **0.004** | |
| STDP A_minus | **0.0048** | LTD:LTP = 1.2:1 |
| STDP τ | 20.0 | |
| Hebbian | **제거** | STDP A_plus로 통합 |
| BCM TARGET_RATE | 25.0 | |
| iSTDP RATE | **0.04** | |
| iSTDP TARGET_RATE | **10.0** | |
| LTP trace 감쇠 | ×0.95/틱 | |
| LTP trace bonus | (trace-1)×0.001 | |

### R-STDP

| 상수 | 값 | 설명 |
|------|-----|------|
| eligibility dw | **±0.006** | |
| τ | 20.0 | |

### 수면 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| SLEEP_FIRE_INTERVAL | **30_000** | |
| SLEEP_TICK_INTERVAL | 500_000 | |
| SLEEP_DURATION_TICKS | 1_000 | |
| SLEEP_WEIGHT_SCALE | 0.92 | fatigue 배율 |
| SLEEP_LTD | **-0.001** | |
| REPLAY_ROUNDS | **5** | 해마 재생 횟수 |
| REPLAY_TICKS | **10** | 재생 당 전파 틱 |

### 축삭발아

| 상수 | 값 | 설명 |
|------|-----|------|
| SPROUT_RADIUS | 5.0 | |
| SPROUT_PROBABILITY | 0.1 | |
| SPROUT_COOLDOWN_TICKS | 500 | |
| MAX_SPROUT_PER_NEURON | 1 | |
| sprout 주기 | 500틱마다 | |
| 발화율 조건 | **1%~5%** | 과활성 뉴런 발아 차단 |

## 현재 상태 (2026-04-14)

### 안정성

- fire 124k+ 달성 (LTD:LTP 1.2:1, threshold 0.72, 이전 구조)
- 새 구조(기억/해마/Map)에서 fire 54k+ 달성
- 시냅스 122k 안정 유지

### 미해결 과제

1. **학습된 경로가 LTD+prune으로 잘림**: 반복 학습한 단어의 경로가 소멸 → 출력 0%
   - 새 단어는 seed 시냅스가 온전해서 출력 100%
   - 학습된 단어만 출력 안 됨 → prune이 경로 중간을 자르는 것이 원인
2. **경로 보호 메커니즘 필요**: 자주 사용되는 경로를 prune/LTD에서 보호
3. **정답률 0%**: 경로 분화 미달

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 발화 판정 (par_iter_mut) |
| serde + serde_json | 직렬화 (스냅샷) |
| rand 0.10 | 확률적 선택 |
