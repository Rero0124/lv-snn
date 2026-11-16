# LV-SNN v2 아키텍처

## 개요

뇌의 구역 구조를 모방한 스파이킹 신경망(SNN) 연구 프로젝트 v2.
연속 틱 모델에서 막전위(potential)가 감쇠/누적되며 임계값을 넘으면 발화하고,
R-STDP(reward-modulated STDP)로 경로를 학습한다.
생물학적으로 깨어있는 동안 누적되는 시냅스 총량을 주기적으로 수면을 통해
재정규화(SHY 가설 기반)하며, 수면 중 해마 재생으로 기억을 고정화한다.

## 설계 철학

- **뉴런은 주체**: 발화 결정, 시냅스 생성, 자체 임계값 조절 모두 뉴런이 함
- **Network는 공간 데이터베이스**: 뉴런이 필요할 때 쿼리하는 정보 제공자
- **모든 STDP는 pre→post 인과관계 기반**: incoming 역참조 테이블로 정확한 방향성

## 파일 구조

```
snn/src/
├── main.rs       # 서버/대화형 모드 진입점
├── server.rs     # HTTP API + 단일 워커 스레드
├── network.rs    # Network — 뉴런 배치/틱 루프/수면/쿼리 응답
├── neuron.rs     # Neuron — 막전위, 불응기, 발화, 자가 임계, sprout 결정
├── synapse.rs    # Synapse — weight, fatigue, 빈도 기반 자동 약화
├── region.rs     # RegionType (Input/Emotion/Reason/Memory/Hippocampus/Output)
├── tokenizer.rs  # 한글 자모 분해/재조합
└── bitmap.rs     # 문자 → 32×32 비트맵 (현재 미사용)

scripts/
├── test_words.py   # teach + fire + feedback 학습 스크립트
├── fast_train.py   # 빠른 학습
├── ai_train.py     # Ollama 기반 자율 학습
└── scripts_util.py # 자모 겹침 평가
```

## 네트워크 구조

```
                    입력 뉴런 (80개, 자모 1개당 1뉴런)
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
                   ↓      ↓      ↓
                    출력 뉴런 (80개)
```

### 영역별 구조

- **입력/출력 뉴런**: 자모당 1쌍
- **감정 (5000)**: Main 80% (4000) + Map 20% (1000)
- **이성 (5000)**: Main 80% (4000) + Map 20% (1000)
- **기억 피질 (2000)**: 느린 감쇠 (decay 0.85)
- **해마 (500)**: 입출력 미연결, Map/기억에만 연결, 느린 감쇠

### 연결 구조

- **입력 → 감정_Main, 이성_Main, 기억** (병렬, w=0.4~0.8 랜덤)
- **감정_Main/이성_Main**: 내부 + → Map + → 출력 (w=0.4~0.8 랜덤)
- **기억**: 내부 + → 출력 + → 해마
- **감정_Map/이성_Map ↔ 기억, 해마**
- **해마 → Map 영역, 기억** (역방향)
- **출력 → Main 영역** (피드백 루프: 운동 신호 복귀, w=0.4~0.8 랜덤)

## 뉴런 구조 (neuron.rs)

```rust
Neuron {
    id: NeuronId,
    potential: f64,
    last_spike_tick: Option<u64>,
    synapses: Vec<Synapse>,          // outgoing 시냅스
    incoming: Vec<(usize, usize)>,   // 역참조 (pre_idx, syn_idx)
    x, y: f32,
    inhibitory: bool,
    skip_refractory: bool,           // 입출력 뉴런
    excitability: f64,               // 항상성 가소성
    fire_count_window: u32,
    decay_rate: f64,                 // 기본 0.7, 기억/해마 0.85
    threshold_scale: f64,            // 기본 1.0, Map 1.7
    self_threshold: f64,             // 자가 적응 임계값 (기본 0.5, 입출력 0.3)
}
```

### 뉴런의 주체적 동작

1. **발화 판정** (`try_fire`)
   - 자기 self_threshold로 결정 (`potential ≥ self_threshold × threshold_scale / excitability`)
   - 발화 시 self_threshold +0.001 (최대 1.0)
   - 신호 받았는데 미발화 시 -0.001 (최소 0.5 / 입출력 0.3)

2. **Sprout 자체 결정** (`wants_sprout`)
   - 발화 시 확률 0.001로 sprout 욕구 결정
   - 욕구 발생 시 Network에 근처 뉴런 쿼리
   - 받은 후보에서 자기가 선택해서 `add_synapse` 직접 생성

3. **불응기** (억제/흥분 분리) — 반감
   - 흥분성: 절대 2틱, 상대 5틱
   - 억제성: 절대 1틱, 상대 2틱 (빠른 재발화로 지속 억제)

4. **항상성 가소성** (10,000틱마다)
   - fire_rate > 5% → excitability × 0.95
   - fire_rate < 1% → excitability × 1.05
   - 범위 0.5~1.5

## 시냅스 구조 (synapse.rs)

```rust
Synapse {
    target: NeuronId,
    weight: f64,
    seed: bool,
    fatigue: f64,
    ltp_trace: f64,
    last_used_tick: u64,
    recent_rate: f64,               // 빈도 기반 자동 약화용
}
```

### 시냅스 자동 조정

- **fire_fatigue** (발화 시): `fatigue *= 0.90`, `recent_rate += 0.1`
  - 추가로 `weight -= 0.01 × recent_rate` (자주 쓸수록 약화)
- **recover** (매 틱): `fatigue += 0.01`, `ltp_trace × 0.95`, `recent_rate × 0.99`

## 틱 루프

```
매 틱:
  1. global_tick += 1
  2. 활성 자극 입력 뉴런 sustain (4틱)
  3. 모든 뉴런 decay
  4. 병렬 발화 판정 (try_fire)
  5. 신호 전달
  6. STDP (Pass 1: pre 기준, Pass 2: post incoming)
  7. iSTDP (Pass 1: 억제성 pre outgoing, Pass 2: post incoming 억제 시냅스)
  8. recent_spikes 기록
  9. 뉴런 주도 sprout (발화 뉴런이 자체 결정 + Network 쿼리)
  10. silent_ticks 업데이트 → 10틱 무발화 시 자극 완료
  11. 완료된 자극 처리 (eligibility 계산, output_spikes 시간순 토큰화)
  12. 항상성 가소성 (10k 틱마다)
```

**자극 종료 조건**: silent_ticks ≥ 10 (한 틱 조용해도 계속 관찰, 10틱 무발화 시 종료)
**STDP 수집**: elapsed_ticks 제한 없음 (모든 발화 STDP 대상)
**출력**: output_spikes를 tick 시간순으로 정렬 → 중복 제거 없이 발화 순서대로 토큰 생성

## 학습 메커니즘

### 일반 STDP (tick 루프 내, 비지도)

**Pass 1 (pre 기준, dt ≤ 0)**:
```
pre 발화 시점 → 각 outgoing 시냅스에 대해
post의 과거 발화 확인:
  dt == 0: 동시 발화 LTP
  dt < 0: post 먼저 발화 → LTD
```

**Pass 2 (post incoming, dt > 0)**:
```
post 발화 시점 → 뉴런의 incoming 역참조로
과거 발화한 pre 뉴런 찾기:
  dt > 0: pre 먼저, post 나중 → 진짜 LTP
```

**양봉 가우시안 soft bound** (모든 STDP 계열 통일):
- **LTP**: peak 0.2 (약한 시냅스 강화 잘 됨)
- **LTD**: peak 1.0 (강한 시냅스 약화 잘 됨)
- σ = 0.4

**BCM 메타가소성 (post 기준)**:
- post 자주 발화 → LTP 어려움, LTD 쉬움
- post 드물게 발화 → LTP 쉬움, LTD 어려움

### iSTDP (억제성 시냅스) — spike-timing + Gaussian

일반 STDP와 동일한 구조. rate-based에서 timing-based로 전환.

**Pass 1 (pre 기준)**: 억제 pre 발화 시, outgoing post 최근 발화 확인
- dt == 0 또는 -50 < dt < 0: **LTD** (post 먼저/동시 → 억제 타이밍 늦음)
- Gaussian LTD peak w=1.0

**Pass 2 (post 기준, incoming 역참조)**: post 발화 시, 억제 pre 최근 발화 확인
- dt > 0, dt < 50: **LTP** (pre 먼저, post 발화 = 억제 부족 → 강화)
- Gaussian LTP peak w=0.2

### R-STDP (feedback 지도학습) — Gaussian 적용

**eligibility trace** (fire 중 계산, incoming 역참조 사용):
- Pass 1 (pre 기준, dt<0): LTD = -0.0144 × exp(-|dt|/20) × f_ltd (peak w=1.0)
- Pass 2 (post incoming, dt>0): LTP = +0.0288 × exp(-|dt|/20) × f_ltp (peak w=0.2)
- LTP:LTD = 2:1

**feedback 시**:
- `change = dw × reward`
- reward = +strength (positive) 또는 -strength (negative)

## 수면 메커니즘

### 트리거
```rust
fires_since_sleep >= 30_000
|| global_tick - last_sleep_tick >= 500_000
```

### 동작

1. **시냅스 fatigue × 0.92 + weight × 0.95** (SHY 균일 다운스케일)
2. 전역 흥분 상태 리셋 (potential=0, excitability=1.0)
3. **해마 재생** (5회 × 10틱):
   - 해마 뉴런 자극 (potential=0.5)
   - 전파 + STDP (Pass 1 + Pass 2)
   - 기억 피질 경로 고정화
4. 휴지기 1000틱

## Sprout (뉴런 주도)

### 흐름
1. 뉴런이 발화 시 `wants_sprout()` 자체 결정 (확률 0.001)
2. 욕구 발생 시 Network에 `query_nearby_neurons()` 호출
3. Network: 반경 5.0 내 후보 리스트 반환 (위치 정보 제공자)
4. 뉴런: 받은 후보 중 선택 → `add_synapse` 직접 생성
5. Network: incoming 역참조 동기화 (뉴런이 만든 시냅스 인덱스 등록)

### 제약
- 입력↔입력, 출력↔출력, 입력↔출력 연결 차단
- MAX_SPROUT_PER_NEURON: 1 (발화당 최대 1개)
- 쿨다운/발화율 조건 없음 (순수 확률)

## HTTP API

```
POST /fire      {"text": "..."}                              → fire_id + output
POST /teach     {"input": "...", "target": "..."}            → fire + 출력 뉴런 pre-자극(+0.6)
POST /feedback  {"fire_id": N, "positive": bool, "strength"} → R-STDP 적용
POST /save
GET  /status    → neurons/synapses/fire_count/threshold/weight_dist/regions
```

## 핵심 상수 일람 (2026-04-17 현재)

### 뉴런 (neuron.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| decay_rate (기본) | 0.7 | 막전위 감쇠 |
| decay_rate (기억/해마) | 0.85 | 느린 감쇠 |
| threshold_scale (기본) | 1.0 | |
| threshold_scale (Map) | 1.7 | 실효 임계값 0.85 |
| self_threshold 초기 (기본) | 0.5 | |
| self_threshold 초기 (입출력) | 0.3 | |
| self_threshold 범위 (기본) | 0.5~1.0 | |
| self_threshold 범위 (입출력) | 0.3~1.0 | |
| self_threshold 조정 | ±0.001 | 발화/미발화 시 |
| fatigue (발화) | ×0.90 | |
| fatigue 회복 | +0.01/틱 | |
| ABSOLUTE_REFRACTORY (흥분/억제) | 2 / 1 | 반감 |
| RELATIVE_REFRACTORY (흥분/억제) | 5 / 2 | 반감 |
| SPROUT_PROBABILITY | 0.001 | 발화 시 sprout 확률 |
| SPROUT_RADIUS | 5.0 | |
| SPROUT_WEIGHT | 0.2 | |

### 시냅스 (synapse.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| fire_fatigue 배율 | ×0.90 | |
| recent_rate 증가 | +0.1 | 발화 시 |
| recent_rate 감쇠 | ×0.99/틱 | |
| 빈도 기반 약화 | weight -= 0.01 × recent_rate | |

### 네트워크 구조 (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| EMOTION_COUNT | 5000 | Main 4000 + Map 1000 |
| REASON_COUNT | 5000 | Main 4000 + Map 1000 |
| MEMORY_COUNT | 2000 | |
| HIPPOCAMPUS_COUNT | 500 | |
| 억제 비율 (감정/이성/기억) | 30% | |
| 억제 비율 (해마) | 20% | |
| INITIAL_WEIGHT | 0.2 | 중간→중간 seed |
| seed weight 랜덤 | 0.4~0.8 | 입력→중간, 중간→출력 등 |
| MIN_WEIGHT | 0.1 | prune 기준 |
| PRUNE_INTERVAL | 5000 fire | |
| initial threshold (전역, 미사용) | 0.50 | self_threshold 사용 |

### STDP (network.rs)

| 상수 | 값 | 설명 |
|------|-----|------|
| STDP_A_PLUS_BASE | 0.006 | LTP |
| STDP_A_MINUS_BASE | 0.003 | LTD — LTP:LTD = 2:1 |
| STDP_TAU | 20.0 | |
| BCM_TARGET_RATE | 25.0 | post 기준 |
| LTP peak (가우시안) | 0.2 | |
| LTD peak (가우시안) | 1.0 | |
| 가우시안 σ | 0.4 | |

### iSTDP (spike-timing 전환)

| 상수 | 값 | 설명 |
|------|-----|------|
| ISTDP_A_PLUS | 0.024 | post 발화 시 억제 pre LTP |
| ISTDP_A_MINUS | 0.012 | pre 억제 발화 시 LTD — 2:1 |
| ISTDP_TAU | 20.0 | |
| LTP peak | 0.2 | |
| LTD peak | 1.0 | |

### R-STDP

| 상수 | 값 | 설명 |
|------|-----|------|
| LTP | +0.0288 | Pass 2 (post incoming) |
| LTD | -0.0144 | Pass 1 (pre) — 2:1 |
| LTP peak | 0.2 | Gaussian |
| LTD peak | 1.0 | Gaussian |

### 수면

| 상수 | 값 | 설명 |
|------|-----|------|
| SLEEP_FIRE_INTERVAL | 30_000 | |
| SLEEP_TICK_INTERVAL | 500_000 | |
| SLEEP_DURATION_TICKS | 1_000 | |
| SLEEP_WEIGHT_SCALE | ×0.92 | fatigue |
| SLEEP_WEIGHT_DECAY | ×0.95 | weight 균일 |
| REPLAY_ROUNDS | 5 | 해마 재생 횟수 |
| REPLAY_TICKS | 10 | 재생당 전파 틱 |

## 수정 이력 (2026-04-17)

### 핵심 구조 변경
- **뉴런 주체성 강화**: wants_sprout(), sprout_from_candidates() — 뉴런이 자체 결정/생성
- **Network는 쿼리 응답자**: query_nearby_neurons() — 위치 정보 제공
- **뉴런 incoming 역참조**: 각 뉴런이 자기 incoming 시냅스 인덱스 소유
- **self_threshold**: 각 뉴런이 자가 적응 임계값 관리 (전역 threshold 대체)

### STDP 전면 통일 (timing + Gaussian)
- **모든 STDP 계열(Regular/Sleep Replay/R-STDP/iSTDP)에 Pass 2 추가**: incoming 역참조로 진짜 pre→post LTP
- **iSTDP를 rate-based → spike-timing 전환**: 타이밍/가우시안 soft-bound 적용
- **LTD peak 통일**: 0.8 → **1.0** (모든 STDP 계열 통일 — 강한 시냅스 약화 우세)
- **LTP peak 0.2 유지**: 약한 시냅스 강화 우세
- **비율 통일**: **LTP:LTD = 2:1** (모든 계열)
- **BCM을 post 기준**: 표준 BCM (post 활동 ↑ → LTP 어려움, LTD 쉬움)

### 네트워크 구조 개선
- **출력 → Main 피드백 루프 추가**: 출력 뉴런이 감정_Main/이성_Main/기억으로 seed 시냅스 생성 (운동 신호 복귀)
- **불응기 반감**: 흥분 4/10 → 2/5, 억제 2/4 → 1/2 (더 많은 재발화 기회)

### 시뮬레이션/출력 로직
- **자극 종료 조건 완화**: silent_ticks ≥ 1 → **10** (한 틱 조용해도 유지)
- **STDP 수집 틱 제한 제거**: `elapsed_ticks <= 14` 제거 → 모든 발화 대상
- **출력 중복 제거 로직 제거**: seen_seqs HashSet 제거 → 발화 시간순 토큰 출력 (반복 허용)

### 학습 스크립트 (test_words.py)
- **출력 있음**: 기본 강화 0.7 + 유사도 보너스 최대 +0.35 = **최대 1.0**
- **정답**: 1.0
- **무출력**: negative 0.3 (LTD)

### 기존 수정 유지
- **sprout 주기 제거**: 500틱 조건 없음, 발화 이벤트마다 확률 체크
- **수면 weight scaling**: -0.001 차감 → ×0.95 균일 (SHY)
- **시냅스 빈도 기반 자동 약화**: recent_rate × 0.01

## 현재 상태

### 순수 STDP 테스트 (feedback 없이 "하이" 50k fire)
- 출력률 13~15% 안정 유지
- weight 분포: 0.1-0.2 (80%) + 중간 구간 + 0.9-1.0 (<1%)
- 양극화 해결됨

### 미해결
- R-STDP가 경로를 부수는 문제 (학습 시 출력 죽음)
- 정답률 0~0.4%

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 발화 판정 |
| serde + serde_json | 직렬화 |
| rand 0.10 | 확률적 선택 |
| fontdue | 비트맵 렌더링 (미사용) |
