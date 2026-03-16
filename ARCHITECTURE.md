# LV-SNN 아키텍처

## 개요

뇌의 구역 구조를 모방한 스파이킹 신경망(SNN) 연구 프로젝트.
텍스트를 다중 토큰(원문, 단어, n-gram, 문자, 자모)으로 분해하여 시냅스에 저장하고,
신호가 구역 사이를 틱 단위로 전파되며, 저장된 토큰 기반으로 응답을 조합한다.
STDP(Spike-Timing-Dependent Plasticity)로 인과적 시냅스를 강화/약화하고,
해마가 뉴런 단위 경로 패턴을 기억 구역에 저장한다.
HTTP API 서버로 동작하며, Ollama 기반 자율 탐색 학습을 지원한다.

## 파일 구조

```
src/
├── main.rs          # 서버/대화형 모드 진입점 (--serve, --port)
├── server.rs        # actix-web HTTP API (fire, teach, feedback, feedback_partial, status, save)
├── tokenizer.rs     # 텍스트 → 다중 토큰 분해 + 결정적 해시 + 자모 재조합 (compose_jamo)
├── region.rs        # RegionType 정의 (Input, Emotion, Reason, Storage, Output), 구역별 연결 규칙
├── neuron.rs        # Neuron 구조체 (확률적 발화(시그모이드), TOP_K 제한, STDP spike 타이밍)
├── synapse.rs       # Synapse + SynapseStore (캐시 + redb 2단 저장 + 토큰 역인덱스 + 프루닝)
├── hippocampus.rs   # 해마 (뉴런 단위 경로 패턴 추적 → 기억 구역 통합, 전달 후 해마에서 삭제)
└── network.rs       # Network (STDP, 발화, 피드백, teach, 쿨다운, 출력 게이트, 패턴 병합, 응답 조합)

scripts/
├── ai_train.py      # 자율 탐색 학습 (Ollama 입력 생성 → SNN 탐색 → O/X 판정, 최대 5회 재시도)
└── fast_train.py    # 빠른 학습 (로컬 1:N 데이터, Ollama 의미 평가 선택)

data/
├── network.redb           # 시냅스 + 네트워크 상태 영속 저장
└── conversation_multi.json # 1:N 학습용 대화 쌍 (59입력 × ~4응답)
```

## 구역 설계 (region.rs)

```
         ┌──────────────────────────────────────────┐
         │              네트워크                      │
         │                                          │
         │   ┌─────┐                                │
         │   │입력 │──┬──→ 감정(0.5) ←──→ 이성(0.7) │
         │   │(128)│  │      ↕             ↕        │
         │   └─────┘  └──→ 기억(128) ──→ 이성       │
         │                   ↕             ↕        │
         │                 출력(64) ←──────┘        │
         │                   ↕                      │
         │              (게이트 닫히면 소멸)           │
         │                                          │
         │   ┌──────┐                               │
         │   │ 해마  │ ← 뉴런 단위 발화 경로 관찰      │
         │   │(패턴) │ → 기억 구역에 패턴 저장         │
         │   │      │   전달 후 해마에서 삭제          │
         │   └──────┘                               │
         └──────────────────────────────────────────┘
```

### 구역 연결 규칙 (region.rs targets())

| 구역 | 연결 대상 | 뉴런 수 | 임계값 |
|------|----------|---------|--------|
| Input | Emotion, Reason, Storage | 128 | 0.5 |
| Emotion | Reason, Storage, Output | 64 | 0.5 |
| Reason | Emotion, Storage, Output | 64 | 0.7 |
| Storage | Emotion, Reason, Output | 128 | 0.5 |
| Output | (나가는 연결 없음) | 64 | 1.0 (게이트) |

## 토큰 시스템 (tokenizer.rs)

### 토큰화

입력 텍스트를 여러 단위로 분해하여 다중 시냅스로 저장:

```
"좋은 아침" → {
    original: "좋은 아침"                    (Original, bonus 0.8x)
    words:    ["좋은", "아침"]               (Word, bonus 2.0x)
    bigrams:  ["좋은", "은 ", " 아", "아침"]   (NGram(2), bonus 1.1x)
    trigrams: ["좋은 ", "은 아", " 아침"]      (NGram(3), bonus 1.0x)
    fourgrams: ["좋은 아", "은 아침"]          (NGram(4), bonus 0.9x)
    chars:    ["좋", "은", "아", "침"]        (Char, bonus 1.3x)
    jamo:     ["ㅈ", "ㅗ", "ㅎ", "ㅡ", ...]   (Jamo, bonus 1.5x)
}
```

### 자모 재조합 (compose_jamo)

자모 배열을 초성+중성(+종성) 패턴 인식으로 한글 음절로 합침:
- 초성 19자 × 중성 21자 × 종성 28자 → 유니코드 0xAC00 기반 계산
- 종성 뒤에 중성이 오면 종성이 아니라 다음 초성으로 판단

## 확률적 뉴런 발화 모델 (neuron.rs)

시그모이드 함수를 사용한 확률적 발화. 결정적 threshold 대신 확률로 판정:

```
뉴런 발화 확률:
  p = sigmoid(activation - threshold, temperature=0.15)
    = 1 / (1 + exp(-(activation - threshold) / 0.15))

  activation >> threshold → p ≈ 1.0 (거의 확실히 발화)
  activation ≈  threshold → p = 0.5 (50% 확률)
  activation << threshold → p ≈ 0.0 (거의 발화 안 함)

시냅스별 전달 확률:
  forward = activation × weight × discount
  p_syn = sigmoid(forward - threshold × 0.5, temperature=0.15)
  → 각 시냅스가 독립적으로 확률적 전달 여부 결정

발산 구간 (기존 유지):
  activation > threshold × DIVERGE_RATIO(1.5) → 신호 × DIVERGE_BOOST(1.3) 증폭
```

- 같은 입력이라도 매번 다른 시냅스 조합이 활성화 → 응답 다양성
- 약한 시냅스도 낮은 확률로 발화 가능 → 탐색 능력
- 강한 시냅스는 거의 항상 발화 → 학습된 경로 안정성
- 각 뉴런은 상위 TOP_K(10)개 시냅스만 발화 (신호 폭발 방지)
- 이미 출력된 토큰의 시냅스는 forward × 0.5 감소 (중복 억제)
- DB에서 로드한 시냅스는 DB_WEIGHT_DISCOUNT(0.5) 적용

## STDP (network.rs)

생물학적 시냅스 가소성 규칙. 뉴런 발화 타이밍 기반 가중치 조정:

```
Δt = t_post - t_pre

Δt > 0 (인과: pre → post 순서):  Δw = +A_plus(0.02) × exp(-Δt / τ(10))   → 강화 (LTP)
Δt < 0 (반인과):                 Δw = -A_minus(0.025) × exp(Δt / τ(10))   → 약화 (LTD)
Δt = 0 (동시):                   Δw = +A_plus × 0.5                       → 약한 강화
```

## 출력 게이트 (network.rs)

Output 뉴런 임계값을 OUTPUT_THRESHOLD(1.0)로 높게 설정하여 초기 잡음 방지:
1. 틱 실행 중 Output 구역에 충분한 신호 축적 확인 (output_ready)
2. GATE_TICKS(5)틱간 게이트 열림 → 임계값을 DEFAULT_THRESHOLD(0.5)로 낮춤
3. 게이트 닫힘 → 임계값 복원 + 활성값 리셋

## 핵심 흐름

### 1. 발화 (fire)

```
입력 텍스트 → 토큰화 (원문, 단어, bigram, trigram, 4gram, 문자, 자모)
           → Input/Storage 시냅스 저장
           → 해시 기반 입력 뉴런 활성화
           → 틱 실행 (최대 MAX_TICKS(50)틱):
             - 확률적 뉴런 발화 (시그모이드) + 발산 증폭
             - TOP_K(10) 제한
             - STDP 가중치 조정 (spike 타이밍 기반)
             - 쿨다운 적용
             - 출력 게이트 메커니즘
           → compose_response (4층 응답 조합)
           → 해마 경로 기록 + maybe_consolidate
           → 패턴 병합 (consolidate_patterns)
           → 쿨다운 기록 (출력 시냅스 1.0, 경로 시냅스 0.5)
```

### 2. 피드백 (feedback)

```
positive=true:
  출력 시냅스: weight + strength × FEEDBACK_LR(0.3)

positive=false:
  출력 시냅스: weight - strength × FEEDBACK_LR(0.3) × 1.5  (강하게 약화)
  경로 시냅스: weight - strength × FEEDBACK_LR(0.3) × 0.5  (다른 경로 탐색 유도)
```

### 3. 부분 피드백 (feedback_partial)

토큰별 개별 점수를 받아 시냅스 세밀 조정:
- token_scores: [("단어", 점수)] — 양수=강화, 음수=약화
- Original 토큰은 모든 점수 평균 적용

### 4. Teach (목표 기반 학습)

```
입력/목표 토큰화 → 시냅스 저장 (Input→Storage, Storage/Emotion/Reason→Output)
               → 조용한 발화 (teach_fire) → 출력 + 경로 길이 수집
               → 점수 계산 → 강화/약화
```

응답 시냅스 저장 시 구역별 다른 가중치:
- Storage → Output: INITIAL_WEIGHT × 1.5 (가장 강함)
- Reason → Output: INITIAL_WEIGHT × 1.2
- Emotion → Output: INITIAL_WEIGHT × 1.0

### 5. 응답 조합 (compose_response)

```
수집된 출력 토큰에 점수 부여:
  score = 시냅스 가중치 × 토큰 길이 보너스 × 입력 관련성 보너스(2.0x)
  가중치 내림차순 정렬 (동점시 먼저 도착한 것 우선)

1층: Original 토큰 (weight ≥ INITIAL_WEIGHT) → 그대로 반환
2층: Word 토큰 상위 8개를 공백으로 연결
3층: Char 토큰 상위 15개를 연결
4층: Jamo 토큰 상위 30개를 compose_jamo로 한글 음절 재조합
폴백: 가중치 상위 3개 토큰 연결
```

### 6. 패턴 병합 (consolidate_patterns)

연속 출력된 토큰을 하나의 시냅스로 통합:
- 대상: Jamo+Jamo, Jamo+Char, Char+Word 등 조합
- 같은 틱 또는 2틱 이내 연속 출현한 토큰 쌍
- 기존 병합 시냅스가 있으면 강화, 없으면 새로 생성 (Storage→Output)

### 7. 해마 통합 (hippocampus.rs)

```
매 발화: 뉴런 경로에서 길이 3 서브패턴 추출 → 빈도 카운트
매 CONSOLIDATION_INTERVAL(5)회: 빈도 ≥ MIN_PATTERN_FREQ(3) 패턴 → 기억 구역 저장
  → 전달된 패턴은 해마에서 삭제 (중복 보유 방지)
  → 나머지 카운터 /= 2 감쇠
```

### 8. 시냅스 프루닝

```
매 10회 발화: 캐시 프루닝 (약한 시냅스 + 중복 시냅스 제거)
저장 시: DB 전체 프루닝 (weight ≤ MIN_WEIGHT 제거, 동일 pre→post+token 중복 제거)
```

## 쿨다운 (최근 사용 억제)

```
fire 시작: 모든 쿨다운 × COOLDOWN_DECAY(0.7)
fire 종료: 출력 시냅스 쿨다운 = 1.0, 경로 시냅스 쿨다운 = 0.5
run_tick: forward_val × (1.0 - cooldown × COOLDOWN_PENALTY(0.15))
```

## HTTP API (server.rs)

```
POST /fire              {"text": "..."}                                    → 발화 + 응답
POST /teach             {"input": "...", "target": "..."}                  → 목표 학습
POST /feedback          {"fire_id": N, "positive": bool, "strength": F}    → 피드백
POST /feedback_partial  {"fire_id": N, "token_scores": [["단어", F], ...]} → 부분 피드백
GET  /status                                                                → 상태
POST /save                                                                  → DB 저장
```

- 서버 설정: keep_alive 300s, client_request_timeout 300s
- /status는 Mutex 없이 AtomicUsize/AtomicU64로 읽어 fire 중에도 즉시 응답

## 학습 방식

### 자율 탐색 학습 (ai_train.py)

```
Ollama(gemma3:12b) 입력 생성 → SNN 자율 탐색:
  1. Ollama가 주제별 자연스러운 입력 생성
  2. SNN /fire → 응답
  3. Ollama가 O/X 판정 (맞다/틀리다만, 점수 없음)
  4. O: 강화 → 다음 입력
  5. X: 약화 (출력 1.5x + 경로 0.5x) → 같은 입력으로 재발화 (다른 경로 탐색)
  6. 시도할수록 약화 강도 증가 (0.5 + attempt × 0.1)
  7. 최대 5회 시도, 모두 실패 시에만 Ollama가 정답 제안 → teach
```

### 빠른 학습 (fast_train.py)

```
로컬 1:N 대화 데이터(conversation_multi.json) 반복 학습:
  매 라운드 전체 데이터 셔플 → fire → 정답 매칭 → feedback + teach
  정확 매칭=1.0, 다른 유효 응답=0.9, --no-llm: 토큰 겹침 기반 평가
```

## 상수 일람

| 상수 | 값 | 위치 | 설명 |
|------|-----|------|------|
| INITIAL_WEIGHT | 1.0 | network.rs | 초기 시냅스 가중치 |
| DECAY_RATE | 0.75 | network.rs | 틱마다 활성값 감쇠 |
| BACKWARD_LR | 0.03 | network.rs | 틱마다 시냅스 가중치 조정 |
| FEEDBACK_LR | 0.3 | network.rs | 피드백 시 가중치 조정 |
| MAX_WEIGHT | 5.0 | network.rs | 시냅스 가중치 상한 |
| MIN_WEIGHT | 0.01 | network.rs | 시냅스 가중치 하한 |
| MAX_TICKS | 50 | network.rs | 최대 틱 수 (발화 1회) |
| OUTPUT_THRESHOLD | 1.0 | network.rs | 출력 뉴런 임계값 (게이트) |
| GATE_TICKS | 5 | network.rs | 출력 게이트 열림 틱 수 |
| EMOTION_THRESHOLD | 0.5 | network.rs | 감정 구역 임계값 |
| REASON_THRESHOLD | 0.7 | network.rs | 이성 구역 임계값 |
| CONSOLIDATION_INTERVAL | 5 | network.rs | 해마 통합 주기 (발화 횟수) |
| MIN_PATTERN_FREQ | 3 | network.rs | 해마 패턴 최소 빈도 |
| DEFAULT_THRESHOLD | 0.5 | neuron.rs | 뉴런 기본 임계값 |
| DIVERGE_RATIO | 1.5 | neuron.rs | 발산 판정 비율 |
| PASS_RATIO | 0.85 | neuron.rs | STDP/출력 수집용 비율 |
| DIVERGE_BOOST | 1.3 | neuron.rs | 발산 시 증폭 비율 |
| SIGMOID_TEMPERATURE | 0.15 | neuron.rs | 시그모이드 온도 (발화 확률 기울기) |
| TOP_K_FIRES | 10 | neuron.rs | 뉴런당 최대 발화 시냅스 수 |
| DB_WEIGHT_DISCOUNT | 0.5 | neuron.rs | DB 로드 시냅스 할인율 |
| MAX_ACTIVATION | 3.0 | neuron.rs | 뉴런 활성값 상한 |
| STDP_A_PLUS | 0.02 | network.rs | LTP 강화 크기 |
| STDP_A_MINUS | 0.025 | network.rs | LTD 약화 크기 (비대칭) |
| STDP_TAU | 10.0 | network.rs | STDP 시간 상수 (틱) |
| COOLDOWN_DECAY | 0.7 | network.rs | 쿨다운 감쇠율 |
| COOLDOWN_PENALTY | 0.15 | network.rs | 쿨다운 전달값 감소율 |
| RELEVANCE_BONUS | 2.0 | network.rs | 입력-출력 토큰 관련성 보너스 |
| PATTERN_LEN | 3 | hippocampus.rs | 패턴 추출 서브시퀀스 길이 |

## 데이터 구조

### Neuron (neuron.rs)

```rust
Neuron {
    id: NeuronId,              // String (UUID)
    outgoing: Vec<SynapseId>,  // 나가는 시냅스 목록
    activation: f64,           // 현재 활성값 (비영속, 상한 3.0)
    threshold: f64,            // 발화 기준치 (기본 0.5)
    last_spike_tick: Option<u64>, // STDP: 마지막 spike 틱 (비영속)
}
```

### Synapse (synapse.rs)

```rust
Synapse {
    id: SynapseId,
    pre_neuron: NeuronId,
    post_neuron: NeuronId,
    weight: f64,
    token: Option<String>,           // 저장된 토큰 문자열
    token_type: Option<TokenType>,   // Original, Word, Char, Jamo, NGram(n)
    memory: Option<PathMemory>,      // 경로 기억
    active: bool,
}
```

### TokenType (tokenizer.rs)

```rust
enum TokenType {
    Original,       // 원문 전체 (bonus 0.8x)
    Word,           // 띄어쓰기 단어 (bonus 2.0x) — 가장 높음
    Char,           // 단일 문자 (bonus 1.3x)
    Jamo,           // 한글 자모 (bonus 1.5x)
    NGram(usize),   // 문자 n-gram (2: 1.1x, 3: 1.0x, 4+: 0.9x)
}
```

## 영속화 (synapse.rs)

단일 redb 파일(`data/network.redb`)에 통합 저장:

```
┌──────────────────────────────────────────────────────┐
│                   redb (network.redb)                 │
│  ┌────────────────────┐    ┌────────────────────────┐│
│  │ "synapses_v2" 테이블│    │  "state" 테이블         ││
│  │  key: SynapseId     │    │  key: "network"        ││
│  │  val: Synapse JSON  │    │  val: NetworkState JSON ││
│  └────────────────────┘    └────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

SynapseStore는 캐시(HashMap) + DB(redb) 2단 구조:
- 최대 50,000개 시냅스 캐시
- 캐시 미스 시 DB에서 로드 (DB_WEIGHT_DISCOUNT 0.5 적용)
- 주기적 프루닝으로 약한/중복 시냅스 제거

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 계산 |
| redb 3.1 | 임베디드 KV DB (시냅스 + 상태 영속) |
| serde + serde_json | 직렬화 |
| signal-hook 0.3 | graceful shutdown (SIGTERM/SIGINT) |
| uuid 1 | 뉴런/시냅스 ID 생성 |
| rand 0.10 | 확률적 선택 |
