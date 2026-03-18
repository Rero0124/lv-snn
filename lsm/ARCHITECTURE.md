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
├── server.rs        # actix-web HTTP API + crossbeam 큐 기반 워커 (fire, teach, feedback, status, save)
├── tokenizer.rs     # 텍스트 → 다중 토큰 분해 + 결정적 해시 + 자모 재조합 (compose_jamo)
├── region.rs        # RegionType 정의 (Input, Emotion, Reason, Storage, Output), 구역별 연결 규칙
├── neuron.rs        # Neuron 구조체 (확률적 발화(시그모이드), TOP_K 제한, STDP spike 타이밍)
├── synapse.rs       # Synapse + SynapseStore (캐시 + redb 2단 저장 + 토큰 역인덱스 + 프루닝)
├── hippocampus.rs   # 해마 (뉴런 단위 경로 패턴 추적 → 기억 구역 통합, 전달 후 해마에서 삭제)
├── fire_engine.rs   # lock-free 병렬 발화 엔진 참조 구현 (AtomicU64, SegQueue, crossbeam deque)
└── network.rs       # Network (병렬 발화, STDP, 피드백, teach, 쿨다운, 축삭 발아, 패턴 병합, 응답 조합)

scripts/
├── ai_train.py      # 자율 탐색 학습 (Ollama 입력 생성 → SNN 탐색 → O/X 판정, 최대 5회 재시도)
└── fast_train.py    # 빠른 학습 (로컬 1:N 데이터, Ollama 의미 평가 선택)

data/
├── network.redb           # 시냅스 + 네트워크 상태 영속 저장
└── conversation_multi.json # 1:N 학습용 대화 쌍 (59입력 × ~4응답)
```

## 서버 아키텍처 (server.rs)

```
     HTTP 클라이언트                     actix-web
          │                                │
     POST /fire ──→ 핸들러 ──→ crossbeam 큐(256) ──→ 워커 스레드
          │                                            │
     oneshot::channel ←─────────────── 응답 전송 ←──── Network 단독 소유
                                                       │
                                              자동 저장 (5분 간격)
                                              recv_timeout(30초)
```

- **Network는 워커 스레드가 단독 소유** — Mutex/Arc 없이 단일 스레드에서 순차 처리
- **crossbeam bounded channel(256)** — HTTP 핸들러가 요청을 큐에 넣고 oneshot으로 응답 대기
- **지연 후처리** — fire 응답 즉시 반환, 후처리(해마/STDP/프루닝)는 큐가 비었을 때(`rx.is_empty()`) 실행
- **자동 저장** — recv_timeout(30초)으로 5분 경과 시 자동 저장, 채널 끊김 시 저장 후 종료

## 구역 설계 (region.rs)

```
         ┌──────────────────────────────────────────┐
         │              네트워크                      │
         │                                          │
         │   ┌─────┐                                │
         │   │입력 │──┬──→ 감정(0.5) ←──→ 이성(0.7) │
         │   │(256)│  │      ↕             ↕        │
         │   └─────┘  └──→ 기억(512) ──→ 이성       │
         │                   ↕             ↕        │
         │                 출력(128) ←──────┘       │
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
| Input | Emotion, Reason, Storage | 256 | 0.5 |
| Emotion | Reason, Storage, Output | 128 | 0.5 |
| Reason | Emotion, Storage, Output | 128 | 0.7 |
| Storage | Emotion, Reason, Output | 512 | 0.5 |
| Output | (나가는 연결 없음) | 128 | 1.0 (게이트) |

## 병렬 발화 엔진 (network.rs — fire_parallel)

기본 `/fire` 엔드포인트가 사용하는 lock-free 병렬 발화 구현.
Mutex 없이 AtomicU64 + CAS loop로 뉴런 활성값을 병렬 업데이트한다.

```
발화 흐름 (fire_parallel):
  1. 입력 토큰화 + 뉴런 활성화
  2. Atomic 배열 + 인덱스 매핑 구축 (NeuronId ↔ idx)
  3. 틱 루프 (최대 50틱):
     ├── collect: atomic 배열에서 활성 뉴런 수집 (val > 0.05)
     ├── sync:    atomic → neuron struct (compute_fires용, 감쇠 전 값)
     ├── decay:   atomic 배열 전체 × DECAY_RATE (감쇠)
     ├── compute: rayon par_iter → compute_fires + CAS add (새 신호는 감쇠 안됨)
     ├── drain:   SegQueue → Vec 변환 (결과 수집)
     ├── sync:    atomic → neuron struct (STDP용)
     ├── sprout:  축삭 발아 (2D 그리드 기반)
     └── stdp:    시냅스 가중치 조정
  4. 응답 조합 (assemble_output)
  5. 후처리 데이터 저장 (PostProcessData) — 지연 실행

순차 발화 (fire_sequential):
  /fire_sequential 엔드포인트로 사용 가능 (백업용)
```

### 감쇠 순서 (순차 버전과 동일)

```
순차: active 수집 → compute_fires → decay → receive (새 신호 감쇠 안됨)
병렬: active 수집 → sync → decay → compute + CAS add (새 신호 감쇠 안됨)

핵심: decay를 compute 전에 적용하여 `기존값 × DECAY + 새신호` 순서 보장
```

### AtomicU64 CAS loop (f64 원자적 덧셈)

```rust
// f64를 AtomicU64에 저장: f64::to_bits() / f64::from_bits()
loop {
    let old_bits = atomic[target_idx].load(Relaxed);
    let old_val = f64::from_bits(old_bits);
    let new_val = (old_val + forward_val).min(1.0);
    if atomic[target_idx]
        .compare_exchange_weak(old_bits, new_val.to_bits(), Relaxed, Relaxed)
        .is_ok()
    { break; }
}
```

### 결과 수집 (SegQueue — lock-free)

```rust
let fired_queue: SegQueue<(NeuronId, SynapseId, NeuronId, f64, f64)> = SegQueue::new();
active.par_iter().for_each(|(_, nid, _)| {
    let fires = neuron.compute_fires(emitted_tokens);
    for (sid, post_id, forward_val, token) in fires {
        fired_queue.push(...);  // lock-free push
    }
});
```

## 축삭 발아 (network.rs — try_sprout)

활성화된 뉴런이 2D 그리드 기반 거리로 근처 뉴런과 자발적 시냅스 형성:

```
뉴런 좌표: idx → (idx % cols, idx / cols)
  Input/Storage: 16×16 / 16×32 격자
  Emotion/Reason/Output: 16×8 격자

발아 확률: p = SPROUT_RATE(0.01) × exp(-d² / (2 × SIGMA²(3.0)))
  d=0:     p ≈ 0.01 (최대)
  d=SIGMA: p ≈ 0.006
  d=3σ:    p ≈ 0.0001 (거의 0)

같은 구역: 그리드 반경(5) 내 뉴런만 탐색 (성능 최적화)
다른 구역: 랜덤 3개만 시도
틱당 최대 2개 발아, 초기 weight = 0.05
```

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
  forward = activation × weight × discount + modifier
  (modifier ≤ 0 → skip)
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
- 시냅스 modifier가 음수면 forward 신호 감소, 0 이하면 해당 시냅스 skip

## STDP (network.rs)

생물학적 시냅스 가소성 규칙. 뉴런 발화 타이밍 기반 가중치 조정:

```
Δt = t_post - t_pre

Δt > 0 (인과: pre → post 순서):  Δw = +A_plus(0.02) × exp(-Δt / τ(10))   → 강화 (LTP)
Δt < 0 (반인과):                 Δw = -A_minus(0.025) × exp(Δt / τ(10))   → 약화 (LTD)
Δt = 0 (동시):                   Δw = +A_plus × 0.5                       → 약한 강화
```

STDP weight 반영은 그룹화 최적화: pre_neuron별로 그룹 → outgoing_cache 1회 순회

## 출력 게이트 (network.rs)

Output 뉴런 임계값을 OUTPUT_THRESHOLD(1.0)로 높게 설정하여 초기 잡음 방지:
1. 틱 실행 중 Output 구역에 충분한 신호 축적 확인 (output_ready)
2. GATE_TICKS(5)틱간 게이트 열림 → 임계값을 DEFAULT_THRESHOLD(0.5)로 낮춤
3. 게이트 닫힘 → 임계값 복원 + 활성값 리셋

## 핵심 흐름

### 1. 발화 (fire — 병렬)

```
입력 텍스트 → 토큰화 (원문, 단어, bigram, trigram, 4gram, 문자, 자모)
           → Input/Storage 시냅스 저장
           → 해시 기반 입력 뉴런 활성화
           → Atomic 배열 구축 (neuron activation → AtomicU64)
           → 틱 실행 (최대 MAX_TICKS(50)틱):
             - 활성 뉴런 수집 (atomic > 0.05)
             - atomic → neuron sync (감쇠 전 값)
             - atomic 감쇠 (× DECAY_RATE)
             - rayon par_iter: compute_fires + CAS add (lock-free)
             - SegQueue drain → 결과 수집
             - atomic → neuron sync
             - 축삭 발아 (2D 그리드 기반)
             - STDP 가중치 조정
             - 출력 게이트 메커니즘
           → assemble_output (4층 응답 조합)
           → PostProcessData 저장 (지연 실행)
```

### 2. 지연 후처리 (run_pending_post_process)

큐가 비었을 때(`rx.is_empty()`) 실행, fire 응답을 블로킹하지 않음:

```
해마 기록 (경로 패턴 + co-firing)
→ 해마 통합 (maybe_consolidate → store_memory)
→ co-firing 시냅스 생성/강화
→ 쿨다운 이력 갱신
→ 패턴 병합 (consolidate_patterns)
→ 출력 토큰 연결 (link_output_tokens)
→ 주기적 프루닝 (매 10회 발화)
```

### 3. 피드백 (feedback) — modifier 조절

피드백은 weight가 아닌 **modifier**를 조절한다. weight는 구조적 연결 강도로 teach/STDP에서만 변경.

```
positive=true:
  출력 시냅스: modifier + strength × FEEDBACK_LR(0.1)

positive=false:
  출력 시냅스: modifier - strength × FEEDBACK_LR(0.1) × 1.5  (강하게 약화)
  경로 시냅스: modifier - strength × FEEDBACK_LR(0.1) × 0.5  (다른 경로 탐색 유도)

modifier 범위: -1.0 ~ 1.0 (clamp)
modifier가 음수여도 시냅스 삭제 안 됨 → 경로 유지
```

### 4. 부분 피드백 (feedback_partial) — modifier 조절

토큰별 개별 점수를 받아 시냅스 modifier 세밀 조정:
- token_scores: [("단어", 점수)] — 양수=강화, 음수=약화
- Original 토큰은 모든 점수 평균 적용
- weight는 변경하지 않고 modifier만 조절

### 5. Teach (목표 기반 학습)

```
입력/목표 토큰화 → 시냅스 저장 (Input→Storage, Storage/Emotion/Reason→Output)
               → 조용한 발화 (teach_fire) → 출력 + 경로 길이 수집
               → 점수 계산 → 강화/약화
```

응답 시냅스 저장 시 구역별 다른 가중치:
- Storage → Output: INITIAL_WEIGHT × 1.5 (가장 강함)
- Reason → Output: INITIAL_WEIGHT × 1.2
- Emotion → Output: INITIAL_WEIGHT × 1.0

### 6. 응답 조합 (assemble_output)

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

### 7. 패턴 병합 (consolidate_patterns)

연속 출력된 토큰을 하나의 시냅스로 통합:
- 대상: Jamo+Jamo, Jamo+Char, Char+Word 등 조합
- 같은 틱 또는 2틱 이내 연속 출현한 토큰 쌍
- 기존 병합 시냅스가 있으면 강화, 없으면 새로 생성 (Storage→Output)

### 8. 해마 통합 (hippocampus.rs)

```
매 발화: 뉴런 경로에서 길이 3 서브패턴 추출 → 빈도 카운트
매 CONSOLIDATION_INTERVAL(5)회: 빈도 ≥ MIN_PATTERN_FREQ(3) 패턴 → 기억 구역 저장
  → 전달된 패턴은 해마에서 삭제 (중복 보유 방지)
  → 나머지 카운터 /= 2 감쇠

co-firing 통합: 동시 발화 뉴런 쌍 추적 → 빈도 높은 쌍에 시냅스 생성/강화
```

### 9. 시냅스 프루닝

```
매 10회 발화: 캐시 프루닝 (약한 시냅스 + 중복 시냅스 제거)
  → cleanup_and_neurogenesis: 고아 뉴런 제거 + 새 뉴런 생성
저장 시: DB 전체 프루닝 (weight ≤ MIN_WEIGHT 제거, 동일 pre→post+token 중복 제거)

fire_records는 최근 50개만 유지 (consolidate_path_modifiers 성능 보장)
```

## 쿨다운 (최근 사용 억제)

```
이력 기반 쿨다운 (COOLDOWN_HISTORY(10)회):
  각 시냅스별 최근 10회 사용 여부 추적 (VecDeque<bool>)
  penalty = 사용 횟수 × 가중 합산 (최근일수록 높은 가중치)
  forward_val × (1.0 - penalty)

미사용 이력은 자동 정리 (retain)
```

## 성능 최적화

### synapse_store.get() 제거

`outgoing_cache.post_neuron` 직접 비교로 Mutex lock 회피:

```rust
// 최적화 전: Mutex lock per synapse (O(n) lock)
neuron.outgoing_cache.iter().any(|os|
    self.synapse_store.get(&os.id).is_some_and(|s| s.post_neuron == *nid_b)
)

// 최적화 후: 직접 비교 (lock-free)
neuron.outgoing_cache.iter().any(|os| os.post_neuron == *nid_b)
```

적용 위치: link_output_tokens, connect_cofiring_pairs, consolidate_path_modifiers, cleanup_and_neurogenesis

### STDP 그룹화

pre_neuron별로 업데이트를 그룹화하여 outgoing_cache를 1회만 순회.

### 스프라우트 그리드 바운딩

같은 구역: 2D 그리드 반경 내 뉴런만 탐색 (전체 구역 순회 대신)
다른 구역: 랜덤 3개만 시도

## HTTP API (server.rs)

```
POST /fire              {"text": "..."}                                    → 병렬 발화 + 응답
POST /fire_sequential   {"text": "..."}                                    → 순차 발화 + 응답 (백업)
POST /fire_debug        {"text": "..."}                                    → 디버그 발화
POST /teach             {"input": "...", "target": "..."}                  → 목표 학습
POST /feedback          {"fire_id": N, "positive": bool, "strength": F}    → 피드백
POST /feedback_partial  {"fire_id": N, "token_scores": [["단어", F], ...]} → 부분 피드백
GET  /status                                                                → 상태
POST /save                                                                  → DB 저장
```

- crossbeam 큐 기반 단일 워커 — Network 단독 소유, Mutex 불필요
- /status는 AtomicUsize/AtomicU64로 읽어 fire 중에도 즉시 응답
- 자동 저장: 5분 간격, 채널 끊김 시 저장 후 종료
- 서버 설정: keep_alive 300s, client_request_timeout 300s

## 학습 방식

### 자율 탐색 학습 (ai_train.py)

```
Ollama(exaone3.5:7.8b) 입력 생성 → SNN 자율 탐색:
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
| SPROUT_RATE | 0.01 | network.rs | 축삭 발아 기본 확률 |
| SPROUT_SIGMA | 3.0 | network.rs | 발아 가우시안 반경 |
| SPROUT_WEIGHT | 0.05 | network.rs | 발아 시냅스 초기 weight |
| MAX_SPROUT_PER_TICK | 2 | network.rs | 틱당 최대 발아 수 |
| SPROUT_SEARCH_RADIUS | 5 | network.rs | 발아 탐색 반경 |
| COOLDOWN_HISTORY | 10 | network.rs | 쿨다운 이력 길이 |
| PATTERN_LEN | 3 | hippocampus.rs | 패턴 추출 서브시퀀스 길이 |

## 데이터 구조

### Neuron (neuron.rs)

```rust
Neuron {
    id: NeuronId,                          // String (UUID)
    outgoing: Vec<SynapseId>,              // 나가는 시냅스 목록
    outgoing_cache: Vec<OutgoingSynapse>,   // 캐시된 시냅스 정보 (post_neuron, weight, modifier 포함)
    activation: f64,                       // 현재 활성값 (비영속, 상한 3.0)
    threshold: f64,                        // 발화 기준치 (기본 0.5)
    last_spike_tick: Option<u64>,          // STDP: 마지막 spike 틱 (비영속)
}
```

### OutgoingSynapse (neuron.rs)

```rust
OutgoingSynapse {
    id: SynapseId,
    post_neuron: NeuronId,    // 직접 참조 — synapse_store.get() 불필요
    weight: f64,
    modifier: f64,
    token: Option<String>,
}
```

### Synapse (synapse.rs)

```rust
Synapse {
    id: SynapseId,
    pre_neuron: NeuronId,
    post_neuron: NeuronId,
    weight: f64,                     // 구조적 연결 강도 (teach/STDP로만 변경)
    modifier: f64,                   // 학습 조절값 (feedback으로만 변경, -1.0~1.0)
    token: Option<String>,           // 저장된 토큰 문자열
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
| rayon 1.10 | 뉴런 병렬 계산 (par_iter) |
| crossbeam 0.8 | lock-free 큐 (SegQueue), 채널 (bounded channel) |
| redb 3.1 | 임베디드 KV DB (시냅스 + 상태 영속) |
| serde + serde_json | 직렬화 |
| signal-hook 0.3 | graceful shutdown (SIGTERM/SIGINT) |
| uuid 1 | 뉴런/시냅스 ID 생성 |
| rand 0.10 | 확률적 선택 |
