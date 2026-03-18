# LV-SNN

뇌의 구역 구조를 모방한 스파이킹 신경망(Spiking Neural Network) 한국어 대화 시스템.

텍스트를 다중 토큰(원문, 단어, n-gram, 문자, 자모)으로 분해하여 시냅스에 저장하고,
신호가 구역 사이를 틱 단위로 전파하며, STDP와 해마 패턴 기반으로 자율 학습한다.
Lock-free 병렬 발화 엔진(AtomicU64 + CAS)으로 뉴런 활성값을 병렬 업데이트한다.

## 구조

```
입력 텍스트 → 토큰화 (원문/단어/n-gram/문자/자모)
           → Input 구역 뉴런 활성화
           → Atomic 배열 구축 + 병렬 틱 전파:
             decay → rayon par_iter compute_fires + CAS add → SegQueue drain
           → 출력 게이트 → assemble_output (4층 조합)
           → 지연 후처리 (해마/STDP/프루닝, 큐 비었을 때 실행)
```

### 구역 (region.rs)

| 구역 | 뉴런 수 | 역할 | 연결 대상 |
|------|---------|------|----------|
| Input | 256 | 토큰 해시 → 뉴런 매핑, 신호 시작 | Emotion, Reason, Storage |
| Emotion | 128 | 낮은 임계값(0.5), 빠른 직관 반응 | Reason, Storage, Output |
| Reason | 128 | 높은 임계값(0.7), 근거 기반 처리 | Emotion, Storage, Output |
| Storage | 512 | 토큰 시냅스 저장, 해마 패턴 보관 | Emotion, Reason, Output |
| Output | 128 | 응답 토큰 수집, 게이트 닫히면 소멸 | (나가는 연결 없음) |

### 핵심 메커니즘

- **병렬 발화 엔진** (network.rs): AtomicU64 + CAS loop로 lock-free 병렬 activation 갱신. SegQueue로 결과 수집. 순차 대비 약 3배 빠름
- **확률적 뉴런 발화** (neuron.rs): 시그모이드 기반 확률적 발화 `p = sigmoid(activation - threshold, 0.15)`. 뉴런당 TOP_K(10)개 시냅스만 발화
- **STDP** (network.rs): 뉴런 발화 타이밍 기반 시냅스 강화(LTP)/약화(LTD). pre→post 인과 = 강화, 역인과 = 약화
- **축삭 발아** (network.rs): 활성 뉴런이 2D 그리드 기반 거리로 근처 뉴런과 자발적 시냅스 형성 (가우시안 감쇠)
- **해마** (hippocampus.rs): 뉴런 단위 경로 패턴(길이 3) 추적 + co-firing 뉴런 쌍 추적 → 기억 구역 통합
- **출력 게이트** (network.rs): Output 뉴런 임계값을 높게(1.0) 설정 → 충분한 신호 축적 후 5틱간 게이트 열림
- **쿨다운**: 최근 10회 사용 이력 기반 시냅스 전달값 감소로 응답 다양성 확보
- **패턴 병합** (consolidate_patterns): 연속 출력된 자모/문자 토큰을 하나의 시냅스로 통합
- **자모 재조합** (tokenizer.rs): 자모 단위 출력을 한글 음절로 재조합 (compose_jamo)
- **다층 응답 조합** (assemble_output): Original(1층) → Word 조합(2층) → Char 조합(3층) → Jamo 재조합(4층) → 폴백

### 피드백 (modifier 시스템)

피드백은 시냅스 weight가 아닌 **modifier** (학습 조절값, -1.0~1.0)를 조절한다.
weight는 구조적 연결 강도로 teach/STDP에서만 변경되며, modifier가 음수여도 시냅스는 삭제되지 않아 경로가 유지된다.

- **positive**: 출력 시냅스 modifier + strength x 0.1
- **negative**: 출력 시냅스 modifier - strength x 0.1 x 1.5 (강하게) + 경로 시냅스 modifier - strength x 0.1 x 0.5
- **partial**: 토큰별 개별 점수로 시냅스 modifier 세밀 조정
- **발화 계산**: `forward = activation × weight × discount + modifier` (modifier ≤ 0이면 skip)

## 서버 아키텍처

```
HTTP 클라이언트 → actix-web 핸들러 → crossbeam 큐(256) → 워커 스레드 (Network 단독 소유)
                                                          ├── 지연 후처리 (큐 비었을 때)
                                                          └── 자동 저장 (5분 간격)
```

- Network는 Mutex 없이 워커 스레드가 단독 소유
- oneshot 채널로 요청-응답 매핑
- 채널 끊김 시 저장 후 종료

## 빌드 & 실행

```bash
cargo build --release

# 서버 모드
./target/release/lv-snn --serve              # http://127.0.0.1:3000
./target/release/lv-snn --serve --port 8080  # 포트 변경

# 대화형 모드
./target/release/lv-snn
```

## HTTP API (server.rs)

| 엔드포인트 | 메서드 | 요청 | 설명 |
|-----------|--------|------|------|
| /fire | POST | `{"text": "안녕"}` | 병렬 발화 + 응답 |
| /fire_sequential | POST | `{"text": "안녕"}` | 순차 발화 + 응답 (백업) |
| /fire_debug | POST | `{"text": "안녕"}` | 디버그 발화 |
| /teach | POST | `{"input": "안녕", "target": "반가워!"}` | 목표 학습 |
| /feedback | POST | `{"fire_id": N, "positive": bool, "strength": 0.8}` | 피드백 |
| /feedback_partial | POST | `{"fire_id": N, "token_scores": [["단어", 0.5]]}` | 부분 피드백 |
| /status | GET | - | 상태 (lock-free atomic) |
| /save | POST | - | DB 저장 |

## 학습 스크립트

### 자율 탐색 학습 (ai_train.py)

Ollama(exaone3.5:7.8b)가 입력을 생성하고, SNN이 스스로 답을 찾을 때까지 탐색.
LLM은 O/X(맞다/틀리다)만 판정하고 정답을 알려주지 않음.

```
입력 → SNN 발화 "시켜 먹을까?" → Ollama "X" → 약화 + 재발화
     → "무슨 일이야?" → Ollama "X" → 약화 + 재발화
     → "맞아 나도 배고파" → Ollama "O" → 강화
     (최대 5회 실패 시에만 Ollama가 정답 제안 → teach)
```

```bash
ollama pull exaone3.5:7.8b
python3 scripts/ai_train.py --topic "음식,여행,감정" --duration 1800
```

### 빠른 학습 (fast_train.py)

로컬 1:N 대화 데이터로 반복 학습. LLM 없이도 동작.

```bash
python3 scripts/fast_train.py --rounds 5 --no-llm    # 토큰 매칭만
python3 scripts/fast_train.py --rounds 10             # Ollama 의미 평가 포함
```

## 데이터

- `data/conversation_multi.json` — 1:N 대화 쌍 (59개 입력, 각 4~5개 응답)
- `data/network.redb` — 시냅스 + 네트워크 상태 영속 저장 (자동 생성)

## 의존성

| 크레이트 | 용도 |
|---------|------|
| actix-web 4 | HTTP API 서버 |
| tokio 1 | 비동기 런타임 |
| rayon 1.10 | 뉴런 병렬 계산 (par_iter) |
| crossbeam 0.8 | lock-free 큐 (SegQueue), 채널 (bounded channel) |
| redb 3.1 | 임베디드 KV DB (시냅스 + 상태 영속) |
| serde + serde_json | 직렬화 |
| signal-hook 0.3 | graceful shutdown |
| uuid 1 | 뉴런/시냅스 ID 생성 |
| rand 0.10 | 확률적 발화 |
