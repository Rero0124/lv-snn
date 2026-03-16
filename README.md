# LV-SNN

뇌의 구역 구조를 모방한 스파이킹 신경망(Spiking Neural Network) 한국어 대화 시스템.

텍스트를 다중 토큰(원문, 단어, n-gram, 문자, 자모)으로 분해하여 시냅스에 저장하고,
신호가 구역 사이를 틱 단위로 전파하며, STDP와 해마 패턴 기반으로 자율 학습한다.

## 구조

```
입력 텍스트 → 토큰화 (원문/단어/n-gram/문자/자모)
           → Input 구역 뉴런 활성화
           → 틱 전파: Input → Emotion/Reason/Storage → Output
           → 출력 게이트 → compose_response (4층 조합)
           → 해마 경로 기록 + 패턴 병합
```

### 구역 (region.rs)

| 구역 | 뉴런 수 | 역할 | 연결 대상 |
|------|---------|------|----------|
| Input | 128 | 토큰 해시 → 뉴런 매핑, 신호 시작 | Emotion, Reason, Storage |
| Emotion | 64 | 낮은 임계값(0.5), 빠른 직관 반응 | Reason, Storage, Output |
| Reason | 64 | 높은 임계값(0.7), 근거 기반 처리 | Emotion, Storage, Output |
| Storage | 128 | 토큰 시냅스 저장, 해마 패턴 보관 | Emotion, Reason, Output |
| Output | 64 | 응답 토큰 수집, 게이트 닫히면 소멸 | (나가는 연결 없음) |

### 핵심 메커니즘

- **3-State 뉴런** (neuron.rs): 소멸(< threshold x 0.85) / 전달 / 발산(> threshold x 1.5, 신호 1.3배 증폭). 뉴런당 TOP_K(10)개 시냅스만 발화
- **STDP** (network.rs): 뉴런 발화 타이밍 기반 시냅스 강화(LTP)/약화(LTD). pre→post 인과 = 강화, 역인과 = 약화
- **해마** (hippocampus.rs): 뉴런 단위 경로 패턴(길이 3) 추적 → 빈도 높은 패턴을 기억 구역에 통합 → 전달 후 해마에서 삭제
- **출력 게이트** (network.rs): Output 뉴런 임계값을 높게(1.0) 설정 → 충분한 신호 축적 후 5틱간 게이트 열림
- **쿨다운**: 최근 사용 시냅스 전달값 감소(0.15)로 응답 다양성 확보
- **패턴 병합** (consolidate_patterns): 연속 출력된 자모/문자 토큰을 하나의 시냅스로 통합 (예: "ㅂ"+"습니다" → "ㅂ습니다")
- **자모 재조합** (tokenizer.rs): 자모 단위 출력을 한글 음절로 재조합 (compose_jamo)
- **다층 응답 조합** (compose_response): Original(1층) → Word 조합(2층) → Char 조합(3층) → Jamo 재조합(4층) → 폴백

### 피드백

- **positive**: 출력 시냅스 가중치 + strength x 0.3
- **negative**: 출력 시냅스 가중치 - strength x 0.3 x 1.5 (강하게) + 경로 시냅스 - strength x 0.3 x 0.5
- **partial**: 토큰별 개별 점수로 시냅스 세밀 조정

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
| /fire | POST | `{"text": "안녕"}` | 발화 + 응답 |
| /teach | POST | `{"input": "안녕", "target": "반가워!"}` | 목표 학습 |
| /feedback | POST | `{"fire_id": N, "positive": bool, "strength": 0.8}` | 피드백 |
| /feedback_partial | POST | `{"fire_id": N, "token_scores": [["단어", 0.5]]}` | 부분 피드백 |
| /status | GET | - | 상태 (lock-free atomic) |
| /save | POST | - | DB 저장 |

## 학습 스크립트

### 자율 탐색 학습 (ai_train.py)

Ollama(gemma3:4b)가 입력을 생성하고, SNN이 스스로 답을 찾을 때까지 탐색.
LLM은 O/X(맞다/틀리다)만 판정하고 정답을 알려주지 않음.

```
입력 → SNN 발화 "시켜 먹을까?" → Ollama "X" → 약화 + 재발화
     → "무슨 일이야?" → Ollama "X" → 약화 + 재발화
     → "맞아 나도 배고파" → Ollama "O" → 강화
     (최대 5회 실패 시에만 Ollama가 정답 제안 → teach)
```

```bash
ollama pull gemma3:4b
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
| rayon 1.10 | 뉴런 병렬 계산 |
| redb 3.1 | 임베디드 KV DB (시냅스 + 상태 영속) |
| serde + serde_json | 직렬화 |
| signal-hook 0.3 | graceful shutdown |
| uuid 1 | 뉴런/시냅스 ID 생성 |
