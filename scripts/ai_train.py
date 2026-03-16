#!/usr/bin/env python3
"""
LV-SNN AI 주도 학습 스크립트 (자율 탐색 방식)
- Ollama가 입력 생성 → SNN이 응답 → Ollama가 맞다/틀리다만 판단
- 틀리면 SNN이 해당 경로 약화 후 재시도 (최대 MAX_RETRY번)
- 재시도마다 다른 경로를 탐색 → "이건가? 저건가?" 자율 학습
- 모두 실패하면 그때만 Ollama가 정답 제안 → teach

사용법:
  python3 scripts/ai_train.py --topic "음식,여행" --duration 1800
"""

import json
import time
import sys
import re
import io as _io
import urllib.request
import argparse

sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:8081"
#OLLAMA_URL = "http://192.168.0.31:11434"
OLLAMA_URL = "http://192.168.55.150:11434"
OLLAMA_MODEL = "gemma3:12b"
MAX_RETRY = 5  # 최대 재시도 횟수


def ollama(prompt, max_tokens=100):
    """Ollama 호출"""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["response"].strip()
    except Exception as e:
        print(f"  [ollama 오류] {e}", file=sys.stderr)
        return None


def generate_input(topic, recent=""):
    """Ollama가 입력 한 줄 생성"""
    ctx = f"최근: {recent}\n" if recent else ""
    prompt = (
        f"{ctx}'{topic}' 관련 자연스러운 한국어 일상 대화 한마디. "
        f"반말이나 존댓말 자유. 1~2문장. 따옴표 없이 텍스트만."
    )
    result = ollama(prompt, max_tokens=60)
    if result:
        result = result.strip().strip('"\'')
        result = re.sub(r'^\d+[\.\)]\s*', '', result)
        if 1 < len(result) < 80:
            return result
    return None


def judge(input_text, output_text):
    """Ollama가 0~1 점수로 평가 (엄격)"""
    if not output_text:
        return 0.0
    prompt = (
        f"A가 말했고 B가 대답했다. B의 응답 품질을 0.0~1.0 점수로 평가.\n"
        f"A:\"{input_text}\"\nB:\"{output_text}\"\n\n"
        f"채점 기준 (엄격하게, 0.7 이상은 정말 잘한 경우만):\n"
        f"- 주제 일치: A의 핵심 주제/키워드에 B가 구체적으로 반응하는가?\n"
        f"- 상황 적합: A의 질문/감정/상황에 맞는 대답인가?\n"
        f"- 구체성: 아무 말에나 통하는 범용 대답이면 0.2 이하\n"
        f"- 자연스러움: 실제 대화처럼 자연스러운가?\n\n"
        f"감점 예시:\n"
        f"- A가 음식 얘기인데 B가 음식 무관 → 0.0~0.1\n"
        f"- B가 두루뭉술하게 맞장구만 → 0.2~0.3\n"
        f"- B가 주제는 맞지만 어색 → 0.4~0.5\n"
        f"- B가 주제에 맞고 자연스러움 → 0.6~0.7\n"
        f"- B가 완벽한 응답 → 0.8~1.0\n\n"
        f"숫자만 출력 (예: 0.3):"
    )
    result = ollama(prompt, max_tokens=5)
    if result:
        try:
            score = float(re.search(r'[01]\.?\d*', result.strip()).group())
            return min(1.0, max(0.0, score))
        except Exception:
            pass
    return 0.0


def judge_tokens(input_text, output_tokens):
    """출력 토큰 중 단어(2글자+) 상위 8개만 평가하여 부분 피드백 생성"""
    words = [t for t in output_tokens if t["length"] >= 2]
    # 가중치 상위 8개만 (LLM 부담 줄이기 + JSON 길이 제한)
    words = sorted(words, key=lambda t: -t["weight"])[:8]
    if not words:
        return []

    word_list = ", ".join(f'"{w["token"]}"' for w in words)
    prompt = (
        f"A가 \"{input_text}\"라고 말했다.\n"
        f"B의 응답 단어들: [{word_list}]\n"
        f"각 단어가 A의 말에 적절한지 -1.0~1.0 점수. "
        f"적절=양수, 무관=0, 부적절=음수.\n"
        f"JSON 배열만 출력 (예: [0.5, -0.3, 0.8]):"
    )
    result = ollama(prompt, max_tokens=80)
    if not result:
        return []
    try:
        scores = json.loads(re.search(r'\[.*?\]', result.strip()).group())
        # 개수 불일치면 짧은 쪽에 맞춤
        pairs = min(len(scores), len(words))
        return [(w["token"], max(-1.0, min(1.0, float(s)))) for w, s in zip(words[:pairs], scores[:pairs])]
    except Exception:
        return []


def reorder_tokens(input_text, tokens):
    """긍정 평가된 토큰들을 자연스러운 순서로 재배열"""
    token_list = ", ".join(f'"{t}"' for t in tokens)
    prompt = (
        f"A가 \"{input_text}\"라고 말했다.\n"
        f"B가 이 단어들로 응답하려 한다: [{token_list}]\n"
        f"이 단어들만 사용해서 자연스러운 한국어 응답 1문장을 만들어라. "
        f"단어를 빼거나 새로 추가하지 말고 순서만 바꿔라. 반말. 따옴표 없이 텍스트만."
    )
    result = ollama(prompt, max_tokens=60)
    if result:
        result = result.strip().strip('"\'')
        if 1 < len(result) < 80:
            return result
    return None


def suggest_response(input_text):
    """모든 재시도 실패 시에만 호출 — Ollama가 정답 제안"""
    prompt = f"\"{input_text}\"에 대한 자연스러운 한국어 응답 1문장. 반말. 따옴표 없이 텍스트만."
    result = ollama(prompt, max_tokens=50)
    if result:
        result = result.strip().strip('"\'')
        if 1 < len(result) < 80:
            return result
    return None


def ssn_request(path, data=None):
    url = f"{SSN_URL}{path}"
    if data is not None:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [SSN 오류] {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="일상 대화")
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=5)
    args = parser.parse_args()

    global MAX_RETRY
    MAX_RETRY = args.max_retry

    status = ssn_request("/status")
    if not status:
        print("LV-SNN 서버가 실행 중이지 않습니다.")
        return

    test = ollama("안녕", max_tokens=3)
    if not test:
        print("Ollama 서버가 실행 중이지 않습니다.")
        return

    topics = [t.strip() for t in args.topic.split(",")]

    print(f"=== LV-SNN 자율 탐색 학습 ({args.duration}초) ===")
    print(f"주제: {', '.join(topics)}")
    print(f"서버: 시냅스 {status['synapses']} | 패턴 {status['patterns']}")
    print(f"LLM: {OLLAMA_MODEL} (0~1 점수 평가)")
    print(f"방식: 입력 → SNN 탐색 (최대 {MAX_RETRY}회) → ≥0.6 강화, 0.5~0.6 무시, <0.5 약화, 전패 시 teach")
    print()

    start = time.time()
    total_inputs = 0
    total_fires = 0
    total_teaches = 0
    total_pos = 0
    total_neg = 0
    found_at = []  # 몇 번째 시도에서 맞았는지
    context = []
    last_report = 0
    topic_idx = 0
    best = {'tries': 0, 'input': '', 'output': ''}

    while time.time() - start < args.duration:
        topic = topics[topic_idx % len(topics)]
        topic_idx += 1

        # 1) 입력 생성
        ctx = " / ".join(context[-3:]) if context else ""
        inp = generate_input(topic, ctx)
        if not inp:
            continue

        total_inputs += 1
        print(f"\n  [{total_inputs}] \"{inp}\"")

        # 2) SNN 자율 탐색: 최대 MAX_RETRY번 시도
        found = False
        tried_outputs = []

        for attempt in range(1, MAX_RETRY + 1):
            resp = ssn_request("/fire", {"text": inp})
            if not resp:
                break
            total_fires += 1
            out = resp.get("output", "")
            fire_id = resp.get("fire_id", 0)

            # 같은 응답이 또 나오면 (경로가 안 바뀜) → 바로 약화하고 다음 시도
            if out in tried_outputs:
                if fire_id:
                    ssn_request("/feedback", {"fire_id": fire_id, "positive": False, "strength": 1.0})
                    total_neg += 1
                print(f"    {attempt}) \"{out}\" (중복, 약화)")
                continue

            tried_outputs.append(out)

            if not out:
                print(f"    {attempt}) (무출력)")
                continue

            # 3) Ollama 점수 평가 (0.0~1.0)
            score = judge(inp, out)

            if score >= 0.6:
                # 좋은 응답 → 점수 비례 강화
                if fire_id:
                    ssn_request("/feedback", {"fire_id": fire_id, "positive": True, "strength": score})
                    total_pos += 1
                found = True
                found_at.append(attempt)
                print(f"    {attempt}) \"{out}\" ✓ {score:.1f}")
                if attempt > (best.get('tries') or 999):
                    best = {'tries': attempt, 'input': inp, 'output': out}
                context.append(f"{inp}→{out}")
                break
            elif score >= 0.3:
                # 애매한 응답 (0.3~0.6) → 토큰별 부분 평가 + 순서 재배열 teach
                output_tokens = resp.get("output_tokens", [])
                token_scores = judge_tokens(inp, output_tokens)
                if token_scores and fire_id:
                    ssn_request("/feedback_partial", {"fire_id": fire_id, "token_scores": token_scores})
                    pos_t = sum(1 for _, s in token_scores if s > 0)
                    neg_t = sum(1 for _, s in token_scores if s < 0)
                    # 긍정 토큰이 3개 이상이면 재배열 teach
                    good_tokens = [tok for tok, s in token_scores if s > 0]
                    if len(good_tokens) >= 3:
                        reordered = reorder_tokens(inp, good_tokens)
                        if reordered:
                            ssn_request("/teach", {"input": inp, "target": reordered})
                            print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t}) → reorder \"{reordered}\"")
                        else:
                            print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t})")
                    else:
                        print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t})")
                else:
                    print(f"    {attempt}) \"{out}\" ~ {score:.1f}")
            else:
                # 나쁜 응답 (<0.3) → 약화 (시도할수록 강하게)
                if fire_id:
                    neg_strength = min(1.0, (1.0 - score) * 0.5 + attempt * 0.1)
                    ssn_request("/feedback", {"fire_id": fire_id, "positive": False, "strength": neg_strength})
                    total_neg += 1
                print(f"    {attempt}) \"{out}\" ✗ {score:.1f}")

        # 4) 모든 시도 실패 → 그때만 LLM이 정답 제안
        if not found:
            suggestion = suggest_response(inp)
            if suggestion:
                ssn_request("/teach", {"input": inp, "target": suggestion})
                total_teaches += 1
                print(f"    → teach \"{suggestion}\"")
            else:
                print(f"    → (제안 실패)")
            context.append(f"{inp}→?")

        if len(context) > 5:
            context.pop(0)

        # 60초 리포트
        elapsed = int(time.time() - start)
        if elapsed >= last_report + 60:
            last_report = elapsed
            st = ssn_request("/status") or {}
            success_rate = len(found_at) / max(total_inputs, 1)
            avg_tries = sum(found_at) / max(len(found_at), 1)
            r10 = found_at[-10:] if len(found_at) >= 10 else found_at or [0]
            print(f"\n--- [{elapsed // 60}분 경과] ---")
            print(f"  입력: {total_inputs} | 발화: {total_fires} | teach: {total_teaches}")
            print(f"  강화: {total_pos} | 약화: {total_neg}")
            print(f"  자력 성공: {success_rate:.0%} ({len(found_at)}/{total_inputs})")
            print(f"  평균 시도: {avg_tries:.1f}회 | 최근10: {sum(r10)/len(r10):.1f}회")
            print(f"  시냅스: {st.get('synapses',0)} | 패턴: {st.get('patterns',0)}")
            print()

    # 저장
    ssn_request("/save")
    elapsed = int(time.time() - start)
    success_rate = len(found_at) / max(total_inputs, 1)
    avg_tries = sum(found_at) / max(len(found_at), 1) if found_at else 0
    print(f"\n{'='*50}")
    print(f"=== 자율 탐색 학습 완료 ({elapsed}초) ===")
    print(f"  입력: {total_inputs} | 발화: {total_fires} | teach: {total_teaches}")
    print(f"  강화: {total_pos} | 약화: {total_neg}")
    print(f"  자력 성공률: {success_rate:.0%} ({len(found_at)}/{total_inputs})")
    if found_at:
        print(f"  평균 시도: {avg_tries:.1f}회")
        by_try = {}
        for t in found_at:
            by_try[t] = by_try.get(t, 0) + 1
        dist = " | ".join(f"{k}번째:{v}회" for k, v in sorted(by_try.items()))
        print(f"  성공 분포: {dist}")
    if len(found_at) > 20:
        f = found_at[:min(20, len(found_at))]
        l = found_at[-min(20, len(found_at)):]
        print(f"  초반 평균: {sum(f)/len(f):.1f}회 → 후반 평균: {sum(l)/len(l):.1f}회")
    print("\n저장 완료.")


if __name__ == "__main__":
    main()
