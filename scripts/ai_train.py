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

SSN_URL = "http://127.0.0.1:3000"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma3:4b"
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
    """Ollama가 맞다/틀리다만 판단 (엄격)"""
    if not output_text:
        return False
    prompt = (
        f"A가 말하고 B가 대답했다. B의 대답이 A의 주제에 정확히 맞는 응답인지 판단.\n"
        f"A:\"{input_text}\"\nB:\"{output_text}\"\n"
        f"엄격 기준 (하나라도 해당되면 X):\n"
        f"- B가 A의 주제와 다른 얘기를 하면 X\n"
        f"- B가 A의 질문/상황에 맞지 않으면 X\n"
        f"- B가 A에 대한 적절한 반응이면 O\n"
        f"O 또는 X 한 글자만:"
    )
    result = ollama(prompt, max_tokens=3)
    if result:
        first = result.strip()[0] if result.strip() else 'X'
        return first in ('O', 'o', '○')
    return False


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
    print(f"LLM: {OLLAMA_MODEL} (판정만: O/X)")
    print(f"방식: 입력 → SNN 탐색 (최대 {MAX_RETRY}회) → 실패 시만 teach")
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

            # 3) Ollama 판정: O/X
            ok = judge(inp, out)

            if ok:
                # 맞음! → 강화
                if fire_id:
                    ssn_request("/feedback", {"fire_id": fire_id, "positive": True, "strength": 0.8})
                    total_pos += 1
                found = True
                found_at.append(attempt)
                print(f"    {attempt}) \"{out}\" ✓")
                if attempt > (best.get('tries') or 999):
                    best = {'tries': attempt, 'input': inp, 'output': out}
                context.append(f"{inp}→{out}")
                break
            else:
                # 틀림! → 약화 (SNN이 다음 시도에서 다른 경로 선택하도록)
                if fire_id:
                    neg_strength = min(1.0, 0.5 + attempt * 0.1)  # 시도할수록 더 강하게 약화
                    ssn_request("/feedback", {"fire_id": fire_id, "positive": False, "strength": neg_strength})
                    total_neg += 1
                print(f"    {attempt}) \"{out}\" ✗")

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
