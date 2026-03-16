#!/usr/bin/env python3
"""
LV-SNN 빠른 학습 스크립트
- conversation_multi.json에서 1:N 대화 쌍을 로드
- 매번 랜덤 응답을 선택해서 teach
- Claude 호출 없이 빠르게 반복 학습
"""

import json
import time
import sys
import random
import io as _io
import urllib.request
import argparse

sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:3000"
OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_evaluate(input_text, output_text):
    """Ollama로 응답 의미 평가 (0.0~1.0)"""
    if not output_text:
        return 0.0

    prompt = (
        f"대화 평가 (엄격하게). 점수만 출력.\n"
        f"A: \"{input_text}\"\nB: \"{output_text}\"\n"
        f"기준: 무관=0.0~0.1, 어색=0.2~0.4, 자연스러움=0.5~0.7, 완벽=0.8~1.0\n점수:"
    )
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({"model": "gemma3:4b", "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())["response"].strip()
            # 숫자 추출
            import re
            m = re.search(r'([01]\.?\d*)', result)
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        pass
    return 0.0


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
        print(f"  [오류] {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=600, help="학습 시간(초)")
    parser.add_argument("--data", type=str, default="data/conversation_multi.json")
    parser.add_argument("--rounds", type=int, default=0, help="라운드 수 (0=시간 기반)")
    parser.add_argument("--no-llm", action="store_true", help="Ollama 없이 토큰 매칭만 사용")
    args = parser.parse_args()

    # 데이터 로드
    with open(args.data) as f:
        pairs = json.load(f)

    total_responses = sum(len(p["responses"]) for p in pairs)
    print(f"=== LV-SNN 빠른 학습 ===")
    print(f"데이터: {len(pairs)}개 입력 × 평균 {total_responses/len(pairs):.1f}개 응답 = {total_responses}쌍")

    # 서버 확인
    status = ssn_request("/status")
    if not status:
        print("서버가 실행 중이지 않습니다.")
        return
    print(f"시냅스: {status['synapses']} | 패턴: {status['patterns']}")
    print()

    start = time.time()
    fires = 0
    teaches = 0
    pos = 0
    neg = 0
    scores = []
    best_score = 0
    best_info = ""
    round_num = 0

    while True:
        round_num += 1
        if args.rounds > 0 and round_num > args.rounds:
            break
        if args.rounds == 0 and time.time() - start >= args.duration:
            break

        # 전체 대화 쌍을 셔플해서 한 라운드
        order = list(range(len(pairs)))
        random.shuffle(order)

        for idx in order:
            if args.rounds == 0 and time.time() - start >= args.duration:
                break

            pair = pairs[idx]
            inp = pair["input"]
            target = random.choice(pair["responses"])

            # 1) fire: 현재 네트워크 응답 확인
            resp = ssn_request("/fire", {"text": inp})
            if not resp:
                continue
            fires += 1
            out = resp.get("output", "")
            fire_id = resp.get("fire_id", 0)

            # 2) 평가: 정확 매칭이면 바로 1.0, 아니면 LLM 의미 평가
            if out.strip() in [r.strip() for r in pair["responses"]]:
                score = 1.0 if out.strip() == target.strip() else 0.9
            elif out and not args.no_llm:
                score = ollama_evaluate(inp, out)
            elif out:
                score = simple_score(out, target, pair["responses"])
            else:
                score = 0.0

            scores.append(score)
            if score > best_score and out:
                best_score = score
                best_info = f'"{inp}" → "{out}"'

            # 3) 피드백 + teach
            if score >= 0.5:
                ssn_request("/feedback", {"fire_id": fire_id, "positive": True, "strength": min(score, 1.0)})
                pos += 1
            elif out:
                ssn_request("/feedback", {"fire_id": fire_id, "positive": False, "strength": max(0.3, 1.0 - score)})
                neg += 1

            # 항상 teach (정답 경로 강화)
            ssn_request("/teach", {"input": inp, "target": target})
            teaches += 1

        # 라운드 끝 리포트
        elapsed = int(time.time() - start)
        r50 = scores[-50:] if scores else [0]
        r10 = scores[-10:] if len(scores) >= 10 else scores or [0]
        st = ssn_request("/status") or {}

        print(f"--- 라운드 {round_num} ({elapsed}초) ---")
        print(f"  발화: {fires} | teach: {teaches} | +{pos} -{neg}")
        print(f"  최근50: {sum(r50)/len(r50):.1%} | 최근10: {sum(r10)/len(r10):.1%}")
        print(f"  시냅스: {st.get('synapses',0)} | 패턴: {st.get('patterns',0)}")
        print(f"  최고: {best_info} ({best_score:.0%})")
        print()

    # 저장
    ssn_request("/save")
    elapsed = int(time.time() - start)
    avg = sum(scores) / max(len(scores), 1)
    print(f"{'='*50}")
    print(f"=== 빠른 학습 완료 ({elapsed}초, {round_num}라운드) ===")
    print(f"  발화: {fires} | teach: {teaches}")
    print(f"  강화: {pos} | 약화: {neg}")
    print(f"  전체 평균: {avg:.1%}")
    if len(scores) > 20:
        f = scores[:min(50, len(scores))]
        l = scores[-min(50, len(scores)):]
        print(f"  초반: {sum(f)/len(f):.1%} → 후반: {sum(l)/len(l):.1%}")
    print(f"  최고: {best_info} ({best_score:.0%})")
    print("\n저장 완료.")


def simple_score(output, target, all_responses):
    """토큰 겹침 기반 간단 평가"""
    if not output:
        return 0.0

    # 정확히 일치하면 1.0
    if output.strip() == target.strip():
        return 1.0

    # 다른 유효 응답과 일치해도 0.9
    for r in all_responses:
        if output.strip() == r.strip():
            return 0.9

    # 단어 겹침 점수
    out_words = set(output.split())
    target_words = set(target.split())
    all_valid_words = set()
    for r in all_responses:
        all_valid_words.update(r.split())

    if not out_words:
        return 0.0

    # 출력 단어 중 유효 응답에 포함된 비율
    overlap = len(out_words & all_valid_words) / len(out_words)

    # 글자 단위 겹침 (짧은 응답 처리)
    out_chars = set(output)
    target_chars = set(target)
    char_overlap = len(out_chars & target_chars) / max(len(out_chars), 1)

    return max(overlap * 0.7, char_overlap * 0.5)


if __name__ == "__main__":
    main()
