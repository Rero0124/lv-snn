#!/usr/bin/env python3
"""fire + feedback 학습 비교 (teach 없음).

  --mode global : 기존 방식 — 전역 보상 (test_words.py 와 같은 점수 계산)
  --mode target : 출력별·오차 기반 보상 — /feedback 에 정답 문자열(target)을 넘김
  --mode rpe    : 전역 보상 + 보상 예측 오차 — 정답을 알려주지 않고, 그 입력의 평소 점수보다
                  이번 출력이 나았으면 강화, 못했으면 약화 (점수 = 출력/정답 자모 집합의 F1)
  --mode length : 전역 보상 + 출력 길이 반영 — 정답보다 길수록 보상을 줄이고, 2배 넘게 길면 음의 보상

정답률(완전 일치)과 함께 자모 집합 지표를 본다:
  재현율 = 정답 자모 중 출력에 나온 비율, 정밀도 = 출력 자모 중 정답에 있는 비율
"""
import argparse, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts_util import decompose_jamo, jamo_overlap

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8081")
ap.add_argument("--mode", choices=["global", "target", "length", "rpe"], default="target")
ap.add_argument("--rounds", type=int, default=2000)
ap.add_argument("--report", type=int, default=200)
ap.add_argument("--task", choices=["words", "simple"], default="words",
                help="words: 단어 5쌍, simple: 자모 하나 → 같은 자모 (ㄱ→ㄱ 등)")
args = ap.parse_args()

def ssn(path, data=None):
    req = urllib.request.Request(f"{args.url}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())

tests = [("안녕", "반가워"), ("고마워", "천만에"), ("사랑해", "나도"), ("잘자", "좋은꿈"), ("배고파", "밥먹자")]
if args.task == "simple":
    tests = [("ㄱ", "ㄱ"), ("ㄴ", "ㄴ"), ("ㄷ", "ㄷ"), ("ㅁ", "ㅁ"), ("ㅂ", "ㅂ")]

def syllable_len(s):
    return sum(1.0 if "가" <= c <= "힣" else 0.3 for c in s)

def global_feedback(fire_id, out, expected):
    if out == expected:
        return ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 1.0})
    if not out:
        return ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.05})
    score = (0.5 if expected in out else 0.0) + (0.2 if out[0] == expected[0] else 0.0)
    score += jamo_overlap(out, expected) * 0.3
    diff = abs(syllable_len(out) - syllable_len(expected))
    if diff <= 1.5:
        score += 0.2 / (1.0 + diff)
    if score > 0:
        ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": min(score, 1.0)})
    else:
        ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.05})

def jamo_len(s):
    """자모 개수 (완성형 음절은 초·중성 2개, 받침 있으면 3개)"""
    return sum((3 if (ord(c) - 0xAC00) % 28 else 2) if "가" <= c <= "힣" else 1 for c in s)

def similarity(out, expected):
    score = (0.5 if expected in out else 0.0) + (0.2 if out[0] == expected[0] else 0.0)
    return min(score + jamo_overlap(out, expected) * 0.3, 1.0)

def length_feedback(fire_id, out, expected):
    if out == expected:
        return ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 1.0})
    if not out:
        return ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.05})
    lo, le = jamo_len(out), jamo_len(expected)
    ratio = min(1.0, le / lo)          # 정답보다 길수록 작아짐
    if lo > 2 * le:                    # 너무 긴 출력은 약화
        ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.3 * (1.0 - ratio)})
    else:                              # 짧을수록(정답 길이에 가까울수록) 강하게 강화
        strength = max(similarity(out, expected), 0.1) * ratio
        ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": strength})

baseline = {}
def rpe_feedback(fire_id, inp, out, expected):
    o, e = decompose_jamo(out), decompose_jamo(expected)
    hit = len(o & e)
    score = 2 * hit / (len(o) + len(e)) if hit else 0.0   # 자모 집합 F1: 군더더기 출력이 많으면 낮아짐
    base = baseline.get(inp, score)
    baseline[inp] = 0.9 * base + 0.1 * score
    delta = score - base
    if delta != 0.0:
        ssn("/feedback", {"fire_id": fire_id, "positive": delta > 0, "strength": min(1.0, abs(delta) * 5.0)})

print(f"mode={args.mode} task={args.task}")
print(f"{'round':>6} {'정답':>5} {'재현율':>6} {'정밀도':>6} {'출력길이':>7}  예시")
for rnd in range(args.rounds + 1):
    if rnd % args.report == 0:  # 평가 (피드백 없이 fire 만)
        exact, rec, prec, length, samples = 0, 0.0, 0.0, 0, []
        for inp, expected in tests:
            out = ssn("/fire", {"text": inp}).get("output", "")
            exact += out == expected
            o, e = decompose_jamo(out), decompose_jamo(expected)
            rec += len(o & e) / len(e)
            prec += len(o & e) / len(o) if o else 0.0
            length += len(out)
            samples.append(out[:12])
        n = len(tests)
        print(f"{rnd:>6} {exact:>3}/{n} {rec/n:>6.2f} {prec/n:>6.2f} {length/n:>7.1f}  {' | '.join(samples)}")
    if rnd == args.rounds:
        break
    for inp, expected in tests:
        resp = ssn("/fire", {"text": inp})
        if args.mode == "target":
            ssn("/feedback", {"fire_id": resp["fire_id"], "target": expected, "strength": 1.0})
        elif args.mode == "rpe":
            rpe_feedback(resp["fire_id"], inp, resp.get("output", ""), expected)
        elif args.mode == "length":
            length_feedback(resp["fire_id"], resp.get("output", ""), expected)
        else:
            global_feedback(resp["fire_id"], resp.get("output", ""), expected)
