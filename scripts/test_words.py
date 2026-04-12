#!/usr/bin/env python3
"""단어 학습 테스트 — teach + fire + feedback (정답률 기반 약화)"""

import json
import urllib.request
import time
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:8081"

def ssn(path, data=None):
    url = f"{SSN_URL}{path}"
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

tests = [
    ("안녕", "반가워"),
    ("고마워", "천만에"),
    ("사랑해", "나도"),
    ("잘자", "좋은꿈"),
    ("배고파", "밥먹자"),
]

DURATION = 3600
start = time.time()
round_count = 0
total_correct = 0
total_tests = 0

print(f"=== 학습 ({DURATION}초) ===\n")

while time.time() - start < DURATION:
    round_count += 1
    accuracy = total_correct / max(total_tests, 1)

    for inp, expected in tests:
        # teach (내부에서 fire 포함)
        resp = ssn("/teach", {"input": inp, "target": expected})
        if not resp:
            continue
        out = resp.get("output", "")
        fire_id = resp.get("fire_id", 0)
        if not fire_id:
            continue

        total_tests += 1

        if out == expected:
            total_correct += 1
            ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 1.0})
        elif not out:
            # 무출력 — 정답률에 따라 약화
            if accuracy > 0.7:
                ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.3})
            elif accuracy > 0.3:
                ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.1})
            # 정답률 30% 이하면 약화 없음
        else:
            score = 0.0
            if expected in out:
                score += 0.5
            if out and out[0] == expected[0]:
                score += 0.2
            len_diff = abs(len(out) - len(expected))
            if len_diff <= 3:
                score += 0.3 / (1.0 + len_diff)

            if score > 0:
                ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": score})
            else:
                # 정답률에 따라 약화
                if accuracy > 0.7:
                    ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.3})
                elif accuracy > 0.3:
                    ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.1})

    elapsed = int(time.time() - start)
    if round_count % 100 == 0:
        accuracy = total_correct / max(total_tests, 1)
        print(f"--- [{elapsed}초, 라운드 {round_count}, 정답률 {accuracy:.1%}] ---")
        for inp, expected in tests:
            resp = ssn("/fire", {"text": inp})
            out = resp.get("output", "") if resp else ""
            mark = "✓" if out == expected else "✗"
            print(f"  {mark} {inp} → \"{out}\"")
        st = ssn("/status") or {}
        print(f"  fire: {st.get('fire_count',0)}, 시냅스: {st.get('synapses',0)}, threshold: {st.get('threshold','?')}")
        print()

ssn("/save", {})
print(f"정답률: {total_correct}/{total_tests} = {total_correct/max(total_tests,1):.1%}")
print("저장 완료.")
