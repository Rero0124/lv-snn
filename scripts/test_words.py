#!/usr/bin/env python3
"""단어 학습 테스트 — fire + feedback만으로 학습"""

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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

tests = [
    ("가", "가"),
    ("나", "나"),
    ("한", "한"),
    ("글", "글"),
    ("말", "말"),
]

DURATION = 600  # 10분
start = time.time()
round_count = 0

print(f"=== fire + feedback 학습 ({DURATION}초) ===\n")

while time.time() - start < DURATION:
    round_count += 1

    for inp, expected in tests:
        resp = ssn("/fire", {"text": inp})
        if not resp:
            continue
        out = resp.get("output", "")
        fire_id = resp.get("fire_id", 0)

        if not fire_id:
            continue

        if out == expected:
            ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 1.0})
        elif out and out[0] == expected[0]:
            ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 0.3})
        elif len(out) == len(expected) and len(out) > 0:
            ssn("/feedback", {"fire_id": fire_id, "positive": True, "strength": 0.1})
        else:
            ssn("/feedback", {"fire_id": fire_id, "positive": False, "strength": 0.5})

    elapsed = int(time.time() - start)
    if round_count % 100 == 0:
        print(f"--- [{elapsed}초, 라운드 {round_count}] ---")
        for inp, expected in tests:
            resp = ssn("/fire", {"text": inp})
            out = resp.get("output", "") if resp else ""
            mark = "✓" if out == expected else "✗"
            print(f"  {mark} {inp} → \"{out}\"")
        st = ssn("/status") or {}
        print(f"  fire: {st.get('fire_count',0)}, 시냅스: {st.get('synapses',0)}, threshold: {st.get('threshold','?')}")
        print()

ssn("/save", {})
print("저장 완료.")
