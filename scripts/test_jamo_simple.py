#!/usr/bin/env python3
"""초성 단순 반복 학습 — ㄱ→ㄱ, ㄴ→ㄴ 이 되는지"""

import json
import urllib.request
import time
import sys
import io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:8081"

def ssn(path, data=None):
    url = f"{SSN_URL}{path}"
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

tests = [
    ("ㄱ", "ㄱ"),
    ("ㄴ", "ㄴ"),
    ("ㄷ", "ㄷ"),
    ("ㅁ", "ㅁ"),
    ("ㅂ", "ㅂ"),
]

DURATION = 600  # 10분
start = time.time()
round_count = 0

print(f"=== 초성 학습 ({DURATION}초) ===\n")

while time.time() - start < DURATION:
    round_count += 1
    for inp, target in tests:
        for _ in range(50):
            ssn("/teach", {"input": inp, "target": target})

    elapsed = int(time.time() - start)
    if elapsed > 0 and round_count % 10 == 0:
        print(f"--- [{elapsed}초, 라운드 {round_count}] ---")
        for inp, expected in tests:
            resp = ssn("/fire", {"text": inp})
            out = resp.get("output", "")
            mark = "✓" if out == expected else "✗"
            print(f"  {mark} {inp} → \"{out}\"")
        print()

ssn("/save")
print("저장 완료.")
