#!/usr/bin/env python3
"""자모/글자 단위 학습 테스트 — 입력한 것이 그대로 출력되는지 확인"""

import json
import urllib.request
import sys

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

# 테스트 쌍: (입력, 기대 출력)
tests = [
    # 1단계: 자모 → 같은 자모
    ("ㄱ", "ㄱ"),
    ("ㄴ", "ㄴ"),
    ("ㅏ", "ㅏ"),
    ("ㅓ", "ㅓ"),
    # 2단계: 한 글자 → 같은 글자
    ("가", "가"),
    ("나", "나"),
    ("다", "다"),
    ("마", "마"),
    # 3단계: 받침 있는 글자
    ("한", "한"),
    ("글", "글"),
    ("말", "말"),
    # 4단계: 두 글자
    ("안녕", "안녕"),
    ("감사", "감사"),
]

TEACH_REPEAT = 20  # teach 반복 횟수

print("=== 자모/글자 학습 테스트 ===\n")

for inp, expected in tests:
    # teach 반복
    for _ in range(TEACH_REPEAT):
        ssn("/teach", {"input": inp, "target": expected})

    # fire로 확인
    resp = ssn("/fire", {"text": inp})
    out = resp.get("output", "")
    match = "✓" if expected in out else "✗"
    print(f"  {match} teach({inp}→{expected}) x{TEACH_REPEAT} → fire({inp}) = \"{out}\"")

print()

# 전체 요약
print("=== 최종 확인 (teach 없이 fire만) ===\n")
for inp, expected in tests:
    resp = ssn("/fire", {"text": inp})
    out = resp.get("output", "")
    match = "✓" if expected in out else "✗"
    print(f"  {match} fire({inp}) = \"{out}\"")

print()
st = ssn("/status")
print(f"뉴런: {st['neurons']}, 시냅스: {st['synapses']}, 어휘: {st['vocab_size']}, fire: {st['fire_count']}")
