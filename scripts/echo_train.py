#!/usr/bin/env python3
"""에코 학습 (fire + feedback) — teach 없이 강화학습식으로 입력→동일 출력 학습.

흐름: fire(입력) → 출력 채점(에코 일치?) → +/- feedback 반복.
출력 뉴런이 노이즈/자발발화로 우연히 정답을 내면 +feedback으로 그 경로를 강화,
틀린 출력은 -feedback으로 억제한다. 역전파 없이 국소 보상 학습이라 수렴이 느리므로
시간/라운드에 따른 정확도 곡선을 추적한다.
"""

import json
import urllib.request
import sys
import time
import argparse
import os
import io as _io

sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts_util import jamo_overlap  # noqa: E402

# 에코 커리큘럼: 단어부터 시작 — 초성 1개는 입력 뉴런 1개라 신호가 약해 출력까지
# 못 간다. 단어는 자모 여러 개로 분해돼 입력 뉴런이 여러 개 켜지므로 신호가 강하다.
# 출력 충돌(공유 자모)을 줄이려 비교적 구별되는 2음절 단어 소수로 시작.
TOKENS = [
    ("안녕", "안녕"),   # ㅇㅏㄴ ㄴㅕㅇ
    # ("감사", "감사"),   # ㄱㅏㅁ ㅅㅏ
    # ("사랑", "사랑"),   # ㅅㅏ ㄹㅏㅇ
    # ("행복", "행복"),   # ㅎㅐㅇ ㅂㅗㄱ
    # ("친구", "친구"),   # ㅊㅣㄴ ㄱㅜ
]


def req(url, path, data=None, timeout=30):
    full = f"{url}{path}"
    if data is not None:
        r = urllib.request.Request(
            full, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
    else:
        r = urllib.request.Request(full)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [오류] {path}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--rounds", type=int, default=15000, help="라운드 수 (0=시간 기반)")
    ap.add_argument("--duration", type=int, default=0, help="학습 시간(초), rounds=0 일 때")
    ap.add_argument("--report", type=int, default=50, help="리포트 주기(라운드)")
    ap.add_argument("--pos-strength", type=float, default=1.0)
    ap.add_argument("--neg-strength", type=float, default=0.5)
    ap.add_argument("--partial-th", type=float, default=0.5,
                    help="이 자모겹침 이상이면 약한 +, 미만이면 -")
    ap.add_argument("--no-neg", action="store_true", help="음성 feedback 끔(정답만 강화)")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="발화 간 간격(초) — fatigue 회복용 (0=연속발화, 억눌림)")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    st = req(url, "/status")
    if not st:
        print("서버가 실행 중이지 않습니다.")
        return
    print("=== 에코 학습 (fire + feedback, teach 없음) ===")
    print(f"토큰 {len(TOKENS)}개 | 뉴런 {st['neurons']} | 시냅스 {st['synapses']} | 어휘 {st['vocab_size']}")
    print(f"보상: +{args.pos_strength} / -{0 if args.no_neg else args.neg_strength} | partial_th={args.partial_th}")
    print()

    start = time.time()
    rnd = 0
    fires = 0
    pos = neg = 0
    # 토큰별 최근 적중 이력(슬라이딩) — 마지막 정확 일치 여부
    last_hit = {inp: 0 for inp, _ in TOKENS}
    window_hits = []  # 라운드별 정확 일치 수

    while True:
        rnd += 1
        if args.rounds > 0 and rnd > args.rounds:
            break
        if args.rounds == 0 and time.time() - start >= args.duration:
            break

        round_hits = 0
        for inp, exp in TOKENS:
            r = req(url, "/fire", {"text": inp})
            if not r:
                continue
            fires += 1
            out = r.get("output", "")
            fid = r.get("fire_id", 0)

            match = exp in out and out != ""
            if match:
                req(url, "/feedback", {"fire_id": fid, "positive": True,
                                       "strength": args.pos_strength})
                pos += 1
                round_hits += 1
                last_hit[inp] = 1
            elif out:
                score = jamo_overlap(out, exp)
                if score >= args.partial_th:
                    req(url, "/feedback", {"fire_id": fid, "positive": True,
                                           "strength": args.pos_strength * score})
                    pos += 1
                elif not args.no_neg:
                    req(url, "/feedback", {"fire_id": fid, "positive": False,
                                           "strength": args.neg_strength})
                    neg += 1
                last_hit[inp] = 0
            else:
                last_hit[inp] = 0
            # 출력 없음(empty)이면 강화할 경로가 없어 feedback 생략
            if args.delay:
                time.sleep(args.delay)  # fatigue 회복용 발화 간격

        window_hits.append(round_hits)

        if rnd % args.report == 0 or (args.rounds and rnd == args.rounds):
            elapsed = int(time.time() - start)
            recent = window_hits[-args.report:]
            acc = sum(recent) / (len(recent) * len(TOKENS))
            cur_hit = sum(last_hit.values())
            st = req(url, "/status") or {}
            print(f"--- 라운드 {rnd} ({elapsed}s) ---")
            print(f"  정확도(최근{len(recent)}라운드): {acc:.1%} | 현재 적중 토큰: {cur_hit}/{len(TOKENS)}")
            print(f"  fire {fires} | +{pos} -{neg} | 시냅스 {st.get('synapses', 0)}")

    # 최종 평가: feedback 없이 fire만
    print()
    print("=== 최종 평가 (feedback 없이 fire만) ===")
    final_hits = 0
    for inp, exp in TOKENS:
        r = req(url, "/fire", {"text": inp}) or {}
        out = r.get("output", "")
        ok = exp in out and out != ""
        if ok:
            final_hits += 1
        print(f"  {'✓' if ok else '✗'} fire({inp}) = \"{out}\"")
        if args.delay:
            time.sleep(args.delay)
    print(f"\n최종: {final_hits}/{len(TOKENS)} 정확 에코")
    req(url, "/save", data={})  # /save 는 POST
    print("저장 완료.")


if __name__ == "__main__":
    main()
