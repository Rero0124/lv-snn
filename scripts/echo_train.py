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
from scripts_util import jamo_overlap, completion_score  # noqa: E402

# 에코 커리큘럼: 단어부터 시작 — 초성 1개는 입력 뉴런 1개라 신호가 약해 출력까지
# 못 간다. 단어는 자모 여러 개로 분해돼 입력 뉴런이 여러 개 켜지므로 신호가 강하다.
# 출력 충돌(공유 자모)을 줄이려 비교적 구별되는 2음절 단어 소수로 시작.
TOKENS = [
    ("안녕", "안녕"),   # ㅇㅏㄴ ㄴㅕㅇ
    ("감사", "감사"),   # ㄱㅏㅁ ㅅㅏ
    ("사랑", "사랑"),   # ㅅㅏ ㄹㅏㅇ
    ("행복", "행복"),   # ㅎㅐㅇ ㅂㅗㄱ
    ("친구", "친구"),   # ㅊㅣㄴ ㄱㅜ
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
    ap.add_argument("--match-bonus", type=float, default=2.0,
                    help="정확 에코 시 추가 보상(+)")
    ap.add_argument("--jamo-reward", type=float, default=1.0,
                    help="자모 겹침 비율 × 이 값 = 보상(+). 자모 1개만 맞아도 + (길이보다 배점 큼)")
    ap.add_argument("--syllable-bonus", type=float, default=0.07,
                    help="완성형 음절을 만들면 추가 보상(+). 음절 정확=최대")
    ap.add_argument("--syllable-partial", type=float, default=0.3,
                    help="자모는 같고 순서만 다른 음절(안↔낭)에 줄 부분 비율 (0~1)")
    ap.add_argument("--len-reward", type=float, default=0.4,
                    help="길이가 목표와 정확히 같을 때 점수(+, 최댓값)")
    ap.add_argument("--len-penalty", type=float, default=0.1,
                    help="목표와 글자 수가 1개 차이날 때마다 감점(-)")
    ap.add_argument("--len-cap", type=float, default=2.0,
                    help="길이 점수 하한(크기)")
    ap.add_argument("--score-max", type=float, default=3.0, help="최종 총점 상한")
    ap.add_argument("--score-min", type=float, default=-2.0, help="최종 총점 하한")
    ap.add_argument("--baseline-ema", type=float, default=0.7,
                    help="점수 baseline(토큰별 이동평균) 갱신 계수. 피드백 = 점수 − baseline")
    ap.add_argument("--no-neg", action="store_true", help="감점(-) 끔: 길이 점수 음수 차단")
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
    print(f"자모(+): 정확 {args.match_bonus} + 겹침×{args.jamo_reward}  (길이보다 배점 큼)")
    print(f"완성형(+): 음절 {args.syllable_bonus} (순서 다르면 ×{args.syllable_partial})")
    print(f"길이(±): 정확/짧음 +{args.len_reward}, 길면 초과글자당 -{args.len_penalty} (하한 -{args.len_cap}{', 음수차단' if args.no_neg else ''})")
    print(f"총점 범위: [{args.score_min}, {args.score_max}]")
    print(f"피드백: (자모+완성형 − 토큰별 평균 baseline EMA {args.baseline_ema}) + 길이(별도)")
    print()

    start = time.time()
    rnd = 0
    fires = 0
    pos = neg = 0
    # 토큰별 최근 적중 이력(슬라이딩) — 마지막 정확 일치 여부
    last_hit = {inp: 0 for inp, _ in TOKENS}
    window_hits = []  # 라운드별 정확 일치 수
    # 피드백 점수 누적 (무출력=0점 포함, fire 단위)
    score_sum = 0.0   # 전체 누적 점수 합
    score_n = 0       # 전체 fire 수
    win_score_sum = 0.0  # 이번 리포트 구간 점수 합
    win_score_n = 0
    # 토큰별 점수 baseline(이동평균) — 최종 피드백은 이 평균 대비 변화량(advantage)
    baseline = {inp: None for inp, _ in TOKENS}

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
            last_hit[inp] = 1 if match else 0
            if match:
                round_hits += 1

            # ── 피드백 = 여러 조건을 합산한 단일 값 ──
            # 무출력(empty)은 강화할 경로도 없고 감점도 주지 않음 → 0점(피드백 생략).
            score = 0.0
            if out:
                # (+) 자모 겹침 보상: 비율 × jamo_reward (1개만 맞아도 +, 정확 일치면 비율 1.0)
                reward = args.jamo_reward * jamo_overlap(out, exp)
                # (+) 정확 에코 추가 보상
                if match:
                    reward += args.match_bonus
                # (+) 완성형 음절 보너스: 조합된 음절을 만들면 가산 (순서 달라도 일부 인정)
                reward += args.syllable_bonus * completion_score(out, exp, args.syllable_partial)
                # (±) 길이 점수 — baseline 과 무관하게 별도 계산.
                #   목표보다 길 때만 감점(짧으면 감점 없음). 정확/짧음 = +len_reward,
                #   초과 1글자당 -len_penalty, 하한 -len_cap. no_neg 면 음수 차단.
                excess = max(0, len(out) - len(exp))
                len_score = args.len_reward - args.len_penalty * excess
                len_score = max(0.0, len_score) if args.no_neg else max(-args.len_cap, len_score)

                # reward(자모+정확+완성형)만 그 토큰의 평균(baseline) 대비 변화량으로,
                # 길이는 그대로 더한다. 평균보다 좋으면 +, 나쁘면 -.
                b = baseline[inp]
                reward_adv = 0.0 if b is None else reward - b
                # baseline(이동평균) 갱신: 첫 출력은 그대로, 이후 EMA (reward 기준)
                baseline[inp] = reward if b is None else (
                    args.baseline_ema * b + (1 - args.baseline_ema) * reward)

                # 최종 피드백 = reward 변화량 + 길이 점수, 총점 상·하한 clamp
                final = max(args.score_min, min(args.score_max, reward_adv + len_score))
                # 리포트용 품질 점수(절대값, baseline 무관): 자모+완성형+길이
                score = max(args.score_min, min(args.score_max, reward + len_score))

                # final 부호로 +/- feedback (0 이면 생략)
                if final > 0:
                    req(url, "/feedback", {"fire_id": fid, "positive": True,
                                           "strength": final})
                    pos += 1
                elif final < 0:
                    req(url, "/feedback", {"fire_id": fid, "positive": False,
                                           "strength": -final})
                    neg += 1

            # 점수 누적 (무출력은 0점으로 포함)
            score_sum += score
            score_n += 1
            win_score_sum += score
            win_score_n += 1
            if args.delay:
                time.sleep(args.delay)  # fatigue 회복용 발화 간격

        window_hits.append(round_hits)

        if rnd % args.report == 0 or (args.rounds and rnd == args.rounds):
            elapsed = int(time.time() - start)
            recent = window_hits[-args.report:]
            acc = sum(recent) / (len(recent) * len(TOKENS))
            cur_hit = sum(last_hit.values())
            st = req(url, "/status") or {}
            win_avg = win_score_sum / win_score_n if win_score_n else 0.0
            tot_avg = score_sum / score_n if score_n else 0.0
            print(f"--- 라운드 {rnd} ({elapsed}s) ---")
            print(f"  정확도(최근{len(recent)}라운드): {acc:.1%} | 현재 적중 토큰: {cur_hit}/{len(TOKENS)}")
            print(f"  평균 점수(품질, baseline 전): 이번턴 {win_avg:+.3f} | 전체 {tot_avg:+.3f}")
            print(f"  fire {fires} | +{pos} -{neg} | 시냅스 {st.get('synapses', 0)}")
            win_score_sum = 0.0  # 이번턴 누적 리셋
            win_score_n = 0

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
