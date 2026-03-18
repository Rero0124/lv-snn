#!/usr/bin/env python3
"""
LV-SNN AI 주도 학습 스크립트 (자율 탐색 방식, 멀티 Ollama 병렬)
- 복수 Ollama 서버에 라운드로빈으로 요청 → SNN 대기 시간 최소화
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
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 소켓 레벨 전역 타임아웃 (Ollama hang 방지)
socket.setdefaulttimeout(120)

sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:8081"

# 멀티 Ollama 서버 설정 (서버당 2스레드)
OLLAMA_SERVERS = [
    {"url": "http://192.168.0.31:11434", "model": "exaone3.5:7.8b"},
    {"url": "http://192.168.1.14:11434", "model": "exaone3.5:7.8b"},
]
THREADS_PER_SERVER = 2
MAX_RETRY = 5  # 최대 재시도 횟수

# 라운드로빈 서버 선택
_server_lock = threading.Lock()
_server_idx = 0

def next_server():
    global _server_idx
    with _server_lock:
        srv = OLLAMA_SERVERS[_server_idx % len(OLLAMA_SERVERS)]
        _server_idx += 1
        return srv


def ollama(prompt, max_tokens=100, server=None):
    """Ollama 호출 (특정 서버 또는 라운드로빈)"""
    srv = server or next_server()
    try:
        req = urllib.request.Request(
            f"{srv['url']}/api/generate",
            data=json.dumps({
                "model": srv["model"],
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


def generate_input(topic, recent="", server=None):
    """Ollama가 입력 한 줄 생성"""
    ctx = f"최근: {recent}\n" if recent else ""
    prompt = (
        f"{ctx}'{topic}' 관련 자연스러운 한국어 일상 대화 한마디. "
        f"반말이나 존댓말 자유. 1~2문장. 따옴표 없이 텍스트만."
    )
    result = ollama(prompt, max_tokens=60, server=server)
    if result:
        result = result.strip().strip('"\'')
        result = re.sub(r'^\d+[\.\)]\s*', '', result)
        if 1 < len(result) < 80:
            return result
    return None


def judge(input_text, output_text, server=None):
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
    result = ollama(prompt, max_tokens=5, server=server)
    if result:
        try:
            score = float(re.search(r'[01]\.?\d*', result.strip()).group())
            return min(1.0, max(0.0, score))
        except Exception:
            pass
    return 0.0


def judge_tokens(input_text, output_tokens, server=None):
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
    result = ollama(prompt, max_tokens=80, server=server)
    if not result:
        return []
    try:
        scores = json.loads(re.search(r'\[.*?\]', result.strip()).group())
        # 개수 불일치면 짧은 쪽에 맞춤
        pairs = min(len(scores), len(words))
        return [(w["token"], max(-1.0, min(1.0, float(s)))) for w, s in zip(words[:pairs], scores[:pairs])]
    except Exception:
        return []


def reorder_tokens(input_text, tokens, server=None):
    """긍정 평가된 토큰들을 자연스러운 순서로 재배열"""
    token_list = ", ".join(f'"{t}"' for t in tokens)
    prompt = (
        f"A가 \"{input_text}\"라고 말했다.\n"
        f"B가 이 단어들로 응답하려 한다: [{token_list}]\n"
        f"이 단어들만 사용해서 자연스러운 한국어 응답 1문장을 만들어라. "
        f"단어를 빼거나 새로 추가하지 말고 순서만 바꿔라. 반말. 따옴표 없이 텍스트만."
    )
    result = ollama(prompt, max_tokens=60, server=server)
    if result:
        result = result.strip().strip('"\'')
        if 1 < len(result) < 80:
            return result
    return None


def find_natural_cutoff(input_text, output_text, server=None):
    """출력에서 앞부분부터 자연스러운 지점까지 잘라서 평가.
    반환: (truncated_text, score) 또는 None"""
    # 공백 기준 토큰 분리
    words = output_text.split()
    if len(words) <= 3:
        return None  # 이미 짧으면 자를 필요 없음

    # LLM에게 앞에서부터 자연스러운 끊김 지점 찾기
    prompt = (
        f"A가 \"{input_text}\"라고 말했다.\n"
        f"B의 응답: \"{output_text}\"\n\n"
        f"B의 응답을 앞에서부터 읽을 때, 자연스러운 문장이 완성되는 지점에서 잘라라.\n"
        f"잘린 응답만 출력. 따옴표 없이 텍스트만."
    )
    result = ollama(prompt, max_tokens=60, server=server)
    if not result:
        return None

    truncated = result.strip().strip('"\'')
    # 잘린 결과가 원본보다 짧고 의미있는 길이인지 확인
    if not truncated or len(truncated) >= len(output_text) or len(truncated) < 2:
        return None

    # 잘린 버전 재평가
    score = judge(input_text, truncated, server=server)
    if score >= 0.5:
        return (truncated, score)
    return None


def suggest_response(input_text, server=None):
    """모든 재시도 실패 시에만 호출 — Ollama가 정답 제안"""
    prompt = f"\"{input_text}\"에 대한 자연스러운 한국어 응답 1문장. 반말. 따옴표 없이 텍스트만."
    result = ollama(prompt, max_tokens=50, server=server)
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [SSN 오류] {e}", file=sys.stderr)
        return None


# SNN 요청은 직렬이므로 lock 사용
_ssn_lock = threading.Lock()

def ssn_request_locked(path, data=None):
    with _ssn_lock:
        return ssn_request(path, data)


def process_one_input(topic, context, stats, server=None):
    """하나의 입력을 처리 (스레드에서 실행, server 고정)"""
    ctx = " / ".join(context[-3:]) if context else ""
    inp = generate_input(topic, ctx, server=server)
    if not inp:
        return None

    with stats["lock"]:
        stats["total_inputs"] += 1
        idx = stats["total_inputs"]

    print(f"\n  [{idx}] \"{inp}\"")

    found = False
    tried_outputs = []

    for attempt in range(1, MAX_RETRY + 1):
        resp = ssn_request_locked("/fire", {"text": inp})
        if not resp:
            break
        with stats["lock"]:
            stats["total_fires"] += 1
        out = resp.get("output", "")
        fire_id = resp.get("fire_id", 0)

        if out in tried_outputs:
            if fire_id:
                ssn_request_locked("/feedback", {"fire_id": fire_id, "positive": False, "strength": 1.0})
                with stats["lock"]:
                    stats["total_neg"] += 1
            print(f"    {attempt}) \"{out}\" (중복, 약화)")
            continue

        tried_outputs.append(out)

        if not out:
            print(f"    {attempt}) (무출력)")
            continue

        score = judge(inp, out, server=server)

        if score >= 0.6:
            if fire_id:
                ssn_request_locked("/feedback", {"fire_id": fire_id, "positive": True, "strength": score})
                with stats["lock"]:
                    stats["total_pos"] += 1
            found = True
            with stats["lock"]:
                stats["found_at"].append(attempt)
            print(f"    {attempt}) \"{out}\" ✓ {score:.1f}")
            with stats["lock"]:
                context.append(f"{inp}→{out}")
            break
        elif score >= 0.3:
            output_tokens = resp.get("output_tokens", [])
            cutoff_done = False

            cutoff = find_natural_cutoff(inp, out, server=server)
            if cutoff and fire_id:
                trunc_text, trunc_score = cutoff
                cut_tokens = []
                for t in output_tokens:
                    if t["length"] < 2:
                        continue
                    if t["token"] in trunc_text:
                        cut_tokens.append((t["token"], 1.0))
                    else:
                        cut_tokens.append((t["token"], -0.5))
                if cut_tokens:
                    ssn_request_locked("/feedback_partial", {"fire_id": fire_id, "token_scores": cut_tokens})
                    if trunc_score >= 0.6:
                        ssn_request_locked("/feedback", {"fire_id": fire_id, "positive": True, "strength": trunc_score * 0.8})
                        with stats["lock"]:
                            stats["total_pos"] += 1
                        found = True
                        with stats["lock"]:
                            stats["found_at"].append(attempt)
                        print(f"    {attempt}) \"{out}\" ~ {score:.1f} → cut \"{trunc_text}\" ✓ {trunc_score:.1f}")
                        with stats["lock"]:
                            context.append(f"{inp}→{trunc_text}")
                        ssn_request_locked("/teach", {"input": inp, "target": trunc_text})
                        cutoff_done = True
                        break
                    else:
                        print(f"    {attempt}) \"{out}\" ~ {score:.1f} → cut \"{trunc_text}\" ({trunc_score:.1f}, +{len(cut_tokens)})")
                        cutoff_done = True

            if not cutoff_done:
                token_scores = judge_tokens(inp, output_tokens, server=server)
                if token_scores and fire_id:
                    ssn_request_locked("/feedback_partial", {"fire_id": fire_id, "token_scores": token_scores})
                    pos_t = sum(1 for _, s in token_scores if s > 0)
                    neg_t = sum(1 for _, s in token_scores if s < 0)
                    good_tokens = [tok for tok, s in token_scores if s > 0]
                    if len(good_tokens) >= 3:
                        reordered = reorder_tokens(inp, good_tokens, server=server)
                        if reordered:
                            ssn_request_locked("/teach", {"input": inp, "target": reordered})
                            print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t}) → reorder \"{reordered}\"")
                        else:
                            print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t})")
                    else:
                        print(f"    {attempt}) \"{out}\" ~ {score:.1f} (토큰 +{pos_t}/-{neg_t})")
                else:
                    print(f"    {attempt}) \"{out}\" ~ {score:.1f}")
        else:
            if fire_id:
                neg_strength = min(1.0, (1.0 - score) * 0.5 + attempt * 0.1)
                ssn_request_locked("/feedback", {"fire_id": fire_id, "positive": False, "strength": neg_strength})
                with stats["lock"]:
                    stats["total_neg"] += 1
            print(f"    {attempt}) \"{out}\" ✗ {score:.1f}")

    if not found:
        suggestion = suggest_response(inp, server=server)
        if suggestion:
            ssn_request_locked("/teach", {"input": inp, "target": suggestion})
            with stats["lock"]:
                stats["total_teaches"] += 1
            print(f"    → teach \"{suggestion}\"")
        else:
            print(f"    → (제안 실패)")
        with stats["lock"]:
            context.append(f"{inp}→?")

    return found


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

    # 모든 Ollama 서버 연결 확인
    available_servers = []
    for srv in OLLAMA_SERVERS:
        try:
            test_req = urllib.request.Request(
                f"{srv['url']}/api/generate",
                data=json.dumps({"model": srv["model"], "prompt": "test", "stream": False, "options": {"num_predict": 1}}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(test_req, timeout=10):
                available_servers.append(srv)
                print(f"  ✓ {srv['url']} ({srv['model']})")
        except Exception as e:
            print(f"  ✗ {srv['url']} ({srv['model']}): {e}")

    if not available_servers:
        print("사용 가능한 Ollama 서버가 없습니다.")
        return

    OLLAMA_SERVERS.clear()
    OLLAMA_SERVERS.extend(available_servers)

    total_threads = len(OLLAMA_SERVERS) * THREADS_PER_SERVER
    topics = [t.strip() for t in args.topic.split(",")]

    print(f"\n=== LV-SNN 자율 탐색 학습 ({args.duration}초) ===")
    print(f"주제: {', '.join(topics)}")
    print(f"서버: 시냅스 {status['synapses']} | 패턴 {status['patterns']}")
    print(f"LLM: {len(OLLAMA_SERVERS)}대 × {THREADS_PER_SERVER}스레드 = {total_threads} 병렬")
    print(f"방식: 입력 → SNN 탐색 (최대 {MAX_RETRY}회) → ≥0.6 강화, 0.5~0.6 무시, <0.5 약화, 전패 시 teach")
    print()

    start = time.time()
    context = []
    last_report = 0
    topic_idx = 0

    stats = {
        "lock": threading.Lock(),
        "total_inputs": 0,
        "total_fires": 0,
        "total_teaches": 0,
        "total_pos": 0,
        "total_neg": 0,
        "found_at": [],
    }

    # 스레드별 전담 서버 배정 (서버당 THREADS_PER_SERVER개)
    server_assignments = []
    for srv in OLLAMA_SERVERS:
        for _ in range(THREADS_PER_SERVER):
            server_assignments.append(srv)
    thread_idx = 0

    with ThreadPoolExecutor(max_workers=total_threads) as executor:
        futures = []

        while time.time() - start < args.duration:
            # 활성 작업이 total_threads 미만이면 새 작업 제출
            futures = [f for f in futures if not f.done()]

            while len(futures) < total_threads and time.time() - start < args.duration:
                topic = topics[topic_idx % len(topics)]
                topic_idx += 1
                srv = server_assignments[thread_idx % len(server_assignments)]
                thread_idx += 1
                future = executor.submit(process_one_input, topic, context, stats, server=srv)
                futures.append(future)

            # 60초 리포트
            elapsed = int(time.time() - start)
            if elapsed >= last_report + 60:
                last_report = elapsed
                st = ssn_request("/status") or {}
                with stats["lock"]:
                    ti = stats["total_inputs"]
                    tf = stats["total_fires"]
                    tt = stats["total_teaches"]
                    tp = stats["total_pos"]
                    tn = stats["total_neg"]
                    fa = list(stats["found_at"])
                success_rate = len(fa) / max(ti, 1)
                avg_tries = sum(fa) / max(len(fa), 1) if fa else 0
                r10 = fa[-10:] if len(fa) >= 10 else fa or [0]
                print(f"\n--- [{elapsed // 60}분 경과] ---")
                print(f"  입력: {ti} | 발화: {tf} | teach: {tt}")
                print(f"  강화: {tp} | 약화: {tn}")
                print(f"  자력 성공: {success_rate:.0%} ({len(fa)}/{ti})")
                print(f"  평균 시도: {avg_tries:.1f}회 | 최근10: {sum(r10)/len(r10):.1f}회")
                print(f"  시냅스: {st.get('synapses',0)} | 패턴: {st.get('patterns',0)}")
                print()

            time.sleep(0.5)  # 폴링 간격

    # 저장
    ssn_request("/save")
    elapsed = int(time.time() - start)
    with stats["lock"]:
        ti = stats["total_inputs"]
        tf = stats["total_fires"]
        tt = stats["total_teaches"]
        tp = stats["total_pos"]
        tn = stats["total_neg"]
        fa = list(stats["found_at"])
    success_rate = len(fa) / max(ti, 1)
    avg_tries = sum(fa) / max(len(fa), 1) if fa else 0
    print(f"\n{'='*50}")
    print(f"=== 자율 탐색 학습 완료 ({elapsed}초) ===")
    print(f"  입력: {ti} | 발화: {tf} | teach: {tt}")
    print(f"  강화: {tp} | 약화: {tn}")
    print(f"  자력 성공률: {success_rate:.0%} ({len(fa)}/{ti})")
    if fa:
        print(f"  평균 시도: {avg_tries:.1f}회")
        by_try = {}
        for t in fa:
            by_try[t] = by_try.get(t, 0) + 1
        dist = " | ".join(f"{k}번째:{v}회" for k, v in sorted(by_try.items()))
        print(f"  성공 분포: {dist}")
    if len(fa) > 20:
        f = fa[:min(20, len(fa))]
        l = fa[-min(20, len(fa)):]
        print(f"  초반 평균: {sum(f)/len(f):.1f}회 → 후반 평균: {sum(l)/len(l):.1f}회")
    print("\n저장 완료.")


if __name__ == "__main__":
    main()
