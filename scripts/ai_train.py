#!/usr/bin/env python3
"""
LV-SNN v2 학습 스크립트
- Ollama가 입력 생성 → SNN 응답 → Ollama 평가
- 점수 높을수록 강하게 강화 (약화 없음)
- 실패 시 teach
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
from concurrent.futures import ThreadPoolExecutor

socket.setdefaulttimeout(120)
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

SSN_URL = "http://127.0.0.1:8081"
OLLAMA_SERVERS = [
#    {"url": "http://192.168.0.31:11434", "model": "exaone3.5:7.8b"},
#    {"url": "http://127.0.0.1:11434", "model": "exaone3.5:7.8b"},
    {"url": "http://192.168.55.150:11434", "model": "exaone3.5:7.8b"},
]
THREADS_PER_SERVER = 2
MAX_RETRY = 5

_server_lock = threading.Lock()
_server_idx = 0

def next_server():
    global _server_idx
    with _server_lock:
        srv = OLLAMA_SERVERS[_server_idx % len(OLLAMA_SERVERS)]
        _server_idx += 1
        return srv


def ollama(prompt, max_tokens=100, server=None):
    srv = server or next_server()
    try:
        req = urllib.request.Request(
            f"{srv['url']}/api/generate",
            data=json.dumps({
                "model": srv["model"], "prompt": prompt,
                "stream": False, "options": {"num_predict": max_tokens}
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["response"].strip()
    except Exception as e:
        print(f"  [ollama 오류] {e}", file=sys.stderr)
        return None


def generate_input(topic, recent="", server=None):
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
    if not output_text or len(output_text) < 2:
        return 0.0
    prompt = (
        f"A가 말했고 B가 대답했다. B의 응답 품질을 0.0~1.0 점수로 평가.\n"
        f"A:\"{input_text}\"\nB:\"{output_text}\"\n\n"
        f"채점 기준:\n"
        f"- 주제 일치, 상황 적합, 자연스러움\n"
        f"- 0.7 이상은 정말 잘한 경우만\n"
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


def suggest_response(input_text, server=None):
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


_ssn_lock = threading.Lock()

def ssn_request_locked(path, data=None):
    with _ssn_lock:
        return ssn_request(path, data)


def process_one_input(topic, context, stats, server=None):
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

        if not out:
            # 무출력만 약화
            if fire_id:
                ssn_request_locked("/feedback", {
                    "fire_id": fire_id, "positive": False, "strength": 0.3
                })
                with stats["lock"]:
                    stats["total_neg"] += 1
            print(f"    {attempt}) (무출력, 약화)")
            continue

        if out in tried_outputs:
            print(f"    {attempt}) \"{out[:40]}\" (중복, skip)")
            continue

        tried_outputs.append(out)
        score = judge(inp, out, server=server)

        # 출력이 나오면 약화 안 함 — 점수에 비례해서 강화
        if fire_id and score > 0:
            ssn_request_locked("/feedback", {
                "fire_id": fire_id, "positive": True, "strength": score
            })
            with stats["lock"]:
                stats["total_pos"] += 1
        elif fire_id and score == 0:
            # 점수 0이라도 출력 있으면 아주 약하게 강화
            ssn_request_locked("/feedback", {
                "fire_id": fire_id, "positive": True, "strength": 0.05
            })
            with stats["lock"]:
                stats["total_pos"] += 1

        if score >= 0.6:
            found = True
            with stats["lock"]:
                stats["found_at"].append(attempt)
            print(f"    {attempt}) \"{out[:60]}\" ✓ {score:.1f}")
            with stats["lock"]:
                context.append(f"{inp}→{out}")
            break
        else:
            print(f"    {attempt}) \"{out[:60]}\" ✗ {score:.1f}")

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

    print(f"\n=== LV-SNN v2 학습 ({args.duration}초) ===")
    print(f"주제: {', '.join(topics)}")
    print(f"서버: 뉴런 {status['neurons']} | 시냅스 {status['synapses']} | 어휘 {status['vocab_size']}")
    print(f"LLM: {len(OLLAMA_SERVERS)}대 × {THREADS_PER_SERVER}스레드 = {total_threads} 병렬")
    print(f"방식: fire → ≥0.4 강화(점수비례), 전패→teach (약화 없음)")
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

    server_assignments = []
    for srv in OLLAMA_SERVERS:
        for _ in range(THREADS_PER_SERVER):
            server_assignments.append(srv)
    thread_idx = 0

    with ThreadPoolExecutor(max_workers=total_threads) as executor:
        futures = []

        while time.time() - start < args.duration:
            futures = [f for f in futures if not f.done()]

            while len(futures) < total_threads and time.time() - start < args.duration:
                topic = topics[topic_idx % len(topics)]
                topic_idx += 1
                srv = server_assignments[thread_idx % len(server_assignments)]
                thread_idx += 1
                future = executor.submit(process_one_input, topic, context, stats, server=srv)
                futures.append(future)

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
                print(f"  입력: {ti} | 발화: {tf} | teach: {tt} | 강화: {tp} | 약화: {tn}")
                print(f"  자력 성공: {success_rate:.0%} ({len(fa)}/{ti})")
                print(f"  평균 시도: {avg_tries:.1f}회 | 최근10: {sum(r10)/len(r10):.1f}회")
                print(f"  뉴런: {st.get('neurons',0)} | 시냅스: {st.get('synapses',0)} | 어휘: {st.get('vocab_size',0)}")
                print()

            time.sleep(0.5)

    ssn_request("/save")
    elapsed = int(time.time() - start)
    with stats["lock"]:
        ti = stats["total_inputs"]
        fa = list(stats["found_at"])
        tt = stats["total_teaches"]
        tp = stats["total_pos"]
    success_rate = len(fa) / max(ti, 1)
    print(f"\n{'='*50}")
    print(f"=== 학습 완료 ({elapsed}초) ===")
    print(f"  입력: {ti} | teach: {tt} | 강화: {tp}")
    print(f"  자력 성공률: {success_rate:.0%} ({len(fa)}/{ti})")
    print("\n저장 완료.")


if __name__ == "__main__":
    main()
