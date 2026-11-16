#!/usr/bin/env python3
"""세 커넥톰의 연결 비율 비교: 예쁜꼬마선충(Cook 2019), 초파리 유충(Winding 2023), 초파리 성체(FlyWire 783).

사용: uv run --with numpy python scripts/compare_connectomes.py data
  data/other/herm_full_edgelist.csv   (openworm/c302)
  data/other/larva/edges.csv          (networks.skewed.de/net/fly_larva)
  data/conn.bin                        (scripts/prepare.py)
화학 시냅스의 방향 있는 연결(뉴런 쌍)만 센다. 연결당 시냅스 수 = 그 쌍 사이의 시냅스 개수.
"""
import csv, sys
from collections import defaultdict
import numpy as np

d = sys.argv[1]

def stats(name, pre, post, cnt, n_neurons):
    cnt = np.asarray(cnt, dtype=np.int64)
    m = len(cnt)
    out_deg = np.bincount(pre, minlength=n_neurons)
    in_deg = np.bincount(post, minlength=n_neurons)
    in_syn = np.bincount(post, weights=cnt, minlength=n_neurons)
    srt = np.sort(cnt)[::-1]
    top1 = srt[: max(1, m // 100)].sum() / cnt.sum()
    # 가장 강한 입력 연결 하나가 그 뉴런 전체 입력 시냅스에서 차지하는 비율 (중앙값)
    strongest = np.zeros(n_neurons)
    np.maximum.at(strongest, post, cnt)
    has_in = in_syn > 0
    return [name, n_neurons, m, int(cnt.sum()),
            f"{m / n_neurons:.1f}", f"{np.median(in_deg[has_in]):.0f}",
            f"{cnt.mean():.1f}", f"{np.median(cnt):.0f}", f"{np.percentile(cnt, 90):.0f}",
            f"{np.percentile(cnt, 99):.0f}", int(cnt.max()),
            f"{100 * (cnt == 1).mean():.0f}%", f"{100 * top1:.0f}%",
            f"{np.median(in_syn[has_in]):.0f}", f"{100 * np.median(strongest[has_in] / in_syn[has_in]):.0f}%"]

rows = []

# 선충: 화학 시냅스, 뉴런(화학 시냅스의 pre 로 등장하는 세포)끼리만
edges = defaultdict(int)
with open(f"{d}/other/herm_full_edgelist.csv") as f:
    for r in csv.DictReader(f):
        if r["Type"].strip() == "chemical":
            edges[(r["Source"].strip(), r["Target"].strip())] += int(r["Weight"])
neurons = sorted({a for a, _ in edges})
idx = {n: i for i, n in enumerate(neurons)}
e = [(idx[a], idx[b], w) for (a, b), w in edges.items() if b in idx]
rows.append(stats("예쁜꼬마선충", *map(np.array, zip(*e)), len(neurons)))

# 유충: 같은 쌍의 병렬 연결(aa/ad/da/dd)을 합침
edges = defaultdict(int)
nodes = set()
with open(f"{d}/other/larva/edges.csv") as f:
    next(f)
    for line in f:
        a, b, c, _ = line.strip().split(",")
        edges[(int(a), int(b))] += int(c)
with open(f"{d}/other/larva/nodes.csv") as f:
    n_larva = sum(1 for _ in f) - 1
e = [(a, b, w) for (a, b), w in edges.items()]
rows.append(stats("초파리 유충", *map(np.array, zip(*e)), n_larva))

# 성체
n_adult = len(np.fromfile(f"{d}/ids.bin", dtype="<u8"))
rec = np.fromfile(f"{d}/conn.bin", dtype=[("pre", "<u4"), ("post", "<u4"), ("w", "<i4")], offset=8)
rows.append(stats("초파리 성체", rec["pre"], rec["post"], np.abs(rec["w"]), n_adult))

head = ["", "뉴런", "연결", "시냅스", "뉴런당 연결", "입력연결 중앙값", "연결당 평균", "중앙값", "상위10%", "상위1%", "최대",
        "시냅스1개 연결", "상위1% 연결의 시냅스 몫", "뉴런당 입력시냅스 중앙값", "최강 입력 1개의 몫(중앙값)"]
for i, h in enumerate(head):
    print(f"{h:<26}" + "".join(f"{str(r[i]):>14}" for r in rows))
