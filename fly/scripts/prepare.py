#!/usr/bin/env python3
"""FlyWire 783 커넥톰(Shiu et al. 공개본) → fly 크레이트용 바이너리 변환.

사용: uv run --with pandas --with pyarrow python scripts/prepare.py data/shiu data

출력 (little-endian):
  data/ids.bin   : u64 × N           — 인덱스 → FlyWire root id
  data/conn.bin  : u64 M, 이후 (u32 pre, u32 post, i32 signed_count) × M
                   signed_count = 시냅스 수 × (+1 흥분 / -1 억제)
"""
import sys
import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
comp = pd.read_csv(f"{src}/Completeness_783.csv", index_col=0)
con = pd.read_parquet(f"{src}/Connectivity_783.parquet")

comp.index.values.astype("<u8").tofile(f"{dst}/ids.bin")

rec = np.empty(len(con), dtype=[("pre", "<u4"), ("post", "<u4"), ("w", "<i4")])
rec["pre"] = con["Presynaptic_Index"].values
rec["post"] = con["Postsynaptic_Index"].values
rec["w"] = con["Excitatory x Connectivity"].values
with open(f"{dst}/conn.bin", "wb") as f:
    np.array([len(rec)], dtype="<u8").tofile(f)
    rec.tofile(f)
print(f"neurons {len(comp)}, connections {len(rec)}")
