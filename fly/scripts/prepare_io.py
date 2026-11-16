#!/usr/bin/env python3
"""전뇌 실험용 입출력 뉴런 목록.

사용: uv run --with pandas python scripts/prepare_io.py data
출력: data/io_neurons.tsv — 전뇌 인덱스, 역할(in/out), 그룹
  in  : 감각뉴런 (super_class == sensory). 그룹 = cell_type. 뉴런 8개 이상인 타입만, 타입당 최대 20개
  out : 하행뉴런/운동뉴런 (super_class in descending, motor). 그룹 = cell_type
"""
import sys
import numpy as np
import pandas as pd

d = sys.argv[1]
ids = np.fromfile(f"{d}/ids.bin", dtype="<u8")
index = {int(r): i for i, r in enumerate(ids)}
a = pd.read_csv(f"{d}/annot/neuron_annotations.tsv", sep="\t", low_memory=False)
a = a[a.root_id.isin(index)].sort_values("root_id")
print("super_class:", a.super_class.value_counts().to_dict())

sens = a[(a.super_class == "sensory") & a.cell_type.notna()]
print("sensory cell_class:", sens.cell_class.value_counts().to_dict())
rows = []
sizes = sens.cell_type.value_counts()
for t in sizes[sizes >= 8].index:
    for r in sens[sens.cell_type == t].root_id.head(20):
        rows.append((index[int(r)], "in", t))
out = a[a.super_class.isin(["descending", "motor"])]
for _, r in out.iterrows():
    rows.append((index[int(r.root_id)], "out", str(r.cell_type)))
df = pd.DataFrame(rows, columns=["idx", "role", "group"])
df.to_csv(f"{d}/io_neurons.tsv", sep="\t", index=False)
print("input groups:", df[df.role == "in"].group.nunique(), "input neurons:", (df.role == "in").sum(),
      "output candidates:", (df.role == "out").sum())
