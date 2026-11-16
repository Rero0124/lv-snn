#!/usr/bin/env python3
"""전뇌 인덱스 → super_class (어느 뉴런 무리가 발화하는지 볼 때 사용).
사용: uv run --with pandas python scripts/prepare_classes.py data  → data/superclass.tsv"""
import sys
import numpy as np, pandas as pd
d = sys.argv[1]
ids = np.fromfile(f"{d}/ids.bin", dtype="<u8")
a = pd.read_csv(f"{d}/annot/neuron_annotations.tsv", sep="\t", low_memory=False, usecols=["root_id", "super_class"])
m = dict(zip(a.root_id, a.super_class))
with open(f"{d}/superclass.tsv", "w") as f:
    for r in ids:
        f.write(f"{m.get(int(r), 'unknown')}\n")
