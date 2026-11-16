#!/usr/bin/env python3
"""버섯체(mushroom body) 서브회로 뉴런 목록 추출.

사용: uv run --with pandas python scripts/prepare_mb.py data
입력: data/ids.bin, data/annot/neuron_annotations.tsv (flyconnectome/flywire_annotations)
출력: data/mb_neurons.tsv — 전뇌 인덱스, 종류(PN/KC/MBON/APL), 세부 타입, 좌우
  PN   : 단일 사구체 투사뉴런 (세부 타입 = 사구체 이름) — 냄새 입력
  KC   : Kenyon cell — 희소 확장 코드
  MBON : 버섯체 출력뉴런 — 판독
  APL  : KC 전체를 되먹임 억제하는 GABA 뉴런
  DAN  : 도파민 뉴런 (PAM, PPL). 조절 뉴런 — 전위를 올리지 않고 학습만 조절
"""
import sys
import numpy as np
import pandas as pd

d = sys.argv[1]
ids = np.fromfile(f"{d}/ids.bin", dtype="<u8")
index = {int(r): i for i, r in enumerate(ids)}
a = pd.read_csv(f"{d}/annot/neuron_annotations.tsv", sep="\t", low_memory=False)
a = a[a.root_id.isin(index)]

rows = []
def add(df, kind, type_of):
    for _, r in df.iterrows():
        rows.append((index[int(r.root_id)], kind, type_of(r), r.side))

pn = a[(a.cell_class == "ALPN") & (a.cell_sub_class == "uniglomerular")]
add(pn, "PN", lambda r: str(r.cell_type).split("_")[0])
add(a[a.cell_class == "Kenyon_Cell"], "KC", lambda r: str(r.cell_type))
add(a[a.cell_class == "MBON"], "MBON", lambda r: str(r.cell_type))
add(a[a.cell_type == "APL"], "APL", lambda r: "APL")
# 도파민 뉴런: 예측 전달물질은 KC 를 도파민성으로 잘못 예측하므로 쓰지 않고 세포 분류(DAN)를 쓴다
add(a[a.cell_class == "DAN"], "DAN", lambda r: str(r.cell_type))

out = pd.DataFrame(rows, columns=["idx", "kind", "type", "side"]).sort_values("idx")
out.to_csv(f"{d}/mb_neurons.tsv", sep="\t", index=False)
print(out.kind.value_counts().to_dict())
print("glomeruli:", out[out.kind == "PN"].type.nunique(), " top_nt of classes:")
for cls in ["ALPN", "Kenyon_Cell", "MBON"]:
    print(" ", cls, a[a.cell_class == cls].top_nt.value_counts().head(3).to_dict())
