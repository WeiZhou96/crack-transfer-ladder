"""Post hoc sensitivity (supplementary S3): PaveDistress background-test comparisons with 100 m route clusters.

Reads the existing per-image accounts only (no inference, no threshold change). Arm density = mean over seeds of
sum(FP components) / sum(Mpx); paired cluster bootstrap: one multinomial resample of clusters per replicate, applied
to every model; 10000 replicates, seed 0; percentile 95% interval of the relative change. The 1 km clustering is
recomputed with the same code as a check against the registered report.
usage: python cluster100m_v1.py <out.json>
"""
import os
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CRACKCF_DATA", "data"))
EVAL = ["students_eval_v1_20261028", "students_eval_ablation_v1_20261028", "students_eval_add3_v1", "students_eval_add6_v1"]
ARMS = ["B0", "B1", "B2", "B3_auto", "B10", "B11", "B0_v6", "A_x01only", "A_x1only", "A_paired_noaux",
        "R50_B0", "R50_B2", "R50_B11", "B1_v5", "A_x01only_B1", "B2_g2", "B2_g7", "B12_auto_posthoc"]
COMP = [("B1", "B0"), ("B2", "B0"), ("B3_auto", "B0"), ("B10", "B0"), ("B11", "B0"), ("B11", "B3_auto"),
        ("B11", "B0_v6"), ("A_x01only", "B0"), ("A_x1only", "B0"), ("A_paired_noaux", "B0"), ("B2_g2", "B0"),
        ("B2_g7", "B0"), ("B12_auto_posthoc", "B0"), ("R50_B11", "R50_B0"), ("R50_B2", "R50_B0"), ("B1_v5", "B0"),
        ("A_x01only_B1", "B1_v5")]
LEG, OP = "pavedistress_background_test", "native_r95"


def find(model):
    for e in EVAL:
        p = ROOT / e / "test" / model / f"{LEG}.csv"
        if p.exists():
            return p
    return None


order, groups, fp, mpx = None, {}, {}, {}
for a in ARMS:
    for s in range(3):
        m = f"{a}_s{s}"
        p = find(m)
        if p is None:
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        ids = [r["image_id"] for r in rows]
        if order is None:
            order = ids
            groups = {g: [r[g] for r in rows] for g in ("bootstrap_group_1km", "bootstrap_group_100m")}
        assert ids == order, m
        fp[m] = np.array([float(r[f"{OP}__fp_components"]) for r in rows])
        mpx[m] = np.array([float(r[f"{OP}__mpx"]) for r in rows])
seeds = {a: [f"{a}_s{s}" for s in range(3) if f"{a}_s{s}" in fp] for a in ARMS}
out = {"leg": LEG, "op": OP, "n_images": len(order), "iters": 10000, "seed": 0, "clusters": {}}
for gcol in ("bootstrap_group_1km", "bootstrap_group_100m"):
    keys = sorted(set(groups[gcol]))
    pos = {g: i for i, g in enumerate(keys)}
    gi = np.array([pos[g] for g in groups[gcol]])
    G = len(keys)
    F = {m: np.bincount(gi, weights=fp[m], minlength=G) for m in fp}
    A = {m: np.bincount(gi, weights=mpx[m], minlength=G) for m in fp}
    rng = np.random.default_rng(0)
    W = rng.multinomial(G, np.full(G, 1 / G), size=10000).astype(float)

    def arm_point(a):
        return float(np.mean([F[m].sum() / A[m].sum() for m in seeds[a]]))

    def arm_boot(a):
        return np.mean([(W @ F[m]) / (W @ A[m]) for m in seeds[a]], axis=0)

    res = {}
    for x, y in COMP:
        if not seeds[x] or not seeds[y]:
            continue
        rc = arm_point(x) / arm_point(y) - 1
        bx, by = arm_boot(x), arm_boot(y)
        ok = by > 0
        lo, hi = np.percentile(bx[ok] / by[ok] - 1, [2.5, 97.5])
        res[f"{x}_vs_{y}"] = {"rel_change": rc, "ci95": [float(lo), float(hi)], "n_seeds": [len(seeds[x]), len(seeds[y])]}
    out["clusters"][gcol] = {"n_groups": G, "comparisons": res}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for gcol, v in out["clusters"].items():
    print(gcol, v["n_groups"])
    for k, r in v["comparisons"].items():
        print(f"  {k:28s} {100*r['rel_change']:+8.1f}  [{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}]")
