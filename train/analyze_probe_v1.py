"""Addendum 5 part A: scene-novel synthetic probe (P14).

Per arm and operator at native_r95 (fixed_0.5 also reported): x01 FP/Mpx, x00 FP/Mpx and operator-induced FP/Mpx
(x01 - x00); arm value = mean of its seeds. Contrasts vs B0: ratio of x01 and difference of operator-induced FP, with a
paired cluster bootstrap over background specimen groups (10000, seed 0; the same resample for every model).
usage: python analyze_probe_v1.py <report_dir> <quad_dir> <eval_dir>=<models.json> [...]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT, QD = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SUB = "quads_" + QD.name
OPS = ["shadow", "joint", "repair_boundary"]
B, SEED = 10000, 0

cluster = {}
for l in open(QD / "specs.jsonl", encoding="utf-8"):
    s = json.loads(l)
    cluster[s["group_id"]] = s["background"]["group_id"]
CL = sorted(set(cluster.values()))
CI = {c: i for i, c in enumerate(CL)}

models, arms = {}, defaultdict(list)
for spec in sys.argv[3:]:
    e, m = spec.split("=", 1)
    for k, v in json.load(open(m, encoding="utf-8")).items():
        p = Path(e) / "strata" / SUB / f"{k}.csv"
        if p.exists():
            models[k] = p
            arms[v["arm"]].append(k)

# per model, operator, threshold: cluster-level sums [fp_x01, fp_x00, mpx]
sums = {}
for k, p in models.items():
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for op in OPS:
        for t in ["native_r95", "fixed_0.5"]:
            a = np.zeros((len(CL), 3))
            for r in rows:
                if r["operator"] != op:
                    continue
                i = CI[cluster[r["group_id"]]]
                a[i] += [float(r[f"{t}__fp_x01"]), float(r[f"{t}__fp_x00"]), float(r["mpx"])]
            sums[k, op, t] = a

rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.integers(0, len(CL), len(CL)), minlength=len(CL)) for _ in range(B)]).astype(float)


def arm_vals(arm, op, t, w=None):
    """returns (x01, x00) arm means; w = bootstrap weight matrix (B x clusters) or None."""
    x1, x0 = [], []
    for k in arms[arm]:
        a = sums[k, op, t]
        if w is None:
            x1.append(a[:, 0].sum() / a[:, 2].sum())
            x0.append(a[:, 1].sum() / a[:, 2].sum())
        else:
            mp = w @ a[:, 2]
            x1.append((w @ a[:, 0]) / mp)
            x0.append((w @ a[:, 1]) / mp)
    return np.mean(x1, axis=0), np.mean(x0, axis=0)


def pct(v):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


table, comps = {}, {}
for t in ["native_r95", "fixed_0.5"]:
    table[t], comps[t] = {}, {}
    for a in sorted(arms):
        table[t][a] = {}
        for op in OPS:
            x1, x0 = arm_vals(a, op, t)
            n = int(sum(1 for g in cluster if g.startswith(op)))
            table[t][a][op] = {"x01": float(x1), "x00": float(x0), "induced": float(x1 - x0), "groups": n,
                               "seeds": len(arms[a])}
    if "B0" not in arms:
        continue
    for a in sorted(arms):
        if a == "B0":
            continue
        comps[t][a] = {}
        for op in OPS:
            x1a, x0a = arm_vals(a, op, t)
            x1b, x0b = arm_vals("B0", op, t)
            bx1a, bx0a = arm_vals(a, op, t, W)
            bx1b, bx0b = arm_vals("B0", op, t, W)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_b = bx1a / bx1b
            comps[t][a][op] = {"x01_ratio": float(x1a / x1b) if x1b > 0 else None, "x01_ratio_ci95": pct(ratio_b),
                               "induced_diff": float((x1a - x0a) - (x1b - x0b)),
                               "induced_diff_ci95": pct((bx1a - bx0a) - (bx1b - bx0b))}

p14 = None
if "B11" in arms and "B0" in arms:
    v = {op: [table["native_r95"]["B11"][op]["x01"], table["native_r95"]["B0"][op]["x01"]] for op in OPS}
    p14 = {"values": v, "verdict": all(b11 <= 0.5 * b0 for b11, b0 in v.values())}
report = {"probe": str(QD), "models": len(models), "clusters": len(CL), "table": table, "comparisons_vs_B0": comps,
          "predictions": {"P14_scene_novel_suppression": p14}, "method": __doc__}
(OUT / "probe_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"models": len(models), "P14": p14,
                  "native_r95": {a: {op: [round(v[op]["x01"], 2), round(v[op]["x00"], 2)] for op in OPS}
                                 for a, v in table["native_r95"].items()}}, indent=1))
