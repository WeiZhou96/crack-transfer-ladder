"""Addendum 4: false-alarm / recall frontier from eval_sweep_v1 outputs; matched-recall comparison (P12, P13).

Per model and threshold: positive-axis component recall R(t) (CrackSC+NHA12D) and A3_leg(t) on the three test legs.
Matched recall: for target r*, A3 is interpolated linearly in log(A3 + 0.01) against R between the two grid points
that bracket r* (R decreases with t); a model whose curve does not bracket r* is reported as not evaluable.
Arm value = mean over its seeds; per-seed values reported. Descriptive: no bootstrap.
usage: python analyze_sweep_v1.py <report_dir> <eval_dir>=<models.json> [...]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODELS, EDIR = {}, {}
for spec in sys.argv[2:]:
    e, m = spec.split("=", 1)
    for k, v in json.load(open(m, encoding="utf-8")).items():
        if (Path(e) / "sweep" / k / "DONE").exists():
            MODELS[k], EDIR[k] = v, Path(e)
LEGS = ["bcl_noise", "pavedistress_background_test", "uav1388_background"]
TARGETS = [0.70, 0.75, 0.80]


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


curves = {}
for m in MODELS:
    d = EDIR[m] / "sweep" / m
    pos = read(d / "positive_axis.csv")
    names = sorted({k.split("__")[0] for k in pos[0] if "__n_hit" in k})
    c = {}
    for n in names:
        rec = sum(int(r[f"{n}__n_hit"]) for r in pos) / sum(int(r[f"{n}__n_gt"]) for r in pos)
        c[n] = {"recall": rec}
    for leg in LEGS:
        rows = read(d / f"{leg}.csv")
        mpx = sum(float(r[f"{names[0]}__mpx"]) for r in rows)
        for n in names:
            c[n][leg] = sum(float(r[f"{n}__fp_components"]) for r in rows) / mpx
    curves[m] = c


def at_recall(c, leg, r):
    pts = sorted(((v["recall"], v[leg]) for k, v in c.items() if k.startswith("t")), key=lambda x: x[0])
    for (r0, a0), (r1, a1) in zip(pts, pts[1:]):
        if r0 <= r <= r1 and r1 > r0:
            w = (r - r0) / (r1 - r0)
            return float(np.exp((1 - w) * np.log(a0 + 0.01) + w * np.log(a1 + 0.01)) - 0.01)
    return None


arms = defaultdict(list)
for m, s in MODELS.items():
    arms[s["arm"]].append(m)
matched = {}
for a, ms in sorted(arms.items()):
    matched[a] = {}
    for leg in LEGS:
        for r in TARGETS:
            vals = [at_recall(curves[m], leg, r) for m in sorted(ms)]
            matched[a][f"{leg}@{r:.2f}"] = {"arm": float(np.mean(vals)) if all(v is not None for v in vals) else None,
                                           "seeds": vals}


def pred(a, b):
    out, ok = {}, True
    for leg in LEGS:
        for r in [0.75, 0.80]:
            x, y = matched.get(a, {}).get(f"{leg}@{r:.2f}", {}).get("arm"), \
                matched.get(b, {}).get(f"{leg}@{r:.2f}", {}).get("arm")
            out[f"{leg}@{r:.2f}"] = [x, y]
            if x is None or y is None:
                ok = None if ok is not False else False
            elif ok is not None and not x > y:
                ok = False
    return {"values": out, "verdict": ok}


report = {"curves": curves, "matched_recall_A3": matched,
          "predictions": {"P12_matched_recall_B11": pred("B11", "B0"), "P13_matched_recall_B10": pred("B10", "B0")},
          "method": __doc__}
(OUT / "sweep_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"models": len(MODELS), "predictions": {k: v["verdict"] for k, v in report["predictions"].items()},
                  "matched_0.75": {a: {leg: matched[a][f"{leg}@0.75"]["arm"] for leg in LEGS} for a in matched}},
                 indent=1))
