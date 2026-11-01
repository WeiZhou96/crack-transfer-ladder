# -*- coding: utf-8 -*-
"""Post hoc (pre-submission review): PaveDistress false alarms at matched IN-DOMAIN recall.

Same estimator as analyze_sweep_overlap_v1.py (Eq. 12): arm curve = seed-mean recall and seed-mean false-alarm density
at each grid threshold (0.02 ... 0.95, native point excluded); over the recall interval shared with the reference arm,
21 points, log(FA + 0.01) interpolated linearly in recall; ratio = exp(mean log ratio). The only change is the recall
axis: component recall on the 3,561 annotated PaveDistress calibration tiles (anchor_recall_sweep_v1.py) instead of the
CrackSC + NHA12D positive axis. False-alarm densities are the PaveDistress background-test values of the existing
threshold sweeps (eval_sweep_v1). Reference: Real-only of the same backbone. Descriptive; no bootstrap.
usage: python -X utf8 analyze_indomain_matched_v1.py [<anchor_sweep_dir or anchor_recall_sweep.json>]
  default: results/posthoc/anchor_recall_sweep.json; writes results/posthoc/indomain_matched.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

C = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else C / "results/posthoc/anchor_recall_sweep.json"
LEG = "pavedistress_background_test"
TS = ["t0.02", "t0.05", "t0.1", "t0.2", "t0.3", "t0.5", "t0.7", "t0.85", "t0.95"]
NAMES = {"B0": "Real-only", "B1": "StrongAug", "B2": "RandNeg", "B3_auto": "MinedNeg", "B10": "CF-Unpaired",
         "B11": "CF-Paired", "B0_v6": "Real-only, retrained", "B1_v5": "StrongAug, retrained",
         "A_x01only": "CF-NegOnly", "A_x1only": "CF-CrackLayers", "A_paired_noaux": "CF-Paired w/o pair losses",
         "B12_auto_posthoc": "MinedNeg + CF", "B2_g7": "RandNeg, 7 groups", "A_x00only": "CF-BgOnly",
         "B2_B1": "RandNeg + StrongAug", "A_x01only_B1": "StrongAug + CF-NegOnly",
         "R50_B0": "Real-only (R50)", "R50_B2": "RandNeg (R50)", "R50_B11": "CF-Paired (R50)"}

fa, ext = {}, {}
for f in sorted((C / "results/sweep").glob("*_sweep_report.json")):
    for m, c in json.load(open(f, encoding="utf-8"))["curves"].items():
        if all(t in c for t in TS):
            new = {t: c[t][LEG] for t in TS}
            if m in fa:
                assert all(abs(fa[m][t] - new[t]) < 1e-9 for t in TS), m   # the same model in several reports agrees
            fa[m] = new
            ext[m] = {t: c[t]["recall"] for t in TS}
rec, nat = {}, {}
recs = [json.load(open(f, encoding="utf-8")) for f in sorted(SRC.glob("*.json"))] if SRC.is_dir() else \
    [dict(v, model=k) for k, v in json.load(open(SRC, encoding="utf-8")).items()]
for d in recs:
    rec[d["model"]] = {t: d["recall"][t] for t in TS}
    nat[d["model"]] = d["recall"]["native_r95"]

arms = defaultdict(list)
for m in sorted(set(fa) & set(rec)):
    arms[m.rsplit("_s", 1)[0]].append(m)


def curve(models, axis):
    r = np.array([np.mean([axis[m][t] for m in models]) for t in TS])
    a = np.array([np.mean([fa[m][t] for m in models]) for t in TS])
    return r, a


def interp(r, a, x):
    o = np.argsort(r)
    return np.exp(np.interp(x, r[o], np.log(a[o] + 0.01)))


def compare(ca, cb):
    (ra, aa), (rb, ab) = ca, cb
    lo, hi = max(ra.min(), rb.min()), min(ra.max(), rb.max())
    if hi <= lo:
        return None
    g = np.linspace(lo, hi, 21)
    return [float(lo), float(hi)], float(np.exp(np.mean(np.log(interp(ra, aa, g)) - np.log(interp(rb, ab, g)))))


out = {}
for a, ms in sorted(arms.items()):
    ref = "R50_B0" if a.startswith("R50_") else "B0"
    row = {"name": NAMES.get(a, a), "models": ms, "reference": NAMES[ref],
           "native_indomain_recall": float(np.mean([nat[m] for m in ms])),
           "indomain_recall_range": [float(min(min(rec[m].values()) for m in ms)), float(max(max(rec[m].values()) for m in ms))]}
    if a != ref:
        for key, axis in (("indomain", rec), ("external", ext)):
            c = compare(curve(ms, axis), curve(arms[ref], axis))
            pairs = [compare(curve([x], axis), curve([y], axis)) for x in ms for y in arms[ref]]
            ok = [p for p in pairs if p is not None]
            row[key] = {"interval": c and c[0], "ratio": c and c[1], "seed_pairs_with_window": len(ok),
                        "seed_pairs": len(pairs), "frac_pairs_above_ref": float(np.mean([p[1] > 1 for p in ok])) if ok else None}
    out[a] = row
(C / "results/posthoc/indomain_matched.json").write_text(json.dumps({"method": __doc__, "arms": out}, indent=1), encoding="utf-8")
print(f"{'arm':28s} n  natRec  in-domain window      ratio  pairs>ref | external window       ratio")
for a, r in out.items():
    i, e = r.get("indomain"), r.get("external")
    fmt = lambda x: "—" if not x or x["ratio"] is None else f"{x['interval'][0]:.3f}-{x['interval'][1]:.3f}  {x['ratio']:6.2f}  {x['frac_pairs_above_ref'] if x['frac_pairs_above_ref'] is None else round(x['frac_pairs_above_ref'],2)}({x['seed_pairs_with_window']}/{x['seed_pairs']})"
    print(f"{r['name'][:28]:28s} {len(r['models'])}  {r['native_indomain_recall']:.3f}  {fmt(i) if i else 'reference':32s} | {fmt(e) if e else ''}")
