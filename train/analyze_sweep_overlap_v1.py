"""Addendum 5 part B (descriptive, post hoc): recall-matched false alarms over the common recall interval.

Input: sweep_report.json from analyze_sweep_v1.py (per-model curves; native_r95 excluded here).
Arm curve = seed-mean recall and seed-mean A3 at each grid threshold. For arm X vs B0 the common recall interval of the
two arm curves is sampled at 21 points; A3(r) is interpolated linearly in log(A3+0.01) against recall. Reported per leg:
interval, mean log ratio log((A3_X+0.01)/(A3_B0+0.01)) and its exp; per-seed-pair dominance over the 9 seed pairs
(fraction with mean log ratio > 0 among pairs with a common interval). No prediction, no bootstrap.
usage: python analyze_sweep_overlap_v1.py <sweep_report.json> <models.json>[,<models.json>...] <out.json>
"""
import json
import sys
from collections import defaultdict

import numpy as np

rep = json.load(open(sys.argv[1], encoding="utf-8"))
spec = {}
for m in sys.argv[2].split(","):
    spec.update(json.load(open(m, encoding="utf-8")))
LEGS = ["bcl_noise", "pavedistress_background_test", "uav1388_background"]
curves = rep["curves"]
arms = defaultdict(list)
for m in curves:
    arms[spec[m]["arm"]].append(m)
TS = sorted({k for c in curves.values() for k in c if k.startswith("t")}, key=lambda k: float(k[1:]))


def curve_of(models):
    rec = np.array([np.mean([curves[m][t]["recall"] for m in models]) for t in TS])
    a3 = {leg: np.array([np.mean([curves[m][t][leg] for m in models]) for t in TS]) for leg in LEGS}
    return rec, a3


def interp(rec, a3, r):
    o = np.argsort(rec)
    return np.exp(np.interp(r, rec[o], np.log(a3[o] + 0.01)))


def compare(ca, cb):
    (ra, aa), (rb, ab) = ca, cb
    lo, hi = max(ra.min(), rb.min()), min(ra.max(), rb.max())
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, 21)
    return {"interval": [float(lo), float(hi)],
            **{leg: float(np.mean(np.log(interp(ra, aa[leg], grid)) - np.log(interp(rb, ab[leg], grid))))
               for leg in LEGS}}


out = {}
base = curve_of(sorted(arms["B0"]))
for a in sorted(arms):
    if a == "B0":
        continue
    c = compare(curve_of(sorted(arms[a])), base)
    pairs = [compare(curve_of([x]), curve_of([y])) for x in sorted(arms[a]) for y in sorted(arms["B0"])]
    ok = [p for p in pairs if p is not None]
    out[a] = {"arm_curve": None if c is None else {"interval": c["interval"],
                                                    **{leg: {"mean_log_ratio": c[leg], "ratio": float(np.exp(c[leg]))}
                                                       for leg in LEGS}},
              "seed_pairs": {"with_common_interval": len(ok), "without": len(pairs) - len(ok),
                             **{leg: {"frac_X_above_B0": float(np.mean([p[leg] > 0 for p in ok])) if ok else None}
                                for leg in LEGS}}}
json.dump({"method": __doc__, "thresholds": TS, "vs_B0": out}, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
print(json.dumps({a: {"interval": v["arm_curve"] and [round(x, 3) for x in v["arm_curve"]["interval"]],
                      **{leg: (v["arm_curve"] and round(v["arm_curve"][leg]["ratio"], 3),
                               v["seed_pairs"][leg]["frac_X_above_B0"]) for leg in LEGS},
                      "pairs": v["seed_pairs"]["with_common_interval"]} for a, v in out.items()}, indent=1))
