"""Addendum 6, descriptive: paired cluster bootstrap CI for A_x01only_B1 vs B1_v5 (and each vs B0) on the A3 legs,
same estimator as analyze_program_v2 (arm A3 = mean over seeds of sum FP / sum Mpx; one resample of groups per leg,
10000 iterations, seed 0). Reads per-image CSVs only."""
import os
import csv
import json
import sys
from pathlib import Path

import numpy as np

D = Path(os.environ.get("CRACKCF_DATA", "data"))
ADD6, PRIM = D / "students_eval_add6_v1", D / "students_eval_v1_20261028"
ARMS = {"A_x01only_B1": (ADD6, ["A_x01only_B1_s0", "A_x01only_B1_s1", "A_x01only_B1_s2"]),
        "B1_v5": (ADD6, ["B1_v5_s0", "B1_v5_s1", "B1_v5_s2"]),
        "B0": (PRIM, ["B0_s0", "B0_s1", "B0_s2"]),
        "B1": (PRIM, ["B1_s0", "B1_s1", "B1_s2"])}
LEGS = {"bcl_noise": ("test/{m}/bcl_noise.csv", "bootstrap_group_1km", "native_r95__fp_components", "native_r95__mpx", None),
        "pavedistress_background_test": ("test/{m}/pavedistress_background_test.csv", "bootstrap_group_1km",
                                         "native_r95__fp_components", "native_r95__mpx", None),
        "uav1388_background": ("test/{m}/uav1388_background.csv", "bootstrap_group_1km", "native_r95__fp_components",
                               "native_r95__mpx", None),
        "sdnet_UP_all": ("sdnet/{m}/sdnet_UP_all.csv", "group_id", "native_r95__fp_components", "native_r95__mpx", None),
        "dacl10k_all": ("strata/dacl10k/{m}.csv", "group_id", "native_r95__fp", "mpx", None),
        "dacl10k_joint": ("strata/dacl10k/{m}.csv", "group_id", "native_r95__assoc__joint", "area_mpx__joint", None),
        "hardproxy_patch": ("strata/hardproxy/{m}.csv", "group_id", "native_r95__fp", "mpx", ("label", "Patch"))}
PAIRS = [("A_x01only_B1", "B1_v5"), ("A_x01only_B1", "B0"), ("B1_v5", "B0"), ("B1_v5", "B1")]
IT = 10000
rng = np.random.default_rng(0)
out = {}
for leg, (tmpl, gcol, num, den, flt) in LEGS.items():
    order, groups, data = None, None, {}
    for arm, (edir, models) in ARMS.items():
        for m in models:
            with open(edir / tmpl.format(m=m), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if flt:
                rows = [r for r in rows if r[flt[0]] == flt[1]]
            ids = [r.get("image_id") or r["group_id"] for r in rows]
            if order is None:
                order, groups = ids, [r[gcol] for r in rows]
            elif ids != order:
                sys.exit("image order mismatch " + leg + " " + m)
            data[m] = (np.array([float(r[num] or 0) for r in rows]), np.array([float(r[den] or 0) for r in rows]))
    ug = sorted(set(groups))
    gi = {g: i for i, g in enumerate(ug)}
    idx = np.array([gi[g] for g in groups])
    G = len(ug)
    agg = {m: (np.bincount(idx, n, G), np.bincount(idx, d, G)) for m, (n, d) in data.items()}

    def arm_a3(arm, w):                      # w: (k, G) group weights -> (k,) arm A3
        return np.mean([(w @ agg[m][0]) / np.maximum(w @ agg[m][1], 1e-12) for m in ARMS[arm][1]], axis=0)
    W = rng.multinomial(G, np.full(G, 1 / G), size=IT).astype(float)
    point = {a: float(arm_a3(a, np.ones((1, G)))[0]) for a in ARMS}
    boots = {a: arm_a3(a, W) for a in ARMS}
    res = {"A3": point, "groups": G}
    for a, b in PAIRS:
        rel = boots[a] / boots[b] - 1
        res[a + "_vs_" + b] = {"rel_change": point[a] / point[b] - 1,
                               "ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                               "p_a_ge_b": float(np.mean(boots[a] >= boots[b]))}
    out[leg] = res
    print(leg, {k: (round(v["rel_change"] * 100, 1), [round(x * 100, 1) for x in v["ci95"]])
                for k, v in res.items() if k.endswith(("v5", "B0", "_B1"))}, flush=True)
json.dump({"method": __doc__, "legs": out}, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
