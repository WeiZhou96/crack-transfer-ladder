"""Post hoc (pre-submission review): rung-4 joint ladder under alternative background and joint definitions.

Reads the existing per-image dacl10k accounts (strata/dacl10k/<model>.csv); no inference. Rung 4 is the joint-
associated false-alarm density per Mpx of dilated joint area minus (excess) or divided by (relative share) the density
of unassociated false alarms per Mpx of class-free area. Variants:
  pooled          background over all label-negative images (as in review_reanalysis_v1.py)
  same_image      background restricted to the images that contain the joint class
  ejoint          joints = expansion joints only (joint tape excluded); background over all images
  ejoint_same     expansion joints only; background restricted to the images that contain expansion joints
Ratios of arm means (mean over seeds of per-seed densities) to Real-only of the same backbone and to CF-BgOnly; paired
image bootstrap, 10000 replicates, seed 0, one resample shared by all models. Native operating point only.
usage: python rung4_variants_v1.py <out.json> <eval_dir>=<models.json> [...]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OP, NB = "native_r95", 10000
MODELS = {}
for s in sys.argv[2:]:
    e, m = s.split("=", 1)
    for k, v in json.load(open(m, encoding="utf-8")).items():
        MODELS[k] = dict(v, edir=Path(e))
ARMS = defaultdict(list)
for k, v in MODELS.items():
    if (v["edir"] / "strata" / "dacl10k" / f"{k}.csv").exists():
        ARMS[v["arm"]].append(k)
FOCUS = ["B1", "B1_v5", "B2", "B3_auto", "B10", "B11", "A_x00only", "A_x01only", "A_x1only", "A_paired_noaux",
         "B12_auto_posthoc", "A_x01only_B1", "B2_B1", "B0_v6", "R50_B2", "R50_B11"]
FOCUS = [a for a in FOCUS if a in ARMS]
REFS = lambda a: (["R50_B0"] if a.startswith("R50_") else ["B0"] + (["A_x00only"] if a != "A_x00only" else []))  # noqa: E731
VARIANTS = {"pooled": ("joint", False), "same_image": ("joint", True),
            "ejoint": ("EJoint", False), "ejoint_same": ("EJoint", True)}

cols = None
data, order = {}, None
for m in sorted(m for ms in ARMS.values() for m in ms):
    with open(MODELS[m]["edir"] / "strata" / "dacl10k" / f"{m}.csv", encoding="utf-8-sig") as f:
        rs = list(csv.DictReader(f))
    key = [r["image"] for r in rs]
    if order is None:
        order = key
    assert key == order, m
    data[m] = {c: np.array([float(r[c]) for r in rs]) for c in
               ["classfree_mpx", f"{OP}__unassoc", "area_mpx__joint", "area_mpx__EJoint",
                f"{OP}__assoc__joint", f"{OP}__assoc__EJoint"]}
n = len(order)
rng = np.random.default_rng(0)
W = rng.multinomial(n, np.full(n, 1 / n), size=NB).astype(float)


def density(m, num, den, mask, w):
    a, b = data[m][num] * mask, data[m][den] * mask
    return (a.sum() / b.sum()) if w is None else (w @ a) / (w @ b)


def arm_stat(arm, z, same, kind, w):
    """Arm-mean densities; the relative share is the ratio of the arm-mean densities, as in review_reanalysis_v1.py."""
    if kind == "relative_risk":
        return arm_stat(arm, z, same, "joint", w) / arm_stat(arm, z, same, "background", w)
    out = []
    for m in ARMS[arm]:
        jm = (data[m][f"area_mpx__{z}"] > 0).astype(float)
        bm = jm if same else np.ones(n)
        J = density(m, f"{OP}__assoc__{z}", f"area_mpx__{z}", jm, w)
        B = density(m, f"{OP}__unassoc", "classfree_mpx", bm, w)
        out.append({"joint": J, "background": B, "excess": J - B}[kind])
    return np.mean(out, axis=0)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size >= 0.9 * NB else [None, None]


res = {"method": __doc__, "n_images": n, "n_images_with": {}, "arms": {a: len(ARMS[a]) for a in ARMS}, "variants": {}}
m0 = ARMS["B0"][0]
for z in ("joint", "EJoint"):
    res["n_images_with"][z] = int((data[m0][f"area_mpx__{z}"] > 0).sum())
for vname, (z, same) in VARIANTS.items():
    v = {"_density": {}}
    for a in FOCUS + ["B0", "R50_B0"]:
        if a in ARMS:
            v["_density"][a] = {k: float(arm_stat(a, z, same, k, None)) for k in ("joint", "background", "excess")}
    for a in FOCUS:
        for r_ in REFS(a):
            if r_ not in ARMS:
                continue
            st = {}
            for kind in ("joint", "background", "excess", "relative_risk"):
                pt = float(arm_stat(a, z, same, kind, None) / arm_stat(r_, z, same, kind, None))
                with np.errstate(divide="ignore", invalid="ignore"):
                    bs = arm_stat(a, z, same, kind, W) / arm_stat(r_, z, same, kind, W)
                st[kind] = {"ratio": pt, "ci": ci(bs)}
            v.setdefault(a, {})[r_] = st
    res["variants"][vname] = v
    print(vname, {a: (round(v[a][REFS(a)[0]]["excess"]["ratio"], 3), round(v[a][REFS(a)[0]]["relative_risk"]["ratio"], 3))
                  for a in ("B10", "B11", "A_x00only", "B1", "B1_v5", "R50_B11") if a in v}, flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1), encoding="utf-8")
