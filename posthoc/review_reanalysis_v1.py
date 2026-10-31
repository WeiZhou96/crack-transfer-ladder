"""Post hoc analyses requested by the independent review (B1, B3, B4). Reads existing per-image / per-group accounts only.

B1  distractor-specific transfer ladder. Rungs 1-3: operator-induced density = (sum FP on x01 - sum FP on x00) / sum Mpx
    of the probe images; rung 4 joints: joint-associated FP per Mpx of dilated joint area minus unassociated FP per Mpx
    of class-free area on the same dacl10k images; rung 4 patches: FP per Mpx on PaveDistress Patch frames minus FP per
    Mpx on PaveDistress background-test tiles. For each arm: ratios to the reference arm of x00, x01 (or joint/patch)
    density, of the specific excess, and of the relative risk (distractor density / background density).
B3  alternative accounts on the three primary legs: component density, phantom length density, false-positive area
    fraction, unfiltered image alert rate, filtered image alert rate (at least one counted component).
B4  (a) two-level bootstrap (seeds resampled with replacement within each arm, clusters resampled) for the main
    comparisons; (b) leave-one-route-block-out on the PaveDistress background test leg.
Arm statistic = mean over seeds of per-seed densities, as in the main analysis. Bootstrap: 10000 replicates, seed 0,
one cluster resample per data source shared by all models. Native operating point only.
usage: python review_reanalysis_v1.py <out.json> <eval_dir>=<models.json> [...]
"""
import os
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(os.environ.get("CRACKCF_DATA", "data"))
OP, NB = "native_r95", 10000
MODELS = {}
for s in sys.argv[2:]:
    e, m = s.split("=", 1)
    for k, v in json.load(open(m, encoding="utf-8")).items():
        MODELS[k] = dict(v, edir=Path(e))
ARMS = defaultdict(list)
for k, v in MODELS.items():
    ARMS[v["arm"]].append(k)
REF = lambda a: "R50_B0" if a.startswith("R50_") else "B0"  # noqa: E731
FOCUS = ["B1", "B2", "B3_auto", "B10", "B11", "A_x01only", "A_x1only", "A_paired_noaux", "B0_v6", "B2_g2", "B2_g7",
         "B12_auto_posthoc", "B1_v5", "A_x01only_B1", "A_x00only", "B2_B1", "R50_B2", "R50_B11"]
FOCUS = [a for a in FOCUS if a in ARMS]
rng = np.random.default_rng(0)


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class Source:
    """cluster-level sums per model for named columns; shared bootstrap weights."""

    def __init__(self, name, path_fn, cluster_fn, cols, filt=None):
        self.sums, order = {}, None
        for m in MODELS:
            p = path_fn(m)
            if not p.exists():
                continue
            rs = [r for r in rows(p) if filt is None or filt(r)]
            key = [r.get("image_id") or r.get("image") or r["group_id"] for r in rs]
            if order is None:
                order = key
                cl = [cluster_fn(r) for r in rs]
                self.keys = sorted(set(cl))
                pos = {c: i for i, c in enumerate(self.keys)}
                self.gi = np.array([pos[c] for c in cl])
            assert key == order, (name, m)
            G = len(self.keys)
            self.sums[m] = {c: np.bincount(self.gi, weights=np.array([float(r[c]) for r in rs]), minlength=G) for c in cols(m)}
        G = len(self.keys)
        self.W = rng.multinomial(G, np.full(G, 1 / G), size=NB).astype(float)
        self.name = name

    def tot(self, m, c, w=None):
        v = self.sums[m][c]
        return v.sum() if w is None else w @ v


def spec_cluster(qdir):
    cl = {}
    for l in open(D / "quadruplets" / qdir / "specs.jsonl", encoding="utf-8"):
        s = json.loads(l)
        cl[s["group_id"]] = s["background"]["group_id"]
    return lambda r: cl[r["group_id"]]


def arm_density(src, arm, num, den, w=None, seeds=None):
    ms = seeds if seeds is not None else ARMS[arm]
    return np.mean([src.tot(m, num, w) / src.tot(m, den, w) for m in ms], axis=0)


def ci(x):
    x = x[np.isfinite(x)]
    if x.size < 0.9 * NB:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


out = {"method": __doc__, "ladder": {}, "accounts": {}, "two_level": {}, "loo_pave": {}}

# ---------------------------------------------------------------- B1 rungs 1-3
RUNGS = {"1_in_sample": ("quads", "v4"), "2_held_out": ("quads_v4_heldout_s20261029", "v4_heldout_s20261029"),
         "3_unseen_scene": ("quads_sdnet_scene_novel_s20261030", "sdnet_scene_novel_s20261030")}
for rung, (sub, qdir) in RUNGS.items():
    clf = spec_cluster(qdir)
    for op in ("joint", "repair_boundary", "shadow"):
        src = Source(f"{rung}_{op}", lambda m: MODELS[m]["edir"] / "strata" / sub / f"{m}.csv", clf,
                     lambda m: ["mpx", f"{OP}__fp_x00", f"{OP}__fp_x01"], filt=lambda r, op=op: r["operator"] == op)
        res = {}
        for a in FOCUS + ["B0", "R50_B0"]:
            if not all(m in src.sums for m in ARMS.get(a, [])) or a not in ARMS:
                continue
            res.setdefault("_density", {})[a] = {
                "x00": float(arm_density(src, a, f"{OP}__fp_x00", "mpx")), "x01": float(arm_density(src, a, f"{OP}__fp_x01", "mpx"))}
        for a in FOCUS:
            r_ = REF(a)
            if a not in res.get("_density", {}) or r_ not in res["_density"]:
                continue
            st = {}
            for key, f_ in (("x00", lambda arm, w: arm_density(src, arm, f"{OP}__fp_x00", "mpx", w)),
                            ("x01", lambda arm, w: arm_density(src, arm, f"{OP}__fp_x01", "mpx", w)),
                            ("excess", lambda arm, w: arm_density(src, arm, f"{OP}__fp_x01", "mpx", w)
                             - arm_density(src, arm, f"{OP}__fp_x00", "mpx", w))):
                pt = float(f_(a, None) / f_(r_, None)) if f_(r_, None) != 0 else float("nan")
                with np.errstate(divide="ignore", invalid="ignore"):
                    bs = f_(a, src.W) / f_(r_, src.W)
                st[key] = {"ratio": pt, "ci": ci(bs)}
            res[a] = st
        out["ladder"][f"{rung}/{op}"] = res
        print(rung, op, {a: round(res[a]["excess"]["ratio"], 3) for a in ("B10", "B11", "A_x01only", "A_x1only", "B1", "R50_B11") if a in res}, flush=True)

# ---------------------------------------------------------------- B1 rung 4 joints (dacl10k)
dac = Source("dacl10k", lambda m: MODELS[m]["edir"] / "strata" / "dacl10k" / f"{m}.csv", lambda r: r["image"],
             lambda m: ["area_mpx__joint", f"{OP}__assoc__joint", "classfree_mpx", f"{OP}__unassoc"])
res = {}
for a in FOCUS:
    r_ = REF(a)
    if a not in ARMS or not all(m in dac.sums for m in ARMS[a] + ARMS[r_]):
        continue
    J = lambda arm, w: arm_density(dac, arm, f"{OP}__assoc__joint", "area_mpx__joint", w)  # noqa: E731
    B = lambda arm, w: arm_density(dac, arm, f"{OP}__unassoc", "classfree_mpx", w)  # noqa: E731
    st = {}
    for key, f_ in (("joint", J), ("background", B), ("excess", lambda arm, w: J(arm, w) - B(arm, w)),
                    ("relative_risk", lambda arm, w: J(arm, w) / B(arm, w))):
        st[key] = {"ratio": float(f_(a, None) / f_(r_, None)), "ci": ci(f_(a, dac.W) / f_(r_, dac.W))}
    res[a] = st
out["ladder"]["4_real/joint"] = res
print("rung4 joint", {a: (round(res[a]["excess"]["ratio"], 3), round(res[a]["relative_risk"]["ratio"], 3)) for a in res}, flush=True)

# ---------------------------------------------------------------- B1 rung 4 patches
hp = Source("patch", lambda m: MODELS[m]["edir"] / "strata" / "hardproxy" / f"{m}.csv", lambda r: r["group_id"],
            lambda m: ["mpx", f"{OP}__fp"], filt=lambda r: r.get("label") == "Patch")
pv = Source("pave", lambda m: MODELS[m]["edir"] / "test" / m / "pavedistress_background_test.csv",
            lambda r: r["bootstrap_group_1km"], lambda m: [f"{OP}__mpx", f"{OP}__fp_components", f"{OP}__phantom_len_px",
                                                           f"{OP}__fp_area_px", f"{OP}__image_fired"])
res = {}
for a in FOCUS:
    r_ = REF(a)
    if a not in ARMS or not all(m in hp.sums and m in pv.sums for m in ARMS[a] + ARMS[r_]):
        continue
    Pp = lambda arm, w: arm_density(hp, arm, f"{OP}__fp", "mpx", w)  # noqa: E731
    Pb = lambda arm, w, wb=None: arm_density(pv, arm, f"{OP}__fp_components", f"{OP}__mpx", wb)  # noqa: E731
    st = {}
    for key, f_, fb in (("patch", lambda arm, w, wb: Pp(arm, w), None), ("background", lambda arm, w, wb: Pb(arm, None, wb), None),
                        ("excess", lambda arm, w, wb: Pp(arm, w) - Pb(arm, None, wb), None),
                        ("relative_risk", lambda arm, w, wb: Pp(arm, w) / Pb(arm, None, wb), None)):
        with np.errstate(divide="ignore", invalid="ignore"):
            st[key] = {"ratio": float(f_(a, None, None) / f_(r_, None, None)), "ci": ci(f_(a, hp.W, pv.W) / f_(r_, hp.W, pv.W))}
    res[a] = st
out["ladder"]["4_real/patch"] = res
print("rung4 patch", {a: round(res[a]["excess"]["ratio"], 3) for a in res}, flush=True)

# ---------------------------------------------------------------- B3 alternative accounts
LEGS = {"bcl_noise": "bcl_noise.csv", "pavedistress_background_test": "pavedistress_background_test.csv",
        "uav1388_background": "uav1388_background.csv"}
for leg, fn in LEGS.items():
    src = pv if leg == "pavedistress_background_test" else Source(
        leg, lambda m, fn=fn: MODELS[m]["edir"] / "test" / m / fn, lambda r: r["bootstrap_group_1km"],
        lambda m: [f"{OP}__mpx", f"{OP}__fp_components", f"{OP}__phantom_len_px", f"{OP}__fp_area_px", f"{OP}__image_fired"])
    # filtered image alert: per-image indicator fp_components > 0, summed per cluster
    filt_rows = {}
    for m in src.sums:
        rs = rows(MODELS[m]["edir"] / "test" / m / fn)
        G = len(src.keys)
        src.sums[m]["img_filtered"] = np.bincount(src.gi, weights=np.array([float(float(r[f"{OP}__fp_components"]) > 0) for r in rs]), minlength=G)
        src.sums[m]["n_img"] = np.bincount(src.gi, minlength=G).astype(float)
    res = {}
    for a in FOCUS:
        r_ = REF(a)
        if a not in ARMS or not all(m in src.sums for m in ARMS[a] + ARMS[r_]):
            continue
        st = {}
        for key, num, den in (("components", f"{OP}__fp_components", f"{OP}__mpx"), ("phantom_length", f"{OP}__phantom_len_px", f"{OP}__mpx"),
                              ("fp_area", f"{OP}__fp_area_px", f"{OP}__mpx"), ("image_alert_unfiltered", f"{OP}__image_fired", "n_img"),
                              ("image_alert_filtered", "img_filtered", "n_img")):
            f_ = lambda arm, w: arm_density(src, arm, num, den, w)  # noqa: E731
            st[key] = {"ratio": float(f_(a, None) / f_(r_, None)), "ci": ci(f_(a, src.W) / f_(r_, src.W)),
                       "value": float(f_(a, None)), "ref_value": float(f_(r_, None))}
        res[a] = st
    out["accounts"][leg] = res
    print("accounts", leg, {a: {k: round(v["ratio"], 2) for k, v in res[a].items()} for a in ("B11", "B10", "B1") if a in res}, flush=True)

    # ------------------------------------------------------------ B4a two-level bootstrap
    two = {}
    for a in FOCUS:
        r_ = REF(a)
        if a not in res:
            continue
        sa, sr = ARMS[a], ARMS[r_]
        reps = np.empty(NB)
        idx_a = rng.integers(0, len(sa), size=(NB, len(sa)))
        idx_r = rng.integers(0, len(sr), size=(NB, len(sr)))
        dens = {m: (src.W @ src.sums[m][f"{OP}__fp_components"]) / (src.W @ src.sums[m][f"{OP}__mpx"]) for m in sa + sr}
        A = np.stack([dens[m] for m in sa], 1)
        R = np.stack([dens[m] for m in sr], 1)
        reps = np.take_along_axis(A, idx_a, 1).mean(1) / np.take_along_axis(R, idx_r, 1).mean(1) - 1
        two[a] = {"rel_change": res[a]["components"]["ratio"] - 1, "ci_clusters_only": [c - 1 for c in res[a]["components"]["ci"]],
                  "ci_two_level": ci(reps)}
    out["two_level"][leg] = two

# ---------------------------------------------------------------- B4b leave one PaveDistress route block out
for a in FOCUS:
    r_ = REF(a)
    if a not in ARMS or not all(m in pv.sums for m in ARMS[a] + ARMS[r_]):
        continue
    vals = {}
    for g, key in enumerate(pv.keys):
        w = np.ones(len(pv.keys)); w[g] = 0
        num = np.mean([pv.tot(m, f"{OP}__fp_components", w) / pv.tot(m, f"{OP}__mpx", w) for m in ARMS[a]])
        den = np.mean([pv.tot(m, f"{OP}__fp_components", w) / pv.tot(m, f"{OP}__mpx", w) for m in ARMS[r_]])
        vals[key] = float(num / den - 1)
    out["loo_pave"][a] = {"range": [min(vals.values()), max(vals.values())], "per_block": vals}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1])
