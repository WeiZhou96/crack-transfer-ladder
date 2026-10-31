"""experiment_program_v1 E2/E3 report: real per-confuser false alarms and H2-public, from eval_strata_v1 outputs.

H2-public (program E3): models B0, B1, B2, B3_auto (never trained on quadruplets), native_r95.
  beta_syn(z)  = mean over groups of log((FP_x01 + 0.5) / (FP_x00 + 0.5)), operator/stratum per program
  beta_real(joint)  = log[(FP touching dilated EJoint|JTape / joint Mpx) / (FP touching no class / class-free Mpx)]
  beta_real(repair) = log[(FP per Mpx on Patch frames) / (FP per Mpx on pavedistress_background_test tiles)]
  pass iff sign(arm-pooled mean beta_syn) == sign(arm-pooled mean beta_real) for both classes.
Bootstrap 95% intervals (10000, seed 0) resample quadruplet groups, dacl10k images, hard-proxy frame groups and
background-test 1 km groups; the same resample is applied to every model.
Per-class tables: lambda_z (associated FP per Mpx of dilated class area) and rho_z (per image Mpx) for every arm.
usage: python analyze_strata_v1.py <eval_dir> <models_primary.json> <report_dir>
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

E, MODELS, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8")), Path(sys.argv[3])
S = E / "strata"
OUT.mkdir(parents=True, exist_ok=True)
OP, ITERS = "native_r95", 10000
H2_ARMS = ["B0", "B1", "B2", "B3_auto"]
SYN = {"joint": ("joint", "rgb_concrete"), "repair": ("repair_boundary", "laser_asphalt")}
rng = np.random.default_rng(0)


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


arms = defaultdict(list)
for m, s in MODELS.items():
    arms[s["arm"]].append(m)
q = {m: read(S / "quads" / f"{m}.csv") for m in MODELS}
hp = {m: read(S / "hardproxy" / f"{m}.csv") for m in MODELS}
dc = {m: read(S / "dacl10k" / f"{m}.csv") for m in MODELS}
bg = {m: read(E / "test" / m / "pavedistress_background_test.csv") for m in MODELS}
f = lambda rows, k: np.array([float(r.get(k) or 0) for r in rows])  # noqa: E731


def resample_index(groups):
    keys = sorted(set(groups))
    pos = {k: i for i, k in enumerate(keys)}
    gi = np.array([pos[g] for g in groups])
    W = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)) for _ in range(ITERS)])
    return gi, W.astype(np.float64), len(keys)


# ---- synthetic beta
syn_rs = {}
for z, (op, st) in SYN.items():
    rows0 = [r for r in q[next(iter(MODELS))] if r["operator"] == op and r["stratum"] == st]
    ids = [r["group_id"] for r in rows0]
    syn_rs[z] = (ids, resample_index(ids))


def beta_syn(m, z):
    op, st = SYN[z]
    rows = {r["group_id"]: r for r in q[m] if r["operator"] == op and r["stratum"] == st}
    ids, (gi, W, n) = syn_rs[z]
    v = np.array([np.log((float(rows[i][f"{OP}__fp_x01"]) + .5) / (float(rows[i][f"{OP}__fp_x00"]) + .5)) for i in ids])
    sums = np.bincount(gi, v, n)
    cnt = np.bincount(gi, None, n)
    return float(v.mean()), (W @ sums) / (W @ cnt)


# ---- real beta: repair (Patch frames vs background-test tiles), joint (dacl10k)
patch_groups = [r["group_id"] for r in hp[next(iter(MODELS))] if r["label"] == "Patch"]
bg_groups = [r["bootstrap_group_1km"] for r in bg[next(iter(MODELS))]]
rs_patch, rs_bg = resample_index(patch_groups), resample_index(bg_groups)
dimgs = [r["group_id"] for r in dc[next(iter(MODELS))]]
rs_d = resample_index(dimgs)


def ratio_boot(num, den, rs):
    gi, W, n = rs
    return (W @ np.bincount(gi, num, n)) / (W @ np.bincount(gi, den, n))


def beta_real(m, z):
    if z == "repair":
        pr = [r for r in hp[m] if r["label"] == "Patch"]
        a_n, a_d = f(pr, f"{OP}__fp"), f(pr, "mpx")
        b_n, b_d = f(bg[m], f"{OP}__fp_components"), f(bg[m], f"{OP}__mpx")
        point = np.log((a_n.sum() / a_d.sum()) / (b_n.sum() / b_d.sum()))
        boot = np.log(ratio_boot(a_n, a_d, rs_patch) / ratio_boot(b_n, b_d, rs_bg))
        return float(point), boot
    rows = dc[m]
    j_n, j_d = f(rows, f"{OP}__assoc__joint"), f(rows, "area_mpx__joint")
    u_n, u_d = f(rows, f"{OP}__unassoc"), f(rows, "classfree_mpx")
    point = np.log((j_n.sum() / j_d.sum()) / (u_n.sum() / u_d.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.log(ratio_boot(j_n, j_d, rs_d) / ratio_boot(u_n, u_d, rs_d))
    return float(point), boot


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


h2 = {}
for z in SYN:
    rec = {"per_arm": {}}
    pooled_s, pooled_r, bs, br = [], [], [], []
    for a in arms:
        ss = [beta_syn(m, z) for m in arms[a]]
        rr = [beta_real(m, z) for m in arms[a]]
        rec["per_arm"][a] = {"beta_syn": float(np.mean([s[0] for s in ss])), "beta_syn_ci": ci(np.mean([s[1] for s in ss], 0)),
                             "beta_real": float(np.mean([r[0] for r in rr])), "beta_real_ci": ci(np.mean([r[1] for r in rr], 0)),
                             "per_model": {m: {"beta_syn": s[0], "beta_real": r[0]} for m, s, r in zip(arms[a], ss, rr)}}
        if a in H2_ARMS:
            pooled_s += [s[0] for s in ss]; pooled_r += [r[0] for r in rr]
            bs += [s[1] for s in ss]; br += [r[1] for r in rr]
    rec["h2_models"] = H2_ARMS
    rec["beta_syn_pooled"], rec["beta_syn_pooled_ci"] = float(np.mean(pooled_s)), ci(np.mean(bs, 0))
    rec["beta_real_pooled"], rec["beta_real_pooled_ci"] = float(np.mean(pooled_r)), ci(np.mean(br, 0))
    rec["sign_agree"] = bool(np.sign(rec["beta_syn_pooled"]) == np.sign(rec["beta_real_pooled"]))
    h2[z] = rec
verdict = "pass" if all(h2[z]["sign_agree"] for z in h2) else "fail"

# ---- per-class real tables (all arms) and hard proxy
classes = sorted({k.split("__")[-1] for k in dc[next(iter(MODELS))][0] if k.startswith(f"{OP}__assoc__")})
per_class = {}
for op in [OP, "fixed_0.5"]:
    per_class[op] = {}
    for a in arms:
        row = {}
        for z in classes:
            num = np.mean([f(dc[m], f"{op}__assoc__{z}").sum() for m in arms[a]])
            area = f(dc[arms[a][0]], f"area_mpx__{z}").sum()
            mpx = f(dc[arms[a][0]], "mpx").sum()
            nimg = int((f(dc[arms[a][0]], f"area_mpx__{z}") > 0).sum())
            row[z] = {"lambda_per_class_mpx": num / area if area else None, "rho_per_image_mpx": num / mpx,
                      "images_with_class": nimg}
        u = np.mean([f(dc[m], f"{op}__unassoc").sum() for m in arms[a]])
        row["_unassociated_per_classfree_mpx"] = u / f(dc[arms[a][0]], "classfree_mpx").sum()
        row["_all_fp_per_mpx"] = np.mean([f(dc[m], f"{op}__fp").sum() for m in arms[a]]) / f(dc[arms[a][0]], "mpx").sum()
        per_class[op][a] = row
hard = {}
for op in [OP, "fixed_0.5"]:
    hard[op] = {}
    for a in arms:
        hard[op][a] = {}
        for lab in ["Patch", "Pothole"]:
            vals = []
            for m in arms[a]:
                rr = [r for r in hp[m] if r["label"] == lab]
                vals.append((f(rr, f"{op}__fp").sum() / f(rr, "mpx").sum(),
                             f(rr, f"{op}__tiles_fired").sum() / f(rr, "tiles").sum()))
            hard[op][a][lab] = {"fp_per_mpx": float(np.mean([v[0] for v in vals])),
                                "tile_far": float(np.mean([v[1] for v in vals]))}
        hard[op][a]["background_test_fp_per_mpx"] = float(np.mean(
            [f(bg[m], f"{op}__fp_components").sum() / f(bg[m], f"{op}__mpx").sum() for m in arms[a]]))
syn_table = {}
for a in arms:
    syn_table[a] = {}
    for opn in ["shadow", "joint", "repair_boundary"]:
        vals = []
        for m in arms[a]:
            rows = [r for r in q[m] if r["operator"] == opn]
            vals.append((f(rows, f"{OP}__fp_x01").sum() / f(rows, "mpx").sum(),
                         f(rows, f"{OP}__fp_x00").sum() / f(rows, "mpx").sum()))
        syn_table[a][opn] = {"fp_per_mpx_x01": float(np.mean([v[0] for v in vals])),
                             "fp_per_mpx_x00": float(np.mean([v[1] for v in vals]))}
report = {"operating_point": OP, "H2_public": {"verdict": verdict, "classes": h2,
          "not_testable": "shadow (no labelled real counterpart); standardized-effect correlation (2 classes)"},
          "dacl10k_per_class": per_class, "dacl10k_images": len(dimgs), "hard_proxy": hard,
          "synthetic_x01_x00": syn_table,
          "notes": "B10/B11 were trained on these quadruplets: their synthetic rows are in-sample diagnostics"}
(OUT / "strata_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float),
                                        encoding="utf-8")
print(json.dumps({"H2_public": verdict, **{z: {k: h2[z][k] for k in ["beta_syn_pooled", "beta_real_pooled",
                  "beta_syn_pooled_ci", "beta_real_pooled_ci", "sign_agree"]} for z in h2}}, indent=1))
