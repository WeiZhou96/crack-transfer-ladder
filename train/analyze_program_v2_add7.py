"""experiment_program_v1 consolidated report: primary (P5) + ablation/post-hoc models, SDNET leg, E6 translation,
scoring of the addendum-1/2 predictions P1-P7 exactly as worded. Reads per-image accounts only; no inference, no
threshold choice (sealed native_r95 and fixed 0.5 from the evaluators).

Legs (A3 = filtered FP components per Mpx; a leg enters for every model whose file exists, image order checked):
  bcl_noise, pavedistress_background_test, uav1388_background   test/<m>/<leg>.csv, group bootstrap_group_1km
  sdnet_staged_negative, sdnet_UP_all                            sdnet/<m>/<leg>.csv, group group_id (overlap group)
  dacl10k_all (all FP / image Mpx), dacl10k_joint (joint-associated FP / dilated joint Mpx)  strata/dacl10k, per image
  hardproxy_patch (FP / Mpx on PaveDistress Patch frames)        strata/hardproxy, group group_id
Rates: positive-axis component recall (CrackSC+NHA12D, group group_id); sdnet_CP image detection rate (any filtered
component). Synthetic probes: x01/x00 FP per Mpx per operator on strata/quads (training draws) and
strata/quads_v4_heldout_s20261029 (addendum 2). Arm = mean of its seeds. Paired cluster bootstrap (10000, seed 0):
one resample per leg applied to every model.
v2 = v1 + addendum-3 arms (B0_v6 trainer control, R50_* second backbone): extra contrasts, predictions P8-P11, and
a held-out-probe sensitivity restricted to groups whose background crop never appears in the v4 training draws.
usage: python analyze_program_v2.py <report_dir> <eval_dir>=<models.json> [<eval_dir>=<models.json> ...]
"""
import os
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
        if k in MODELS:
            raise SystemExit("duplicate model " + k)
        MODELS[k], EDIR[k] = v, Path(e)
OPS, ITERS = ["native_r95", "fixed_0.5"], 10000
HELD = "quads_v4_heldout_s20261029"
PX_MM = 1.012                                    # PaveDistress ground sampling (program E6)
M2_PER_MPX = 1e6 * (PX_MM / 1000) ** 2
LANE_KM_M2 = 3.75 * 1000

arms = defaultdict(list)
for m, s in MODELS.items():
    arms[s["arm"]].append(m)
for a in arms:
    arms[a].sort()


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


LEGS = {
    "bcl_noise": ("test/{m}/bcl_noise.csv", "bootstrap_group_1km", "{op}__fp_components", "{op}__mpx", None),
    "pavedistress_background_test": ("test/{m}/pavedistress_background_test.csv", "bootstrap_group_1km",
                                     "{op}__fp_components", "{op}__mpx", None),
    "uav1388_background": ("test/{m}/uav1388_background.csv", "bootstrap_group_1km", "{op}__fp_components",
                           "{op}__mpx", None),
    "sdnet_staged_negative": ("sdnet/{m}/sdnet_staged_negative.csv", "group_id", "{op}__fp_components", "{op}__mpx",
                              None),
    "sdnet_UP_all": ("sdnet/{m}/sdnet_UP_all.csv", "group_id", "{op}__fp_components", "{op}__mpx", None),
    "dacl10k_all": ("strata/dacl10k/{m}.csv", "group_id", "{op}__fp", "mpx", None),
    "dacl10k_joint": ("strata/dacl10k/{m}.csv", "group_id", "{op}__assoc__joint", "area_mpx__joint", None),
    "hardproxy_patch": ("strata/hardproxy/{m}.csv", "group_id", "{op}__fp", "mpx", ("label", "Patch")),
}
rng = np.random.default_rng(0)


def load_leg(leg):
    tmpl, gcol, num, den, flt = LEGS[leg]
    order, groups, data = None, None, {}
    for m in MODELS:
        p = EDIR[m] / tmpl.format(m=m)
        if not p.exists():
            continue
        rows = read(p)
        if flt:
            rows = [r for r in rows if r[flt[0]] == flt[1]]
        ids = [r.get("image_id") or r["group_id"] for r in rows]
        if order is None:
            order, groups = ids, [r[gcol] for r in rows]
        elif ids != order:
            raise SystemExit(f"image order differs: {m} {leg}")
        data[m] = {op: (np.array([float(r.get(num.format(op=op)) or 0) for r in rows]),
                        np.array([float(r[den.format(op=op)]) for r in rows]),
                        np.array([float(r.get(f"{op}__phantom_len_px") or 0) for r in rows]),
                        np.array([float(r.get(f"{op}__image_fired") or 0) for r in rows])) for op in OPS}
    if not data:
        return None
    keys = sorted(set(groups))
    pos = {k: i for i, k in enumerate(keys)}
    gi = np.array([pos[g] for g in groups])
    W = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                  for _ in range(ITERS)]).astype(np.float32)
    return {"n_images": len(order), "n_groups": len(keys), "gi": gi, "W": W, "data": data}


def complete(arm, have):
    return arm in arms and all(m in have for m in arms[arm])


def arm_point(L, arm, op, k=0):
    d = L["data"]
    return float(np.mean([d[m][op][k].sum() / d[m][op][1].sum() for m in arms[arm]]))


def arm_boot(L, arm, op, k=0):
    gi, W, n = L["gi"], L["W"], L["n_groups"]
    return np.mean([(W @ np.bincount(gi, L["data"][m][op][k], n)) / (W @ np.bincount(gi, L["data"][m][op][1], n))
                    for m in arms[arm]], 0)


def pct(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


# ---------------------------------------------------------------- negative legs
COMPARE = [("B11", "B0"), ("B10", "B0"), ("B11", "B3_auto"), ("B11", "B10"), ("B1", "B0"), ("B2", "B0"),
           ("B3_auto", "B0"), ("B3_auto", "B2"), ("A_x1only", "B0"), ("A_x01only", "B0"), ("A_paired_noaux", "B0"),
           ("A_paired_noaux", "B11"), ("A_x1only", "B10"), ("A_x01only", "B10"), ("B12_auto_posthoc", "B3_auto"),
           ("B12_auto_posthoc", "B11"), ("B12_auto_posthoc", "B0"), ("B2_g2", "B0"), ("B2_g7", "B0"),
           ("B2_g2", "B2"), ("B2_g7", "B2"), ("B11_tauhigh", "B11_taulow"),
           ("B0_v6", "B0"), ("B11", "B0_v6"), ("B10", "B0_v6"), ("A_x1only", "B0_v6"), ("A_x01only", "B0_v6"),
           ("A_paired_noaux", "B0_v6"), ("B2_g2", "B0_v6"), ("B2_g7", "B0_v6"), ("B12_auto_posthoc", "B0_v6"),
           ("R50_B11", "R50_B0"), ("R50_B2", "R50_B0"), ("R50_B0", "B0")]
COMPARE += [("A_x00only", "B0"), ("A_x00only", "A_x01only"), ("A_x00only", "B2"), ("A_x00only", "B0_v6"),
            ("B2_B1", "B1_v5"), ("B2_B1", "B0"), ("B2_B1", "B2"), ("B2_B1", "B1"), ("B2_B1", "B0_v6"),
            ("B1_v5", "B0"), ("A_x01only_B1", "B1_v5")]   # addendum 7 (post hoc)
legs, arm_A3, comps, e6 = {}, {}, {}, {}
for leg in LEGS:
    L = load_leg(leg)
    if L is None:
        continue
    have = set(L["data"])
    legs[leg] = {"n_images": L["n_images"], "n_groups": L["n_groups"],
                 "models": sorted(have), "complete_arms": sorted(a for a in arms if complete(a, have))}
    arm_A3[leg] = {op: {a: arm_point(L, a, op) for a in arms if complete(a, have)} for op in OPS}
    if leg == "pavedistress_background_test":
        for op in OPS:
            e6[op] = {}
            for a in arms:
                if complete(a, have):
                    a3, ph = arm_point(L, a, op), arm_point(L, a, op, 2)
                    e6[op][a] = {"fp_per_100m2": a3 / M2_PER_MPX * 100,
                                 "phantom_m_per_100m2": ph * PX_MM / 1000 / M2_PER_MPX * 100,
                                 "fp_per_lane_km": a3 / M2_PER_MPX * LANE_KM_M2,
                                 "tile_alert_rate": float(np.mean([L["data"][m][op][3].mean() for m in arms[a]]))}
    comps[leg] = {}
    for op in OPS:
        comps[leg][op] = {}
        for a, b in COMPARE:
            if not (complete(a, have) and complete(b, have)):
                continue
            xa, xb = arm_boot(L, a, op), arm_boot(L, b, op)
            ma, mb = arm_point(L, a, op), arm_point(L, b, op)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (xa - xb) / xb
            comps[leg][op][f"{a}_vs_{b}"] = {
                "A3_a": ma, "A3_b": mb, "rel_change": (ma - mb) / mb if mb > 0 else None, "rel_change_ci95": pct(rel),
                "p_a_ge_b": float(np.mean(xa >= xb)),
                "seeds": [{"a": x, "b": y, "a3_a": float(L["data"][x][op][0].sum() / L["data"][x][op][1].sum()),
                           "a3_b": float(L["data"][y][op][0].sum() / L["data"][y][op][1].sum())}
                          for x, y in zip(arms[a], arms[b])]}
    del L

# ---------------------------------------------------------------- rates: positive axis recall, SDNET CP detection
def load_rate(kind):
    order, groups, data = None, None, {}
    for m in MODELS:
        p = EDIR[m] / ("test/{m}/positive_axis.csv" if kind == "pos" else "sdnet/{m}/sdnet_CP.csv").format(m=m)
        if not p.exists():
            continue
        rows = read(p)
        ids = [r["image_id"] for r in rows]
        if order is None:
            order, groups = ids, [r["group_id"] for r in rows]
        elif ids != order:
            raise SystemExit(f"image order differs: {m} {kind}")
        if kind == "pos":
            data[m] = {op: (np.array([float(r[f"{op}__n_hit"]) for r in rows]),
                            np.array([float(r[f"{op}__n_gt"]) for r in rows])) for op in OPS}
        else:
            data[m] = {op: (np.array([float(float(r[f"{op}__fp_components"]) > 0) for r in rows]),
                            np.ones(len(rows))) for op in OPS}
    if not data:
        return None
    keys = sorted(set(groups))
    pos = {k: i for i, k in enumerate(keys)}
    gi = np.array([pos[g] for g in groups])
    W = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                  for _ in range(ITERS)]).astype(np.float32)
    return {"n_images": len(order), "n_groups": len(keys), "gi": gi, "W": W, "data": data}


rates, rate_comps = {}, {}
for kind, name in [("pos", "positive_axis_recall"), ("cp", "sdnet_CP_detection")]:
    L = load_rate(kind)
    if L is None:
        continue
    have = set(L["data"])
    rates[name] = {op: {a: arm_point(L, a, op) for a in arms if complete(a, have)} for op in OPS}
    rate_comps[name] = {}
    for op in OPS:
        rate_comps[name][op] = {}
        for a, b in COMPARE:
            if complete(a, have) and complete(b, have):
                d = arm_boot(L, a, op) - arm_boot(L, b, op)
                rate_comps[name][op][f"{a}_vs_{b}"] = {"diff_pp": 100 * (arm_point(L, a, op) - arm_point(L, b, op)),
                                                       "diff_pp_ci95": [100 * v for v in pct(d)]}
    del L

# ---------------------------------------------------------------- synthetic probes
QD = Path(os.environ.get("CRACKCF_DATA", "data")) / "quadruplets"
_train_bg = {json.loads(l)["background"]["crop_id"] for l in open(QD / "v4/specs.jsonl", encoding="utf-8")}
NOVEL = {json.loads(l)["group_id"] for l in open(QD / "v4_heldout_s20261029/specs.jsonl", encoding="utf-8")
         if json.loads(l)["background"]["crop_id"] not in _train_bg}
synth = {}
for sub, keep in [("quads", None), (HELD, None), (HELD + "__novel_background", NOVEL)]:
    t = {}
    for a in arms:
        ps = [EDIR[m] / "strata" / sub.split("__")[0] / f"{m}.csv" for m in arms[a]]
        if not all(p.exists() for p in ps):
            continue
        t[a] = {}
        for opn in ["shadow", "joint", "repair_boundary"]:
            v1, v0, n = [], [], 0
            for p in ps:
                rows = [r for r in read(p) if r["operator"] == opn and (keep is None or r["group_id"] in keep)]
                n = len(rows)
                mpx = sum(float(r["mpx"]) for r in rows)
                v1.append(sum(float(r["native_r95__fp_x01"]) for r in rows) / mpx if mpx else float("nan"))
                v0.append(sum(float(r["native_r95__fp_x00"]) for r in rows) / mpx if mpx else float("nan"))
            t[a][opn] = {"fp_per_mpx_x01": float(np.mean(v1)), "fp_per_mpx_x00": float(np.mean(v0)), "groups": n}
    synth[sub] = t


# ---------------------------------------------------------------- predictions (addendum 1 P1-P6, addendum 2 P7)
def g(leg, a, op="native_r95"):
    return arm_A3.get(leg, {}).get(op, {}).get(a)


def need(*vals):
    return all(v is not None for v in vals)


pred = {}
B, U, P, S = "bcl_noise", "uav1388_background", "pavedistress_background_test", "sdnet_staged_negative"
v = [g(B, "A_x1only"), g(U, "A_x1only"), g(B, "A_x01only"), g(U, "A_x01only"), g(B, "B0"), g(U, "B0")]
pred["P1_mechanism"] = {"values": v, "verdict": None if not need(*v) else bool(
    v[0] >= 1.10 * v[4] and v[1] >= 1.10 * v[5] and v[2] <= v[4] and v[3] <= v[5])}
v = [g(B, "B11_tauhigh"), g(B, "B11_taulow"), g(U, "B11_tauhigh"), g(U, "B11_taulow")]
pred["P2_tau"] = {"values": v, "note": "1 seed per tau level",
                  "verdict": None if not need(*v) else bool(v[0] < v[1] and v[2] < v[3])}
r = rates.get("positive_axis_recall", {}).get("native_r95", {})
v = [g(P, "B12_auto_posthoc"), g(P, "B3_auto"), r.get("B12_auto_posthoc"), r.get("B3_auto")]
pred["P3_complementarity"] = {"values": v, "verdict": None if not need(*v) else bool(
    v[0] <= 2 * v[1] and v[2] >= v[3] + 0.02)}
v = [g(P, "B0"), g(P, "B2"), g(P, "B2_g7"), g(P, "B2_g2")]
if need(*v):
    red = v[0] - v[1]
    pred["P4_data_efficiency"] = {"values": v, "kept_g7": (v[0] - v[2]) / red, "kept_g2": (v[0] - v[3]) / red,
                                  "verdict": bool((v[0] - v[2]) >= 0.9 * red and (v[0] - v[3]) >= 0.5 * red)}
else:
    pred["P4_data_efficiency"] = {"values": v, "verdict": None}
rows5, ok5 = {}, True
for a in ["B2", "B3_auto", "B2_g2", "B2_g7"]:
    for leg in [B, U, S]:
        x, b0 = g(leg, a), g(leg, "B0")
        if not need(x, b0):
            ok5 = None
            continue
        rows5[f"{a}|{leg}"] = x / b0 - 1
        if ok5 is not None and abs(x / b0 - 1) > 0.15:
            ok5 = False
pred["P5_locality"] = {"rel_change_vs_B0": rows5, "verdict": ok5}
v = [g(S, "B11"), g(S, "B0")]
pred["P6_confirmation"] = {"values": v, "verdict": None if not need(*v) else bool(v[0] > v[1])}
h = synth.get(HELD, {})
if "B11" in h and "B0" in h:
    pred["P7_operator_generalisation"] = {
        "values": {o: [h["B11"][o]["fp_per_mpx_x01"], h["B0"][o]["fp_per_mpx_x01"]] for o in h["B0"]},
        "verdict": all(h["B11"][o]["fp_per_mpx_x01"] <= 0.5 * h["B0"][o]["fp_per_mpx_x01"] for o in h["B0"])}
else:
    pred["P7_operator_generalisation"] = {"verdict": None}

r = rates.get("positive_axis_recall", {}).get("native_r95", {})
v = {leg: [g(leg, "B0_v6"), g(leg, "B0")] for leg in [B, P, U]}
if all(need(*x) for x in v.values()) and need(r.get("B0_v6"), r.get("B0")):
    pred["P8_trainer_equivalence"] = {"values": v, "recall_pp": 100 * (r["B0_v6"] - r["B0"]), "verdict": bool(
        all(abs(x[0] / x[1] - 1) <= 0.15 for x in v.values()) and abs(r["B0_v6"] - r["B0"]) <= 0.02)}
else:
    pred["P8_trainer_equivalence"] = {"verdict": None}
v = {leg: [g(leg, "R50_B11"), g(leg, "R50_B0")] for leg in [B, P, U]}
pred["P9_backbone_direction"] = {"values": v, "verdict": None if not all(need(*x) for x in v.values()) else
                                 bool(all(x[0] > x[1] for x in v.values()))}
v = {leg: [g(leg, "R50_B2"), g(leg, "R50_B0")] for leg in [B, P, U]}
pred["P10_backbone_locality"] = {"values": v, "verdict": None if not all(need(*x) for x in v.values()) else bool(
    v[P][0] <= 0.1 * v[P][1] and abs(v[B][0] / v[B][1] - 1) <= 0.15 and abs(v[U][0] / v[U][1] - 1) <= 0.15)}
v = [r.get("R50_B11"), r.get("R50_B0")]
pred["P11_backbone_recall"] = {"values": v, "verdict": None if not need(*v) else bool(v[0] >= v[1])}

report = {"models": {m: {**MODELS[m], "eval_dir": str(EDIR[m])} for m in MODELS}, "legs": legs, "arm_A3": arm_A3,
          "comparisons": comps, "rates": rates, "rate_comparisons": rate_comps, "synthetic_probes": synth,
          "E6_pavedistress": {"px_mm": PX_MM, "m2_per_mpx": M2_PER_MPX, "lane_km_m2": LANE_KM_M2, "arms": e6},
          "predictions": pred, "bootstrap": {"iters": ITERS, "seed": 0, "paired": True}}
(OUT / "program_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float),
                                         encoding="utf-8")
with open(OUT / "arm_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    cols = list(arm_A3) + list(rates)
    w.writerow(["op", "arm", "n_seeds"] + cols)
    for op in OPS:
        for a in sorted(arms):
            w.writerow([op, a, len(arms[a])] + [arm_A3[c][op].get(a, "") if c in arm_A3 else rates[c][op].get(a, "")
                                                for c in cols])
print(json.dumps({"legs": {k: (v["n_images"], len(v["models"])) for k, v in legs.items()},
                  "predictions": {k: v["verdict"] for k, v in pred.items()}}, indent=1))
