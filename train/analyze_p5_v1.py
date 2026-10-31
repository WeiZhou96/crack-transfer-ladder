"""P5 report (prereg_v6 go_no_go): read eval_students_v1 per-image accounts; no inference, no threshold choice.

Per model / leg / operating point: A1 image FAR, A2 FP area ratio (raw binary), A3 filtered FP components per Mpx,
PhantomLen px/Mpx; positive axis (CrackSC+NHA12D) component recall, clDice, clIoU@4. Arm = mean of the 3 seeds.
H1-public (prereg_v6): on both primary legs A3(B11) < A3(B3_auto) with relative drop >= 0.15 and positive-axis
component-recall difference >= -2 pp; |drop - 0.15| <= 0.05 is flagged as borderline for manual review.
Cluster bootstrap over bootstrap_group_1km (10000, seed 0): images of a resampled group enter every model together,
so arm differences are paired. Secondary arms are reported, never substituted into the criterion.
usage: python analyze_p5_v1.py <eval_out_dir> <models.json> <report_dir>
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

E, MODELS, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8")), Path(sys.argv[3])
PREREG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "prereg_v6.yaml", encoding="utf-8"))
G = PREREG["go_no_go"]
DROP, FLOOR, BAND = float(G["H1_public"]["min_relative_drop"]), float(G["H1_public"]["recall_floor_pp"]), \
    float(G["borderline_band_relative"])
LEGS = [l["id"] for l in PREREG["evaluation_negative_legs"]]
PRIMARY = [l["id"] for l in PREREG["evaluation_negative_legs"] if l["role"] == "primary_real_negative"]
OPS = ["native_r95", "fixed_0.5"]
ITERS = 10000
OUT.mkdir(parents=True, exist_ok=True)


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


arms = defaultdict(list)
for m, s in MODELS.items():
    arms[s["arm"]].append(m)
for a in arms:
    arms[a].sort()

# ---- negative legs: per model arrays aligned on a common image order
neg = {}
for leg in LEGS:
    order, groups, data = None, None, {}
    for m in MODELS:
        rows = read(E / "test" / m / f"{leg}.csv")
        ids = [r["image_id"] for r in rows]
        if order is None:
            order, groups = ids, [r["bootstrap_group_1km"] for r in rows]
        elif ids != order:
            raise SystemExit(f"image order differs: {m} {leg}")
        data[m] = {op: {k: np.array([float(r[f"{op}__{k}"]) for r in rows]) for k in
                        ["mpx", "fp_components", "phantom_len_px", "fp_area_px", "image_fired"]} for op in OPS}
    neg[leg] = (order, groups, data)

pos = {}
for m in MODELS:
    rows = read(E / "test" / m / "positive_axis.csv")
    pos[m] = {op: {"hit": sum(int(r[f"{op}__n_hit"]) for r in rows), "gt": sum(int(r[f"{op}__n_gt"]) for r in rows),
                   "cldice": float(np.mean([float(r[f"{op}__cldice"]) for r in rows])),
                   "cliou4": float(np.mean([float(r[f"{op}__cliou4"]) for r in rows])),
                   "by_dataset": {ds: {"cldice": float(np.mean([float(r[f"{op}__cldice"]) for r in rows
                                                                if r["source_dataset"] == ds])),
                                       "recall": sum(int(r[f"{op}__n_hit"]) for r in rows if r["source_dataset"] == ds)
                                       / max(1, sum(int(r[f"{op}__n_gt"]) for r in rows if r["source_dataset"] == ds))}
                                  for ds in sorted({r["source_dataset"] for r in rows})}}
              for op in OPS}
thr = {m: json.loads((E / "dev" / m / "threshold.json").read_text())["native_r95"] for m in MODELS}

# ---- per-model table
table = []
for m, s in MODELS.items():
    for leg in LEGS:
        _, _, data = neg[leg]
        for op in OPS:
            d = data[m][op]
            table.append({"model": m, "arm": s["arm"], "seed": s["seed"], "role": s["role"], "leg": leg, "op": op,
                          "threshold": thr[m] if op == "native_r95" else 0.5, "n_images": len(d["mpx"]),
                          "A1_image_far": d["image_fired"].mean(),
                          "A2_fp_area_ratio": d["fp_area_px"].sum() / (d["mpx"].sum() * 1e6),
                          "A3_fp_per_mpx": d["fp_components"].sum() / d["mpx"].sum(),
                          "phantom_px_per_mpx": d["phantom_len_px"].sum() / d["mpx"].sum(),
                          "pos_component_recall": pos[m][op]["hit"] / pos[m][op]["gt"],
                          "pos_cldice": pos[m][op]["cldice"], "pos_cliou4": pos[m][op]["cliou4"]})
with open(OUT / "per_model.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)


# ---- arm means and paired cluster bootstrap
def arm_a3(leg, op, arm):
    _, _, data = neg[leg]
    return float(np.mean([data[m][op]["fp_components"].sum() / data[m][op]["mpx"].sum() for m in arms[arm]]))


def arm_recall(op, arm):
    return float(np.mean([pos[m][op]["hit"] / pos[m][op]["gt"] for m in arms[arm]]))


# group-level sums and multinomial resampling counts (identical for every model: paired bootstrap)
gsum, counts = {}, {}
rng = np.random.default_rng(0)
for leg in LEGS:
    _, groups, data = neg[leg]
    keys = sorted(set(groups))
    gi = np.array([keys.index(g) for g in groups]) if len(keys) < 2000 else \
        np.array([{k: i for i, k in enumerate(keys)}[g] for g in groups])
    counts[leg] = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                            for _ in range(ITERS)]).astype(np.float64)
    gsum[leg] = {m: {op: (np.bincount(gi, data[m][op]["fp_components"], len(keys)),
                          np.bincount(gi, data[m][op]["mpx"], len(keys))) for op in OPS} for m in data}


def boot_arm_a3(leg, op, arm):
    return np.mean([(counts[leg] @ gsum[leg][m][op][0]) / (counts[leg] @ gsum[leg][m][op][1]) for m in arms[arm]], 0)


def compare(a, b, op):
    out = {}
    for leg in LEGS:
        ma, mb = arm_a3(leg, op, a), arm_a3(leg, op, b)
        xa, xb = boot_arm_a3(leg, op, a), boot_arm_a3(leg, op, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            drops = np.where(xb > 0, (xb - xa) / xb, np.nan)
        _, _, data = neg[leg]
        per_seed = [{"seed": MODELS[x]["seed"], "a3_a": data[x][op]["fp_components"].sum() / data[x][op]["mpx"].sum(),
                     "a3_b": data[y][op]["fp_components"].sum() / data[y][op]["mpx"].sum()}
                    for x, y in zip(arms[a], arms[b])]
        out[leg] = {"A3_a": ma, "A3_b": mb, "relative_drop": (mb - ma) / mb if mb > 0 else None,
                    "drop_ci95": [float(np.nanpercentile(drops, 2.5)), float(np.nanpercentile(drops, 97.5))],
                    "p_drop_le_0": float(np.mean(drops <= 0)),
                    "seeds_a_lower": sum(p["a3_a"] < p["a3_b"] for p in per_seed), "per_seed": per_seed}
    out["recall_diff_pp"] = 100 * (arm_recall(op, a) - arm_recall(op, b))
    return out


report = {"prereg": PREREG["version"], "criteria": {"min_relative_drop": DROP, "recall_floor_pp": FLOOR,
                                                     "borderline_band": BAND, "primary_legs": PRIMARY},
          "bootstrap": {"unit": "bootstrap_group_1km", "iters": ITERS, "seed": 0, "paired": True},
          "arm_A3": {op: {leg: {a: arm_a3(leg, op, a) for a in arms} for leg in LEGS} for op in OPS},
          "arm_recall": {op: {a: arm_recall(op, a) for a in arms} for op in OPS},
          "arm_A1": {op: {leg: {a: float(np.mean([neg[leg][2][m][op]["image_fired"].mean() for m in arms[a]]))
                                for a in arms} for leg in LEGS} for op in OPS},
          "comparisons": {}}
for op in OPS:
    report["comparisons"][op] = {f"{a}_vs_{b}": compare(a, b, op) for a, b in
                                 [("B11", "B3_auto"), ("B11", "B10"), ("B11", "B0"), ("B11", "B2"), ("B10", "B3_auto"),
                                  ("B3_auto", "B2"), ("B11", "B1")]}
h = report["comparisons"]["native_r95"]["B11_vs_B3_auto"]
legs_pass = {leg: h[leg]["relative_drop"] is not None and h[leg]["relative_drop"] >= DROP for leg in PRIMARY}
border = {leg: h[leg]["relative_drop"] is not None and abs(h[leg]["relative_drop"] - DROP) <= BAND for leg in PRIMARY}
recall_ok = h["recall_diff_pp"] >= FLOOR
report["H1_public"] = {"legs_pass": legs_pass, "borderline": border, "recall_diff_pp": h["recall_diff_pp"],
                       "recall_ok": recall_ok,
                       "verdict": ("borderline_human_review" if any(border.values()) else
                                   "pass" if all(legs_pass.values()) and recall_ok else "fail")}
p = report["comparisons"]["native_r95"]["B11_vs_B10"]
report["pairing_value"] = {leg: {"relative_drop": p[leg]["relative_drop"], "ci95": p[leg]["drop_ci95"]} for leg in LEGS}
(OUT / "p5_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=lambda o: o.item()),
                                    encoding="utf-8")
print(json.dumps({"H1_public": report["H1_public"], "arm_A3_native": report["arm_A3"]["native_r95"],
                  "arm_recall_native": report["arm_recall"]["native_r95"]}, indent=1, ensure_ascii=False,
                 default=lambda o: o.item()))
