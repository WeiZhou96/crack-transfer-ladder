"""E3 three-state screen under R20 (see marking_r20.py). Identical to the frozen poc/s05_negpool_screen.py (same
committee.yaml, same predictions, same functions imported from s05, same thresholds and quality check), except that
for PaveDistress frames each screen member's components are masked by NOT marking before the cross-family
same-location consensus. Output: <data_root>/negpool_r20/{accept,hard_negative,drop,pending}.csv, h_telemetry.csv,
summary.json (with the frozen-rule counts for comparison and per-state marking rates).
usage: python negpool_screen_r20.py <code_dir> <data_root> <pred_root> [--workers 32]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("pred_root")
ap.add_argument("--workers", type=int, default=32)
args = ap.parse_args()
code, root, pred_root = Path(args.code_dir), Path(args.data_root), Path(args.pred_root)
sys.path.insert(0, str(code / "poc"))
sys.path.insert(0, str(code / "train"))
from common import DEFAULT_MIN_SKEL_PX, filter_components, load_verified_freeze, read_jsonl, sha256_file  # noqa: E402
from s05_negpool_screen import (flat, load_train_negative_rows, member_components, pair_same_location,  # noqa: E402
                                quality_ok)
from marking_r20 import LASER_SOURCES, marking_mask  # noqa: E402

committee_path = (code / "configs/committee.yaml").resolve()
com = yaml.safe_load(committee_path.read_text(encoding="utf-8"))["members"]
screen = [m for m in com if m.get("role", "screen") == "screen"]
baselines = [m for m in com if m.get("role") == "baseline"]
_, freeze = load_verified_freeze(root)
if freeze.get("committee_sha256") != sha256_file(committee_path):
    raise SystemExit("committee.yaml hash differs from the freeze")
rows = load_train_negative_rows(root / "splits" / freeze.get("split_assignments_file"))
reg = {r["image_id"]: r for r in read_jsonl(root / "registry" / "registry.jsonl")}


def one(r):
    iid = r["image_id"]
    g = reg[iid]
    q_ok, q = quality_ok(Path(g["path"]))
    masks = {m["name"]: member_components(pred_root / m["name"] / f"{flat(iid)}.png", m.get("threshold", 0.5))
             for m in screen + baselines}
    missing = [n for n, v in masks.items() if v is None]
    mark = marking_mask(g["path"]) if g["source_dataset"] in LASER_SOURCES else None
    marking_frac = float(mark.mean()) if mark is not None else 0.0
    def outside(v):              # mask out markings, then re-apply the same component filter as member_components
        if v is None or mark is None:
            return v
        kept, comps = filter_components((v.astype(bool) & ~mark).astype(np.uint8), min_area_px=0,
                                        min_skel_px=DEFAULT_MIN_SKEL_PX)
        return kept if comps else np.zeros_like(v)
    screened = {n: outside(v) for n, v in masks.items()}
    consensus = False
    sn = [m for m in screen if screened.get(m["name"]) is not None]
    for i in range(len(sn)):
        for j in range(i + 1, len(sn)):
            if sn[i]["arch_family"] != sn[j]["arch_family"] and \
                    pair_same_location(screened[sn[i]["name"]], screened[sn[j]["name"]]):
                consensus = True
                break
        if consensus:
            break
    baseline_fired = any(masks[m["name"]] is not None and masks[m["name"]].sum() > 0 for m in baselines)
    if missing:
        state, reason = "pending", "missing_prediction:" + ",".join(missing)
    elif not q_ok:
        state, reason = "drop", "quality"
    elif consensus:
        state, reason = "drop", "consensus_outside_marking"
    elif baseline_fired:
        state, reason = "hard_negative", "baseline_fired_unsupported"
    else:
        state, reason = "accept", ""
    res = {"image_id": iid, "group_id": r["group_id"], "split": r["split"], "state": state, "reason": reason, **q,
           "has_marking": bool(mark is not None and mark.any()), "marking_frac": round(marking_frac, 5)}
    tel = {"image_id": iid, **{f"fp_px_{n}": int(v.sum()) if v is not None else None for n, v in masks.items()},
           **{f"fp_px_outside_marking_{n}": int(v.sum()) if v is not None else None for n, v in screened.items()}}
    return res, tel


with Pool(args.workers) as p:
    out = p.map(one, rows, chunksize=4)
results = [a for a, _ in out]
tele = [b for _, b in out]
od = root / "negpool_r20"
od.mkdir(exist_ok=False)
fields = ["image_id", "group_id", "split", "state", "reason", "lap_var", "mean_gray", "has_marking", "marking_frac"]
for state in ("accept", "hard_negative", "drop", "pending"):
    with open(od / f"{state}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([x for x in results if x["state"] == state])
with open(od / "h_telemetry.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(tele[0]))
    w.writeheader()
    w.writerows(tele)
frozen = {s: sum(1 for _ in csv.DictReader(open(root / "negpool" / f"{s}.csv", encoding="utf-8-sig")))
          for s in ("accept", "hard_negative", "drop", "pending")}
summary = {"rule": "R20: frozen s05 + lane-marking exclusion from committee consensus on PaveDistress frames",
           "counts_r20": dict(Counter(x["state"] for x in results)), "counts_frozen_rule": frozen,
           "marking_rate_by_state": {s: round(float(np.mean([x["has_marking"] for x in results if x["state"] == s])), 4)
                                     for s in set(x["state"] for x in results)},
           "reasons": dict(Counter(x["reason"] for x in results))}
(od / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=1, ensure_ascii=False))
