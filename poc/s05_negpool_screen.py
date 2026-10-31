# -*- coding: utf-8 -*-
"""Step 3: three-state screening of the negative pool (fully automatic admission).

Four sources of evidence: source label (images entering this script already carry a non-crack
source label), a committee of crack models, a VLM verdict (optional, external), and
morphology/quality checks. Three-state rule:
  accept        fewer than 2 heterogeneous committee members "form a long component at the same
                location", the VLM does not report a crack, and quality is acceptable
  hard_negative the designated baseline member fires, but neither the other members nor the VLM
                support a "real crack"
  drop          >= 2 heterogeneous members agree on a detection at the same location, or the VLM
                reports a possible crack, or quality is unacceptable (when in doubt, discard)

Q/H separation: committee firing alone is not a reason to remove an image -- only agreeing
co-located detections escalate it to "suspected real crack"; the responses of all members are
recorded in the H telemetry output, since high-response samples are exactly what makes
hard_negative valuable.

Committee prediction directory contract: <pred_root>/<member_name>/<flattened image_id>.png
(16-bit 0-65535 probability). Flattened image_id = image_id with '/' replaced by '__'.
Members are declared in configs/committee.yaml: name / arch_family / threshold / role(baseline|screen).
The admission committee and the B3-auto crack-exclusion committee must have disjoint members
(this script consumes only screen-role members + baseline).

Usage:
  python -X utf8 s05_negpool_screen.py --data-root <ROOT> --committee ../configs/committee.yaml \
      --pred-root <PRED_ROOT> [--vlm vlm_results.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from common import (DEFAULT_MIN_SKEL_PX, binarize_prediction, dilate, filter_components,
                    load_gray, load_probability_png, load_verified_freeze, read_jsonl,
                    setup_logger, sha256_file)

LOG = setup_logger("s05")

AGREE_IOU = 0.30          # long components of two members count as "same location" if their dilated IoU reaches this
AGREE_DILATE_PX = 4
BLUR_MIN_LAPVAR = 25.0    # quality floor: Laplacian variance (too blurry) and exposure out of range
EXPOSURE_LO, EXPOSURE_HI = 10, 245


def flat(image_id: str) -> str:
    return image_id.replace("/", "__")


def member_components(pred_path: Path, threshold: float) -> np.ndarray | None:
    if not pred_path.exists():
        return None
    arr = load_probability_png(pred_path)
    binary = binarize_prediction(arr, threshold)
    filtered, comps = filter_components(binary, min_area_px=0, min_skel_px=DEFAULT_MIN_SKEL_PX)
    return filtered if comps else np.zeros_like(binary)


def pair_same_location(a: np.ndarray, b: np.ndarray) -> bool:
    if a is None or b is None or a.sum() == 0 or b.sum() == 0:
        return False
    na, la = cv2.connectedComponents(a.astype(np.uint8), connectivity=8)
    nb, lb = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    for ia in range(1, na):
        da = dilate((la == ia).astype(np.uint8), AGREE_DILATE_PX) > 0
        for ib in range(1, nb):
            db = dilate((lb == ib).astype(np.uint8), AGREE_DILATE_PX) > 0
            inter = np.logical_and(da, db).sum()
            union = np.logical_or(da, db).sum()
            if union > 0 and inter / union >= AGREE_IOU:
                return True
    return False


def quality_ok(img_path: Path) -> tuple[bool, dict]:
    g = load_gray(img_path).astype(np.float64)
    lapvar = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean = float(g.mean())
    ok = lapvar >= BLUR_MIN_LAPVAR and EXPOSURE_LO <= mean <= EXPOSURE_HI
    return ok, {"lap_var": round(lapvar, 1), "mean_gray": round(mean, 1)}


def load_train_negative_rows(assignments_path: Path) -> list[dict]:
    rows = []
    with open(assignments_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["pool"] == "train_negative" and row["split"] == "train":
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--committee", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--vlm", default="", help="Optional: VLM verdict jsonl ({image_id, has_crack: bool})")
    args = ap.parse_args()

    root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    committee_path = Path(args.committee).resolve()
    com = yaml.safe_load(committee_path.read_text(encoding="utf-8"))["members"]
    screen = [m for m in com if m.get("role", "screen") == "screen"]
    baselines = [m for m in com if m.get("role") == "baseline"]
    fams = {m["arch_family"] for m in screen}
    LOG.info(f"screening committee: {len(screen)} members (architecture families {sorted(fams)}); {len(baselines)} baseline(s)")
    if len(fams) < 2:
        raise SystemExit("screening committee has < 2 architecture families; refusing to run three-state screening")

    vlm = {}
    if args.vlm:
        for r in read_jsonl(Path(args.vlm)):
            vlm[r["image_id"]] = bool(r.get("has_crack"))

    # The pool is drawn strictly from the frozen train side; test must not enter admission/B2/B3-auto/training artifacts.
    _, freeze = load_verified_freeze(root)
    if freeze.get("committee_sha256") != sha256_file(committee_path):
        raise SystemExit("committee.yaml hash does not match the freeze snapshot; refusing to screen")
    assignments_path = root / "splits" / freeze.get("split_assignments_file", "split_assignments.csv")
    rows = load_train_negative_rows(assignments_path)
    reg = {r["image_id"]: r for r in read_jsonl(root / "registry" / "registry.jsonl")}
    LOG.info(f"{len(rows)} negative images to screen")

    out = root / "negpool"
    out.mkdir(parents=True, exist_ok=True)
    results, telemetry = [], []

    for r in tqdm(rows, unit="img"):
        iid = r["image_id"]
        img_path = Path(reg[iid]["path"])
        q_ok, q = quality_ok(img_path)

        masks = {}
        for m in screen + baselines:
            masks[m["name"]] = member_components(pred_root / m["name"] / f"{flat(iid)}.png",
                                                 m.get("threshold", 0.5))
        missing = [n for n, v in masks.items() if v is None]
        fired = {n: v for n, v in masks.items() if v is not None and v.sum() > 0}

        # Heterogeneous agreement: two screening members from different architecture families form a long component at the same location
        consensus = False
        sn = [m for m in screen if masks.get(m["name"]) is not None]
        for i in range(len(sn)):
            for j in range(i + 1, len(sn)):
                if sn[i]["arch_family"] == sn[j]["arch_family"]:
                    continue
                if pair_same_location(masks[sn[i]["name"]], masks[sn[j]["name"]]):
                    consensus = True
                    break
            if consensus:
                break

        vlm_missing = bool(args.vlm) and iid not in vlm
        vlm_crack = vlm.get(iid, False)
        baseline_fired = any(m["name"] in fired for m in baselines)

        if missing or vlm_missing:
            missing_reason = [f"missing_prediction:{','.join(missing)}"] if missing else []
            if vlm_missing:
                missing_reason.append("missing_vlm_verdict")
            state, reason = "pending", ";".join(missing_reason)
        elif not q_ok:
            state, reason = "drop", "quality"
        elif consensus or vlm_crack:
            state, reason = "drop", "consensus" if consensus else "vlm"
        elif baseline_fired:
            state, reason = "hard_negative", "baseline_fired_unsupported"
        else:
            state, reason = "accept", ""

        results.append({"image_id": iid, "group_id": r["group_id"], "split": r["split"],
                        "state": state, "reason": reason, **q})
        telemetry.append({"image_id": iid, "group_id": r["group_id"], "split": r["split"],
                          **{f"fp_px_{n}": int(v.sum()) if v is not None else None
                             for n, v in masks.items()}})

    for state in ("accept", "hard_negative", "drop", "pending"):
        sub = [x for x in results if x["state"] == state]
        with open(out / f"{state}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "group_id", "split", "state", "reason",
                                                  "lap_var", "mean_gray"])
            w.writeheader()
            w.writerows(sub)
        LOG.info(f"  {state}: {len(sub)}")
    with open(out / "h_telemetry.csv", "w", newline="", encoding="utf-8-sig") as f:
        if telemetry:
            w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
            w.writeheader()
            w.writerows(telemetry)
    LOG.info(f"three-state lists and H telemetry -> {out} (proportions are reported in the paper; rerun once predictions for pending images are complete)")


if __name__ == "__main__":
    main()
