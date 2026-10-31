# -*- coding: utf-8 -*-
"""E-D1-public: public-negative version of the leaderboard-blindness experiment
(public-data supplementary panel for E-D1).

Inputs:
  --neg-list   csv (columns image_id[, mpx]) -- independent real negative list (PaveDistress
               background-test, SDNET non-crack samples, etc.; taken from splits/negpool outputs)
  --pred-root  <PRED_ROOT>/<model>/<flat(image_id)>.png (16-bit 0-65535 continuous probability)
  --models     comma-separated model names (same as the pred-root subdirectories)
  --val-list    Pavementscapes-val positive list (must contain image_id,group_id,mask_path, or use --gt-root)
                -- per-model calibration of t^proxy_R95
  --pos-list    unified CrackSC∪NHA12D positive test list, used to report clDice, clIoU@4px and component recall

Three accounts per model (preregistered visualisation of the unit mismatch):
  A1 image-level FAR = fraction of images that fire (any component surviving the filter fires)
  A2 FP area ratio = FP pixels / total pixels
  A3 component-level FP components/Mpx (primary measure) + PhantomLen px/Mpx + CleanImg%
Primary report: 1 km block-clustered bootstrap 95% CI, with 100 m/1 km sensitivity alongside; Spearman
is computed only between this project's unified positive axis and the false-alarm axis, and
self-reported scores from papers must not be used as input; secondary report at fixed t=0.5.

Usage:
  python -X utf8 s06_ed1_public.py --data-root <ROOT> --neg-list <csv> --pred-root <DIR> \
      --models deepcrack,hrsegnet,... [--threshold 0.5] [--leaderboard lb.csv] [--plot]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from common import (binarize_prediction, component_recall, dilate, load_binary_mask,
                    load_probability_png, load_verified_freeze, negative_image_accounts,
                    setup_logger, sha256_file, skeletonize_mask)

LOG = setup_logger("s06")


class TargetRecallUnreachable(RuntimeError):
    pass


def flat(image_id: str) -> str:
    return image_id.replace("/", "__")


def read_list(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group_bootstrap_ci(per_image: list[dict], num: str, den: str, iters: int = 10000,
                       seed: int = 0, group_key: str = "group_id") -> tuple[float, float, float]:
    """Bootstrap 95% CI of the ratio metric sum(num)/sum(den), clustered by the given correlation block."""
    grouped = defaultdict(lambda: [0.0, 0.0])
    for row in per_image:
        gid = row.get(group_key)
        if not gid:
            raise ValueError(f"evaluation list lacks {group_key}; tiles/frames must not be treated as independent samples")
        grouped[gid][0] += float(row[num])
        grouped[gid][1] += float(row[den])
    rng = np.random.default_rng(seed)
    nums = np.array([v[0] for v in grouped.values()], dtype=np.float64)
    dens = np.array([v[1] for v in grouped.values()], dtype=np.float64)
    point = nums.sum() / dens.sum()
    n = len(nums)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        d = dens[idx].sum()
        stats.append(nums[idx].sum() / d if d > 0 else 0.0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def resolve_gt(row: dict, gt_root: Path | None) -> Path:
    declared = row.get("mask_path") or row.get("gt_path")
    if declared:
        return Path(declared)
    if gt_root is not None:
        return gt_root / f"{flat(row['image_id'])}.png"
    raise ValueError(f"{row.get('image_id')}: list has no mask_path/gt_path and --gt-root was not given")


def load_positive_pairs(rows: list[dict], pred_root: Path, model: str,
                        gt_root: Path | None) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    pairs, missing = [], []
    for row in rows:
        pred_path = pred_root / model / f"{flat(row['image_id'])}.png"
        gt_path = resolve_gt(row, gt_root)
        if not pred_path.exists() or not gt_path.exists():
            missing.append(f"{row['image_id']} pred={pred_path.exists()} gt={gt_path.exists()}")
            continue
        pairs.append((row, load_probability_png(pred_path), load_binary_mask(gt_path)))
    if missing:
        raise FileNotFoundError(
            f"{model}: {len(missing)} positive predictions/GT missing (e.g. {missing[0]}); formal evaluation must not shrink the denominator by dropping images"
        )
    return pairs


def recall_at_threshold(pairs: list[tuple[dict, np.ndarray, np.ndarray]], threshold: float,
                        min_area_px: int, min_skel_px: int,
                        dilate_px: int, cover: float) -> tuple[float, int, int]:
    hits = total = 0
    for _, pred_arr, gt in pairs:
        cr = component_recall(
            gt, binarize_prediction(pred_arr, threshold),
            dilate_px=dilate_px, cover=cover,
            min_area_px=min_area_px, min_skel_px=min_skel_px,
        )
        hits += cr["n_hit"]
        total += cr["n_gt"]
    if total == 0:
        raise ValueError("validation positive list has no GT component that passes the component filter")
    return hits / total, hits, total


def safe_ratio(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    return numerator / denominator if denominator > 0 else empty_value


def centerline_metrics(prediction: np.ndarray, ground_truth: np.ndarray,
                       tolerance_px: int = 4) -> tuple[float, float]:
    """Return standard clDice and a tolerance clIoU defined by bidirectional skeleton matching."""
    pred = (prediction > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    pred_skel = skeletonize_mask(pred)
    gt_skel = skeletonize_mask(gt)

    topology_precision = safe_ratio(float((pred_skel & gt).sum()), float(pred_skel.sum()),
                                    empty_value=float(not np.any(gt)))
    topology_sensitivity = safe_ratio(float((gt_skel & pred).sum()), float(gt_skel.sum()),
                                      empty_value=float(not np.any(pred)))
    denominator = topology_precision + topology_sensitivity
    cldice = (2.0 * topology_precision * topology_sensitivity / denominator
              if denominator > 0 else 0.0)

    pred_match = safe_ratio(float((pred_skel & dilate(gt_skel, tolerance_px)).sum()),
                            float(pred_skel.sum()), empty_value=float(not np.any(gt_skel)))
    gt_match = safe_ratio(float((gt_skel & dilate(pred_skel, tolerance_px)).sum()),
                          float(gt_skel.sum()), empty_value=float(not np.any(pred_skel)))
    cliou_denominator = pred_match + gt_match - pred_match * gt_match
    cliou = (pred_match * gt_match / cliou_denominator
             if cliou_denominator > 0 else 0.0)
    return cldice, cliou


def evaluate_unified_positive_axis(
        pairs: list[tuple[dict, np.ndarray, np.ndarray]], threshold: float,
        tolerance_px: int = 4) -> dict:
    by_dataset = defaultdict(list)
    for row, probability, gt in pairs:
        pred = binarize_prediction(probability, threshold)
        by_dataset[row["source_dataset"]].append(centerline_metrics(pred, gt, tolerance_px))
    per_dataset = {
        dataset: {
            "n": len(values),
            "cldice": float(np.mean([value[0] for value in values])),
            "cliou_tolerance": float(np.mean([value[1] for value in values])),
        }
        for dataset, values in sorted(by_dataset.items())
    }
    all_values = [value for values in by_dataset.values() for value in values]
    return {
        "n": len(all_values),
        "cldice": float(np.mean([value[0] for value in all_values])),
        "cliou_tolerance": float(np.mean([value[1] for value in all_values])),
        "tolerance_px": tolerance_px,
        "by_dataset": per_dataset,
    }


def validate_unified_positive_sources(rows: list[dict], required_sources: set[str]) -> None:
    observed = {row.get("source_dataset") for row in rows}
    if observed != required_sources:
        raise SystemExit(
            f"unified positive axis sources must be exactly {sorted(required_sources)}, got {sorted(observed)}"
        )


def bootstrap_spearman_ci(x: list[float], y: list[float], iters: int = 10000,
                          seed: int = 20260831) -> tuple[float, float, float]:
    from scipy.stats import spearmanr
    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    statistics = []
    n = len(x)
    for _ in range(iters):
        indices = rng.integers(0, n, n)
        statistic = float(spearmanr(np.asarray(x)[indices], np.asarray(y)[indices]).statistic)
        if np.isfinite(statistic):
            statistics.append(statistic)
    if not statistics:
        return point, float("nan"), float("nan")
    low, high = np.percentile(statistics, [2.5, 97.5])
    return point, float(low), float(high)


def select_t_proxy_r95(pairs: list[tuple[dict, np.ndarray, np.ndarray]], target: float,
                       min_area_px: int, min_skel_px: int,
                       dilate_px: int, cover: float) -> tuple[float, float]:
    """Choose the highest threshold, among the actual positive probability values of the predictions, that meets the target recall."""
    unique_parts = [np.unique(pred[pred > 0]) for _, pred, _ in pairs if np.any(pred > 0)]
    if not unique_parts:
        raise TargetRecallUnreachable("all validation predictions are 0; R95 is unreachable")
    thresholds = np.unique(np.concatenate(unique_parts)).astype(np.float64)
    lowest_recall, _, _ = recall_at_threshold(
        pairs, float(thresholds[0]), min_area_px, min_skel_px, dilate_px, cover)
    if lowest_recall < target:
        raise TargetRecallUnreachable(
            f"component recall at the lowest positive threshold {thresholds[0]:.8f} is {lowest_recall:.6f} < {target:.6f}"
        )
    lo, hi, best_index, best_recall = 0, len(thresholds) - 1, 0, lowest_recall
    while lo <= hi:
        index = (lo + hi) // 2
        recall, _, _ = recall_at_threshold(
            pairs, float(thresholds[index]), min_area_px, min_skel_px, dilate_px, cover)
        if recall >= target:
            best_index, best_recall = index, recall
            lo = index + 1
        else:
            hi = index - 1
    return float(thresholds[best_index]), best_recall


def evaluate_negatives(rows: list[dict], pred_root: Path, model: str, threshold: float,
                       min_area_px: int, min_skel_px: int) -> list[dict]:
    per_image, missing = [], []
    for row in rows:
        pred_path = pred_root / model / f"{flat(row['image_id'])}.png"
        if not pred_path.exists():
            missing.append(row["image_id"])
            continue
        pred = binarize_prediction(load_probability_png(pred_path), threshold)
        acc = negative_image_accounts(pred, min_area_px=min_area_px, min_skel_px=min_skel_px)
        acc.update({
            "image_id": row["image_id"],
            "group_id": row.get("group_id"),
            "bootstrap_group_100m": row.get("bootstrap_group_100m") or row.get("group_id"),
            "bootstrap_group_1km": row.get("bootstrap_group_1km") or row.get("group_id"),
        })
        per_image.append(acc)
    if missing:
        raise FileNotFoundError(
            f"{model}: predictions missing for {len(missing)}/{len(rows)} negatives (e.g. {missing[0]}); "
            "formal evaluation must not shrink the denominator by dropping images"
        )
    return per_image


def validate_frozen_split(rows: list[dict], label: str, expected_split: str | set[str],
                          assignments: dict[str, str]) -> None:
    expected = {expected_split} if isinstance(expected_split, str) else set(expected_split)
    if not rows:
        raise SystemExit(f"{label} is empty")
    for row in rows:
        image_id = row.get("image_id", "")
        declared = row.get("split")
        if declared not in expected:
            raise SystemExit(
                f"{label}: {image_id} declares split={declared!r}, required {sorted(expected)!r}"
            )
        source_id = row.get("source_image") or image_id
        frozen = assignments.get(source_id)
        if frozen is None:
            raise SystemExit(f"{label}: {image_id} cannot be mapped to a frozen split (source={source_id})")
        if frozen not in expected:
            raise SystemExit(
                f"{label}: {image_id} has frozen split={frozen!r}, required {sorted(expected)!r}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--neg-list", required=True)
    ap.add_argument("--val-list", required=True, help="Positive list used for per-model calibration of t^proxy_R95")
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fixed-threshold", type=float, default=0.5)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--bootstrap-iters", type=int, default=None,
                    help="Override prereg metrics.bootstrap.iters; production default 10000")
    ap.add_argument("--pos-list", default="")
    ap.add_argument("--gt-root", default="")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    prereg_path = Path(args.prereg)
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    _, freeze = load_verified_freeze(root)
    if freeze.get("prereg_sha256") != sha256_file(prereg_path):
        raise SystemExit("prereg.yaml hash does not match the freeze snapshot; refusing to evaluate")
    metric_cfg = prereg["metrics"]
    bootstrap_cfg = metric_cfg.get("bootstrap", {})
    bootstrap_iters = (args.bootstrap_iters if args.bootstrap_iters is not None
                       else int(bootstrap_cfg.get("iters", 10000)))
    primary_bootstrap_key = bootstrap_cfg.get("primary_group", "bootstrap_group_1km")
    sensitivity_keys = bootstrap_cfg.get("sensitivity_groups", {
        "100m": "bootstrap_group_100m", "1km": "bootstrap_group_1km",
    })
    positive_axis_cfg = prereg.get("ed1", {}).get("positive_axis", {})
    filt = metric_cfg["component_filter_px"]
    det = metric_cfg["detected_criterion"]
    min_area_px, min_skel_px = int(filt["min_area"]), int(filt["min_skel"])
    dilate_px, cover = int(det["dilate_px"]), float(det["skeleton_cover"])
    pred_root = Path(args.pred_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    negs = read_list(Path(args.neg_list))
    vals = read_list(Path(args.val_list))
    assignments_path = root / "splits" / freeze.get("split_assignments_file", "split_assignments.csv")
    assignments = {
        row["image_id"]: row["split"] for row in read_list(assignments_path)
    }
    validate_frozen_split(vals, "val-list", "proxy_val", assignments)
    validate_frozen_split(negs, "neg-list", {"test", "proxy_test"}, assignments)
    pos = read_list(Path(args.pos_list)) if args.pos_list else []
    if positive_axis_cfg.get("required") and not pos:
        raise SystemExit("preregistration requires the unified CrackSC∪NHA12D positive axis; --pos-list must be given")
    if pos:
        validate_frozen_split(pos, "pos-list", "test", assignments)
        required_sources = set(positive_axis_cfg.get("sources", ["cracksc", "nha12d"]))
        validate_unified_positive_sources(pos, required_sources)
    for label, rows in (("neg-list", negs), ("val-list", vals)):
        missing_group = [r.get("image_id", "?") for r in rows if not r.get("group_id")]
        if missing_group:
            raise SystemExit(f"{label} lacks group_id (e.g. {missing_group[0]})")
    overlap = {r["group_id"] for r in negs} & {r["group_id"] for r in vals}
    if overlap:
        raise SystemExit(f"val and negative test share groups (e.g. {sorted(overlap)[0]})")
    LOG.info(f"{len(negs)} negatives × {len(models)} models")

    summary, threshold_failures = [], []
    for model in models:
        val_pairs = load_positive_pairs(vals, pred_root, model,
                                        Path(args.gt_root) if args.gt_root else None)
        try:
            threshold, val_recall = select_t_proxy_r95(
                val_pairs, args.target_recall, min_area_px, min_skel_px, dilate_px, cover)
        except TargetRecallUnreachable as error:
            threshold_failures.append({
                "model": model,
                "status": "t_proxy_r95_unreachable",
                "reason": str(error),
            })
            LOG.error(f"{model}: {error}")
            continue
        per_image = evaluate_negatives(negs, pred_root, model, threshold, min_area_px, min_skel_px)
        fixed = evaluate_negatives(negs, pred_root, model, args.fixed_threshold,
                                   min_area_px, min_skel_px)

        fp_mpx, lo, hi = group_bootstrap_ci(
            per_image, "fp_components", "mpx", bootstrap_iters,
            group_key=primary_bootstrap_key)
        ph_mpx, plo, phi = group_bootstrap_ci(
            per_image, "phantom_len_px", "mpx", bootstrap_iters,
            group_key=primary_bootstrap_key)
        sensitivity = {}
        for label, group_key in sensitivity_keys.items():
            _, fp_lo, fp_hi = group_bootstrap_ci(
                per_image, "fp_components", "mpx", bootstrap_iters,
                group_key=group_key)
            _, ph_lo, ph_hi = group_bootstrap_ci(
                per_image, "phantom_len_px", "mpx", bootstrap_iters,
                group_key=group_key)
            sensitivity[label] = {
                "group_key": group_key,
                "n_groups": len({r[group_key] for r in per_image}),
                "A3_ci": f"[{fp_lo:.3f},{fp_hi:.3f}]",
                "phantom_ci": f"[{ph_lo:.1f},{ph_hi:.1f}]",
            }
        n = len(per_image)
        row = {
            "model": model, "n_images": n, "n_groups": len({r["group_id"] for r in per_image}),
            "bootstrap_primary": primary_bootstrap_key,
            "bootstrap_iters": bootstrap_iters,
            "t_proxy_r95": round(threshold, 6), "val_component_recall": round(val_recall, 4),
            "A3_fp_per_mpx": round(fp_mpx, 3), "A3_ci": f"[{lo:.3f},{hi:.3f}]",
            "phantom_px_per_mpx": round(ph_mpx, 1), "phantom_ci": f"[{plo:.1f},{phi:.1f}]",
            "A1_image_far": round(sum(r["image_fired"] for r in per_image) / n, 4),
            "A2_fp_area_ratio": round(sum(r["fp_area_px"] for r in per_image)
                                      / sum(r["mpx"] * 1e6 for r in per_image), 6),
            "clean_img_pct": round(100 * sum(1 - r["image_fired"] for r in per_image) / n, 2),
            "fixed_t": args.fixed_threshold,
            "fixed_A3_fp_per_mpx": round(sum(r["fp_components"] for r in fixed)
                                             / sum(r["mpx"] for r in fixed), 3),
            "fixed_A1_image_far": round(sum(r["image_fired"] for r in fixed) / len(fixed), 4),
        }
        for label, values in sensitivity.items():
            row[f"bootstrap_groups_{label}"] = values["n_groups"]
            row[f"A3_ci_{label}"] = values["A3_ci"]
            row[f"phantom_ci_{label}"] = values["phantom_ci"]

        if pos:
            pos_pairs = load_positive_pairs(pos, pred_root, model,
                                            Path(args.gt_root) if args.gt_root else None)
            pos_recall, _, _ = recall_at_threshold(
                pos_pairs, threshold, min_area_px, min_skel_px, dilate_px, cover)
            row["pos_component_recall"] = round(pos_recall, 4)
            positive_axis = evaluate_unified_positive_axis(
                pos_pairs, threshold,
                tolerance_px=int(positive_axis_cfg.get("cliou_tolerance_px", 4)))
            row["positive_axis_n"] = positive_axis["n"]
            row["positive_cldice"] = round(positive_axis["cldice"], 6)
            row["positive_cliou_4px"] = round(positive_axis["cliou_tolerance"], 6)
            row["positive_by_dataset_json"] = json.dumps(
                positive_axis["by_dataset"], ensure_ascii=False, sort_keys=True)
        summary.append(row)

    out = root / "ed1"
    out.mkdir(parents=True, exist_ok=True)
    if threshold_failures:
        with open(out / "threshold_failures.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "status", "reason"])
            writer.writeheader()
            writer.writerows(threshold_failures)
    elif (out / "threshold_failures.csv").exists():
        (out / "threshold_failures.csv").unlink()
    if not summary:
        raise SystemExit("no model can reach t^proxy_R95; see ed1/threshold_failures.csv")
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Rankings on the three accounts + Spearman against this project's unified positive axis
    from scipy.stats import spearmanr

    def ranks(key, reverse=False):
        order = sorted(summary, key=lambda r: r[key], reverse=reverse)
        return {r["model"]: i + 1 for i, r in enumerate(order)}

    rk = {"A1": ranks("A1_image_far"), "A2": ranks("A2_fp_area_ratio"), "A3": ranks("A3_fp_per_mpx")}
    lines = ["# E-D1-public report", "",
             f"- {len(negs)} negatives; primary CI by bootstrap clustered on {primary_bootstrap_key}, "
             f"{bootstrap_iters} iterations; 100 m/1 km sensitivity alongside",
             f"- Primary operating point: t^proxy_R95 calibrated per model on the validation positive layer; secondary report at fixed t={args.fixed_threshold}", "",
             "## Rankings on the three accounts (1 = fewest false alarms)", "",
             "| model | A1 image-level FAR | A2 area ratio | A3 components/Mpx |", "|---|---|---|---|"]
    if threshold_failures:
        lines.insert(3, f"- Models for which t^proxy_R95 is unreachable: {', '.join(r['model'] for r in threshold_failures)}")
    for r in summary:
        m = r["model"]
        lines.append(f"| {m} | {rk['A1'][m]} | {rk['A2'][m]} | {rk['A3'][m]} |")
    mods = [r["model"] for r in summary]
    for a, b in (("A1", "A3"), ("A2", "A3"), ("A1", "A2")):
        rho, p = spearmanr([rk[a][m] for m in mods], [rk[b][m] for m in mods])
        lines.append(f"\nAccount {a} vs {b} ranking Spearman ρ = {rho:.3f} (p={p:.3f})")

    if pos and len(summary) >= 4:
        positive_rank = ranks("positive_cldice", reverse=True)
        rho, rho_low, rho_high = bootstrap_spearman_ci(
            [positive_rank[model] for model in mods], [rk["A3"][model] for model in mods],
            iters=bootstrap_iters)
        lines.append(
            f"\n**Unified positive-axis clDice ranking vs component-level FA ranking Spearman ρ = {rho:.3f} "
            f"(bootstrap 95% CI [{rho_low:.3f},{rho_high:.3f}], n={len(mods)})**"
        )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.info(f"results -> {out/'results.csv'}; report -> {out/'report.md'}")

    if args.plot and pos:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs, ys, names = [], [], []
        for r in summary:
            xs.append(r["positive_cldice"])
            ys.append(r["A3_fp_per_mpx"])
            names.append(r["model"])
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.scatter(xs, ys, s=28)
        for x, y, nm in zip(xs, ys, names):
            ax.annotate(nm, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Unified CrackSC + NHA12D clDice")
        ax.set_ylabel("FP components / Mpx (real negatives)")
        fig.tight_layout()
        fig.savefig(out / "scatter.png")
        LOG.info(f"scatter plot -> {out/'scatter.png'}")


if __name__ == "__main__":
    main()
