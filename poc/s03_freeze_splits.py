# -*- coding: utf-8 -*-
"""Step 1c: firewall split freeze + preregistration snapshot.

Rules:
1) Datasets with an official split keep it (registry.original_split);
2) Datasets that need a local split (local_split: {train: 0.7, test: 0.3, seed: N} in datasets.yaml)
   are split randomly by whole group_id -- a group never spans splits;
3) Apply dedup automatic exclusions and adjudicated exclusions (rows of adjudication_todo.csv
   with a non-empty decision);
4) Any image visually inspected before the freeze is expanded to its whole group via
   development_exposures.yaml, marked development_exposed, and must not enter train/val/test;
5) Check: any group spanning {train,val,test} -> listed in violations, freeze fails;
6) On success write FREEZE.json (registry hash, prereg.yaml hash, seeds, per-pool counts); if it
   already exists, overwriting is refused (only --new-version saves a new version); changes after
   the freeze require a new version, and old versions are never deleted.

Usage:
  python -X utf8 s03_freeze_splits.py --config ../configs/datasets.yaml --prereg ../configs/prereg.yaml --data-root <ROOT>
"""
from __future__ import annotations

import argparse
import csv
import json
import itertools
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from common import (APPLICATION_ROLES, EVAL_ROLES, PROXY_ROLES, QUARANTINE_ROLES,
                    ROLE_WHITELIST, TRAIN_ROLES, read_jsonl, setup_logger, sha256_file)
from anchor_acceptance import validate_anchor_acceptance

LOG = setup_logger("s03")

SPLIT_CANON = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test", "all": "all"}
DEDUP_REQUIRED_FILES = (
    "phash.jsonl", "pairs_phash.csv", "pair_summary.csv", "exclusions_auto.csv",
    "pairs_group_protected.csv", "adjudication_todo.csv", "report.md", "run_metadata.json",
)


def validate_roles(config: dict) -> None:
    unknown = {
        dataset_id: dataset.get("role")
        for dataset_id, dataset in config["datasets"].items()
        if dataset.get("role") not in ROLE_WHITELIST
    }
    if unknown:
        raise SystemExit(f"datasets.yaml contains unknown roles: {unknown}")


def validate_proxy_anchor_ready(config: dict, prereg: dict, rows: list[dict], data_root=None) -> list[Path]:
    anchor = prereg.get("proxy_anchor")
    if not anchor:
        return []
    dataset_id = anchor["dataset"]
    dataset = config["datasets"].get(dataset_id, {})
    if dataset.get("status") != "annotation_accepted":
        raise SystemExit(f"{dataset_id}: R7 calibration anchor not yet formally accepted; freezing is not allowed")
    selected = [row for row in rows if row["source_dataset"] == dataset_id]
    # prereg-3 (R16): registered anchors may be fewer than sample_size only by declared geometry_unresolved
    # entries; anchor_acceptance checks that every sampled ID is covered exactly once and the cap holds.
    if not 0 < len(selected) <= int(anchor["sample_size"]):
        raise SystemExit(f"{dataset_id}: anchor count {len(selected)} outside (0, {anchor['sample_size']}]")
    for row in selected:
        if (row.get("pool") != "proxy_bench" or row.get("original_split") != "val"
                or row.get("crack_annotation") != "sparse_centerline"
                or not row.get("mask_path") or not Path(row["mask_path"]).is_file()):
            raise SystemExit(f"{dataset_id}: anchors must be proxy val images with a centerline mask: {row['image_id']}")
    try:
        return validate_anchor_acceptance(dataset_id, dataset, anchor, selected, data_root)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        raise SystemExit(f"{dataset_id}: R7 geometric acceptance evidence incomplete: {exc}") from exc


def split_for_role(role: str, original_split: str) -> str:
    canonical = SPLIT_CANON.get(str(original_split).lower(), str(original_split).lower())
    if role in TRAIN_ROLES:
        return canonical
    if role in PROXY_ROLES:
        return {
            "train": "proxy_train", "val": "proxy_val", "test": "proxy_test",
            "all": "proxy_unassigned",
        }.get(canonical, f"proxy_{canonical}")
    if role in EVAL_ROLES:
        return "test"
    if role in QUARANTINE_ROLES:
        return "quarantine"
    if role in APPLICATION_ROLES:
        return "unavailable"
    raise SystemExit(f"unknown role={role!r}")


def leakage_partition(split: str) -> str | None:
    return {
        "train": "train", "proxy_train": "train",
        "val": "val", "proxy_val": "val",
        "test": "test", "proxy_test": "test",
    }.get(split)


def validate_role_splits(assignments: list[dict]) -> None:
    invalid = [
        row for row in assignments
        if row["pool"] in (EVAL_ROLES | PROXY_ROLES | QUARANTINE_ROLES | APPLICATION_ROLES)
        and row["split"] in {"train", "val"}
    ]
    if invalid:
        row = invalid[0]
        raise SystemExit(
            f"non-training role assigned to train/val: {row['image_id']} pool={row['pool']} split={row['split']}"
        )


def candidate_pairs(dedup_dir: Path):
    seen = set()
    for filename in ("pairs_phash.csv", "pairs_dino.csv"):
        path = dedup_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = tuple(sorted((row["a"], row["b"])))
                if key not in seen:
                    seen.add(key)
                    yield key


def validate_candidate_firewall(dedup_dir: Path, assignments: list[dict]) -> None:
    split_by_image = {row["image_id"]: row["split"] for row in assignments}
    violations = []
    for a, b in candidate_pairs(dedup_dir):
        if a not in split_by_image or b not in split_by_image:
            raise SystemExit(f"candidate pair references unregistered image: {a} <-> {b}")
        pa = leakage_partition(split_by_image[a])
        pb = leakage_partition(split_by_image[b])
        if {pa, pb} & {"train", "val"} and "test" in {pa, pb}:
            violations.append((a, split_by_image[a], b, split_by_image[b]))
    if violations:
        a, sa, b, sb = violations[0]
        raise SystemExit(
            f"near-duplicate candidates cross the development/test firewall: {len(violations)} pairs; e.g. {a}({sa}) <-> {b}({sb})"
        )


def load_exclusions(dedup_dir: Path) -> set[str]:
    drops = set()
    auto = dedup_dir / "exclusions_auto.csv"
    if auto.exists():
        with open(auto, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                drops.add(row["drop_image_id"])
    adj = dedup_dir / "adjudication_todo.csv"
    if adj.exists():
        with open(adj, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                d = (row.get("decision(keep/drop_a/drop_b)") or "").strip().lower()
                if d == "drop_a":
                    drops.add(row["a"])
                elif d == "drop_b":
                    drops.add(row["b"])
    return drops


def validate_adjudications(dedup_dir: Path) -> None:
    """Before freezing, every near-duplicate candidate must have a valid adjudication."""
    adj = dedup_dir / "adjudication_todo.csv"
    if not adj.exists():
        raise SystemExit(f"missing {adj}; run the s02 dedup scan first")
    unresolved, invalid = [], []
    with open(adj, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            decision = (row.get("decision(keep/drop_a/drop_b)") or "").strip().lower()
            pair = f"{row.get('a')} <-> {row.get('b')}"
            if not decision:
                unresolved.append(pair)
            elif decision not in {"keep", "drop_a", "drop_b"}:
                invalid.append(f"{pair}: {decision}")
    if unresolved or invalid:
        msg = ["Dedup adjudication incomplete; refusing to freeze:"]
        if unresolved:
            msg.append(f"  empty decision: {len(unresolved)} (e.g. {unresolved[0]})")
        if invalid:
            msg.append(f"  invalid decision: {len(invalid)} (e.g. {invalid[0]})")
        raise SystemExit("\n".join(msg))


def validate_registry(root: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("Registry is empty; refusing to freeze")
    missing_hash = [r["image_id"] for r in rows if not r.get("sha256")]
    if missing_hash:
        raise SystemExit(
            f"Registry still has {len(missing_hash)} images without sha256 (e.g. {missing_hash[0]}); "
            "rerun s01 without --no-hash"
        )
    problems = root / "registry" / "problems.txt"
    if problems.exists() and problems.read_text(encoding="utf-8").strip():
        raise SystemExit(f"{problems} is not empty; resolve all listed problems first")


def load_development_exposure_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    exposures = config.get("exposures", [])
    ids = set()
    for item in exposures:
        image_id = item if isinstance(item, str) else item.get("image_id")
        if not image_id:
            raise SystemExit(f"{path} contains an exposure entry without image_id")
        ids.add(str(image_id))
    return ids


def row_numeric_value(row: dict) -> float:
    match = re.match(r"^\d+(?:\.\d+)?", str(row["original_image_id"]))
    if not match:
        raise SystemExit(f"original_image_id of {row['image_id']} has no numeric prefix")
    return float(match.group(0))


def resolve_exposure_groups(rows: list[dict], exposure_ids: set[str],
                            numeric_radii: dict[str, float] | None = None) -> set[str]:
    image_to_group = {row["image_id"]: row["group_id"] for row in rows}
    image_to_row = {row["image_id"]: row for row in rows}
    unknown = sorted(exposure_ids - image_to_group.keys())
    if unknown:
        raise SystemExit(f"development exposure list contains {len(unknown)} unregistered image_id(s) (e.g. {unknown[0]})")
    groups = {image_to_group[image_id] for image_id in exposure_ids}
    numeric_radii = numeric_radii or {}
    numeric_targets = defaultdict(list)
    for image_id in exposure_ids:
        row = image_to_row[image_id]
        radius = float(numeric_radii.get(row["source_dataset"], 0.0))
        if radius > 0:
            numeric_targets[row["source_dataset"]].append((row_numeric_value(row), radius))
    for row in rows:
        targets = numeric_targets.get(row["source_dataset"])
        if not targets:
            continue
        value = row_numeric_value(row)
        if any(abs(value - target) <= radius for target, radius in targets):
            groups.add(row["group_id"])
    return groups


def blocked_numeric_assignments(rows: list[dict], dataset_id: str, settings: dict,
                                excluded_groups: set[str]) -> dict[str, str]:
    group_ranges = {}
    for row in rows:
        if row["source_dataset"] != dataset_id or row["group_id"] in excluded_groups:
            continue
        value = row_numeric_value(row)
        if row["group_id"] not in group_ranges:
            group_ranges[row["group_id"]] = [value, value]
        else:
            group_ranges[row["group_id"]][0] = min(group_ranges[row["group_id"]][0], value)
            group_ranges[row["group_id"]][1] = max(group_ranges[row["group_id"]][1], value)
    ordered = sorted(
        ((group_id, bounds[0], bounds[1]) for group_id, bounds in group_ranges.items()),
        key=lambda item: (item[1] + item[2]) / 2.0)
    if not ordered:
        return {}

    sequence_gap = float(settings.get("sequence_gap", 1.0))
    sequences, current = [], []
    for item in ordered:
        if current and item[1] - current[-1][2] > sequence_gap:
            sequences.append(current)
            current = []
        current.append(item)
    sequences.append(current)

    train_frac = float(settings["train"])
    test_frac = float(settings["test"])
    total = train_frac + test_frac
    buffer = float(settings.get("buffer", 0.0))
    assignments = {}
    for sequence in sequences:
        if len(sequence) == 1:
            assignments[sequence[0][0]] = "train"
            continue
        n_train = max(1, min(len(sequence) - 1, int(round(len(sequence) * train_frac / total))))
        cut = (sequence[n_train - 1][2] + sequence[n_train][1]) / 2.0
        sequence_assignments = {}
        for group_id, minimum, maximum in sequence:
            if maximum < cut - buffer:
                sequence_assignments[group_id] = "train"
            elif minimum > cut + buffer:
                sequence_assignments[group_id] = "test"
            else:
                sequence_assignments[group_id] = "development_buffer"
        sequence_splits = set(sequence_assignments.values())
        if "train" not in sequence_splits or "test" not in sequence_splits:
            sequence_assignments = {group_id: "train" for group_id, _, _ in sequence}
        assignments.update(sequence_assignments)
    return assignments


def interleaved_numeric_plan(rows: list[dict], dataset_id: str, settings: dict,
                             excluded_groups: set[str]) -> tuple[dict[str, str], list[dict]]:
    """Build 4-6 interleaved blocks of about 2 km per continuous chainage sequence, with a buffer on both sides of each block boundary."""
    group_ranges = {}
    group_sizes = Counter()
    for row in rows:
        if row["source_dataset"] != dataset_id or row["group_id"] in excluded_groups:
            continue
        value = row_numeric_value(row)
        group_sizes[row["group_id"]] += 1
        bounds = group_ranges.setdefault(row["group_id"], [value, value])
        bounds[0] = min(bounds[0], value)
        bounds[1] = max(bounds[1], value)
    ordered = sorted(
        ((group_id, bounds[0], bounds[1]) for group_id, bounds in group_ranges.items()),
        key=lambda item: (item[1] + item[2]) / 2.0)
    if not ordered:
        return {}, []

    sequence_gap = float(settings.get("sequence_gap", 1.0))
    sequences, current = [], []
    for item in ordered:
        if current and item[1] - current[-1][2] > sequence_gap:
            sequences.append(current)
            current = []
        current.append(item)
    sequences.append(current)

    block_width = float(settings.get("block_width", 2.0))
    buffer = float(settings.get("buffer", 0.1))
    min_blocks = int(settings.get("min_blocks", 4))
    max_blocks = int(settings.get("max_blocks", 6))
    train_frac = float(settings["train"])
    test_frac = float(settings["test"])
    seed = int(settings.get("seed", 20260831))
    assignments, plan = {}, []

    for sequence_index, sequence in enumerate(sequences):
        sequence_start = min(item[1] for item in sequence)
        sequence_stop = max(item[2] for item in sequence)
        span = max(0.0, sequence_stop - sequence_start)
        natural_blocks = max(1, int(np.ceil(span / block_width)))
        if natural_blocks < min_blocks:
            for group_id, _, _ in sequence:
                assignments[group_id] = "train"
            plan.append({
                "sequence": sequence_index,
                "block": 0,
                "start": sequence_start,
                "stop": sequence_stop,
                "split": "train",
                "groups": len(sequence),
                "images": sum(group_sizes[item[0]] for item in sequence),
                "reason": "short_sequence_train_only",
            })
            continue

        n_blocks = min(natural_blocks, max_blocks)
        width = max(block_width, span / n_blocks)
        n_test = max(1, min(n_blocks - 1,
                            int(round(n_blocks * test_frac / (train_frac + test_frac)))))
        combinations = list(itertools.combinations(range(n_blocks), n_test))
        max_transitions = max(
            sum((index in combo) != (index - 1 in combo) for index in range(1, n_blocks))
            for combo in combinations)
        candidates = [
            combo for combo in combinations
            if sum((index in combo) != (index - 1 in combo)
                   for index in range(1, n_blocks)) == max_transitions
        ]
        rng = random.Random(seed + sequence_index)
        test_blocks = set(candidates[rng.randrange(len(candidates))])
        block_splits = ["test" if index in test_blocks else "train"
                        for index in range(n_blocks)]
        boundaries = [sequence_start + width * index for index in range(1, n_blocks)]
        block_groups = [set() for _ in range(n_blocks)]
        block_images = [0 for _ in range(n_blocks)]
        for group_id, minimum, maximum in sequence:
            if any(minimum <= boundary + buffer and maximum >= boundary - buffer
                   for boundary in boundaries):
                assignments[group_id] = "development_buffer"
                continue
            center = (minimum + maximum) / 2.0
            block_index = min(n_blocks - 1, max(0, int((center - sequence_start) // width)))
            assignments[group_id] = block_splits[block_index]
            block_groups[block_index].add(group_id)
            block_images[block_index] += group_sizes[group_id]
        for block_index in range(n_blocks):
            plan.append({
                "sequence": sequence_index,
                "block": block_index,
                "start": sequence_start + block_index * width,
                "stop": min(sequence_stop, sequence_start + (block_index + 1) * width),
                "split": block_splits[block_index],
                "groups": len(block_groups[block_index]),
                "images": block_images[block_index],
                "buffer_km": buffer,
                "reason": "interleaved_seeded_block",
            })
    return assignments, plan


def versioned_targets(out_dir: Path, new_version: bool) -> tuple[Path, Path]:
    freeze = out_dir / "FREEZE.json"
    assignments = out_dir / "split_assignments.csv"
    if not freeze.exists():
        return freeze, assignments
    if not new_version:
        raise SystemExit(f"freeze already exists: {freeze}; changing splits requires --new-version, overwriting is forbidden")
    ver = 2
    while (out_dir / f"FREEZE.v{ver}.json").exists():
        ver += 1
    return out_dir / f"FREEZE.v{ver}.json", out_dir / f"split_assignments.v{ver}.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--exposures", default="", help="Defaults to development_exposures.yaml beside --config")
    ap.add_argument("--auto-exposures", default="",
                    help="Defaults to <data-root>/development_exposures.auto.yaml when present")
    ap.add_argument("--committee", default="", help="Defaults to committee.yaml beside --config")
    ap.add_argument("--models-config", default="",
                    help="Defaults to models_public.yaml beside --config when present")
    ap.add_argument("--new-version", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    config_path = Path(args.config).resolve()
    prereg_path = Path(args.prereg).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    validate_roles(cfg)
    reg_path = root / "registry" / "registry.jsonl"
    rows = list(read_jsonl(reg_path))
    anchor_review_artifacts = validate_proxy_anchor_ready(cfg, prereg, rows, root)
    validate_registry(root, rows)
    validate_adjudications(root / "dedup")
    drops = load_exclusions(root / "dedup")
    exposures_path = (Path(args.exposures) if args.exposures else
                      config_path.with_name("development_exposures.yaml"))
    auto_exposures_path = (Path(args.auto_exposures) if args.auto_exposures else
                           root / "development_exposures.auto.yaml")
    committee_path = (Path(args.committee).resolve() if args.committee else
                      config_path.with_name("committee.yaml"))
    if not committee_path.exists():
        raise SystemExit(f"missing committee config {committee_path}")
    models_config_path = (Path(args.models_config).resolve() if args.models_config else
                          config_path.with_name("models_public.yaml"))
    if args.models_config and not models_config_path.exists():
        raise SystemExit(f"missing models config {models_config_path}")
    exposure_ids = load_development_exposure_ids(exposures_path)
    auto_exposure_ids = load_development_exposure_ids(auto_exposures_path)
    exposure_ids |= auto_exposure_ids
    numeric_radii = {
        ds_id: float(ds.get("development_exposure_numeric_radius", 0.0))
        for ds_id, ds in cfg["datasets"].items()
    }
    exposure_groups = resolve_exposure_groups(rows, exposure_ids, numeric_radii)
    LOG.info(f"registry {len(rows)} rows; excluded {len(drops)} images (dedup automatic + adjudicated); "
             f"development-exposed {len(exposure_ids)} images / {len(exposure_groups)} groups")

    out_dir = root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    target, assignments_path = versioned_targets(out_dir, args.new_version)

    # Local splits (group level)
    local_assign: dict[str, str] = {}
    split_plans = {}
    for ds_id, ds in cfg["datasets"].items():
        ls = ds.get("local_split")
        if not ls:
            continue
        if ds.get("role") not in TRAIN_ROLES:
            raise SystemExit(f"{ds_id}: non-training role={ds.get('role')} must not configure local_split")
        if ds.get("local_split_mode") == "blocked_numeric":
            settings = {
                **ls,
                "sequence_gap": ds.get("local_split_sequence_gap", 1.0),
                "buffer": ds.get("local_split_buffer", 0.0),
            }
            blocked = blocked_numeric_assignments(rows, ds_id, settings, exposure_groups)
            local_assign.update(blocked)
            split_counts = Counter(blocked.values())
            LOG.info(f"{ds_id}: blocked chainage split of {len(blocked)} groups -> {dict(split_counts)}")
            continue
        if ds.get("local_split_mode") == "interleaved_numeric":
            settings = {
                **ls,
                "sequence_gap": ds.get("local_split_sequence_gap", 1.0),
                "buffer": ds.get("local_split_buffer", 0.1),
                "block_width": ds.get("local_split_block_width", 2.0),
                "min_blocks": ds.get("local_split_min_blocks", 4),
                "max_blocks": ds.get("local_split_max_blocks", 6),
            }
            interleaved, plan = interleaved_numeric_plan(
                rows, ds_id, settings, exposure_groups)
            local_assign.update(interleaved)
            split_plans[ds_id] = plan
            split_counts = Counter(interleaved.values())
            LOG.info(f"{ds_id}: interleaved chainage-block split of {len(interleaved)} groups -> {dict(split_counts)}")
            continue
        groups = sorted({r["group_id"] for r in rows
                         if r["source_dataset"] == ds_id and r["image_id"] not in drops
                         and r["group_id"] not in exposure_groups})
        rng = random.Random(ls.get("seed", 20260831))
        rng.shuffle(groups)
        fracs = [(k, v) for k, v in ls.items() if k in ("train", "val", "test")]
        total = sum(v for _, v in fracs)
        start = 0
        for i, (name, frac) in enumerate(fracs):
            n = len(groups) - start if i == len(fracs) - 1 else int(round(len(groups) * frac / total))
            for g in groups[start:start + n]:
                local_assign[g] = name
            start += n
        LOG.info(f"{ds_id}: local group split of {len(groups)} groups -> " +
                 ", ".join(f"{k}:{v}" for k, v in ls.items() if k != 'seed'))

    assignments, group_splits = [], defaultdict(set)
    for r in rows:
        if r["image_id"] in drops:
            assignments.append({**{k: r[k] for k in ("image_id", "source_dataset", "pool", "group_id")},
                                "split": "excluded"})
            continue
        if r["group_id"] in exposure_groups:
            split = "development_exposed"
        elif r["group_id"] in local_assign:
            split = local_assign[r["group_id"]]
        else:
            split = split_for_role(r["pool"], r["original_split"])
        bootstrap_100m = r["group_id"]
        bootstrap_1km = r["group_id"]
        if r["source_dataset"] == "pavedistress":
            value = row_numeric_value(r)
            bootstrap_1km = f"pavedistress::distance_1km_{int(value // 1.0):06d}"
        assignments.append({**{k: r[k] for k in ("image_id", "source_dataset", "pool", "group_id")},
                            "split": split, "bootstrap_group_100m": bootstrap_100m,
                            "bootstrap_group_1km": bootstrap_1km})
        partition = leakage_partition(split)
        if partition:
            group_splits[r["group_id"]].add(partition)

    violations = [(g, sorted(s)) for g, s in group_splits.items() if len(s) > 1]
    with open(assignments_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_id", "source_dataset", "pool", "group_id", "split",
                  "bootstrap_group_100m", "bootstrap_group_1km"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(assignments)

    if violations:
        vp = out_dir / "violations.txt"
        vp.write_text("\n".join(f"{g}: {s}" for g, s in violations), encoding="utf-8")
        raise SystemExit(f"firewall violation: {len(violations)} groups span splits ({vp}). Not frozen.")

    validate_role_splits(assignments)
    validate_candidate_firewall(root / "dedup", assignments)

    counts = defaultdict(int)
    for a in assignments:
        counts[f"{a['pool']}/{a['split']}"] += 1
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        path.relative_to(code_root).as_posix(): sha256_file(path)
        for path in sorted((code_root / "poc").rglob("*.py"))
    }
    dedup_dir = root / "dedup"
    missing_dedup = [name for name in DEDUP_REQUIRED_FILES if not (dedup_dir / name).exists()]
    if missing_dedup:
        raise SystemExit(f"dedup artifacts required for freezing are missing: {missing_dedup}")
    artifact_paths = [reg_path, assignments_path]
    artifact_paths.extend(anchor_review_artifacts)
    # Centerline annotations directly determine calibration; bind their actual masks.
    artifact_paths.extend(Path(row["mask_path"]) for row in rows
                          if row.get("crack_annotation") == "sparse_centerline"
                          and row.get("mask_path"))
    for dataset_id, dataset in cfg["datasets"].items():
        group_map_csv = dataset.get("group_map_csv")
        if group_map_csv:
            group_map_path = root / group_map_csv
            if not group_map_path.exists():
                raise SystemExit(f"{dataset_id}: group_map_csv required for freezing is missing: {group_map_path}")
            artifact_paths.append(group_map_path)
        for field in ("manifest", "double_annotation_manifest", "annotation_protocol"):
            declared = dataset.get(field)
            if not declared:
                continue
            declared_path = root / declared
            if not declared_path.exists():
                raise SystemExit(f"{dataset_id}: {field} required for freezing is missing: {declared_path}")
            artifact_paths.append(declared_path)
        negative_manifest = (dataset.get("negative_leg") or {}).get("manifest")
        if negative_manifest:
            negative_path = root / negative_manifest
            if not negative_path.exists():
                raise SystemExit(f"{dataset_id}: negative_leg manifest required for freezing is missing: {negative_path}")
            artifact_paths.append(negative_path)
    for dataset_id, plan in split_plans.items():
        plan_path = out_dir / f"{assignments_path.stem}.{dataset_id}_block_plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact_paths.append(plan_path)
    artifact_paths.extend(dedup_dir / name for name in DEDUP_REQUIRED_FILES)
    optional_dino = dedup_dir / "pairs_dino.csv"
    if optional_dino.exists():
        artifact_paths.append(optional_dino)
    artifacts_sha256 = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in artifact_paths
    }
    config_paths = [config_path, prereg_path, committee_path]
    if models_config_path.exists():
        config_paths.append(models_config_path.resolve())
    if exposures_path.exists():
        config_paths.append(exposures_path.resolve())
    if auto_exposures_path.exists():
        config_paths.append(auto_exposures_path.resolve())
    config_sha256 = {str(path.resolve()): sha256_file(path) for path in config_paths}
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": sha256_file(reg_path),
        "prereg_sha256": sha256_file(prereg_path),
        "prereg_file": prereg_path.name,
        "datasets_config_sha256": sha256_file(config_path),
        "datasets_config_file": config_path.name,
        "development_exposures_sha256": sha256_file(exposures_path) if exposures_path.exists() else None,
        "development_exposures_file": exposures_path.name if exposures_path.exists() else None,
        "development_exposures_auto_sha256": (
            sha256_file(auto_exposures_path) if auto_exposures_path.exists() else None),
        "development_exposures_auto_file": (
            auto_exposures_path.name if auto_exposures_path.exists() else None),
        "n_auto_development_exposure_images": len(auto_exposure_ids),
        "committee_sha256": sha256_file(committee_path),
        "committee_file": committee_path.name,
        "models_config_sha256": (
            sha256_file(models_config_path) if models_config_path.exists() else None),
        "models_config_file": models_config_path.name if models_config_path.exists() else None,
        "n_development_exposure_images": len(exposure_ids),
        "n_development_exposure_groups": len(exposure_groups),
        "code_sha256": code_hashes,
        "artifacts_sha256": artifacts_sha256,
        "config_sha256": config_sha256,
        "split_assignments_file": assignments_path.name,
        "split_assignments_sha256": sha256_file(assignments_path),
        "n_images": len(rows),
        "n_excluded": len(drops),
        "counts": dict(sorted(counts.items())),
    }
    target.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info(f"freeze complete -> {target}")
    for k, v in freeze["counts"].items():
        LOG.info(f"  {k}: {v}")
    LOG.info("From now on, no training or tuning may read split=test images; criteria follow the frozen prereg.yaml.")


if __name__ == "__main__":
    main()
