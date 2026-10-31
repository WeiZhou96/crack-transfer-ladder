# -*- coding: utf-8 -*-
"""Step 1a: build the data registry.

One record per image:
  image_id, source_dataset, original_split, original_image_id, pool(role),
  material, sensor, gsd_mm_per_pixel, crack_annotation, license,
  path, mask_path, width, height, sha256, group_id

layout contract (per dataset in datasets.yaml):
  layout:
    root: raw/<id>/...            # relative to data_root
    entries:
      - split: train              # official split name; use "all" if there is no official split
        images_glob: "images/*.jpg"
        mask_map: {from: "images/", to: "masks/", ext: ".png"}   # optional (negatives / image-level labels)
  group_rule: "^(?P<g>...)$"      # regex on the file stem; named group g becomes group_id; default = stem

Checks: mask existence and size consistency (mismatches are only reported, files are not modified).
Usage:
  python -X utf8 s01_build_registry.py --config ../configs/datasets.yaml --data-root <ROOT> [--only ...] [--no-hash]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from common import ROLE_WHITELIST, image_size, setup_logger, sha256_file, write_jsonl

LOG = setup_logger("s01")


def derive_group_local(stem: str, group_re, dataset: dict) -> str:
    if group_re:
        match = group_re.match(stem)
        if not match or "g" not in match.groupdict() or match.group("g") is None:
            raise ValueError(f"group_rule does not match stem={stem!r} with named group 'g'")
        group = match.group("g")
    else:
        group = stem
    bucket = dataset.get("group_numeric_bucket")
    if bucket is not None:
        try:
            index = int(float(group) // float(bucket))
        except ValueError as exc:
            raise ValueError(f"group {group!r} is not numeric for bucket={bucket}") from exc
        label = str(dataset.get("group_numeric_bucket_label", "numeric_bucket"))
        return f"{label}_{index:08d}"
    return group


def load_group_map(root: Path, dataset_id: str, dataset: dict) -> dict[str, str] | None:
    declared = dataset.get("group_map_csv")
    if not declared:
        return None
    path = root / declared
    if not path.exists():
        raise SystemExit(f"{dataset_id}: missing group_map_csv -> {path}")
    mapping = {}
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_id = (row.get("image_id") or "").strip()
            group_local = (row.get("group_local") or "").strip()
            if not image_id or not group_local:
                raise SystemExit(f"{dataset_id}: {path} contains an empty image_id/group_local")
            previous = mapping.setdefault(image_id, group_local)
            if previous != group_local:
                raise SystemExit(
                    f"{dataset_id}: {image_id} maps to multiple groups in {path}: {previous}, {group_local}"
                )
    if not mapping:
        raise SystemExit(f"{dataset_id}: group_map_csv is empty -> {path}")
    return mapping


def map_mask(img: Path, mask_map: dict, ds_root: Path) -> Path | None:
    rel = img.relative_to(ds_root).as_posix()
    if mask_map.get("from") is not None:
        rel = rel.replace(mask_map["from"], mask_map["to"], 1)
    p = ds_root / rel
    ext = mask_map.get("ext")
    if ext:
        p = p.with_suffix(ext)
    suffix = mask_map.get("stem_suffix")
    if suffix:
        p = p.with_name(p.stem + suffix + p.suffix)
    return p if p.exists() else None


def expected_registry_actual(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "masks": sum(row.get("mask_path") is not None for row in rows),
        "splits": dict(sorted(Counter(row["original_split"] for row in rows).items())),
        "semantic_labels": dict(sorted(Counter(
            label for row in rows for label in row.get("semantic_labels", [])).items())),
    }


def validate_expected_registry(dataset_id: str, dataset: dict, rows: list[dict]) -> dict:
    expected = dataset.get("expected_registry")
    actual = expected_registry_actual(rows)
    if not expected:
        return actual
    failures = []
    if "total" in expected and len(rows) != int(expected["total"]):
        failures.append(f"total={len(rows)} != {expected['total']}")
    mask_count = sum(row.get("mask_path") is not None for row in rows)
    if "masks" in expected and mask_count != int(expected["masks"]):
        failures.append(f"masks={mask_count} != {expected['masks']}")
    observed_splits = Counter(row["original_split"] for row in rows)
    for split, count in (expected.get("splits") or {}).items():
        if observed_splits[split] != int(count):
            failures.append(f"split[{split}]={observed_splits[split]} != {count}")
    observed_labels = Counter(
        label for row in rows for label in row.get("semantic_labels", []))
    for label, count in (expected.get("semantic_labels") or {}).items():
        if observed_labels[label] != int(count):
            failures.append(f"semantic_labels[{label}]={observed_labels[label]} != {count}")
    if failures:
        raise SystemExit(f"{dataset_id}: expected_registry count gate failed: " + "; ".join(failures))
    return actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-hash", action="store_true", help="Skip sha256 (for quick inventory; a full hashing run is required before freezing)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.data_root)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    rows, problems, expected_audit = [], [], {}
    for ds_id, ds in cfg["datasets"].items():
        dataset_row_start = len(rows)
        role = ds.get("role")
        if role not in ROLE_WHITELIST:
            raise SystemExit(f"{ds_id}: unknown role={role!r}; allowed values={sorted(ROLE_WHITELIST)}")
        if only and ds_id not in only:
            continue
        layout = ds.get("layout")
        if not layout:
            LOG.warning(f"{ds_id}: no layout (not downloaded or not configured), skipping")
            continue
        ds_root = root / layout["root"]
        if not ds_root.exists():
            LOG.warning(f"{ds_id}: directory does not exist {ds_root}, skipping")
            continue
        grp_re = re.compile(ds["group_rule"]) if ds.get("group_rule") else None
        group_map = load_group_map(root, ds_id, ds)
        seen_group_map_ids = set()
        rows_by_stem = {}

        for entry in layout["entries"]:
            imgs = sorted(ds_root.glob(entry["images_glob"]))
            if not imgs:
                problems.append(f"{ds_id}/{entry.get('split')}: images_glob matched 0 images -> {entry['images_glob']}")
                continue
            for img in tqdm(imgs, desc=f"{ds_id}:{entry.get('split', 'all')}", unit="img"):
                stem = img.stem
                semantic_label = entry.get("label")
                if ds.get("deduplicate_by_stem") and stem in rows_by_stem:
                    existing = rows_by_stem[stem]
                    if semantic_label and semantic_label not in existing["semantic_labels"]:
                        existing["semantic_labels"].append(semantic_label)
                        existing["semantic_labels"].sort()
                    existing["source_paths"].append(str(img))
                    continue
                image_id = f"{ds_id}/{img.relative_to(ds_root).as_posix()}"
                if group_map is not None:
                    group_local = group_map.get(image_id)
                    if group_local is None:
                        problems.append(f"{ds_id}: group_map_csv lacks image_id -> {image_id}")
                        group_local = f"unmapped_{stem}"
                    else:
                        seen_group_map_ids.add(image_id)
                else:
                    try:
                        group_local = derive_group_local(stem, grp_re, ds)
                    except ValueError as error:
                        problems.append(f"{ds_id}: group_rule does not match {img.name}: {error}")
                        group_local = stem
                mask_path = None
                if entry.get("mask_map"):
                    mask_path = map_mask(img, entry["mask_map"], ds_root)
                    if mask_path is None:
                        problems.append(f"{ds_id}: missing mask -> {img.name}")
                w, h = image_size(img)
                if mask_path is not None:
                    mw, mh = image_size(mask_path)
                    if (mw, mh) != (w, h):
                        problems.append(f"{ds_id}: size mismatch {img.name} img={w}x{h} mask={mw}x{mh}")
                row = {
                    "image_id": image_id,
                    "source_dataset": ds_id,
                    "original_split": entry.get("split", "all"),
                    "original_image_id": stem,
                    "pool": ds.get("role", "unassigned"),
                    "material": ds.get("material", "unknown"),
                    "sensor": ds.get("sensor", "unknown"),
                    "gsd_mm_per_pixel": ds.get("gsd_mm_per_pixel"),
                    "crack_annotation": ds.get("crack_annotation", "none"),
                    "license": ds.get("license", "unstated"),
                    "path": str(img),
                    "mask_path": str(mask_path) if mask_path else None,
                    "width": w, "height": h,
                    "sha256": None if args.no_hash else sha256_file(img),
                    "group_id": f"{ds_id}::{group_local}",
                    "semantic_labels": [semantic_label] if semantic_label else [],
                    "source_paths": [str(img)],
                }
                rows.append(row)
                if ds.get("deduplicate_by_stem"):
                    rows_by_stem[stem] = row
        if group_map is not None:
            unused = sorted(set(group_map) - seen_group_map_ids)
            if unused:
                problems.append(
                    f"{ds_id}: group_map_csv contains {len(unused)} image_id(s) not matched by the current layout"
                    f" (e.g. {unused[0]})"
                )
        if ds.get("expected_registry"):
            actual = validate_expected_registry(ds_id, ds, rows[dataset_row_start:])
            expected_audit[ds_id] = {
                "status": "pass", "expected": ds["expected_registry"], "actual": actual,
            }

    out_dir = root / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "registry.jsonl", rows)
    with open(out_dir / "registry.csv", "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    (out_dir / "problems.txt").write_text("\n".join(problems), encoding="utf-8")
    (out_dir / "expected_audit.json").write_text(
        json.dumps(expected_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info(f"registry: {len(rows)} rows -> {out_dir/'registry.jsonl'}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["source_dataset"], [0, 0])
        by_ds[r["source_dataset"]][0] += 1
        by_ds[r["source_dataset"]][1] += int(r["mask_path"] is not None)
    for k, (n, nm) in sorted(by_ds.items()):
        LOG.info(f"  {k}: {n} images ({nm} with mask)")
    if problems:
        LOG.warning(f"{len(problems)} problems -> {out_dir/'problems.txt'} (must be cleared or individually waived before freezing)")


if __name__ == "__main__":
    main()
