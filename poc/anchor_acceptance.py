"""Validate hash-bound declared anchor reviews, not the truth of human judgments."""
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from common import image_size, load_binary_mask, sha256_file
from tools.validate_pavedistress_submissions import centerline_mask, parse_annotation


def validate_anchor_acceptance(dataset_id, dataset, anchor, rows, data_root):
    if data_root is None:
        raise ValueError("R7 acceptance evidence needs the explicit data root")
    root = Path(data_root).resolve()
    artifacts = []

    def bound_file(value, expected_sha):
        if not isinstance(value, str) or not value:
            raise ValueError("R7 review evidence lacks a file path")
        raw = Path(value)
        path = (raw if raw.is_absolute() else root / raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("R7 evidence path is missing or outside the data root")
        if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            raise ValueError("R7 evidence hash mismatch")
        artifacts.append(path)
        return path

    declaration = dataset.get("acceptance_evidence") or {}
    receipt_path = bound_file(declaration.get("path"), declaration.get("sha256"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != 1 or receipt.get("dataset") != dataset_id
            or receipt.get("status") != "accepted" or receipt.get("geometry_review_complete") is not True
            or receipt.get("scope") != "full_anchor_native_geometry"
            or receipt.get("uncertainty_policy") != "visible_segments_only_unlabeled_not_negative"):
        raise ValueError("R7 requires explicit full-anchor native-geometry acceptance, not staging")
    reviewer = receipt.get("reviewer") or {}
    if (not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip()
            or reviewer.get("kind") not in {"human", "human_ai_assisted", "ai_authorized"}):
        raise ValueError("R7 acceptance must disclose reviewer identity and kind")
    reviewed_at = datetime.fromisoformat(receipt.get("reviewed_at", ""))
    if reviewed_at.tzinfo is None:
        raise ValueError("R7 review timestamp needs a timezone")
    entries = receipt.get("images") or []
    indexed = {entry["image_id"]: entry for entry in entries}
    # prereg-3 (R16): a declared, capped number of anchors may be reviewed as geometry_unresolved. They are not
    # registered, contribute no positives, and are never read as no_visible_crack; every sampled ID stays accounted for.
    unresolved_policy = anchor.get("geometry_unresolved") or {}
    max_unresolved = int(unresolved_policy.get("max_images", 0)) if unresolved_policy.get("allowed") is True else 0
    unresolved = [entry for entry in entries if entry.get("decision") == "geometry_unresolved"]
    row_ids = {row["image_id"] for row in rows}
    accepted_ids = set(indexed) - {entry["image_id"] for entry in unresolved}
    if (len(indexed) != len(entries) or len(entries) != int(anchor["sample_size"])
            or len(accepted_ids) != len(rows) or accepted_ids != row_ids):
        raise ValueError("R7 evidence must cover each registered anchor ID exactly once")
    if len(unresolved) > max_unresolved:
        raise ValueError("R7 geometry_unresolved anchors exceed the preregistered cap")
    for entry in unresolved:
        if (entry.get("geometry_reviewed") is not True or entry.get("no_visible_crack") is not False
                or not isinstance(entry.get("reason"), str) or not entry["reason"].strip()
                or any(key in entry for key in ("label_path", "label_sha256", "mask_sha256"))):
            raise ValueError("R7 geometry_unresolved anchor needs review, a reason and no label or mask")
        bound_file(entry.get("source_path"), entry.get("image_sha256"))
    for row in rows:
        entry = indexed[row["image_id"]]
        if entry.get("geometry_reviewed") is not True or entry.get("decision") != "accepted_visible_segments":
            raise ValueError("R7 contains unreviewed or pending geometry")
        if entry.get("image_sha256") != row.get("sha256"):
            raise ValueError("R7 image hash differs from the registry")
        source_path = bound_file(row.get("path"), entry["image_sha256"])
        mask_path = bound_file(row.get("mask_path"), entry.get("mask_sha256"))
        label_path = bound_file(entry.get("label_path"), entry.get("label_sha256"))
        width, height = image_size(source_path)
        task = {"workspace_filename": entry["workspace_filename"], "width": width, "height": height}
        label, errors, _ = parse_annotation(label_path.read_bytes(), task, row["image_id"])
        if errors:
            raise ValueError("R7 accepted label violates its annotation contract")
        expected_mask = centerline_mask(label, width, height)
        if not np.array_equal(load_binary_mask(mask_path), expected_mask):
            raise ValueError("R7 mask does not equal the accepted 1px centerline raster")
        prior = label.get("codex_adjudication") or {}
        if prior.get("formal_r7_accepted") is False or prior.get("geometry_review_complete") is False:
            raise ValueError("R7 staging label cannot be promoted by changing only dataset status")
        if label_path.stem + ".jpg" != entry.get("workspace_filename"):
            raise ValueError("R7 workspace name does not match its accepted label")
    fraction = float(anchor.get("double_annotation_fraction", 0))
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid registered double-annotation fraction")
    expected_count = round(int(anchor["sample_size"]) * fraction)
    double = receipt.get("double_annotation_reviews") or []
    if len(double) != expected_count or len({item["workspace_filename"] for item in double}) != len(double):
        raise ValueError("R7 double-annotation review count differs from the protocol")
    if expected_count:
        manifest_name = dataset.get("double_annotation_manifest")
        manifest_path = bound_file(manifest_name, receipt.get("double_annotation_manifest_sha256"))
        with manifest_path.open(encoding="utf-8-sig") as stream:
            assigned = {item["workspace_filename"] for item in csv.DictReader(stream)}
        actual = {item["workspace_filename"] for item in double}
        accepted_names = {entry["workspace_filename"] for entry in entries if entry["image_id"] in accepted_ids}
        if actual != assigned or not actual.issubset(accepted_names):
            raise ValueError("R7 double review changed the originally assigned sample IDs")
    for item in double:
        if (item.get("geometry_reviewed") is not True
                or item.get("review_group") not in {"independent_review", "calibration_exposed", "assisted_or_exposed"}):
            raise ValueError("R7 double review is pending or has no independence disclosure")
        label_path = bound_file(item.get("label_path"), item.get("label_sha256"))
        accepted = next(entry for entry in entries if entry["workspace_filename"] == item["workspace_filename"])
        primary = json.loads((root / accepted["label_path"]).read_text(encoding="utf-8-sig"))
        _, errors, _ = parse_annotation(label_path.read_bytes(), {"workspace_filename": item["workspace_filename"],
            "width": primary["imageWidth"], "height": primary["imageHeight"]}, "double-review")
        if errors:
            raise ValueError("R7 double-reviewed label violates its annotation contract")
    # Preserve the caller's root alias so s03 can serialize root-relative paths.
    return [Path(data_root) / path.relative_to(root) for path in sorted(set(artifacts))]
