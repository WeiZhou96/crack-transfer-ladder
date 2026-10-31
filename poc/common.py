# -*- coding: utf-8 -*-
"""Shared primitives for the POC scripts: IO, hashing, binary masks, connected components,
skeletons, metrics.

Conventions:
- Binary masks are always uint8 {0,1} in memory; class label maps (Pavementscapes/GAPs) keep
  their original integer values and are read with load_label_raw.
- Component analysis always uses 8-connectivity.
- Default component filter (pixels): area >= 64 px and 8-neighbour weighted skeleton length
  >= 20 px; compact large blobs with area >= 900 px are exempt from the skeleton threshold
  (a conservative area gate corresponding to (30 mm)^2 at about 1 mm/px).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # GAPs 10m images are 5030x11505, above PIL's default decompression-bomb limit

DEFAULT_MIN_AREA_PX = 64
DEFAULT_MIN_SKEL_PX = 20
DEFAULT_LARGE_AREA_PX = 900
PREDICTION_SCALE_U16 = 65535

TRAIN_ROLES = frozenset({"train_positive", "train_positive_optional", "train_negative"})
PROXY_ROLES = frozenset({"proxy_bench"})
EVAL_ROLES = frozenset({
    "eval_hard_positive", "eval_hard_positive_optional", "eval_negative",
    "eval_confuser", "eval_unseen_sensor_auxiliary",
})
QUARANTINE_ROLES = frozenset({"quarantine_unverified"})
APPLICATION_ROLES = frozenset({"application_parallel"})
ROLE_WHITELIST = TRAIN_ROLES | PROXY_ROLES | EVAL_ROLES | QUARANTINE_ROLES | APPLICATION_ROLES


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------- IO / hashing

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - only for checking publisher checksums, not for security
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) without decoding the full image."""
    with Image.open(path) as im:
        return im.size


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_label_raw(path: Path) -> np.ndarray:
    """Class label map: keep original integer values (P or L mode becomes an integer array; no binarisation)."""
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im)
        return np.array(im.convert("L"))


def load_binary_mask(path: Path) -> np.ndarray:
    """Read a binary mask and normalise it to {0,1}.

    The value range is detected automatically: masks with max<=1 are thresholded at >0,
    otherwise at >127 (covers both 0/1 and 0/255 release conventions; class label maps
    must not be read with this function).
    """
    arr = load_gray(path)
    maximum = int(arr.max())
    if maximum <= 1:
        return (arr > 0).astype(np.uint8)
    if maximum <= 127:
        raise ValueError(
            f"Ambiguous binary mask value range (max={maximum}): only 0/1 or 0/255 is accepted; "
            f"class label maps must use load_label_raw: {path}"
        )
    return (arr > 127).astype(np.uint8)


def encode_probability_u16(probability: np.ndarray) -> np.ndarray:
    """Encode a continuous [0,1] probability map as 16-bit PNG pixel values."""
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Probability map must be a 2-D array, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Probability map contains NaN/Inf")
    return np.rint(np.clip(array, 0.0, 1.0) * PREDICTION_SCALE_U16).astype(np.uint16)


def save_probability_png(path: Path, probability: np.ndarray, compress_level: int = 1) -> None:
    """Write a 16-bit grayscale PNG following the project's prediction contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_probability_u16(probability)
    Image.fromarray(encoded).save(path, compress_level=compress_level)


def load_probability_png(path: Path) -> np.ndarray:
    """Read a project 16-bit prediction PNG and return float32 in [0,1]; legacy 8-bit maps are rejected."""
    with Image.open(path) as image:
        mode = image.mode
        encoded = np.asarray(image)
    if mode not in {"I;16", "I;16L", "I;16B"} and encoded.dtype != np.uint16:
        raise ValueError(
            f"Prediction map is not a 16-bit grayscale PNG (mode={mode}, dtype={encoded.dtype}): {path}"
        )
    if encoded.ndim != 2:
        raise ValueError(f"Prediction map must be a 2-D grayscale image, got shape={encoded.shape}: {path}")
    return encoded.astype(np.float32) / PREDICTION_SCALE_U16


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def default_development_exposures_auto(registry_path: Path) -> Path:
    return registry_path.resolve().parent.parent / "development_exposures.auto.yaml"


def append_development_exposures(path: Path, rows, *, reason: str,
                                 source: str, model: str | None = None) -> int:
    """Idempotently append pre-freeze QA inputs that will be inspected by humans to the automatic exposure list."""
    path = path.resolve()
    if path.exists():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        document = {
            "version": 1,
            "policy": "Generated by QA tools; merge with configs/development_exposures.yaml before freeze.",
            "exposures": [],
        }
    exposures = document.setdefault("exposures", [])
    exposures[:] = [
        {"image_id": item, "reason": "legacy auto exposure", "source": "unknown"}
        if isinstance(item, str) else item
        for item in exposures
    ]
    existing = {
        (str(item.get("image_id")), str(item.get("source")), str(item.get("model") or ""))
        for item in exposures if isinstance(item, dict)
    }
    added = 0
    for row in rows:
        image_id = str(row["image_id"] if isinstance(row, dict) else row)
        key = (image_id, source, model or "")
        if key in existing:
            continue
        item = {"image_id": image_id, "reason": reason, "source": source}
        if model:
            item["model"] = model
        exposures.append(item)
        existing.add(key)
        added += 1
    exposures.sort(key=lambda item: (
        str(item.get("image_id", "")), str(item.get("source", "")), str(item.get("model", ""))))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return added


def freeze_version(path: Path) -> int:
    if path.name == "FREEZE.json":
        return 1
    match = re.fullmatch(r"FREEZE\.v(\d+)\.json", path.name)
    return int(match.group(1)) if match else -1


def latest_freeze_path(data_root: Path) -> Path:
    candidates = [
        path for path in (data_root / "splits").glob("FREEZE*.json")
        if freeze_version(path) > 0
    ]
    if not candidates:
        return data_root / "splits" / "FREEZE.json"
    return max(candidates, key=freeze_version)


def load_verified_freeze(data_root: Path, explicit: Path | None = None) -> tuple[Path, dict]:
    """Load the highest-version FREEZE and verify all recorded artifact, config and (recursive) code hashes."""
    latest = latest_freeze_path(data_root)
    freeze_path = explicit.resolve() if explicit else latest.resolve()
    if not freeze_path.exists():
        raise SystemExit(f"Missing freeze snapshot {freeze_path}; run s03 first")
    if latest.exists() and freeze_path != latest.resolve():
        raise SystemExit(f"Refusing to use outdated freeze {freeze_path.name}; latest version is {latest.name}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {"artifacts_sha256", "config_sha256", "code_sha256"}
    missing_sections = sorted(required - freeze.keys())
    if missing_sections:
        raise SystemExit(f"Freeze snapshot is missing integrity sections: {missing_sections}")

    for relative, expected in freeze["artifacts_sha256"].items():
        path = data_root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise SystemExit(f"Frozen artifact hash mismatch: {relative}")
    for absolute, expected in freeze["config_sha256"].items():
        path = Path(absolute)
        if not path.exists() or sha256_file(path) != expected:
            raise SystemExit(f"Frozen config hash mismatch: {path}")
    code_root = Path(__file__).resolve().parents[1]
    for relative, expected in freeze["code_sha256"].items():
        path = code_root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise SystemExit(f"Frozen code hash mismatch: {relative}")
    return freeze_path, freeze


def ensure_prediction_output_compatible(output_dir: Path, weights_sha256: str,
                                        overwrite: bool) -> None:
    """Existing predictions may be skipped only if their metadata weights hash matches the current weights."""
    if overwrite or not output_dir.exists() or not next(output_dir.glob("*.png"), None):
        return
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"{output_dir} contains predictions but no metadata.json; refusing to skip without --overwrite"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous = metadata.get("weights_sha256")
    if previous != weights_sha256:
        raise RuntimeError(
            f"{output_dir}: weights hash of existing predictions {previous!r} differs from current {weights_sha256!r}; "
            "recompute explicitly with --overwrite"
        )


# ---------------------------------------------------------- components / skel

def connected_components(mask: np.ndarray):
    """8-connectivity. Returns (labels int32, n_components)."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, n - 1


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def skeleton_length_8n(skeleton: np.ndarray) -> float:
    """8-neighbour skeleton length on the pixel-centre graph; orthogonal edges weigh 1, diagonal edges sqrt(2).

    Each skeleton connected component adds a 1 px baseline, so a single-pixel skeleton has length 1
    and an N-pixel straight line still has length N.
    """
    skel = skeleton.astype(bool)
    if not np.any(skel):
        return 0.0
    horizontal = int(np.count_nonzero(skel[:, 1:] & skel[:, :-1]))
    vertical = int(np.count_nonzero(skel[1:, :] & skel[:-1, :]))
    diagonal = int(np.count_nonzero(skel[1:, 1:] & skel[:-1, :-1]))
    diagonal += int(np.count_nonzero(skel[1:, :-1] & skel[:-1, 1:]))
    _, components = connected_components(skel.astype(np.uint8))
    return float(components + horizontal + vertical + np.sqrt(2.0) * diagonal)


def component_stats(mask: np.ndarray, skeleton: np.ndarray | None = None):
    """Per-component statistics; skel_px is the 8-neighbour weighted length, skel_pixels the raw skeleton pixel count."""
    if skeleton is None:
        skeleton = skeletonize_mask(mask)
    labels, n = connected_components(mask)
    if n == 0:
        return [], labels
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    stats = []
    for i in range(1, n + 1):
        component_skeleton = (labels == i) & (skeleton > 0)
        stats.append({
            "label": i,
            "area_px": int(areas[i]),
            "skel_px": skeleton_length_8n(component_skeleton),
            "skel_pixels": int(component_skeleton.sum()),
        })
    return stats, labels


def filter_components(mask: np.ndarray,
                      min_area_px: int = DEFAULT_MIN_AREA_PX,
                      min_skel_px: int = DEFAULT_MIN_SKEL_PX,
                      large_area_px: int = DEFAULT_LARGE_AREA_PX):
    """Component filter: regular components must pass both the area and skeleton gates; compact large blobs are kept by the area gate alone."""
    stats, labels = component_stats(mask)
    keep = [
        stat for stat in stats
        if stat["area_px"] >= large_area_px
        or (stat["area_px"] >= min_area_px and stat["skel_px"] >= min_skel_px)
    ]
    if not keep:
        return np.zeros_like(mask), []
    keep_labels = np.zeros(labels.max() + 1, dtype=bool)
    for s in keep:
        keep_labels[s["label"]] = True
    return keep_labels[labels].astype(np.uint8), keep


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.dilate(mask.astype(np.uint8), k)


# ------------------------------------------------------------------- metrics

def component_recall(gt_mask: np.ndarray, pred_mask: np.ndarray,
                     dilate_px: int = 4, cover: float = 0.5,
                     min_area_px: int = DEFAULT_MIN_AREA_PX,
                     min_skel_px: int = DEFAULT_MIN_SKEL_PX) -> dict:
    """Detection criterion for a GT component (pixel version): at least a `cover` fraction of its skeleton lies inside dilate(pred, dilate_px)."""
    gt_f, _ = filter_components(gt_mask, min_area_px, min_skel_px)
    skel = skeletonize_mask(gt_f)
    gt_keep, labels = component_stats(gt_f, skeleton=skel)
    if not gt_keep:
        return {"n_gt": 0, "n_hit": 0, "recall": None}
    pred_d = dilate(pred_mask, dilate_px)
    n_hit = 0
    for stat in gt_keep:
        comp_skel = (labels == stat["label"]) & (skel > 0)
        total = int(comp_skel.sum())
        if total == 0:
            continue
        hit = int((comp_skel & (pred_d > 0)).sum())
        if hit / total >= cover:
            n_hit += 1
    return {"n_gt": len(gt_keep), "n_hit": n_hit, "recall": n_hit / len(gt_keep)}


def negative_image_accounts(pred_mask: np.ndarray,
                            min_area_px: int = DEFAULT_MIN_AREA_PX,
                            min_skel_px: int = DEFAULT_MIN_SKEL_PX) -> dict:
    """Raw quantities of the three false-alarm accounts on a negative image (component / image / area level). The denominator (Mpx) is aggregated by the caller."""
    h, w = pred_mask.shape[:2]
    raw = (pred_mask > 0).astype(np.uint8)
    filtered, comps = filter_components(raw, min_area_px, min_skel_px)
    skel = skeletonize_mask(filtered) if comps else np.zeros_like(filtered)
    return {
        "mpx": h * w / 1e6,
        "fp_components": len(comps),
        "phantom_len_px": skeleton_length_8n(skel),
        "fp_area_px": int(raw.sum()),
        "image_fired": int(np.any(raw)),
    }


def binarize_prediction(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise a prediction map; the standard path passes [0,1] floats, uint16 is accepted only for low-level compatibility."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0,1], got {threshold}")
    if arr.dtype == np.uint8:
        if int(arr.max()) <= 1:
            raise ValueError(
                "uint8 prediction map contains only 0/1, so a quantised probability map cannot be told apart from a binary map; "
                "prediction exports must keep the continuous probability dynamic range."
            )
        raise ValueError("The 8-bit prediction contract is retired; re-export as 16-bit probability PNG")
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / PREDICTION_SCALE_U16
    array = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Prediction probabilities contain NaN/Inf")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("Prediction probabilities outside [0,1]")
    return (array >= threshold).astype(np.uint8)
