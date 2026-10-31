# -*- coding: utf-8 -*-
"""Crack brush P_M (real crack texture transfer).

Geometry: the target mask M is a mask-library instance mask under a dihedral transform (8 variants: rotation by 90°×k
      + flip), translated onto the canvas; parts outside the canvas are cropped (in real images cracks are likewise cut
      by the frame). No scaling and no arbitrary-angle rotation, so line width and morphology stay real and source and
      target geometry correspond pixel by pixel; thin-plate-spline alignment therefore reduces to the identity here
      (alignment holds by construction).
Appearance: the crack's "transmittance field" is transferred, not the pixels themselves:
      R = L_src / L̂_src, where L is the linear luminance of the source image and L̂ is the background estimate obtained
      by inpainting all source GT (dilated by 2px) (Telea); R takes values only inside dilate(M_inst, 2px) and is 1
      elsewhere; rendering is P(B', M) = B' ⊙ R (linear domain, all three channels multiplied equally).
  This is the log-domain special case of Poisson blending (log R as guidance field, zero boundary). Unlike a direct
  Poisson copy, the source background texture does not enter the target image: outside the band R≡1, and inside the
  band only the crack's brightness ratio to its local background remains. Rationale (source filtering):
    · local illumination normalisation: achieved by construction through division by L̂;
    · confuser exclusion: the source datasets have no distractor polygons (confuser_annotation_available=false is
      recorded when the mask library is built); ratio transfer keeps shadows/stains/joints in the source image that do
      not coincide with the crack out of the target image; those coinciding with the crack still get in and can only be
      caught by the source-ID probe;
    · material matching: the assembler selects instances by material_pool.
Pairing purity: R and M depend only on (instance, dihedral, position), with no randomness; x₁₀/x₁₁ of a group use the
same cached R, so the crack realisation is pixel-identical (paired RNG).
In the composite x₁ₓ = B'(1−G_M) + P(B',M)·G_M, pixels outside the band equal B' bit-exactly: here R≡1 outside the band,
so this holds by construction (support Ω_M = pixels where R≠1).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .operators.base import as_rgb_float, linear_to_srgb, skeletonize_mask, srgb_to_linear

PAD = 16            # padding of the source crop (inpainting needs context)
BAND_PX = 2         # band where the transmittance field takes values: dilate(M_inst, 2)
HOLE_PX = 2         # inpainting hole: dilate(all GT, 2)
R_CLIP = (0.02, 2.0)


@dataclass
class BrushField:
    mask: np.ndarray        # uint8 0/1, instance mask (source-crop coordinates)
    ratio: np.ndarray       # float32, transmittance R (= 1 outside the band)
    stats: dict


def _lum(rgb_lin: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]).astype(np.float32)


def load_binary_gt(path: str) -> np.ndarray:
    """Same value-range rule as poc/common.load_binary_mask (max≤1 → >0; otherwise >127; 1<max≤127 is rejected)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    m = int(arr.max())
    if m <= 1:
        return arr > 0
    if m <= 127:
        raise ValueError(f"ambiguous value range for a binary mask (max={m}): {path}")
    return arr > 127


def extract_field(source_path: str, source_mask_path: str, bbox_xyxy: list[int],
                  instance_mask_crop: np.ndarray) -> BrushField:
    """bbox_xyxy: crop box recorded in the mask library (already padded by 8px, bottom-right exclusive); instance_mask_crop is masks/<id>.png."""
    with Image.open(source_path) as im:
        src = as_rgb_float(np.asarray(im.convert("RGB")))
    gt_all = load_binary_gt(source_mask_path)
    h, w = gt_all.shape
    if src.shape[:2] != (h, w):
        raise ValueError(f"source image and GT sizes differ: {source_path}")
    x0, y0, x1, y1 = bbox_xyxy
    if instance_mask_crop.shape != (y1 - y0, x1 - x0):
        raise ValueError("instance crop size does not match bbox_xyxy")
    inst_full = np.zeros((h, w), np.uint8)
    inst_full[y0:y1, x0:x1] = instance_mask_crop > 0
    if (inst_full & ~gt_all).any():
        raise ValueError("instance mask is not inside the source GT")
    X0, Y0 = max(x0 - PAD, 0), max(y0 - PAD, 0)
    X1, Y1 = min(x1 + PAD, w), min(y1 + PAD, h)
    rgb = src[Y0:Y1, X0:X1]
    inst = inst_full[Y0:Y1, X0:X1]
    hole = cv2.dilate(gt_all[Y0:Y1, X0:X1].astype(np.uint8), np.ones((2 * HOLE_PX + 1,) * 2, np.uint8))
    lum = _lum(srgb_to_linear(rgb))
    # Telea inpainting runs on 16-bit integers: float32 Telea in OpenCV 4.11 produces negative values inside the hole (observed), so it is unusable
    lum16 = np.round(np.clip(lum, 0.0, 1.0) * 65535.0).astype(np.uint16)
    background = cv2.inpaint(lum16, hole, 5, cv2.INPAINT_TELEA).astype(np.float32) / 65535.0
    band = cv2.dilate(inst, np.ones((2 * BAND_PX + 1,) * 2, np.uint8)) > 0
    ratio = np.ones_like(lum)
    ratio[band] = np.clip(lum[band] / np.maximum(background[band], 1e-4), *R_CLIP)
    inside = inst > 0
    stats = {"ratio_median_in_mask": round(float(np.median(ratio[inside])), 4),
             "ratio_p10_in_mask": round(float(np.percentile(ratio[inside], 10)), 4),
             "band_px": int(band.sum()), "mask_px": int(inside.sum())}
    return BrushField(inst, ratio, stats)


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """k∈0..7: k%4 counter-clockwise 90° rotations, preceded by a left-right flip when k≥4."""
    if k >= 4:
        a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, k % 4))


def place(field: BrushField, k: int, ox: int, oy: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Place the top-left corner of the (dihedrally transformed) crop at canvas (ox, oy); return (M uint8, R float32) in canvas coordinates."""
    m, r = dihedral(field.mask, k), dihedral(field.ratio, k)
    H, W = shape
    M = np.zeros((H, W), np.uint8)
    R = np.ones((H, W), np.float32)
    h, w = m.shape
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    dx1, dy1 = min(W, ox + w), min(H, oy + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return M, R
    M[dy0:dy1, dx0:dx1] = m[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    R[dy0:dy1, dx0:dx1] = r[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    return M, R


def placement_ok(M: np.ndarray, total_skel_px: float, canvas_side: int, min_area: int = 64) -> tuple[bool, dict]:
    skel_in = int(skeletonize_mask(M > 0).sum()) if M.any() else 0
    need = max(30.0, min(0.5 * total_skel_px, 0.75 * canvas_side))
    info = {"skel_in_canvas_px": skel_in, "area_in_canvas_px": int(M.sum()), "skel_needed_px": round(need, 1)}
    return skel_in >= need and int(M.sum()) >= min_area, info


def apply(background: np.ndarray, R: np.ndarray, polarity: str = "dark") -> tuple[np.ndarray, np.ndarray]:
    """P(B', M) = B' ⊙ R (linear domain); return (image, support Ω_M = {R ≠ 1}). Outside Ω_M the output equals the input bit-exactly.
    polarity="bright": in laser line-scan grey frames cracks appear as bright lines (on PaveDistress samples the centreline
    is ~40% brighter than the surrounding ring); the same transmittance is then applied in the 8-bit inverted domain 1−x
    and inverted back, i.e. the inverse of the inversion used to measure bright-crack contrast;
    outside Ω_M the input is still taken bit-exactly (np.where keeps the original value, with no floating-point round trip through two inversions)."""
    x = as_rgb_float(background)
    omega = R != 1.0
    if polarity == "dark":
        rendered = linear_to_srgb(srgb_to_linear(x) * R[..., None])
    elif polarity == "bright":
        rendered = 1.0 - linear_to_srgb(srgb_to_linear(1.0 - x) * R[..., None])
    else:
        raise ValueError(f"unknown polarity {polarity!r}")
    out = np.where(omega[..., None], np.clip(rendered, 0.0, 1.0), x).astype(np.float32)
    return out, omega
