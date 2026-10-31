# -*- coding: utf-8 -*-
"""Visibility term V: after all operators are applied, the local Weber contrast in a 15×15 neighbourhood of the crack centreline,
C_w = |L_bg − L_crack| / L_bg (L is linear-domain luminance), V = 1[C_w ≥ τ_c], τ_c = 0.02 by default.
V is computed on skeleton points and propagated to the whole crack mask via the nearest skeleton point; y = M̃ ⊙ V, ignore ⊇ M̃ ⊙ ¬V.
Note: a uniform multiplicative shadow does not change the Weber contrast (physically correct); deep shadow makes a crack
indiscernible only through A_ν (noise, quantisation, blur), so V must be computed on the final layer (including A_ν)."""
from __future__ import annotations

import cv2
import numpy as np


from .base import as_rgb_float, skeletonize_mask, srgb_to_linear


def luminance(image: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(as_rgb_float(image))
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)


def weber_contrast_on_skeleton(image: np.ndarray, crack_mask: np.ndarray, window_px: int = 15) -> tuple[np.ndarray, np.ndarray]:
    crack = (crack_mask > 0).astype(np.float32)
    lum = luminance(image)
    k = (window_px, window_px)
    s_crack = cv2.boxFilter(lum * crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    n_crack = cv2.boxFilter(crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    s_bg = cv2.boxFilter(lum * (1 - crack), -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    n_bg = cv2.boxFilter(1 - crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    l_crack = s_crack / np.maximum(n_crack, 1e-6)
    l_bg = s_bg / np.maximum(n_bg, 1e-6)
    contrast = np.abs(l_bg - l_crack) / np.maximum(l_bg, 1e-4)
    skel = skeletonize_mask(crack_mask > 0)
    contrast[(n_bg < 1) | (skel == 0)] = 0.0
    return contrast, skel


def visibility_mask(image: np.ndarray, crack_mask: np.ndarray, tau_c: float = 0.02,
                    window_px: int = 15) -> tuple[np.ndarray, dict]:
    """Return (bool map of V on the crack mask, statistics)."""
    crack = crack_mask > 0
    if not crack.any():
        return np.zeros(crack.shape, bool), {"crack_px": 0, "visible_frac": None}
    contrast, skel = weber_contrast_on_skeleton(image, crack_mask, window_px)
    visible_skel = (skel > 0) & (contrast >= tau_c)
    # nearest-skeleton propagation: distance transform over non-skeleton pixels, taking the label of the nearest skeleton point
    _, labels = cv2.distanceTransformWithLabels((skel == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(skel)
    label_visible = np.zeros(int(labels.max()) + 1, bool)
    label_visible[labels[ys, xs]] = visible_skel[ys, xs]
    v = crack & label_visible[labels]
    return v, {"crack_px": int(crack.sum()), "visible_frac": round(float(v.sum() / crack.sum()), 5),
               "skeleton_px": int(skel.sum()), "tau_c": tau_c}


def signed_contrast_on_skeleton(image: np.ndarray, crack_mask: np.ndarray, window_px: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """(L_bg − L_crack) / L_bg on skeleton points (positive = crack darker than its surround). Same windows as the frozen C_w."""
    crack = (crack_mask > 0).astype(np.float32)
    lum = luminance(image)
    k = (window_px, window_px)
    l_crack = cv2.boxFilter(lum * crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT) / np.maximum(
        cv2.boxFilter(crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT), 1e-6)
    n_bg = cv2.boxFilter(1 - crack, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    l_bg = cv2.boxFilter(lum * (1 - crack), -1, k, normalize=False, borderType=cv2.BORDER_REFLECT) / np.maximum(n_bg, 1e-6)
    contrast = (l_bg - l_crack) / np.maximum(l_bg, 1e-4)
    skel = skeletonize_mask(crack_mask > 0)
    contrast[(n_bg < 1) | (skel == 0)] = 0.0
    return contrast, skel


def _propagate(crack: np.ndarray, skel: np.ndarray, visible_skel: np.ndarray) -> np.ndarray:
    _, labels = cv2.distanceTransformWithLabels((skel == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(skel)
    label_visible = np.zeros(int(labels.max()) + 1, bool)
    label_visible[labels[ys, xs]] = visible_skel[ys, xs]
    return crack & label_visible[labels]


def visibility_mask_paired(image_with: np.ndarray, image_without: np.ndarray, crack_mask: np.ndarray,
                           tau_delta: float, window_px: int = 15) -> tuple[np.ndarray, dict]:
    """Texture-normalised visibility (τ_Δ requires human visibility calibration; the frozen definition visibility_mask is kept unchanged).
    Uses the pairing structure of the quadruplet: same background, same Z, same A_ν realisation (paired RNG), differing only
    in the crack. Along the same skeleton, ΔC = C_w(x_with) − C_w(x_without), i.e. the contrast contributed by the crack itself;
    texture and lighting are identical in both images, so ΔC ≡ 0 without a crack and texture alone can no longer "create"
    visibility (in a zero-contrast diagnostic, texture alone pushed 65–87% of points over the frozen threshold τ_c=0.02).
    V = 1[ΔC ≥ τ_Δ], propagated to the whole mask via the nearest skeleton point."""
    crack = crack_mask > 0
    if not crack.any():
        return np.zeros(crack.shape, bool), {"crack_px": 0, "visible_frac": None}
    # signed (dark-crack) contrast: an absolute |ΔL| would make ΔC negative wherever the path happens to lie on a locally
    # bright streak, because darkening first reduces |ΔL|.
    c_with, skel = signed_contrast_on_skeleton(image_with, crack_mask, window_px)
    c_without, _ = signed_contrast_on_skeleton(image_without, crack_mask, window_px)
    delta = np.where(skel > 0, c_with - c_without, 0.0)
    v = _propagate(crack, skel, (skel > 0) & (delta >= tau_delta))
    d = delta[skel > 0]
    return v, {"crack_px": int(crack.sum()), "visible_frac": round(float(v.sum() / crack.sum()), 5),
               "skeleton_px": int(skel.sum()), "tau_delta": tau_delta,
               "delta_median": round(float(np.median(d)), 5), "delta_p10": round(float(np.percentile(d, 10)), 5)}
