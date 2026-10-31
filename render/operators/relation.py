# -*- coding: utf-8 -*-
"""Spatial relation r = (d, Δφ, n) (only for linear distractors × the crack layer x₁₁; fixed measurement definitions).

  d  = nearest distance between the distractor skeleton and the crack skeleton / median width of that crack;
       bins overlap (distractor band intersects the crack mask) / near (0<d<2) / excluded (2≤d≤10, not sampled
       synthetically) / far (d>10)
  Δφ = angle between the two skeleton tangents at the nearest points (0–90°), bins <15 / 15–60 / >60
  n  = number of transverse crossings (connected components of distractor band ∩ crack skeleton), bins 0 / ≥1
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import skeletonize_mask



def _tangent_deg(points: np.ndarray, center: np.ndarray, radius: float = 7.0) -> float | None:
    near = points[np.linalg.norm(points - center, axis=1) <= radius]
    if len(near) < 3:
        return None
    centered = near - near.mean(axis=0)
    _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    dy, dx = vt[0]
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def measure(distractor_centerline: np.ndarray, distractor_support: np.ndarray,
            crack_mask: np.ndarray) -> dict:
    crack = (crack_mask > 0).astype(np.uint8)
    if not crack.any() or not distractor_centerline.any():
        raise ValueError("r is defined only for samples containing both a crack and a linear distractor")
    crack_skel = skeletonize_mask(crack)
    dist_crack = cv2.distanceTransform(crack, cv2.DIST_L2, 3)
    widths = 2.0 * dist_crack[crack_skel > 0]
    width_median = float(np.median(widths)) if len(widths) else 1.0
    # nearest distance from the crack skeleton to the distractor centreline
    inv = (distractor_centerline == 0).astype(np.uint8)
    dist_to_distractor = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    skel_pts = np.argwhere(crack_skel > 0)
    d_px = dist_to_distractor[crack_skel > 0]
    nearest = int(np.argmin(d_px))
    d = float(d_px[nearest]) / max(width_median, 1e-6)
    overlap = bool(((distractor_support > 0) & (crack > 0)).any())
    if overlap:
        d_bin = "overlap"
    elif d < 2:
        d_bin = "near"
    elif d <= 10:
        d_bin = "excluded"
    else:
        d_bin = "far"
    p_crack = skel_pts[nearest].astype(np.float64)
    dist_pts = np.argwhere(distractor_centerline > 0)
    p_dist = dist_pts[int(np.argmin(np.linalg.norm(dist_pts - p_crack, axis=1)))].astype(np.float64)
    phi_c, phi_d = _tangent_deg(skel_pts, p_crack), _tangent_deg(dist_pts, p_dist)
    dphi = None if phi_c is None or phi_d is None else float(min(abs(phi_c - phi_d), 180 - abs(phi_c - phi_d)))
    dphi_bin = None if dphi is None else ("<15" if dphi < 15 else ("15-60" if dphi <= 60 else ">60"))
    crossings = (distractor_support > 0) & (crack_skel > 0)
    n = int(cv2.connectedComponents(crossings.astype(np.uint8), connectivity=8)[0] - 1) if crossings.any() else 0
    return {
        "d": round(d, 4), "d_px": round(float(d_px[nearest]), 3), "crack_width_median_px": round(width_median, 3),
        "d_bin": d_bin, "dphi_deg": None if dphi is None else round(dphi, 2), "dphi_bin": dphi_bin,
        "n_cross": n, "n_bin": "0" if n == 0 else ">=1",
    }
