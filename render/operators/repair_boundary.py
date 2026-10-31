# -*- coding: utf-8 -*-
"""repair_boundary: boundary of an old patch/repair (restricted to the S_pre slot: the repair predates the crack; label-preserving).

θ_p = (patch area, boundary length, inside/outside brightness difference, boundary shadow width). The geometry is the
signed distance field of a rotated rounded rectangle plus a low-frequency displacement for an irregular edge;
photometry (multiplicative, near-linear domain):
  interior    : × (1 + brightness_delta), blended with a blurred copy by texture_smooth (patch material is newer and more uniform);
  seam        : × (1 − seam_darkness) where |sdf| < seam_w/2 (saw cut / sealant line);
  step shadow : the patch surface sits slightly above the original pavement; on the side facing away from the light,
                the outer rim 0 < sdf < step_shadow_px is darkened, fading linearly outward.
Measurements (measured): area_frac, boundary_len_px (8-connected skeleton length), inner_outer_ratio (linear-domain luminance ratio).
Ω_Z = sdf < max(seam_w/2, step_shadow_px) + 1.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import (Context, OperatorResult, as_rgb_float, check_material, finalize, linear_to_srgb,
                   smooth_noise, srgb_to_linear, uniform)

NAME = "repair_boundary"


def sample_theta(rng: np.random.Generator, ctx: Context, domain: dict, shape: tuple[int, int]) -> dict:
    h, w = shape
    return {
        "operator": NAME,
        "area_frac": uniform(rng, domain["area_frac"]),
        "aspect": uniform(rng, domain["aspect"]),
        "angle_deg": float(rng.uniform(-20.0, 20.0)),
        "center_frac": [float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7))],
        "corner_radius_frac": float(rng.uniform(0.0, 0.25)),
        "brightness_delta": uniform(rng, domain["brightness_delta"]),
        "texture_smooth": uniform(rng, domain["texture_smooth"]),
        "seam_width_px": uniform(rng, domain["seam_width_px"]),
        "seam_darkness": uniform(rng, domain["seam_darkness"]),
        "step_shadow_px": uniform(rng, domain["step_shadow_px"]),
        "edge_jitter_px": uniform(rng, domain["edge_jitter_px"]),
        "light_azimuth_deg": float(rng.uniform(0, 360)),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }


def signed_distance(theta: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    area = theta["area_frac"] * h * w
    half_w = np.sqrt(area * theta["aspect"]) / 2.0
    half_h = np.sqrt(area / theta["aspect"]) / 2.0
    radius = theta["corner_radius_frac"] * min(half_w, half_h)
    cx, cy = theta["center_frac"][0] * (w - 1), theta["center_frac"][1] * (h - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(theta["angle_deg"])
    x = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    y = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    qx, qy = np.abs(x) - (half_w - radius), np.abs(y) - (half_h - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    sdf = outside + inside - radius
    rng = np.random.default_rng(theta["seed"])
    return (sdf + theta["edge_jitter_px"] * smooth_noise(shape, 12.0, rng)).astype(np.float32)


def render(image: np.ndarray, theta: dict, ctx: Context, domain: dict) -> OperatorResult:
    check_material(NAME, domain["materials"], ctx)
    x = as_rgb_float(image)
    h, w = x.shape[:2]
    sdf = signed_distance(theta, (h, w))
    lin = srgb_to_linear(x)
    inside = sdf < 0
    smooth = cv2.GaussianBlur(lin, (0, 0), 2.0, borderType=cv2.BORDER_REFLECT)
    mix = theta["texture_smooth"]
    patch = ((1 - mix) * lin + mix * smooth) * (1.0 + theta["brightness_delta"])
    out_lin = np.where(inside[..., None], patch, lin)
    # step shadow: the side where the outward normal makes > 90° with the illumination direction (shaded side)
    gy, gx = np.gradient(sdf)
    light = np.array([np.cos(np.deg2rad(theta["light_azimuth_deg"])),
                      np.sin(np.deg2rad(theta["light_azimuth_deg"]))], np.float32)
    away = (gx * light[0] + gy * light[1]) > 0      # sdf gradient = outward normal; aligned with the incoming light direction = shaded side
    step = theta["step_shadow_px"]
    shadow_zone = (sdf > 0) & (sdf < step) & away
    shade = 1.0 - 0.35 * (1.0 - sdf / step)
    out_lin = np.where(shadow_zone[..., None], out_lin * shade[..., None], out_lin)
    seam = np.abs(sdf) < theta["seam_width_px"] / 2.0
    out_lin = np.where(seam[..., None], out_lin * (1.0 - theta["seam_darkness"]), out_lin)
    omega = sdf < max(theta["seam_width_px"] / 2.0, step) + 1.0
    out = finalize(x, linear_to_srgb(out_lin), omega)
    boundary = (np.abs(sdf) <= 0.5 + 1e-6).astype(np.uint8)
    from skimage.morphology import skeletonize
    skel = skeletonize(cv2.dilate(boundary, np.ones((3, 3), np.uint8)) > 0)
    ring = (sdf > 6) & (sdf < 18)
    inner = (sdf < -6)
    lum = srgb_to_linear(out).mean(axis=2)
    measured = {
        "area_frac": round(float(inside.mean()), 5),
        "boundary_len_px": int(skel.sum()),
        "inner_outer_ratio": (round(float(lum[inner].mean() / max(lum[ring].mean(), 1e-6)), 4)
                              if inner.any() and ring.any() else None),
        "support_fraction": round(float(omega.mean()), 5),
    }
    return OperatorResult(out, omega, dict(theta), "S_pre", None, measured)
