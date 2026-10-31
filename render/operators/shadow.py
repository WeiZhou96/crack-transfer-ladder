# -*- coding: utf-8 -*-
"""shadow: cast shadow / linear shadow (slot L^light, applied after the brush).

Lighting model (near-linear domain): x' = x · L, L = 1 − (1 − k)·S, where S∈[0,1] is a soft shadow field;
the shadowed region is lit by sky scatter and is bluish: the channel gains are further multiplied by (1 − c·S, 1, 1 + c·S).
The penumbra grows with distance from the occluder: along the projection direction it goes linearly from penumbra/4 at
the near end to penumbra at the far end (blend of two Gaussian levels).
  cast   : canopy-like blobs from thresholded low-frequency noise; core coverage = coverage (hit exactly via a quantile);
  linear : linear shadow of a pole/wire, width line_width_px, crossing the whole image.
Ω_Z = {S > 0} (S below 1e-3 is zeroed); label-preserving, V is coupled by visibility.py after all operators.
"""
from __future__ import annotations

import numpy as np

from .base import (Context, OperatorResult, as_rgb_float, check_material, choice, finalize,
                   gaussian_soften, line_coordinates, linear_to_srgb, smooth_noise, srgb_to_linear,
                   uniform)

NAME = "shadow"


def sample_theta(rng: np.random.Generator, ctx: Context, domain: dict, shape: tuple[int, int]) -> dict:
    h, w = shape
    kind = choice(rng, domain["kind"])
    theta = {
        "operator": NAME,
        "kind": kind,
        "transmittance_k": uniform(rng, domain["transmittance_k"]),
        "penumbra_px": uniform(rng, domain["penumbra_px"]),
        "sky_blue_shift": uniform(rng, domain["sky_blue_shift"]),
        "light_azimuth_deg": float(rng.uniform(0, 360)),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }
    if kind == "cast":
        theta["coverage"] = uniform(rng, domain["coverage"])
        theta["blob_scale_frac"] = uniform(rng, domain["cast_blob_scale_frac"])
    else:
        theta["line_width_px"] = uniform(rng, domain["line_width_px"])
        theta["line_angle_deg"] = float(rng.uniform(0, 180))
        theta["line_offset_px"] = float(rng.uniform(-0.35, 0.35) * min(h, w))
    return theta


def _hard_mask(theta: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = shape
    if theta["kind"] == "cast":
        rng = np.random.default_rng(theta["seed"])
        field_ = smooth_noise(shape, theta["blob_scale_frac"] * min(h, w), rng)
        threshold = np.quantile(field_, 1.0 - theta["coverage"])
        return (field_ > threshold).astype(np.uint8), None
    s, _ = line_coordinates(shape, theta["line_angle_deg"], theta["line_offset_px"])
    mask = (np.abs(s) <= theta["line_width_px"] / 2.0).astype(np.uint8)
    return mask, (np.abs(s) <= 0.5 + 1e-6).astype(np.uint8)


def render(image: np.ndarray, theta: dict, ctx: Context, domain: dict) -> OperatorResult:
    check_material(NAME, domain["materials"], ctx)
    x = as_rgb_float(image)
    h, w = x.shape[:2]
    hard, centerline = _hard_mask(theta, (h, w))
    near = gaussian_soften(hard, max(theta["penumbra_px"] / 4.0, 1.0))
    far = gaussian_soften(hard, theta["penumbra_px"])
    # normalised distance along the projection direction: 0 = near the occluder, 1 = far end
    a = np.deg2rad(theta["light_azimuth_deg"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(a) + yy * np.sin(a)
    weight = (proj - proj.min()) / max(float(proj.max() - proj.min()), 1e-6)
    soft = (1.0 - weight) * near + weight * far
    soft[soft < 1e-3] = 0.0
    omega = soft > 0
    k, c = theta["transmittance_k"], theta["sky_blue_shift"]
    gain = 1.0 - (1.0 - k) * soft
    tint = np.stack([1.0 - c * soft, np.ones_like(soft), 1.0 + c * soft], axis=2)
    lin = srgb_to_linear(x) * gain[..., None] * tint
    out = finalize(x, linear_to_srgb(lin), omega)
    measured = {
        "core_coverage": round(float((soft >= 0.9).mean()), 5),
        "support_fraction": round(float(omega.mean()), 5),
    }
    return OperatorResult(out, omega, dict(theta), "L_light", centerline, measured)
