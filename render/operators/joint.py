# -*- coding: utf-8 -*-
"""joint: concrete slab joint (slot S_pre, drawn beneath the crack; photometric groove rendering, not a painted black line).

Typed precondition: concrete background only (ctx.material == "concrete"), otherwise PreconditionError.
Geometry: straight joints crossing the whole image; with known GSD the number and spacing follow a 4–6 m period,
without GSD 1–2 joints (default).
Photometry (multiplicative, keeps the background texture):
  groove floor: |s| < w/2, multiply by 1 − darkness·profile(s); profile is 1 at the groove centre and falls smoothly to 0 at the edge;
  lips        : w/2 ≤ |s| < w/2 + b (b = max(1, w/4)); the lit side is multiplied by 1 + lip, the shaded side by 1 − lip;
  walls       : the shaded wall is darkened by an extra lip relative to the lit wall (groove self-shadowing).
Edge undulation: a low-frequency along-line displacement (edge_jitter_px) is added to the normal distance s. Ω_Z = |s| < w/2 + b + 1.
"""
from __future__ import annotations

import numpy as np

from .base import (Context, OperatorResult, as_rgb_float, check_material, choice, finalize,
                   line_coordinates, smooth_noise_1d, uniform)

NAME = "joint"


def sample_theta(rng: np.random.Generator, ctx: Context, domain: dict, shape: tuple[int, int]) -> dict:
    h, w = shape
    angle = float(rng.choice([0.0, 90.0]) + rng.uniform(-8.0, 8.0))  # joints are mostly longitudinal/transverse; allow camera-angle skew
    theta = {
        "operator": NAME,
        "angle_deg": angle,
        "width_px": uniform(rng, domain["width_px"]),
        "groove_darkness": uniform(rng, domain["groove_darkness"]),
        "lip_contrast": uniform(rng, domain["lip_contrast"]),
        "edge_jitter_px": uniform(rng, domain["edge_jitter_px"]),
        "light_azimuth_deg": float(rng.uniform(0, 360)),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }
    extent = abs(np.cos(np.deg2rad(angle))) * h + abs(np.sin(np.deg2rad(angle))) * w  # extent along the normal
    if ctx.gsd_mm_per_pixel:
        period_px = uniform(rng, domain["period_m"]) * 1000.0 / ctx.gsd_mm_per_pixel
        theta["period_px"] = period_px
        first = float(rng.uniform(-period_px / 2, period_px / 2))
        offsets = [first + i * period_px for i in range(-20, 21)
                   if abs(first + i * period_px) < extent / 2 - theta["width_px"]]
        if not offsets:                              # field of view smaller than one period: no joint in this image
            offsets = []
    else:
        n = int(choice(rng, domain["n_joints_without_gsd"]))
        span = extent / 2 - theta["width_px"] - 4
        offsets = sorted(float(v) for v in rng.uniform(-0.8 * span, 0.8 * span, n))
        if n == 2 and abs(offsets[1] - offsets[0]) < 0.3 * extent:
            offsets = [offsets[0]]                   # two joints this close are not physical: fall back to one (actual count is recorded)
    theta["offsets_px"] = [round(v, 3) for v in offsets]
    theta["n_joints"] = len(offsets)
    return theta


def render(image: np.ndarray, theta: dict, ctx: Context, domain: dict) -> OperatorResult:
    check_material(NAME, domain["materials"], ctx)
    x = as_rgb_float(image)
    h, w = x.shape[:2]
    width, dark, lip = theta["width_px"], theta["groove_darkness"], theta["lip_contrast"]
    band = max(1.0, width / 4.0)
    rng = np.random.default_rng(theta["seed"])
    gain = np.ones((h, w), np.float32)
    omega = np.zeros((h, w), bool)
    centerline = np.zeros((h, w), np.uint8)
    a = np.deg2rad(theta["angle_deg"])
    normal = np.array([-np.sin(a), np.cos(a)])
    light = np.array([np.cos(np.deg2rad(theta["light_azimuth_deg"])),
                      np.sin(np.deg2rad(theta["light_azimuth_deg"]))])
    lit_sign = 1.0 if float(normal @ light) < 0 else -1.0  # light comes from the −normal side → the +s lip is lit
    for offset in theta["offsets_px"]:
        s, t = line_coordinates((h, w), theta["angle_deg"], offset)
        t_index = np.clip(np.round(t - t.min()).astype(int), 0, None)
        jitter = smooth_noise_1d(int(t_index.max()) + 1, 6.0, rng) * theta["edge_jitter_px"]
        s = s - jitter[t_index]
        u = np.abs(s) / (width / 2.0)
        floor = np.clip(1.0 - u ** 4, 0.0, 1.0)                        # groove floor profile
        wall_side = np.where(s * lit_sign > 0, 0.0, 1.0)               # shaded wall
        g = 1.0 - dark * floor - lip * wall_side * floor
        lip_zone = (u >= 1.0) & (np.abs(s) < width / 2.0 + band)
        ramp = 1.0 - (np.abs(s) - width / 2.0) / band
        g = np.where(lip_zone, 1.0 + np.sign(s * lit_sign) * lip * np.clip(ramp, 0, 1), g)
        zone = np.abs(s) < width / 2.0 + band + 1.0
        gain = np.where(zone, gain * g, gain)
        omega |= zone
        centerline |= (np.abs(s) <= 0.5 + 1e-6).astype(np.uint8)
    out = finalize(x, x * gain[..., None], omega)
    measured = {"n_joints": len(theta["offsets_px"]), "support_fraction": round(float(omega.mean()), 5)}
    return OperatorResult(out, omega, dict(theta), "S_pre", centerline if omega.any() else None, measured)
