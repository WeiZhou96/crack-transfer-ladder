# -*- coding: utf-8 -*-
"""acquisition: acquisition operator A_ν (slot A_nu, applied last, in a near-linear domain; used as a weak augmentation).

Components (each enabled with its own probability; θ records whether it is enabled and its parameters):
  exposure : linear domain × 2^EV;
  motion   : linear PSF (normalised line-segment kernel of length L pixels and direction φ);
  defocus  : uniform disc kernel (circle-of-confusion radius r);
  noise    : shot + read, σ² = a·x + b² (linear domain, per-pixel Gaussian approximation); the noise realisation is set by seed;
  jpeg     : convert to sRGB, then encode at quality Q and decode (last step).
Pairing: the four images of a quadruplet share the same θ (including seed), so the noise field is pixel-identical; the
difference between two images therefore comes only from their difference before A_ν, spread by blur/JPEG into its
neighbourhood (this is what the "final-layer dilation + tolerance" of the Q2 check accounts for). Ω = whole image (global operator).
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import as_rgb_float, linear_to_srgb, srgb_to_linear, uniform

NAME = "acquisition"


def sample_theta(rng: np.random.Generator, domain: dict) -> dict:
    theta = {"operator": NAME, "seed": int(rng.integers(0, 2**31 - 1))}
    for part in ("exposure", "motion", "defocus", "noise", "jpeg"):
        theta[f"{part}_on"] = bool(rng.random() < domain[part]["prob"])
    theta["exposure_ev"] = uniform(rng, domain["exposure"]["ev"]) if theta["exposure_on"] else 0.0
    if theta["motion_on"]:
        theta["motion_len_px"] = uniform(rng, domain["motion"]["len_px"])
        theta["motion_angle_deg"] = float(rng.uniform(0, 180))
    if theta["defocus_on"]:
        theta["defocus_radius_px"] = uniform(rng, domain["defocus"]["radius_px"])
    if theta["noise_on"]:
        theta["shot_gain"] = uniform(rng, domain["noise"]["shot_gain"])
        theta["read_sigma"] = uniform(rng, domain["noise"]["read_sigma"])
    if theta["jpeg_on"]:
        theta["jpeg_q"] = int(round(uniform(rng, domain["jpeg"]["quality"])))
    return theta


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    size = int(np.ceil(length)) | 1
    k = np.zeros((size, size), np.float32)
    c = (size - 1) / 2.0
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, max(int(np.ceil(length)) * 4, 2)):
        x, y = c + t * np.cos(a), c + t * np.sin(a)
        k[int(round(y)), int(round(x))] = 1.0
    return k / k.sum()


def disk_kernel(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx ** 2 + yy ** 2 <= radius ** 2 + 1e-6).astype(np.float32)
    return k / k.sum()


def render(image: np.ndarray, theta: dict) -> np.ndarray:
    x = srgb_to_linear(as_rgb_float(image))
    if theta["exposure_on"]:
        x = x * (2.0 ** theta["exposure_ev"])
    if theta["motion_on"]:
        x = cv2.filter2D(x, -1, motion_kernel(theta["motion_len_px"], theta["motion_angle_deg"]),
                         borderType=cv2.BORDER_REFLECT)
    if theta["defocus_on"]:
        x = cv2.filter2D(x, -1, disk_kernel(theta["defocus_radius_px"]), borderType=cv2.BORDER_REFLECT)
    x = np.clip(x, 0.0, 1.0)
    if theta["noise_on"]:
        rng = np.random.default_rng(theta["seed"])
        z = rng.standard_normal(x.shape[:2]).astype(np.float32)[..., None]   # one noise field shared by all channels (luminance noise)
        sigma = np.sqrt(theta["shot_gain"] * x + theta["read_sigma"] ** 2)
        x = np.clip(x + sigma * z, 0.0, 1.0)
    out = linear_to_srgb(x)
    if theta["jpeg_on"]:
        u8 = (out * 255.0 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, theta["jpeg_q"]])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(out, dtype=np.float32)
