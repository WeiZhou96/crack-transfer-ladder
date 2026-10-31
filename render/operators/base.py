# -*- coding: utf-8 -*-
"""Shared operator utilities: result contract, typed preconditions, θ sampling, sRGB↔linear, constructive guarantee of the support Ω_Z.

Contract (including the Q2 check):
  - Input and output are float32 RGB in [0,1], HxWx3; operators do not change the image size.
  - Every render returns a declared support omega (bool HxW); `finalize` restores pixels outside omega bit-exactly
    to the input, so "pixel differences in the layers before A_ν lie strictly inside Ω_Z" holds by construction
    (the strict-zero Q2 check is done at this layer).
  - Re-rendering with the same θ (including seed) is pixel-identical; θ contains only JSON-serialisable scalars and is
    saved with the manifest.
  - All three operators are label-preserving: y is unchanged and is coupled through the visibility term V (visibility.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


class PreconditionError(ValueError):
    """Operator precondition (material / event order) not met: the renderer must reject when assembling σ, not silently degrade."""


@dataclass
class Context:
    material: str                          # asphalt / concrete (background material, from the registry or per-image labels)
    gsd_mm_per_pixel: float | None = None  # None when no reliable GSD is available; quantities are then in pixels
    has_crack: bool = False                # needed only by S_post operators; the three operators here do not use it


@dataclass
class OperatorResult:
    image: np.ndarray                      # float32 HxWx3 [0,1]
    omega: np.ndarray                      # bool HxW, declared support Ω_Z
    theta: dict                            # θ actually used (including seed and derived quantities)
    slot: str
    centerline: np.ndarray | None = None   # centreline of a linear distractor (uint8 0/1), for measuring r
    measured: dict = field(default_factory=dict)  # post-render measurements (coverage, perimeter, etc.)


def check_material(name: str, allowed: list[str], ctx: Context) -> None:
    if ctx.material not in allowed:
        raise PreconditionError(f"{name}: background material {ctx.material!r} not in allowed set {allowed}")


def uniform(rng: np.random.Generator, spec: dict) -> float:
    lo, hi = spec["range"]
    return float(rng.uniform(lo, hi))


def choice(rng: np.random.Generator, spec: dict):
    values, probs = spec["values"], spec.get("probs")
    index = int(rng.choice(len(values), p=probs))
    value = values[index]
    return value.item() if hasattr(value, "item") else value


def in_domain(value: float, spec: dict, tol: float = 1e-9) -> bool:
    lo, hi = spec["range"]
    return lo - tol <= value <= hi + tol


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def as_rgb_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxW or HxWx3 image, got {image.shape}")
    return image


def smooth_noise(shape: tuple[int, int], scale_px: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-std low-frequency noise field (Gaussian-filtered white noise), used for blobs and edge undulation."""
    sigma = max(float(scale_px), 0.5)
    field_ = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma,
                              borderType=cv2.BORDER_REFLECT)
    std = float(field_.std())
    return field_ / std if std > 0 else field_


def smooth_noise_1d(length: int, scale_px: float, rng: np.random.Generator) -> np.ndarray:
    return smooth_noise((1, length), scale_px, rng)[0] if length > 1 else np.zeros(length, np.float32)


def gaussian_soften(mask: np.ndarray, transition_px: float, cutoff: float = 1e-3) -> np.ndarray:
    """Binary → soft mask: transition_px is the 10%-90% transition width (Gaussian edge ≈ 2.563σ); values below cutoff are zeroed to bound the support."""
    sigma = max(transition_px / 2.563, 0.3)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    soft[soft < cutoff] = 0.0
    return np.clip(soft, 0.0, 1.0)


def finalize(original: np.ndarray, rendered: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Restore the input bit-exactly outside Ω_Z; clip to [0,1] inside Ω_Z."""
    out = np.where(omega[..., None], np.clip(rendered, 0.0, 1.0), original).astype(np.float32)
    return out


def line_coordinates(shape: tuple[int, int], angle_deg: float, offset_px: float) -> tuple[np.ndarray, np.ndarray]:
    """With the image centre as origin, return (s, t): s is the signed (normal) distance to the line, t the coordinate along it.
    The line direction is angle_deg (relative to the x axis); offset_px is the line's displacement from the centre along the normal."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx - (w - 1) / 2.0, yy - (h - 1) / 2.0
    a = np.deg2rad(angle_deg)
    direction = np.array([np.cos(a), np.sin(a)], np.float32)
    normal = np.array([-np.sin(a), np.cos(a)], np.float32)
    s = x * normal[0] + y * normal[1] - offset_px
    t = x * direction[0] + y * direction[1]
    return s.astype(np.float32), t.astype(np.float32)


def centerline_from_distance(s: np.ndarray) -> np.ndarray:
    """One-pixel-wide centreline where |s| is minimal (local minimum along the normal), used for measuring r."""
    a = np.abs(s)
    return (a <= 0.5 + 1e-6).astype(np.uint8)


def skeletonize_mask(mask):
    """Same as poc/common.skeletonize_mask (skimage skeletonisation → uint8 0/1); bundled so the rendering package does not depend on the frozen evaluation code."""
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool)).astype(np.uint8)
