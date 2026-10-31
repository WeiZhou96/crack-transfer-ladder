# -*- coding: utf-8 -*-
"""Distractor operator library (shadow / joint / repair_boundary; all label-preserving).

Usage:
    from operators import OPERATORS, load_domains, Context
    domains = load_domains(<configs/operators.yaml>)
    op = OPERATORS["joint"]
    theta = op.sample_theta(rng, ctx, domains["joint"], image.shape[:2])
    result = op.render(image, theta, ctx, domains["joint"])   # -> OperatorResult(image, omega, theta, slot, ...)
Rendering order: S_pre (joint, repair_boundary) → brush P_M → … → L^light (shadow) → A_ν.
Quadruplet assembly, brush caching and manifests belong to render/quadruplet.py, not to this package.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import joint, repair_boundary, shadow
from .base import Context, OperatorResult, PreconditionError

OPERATORS = {m.NAME: m for m in (shadow, joint, repair_boundary)}
SLOT_ORDER = ["S_pre", "P_M", "S_post", "T_mat", "O_occ", "L_light", "A_nu"]


def load_domains(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [name for name in OPERATORS if name not in cfg]
    if missing:
        raise ValueError(f"operators.yaml is missing operators: {missing}")
    return cfg


__all__ = ["OPERATORS", "SLOT_ORDER", "Context", "OperatorResult", "PreconditionError", "load_domains"]
