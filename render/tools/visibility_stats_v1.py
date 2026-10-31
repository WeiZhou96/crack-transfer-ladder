"""Label statistics of a quadruplet set under per-stratum tau_delta (R20 acceptance table; no training, no test data).

For every group: labels_from_dc on dc10/dc11 -> empty x11 positive label, visible / ignore crack-pixel fractions.
Aggregated by stratum x operator x alpha band (natural: alpha = 1 in v3; attenuated otherwise) and written as JSON.
usage: python visibility_stats_v1.py <code_dir> <quad_dir> <out_json> --prereg PREREG [--tau-key KEY]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("quad_dir")
ap.add_argument("out_json")
ap.add_argument("--prereg", required=True)
args = ap.parse_args()
sys.path.insert(0, args.code_dir)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from render.quadruplet import labels_from_dc  # noqa: E402

TAU = yaml.safe_load(open(args.prereg, encoding="utf-8"))["v6_execution_decisions"]["tau_delta_by_stratum"]
qd = Path(args.quad_dir)
acc = defaultdict(lambda: {"groups": 0, "empty_x10": 0, "empty_x11": 0, "crack_px": 0, "visible_px_11": 0})
for line in open(qd / "index.jsonl", encoding="utf-8"):
    g = json.loads(line)
    d = qd / g["operator"] / g["group_id"]
    spec = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["spec"]
    tau = float(TAU[spec["stratum"]])
    st = spec["crack"].get("alpha_stratify")
    v3_alpha = st["v3_alpha"] if st else spec["crack"].get("contrast_alpha", 1.0)
    band = "natural" if v3_alpha == 1.0 else "attenuated"
    M = np.asarray(Image.open(d / "M.png")) > 0
    dc = np.load(d / "dc.npz")
    y10, _ = labels_from_dc(M, dc["skel_yx"], dc["dc10"], tau)
    y11, _ = labels_from_dc(M, dc["skel_yx"], dc["dc11"], tau)
    for key in [(spec["stratum"], g["operator"], band), (spec["stratum"], "all", "all"), ("all", "all", "all")]:
        a = acc["|".join(key)]
        a["groups"] += 1
        a["empty_x10"] += int(y10.sum() == 0)
        a["empty_x11"] += int(y11.sum() == 0)
        a["crack_px"] += int(M.sum())
        a["visible_px_11"] += int(y11.sum())
out = {k: {**v, "visible_pixel_fraction_11": round(v["visible_px_11"] / max(v["crack_px"], 1), 4),
           "empty_x11_fraction": round(v["empty_x11"] / max(v["groups"], 1), 4)} for k, v in sorted(acc.items())}
Path(args.out_json).write_text(json.dumps({"quad_dir": str(qd), "tau_delta_by_stratum": TAU, "test_images_read": 0,
                                           "cells": out}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: (v["groups"], v["empty_x11"], v["visible_pixel_fraction_11"]) for k, v in out.items()
                  if k.endswith("all|all")}, ensure_ascii=False))
