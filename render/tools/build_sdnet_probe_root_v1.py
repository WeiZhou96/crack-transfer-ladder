"""Addendum 5 part A: background pool and data root for the scene-novel synthetic probe.

Backgrounds = SDNET2018 P/UP images of the double-annotator clean staged-negative leg (never in any training set).
Each 256x256 image -> four 128x128 crops (rgb_concrete tier of background_pool_v3), same crop QC as background_pool_v3.
The data root holds registry/registry.jsonl (original registry + one entry per SDNET image) and a symlink to the original
mask_library, so render_quadruplets_v1.py runs unchanged. Nothing under the original data root is modified.
usage: python build_sdnet_probe_root_v1.py <data_root> <staged_negative_csv> <out_root> <pool_dir>
"""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

root, neg_csv, out_root, pool_dir = map(Path, sys.argv[1:5])
out_root.mkdir(parents=True, exist_ok=False)
pool_dir.mkdir(parents=True, exist_ok=False)
(out_root / "registry").mkdir()
os.symlink(root / "mask_library", out_root / "mask_library")
SIDE = 128

rows, reg_new, skipped = [], [], Counter()
for r in csv.DictReader(open(neg_csv, encoding="utf-8")):
    iid = "sdnet2018/" + r["image_id"]
    img = np.asarray(Image.open(r["path"]).convert("RGB"))
    H, W = img.shape[:2]
    reg_new.append({"image_id": iid, "source_dataset": "sdnet2018", "pool": "probe_background", "material": "concrete",
                    "crack_annotation": "image-level", "path": r["path"], "mask_path": None, "width": W, "height": H})
    for y in range(0, H - SIDE + 1, SIDE):
        for x in range(0, W - SIDE + 1, SIDE):
            crop = img[y:y + SIDE, x:x + SIDE]
            sat = ((crop >= 250).all(axis=2) | (crop <= 5).all(axis=2)).mean()
            g = crop.mean(axis=2)
            if sat > 0.01 or g.std() < 2:
                skipped["crop_qc"] += 1
                continue
            if (g >= max(178.0, float(np.median(g)) + 77.0)).mean() > 0.02:
                skipped["crop_bright_structure"] += 1
                continue
            rows.append({"crop_id": f"{iid}#y{y}_x{x}_s{SIDE}", "image_id": iid, "source_dataset": "sdnet2018",
                         "group_id": r["group_id"], "domain": "rgb_photo", "material": "concrete", "surface": "",
                         "x": x, "y": y, "side": SIDE, "tier": "interim_small", "admission": "probe_only",
                         "screened": "double_annotator_clean_image"})

with open(out_root / "registry/registry.jsonl", "w", encoding="utf-8") as f:
    for line in open(root / "registry/registry.jsonl", encoding="utf-8"):
        f.write(line if line.endswith("\n") else line + "\n")
    for e in reg_new:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(pool_dir / "candidates.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summ = {"images": len(reg_new), "groups": len({r["group_id"] for r in rows}), "crops": len(rows),
        "skipped": dict(skipped), "side": SIDE, "source": str(neg_csv)}
(pool_dir / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
print(json.dumps(summ, indent=1))
