"""In-domain anchor recall summary (post hoc): arm recall = mean over seeds of per-seed component recall on the 147
PaveDistress calibration frames at the native R95 threshold; difference to the backbone's Real-only arm with a paired
frame bootstrap (10000 replicates, seed 0, one frame resample shared by all models).
usage: python anchor_recall_summary_v2.py <anchor_dir> <out.json>
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
OP, NB = "native_r95", 10000
recs = [json.load(open(p, encoding="utf-8")) for p in sorted(src.glob("*.json"))]
lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else v  # noqa: E731
frames = sorted(lit(recs[0]["per_source_image"]))
arms = defaultdict(list)
H, G = {}, {}
for r in recs:
    ps = lit(r["per_source_image"])
    assert sorted(ps) == frames, r["model"]
    H[r["model"]] = np.array([ps[f][OP][0] for f in frames], float)
    G[r["model"]] = np.array([ps[f][OP][1] for f in frames], float)
    arms[r["arm"]].append(r["model"])
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, len(frames), len(frames)), minlength=len(frames)) for _ in range(NB)]).astype(float)


def arm_point(a):
    return float(np.mean([H[m].sum() / G[m].sum() for m in arms[a]]))


def arm_boot(a):
    return np.mean([(W @ H[m]) / (W @ G[m]) for m in arms[a]], axis=0)


res = {}
for a in sorted(arms):
    ref = "R50_B0" if a.startswith("R50_") else "B0"
    d = 100 * (arm_boot(a) - arm_boot(ref))
    res[a] = {"recall": arm_point(a), "ref": arm_point(ref), "diff_pp": 100 * (arm_point(a) - arm_point(ref)),
              "ci_pp": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
              "n_gt": int(G[arms[a][0]].sum()), "frames": len(frames), "seeds": len(arms[a])}
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
for a, v in res.items():
    print(f"{a:18s} {v['recall']:.3f}  {v['diff_pp']:+6.2f} [{v['ci_pp'][0]:+.2f}, {v['ci_pp'][1]:+.2f}]")
