"""R20 (prereg_v6): re-solve contrast_alpha of attenuated v3 groups around the stratum tau_delta.

v3 sampled the attenuated half as alpha ~ log-uniform[0.02, 1] without reference to tau_delta, leaving 1157/3000 x11
layers without positive labels at the point thresholds. Here every v3 spec is kept (seeds, geometry, theta, nu); only
groups with contrast_alpha != 1 get a new alpha, solved by bisection on log(alpha) so that the median x11 skeleton
dc (8-bit, as written to disk and seen in training) hits a target drawn per group:
  70% U[tau, 2 tau]  (hard-but-visible band)   30% U[0.5 tau, tau)  (kept below tau: ignore stays disclosed)
alpha <= 1 always (no contrast exaggeration); if alpha = 1 cannot reach the target the group keeps alpha = 1 and is
marked capped. Natural groups (alpha = 1) are copied unchanged. Output: <out>/specs.jsonl + run_meta.json, to be
rendered with render_quadruplets_v1.py render (which reuses an existing specs.jsonl).
usage: python alpha_stratify_v1.py <code_dir> <data_root> <src_quad_dir> <out_dir> --prereg PREREG [--workers W]
       [--limit N]
"""
import argparse
import copy
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("src_quad_dir")
ap.add_argument("out_dir")
ap.add_argument("--prereg", required=True)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--limit", type=int, default=None, help="dry run on the first N attenuated groups")
args = ap.parse_args()
sys.path.insert(0, args.code_dir)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from render import quadruplet as Q  # noqa: E402
from render.operators import load_domains  # noqa: E402

cv2.setNumThreads(1)
OPS = ["shadow", "joint", "repair_boundary"]
SEED = 20261028
P_VISIBLE = 0.7
LOG_LO = float(np.log(1e-3))
STEPS = 12
src, out = Path(args.src_quad_dir), Path(args.out_dir)
meta = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
domains = load_domains(Path(args.code_dir) / "configs/operators.yaml")
TAU = yaml.safe_load(open(args.prereg, encoding="utf-8"))["v6_execution_decisions"]["tau_delta_by_stratum"]
RES = None


def resources():
    global RES
    if RES is None:
        RES = Q.load_resources(Path(args.data_root), domains, Path(meta["pool_csv"]), Path(meta["watermark_csv"]))
    return RES


def dc11_median(spec, alpha):
    s = copy.deepcopy(spec)
    s["crack"]["contrast_alpha"] = float(alpha)
    return float(np.median(Q.render_group(s, resources())["dc11"]))


def solve(spec):
    op, gid = spec["operator"], spec["group_id"]
    tau = float(TAU[spec["stratum"]])
    rng = np.random.default_rng(np.random.SeedSequence([SEED, OPS.index(op), int(gid.rsplit("_", 1)[1])]))
    visible = bool(rng.random() < P_VISIBLE)
    target = float(rng.uniform(tau, 2 * tau) if visible else rng.uniform(0.5 * tau, tau))
    m_nat = dc11_median(spec, 1.0)
    capped = floored = False
    if m_nat <= target:
        alpha, achieved, capped = 1.0, m_nat, True
    else:
        lo, hi, m_hi = LOG_LO, 0.0, m_nat
        m_lo = dc11_median(spec, np.exp(lo))
        if m_lo >= target:
            hi, m_hi, floored = lo, m_lo, True
        else:
            for _ in range(STEPS):
                mid = 0.5 * (lo + hi)
                m = dc11_median(spec, np.exp(mid))
                if m >= target:
                    hi, m_hi = mid, m
                else:
                    lo = mid
        alpha, achieved = (1.0 if hi == 0.0 else float(np.exp(hi))), m_hi
    s = copy.deepcopy(spec)
    s["crack"]["contrast_alpha"] = alpha
    s["crack"]["alpha_stratify"] = {
        "rule": "prereg_v6 R20", "v3_alpha": spec["crack"]["contrast_alpha"], "tau_delta": tau,
        "band": "hard_visible" if visible else "sub_threshold", "target_dc11_median": round(target, 6),
        "achieved_dc11_median": round(achieved, 6), "natural_dc11_median": round(m_nat, 6),
        "capped": capped, "floored": floored}
    return s


specs = [json.loads(l) for l in open(src / "specs.jsonl", encoding="utf-8")]
attenuated = [i for i, s in enumerate(specs) if s["crack"].get("contrast_alpha", 1.0) != 1.0]
if args.limit:
    attenuated = attenuated[:args.limit]
out.mkdir(parents=True, exist_ok=False)
t0 = time.time()
with Pool(args.workers) as p:
    solved = p.map(solve, [specs[i] for i in attenuated], chunksize=2)
new = list(specs) if not args.limit else [specs[i] for i in attenuated]
if not args.limit:
    for i, s in zip(attenuated, solved):
        new[i] = s
else:
    new = solved
with open(out / "specs.jsonl", "w", encoding="utf-8") as f:
    for s in new:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
info = [s["crack"]["alpha_stratify"] for s in solved]
err = [abs(a["achieved_dc11_median"] - a["target_dc11_median"]) for a in info if not a["capped"] and not a["floored"]]
summary = {
    "rule": "prereg_v6 R20", "seed": SEED, "p_visible": P_VISIBLE, "log_alpha_floor": LOG_LO, "steps": STEPS,
    "tau_delta_by_stratum": TAU, "groups_total": len(new), "attenuated_resolved": len(solved),
    "natural_unchanged": len(new) - len(solved), "solve_s": round(time.time() - t0, 1),
    "by_stratum_band": Counter(f"{s['stratum']}|{s['crack']['alpha_stratify']['band']}" for s in solved),
    "capped_by_stratum": Counter(s["stratum"] for s in solved if s["crack"]["alpha_stratify"]["capped"]),
    "floored": sum(a["floored"] for a in info),
    "abs_target_error_median": float(np.median(err)) if err else None,
    "abs_target_error_p95": float(np.percentile(err, 95)) if err else None,
    "source_specs": str(src / "specs.jsonl"), "source_specs_sha256": Q.sha256_bytes((src / "specs.jsonl").read_bytes()),
}
(out / "alpha_stratify_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
(out / "run_meta.json").write_text(json.dumps({**meta, "derived_from": str(src), "alpha_stratify": "prereg_v6 R20",
                                               "alpha_stratify_tool": "render/tools/alpha_stratify_v1.py"},
                                              indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=1, ensure_ascii=False))
