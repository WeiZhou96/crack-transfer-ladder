"""Batch: sample specs and render quadruplets (3 operators x N groups).

Phase 1 (serial, deterministic): sample all specs from one master seed (per-group SeedSequence) -> specs.jsonl.
Phase 2 (parallel): render each spec -> <out>/<operator>/<group_id>/{x00,x01,x10,x11,M,omega_z,omega_m}.png, dc.npz,
         manifest.json; index.jsonl collects manifests (without spec duplication).
--verify K: re-render K random groups from their manifest specs and compare every PNG sha256 (acceptance:
         pixel-identical re-render).
usage:
  python render_quadruplets_v1.py render <code_dir> <data_root> <pool_csv> <watermark_csv> <out_dir> [--n N] [--seed S] [--workers W]
  python render_quadruplets_v1.py verify <code_dir> <data_root> <pool_csv> <watermark_csv> <out_dir> [--k K]
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["render", "verify"])
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("pool_csv")
ap.add_argument("watermark_csv")
ap.add_argument("out_dir")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--seed", type=int, default=20260928)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--k", type=int, default=30)
args = ap.parse_args()
sys.path.insert(0, args.code_dir)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from render import quadruplet as Q  # noqa: E402
from render.operators import load_domains  # noqa: E402

cv2.setNumThreads(1)
OPS = ["shadow", "joint", "repair_boundary"]
domains = load_domains(Path(args.code_dir) / "configs/operators.yaml")
out = Path(args.out_dir)
RES = None


def resources():
    global RES
    if RES is None:
        RES = Q.load_resources(Path(args.data_root), domains, Path(args.pool_csv), Path(args.watermark_csv))
    return RES


def sample_one(job):
    oi, op, i = job
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, oi, i]))
    return Q.sample_spec(rng, op, i, resources())


def work(spec):
    res = resources()
    d = out / spec["operator"] / spec["group_id"]
    if (d / "manifest.json").exists():
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    try:
        return Q.write_group(d, spec, Q.render_group(spec, res))
    except Exception as e:  # noqa: BLE001
        return {"spec": spec, "error": repr(e)}


if args.mode == "render":
    n = args.n or domains["quadruplet"]["n_per_operator"]
    specs_path = out / "specs.jsonl"
    if specs_path.exists():
        specs = [json.loads(l) for l in open(specs_path, encoding="utf-8")]
        print("reusing", len(specs), "specs")
    else:
        out.mkdir(parents=True, exist_ok=False)
        res = resources()
        t0 = time.time()
        jobs = [(oi, op, i) for oi, op in enumerate(OPS) for i in range(n)]
        with Pool(args.workers) as p:           # one SeedSequence([seed, operator index, group index]) per group: parallelism does not affect determinism
            specs = p.map(sample_one, jobs, chunksize=4)
        print("specs sampled", len(specs), round(time.time() - t0, 1), "s", flush=True)
        with open(specs_path, "w", encoding="utf-8") as f:
            for s in specs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        (out / "run_meta.json").write_text(json.dumps({
            "seed": args.seed, "n_per_operator": n, "operators": OPS, "operators_yaml_version": domains["version"],
            "pool_csv": args.pool_csv, "watermark_csv": args.watermark_csv,
            "pool_sizes": {k: len(v) for k, v in res.pools.items()},
            "library_sizes": {k: len(v) for k, v in res.library.items()}}, indent=1), encoding="utf-8")
    t0 = time.time()
    with Pool(args.workers) as p:
        manifests = p.map(work, specs, chunksize=4)
    errors = [m for m in manifests if "error" in m]
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for m in manifests:
            if "error" not in m:
                f.write(json.dumps({"group_id": m["spec"]["group_id"], "operator": m["spec"]["operator"],
                                    "stratum": m["spec"]["stratum"], "measured": m["measured"]},
                                   ensure_ascii=False) + "\n")
    q2 = max(max(m["measured"]["q2_pre_max_abs_outside"].values()) for m in manifests if "error" not in m)
    summary = {"groups": len(manifests), "errors": len(errors), "render_s": round(time.time() - t0, 1),
               "q2_pre_max_abs_outside_all": q2,
               "by_operator_stratum": Counter(f"{m['spec']['operator']}|{m['spec']['stratum']}" for m in manifests),
               "r_realized": Counter(f"{m['spec']['operator']}|{m['spec'].get('r_realized_bin')}"
                                     for m in manifests if m["spec"].get("r_target")),
               "r_fallback_groups": sum(1 for m in manifests if m["spec"].get("r_fallback"))}
    (out / "render_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    if errors:
        (out / "errors.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in errors), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
else:
    res = resources()
    groups = [json.loads(l)["group_id"] for l in open(out / "index.jsonl", encoding="utf-8")]
    pick = random.Random(args.seed).sample(groups, min(args.k, len(groups)))
    bad = []
    for g in pick:
        op = g.rsplit("_", 1)[0]
        m = json.loads((out / op / g / "manifest.json").read_text(encoding="utf-8"))
        again = Q.rerender_hashes(m["spec"], res)
        diff = [k for k, v in again.items() if m["files_sha256"].get(k) != v]
        if diff:
            bad.append({"group": g, "files": diff})
    result = {"checked": len(pick), "mismatched": bad}
    (out / f"verify_rerender_k{len(pick)}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
