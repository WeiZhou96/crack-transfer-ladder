"""experiment_program_v1 E2/E3: real confuser strata and synthetic paired response, at the sealed thresholds.

stage dacl10k : label-negative dacl10k images (no Crack/ACrack polygon; train+validation annotations). Images are
                resized (INTER_AREA) so the long side is 1280 px (the modal native size; keeps 4032x3024 frames
                within GPU memory), polygons scaled alike. A filtered FP component is associated with class z if it
                overlaps dilate(polygon_z, 8 px). Per image: Mpx, FP components, per-class associated components,
                per-class dilated area, class-free area and FP components touching no class.
stage hardproxy: PaveDistress Patch/Pothole proxy_test frames, cut into the same non-overlapping 512 grid as the
                background-test leg (partial edges dropped), inverted (R18). Per frame: Mpx, FP components, fired.
stage quads   : v4 quadruplets x00/x01 (no crack in either), laser layers inverted (R18); per group FP components
                on x00 and x01 plus the operator intensity scalar from the spec. Training-exposed for B10/B11/ablations.
All stages: native_r95 (sealed) and fixed 0.5; same inference as eval_students_v1. No threshold or subset choice.
usage: python eval_strata_v1.py dacl10k|hardproxy|quads --models models.json --out DIR --gpu G [--only NAME]
"""
import os
import argparse
import csv
import glob
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["dacl10k", "hardproxy", "quads"])
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--only", default=None)
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
C, D = R, Path(os.environ.get("CRACKCF_DATA", "data"))
sys.path[:0] = [str(C / "poc")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from common import binarize_prediction, filter_components  # noqa: E402

cv2.setNumThreads(2)
PREREG = yaml.safe_load(open(C / "configs/prereg_v6.yaml", encoding="utf-8"))
FILT = PREREG["metrics"]["component_filter_px"]
MIN_AREA, MIN_SKEL, LARGE = int(FILT["min_area"]), int(FILT["min_skel"]), int(FILT["large_area"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DACL = D / "raw/dacl10k/extracted/dev/dacl10k_v2_devphase"
LONG_SIDE, ASSOC_DIL = 1280, 8
out = Path(args.out)
E = Path(os.environ.get("CRACKCF_DATA", "data")) / "students_eval_v1_20261028"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predictor(spec):
    import segmentation_models_pytorch as smp
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(f"cuda:{args.gpu}")
    net = smp.UPerNet("tu-convnext_tiny", encoder_weights=None, classes=1).to(dev)
    net.load_state_dict(torch.load(os.path.expandvars(spec["ckpt"]), map_location=dev, weights_only=True))
    net.eval()

    def run(img):
        h, w = img.shape[:2]
        H, W = max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32)
        x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
        with torch.inference_mode():
            p = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)))[0, 0, :h, :w]
        return p.float().cpu().numpy()
    return run


def fp_labels(prob, t):
    filt, comps = filter_components(binarize_prediction(prob, t), MIN_AREA, MIN_SKEL, LARGE)
    n, lab = cv2.connectedComponents(filt, connectivity=8)
    return n - 1, lab, filt


# ------------------------------------------------------------------ dacl10k
def dacl_items():
    items = []
    for split in ["train", "validation"]:
        for f in sorted(glob.glob(str(DACL / "annotations" / split / "*.json"))):
            a = json.load(open(f, encoding="utf-8"))
            labels = {s["label"] for s in a.get("shapes", [])}
            if labels & {"Crack", "ACrack"}:
                continue
            items.append({"ann": f, "img": str(DACL / a["imagePath"]), "group_id": "dacl10k::" + Path(f).stem})
    return items


def dacl_load(item):
    a = json.load(open(item["ann"], encoding="utf-8"))
    with Image.open(item["img"]) as im:
        img = np.asarray(im.convert("RGB"), np.uint8)
    h, w = img.shape[:2]
    s = LONG_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    hh, ww = img.shape[:2]
    masks = {}
    for sh in a.get("shapes", []):
        pts = np.round(np.array(sh["points"], np.float64) * s * (np.array([ww / (w * s), hh / (h * s)]))).astype(np.int32)
        m = masks.setdefault(sh["label"], np.zeros((hh, ww), np.uint8))
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
        elif len(pts) == 2:
            cv2.line(m, tuple(pts[0]), tuple(pts[1]), 1, 3)
    return img, masks


def dacl_job(a):
    prob, item, thr = a
    _, masks = dacl_load(item)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ASSOC_DIL + 1, 2 * ASSOC_DIL + 1))
    dil = {z: cv2.dilate(m, k) > 0 for z, m in masks.items()}
    joint = None
    for z in ("EJoint", "JTape"):
        if z in dil:
            joint = dil[z] if joint is None else (joint | dil[z])
    if joint is not None:
        dil["joint"] = joint
    any_cls = np.zeros(prob.shape, bool)
    for z, m in dil.items():
        if z != "joint":
            any_cls |= m
    res = {"image": Path(item["img"]).name, "group_id": item["group_id"], "mpx": prob.size / 1e6,
           "classfree_mpx": float((~any_cls).sum()) / 1e6}
    for z, m in dil.items():
        res[f"area_mpx__{z}"] = float(m.sum()) / 1e6
    for name, t in thr.items():
        n, lab, _ = fp_labels(prob, t)
        res[f"{name}__fp"] = n
        touched = np.zeros(n + 1, bool)
        for z, m in dil.items():
            ids = np.unique(lab[m & (lab > 0)])
            res[f"{name}__assoc__{z}"] = int(len(ids))
            if z != "joint":
                touched[ids] = True
        res[f"{name}__unassoc"] = int(n - touched[1:].sum())
    return res


# ------------------------------------------------------------------ hard proxy
def hp_items():
    items = []
    for l in open(D / "registry/registry.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["source_dataset"] == "pavedistress_hard_proxy":
            items.append({"image_id": r["image_id"], "path": r["path"], "group_id": r["group_id"],
                          "label": r["image_id"].split("/")[1]})
    return sorted(items, key=lambda x: x["image_id"])


def hp_job(a):
    probs, meta, thr = a
    res = dict(meta)
    res["mpx"] = sum(p.size for p in probs) / 1e6
    res["tiles"] = len(probs)
    for name, t in thr.items():
        fp = fired = 0
        for p in probs:
            n, _, _ = fp_labels(p, t)
            fp += n
            fired += int((p >= t).any())
        res[f"{name}__fp"] = fp
        res[f"{name}__tiles_fired"] = fired
    return res


# ------------------------------------------------------------------ quadruplets
INTENSITY = {"shadow": "transmittance_k", "joint": None, "repair_boundary": None}


def quad_job(a):
    p00, p01, meta, thr = a
    res = dict(meta)
    res["mpx"] = p00.size / 1e6
    for name, t in thr.items():
        res[f"{name}__fp_x00"] = fp_labels(p00, t)[0]
        res[f"{name}__fp_x01"] = fp_labels(p01, t)[0]
    return res


models = json.load(open(args.models, encoding="utf-8"))
sealed = json.loads((E / "dev_thresholds_sealed.json").read_text(encoding="utf-8"))
names = [args.only] if args.only else list(models)
(out / args.stage).mkdir(parents=True, exist_ok=True)
pool = get_context("fork").Pool(args.workers)
for name in names:
    spec = models[name]
    s = sealed["models"].get(name) or json.loads((E / "dev" / name / "threshold.json").read_text())
    if s["ckpt_sha256"] != sha(os.path.expandvars(spec["ckpt"])):
        raise SystemExit("checkpoint differs from dev stage: " + name)
    thr = {"native_r95": s["native_r95"], "fixed_0.5": 0.5}
    dst = out / args.stage / f"{name}.csv"
    if dst.exists():
        print("skip", name, flush=True)
        continue
    run = predictor(spec)
    t0 = time.time()
    jobs = []
    if args.stage == "dacl10k":
        for it in dacl_items():
            img, _ = dacl_load(it)
            jobs.append(pool.apply_async(dacl_job, ((run(img), it, thr),)))
            if len(jobs) > 600:
                jobs[-600].wait()
    elif args.stage == "hardproxy":
        for it in hp_items():
            with Image.open(it["path"]) as im:
                img = 255 - np.asarray(im.convert("RGB"), np.uint8)
            h, w = img.shape[:2]
            probs = [run(img[y:y + 512, x:x + 512]) for y in range(0, h - 511, 512) for x in range(0, w - 511, 512)]
            jobs.append(pool.apply_async(hp_job, ((probs, it, thr),)))
            if len(jobs) > 60:
                jobs[-60].wait()
    else:
        qd = D / "quadruplets/v4"
        for l in open(qd / "index.jsonl", encoding="utf-8"):
            g = json.loads(l)
            gd = qd / g["operator"] / g["group_id"]
            man = json.loads((gd / "manifest.json").read_text(encoding="utf-8"))
            inv = man["spec"]["background"]["domain"] == "laser_gray"
            ims = []
            for k in ("x00", "x01"):
                with Image.open(gd / f"{k}.png") as im:
                    a = np.asarray(im.convert("RGB"), np.uint8)
                ims.append(255 - a if inv else a)
            th = man["spec"]["theta_z"]
            meta = {"group_id": g["group_id"], "operator": g["operator"], "stratum": g["stratum"],
                    "r_bin": man["spec"].get("r_realized_bin"),
                    "intensity": th.get(INTENSITY.get(g["operator"]) or "", None),
                    "omega_z_frac": man["measured"]["omega_z_frac"], "theta_z": json.dumps(th)}
            jobs.append(pool.apply_async(quad_job, ((run(ims[0]), run(ims[1]), meta, thr),)))
    res = [j.get() for j in jobs]
    keys = sorted({k for r in res for k in r}, key=lambda k: (k.count("__"), k))
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(res)
    print(json.dumps({"model": name, "stage": args.stage, "n": len(res), "s": round(time.time() - t0),
                      "evaluator_sha256": sha(__file__)}), flush=True)
pool.close()
