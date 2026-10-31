"""experiment_program_v1_addendum1 confirmation leg: SDNET2018 pavement, never used by any student before.

Lists (ed1_inputs_sdnet_20261028): sdnet_staged_negative (122 double-reviewed crack-free UP), sdnet_UP_all
(21,726 UP, noisy image labels), sdnet_CP (2,608 cracked). 256x256 RGB, no inversion. Thresholds: sealed dev
thresholds (students_eval_v1_20261028 or a later sealed file passed by --sealed). Per image at native_r95 and 0.5:
negative accounts (filtered FP components, PhantomLen, raw FP area, fired) for every list; for CP the same
numbers are read as image-level detection (any filtered component). Same inference as eval_students_v1.
usage: python eval_sdnet_v1.py --models models.json --sealed sealed.json --out DIR --gpu G [--only NAME]
"""
import os
import argparse
import csv
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--sealed", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--only", default=None)
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
C, D = R, Path(os.environ.get("CRACKCF_DATA", "data"))
sys.path[:0] = [str(C / "poc")]
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from common import binarize_prediction, negative_image_accounts  # noqa: E402

PREREG = yaml.safe_load(open(C / "configs/prereg_v6.yaml", encoding="utf-8"))
FILT = PREREG["metrics"]["component_filter_px"]
MIN_AREA, MIN_SKEL = int(FILT["min_area"]), int(FILT["min_skel"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LISTS = ["sdnet_staged_negative", "sdnet_CP", "sdnet_UP_all"]
I = D / "ed1_inputs_sdnet_20261028"


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

    def run(path):
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), np.uint8)
        h, w = img.shape[:2]
        H, W = max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32)
        x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
        with torch.inference_mode():
            p = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)))[0, 0, :h, :w]
        return p.float().cpu().numpy()
    return run


def job(a):
    prob, meta, thr = a
    res = dict(meta)
    for name, t in thr.items():
        for k, v in negative_image_accounts(binarize_prediction(prob, t), MIN_AREA, MIN_SKEL).items():
            res[f"{name}__{k}"] = v
    return res


models = json.load(open(args.models, encoding="utf-8"))
sealed = json.load(open(args.sealed, encoding="utf-8"))["models"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pool = get_context("fork").Pool(args.workers)
for name in ([args.only] if args.only else list(models)):
    spec = models[name]
    if sealed[name]["ckpt_sha256"] != sha(os.path.expandvars(spec["ckpt"])):
        raise SystemExit("checkpoint differs from sealed dev stage: " + name)
    thr = {"native_r95": sealed[name]["native_r95"], "fixed_0.5": 0.5}
    mdir = out / name
    if (mdir / "DONE").exists():
        print("skip", name, flush=True)
        continue
    mdir.mkdir(parents=True, exist_ok=True)
    run = predictor(spec)
    t0 = time.time()
    for lst in LISTS:
        with open(I / f"{lst}.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        jobs = []
        for r in rows:
            jobs.append(pool.apply_async(job, ((run(r["path"]), {"image_id": r["image_id"], "group_id": r["group_id"]},
                                                thr),)))
            if len(jobs) > 3000:
                jobs[-3000].wait()
        res = [j.get() for j in jobs]
        with open(mdir / f"{lst}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
    (mdir / "DONE").write_text(json.dumps({"model": name, "thresholds": thr, "seconds": round(time.time() - t0),
                                           "evaluator_sha256": sha(__file__)}), encoding="utf-8")
    print(json.dumps({"model": name, "s": round(time.time() - t0)}), flush=True)
pool.close()
