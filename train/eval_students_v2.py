"""v2 = v1 with arch "upernet_resnet50" accepted (addendum 3 second backbone); logic unchanged.
P4/P5 (prereg_v6 R21): one evaluation path for every student model.

stage dev  : full-image inference on the native validation set (split_assignments.v3: pool=train_positive, split=val);
             per-model R95 threshold by the frozen s06 component-recall search; thresholds.json is written and sealed.
stage test : refuses to run unless the sealed thresholds of every listed model exist; one pass over the three
             negative legs and the CrackSC+NHA12D positive axis at native R95 and fixed 0.5. Per-image accounts are
             written for bootstrap; probabilities are not stored. No threshold, model or list is chosen here.
Inference matches student_v1.validate / export_float32_v2: RGB, ImageNet norm, reflect pad to /32, sigmoid, float32.
R18: PaveDistress-sourced inputs are read from the pre-inverted registry copies (pred_registry_v2_full.jsonl).
usage: python eval_students_v1.py dev|test --models models.json --out DIR --gpu G [--only NAME] [--workers W]
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
ap.add_argument("stage", choices=["dev", "test"])
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--only", default=None)
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
C, D = R, Path(os.environ.get("CRACKCF_DATA", "data"))
sys.path[:0] = [str(C / "poc")]
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from common import binarize_prediction, component_recall, load_binary_mask, negative_image_accounts  # noqa: E402
from s06_ed1_public import centerline_metrics, select_t_proxy_r95  # noqa: E402

PREREG = yaml.safe_load(open(C / "configs/prereg_v6.yaml", encoding="utf-8"))
FILT, DET = PREREG["metrics"]["component_filter_px"], PREREG["metrics"]["detected_criterion"]
MIN_AREA, MIN_SKEL = int(FILT["min_area"]), int(FILT["min_skel"])
DIL, COVER = int(DET["dilate_px"]), float(DET["skeleton_cover"])
INV = set(yaml.safe_load(open(C / "configs/e1_arms.yaml", encoding="utf-8"))["preprocess"]["r18_invert_sources"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
I2 = D / "ed1_inputs_v2"
LEGS = {"bcl_noise": I2 / "neg_bcl_noise.csv", "pavedistress_background_test": I2 / "neg_pavedistress_test.csv",
        "uav1388_background": I2 / "neg_uav1388_background.csv"}
POS = I2 / "pos_cracksc_nha12d.csv"
out = Path(args.out)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def native_val():
    reg = {json.loads(l)["image_id"]: json.loads(l) for l in open(D / "registry/registry.jsonl", encoding="utf-8")}
    vals = []
    for r in rows(D / "splits/split_assignments.v3.csv"):
        if r["pool"] == "train_positive" and r["split"] == "val":
            g = reg[r["image_id"]]
            vals.append({"image_id": r["image_id"], "group_id": r["group_id"], "path": g["path"],
                         "mask_path": g["mask_path"], "invert": g["source_dataset"] in INV,
                         "source_dataset": g["source_dataset"]})
    return vals


def build(spec):
    import segmentation_models_pytorch as smp
    arch = spec.get("arch", "upernet_convnext_tiny")
    if arch == "upernet_convnext_tiny":
        return smp.UPerNet("tu-convnext_tiny", encoder_weights=None, classes=1)
    if arch == "upernet_resnet50":
        return smp.UPerNet("resnet50", encoder_weights=None, classes=1)
    raise SystemExit("unknown arch " + arch)


def predictor(spec):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(f"cuda:{args.gpu}")
    net = build(spec).to(dev)
    state = torch.load(os.path.expandvars(spec["ckpt"]), map_location=dev, weights_only=True)
    net.load_state_dict(state)
    net.eval()

    def run(path, invert):
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), np.uint8)
        if invert:
            img = 255 - img
        h, w = img.shape[:2]
        H, W = max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32)
        x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
        with torch.inference_mode():
            p = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)))[0, 0, :h, :w]
        return p.float().cpu().numpy()
    return run


def neg_job(a):
    prob, thresholds, meta = a
    res = dict(meta)
    for name, t in thresholds.items():
        acc = negative_image_accounts(binarize_prediction(prob, t), MIN_AREA, MIN_SKEL)
        for k, v in acc.items():
            res[f"{name}__{k}"] = v
    return res


def pos_job(a):
    prob, gt_path, thresholds, meta = a
    gt = load_binary_mask(gt_path)
    res = dict(meta)
    for name, t in thresholds.items():
        pred = binarize_prediction(prob, t)
        cr = component_recall(gt, pred, dilate_px=DIL, cover=COVER, min_area_px=MIN_AREA, min_skel_px=MIN_SKEL)
        cld, cli = centerline_metrics(pred, gt, 4)
        res.update({f"{name}__n_hit": cr["n_hit"], f"{name}__n_gt": cr["n_gt"], f"{name}__cldice": cld,
                    f"{name}__cliou4": cli})
    return res


models = json.load(open(args.models, encoding="utf-8"))
names = [args.only] if args.only else list(models)
out.mkdir(parents=True, exist_ok=True)

if args.stage == "dev":
    vals = native_val()
    for name in names:
        spec = models[name]
        mdir = out / "dev" / name
        if (mdir / "threshold.json").exists():
            print("skip", name, flush=True)
            continue
        mdir.mkdir(parents=True, exist_ok=True)
        ck = sha(os.path.expandvars(spec["ckpt"]))
        run = predictor(spec)
        t0 = time.time()
        pairs = [(v, run(v["path"], v["invert"]), load_binary_mask(v["mask_path"])) for v in vals]
        t_r95, rec = select_t_proxy_r95(pairs, 0.95, MIN_AREA, MIN_SKEL, DIL, COVER)
        per_source = {}
        for t_name, t in {"native_r95": t_r95, "fixed_0.5": 0.5}.items():
            for v, p, gt in pairs:
                cr = component_recall(gt, binarize_prediction(p, t), dilate_px=DIL, cover=COVER,
                                      min_area_px=MIN_AREA, min_skel_px=MIN_SKEL)
                s = per_source.setdefault(t_name, {}).setdefault(v["source_dataset"], [0, 0])
                s[0] += cr["n_hit"]
                s[1] += cr["n_gt"]
        np.savez_compressed(mdir / "native_val_prob_f16.npz",
                            **{v["image_id"].replace("/", "__"): p.astype(np.float16) for v, p, _ in pairs})
        rec_json = {"model": name, "ckpt": os.path.expandvars(spec["ckpt"]), "ckpt_sha256": ck, "native_r95": t_r95,
                    "native_r95_float32_bits": int(np.float32(t_r95).view(np.uint32)),
                    "native_val_component_recall": rec, "n_val_images": len(vals),
                    "per_source_component_hits": per_source, "prereg_v6_sha256": sha(C / "configs/prereg_v6.yaml"),
                    "evaluator_sha256": sha(__file__), "seconds": round(time.time() - t0, 1),
                    "test_images_read": 0}
        (mdir / "threshold.json").write_text(json.dumps(rec_json, indent=1), encoding="utf-8")
        print(json.dumps({k: rec_json[k] for k in ["model", "native_r95", "native_val_component_recall", "seconds"]}),
              flush=True)
else:
    seal = out / "dev_thresholds_sealed.json"
    if not seal.exists():
        raise SystemExit("thresholds are not sealed; run the dev stage for all models and seal first")
    sealed = json.loads(seal.read_text(encoding="utf-8"))
    reg = {json.loads(l)["image_id"]: json.loads(l)["path"]
           for l in open(I2 / "pred_registry_v2_full.jsonl", encoding="utf-8")}
    pool = get_context("fork").Pool(args.workers)
    for name in names:
        spec = models[name]
        if sealed["models"][name]["ckpt_sha256"] != sha(os.path.expandvars(spec["ckpt"])):
            raise SystemExit("checkpoint differs from sealed dev stage: " + name)
        thr = {"native_r95": sealed["models"][name]["native_r95"], "fixed_0.5": 0.5}
        mdir = out / "test" / name
        if (mdir / "DONE").exists():
            print("skip", name, flush=True)
            continue
        mdir.mkdir(parents=True, exist_ok=True)
        run = predictor(spec)
        t0 = time.time()
        for leg, lst in LEGS.items():
            jobs = []
            for r in rows(lst):
                meta = {"image_id": r["image_id"], "group_id": r["group_id"],
                        "bootstrap_group_1km": r["bootstrap_group_1km"],
                        "bootstrap_group_100m": r["bootstrap_group_100m"]}
                jobs.append(pool.apply_async(neg_job, ((run(reg[r["image_id"]], False), thr, meta),)))
                if len(jobs) > 1500:                       # bound queued probability maps
                    jobs[-1500].wait()
            res = [j.get() for j in jobs]
            with open(mdir / f"{leg}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(res[0]))
                w.writeheader()
                w.writerows(res)
            print(json.dumps({"model": name, "leg": leg, "n": len(res), "s": round(time.time() - t0)}), flush=True)
        jobs = []
        for r in rows(POS):
            meta = {"image_id": r["image_id"], "source_dataset": r["source_dataset"], "group_id": r["group_id"]}
            jobs.append(pool.apply_async(pos_job, ((run(reg[r["image_id"]], False), r["mask_path"], thr, meta),)))
        res = [j.get() for j in jobs]
        with open(mdir / "positive_axis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        (mdir / "DONE").write_text(json.dumps({"model": name, "thresholds": thr, "seconds": round(time.time() - t0),
                                               "evaluator_sha256": sha(__file__)}), encoding="utf-8")
        print(json.dumps({"model": name, "done": True, "s": round(time.time() - t0)}), flush=True)
    pool.close()
