"""Post hoc (independent review, item B2): in-domain crack recall on PaveDistress.

Component recall of each student model on the 3,561 PaveDistress calibration tiles (147 laser frames with sparse
centreline annotation, routes outside every training and test block), at the model's sealed native R95 threshold and
at 0.5. Same inference path as eval_students_v2 (RGB, ImageNet norm, reflect pad to /32, sigmoid, float32; PaveDistress
inverted at load as in training) and the same component-recall rule (prereg_v6 filter and detection criterion).
Calibration tiles only: no test image is read, nothing is tuned; probabilities are not stored.
usage: python anchor_recall_v1.py <out_dir> --gpu G <eval_dir>=<models.json> [...] [--only NAME ...]
"""
import os
import argparse
import csv
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

C = Path(__file__).resolve().parents[1]
D = Path(os.environ.get("CRACKCF_DATA", "data"))
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "poc"))
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from common import binarize_prediction, component_recall, load_binary_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("specs", nargs="+")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()
PREREG = yaml.safe_load(open(C / "configs/prereg_v6.yaml", encoding="utf-8"))
FILT, DET = PREREG["metrics"]["component_filter_px"], PREREG["metrics"]["detected_criterion"]
MIN_AREA, MIN_SKEL, DIL, COVER = int(FILT["min_area"]), int(FILT["min_skel"]), int(DET["dilate_px"]), float(DET["skeleton_cover"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
with open(D / "ed1_inputs_v2/val_anchor.csv", encoding="utf-8-sig") as f:
    TILES = list(csv.DictReader(f))
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

MODELS = {}
for s in args.specs:
    e, m = s.split("=", 1)
    sealed = json.load(open(Path(e) / "dev_thresholds_sealed.json", encoding="utf-8"))["models"]
    for k, v in json.load(open(m, encoding="utf-8")).items():
        if k in sealed:
            MODELS[k] = dict(v, native_r95=sealed[k]["native_r95"])
names = args.only or sorted(MODELS)


def build(spec):
    import segmentation_models_pytorch as smp
    if spec.get("arch", "upernet_convnext_tiny") == "upernet_convnext_tiny":
        return smp.UPerNet("tu-convnext_tiny", encoder_weights=None, classes=1)
    return smp.UPerNet("resnet50", encoder_weights=None, classes=1)


def job(a):
    prob, gt_path, th = a
    gt = load_binary_mask(gt_path)
    res = {}
    for name, t in th.items():
        cr = component_recall(gt, binarize_prediction(prob, t), dilate_px=DIL, cover=COVER, min_area_px=MIN_AREA,
                              min_skel_px=MIN_SKEL)
        res[name] = (cr["n_hit"], cr["n_gt"])
    return res


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = torch.device(f"cuda:{args.gpu}")
    pool = get_context("fork").Pool(12)
    for name in names:
        dst = OUT / f"{name}.json"
        if dst.exists():
            continue
        spec = MODELS[name]
        net = build(spec).to(dev)
        net.load_state_dict(torch.load(os.path.expandvars(spec["ckpt"]), map_location=dev, weights_only=True)); net.eval()
        th = {"native_r95": spec["native_r95"], "fixed_0.5": 0.5}
        t0, per_img, batch = time.time(), {}, []

        def flush():
            for (row, _), r in zip(batch, pool.map(job, [(p, row["mask_path"], th) for row, p in batch])):
                g = per_img.setdefault(row["source_image"], {k: [0, 0] for k in th})
                for k, (h, n) in r.items():
                    g[k][0] += h; g[k][1] += n
            batch.clear()

        for row in TILES:
            img = 255 - np.asarray(Image.open(row["image_path"]).convert("RGB"), np.uint8)
            h, w = img.shape[:2]
            H, W = max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32)
            x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
            with torch.inference_mode():
                p = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)))[0, 0, :h, :w]
            batch.append((row, p.float().cpu().numpy()))
            if len(batch) >= 48:
                flush()
        flush()
        tot = {k: [sum(v[k][0] for v in per_img.values()), sum(v[k][1] for v in per_img.values())] for k in th}
        rec = {"model": name, "arm": spec["arm"], "seed": spec["seed"], "thresholds": th,
               "recall": {k: tot[k][0] / tot[k][1] for k in th}, "hits_total": tot, "per_source_image": per_img,
               "n_tiles": len(TILES), "seconds": round(time.time() - t0, 1)}
        dst.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps({k: rec[k] for k in ("model", "recall", "seconds")}), flush=True)
        del net; torch.cuda.empty_cache()
