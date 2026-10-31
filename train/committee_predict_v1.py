"""E3 committee predictions on the train-side negative pool, under the s05 prediction contract
(<pred_root>/<member>/<image_id with '/' -> '__'>.png, uint16 probability via poc/common.save_probability_png).

Members = the names frozen in configs/committee.yaml (FREEZE.v3); their weights are bound here and recorded in
<pred_root>/<member>/weights.json (committee.yaml itself is frozen with "TODO" weights and is re-registered at FREEZE.v4):
  unet_r34 / deeplabv3p_r50 / segformer_b2 : teachers_v1 (real train positives only, frozen)
  third_party_public                        : deepcrack_cracktree260 public weights (models_public.yaml; sha256 945239...),
                                              same conventions as poc/adapters/deepcrack_export.py (BGR [0,1], primary
                                              logit) except that frames are tiled at 512 instead of resized whole
  b0_student                                : E1 arm B0 seed 0 final checkpoint (UPerNet-ConvNeXt-T)
Preprocessing: R18 inversion (255 - x) for PaveDistress-sourced frames, for every member. Inference on 512x512 tiles
(grid step 512, last row/column snapped to the border), overlaps merged by max.
usage: python committee_predict_v1.py <code_dir> <data_root> <pred_root> --members a,b,... [--gpu 0]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("pred_root")
ap.add_argument("--members", required=True)
ap.add_argument("--gpu", default="0")
args = ap.parse_args()
code, root, pred_root = Path(args.code_dir), Path(args.data_root), Path(args.pred_root)
sys.path.insert(0, str(code / "poc"))
from common import save_probability_png  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

dev = torch.device(f"cuda:{args.gpu}")
T = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
INV = {"pavedistress", "pavedistress_anchor", "pavedistress_hard_proxy"}
reg = {json.loads(l)["image_id"]: json.loads(l) for l in open(root / "registry/registry.jsonl", encoding="utf-8")}
ids = [r["image_id"] for r in csv.DictReader(open(root / "splits/split_assignments.v3.csv", encoding="utf-8-sig"))
       if r["pool"] == "train_negative" and r["split"] == "train"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def member(name):
    if name in ("unet_r34", "deeplabv3p_r50", "segformer_b2"):
        b = {"unet_r34": lambda: smp.Unet("resnet34", encoder_weights=None, classes=1),
             "deeplabv3p_r50": lambda: smp.DeepLabV3Plus("resnet50", encoder_weights=None, classes=1),
             "segformer_b2": lambda: smp.Segformer("mit_b2", encoder_weights=None, classes=1)}[name]()
        w = root / "teachers_v1" / name / "best.pt"
        b.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        mode = "imagenet"
    elif name == "b0_student":
        b = smp.UPerNet("tu-convnext_tiny", encoder_weights=None, classes=1)
        w = root / "students_v1" / "B0_s0" / "final.pt"
        b.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        mode = "imagenet"
    elif name == "third_party_public":
        v = code / "vendor_models"
        sys.path.insert(0, str(v / "deepcrack_qinnzou_src" / "codes"))
        from model.deepcrack import DeepCrack  # noqa: E402
        b = DeepCrack()
        w = v / "deepcrack_qinnzou" / "checkpoints" / "DeepCrack_CT260.pth"
        st = torch.load(w, map_location="cpu", weights_only=True)
        for k in ("model", "state_dict", "model_state_dict"):
            if k in st and isinstance(st[k], dict):
                st = st[k]
                break
        b.load_state_dict({k.removeprefix("module."): t for k, t in st.items()}, strict=True)
        mode = "bgr01"
    else:
        raise SystemExit(f"unknown member {name}")
    return b.to(dev).eval(), mode, w


@torch.no_grad()
def predict(net, mode, img):
    h, w = img.shape[:2]
    ys = sorted(set(list(range(0, max(h - T, 0) + 1, T)) + [max(h - T, 0)]))
    xs = sorted(set(list(range(0, max(w - T, 0) + 1, T)) + [max(w - T, 0)]))
    out = np.zeros((h, w), np.float32)
    for y in ys:
        for x in xs:
            t = img[y:y + T, x:x + T].astype(np.float32) / 255
            if mode == "imagenet":
                t = (t - MEAN) / STD
            else:
                t = t[:, :, ::-1]
            tt = torch.tensor(np.ascontiguousarray(t.transpose(2, 0, 1))[None], device=dev)
            o = net(tt)
            logit = o[0] if isinstance(o, (list, tuple)) else o
            p = torch.sigmoid(logit)[0, 0].cpu().numpy()
            out[y:y + T, x:x + T] = np.maximum(out[y:y + T, x:x + T], p[:t.shape[0], :t.shape[1]])
    return out


for name in args.members.split(","):
    net, mode, wpath = member(name)
    od = pred_root / name
    od.mkdir(parents=True, exist_ok=True)
    (od / "weights.json").write_text(json.dumps({"member": name, "weights": str(wpath), "sha256": sha(wpath),
                                                  "preprocess": "R18 invert for PaveDistress; 512 tiles, max merge",
                                                  "input_mode": mode, "images": len(ids)}, indent=1), encoding="utf-8")
    for iid in tqdm(ids, desc=name):
        p = od / f"{iid.replace('/', '__')}.png"
        if p.exists():
            continue
        g = reg[iid]
        img = np.asarray(Image.open(g["path"]).convert("RGB"), np.uint8)
        if g["source_dataset"] in INV:
            img = 255 - img
        save_probability_png(p, predict(net, mode, img))
