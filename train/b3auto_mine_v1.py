"""E2: B3-auto mining (committee + VLM crack exclusion -> DINO clustering -> top-K; no human adjudication, which
makes B3-auto a weaker form of mining than one with manual review).

Sources (prereg b3_auto_mining): (a) consensus-screened train negatives = E3 states accept + hard_negative;
(b) train-positive images, crops that do not intersect dilate(GT, 30 px) (SDNET2018 fallback, prereg R08).
1. Scanner = the real-only student B0 (seed 0). Components at t = 0.5 filtered by the frozen component filter
   (poc/common.filter_components). Each component -> one 256x256 candidate crop centred on it (clipped into the image;
   for (b) rejected if it touches dilate(GT,30)); score = max probability in the component; per image, greedy NMS
   (crop IoU > 0.3 suppressed, higher score first).
2. Crack exclusion by the B3-auto committee (configs/committee_b3auto.yaml; members disjoint from the admission
   committee, three architecture families): candidate crops are written as PNG (R18-inverted for PaveDistress, as
   every model sees them) and scored by the frozen public-model adapters; a candidate is dropped when >= 2 members of
   different families form long components at the same location (s05 consensus rule). VLM: not run in this study.
3. DINOv2-ViT-S/14 embeddings (square 224, as s02) -> k-means (k = 50) -> top-K by score, round-robin over clusters.
   K = 3,000 (POC default = number of quadruplet groups per set). One mining round (iterative re-mining: not in POC).
Output: <out>/b3auto_crops.csv (image_id, x, y, side, score, cluster, source) -> student_v1.py --neg-list.
usage: python b3auto_mine_v1.py <code_dir> <data_root> <out_dir> --b0 <final.pt> [--gpu 0]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("out_dir")
ap.add_argument("--b0", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--k", type=int, default=3000)
ap.add_argument("--negpool-dir", default="negpool", help="E3 output dir under data_root (negpool = frozen rule, negpool_r20 = R20)")
ap.add_argument("--r20", action="store_true", help="R20: lane-marking exclusion from the committee consensus on laser crops")
args = ap.parse_args()
code, root, out = Path(args.code_dir), Path(args.data_root), Path(args.out_dir)
sys.path.insert(0, str(code / "poc"))
from common import DEFAULT_MIN_SKEL_PX, dilate, filter_components, load_probability_png  # noqa: E402
sys.path.insert(0, str(code / "train"))
from marking_r20 import LASER_SOURCES, marking_mask  # noqa: E402
from s05_negpool_screen import member_components, pair_same_location  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
(out / "crops").mkdir(exist_ok=True)
dev = torch.device(f"cuda:{args.gpu}")
S = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
INV = {"pavedistress", "pavedistress_anchor", "pavedistress_hard_proxy"}
reg = {json.loads(l)["image_id"]: json.loads(l) for l in open(root / "registry/registry.jsonl", encoding="utf-8")}
split = {r["image_id"]: r for r in csv.DictReader(open(root / "splits/split_assignments.v3.csv", encoding="utf-8-sig"))}
com = yaml.safe_load(open(code / "configs/committee_b3auto.yaml", encoding="utf-8"))


def load_img(iid):
    g = reg[iid]
    a = np.asarray(Image.open(g["path"]).convert("RGB"), np.uint8)
    return 255 - a if g["source_dataset"] in INV else a


net = smp.UPerNet("tu-convnext_tiny", encoder_weights=None, classes=1)
net.load_state_dict(torch.load(args.b0, map_location="cpu", weights_only=True))
net = net.to(dev).eval()


@torch.no_grad()
def b0_prob(img):
    h, w = img.shape[:2]
    H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
    return torch.sigmoid(net(torch.tensor(x.transpose(2, 0, 1)[None], device=dev)))[0, 0, :h, :w].cpu().numpy()


# ---------------------------------------------------------------- 1. candidates
sources = []
for state in ("accept", "hard_negative"):
    for r in csv.DictReader(open(root / args.negpool_dir / f"{state}.csv", encoding="utf-8-sig")):
        sources.append((r["image_id"], "negpool_" + state))
for iid, r in split.items():
    if r["pool"] == "train_positive" and r["split"] == "train":
        sources.append((iid, "positive_outside_gt30"))
cands = []
for iid, src in tqdm(sources, desc="scan"):
    img = load_img(iid)
    h, w = img.shape[:2]
    if src.startswith("negpool"):
        p = load_probability_png(root / "negpool_preds_v1" / "b0_student" / f"{iid.replace('/', '__')}.png")
        forbid = np.zeros((h, w), bool)
    else:
        p = b0_prob(img)
        m = np.asarray(Image.open(reg[iid]["mask_path"]).convert("L"))
        gt = (m > 0) if m.max() <= 1 else (m > 127)
        forbid = dilate(gt.astype(np.uint8), 30) > 0
    mark_full = marking_mask(reg[iid]["path"]) if (args.r20 and reg[iid]["source_dataset"] in LASER_SOURCES) else None
    comp, stats = filter_components((p >= 0.5).astype(np.uint8))
    if not stats:
        continue
    n, lab = cv2.connectedComponents(comp, connectivity=8)
    boxes = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        cy, cx = int(ys.mean()), int(xs.mean())
        x0 = int(np.clip(cx - S // 2, 0, max(w - S, 0)))
        y0 = int(np.clip(cy - S // 2, 0, max(h - S, 0)))
        if forbid[y0:y0 + S, x0:x0 + S].any():
            continue
        boxes.append((float(p[lab == k].max()), x0, y0))
    kept = []
    for sc, x0, y0 in sorted(boxes, reverse=True):
        if all(max(0, S - abs(x0 - a)) * max(0, S - abs(y0 - b)) / (S * S) <= 0.3 for _, a, b in kept):
            kept.append((sc, x0, y0))
    for sc, x0, y0 in kept:
        cid = f"{iid}#y{y0}_x{x0}_s{S}"
        path = out / "crops" / (cid.replace("/", "__").replace("#", "__") + ".png")
        if not path.exists():
            Image.fromarray(img[y0:y0 + S, x0:x0 + S]).save(path)
        mk = None
        if mark_full is not None:
            mk = mark_full[y0:y0 + S, x0:x0 + S]
            mk_path = out / "crops" / (path.stem + "_marking.png")
            Image.fromarray(mk.astype(np.uint8) * 255).save(mk_path)
        cands.append({"crop_id": cid, "image_id": iid, "x": x0, "y": y0, "side": S, "score": round(sc, 4),
                      "source": src, "path": str(path), "marking_path": str(mk_path) if mk is not None else "",
                      "marking_frac": round(float(mk.mean()), 4) if mk is not None else 0.0})
print("candidates", len(cands), flush=True)

# ---------------------------------------------------------------- 2. committee exclusion
reg_path = out / "candidates_registry.jsonl"
with open(reg_path, "w", encoding="utf-8") as f:
    for c in cands:
        f.write(json.dumps({"image_id": c["crop_id"], "path": c["path"], "width": S, "height": S,
                            "source_dataset": "b3auto_candidates"}) + "\n")
pred_root = out / "committee_pred"
for m in com["members"]:
    # venv interpreter: normalise the path but do NOT resolve symlinks (bin/python links to the base interpreter,
    # and resolving it would drop the venv and its mamba_ssm runtime)
    interp = os.path.normpath(str(code / m["python"])) if m.get("python") else sys.executable
    cmd = [interp, "-X", "utf8", str(code / "poc" / m["adapter"]), "--registry", str(reg_path),
           "--repo", str(code / m["repo"]), "--weights", str(code / m["weights"]), "--pred-root", str(pred_root),
           "--device", f"cuda:{args.gpu}", "--batch-size", str(m.get("batch_size", 4)), "--model-name", m["name"]]
    subprocess.run(cmd, check=True, cwd=str(code / "poc"))
members = com["members"]
for c in tqdm(cands, desc="exclusion"):
    masks = {m["name"]: member_components(pred_root / m["name"] / f"{c['crop_id'].replace('/', '__')}.png",
                                          m.get("threshold", 0.5)) for m in members}
    if c["marking_path"]:                               # R20: consensus counted outside lane markings only
        mk = np.asarray(Image.open(c["marking_path"])) > 0
        for n_, v in list(masks.items()):
            if v is not None:
                kept, comps = filter_components((v.astype(bool) & ~mk).astype(np.uint8), min_area_px=0,
                                                min_skel_px=DEFAULT_MIN_SKEL_PX)
                masks[n_] = kept if comps else np.zeros_like(v)
    agree = False
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if members[i]["arch_family"] != members[j]["arch_family"] and \
                    pair_same_location(masks[members[i]["name"]], masks[members[j]["name"]]):
                agree = True
    c["committee_drop"] = agree
survivors = [c for c in cands if not c["committee_drop"]]
print("survivors", len(survivors), "of", len(cands), flush=True)

# ---------------------------------------------------------------- 3. DINO clustering + top-K
from torchvision import transforms  # noqa: E402
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
feats = []
with torch.no_grad():
    for b in range(0, len(survivors), 128):
        x = torch.stack([tf(Image.open(c["path"]).convert("RGB")) for c in survivors[b:b + 128]]).to(dev)
        feats.append(torch.nn.functional.normalize(dino(x), dim=1).cpu().numpy())
F_ = np.concatenate(feats) if feats else np.zeros((0, 384), np.float32)
k = min(50, len(survivors))
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
cv2.setRNGSeed(0)
_, labels, _ = cv2.kmeans(F_.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
labels = labels.ravel()
by = {}
for c, l in zip(survivors, labels):
    c["cluster"] = int(l)
    by.setdefault(int(l), []).append(c)
for v in by.values():
    v.sort(key=lambda c: -c["score"])
chosen, rnd = [], 0
while len(chosen) < min(args.k, len(survivors)):
    for l in sorted(by):
        if rnd < len(by[l]) and len(chosen) < args.k:
            chosen.append(by[l][rnd])
    rnd += 1
with open(out / "b3auto_crops.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image_id", "x", "y", "side", "score", "cluster", "source", "crop_id", "marking_frac"])
    w.writeheader()
    for c in chosen:
        w.writerow({k_: c[k_] for k_ in w.fieldnames})
summary = {"sources": len(sources), "candidates": len(cands), "committee_dropped": len(cands) - len(survivors),
           "survivors": len(survivors), "k": len(chosen), "clusters": k,
           "chosen_by_source": {s: sum(1 for c in chosen if c["source"] == s) for s in {c["source"] for c in chosen}},
           "committee": [m["name"] for m in members], "vlm": "not run (POC)", "rounds": 1,
           "negpool_dir": args.negpool_dir, "r20": args.r20,
           "chosen_with_marking": sum(1 for c in chosen if c.get("marking_frac", 0) > 0)}
(out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
