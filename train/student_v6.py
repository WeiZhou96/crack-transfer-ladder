"""E1 student trainer v6 (experiment_program_v1 addendum 3): student_v5 plus --backbone. upernet_resnet50 =
smp.UPerNet("resnet50") whose encoder is initialised from the cached torchvision ImageNet checkpoint
(resnet50-11ad3fa6.pth, fc dropped, strict load; no network access). Recipe, loss, data and arms unchanged; with the
default backbone v6 == v5.

E1 student trainer v5 (experiment_program_v1 ablations): student_v4 with arms read from --arms-config.

Recipe and loss constants still come from configs/e1_arms.yaml; an arm may only (a) restrict unpaired layers
(synth.layers), (b) override alpha_pair / beta_marg, or (c) mix mined real negatives with paired quadruplets
(B12_auto: other half = 8 mined negatives + 2 aligned groups). With the default arms config v5 == v4.

E1 student trainer v4 (prereg_v6 R20): student_v1 recipe and loss unchanged; tau_delta per stratum from prereg_v6.

Differences from v1: --tau-prereg/--tau-level replace the scalar --tau-delta (tau looked up by manifest stratum);
decoded quadruplet groups are cached in RAM as uint8 (values identical, cast to float32 after cropping).

E1 student trainer for the six arms (loss and training recipe as described in the paper). One script, arms from configs/e1_arms.yaml.

Batch = 16 real positive crops + 16 "other" (1:1): nothing (B0/B1: all 32 real), real negative crops (B2/B3-auto),
four unpaired quadruplet layers x4 (B10) or four aligned quadruplets (B11: x00,x01,x10,x11 cropped identically).
Loss per sample (v6 eq. 5): GT non-empty -> Dice + BCE with pixel weight w (0 on ignore/padding);
GT empty -> kappa_s-weighted BCE (kappa_s inside Omega_Z, 1 elsewhere), Dice skipped.
B11 adds alpha * L_pair (eq. 6: |f(x11)-f(x10)| on W_stable = valid & not ignore(x10|x11), |f(x01)-f(x00)| on W_bg = valid)
and beta * L_marg (eq. 7: max(0, m - [<f(x11)>_{M&V} - <f(x01)>_M])).
R18: PaveDistress-sourced real images and laser-domain synthetic layers are inverted (255 - x) at load for every model.
Synthetic labels: y = M & V, ignore = M & ~V via labels_from_dc, tau_delta per stratum (prereg_v6 v6_execution_decisions).
Checkpoints at fixed steps; the final step is the model (validation on train_positive/val is logged, never used to select).
usage: python student_v4.py <code_dir> <data_root> <arm> <seed> <out_dir> [--gpu 0] [--iters N]
       [--quad-dir DIR --tau-prereg configs/prereg_v6.yaml [--tau-level point|ci_low|ci_high]] [--neg-list CSV]
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("code_dir")
ap.add_argument("data_root")
ap.add_argument("arm")
ap.add_argument("seed", type=int)
ap.add_argument("out_dir")
ap.add_argument("--gpu", default="0")
ap.add_argument("--iters", type=int, default=None)
ap.add_argument("--quad-dir", default=None)
ap.add_argument("--tau-prereg", default=None, help="prereg_v6.yaml holding v6_execution_decisions.tau_delta_by_stratum")
ap.add_argument("--arms-config", default=None, help="yaml with an arms: mapping (default configs/e1_arms.yaml)")
ap.add_argument("--tau-level", choices=["point", "ci_low", "ci_high"], default="point", help="ci_*: R20 sensitivity")
ap.add_argument("--neg-list", default=None, help="csv with image_id column (B2 accept pool / B3-auto mined)")
ap.add_argument("--backbone", choices=["upernet_convnext_tiny", "upernet_resnet50"], default=None)
args = ap.parse_args()
sys.path.insert(0, args.code_dir)
import segmentation_models_pytorch as smp  # noqa: E402

cfg = yaml.safe_load(open(Path(args.code_dir) / "configs/e1_arms.yaml", encoding="utf-8"))
arm = (yaml.safe_load(open(args.arms_config, encoding="utf-8"))["arms"] if args.arms_config else cfg["arms"])[args.arm]
rc, lc, pc = cfg["recipe"], cfg["loss"], cfg["preprocess"]
if args.backbone:
    rc = dict(rc, backbone=args.backbone)
R50_CKPT = Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
iters = args.iters or rc["iters"]
root, out = Path(args.data_root), Path(args.out_dir)
out.mkdir(parents=True, exist_ok=False)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device(f"cuda:{args.gpu}")
C = rc["crop"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
reg = {json.loads(l)["image_id"]: json.loads(l) for l in open(root / "registry/registry.jsonl", encoding="utf-8")}
INV = set(pc["r18_invert_sources"])


def load_binary_mask(path):
    a = np.asarray(Image.open(path).convert("L"))
    m = int(a.max())
    if m <= 1:
        return a > 0
    if m <= 127:
        raise ValueError(f"ambiguous mask value domain: {path}")
    return a > 127


def load_real(image_id):
    g = reg[image_id]
    img = np.asarray(Image.open(g["path"]).convert("RGB"), np.uint8)
    if g["source_dataset"] in INV:
        img = 255 - img
    return img


# ------------------------------------------------------------------ data
split = {"train": [], "val": []}
for r in csv.DictReader(open(root / "splits/split_assignments.v3.csv", encoding="utf-8-sig")):
    if r["pool"] == "train_positive" and r["split"] in split:
        split[r["split"]].append(r["image_id"])
POS = [(load_real(i), load_binary_mask(reg[i]["mask_path"])) for i in split["train"]]
VAL = [(load_real(i), load_binary_mask(reg[i]["mask_path"])) for i in split["val"]]
NEG = []          # image-level list (B2: random crops of admitted frames) or crop list (B3-auto: mined crops)
TRAIN_SPLIT = {r["image_id"]: r for r in csv.DictReader(open(root / "splits/split_assignments.v3.csv", encoding="utf-8-sig"))}
if arm.get("real_neg"):
    if not args.neg_list:
        raise SystemExit(f"arm {args.arm} needs --neg-list ({arm['real_neg']})")
    NEG = list(csv.DictReader(open(args.neg_list, encoding="utf-8-sig")))
    for r in NEG:
        s = TRAIN_SPLIT[r["image_id"]]
        crop_mode = "x" in r
        ok = s["split"] == "train" and (s["pool"] == "train_negative" or (crop_mode and s["pool"] == "train_positive"))
        if not ok:                     # image-level negatives must be train negatives; mined crops may also come from
            raise SystemExit(f"negative list row not allowed: {r}")   # train-positive images outside dilate(GT,30)
_cache = {}


def load_cached(image_id):
    if image_id not in _cache:
        if len(_cache) >= 128:
            _cache.pop(next(iter(_cache)))
        _cache[image_id] = load_real(image_id)
    return _cache[image_id]
GROUPS = []
TAU = None
if arm.get("synth"):
    if args.quad_dir is None or args.tau_prereg is None:
        raise SystemExit("synthetic arms need --quad-dir and --tau-prereg")
    _dec = yaml.safe_load(open(args.tau_prereg, encoding="utf-8"))["v6_execution_decisions"]
    TAU = dict(_dec["tau_delta_by_stratum"]) if args.tau_level == "point" else         {k: v[0 if args.tau_level == "ci_low" else 1] for k, v in _dec["tau_delta_sensitivity"].items()}
    print(json.dumps({"tau_level": args.tau_level, "tau_delta_by_stratum": TAU}), flush=True)
    from render.quadruplet import labels_from_dc  # noqa: E402
    qd = Path(args.quad_dir)
    for l in open(qd / "index.jsonl", encoding="utf-8"):
        g = json.loads(l)
        GROUPS.append(qd / g["operator"] / g["group_id"])
SYN_INV = set(pc["r18_invert_synthetic_domains"])


def rand_crop(arrs, pad_valid=True):
    """Same random CxC crop for all arrays (HxW[xc]); arrays smaller than C are zero-padded, valid=0 there."""
    h, w = arrs[0].shape[:2]
    y0 = random.randint(0, max(h - C, 0))
    x0 = random.randint(0, max(w - C, 0))
    outl = []
    for a in arrs:
        c = a[y0:y0 + C, x0:x0 + C]
        if c.shape[0] < C or c.shape[1] < C:
            pad = [(0, C - c.shape[0]), (0, C - c.shape[1])] + [(0, 0)] * (a.ndim - 2)
            c = np.pad(c, pad)
        outl.append(c)
    valid = np.zeros((C, C), np.float32)
    valid[:min(C, h - y0), :min(C, w - x0)] = 1.0
    return outl, valid


def geo(k, flip, *arrs):
    res = []
    for a in arrs:
        a = np.rot90(a, k)
        if flip:
            a = a[:, ::-1]
        res.append(np.ascontiguousarray(a))
    return res


def strong_aug(img):
    """B1: generic photometric + degradation augmentation (no confuser operators)."""
    x = img.astype(np.float32) / 255
    x = (x - 0.5) * random.uniform(0.6, 1.4) + 0.5 + random.uniform(-0.15, 0.15)
    x = x * np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if random.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.5, 2.0))
    if random.random() < 0.3:
        x = x + np.random.normal(0, random.uniform(0.005, 0.04), x.shape).astype(np.float32)
    x = np.clip(x, 0, 1)
    if random.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])
        x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    return (x * 255).astype(np.uint8)


def sample_pos():
    img, m = random.choice(POS)
    (ci, cm), valid = rand_crop([img, m.astype(np.float32)])
    k, fl = random.randint(0, 3), random.random() < 0.5
    ci, cm, valid = geo(k, fl, ci, cm, valid)
    if arm.get("strong_aug"):
        ci = strong_aug(ci)
    return ci, cm, np.zeros_like(cm), valid, np.zeros_like(cm)        # img, y, ignore, valid, omega_z


_crops = {}


def sample_neg():
    j = random.randrange(len(NEG))
    r = NEG[j]
    if "x" in r:                                       # mined crop: train inside its box only (crop cached, small)
        if j not in _crops:
            x, y, s = int(r["x"]), int(r["y"]), int(r["side"])
            _crops[j] = load_real(r["image_id"])[y:y + s, x:x + s].copy()
        img = _crops[j]
    else:                                              # admitted frame: whole frame cached (JPEG decode dominates)
        if r["image_id"] not in _cache:
            _cache[r["image_id"]] = load_real(r["image_id"])
        img = _cache[r["image_id"]]
    (ci,), valid = rand_crop([img])
    k, fl = random.randint(0, 3), random.random() < 0.5
    ci, valid = geo(k, fl, ci, valid)
    z = np.zeros((C, C), np.float32)
    return ci, z, z, valid, z


_gcache = {}


def load_group(d):
    if d in _gcache:
        return _gcache[d]
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    tau = TAU[man["spec"]["stratum"]]
    inv = man["spec"]["background"]["domain"] in SYN_INV
    ims = {k: np.asarray(Image.open(d / f"{k}.png").convert("RGB"), np.uint8) for k in ("x00", "x01", "x10", "x11")}
    if inv:
        ims = {k: 255 - v for k, v in ims.items()}
    M = np.asarray(Image.open(d / "M.png")) > 0
    oz = (np.asarray(Image.open(d / "omega_z.png")) > 0).astype(np.uint8)
    dc = np.load(d / "dc.npz")
    y10, i10 = labels_from_dc(M, dc["skel_yx"], dc["dc10"], tau)
    y11, i11 = labels_from_dc(M, dc["skel_yx"], dc["dc11"], tau)
    z = np.zeros_like(oz)
    lab = {"x00": (z, z, z), "x01": (z, z, oz), "x10": (y10.astype(np.uint8), i10.astype(np.uint8), z),
           "x11": (y11.astype(np.uint8), i11.astype(np.uint8), oz)}
    _gcache[d] = (ims, lab, M)
    return ims, lab, M


def sample_group_aligned():
    ims, lab, M = load_group(random.choice(GROUPS))
    keys = ("x00", "x01", "x10", "x11")
    arrs = [ims[k] for k in keys] + [a for k in keys for a in lab[k]] + [M.astype(np.float32)]
    crops, valid = rand_crop(arrs)
    crops = crops[:4] + [c.astype(np.float32) for c in crops[4:]]
    k, fl = random.randint(0, 3), random.random() < 0.5
    crops = geo(k, fl, *crops)
    valid = geo(k, fl, valid)[0]
    out_ = []
    for j, key in enumerate(keys):
        y, ig, oz = crops[4 + 3 * j:7 + 3 * j]
        out_.append((crops[j], y, ig, valid, oz))
    return out_, crops[-1]


def sample_layer_unpaired():
    ims, lab, _ = load_group(random.choice(GROUPS))
    key = random.choice(tuple(arm["synth"].get("layers", ("x00", "x01", "x10", "x11"))))
    y, ig, oz = lab[key]
    (ci, cy, cig, coz), valid = rand_crop([ims[key], y, ig, oz])
    cy, cig, coz = cy.astype(np.float32), cig.astype(np.float32), coz.astype(np.float32)
    k, fl = random.randint(0, 3), random.random() < 0.5
    ci, cy, cig, coz, valid = geo(k, fl, ci, cy, cig, coz, valid)
    return ci, cy, cig, valid, coz


def to_tensor(samples):
    x = np.stack([((s[0].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for s in samples])
    rest = [np.stack([s[i] for s in samples])[:, None] for i in range(1, 5)]
    return [torch.tensor(a, dtype=torch.float32, device=dev) for a in [x] + rest]


# ------------------------------------------------------------------ model / loss
def build():
    if rc["backbone"] == "upernet_convnext_tiny":
        return smp.UPerNet("tu-convnext_tiny", encoder_weights=rc["encoder_weights"], classes=1)
    if rc["backbone"] == "upernet_resnet50":
        m = smp.UPerNet("resnet50", encoder_weights=None, classes=1)
        sd = {k: v for k, v in torch.load(R50_CKPT, map_location="cpu", weights_only=True).items()
              if not k.startswith("fc.")}
        m.encoder.load_state_dict(sd, strict=True)
        return m
    raise SystemExit(f"unknown backbone {rc['backbone']}")


def seg_loss(logit, y, ig, valid, oz):
    w = valid * (1 - ig)
    prob = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    has = (y * valid).flatten(1).sum(1) > 0
    kappa = 1 + (lc["kappa_s"] - 1) * oz
    wn = torch.where(has[:, None, None, None], w, w * kappa)
    l_bce = (bce * wn).flatten(1).sum(1) / wn.flatten(1).sum(1).clamp_min(1.0)
    inter = (prob * y * w).flatten(1).sum(1)
    den = (prob * w).flatten(1).sum(1) + (y * w).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (den + 1)
    l = l_bce + torch.where(has, dice, torch.zeros_like(dice))       # GT-empty samples skip Dice (v6 eq. 5)
    return l.mean()


net = build().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / iters) ** rc["poly_power"])
half = rc["batch"] // 2
log, t0 = [], time.time()


@torch.no_grad()
def validate():
    net.eval()
    tp = fp = fn = 0
    for img, m in VAL:
        h, w = m.shape
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        x = np.pad((img.astype(np.float32) / 255 - MEAN) / STD, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
        p = torch.sigmoid(net(torch.tensor(x.transpose(2, 0, 1)[None], device=dev)))[0, 0, :h, :w].cpu().numpy() >= 0.5
        tp += (p & m).sum(); fp += (p & ~m).sum(); fn += (~p & m).sum()
    net.train()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


net.train()
for it in range(1, iters + 1):
    other = rc["batch"] - half
    real = [sample_pos() for _ in range(half)]
    extra_loss = torch.zeros((), device=dev)
    goff = half
    if arm.get("synth") and arm["synth"]["paired"]:
        negs = [sample_neg() for _ in range(other // 2)] if NEG else []
        goff = half + len(negs)
        groups, Ms = [], []
        for _ in range((other - len(negs)) // 4):
            g, Mc = sample_group_aligned()
            groups.extend(g); Ms.append(Mc)
        batch = real + negs + groups
    elif arm.get("synth"):
        batch = real + [sample_layer_unpaired() for _ in range(other)]
    elif NEG:
        batch = real + [sample_neg() for _ in range(other)]
    else:
        batch = real + [sample_pos() for _ in range(other)]
    x, y, ig, valid, oz = to_tensor(batch)
    logit = net(x)
    loss = seg_loss(logit, y, ig, valid, oz)
    if arm.get("synth") and arm["synth"]["paired"]:
        f = torch.sigmoid(logit[goff:]).view(-1, 4, 1, C, C)      # [group, (x00,x01,x10,x11), ...]
        v = valid[goff:].view(-1, 4, 1, C, C)[:, 0]
        igs = ig[goff:].view(-1, 4, 1, C, C)
        ys = y[goff:].view(-1, 4, 1, C, C)
        w_stable = v * (1 - torch.clamp(igs[:, 2] + igs[:, 3], 0, 1))
        l_pair = ((f[:, 3] - f[:, 2]).abs() * w_stable).flatten(1).sum(1) / w_stable.flatten(1).sum(1).clamp_min(1) \
            + ((f[:, 1] - f[:, 0]).abs() * v).flatten(1).sum(1) / v.flatten(1).sum(1).clamp_min(1)
        Mt = torch.tensor(np.stack(Ms)[:, None], dtype=torch.float32, device=dev) * v
        mv = ys[:, 3] * v
        crack_resp = (f[:, 3] * mv).flatten(1).sum(1) / mv.flatten(1).sum(1).clamp_min(1)
        bg_resp = (f[:, 1] * Mt).flatten(1).sum(1) / Mt.flatten(1).sum(1).clamp_min(1)
        has = (mv.flatten(1).sum(1) > 0) & (Mt.flatten(1).sum(1) > 0)
        l_marg = torch.where(has, F.relu(lc["margin_m"] - (crack_resp - bg_resp)), torch.zeros_like(crack_resp))
        extra_loss = (arm["synth"].get("alpha_pair", lc["alpha_pair"]) * l_pair.mean()
                      + arm["synth"].get("beta_marg", lc["beta_marg"]) * l_marg.mean())
        loss = loss + extra_loss
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % rc["val_every"] == 0 or it == iters:
        rec = {"iter": it, "loss": round(float(loss), 4), "extra": round(float(extra_loss), 4),
               "val_f1_at_0.5": round(validate(), 4), "elapsed_s": round(time.time() - t0)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if it % rc["checkpoint_every"] == 0 or it == iters:
        torch.save(net.state_dict(), out / ("final.pt" if it == iters else f"it{it}.pt"))
meta = {"arm": args.arm, "arm_config": arm, "seed": args.seed, "iters": iters, "recipe": rc, "loss": lc,
        "preprocess": pc, "config_version": cfg["version"], "train_pos": len(POS), "neg_list": args.neg_list,
        "neg_images": len(NEG), "quad_dir": args.quad_dir, "tau_prereg": args.tau_prereg, "tau_level": args.tau_level,
        "tau_delta_by_stratum": TAU, "trainer": "student_v6", "backbone": rc["backbone"], "arms_config": args.arms_config, "groups": len(GROUPS),
        "smp_version": smp.__version__, "log": log, "model": "final.pt (fixed step; no validation-based selection)"}
(out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
