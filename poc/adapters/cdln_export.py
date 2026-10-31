# -*- coding: utf-8 -*-
"""Export CrackDenseLinkNet (Manjunatha et al., Structural Health Monitoring 2023; official repo
preethamam/CrackDenseLinkNet-DeepLearning-CrackSegmentation) probability maps under the project's 16-bit probability PNG prediction contract.

Official path (test.py + torch_utils.Dataset): cv2.imread (BGR, no colour conversion) -> cv2.resize to 512x512 ->
segmentation_models_pytorch `get_preprocessing_fn('densenet169', 'imagenet')` (x/255, ImageNet mean/std, applied to the
BGR array exactly as the official code does) -> `smp.Linknet('densenet169', classes=1, activation='sigmoid')`,
checkpoint['model_state_dict'] loaded strictly (a leading 'module.' from DataParallel is stripped).
Output: the sigmoid probability (official code rounds it to a mask).
Runtime: smp 0.3.2 as pinned; timm 0.6.13 instead of the pinned 0.6.12 (0.6.12 fails to import on Python 3.11;
the densenet169 encoder does not use timm).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (append_development_exposures, default_development_exposures_auto,
                    ensure_prediction_output_compatible, save_probability_png)  # noqa: E402

MODEL_NAME = "cdln_liu_fcn_csn_usc"
INPUT_SIZE = 512


def flat(image_id: str) -> str:
    return image_id.replace("/", "__")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry(path: Path, datasets: set[str]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not datasets or row["source_dataset"] in datasets:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import segmentation_models_pytorch as smp
    import timm
    import torch
    import torch.nn.functional as functional
    from segmentation_models_pytorch.encoders import get_preprocessing_fn

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    weights = Path(args.weights).resolve()

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    model = smp.Linknet("densenet169", encoder_weights=None, classes=1, activation="sigmoid")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    preprocess = get_preprocessing_fn("densenet169", pretrained="imagenet")

    output_dir = Path(args.pred_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = file_sha256(weights)
    ensure_prediction_output_compatible(output_dir, weights_sha256, args.overwrite)
    manifest = []
    for start in tqdm(range(0, len(rows), max(1, args.batch_size)), unit="batch"):
        batch_rows = rows[start:start + max(1, args.batch_size)]
        tensors = []
        for row in batch_rows:
            image = cv2.imread(row["path"], cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"cannot read {row['path']}")
            image = cv2.resize(image, dsize=(INPUT_SIZE, INPUT_SIZE))
            array = preprocess(image).astype(np.float32).transpose(2, 0, 1)
            tensors.append(torch.from_numpy(array))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            probabilities = model(inputs)
        for row, probability in zip(batch_rows, probabilities):
            output = output_dir / f"{flat(row['image_id'])}.png"
            if args.overwrite or not output.exists():
                restored = functional.interpolate(
                    probability.unsqueeze(0).float(), size=(int(row["height"]), int(row["width"])),
                    mode="bilinear", align_corners=False)[0, 0].cpu()
                save_probability_png(output, restored.clamp(0, 1).numpy())
            manifest.append({"image_id": row["image_id"], "source_dataset": row["source_dataset"],
                             "input_path": row["path"], "output_path": str(output),
                             "width": row["width"], "height": row["height"]})

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    exposure_auto = None
    if args.limit:
        exposure_auto = default_development_exposures_auto(Path(args.registry))
        append_development_exposures(exposure_auto, rows, reason="adapter smoke run",
                                     source=Path(__file__).name, model=args.model_name)
    metadata = {
        "model": args.model_name,
        "official_repo": "https://github.com/preethamam/CrackDenseLinkNet-DeepLearning-CrackSegmentation",
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "Liu (DeepCrack-537) + FCN (Yang) + CrackSegNet + USC CDLN concrete (README)",
        "license": "MIT",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "smp": smp.__version__,
                    "timm": timm.__version__, "device": str(device)},
        "preprocess": {"color": "BGR as read by cv2 (official code does not convert)", "resize": [INPUT_SIZE, INPUT_SIZE],
                       "normalization": "smp get_preprocessing_fn('densenet169','imagenet')", "test_time_augmentation": False},
        "deviations": ["sigmoid probability exported instead of rounded mask", "timm 0.6.13 instead of pinned 0.6.12 (py3.11)"],
        "postprocess": "bilinear resize to original dimensions; uint16 PNG",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
