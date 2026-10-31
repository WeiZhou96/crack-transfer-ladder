# -*- coding: utf-8 -*-
"""Export DeepCrack (Liu et al., Neurocomputing 2019; official PyTorch code yhlleo/DeepSegmentor) probability maps
under the project's 16-bit probability PNG prediction contract. Distinct from Zou et al.'s DeepCrack (deepcrack_export.py).

Network: official `models/deepcrack_networks.DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch')`,
strict load of the released `pretrained_net_G.pth` (trained on the DeepCrack-537 benchmark).
Preprocessing follows the official dataset loader: RGB, ToTensor, Normalize(0.5, 0.5, 0.5).
Output: sigmoid of the fused logit (the official test path visualises (sigmoid(fused)-0.5)/0.5; the probability
itself is exported). Deviation: anisotropic square resize to 512x512 (official default is native size),
the unified convention used by all project adapters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (append_development_exposures, default_development_exposures_auto,
                    ensure_prediction_output_compatible, save_probability_png)  # noqa: E402

MODEL_NAME = "deepcrackliu_deepcrack537"
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
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    sys.path.insert(0, str(repo))
    from models.deepcrack_networks import DeepCrackNet  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    model = DeepCrackNet(3, 1, 64, norm="batch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    output_dir = Path(args.pred_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = file_sha256(weights)
    ensure_prediction_output_compatible(output_dir, weights_sha256, args.overwrite)
    manifest = []
    for start in tqdm(range(0, len(rows), max(1, args.batch_size)), unit="batch"):
        batch_rows = rows[start:start + max(1, args.batch_size)]
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
                array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensors.append(torch.from_numpy((array - 0.5) / 0.5))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(inputs)[-1])
        for row, probability in zip(batch_rows, probabilities):
            output = output_dir / f"{flat(row['image_id'])}.png"
            if args.overwrite or not output.exists():
                restored = functional.interpolate(
                    probability.unsqueeze(0), size=(int(row["height"]), int(row["width"])),
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
        "official_repo": "https://github.com/yhlleo/DeepSegmentor",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "DeepCrack-537 (Liu et al.)",
        "license": "BSD-2-Clause (repository LICENSE)",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "preprocess": {"color": "RGB", "resize": [INPUT_SIZE, INPUT_SIZE],
                       "normalization_mean": [0.5, 0.5, 0.5], "normalization_std": [0.5, 0.5, 0.5],
                       "test_time_augmentation": False},
        "deviations": ["anisotropic square resize to 512 (official default is native size)"],
        "postprocess": "sigmoid of fused logit; bilinear resize to original dimensions; uint16 PNG",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
