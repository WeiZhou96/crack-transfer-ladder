# -*- coding: utf-8 -*-
"""Export CT-CrackSeg probability maps under the project's 16-bit probability PNG prediction contract."""
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


MODEL_NAME = "ct_crackseg_deepcrack"
INPUT_SIZE = 256
MEAN = (0.598, 0.584, 0.565)
STD = (0.104, 0.103, 0.103)


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


def normalize_state_dict(state: dict) -> dict:
    for key in ("model_weights", "model", "state_dict", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    return {key.removeprefix("module."): value for key, value in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated source_dataset filter")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    sys.path.insert(0, str(repo))
    from model.TransMUNet import TransMUNet  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = TransMUNet(n_classes=1).to(device)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(normalize_state_dict(state), strict=True)
    model.eval()

    output_dir = Path(args.pred_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = file_sha256(weights)
    ensure_prediction_output_compatible(output_dir, weights_sha256, args.overwrite)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    manifest = []

    for start in tqdm(range(0, len(rows), max(1, args.batch_size)), unit="batch"):
        batch_rows = rows[start:start + max(1, args.batch_size)]
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                resized = image.convert("RGB").resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
                array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensors.append((torch.from_numpy(array) - mean) / std)
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(inputs))

        for row, probability in zip(batch_rows, probabilities):
            output = output_dir / f"{flat(row['image_id'])}.png"
            if args.overwrite or not output.exists():
                restored = functional.interpolate(
                    probability.unsqueeze(0),
                    size=(int(row["height"]), int(row["width"])),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu()
                save_probability_png(output, restored.clamp(0, 1).numpy())
            manifest.append({
                "image_id": row["image_id"],
                "source_dataset": row["source_dataset"],
                "input_path": row["path"],
                "output_path": str(output),
                "width": row["width"],
                "height": row["height"],
            })

    fields = list(manifest[0])
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    exposure_auto = None
    if args.limit:
        exposure_auto = default_development_exposures_auto(Path(args.registry))
        append_development_exposures(
            exposure_auto, rows, reason="pre-freeze adapter smoke run",
            source=Path(__file__).name, model=args.model_name)
    metadata = {
        "model": args.model_name,
        "official_repo": "https://github.com/HqiTao/CT-crackseg",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "DeepCrack",
        "license": "MIT",
        "preprocess": {
            "color": "RGB",
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "normalization_mean": MEAN,
            "normalization_std": STD,
            "test_time_augmentation": False,
        },
        "postprocess": "sigmoid logits; bilinear resize to original dimensions; uint16 PNG 0-65535",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
