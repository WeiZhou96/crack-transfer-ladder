# -*- coding: utf-8 -*-
"""Export Crackmer probability maps under the project's 16-bit probability PNG prediction contract."""
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


MODEL_NAME = "crackmer_deepcrack"
INPUT_SIZE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def flat(image_id: str) -> str:
    return image_id.replace("/", "__")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry(path: Path, datasets: set[str]) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if not datasets or row["source_dataset"] in datasets:
                rows.append(row)
    return rows


def normalize_state_dict(state: dict) -> dict:
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--datasets", default="", help="Comma-separated source_dataset filter")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--audit-float-root", default="", help="Smoke tests only: also save float32 .npy for quantisation AUC audit")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as functional

    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    sys.path.insert(0, str(repo / "src"))
    from Net import Net  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = Net().to(device)
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
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
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
                restored_array = restored.clamp(0, 1).numpy()
                save_probability_png(output, restored_array)
                if args.audit_float_root:
                    audit_path = (Path(args.audit_float_root) / args.model_name
                                  / f"{flat(row['image_id'])}.npy")
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(audit_path, restored_array.astype(np.float32), allow_pickle=False)
            manifest.append({
                "image_id": row["image_id"],
                "source_dataset": row["source_dataset"],
                "input_path": row["path"],
                "output_path": str(output),
                "width": row["width"],
                "height": row["height"],
            })

    fields = list(manifest[0])
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
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
        "official_repo": "https://github.com/zZhiG/crackmer",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "DeepCrack",
        "license": "unstated in repository",
        "preprocess": {
            "color": "RGB",
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "normalization_mean": MEAN,
            "normalization_std": STD,
            "test_time_augmentation": False,
        },
        "postprocess": "sigmoid logits; bilinear resize to original dimensions; uint16 PNG 0-65535",
        "prediction_contract": "probability_png_u16_v1",
        "audit_float_root": args.audit_float_root or None,
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
