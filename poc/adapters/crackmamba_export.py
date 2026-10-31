# -*- coding: utf-8 -*-
"""Export CrackMamba (CrackSeg9k checkpoint) probability maps under the project's 16-bit probability PNG prediction contract.

The official code is an nnU-Net v2 fork. Inference here rebuilds the network directly from the official
`nnunetv2/nets/CrackMamba.py` (only the plans-handler import is stubbed; it is used for type hints) and
strictly loads `network_weights` from the official checkpoint. Deviations from `nnUNetv2_predict`
(disclosed in metadata): anisotropic square resize to the training patch size 448x448 instead of
sliding-window patches with Gaussian weighting (same unified convention as all project adapters);
TTA disabled, as in the official predict script (`--disable_tta`).
Preprocessing follows the logged plans: RGB, per-image per-channel z-score (ZScoreNormalization,
use_mask_for_norm=False). Output = softmax probability of the crack class.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (append_development_exposures, default_development_exposures_auto,
                    ensure_prediction_output_compatible, save_probability_png)  # noqa: E402

MODEL_NAME = "crackmamba_crackseg9k"
INPUT_SIZE = 448


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


def stub_plans_handler() -> None:
    names = ["nnunetv2.utilities", "nnunetv2.utilities.plans_handling",
             "nnunetv2.utilities.plans_handling.plans_handler"]
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)
    handler = sys.modules["nnunetv2.utilities.plans_handling.plans_handler"]
    handler.ConfigurationManager = type("ConfigurationManager", (), {})
    handler.PlansManager = type("PlansManager", (), {})


def load_network_weights(path: Path) -> tuple[dict, dict]:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["network_weights"] if "network_weights" in checkpoint else checkpoint
    cleaned = {}
    for key, value in state.items():
        for prefix in ("_orig_mod.", "module."):
            key = key.removeprefix(prefix)
        cleaned[key] = value
    info = {k: checkpoint.get(k) for k in ("trainer_name", "current_epoch", "inference_allowed_mirroring_axes")
            if isinstance(checkpoint, dict) and k in checkpoint}
    return cleaned, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated source_dataset filter")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
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
    stub_plans_handler()
    from nnunetv2.nets.CrackMamba import CrackMamba  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    model = CrackMamba(in_chans=3, out_chans=2, feat_size=[48, 96, 192, 384, 768],
                       deep_supervision=True, hidden_size=768)
    state, checkpoint_info = load_network_weights(weights)
    model.load_state_dict(state, strict=True)
    model.deep_supervision = False          # nnU-Net inference uses the full-resolution head only
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
                array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
            mean = array.reshape(3, -1).mean(axis=1)[:, None, None]
            std = array.reshape(3, -1).std(axis=1)[:, None, None]
            tensors.append(torch.from_numpy((array - mean) / np.maximum(std, 1e-8)))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            logits = model(inputs)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1:2]

        for row, probability in zip(batch_rows, probabilities):
            output = output_dir / f"{flat(row['image_id'])}.png"
            if args.overwrite or not output.exists():
                restored = functional.interpolate(
                    probability.unsqueeze(0), size=(int(row["height"]), int(row["width"])),
                    mode="bilinear", align_corners=False)[0, 0].cpu()
                save_probability_png(output, restored.clamp(0, 1).numpy())
            manifest.append({
                "image_id": row["image_id"], "source_dataset": row["source_dataset"],
                "input_path": row["path"], "output_path": str(output),
                "width": row["width"], "height": row["height"],
            })

    fields = list(manifest[0])
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    exposure_auto = None
    if args.limit:
        exposure_auto = default_development_exposures_auto(Path(args.registry))
        append_development_exposures(exposure_auto, rows, reason="adapter smoke run",
                                     source=Path(__file__).name, model=args.model_name)
    import monai
    metadata = {
        "model": args.model_name,
        "official_repo": "https://github.com/shengyu27/CrackMamba",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "CrackSeg9k",
        "checkpoint_info": {k: (v if isinstance(v, (int, float, str, list, type(None))) else str(v))
                            for k, v in checkpoint_info.items()},
        "license": "not_declared",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "monai": monai.__version__,
                    "selective_scan": "upstream fallback to mamba-ssm selective_scan_cuda (oflex/core kernels absent)",
                    "device": str(device)},
        "preprocess": {"color": "RGB", "resize": [INPUT_SIZE, INPUT_SIZE],
                       "normalization": "per-image per-channel z-score (nnU-Net ZScoreNormalization, no mask)",
                       "test_time_augmentation": False},
        "deviations": ["anisotropic square resize to 448 instead of nnU-Net sliding window with Gaussian weighting",
                       "plans handler stubbed (type hints only); network built from official CrackMamba.py"],
        "postprocess": "softmax crack-class probability; bilinear resize to original dimensions; uint16 PNG",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
