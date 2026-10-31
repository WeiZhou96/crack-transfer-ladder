# -*- coding: utf-8 -*-
"""Export CrackSAM (Ge et al., Construction and Building Materials 2024; official repo KG-TSI-Civil/CrackSAM)
probability maps under the project's 16-bit probability PNG prediction contract.

Official path (test.py / utils.test_single_volume): SAM ViT-H backbone (Meta sam_vit_h_4b8939.pth) built by the repo's
`sam_model_registry['vit_h'](image_size=448, num_classes=1, pixel_mean=[0,0,0], pixel_std=[1,1,1])`, wrapped by
`delta.sam_adapter_image_encoder.Adapter_Sam(sam, middle_dim=32, scaling_factor=0.2)` (README test example for
CrackSAM_adapter_d32.pth), delta parameters injected by the official `load_delta_parameters`. Input: RGB / 255 at
448x448; output `masks` (B,2,448,448) -> softmax channel 1 = crack probability.
Strictness: the official loader raises on any missing key; this adapter additionally checks that every key in the delta
checkpoint is consumed (adapter weights + prompt encoder + mask decoder), otherwise it refuses.
Deviations: PIL bicubic resize instead of scipy.ndimage.zoom(order=3) (both cubic, anisotropic to 448); probability
exported instead of argmax mask; runtime torch version differs from the repo's pinned 1.13.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (append_development_exposures, default_development_exposures_auto,
                    ensure_prediction_output_compatible, save_probability_png)  # noqa: E402

MODEL_NAME = "cracksam_adapter_khanhha"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repo", required=True, help="path to the CrackSAM/ folder of the official repo")
    parser.add_argument("--weights", required=True, help="CrackSAM_adapter_d32.pth")
    parser.add_argument("--sam-backbone", required=True, help="sam_vit_h_4b8939.pth")
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="")
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
    backbone = Path(args.sam_backbone).resolve()
    sys.path.insert(0, str(repo))
    from segment_anything import sam_model_registry  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    sam, _ = sam_model_registry["vit_h"](image_size=INPUT_SIZE, num_classes=1, checkpoint=str(backbone),
                                         pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    net = import_module("delta.sam_adapter_image_encoder").Adapter_Sam(sam, 32, 0.2)
    delta = torch.load(weights, map_location="cpu")
    expected = set()
    for prefix in ("w_a", "w_b", "w_c", "w_d"):
        for i in range(len(net.w_down_attn)):
            expected |= {f"{prefix}_{i:03d}", f"{prefix}_{i:03d}_bia"}
    expected |= {k for k in net.sam.state_dict() if "prompt_encoder" in k or "mask_decoder" in k}
    unused, missing = set(delta) - expected, expected - set(delta)
    if unused or missing:
        raise SystemExit(f"delta checkpoint key mismatch: unused={sorted(unused)[:5]} missing={sorted(missing)[:5]}")
    net.load_delta_parameters(str(weights))
    net.to(device)
    net.eval()

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
            tensors.append(torch.from_numpy(array))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            masks = net(inputs, False, INPUT_SIZE)["masks"]
            if masks.shape[-1] != INPUT_SIZE:
                masks = functional.interpolate(masks, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            probabilities = torch.softmax(masks.float(), dim=1)[:, 1:2]
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
        "official_repo": "https://github.com/KG-TSI-Civil/CrackSAM",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "sam_backbone_path": str(backbone),
        "sam_backbone_sha256": file_sha256(backbone),
        "weights_training_dataset": "Khanhha crack_segmentation (repository default; not stated per checkpoint)",
        "license": "MIT (code); SAM backbone Apache-2.0",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "preprocess": {"color": "RGB", "resize": [INPUT_SIZE, INPUT_SIZE], "scale": "x/255",
                       "pixel_mean": [0, 0, 0], "pixel_std": [1, 1, 1], "test_time_augmentation": False},
        "delta": {"type": "adapter", "middle_dim": 32, "scaling_factor": 0.2, "vit": "vit_h",
                  "all_checkpoint_keys_consumed": True},
        "deviations": ["PIL bicubic resize instead of scipy zoom(order=3)", "softmax probability instead of argmax mask",
                       f"torch {torch.__version__} instead of repository-pinned 1.13"],
        "postprocess": "softmax crack-class probability; bilinear resize to original dimensions; uint16 PNG",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
