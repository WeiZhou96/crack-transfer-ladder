# -*- coding: utf-8 -*-
"""Export HrSegNet-B16 probability maps under the project's 16-bit probability PNG prediction contract."""
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


MODEL_NAME = "hrsegnet_b16"
INPUT_SIZE = 400
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


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


def install_paddleseg_stubs(paddle) -> None:
    class Registry:
        @staticmethod
        def add_component(component):
            return component

    def kaiming_normal_init(parameter) -> None:
        paddle.nn.initializer.KaimingNormal()(parameter)

    def constant_init(parameter, value=0) -> None:
        paddle.nn.initializer.Constant(value)(parameter)

    def load_entire_model(_model, _path) -> None:
        raise RuntimeError("Adapter loads and validates the state dict explicitly")

    paddleseg = types.ModuleType("paddleseg")
    utils_pkg = types.ModuleType("paddleseg.utils")
    utils_pkg.utils = types.SimpleNamespace(load_entire_model=load_entire_model)
    cvlibs = types.ModuleType("paddleseg.cvlibs")
    cvlibs.manager = types.SimpleNamespace(MODELS=Registry())
    cvlibs.param_init = types.SimpleNamespace(
        kaiming_normal_init=kaiming_normal_init,
        constant_init=constant_init,
    )
    models_pkg = types.ModuleType("paddleseg.models")
    layers_pkg = types.ModuleType("paddleseg.models.layers")
    layer_libs = types.ModuleType("paddleseg.models.layers.layer_libs")
    layer_libs.SyncBatchNorm = paddle.nn.SyncBatchNorm
    sys.modules.update({
        "paddleseg": paddleseg,
        "paddleseg.utils": utils_pkg,
        "paddleseg.cvlibs": cvlibs,
        "paddleseg.models": models_pkg,
        "paddleseg.models.layers": layers_pkg,
        "paddleseg.models.layers.layer_libs": layer_libs,
    })


def validate_state_dict(model_state: dict, state: dict) -> None:
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatch = []
    for key in sorted(set(model_state) & set(state)):
        if tuple(model_state[key].shape) != tuple(state[key].shape):
            shape_mismatch.append((key, tuple(model_state[key].shape), tuple(state[key].shape)))
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"State dict mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"shape={shape_mismatch[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated source_dataset filter")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import paddle

    paddle.set_device(args.device)
    install_paddleseg_stubs(paddle)
    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    sys.path.insert(0, str(repo / "models"))
    from hrsegnet_b16 import HrSegNetB16  # noqa: E402

    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    rows = read_registry(Path(args.registry), datasets)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No registry rows match the requested datasets")

    model = HrSegNetB16(num_classes=2)
    state = paddle.load(str(weights))
    validate_state_dict(model.state_dict(), state)
    model.set_state_dict(state)
    model.eval()
    parameter_count = sum(int(np.prod(parameter.shape)) for parameter in model.parameters())

    output_dir = Path(args.pred_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = file_sha256(weights)
    ensure_prediction_output_compatible(output_dir, weights_sha256, args.overwrite)
    mean = np.asarray(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(STD, dtype=np.float32).reshape(3, 1, 1)
    manifest = []

    for start in tqdm(range(0, len(rows), max(1, args.batch_size)), unit="batch"):
        batch_rows = rows[start:start + max(1, args.batch_size)]
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                resized = image.convert("RGB").resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensors.append((array - mean) / std)
        inputs = paddle.to_tensor(np.stack(tensors), dtype="float32")
        with paddle.no_grad():
            logits = model(inputs)[0]
            probabilities = paddle.nn.functional.softmax(logits, axis=1)[:, 1].numpy()

        for row, probability in zip(batch_rows, probabilities):
            output = output_dir / f"{flat(row['image_id'])}.png"
            if args.overwrite or not output.exists():
                restored = Image.fromarray(probability.astype(np.float32), mode="F").resize(
                    (int(row["width"]), int(row["height"])), Image.Resampling.BILINEAR)
                save_probability_png(output, np.asarray(restored).clip(0, 1))
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
        "official_repo": "https://github.com/CHDyshli/HrSegNet4CrackSegmentation",
        "repo_path": str(repo),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "weights_training_dataset": "CrackSeg9k",
        "state_tensor_count": len(state),
        "parameter_count": parameter_count,
        "license": "Apache-2.0",
        "runtime": {"framework": "PaddlePaddle", "device": args.device},
        "preprocess": {
            "color": "RGB",
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "normalization_mean": MEAN,
            "normalization_std": STD,
            "test_time_augmentation": False,
        },
        "postprocess": "softmax crack class 1; bilinear resize to original dimensions; uint16 PNG 0-65535",
        "prediction_contract": "probability_png_u16_v1",
        "images": len(manifest),
        "development_exposures_auto": str(exposure_auto) if exposure_auto else None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
