#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np
from PIL import Image, ImageDraw


EXPECTED_LABEL = "crack_centerline"


def archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members = {}
    for raw_name in archive.namelist():
        normalized = raw_name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or PureWindowsPath(raw_name).drive:
            raise ValueError(f"ZIP contains unsafe path: {raw_name}")
        if normalized in members:
            raise ValueError(f"ZIP contains duplicate normalized path: {normalized}")
        members[normalized] = raw_name
    return members


def read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    members = archive_members(archive)
    if name not in members:
        raise KeyError(f"ZIP member not found: {name}")
    return archive.read(members[name])


def read_csv(archive: zipfile.ZipFile, name: str) -> tuple[bytes, list[dict]]:
    payload = read_member(archive, name)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    return payload, rows


def safe_members(archive: zipfile.ZipFile) -> list[str]:
    names = list(archive_members(archive))
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    return names


def task_index(rows: list[dict]) -> dict[str, dict]:
    index = {row["expected_json"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("task.csv contains duplicate expected_json")
    return index


def parse_annotation(payload: bytes, task: dict, prefix: str) -> tuple[dict, list[str], list[str]]:
    errors, warnings = [], []
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except Exception as exc:
        return {}, [f"{prefix}: invalid JSON: {exc!r}"], []
    if not isinstance(data, dict):
        return {}, [f"{prefix}: JSON root must be an object"], []
    filename = task["workspace_filename"]
    if PurePosixPath(str(data.get("imagePath", "")).replace("\\", "/")).name != filename:
        errors.append(f"{prefix}: imagePath does not match {filename}")
    width, height = int(task["width"]), int(task["height"])
    if any(type(data.get(key)) is not int for key in ("imageWidth", "imageHeight")):
        errors.append(f"{prefix}: image dimensions must be integers")
    elif data["imageWidth"] != width or data["imageHeight"] != height:
        errors.append(f"{prefix}: image size differs from task.csv")
    flags = data.get("flags", {})
    shapes = data.get("shapes")
    if not isinstance(flags, dict):
        errors.append(f"{prefix}: flags must be an object")
        flags = {}
    for key in ("no_visible_crack", "uncertain"):
        if key in flags and type(flags[key]) is not bool:
            errors.append(f"{prefix}: flag {key} must be boolean")
    if not isinstance(shapes, list):
        return data, errors + [f"{prefix}: shapes must be a list"], warnings
    if not shapes and not flags.get("no_visible_crack", False):
        errors.append(f"{prefix}: empty annotation lacks no_visible_crack=true")
    if shapes and flags.get("no_visible_crack", False):
        errors.append(f"{prefix}: has lines but no_visible_crack=true")
    if data.get("imageData"):
        warnings.append(f"{prefix}: imageData is embedded")

    duplicate_shapes = Counter()
    for shape_index, shape in enumerate(shapes, start=1):
        shape_prefix = f"{prefix} shape#{shape_index}"
        if not isinstance(shape, dict):
            errors.append(f"{shape_prefix}: shape must be an object")
            continue
        if shape.get("label") != EXPECTED_LABEL:
            errors.append(f"{shape_prefix}: label={shape.get('label')!r}")
        if shape.get("shape_type") != "linestrip":
            errors.append(f"{shape_prefix}: shape_type={shape.get('shape_type')!r}")
        points = shape.get("points")
        if not isinstance(points, list):
            errors.append(f"{shape_prefix}: points must be a list")
            continue
        if len(points) < 2:
            errors.append(f"{shape_prefix}: fewer than 2 points")
        canonical_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                errors.append(f"{shape_prefix}: invalid point={point!r}")
                continue
            x, y = point
            if not all(type(value) in (int, float) and math.isfinite(value)
                       for value in (x, y)):
                errors.append(f"{shape_prefix}: non-finite point={point!r}")
                continue
            elif not (0 <= x < width and 0 <= y < height):
                errors.append(f"{shape_prefix}: point outside image={point!r}")
            canonical_points.append((round(float(x), 4), round(float(y), 4)))
        if len(canonical_points) >= 2:
            duplicate_shapes[tuple(canonical_points)] += 1
            for left, right in zip(canonical_points, canonical_points[1:]):
                if left == right:
                    errors.append(f"{shape_prefix}: duplicate consecutive points")
    if any(count > 1 for count in duplicate_shapes.values()):
        warnings.append(f"{prefix}: contains exact duplicate line strips")
    return data, errors, warnings


def shape_statistics(data: dict) -> dict:
    point_counts = [len(shape.get("points") or []) for shape in data.get("shapes") or []]
    segment_lengths = []
    for shape in data.get("shapes") or []:
        points = np.asarray(shape.get("points") or [], dtype=np.float64)
        if len(points) >= 2:
            segment_lengths.extend(np.linalg.norm(np.diff(points, axis=0), axis=1).tolist())
    return {
        "shapes": len(point_counts),
        "points": sum(point_counts),
        "max_points_per_shape": max(point_counts, default=0),
        "max_segment_px": max(segment_lengths, default=0.0),
        "empty": int(not point_counts),
        "uncertain": int(bool((data.get("flags") or {}).get("uncertain", False))),
    }


def validate_submission(master: zipfile.ZipFile, submission_path: Path,
                        role: str, master_task_name: str | None = None) -> tuple[dict[str, dict], dict]:
    master_task_payload, master_rows = read_csv(master, master_task_name or f"{role}/task.csv")
    expected = task_index(master_rows)
    with zipfile.ZipFile(submission_path) as submission:
        names = safe_members(submission)
        errors, warnings = [], []
        try:
            task_payload, submitted_rows = read_csv(submission, "task.csv")
            submitted_index = task_index(submitted_rows)
        except (KeyError, ValueError, UnicodeError) as exc:
            task_payload, submitted_index = b"", {}
            errors.append(f"{role}: invalid or missing task.csv: {exc}")
        if task_payload != master_task_payload:
            errors.append(f"{role}: submitted task.csv differs from issued task.csv")
        if submitted_index != expected:
            errors.append(f"{role}: parsed task.csv rows differ from issued rows")
        json_names = sorted(name for name in names if name.startswith("labels/")
                            and name.lower().endswith(".json"))
        expected_names = {f"labels/{name}" for name in expected}
        missing = sorted(expected_names - set(json_names))
        unexpected = sorted(set(json_names) - expected_names)
        if missing:
            errors.append(f"{role}: missing {len(missing)} JSON; example={missing[0]}")
        if unexpected:
            errors.append(f"{role}: unexpected {len(unexpected)} JSON; example={unexpected[0]}")

        annotations = {}
        per_image = []
        received_json = 0
        for expected_json, task in expected.items():
            member = f"labels/{expected_json}"
            if member not in names:
                continue
            received_json += 1
            data, row_errors, row_warnings = parse_annotation(
                read_member(submission, member), task, f"{role}/{expected_json}")
            errors.extend(row_errors)
            warnings.extend(row_warnings)
            if row_errors:
                continue
            annotations[expected_json] = data
            per_image.append({"expected_json": expected_json, **shape_statistics(data)})
    points = [row["points"] for row in per_image]
    shapes = [row["shapes"] for row in per_image]
    summary = {
        "role": role,
        "submission": str(submission_path),
        "zip_crc": "pass",
        "expected_json": len(expected),
        "received_json": received_json,
        "valid_json": len(annotations),
        "task_csv_sha256": hashlib.sha256(task_payload).hexdigest(),
        "issued_task_csv_sha256": hashlib.sha256(master_task_payload).hexdigest(),
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "errors": errors,
        "warnings": warnings,
        "empty_images": sum(row["empty"] for row in per_image),
        "uncertain_images": sum(row["uncertain"] for row in per_image),
        "total_shapes": sum(shapes),
        "total_points": sum(points),
        "median_shapes_per_image": statistics.median(shapes) if shapes else None,
        "median_points_per_image": statistics.median(points) if points else None,
        "max_points_in_image": max(points, default=0),
        "max_points_per_shape": max((row["max_points_per_shape"] for row in per_image), default=0),
        "max_segment_px": max((row["max_segment_px"] for row in per_image), default=0.0),
        "per_image": per_image,
    }
    return annotations, summary


def centerline_mask(data: dict, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes") or []:
        points = np.rint(np.asarray(shape["points"], dtype=np.float64)).astype(np.int32)
        cv2.polylines(mask, [points], isClosed=False, color=1, thickness=1,
                      lineType=cv2.LINE_8)
    return mask


def directional_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not np.any(source):
        return np.asarray([], dtype=np.float32)
    if not np.any(target):
        return np.full(int(source.sum()), np.inf, dtype=np.float32)
    distance = cv2.distanceTransform((1 - target).astype(np.uint8), cv2.DIST_L2, 5)
    return distance[source.astype(bool)]


def agreement_metrics(a_data: dict, b_data: dict, task: dict) -> dict:
    width, height = int(task["width"]), int(task["height"])
    a_mask = centerline_mask(a_data, width, height)
    b_mask = centerline_mask(b_data, width, height)
    a_to_b = directional_distances(a_mask, b_mask)
    b_to_a = directional_distances(b_mask, a_mask)
    result = {
        "a_pixels": int(a_mask.sum()),
        "b_pixels": int(b_mask.sum()),
        "a_shapes": len(a_data.get("shapes") or []),
        "b_shapes": len(b_data.get("shapes") or []),
        "both_empty": bool(not np.any(a_mask) and not np.any(b_mask)),
    }
    finite = np.concatenate([a_to_b, b_to_a])
    if finite.size and np.all(np.isfinite(finite)):
        result["symmetric_mean_distance_px"] = float(finite.mean())
        result["symmetric_p95_distance_px"] = float(np.percentile(finite, 95))
    elif result["both_empty"]:
        result["symmetric_mean_distance_px"] = 0.0
        result["symmetric_p95_distance_px"] = 0.0
    else:
        result["symmetric_mean_distance_px"] = None
        result["symmetric_p95_distance_px"] = None
    for tolerance in (4, 8, 16):
        if result["both_empty"]:
            precision = recall = 1.0
        else:
            precision = float(np.mean(a_to_b <= tolerance)) if a_to_b.size else 0.0
            recall = float(np.mean(b_to_a <= tolerance)) if b_to_a.size else 0.0
        denominator = precision + recall - precision * recall
        result[f"cliou_{tolerance}px"] = (
            precision * recall / denominator if denominator > 0 else 0.0)
    return result


def load_image(master: zipfile.ZipFile, filename: str) -> Image.Image:
    payload = read_member(master, f"annotator_a/images/{filename}")
    return Image.open(io.BytesIO(payload)).convert("RGB")


def overlay(image: Image.Image, data: dict, color: tuple[int, int, int],
            size: tuple[int, int]) -> Image.Image:
    rendered = image.copy()
    scale = min(size[0] / image.width, size[1] / image.height)
    rendered = rendered.resize((max(1, round(image.width * scale)),
                                max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(rendered)
    for shape in data.get("shapes") or []:
        points = [(float(x) * scale, float(y) * scale) for x, y in shape["points"]]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")
    return rendered


def combined_overlay(image: Image.Image, a_data: dict, b_data: dict,
                     size: tuple[int, int]) -> Image.Image:
    rendered = overlay(image, a_data, (235, 40, 45), size)
    scale = rendered.width / image.width
    draw = ImageDraw.Draw(rendered)
    for shape in b_data.get("shapes") or []:
        points = [(float(x) * scale, float(y) * scale) for x, y in shape["points"]]
        if len(points) >= 2:
            draw.line(points, fill=(0, 210, 220), width=2, joint="curve")
    return rendered


def make_double_sheets(master: zipfile.ZipFile, tasks: list[dict],
                       a_annotations: dict[str, dict], b_annotations: dict[str, dict],
                       metrics: list[dict], output_dir: Path) -> list[str]:
    outputs = []
    panel_size = (360, 200)
    caption_height = 34
    by_json = {row["expected_json"]: row for row in metrics}
    for page_start in range(0, len(tasks), 10):
        page_tasks = tasks[page_start:page_start + 10]
        canvas = Image.new("RGB", (3 * panel_size[0], len(page_tasks) *
                                   (panel_size[1] + caption_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, task in enumerate(page_tasks):
            name = task["expected_json"]
            image = load_image(master, task["workspace_filename"])
            panels = [
                overlay(image, a_annotations[name], (235, 40, 45), panel_size),
                overlay(image, b_annotations[name], (0, 190, 210), panel_size),
                combined_overlay(image, a_annotations[name], b_annotations[name], panel_size),
            ]
            y0 = row_index * (panel_size[1] + caption_height)
            for column, panel in enumerate(panels):
                x0 = column * panel_size[0]
                canvas.paste(panel, (x0 + (panel_size[0] - panel.width) // 2,
                                     y0 + (panel_size[1] - panel.height) // 2))
            metric = by_json[name]
            caption = (f"{name}  A/B shapes={metric['a_shapes']}/{metric['b_shapes']}  "
                       f"clIoU4/8/16={metric['cliou_4px']:.3f}/"
                       f"{metric['cliou_8px']:.3f}/{metric['cliou_16px']:.3f}")
            draw.text((4, y0 + panel_size[1] + 4), caption, fill=(20, 20, 20))
        path = output_dir / f"double_agreement_{page_start // 10 + 1:02d}.jpg"
        canvas.save(path, quality=92)
        outputs.append(str(path))
    return outputs


def make_a_sheet(master: zipfile.ZipFile, tasks: list[dict], annotations: dict[str, dict],
                 per_image: list[dict], output: Path) -> None:
    by_json = {row["expected_json"]: row for row in per_image}
    selected_names = []
    ordered = [task["expected_json"] for task in tasks]
    for index in np.linspace(0, len(ordered) - 1, 20, dtype=int):
        selected_names.append(ordered[int(index)])
    selected_names.extend(row["expected_json"] for row in per_image if row["empty"])
    selected_names.extend(row["expected_json"] for row in sorted(
        per_image, key=lambda item: item["points"], reverse=True)[:4])
    selected_names = list(dict.fromkeys(selected_names))[:28]
    task_by_json = {task["expected_json"]: task for task in tasks}
    columns, tile_size, caption_height = 4, (360, 200), 28
    rows = (len(selected_names) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size[0], rows *
                               (tile_size[1] + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(selected_names):
        task = task_by_json[name]
        image = load_image(master, task["workspace_filename"])
        panel = overlay(image, annotations[name], (235, 40, 45), tile_size)
        x0 = (index % columns) * tile_size[0]
        y0 = (index // columns) * (tile_size[1] + caption_height)
        canvas.paste(panel, (x0 + (tile_size[0] - panel.width) // 2,
                             y0 + (tile_size[1] - panel.height) // 2))
        stats = by_json[name]
        draw.text((x0 + 3, y0 + tile_size[1] + 3),
                  f"{name} shapes={stats['shapes']} points={stats['points']}",
                  fill=(20, 20, 20))
    canvas.save(output, quality=92)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-package", required=True)
    parser.add_argument("--submission-a", required=True)
    parser.add_argument("--submission-b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.master_package) as master:
        safe_members(master)
        a_annotations, a_summary = validate_submission(
            master, Path(args.submission_a), "annotator_a")
        b_annotations, b_summary = validate_submission(
            master, Path(args.submission_b), "annotator_b")
        _, b_tasks = read_csv(master, "annotator_b/task.csv")
        double_metrics = []
        for task in b_tasks:
            name = task["expected_json"]
            if name not in a_annotations or name not in b_annotations:
                continue
            double_metrics.append({"expected_json": name, **agreement_metrics(
                a_annotations[name], b_annotations[name], task)})
        agreement_fields = [
            "expected_json", "a_pixels", "b_pixels", "a_shapes", "b_shapes",
            "both_empty", "symmetric_mean_distance_px", "symmetric_p95_distance_px",
            "cliou_4px", "cliou_8px", "cliou_16px",
        ]
        write_csv(output / "double_agreement.csv", double_metrics, agreement_fields)
        comparable_tasks = [task for task in b_tasks
                            if task["expected_json"] in a_annotations
                            and task["expected_json"] in b_annotations]
        sheets = make_double_sheets(
            master, comparable_tasks, a_annotations, b_annotations,
            double_metrics, output) if comparable_tasks else []
        _, a_tasks = read_csv(master, "annotator_a/task.csv")
        available_a_tasks = [task for task in a_tasks
                             if task["expected_json"] in a_annotations]
        if available_a_tasks:
            make_a_sheet(master, available_a_tasks, a_annotations,
                         a_summary["per_image"],
                         output / "annotator_a_visual_audit.jpg")

    cliou_summary = {
        f"cliou_{tolerance}px": {
            "mean": (float(np.mean([row[f"cliou_{tolerance}px"]
                                     for row in double_metrics]))
                     if double_metrics else None),
            "median": (float(np.median([row[f"cliou_{tolerance}px"]
                                         for row in double_metrics]))
                       if double_metrics else None),
            "minimum": (float(np.min([row[f"cliou_{tolerance}px"]
                                       for row in double_metrics]))
                        if double_metrics else None),
        }
        for tolerance in (4, 8, 16)
    }
    report = {
        "contract": {
            "label": EXPECTED_LABEL,
            "shape_type": "linestrip",
            "annotator_a_expected": a_summary["expected_json"],
            "annotator_b_expected": b_summary["expected_json"],
        },
        "annotator_a": a_summary,
        "annotator_b": b_summary,
        "double_annotation": {
            "pairs": len(double_metrics),
            "cliou": cliou_summary,
            "both_empty": sum(row["both_empty"] for row in double_metrics),
            "one_empty": sum((row["a_pixels"] == 0) != (row["b_pixels"] == 0)
                             for row in double_metrics),
            "visual_sheets": sheets,
        },
        "automatic_format_pass": not a_summary["errors"] and not b_summary["errors"],
        "visual_review_status": "pending",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "automatic_format_pass": report["automatic_format_pass"],
        "a_received": a_summary["received_json"],
        "b_received": b_summary["received_json"],
        "a_errors": len(a_summary["errors"]),
        "b_errors": len(b_summary["errors"]),
        "a_empty": a_summary["empty_images"],
        "b_empty": b_summary["empty_images"],
        "double_pairs": len(double_metrics),
        "cliou": cliou_summary,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    if not report["automatic_format_pass"]:
        raise SystemExit("submission format validation failed")


if __name__ == "__main__":
    main()
