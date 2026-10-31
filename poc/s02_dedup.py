# -*- coding: utf-8 -*-
"""Step 1b: pHash + DINOv2 near-duplicate scan (train/test contamination control).

Outputs (<data_root>/dedup/):
  phash.jsonl            64-bit pHash per image (resumable)
  pairs_phash.csv        candidate pairs with Hamming distance <= --phash-th (cross-dataset pairs flagged)
  pairs_dino.csv         (with --dino) candidate pairs with cosine similarity >= --dino-th
  exclusions_auto.csv    automatic exclusions: identical sha256, or pHash distance 0 across train/test;
                         the test-side member is kept and the training-side member is removed
  adjudication_todo.csv  remaining candidate pairs, to be adjudicated manually / by preregistered rules
                         (fill in the keep/drop column)
  report.md              summary

pHash candidate retrieval uses a complete 9-band index (band widths PHASH_BAND_WIDTHS, 64 bits in total):
any pair with Hamming distance <= 8 shares at least one identical band.
Usage:
  python -X utf8 s02_dedup.py --data-root <ROOT> --prereg ../configs/prereg.yaml [--dino]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from common import read_jsonl, setup_logger, sha256_file, write_jsonl

LOG = setup_logger("s02")
PHASH_CACHE_VERSION = "imagehash.phash-default-v1"
DINO_CACHE_VERSION = "dinov2_vits14-square224-imagenet-v2"
PHASH_BAND_WIDTHS = (8, 7, 7, 7, 7, 7, 7, 7, 7)


def compute_phash_one(row: dict) -> dict:
    import imagehash
    try:
        with Image.open(row["path"]) as image:
            phash = str(imagehash.phash(image))
        return {"image_id": row["image_id"], "source_sha256": row.get("sha256"),
                "algorithm": PHASH_CACHE_VERSION, "phash": phash, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"image_id": row["image_id"], "source_sha256": row.get("sha256"),
                "algorithm": PHASH_CACHE_VERSION, "phash": None, "error": repr(exc)}


def compute_phashes(rows, cache: Path, workers: int = 1):
    current = {r["image_id"]: r for r in rows}
    done = {}
    if cache.exists():
        for r in read_jsonl(cache):
            image_id = r["image_id"]
            if (image_id in current
                    and r.get("source_sha256") == current[image_id].get("sha256")
                    and r.get("algorithm", PHASH_CACHE_VERSION) == PHASH_CACHE_VERSION):
                done[image_id] = r["phash"]
    todo = [r for r in rows if r["image_id"] not in done]
    LOG.info(f"pHash: {len(done)} cached, {len(todo)} to compute")
    cache.parent.mkdir(parents=True, exist_ok=True)
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(compute_phash_one, todo, chunksize=8)
    else:
        results = map(compute_phash_one, todo)
    with open(cache, "a", encoding="utf-8") as f:
        for index, record in enumerate(tqdm(results, total=len(todo), unit="img"), 1):
            if not record["error"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 100 == 0:
                    f.flush()
                done[record["image_id"]] = record["phash"]
            else:
                LOG.warning(f"pHash failed {record['image_id']}: {record['error']}")
        f.flush()
    if executor is not None:
        executor.shutdown()
    missing = sorted(set(current) - set(done))
    if missing:
        raise RuntimeError(
            f"Incomplete pHash coverage: missing {len(missing)}/{len(rows)} (e.g. {missing[0]})"
        )
    return done  # image_id -> hex str


def phash_pairs(hashes: dict, phash_th: int):
    """Complete 9-band candidate retrieval; every pair with Hamming<=8 shares at least one band."""
    if phash_th > len(PHASH_BAND_WIDTHS) - 1:
        raise ValueError("The 9-band index guarantees completeness only for Hamming<=8")
    ints = {k: int(v, 16) for k, v in hashes.items()}
    bands = defaultdict(list)
    for k, v in ints.items():
        shift = 0
        for band_index, width in enumerate(PHASH_BAND_WIDTHS):
            bands[(band_index, (v >> shift) & ((1 << width) - 1))].append(k)
            shift += width
    seen, pairs = set(), []
    for bucket in bands.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = sorted((bucket[i], bucket[j]))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                d = bin(ints[a] ^ ints[b]).count("1")
                if d <= phash_th:
                    pairs.append((a, b, d))
    return pairs


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def stable_pair_id(a: str, b: str) -> str:
    first, second = canonical_pair(a, b)
    return hashlib.sha1(f"{first}\0{second}".encode("utf-8")).hexdigest()[:12]


def load_previous_adjudications(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    previous = {}
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            previous[canonical_pair(row["a"], row["b"])] = {
                "decision": (row.get("decision(keep/drop_a/drop_b)") or "").strip().lower(),
                "rationale": (row.get("rationale") or "").strip(),
            }
    return previous


def backup_adjudication(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"adjudication_todo.{timestamp}.csv"
    shutil.copy2(path, backup)
    return backup


def load_dino_shards(rows: list[dict], shard_dir: Path) -> dict[str, np.ndarray]:
    current = {row["image_id"]: row.get("sha256") for row in rows}
    cached = {}
    if not shard_dir.exists():
        return cached
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as data:
            if str(data["version"].item()) != DINO_CACHE_VERSION:
                continue
            for image_id, source_sha256, embedding in zip(
                    data["ids"].astype(str), data["source_sha256"].astype(str), data["emb"]):
                if image_id in current and current[image_id] == source_sha256:
                    cached[image_id] = embedding.astype(np.float32, copy=False)
    return cached


def save_dino_shard(shard_dir: Path, index: int,
                    records: list[dict], embeddings: np.ndarray) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    target = shard_dir / f"shard_{index:06d}.npz"
    tmp = target.with_suffix(".tmp")
    with open(tmp, "wb") as file:
        np.savez_compressed(
            file,
            version=np.asarray(DINO_CACHE_VERSION),
            ids=np.asarray([record["image_id"] for record in records]),
            source_sha256=np.asarray([record["sha256"] for record in records]),
            emb=embeddings.astype(np.float32, copy=False),
        )
    tmp.replace(target)


def mutual_knn_pairs(ids: list[str], emb: np.ndarray, th: float,
                     topk: int, step: int = 2048) -> list[tuple[str, str, float]]:
    n = len(ids)
    if n < 2:
        return []
    k = min(max(1, topk), n - 1)
    neighbor_indices = np.empty((n, k), dtype=np.int32)
    neighbor_scores = np.empty((n, k), dtype=np.float32)
    emb_t = emb.T
    for start in tqdm(range(0, n, step), unit="chunk"):
        stop = min(start + step, n)
        similarities = emb[start:stop] @ emb_t
        local = np.arange(stop - start)
        similarities[local, start + local] = -np.inf
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        scores = np.take_along_axis(similarities, indices, axis=1)
        order = np.argsort(scores, axis=1)[:, ::-1]
        neighbor_indices[start:stop] = np.take_along_axis(indices, order, axis=1)
        neighbor_scores[start:stop] = np.take_along_axis(scores, order, axis=1)

    neighbor_sets = [set(row.tolist()) for row in neighbor_indices]
    pairs = []
    for i in range(n):
        for position, j in enumerate(neighbor_indices[i]):
            j = int(j)
            similarity = float(neighbor_scores[i, position])
            if i < j and similarity >= th and i in neighbor_sets[j]:
                pairs.append((ids[i], ids[j], similarity))
    return pairs


def partitioned_mutual_knn_pairs(
    ids: list[str], emb: np.ndarray, training_side: np.ndarray, th: float,
    cross_topk: int = 5, same_topk: int = 1, step: int = 2048,
) -> list[tuple[str, str, float, str]]:
    """Mutual top-k across the training/non-training pools, mutual top-1 within the same side; returns the union."""
    n = len(ids)
    if n < 2:
        return []
    side = np.asarray(training_side, dtype=bool)
    if side.shape != (n,):
        raise ValueError("training_side shape mismatch")
    emb_t = emb.T

    def neighbors(cross: bool, topk: int):
        k = min(max(1, topk), n - 1)
        indices_all = np.full((n, k), -1, dtype=np.int32)
        scores_all = np.full((n, k), -np.inf, dtype=np.float32)
        for start in tqdm(range(0, n, step), unit="chunk"):
            stop = min(start + step, n)
            similarities = emb[start:stop] @ emb_t
            eligible = side[start:stop, None] != side[None, :] if cross else (
                side[start:stop, None] == side[None, :])
            local = np.arange(stop - start)
            eligible[local, start + local] = False
            similarities[~eligible] = -np.inf
            indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
            scores = np.take_along_axis(similarities, indices, axis=1)
            order = np.argsort(scores, axis=1)[:, ::-1]
            indices_all[start:stop] = np.take_along_axis(indices, order, axis=1)
            scores_all[start:stop] = np.take_along_axis(scores, order, axis=1)
        return indices_all, scores_all

    found = {}
    for relation, cross, topk in (
        (f"cross_pool_mutual_top{cross_topk}", True, cross_topk),
        (f"same_pool_mutual_top{same_topk}", False, same_topk),
    ):
        indices, scores = neighbors(cross, topk)
        neighbor_sets = [
            {int(index) for index, score in zip(row_indices, row_scores) if np.isfinite(score)}
            for row_indices, row_scores in zip(indices, scores)
        ]
        for i in range(n):
            for j, similarity in zip(indices[i], scores[i]):
                j = int(j)
                if j < 0 or not np.isfinite(similarity) or i >= j:
                    continue
                if float(similarity) >= th and i in neighbor_sets[j]:
                    key = canonical_pair(ids[i], ids[j])
                    candidate = (key[0], key[1], float(similarity), relation)
                    if key not in found or candidate[2] > found[key][2]:
                        found[key] = candidate
    return [found[key] for key in sorted(found)]


def dino_pairs(rows, cache_dir: Path, th: float, batch: int = 64,
               io_workers: int = 8, cross_topk: int = 5, same_topk: int = 1):
    import torch
    shard_dir = cache_dir / "dino_shards"
    cached = load_dino_shards(rows, shard_dir)
    todo = [row for row in rows if row["image_id"] not in cached]
    LOG.info(f"DINOv2: {len(cached)} cached, {len(todo)} to compute")

    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(device).eval()
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        existing_indices = [int(path.stem.split("_")[-1])
                            for path in shard_dir.glob("shard_*.npz")]
        shard_index = max(existing_indices, default=-1) + 1
        def load_tensor(row):
            try:
                with Image.open(row["path"]) as image:
                    return row, transform(image.convert("RGB")), None
            except Exception as exc:  # noqa: BLE001
                return row, None, repr(exc)

        with ThreadPoolExecutor(max_workers=max(1, io_workers)) as io_executor:
            with torch.no_grad():
                for start in tqdm(range(0, len(todo), batch), unit="batch"):
                    chunk = todo[start:start + batch]
                    tensors, valid = [], []
                    for row, tensor, error in io_executor.map(load_tensor, chunk):
                        if error is None:
                            tensors.append(tensor)
                            valid.append(row)
                        else:
                            LOG.error(f"DINO image read failed {row['image_id']}: {error}")
                    if not tensors:
                        continue
                    tensor = torch.stack(tensors).to(device)
                    features = torch.nn.functional.normalize(model(tensor), dim=1).cpu().numpy()
                    save_dino_shard(shard_dir, shard_index, valid, features)
                    for row, feature in zip(valid, features):
                        cached[row["image_id"]] = feature.astype(np.float32, copy=False)
                    shard_index += 1

    missing = [row["image_id"] for row in rows if row["image_id"] not in cached]
    if missing:
        raise RuntimeError(f"DINOv2 features missing for {len(missing)} images (e.g. {missing[0]})")
    ids = [row["image_id"] for row in rows]
    emb = np.stack([cached[image_id] for image_id in ids]).astype(np.float32, copy=False)
    training_side = np.asarray([row["pool"] in TRAIN_POOLS for row in rows], dtype=bool)
    return partitioned_mutual_knn_pairs(
        ids, emb, training_side, th=th, cross_topk=cross_topk, same_topk=same_topk)


TRAIN_POOLS = {"train_positive", "train_positive_optional", "train_negative"}


def build_exclusion_and_todo(pp: list[tuple[str, str, int]], rows: list[dict], meta: dict):
    def train_test_pair(a, b):
        pa, pb = meta[a]["pool"] in TRAIN_POOLS, meta[b]["pool"] in TRAIN_POOLS
        return pa != pb

    exclusion_map = {}
    todo = []
    protected = []
    for a, b, distance in pp:
        key = tuple(sorted((a, b)))
        if distance == 0 and train_test_pair(a, b):
            drop = a if meta[a]["pool"] in TRAIN_POOLS else b
            exclusion_map[key] = (a, b, distance, drop, "phash0_train_test")
        elif meta[a]["group_id"] == meta[b]["group_id"]:
            protected.append((a, b, distance, "same_group_split_protection"))
        else:
            todo.append((a, b, distance, "phash"))

    by_sha = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            by_sha[row["sha256"]].append(row["image_id"])
    for group in (group for group in by_sha.values() if len(group) > 1):
        keep = sorted(
            group,
            key=lambda image_id: (
                meta[image_id]["pool"] in TRAIN_POOLS,
                image_id,
            ),
        )[0]
        for duplicate in group:
            if duplicate == keep:
                continue
            key = tuple(sorted((keep, duplicate)))
            exclusion_map[key] = (keep, duplicate, 0, duplicate, "sha256_exact")

    excluded_keys = set(exclusion_map)
    todo = [item for item in todo if tuple(sorted(item[:2])) not in excluded_keys]
    protected = [item for item in protected if tuple(sorted(item[:2])) not in excluded_keys]
    exclusions = [exclusion_map[key] for key in sorted(exclusion_map)]
    return exclusions, protected, todo


def merge_dino_candidates(exclusions: list, protected: list, todo: list,
                          dino_pairs_found: list[tuple], meta: dict):
    excluded_keys = {tuple(sorted(item[:2])) for item in exclusions}
    protected_map = {tuple(sorted(item[:2])): item for item in protected}
    todo_map = {tuple(sorted(item[:2])): item for item in todo}
    for candidate in dino_pairs_found:
        a, b, similarity = candidate[:3]
        key = tuple(sorted((a, b)))
        if key in excluded_keys:
            continue
        if meta[a]["group_id"] == meta[b]["group_id"]:
            protected_map.setdefault(
                key, (a, b, f"{similarity:.4f}", "same_group_split_protection"))
            todo_map.pop(key, None)
            continue
        if key in todo_map:
            previous = todo_map[key]
            todo_map[key] = (
                previous[0], previous[1],
                f"phash={previous[2]};dino={similarity:.4f}", "phash+dino")
        else:
            todo_map[key] = (a, b, f"{similarity:.4f}", "dino")
    return ([protected_map[key] for key in sorted(protected_map)],
            [todo_map[key] for key in sorted(todo_map)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--phash-th", type=int, default=None,
                    help="Consistency check only; must equal prereg.yaml")
    ap.add_argument("--phash-workers", type=int, default=8)
    ap.add_argument("--dino", action="store_true")
    ap.add_argument("--dino-th", type=float, default=None,
                    help="Consistency check only; must equal prereg.yaml")
    ap.add_argument("--dino-batch", type=int, default=64)
    ap.add_argument("--dino-io-workers", type=int, default=8)
    ap.add_argument("--dino-topk", type=int, default=None,
                    help="Consistency check only; must equal mutual_topK in prereg.yaml")
    args = ap.parse_args()

    root = Path(args.data_root)
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))
    dedup_cfg = prereg["dedup"]
    phash_th = int(dedup_cfg["phash_hamming_max"])
    dino_th = float(dedup_cfg["dino_cosine_min"])
    retrieval = dedup_cfg["dino_retrieval"]
    if not isinstance(retrieval, dict):
        raise SystemExit("dedup.dino_retrieval must declare cross_pool and same_pool separately")
    cross_match = re.fullmatch(r"mutual_top(?P<k>\d+)", str(retrieval.get("cross_pool")))
    same_match = re.fullmatch(r"mutual_top(?P<k>\d+)", str(retrieval.get("same_pool")))
    if not cross_match or not same_match:
        raise SystemExit(f"Unsupported dedup.dino_retrieval={retrieval!r}")
    dino_cross_topk = int(cross_match.group("k"))
    dino_same_topk = int(same_match.group("k"))
    if args.phash_th is not None and args.phash_th != phash_th:
        raise SystemExit(f"--phash-th={args.phash_th} does not match prereg={phash_th}")
    if args.dino_th is not None and args.dino_th != dino_th:
        raise SystemExit(f"--dino-th={args.dino_th} does not match prereg={dino_th}")
    if args.dino_topk is not None and args.dino_topk != dino_cross_topk:
        raise SystemExit(f"--dino-topk={args.dino_topk} does not match prereg cross_pool={dino_cross_topk}")
    rows = list(read_jsonl(root / "registry" / "registry.jsonl"))
    missing_sha = [r["image_id"] for r in rows if not r.get("sha256")]
    if missing_sha:
        raise SystemExit(
            f"Registry has {len(missing_sha)} rows without sha256 (e.g. {missing_sha[0]}); "
            "rerun s01 without --no-hash before the production dedup run")
    meta = {r["image_id"]: r for r in rows}
    out = root / "dedup"
    out.mkdir(parents=True, exist_ok=True)

    hashes = compute_phashes(rows, out / "phash.jsonl", workers=max(1, args.phash_workers))
    pp = phash_pairs(hashes, phash_th)
    LOG.info(f"pHash candidate pairs (<= {phash_th}): {len(pp)}")

    def cross(a, b):
        return meta[a]["source_dataset"] != meta[b]["source_dataset"]

    def train_test_pair(a, b):
        pa, pb = meta[a]["pool"] in TRAIN_POOLS, meta[b]["pool"] in TRAIN_POOLS
        return pa != pb

    with open(out / "pairs_phash.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "hamming", "cross_dataset", "train_test", "same_group"])
        for a, b, d in sorted(pp, key=lambda t: t[2]):
            w.writerow([a, b, d, int(cross(a, b)), int(train_test_pair(a, b)),
                        int(meta[a]["group_id"] == meta[b]["group_id"])])

    pair_summary = defaultdict(
        lambda: {"pairs": 0, "train_test": 0, "hamming0": 0, "cross_group": 0})
    for a, b, distance in pp:
        key = tuple(sorted((meta[a]["source_dataset"], meta[b]["source_dataset"])))
        pair_summary[key]["pairs"] += 1
        pair_summary[key]["train_test"] += int(train_test_pair(a, b))
        pair_summary[key]["hamming0"] += int(distance == 0)
        pair_summary[key]["cross_group"] += int(meta[a]["group_id"] != meta[b]["group_id"])
    with open(out / "pair_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_a", "dataset_b", "pairs", "train_test", "hamming0", "cross_group"])
        for (dataset_a, dataset_b), counts in sorted(
                pair_summary.items(), key=lambda item: (-item[1]["pairs"], item[0])):
            writer.writerow([dataset_a, dataset_b, counts["pairs"],
                             counts["train_test"], counts["hamming0"], counts["cross_group"]])

    exclusions, protected, todo = build_exclusion_and_todo(pp, rows, meta)

    dp = []
    if args.dino:
        dp = dino_pairs(rows, out, dino_th, batch=max(1, args.dino_batch),
                        io_workers=max(1, args.dino_io_workers),
                        cross_topk=dino_cross_topk, same_topk=dino_same_topk)
        LOG.info(f"DINOv2 candidate pairs (>= {dino_th}): {len(dp)}")
        with open(out / "pairs_dino.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "cosine", "retrieval", "cross_dataset", "train_test"])
            for a, b, s, relation in sorted(dp, key=lambda t: -t[2]):
                w.writerow([a, b, f"{s:.4f}", relation,
                            int(cross(a, b)), int(train_test_pair(a, b))])
        protected, todo = merge_dino_candidates(exclusions, protected, todo, dp, meta)

    with open(out / "exclusions_auto.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "dist", "drop_image_id", "reason"])
        w.writerows(exclusions)
    with open(out / "pairs_group_protected.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "score", "reason"])
        w.writerows(protected)
    adjudication_path = out / "adjudication_todo.csv"
    previous = load_previous_adjudications(adjudication_path)
    backup_path = backup_adjudication(adjudication_path)
    preserved = 0
    with open(adjudication_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["pair_id", "a", "b", "score", "channel",
                  "decision(keep/drop_a/drop_b)", "rationale"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a, b, score, channel in todo:
            first, second = canonical_pair(a, b)
            old = previous.get((first, second), {})
            preserved += int(bool(old.get("decision")))
            writer.writerow({
                "pair_id": stable_pair_id(first, second),
                "a": first,
                "b": second,
                "score": score,
                "channel": channel,
                "decision(keep/drop_a/drop_b)": old.get("decision", ""),
                "rationale": old.get("rationale", ""),
            })

    n_tt = sum(1 for a, b, d in pp if train_test_pair(a, b))
    top_pairs = sorted(pair_summary.items(), key=lambda item: (-item[1]["pairs"], item[0]))[:10]
    top_lines = "\n".join(
        f"  - {a} × {b}: {counts['pairs']} (cross-group {counts['cross_group']}, "
        f"train-test cross-pool {counts['train_test']}, hamming=0 {counts['hamming0']})"
        for (a, b), counts in top_pairs)
    (out / "report.md").write_text(
        "# Deduplication scan report\n\n"
        f"- Registered images: {len(rows)}\n"
        f"- pHash candidate pairs (<= {phash_th}): {len(pp)}, of which train-test cross-pool {n_tt}\n"
        f"- DINOv2 cross-pool mutual top-{dino_cross_topk} / same-pool mutual top-{dino_same_topk} "
        f"candidate pairs: {len(dp) if args.dino else 'disabled'}\n"
        f"- Automatic exclusions (SHA256 exact, or pHash0 across train-test): {len(exclusions)}\n"
        f"- Same-group near-duplicates (covered by group split, no pairwise adjudication needed): {len(protected)}\n"
        f"- To adjudicate: {len(todo)} ({preserved} existing decisions preserved; backup {backup_path or 'no previous file'})\n"
        "- Dataset pairs with the most candidates:\n"
        f"{top_lines}\n",
        encoding="utf-8")
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": sha256_file(root / "registry" / "registry.jsonl"),
        "prereg_sha256": sha256_file(Path(args.prereg)),
        "phash_cache_version": PHASH_CACHE_VERSION,
        "phash_index": "complete_9band_hamming_le_8",
        "dino_enabled": bool(args.dino),
        "dino_cache_version": DINO_CACHE_VERSION,
        "dino_retrieval": retrieval,
        "counts": {"registry": len(rows), "phash_pairs": len(pp), "dino_pairs": len(dp),
                   "auto_exclusions": len(exclusions), "protected": len(protected), "todo": len(todo),
                   "preserved_decisions": preserved},
        "adjudication_backup": str(backup_path) if backup_path else None,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(f"done -> {out}/report.md; {len(exclusions)} automatic exclusions, {len(todo)} to adjudicate")


if __name__ == "__main__":
    main()
