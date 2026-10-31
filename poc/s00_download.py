# -*- coding: utf-8 -*-
"""Step 0: fetch data according to configs/datasets.yaml, archive license pages, write a sha256 manifest.

Only the automatable parts are automated:
- access: direct  -> download each URL to <data_root>/raw/<id>/
- access: repo    -> git clone --depth 1 to <data_root>/raw/<id>/repo/ (cloud-drive links still need manual download)
- access: form / email / manual / pending-verification -> print instructions, make no requests
- license_page is always archived to <data_root>/licenses/<id>.html if possible (failure only warns)

Usage:
  python -X utf8 s00_download.py --config ../configs/datasets.yaml --data-root <DATA_ROOT> [--only id1,id2]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from common import md5_file, setup_logger, sha256_file

LOG = setup_logger("s00")
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def fetch_to(urls: str | list[str], dest: Path, referer: str = "", retries: int = 4) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info(f"  already exists, skipping: {dest.name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    candidates = [urls] if isinstance(urls, str) else list(urls)
    last_error = None
    for url in candidates:
        for attempt in range(1, retries + 1):
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = dict(UA)
            if referer:
                headers["Referer"] = referer
            if existing:
                headers["Range"] = f"bytes={existing}-"
            try:
                with requests.get(url, headers=headers, stream=True,
                                  timeout=(30, 300), allow_redirects=True) as r:
                    if r.status_code == 416 and existing:
                        tmp.rename(dest)
                        LOG.info(f"  server confirms partial file is complete: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
                        return True
                    r.raise_for_status()
                    append = existing > 0 and r.status_code == 206
                    mode = "ab" if append else "wb"
                    if existing and not append:
                        LOG.info(f"  server does not support Range, restarting download: {dest.name}")
                    with open(tmp, mode) as f:
                        for chunk in r.iter_content(1 << 18):
                            if chunk:
                                f.write(chunk)
                tmp.rename(dest)
                LOG.info(f"  download complete: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
                return True
            except Exception as e:  # noqa: BLE001 - bounded retries, partial file kept for resume
                last_error = e
                LOG.warning(
                    f"  download failed attempt={attempt}/{retries} url={url}: {e}; "
                    f"partial file kept {tmp.stat().st_size/1e6:.1f} MB" if tmp.exists()
                    else f"  download failed attempt={attempt}/{retries} url={url}: {e}"
                )
                if attempt < retries:
                    time.sleep((1, 3, 10)[min(attempt - 1, 2)])
    LOG.warning(f"  all official sources failed: {dest.name}; last_error={last_error}")
    return False


def archive_license(ds_id: str, url: str, lic_dir: Path) -> None:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        out = lic_dir / f"{ds_id}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"<!-- archived {datetime.now(timezone.utc).isoformat()} from {url} -->\n"
        out.write_text(stamp + r.text, encoding="utf-8", errors="replace")
        LOG.info(f"  license page archived: {out.name}")
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"  license page archiving failed {url}: {e}")


def git_clone(repo: str, dest: Path) -> bool:
    if dest.exists():
        LOG.info(f"  repository already exists, skipping clone: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)],
                       check=True, capture_output=True, text=True)
        LOG.info(f"  clone complete: {repo}")
        return True
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"  clone failed {repo}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", default="", help="Comma-separated subset of dataset ids")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.data_root)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    manual_todo, failed, ok = [], [], []
    checksums = {}

    for ds_id, ds in cfg["datasets"].items():
        if only and ds_id not in only:
            continue
        LOG.info(f"== {ds_id} [{ds.get('access')}] {ds.get('name', '')}")
        raw = root / "raw" / ds_id

        if ds.get("license_page"):
            archive_license(ds_id, ds["license_page"], root / "licenses")

        access = ds.get("access")
        if access == "direct":
            good = True
            for item in ds.get("downloads", []):
                dest = raw / item["filename"]
                urls = item.get("urls", item.get("url"))
                downloaded = fetch_to(urls, dest, referer=item.get("referer", ds.get("url", "")))
                good = downloaded and good
                if dest.exists():
                    actual_sha256 = sha256_file(dest)
                    actual_md5 = md5_file(dest) if item.get("md5") else None
                    verified = True
                    if item.get("sha256") and actual_sha256.lower() != item["sha256"].lower():
                        LOG.warning(f"  SHA256 mismatch: {dest.name}")
                        verified = False
                    if item.get("md5") and actual_md5.lower() != item["md5"].lower():
                        LOG.warning(f"  MD5 mismatch: {dest.name}")
                        verified = False
                    good = good and verified
                    checksums[f"{ds_id}/{item['filename']}"] = {
                        "sha256": actual_sha256,
                        "md5": actual_md5,
                        "expected_sha256": item.get("sha256"),
                        "expected_md5": item.get("md5"),
                        "source_checksum_verified": verified,
                    }
            (ok if good else failed).append(ds_id)
        elif access == "repo":
            good = git_clone(ds["url"], raw / "repo")
            note = ds.get("manual_note")
            if note:
                manual_todo.append((ds_id, note))
            (ok if good else failed).append(ds_id)
        else:
            manual_todo.append((ds_id, ds.get("manual_note", ds.get("url", "see yaml entry"))))

    if checksums:
        cs = root / "raw" / "checksums.json"
        old = json.loads(cs.read_text(encoding="utf-8")) if cs.exists() else {}
        old = {k: ({"sha256": v} if isinstance(v, str) else v) for k, v in old.items()}
        old.update(checksums)
        cs.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info(f"sha256 manifest updated: {cs}")

    LOG.info("---------------- summary ----------------")
    LOG.info(f"automated: {', '.join(ok) if ok else '(none)'}")
    if failed:
        LOG.warning(f"automatic fetch failed (retry or download manually): {', '.join(failed)}")
    if manual_todo:
        LOG.info("manual action required (application / form / cloud drive):")
        for ds_id, note in manual_todo:
            LOG.info(f"  - {ds_id}: {note}")


if __name__ == "__main__":
    main()
