#!/usr/bin/env python3
"""Download COCO images from coco_url (32 threads, retries, PIL verification).

Usage:
  python scripts/fetch_images.py --subset data/annotations/instances_train2017.stripped.json --dest data/images/train2017 --jobs 32
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("fetch_images")


def download_one(entry, dest: Path, timeout: int = 30, retries: int = 3):
    file_name = entry["file_name"]
    url = entry.get("coco_url")
    out = dest / file_name
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        try:
            with Image.open(out) as im:
                im.verify()
            return file_name, "skipped", None
        except Exception:
            out.unlink(missing_ok=True)

    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "openworld-fetch/1.0"})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
            with Image.open(BytesIO(data)) as im:
                im.verify()
            with open(out, "wb") as f:
                f.write(data)
            return file_name, "ok", None
        except Exception as e:
            last_err = e
    return file_name, "failed", str(last_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="stripped COCO json with coco_url")
    ap.add_argument("--dest", required=True, help="destination image folder")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    with open(args.subset) as f:
        data = json.load(f)
    entries = data["images"]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(download_one, e, dest) for e in entries]
        for fut in as_completed(futures):
            name, status, err = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append((name, err))
    log.info(
        f"done: ok={ok} skipped={skipped} failed={failed} out of {len(entries)}"
    )
    if failures:
        with open(dest.parent / "fetch_failures.log", "w") as f:
            for name, err in failures:
                f.write(f"{name}\t{err}\n")
        log.warning(f"wrote {len(failures)} failures to fetch_failures.log")


if __name__ == "__main__":
    main()
