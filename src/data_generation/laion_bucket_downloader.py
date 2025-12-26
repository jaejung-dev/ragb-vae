#!/usr/bin/env python3
"""
Download a subset of laion/laion2B-en-aesthetic, filter by min side,
bucket-resize (same rules as prepare_rgba_buckets), and save RGB images
as PNG into bucketed folders with a manifest.

Notes:
- Dataset is gated; you must have accepted access at HF.
- Uses streaming iteration to avoid full download.
- Skips samples with min(width, height) < --min-side (default 512).
- Saves PNG (lossless).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Bucket rules (aligned with prepare_rgba_buckets.py)
MAX_SIDE = 1408
MAX_PIXELS = 1408 * 768
MULTIPLE = 64
MIN_BUCKET_SIDE = MULTIPLE
DEFAULT_MIN_SIDE = 512  # user requirement: skip < 512
FILTER_MAX_AR = 2.0  # same as prepare_rgba_buckets.py


def round_to_multiple(value: float, multiple: int = MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def bucket_for_size(width: int, height: int) -> Tuple[int, int]:
    scale_side = min(MAX_SIDE / width, MAX_SIDE / height, 1.0)
    scale_pixels = min(math.sqrt(MAX_PIXELS / float(width * height)), 1.0)
    scale = min(scale_side, scale_pixels)
    sw, sh = width * scale, height * scale
    bucket_w = max(round_to_multiple(sw), MIN_BUCKET_SIDE)
    bucket_h = max(round_to_multiple(sh), MIN_BUCKET_SIDE)
    return int(bucket_w), int(bucket_h)


def bucket_assignment(size: Tuple[int, int], min_side: int = DEFAULT_MIN_SIDE):
    w, h = size
    if w <= 0 or h <= 0:
        return None, "invalid_dimensions"
    if min(w, h) < min_side:
        return None, f"too_small(<{min_side})"
    larger = max(w, h)
    smaller = min(w, h)
    if smaller <= 0 or larger / smaller >= FILTER_MAX_AR:
        return None, f"extreme_aspect_ratio(>={FILTER_MAX_AR})"
    # Use the same aspect/scale logic as prepare_rgba_buckets, but rely on MAX_SIDE/MAX_PIXELS
    bucket_dims = bucket_for_size(w, h)
    bucket_key = f"w{bucket_dims[0]}-h{bucket_dims[1]}"
    return (bucket_key, bucket_dims), None


def safe_image_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def download_image(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        return img.convert("RGB")
    except Exception:
        return None


def process_row(
    row: Dict,
    output_root: Path,
    min_side: int,
) -> Optional[Dict]:
    url = row.get("URL") or row.get("url")
    if not url:
        return None
    img = download_image(url)
    if img is None:
        return None
    assignment, reason = bucket_assignment(img.size, min_side=min_side)
    if assignment is None:
        return None
    bucket_name, bucket_dims = assignment
    bucket_dir = output_root / "train" / bucket_name
    bucket_dir.mkdir(parents=True, exist_ok=True)

    img_id = safe_image_id(url)
    out_path = bucket_dir / f"{img_id}.png"
    img.resize(bucket_dims, resample=Image.LANCZOS).save(out_path, "PNG")

    return {
        "url": url,
        "id": img_id,
        "bucket": bucket_name,
        "bucket_dims": bucket_dims,
        "original_size": img.size,
        "path": str(out_path.relative_to(output_root)),
    }


def main():
    parser = argparse.ArgumentParser(description="Bucket laion2B-en-aesthetic subset into RGB buckets.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--max-samples", type=int, default=1_000_000, help="Max number of kept images.")
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE, help="Skip images smaller than this.")
    parser.add_argument("--num-workers", type=int, default=16, help="Download/resize thread pool size.")
    parser.add_argument("--hf-cache", type=Path, default=None, help="Optional HF cache dir (sets HF_HOME/HF_DATASETS_CACHE).")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = str(args.hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(args.hf_cache)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # streaming prevents local full download
    ds = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)

    manifest = []
    futures = []
    kept = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex, tqdm(
        total=args.max_samples, unit="img", desc="kept"
    ) as pbar:
        for row in ds:
            if kept >= args.max_samples:
                break
            fut = ex.submit(
                process_row,
                row,
                output_root,
                args.min_side,
            )
            futures.append(fut)
            # throttle queue to avoid unbounded memory
            if len(futures) >= args.num_workers * 4:
                for f in as_completed(futures):
                    res = f.result()
                    if res:
                        manifest.append(res)
                        kept += 1
                        pbar.update(1)
                        if kept >= args.max_samples:
                            break
                futures = []
            if kept >= args.max_samples:
                break

        # drain remaining
        for f in as_completed(futures):
            res = f.result()
            if res:
                manifest.append(res)
                kept += 1
                pbar.update(1)
                if kept >= args.max_samples:
                    break

    manifest_path = output_root / "metadata" / "laion_aesthetic_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. kept={kept}, manifest={manifest_path}")


if __name__ == "__main__":
    main()

