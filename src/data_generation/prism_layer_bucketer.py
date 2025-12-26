#!/usr/bin/env python3
"""
Minimal PrismLayersReal bucketer for RGBA-VAE.

Loads HuggingFace dataset samples (base + whole + layer_n + layer_n_box),
restores each cropped layer onto a full-size transparent canvas, assigns a
bucket (wXXX-hYYY) based on base size, resizes to that bucket, and saves:
  - base (background) RGBA
  - whole (full composite) RGBA
  - each layer_n (full-canvas RGBA)
Writes a simple manifest with relative paths.

This is a trimmed, single-process variant inspired by prepare_rgba_buckets.py.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image

# Bucket rules (aligned with prepare_rgba_buckets defaults)
MAX_SIDE = 1408
MAX_PIXELS = 1408 * 768
MULTIPLE = 32
MIN_BUCKET_SIDE = MULTIPLE
FILTER_MIN_SIDE = 256
FILTER_MAX_AR = 2.3


def decode_image_or_passthrough(val):
    """Handle PIL.Image or base64 data URI -> RGBA PIL.Image."""
    if val is None or val == "":
        return None
    if isinstance(val, Image.Image):
        return val.convert("RGBA")
    b64_str = val
    if isinstance(b64_str, bytes):
        b64_str = b64_str.decode("utf-8")
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def layer_to_full_canvas(layer_val, box, canvas_size):
    """Place cropped layer onto a transparent canvas matching the base size."""
    layer_img = decode_image_or_passthrough(layer_val)
    if layer_img is None:
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    expected_size = (x1 - x0, y1 - y0)
    if layer_img.size != expected_size:
        print(f"Warning: layer size {layer_img.size} != box size {expected_size}")
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.paste(layer_img, (x0, y0), layer_img)
    return canvas


def round_to_multiple(value: float, multiple: int = MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def should_exclude_size(width: int, height: int) -> Optional[str]:
    smaller = min(width, height)
    larger = max(width, height)
    if smaller < FILTER_MIN_SIDE:
        return "too_small"
    if larger / max(1, smaller) >= FILTER_MAX_AR:
        return "extreme_aspect_ratio"
    return None


def bucket_for_size(width: int, height: int) -> Tuple[int, int]:
    scale_side = min(MAX_SIDE / width, MAX_SIDE / height, 1.0)
    scale_pixels = min(math.sqrt(MAX_PIXELS / float(width * height)), 1.0)
    scale = min(scale_side, scale_pixels)
    sw, sh = width * scale, height * scale
    bucket_w = max(round_to_multiple(sw), MIN_BUCKET_SIDE)
    bucket_h = max(round_to_multiple(sh), MIN_BUCKET_SIDE)
    return int(bucket_w), int(bucket_h)


def bucket_assignment(size: Tuple[int, int]) -> Tuple[Optional[Tuple[str, Tuple[int, int]]], Optional[str]]:
    w, h = size
    if w <= 0 or h <= 0:
        return None, "invalid_dimensions"
    reason = should_exclude_size(w, h)
    if reason:
        return None, reason
    bucket_dims = bucket_for_size(w, h)
    bucket_key = f"w{bucket_dims[0]}-h{bucket_dims[1]}"
    return (bucket_key, bucket_dims), None


def save_rgba(img: Image.Image, path: Path, size: Tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGBA").resize(size, resample=Image.LANCZOS).save(path)


def process_sample(
    sample: Dict,
    sample_idx: int,
    output_root: Path,
    split: str = "train",
) -> Optional[Dict]:
    sample_id = sample.get("id") or sample.get("sample_id") or f"sample_{sample_idx:06d}"
    base_img = decode_image_or_passthrough(sample.get("base_image"))
    whole_img = decode_image_or_passthrough(sample.get("whole_image"))
    if base_img is None or whole_img is None:
        print(f"[skip] {sample_id}: missing base or whole")
        return None

    assignment, reason = bucket_assignment(base_img.size)
    if assignment is None:
        print(f"[skip] {sample_id}: {reason}")
        return None
    bucket_name, bucket_dims = assignment

    bucket_dir = output_root / split / bucket_name
    base_path = bucket_dir / f"{sample_id}_base.png"
    whole_path = bucket_dir / f"{sample_id}_whole.png"

    save_rgba(base_img, base_path, bucket_dims)
    save_rgba(whole_img, whole_path, bucket_dims)

    layer_paths: List[str] = []
    layer_count = int(sample.get("layer_count") or 0)
    for i in range(layer_count):
        base_key = f"layer_{i:02}"
        img_key = f"{base_key}_image" if f"{base_key}_image" in sample else base_key
        box_key = f"{base_key}_box"
        layer_val = sample.get(img_key)
        box = sample.get(box_key, [0, 0, 0, 0])
        canvas = layer_to_full_canvas(layer_val, box, base_img.size)
        if canvas is None:
            continue
        layer_path = bucket_dir / f"{sample_id}_{base_key}.png"
        save_rgba(canvas, layer_path, bucket_dims)
        layer_paths.append(str(layer_path.relative_to(output_root)))

    entry = {
        "id": sample_id,
        "split": split,
        "bucket": bucket_name,
        "bucket_dims": list(bucket_dims),
        "base_path": str(base_path.relative_to(output_root)),
        "whole_path": str(whole_path.relative_to(output_root)),
        "layer_paths": layer_paths,
        "original_size": list(base_img.size),
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description="Bucket PrismLayersReal samples (base/whole/layers).")
    parser.add_argument("--output-root", type=Path, required=True, help="Where to write buckets.")
    parser.add_argument("--split", type=str, default="train", help="Split name for output path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests.")
    args = parser.parse_args()

    print("Loading dataset artplus/PrismLayersReal ...")
    ds = load_dataset("artplus/PrismLayersReal", split="train")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    total = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    for idx in range(total):
        entry = process_sample(ds[idx], idx, output_root=output_root, split=args.split)
        if entry:
            manifest.append(entry)

    manifest_path = output_root / "metadata" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved {len(manifest)} samples. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

