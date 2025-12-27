#!/usr/bin/env python3
"""
PrismLayersPro bucketer with non-overlap + alpha-weighted representative export.

기능 요약
- HuggingFace artplus/PrismLayersPro 전체 split을 불러와서 버킷에 맞게 리사이즈.
- base, composite(all layers), non-overlap fg, representative fg(알파합 가중 랜덤) 저장.
- whole 이미지는 저장하지 않음.
- 단일 프로세스, LANCZOS 리사이즈, 단순 비겹침 선택(마스크 기반).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Bucket 규칙 (prepare_rgba_buckets와 호환)
MAX_SIDE = 1408
MAX_PIXELS = 1408 * 768
MULTIPLE = 64
MIN_BUCKET_SIDE = MULTIPLE
FILTER_MIN_SIDE = 384
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
        layer_img = layer_img.resize(expected_size, Image.LANCZOS)
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


def find_nonoverlap(idxs: Iterable[int], masks: Dict[int, np.ndarray]) -> List[int]:
    covered = np.zeros_like(next(iter(masks.values())), dtype=bool) if masks else None
    if covered is None:
        return []
    picks: List[int] = []
    for idx in reversed(list(idxs)):  # 뒤에서 앞으로 훑어 겹치지 않는 것만 선택
        m = masks.get(idx)
        if m is None:
            continue
        if not np.any(m & covered):
            picks.append(idx)
            covered |= m
    picks.reverse()
    return picks


def process_sample(sample: Dict, sample_idx: int, output_root: Path, split: str, rng: np.random.Generator) -> Optional[Dict]:
    sample_id = sample.get("id") or sample.get("sample_id") or f"{split}_{sample_idx:06d}"
    file_id = f"{split}_{sample_id}"
    base_img = decode_image_or_passthrough(sample.get("base_image"))
    if base_img is None:
        print(f"[skip] {sample_id}: missing base")
        return None

    assignment, reason = bucket_assignment(base_img.size)
    if assignment is None:
        print(f"[skip] {sample_id}: {reason}")
        return None
    bucket_name, bucket_dims = assignment
    # Always write under a unified train bucket path regardless of dataset split
    bucket_dir = output_root / "train" / bucket_name

    # Build full-canvas layers + masks
    layers: List[Tuple[int, Image.Image]] = []
    masks: Dict[int, np.ndarray] = {}
    alpha_sums: Dict[int, int] = {}
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
        layers.append((i, canvas))
        mask = np.asarray(canvas, dtype=np.uint8)[..., 3] > 0
        if mask.any():
            masks[i] = mask
            alpha_sums[i] = int(mask.sum())

    # Composite all layers
    composite_all = base_img.convert("RGBA")
    for _, canvas in layers:
        composite_all = Image.alpha_composite(composite_all, canvas)

    # Non-overlap picks and fg
    remaining = [idx for idx, _ in layers if idx in masks]
    non_overlap_idxs = find_nonoverlap(remaining, masks)
    fg_non_overlap = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    for idx, canvas in layers:
        if idx in non_overlap_idxs:
            fg_non_overlap = Image.alpha_composite(fg_non_overlap, canvas)

    # Representative (alpha-weighted)
    rep_idx = None
    rep_fg = None
    if non_overlap_idxs:
        weights = np.array([alpha_sums[i] for i in non_overlap_idxs], dtype=np.float64)
        if weights.sum() > 0:
            rep_idx = int(rng.choice(non_overlap_idxs, p=weights / weights.sum()))
            rep_canvas = next(c for i, c in layers if i == rep_idx)
            rep_fg = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            rep_fg = Image.alpha_composite(rep_fg, rep_canvas)

    # Save (no whole)
    base_path = bucket_dir / f"{file_id}_base.png"
    comp_path = bucket_dir / f"{file_id}_composite.png"
    nonoverlap_path = bucket_dir / f"{file_id}_fg_non_overlap.png"
    rep_path = bucket_dir / f"{file_id}_rep.png" if rep_fg is not None else None

    save_rgba(base_img, base_path, bucket_dims)
    save_rgba(composite_all, comp_path, bucket_dims)
    save_rgba(fg_non_overlap, nonoverlap_path, bucket_dims)
    if rep_fg is not None and rep_path is not None:
        save_rgba(rep_fg, rep_path, bucket_dims)

    entry = {
        "id": sample_id,
        "split": split,
        "bucket": bucket_name,
        "bucket_dims": list(bucket_dims),
        "base_path": str(base_path.relative_to(output_root)),
        "composite_path": str(comp_path.relative_to(output_root)),
        "fg_non_overlap_path": str(nonoverlap_path.relative_to(output_root)),
        "rep_path": str(rep_path.relative_to(output_root)) if rep_path else None,
        "rep_layer_idx": rep_idx,
        "non_overlap_layer_indices": non_overlap_idxs,
        "original_size": list(base_img.size),
    }
    return entry


DEFAULT_CACHE = Path("/mnt/local")


def main():
    parser = argparse.ArgumentParser(description="Bucket PrismLayersPro (base/composite/non-overlap/rep).")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for buckets and manifest.")
    parser.add_argument("--splits", type=str, default="all", help="Comma-separated splits to process, or 'all'.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for quick tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"HF cache dir (default: {DEFAULT_CACHE})",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Total number of nodes/processes for sharding.")
    parser.add_argument("--rank", type=int, default=0, help="This worker rank (0-indexed).")
    args = parser.parse_args()

    if args.world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if not (0 <= args.rank < args.world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 기본 캐시 경로도 동일하게 맞춰둔다 (환경변수 미설정 시)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    ds = load_dataset("artplus/PrismLayersPro", cache_dir=str(cache_dir))

    if args.splits.strip().lower() == "all":
        split_names = list(ds.keys())
    else:
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    output_root = Path(args.output_root)
    manifest: List[Dict] = []
    rng = np.random.default_rng(args.seed)

    for split in split_names:
        if split not in ds:
            print(f"[warn] split {split} not found; skipping.")
            continue
        split_ds = ds[split]
        limit = len(split_ds) if args.max_samples is None else min(len(split_ds), args.max_samples)
        indices = [i for i in range(limit) if i % args.world_size == args.rank]
        for idx in tqdm(indices, desc=f"{split} [rank {args.rank}/{args.world_size}]"):
            entry = process_sample(split_ds[idx], idx, output_root=output_root, split=split, rng=rng)
            if entry:
                manifest.append(entry)

    manifest_path = output_root / "metadata" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved {len(manifest)} entries. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

