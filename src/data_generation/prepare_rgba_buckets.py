#!/usr/bin/env python3
"""
Prepare RGBA component + composite pairs for RGBA-VAE training using resolution buckets.

Each saved sample now contains:
- Component RGBA layer (resized to bucket dims)
- Corresponding composite RGBA (same sample, same bucket)

Counts are based on number of components (train/train_count, val/val_count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from tqdm import tqdm

# Defaults align with previous data pipelines
DEFAULT_RENDERED_ROOT = Path("/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered")
DEFAULT_OUTPUT_ROOT = Path("/home/ubuntu/qwen-image-layered/data/rgba_layers")
DEFAULT_VALIDATION_LIST = Path("/home/ubuntu/ragb-vae/validation_list.txt")

# Lower bounds to keep each resized sample <=1.1M pixels (lighter batches).
MAX_SIDE = 1408
MAX_PIXELS = 1408 * 768
MULTIPLE = 32
MIN_BUCKET_SIDE = MULTIPLE
FILTER_MIN_SIDE = 256
FILTER_MAX_AR = 2.3
BACKGROUND_VISIBILITY_THRESHOLD = 0.01


_WORKER_STATE: Dict[str, Any] | None = None
_TRAIN_COUNTER: Optional["mp.Value"] = None
_VAL_COUNTER: Optional["mp.Value"] = None
_COUNTER_LOCK: Optional[mp.Lock] = None


def find_component_paths(sample_dir: Path) -> List[Path]:
    patterns = [
        "component_*.png",
        f"{sample_dir.name}_component_*.png",
        "*_component_*.png",
    ]
    for pattern in patterns:
        indexed: List[Tuple[int, Path]] = []
        for path in sample_dir.glob(pattern):
            if "thumbnail" in path.name.lower():
                continue
            stem_parts = path.stem.split("_")
            numeric = next((part for part in reversed(stem_parts) if part.isdigit()), None)
            if numeric is None:
                continue
            indexed.append((int(numeric), path))
        if indexed:
            indexed.sort(key=lambda item: item[0])
            return [p for _, p in indexed]
    return []


def load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGBA")


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


def resolve_background_path(sample_dir: Path) -> Path:
    direct = sample_dir / "background.png"
    if direct.exists():
        return direct
    prefixed = sample_dir / f"{sample_dir.name}_background.png"
    if prefixed.exists():
        return prefixed
    for candidate in sorted(sample_dir.glob("*_background.png")):
        if "thumbnail" in candidate.name.lower():
            continue
        return candidate
    raise FileNotFoundError(f"Background not found for {sample_dir}")


def composite_layers(background: Image.Image, components: Sequence[Image.Image]) -> Image.Image:
    composite = background.convert("RGBA")
    for layer in components:
        composite = Image.alpha_composite(composite, layer.convert("RGBA"))
    return composite


def save_component(
    img: Image.Image,
    split: str,
    sample_name: str,
    comp_idx: int,
    bucket_name: str,
    bucket_dims: Tuple[int, int],
    output_root: Path,
) -> Path:
    bucket_root = output_root / split / bucket_name
    bucket_root.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_name}_fg{comp_idx:03d}.png"
    out_path = bucket_root / filename
    resized = img.resize(bucket_dims, resample=Image.LANCZOS)
    resized.save(out_path)
    return out_path.relative_to(output_root)



def _component_alpha_mask(image: Image.Image) -> np.ndarray:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = np.asarray(image, dtype=np.uint8)[..., 3]
    return alpha > 0


def _build_component_masks(components: Sequence[Image.Image]) -> Dict[int, np.ndarray]:
    masks: Dict[int, np.ndarray] = {}
    for idx, image in enumerate(components):
        mask = _component_alpha_mask(image)
        if np.any(mask):
            masks[idx] = mask
    return masks


def _erode_masks(masks: Dict[int, np.ndarray], iterations: int) -> Dict[int, np.ndarray]:
    if iterations <= 0:
        return {idx: mask.copy() for idx, mask in masks.items()}
    structure = np.ones((3, 3), dtype=bool)
    eroded: Dict[int, np.ndarray] = {}
    for idx, mask in masks.items():
        eroded_mask = binary_erosion(mask, structure=structure, iterations=iterations)
        if not np.any(eroded_mask):
            eroded_mask = mask.copy()
        eroded[idx] = eroded_mask
    return eroded


def _background_visible_ratio(masks: Dict[int, np.ndarray]) -> float:
    """
    Estimate how much of the background remains visible (not covered by components).
    """
    if not masks:
        return 1.0
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        union |= mask
    total = union.size
    if total <= 0:
        return 1.0
    visible = total - int(union.sum())
    return float(visible) / float(total)


def _find_unoverlapped_indices(remaining: Sequence[int], eroded_masks: Dict[int, np.ndarray]) -> List[int]:
    if not remaining:
        return []
    sample_mask = next(iter(eroded_masks.values()))
    covered = np.zeros_like(sample_mask, dtype=bool)
    picks: List[int] = []
    for idx in reversed(remaining):
        mask = eroded_masks.get(idx)
        if mask is None:
            continue
        if not np.any(mask & covered):
            picks.append(idx)
            covered |= mask
    picks.reverse()
    return picks


def _composite_subset(components: Sequence[Image.Image], indices: Sequence[int], canvas_size: Tuple[int, int]) -> Image.Image:
    fg = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for idx in indices:
        fg = Image.alpha_composite(fg, components[idx].convert("RGBA"))
    return fg


def iterate_foreground_groups(
    background: Image.Image,
    components: Sequence[Image.Image],
    *,
    erosion_iterations: int,
    max_groups: Optional[int],
    masks: Optional[Dict[int, np.ndarray]] = None,
):
    masks = masks if masks is not None else _build_component_masks(components)
    if not masks:
        return
    eroded_masks = _erode_masks(masks, iterations=erosion_iterations)
    remaining = [idx for idx in range(len(components)) if idx in masks]
    stage = 0
    while remaining:
        picks = _find_unoverlapped_indices(remaining, eroded_masks)
        if not picks:
            break
        base_image = composite_layers(background, [components[idx] for idx in remaining])
        fg_image = _composite_subset(components, picks, background.size)
        yield stage, picks, base_image, fg_image
        remaining = [idx for idx in remaining if idx not in picks]
        stage += 1
        if max_groups is not None and stage >= max_groups:
            break


def _claim_split_mp(sample_name: str) -> Optional[str]:
    if _COUNTER_LOCK is None or _TRAIN_COUNTER is None or _VAL_COUNTER is None:
        raise RuntimeError("Multiprocessing counters are not initialized.")
    validation_set = set(_WORKER_STATE.get("validation_set", [])) if _WORKER_STATE else set()

    def take(split: str) -> Optional[str]:
        if split == "val":
            if _VAL_COUNTER.value == -1:
                return "val"
            if _VAL_COUNTER.value > 0:
                _VAL_COUNTER.value -= 1
                return "val"
            return None
        if _TRAIN_COUNTER.value == -1:
            return "train"
        if _TRAIN_COUNTER.value > 0:
            _TRAIN_COUNTER.value -= 1
            return "train"
        return None

    with _COUNTER_LOCK:
        if validation_set and sample_name in validation_set:
            return take("val")
        return take("train")


def _init_worker(state: Dict[str, Any], train_counter, val_counter, lock) -> None:
    global _WORKER_STATE, _TRAIN_COUNTER, _VAL_COUNTER, _COUNTER_LOCK
    _WORKER_STATE = state
    _TRAIN_COUNTER = train_counter
    _VAL_COUNTER = val_counter
    _COUNTER_LOCK = lock


def _make_sample_rng(sample_name: str, base_seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{sample_name}|{base_seed}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _pick_component_by_alpha(
    indices: Sequence[int],
    alpha_sums: Dict[int, int],
    rng: np.random.Generator,
) -> Optional[int]:
    if not indices:
        return None
    weights = np.array([alpha_sums.get(idx, 0) for idx in indices], dtype=np.float64)
    probs = None
    if np.any(weights):
        probs = weights / weights.sum()
    return int(rng.choice(indices, p=probs))


def _worker_process(sample_dir: Path) -> List[Dict[str, Any]]:
    if _TRAIN_COUNTER is not None and _VAL_COUNTER is not None:
        train_exhausted = _TRAIN_COUNTER.value == 0
        val_exhausted = _VAL_COUNTER.value == 0
        if train_exhausted and val_exhausted:
            return []
    try:
        return _process_sample(sample_dir)
    except Exception:  # noqa: BLE001
        logging.exception("Failed to process %s", sample_dir)
        return []


def _process_sample(
    sample_dir: Path,
    state: Optional[Dict[str, Any]] = None,
    claim_split: Optional[Callable[[str], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    active_state = state or _WORKER_STATE
    if active_state is None:
        raise RuntimeError("Worker state is not initialized.")

    component_paths = find_component_paths(sample_dir)
    if not component_paths:
        return []

    with Image.open(resolve_background_path(sample_dir)) as bg:
        background = bg.convert("RGBA")
    component_images = [load_rgba(path) for path in component_paths]
    component_masks = _build_component_masks(component_images)
    if not component_masks:
        return []
    component_alpha_sums = {idx: int(mask.sum()) for idx, mask in component_masks.items()}
    visible_ratio = _background_visible_ratio(component_masks)
    background_visible = visible_ratio > BACKGROUND_VISIBILITY_THRESHOLD

    assignment, reason = bucket_assignment(background.size)
    if assignment is None:
        logging.debug("Skipping %s due to bucket exclusion: %s", sample_dir.name, reason)
        return []
    bucket_name, bucket_dims = assignment

    groups: List[Tuple[int, List[int], Image.Image, Image.Image]] = []
    rng = _make_sample_rng(sample_dir.name, int(active_state.get("seed", 0)))

    for stage_idx, picks, base_image, fg_image in iterate_foreground_groups(
        background,
        component_images,
        erosion_iterations=active_state["fg_erosion_iterations"],
        max_groups=active_state["fg_max_groups"],
        masks=component_masks,
    ):
        groups.append((stage_idx, list(picks), base_image, fg_image))

    if not groups:
        return []

    splitter = claim_split or _claim_split_mp
    split = splitter(sample_dir.name)
    if split is None:
        logging.debug("Capacity reached; skipping sample %s", sample_dir.name)
        return []

    output_root = Path(active_state["output_root"])
    composite_candidate = output_root / split / bucket_name / f"{sample_dir.name}_fg000_composite.png"
    if composite_candidate.exists():
        logging.debug("Sample %s already processed for %s; skipping", sample_dir.name, split)
        return []

    background_rel: Optional[Path] = None
    if background_visible:
        background_rel = save_background(
            background,
            split=split,
            sample_name=sample_dir.name,
            bucket_name=bucket_name,
            bucket_dims=bucket_dims,
            output_root=output_root,
        )
    else:
        logging.debug(
            "Skipping background save for %s (visible_ratio=%.4f <= %.3f)",
            sample_dir.name,
            visible_ratio,
            BACKGROUND_VISIBILITY_THRESHOLD,
        )

    entries: List[Dict[str, Any]] = []
    composite_rel: Optional[str] = None
    composite_stage: Optional[int] = None
    last_stage = groups[-1][0]

    for stage_idx, group_indices, base_image, fg_image in groups:
        do_select = stage_idx != last_stage
        selected_indices: List[int] = []
        selected_paths: List[Path] = []
        if do_select and group_indices:
            first_idx = _pick_component_by_alpha(group_indices, component_alpha_sums, rng)
            if first_idx is not None:
                selected_indices.append(first_idx)
                selected_paths.append(
                    save_selected_component(
                        component_images[first_idx],
                        split=split,
                        sample_name=sample_dir.name,
                        comp_idx=stage_idx,
                        bucket_name=bucket_name,
                        bucket_dims=bucket_dims,
                        output_root=output_root,
                        selection_rank=0,
                    )
                )
                remaining = [idx for idx in group_indices if idx != first_idx]
                if remaining:
                    second_idx = _pick_component_by_alpha(
                        remaining, component_alpha_sums, rng
                    )
                    if second_idx is not None:
                        selected_indices.append(second_idx)
                        selected_paths.append(
                            save_selected_component(
                                component_images[second_idx],
                                split=split,
                                sample_name=sample_dir.name,
                                comp_idx=stage_idx,
                                bucket_name=bucket_name,
                                bucket_dims=bucket_dims,
                                output_root=output_root,
                                selection_rank=1,
                            )
                        )
        comp_rel = save_component(
            fg_image,
            split=split,
            sample_name=sample_dir.name,
            comp_idx=stage_idx,
            bucket_name=bucket_name,
            bucket_dims=bucket_dims,
            output_root=output_root,
        )
        if composite_rel is None:
            composite_rel = save_composite(
                base_image,
                split=split,
                sample_name=sample_dir.name,
                comp_idx=stage_idx,
                bucket_name=bucket_name,
                bucket_dims=bucket_dims,
                output_root=output_root,
            )
            composite_stage = stage_idx
        fg_size = list(fg_image.size)
        entries.append(
            {
                "split": split,
                "bucket": bucket_name,
                "bucket_dims": list(bucket_dims),
                "component_path": str(comp_rel),
                "composite_path": str(composite_rel),
                "background_path": str(background_rel) if background_rel else None,
                "source_sample": sample_dir.name,
                "component_index": stage_idx,
                "composite_stage": composite_stage,
                "group_size": len(group_indices),
                "group_indices": group_indices,
                "original_size": fg_size,
                "selected_component_index": selected_indices[0]
                if selected_indices
                else None,
                "selected_component_path": str(selected_paths[0])
                if selected_paths
                else None,
                "selected_component_indices": selected_indices,
                "selected_component_paths": [str(p) for p in selected_paths],
            }
        )
        base_image.close()
        fg_image.close()

    logging.info(
        "Processed %s -> %s (groups=%d)", sample_dir.name, split, len(entries)
    )
    return entries



def save_composite(
    composite: Image.Image,
    split: str,
    sample_name: str,
    comp_idx: int,
    bucket_name: str,
    bucket_dims: Tuple[int, int],
    output_root: Path,
) -> Path:
    bucket_root = output_root / split / bucket_name
    bucket_root.mkdir(parents=True, exist_ok=True)
    out_path = bucket_root / f"{sample_name}_fg{comp_idx:03d}_composite.png"
    composite.resize(bucket_dims, resample=Image.LANCZOS).save(out_path)
    return out_path.relative_to(output_root)


def save_background(
    background: Image.Image,
    split: str,
    sample_name: str,
    bucket_name: str,
    bucket_dims: Tuple[int, int],
    output_root: Path,
) -> Path:
    """
    Save the resized background once per sample to align with bucket dimensions.
    """
    bucket_root = output_root / split / bucket_name
    bucket_root.mkdir(parents=True, exist_ok=True)
    out_path = bucket_root / f"{sample_name}_background.png"
    background.resize(bucket_dims, resample=Image.LANCZOS).save(out_path)
    return out_path.relative_to(output_root)


def save_selected_component(
    img: Image.Image,
    split: str,
    sample_name: str,
    comp_idx: int,
    bucket_name: str,
    bucket_dims: Tuple[int, int],
    output_root: Path,
    selection_rank: int = 0,
) -> Path:
    bucket_root = output_root / split / bucket_name
    bucket_root.mkdir(parents=True, exist_ok=True)
    suffix = (
        "selected.png" if selection_rank == 0 else f"selected{selection_rank}.png"
    )
    filename = f"{sample_name}_fg{comp_idx:03d}_{suffix}"
    out_path = bucket_root / filename
    img.resize(bucket_dims, resample=Image.LANCZOS).save(out_path)
    return out_path.relative_to(output_root)


def write_manifest(records: List[Dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_validation_set(path: Optional[Path]) -> Set[str]:
    """
    Load validation sample names (one per line). Blank lines are ignored.
    """
    if path is None:
        return set()
    if not path.exists():
        logging.info("Validation list %s not found; all samples will default to train.", path)
        return set()
    with path.open("r", encoding="utf-8") as f:
        names = {line.strip() for line in f if line.strip()}
    logging.info("Loaded %d validation sample names from %s", len(names), path)
    return names


def _flatten_structure(records: List[Dict[str, Any]], output_root: Path) -> None:
    """
    Ensure component/composite files live directly under bucket directories and
    update manifest paths accordingly.
    """
    for entry in records:
        split = entry["split"]
        bucket = entry["bucket"]
        bucket_root = output_root / split / bucket
        bucket_root.mkdir(parents=True, exist_ok=True)

        comp_name = Path(entry["component_path"]).name
        comp_dst = bucket_root / comp_name
        comp_candidates = [
            output_root / entry["component_path"],
            bucket_root / "components" / comp_name,
        ]
        for src in comp_candidates:
            if src.exists():
                if src != comp_dst:
                    comp_dst.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(comp_dst)
                break
        entry["component_path"] = str(Path(split) / bucket / comp_name)

        sample_name = entry["source_sample"]
        raw_composite_name = Path(entry["composite_path"]).name
        composite_name = raw_composite_name if raw_composite_name.endswith("_composite.png") else f"{sample_name}_composite.png"
        compo_dst = bucket_root / composite_name
        compo_candidates = [
            output_root / entry["composite_path"],
            bucket_root / "composite" / raw_composite_name,
        ]
        for src in compo_candidates:
            if src.exists():
                if src != compo_dst:
                    compo_dst.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(compo_dst)
                break
        entry["composite_path"] = str(Path(split) / bucket / composite_name)

        background_rel = entry.get("background_path")
        if background_rel:
            bg_name = Path(background_rel).name
            bg_dst = bucket_root / bg_name
            bg_candidates = [
                output_root / background_rel,
                bucket_root / "background" / bg_name,
            ]
            for src in bg_candidates:
                if src.exists():
                    if src != bg_dst:
                        bg_dst.parent.mkdir(parents=True, exist_ok=True)
                        src.replace(bg_dst)
                    break
            entry["background_path"] = str(Path(split) / bucket / bg_name)

    # Remove empty legacy directories
    for split_dir in (output_root / "train", output_root / "val"):
        if not split_dir.exists():
            continue
        for bucket_dir in split_dir.iterdir():
            if not bucket_dir.is_dir():
                continue
            for legacy in ("components", "composite"):
                legacy_dir = bucket_dir / legacy
                if legacy_dir.exists() and legacy_dir.is_dir():
                    for child in legacy_dir.iterdir():
                        child.unlink()
                    legacy_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket RGBA component layers for VAE training.")
    parser.add_argument("--rendered-root", type=Path, default=DEFAULT_RENDERED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validation-list",
        type=Path,
        default=DEFAULT_VALIDATION_LIST,
        help="File containing sample names for validation split (one per line). If missing or empty, all go to train unless counts force otherwise.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Optional cap on number of training composites. Omit for unlimited.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Optional cap on number of validation composites. Omit for unlimited.",
    )
    parser.add_argument("--fg-max-groups", type=int, default=None, help="Optional cap on the number of foreground groups to emit per sample.")
    parser.add_argument("--fg-erosion-iterations", type=int, default=1, help="3x3 erosion iterations before overlap grouping (default: 1).")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes to use for sample generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on sample directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    validation_set = load_validation_set(args.validation_list)
    train_limit = args.train_count if args.train_count is not None else -1
    val_limit = args.val_count if args.val_count is not None else -1

    sample_dirs = sorted(d for d in args.rendered_root.iterdir() if d.is_dir())
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(sample_dirs))
    rng.shuffle(indices)
    shuffled_dirs = [sample_dirs[i] for i in indices]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    state = {
        "output_root": str(output_root),
        "fg_max_groups": args.fg_max_groups,
        "fg_erosion_iterations": args.fg_erosion_iterations,
        "seed": args.seed,
        "validation_set": validation_set,
    }

    manifest_records: List[Dict[str, Any]] = []

    if args.num_workers <= 1:
        train_remaining: Optional[int] = None if train_limit < 0 else train_limit
        val_remaining: Optional[int] = None if val_limit < 0 else val_limit

        def claim_split_local(sample_name: str) -> Optional[str]:
            nonlocal train_remaining, val_remaining
            wants_val = sample_name in validation_set
            if wants_val:
                if val_remaining is None:
                    return "val"
                if val_remaining > 0:
                    val_remaining -= 1
                    return "val"
                return None
            if train_remaining is None:
                return "train"
            if train_remaining > 0:
                train_remaining -= 1
                return "train"
            return None

        def limits_exhausted() -> bool:
            return (train_remaining is not None and train_remaining <= 0) and (
                val_remaining is not None and val_remaining <= 0
            )

        for sample_dir in tqdm(shuffled_dirs, desc="Processing samples"):
            if limits_exhausted():
                break
            entries = _process_sample(sample_dir, state, claim_split_local)
            manifest_records.extend(entries)
    else:
        train_counter = mp.Value("i", train_limit)
        val_counter = mp.Value("i", val_limit)
        lock = mp.Lock()
        pool = mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(state, train_counter, val_counter, lock),
        )
        terminated = False
        try:
            iterator = pool.imap_unordered(_worker_process, shuffled_dirs)
            for entries in tqdm(iterator, total=len(shuffled_dirs), desc="Processing samples"):
                manifest_records.extend(entries)
                train_exhausted = train_counter.value == 0
                val_exhausted = val_counter.value == 0
                if train_exhausted and val_exhausted:
                    pool.terminate()
                    terminated = True
                    break
        finally:
            if not terminated:
                pool.close()
            pool.join()
        train_remaining = train_counter.value
        val_remaining = val_counter.value

    def is_unlimited(value: Optional[int]) -> bool:
        return value is None or value == -1

    if (not is_unlimited(train_remaining) and train_remaining > 0) or (
        not is_unlimited(val_remaining) and val_remaining > 0
    ):
        logging.warning(
            "Did not reach requested counts (train_remaining=%d, val_remaining=%d)",
            train_remaining,
            val_remaining,
        )

    _flatten_structure(manifest_records, output_root)
    manifest_path = output_root / "metadata" / "manifest.json"
    write_manifest(manifest_records, manifest_path)
    logging.info("Manifest written to %s with %d entries.", manifest_path, len(manifest_records))


if __name__ == "__main__":
    main()