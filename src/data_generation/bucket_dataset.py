from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset

from .rgba_component_dataset import _pil_to_tensor

# Allow PNG text/iCCP chunks up to this size (default 64MB) to avoid Pillow
# safety guard failures on large embedded profiles in the dataset.
PNG_TEXT_CHUNK_LIMIT = int(os.environ.get("PNG_MAX_TEXT_CHUNK", 64 * 1024 * 1024))
if hasattr(PngImagePlugin, "MAX_TEXT_CHUNK"):
    PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, PNG_TEXT_CHUNK_LIMIT)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_bucket_dims(bucket: str) -> Tuple[int, int]:
    # expected format: w{width}-h{height}
    if not bucket.startswith("w") or "-h" not in bucket:
        raise ValueError(f"Invalid bucket format: {bucket}")
    width_part = bucket[1 : bucket.index("-h")]
    height_part = bucket[bucket.index("-h") + 2 :]
    return int(width_part), int(height_part)


def _normalize_entry_bucket(entry: Dict[str, Any]) -> Tuple[str, Tuple[int, int]]:
    bucket = entry.get("bucket")
    bucket_dims = entry.get("bucket_dims")
    if bucket_dims is not None:
        return bucket, tuple(bucket_dims)  # type: ignore[arg-type]
    if bucket is None:
        raise ValueError("Entry must contain either bucket or bucket_dims")
    return bucket, _parse_bucket_dims(bucket)


def _standardize_components_manifest(
    data: List[Dict[str, Any]],
    *,
    split: str,
    root: Path,
    respect_split: bool = True,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in data:
        if respect_split and item.get("split") != split:
            continue
        bucket, bucket_dims = _normalize_entry_bucket(item)
        paths: List[Tuple[str, str]] = []
        if item.get("component_path"):
            paths.append(("component", item["component_path"]))
        if item.get("composite_path"):
            paths.append(("composite", item["composite_path"]))
        if item.get("background_path"):
            paths.append(("background", item["background_path"]))
        for sel_path in item.get("selected_component_paths", []):
            paths.append(("selected_component", sel_path))
        for variant, path in paths:
            entries.append(
                {
                    "split": split,
                    "root_dir": str(root),
                    "bucket": bucket,
                    "bucket_dims": bucket_dims,
                    "image_path": path,
                    "source_sample": item.get("source_sample"),
                    "variant": variant,
                }
            )
    return entries


def _standardize_prism_real(
    data: List[Dict[str, Any]],
    *,
    split: str,
    root: Path,
    respect_split: bool = True,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in data:
        if respect_split and item.get("split") != split:
            continue
        bucket, bucket_dims = _normalize_entry_bucket(item)
        layer_paths = item.get("layer_paths") or []
        candidates: List[Tuple[str, str]] = []
        if item.get("base_path"):
            candidates.append(("base", item["base_path"]))
        if item.get("whole_path"):
            candidates.append(("whole", item["whole_path"]))
        for layer_path in layer_paths:
            candidates.append(("layer", layer_path))
        for variant, path in candidates:
            entries.append(
                {
                    "split": split,
                    "root_dir": str(root),
                    "bucket": bucket,
                    "bucket_dims": bucket_dims,
                    "image_path": path,
                    "source_sample": item.get("id"),
                    "variant": variant,
                }
            )
    return entries


def _standardize_prism_pro(
    data: List[Dict[str, Any]],
    *,
    split: str,
    use_fg: bool,
    use_rep: bool,
    root: Path,
    respect_split: bool = True,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in data:
        if respect_split and item.get("split") != split:
            continue
        bucket, bucket_dims = _normalize_entry_bucket(item)
        candidates: List[Tuple[str, str]] = []
        if item.get("base_path"):
            candidates.append(("base", item["base_path"]))
        if item.get("composite_path"):
            candidates.append(("composite", item["composite_path"]))
        if use_fg and item.get("fg_non_overlap_path"):
            candidates.append(("fg_non_overlap", item["fg_non_overlap_path"]))
        if use_rep and item.get("rep_path"):
            candidates.append(("rep", item["rep_path"]))
        for variant, path in candidates:
            entries.append(
                {
                    "split": split,
                    "root_dir": str(root),
                    "bucket": bucket,
                    "bucket_dims": bucket_dims,
                    "image_path": path,
                    "source_sample": item.get("id"),
                    "variant": variant,
                }
            )
    return entries


def _collect_laion_rgb(
    root: Path, *, split: str, max_count: Optional[int] = None
) -> List[Dict[str, Any]]:
    split_root = root / split
    if not split_root.exists():
        return []
    entries: List[Dict[str, Any]] = []
    # pattern: {split}/wXXX-hYYY/*.png
    for bucket_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        bucket = bucket_dir.name
        bucket_dims = _parse_bucket_dims(bucket)
        for idx, path in enumerate(sorted(bucket_dir.glob("*.png"))):
            entries.append(
                {
                    "split": split,
                    "root_dir": str(root),
                    "bucket": bucket,
                    "bucket_dims": bucket_dims,
                    "image_path": str(Path(split) / bucket / path.name),
                    "source_sample": path.stem,
                    "variant": "rgb_only",
                }
            )
            if max_count is not None and len(entries) >= max_count:
                return entries
    return entries


def build_bucket_entries(
    dataset_cfgs: Sequence[Dict[str, Any]], *, split: str
) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    for cfg in dataset_cfgs:
        allowed_splits = cfg.get("splits")
        if allowed_splits is not None and split not in allowed_splits:
            continue
        dtype = cfg.get("type", "components")
        root = Path(cfg["root"])
        manifest_path = cfg.get("manifest")
        target_split = cfg.get("split", split)
        respect_split = bool(cfg.get("respect_manifest_split", True))
        use_fg = bool(cfg.get("use_fg_non_overlap", True))
        use_rep = bool(cfg.get("use_rep", True))
        max_count = cfg.get("max_count")

        if dtype == "components":
            manifest = Path(manifest_path or (root / "metadata" / "manifest.json"))
            data = _load_json(manifest)
            combined.extend(
                _standardize_components_manifest(
                    data, split=target_split, root=root, respect_split=respect_split
                )
            )
        elif dtype == "prism_real":
            manifest = Path(manifest_path or (root / "metadata" / "manifest.json"))
            data = _load_json(manifest)
            combined.extend(
                _standardize_prism_real(
                    data, split=target_split, root=root, respect_split=respect_split
                )
            )
        elif dtype == "prism_pro":
            manifest = Path(manifest_path or (root / "metadata" / "manifest.json"))
            data = _load_json(manifest)
            combined.extend(
                _standardize_prism_pro(
                    data,
                    split=target_split,
                    use_fg=use_fg,
                    use_rep=use_rep,
                    root=root,
                    respect_split=respect_split,
                )
            )
        elif dtype == "laion_rgb":
            combined.extend(
                _collect_laion_rgb(root, split=target_split, max_count=max_count)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")
    return combined


class MixedBucketDataset(Dataset):
    """
    Unified bucketed dataset that can consume mixed manifest schemas.
    Entries must contain:
      - split, bucket, bucket_dims
      - image_path (required)
    """

    def __init__(
        self,
        root_dir: Path | str,
        entries: Sequence[Dict[str, Any]],
        *,
        include_metadata: bool = False,
        include_background: bool = False,
        blend_component_to_white: bool = False,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.entries: List[Dict[str, Any]] = list(entries)
        self.include_metadata = include_metadata
        self.include_background = include_background  # kept for compatibility; ignored
        self.blend_component_to_white = blend_component_to_white  # ignored
        self.transform = transform

        self.bucket_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.entries):
            bucket = entry["bucket"]
            self.bucket_to_indices.setdefault(bucket, []).append(idx)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_tensor(self, path: Path) -> Any:
        try:
            with Image.open(path) as img:
                img_rgba = img.convert("RGBA")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            # Expose the offending file path in the error for faster triage
            if isinstance(exc, ValueError) and "MAX_TEXT_CHUNK" in str(exc):
                raise RuntimeError(
                    f"PNG text chunk too large (iCCP) in file: {path}. "
                    f"Consider sanitizing the image or increasing PNG_MAX_TEXT_CHUNK."
                ) from exc
            raise RuntimeError(f"Failed to load image at {path}: {exc}") from exc
        return _pil_to_tensor(img_rgba)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        image_path = entry.get("image_path")
        root_dir = Path(entry.get("root_dir", self.root_dir))

        if image_path is None:
            raise ValueError("image_path is required for each entry.")

        composite = self._load_tensor(root_dir / image_path)
        sample: Dict[str, Any] = {"composite": composite}

        if self.include_metadata:
            sample.update(
                {
                    "bucket": entry.get("bucket"),
                    "bucket_dims": tuple(entry.get("bucket_dims") or ()),
                    "source_sample": entry.get("source_sample"),
                    "image_path": image_path,
                    "variant": entry.get("variant"),
                }
            )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class BucketBatchSampler:
    """
    Yield batches that are bucket-pure (same resolution per batch).
    """

    def __init__(
        self,
        bucket_to_indices: Dict[str, List[int]],
        *,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        interleave: bool = False,
    ) -> None:
        self.bucket_to_indices = {k: list(v) for k, v in bucket_to_indices.items()}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.interleave = interleave

    def __iter__(self):
        if not self.interleave:
            bucket_keys = list(self.bucket_to_indices.keys())
            if self.shuffle:
                random.shuffle(bucket_keys)
            for bucket in bucket_keys:
                indices = list(self.bucket_to_indices[bucket])
                if self.shuffle:
                    random.shuffle(indices)
                total = len(indices)
                step = self.batch_size
                max_len = total - (total % step) if self.drop_last else total
                for start in range(0, max_len, step):
                    batch = indices[start : start + step]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    yield batch
            return

        # Interleaved path: pick buckets repeatedly (proportional to remaining size when shuffled)
        bucket_to_indices = {k: list(v) for k, v in self.bucket_to_indices.items()}
        for k, v in bucket_to_indices.items():
            if self.shuffle:
                random.shuffle(v)

        active = [k for k, v in bucket_to_indices.items() if v]
        while active:
            if self.shuffle and len(active) > 1:
                weights = [len(bucket_to_indices[k]) for k in active]
                bucket = random.choices(active, weights=weights, k=1)[0]
            else:
                bucket = active[0]

            indices = bucket_to_indices[bucket]
            if len(indices) < self.batch_size:
                if self.drop_last:
                    active.remove(bucket)
                    continue
                batch = indices[:]
                bucket_to_indices[bucket] = []
            else:
                batch = indices[: self.batch_size]
                bucket_to_indices[bucket] = indices[self.batch_size :]

            if not bucket_to_indices[bucket]:
                active.remove(bucket)

            if batch:
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.bucket_to_indices.values():
            if self.drop_last:
                total += math.floor(len(indices) / self.batch_size)
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total

