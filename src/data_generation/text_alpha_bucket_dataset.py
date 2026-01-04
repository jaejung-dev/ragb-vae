from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


BUCKET_RE = re.compile(r"^w(?P<w>\d+)-h(?P<h>\d+)$")


def _parse_bucket_dims(bucket_dir: Path) -> Tuple[int, int]:
    m = BUCKET_RE.match(bucket_dir.name)
    if not m:
        raise ValueError(f"Invalid bucket directory name: {bucket_dir.name}")
    return int(m.group("w")), int(m.group("h"))


def _load_rgba(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    arr = torch.from_numpy(np.array(rgba, dtype=np.uint8)).float()  # type: ignore[name-defined]
    # HWC -> CHW, [0,1]
    return arr.permute(2, 0, 1) / 255.0


def _gather_pairs(split_root: Path) -> List[Dict]:
    entries: List[Dict] = []
    for bucket_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        try:
            bucket_dims = _parse_bucket_dims(bucket_dir)
        except ValueError:
            continue

        gt_dir = bucket_dir / "gt"
        text_alpha_dir = bucket_dir / "text_alpha"
        if not gt_dir.exists() or not text_alpha_dir.exists():
            continue

        for gt_path in sorted(gt_dir.glob("*.png")):
            name = gt_path.stem
            ta_path = text_alpha_dir / f"{name}.png"
            if not ta_path.exists():
                continue
            entries.append(
                {
                    "bucket": bucket_dir.name,
                    "bucket_dims": bucket_dims,
                    "gt_path": gt_path,
                    "text_alpha_path": ta_path,
                    "sample_name": name,
                }
            )
    if not entries:
        raise ValueError(f"No gt/text_alpha pairs found under {split_root}")
    return entries


class TextAlphaBucketDataset(Dataset):
    """Bucketed dataset yielding (gt, text_alpha) RGBA tensors."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.split_root = Path(root) / split
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {self.split_root}")
        self.entries = _gather_pairs(self.split_root)

        self.bucket_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.entries):
            bucket = entry["bucket"]
            self.bucket_to_indices.setdefault(bucket, []).append(idx)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        gt = _load_rgba(entry["gt_path"])
        text_alpha = _load_rgba(entry["text_alpha_path"])
        return {
            "gt": gt,
            "text_alpha": text_alpha,
            "bucket": entry["bucket"],
            "bucket_dims": torch.tensor(entry["bucket_dims"], dtype=torch.int64),
            "sample_name": entry["sample_name"],
        }


class BucketBatchSampler:
    """Bucket-pure sampler with optional interleaving proportional to bucket size."""

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
        import random

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
        import math

        total = 0
        for indices in self.bucket_to_indices.values():
            if self.drop_last:
                total += math.floor(len(indices) / self.batch_size)
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total

