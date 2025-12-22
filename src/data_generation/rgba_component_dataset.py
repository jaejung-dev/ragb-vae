"""
PyTorch dataset/dataloader utilities for the bucketed RGBA component dataset.

Expected directory structure (per `prepare_rgba_buckets.py`):
data_root/
  ├── train/
  │     └── w1088-h1088/
  │            ├── sample_comp000.png
  │            ├── sample_comp001.png
  │            └── sample_composite.png
  ├── val/
  │     └── ...
  └── metadata/manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 3:
        alpha = np.ones_like(arr[..., :1], dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def _blend_to_white(rgba: torch.Tensor) -> torch.Tensor:
    rgb = rgba[:3]
    alpha = rgba[3:4]
    blended = rgb * alpha + (1.0 - alpha)
    return torch.cat([blended, torch.ones_like(alpha)], dim=0)


class RgbaComponentDataset(Dataset):
    """
    Yields (component RGBA, composite RGBA) pairs along with metadata needed for training.
    """

    def __init__(
        self,
        root_dir: Path | str = "data/rgba_layers",
        manifest_path: Optional[Path | str] = None,
        split: str = "train",
        limit: Optional[int] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        include_metadata: bool = True,
        blend_component_to_white: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest_path = Path(manifest_path or (self.root_dir / "metadata" / "manifest.json"))
        with manifest_path.open("r", encoding="utf-8") as f:
            entries: List[Dict[str, Any]] = json.load(f)
        self.entries = [entry for entry in entries if entry["split"] == split]
        if limit is not None:
            self.entries = self.entries[:limit]
        self.transform = transform
        self.include_metadata = include_metadata
        self.blend_component_to_white = blend_component_to_white

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        component_path = self.root_dir / entry["component_path"]
        composite_path = self.root_dir / entry["composite_path"]

        component = Image.open(component_path).convert("RGBA")
        composite = Image.open(composite_path).convert("RGBA")

        component_tensor = _pil_to_tensor(component)
        composite_tensor = _pil_to_tensor(composite)

        sample: Dict[str, Any] = {
            "component": component_tensor,
            "composite": composite_tensor,
        }

        if self.blend_component_to_white:
            sample["component_white"] = _blend_to_white(component_tensor.clone())

        if self.include_metadata:
            sample.update(
                {
                    "bucket": entry["bucket"],
                    "bucket_dims": tuple(entry["bucket_dims"]),
                    "source_sample": entry["source_sample"],
                    "component_index": entry["component_index"],
                    "original_size": tuple(entry["original_size"]),
                    "component_path": entry["component_path"],
                    "composite_path": entry["composite_path"],
                }
            )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def create_component_dataloader(
    root_dir: Path | str = "data/rgba_layers",
    manifest_path: Optional[Path | str] = None,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    limit: Optional[int] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    **loader_kwargs: Any,
) -> DataLoader:
    dataset_kwargs = dataset_kwargs or {}
    dataset = RgbaComponentDataset(
        root_dir=root_dir,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        transform=transform,
        **dataset_kwargs,
    )
    use_pad_collate = not dataset_kwargs.get("include_metadata", False)
    collate_fn = _pad_collate_tensors if use_pad_collate else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


def _pad_collate_tensors(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensor_keys = [k for k, v in batch[0].items() if torch.is_tensor(v)]
    max_hw: Dict[str, Tuple[int, int]] = {k: (0, 0) for k in tensor_keys}
    for item in batch:
        for key in tensor_keys:
            _, h, w = item[key].shape
            max_h, max_w = max_hw[key]
            max_hw[key] = (max(max_h, h), max(max_w, w))

    collated: Dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        target_h, target_w = max_hw[key]
        padded = [_pad_to_hw(item[key], target_h, target_w) for item in batch]
        collated[key] = torch.stack(padded, dim=0)
    return collated


def _pad_to_hw(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, h, w = tensor.shape
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return tensor
    pad = (0, pad_w, 0, pad_h)
    return F.pad(tensor, pad, mode="constant", value=0.0)

