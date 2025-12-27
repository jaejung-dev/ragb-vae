"""
Torch dataset for multilayer RGBA samples with layout metadata.

Each sample directory under RENDERED_ROOT is expected to contain:
- background.png (or *_background.png variants)
- component_*.png (RGBA) ordered by numeric suffix

Corresponding JSON under JSON_ROOT provides layout metadata (descriptions, types).
Composite is either provided or computed via alpha compositing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .paths import RENDERED_ROOT, JSON_ROOT


def _component_index_key(path: Path) -> int:
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Component filename lacks numeric suffix: {path.name}")


def _resolve_background_path(sample_dir: Path) -> Path:
    direct_path = sample_dir / "background.png"
    if direct_path.exists():
        return direct_path

    prefixed_path = sample_dir / f"{sample_dir.name}_background.png"
    if prefixed_path.exists():
        return prefixed_path

    for candidate in sorted(sample_dir.glob("*_background.png")):
        if "thumbnail" in candidate.name.lower():
            continue
        return candidate

    raise FileNotFoundError(f"Background image not found in {sample_dir}")


def _find_component_paths(sample_dir: Path) -> List[Path]:
    patterns = [
        "component_*.png",
        f"{sample_dir.name}_component_*.png",
        "*_component_*.png",
    ]

    for pattern in patterns:
        indexed_candidates: List[Tuple[int, Path]] = []
        for path in sample_dir.glob(pattern):
            if "thumbnail" in path.name.lower():
                continue
            try:
                index = _component_index_key(path)
            except ValueError:
                continue
            indexed_candidates.append((index, path))

        if indexed_candidates:
            indexed_candidates.sort(key=lambda item: item[0])
            return [path for _, path in indexed_candidates]

    return []


def _load_image(path: Path, mode: str = "RGBA") -> Image.Image:
    with Image.open(path) as img:
        return img.convert(mode)


def _composite_layers(background: Image.Image, components: Sequence[Image.Image]) -> Image.Image:
    composite = background.convert("RGBA") if background.mode != "RGBA" else background.copy()
    for component in components:
        overlay = component if component.mode == "RGBA" else component.convert("RGBA")
        if overlay.size != composite.size:
            raise ValueError(f"Component size {overlay.size} does not match background {composite.size}")
        composite = Image.alpha_composite(composite, overlay)
    return composite


def _to_tensor_rgba(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.ones_like(arr)], axis=-1)
    if arr.shape[-1] == 3:
        alpha = np.ones_like(arr[..., :1])
        arr = np.concatenate([arr, alpha], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    return torch.from_numpy(arr)


def _visible_alpha_mask(img: Image.Image, alpha_threshold: int) -> torch.Tensor:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.asarray(img, dtype=np.uint8)[..., 3]
    return torch.from_numpy(alpha >= alpha_threshold)


@dataclass
class MultiLayerSample:
    sample_dir: Path
    background: torch.Tensor  # (4,H,W)
    components: List[torch.Tensor]  # list of (4,H,W)
    composite: torch.Tensor  # (4,H,W)
    layout: Dict[str, Any]
    visible_masks: List[torch.Tensor]  # list of (H,W) bool


class MultiLayerDataset(Dataset):
    def __init__(
        self,
        rendered_root: Path = RENDERED_ROOT,
        json_root: Path = JSON_ROOT,
        alpha_threshold: int = 100,
        max_samples: Optional[int] = None,
    ) -> None:
        self.rendered_root = Path(rendered_root)
        self.json_root = Path(json_root)
        self.alpha_threshold = alpha_threshold

        if not self.rendered_root.exists():
            raise FileNotFoundError(f"Rendered root not found: {self.rendered_root}")
        self.sample_dirs = sorted(p for p in self.rendered_root.iterdir() if p.is_dir())
        if max_samples is not None:
            self.sample_dirs = self.sample_dirs[:max_samples]
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories under {self.rendered_root}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> MultiLayerSample:
        sample_dir = self.sample_dirs[index]
        background_path = _resolve_background_path(sample_dir)
        component_paths = _find_component_paths(sample_dir)

        background = _load_image(background_path, mode="RGBA")
        components = [_load_image(p, mode="RGBA") for p in component_paths]
        composite = _composite_layers(background, components)

        background_t = _to_tensor_rgba(background)
        components_t = [_to_tensor_rgba(comp) for comp in components]
        composite_t = _to_tensor_rgba(composite)
        visible_masks = [_visible_alpha_mask(comp, self.alpha_threshold) for comp in components]

        json_path = self.json_root / f"{sample_dir.name}.json"
        layout: Dict[str, Any] = {"layout_config": {"components": []}}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                layout = json.load(f)

        return MultiLayerSample(
            sample_dir=sample_dir,
            background=background_t,
            components=components_t,
            composite=composite_t,
            layout=layout,
            visible_masks=visible_masks,
        )


def multilayer_collate(batch: List[MultiLayerSample]) -> Dict[str, Any]:
    """Pad variable-length component stacks and emit masks."""
    if not batch:
        return {}
    max_components = max(len(item.components) for item in batch)

    backgrounds = []
    composites = []
    components_padded = []
    component_mask = []
    visible_masks = []
    sample_dirs = []
    layouts = []

    for item in batch:
        backgrounds.append(item.background)
        composites.append(item.composite)
        layouts.append(item.layout)
        sample_dirs.append(str(item.sample_dir))

        comps = item.components
        vis_masks = item.visible_masks
        if not comps:
            # If no components, create a single zero layer to avoid empty stacks.
            zero = torch.zeros_like(item.background)
            comps = [zero]
            vis_masks = [torch.zeros(item.background.shape[1:], dtype=torch.bool)]

        pad_count = max_components - len(comps)
        if pad_count > 0:
            zero = torch.zeros_like(comps[0])
            comps = comps + [zero] * pad_count
            vis_mask_zero = torch.zeros_like(vis_masks[0])
            vis_masks = vis_masks + [vis_mask_zero] * pad_count

        components_padded.append(torch.stack(comps, dim=0))  # (L,4,H,W)
        visible_masks.append(torch.stack(vis_masks, dim=0))  # (L,H,W)

        mask = torch.zeros(max_components, dtype=torch.bool)
        mask[: len(item.components)] = True
        component_mask.append(mask)

    return {
        "background": torch.stack(backgrounds, dim=0),  # (B,4,H,W)
        "composite": torch.stack(composites, dim=0),  # (B,4,H,W)
        "components": torch.stack(components_padded, dim=0),  # (B,L,4,H,W)
        "component_mask": torch.stack(component_mask, dim=0),  # (B,L)
        "visible_masks": torch.stack(visible_masks, dim=0),  # (B,L,H,W)
        "layout": layouts,  # raw dicts (variable structure)
        "sample_dirs": sample_dirs,
    }



