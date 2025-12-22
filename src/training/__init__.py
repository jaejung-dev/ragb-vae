"""Training utilities and stage entrypoints."""

from __future__ import annotations

from typing import Any, Dict

from .rgba_vae_stage import (
    build_dataloader,
    compute_model_loss,
    evaluate_rgba_vae,
    save_checkpoints,
    train_rgba_vae,
)

__all__ = [
    "build_dataloader",
    "compute_model_loss",
    "evaluate_rgba_vae",
    "save_checkpoints",
    "train_rgba_vae",
    "train_decomposition",
    "train_refine",
]


def train_decomposition(cfg: Dict[str, Any]) -> None:  # pragma: no cover - placeholder
    dataloader = build_dataloader(cfg)
    _ = dataloader
    raise NotImplementedError("Decomposition training is pending implementation.")


def train_refine(cfg: Dict[str, Any]) -> None:  # pragma: no cover - placeholder
    dataloader = build_dataloader(cfg)
    _ = dataloader
    raise NotImplementedError("Refinement stage is pending implementation.")
