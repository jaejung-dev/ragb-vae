#!/usr/bin/env python3
"""
Entry point for stage-based training matching the paper flow:
- Stage 1: RGBA-VAE adaptation/pretraining
- Stage 2: Variable Layers Decomposition (VLD-MMDiT)
- Stage 3: Task-specific refinement / editing alignment
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_decomposition, train_refine, train_rgba_vae
from src.training.flux_kontext_textalpha_lora import train_from_config as train_kontext_textalpha_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stage = cfg.get("training", {}).get("stage", "rgba_vae")
    if stage == "rgba_vae":
        train_rgba_vae(cfg)
    elif stage == "decompose":
        train_decomposition(cfg)
    elif stage == "refine":
        train_refine(cfg)
    elif stage == "kontext_textalpha_lora":
        train_kontext_textalpha_lora(cfg)
    else:
        raise ValueError(f"Unknown training stage: {stage}")


if __name__ == "__main__":
    main()