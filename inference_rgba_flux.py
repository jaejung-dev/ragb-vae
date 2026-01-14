#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.models import FluxTextAlphaModel


def load_rgba(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0  # CHW
    return tensor


def save_rgba(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.clamp(0, 1).detach().cpu().float()  # ensure numpy supports dtype
    img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img, mode="RGBA").save(path)


def load_lora_metadata(lora_dir: Path):
    """
    Try to load metadata.json written during training so we can auto-fill rank/alpha.
    """
    metadata_path = lora_dir / "metadata.json"
    if not metadata_path.exists():
        return None, metadata_path
    try:
        import json

        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta, metadata_path
    except Exception as e:  # pragma: no cover - best-effort helper
        print(f"Warning: failed to read LoRA metadata at {metadata_path}: {e}")
        return None, metadata_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: predict text_alpha from RGBA input using FluxTextAlphaModel")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--rgba_vae_path", type=str, required=True)
    p.add_argument("--vae_subfolder", type=str, default="ae")
    p.add_argument("--lora_path", type=str, default=None, help="Directory containing saved LoRA weights (FluxPipeline.save_lora_weights format).")
    p.add_argument("--rank", type=int, default=96, help="LoRA rank (must match training).")
    p.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha (must match training).")
    p.add_argument("--input_image", type=str, required=True, help="Path to RGBA input image (condition).")
    p.add_argument("--output_path", type=str, required=True, help="Where to save predicted text_alpha RGBA.")
    p.add_argument("--steps", type=int, default=20, help="Number of flow steps during sampling.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic sampling.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--hf_token", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    generator = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)

    lora_metadata = None
    lora_metadata_path = None
    if args.lora_path:
        lora_metadata, lora_metadata_path = load_lora_metadata(Path(args.lora_path))
        if lora_metadata:
            meta_rank = lora_metadata.get("rank")
            meta_alpha = lora_metadata.get("alpha")
            if meta_rank is not None:
                args.rank = int(meta_rank)
            if meta_alpha is not None:
                args.lora_alpha = int(meta_alpha)
            print(
                f"Loaded LoRA metadata from {lora_metadata_path}: rank={args.rank} alpha={args.lora_alpha}"
            )

    model = FluxTextAlphaModel(
        args.pretrained_model_name_or_path,
        vae_path=args.rgba_vae_path,
        vae_subfolder=args.vae_subfolder,
        token=args.hf_token,
        device=device,
        weight_dtype=weight_dtype,
    )

    if args.lora_path:
        model.load_lora(args.lora_path, rank=args.rank, lora_alpha=args.lora_alpha)

    # Match validation mode: disable dropout, etc.
    model.transformer.eval()
    model.vae.eval()

    inp = load_rgba(Path(args.input_image)).unsqueeze(0)  # B=1
    with torch.no_grad():
        pred = model.sample(inp, num_inference_steps=args.steps, generator=generator)
    # pred shape: [1, 4, H, W]
    save_rgba(pred[0], Path(args.output_path))
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()

