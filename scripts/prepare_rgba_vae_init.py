#!/usr/bin/env python3
"""
Helper to convert a Qwen or Flux VAE into RGBA format and stash it under
`checkpoints/rgba_vae_init` for Stage 1 training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from convert_qwen_vae_to_rgba import convert_flux, convert_qwen


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RGBA-ready VAE weights.")
    parser.add_argument("--source", required=True, help="Hugging Face repo or local path to the RGB VAE.")
    parser.add_argument("--arch", default="qwen", choices=["qwen", "flux"], help="Base model family.")
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Subfolder containing the VAE (defaults to 'vae' for Qwen or 'ae' for Flux).",
    )
    parser.add_argument("--alpha-bias-init", type=float, default=0.0, help="Initial bias for the alpha channel.")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"], help="Load dtype.")
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR.parent / "checkpoints" / "rgba_vae_init"),
        help="Target directory to store the converted VAE.",
    )
    parser.add_argument("--state-dict", action="store_true", help="Also dump pytorch_model.bin.")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    default_subfolder = "ae" if args.arch == "flux" else "vae"
    subfolder = args.subfolder or default_subfolder

    if args.arch == "flux":
        vae = convert_flux(
            source=args.source,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            alpha_bias_init=args.alpha_bias_init,
        )
    else:
        vae = convert_qwen(
            source=args.source,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            alpha_bias_init=args.alpha_bias_init,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(output_dir)
    if args.state_dict:
        torch.save(vae.state_dict(), output_dir / "pytorch_model.bin")
    print(f"[prepare_rgba_vae_init] Saved {args.arch} RGBA VAE to {output_dir}")


if __name__ == "__main__":
    main()

