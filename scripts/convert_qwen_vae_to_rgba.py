#!/usr/bin/env python3
"""
Convert a pretrained RGB VAE checkpoint (Qwen or Flux) into an RGBA-ready directory.

The logic mirrors AlphaVAE's convert.py: widen the first encoder conv (input 4 channels)
and the last decoder conv (output 4 channels) by copying RGB weights and initializing
the alpha path to zeros (bias optionally set via --alpha-bias-init).
"""
import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from src.models.rgba_vae import adapt_vae_to_rgba


def convert_qwen(
    source: str,
    subfolder: str,
    torch_dtype: torch.dtype,
    alpha_bias_init: float,
) -> AutoencoderKL:
    """Load a Qwen-Image AutoencoderKL and adapt it to RGBA."""
    vae = AutoencoderKL.from_pretrained(
        source,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    adapt_vae_to_rgba(vae, alpha_bias_init=alpha_bias_init)
    return vae


def convert_flux(
    source: str,
    subfolder: str,
    torch_dtype: torch.dtype,
    alpha_bias_init: float,
) -> AutoencoderKL:
    """
    Load a Flux.1 AutoencoderKL and adapt it to RGBA.

    Flux checkpoints typically store the VAE under the `ae` subfolder.
    """
    vae = AutoencoderKL.from_pretrained(
        source,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    adapt_vae_to_rgba(vae, alpha_bias_init=alpha_bias_init)
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Hugging Face repo or local directory with the RGB VAE.")
    parser.add_argument(
        "--arch",
        default="qwen",
        choices=["qwen", "flux"],
        help="Base VAE family to convert.",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Subfolder inside --source (defaults to 'vae' for Qwen or 'ae' for Flux).",
    )
    parser.add_argument("--alpha-bias-init", type=float, default=0.0, help="Initial bias for alpha channel.")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-dir", required=True, help="Directory to save the converted RGBA VAE (HuggingFace format).")
    parser.add_argument("--state-dict", action="store_true", help="Also dump pytorch_model.bin alongside the config.")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    default_subfolder = "ae" if args.arch == "flux" else "vae"
    subfolder = args.subfolder or default_subfolder

    if args.arch == "flux":
        vae = convert_flux(
            args.source,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            alpha_bias_init=args.alpha_bias_init,
        )
    else:
        vae = convert_qwen(
            args.source,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            alpha_bias_init=args.alpha_bias_init,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(output_dir)
    if args.state_dict:
        torch.save(vae.state_dict(), output_dir / "pytorch_model.bin")
    print(f"Saved RGBA VAE to {output_dir}")


if __name__ == "__main__":
    main()

