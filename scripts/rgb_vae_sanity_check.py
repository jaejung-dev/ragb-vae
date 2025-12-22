#!/usr/bin/env python3
"""
Sanity-check reconstruction quality using the original RGB Qwen-Image VAE.

This mirrors the RGBA-VAE visualization but drops alpha and feeds RGB only.
Saves a checkerboard grid comparing each component layer vs. its reconstruction.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL

from src.data.multilayer_dataset import MultiLayerDataset


def tensor_to_rgba(tensor: torch.Tensor) -> np.ndarray:
    """Convert (4,H,W) tensor in [0,1] to numpy array for visualization."""
    arr = tensor.detach().cpu().numpy()
    rgba = np.transpose(arr, (1, 2, 0))
    rgba = np.clip(rgba, 0.0, 1.0)
    return np.nan_to_num(rgba, nan=0.0)


def checkerboard(height: int, width: int, cell: int = 32) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pattern = ((yy // cell) + (xx // cell)) % 2
    board = np.where(pattern[..., None] == 0, 0.8, 0.6)
    return board.astype(np.float32)


def load_rgb_vae(repo: str, subfolder: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        repo,
        subfolder=subfolder,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    return vae.to(device)


def reconstruct_rgb(vae: AutoencoderKL, component: torch.Tensor) -> torch.Tensor:
    """component: (1,3,H,W) in [0,1]; returns reconstruction (1,3,H,W) in [0,1]."""
    with torch.no_grad():
        vae_input = component * 2.0 - 1.0
        posterior = vae.encode(vae_input).latent_dist
        latents = posterior.sample()
        recon = vae.decode(latents).sample
        recon = torch.clamp((recon + 1.0) * 0.5, 0.0, 1.0)
        return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered-root", type=str, default="/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered")
    parser.add_argument("--json-root", type=str, default="/home/ubuntu/jjseol/layer_data/inpainting_250k_subset")
    parser.add_argument("--sample-index", type=int, default=0, help="Index into the rendered dataset.")
    parser.add_argument("--max-components", type=int, default=12, help="Number of components to visualize.")
    parser.add_argument("--rgb-vae", type=str, default="Qwen/Qwen-Image", help="HF repo or local dir of the RGB VAE.")
    parser.add_argument("--vae-subfolder", type=str, default="vae", help="Subfolder inside the RGB VAE repo.")
    parser.add_argument("--overlay-background", action="store_true", help="Composite component over background before encoding.")
    args = parser.parse_args()

    ds = MultiLayerDataset(
        rendered_root=Path(args.rendered_root),
        json_root=Path(args.json_root),
        alpha_threshold=0,
    )
    sample = ds[args.sample_index]

    components = sample.components
    if not components:
        print("Sample has no components to visualize.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_rgb_vae(args.rgb_vae, args.vae_subfolder, device)
    vae.eval()

    cards = []
    background = sample.background

    for idx, component in enumerate(components[: args.max_components]):
        comp_rgba = component.clone()
        if args.overlay_background:
            alpha = comp_rgba[3:4]
            bg_rgb = background[:3]
            comp_rgba[:3] = comp_rgba[:3] * alpha + bg_rgb * (1.0 - alpha)
            comp_rgba[3:] = torch.ones_like(alpha)

        rgb = comp_rgba[:3].unsqueeze(0).to(device)
        recon = reconstruct_rgb(vae, rgb).cpu()
        rgba = tensor_to_rgba(comp_rgba)

        recon_rgba = np.zeros_like(rgba)
        recon_rgba[..., :3] = np.transpose(recon.squeeze(0).numpy(), (1, 2, 0))
        recon_rgba[..., 3] = rgba[..., 3]

        cards.append((idx, rgba, recon_rgba))

    if not cards:
        print("No components selected for visualization.")
        return

    h, w = cards[0][1].shape[:2]
    board = checkerboard(h, w)
    rows = len(cards)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (idx, rgba, recon_rgba) in enumerate(cards):
        for col, (title, image) in enumerate(
            (
                (f"Component #{idx}", rgba),
                ("RGB VAE recon", recon_rgba),
            )
        ):
            ax = axes[row, col]
            ax.imshow(board, cmap="gray", interpolation="nearest")
            ax.imshow(image[..., :3], alpha=image[..., 3], interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rgb_vae_component_recon_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved RGB VAE reconstruction grid to {out_path}")


if __name__ == "__main__":
    main()

