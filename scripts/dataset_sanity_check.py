#!/usr/bin/env python3
"""
Quick sanity check for the multilayer dataset and dataloader.

Usage:
    python scripts/dataset_sanity_check.py --max-samples 2 --batch-size 1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.multilayer_dataset import MultiLayerDataset, multilayer_collate
from src.models import RgbaVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--alpha-threshold", type=int, default=100)
    args = parser.parse_args()

    ds = MultiLayerDataset(alpha_threshold=args.alpha_threshold, max_samples=args.max_samples)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multilayer_collate,
    )

    batch = next(iter(dl))
    print("Batch keys:", list(batch.keys()))
    print("background:", batch["background"].shape)
    print("composite:", batch["composite"].shape)
    print("components:", batch["components"].shape)
    print("component_mask:", batch["component_mask"].shape, batch["component_mask"].dtype)
    print("visible_masks:", batch["visible_masks"].shape)
    print("sample_dirs:", batch["sample_dirs"])

    # Quick reconstruction sanity check using RGBA-VAE (if checkpoint exists)
    ckpt_dir = Path("checkpoints/rgba_vae_init")
    if ckpt_dir.exists():
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae = RgbaVAE.from_pretrained_rgb(
                model_name_or_path=str(ckpt_dir),
                subfolder=None,
                torch_dtype=torch.float32,
                device=device,
            )
            vae.eval()
            component_mask = batch["component_mask"][0]
            valid_indices = torch.nonzero(component_mask, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                print("No valid components in batch; skipping reconstruction test.")
                return

            def tensor_to_rgba(t: torch.Tensor) -> np.ndarray:
                arr = t.squeeze(0).permute(1, 2, 0).numpy()
                arr = np.clip(arr, 0.0, 1.0)
                return np.nan_to_num(arr, nan=0.0)

            def checkerboard(h: int, w: int, cell: int = 32) -> np.ndarray:
                yy, xx = np.indices((h, w))
                pattern = ((yy // cell) + (xx // cell)) % 2
                board = np.where(pattern[..., None] == 0, 0.8, 0.6)
                return board.astype(np.float32)

            comps = []
            recons = []
            for idx in valid_indices.tolist():
                sample = batch["components"][0, idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    recon, _ = vae(sample)
                comps.append(tensor_to_rgba(sample.cpu()))
                recons.append(tensor_to_rgba(recon.cpu()))

            h, w = comps[0].shape[:2]
            board = checkerboard(h, w)
            rows = len(comps)
            fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
            if rows == 1:
                axes = np.expand_dims(axes, axis=0)
            for row, (orig_rgba, recon_rgba) in enumerate(zip(comps, recons)):
                for col, title, rgba in (
                    (0, f"Component #{valid_indices[row].item()}", orig_rgba),
                    (1, "Reconstruction", recon_rgba),
                ):
                    ax = axes[row, col]
                    ax.imshow(board, cmap="gray", interpolation="nearest")
                    ax.imshow(rgba[..., :3], alpha=rgba[..., 3], interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")
            plt.tight_layout()
            out_path = Path("outputs") / "rgba_vae_component_recon_grid.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved reconstruction preview to {out_path}")
        except FileNotFoundError:
            print("RGBA VAE checkpoint not found; skipping reconstruction test.")
    else:
        print("checkpoints/rgba_vae_init not found; skipping VAE sanity check.")


if __name__ == "__main__":
    main()


