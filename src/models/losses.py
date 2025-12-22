from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

try:  # pragma: no cover - optional dependency
    import lpips as lpips_lib  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    lpips_lib = None


class AlphaVaeLoss(nn.Module):
    """
    Loss bundle that mirrors AlphaVAE's Appendix (RGBAVAELoss) without the GAN terms.

    - Reconstruction loss follows Eq. (9) using Eb/Eb² channel priors.
    - Optional LPIPS perceptual loss blends over black/white backgrounds.
    - KL and reference-KL (against frozen VAE) share identical reduction logic.
    """

    def __init__(
        self,
        *,
        reduce_mean: bool = False,
        use_naive_mse: bool = False,
        use_lpips: bool = False,
        custom_eb: Optional[Sequence[float]] = None,
        custom_eb2: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if custom_eb is None:
            custom_eb = (-0.0357, -0.0811, -0.1797)
        if custom_eb2 is None:
            custom_eb2 = (0.3163, 0.3060, 0.3634)
        if len(custom_eb) != 3 or len(custom_eb2) != 3:
            raise ValueError("custom_eb/custom_eb2 must each provide three channel weights.")

        self.reduce_mean = reduce_mean
        self.use_naive_mse = use_naive_mse
        self.use_lpips = use_lpips

        eb_tensor = torch.tensor(custom_eb, dtype=torch.float32).view(1, 3, 1, 1)
        eb2_tensor = torch.tensor(custom_eb2, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("eb", eb_tensor, persistent=False)
        self.register_buffer("eb2", eb2_tensor, persistent=False)

        if self.use_lpips:
            if lpips_lib is None:
                raise ImportError(
                    "LPIPS loss requested but the 'lpips' package is not installed. "
                    "Install it via `pip install lpips` or disable lpips_scale in the config."
                )
            self.lpips = lpips_lib.LPIPS(net="vgg")
            self.lpips.requires_grad_(False)
            self.lpips.eval()

    # PyTorch's nn.Module.to does not automatically move nested LPIPS modules when created lazily.
    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        if self.use_lpips:
            self.lpips.to(*args, **kwargs)
        return self

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        `pred` / `target` are expected in [-1, 1] with channels ordered as RGBA.
        """
        if self.use_naive_mse:
            return self._reduce((pred - target).pow(2))

        target_rgb = target[:, :3]
        pred_rgb = pred[:, :3]
        target_alpha = (target[:, 3:] + 1.0) * 0.5
        pred_alpha = (pred[:, 3:] + 1.0) * 0.5

        rgba_diff = target_rgb * target_alpha - pred_rgb * pred_alpha
        alpha_diff = target_alpha - pred_alpha

        loss = rgba_diff.pow(2) - 2.0 * self.eb * rgba_diff * alpha_diff + self.eb2 * alpha_diff.pow(2)
        return self._reduce(loss)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_lpips:
            raise RuntimeError("perceptual_loss called while LPIPS is disabled.")

        target_rgb = target[:, :3]
        pred_rgb = pred[:, :3]
        target_alpha = (target[:, 3:] + 1.0) * 0.5
        pred_alpha = (pred[:, 3:] + 1.0) * 0.5

        target_black = target_rgb * target_alpha
        pred_black = pred_rgb * pred_alpha
        target_white = target_rgb * target_alpha + (1.0 - target_alpha)
        pred_white = pred_rgb * pred_alpha + (1.0 - pred_alpha)

        # LPIPS expects float32 tensors in [-1, 1].
        target_black = target_black.to(torch.float32)
        pred_black = pred_black.to(torch.float32)
        target_white = target_white.to(torch.float32)
        pred_white = pred_white.to(torch.float32)

        loss_black = self.lpips(target_black, pred_black)
        loss_white = self.lpips(target_white, pred_white)
        return 0.5 * (loss_black.mean() + loss_white.mean())

    def kl_loss(
        self,
        posterior: DiagonalGaussianDistribution,
        reference: Optional[DiagonalGaussianDistribution] = None,
    ) -> torch.Tensor:
        loss = posterior.kl(reference)
        return self._reduce(loss)

    def _reduce(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value
        if self.reduce_mean:
            return value.mean()
        value = value.view(value.shape[0], -1)
        return value.sum(dim=1).mean()

