"""
RGBA-VAE built directly on top of the pretrained Qwen-Image VAE (AutoencoderKL).

Following §3.1 of Qwen-Image-Layered and AlphaVAE [wang2025alphavae]:
- extend the first encoder convolution and last decoder convolution from 3→4 channels
  by copying RGB weights and zero-initializing the new alpha path (optionally biased).
- train on both RGB (alpha=1) and RGBA data, sharing a single latent space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

try:  # pragma: no cover - optional dependency
    from safetensors.torch import load_file as load_safetensors  # type: ignore
except Exception:  # pragma: no cover - handled when loading weights
    load_safetensors = None


def _ensure_alpha(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 4:
        return x
    alpha = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    return torch.cat([x, alpha], dim=1)


def _to_vae_range(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _from_vae_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


def _normalize_background(
    background: Union[float, int, Sequence[float], torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Ensure the background tensor matches the spatial dimensions of the reference RGB tensor.
    """
    device = reference.device
    dtype = reference.dtype
    batch, _, height, width = reference.shape

    if isinstance(background, torch.Tensor):
        bg = background.to(device=device, dtype=dtype)
        if bg.dim() == 3:
            bg = bg.unsqueeze(0)
        if bg.dim() != 4:
            raise ValueError(f"Background tensor must have 3 or 4 dimensions, got {bg.dim()}")
        if bg.shape[0] == 1 and batch > 1:
            bg = bg.expand(batch, -1, -1, -1)
        if bg.shape[1] == 1:
            bg = bg.repeat(1, 3, 1, 1)
        if bg.shape[2] != height or bg.shape[3] != width:
            raise ValueError("Background tensor spatial size must match the RGBA tensor.")
        return bg

    if isinstance(background, Sequence):
        if len(background) != 3:
            raise ValueError("Background color sequence must contain exactly three values.")
        color = torch.tensor(background, dtype=dtype, device=device).view(1, 3, 1, 1)
        return color.expand(batch, -1, height, width)

    value = float(background)
    return torch.full((batch, 3, height, width), value, dtype=dtype, device=device)


def composite_over_background(
    rgba: torch.Tensor,
    background: Union[float, int, Sequence[float], torch.Tensor],
) -> torch.Tensor:
    """Alpha composite RGBA tensors over a background color or tensor."""
    rgba = _ensure_alpha(rgba)
    rgb = rgba[:, :3]
    alpha = rgba[:, 3:4]
    bg = _normalize_background(background, rgb)
    return rgb * alpha + bg * (1.0 - alpha)


def composite_over_white(rgba: torch.Tensor) -> torch.Tensor:
    return composite_over_background(rgba, 1.0)


def composite_over_black(rgba: torch.Tensor) -> torch.Tensor:
    return composite_over_background(rgba, 0.0)


def adapt_vae_to_rgba(vae: AutoencoderKL, alpha_bias_init: float = 0.0) -> None:
    """Mutate a pretrained RGB AutoencoderKL into RGBA by widening conv layers."""
    conv_in: nn.Conv2d = vae.encoder.conv_in
    if conv_in.in_channels != 4:
        weight = conv_in.weight.data
        new_weight = torch.zeros(weight.size(0), 4, *weight.shape[2:], dtype=weight.dtype, device=weight.device)
        new_weight[:, :3] = weight
        conv_in.in_channels = 4
        conv_in.weight = nn.Parameter(new_weight)
        if conv_in.bias is not None:
            conv_in.bias = nn.Parameter(conv_in.bias.data.clone())

    conv_out: nn.Conv2d = vae.decoder.conv_out
    if conv_out.out_channels != 4:
        weight = conv_out.weight.data
        new_weight = torch.zeros(4, weight.size(1), *weight.shape[2:], dtype=weight.dtype, device=weight.device)
        new_weight[:3] = weight
        conv_out.out_channels = 4
        conv_out.weight = nn.Parameter(new_weight)
        if conv_out.bias is not None:
            new_bias = torch.zeros(4, dtype=conv_out.bias.dtype, device=conv_out.bias.device)
            new_bias[:3] = conv_out.bias.data
            new_bias[3] = alpha_bias_init
            conv_out.bias = nn.Parameter(new_bias)
        else:
            conv_out.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, alpha_bias_init], dtype=new_weight.dtype, device=new_weight.device))

    vae.config.in_channels = 4
    vae.config.out_channels = 4


def _resolve_checkpoint_dir(model_name_or_path: Union[str, Path], subfolder: Optional[str]) -> Optional[Path]:
    base_path = Path(model_name_or_path)
    if not base_path.exists():
        return None
    if subfolder is not None and subfolder != "":
        base_path = base_path / subfolder
    return base_path if base_path.exists() else None


def _locate_weight_file(directory: Path) -> Optional[Path]:
    for filename in ("diffusion_pytorch_model.safetensors", "pytorch_model.bin"):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _maybe_restore_rgba_convs(vae: AutoencoderKL, model_name_or_path: str, subfolder: Optional[str]) -> None:
    """
    When the checkpoint already contains RGBA convolutions (4 input/output channels),
    reload those tensors manually to avoid NaNs introduced by ignore_mismatched_sizes.
    """
    checkpoint_dir = _resolve_checkpoint_dir(model_name_or_path, subfolder)
    if checkpoint_dir is None:
        return

    weight_file = _locate_weight_file(checkpoint_dir)
    if weight_file is None:
        return

    try:
        if weight_file.suffix == ".safetensors":
            if load_safetensors is None:
                return
            state_dict = load_safetensors(str(weight_file))
        else:
            state_dict = torch.load(weight_file, map_location="cpu")
    except Exception:
        return

    conv_in_weight = state_dict.get("encoder.conv_in.weight")
    conv_in_bias = state_dict.get("encoder.conv_in.bias")
    conv_out_weight = state_dict.get("decoder.conv_out.weight")
    conv_out_bias = state_dict.get("decoder.conv_out.bias")

    if conv_in_weight is None or conv_out_weight is None:
        return
    if conv_in_weight.shape[1] != 4 or conv_out_weight.shape[0] != 4:
        return

    conv_in = vae.encoder.conv_in
    conv_out = vae.decoder.conv_out
    with torch.no_grad():
        conv_in.weight.copy_(conv_in_weight.to(conv_in.weight.dtype, copy=False))
        if conv_in_bias is not None and conv_in.bias is not None:
            conv_in.bias.copy_(conv_in_bias.to(conv_in.bias.dtype, copy=False))
        conv_out.weight.copy_(conv_out_weight.to(conv_out.weight.dtype, copy=False))
        if conv_out_bias is not None and conv_out.bias is not None:
            conv_out.bias.copy_(conv_out_bias.to(conv_out.bias.dtype, copy=False))

    for name, tensor in {
        "encoder.conv_in.weight": conv_in.weight,
        "decoder.conv_out.weight": conv_out.weight,
    }.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"{name} contains NaN/Inf after loading RGBA checkpoint.")


class RgbaVAE(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        beta: float = 0.25,
        alpha_loss_weight: float = 1.0,
        alpha_l1_weight: float = 0.0,
        rgb_loss_weight: float = 1.0,
        white_bg_weight: float = 0.0,
        black_bg_weight: float = 0.0,
        loss_reduce_mean: bool = False,
        use_naive_mse: bool = False,
        custom_eb: Optional[Sequence[float]] = None,
        custom_eb2: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.alpha_loss_weight = alpha_loss_weight
        self.alpha_l1_weight = alpha_l1_weight
        self.rgb_loss_weight = rgb_loss_weight
        self.white_bg_weight = white_bg_weight
        self.black_bg_weight = black_bg_weight
        self.loss_reduce_mean = loss_reduce_mean
        self.use_naive_mse = use_naive_mse
        if custom_eb is None:
            custom_eb = (-0.0357, -0.0811, -0.1797)
        if custom_eb2 is None:
            custom_eb2 = (0.3163, 0.3060, 0.3634)
        if len(custom_eb) != 3 or len(custom_eb2) != 3:
            raise ValueError("custom_eb and custom_eb2 must each provide three channel weights.")
        eb_tensor = torch.tensor(custom_eb, dtype=torch.float32).view(1, 3, 1, 1)
        eb2_tensor = torch.tensor(custom_eb2, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("alphavae_eb", eb_tensor, persistent=False)
        self.register_buffer("alphavae_eb2", eb2_tensor, persistent=False)

    @classmethod
    def from_pretrained_rgb(
        cls,
        model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        torch_dtype: Optional[torch.dtype] = torch.float32,
        alpha_bias_init: float = 0.0,
        beta: float = 0.25,
        alpha_loss_weight: float = 1.0,
        alpha_l1_weight: float = 0.0,
        rgb_loss_weight: float = 1.0,
        white_bg_weight: float = 0.0,
        black_bg_weight: float = 0.0,
        device: Optional[torch.device] = None,
        loss_reduce_mean: bool = False,
        use_naive_mse: bool = False,
        custom_eb: Optional[Sequence[float]] = None,
        custom_eb2: Optional[Sequence[float]] = None,
    ) -> "RgbaVAE":
        vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
        adapt_vae_to_rgba(vae, alpha_bias_init=alpha_bias_init)
        _maybe_restore_rgba_convs(vae, model_name_or_path, subfolder)
        if device is not None:
            vae = vae.to(device)
        return cls(
            vae=vae,
            beta=beta,
            alpha_loss_weight=alpha_loss_weight,
            alpha_l1_weight=alpha_l1_weight,
            rgb_loss_weight=rgb_loss_weight,
            white_bg_weight=white_bg_weight,
            black_bg_weight=black_bg_weight,
            loss_reduce_mean=loss_reduce_mean,
            use_naive_mse=use_naive_mse,
            custom_eb=custom_eb,
            custom_eb2=custom_eb2,
        )

    def forward(self, x: torch.Tensor):
        x_rgba = _ensure_alpha(x)
        vae_input = _to_vae_range(x_rgba)
        posterior = self.vae.encode(vae_input).latent_dist
        z = posterior.sample()
        recon = self.vae.decode(z).sample
        recon = torch.clamp(_from_vae_range(recon), 0.0, 1.0)
        return recon, posterior

    def loss(self, recon: torch.Tensor, target: torch.Tensor, posterior) -> torch.Tensor:
        target_rgba = _ensure_alpha(target)
        recon_rgba = _ensure_alpha(recon)
        target_scaled = target_rgba * 2.0 - 1.0
        recon_scaled = recon_rgba * 2.0 - 1.0

        losses = []
        if self.rgb_loss_weight > 0.0:
            if self.use_naive_mse:
                mse_value = (recon_rgba[:, :3] - target_rgba[:, :3]).pow(2)
                base = self._reduce_loss(mse_value)
            else:
                base = self._alphavae_reconstruction_loss(recon_scaled, target_scaled)
            losses.append(self.rgb_loss_weight * base)

        if self.white_bg_weight > 0.0:
            recon_white = composite_over_white(recon_rgba)
            target_white = composite_over_white(target_rgba)
            losses.append(self.white_bg_weight * F.mse_loss(recon_white, target_white))

        if self.black_bg_weight > 0.0:
            recon_black = composite_over_black(recon_rgba)
            target_black = composite_over_black(target_rgba)
            losses.append(self.black_bg_weight * F.mse_loss(recon_black, target_black))

        if self.alpha_loss_weight > 0.0 and recon_rgba.shape[1] > 3:
            losses.append(self.alpha_loss_weight * F.mse_loss(recon_rgba[:, 3:], target_rgba[:, 3:]))

        if self.alpha_l1_weight > 0.0 and recon_rgba.shape[1] > 3:
            losses.append(self.alpha_l1_weight * F.l1_loss(recon_rgba[:, 3:], target_rgba[:, 3:]))

        kl = posterior.kl().mean()
        losses.append(self.beta * kl)
        return sum(losses)

    def _reduce_loss(self, value: torch.Tensor) -> torch.Tensor:
        if self.loss_reduce_mean:
            return value.mean()
        value = value.view(value.shape[0], -1)
        return value.sum(dim=1).mean()

    def _alphavae_reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_rgb = target[:, :3]
        pred_rgb = recon[:, :3]
        target_alpha = (target[:, 3:] + 1.0) * 0.5
        pred_alpha = (recon[:, 3:] + 1.0) * 0.5
        rgba_diff = target_rgb * target_alpha - pred_rgb * pred_alpha
        alpha_diff = target_alpha - pred_alpha
        loss = (
            rgba_diff.pow(2)
            - 2.0 * self.alphavae_eb * rgba_diff * alpha_diff
            + self.alphavae_eb2 * alpha_diff.pow(2)
        )
        return self._reduce_loss(loss)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return recon

