from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from torch.nn.utils import clip_grad_norm_
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.multilayer_dataset import MultiLayerDataset, multilayer_collate
from src.data_generation import create_component_dataloader
from src.models import AlphaVaeLoss, RgbaVAE
from src.models.rgba_vae import (
    composite_over_background,
    composite_over_black,
    composite_over_white,
)


TensorDict = Dict[str, torch.Tensor]


def _ensure_finite(value: torch.Tensor, name: str, *, epoch: int, step: int, accelerator: Accelerator) -> None:
    if not torch.isfinite(value).all():
        accelerator.print(
            f"[NaNGuard] epoch={epoch} step={step} detected non-finite '{name}' "
            f"(min={value.min().item():.4e} max={value.max().item():.4e})"
        )
        raise RuntimeError(f"Non-finite tensor encountered in '{name}'")


class RandomBackgroundBlend:
    """
    Blend RGBA tensors onto a random opaque background with a small probability.
    The operation follows AlphaVAE's detail-augmentation trick to expose the VAE
    to background-only supervision.
    """

    def __init__(
        self,
        prob: float = 0.1,
        keys: Sequence[str] = ("component",),
        color_range: Tuple[float, float] = (0.2, 0.9),
    ) -> None:
        self.prob = prob
        self.keys = tuple(keys)
        if color_range[0] >= color_range[1]:
            raise ValueError("color_range lower bound must be < upper bound.")
        self.color_range = color_range

    def __call__(self, sample: TensorDict) -> TensorDict:
        if random.random() >= self.prob:
            return sample
        augmented = dict(sample)
        for key in self.keys:
            tensor = augmented.get(key)
            if tensor is None:
                continue
            augmented[key] = self._blend_tensor(tensor)
        augmented["background_augmented"] = True
        return augmented

    def _blend_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        rgb = tensor[:3]
        alpha = tensor[3:4]
        device = tensor.device
        dtype = tensor.dtype
        color = torch.empty((3, 1, 1), dtype=dtype, device=device).uniform_(*self.color_range)
        bg = color.expand_as(rgb)
        blended = rgb * alpha + bg * (1.0 - alpha)
        new_alpha = torch.ones_like(alpha)
        return torch.cat([blended, new_alpha], dim=0)


def build_dataloader(cfg: Dict[str, Any], *, split: Optional[str] = None) -> DataLoader:
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "multilayer")
    target_split = split or "train"
    train_mode = target_split == "train"
    val_shuffle = bool(data_cfg.get("val_shuffle", False))

    if source == "bucket":
        dataset_kwargs = data_cfg.get("dataset_kwargs", {"include_metadata": False})
        val_dataset_kwargs = data_cfg.get("val_dataset_kwargs", dataset_kwargs)

        if target_split == "val":
            split_name = data_cfg.get("bucket_val_split", "val")
            shuffle = val_shuffle
            extra_kwargs = val_dataset_kwargs
        else:
            split_name = data_cfg.get("bucket_split", "train")
            shuffle = data_cfg.get("shuffle", True)
            extra_kwargs = dataset_kwargs

        transform = None
        if train_mode:
            blend_prob = float(data_cfg.get("background_blend_prob", 0.0))
            if blend_prob > 0.0:
                targets = data_cfg.get("background_blend_targets", ["component", "composite"])
                color_range = tuple(data_cfg.get("background_color_range", [0.2, 0.9]))
                transform = RandomBackgroundBlend(prob=blend_prob, keys=targets, color_range=color_range)  # type: ignore[arg-type]

        return create_component_dataloader(
            root_dir=data_cfg.get("bucket_root", "data/rgba_layers"),
            manifest_path=data_cfg.get("bucket_manifest"),
            split=split_name,
            batch_size=data_cfg.get("batch_size", 4),
            shuffle=shuffle,
            num_workers=data_cfg.get("num_workers", 4),
            limit=data_cfg.get("limit"),
            transform=transform,
            dataset_kwargs=extra_kwargs,
        )

    ds = MultiLayerDataset(
        rendered_root=Path(data_cfg["rendered_root"]),
        json_root=Path(data_cfg["json_root"]),
        alpha_threshold=data_cfg.get("alpha_threshold", 100),
        max_samples=data_cfg.get("max_samples"),
    )
    should_shuffle = train_mode or (target_split == "val" and val_shuffle)
    dl = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=should_shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=multilayer_collate,
        pin_memory=True,
    )
    return dl


def train_rgba_vae(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    mixed_precision = train_cfg.get("mixed_precision", "no")
    if isinstance(mixed_precision, bool):
        mixed_precision = "fp16" if mixed_precision else "no"

    deepspeed_plugin: Optional[DeepSpeedPlugin] = None
    ds_config_path = train_cfg.get("deepspeed_config")
    if ds_config_path:
        resolved_path = Path(ds_config_path)
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"training.deepspeed_config file not found at {resolved_path}")
        with resolved_path.open("r", encoding="utf-8") as f:
            ds_config = json.load(f)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )
    device = accelerator.device

    rgb_ckpt = model_cfg.get("rgb_checkpoint")
    if not rgb_ckpt:
        raise ValueError("model.rgb_checkpoint must point to the converted VAE directory.")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "float32"), torch.float32)

    base_arch = model_cfg.get("base_arch", "qwen").lower()
    default_subfolder = "ae" if "flux" in base_arch else "vae"
    rgb_subfolder = model_cfg.get("rgb_subfolder")
    subfolder = default_subfolder if rgb_subfolder is None else rgb_subfolder

    model = RgbaVAE.from_pretrained_rgb(
        model_name_or_path=rgb_ckpt,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        alpha_bias_init=model_cfg.get("alpha_bias_init", 0.0),
        beta=model_cfg.get("beta", 0.25),
        alpha_loss_weight=model_cfg.get("alpha_loss_weight", 1.0),
        alpha_l1_weight=model_cfg.get("alpha_l1_weight", 0.0),
        rgb_loss_weight=model_cfg.get("rgb_loss_weight", 1.0),
        white_bg_weight=model_cfg.get("white_bg_loss_weight", 0.0),
        black_bg_weight=model_cfg.get("black_bg_loss_weight", 0.0),
        device=device,
    )

    if train_cfg.get("vae_tiling", True):
        model.vae.enable_tiling()
    else:
        model.vae.disable_tiling()

    if train_cfg.get("vae_slicing", True):
        model.vae.enable_slicing()
    else:
        model.vae.disable_slicing()

    if train_cfg.get("vae_gradient_checkpointing", False):
        model.vae.enable_gradient_checkpointing()

    train_loader = build_dataloader(cfg, split="train")
    try:
        train_batches_total = len(train_loader)
    except TypeError:
        train_batches_total = None
    val_loader = None
    if train_cfg.get("run_validation", True):
        try:
            val_loader = build_dataloader(cfg, split="val")
        except Exception:
            val_loader = None

    lr = float(train_cfg.get("learning_rate", 1e-4))
    epochs = train_cfg.get("epochs", 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,  betas=(0.5, 0.9))

    loss_module = AlphaVaeLoss(
        reduce_mean=train_cfg.get("loss_reduce_mean", False),
        use_naive_mse=train_cfg.get("use_naive_mse", False),
        use_lpips=float(train_cfg.get("lpips_scale", 0.0) or 0.0) > 0.0,
        custom_eb=model_cfg.get("loss_eb"),
        custom_eb2=model_cfg.get("loss_eb2"),
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    try:
        progress_total = len(train_loader)
    except TypeError:
        progress_total = train_batches_total
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    loss_module = loss_module.to(accelerator.device)
    lpips_scale = float(train_cfg.get("lpips_scale", 0.0) or 0.0)
    kl_scale = train_cfg.get("kl_scale")
    if kl_scale is not None:
        kl_scale = float(kl_scale)
    ref_kl_scale = train_cfg.get("ref_kl_scale")
    if ref_kl_scale is not None:
        ref_kl_scale = float(ref_kl_scale)

    log_every = train_cfg.get("log_every", 50)
    max_grad_norm = train_cfg.get("max_grad_norm")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
    ckpt_every_steps = int(train_cfg.get("ckpt_every_steps", 0) or 0)
    val_every_steps = int(train_cfg.get("val_every_steps", 500))
    background_sample_prob = float(data_cfg.get("background_sample_prob", 0.0))
    global_step = 0
    performed_validation = False

    weight_dtype = next(model.parameters()).dtype
    ref_vae = None
    ref_vae_dtype: Optional[torch.dtype] = None
    if ref_kl_scale and ref_kl_scale > 0.0:
        ref_ckpt = model_cfg.get("ref_rgb_checkpoint") or rgb_ckpt
        ref_rgb_subfolder = model_cfg.get("ref_rgb_subfolder")
        ref_subfolder = subfolder if ref_rgb_subfolder is None else ref_rgb_subfolder
        ref_model = RgbaVAE.from_pretrained_rgb(
            model_name_or_path=ref_ckpt,
            subfolder=ref_subfolder,
            torch_dtype=weight_dtype,
            alpha_bias_init=model_cfg.get("alpha_bias_init", 0.0),
            beta=model_cfg.get("beta", 0.25),
            alpha_loss_weight=model_cfg.get("alpha_loss_weight", 1.0),
            alpha_l1_weight=model_cfg.get("alpha_l1_weight", 0.0),
            rgb_loss_weight=model_cfg.get("rgb_loss_weight", 1.0),
            white_bg_weight=model_cfg.get("white_bg_loss_weight", 0.0),
            black_bg_weight=model_cfg.get("black_bg_loss_weight", 0.0),
            device=accelerator.device,
        )
        ref_vae = ref_model.vae
        ref_vae.eval()
        ref_vae_dtype = next(ref_vae.parameters()).dtype
        for param in ref_vae.parameters():
            param.requires_grad_(False)

    for epoch in range(epochs):
        progress_bar = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        model.train()
        for batch in progress_bar:
            with accelerator.accumulate(model):
                current_step = global_step + 1
                inputs = build_training_batch(
                    batch,
                    accelerator.device,
                    background_sample_prob=background_sample_prob,
                ).to(dtype=weight_dtype)
                target = torch.clamp(inputs, 0.0, 1.0)
                target_vae = target * 2.0 - 1.0
                composed_target = build_detail_augmented_triplet(target_vae)

                unwrapped = accelerator.unwrap_model(model)
                vae = unwrapped.vae

                with accelerator.autocast():
                    posterior_all = vae.encode(composed_target).latent_dist
                    posterior, posterior_black, posterior_white = split_triplet_distribution(posterior_all)
                    z = posterior.sample()
                    pred = vae.decode(z).sample

                    recon_loss = loss_module.reconstruction_loss(pred, target_vae)
                    _ensure_finite(
                        recon_loss.detach(),
                        "recon_loss",
                        epoch=epoch,
                        step=current_step,
                        accelerator=accelerator,
                    )
                    total_loss = recon_loss
                    step_metrics: Dict[str, torch.Tensor] = {"train/recon": recon_loss.detach()}

                    if lpips_scale > 0.0 and loss_module.use_lpips:
                        lpips_loss = loss_module.perceptual_loss(pred, target_vae)
                        _ensure_finite(
                            lpips_loss.detach(),
                            "lpips_loss",
                            epoch=epoch,
                            step=current_step,
                            accelerator=accelerator,
                        )
                        total_loss = total_loss + lpips_scale * lpips_loss
                        step_metrics["train/lpips"] = lpips_loss.detach()

                    if kl_scale is not None and kl_scale > 0.0:
                        kl_loss = loss_module.kl_loss(posterior)
                        _ensure_finite(
                            kl_loss.detach(),
                            "kl_loss",
                            epoch=epoch,
                            step=current_step,
                            accelerator=accelerator,
                        )
                        total_loss = total_loss + kl_scale * kl_loss
                        step_metrics["train/kl"] = kl_loss.detach()

                    if ref_vae is not None and ref_kl_scale and ref_kl_scale > 0.0:
                        with torch.no_grad():
                            ref_input = composed_target
                            if ref_vae_dtype is not None and ref_vae_dtype != ref_input.dtype:
                                ref_input = ref_input.to(dtype=ref_vae_dtype)
                            ref_posterior_all = ref_vae.encode(ref_input).latent_dist
                        _, ref_black, ref_white = split_triplet_distribution(ref_posterior_all)
                        ref_kl_loss = 0.5 * (
                            loss_module.kl_loss(posterior_black, ref_black)
                            + loss_module.kl_loss(posterior_white, ref_white)
                        )
                        _ensure_finite(
                            ref_kl_loss.detach(),
                            "ref_kl_loss",
                            epoch=epoch,
                            step=current_step,
                            accelerator=accelerator,
                        )
                        total_loss = total_loss + ref_kl_scale * ref_kl_loss
                        step_metrics["train/ref_kl"] = ref_kl_loss.detach()

                    _ensure_finite(
                        total_loss.detach(),
                        "total_loss",
                        epoch=epoch,
                        step=current_step,
                        accelerator=accelerator,
                    )
                    step_metrics["train/loss"] = total_loss.detach()

                accelerator.backward(total_loss)
                if max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                metrics_to_log = {
                    key: accelerator.gather_for_metrics(value).mean().item() for key, value in step_metrics.items()
                }
                accelerator.log(metrics_to_log, step=global_step)

                if global_step % log_every == 0:
                    loss_value = metrics_to_log.get("train/loss", 0.0)
                    progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
                    accelerator.print(f"[RGBA-VAE] epoch {epoch} step {global_step} loss {loss_value:.4f}")

                if (
                    train_cfg.get("run_validation", True)
                    and val_loader is not None
                    and val_every_steps > 0
                    and global_step % val_every_steps == 0
                ):
                    evaluate_rgba_vae(
                        accelerator,
                        model,
                        val_loader,
                        epoch=epoch,
                        eval_cfg=train_cfg,
                        global_step=global_step,
                    )
                    performed_validation = True

                if ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                    save_checkpoints(accelerator, model, cfg, step=global_step)

        progress_bar.close()

    if (
        train_cfg.get("run_validation", True)
        and val_loader is not None
        and not performed_validation
    ):
        evaluate_rgba_vae(
            accelerator,
            model,
            val_loader,
            epoch=epochs - 1,
            eval_cfg=train_cfg,
            global_step=global_step,
        )

    save_checkpoints(accelerator, model, cfg, step=global_step)


def build_training_batch(
    batch: TensorDict,
    device: torch.device,
    *,
    background_sample_prob: float = 0.0,
) -> torch.Tensor:
    """
    Combine component/composite tensors and optionally inject background frames.
    """
    tensors: List[torch.Tensor] = []
    if "component" in batch and "composite" in batch:
        tensors.extend([batch["component"], batch["composite"]])
    elif "composite" in batch:
        tensors.append(batch["composite"])
    else:
        raise ValueError("Batch must contain 'composite' tensor for training.")

    inputs = torch.cat([tensor.to(device) for tensor in tensors], dim=0)

    if background_sample_prob > 0.0 and "background" in batch:
        background = batch["background"].to(device)
        if background.dim() == 3:
            background = background.unsqueeze(0)
        if background.shape[1] != 4:
            raise ValueError("Background tensor is expected to have 4 channels (RGBA).")
        mask = torch.rand(background.shape[0], device=device) < background_sample_prob
        if mask.any():
            inputs = torch.cat([inputs, background[mask]], dim=0)
    return inputs


def build_detail_augmented_triplet(target: torch.Tensor) -> torch.Tensor:
    """
    Create (original, black-bg, white-bg) triplets following AlphaVAE's detail augmentation.
    Expects `target` in [-1, 1] range with channels (RGBA).
    """
    if target.shape[1] < 4:
        raise ValueError("detail augmentation expects RGBA tensors.")

    fg_alpha = (1.0 + target[:, 3:4]) * 0.5
    bg_alpha = (1.0 - target[:, 3:4]) * 0.5

    black = target * fg_alpha - bg_alpha
    white = target * fg_alpha + bg_alpha

    black = black.clone()
    white = white.clone()
    black[:, 3:] = 1.0
    white[:, 3:] = 1.0

    return torch.cat([target, black, white], dim=0)


def split_triplet_distribution(
    posterior: DiagonalGaussianDistribution,
) -> Tuple[DiagonalGaussianDistribution, DiagonalGaussianDistribution, DiagonalGaussianDistribution]:
    """
    Split a concatenated posterior into (original, black, white) chunks by batch dimension.
    """
    params = posterior.parameters
    chunks = torch.chunk(params, 3, dim=0)
    if len(chunks) != 3:
        raise ValueError("Posterior batch dimension must be divisible by 3 for triplet splits.")
    return tuple(DiagonalGaussianDistribution(chunk) for chunk in chunks)  # type: ignore[return-value]


def compute_model_loss(model: torch.nn.Module, recon: torch.Tensor, inputs: torch.Tensor, posterior) -> torch.Tensor:
    target = model
    if not hasattr(target, "loss"):
        target = getattr(model, "module", None)
    if target is None or not hasattr(target, "loss"):
        raise AttributeError("Underlying model does not expose a `loss` method.")
    return target.loss(recon, inputs, posterior)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-8)
    return -10.0 * torch.log10(mse)


def evaluate_rgba_vae(
    accelerator: Accelerator,
    model: RgbaVAE,
    dataloader: DataLoader,
    epoch: int,
    eval_cfg: Dict[str, Any],
    *,
    global_step: Optional[int] = None,
) -> None:
    model.eval()
    background_specs = eval_cfg.get("val_background_colors", ["white", "black"])
    resolved_backgrounds = [resolve_background_spec(spec) for spec in background_specs]
    psnr_records: Dict[str, List[torch.Tensor]] = {spec: [] for spec in background_specs}
    alpha_l1_values: List[torch.Tensor] = []
    viz_samples: List[Dict[str, torch.Tensor]] = []
    max_batches = eval_cfg.get("val_max_batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = build_training_batch(batch, accelerator.device, background_sample_prob=0.0)
            model_dtype = next(model.parameters()).dtype
            inputs = inputs.to(dtype=model_dtype)
            recon, _ = model(inputs)

            for spec, bg in zip(background_specs, resolved_backgrounds):
                gt_bg = composite_over_background(inputs, bg)
                recon_bg = composite_over_background(recon, bg)
                psnr = compute_psnr(recon_bg, gt_bg)
                gathered = accelerator.gather(psnr)
                psnr_records[spec].append(gathered.cpu())

            if recon.shape[1] > 3:
                alpha_gt = inputs[:, 3:]
                alpha_pred = recon[:, 3:]
                alpha_mae = torch.mean(torch.abs(alpha_pred - alpha_gt), dim=(1, 2, 3))
                alpha_l1_values.append(accelerator.gather(alpha_mae).cpu())

            if accelerator.is_main_process and len(viz_samples) < eval_cfg.get("val_visual_rows", 8):
                viz_samples.append(
                    {
                        "gt": inputs[0].detach().cpu().float(),
                        "recon": recon[0].detach().cpu().float(),
                    }
                )

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    for spec, values in psnr_records.items():
        if not values:
            continue
        stacked = torch.cat(values)
        accelerator.print(f"[RGBA-VAE][val] epoch {epoch} PSNR ({spec} background): {stacked.mean().item():.2f} dB")

    if alpha_l1_values:
        alpha_mean = torch.cat(alpha_l1_values).mean().item()
        accelerator.print(f"[RGBA-VAE][val] epoch {epoch} alpha MAE: {alpha_mean:.4f}")

    if accelerator.is_main_process and viz_samples:
        save_validation_grid(
            viz_samples,
            epoch=epoch,
            step=global_step,
            output_dir=eval_cfg.get("val_output_dir", "outputs"),
        )

    model.train()


def resolve_background_spec(spec: Union[str, Sequence[float], float]) -> Union[float, Sequence[float]]:
    if isinstance(spec, str):
        lowered = spec.lower()
        if lowered == "white":
            return 1.0
        if lowered == "black":
            return 0.0
        raise ValueError(f"Unknown background spec '{spec}'.")
    return spec


def save_validation_grid(
    samples: List[Dict[str, torch.Tensor]],
    *,
    epoch: int,
    step: Optional[int],
    output_dir: str,
) -> None:
    rows = len(samples)
    columns = ("GT (white)", "Recon (white)", "GT (black)", "Recon (black)", "Alpha diff")
    fig, axes = plt.subplots(rows, len(columns), figsize=(4 * len(columns), 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        gt = sample["gt"]
        recon = sample["recon"]
        alpha_diff = torch.abs(gt[3:] - recon[3:])
        visuals = (
            composite_over_white(gt.unsqueeze(0))[0],
            composite_over_white(recon.unsqueeze(0))[0],
            composite_over_black(gt.unsqueeze(0))[0],
            composite_over_black(recon.unsqueeze(0))[0],
            alpha_diff.repeat(3, 1, 1),
        )
        for col, (title, tensor) in enumerate(zip(columns, visuals)):
            img = tensor.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0.0, 1.0)
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if step is not None:
        filename = f"val_recon_epoch_{epoch}_step_{step}.png"
    else:
        filename = f"val_recon_epoch_{epoch}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[RGBA-VAE][val] saved visualization to {out_path}")


def save_checkpoints(
    accelerator: Accelerator,
    model: RgbaVAE,
    cfg: Dict[str, Any],
    *,
    step: Optional[int] = None,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    ckpt_dir = Path(cfg.get("training", {}).get("ckpt_dir", "checkpoints"))
    target_dir = ckpt_dir if step is None else ckpt_dir / f"step_{step:07d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)
    torch.save(state_dict, target_dir / "rgba_vae.pt")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.vae.save_pretrained(target_dir / "rgba_vae_hf")
    step_msg = f" (step {step})" if step is not None else ""
    accelerator.print(f"Saved RGBA-VAE checkpoints to {target_dir}{step_msg}")

