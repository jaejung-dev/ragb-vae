from __future__ import annotations

import argparse
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from diffusers import FluxPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_generation.text_alpha_bucket_dataset import BucketBatchSampler, TextAlphaBucketDataset
from src.models import (
    FluxTextAlphaModel,
    encode_empty_prompt,
    load_rgba_vae_from_path,
    load_scheduler,
    load_transformer,
)


def _resolve_env_token(value: str | None) -> str | None:
    """
    Expand ${env:VAR_NAME} placeholders from YAML into real env values.
    Returns None if the placeholder is missing or the env var is unset.
    """
    if value is None:
        return None
    if value.startswith("${env:") and value.endswith("}"):
        env_name = value[len("${env:") : -1]
        return os.environ.get(env_name)
    return value


def _dtype_to_str(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "torch.bfloat16"
    if dtype is torch.float16:
        return "torch.float16"
    if dtype is torch.float32:
        return "torch.float32"
    return str(dtype)


def _write_lora_metadata(save_dir: Path, *, step: int | str, args: argparse.Namespace, dtype: torch.dtype) -> None:
    """
    Persist lightweight metadata next to LoRA weights so inference can auto-read rank/alpha/etc.
    """
    meta = {
        "model_id": args.pretrained_model_name_or_path,
        "revision": None,
        "adapter_name": "default",
        "rank": int(args.rank),
        "alpha": int(args.lora_alpha),
        "dropout": 0.0,
        "dtype": _dtype_to_str(dtype),
        "step": step,
        "weights": "pytorch_lora_weights.safetensors",
    }
    metadata_path = save_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args(args: list[str] | None = None, *, allow_missing: bool = False) -> argparse.Namespace:
    """
    `allow_missing=True` lets config-driven runs fetch defaults without argparse
    exiting on required fields. CLI usage keeps required checks.
    """
    parser = argparse.ArgumentParser(description="FLUX-Kontext LoRA for text_alpha latent prediction.")
    required = not allow_missing
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=required, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--rgba_vae_path", type=str, required=required, default=None)
    parser.add_argument("--vae_subfolder", type=str, default="ae")
    parser.add_argument("--data_root", type=str, required=required, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/flux_kontext_textalpha_lora")
    parser.add_argument("--output_dir", type=str, default="outputs/flux_kontext_textalpha_lora")
    parser.add_argument("--val_output_dir", type=str, default="outputs/flux_kontext_textalpha_lora/val_samples")
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_max_samples", type=int, default=100)
    parser.add_argument("--val_num_inference_steps", type=int, default=20)
    parser.add_argument("--run_validation_on_start", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--interleave_buckets", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args(args=args)


def train(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("medium")
    accelerator_kwargs = dict(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    ds_plugin = None
    ds_config_path: Path | None = None
    if args.deepspeed_config:
        ds_config_path = Path(args.deepspeed_config)
        if not ds_config_path.is_absolute():
            ds_config_path = Path.cwd() / ds_config_path
        ds_plugin = DeepSpeedPlugin(hf_ds_config=str(ds_config_path))
        accelerator_kwargs["deepspeed_plugin"] = ds_plugin

    accelerator = Accelerator(**accelerator_kwargs)
    device = accelerator.device

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed + accelerator.process_index)

    token = _resolve_env_token(args.hf_token) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    model = FluxTextAlphaModel(
        args.pretrained_model_name_or_path,
        vae_path=args.rgba_vae_path,
        vae_subfolder=args.vae_subfolder,
        token=token,
        device=device,
        weight_dtype=weight_dtype,
    )
    model.add_lora(args.rank, args.lora_alpha)

    # Datasets
    train_ds = TextAlphaBucketDataset(Path(args.data_root), split=args.train_split)
    val_ds: Optional[TextAlphaBucketDataset] = None
    if args.val_split:
        val_ds = TextAlphaBucketDataset(Path(args.data_root), split=args.val_split)

    train_sampler = BucketBatchSampler(
        train_ds.bucket_to_indices,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        interleave=args.interleave_buckets,
    )
    loader_kwargs: Dict[str, Any] = {"pin_memory": True}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        **loader_kwargs,
    )
    val_dl: Optional[DataLoader] = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=min(4, args.num_workers),
            pin_memory=True,
            **({"persistent_workers": True, "prefetch_factor": 2} if min(4, args.num_workers) > 0 else {}),
        )

    # Optimizer & LR
    params_to_optimize = model.lora_parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    transformer_wrapped, optimizer, train_dl = accelerator.prepare(model.transformer, optimizer, train_dl)
    model.transformer = transformer_wrapped
    model.to_device(device)

    # Report effective batch sizes to avoid confusion between configs.
    world_size = accelerator.num_processes
    per_device_batch = args.batch_size
    grad_accum = max(1, args.grad_accum_steps)
    effective_global_batch = per_device_batch * grad_accum * max(1, world_size)
    ds_micro = None
    if ds_config_path is not None and ds_config_path.exists():
        try:
            with ds_config_path.open("r", encoding="utf-8") as f:
                ds_cfg = json.load(f)
            ds_micro = ds_cfg.get("train_micro_batch_size_per_gpu")
        except Exception:
            ds_micro = "unreadable"
    accelerator.print(
        "[Batch] per_device=%s grad_accum=%s world_size=%s effective_per_step=%s ds_micro=%s"
        % (per_device_batch, grad_accum, world_size, effective_global_batch, ds_micro)
    )

    total_steps = 0
    accelerator.print(f"[Train] {len(train_ds)} samples across {len(train_ds.bucket_to_indices)} buckets.")
    if val_ds is not None:
        accelerator.print(f"[Val]   {len(val_ds)} samples.")
    else:
        accelerator.print("[Val]   (disabled: no val_split provided)")

    pbar = None
    if accelerator.is_local_main_process:
        pbar = tqdm(total=args.max_train_steps, desc="train", dynamic_ncols=True)
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    def _rgba_to_uint8(tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.clamp(0, 1)
            .permute(1, 2, 0)
            .to(torch.float32, copy=True)
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

    def _save_pair(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor, path: Path) -> None:
        gt_img = Image.fromarray(_rgba_to_uint8(gt_tensor), mode="RGBA")
        pred_img = Image.fromarray(_rgba_to_uint8(pred_tensor), mode="RGBA")
        w, h = gt_img.size
        canvas = Image.new("RGBA", (w * 2, h))
        canvas.paste(gt_img, (0, 0))
        canvas.paste(pred_img, (w, 0))
        canvas.save(path)

    def run_validation(step_label: str) -> None:
        if val_dl is None or not accelerator.is_main_process:
            return
        model.transformer.eval()
        out_dir = Path(args.val_output_dir) / f"step-{step_label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        max_samples = args.val_max_samples
        val_total_batches = min(len(val_dl), (max_samples + args.val_batch_size - 1) // args.val_batch_size)
        vbar = tqdm(total=val_total_batches, desc=f"val-{step_label}", dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_dl:
                if saved >= max_samples:
                    break
                gt = batch["gt"].to(device, dtype=model.vae.dtype)
                decoded = model.sample(gt, num_inference_steps=args.val_num_inference_steps)

                names = batch.get("sample_name", ["val"])
                if isinstance(names, str):
                    names = [names]

                for i in range(decoded.shape[0]):
                    if saved >= max_samples:
                        break
                    name = names[i] if i < len(names) else f"val_{saved}"
                    _save_pair(gt[i], decoded[i], out_dir / f"{name}_pair.png")
                    saved += 1
                vbar.update(1)
        model.transformer.train()
        vbar.close()

    # Initial sanity-check validation
    if args.run_validation_on_start:
        run_validation("start")

    while total_steps < args.max_train_steps:
        for batch in train_dl:
            with accelerator.accumulate(model.transformer):
                gt = batch["gt"]
                ta = batch["text_alpha"]

                loss, _ = model.compute_loss(gt, ta)

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_steps += 1
                if pbar is not None:
                    current_lr = optimizer.param_groups[0]["lr"]
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
                if accelerator.is_main_process and total_steps % args.log_every == 0:
                    accelerator.print(f"[step {total_steps}] loss={loss.item():.4f} lr={current_lr:.6f}")

                if accelerator.is_main_process and args.save_every and total_steps % args.save_every == 0:
                    save_dir = Path(args.ckpt_dir) / f"checkpoint-{total_steps}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    lora_state = model.lora_state_dict()
                    FluxPipeline.save_lora_weights(
                        save_dir,
                        transformer_lora_layers=lora_state,
                    )
                    _write_lora_metadata(save_dir, step=total_steps, args=args, dtype=weight_dtype)

                # Run validation on schedule
                if args.val_every and total_steps % args.val_every == 0 and total_steps > 0:
                    run_validation(str(total_steps))

                if total_steps >= args.max_train_steps:
                    break
        # allow re-iteration over the dataloader until max steps are reached

    if pbar is not None:
        pbar.close()

    if accelerator.is_main_process:
        final_dir = Path(args.ckpt_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model.transformer)
        lora_state = get_peft_model_state_dict(unwrapped)
        FluxPipeline.save_lora_weights(
            final_dir,
            transformer_lora_layers=lora_state,
        )
        _write_lora_metadata(final_dir, step=args.max_train_steps, args=args, dtype=weight_dtype)
    accelerator.print("Done.")


def build_args_from_cfg(cfg: Dict[str, Any]) -> argparse.Namespace:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    defaults = parse_args(args=[], allow_missing=True)
    # Build a namespace manually to avoid argparse re-parsing
    args = argparse.Namespace(**vars(defaults))

    # Model
    if model_cfg.get("pretrained_model_name_or_path"):
        args.pretrained_model_name_or_path = model_cfg["pretrained_model_name_or_path"]
    if model_cfg.get("hf_token"):
        args.hf_token = _resolve_env_token(model_cfg.get("hf_token"))
    if model_cfg.get("rgba_vae_path"):
        args.rgba_vae_path = model_cfg["rgba_vae_path"]
    if model_cfg.get("vae_subfolder") is not None:
        args.vae_subfolder = model_cfg["vae_subfolder"]

    # Data
    if data_cfg.get("root"):
        args.data_root = data_cfg["root"]
    if data_cfg.get("train_split") is not None:
        args.train_split = data_cfg["train_split"]
    if data_cfg.get("val_split") is not None:
        args.val_split = data_cfg["val_split"]
    if data_cfg.get("batch_size") is not None:
        args.batch_size = int(data_cfg["batch_size"])
    if data_cfg.get("val_batch_size") is not None:
        args.val_batch_size = int(data_cfg["val_batch_size"])
    if data_cfg.get("num_workers") is not None:
        args.num_workers = int(data_cfg["num_workers"])
    if data_cfg.get("drop_last") is not None:
        args.drop_last = bool(data_cfg["drop_last"])
    if data_cfg.get("interleave_buckets") is not None:
        args.interleave_buckets = bool(data_cfg["interleave_buckets"])

    # Training
    if train_cfg.get("mixed_precision") is not None:
        args.mixed_precision = train_cfg["mixed_precision"]
    if train_cfg.get("grad_accum_steps") is not None:
        args.grad_accum_steps = int(train_cfg["grad_accum_steps"])
    if train_cfg.get("learning_rate") is not None:
        args.learning_rate = float(train_cfg["learning_rate"])
    if train_cfg.get("weight_decay") is not None:
        args.weight_decay = float(train_cfg["weight_decay"])
    if train_cfg.get("adam_beta1") is not None:
        args.adam_beta1 = float(train_cfg["adam_beta1"])
    if train_cfg.get("adam_beta2") is not None:
        args.adam_beta2 = float(train_cfg["adam_beta2"])
    if train_cfg.get("adam_eps") is not None:
        args.adam_eps = float(train_cfg["adam_eps"])
    if train_cfg.get("max_train_steps") is not None:
        args.max_train_steps = int(train_cfg["max_train_steps"])
    if train_cfg.get("log_every") is not None:
        args.log_every = int(train_cfg["log_every"])
    if train_cfg.get("save_every") is not None:
        args.save_every = int(train_cfg["save_every"])
    if train_cfg.get("ckpt_every_steps") is not None:
        args.save_every = int(train_cfg["ckpt_every_steps"])
    if train_cfg.get("ckpt_dir") is not None:
        args.ckpt_dir = str(train_cfg["ckpt_dir"])
    if train_cfg.get("output_dir") is not None:
        args.output_dir = str(train_cfg["output_dir"])
    if train_cfg.get("val_output_dir") is not None:
        args.val_output_dir = str(train_cfg["val_output_dir"])
    if train_cfg.get("val_every") is not None:
        args.val_every = int(train_cfg["val_every"])
    if train_cfg.get("val_every_steps") is not None:
        args.val_every = int(train_cfg["val_every_steps"])
    if train_cfg.get("val_max_samples") is not None:
        args.val_max_samples = int(train_cfg["val_max_samples"])
    if train_cfg.get("val_max_batches") is not None:
        args.val_max_samples = int(train_cfg["val_max_batches"]) * args.val_batch_size
    if train_cfg.get("val_num_inference_steps") is not None:
        args.val_num_inference_steps = int(train_cfg["val_num_inference_steps"])
    if train_cfg.get("run_validation_on_start") is not None:
        args.run_validation_on_start = bool(train_cfg["run_validation_on_start"])
    if train_cfg.get("rank") is not None:
        args.rank = int(train_cfg["rank"])
    if train_cfg.get("lora_alpha") is not None:
        args.lora_alpha = int(train_cfg["lora_alpha"])
    if train_cfg.get("max_grad_norm") is not None:
        args.max_grad_norm = float(train_cfg["max_grad_norm"])
    if train_cfg.get("deepspeed_config") is not None:
        args.deepspeed_config = train_cfg["deepspeed_config"]
    if train_cfg.get("seed") is not None:
        args.seed = int(train_cfg["seed"])

    missing = []
    if not args.pretrained_model_name_or_path:
        missing.append("model.pretrained_model_name_or_path")
    if not args.rgba_vae_path:
        missing.append("model.rgba_vae_path")
    if not args.data_root:
        missing.append("data.root")
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    return args


def train_from_config(cfg: Dict[str, Any]) -> None:
    args = build_args_from_cfg(cfg)
    train(args)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

