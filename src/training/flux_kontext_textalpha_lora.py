from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

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
from torch.utils.data import DataLoader

from src.data_generation.text_alpha_bucket_dataset import BucketBatchSampler, TextAlphaBucketDataset
from src.models import (
    encode_empty_prompt,
    load_rgba_vae_from_path,
    load_scheduler,
    load_transformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX-Kontext LoRA for text_alpha latent prediction.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--rgba_vae_path", type=str, required=True)
    parser.add_argument("--vae_subfolder", type=str, default="ae")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/flux_kontext_textalpha_lora")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--interleave_buckets", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    accelerator_kwargs = dict(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    ds_plugin = None
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

    token = args.hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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
    train_sampler = BucketBatchSampler(
        train_ds.bucket_to_indices,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        interleave=args.interleave_buckets,
    )
    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
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

    total_steps = 0
    accelerator.print(f"[Buckets] {len(train_ds.bucket_to_indices)} buckets, {len(train_ds)} samples.")
    progress = range(args.max_train_steps)

    for step in progress:
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
                if accelerator.is_main_process and total_steps % args.log_every == 0:
                    accelerator.print(f"[step {total_steps}] loss={loss.item():.4f}")

                if accelerator.is_main_process and args.save_every and total_steps % args.save_every == 0:
                    save_dir = Path(args.output_dir) / f"checkpoint-{total_steps}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    lora_state = model.lora_state_dict()
                    FluxPipeline.save_lora_weights(
                        save_dir,
                        transformer_lora_layers=lora_state,
                    )

                if total_steps >= args.max_train_steps:
                    break
        if total_steps >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model.transformer)
        lora_state = get_peft_model_state_dict(unwrapped)
        FluxPipeline.save_lora_weights(
            final_dir,
            transformer_lora_layers=lora_state,
        )
    accelerator.print("Done.")


def build_args_from_cfg(cfg: Dict[str, Any]) -> argparse.Namespace:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    defaults = parse_args()
    # Build a namespace manually to avoid argparse re-parsing
    args = argparse.Namespace(**vars(defaults))

    # Model
    if model_cfg.get("pretrained_model_name_or_path"):
        args.pretrained_model_name_or_path = model_cfg["pretrained_model_name_or_path"]
    if model_cfg.get("hf_token"):
        args.hf_token = model_cfg["hf_token"]
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
    if train_cfg.get("output_dir") is not None:
        args.output_dir = str(train_cfg["output_dir"])
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

    return args


def train_from_config(cfg: Dict[str, Any]) -> None:
    args = build_args_from_cfg(cfg)
    train(args)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

