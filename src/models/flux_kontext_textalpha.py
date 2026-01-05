from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from src.models.rgba_vae import _maybe_restore_rgba_convs, adapt_vae_to_rgba

# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------
def load_transformer(
    model_id: str,
    *,
    token: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> FluxTransformer2DModel:
    return FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        revision=revision,
        variant=variant,
        torch_dtype=dtype,
        token=token,
    )


def load_scheduler(
    model_id: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        revision=revision,
        variant=variant,
        token=token,
    )


def load_rgba_vae_from_path(
    vae_path: str,
    *,
    subfolder: str = "ae",
    dtype: torch.dtype = torch.float32,
) -> AutoencoderKL:
    """
    Load an RGBA-ready VAE checkpoint.

    - If the saved config is still RGB (in/out_channels=3), transparently widen to 4
      channels and restore RGBA weights when present.
    - Uses ignore_mismatched_sizes to avoid shape errors when weights are already RGBA.
    """
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=subfolder,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    in_c = getattr(vae.config, "in_channels", None)
    out_c = getattr(vae.config, "out_channels", None)
    if in_c == 3 or out_c == 3:
        adapt_vae_to_rgba(vae, alpha_bias_init=0.0)
        _maybe_restore_rgba_convs(vae, vae_path, subfolder)
        vae.config.in_channels = 4
        vae.config.out_channels = 4
    return vae


def encode_empty_prompt(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (prompt_embeds, pooled_prompt_embeds, text_ids) for the empty prompt."""

    tokenizer_one = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision, variant=variant, token=token
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        model_id, subfolder="tokenizer_2", revision=revision, variant=variant, token=token
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, variant=variant, token=token
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", revision=revision, variant=variant, token=token
    )

    for m in (text_encoder_one, text_encoder_two):
        m.to(device, dtype=dtype)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    empty_prompt = [""]
    text_inputs_one = tokenizer_one(
        empty_prompt,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    text_inputs_two = tokenizer_two(
        empty_prompt,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder_one(**text_inputs_one).last_hidden_state
        prompt_embeds_2 = text_encoder_two(**text_inputs_two).last_hidden_state
        pooled = text_encoder_one.text_model.final_layer_norm(prompt_embeds)[:, 0]

    # FLUX-Kontext uses two encoders with different hidden sizes (CLIP: 768, T5: 4096).
    # If the hidden dims mismatch, fall back to using the T5 stream only to avoid cat errors.
    if prompt_embeds.shape[-1] == prompt_embeds_2.shape[-1]:
        prompt = torch.cat([prompt_embeds, prompt_embeds_2], dim=1)
    else:
        prompt = prompt_embeds_2

    # FLUX expects txt_ids as positional ids of shape (seq_len, 3) without a batch dim.
    text_ids = torch.zeros(prompt.shape[1], 3, device=device, dtype=prompt.dtype)
    return prompt, pooled, text_ids


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------
def add_lora_to_transformer(
    transformer: FluxTransformer2DModel,
    *,
    rank: int,
    lora_alpha: int,
    adapter_name: str = "default",
) -> None:
    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_cfg, adapter_name=adapter_name)
    transformer.enable_adapters()


def load_lora_weights_into_transformer(
    transformer: FluxTransformer2DModel,
    lora_dir: str | Path,
    *,
    rank: int,
    lora_alpha: int,
    adapter_name: str = "default",
) -> None:
    """Attach adapters (if missing) then load LoRA weights saved via FluxPipeline.save_lora_weights."""
    add_lora_to_transformer(transformer, rank=rank, lora_alpha=lora_alpha, adapter_name=adapter_name)
    lora_state = FluxPipeline.lora_state_dict(lora_dir)
    transformer.load_state_dict(lora_state, strict=False)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------
class FluxTextAlphaModel:
    """
    Wrapper that holds transformer + VAE + scheduler + prompt embeddings
    for text_alpha prediction. Provides loss computation and sampling helpers.
    """

    def __init__(
        self,
        model_id: str,
        *,
        vae_path: str,
        vae_subfolder: str = "ae",
        token: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        weight_dtype: torch.dtype = torch.float32,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        self.device = device
        self.weight_dtype = weight_dtype
        # Default embedded guidance scale used by FLUX guidance-distilled checkpoints.
        # Matches the diffusers pipeline default (3.5) unless overridden manually.
        self.guidance_scale = 3.5

        self.transformer = load_transformer(model_id, token=token, dtype=weight_dtype)
        self.vae = load_rgba_vae_from_path(vae_path, subfolder=vae_subfolder, dtype=weight_dtype)
        self.scheduler = load_scheduler(model_id, token=token)
        self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = encode_empty_prompt(
            model_id, device=device, dtype=weight_dtype, token=token
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))

        if use_gradient_checkpointing and hasattr(self.transformer, "enable_gradient_checkpointing"):
            # Reduce activation memory; critical for LoRA fine-tuning at higher resolutions.
            self.transformer.enable_gradient_checkpointing()

        self.to_device(device)
        self._set_timesteps(device, num_train_timesteps=self.scheduler.config.num_train_timesteps)

    # ----------------------------
    # Device / adapters
    # ----------------------------
    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.transformer.to(device)
        self.vae.to(device)
        self.prompt_embeds = self.prompt_embeds.to(device)
        self.pooled_prompt_embeds = self.pooled_prompt_embeds.to(device)
        self.text_ids = self.text_ids.to(device)
        # timesteps need to match device
        self._set_timesteps(device)

    def _calc_mu(self, seq_len: Optional[int] = None) -> Optional[float]:
        cfg = self.scheduler.config
        if not getattr(cfg, "use_dynamic_shifting", False):
            return None

        base_seq = getattr(cfg, "base_image_seq_len", 256) or 256
        max_seq = getattr(cfg, "max_image_seq_len", 4096) or 4096
        base_shift = getattr(cfg, "base_shift", 0.5) or 0.5
        max_shift = getattr(cfg, "max_shift", 1.15) or 1.15

        # Estimate sequence length if not provided:
        # use VAE sample_size reduced by scale factor; clamp to config bounds.
        if seq_len is None:
            sample_size = getattr(self.vae.config, "sample_size", 256) or 256
            h = max(int(sample_size // self.vae_scale_factor), 1)
            seq_len = h * h
        seq_len = int(seq_len)
        seq_len = max(min(seq_len, max_seq), base_seq)

        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        return float(seq_len * m + b)

    def _set_timesteps(self, device: torch.device, *, num_train_timesteps: Optional[int] = None) -> None:
        mu = self._calc_mu()
        num_steps = num_train_timesteps or self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_steps, device=device, mu=mu)

    def _unwrap_transformer(self):
        """
        Return the base transformer module (handles Accelerator/DeepSpeed wrappers).
        """
        if hasattr(self.transformer, "module"):
            return self.transformer.module
        return self.transformer

    def _transformer_config(self):
        """
        Return the underlying transformer config even if wrapped by Accelerator/DeepSpeed.
        """
        base = self._unwrap_transformer()
        if hasattr(base, "config"):
            return base.config
        return None

    def _prepare_guidance(self, batch_size: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return a guidance tensor when the transformer expects guidance embeddings."""
        cfg = self._transformer_config()
        base = self._unwrap_transformer()
        time_embed = getattr(base, "time_text_embed", None)
        guidance_needed = getattr(cfg, "guidance_embeds", False) or isinstance(
            time_embed, CombinedTimestepGuidanceTextProjEmbeddings
        )
        if not guidance_needed:
            return None
        scale = float(getattr(self, "guidance_scale", 3.5))
        return torch.full((batch_size,), scale, device=self.device, dtype=dtype)

    def add_lora(self, rank: int, lora_alpha: int) -> None:
        add_lora_to_transformer(self.transformer, rank=rank, lora_alpha=lora_alpha)

    def load_lora(self, lora_dir: str | Path, rank: int, lora_alpha: int) -> None:
        load_lora_weights_into_transformer(self.transformer, lora_dir, rank=rank, lora_alpha=lora_alpha)

    def lora_parameters(self):
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def lora_state_dict(self):
        return get_peft_model_state_dict(self.transformer)

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode((x * 2.0 - 1.0)).latent_dist.sample()
        return (latents - self.shift_factor) * self.scaling_factor

    def _pack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return FluxPipeline._pack_latents(
            latent,
            batch_size=latent.shape[0],
            num_channels_latents=latent.shape[1],
            height=latent.shape[2],
            width=latent.shape[3],
        )

    def _unpack_target(self, model_pred_tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return FluxPipeline._unpack_latents(
            model_pred_tokens,
            height=height,
            width=width,
            vae_scale_factor=self.vae_scale_factor,
        )

    # ----------------------------
    # Training loss
    # ----------------------------
    def compute_loss(self, gt: torch.Tensor, text_alpha: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        gt = gt.to(self.device, dtype=self.vae.dtype)
        text_alpha = text_alpha.to(self.device, dtype=self.vae.dtype)

        with torch.no_grad():
            cond_latent = self._encode_latents(gt)
            target_latent = self._encode_latents(text_alpha)

        noise = torch.randn_like(target_latent)
        bsz = target_latent.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.0,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        # Guard against rare out-of-range indices -> clamp by actual buffer lengths.
        max_idx = min(self.scheduler.timesteps.shape[0] - 1, self.scheduler.sigmas.shape[0] - 1)
        indices = indices.clamp(max=max_idx)

        timesteps = self.scheduler.timesteps[indices].to(self.device)
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=target_latent.dtype)[indices]
        while len(sigmas.shape) < target_latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_target = (1.0 - sigmas) * target_latent + sigmas * noise
        packed_cond = self._pack_latent(cond_latent)
        packed_tgt = self._pack_latent(noisy_target)
        packed = torch.cat([packed_cond, packed_tgt], dim=1)
        latent_h, latent_w = target_latent.shape[2], target_latent.shape[3]
        latent_image_ids_single = FluxPipeline._prepare_latent_image_ids(
            packed.shape[0], latent_h // 2, latent_w // 2, self.device, packed.dtype
        )
        latent_image_ids = torch.cat([latent_image_ids_single, latent_image_ids_single], dim=0)

        guidance = self._prepare_guidance(bsz, target_latent.dtype)

        model_pred = self.transformer(
            hidden_states=packed,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        seq_len_cond = packed_cond.shape[1]
        model_pred_tgt_tokens = model_pred[:, seq_len_cond:, :]
        model_pred_target = self._unpack_target(model_pred_tgt_tokens, gt.shape[2], gt.shape[3])
        loss_target = noise - target_latent

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="logit_normal",
            sigmas=sigmas,
        )
        loss = torch.mean(
            (weighting.float() * (model_pred_target.float() - loss_target.float()) ** 2).reshape(
                loss_target.shape[0], -1
            ),
            1,
        ).mean()

        stats = {
            "timesteps_mean": float(timesteps.float().mean().item()),
            "sigmas_mean": float(sigmas.float().mean().item()),
        }
        return loss, stats

    # ----------------------------
    # Inference / sampling
    # ----------------------------
    @torch.no_grad()
    def sample(
        self,
        gt: torch.Tensor,
        *,
        num_inference_steps: int = 20,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        gt = gt.to(self.device, dtype=self.vae.dtype)
        cond_latent = self._encode_latents(gt)

        # Prepare scheduler for inference steps
        self._set_timesteps(self.device, num_train_timesteps=num_inference_steps)

        if generator is None:
            latents = torch.randn_like(cond_latent)
        else:
            latents = torch.randn(
                cond_latent.shape,
                device=self.device,
                dtype=cond_latent.dtype,
                generator=generator,
            )
        cond_c = cond_latent.shape[1]
        bsz = gt.shape[0]
        guidance = self._prepare_guidance(bsz, latents.dtype)

        for timestep in self.scheduler.timesteps:
            sigmas = self.scheduler.sigmas.to(device=self.device, dtype=latents.dtype)
            sigma = sigmas[(self.scheduler.timesteps == timestep).nonzero().item()]
            while len(sigma.shape) < latents.ndim:
                sigma = sigma.unsqueeze(-1)

            noisy_target = (1.0 - sigma) * latents + sigma * torch.randn_like(latents)
            packed_cond = self._pack_latent(cond_latent)
            packed_tgt = self._pack_latent(noisy_target)
            packed = torch.cat([packed_cond, packed_tgt], dim=1)
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            latent_image_ids_single = FluxPipeline._prepare_latent_image_ids(
                packed.shape[0], latent_h // 2, latent_w // 2, self.device, packed.dtype
            )
            latent_image_ids = torch.cat([latent_image_ids_single, latent_image_ids_single], dim=0)

            # Expand scalar timestep to per-example 1D tensor as expected by diffusers embeddings.
            timestep_batched = timestep.expand(latents.shape[0]).to(latents.dtype)

            model_pred = self.transformer(
                hidden_states=packed,
                timestep=timestep_batched / 1000,
                guidance=guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            seq_len_cond = packed_cond.shape[1]
            model_pred_tgt_tokens = model_pred[:, seq_len_cond:, :]
            model_pred_target = self._unpack_target(model_pred_tgt_tokens, gt.shape[2], gt.shape[3])
            latents = self.scheduler.step(
                model_pred_target,
                timestep,
                latents,
                return_dict=False,
            )[0]

        decoded = self.vae.decode(latents / self.scaling_factor + self.shift_factor).sample
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        return decoded

