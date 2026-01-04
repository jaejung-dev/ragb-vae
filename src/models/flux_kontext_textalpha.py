from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.models.autoencoders.vae import AutoencoderKL
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


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
    return AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )


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

    prompt = torch.cat([prompt_embeds, prompt_embeds_2], dim=1)
    txt_ids = torch.cat([text_inputs_one.input_ids, text_inputs_two.input_ids], dim=1)
    return prompt, pooled, txt_ids


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
    ) -> None:
        self.device = device
        self.weight_dtype = weight_dtype

        self.transformer = load_transformer(model_id, token=token, dtype=weight_dtype)
        self.vae = load_rgba_vae_from_path(vae_path, subfolder=vae_subfolder, dtype=weight_dtype)
        self.scheduler = load_scheduler(model_id, token=token)
        self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = encode_empty_prompt(
            model_id, device=device, dtype=weight_dtype, token=token
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))

        self.to_device(device)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)

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
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)

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

    def _pack(
        self,
        cond_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([cond_latent, target_latent], dim=1)
        return FluxPipeline._pack_latents(
            combined,
            batch_size=combined.shape[0],
            num_channels_latents=combined.shape[1],
            height=combined.shape[2],
            width=combined.shape[3],
        )

    def _unpack_target(self, model_pred: torch.Tensor, cond_channels: int, height: int, width: int) -> torch.Tensor:
        unpacked = FluxPipeline._unpack_latents(
            model_pred,
            height=height * self.vae_scale_factor,
            width=width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        return unpacked[:, cond_channels:, ...]

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
        indices = (u * self.scheduler.config.num_train_timesteps).long().clamp(
            max=self.scheduler.config.num_train_timesteps - 1
        )
        timesteps = self.scheduler.timesteps[indices].to(self.device)
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=target_latent.dtype)[indices]
        while len(sigmas.shape) < target_latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_target = (1.0 - sigmas) * target_latent + sigmas * noise
        packed = self._pack(cond_latent, noisy_target)
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            packed.shape[0],
            packed.shape[2] // 2,
            packed.shape[3] // 2,
            self.device,
            packed.dtype,
        )

        model_pred = self.transformer(
            hidden_states=packed,
            timestep=timesteps / 1000,
            guidance=None,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        cond_c = cond_latent.shape[1]
        model_pred_target = self._unpack_target(model_pred, cond_c, target_latent.shape[2], target_latent.shape[3])
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
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        latents = torch.randn_like(cond_latent, generator=generator)
        cond_c = cond_latent.shape[1]

        for timestep in self.scheduler.timesteps:
            sigmas = self.scheduler.sigmas.to(device=self.device, dtype=latents.dtype)
            sigma = sigmas[(self.scheduler.timesteps == timestep).nonzero().item()]
            while len(sigma.shape) < latents.ndim:
                sigma = sigma.unsqueeze(-1)

            noisy_target = (1.0 - sigma) * latents + sigma * torch.randn_like(latents)
            packed = self._pack(cond_latent, noisy_target)
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                packed.shape[0],
                packed.shape[2] // 2,
                packed.shape[3] // 2,
                self.device,
                packed.dtype,
            )

            model_pred = self.transformer(
                hidden_states=packed,
                timestep=timestep / 1000,
                guidance=None,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            model_pred_target = self._unpack_target(model_pred, cond_c, latents.shape[2], latents.shape[3])
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

