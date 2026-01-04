"""
Model package placeholder.

Exports for model components.
"""
from .losses import AlphaVaeLoss
from .rgba_vae import RgbaVAE, composite_over_background, composite_over_black, composite_over_white
from .flux_kontext_textalpha import (
    load_transformer,
    load_scheduler,
    load_rgba_vae_from_path,
    encode_empty_prompt,
    add_lora_to_transformer,
    load_lora_weights_into_transformer,
    FluxTextAlphaModel,
)

__all__ = [
    "AlphaVaeLoss",
    "RgbaVAE",
    "composite_over_background",
    "composite_over_white",
    "composite_over_black",
    "load_transformer",
    "load_scheduler",
    "load_rgba_vae_from_path",
    "encode_empty_prompt",
    "add_lora_to_transformer",
    "load_lora_weights_into_transformer",
    "FluxTextAlphaModel",
]

