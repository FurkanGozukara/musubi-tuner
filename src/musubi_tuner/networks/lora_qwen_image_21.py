# LoRA module for Qwen-Image 2.1

from typing import Optional

import torch
from torch import nn

from musubi_tuner.networks import lora

QWEN_IMAGE_21_TARGET_REPLACE_MODULES = ["QwenImage21TransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: list[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create LoRA modules for the attention and MLP projections in each block."""
    return lora.create_network(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders: Optional[list[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    """Restore LoRA modules from a saved adapter state dict."""
    return lora.create_network_from_weights(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )


def convert_weight_keys(weights_sd: dict[str, torch.Tensor]) -> Optional[dict[str, torch.Tensor]]:
    """ComfyUI / ai-toolkit / Diffusers keys (``diffusion_model.`` or ``transformer.`` + module + ``.lora_A/B.weight``) to
    this network's names (``lora_unet_<module>.lora_down/up.weight`` + ``alpha``); None for weights already in that form.

    ComfyUI's Qwen-Image 2.1 files fuse each block's MLP gate and up projections (``img_mlp.gate_up`` = [gate_layer; proj]
    on the output axis), which this model keeps as two layers: a fused LoRA becomes two with the same down matrix and
    the halves of its up matrix (exact: B A x split by output rows), as ostris' qwen_image_2_turbo_training_adapter needs.
    A fused input-axis DoRA magnitude belongs to the fused layer's column norms and cannot be split (Guardrail 18)."""
    keys = list(weights_sd)
    if not keys or not any(key.startswith(("diffusion_model.", "transformer.")) for key in keys):
        return None
    from musubi_tuner.convert_lora import convert_from_diffusers

    converted = convert_from_diffusers("lora_unet_", weights_sd)
    result: dict[str, torch.Tensor] = {}
    for key, value in converted.items():
        name, _, suffix = key.partition(".")
        if not name.endswith("_img_mlp_gate_up"):
            result[key] = value
            continue
        stem = name[: -len("gate_up")]
        if suffix == "lora_up.weight":
            if value.shape[0] % 2:
                raise ValueError(f"{name}: fused gate_up LoRA with an odd output size {tuple(value.shape)}")
            gate, up = value.chunk(2, dim=0)
            result[f"{stem}gate_layer.lora_up.weight"] = gate.contiguous()
            result[f"{stem}proj.lora_up.weight"] = up.contiguous()
        elif suffix in ("lora_down.weight", "alpha"):
            result[f"{stem}gate_layer.{suffix}"] = value
            result[f"{stem}proj.{suffix}"] = value.clone()
        else:
            raise ValueError(f"{name}.{suffix}: a fused gate_up {suffix} cannot be split into gate_layer / proj")
    return result
