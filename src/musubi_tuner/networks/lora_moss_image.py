# LoRA module for MOSS-Image

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# q / k / v / o, ffn_up_gate and ffn_down of the 42 blocks. q / k / v are separate modules (one DoRA magnitude each):
# ComfyUI maps ``lora_unet_layers_0_proj_q`` onto the row slice of its fused ``proj.qkv`` weight. The embedders, the
# shared modulation, the text refiner and the output head sit outside these blocks.
MOSS_IMAGE_TARGET_REPLACE_MODULES = ["MossImageBlock"]
# network_args include_text_embedder=True adds the text refiner's two blocks (fused qkv, proj, ffn_gate / up / down), as
# the colour / depth control adapters ComfyUI loads do (moss_image/moss_control.py)
MOSS_TEXT_EMBEDDER_TARGET_MODULES = ["MossTextBlock"]


def _targets(include_text_embedder) -> list:
    on = str(include_text_embedder).strip().lower() in ("1", "true", "yes", "on")
    return MOSS_IMAGE_TARGET_REPLACE_MODULES + (MOSS_TEXT_EMBEDDER_TARGET_MODULES if on else [])


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'norm' in the name of the module
    exclude_patterns.append(r".*(norm).*")

    kwargs["exclude_patterns"] = exclude_patterns
    targets = _targets(kwargs.pop("include_text_embedder", False))

    return lora.create_network(
        targets,
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
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    kwargs.pop("include_text_embedder", None)
    targets = _targets(any(key.startswith("lora_unet_text_embedder_") for key in weights_sd))
    return lora.create_network_from_weights(targets, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs)
