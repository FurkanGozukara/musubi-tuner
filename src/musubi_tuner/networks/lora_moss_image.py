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
# network_args include_text_proj=True adds text_embedder.proj, the projection of the Qwen3.5 states (4096 -> 3072) into
# the DiT (ComfyUI maps lora_unet_text_embedder_proj onto diffusion_model.text_embedder.proj.weight). kohya sd-scripts'
# train_text_embedder=True means both (the projection and the refiner blocks) and is accepted as that alias.
MOSS_TEXT_PROJECTION_TARGET_MODULES = ["MossTextProjection"]
_TEXT_REFINER_PATTERN = r"text_embedder\.blocks\..*"


def _on(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _targets(include_text_embedder, include_text_proj=False) -> tuple[list, list]:
    """(target classes, extra exclude patterns). The projection's class holds the refiner blocks as well, so with the
    projection alone the blocks are excluded by name, and with both one target covers them (no module twice)."""
    blocks, proj = _on(include_text_embedder), _on(include_text_proj)
    if proj:
        return MOSS_IMAGE_TARGET_REPLACE_MODULES + MOSS_TEXT_PROJECTION_TARGET_MODULES, ([] if blocks else [_TEXT_REFINER_PATTERN])
    return MOSS_IMAGE_TARGET_REPLACE_MODULES + (MOSS_TEXT_EMBEDDER_TARGET_MODULES if blocks else []), []


def _text_options(kwargs: dict) -> tuple[bool, bool]:
    both = _on(kwargs.pop("train_text_embedder", False))
    return both or _on(kwargs.pop("include_text_embedder", False)), both or _on(kwargs.pop("include_text_proj", False))


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

    targets, extra_excludes = _targets(*_text_options(kwargs))
    exclude_patterns.extend(extra_excludes)
    kwargs["exclude_patterns"] = exclude_patterns

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
    _text_options(kwargs)
    blocks = any(key.startswith("lora_unet_text_embedder_blocks_") for key in weights_sd)
    proj = any(key.startswith("lora_unet_text_embedder_proj.") for key in weights_sd)
    # the modules come from the weights' names (modules_dim), so the projection's class covers the blocks without a filter
    targets = _targets(blocks, proj)[0]
    return lora.create_network_from_weights(targets, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs)
