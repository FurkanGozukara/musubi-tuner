# LoRA module for FLUX.3 [klein]

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# q / k / v, mlp_in, attn_out and mlp_out of the per-mode (image, image_cond, txt) and joint blocks. The modulations,
# the input embeddings and the final layer sit outside these blocks; the names match BFL's checkpoint and ComfyUI's
# model, so ``lora_unet_single_blocks_0_q_proj`` loads there without conversion.
FLUX_3_TARGET_REPLACE_MODULES = ["Flux3ModeBlock", "Flux3JointBlock"]


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

    # text-to-image training never runs the reference-image stream (flux_3_train_network sets this from the datasets)
    if getattr(unet, "trains_references", True) is False:
        exclude_patterns.append(r"content_mode_blocks\.image_cond\..*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        FLUX_3_TARGET_REPLACE_MODULES,
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
    return lora.create_network_from_weights(
        FLUX_3_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
