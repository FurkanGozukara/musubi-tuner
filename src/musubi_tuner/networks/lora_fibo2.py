# LoRA module for FIBO 2.0

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Every Linear of the 44 blocks, the two noise-refiner blocks and the Perceiver's two layers (ai-toolkit's fibo2 targets:
# its transformer_block_names noise_refiner / perceiver.layers / layers): the fused attention.qkv, attention.out,
# feed_forward.w13 / w2, adaLN_modulation.0 and the injection blocks' cross_attention.to_q / to_kv / to_out. The names are
# ComfyUI's fused modules (``lora_unet_layers_0_attention_qkv``), so ComfyUI's own loader maps every key, and each DoRA
# module keeps its own magnitude (Guardrail 18). The embedders, the latent queries and the final layer stay out.
FIBO2_TARGET_REPLACE_MODULES = ["Fibo2Block", "Fibo2PerceiverLayer"]


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

    return lora.create_network(
        FIBO2_TARGET_REPLACE_MODULES,
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
        FIBO2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
