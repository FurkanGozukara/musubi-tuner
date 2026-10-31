# LoRA module for Anima

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# q / k / v / output_proj of the self- and cross-attention and the two MLP layers of the 28 blocks, under the checkpoint's
# names (``lora_unet_blocks_0_self_attn_q_proj``), which ComfyUI maps onto ``diffusion_model.blocks.0.self_attn.q_proj``
# without conversion. The AdaLN modulations, the embedders and the final layer are left out, as in kohya-ss sd-scripts'
# networks/lora_anima.py.
ANIMA_TARGET_REPLACE_MODULES = ["AnimaBlock"]
# network_args train_llm_adapter=True adds the LLM adapter's six blocks. The model card advises against it ("the adapter
# itself contains a surprising amount of knowledge and is easy to degrade by training it"); kohya and diffusion-pipe
# leave it off by default as well.
ANIMA_ADAPTER_TARGET_MODULES = ["AnimaAdapterBlock"]
DEFAULT_EXCLUDE = r".*(_modulation|_norm|_embedder|final_layer).*"


def _on(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _targets(train_llm_adapter) -> list:
    return ANIMA_TARGET_REPLACE_MODULES + (ANIMA_ADAPTER_TARGET_MODULES if _on(train_llm_adapter) else [])


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
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns = list(exclude_patterns) if isinstance(exclude_patterns, (list, tuple)) else [exclude_patterns]
    exclude_patterns.append(DEFAULT_EXCLUDE)
    kwargs["exclude_patterns"] = exclude_patterns
    targets = _targets(kwargs.pop("train_llm_adapter", False))

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
    kwargs.pop("train_llm_adapter", None)
    targets = _targets(any(key.startswith("lora_unet_llm_adapter_") for key in weights_sd))
    return lora.create_network_from_weights(targets, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs)
