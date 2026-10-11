"""Ideogram 4 with activation CPU offloading: the last block's output comes back on the CPU.

The offloading wrapper returns every block's output on the CPU; the final layer then failed with
"Expected all tensors to be on the same device" (cuda:0 and cpu) in every LoRA run that combined
activation CPU offload with block swap.
"""

import pytest
import torch

from musubi_tuner.ideogram4.ideogram4_model import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Config,
    Ideogram4Transformer,
)


def _tiny_model() -> Ideogram4Transformer:
    torch.manual_seed(0)
    config = Ideogram4Config(
        emb_dim=32, num_layers=2, num_heads=4, intermediate_size=64, adanln_dim=16, in_channels=4, llm_features_dim=8, mrope_section=(2, 1, 1)
    )
    return Ideogram4Transformer(config, attn_mode="torch")


def _inputs(device):
    indicator = torch.tensor([[OUTPUT_IMAGE_INDICATOR, OUTPUT_IMAGE_INDICATOR, LLM_TOKEN_INDICATOR, LLM_TOKEN_INDICATOR]], device=device)
    return dict(
        llm_features=torch.randn(1, 4, 8, device=device),
        x=torch.randn(1, 4, 4, device=device),
        t=torch.rand(1, device=device),
        position_ids=torch.arange(4, device=device).view(1, 4, 1).expand(-1, -1, 3),
        attention_mask=torch.ones(1, 2, device=device, dtype=torch.bool),
        indicator=indicator,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.device_count(), reason="needs a CUDA device"))],
)
def test_activation_offload_forward_and_backward(device):
    model = _tiny_model().to(device)
    model.train()
    model.enable_gradient_checkpointing(activation_cpu_offloading=True)
    output = model(**_inputs(device))
    assert output.device.type == device
    output.square().mean().backward()
    assert model.input_proj.weight.grad is not None
