"""HiDream-O1 vendored Qwen3-VL text stack runs forward + backward with transformers 5 (tiny CPU model)."""

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from musubi_tuner.hidream_o1.qwen3_vl_transformers import Qwen3VLTextModel


def _tiny_config():
    config = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2], "mrope_interleaved": True, "rope_theta": 5000000},
    )
    config._attn_implementation = "sdpa"
    return config


def test_text_model_forward_backward():
    torch.manual_seed(0)
    model = Qwen3VLTextModel(_tiny_config()).float()
    inputs_embeds = torch.randn(1, 12, 64, requires_grad=True)
    position_ids = torch.arange(12).view(1, 1, -1).expand(3, 1, -1)
    output = model(inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False)
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    assert hidden.shape == (1, 12, 64)
    hidden.float().pow(2).mean().backward()
    assert inputs_embeds.grad is not None and torch.isfinite(inputs_embeds.grad).all()