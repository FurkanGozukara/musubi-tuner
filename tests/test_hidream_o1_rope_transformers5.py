"""HiDream-O1's vendored Qwen3-VL rotary embedding must work with transformers >= 5."""

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from musubi_tuner.hidream_o1.qwen3_vl_transformers import Qwen3VLTextRotaryEmbedding


def _config():
    return Qwen3VLTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=1,
        max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2], "mrope_interleaved": True, "rope_theta": 5000000},
    )


def test_default_rope_matches_transformers():
    config = _config()
    ours = Qwen3VLTextRotaryEmbedding(config)
    reference = hf.Qwen3VLTextRotaryEmbedding(config)
    assert torch.allclose(ours.inv_freq.float(), reference.inv_freq.float())
    assert ours.attention_scaling == 1.0


def test_rope_hook_used_by_transformers_weight_init():
    inv_freq, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(_config())
    assert inv_freq.shape == (8,) and scaling == 1.0