"""LTX-2 Gemma loader: transformers 5 per-layer-type rotary buffers are rebuilt from config."""

import torch
from transformers.models.gemma3 import modeling_gemma3 as hf
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import _materialize_meta_buffer


def test_gemma3_rotary_buffers_rebuilt_from_meta():
    config = Gemma3TextConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=128,
    )
    reference = hf.Gemma3RotaryEmbedding(config)
    with torch.device("meta"):
        rebuilt = hf.Gemma3RotaryEmbedding(config)
    root = torch.nn.Module()
    root.rotary_emb = rebuilt
    names = [name for name, buffer in root.named_buffers() if buffer.device.type == "meta"]
    assert any(name.endswith("_inv_freq") and not name.endswith(".inv_freq") for name in names)
    for name in names:
        _materialize_meta_buffer(root, name, dict(root.named_buffers())[name], device=torch.device("cpu"))
    for name, buffer in reference.named_buffers():
        assert torch.allclose(getattr(rebuilt, name), buffer), name