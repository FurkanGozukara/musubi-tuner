"""attention() without data-dependent graph breaks under torch.compile (V39, kohya-ss/sd-scripts 4e3905f / 5f20b9b).

`create_attention_params_from_mask` turns the sequence lengths into Python ints once per model forward (one device sync)
and records `uniform_seqlen`; `attention()` reads only these ints, so a compiled block no longer breaks at
`torch.all(seqlens == seqlens[0])` / `.item()` (2-3 breaks per attention call before).
"""

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.modules import attention

B, IMG, L, H, D = 2, 12, 8, 4, 16

MASKS = {
    "uniform": torch.tensor([[1] * 5 + [0] * 3, [1] * 5 + [0] * 3]),
    "nonuniform": torch.tensor([[1] * 6 + [0] * 2, [1] * 3 + [0] * 5]),
    "allvalid": torch.ones(B, L, dtype=torch.long),
}


def _qkv(seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, IMG + L, H, D)
    return q, torch.randn_like(q), torch.randn_like(q)


def test_mask_params_hold_python_ints_and_the_uniform_length():
    params = attention.AttentionParams.create_attention_params_from_mask("torch", False, IMG, MASKS["uniform"])
    assert params.seqlens == [IMG + 5, IMG + 5] and all(type(n) is int for n in params.seqlens)
    assert params.uniform_seqlen == IMG + 5
    assert params.cu_seqlens.dtype == torch.int32
    assert params.cu_seqlens.tolist() == [0, IMG + 5, IMG + L, 2 * (IMG + L) - (L - 5), 2 * (IMG + L)]
    nonuniform = attention.AttentionParams.create_attention_params_from_mask("torch", False, IMG, MASKS["nonuniform"])
    assert nonuniform.uniform_seqlen is None and nonuniform.seqlens == [IMG + 6, IMG + 3]
    split = attention.AttentionParams.create_attention_params_from_mask("torch", True, IMG, MASKS["uniform"])
    assert split.uniform_seqlen is None and split.seqlens == [IMG + 5, IMG + 5]


@pytest.mark.parametrize("name", list(MASKS))
@pytest.mark.parametrize("split", [False, True])
def test_valid_positions_match_masked_sdpa(name, split):
    mask = MASKS[name]
    q, k, v = _qkv()
    params = attention.AttentionParams.create_attention_params_from_mask("torch", split, IMG, mask)
    out = attention.attention(q.clone(), k.clone(), v.clone(), params)
    full_mask = F.pad(mask, (IMG, 0), value=1).to(torch.bool)  # [B, IMG + L]
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=full_mask[:, None, None, :]
    ).transpose(1, 2)
    ref = ref.reshape(B, IMG + L, H * D)
    assert out.shape == ref.shape
    for i, n in enumerate(params.seqlens):
        torch.testing.assert_close(out[i, :n], ref[i, :n], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(MASKS))
@pytest.mark.parametrize("split", [False, True])
def test_attention_has_no_graph_breaks(name, split):
    params = attention.AttentionParams.create_attention_params_from_mask("torch", split, IMG, MASKS[name])
    q, k, v = _qkv()

    def block(q, k, v):
        return attention.attention(q * 1.0, k, v, params) * 2

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(block)(q, k, v)
    assert explanation.graph_break_count == 0, explanation.break_reasons


def test_split_attention_without_mask_has_no_graph_breaks():
    q, k, v = _qkv()

    def block(q, k, v):
        return attention.attention(q * 1.0, k, v, attention.AttentionParams.create_attention_params("torch", True)) * 2

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(block)(q, k, v)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(block(q, k, v), expected.reshape(B, IMG + L, H * D) * 2, rtol=1e-5, atol=1e-5)
