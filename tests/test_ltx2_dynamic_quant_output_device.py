"""The dynamic LTX-2 quantization loaders keep their results on ``output_device``.

Block swap loads the LTX-2 transformer on the CPU while the quantization runs on the GPU; the
quantized weights must follow the load device, or the whole 22B model sits on the GPU before
block swap can move any block off it (25.8 GB at every block-swap setting).
"""

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.ltx2_model_loading import (
    load_safetensors_dynamic_int4_convrot,
    load_safetensors_dynamic_int8,
    load_safetensors_dynamic_int8_convrot,
)

LOADERS = {
    "int8": load_safetensors_dynamic_int8,
    "int8_convrot": load_safetensors_dynamic_int8_convrot,
    "int4_convrot": load_safetensors_dynamic_int4_convrot,
}
QUANTIZED_KEY = "transformer_blocks.0.attn1.to_q.weight"


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "ltx.safetensors"
    save_file(
        {
            QUANTIZED_KEY: torch.randn(64, 128).to(torch.bfloat16),
            "transformer_blocks.0.attn1.to_q.bias": torch.randn(64).to(torch.bfloat16),
            "transformer_blocks.0.norm.weight": torch.ones(128, dtype=torch.float32),
        },
        str(path),
    )
    return [str(path)]


@pytest.mark.parametrize("kind", sorted(LOADERS))
def test_results_follow_output_device(tmp_path, kind):
    files = _checkpoint(tmp_path)
    load = LOADERS[kind]
    kept = load(files, target_keys=["transformer_blocks"], exclude_keys=[], calc_device="cpu")
    # "meta" stands in for the CPU of a block-swap load without needing a GPU: whatever device the
    # quantization ran on, every stored tensor must land on output_device.
    moved = load(files, target_keys=["transformer_blocks"], exclude_keys=[], calc_device="cpu", output_device="meta")

    assert {tensor.device.type for tensor in kept.values()} == {"cpu"}
    assert {tensor.device.type for tensor in moved.values()} == {"meta"}
    assert sorted(moved) == sorted(kept)
    assert kept[QUANTIZED_KEY].dtype not in (torch.bfloat16, torch.float32), "the target weight was not quantized"
    for key, tensor in kept.items():
        assert (moved[key].dtype, moved[key].shape) == (tensor.dtype, tensor.shape), key
