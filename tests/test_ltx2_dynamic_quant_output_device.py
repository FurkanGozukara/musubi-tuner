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


def test_transfers_are_batched_and_finished():
    """Copies to the output device are batched (one synchronous copy per tensor would serialize GPU quantization
    with reading the checkpoint); finish() moves the rest and returns a plain dict."""
    from musubi_tuner.ltx2_model_loading import _StateDictOnDevice

    sd = _StateDictOnDevice(torch.device("meta"), batch_bytes=64)
    sd["a"] = torch.zeros(8, dtype=torch.float32)  # 32 bytes: waits for the batch
    assert sd["a"].device.type == "cpu"
    sd["b"] = torch.zeros(8, dtype=torch.float32)  # 64 bytes: the batch moves
    assert {tensor.device.type for tensor in sd.values()} == {"meta"}
    sd["c"] = torch.zeros(2)
    result = sd.finish()
    assert type(result) is dict
    assert sorted(result) == ["a", "b", "c"]
    assert {tensor.device.type for tensor in result.values()} == {"meta"}


def test_model_key_filter_loads_only_the_variants_tensors():
    """A video-only model skips the checkpoint's audio branch, VAEs and vocoder (~18.5 GB of BF16 in LTX-2.3)."""
    from musubi_tuner.ltx2_model_loading import make_ltx2_model_key_filter

    keep = make_ltx2_model_key_filter({"transformer_blocks.0.attn1.to_q.weight", "transformer_blocks.0.attn1.to_q.bias", "proj_out.weight"})
    assert keep("transformer_blocks.0.attn1.to_q.weight")
    assert keep("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight")  # Comfy prefix
    assert keep("transformer_blocks.0.attn1.to_q.weight_scale") and keep("transformer_blocks.0.attn1.to_q.comfy_quant")
    for skipped in ("transformer_blocks.0.audio_attn1.to_q.weight", "transformer_blocks.0.audio_scale_shift_table", "vocoder.conv.weight", "vae.decoder.weight"):
        assert not keep(skipped), skipped
