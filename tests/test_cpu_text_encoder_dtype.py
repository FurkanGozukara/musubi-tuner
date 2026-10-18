"""CPU text encoding avoids BF16 / FP16 where the CPU is slow at it (utils.model_utils.cpu_text_encoder_dtype).

A Broadwell Xeon (AVX2, no native BF16) ran BF16 matmuls about 850x slower than FP32: one caption of a 9B text encoder
took 1035 s in BF16 and 21.5 s in FP32. The switch to FP32 doubles the weights' RAM, so it needs the room for them.
"""

import argparse
import inspect
import types

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.utils import model_utils


@pytest.fixture
def slow_cpu(monkeypatch):
    monkeypatch.setenv("MUSUBI_CPU_HALF_SLOW", "1")
    model_utils.cpu_dtype_is_slow.cache_clear()
    yield
    model_utils.cpu_dtype_is_slow.cache_clear()


def _checkpoint(tmp_path, name="te.safetensors"):
    path = tmp_path / name
    save_file({"a": torch.zeros(10, 20, dtype=torch.bfloat16), "b": torch.zeros(30, dtype=torch.bfloat16)}, str(path))
    return str(path)


def test_parameters_come_from_the_safetensors_headers(tmp_path):
    single = _checkpoint(tmp_path)
    assert model_utils.checkpoint_parameter_count(single) == 230
    folder = tmp_path / "shards"
    folder.mkdir()
    _checkpoint(folder, "1.safetensors")
    _checkpoint(folder, "2.safetensors")
    assert model_utils.checkpoint_parameter_count(str(folder)) == 460
    other = tmp_path / "t5.pth"
    other.write_bytes(b"\0" * 1000)
    assert model_utils.checkpoint_parameter_count(str(other)) == 500  # counted as half precision
    assert model_utils.checkpoint_parameter_count("org/model-id") is None
    assert model_utils.checkpoint_parameter_count(None) is None


def test_slow_cpu_bf16_becomes_fp32_when_the_ram_holds_it(tmp_path, slow_cpu):
    path = _checkpoint(tmp_path)
    assert model_utils.cpu_text_encoder_dtype("cpu", torch.bfloat16, path) == torch.float32
    assert model_utils.cpu_text_encoder_dtype(torch.device("cpu"), torch.float16, path) == torch.float32
    # the GPU, FP32 and FP8 are never touched
    assert model_utils.cpu_text_encoder_dtype("cuda", torch.bfloat16, path) == torch.bfloat16
    assert model_utils.cpu_text_encoder_dtype("cpu", torch.float32, path) == torch.float32
    assert model_utils.cpu_text_encoder_dtype("cpu", torch.float8_e4m3fn, path) == torch.float8_e4m3fn


def test_slow_cpu_keeps_bf16_without_room_or_size(tmp_path, slow_cpu, monkeypatch):
    import psutil

    path = _checkpoint(tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=100))
    assert model_utils.cpu_text_encoder_dtype("cpu", torch.bfloat16, path) == torch.bfloat16  # FP32 needs 1012 bytes
    monkeypatch.undo()
    monkeypatch.setenv("MUSUBI_CPU_HALF_SLOW", "1")
    assert model_utils.cpu_text_encoder_dtype("cpu", torch.bfloat16, "org/model-id") == torch.bfloat16  # size unknown


def test_fast_cpu_keeps_bf16(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_CPU_HALF_SLOW", "0")
    model_utils.cpu_dtype_is_slow.cache_clear()
    try:
        assert model_utils.cpu_text_encoder_dtype("cpu", torch.bfloat16, _checkpoint(tmp_path)) == torch.bfloat16
    finally:
        model_utils.cpu_dtype_is_slow.cache_clear()


def test_cast_for_cache_restores_the_requested_dtype_only_after_a_fallback():
    owner = types.SimpleNamespace()
    value = {"embeds": torch.ones(2), "rest": [torch.ones(1, dtype=torch.int64), (torch.ones(1),)]}
    assert model_utils.cast_for_cache(owner, value) is value  # nothing recorded
    model_utils.mark_cpu_fp32_fallback(owner, torch.bfloat16, torch.bfloat16)
    assert model_utils.cast_for_cache(owner, value) is value  # no fallback happened
    model_utils.mark_cpu_fp32_fallback(owner, torch.bfloat16, torch.float32)
    cast = model_utils.cast_for_cache(owner, value)
    assert cast["embeds"].dtype == torch.bfloat16 and cast["rest"][1][0].dtype == torch.bfloat16
    assert cast["rest"][0].dtype == torch.int64 and isinstance(cast["rest"][1], tuple)


def test_training_sample_prompts_on_a_slow_cpu_encode_in_fp32_and_keep_bf16(tmp_path, slow_cpu, monkeypatch):
    """--sample_text_encoder_device cpu hands process_sample_prompts the CPU: Krea 2 stands for the trainers."""
    from musubi_tuner import krea2_train_network
    from musubi_tuner.krea2 import krea2_utils

    loaded = {}

    def load(path, dtype, device):
        loaded.update(dtype=dtype, device=device)
        return types.SimpleNamespace()

    def embeds(encoder, prompts):
        return torch.ones(1, 3, 2, 4, dtype=loaded["dtype"]), torch.tensor([[True, True, False]])

    monkeypatch.setattr(krea2_utils, "load_krea2_text_encoder", load)
    monkeypatch.setattr(krea2_utils, "get_krea2_prompt_embeds", embeds)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a photo of a dog --n blurry\n", encoding="utf-8")
    trainer = krea2_train_network.Krea2NetworkTrainer()
    args = argparse.Namespace(text_encoder=_checkpoint(tmp_path))
    params = trainer.process_sample_prompts(args, types.SimpleNamespace(device=torch.device("cpu")), str(prompts))
    assert loaded["dtype"] == torch.float32
    assert params[0]["krea2_vl_embed"].dtype == torch.bfloat16 and params[0]["krea2_vl_embed"].shape == (2, 2, 4)
    assert params[0]["negative_krea2_vl_embed"].dtype == torch.bfloat16


def test_kandinsky6_sampling_embedder_stays_bf16():
    """Training samples build the embedder on the CPU and encode on the GPU: only the caching script, which encodes
    where it loads, may ask for FP32 (a 7B Qwen2.5-VL in FP32 is about 30 GB)."""
    from musubi_tuner.kandinsky6.runtime.core.components.text_embedder import Kandinsky6TextEmbedder

    assert inspect.signature(Kandinsky6TextEmbedder).parameters["qwen_dtype"].default is torch.bfloat16
