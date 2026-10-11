"""The FP8 Gemma loader places offloaded FP8 Linear weights on the CPU while loading.

With FP8 weight offload (the default), FP8Linear keeps its weight in CPU RAM and moves it per forward, but the
loader used to materialize every tensor on the GPU first: LTX-2.3 text caching with the single-file FP8 Gemma 3
12B peaked at 12.8 GB before replace_linear_with_fp8 moved the weights, which kept LTX-2.3 out of 8-12 GB GPUs.
"""

import torch

from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import _fp8_weight_load_device

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def test_offloaded_fp8_linear_weights_load_on_the_cpu():
    linear = torch.nn.Linear(4, 4)
    fp8 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    assert _fp8_weight_load_device(fp8, linear, "weight", CUDA, True) == CPU
    assert _fp8_weight_load_device(fp8.to(torch.float8_e5m2), linear, "weight", CUDA, True) == CPU


def test_everything_else_keeps_the_load_device():
    linear = torch.nn.Linear(4, 4)
    fp8 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    # offload disabled: FP8Linear keeps the weight in VRAM
    assert _fp8_weight_load_device(fp8, linear, "weight", CUDA, False) == CUDA
    # BF16 weights (norms, embeddings, unreplaced Linear layers) stay where the model runs
    assert _fp8_weight_load_device(torch.zeros(4, 4, dtype=torch.bfloat16), linear, "weight", CUDA, True) == CUDA
    # replace_linear_with_fp8 only wraps nn.Linear weights, so an FP8 tensor of another module stays put
    assert _fp8_weight_load_device(fp8, torch.nn.Embedding(4, 4), "weight", CUDA, True) == CUDA
    assert _fp8_weight_load_device(fp8[0], linear, "bias", CUDA, True) == CUDA
