"""LTX-2 full fine-tuning honours --ltx2_low_ram_load (AkaneTendo25 850015b2).

With block swap the fine-tuner always loaded the transformer on the CPU, where the loader does not place
weights by device, so the checkpoint was staged whole in main RAM whatever Low-RAM Load said. With the option
the transformer is loaded on the GPU, and the loader streams the swapped blocks to the CPU.
"""

import argparse
import inspect

import pytest
import torch

from musubi_tuner import ltx2_train
from musubi_tuner.ltx2_train import _transformer_loading_device

GPU = torch.device("cuda")  # only compared, never allocated on


def _args(**overrides) -> argparse.Namespace:
    values = {"ltx2_low_ram_load": False, "blocks_to_swap": 0}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_block_swap_stages_the_transformer_on_the_cpu_without_low_ram_load():
    assert _transformer_loading_device(_args(blocks_to_swap=10), 10, False, GPU) == "cpu"
    assert _transformer_loading_device(_args(), 0, False, GPU) == GPU


def test_low_ram_load_loads_on_the_gpu_for_streamed_placement():
    args = _args(ltx2_low_ram_load=True, blocks_to_swap=10)

    assert _transformer_loading_device(args, 10, False, GPU) == GPU


def test_low_ram_load_is_refused_with_features_that_stage_the_transformer_on_the_cpu():
    with pytest.raises(ValueError, match="stage the transformer on CPU"):
        _transformer_loading_device(_args(ltx2_low_ram_load=True, blocks_to_swap=10), 10, True, GPU)
    # without the option they keep loading on the CPU
    assert _transformer_loading_device(_args(blocks_to_swap=10), 10, True, GPU) == "cpu"
    assert _transformer_loading_device(_args(), 0, True, GPU) == "cpu"


def test_low_ram_load_is_validated_as_in_lora_training():
    with pytest.raises(ValueError, match="requires --blocks_to_swap"):
        _transformer_loading_device(_args(ltx2_low_ram_load=True), 0, False, GPU)
    activation_offload = _args(ltx2_low_ram_load=True, blocks_to_swap=10, gradient_checkpointing_cpu_offload=True)
    with pytest.raises(ValueError, match="gradient_checkpointing_cpu_offload"):
        _transformer_loading_device(activation_offload, 10, False, GPU)


def test_the_fine_tuner_loads_where_it_says():
    source = inspect.getsource(ltx2_train.main)

    assert "loading_device = _transformer_loading_device(args, blocks_to_swap, force_cpu_load, accelerator.device)" in source
    assert "loading_device=loading_device" in source
