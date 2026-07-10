import argparse

import pytest
import torch

from musubi_tuner.training.full_finetune import (
    TrainingProgress,
    add_full_finetune_args,
    normalize_compiled_state_dict,
    resolve_trainable_dtype,
    validate_full_finetune_args,
)


def test_training_progress_round_trip():
    progress = TrainingProgress(global_step=7, epoch=2, next_batch=3)
    restored = TrainingProgress()
    restored.load_state_dict(progress.state_dict())
    assert restored == progress


def test_normalize_compiled_state_dict_rejects_collision():
    state = {"blocks.0.weight": torch.ones(1), "blocks.0._orig_mod.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="collision"):
        normalize_compiled_state_dict(state)


def test_normalize_compiled_state_dict_removes_compile_wrappers():
    value = torch.ones(1)
    state = {"_orig_mod.blocks.0._orig_mod.weight": value}

    assert normalize_compiled_state_dict(state) == {"blocks.0.weight": value}


@pytest.mark.parametrize(
    ("full_bf16", "expected"),
    [(False, torch.float32), (True, torch.bfloat16)],
)
def test_resolve_trainable_dtype(full_bf16, expected):
    args = argparse.Namespace(full_bf16=full_bf16)

    assert resolve_trainable_dtype(args) == expected


def test_add_full_finetune_args_defaults_and_flags():
    parser = add_full_finetune_args(argparse.ArgumentParser())

    defaults = parser.parse_args([])
    assert defaults.full_bf16 is False
    assert defaults.fused_backward_pass is False
    assert defaults.mem_eff_save is False
    assert defaults.block_swap_optimizer_patch_params is False

    enabled = parser.parse_args(["--full_bf16", "--fused_backward_pass", "--mem_eff_save", "--block_swap_optimizer_patch_params"])
    assert enabled.full_bf16 is True
    assert enabled.fused_backward_pass is True
    assert enabled.mem_eff_save is True
    assert enabled.block_swap_optimizer_patch_params is True


def make_full_finetune_args(**overrides):
    defaults = {
        "network_weights": None,
        "network_module": None,
        "network_dim": None,
        "network_alpha": 1,
        "network_dropout": None,
        "network_args": None,
        "dim_from_weights": False,
        "scale_weight_norms": None,
        "base_weights": None,
        "base_weights_multiplier": None,
        "fp8_base": False,
        "fp8_scaled": False,
        "block_swap_h2d_only": False,
        "blocks_to_swap": None,
        "fused_backward_pass": False,
        "optimizer_type": "AdamW",
        "gradient_accumulation_steps": 1,
        "save_precision": None,
        "full_bf16": False,
        "mixed_precision": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("network_weights", "lora.safetensors"),
        ("network_module", "networks.lora"),
        ("network_dim", 16),
        ("network_alpha", 2),
        ("network_dropout", 0.1),
        ("network_args", ["rank_dropout=0.1"]),
        ("dim_from_weights", True),
        ("scale_weight_norms", 1.0),
        ("base_weights", ["base.safetensors"]),
        ("base_weights_multiplier", [0.5]),
    ],
)
def test_validate_full_finetune_args_rejects_non_default_lora_options(name, value):
    args = make_full_finetune_args(**{name: value})

    with pytest.raises(ValueError, match=f"--{name}"):
        validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize("name", ["fp8_base", "fp8_scaled"])
def test_validate_full_finetune_args_rejects_fp8_flags(name):
    args = make_full_finetune_args(**{name: True})

    with pytest.raises(ValueError, match=f"--{name}"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_h2d_only_swap():
    args = make_full_finetune_args(block_swap_h2d_only=True)

    with pytest.raises(ValueError, match="--block_swap_h2d_only"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_multi_process_block_swap():
    args = make_full_finetune_args(blocks_to_swap=1)

    with pytest.raises(ValueError, match="--blocks_to_swap"):
        validate_full_finetune_args(args, num_processes=2)


def test_validate_full_finetune_args_rejects_fused_backward_for_non_adafactor():
    args = make_full_finetune_args(fused_backward_pass=True, optimizer_type="AdamW")

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_multi_process_fused_backward():
    args = make_full_finetune_args(fused_backward_pass=True, optimizer_type="Adafactor")

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=2)


def test_validate_full_finetune_args_rejects_accumulated_fused_backward():
    args = make_full_finetune_args(
        fused_backward_pass=True,
        optimizer_type="Adafactor",
        gradient_accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_mismatched_save_precision():
    args = make_full_finetune_args(save_precision="bf16")

    with pytest.raises(ValueError, match="--save_precision"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_full_bf16_without_bf16_mixed_precision():
    args = make_full_finetune_args(full_bf16=True, mixed_precision="no")

    with pytest.raises(ValueError, match="--full_bf16"):
        validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize(
    ("overrides", "num_processes"),
    [
        ({}, 1),
        ({"full_bf16": True, "mixed_precision": "bf16"}, 1),
        ({"blocks_to_swap": 1}, 1),
        ({"fused_backward_pass": True, "optimizer_type": "Adafactor"}, 1),
    ],
)
def test_validate_full_finetune_args_accepts_supported_combinations(overrides, num_processes):
    args = make_full_finetune_args(**overrides)

    validate_full_finetune_args(args, num_processes)
