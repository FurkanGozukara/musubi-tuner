import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from musubi_tuner.gui_dashboard.command_builder import (
    build_full_finetune_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import (
    validate_full_finetune_config,
    validate_training_config,
)
from musubi_tuner.ltx2_args import ltx2_setup_parser
from musubi_tuner.ltx2_generate_video import parse_args as parse_generate_args
from musubi_tuner.ltx_2.model.transformer.attention import PytorchAttention
from musubi_tuner.ltx_2.model.transformer.transformer_args import (
    TransformerArgsPreprocessor,
)
from musubi_tuner.networks.lora_ltx2 import build_temporal_causal_attention_mask


def _positions(times):
    positions = torch.zeros((1, 3, len(times), 2), dtype=torch.float32)
    positions[:, 0, :, 0] = torch.tensor(times)
    positions[:, 0, :, 1] = torch.tensor(times) + 1
    return positions


def test_causal_mask_is_lower_triangular_by_frame_not_token():
    mask = build_temporal_causal_attention_mask(_positions([0, 0, 1, 1, 2]))
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask[0], expected)


def test_causal_attention_output_does_not_depend_on_future_values():
    torch.manual_seed(7)
    q = torch.randn(1, 4, 2)
    k = torch.randn(1, 4, 2)
    v = torch.randn(1, 4, 2)
    mask = build_temporal_causal_attention_mask(_positions([0, 1, 2, 3]))
    bias = TransformerArgsPreprocessor._prepare_self_attention_mask(None, mask, torch.float32)

    baseline = PytorchAttention()(q, k, v, heads=1, mask=bias)
    changed = v.clone()
    changed[:, 1:] += 1000
    causal_changed = PytorchAttention()(q, k, changed, heads=1, mask=bias)
    bidirectional_changed = PytorchAttention()(q, k, changed, heads=1)

    assert torch.equal(baseline[:, 0], causal_changed[:, 0])
    assert not torch.allclose(baseline[:, 0], bidirectional_changed[:, 0])


def test_causal_temporal_attention_is_disabled_by_default():
    args = ltx2_setup_parser(argparse.ArgumentParser()).parse_args([])
    assert args.ltx2_causal_temporal_attention is False


def _parse_generate_args(*extra_args):
    argv = [
        "ltx2_generate_video.py",
        "--prompt",
        "test",
        "--gemma_root",
        "gemma",
        *extra_args,
    ]
    with patch.object(sys, "argv", argv):
        return parse_generate_args()


def test_standalone_generation_causal_temporal_attention_is_opt_in():
    assert _parse_generate_args().ltx2_causal_temporal_attention is False
    args = _parse_generate_args(
        "--ltx2_causal_temporal_attention",
        "--attn_mode",
        "sdpa",
    )
    assert args.ltx2_causal_temporal_attention is True


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--ltx2_causal_temporal_attention",),
        (
            "--ltx2_causal_temporal_attention",
            "--attn_mode",
            "sdpa",
            "--ltx2_mode",
            "audio",
        ),
    ],
)
def test_standalone_generation_causal_temporal_attention_rejects_invalid_paths(
    extra_args,
):
    with pytest.raises(ValueError, match="requires"):
        _parse_generate_args(*extra_args)


def _build_dashboard_command(section, values):
    config = ProjectConfig(**{section: values})
    builder = build_training_cmd if section == "training" else build_full_finetune_cmd
    with (
        patch(
            "musubi_tuner.gui_dashboard.command_builder.export_dataset_toml",
            return_value=Path("dataset.toml"),
        ),
        patch(
            "musubi_tuner.gui_dashboard.command_builder._find_script",
            side_effect=lambda name: name,
        ),
    ):
        return builder(config)


@pytest.mark.parametrize("section", ["training", "full_finetune"])
def test_dashboard_causal_temporal_attention_is_opt_in(section):
    assert "--ltx2_causal_temporal_attention" not in _build_dashboard_command(section, {})
    command = _build_dashboard_command(section, {"ltx2_causal_temporal_attention": True, "sdpa": True})
    assert "--ltx2_causal_temporal_attention" in command


@pytest.mark.parametrize(
    ("section", "validator", "field_prefix"),
    [
        ("training", validate_training_config, "training"),
        ("full_finetune", validate_full_finetune_config, "full_finetune"),
    ],
)
def test_dashboard_causal_temporal_attention_requires_sdpa(section, validator, field_prefix):
    config = ProjectConfig(**{section: {"ltx2_causal_temporal_attention": True, "sdpa": False}})
    result = validator(config)
    assert any(issue["field"] == f"{field_prefix}.ltx2_causal_temporal_attention" for issue in result["errors"])
