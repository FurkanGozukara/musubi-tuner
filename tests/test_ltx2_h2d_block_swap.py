"""LTX-2 H2D-only block swap on the shared LoRAStreamOffloader.

The LTX code called the offloader's former ``_modules`` / ``_swap_modules`` API, which the shared offloader
replaced with swap jobs (``_jobs`` and a ``swap_tensor_selector``): ``--block_swap_h2d_only`` stopped at
``enable_block_swap`` with an AttributeError and LTX's swap mask was no longer applied. The copiers are the
offloader's only CUDA-bound parts, so these checks run on the CPU with a recording copier.
"""

import pytest
import torch

from musubi_tuner.ltx_2.model.transformer.model import LTXModel, LTXModelType
from musubi_tuner.modules import custom_offloading_utils
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


class _RecordingCopier:
    def __init__(self, device, *args, **kwargs):
        self.submitted = []

    def submit(self, key, dst_flat, src_flat, gate_event=None):
        self.submitted.append(key)
        dst_flat.copy_(src_flat)

    def wait(self, key):
        return None

    def pop_xfer_timing(self, key):
        return None

    def sync(self):
        pass

    def reset(self):
        pass


@pytest.fixture
def recording_copier(monkeypatch):
    monkeypatch.setattr(custom_offloading_utils, "_DirectCopier", _RecordingCopier)
    monkeypatch.setattr(custom_offloading_utils, "_StagedCopier", _RecordingCopier)


def _tiny_ltx(num_layers: int = 4) -> LTXModel:
    torch.manual_seed(0)
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=8,
        out_channels=8,
        num_layers=num_layers,
        cross_attention_dim=16,
        caption_channels=16,
    )
    model.requires_grad_(False)  # H2D-only swap streams a frozen base
    return model


def _h2d_config() -> BlockSwapConfig:
    # the device only has to read as CUDA here: nothing is allocated on it before the first prepare
    return BlockSwapConfig(device=torch.device("cuda"), supports_backward=True, h2d_only=True)


def _linears(block: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in block.modules() if module.__class__.__name__.endswith("Linear")]


def test_enable_h2d_block_swap_marks_the_streamed_linears(recording_copier):
    model = _tiny_ltx()

    model.enable_block_swap(2, _h2d_config())

    streamed = set(model.offloader.stream_idx)
    assert streamed == {1, 3}
    for idx, block in enumerate(model.transformer_blocks):
        assert _linears(block)
        for module in _linears(block):
            assert module._h2d_stream_managed == (idx in streamed)


def test_h2d_block_swap_streams_only_the_masked_linears(recording_copier, monkeypatch):
    monkeypatch.setenv("LTX2_FULL_FT_SWAP_MASK", "ff")
    model = _tiny_ltx()

    model.enable_block_swap(2, _h2d_config())

    block = model.transformer_blocks[1]
    feed_forward = {id(block.ff.net[0].proj), id(block.ff.net[2])}
    assert {id(module) for module, _name, _is_param in model.offloader._jobs(1)} == feed_forward
    for module in _linears(block):
        assert module._h2d_stream_managed == (id(module) in feed_forward)
