"""MiniMax-H3 probe forwards (guidance uncond, teacher) reset the standard block-swap placement.

The standard offloader streams blocks during a forward and relies on the backward pass to bring them back.
After the no-grad guidance probe the training forward found block 0 on the CPU and stopped with
"block 0 ... is on cpu, expected cuda after wait"; H2D-only streaming handles repeated forwards itself.
"""

from types import SimpleNamespace

from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader, ModelOffloader


class _Model:
    def __init__(self, offloader, blocks_to_swap=48):
        self.offloader = offloader
        self.blocks_to_swap = blocks_to_swap
        self.prepared = 0

    def prepare_block_swap_before_forward(self):
        self.prepared += 1


def _reset(model):
    accelerator = SimpleNamespace(unwrap_model=lambda module: module)
    MiniMaxH3NetworkTrainer._reset_block_swap_after_probe(accelerator, model)
    return model.prepared


def test_standard_offloader_is_reset_after_a_probe_forward():
    assert _reset(_Model(object.__new__(ModelOffloader))) == 1


def test_h2d_streaming_and_no_swap_are_left_alone():
    assert _reset(_Model(object.__new__(LoRAStreamOffloader))) == 0
    assert _reset(_Model(object.__new__(ModelOffloader), blocks_to_swap=0)) == 0
    assert _reset(_Model(None)) == 0
