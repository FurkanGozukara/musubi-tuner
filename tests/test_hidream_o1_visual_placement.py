import torch
from torch import nn

from musubi_tuner.hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration


class _Inner(nn.Module):
    def __init__(self, skip):
        super().__init__()
        self.visual = nn.Linear(2, 2)
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.skip_t2i_visual_dummy = skip


class _Model(nn.Module):
    visual_unused_in_training = Qwen3VLForConditionalGeneration.visual_unused_in_training
    move_to_device_except_swap_blocks = Qwen3VLForConditionalGeneration.move_to_device_except_swap_blocks

    def __init__(self, *, swap=2, skip=True, task="t2i"):
        super().__init__()
        self.model = _Inner(skip)
        self.lm_head = nn.Linear(2, 2)
        self.blocks_to_swap = swap
        self.hidream_o1_task = task


def _devices(model):
    return model.lm_head.weight.device.type, model.model.visual.weight.device.type, model.model.language_model.layers[0].weight.device.type


def test_unused_vision_encoder_stays_on_cpu_with_block_swap():
    model = _Model()
    model.move_to_device_except_swap_blocks(torch.device("meta"))
    assert _devices(model) == ("meta", "cpu", "cpu")  # swapped layers stay with the offloader, visual on the CPU


def test_vision_encoder_moves_when_it_can_run():
    for model in (_Model(task="i2i"), _Model(skip=False), _Model(swap=0)):
        model.move_to_device_except_swap_blocks(torch.device("meta"))
        assert model.model.visual.weight.device.type == "meta"
