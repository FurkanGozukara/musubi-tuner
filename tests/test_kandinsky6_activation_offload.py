import copy

import pytest
import torch

from musubi_tuner.kandinsky6.dit import DiffusionTransformer3D


class _CheckpointHarness(torch.nn.Module):
    _checkpoint_block = DiffusionTransformer3D._checkpoint_block
    _run_visual_block = DiffusionTransformer3D._run_visual_block

    def __init__(self, blocks, *, activation_cpu_offloading, offloader=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.gradient_checkpointing = True
        self.is_multimodal = False
        self.activation_cpu_offloading = activation_cpu_offloading
        self._activation_offload_streams = {}
        self.offloader = offloader
        self.blocks_to_swap = int(offloader is not None)

    def forward(self, value):
        for index, block in enumerate(self.blocks):
            if self.offloader is not None:
                self.offloader.wait_for_block(index)
            value = self._run_visual_block(block, value, block_index=index)
            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.blocks, index)
        return value


class _Block(torch.nn.Module):
    def __init__(self, width, residency=None, index=None):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.residency = residency
        self.index = index

    def forward(self, value):
        if self.residency is not None and not self.residency[self.index]:
            raise RuntimeError(f"block {self.index} was recomputed before it became resident")
        return torch.nn.functional.silu(self.linear(value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for activation CPU offload")
def test_checkpoint_input_offload_preserves_cuda_outputs_gradients_and_update():
    torch.manual_seed(123)
    resident = _CheckpointHarness([_Block(64) for _ in range(4)], activation_cpu_offloading=False).cuda().to(torch.bfloat16)
    offloaded = copy.deepcopy(resident)
    offloaded.activation_cpu_offloading = True
    resident.train()
    offloaded.train()
    expected_input = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)

    expected = resident(expected_input)
    actual = offloaded(actual_input)
    assert actual.is_cuda
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    for expected_parameter, actual_parameter in zip(resident.parameters(), offloaded.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)

    for model in (resident, offloaded):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
    for expected_parameter, actual_parameter in zip(resident.parameters(), offloaded.parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for activation CPU offload")
@pytest.mark.parametrize("is_multimodal", [False, True])
def test_visual_checkpoint_offloads_activations_but_not_shared_conditioning(monkeypatch, is_multimodal):
    model = _CheckpointHarness([], activation_cpu_offloading=True).cuda().train()
    model.is_multimodal = is_multimodal
    video = torch.randn(2, 3, 5, device="cuda", requires_grad=True)
    audio = torch.randn(2, 3, 7, device="cuda", requires_grad=True)
    conditioning = torch.randn(2, 3, 11, device="cuda", requires_grad=True)
    packed_shapes = []
    empty_like = torch.empty_like

    def tracked_empty_like(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("pin_memory"):
            packed_shapes.append(tensor.shape)
        return empty_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", tracked_empty_like)

    def block(video_value, audio_value, shared):
        return video_value.sin() + shared[..., :5], audio_value.cos() + shared[..., :7]

    video_output, audio_output = model._run_visual_block(block, video, audio, conditioning, block_index=0)
    (video_output.sum() + audio_output.sum()).backward()

    assert packed_shapes == ([video.shape, audio.shape] if is_multimodal else [video.shape])
    assert video.grad is not None and audio.grad is not None and conditioning.grad is not None


class _ResidencyOffloader:
    recompute_requires_wait = True

    def __init__(self, count):
        self.resident = [False] * count
        self.waits = [0] * count

    def wait_for_block(self, index):
        self.resident[index] = True
        self.waits[index] += 1

    def submit_move_blocks_forward(self, blocks, index):
        del blocks
        self.resident[index] = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for activation CPU offload")
def test_activation_offload_waits_for_ring_residency_during_non_reentrant_recompute():
    offloader = _ResidencyOffloader(3)
    blocks = [_Block(32, offloader.resident, index) for index in range(3)]
    blocks[0].requires_grad_(False)
    model = _CheckpointHarness(blocks, activation_cpu_offloading=True, offloader=offloader).cuda().to(torch.bfloat16).train()
    value = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    model(value).float().square().mean().backward()

    assert all(wait_count == 2 for wait_count in offloader.waits)
    assert value.grad is not None
    assert all(parameter.grad is None for parameter in model.blocks[0].parameters())
    assert all(parameter.grad is not None for parameter in model.blocks[1].parameters())
