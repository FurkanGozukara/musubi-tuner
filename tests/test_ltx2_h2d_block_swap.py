"""LTX-2 H2D-only block swap on the shared LoRAStreamOffloader.

The LTX code called the offloader's former ``_modules`` / ``_swap_modules`` API, which the shared offloader
replaced with swap jobs (``_jobs`` and a ``swap_tensor_selector``): ``--block_swap_h2d_only`` stopped at
``enable_block_swap`` with an AttributeError and LTX's swap mask was no longer applied.

``Module.to`` moves registered Parameters in place, so a device move of a block bound to the H2D ring also moves
the ring's own Parameters off the GPU, and it takes the streamed blocks' non-swap tensors to the CPU. Training
samples ended with such a move after the ring was prepared (AkaneTendo25 12eedd9f): the ring is now prepared after
the last placement move, and the move rebuilds it.

The copiers are the offloader's only CUDA-bound parts, so most checks run on the CPU with a recording copier; the
CUDA tests run the real offloader.
"""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from musubi_tuner import ltx2_sampling
from musubi_tuner.ltx2_sampling import LTX2SamplingMixin
from musubi_tuner.ltx_2.model.transformer.model import LTXModel, LTXModelType
from musubi_tuner.modules import custom_offloading_utils
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, LoRAStreamOffloader


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


def _h2d_config(device: torch.device = torch.device("cuda"), supports_backward: bool = True) -> BlockSwapConfig:
    # on the CPU the device only has to read as CUDA: nothing is allocated on it before the first prepare
    return BlockSwapConfig(device=device, supports_backward=supports_backward, h2d_only=True)


def _linears(block: nn.Module) -> list[nn.Module]:
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


# ---------------------------------------------------------------- sampling order


class _RecordingTransformer:
    """Records placement moves and block-ring preparations the way the LTX-2 wrapper exposes them."""

    def __init__(self, log: list, device: str):
        self.log = log
        self.device = torch.device(device)
        self.blocks_to_swap = 2

    def parameters(self):
        yield SimpleNamespace(device=self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.log.append(("move", self.device.type))
        return self

    def move_to_device_except_swap_blocks(self, device):
        self.device = torch.device(device)
        self.log.append(("move", self.device.type))

    def switch_block_swap_for_inference(self):
        self.log.append(("prepare", "inference"))

    def switch_block_swap_for_training(self):
        self.log.append(("prepare", "training"))

    def prepare_block_swap_before_forward(self):
        self.log.append(("prepare", None))


class _Accelerator:
    def __init__(self, device: str):
        self.device = torch.device(device)

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return contextlib.nullcontext()


class _Sampler(LTX2SamplingMixin):
    _ltx_mode = "video"
    _audio_video = False
    default_guidance_scale = 1.0

    def __init__(self, log: list):
        self.log = log

    def sample_image_inference(self, *args, **kwargs):
        self.log.append(("sample",))

    def _load_vae_impl(self, *args, **kwargs):
        return SimpleNamespace(to_device=lambda device: None)

    @staticmethod
    def _cleanup_cuda(device):
        pass


def _sample(tmp_path, monkeypatch, *, transformer_device: str, offload: bool) -> list:
    monkeypatch.setattr(ltx2_sampling, "should_sample_images", lambda *args: True)
    monkeypatch.setattr(ltx2_sampling, "clean_memory_on_device", lambda device: None)
    log = []
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        sample_prompts="prompts.txt",
        sample_with_offloading=offload,
        vae=None,
        vae_dtype=None,
        ltx_mode="video",
    )
    # the offloading path runs on CUDA only; the device just has to read as one
    accelerator = _Accelerator("cuda" if offload else "cpu")
    prompts = [{"prompt_embeds": torch.zeros(1), "cfg_scale": 1.0} for _ in range(2)]
    transformer = _RecordingTransformer(log, transformer_device)
    _Sampler(log).sample_images(accelerator, args, 1, 10, None, transformer, prompts, torch.float32)
    return log


def _assert_ring_prepared_after_every_move(log: list) -> None:
    mode, ready = "training", True  # the training loop prepared the ring before sampling
    for event in log:
        if event[0] == "move":
            ready = False  # a move rebinds the weights a ring has bound
        elif event[0] == "prepare":
            mode, ready = event[1] or mode, True
        elif event[0] == "sample":
            assert ready and mode == "inference", log
    assert ready and mode == "training", log
    # leaving inference mode (which finishes the transfers in flight) comes before the closing placement move
    assert log[-3] == ("prepare", "training") and log[-2][0] == "move" and log[-1] == ("prepare", None), log


@pytest.mark.parametrize("transformer_device", ["cuda", "cpu"])
def test_sampling_prepares_the_block_ring_after_the_placement_moves(tmp_path, monkeypatch, transformer_device):
    # "cpu": the first parameter is on the CPU, so sampling starts with a placement move
    log = _sample(tmp_path, monkeypatch, transformer_device=transformer_device, offload=False)

    assert log.count(("sample",)) == 2
    _assert_ring_prepared_after_every_move(log)


def test_offloaded_sampling_prepares_the_block_ring_after_each_move_to_the_gpu(tmp_path, monkeypatch):
    log = _sample(tmp_path, monkeypatch, transformer_device="cuda", offload=True)

    assert log.count(("sample",)) == 2
    assert log.count(("move", "cpu")) >= 3  # before the prompt loop and after every prompt
    _assert_ring_prepared_after_every_move(log)


def test_h2d_block_swap_allows_sample_offloading_but_not_the_block_cpu_offloads():
    # the offloaded sampler's moves rebuild the ring (above), so only the two block CPU offloads stay refused
    base = {"block_swap_h2d_only": True, "gradient_checkpointing": True}
    device = torch.device("cuda")

    config = BlockSwapConfig.from_args(SimpleNamespace(**base, sample_with_offloading=True), device, supports_backward=True)

    assert config.h2d_only
    for flag in ("blockwise_checkpointing", "gradient_checkpointing_cpu_offload"):
        with pytest.raises(ValueError, match=flag):
            BlockSwapConfig.from_args(SimpleNamespace(**base, **{flag: True}), device, supports_backward=True)


# ---------------------------------------------------------------- the ring after a device move


def test_placement_move_rebuilds_the_ring_after_moving_the_blocks(recording_copier, monkeypatch):
    model = _tiny_ltx()
    model.enable_block_swap(2, _h2d_config())
    log = []
    for idx, block in enumerate(model.transformer_blocks):
        monkeypatch.setattr(block, "to", lambda *args, _idx=idx, **kwargs: log.append(("to", _idx)))
    monkeypatch.setattr(model.offloader, "invalidate_ring_bindings_after_device_move", lambda: log.append("rebuild"))

    model.move_to_device_except_swap_blocks(torch.device("cpu"))

    assert sorted(log[:-1]) == [("to", idx) for idx in range(4)]
    assert log[-1] == "rebuild"


class _Block(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(8, 8, generator=generator))
            self.proj.bias.copy_(torch.randn(8, generator=generator))
            self.norm.weight.copy_(torch.rand(8, generator=generator) + 0.5)
        self.requires_grad_(False)

    def forward(self, x):
        return self.norm(self.proj(x))


def _cpu_offloader(blocks: list[nn.Module], blocks_to_swap: int) -> LoRAStreamOffloader:
    offloader = LoRAStreamOffloader(
        "test", blocks, len(blocks), blocks_to_swap, False, torch.device("cuda"), ring_size=2, use_pinned_memory=False
    )
    offloader.device = torch.device("cpu")  # the CPU stands in for the GPU: tensors are told apart by their storage
    return offloader


def _on(tensor: torch.Tensor, flat: torch.Tensor) -> bool:
    return tensor.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()


def _move_from_the_gpu(block: nn.Module, offloader: LoRAStreamOffloader) -> None:
    # block.to("cpu") as it goes through Module._apply: every Parameter keeps its object and gets the moved tensor
    # as .data. Moving from the GPU makes new storage (a clone here); the CPU masters stay where they are.
    masters = {flat.untyped_storage().data_ptr() for flat in offloader.cpu_flat.values()}
    block._apply(lambda tensor: tensor if tensor.untyped_storage().data_ptr() in masters else tensor.clone())


def test_device_move_rebuilds_the_ring_views(recording_copier):
    blocks = [_Block(seed) for seed in range(4)]
    weights = [block.proj.weight.detach().clone() for block in blocks]
    offloader = _cpu_offloader(blocks, blocks_to_swap=4)
    offloader.prepare_block_devices_before_forward(blocks)
    assert offloader.in_slot == [0, 1]

    for block in blocks:
        _move_from_the_gpu(block, offloader)
    assert not _on(offloader.ring_param[0][0], offloader.ring_flat[0])  # the move took the ring's Parameter along

    offloader.copier.submitted.clear()
    offloader.invalidate_ring_bindings_after_device_move()
    offloader.prepare_block_devices_before_forward(blocks)

    assert sorted(offloader.copier.submitted) == [0, 1]  # every slot refilled from its master
    for slot, flat in enumerate(offloader.ring_flat):
        assert all(_on(tensor, flat) for tensor in offloader.ring_param[slot])
    for idx, block in enumerate(blocks):
        bound = offloader.ring_param[idx][0] if idx < 2 else offloader.cpu_master[idx][0]
        assert block.proj.weight is bound
        assert torch.equal(block.proj.weight, weights[idx])
        assert _on(offloader.cpu_master[idx][0], offloader.cpu_flat[idx])


def test_prepare_after_a_device_move_places_the_blocks_with_their_masters_unbound(recording_copier, monkeypatch):
    blocks = [_Block(seed) for seed in range(4)]
    offloader = _cpu_offloader(blocks, blocks_to_swap=2)
    offloader.prepare_block_devices_before_forward(blocks)
    assert offloader.stream_idx == [1, 3]
    moves = []
    for idx, block in enumerate(blocks):
        original = block.to

        def to(*args, _idx=idx, _block=block, _original=original, **kwargs):
            moves.append((_idx, _block.proj._parameters["weight"] is None))
            return _original(*args, **kwargs)

        monkeypatch.setattr(block, "to", to)

    offloader.prepare_block_devices_before_forward(blocks)
    assert moves == []  # a plain re-prepare leaves the placement alone

    offloader.invalidate_ring_bindings_after_device_move()
    offloader.prepare_block_devices_before_forward(blocks)
    # resident blocks move whole; streamed blocks move without their swap tensors, so the masters stay on the CPU
    assert sorted(moves) == [(0, False), (1, True), (2, False), (3, True)]
    assert blocks[1].proj.weight is offloader.ring_param[0][0]
    assert blocks[3].proj.weight is offloader.ring_param[1][0]


# ---------------------------------------------------------------- CUDA


def _run_blocks(blocks: list[nn.Module], offloader, x: torch.Tensor) -> torch.Tensor:
    for idx, block in enumerate(blocks):
        offloader.wait_for_block(idx)
        x = block(x)
        offloader.submit_move_blocks_forward(blocks, idx)
    return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LoRAStreamOffloader")
@pytest.mark.parametrize("use_pinned_memory", [False, True])
@pytest.mark.parametrize("whole_move_first", [False, True])
def test_cuda_ring_survives_a_device_move_of_the_streamed_blocks(use_pinned_memory, whole_move_first):
    device = torch.device("cuda")
    blocks = [_Block(seed) for seed in range(6)]
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(7)).to(device)
    with torch.no_grad():
        reference = x
        for block in blocks:
            reference = copy.deepcopy(block).to(device)(reference)  # the same weights on the same device
    offloader = LoRAStreamOffloader("test", blocks, 6, 3, False, device, ring_size=2, use_pinned_memory=use_pinned_memory)

    with torch.no_grad():
        offloader.prepare_block_devices_before_forward(blocks)
        first = _run_blocks(blocks, offloader, x)
        if whole_move_first:  # --sample_with_offloading moves the whole transformer to the CPU before that
            for block in blocks:
                block.to("cpu")
        # what LTX-2's placement helper does after sampling: streamed blocks to the CPU, the others to the GPU
        for idx, block in enumerate(blocks):
            block.to("cpu" if offloader.is_stream[idx] else device)
        offloader.invalidate_ring_bindings_after_device_move()
        offloader.prepare_block_devices_before_forward(blocks)

        for block in blocks:
            assert block.norm.weight.device.type == "cuda"
            assert block.proj.bias.device.type == "cuda"
        for idx in offloader.stream_idx:
            master = offloader.cpu_master[idx][0]
            assert master.device.type == "cpu"
            assert master.is_pinned() == use_pinned_memory
        second = _run_blocks(blocks, offloader, x)
    torch.cuda.synchronize()

    torch.testing.assert_close(first, reference)
    torch.testing.assert_close(second, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LoRAStreamOffloader")
def test_cuda_ltx_sampling_end_keeps_the_h2d_layout():
    device = torch.device("cuda")
    model = _tiny_ltx(num_layers=6)
    model.enable_block_swap(4, _h2d_config(device, supports_backward=False))
    model.move_to_device_except_swap_blocks(device)
    model.switch_block_swap_for_inference()

    # the end of LTX2SamplingMixin.sample_images
    model.switch_block_swap_for_training()
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    torch.cuda.synchronize()

    _assert_h2d_layout(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LoRAStreamOffloader")
def test_cuda_ltx_offloaded_sampling_keeps_the_h2d_layout():
    device = torch.device("cuda")
    model = _tiny_ltx(num_layers=6)
    model.enable_block_swap(4, _h2d_config(device))
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()

    # LTX2SamplingMixin.sample_images with --sample_with_offloading, one prompt
    model.to("cpu")  # before the prompt loop
    model.move_to_device_except_swap_blocks(device)  # for the denoise
    model.switch_block_swap_for_inference()
    model.move_to_device_except_swap_blocks(torch.device("cpu"))  # for the VAE decode
    model.to("cpu")  # after the prompt
    model.move_to_device_except_swap_blocks(device)  # back on the training device
    model.switch_block_swap_for_training()
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    torch.cuda.synchronize()

    _assert_h2d_layout(model)


def _assert_h2d_layout(model: LTXModel) -> None:
    offloader = model.offloader
    assert offloader.stream_idx == [0, 2, 3, 5]
    assert offloader.in_slot == [0, 2]  # blocks 3 and 5 stay on their CPU masters
    for idx, block in enumerate(model.transformer_blocks):
        swap = {(id(module), name) for module, name, _ in offloader._jobs(idx)} if offloader.is_stream[idx] else set()
        in_ring = idx in offloader.in_slot
        for module in block.modules():
            tensors = list(module._parameters.items()) + list(module._buffers.items())
            for name, tensor in tensors:
                if tensor is None:
                    continue
                expected = "cpu" if (id(module), name) in swap and not in_ring else "cuda"
                assert tensor.device.type == expected, (idx, type(module).__name__, name)
