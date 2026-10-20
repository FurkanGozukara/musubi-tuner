"""H2D-only block swap's pageable -> pinned staging copy (custom_offloading_utils._host_memcpy, kohya-ss/sd-scripts
3a082f5): one memmove on the worker thread instead of torch's CPU copy_, which spread every block's copy over all
intra-op threads (46 of 64 cores busy in a MOSS-Image run). Same bytes; anything unusual still goes through copy_."""

import inspect

import pytest
import torch

from musubi_tuner.modules import custom_offloading_utils as offload


def test_memmove_copies_the_same_bytes():
    torch.manual_seed(0)
    src = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8)
    dst = torch.zeros_like(src)
    offload._host_memcpy(dst, src)
    assert torch.equal(dst, src)
    floats = torch.randn(1000, dtype=torch.bfloat16)
    target = torch.empty(2000, dtype=torch.uint8)
    offload._host_memcpy(target, floats)  # equal byte size: copied byte for byte
    assert torch.equal(target.view(torch.bfloat16), floats)


def test_unusual_tensors_fall_back_to_copy():
    src = torch.arange(12, dtype=torch.uint8).view(3, 4).t()  # not contiguous
    dst = torch.empty(4, 3, dtype=torch.uint8)
    offload._host_memcpy(dst, src)
    assert torch.equal(dst, src)
    with pytest.raises(RuntimeError):  # a size mismatch raises as copy_ does, never writes past the buffer
        offload._host_memcpy(torch.empty(8, dtype=torch.uint8), torch.empty(16, dtype=torch.uint8))


def test_the_staged_copier_uses_it():
    assert "_host_memcpy(staging, src_flat)" in inspect.getsource(offload._StagedCopier._run)
