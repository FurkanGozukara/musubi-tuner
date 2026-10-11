"""_int_mm_allow_small_m pads the rows torch._int_mm cannot take and slices them off again.

On an RTX A6000 (Ampere) cuBLASLt's INT8 GEMM with a row-major right-hand side only accepts a row count that is a
multiple of 32: the LTX-2.3 INT8 ConvRot LoRA backward stopped with CUBLAS_STATUS_NOT_SUPPORTED for 935 tokens
(512x384, 33 frames). The forward pass's column-major right-hand side takes any row count.
"""

import pytest
import torch

from musubi_tuner.modules.w8a8_optimization_utils import _int_mm_allow_small_m


def _reference(a, b):
    return (a.double() @ b.double()).to(torch.int32)


@pytest.mark.parametrize("m", [1, 16, 17, 935])
def test_cpu_results_match_a_plain_matmul(m):
    torch.manual_seed(0)
    a = torch.randint(-127, 127, (m, 64), dtype=torch.int8)
    b = torch.randint(-127, 127, (64, 48), dtype=torch.int8)
    out = _int_mm_allow_small_m(a, b)
    assert out.shape == (m, 48)
    assert torch.equal(out, _reference(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m", [5, 17, 935, 944, 960, 1937])
def test_cuda_row_counts_that_cublaslt_rejects(m):
    torch.manual_seed(0)
    a = torch.randint(-127, 127, (m, 256), dtype=torch.int8, device="cuda")
    row_major = torch.randint(-127, 127, (256, 512), dtype=torch.int8, device="cuda")
    col_major = torch.randint(-127, 127, (512, 256), dtype=torch.int8, device="cuda").t()
    for b in (row_major, col_major):
        out = _int_mm_allow_small_m(a, b)
        assert out.shape == (m, 512)
        assert torch.equal(out, _reference(a, b))
