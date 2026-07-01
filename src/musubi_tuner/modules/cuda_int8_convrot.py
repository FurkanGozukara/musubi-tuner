"""Optional CUDA kernels for INT8 ConvRot activation quantization.

These kernels are intentionally separate from the CUTLASS int8 GEMM extension:
they only need the PyTorch CUDA extension toolchain and accelerate the online
ConvRot rotate + row-wise activation quantization step.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_EXTENSION = None
_LOAD_ERROR: Exception | None = None


def _load_extension():
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        _LOAD_ERROR = exc
        raise

    source_dir = Path(__file__).resolve().parent / "cuda_int8_convrot_ext"
    try:
        _EXTENSION = load(
            name="ltx2_cuda_int8_convrot",
            sources=[str(source_dir / "binding.cpp"), str(source_dir / "int8_convrot_kernels.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            with_cuda=True,
            verbose=os.getenv("LTX2_CUDA_INT8_CONVROT_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:  # pragma: no cover - build environment dependent
        _LOAD_ERROR = exc
        raise
    return _EXTENSION


def is_available() -> bool:
    try:
        _load_extension()
    except Exception as exc:
        logger.debug("CUDA INT8 ConvRot extension unavailable: %s", exc)
        return False
    return True


def quantize_rowwise_convrot(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q_int8, row_scale)`` for ``rotate_convrot(x)``.

    ``x`` must be a contiguous CUDA matrix with feature dimension divisible by
    ``group_size``. Supported group sizes are 16, 64, and 256.
    """

    q, scale = _load_extension().quantize_rowwise_convrot(x, int(group_size))
    return q, scale


def quantize_rowwise(x: torch.Tensor, col_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row int8 quantization of ``x`` with optional column scale folded in."""

    q, scale = _load_extension().quantize_rowwise(x, col_scale)
    return q, scale


def dequant_epilogue(
    out_int32: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply row/column scales and optional bias to an int32 matmul result."""

    return _load_extension().dequant_epilogue(out_int32, row_scale, col_scale, bias, _dtype_code(out_dtype))


def dequant_epilogue_convrot(
    out_int32: torch.Tensor,
    row_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply row scale and inverse ConvRot to an int32 grad-input matmul result."""

    return _load_extension().dequant_epilogue_convrot(out_int32, row_scale, int(group_size), _dtype_code(out_dtype))


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"CUDA INT8 ConvRot kernels do not support output dtype {dtype}")
