"""Optional CUDA kernels for INT4 ConvRot W4A4 linear layers."""

from __future__ import annotations

import logging
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

    source_dir = Path(__file__).resolve().parent / "cuda_int4_convrot_ext"
    try:
        _EXTENSION = load(
            name="ltx2_cuda_int4_convrot",
            sources=[str(source_dir / "binding.cpp"), str(source_dir / "int4_convrot_kernels.cu")],
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
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
    except Exception as exc:  # pragma: no cover - build environment dependent
        _LOAD_ERROR = exc
        raise
    return _EXTENSION


def is_available() -> bool:
    try:
        _load_extension()
    except Exception as exc:
        logger.debug("CUDA INT4 ConvRot extension unavailable: %s", exc)
        return False
    return True


def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed signed-int4 rowwise quantization of ``x`` and fp32 row scales."""

    q, scale = _load_extension().quantize_rowwise(x.contiguous())
    return q, scale


def unpack_to_int8(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack signed-int4 nibbles to an int8 CUDA matrix."""

    return _load_extension().unpack_to_int8(packed.contiguous(), int(num_values))


def transpose_packed(packed: torch.Tensor, logical_cols: int) -> torch.Tensor:
    """Transpose a packed signed-int4 CUDA matrix without unpacking to int8."""

    return _load_extension().transpose_packed(packed.contiguous(), int(logical_cols))


def wmma_int4_mm(a_packed: torch.Tensor, b_t_packed: torch.Tensor, k: int) -> torch.Tensor:
    """Compute ``a @ b_t.T`` with CUDA WMMA signed-int4 tensor cores."""

    return _load_extension().wmma_int4_mm(a_packed.contiguous(), b_t_packed.contiguous(), int(k))


def dequant_epilogue(
    acc: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply row/column scales and optional bias to int32 accumulators."""

    return _load_extension().dequant_epilogue(
        acc.contiguous(),
        row_scale.reshape(-1).contiguous(),
        col_scale.reshape(-1).contiguous() if col_scale is not None else None,
        bias.contiguous() if bias is not None else None,
        _dtype_code(out_dtype),
    )


def linear_forward(
    qx: torch.Tensor,
    qw: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run packed int4 activation x packed int4 weight matmul and dequantize."""

    return _load_extension().linear_forward(
        qx.contiguous(),
        qw.contiguous(),
        row_scale.reshape(-1).contiguous(),
        col_scale.reshape(-1).contiguous(),
        bias.contiguous() if bias is not None else None,
        _dtype_code(out_dtype),
    )


def linear_backward_input(
    qg: torch.Tensor,
    qw: torch.Tensor,
    row_scale: torch.Tensor,
    padded_features: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run packed int4 grad-output x packed int4 weight matmul for input grads."""

    return _load_extension().linear_backward_input(
        qg.contiguous(),
        qw.contiguous(),
        row_scale.reshape(-1).contiguous(),
        int(padded_features),
        _dtype_code(out_dtype),
    )


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"CUDA INT4 ConvRot kernels do not support output dtype {dtype}")
