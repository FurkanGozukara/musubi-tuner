"""Optional CUTLASS-backed native int4 GEMM for W4A4 linear layers.

The extension consumes the repository's signed low/high nibble packing directly
as CUTLASS ``int4b_t`` operands. It is built lazily and is only required when
``LTX2_INT4_CONVROT_BACKEND=cutlass`` is selected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from musubi_tuner.modules.cutlass_int8 import _candidate_include_dirs

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

    source_dir = Path(__file__).resolve().parent / "cutlass_int4_ext"
    include_dirs = _candidate_include_dirs()
    if not include_dirs:
        _LOAD_ERROR = RuntimeError("CUTLASS headers were not found. Set LTX2_CUTLASS_INCLUDE_DIR to the CUTLASS include directory.")
        raise _LOAD_ERROR

    try:
        _EXTENSION = load(
            name="ltx2_cutlass_int4_gemm",
            sources=[str(source_dir / "binding.cpp"), str(source_dir / "cutlass_int4_gemm.cu")],
            extra_include_paths=include_dirs,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            with_cuda=True,
            verbose=os.getenv("LTX2_CUTLASS_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:  # pragma: no cover - build environment dependent
        _LOAD_ERROR = exc
        raise
    return _EXTENSION


def is_available() -> bool:
    try:
        _load_extension()
    except Exception as exc:
        logger.debug("CUTLASS int4 extension unavailable: %s", exc)
        return False
    return True


def int4_mm(a_packed: torch.Tensor, b_t_packed: torch.Tensor, k: int) -> torch.Tensor:
    """Compute ``a @ b_t.T`` from packed signed-int4 CUDA matrices.

    ``a_packed`` is shaped ``[M, ceil(K / 2)]`` and ``b_t_packed`` is shaped
    ``[N, ceil(K / 2)]``. The returned tensor is int32 ``[M, N]``.
    """

    return _load_extension().int4_mm(a_packed.contiguous(), b_t_packed.contiguous(), int(k))


def linear(
    a_packed: torch.Tensor,
    b_t_packed: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    k: int,
) -> torch.Tensor:
    """Compute packed int4 GEMM and return scaled floating output."""

    return _load_extension().linear(
        a_packed.contiguous(),
        b_t_packed.contiguous(),
        row_scale.reshape(-1).contiguous(),
        col_scale.reshape(-1).contiguous() if col_scale is not None else None,
        bias.contiguous() if bias is not None else None,
        _dtype_code(out_dtype),
        int(k),
    )


def linear_backward_input(
    qg_packed: torch.Tensor,
    packed_weight: torch.Tensor,
    row_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
    padded_features: int,
) -> torch.Tensor:
    """Compute INT4 grad-input contraction with native packed transpose."""

    return _load_extension().linear_backward_input(
        qg_packed.contiguous(),
        packed_weight.contiguous(),
        row_scale.reshape(-1).contiguous(),
        _dtype_code(out_dtype),
        int(out_features),
        int(padded_features),
    )


def transpose_packed(packed: torch.Tensor, logical_cols: int) -> torch.Tensor:
    """Transpose a packed signed-int4 row-major matrix without unpacking to int8."""

    return _load_extension().transpose_packed(packed.contiguous(), int(logical_cols))


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"CUTLASS INT4 kernels do not support output dtype {dtype}")
