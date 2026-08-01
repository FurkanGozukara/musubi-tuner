"""Optional fused CUTLASS int4 GEMM with an in-epilogue dequant (Epilogue Visitor Tree).

This is the fused sibling of :mod:`cutlass_int4`. The unfused path materializes the
int32 ``[M, N]`` accumulator in global memory and reads it back in a separate dequant
kernel; here the per-row activation scale, per-column weight scale, optional bias, and
the int32->output cast are folded into an Sm80 EVT so the accumulator never leaves the
tensor-core epilogue. Built lazily; only required when the W4A4 CUDA fusion is enabled
via ``LTX2_INT4_CONVROT_FUSE_CUDA=1``.
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

    source_dir = Path(__file__).resolve().parent / "cutlass_int4_fused_ext"
    include_dirs = _candidate_include_dirs()
    if not include_dirs:
        _LOAD_ERROR = RuntimeError("CUTLASS headers were not found. Set LTX2_CUTLASS_INCLUDE_DIR to the CUTLASS include directory.")
        raise _LOAD_ERROR
    # The EVT visitor path transitively includes CUTLASS 3.8 SM100 (Blackwell) kernel
    # headers that fail to compile under nvcc + MSVC (constexpr dim3). Prepend a shim
    # dir of empty stubs that shadow those four headers; the aggregator never names the
    # SM100 types, and we only instantiate a 2.x kernel, so this is a no-op elsewhere.
    shim_dir = source_dir / "cutlass_shim"
    include_dirs = [str(shim_dir), *include_dirs]

    try:
        _EXTENSION = load(
            name="ltx2_cutlass_int4_fused_gemm",
            sources=[str(source_dir / "binding.cpp"), str(source_dir / "cutlass_int4_fused_gemm.cu")],
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
        logger.debug("Fused CUTLASS int4 extension unavailable: %s", exc)
        return False
    return True


def linear_fused(
    a_packed: torch.Tensor,
    b_t_packed: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    k: int,
) -> torch.Tensor:
    """Fused packed int4 GEMM ``out = row_scale * col_scale * (a @ b_t.T) + bias``.

    ``a_packed`` is ``[M, ceil(k / 2)]`` and ``b_t_packed`` is ``[N, ceil(k / 2)]``.
    ``row_scale`` is per-row (``[M]``, activation scale); ``col_scale`` is per-column
    (``[N]``, weight scale) or ``None`` (identity); ``bias`` is ``[N]`` or ``None``.
    """

    return _load_extension().linear_fused(
        a_packed.contiguous(),
        b_t_packed.contiguous(),
        row_scale.reshape(-1).contiguous(),
        col_scale.reshape(-1).contiguous() if col_scale is not None else None,
        bias.reshape(-1).contiguous() if bias is not None else None,
        _dtype_code(out_dtype),
        int(k),
    )


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"Fused CUTLASS INT4 kernels do not support output dtype {dtype}")
