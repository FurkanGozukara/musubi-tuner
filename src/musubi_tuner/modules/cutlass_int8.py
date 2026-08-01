"""Optional CUTLASS-backed int8 GEMM for W8A8 linear layers.

The extension is built lazily with torch.utils.cpp_extension and is intentionally
not required for normal imports. Set LTX2_CUTLASS_INCLUDE_DIR or
LTX2_CUTLASS_INCLUDE_DIRS to the CUTLASS include root when it is not installed
in a standard location.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_EXTENSION = None
_LOAD_ERROR: Exception | None = None


def _split_env_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _candidate_include_dirs() -> list[str]:
    candidates: list[Path] = []
    candidates.extend(_split_env_paths(os.getenv("LTX2_CUTLASS_INCLUDE_DIRS")))
    candidates.extend(_split_env_paths(os.getenv("LTX2_CUTLASS_INCLUDE_DIR")))
    candidates.extend(
        [
            Path("/usr/local/cutlass"),
            Path("/usr/local/include"),
            Path("/usr/local/cuda/include"),
        ]
    )

    include_dirs: list[Path] = []
    for candidate in candidates:
        if (candidate / "cutlass").is_dir():
            include_dirs.append(candidate)
        if (candidate / "include" / "cutlass").is_dir():
            include_dirs.append(candidate / "include")
        if (candidate / "tools" / "util" / "include").is_dir():
            include_dirs.append(candidate / "tools" / "util" / "include")
        if (candidate.parent / "tools" / "util" / "include").is_dir():
            include_dirs.append(candidate.parent / "tools" / "util" / "include")

    deduped: list[str] = []
    seen: set[str] = set()
    for path in include_dirs:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


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

    source_dir = Path(__file__).resolve().parent / "cutlass_int8_ext"
    include_dirs = _candidate_include_dirs()
    if not include_dirs:
        _LOAD_ERROR = RuntimeError("CUTLASS headers were not found. Set LTX2_CUTLASS_INCLUDE_DIR to the CUTLASS include directory.")
        raise _LOAD_ERROR

    try:
        _EXTENSION = load(
            name="ltx2_cutlass_int8_gemm",
            sources=[str(source_dir / "binding.cpp"), str(source_dir / "cutlass_int8_gemm.cu")],
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
        logger.debug("CUTLASS int8 extension unavailable: %s", exc)
        return False
    return True


def int8_mm(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b_t.T`` for contiguous CUDA int8 matrices and return int32."""
    return _load_extension().int8_mm(a, b_t)


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"CUTLASS scaled int8 GEMM does not support output dtype {dtype}")


def int8_mm_scaled(
    a: torch.Tensor,
    b_t: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``a @ b_t.T`` and apply row/column scales before returning."""
    return _load_extension().int8_mm_scaled(a, b_t, row_scale, col_scale, bias, _dtype_code(output_dtype))


int8_mm.scaled = int8_mm_scaled  # type: ignore[attr-defined]
