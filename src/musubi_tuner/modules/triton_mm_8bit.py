"""Triton int8 matmul used by W8A8 backward."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
    ],
    key=["QUANTIZED_M", "N", "K", "stride_bk"],
)
@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANTIZED_M,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
    ],
    key=["QUANTIZED_M", "N", "K", "stride_bk", "HAS_COL", "HAS_BIAS"],
)
@triton.jit
def _mm_scaled_kernel(
    a_ptr,
    b_ptr,
    row_scale_ptr,
    col_scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANTIZED_M,
    HAS_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    out = accumulator.to(tl.float32) * tl.load(row_scale_ptr + offs_am, mask=offs_am < M, other=0.0)[:, None]
    if HAS_COL:
        out = out * tl.load(col_scale_ptr + offs_bn, mask=offs_bn < N, other=0.0)[None, :]
    if HAS_BIAS:
        out = out + tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)[None, :]

    out_ptrs = out_ptr + stride_om * offs_am[:, None] + stride_on * offs_bn[None, :]
    out_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=out_mask)


def mm_8bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return int32 matmul for int8 matrices where A is contiguous."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions: {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_contiguous():
        raise ValueError("Matrix A must be contiguous")
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: {a.dtype} and {b.dtype}")
    if a.dtype != torch.int8:
        raise ValueError(f"Unsupported dtype: {a.dtype}")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.int32)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE_N"]), triton.cdiv(m, meta["BLOCK_SIZE_M"]))

    _mm_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        QUANTIZED_M=m // 64,
    )
    return c


def mm_8bit_scaled(
    a: torch.Tensor,
    b: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Return dequantized int8 matmul: ``a @ b * row_scale * col_scale + bias``."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions: {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_contiguous():
        raise ValueError("Matrix A must be contiguous")
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: {a.dtype} and {b.dtype}")
    if a.dtype != torch.int8:
        raise ValueError(f"Unsupported dtype: {a.dtype}")

    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    row_scale = row_scale.reshape(m).contiguous()
    col_scale_arg = col_scale.reshape(n).contiguous() if col_scale is not None else row_scale
    bias_arg = bias.contiguous() if bias is not None else row_scale

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE_N"]), triton.cdiv(m, meta["BLOCK_SIZE_M"]))

    _mm_scaled_kernel[grid](
        a,
        b,
        row_scale,
        col_scale_arg,
        bias_arg,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        QUANTIZED_M=m // 64,
        HAS_COL=col_scale is not None,
        HAS_BIAS=bias is not None,
    )
    return out


mm_8bit.scaled = mm_8bit_scaled  # type: ignore[attr-defined]
