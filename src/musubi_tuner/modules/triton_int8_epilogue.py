"""Fused dequant epilogue for int8 W8A8 matmuls.

The eager rescale ``out_int32.float() * x_scale * weight_scale + bias`` makes
several full passes over the [M, N] output (int32->fp32, two scale multiplies,
optional bias, cast to bf16). This Triton kernel does it in one pass: read int32
once, apply the per-row activation scale, the optional per-column weight scale and
bias, write the compute dtype once. Keeps the matmul on cuBLAS ``torch._int_mm``.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton optional
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _dequant_epilogue_kernel(
        acc_ptr,
        xs_ptr,
        ws_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        HAS_COL: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        base = pid_m * N + offs_n
        out = tl.load(acc_ptr + base, mask=mask, other=0).to(tl.float32)
        out = out * tl.load(xs_ptr + pid_m)
        if HAS_COL:
            out = out * tl.load(ws_ptr + offs_n, mask=mask, other=0.0)
        if HAS_BIAS:
            out = out + tl.load(bias_ptr + offs_n, mask=mask, other=0.0)
        tl.store(out_ptr + base, out.to(out_ptr.dtype.element_ty), mask=mask)


def have_triton() -> bool:
    return _HAVE_TRITON


def dequant_epilogue(out_int32, row_scale, col_scale, bias, out_dtype):
    """Fused: out_int32[M,N] * row_scale[M,1] * (col_scale[N] if given) (+ bias[N]) -> out_dtype.

    ``col_scale`` / ``bias`` may be None (e.g. the backward grad_input rescale folds
    the weight scale before the matmul and has no bias).
    """
    out_int32 = out_int32.contiguous()
    M, N = out_int32.shape
    out = torch.empty((M, N), device=out_int32.device, dtype=out_dtype)
    grid = (M, triton.cdiv(N, 512))
    _dequant_epilogue_kernel[grid](
        out_int32,
        row_scale.reshape(M).contiguous(),
        (col_scale.reshape(N).contiguous() if col_scale is not None else out_int32),
        (bias.contiguous() if bias is not None else out_int32),
        out,
        M,
        N,
        HAS_COL=col_scale is not None,
        HAS_BIAS=bias is not None,
        BLOCK_N=512,
    )
    return out
