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
    from triton.language.extra import libdevice

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton optional
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _convrot_hadamard4_stage(y, BLOCK: tl.constexpr, GROUP: tl.constexpr, STEP: tl.constexpr):
        z = tl.reshape(y, (BLOCK // GROUP, GROUP // (4 * STEP), 4, STEP))
        z = tl.permute(z, (0, 1, 3, 2))
        z = tl.reshape(z, (BLOCK // GROUP, GROUP // (4 * STEP), STEP, 2, 2))
        ab, cd = tl.split(z)
        a, b = tl.split(ab)
        c, d = tl.split(cd)
        o0 = a + b + c - d
        o1 = a + b - c + d
        o2 = a - b + c + d
        o3 = -a + b + c + d
        left = tl.join(o0, o1)
        right = tl.join(o2, o3)
        out = tl.join(left, right)
        out = tl.reshape(out, (BLOCK // GROUP, GROUP // (4 * STEP), STEP, 4))
        out = tl.permute(out, (0, 1, 3, 2))
        return tl.reshape(out, (BLOCK // GROUP, GROUP))

    @triton.jit
    def _convrot_hadamard4(y, BLOCK: tl.constexpr, GROUP: tl.constexpr):
        y = tl.reshape(y, (BLOCK // GROUP, GROUP))
        if GROUP >= 4:
            y = _convrot_hadamard4_stage(y, BLOCK, GROUP, 1)
        if GROUP >= 16:
            y = _convrot_hadamard4_stage(y, BLOCK, GROUP, 4)
        if GROUP >= 64:
            y = _convrot_hadamard4_stage(y, BLOCK, GROUP, 16)
        if GROUP >= 256:
            y = _convrot_hadamard4_stage(y, BLOCK, GROUP, 64)
        return tl.reshape(y, (BLOCK,)) / tl.sqrt(GROUP + 0.0)

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

    @triton.jit
    def _quantize_rowwise_kernel(
        x_ptr,
        cs_ptr,
        q_ptr,
        s_ptr,
        N,
        HAS_COL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_COL:
            x = x * tl.load(cs_ptr + offs, mask=mask, other=0.0)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-30)
        q = tl.minimum(tl.maximum(libdevice.rint(x / scale), -127.0), 127.0)
        tl.store(q_ptr + row * N + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _quantize_rowwise_convrot_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        N,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        x = _convrot_hadamard4(x, BLOCK, GROUP)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-30)
        q = tl.minimum(tl.maximum(libdevice.rint(x / scale), -127.0), 127.0)
        tl.store(q_ptr + row * N + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _dequant_epilogue_convrot_kernel(
        acc_ptr,
        xs_ptr,
        out_ptr,
        N,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        out = tl.load(acc_ptr + row * N + offs, mask=mask, other=0).to(tl.float32)
        out = out * tl.load(xs_ptr + row)
        out = _convrot_hadamard4(out, BLOCK, GROUP)
        tl.store(out_ptr + row * N + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def have_triton() -> bool:
    return _HAVE_TRITON


def quantize_rowwise(x, col_scale=None, *, convrot_groupsize=0):
    """Fused per-token int8 quant in one pass; optionally folds a per-column scale
    (``col_scale[N]``) before quantizing (used by the backward to absorb the weight
    scale). Returns (int8 [M,N], fp32 scale [M,1]).

    ``convrot_groupsize`` applies the ConvRot Hadamard activation rotation before
    quantizing. It is only valid for forward activations, so it cannot be
    combined with ``col_scale``.
    """
    x = x.contiguous()
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    convrot_groupsize = int(convrot_groupsize or 0)
    if convrot_groupsize > 0:
        if col_scale is not None:
            raise ValueError("ConvRot fused quantization cannot be combined with col_scale")
        if N % convrot_groupsize != 0:
            raise ValueError(f"ConvRot group size {convrot_groupsize} does not divide N={N}")
        if convrot_groupsize > 256:
            raise ValueError(f"ConvRot fused quantization supports group sizes up to 256, got {convrot_groupsize}")
        _quantize_rowwise_convrot_kernel[(M,)](
            x,
            q,
            s,
            N,
            GROUP=convrot_groupsize,
            BLOCK=triton.next_power_of_2(N),
        )
        return q, s
    _quantize_rowwise_kernel[(M,)](
        x,
        col_scale.reshape(N).contiguous() if col_scale is not None else x,
        q,
        s,
        N,
        HAS_COL=col_scale is not None,
        BLOCK=triton.next_power_of_2(N),
    )
    return q, s


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


def dequant_epilogue_convrot(out_int32, row_scale, group_size, out_dtype):
    """Fused backward epilogue for ConvRot grad_input.

    Applies ``out_int32[M,N] * row_scale[M,1]`` and the inverse ConvRot
    Hadamard in one pass. The Hadamard is symmetric, so the inverse transform is
    the same as the forward activation rotation.
    """
    out_int32 = out_int32.contiguous()
    M, N = out_int32.shape
    group_size = int(group_size)
    if N % group_size != 0:
        raise ValueError(f"ConvRot group size {group_size} does not divide N={N}")
    if group_size > 256:
        raise ValueError(f"ConvRot fused epilogue supports group sizes up to 256, got {group_size}")
    out = torch.empty((M, N), device=out_int32.device, dtype=out_dtype)
    _dequant_epilogue_convrot_kernel[(M,)](
        out_int32,
        row_scale.reshape(M).contiguous(),
        out,
        N,
        GROUP=group_size,
        BLOCK=triton.next_power_of_2(N),
    )
    return out
