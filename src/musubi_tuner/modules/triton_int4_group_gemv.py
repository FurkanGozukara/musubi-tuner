"""Small-M fused GEMV for packed INT4 ConvRot group scales.

The video path normally uses large activation matrices and the native
unpack-to-INT8 plus ``torch._int_mm`` backend. Decode-like calls with at most
16 rows are instead bandwidth/launch bound. This kernel quantizes activations
internally, reads packed INT4 weights directly, applies group ratios in
registers, accumulates with INT8 tensor-core dots, and writes the scaled output
in one launch without materializing an INT8 weight matrix. Its fast Triton
activation quantizer is intentionally not bit-exact with the eager PyTorch
fallback.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - Triton is optional
    _HAVE_TRITON = False


MAX_FUSED_GEMV_ROWS = 16
_FUSED_BLOCK_N = 16
_FUSED_NUM_WARPS = 8


if _HAVE_TRITON:

    @triton.jit
    def _decode_group_lane(word, ratio, lane: tl.constexpr):
        nibble = (word >> (4 * lane)) & 0x0F
        code = (nibble ^ 0x08) - 0x08
        mapped = libdevice.rint(code.to(tl.float32) * ratio)
        return tl.minimum(tl.maximum(mapped, -127.0), 127.0).to(tl.int8)

    @triton.jit
    def _int4_group_gemv_kernel(
        x_ptr,
        w_ptr,
        ratio_ptr,
        weight_scale_ptr,
        bias_ptr,
        out_ptr,
        M,
        K,
        N,
        packed_stride,
        groups_per_row,
        scale_group_size,
        HAS_BIAS: tl.constexpr,
        RATIO_Q8: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_m = tl.arange(0, 16)
        mask_m = offs_m < M

        amax = tl.zeros((16,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
        activation_scale = tl.maximum(amax / 127.0, 1.0e-30)

        acc = tl.zeros((16, BLOCK_N), tl.int32)
        for k0 in range(0, K, BLOCK_K):
            offs_word = k0 // 8 + tl.arange(0, BLOCK_K // 8)
            mask_word = offs_word * 8 < K
            word = tl.zeros((BLOCK_N, BLOCK_K // 8), tl.int32)
            for byte_lane in tl.static_range(4):
                byte = tl.load(
                    w_ptr + offs_n[:, None] * packed_stride + offs_word[None, :] * 4 + byte_lane,
                    mask=mask_n[:, None] & mask_word[None, :],
                    other=0,
                ).to(tl.int32)
                word += byte << (8 * byte_lane)
            ratio_word = tl.load(
                ratio_ptr + offs_n[:, None] * groups_per_row + ((offs_word * 8) // scale_group_size)[None, :],
                mask=mask_n[:, None] & mask_word[None, :],
                other=1.0,
            )
            if RATIO_Q8:
                ratio_word = ratio_word.to(tl.float32) * (1.0 / 256.0)
            c0 = _decode_group_lane(word, ratio_word, 0)
            c1 = _decode_group_lane(word, ratio_word, 1)
            c2 = _decode_group_lane(word, ratio_word, 2)
            c3 = _decode_group_lane(word, ratio_word, 3)
            c4 = _decode_group_lane(word, ratio_word, 4)
            c5 = _decode_group_lane(word, ratio_word, 5)
            c6 = _decode_group_lane(word, ratio_word, 6)
            c7 = _decode_group_lane(word, ratio_word, 7)
            even = tl.interleave(tl.interleave(c0, c4), tl.interleave(c2, c6))
            odd = tl.interleave(tl.interleave(c1, c5), tl.interleave(c3, c7))
            weight = tl.interleave(even, odd)

            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            activation = libdevice.rint(x / activation_scale[:, None])
            activation = tl.minimum(tl.maximum(activation, -127.0), 127.0).to(tl.int8)
            acc = tl.dot(activation, tl.trans(weight), acc, out_dtype=tl.int32)

        weight_scale = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=0.0)
        out = acc.to(tl.float32) * activation_scale[:, None]
        out = out * weight_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            out += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def have_triton() -> bool:
    return _HAVE_TRITON


def int4_group_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    group_scale_ratio: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    scale_group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return the fused result, or ``None`` when the shape/backend is ineligible."""

    if not _HAVE_TRITON or not x.is_cuda or x.ndim != 2:
        return None
    rows, features = x.shape
    out_features = packed_weight.shape[0]
    scale_group_size = int(scale_group_size)
    if (
        rows < 1
        or rows > MAX_FUSED_GEMV_ROWS
        or packed_weight.dtype != torch.uint8
        or packed_weight.device != x.device
        or group_scale_ratio.device != x.device
        or group_scale_ratio.dtype not in (torch.float32, torch.int16)
        or weight_scale.device != x.device
        or (bias is not None and bias.device != x.device)
        or features % 256
        or scale_group_size < 8
        or scale_group_size % 8
        or features % scale_group_size
    ):
        return None
    groups_per_row = features // scale_group_size
    if tuple(group_scale_ratio.shape) != (out_features, groups_per_row):
        return None

    x = x.contiguous()
    packed_weight = packed_weight.contiguous()
    group_scale_ratio = group_scale_ratio.contiguous()
    weight_scale = weight_scale.float().reshape(out_features).contiguous()
    bias_arg = bias.contiguous() if bias is not None else weight_scale
    out = torch.empty((rows, out_features), device=x.device, dtype=out_dtype)
    _int4_group_gemv_kernel[(triton.cdiv(out_features, _FUSED_BLOCK_N),)](
        x,
        packed_weight,
        group_scale_ratio,
        weight_scale,
        bias_arg,
        out,
        rows,
        features,
        out_features,
        packed_weight.shape[1],
        groups_per_row,
        scale_group_size,
        HAS_BIAS=bias is not None,
        RATIO_Q8=group_scale_ratio.dtype == torch.int16,
        BLOCK_N=_FUSED_BLOCK_N,
        BLOCK_K=256,
        num_warps=_FUSED_NUM_WARPS,
        num_stages=2,
    )
    return out
