"""Optional fused qk-norm + RoPE CUDA kernel.

Wraps a fused RMSNorm + RoPE CUDA kernel as the forward of a
:class:`torch.autograd.Function`. Both RoPE conventions are covered: the interleaved
(GPT-J pairwise) layout and the split (GPT-NeoX half-split) layout used by the
LTX-2.3 22B model. The backward is a closed-form analytical gradient (elementwise
math plus one reduction per row, no forward recompute): the rotation is inverted by
its transpose, the affine grad is a row reduction, and the RMSNorm grad uses the
standard mean-free formula. The extension is built with CUDA fast math and has no
documented numerical- or training-equivalence guarantee relative to eager execution.

Opt-in via ``LTX2_FUSED_NORM_ROPE=1`` (default OFF). When the flag is unset the
caller retains the existing eager path. When the flag is set but tensors are ineligible,
the process is compiling, or the CUDA extension cannot be built, the caller falls back
to eager (with a one-time warning for extension-load failure).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_EXTENSION = None
_LOAD_ERROR: Exception | None = None
_WARNED_UNAVAILABLE = False
_LOGGED_ENGAGED = False

_MAX_FUSED_DIM = 32768


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

    source_dir = Path(__file__).resolve().parent / "cuda_fused_norm_rope_ext"
    try:
        _EXTENSION = load(
            name="ltx2_cuda_fused_norm_rope",
            sources=[
                str(source_dir / "binding.cpp"),
                str(source_dir / "rms_norm_rope.cpp"),
                str(source_dir / "rms_norm_rope_cuda.cu"),
                str(source_dir / "rms_norm_split_rope.cpp"),
                str(source_dir / "rms_norm_split_rope_cuda.cu"),
            ],
            extra_include_paths=[str(source_dir)],
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
    """True if the fused CUDA extension is (or can be) built on this machine."""
    try:
        _load_extension()
    except Exception as exc:
        logger.debug("CUDA fused norm+rope extension unavailable: %s", exc)
        return False
    return True


def is_enabled() -> bool:
    """Opt-in gate. Reads ``LTX2_FUSED_NORM_ROPE`` live so it can be toggled per run."""
    return os.getenv("LTX2_FUSED_NORM_ROPE", "0").strip().lower() in ("1", "true", "yes", "on")


def _warn_unavailable_once() -> None:
    global _WARNED_UNAVAILABLE
    if not _WARNED_UNAVAILABLE:
        _WARNED_UNAVAILABLE = True
        err = _LOAD_ERROR
        logger.warning(
            "LTX2_FUSED_NORM_ROPE=1 but the fused CUDA kernel is unavailable (%s); using the eager qk-norm+rope path.",
            err,
        )


def _log_engaged_once(kind: str) -> None:
    global _LOGGED_ENGAGED
    if not _LOGGED_ENGAGED:
        _LOGGED_ENGAGED = True
        logger.info("LTX2_FUSED_NORM_ROPE=1: using fused qk-norm + %s RoPE kernel", kind)


def _eager_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Exact eager math: fp32 RMSNorm island + interleaved rotation.

    Mirrors ``Attention.forward``: ``q_norm(x)`` followed by
    ``apply_rotary_emb(x, (cos, sin), INTERLEAVED)``. Used for the recompute-based
    backward so gradients equal the eager autograd grads.
    """
    from musubi_tuner.ltx_2.model.transformer.rope import apply_interleaved_rotary_emb
    from musubi_tuner.ltx_2.utils import rms_norm

    normed = rms_norm(x, weight, eps)
    return apply_interleaved_rotary_emb(normed, cos, sin)


def _split_eager_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Exact eager math: fp32 RMSNorm island + split (half-split) rotation.

    Mirrors ``Attention.forward`` for the split ``rope_type``: ``q_norm(x)`` followed
    by ``apply_rotary_emb(x, (cos, sin), SPLIT)``. ``x`` is ``[B, T, inner_dim]`` and
    cos/sin are ``[B, H, T, head_dim//2]``.
    """
    from musubi_tuner.ltx_2.model.transformer.rope import apply_split_rotary_emb
    from musubi_tuner.ltx_2.utils import rms_norm

    normed = rms_norm(x, weight, eps)
    return apply_split_rotary_emb(normed, cos, sin)


def _rope_grad_input(grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Gradient of interleaved RoPE w.r.t. its input (the adjoint of the forward map).

    Forward RoPE is the linear map ``a -> a*cos + rot(a)*sin`` with
    ``rot(a) = [-a1, a0, -a3, a2, ...]`` per ``(2i, 2i+1)`` pair. Its adjoint is
    ``gy*cos - rot(gy*sin)``, which is exact for arbitrary cos/sin layouts (no
    per-pair orthogonality assumed). For the model's cos/sin (``repeat_interleave``
    of a shared per-pair angle) the map is orthonormal, so the adjoint equals the
    inverse rotation.
    """
    gys = (grad_out * sin).unflatten(-1, (grad_out.shape[-1] // 2, 2))
    rot_gys = torch.stack((-gys[..., 1], gys[..., 0]), dim=-1).flatten(-2)
    return grad_out * cos - rot_gys


def _split_rope_grad_input(grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Gradient of split (half-split) RoPE w.r.t. its input (the adjoint of the forward map).

    Forward split RoPE rotates the head-split halves ``(first, second)`` by
    ``out_first = first*cos - second*sin`` and ``out_second = second*cos + first*sin``.
    Its adjoint maps an upstream grad ``(g_first, g_second)`` to
    ``d_first = g_first*cos + g_second*sin`` and ``d_second = -g_first*sin + g_second*cos``.
    That is exactly the forward map with ``sin`` negated, so we reuse
    ``apply_split_rotary_emb`` to guarantee the identical per-head reshape/swapaxes the
    forward used. For orthonormal cos/sin the map is orthogonal, so the adjoint equals
    the inverse rotation.
    """
    from musubi_tuner.ltx_2.model.transformer.rope import apply_split_rotary_emb

    return apply_split_rotary_emb(grad_out, cos, -sin)


def _rmsnorm_affine_backward(
    da: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    needs_x: bool,
    needs_w: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Closed-form RMSNorm(+affine) backward given ``da`` = grad w.r.t. the rope input.

    ``da`` is the gradient w.r.t. the affine-normalized activation ``normed*weight``
    (i.e. the rope adjoint already applied). Returns ``(grad_x, grad_weight)`` using the
    standard mean-free RMSNorm formula. Runs in fp32; casts back to the input dtypes.
    """
    dim = x.shape[-1]
    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)

    gx = None
    gw = None

    if weight is not None and needs_w:
        n = xf * inv_rms  # normalized input
        reduce_dims = tuple(range(da.ndim - 1))
        gw = (da * n).sum(dim=reduce_dims).to(weight.dtype)

    if needs_x:
        g_n = da * weight.float() if weight is not None else da
        row_dot = (g_n * xf).sum(dim=-1, keepdim=True)
        dx = inv_rms * g_n - (inv_rms.pow(3) * xf / dim) * row_dot
        gx = dx.to(x.dtype)

    return gx, gw


class _FusedNormRopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):  # noqa: ANN001
        ext = _load_extension()
        cos_k = cos.expand(x.shape).contiguous().to(torch.bfloat16)
        sin_k = sin.expand(x.shape).contiguous().to(torch.bfloat16)
        w_k = weight.to(torch.bfloat16).contiguous() if weight is not None else None
        out = ext.rms_norm_rope(x.contiguous(), w_k, cos_k, sin_k, True)
        ctx.save_for_backward(x, weight, cos, sin)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: ANN001
        x, weight, cos, sin = ctx.saved_tensors
        needs_x, needs_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]

        # d(a) is the adjoint of the interleaved rotation applied to the upstream grad.
        da = _rope_grad_input(grad_out.float(), cos.float(), sin.float())
        gx, gw = _rmsnorm_affine_backward(da, x, weight, ctx.eps, needs_x, needs_w)
        return gx, gw, None, None, None


class _FusedNormSplitRopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):  # noqa: ANN001
        ext = _load_extension()
        # cos/sin are [B, H, T, D//2]; the kernel reads them with their own strides,
        # so no expand/contiguous is needed (a swapaxes view keeps inner stride 1).
        cos_k = cos.to(torch.bfloat16)
        sin_k = sin.to(torch.bfloat16)
        w_k = weight.to(torch.bfloat16).contiguous()
        # Binding order is (x, sin, cos, weights, out_fp8); out_fp8=False -> bf16 out.
        out = ext.rms_norm_split_rope(x.contiguous(), sin_k, cos_k, w_k, False)
        ctx.save_for_backward(x, weight, cos, sin)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: ANN001
        x, weight, cos, sin = ctx.saved_tensors
        needs_x, needs_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]

        # d(a) is the adjoint of the split rotation applied to the upstream grad.
        da = _split_rope_grad_input(grad_out.float(), cos.float(), sin.float())
        gx, gw = _rmsnorm_affine_backward(da, x, weight, ctx.eps, needs_x, needs_w)
        return gx, gw, None, None, None


def fused_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm(+affine) then interleaved RoPE on ``x`` (bf16 in, bf16 out)."""
    return _FusedNormRopeFn.apply(x, weight, cos, sin, eps)


def fused_norm_split_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm(+affine) then split (half-split) RoPE on ``x`` (bf16 in, bf16 out)."""
    return _FusedNormSplitRopeFn.apply(x, weight, cos, sin, eps)


def _is_eligible(x: object) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 8 == 0
        and x.shape[-1] <= _MAX_FUSED_DIM
    )


def _freqs_ok(cos: object, sin: object, x: torch.Tensor) -> bool:
    """Interleaved freq check: cos/sin match the full per-token hidden dim."""
    return (
        isinstance(cos, torch.Tensor)
        and isinstance(sin, torch.Tensor)
        and cos.shape[-1] == x.shape[-1]
        and sin.shape[-1] == x.shape[-1]
    )


def _freqs_ok_split(cos: object, sin: object, x: torch.Tensor) -> bool:
    """Split freq check for cos/sin shaped ``[B, H, T, head_dim//2]``.

    Validates the layout the split kernel assumes: 4D per-head freqs whose batch and
    sequence axes match ``x`` (shape ``[B, T, inner_dim]``), and a head split that
    satisfies the kernel's per-head thread tiling (``head_dim`` a multiple of 16 and
    warp-aligned). Any mismatch falls back to eager.
    """
    if not (isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor)):
        return False
    if cos.ndim != 4 or sin.ndim != 4 or cos.shape != sin.shape:
        return False
    if x.ndim != 3:
        return False
    b, n, t, d2 = cos.shape
    inner = x.shape[-1]
    if b != x.shape[0] or t != x.shape[1]:
        return False
    if n <= 0 or n * d2 * 2 != inner:
        return False
    if inner // 8 > 1024:  # kernel launches inner/8 threads per block
        return False
    head_dim = inner // n
    threads_per_head = head_dim // 8
    if head_dim % 16 != 0 or head_dim > 256:
        return False
    if threads_per_head == 0 or 32 % threads_per_head != 0:
        return False
    return True


def _rope_kind(rope_type: object) -> str:
    """Map an ``LTXRopeType`` (or its value) to ``'interleaved'`` / ``'split'`` / ''."""
    val = getattr(rope_type, "value", rope_type)
    if val in ("interleaved", "split"):
        return val
    return ""


def try_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    pe: tuple[torch.Tensor, torch.Tensor],
    k_pe: tuple[torch.Tensor, torch.Tensor] | None,
    eps: float,
    rope_type: object,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse q/k RMSNorm + RoPE, or return ``None`` to fall back to eager.

    Handles both the interleaved and split (LTX-2.3) RoPE conventions, dispatched on
    ``rope_type``. Callers should gate on :func:`is_enabled` and a non-None ``pe``.
    Returns ``None`` whenever the kernel is unavailable or the tensors are not
    eligible, so the caller uses its existing eager implementation.
    """
    # PyBind extension calls cannot be captured by Dynamo in a full graph and
    # introduce a graph break otherwise. Let the existing eager tensor path be
    # compiled instead; the fused CUDA path remains active outside torch.compile.
    if torch.compiler.is_compiling():
        return None

    kind = _rope_kind(rope_type)
    if kind == "":
        return None
    if not (_is_eligible(q) and _is_eligible(k)):
        return None
    cos_q, sin_q = pe
    cos_k, sin_k = k_pe if k_pe is not None else pe

    if kind == "split":
        # The split kernel applies the affine weight unconditionally; require it.
        if q_weight is None or k_weight is None:
            return None
        if not (_freqs_ok_split(cos_q, sin_q, q) and _freqs_ok_split(cos_k, sin_k, k)):
            return None
        if not is_available():
            _warn_unavailable_once()
            return None
        _log_engaged_once("split")
        q_out = fused_norm_split_rope(q, q_weight, cos_q, sin_q, eps)
        k_out = fused_norm_split_rope(k, k_weight, cos_k, sin_k, eps)
        return q_out, k_out

    if not (_freqs_ok(cos_q, sin_q, q) and _freqs_ok(cos_k, sin_k, k)):
        return None
    if not is_available():
        _warn_unavailable_once()
        return None
    _log_engaged_once("interleaved")
    q_out = fused_norm_rope(q, q_weight, cos_q, sin_q, eps)
    k_out = fused_norm_rope(k, k_weight, cos_k, sin_k, eps)
    return q_out, k_out
