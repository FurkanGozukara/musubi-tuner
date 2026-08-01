"""Int8 weight-only quantized full fine-tuning for LTX-2.

Trainable weights are stored as int8 (1 byte/param, no bf16 master); the optimizer
updates them in place with stochastic rounding, since a small bf16 update would round
to zero under nearest rounding. Forward/backward dequantize to bf16 for the GEMM, so
this needs no FP8 tensor cores. Full bf16 gradients (no low-rank projection).

An optional W8A8 mode (w8a8_compute) runs the forward and grad-input GEMMs in int8
via torch._int_mm, with the ConvRot rotation applied on the activation side; the
grad-weight GEMM stays bf16 so the stochastic-rounding weight updates are unchanged.

Granularity is selectable: row-wise (group_size=0, one scale per output channel) or
group-wise (group_size>0, one scale per G input elements — finer scales, lower
per-step quantization error).

Based on the int8 quantized-training recipe from PyTorch torchao (BSD-3-Clause).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from musubi_tuner.modules.int8_convrot_utils import (
    best_int8_convrot_groupsize,
    build_hadamard,
    parse_int8_convrot_groupsizes,
    rotate_activation_by_groupsize,
    rotate_weight,
)
from musubi_tuner.modules.w8a8_optimization_utils import (
    _int_mm_allow_small_m,
    _quantize_int8_per_token,
    _supports_int_mm,
)

aten = torch.ops.aten


def _resolve_convrot_group(in_features: int, spec) -> int:
    """Resolve a ConvRot rotation group spec ('auto' / int / comma-list / 0) to the largest
    power-of-4 group that divides in_features, or 0 (rotation off) when none fits."""
    if spec in (None, 0, "", "0"):
        return 0
    return best_int8_convrot_groupsize(in_features, parse_int8_convrot_groupsizes(spec)) or 0


@torch.no_grad()
def quantize_int8(
    tensor: Tensor,
    group_size: int = 0,
    stochastic_rounding: bool = False,
    eps: float = 1e-12,
    outlier_clip_quantile: float = 1.0,
    sparse_ratio: float = 0.0,
    convrot_group: int = 0,
):
    """Symmetric int8 quantization.

    Returns (int8 data [out,in], scale) when sparse_ratio<=0, or
    (int8 data, scale, sparse_idx, sparse_val) when sparse_ratio>0.

    group_size<=0 -> row-wise: scale shape (out,). group_size>0 -> group-wise along the
    input dim: scale shape (out, in//group_size). Stochastic rounding rounds up with
    probability x-floor(x) so that small updates that would round to zero under nearest
    rounding still make stochastic progress.

    outlier_clip_quantile<1.0 sets the scale from a per-row/group quantile of |w| rather
    than the absmax: the top (1 - q) of weights gets clipped to ±127·scale, the rest gets
    a tighter grid. Default 1.0 uses the absmax.

    sparse_ratio>0 (dense-and-sparse) holds the top
    ``sparse_ratio`` fraction of |w| as an exact fp32 side-vector (sparse_idx flat
    indices, sparse_val values) and zeroes those positions BEFORE computing the int8
    scale, so a few heavy outliers no longer inflate the grid for the bulk. Outliers are
    scattered back on dequantize. Returns the extra (sparse_idx int64, sparse_val fp32).
    Orthogonal to outlier_clip_quantile (clip discards the tail; sparse preserves it
    exactly) — use one or the other.
    """
    out_features, in_features = tensor.shape
    if convrot_group and convrot_group > 0:
        # ConvRot: rotate the weight into Hadamard space before the int8 grid so per-group
        # outliers are spread out; dequantize() applies the inverse rotation. The rotation is
        # exact/loss-free (H orthonormal), so it does not change what the bf16 GEMM sees.
        if in_features % convrot_group != 0:
            raise ValueError(f"ConvRot group {convrot_group} does not divide in_features={in_features}")
        _H = build_hadamard(convrot_group, device=tensor.device, dtype=tensor.dtype)
        tensor = rotate_weight(tensor, _H, convrot_group)
    sparse_idx = None
    sparse_val = None
    work = tensor
    if sparse_ratio and sparse_ratio > 0.0:
        flat = tensor.reshape(-1)
        k = int(flat.numel() * float(sparse_ratio))
        if k > 0:
            sparse_idx = flat.abs().topk(k, sorted=False).indices
            sparse_val = flat[sparse_idx].to(torch.float32).clone()
            work = flat.clone()
            work[sparse_idx] = 0.0  # exclude outliers from the dense int8 grid
            work = work.reshape(out_features, in_features)
    use_group = bool(group_size) and group_size > 0 and in_features % group_size == 0 and group_size < in_features
    if use_group:
        t = work.reshape(out_features, in_features // group_size, group_size)
        if outlier_clip_quantile >= 1.0:
            scale = t.abs().amax(-1) / 127
        else:
            scale = torch.quantile(t.abs().float(), outlier_clip_quantile, dim=-1).to(t.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).unsqueeze(-1)
        q = t.float() * inv
    else:
        if outlier_clip_quantile >= 1.0:
            scale = work.abs().amax(1) / 127
        else:
            scale = torch.quantile(work.abs().float(), outlier_clip_quantile, dim=1).to(work.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).view(-1, 1)
        q = work.float() * inv
    if stochastic_rounding:
        q = (q + torch.rand_like(q)).floor()
    else:
        q = q.round()
    q = q.clip(-128, 127).reshape(out_features, in_features).to(torch.int8)
    if sparse_ratio and sparse_ratio > 0.0:
        if sparse_idx is None:  # k==0 (tiny tensor): empty side-vectors
            sparse_idx = tensor.new_empty(0, dtype=torch.int64)
            sparse_val = tensor.new_empty(0, dtype=torch.float32)
        return q, scale, sparse_idx, sparse_val
    return q, scale


def _dequantize(int_data: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim == 1:  # row-wise
        return int_data * scale.view(-1, 1)
    out_features, in_features = int_data.shape
    n_groups = scale.shape[-1]
    return (int_data.reshape(out_features, n_groups, in_features // n_groups) * scale.unsqueeze(-1)).reshape(
        out_features, in_features
    )


def _w8a8_supported(weight: "Int8QTWeight", x: Tensor) -> bool:
    """Whether the W8A8 int8-compute path can run for this weight/activation.

    Requires per-row weight scale (group_size == 0; a per-group input-dim scale cannot
    factor out of the int8 GEMM reduction), no dense-sparse side vector, a CUDA activation
    with torch._int_mm support, and a real float activation dtype. group_size/sparse are
    rejected at conversion time; this guard also covers device/dtype at runtime.
    """
    if not getattr(weight, "w8a8_compute", False):
        return False
    if weight.group_size != 0 or weight.scale.ndim != 1:
        return False
    if weight.sparse_ratio != 0.0 or weight.sparse_val is not None:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    return bool(x.is_cuda and _supports_int_mm(x.device))


def _w8a8_forward(x: Tensor, weight: "Int8QTWeight", bias: Tensor | None) -> Tensor:
    """Forward y = x·Wᵀ with the GEMM in int8.

    int_data holds quant(rotate(W)); rotating the activation (xr = x·H) and quantizing it
    per token pairs with int_data because H·H = I: (x·H)·(W·Hᵀ)ᵀ == x·Wᵀ. The epilogue
    rescales the int32 accumulator by the per-token and per-row scales in fp32.
    """
    int_data = weight.int_data  # [out, in] int8, stored in ConvRot space when convrot_group>0
    w_scale = weight.scale  # [out] per-row fp32
    g = weight.convrot_group
    x2d = x.reshape(-1, x.shape[-1])
    xr = rotate_activation_by_groupsize(x2d, g) if g > 0 else x2d
    x_int8, s_x = _quantize_int8_per_token(xr)  # [M, in] int8, [M, 1] fp32
    # _int_mm handles the transposed (column-major) second operand natively; reduction is over `in`.
    acc = _int_mm_allow_small_m(x_int8.contiguous(), int_data.t())  # [M, out] int32
    out = (acc.float() * s_x * w_scale.reshape(1, -1).float()).to(x.dtype)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], int_data.shape[0])


def _w8a8_grad_input(grad_output: Tensor, weight: "Int8QTWeight", dt: torch.dtype) -> Tensor:
    """Grad-input grad_x = grad_output·W with the GEMM in int8.

    The per-out-row weight scale lies on the reduced (out) dimension, so it is folded into
    grad_output before the per-token int8 quantization rather than applied as an output
    scale. The int GEMM then contracts against int_data directly (row-major RHS). For ConvRot
    the result is in rotated-input space; one more rotation inverts it (H·H = I).
    """
    int_data = weight.int_data  # [out, in] int8
    w_scale = weight.scale  # [out] per-row fp32
    g = weight.convrot_group
    out_features, in_features = int_data.shape
    go = grad_output.reshape(-1, out_features)
    g_scaled = go.float() * w_scale.reshape(1, -1).float()
    g_int8, s_g = _quantize_int8_per_token(g_scaled)  # [M, out] int8, [M, 1] fp32
    acc = _int_mm_allow_small_m(g_int8.contiguous(), int_data)  # [M, in] int32
    grad_input = (acc.float() * s_g).to(dt)
    if g > 0:
        grad_input = rotate_activation_by_groupsize(grad_input, g, inverse=True)
    return grad_input.reshape(*grad_output.shape[:-1], in_features)


class _Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: "Int8QTWeight", bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        if _w8a8_supported(weight, input):
            ctx.w8a8 = True
            return _w8a8_forward(input, weight, bias)
        ctx.w8a8 = False
        w = weight.dequantize()  # bf16 weight; group-wise scale can't factor out of the GEMM
        out = input @ w.T
        return out + bias if bias is not None else out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight = ctx.saved_tensors
        # Under mixed precision grad_output can arrive as fp32 while the saved activation and the
        # dequantized weight are bf16; align all three to the forward compute dtype so the GEMMs match.
        dt = input.dtype
        grad_output = grad_output.to(dt)
        if getattr(ctx, "w8a8", False):
            grad_input = _w8a8_grad_input(grad_output, weight, dt)
        else:
            w = weight.dequantize().to(dt)
            grad_input = grad_output @ w
        # grad-weight and grad-bias stay bf16 over the real-space saved input: int8 here would
        # coarsen the stochastic-rounding weight updates.
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(-1, weight.shape[1])
        grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias


_OPS: dict = {}


def _implements(ops):
    ops = ops if isinstance(ops, (list, tuple)) else [ops]

    def deco(fn):
        for op in ops:
            _OPS[op] = fn
        return fn

    return deco


class Int8QTWeight(Tensor):
    """Int8 weight that updates in place with stochastic rounding (row- or group-wise).

    outlier_clip_quantile (default 1.0 = absmax) is preserved on the weight and applied
    on every in-place requantize so the scale stays tight against the bulk distribution
    during training instead of being inflated by a few outliers.
    """

    @staticmethod
    def __new__(
        cls,
        int_data: Tensor,
        scale: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_idx: Tensor | None = None,
        sparse_val: Tensor | None = None,
        sparse_ratio: float = 0.0,
        convrot_group: int = 0,
        w8a8_compute: bool = False,
    ):
        return Tensor._make_wrapper_subclass(cls, int_data.shape, dtype=scale.dtype, device=int_data.device, requires_grad=False)

    def __init__(
        self,
        int_data: Tensor,
        scale: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_idx: Tensor | None = None,
        sparse_val: Tensor | None = None,
        sparse_ratio: float = 0.0,
        convrot_group: int = 0,
        w8a8_compute: bool = False,
    ):
        assert int_data.dtype is torch.int8 and int_data.ndim == 2
        self.int_data = int_data
        self.scale = scale
        self.group_size = int(group_size)
        self.outlier_clip_quantile = float(outlier_clip_quantile)
        self.sparse_ratio = float(sparse_ratio)
        # int_data is stored in ConvRot-rotated space when convrot_group>0; dequantize() inverts it.
        self.convrot_group = int(convrot_group)
        # When True the forward and grad-input GEMMs run in int8 (torch._int_mm); grad-weight stays bf16.
        self.w8a8_compute = bool(w8a8_compute)
        # dense-and-sparse outlier side-vectors (None when sparse_ratio<=0).
        self.sparse_idx = sparse_idx
        self.sparse_val = sparse_val

    def __tensor_flatten__(self):
        inner = ["int_data", "scale"]
        if self.sparse_val is not None:
            inner = inner + ["sparse_idx", "sparse_val"]
        return inner, [self.group_size, self.outlier_clip_quantile, self.sparse_ratio, self.convrot_group, self.w8a8_compute]

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        ocq = meta[1] if len(meta) > 1 else 1.0
        sr = meta[2] if len(meta) > 2 else 0.0
        cg = meta[3] if len(meta) > 3 else 0
        w8a8 = meta[4] if len(meta) > 4 else False
        return cls(
            inner_tensors["int_data"],
            inner_tensors["scale"],
            meta[0],
            ocq,
            inner_tensors.get("sparse_idx"),
            inner_tensors.get("sparse_val"),
            sr,
            cg,
            w8a8,
        )

    def dequantize(self) -> Tensor:
        w = _dequantize(self.int_data, self.scale)
        if self.sparse_val is not None and self.sparse_val.numel() > 0:
            # w is fresh (int_data*scale); scatter outliers in place, no clone
            w = w.contiguous()
            w.view(-1)[self.sparse_idx] = self.sparse_val.to(w.dtype)
        if self.convrot_group > 0:
            # invert the encode-time ConvRot rotation to recover the real-space weight for the GEMM
            _H = build_hadamard(self.convrot_group, device=w.device, dtype=w.dtype)
            w = rotate_weight(w, _H, self.convrot_group, inverse=True)
        return w

    @torch.no_grad()
    def requantize_(self, t: Tensor, stochastic_rounding: bool = True) -> "Int8QTWeight":
        """In-place write of ``t`` into the int8 storage.

        Replaces int_data/scale (and the dense-sparse outlier side-vectors when
        sparse_ratio>0) with a fresh quantization of ``t`` using this weight's
        group_size / outlier_clip_quantile / sparse_ratio. stochastic_rounding=True
        (default) is the per-step SR update path; False writes deterministic
        round-to-nearest, for optimizers that handle the rounding residual themselves.
        """
        if self.sparse_ratio > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                t,
                self.group_size,
                stochastic_rounding=stochastic_rounding,
                outlier_clip_quantile=self.outlier_clip_quantile,
                sparse_ratio=self.sparse_ratio,
                convrot_group=self.convrot_group,
            )
            self.int_data.copy_(int_data)
            self.scale.copy_(scale)
            # k = floor(numel*ratio) is constant for a fixed weight, so the side-vector
            # sizes are stable across updates and copy_ is in place.
            self.sparse_idx.copy_(sidx)
            self.sparse_val.copy_(sval)
            return self
        int_data, scale = quantize_int8(
            t,
            self.group_size,
            stochastic_rounding=stochastic_rounding,
            outlier_clip_quantile=self.outlier_clip_quantile,
            convrot_group=self.convrot_group,
        )
        self.int_data.copy_(int_data)
        self.scale.copy_(scale)
        return self

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        weight: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_ratio: float = 0.0,
        convrot_group: int = 0,
        w8a8_compute: bool = False,
    ) -> "Int8QTWeight":
        cg = _resolve_convrot_group(weight.shape[1], convrot_group)
        if sparse_ratio and sparse_ratio > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                weight.detach(),
                group_size,
                stochastic_rounding=False,
                outlier_clip_quantile=outlier_clip_quantile,
                sparse_ratio=sparse_ratio,
                convrot_group=cg,
            )
            return cls(int_data, scale, group_size, outlier_clip_quantile, sidx, sval, sparse_ratio, cg, w8a8_compute)
        int_data, scale = quantize_int8(
            weight.detach(), group_size, stochastic_rounding=False, outlier_clip_quantile=outlier_clip_quantile, convrot_group=cg
        )
        return cls(int_data, scale, group_size, outlier_clip_quantile, None, None, 0.0, cg, w8a8_compute)

    def __repr__(self):
        return f"Int8QTWeight(shape={tuple(self.shape)}, group_size={self.group_size}, dtype={self.dtype})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            return _Int8Linear.apply(args[0], args[1], args[2] if len(args) > 2 else None)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        handler = _OPS.get(func)
        if handler is None:
            raise NotImplementedError(f"Int8QTWeight: unimplemented op {func}")
        return handler(func, types, args, kwargs)


@_implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    a = args[0]
    sidx = getattr(a, "sparse_idx", None)
    sval = getattr(a, "sparse_val", None)
    out = Int8QTWeight(
        func(a.int_data),
        func(a.scale),
        a.group_size,
        getattr(a, "outlier_clip_quantile", 1.0),
        func(sidx) if sidx is not None else None,
        func(sval) if sval is not None else None,
        getattr(a, "sparse_ratio", 0.0),
        getattr(a, "convrot_group", 0),
        getattr(a, "w8a8_compute", False),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@_implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    a = args[0]
    sidx = getattr(a, "sparse_idx", None)
    sval = getattr(a, "sparse_val", None)
    out = Int8QTWeight(
        a.int_data.to(device=device),
        a.scale.to(device=device, dtype=dtype),
        a.group_size,
        getattr(a, "outlier_clip_quantile", 1.0),
        sidx.to(device=device) if sidx is not None else None,  # idx stays int64
        sval.to(device=device) if sval is not None else None,  # val stays fp32
        getattr(a, "sparse_ratio", 0.0),
        getattr(a, "convrot_group", 0),
        getattr(a, "w8a8_compute", False),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@_implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)  # plain tensor for optimizer state


@_implements([aten.sub.Tensor, aten.mul.Tensor, aten.linalg_vector_norm.default])
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, Int8QTWeight) else x for x in args]
    return func(*args, **kwargs)


@_implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, Int8QTWeight) and isinstance(src, Int8QTWeight):
        dst.int_data.copy_(src.int_data)
        dst.scale.copy_(src.scale)
        if getattr(dst, "sparse_val", None) is not None and getattr(src, "sparse_val", None) is not None:
            dst.sparse_idx.copy_(src.sparse_idx)
            dst.sparse_val.copy_(src.sparse_val)
    elif isinstance(dst, Int8QTWeight):
        if getattr(dst, "sparse_ratio", 0.0) > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                src,
                dst.group_size,
                stochastic_rounding=True,  # SR on update
                outlier_clip_quantile=getattr(dst, "outlier_clip_quantile", 1.0),
                sparse_ratio=dst.sparse_ratio,
                convrot_group=getattr(dst, "convrot_group", 0),
            )
            dst.int_data.copy_(int_data)
            dst.scale.copy_(scale)
            dst.sparse_idx.copy_(sidx)
            dst.sparse_val.copy_(sval)
        else:
            int_data, scale = quantize_int8(
                src,
                dst.group_size,
                stochastic_rounding=True,  # SR on update
                outlier_clip_quantile=getattr(dst, "outlier_clip_quantile", 1.0),
                convrot_group=getattr(dst, "convrot_group", 0),
            )
            dst.int_data.copy_(int_data)
            dst.scale.copy_(scale)
    else:
        dst.copy_(src.dequantize())
    return dst


@_implements(
    [
        aten.add_.Tensor,
        aten.sub_.Tensor,
        aten.addcdiv_.default,
        aten.addcmul_.default,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.lerp_.Scalar,
        aten.lerp_.Tensor,
    ]
)
def _(func, types, args, kwargs):
    # Dequantize every Int8QTWeight in args, not just args[0]: ops like
    # grad.sub_(param) put the weight at args[1], which would otherwise recurse
    # through __torch_dispatch__ until the recursion limit.
    original = args[0]
    deq_args = tuple(a.dequantize() if isinstance(a, Int8QTWeight) else a for a in args)
    out = func(*deq_args, **kwargs)
    return original.copy_(out)


def convert_to_int8_training(
    model: nn.Module,
    *,
    filter_fn=None,
    group_size: int = 0,
    outlier_clip_quantile: float = 1.0,
    sparse_ratio: float = 0.0,
    convrot_group=0,
    w8a8: bool = False,
) -> int:
    """Swap eligible nn.Linear weights to Int8QTWeight in place. Returns count.

    w8a8=True marks each converted weight for the int8-compute forward/grad-input path.
    That path needs a per-row weight scale and a dense grid, so group_size and sparse_ratio
    must both be 0; otherwise the int8 GEMM cannot reproduce the dequantized result.
    """
    if w8a8:
        if group_size and group_size > 0:
            raise ValueError(
                "int8 W8A8 compute requires group_size 0 (per-row weight scale); a per-group input-dim "
                "scale cannot factor out of the int8 GEMM reduction."
            )
        if sparse_ratio and sparse_ratio > 0.0:
            raise ValueError(
                "int8 W8A8 compute requires sparse_ratio 0; a dense-sparse outlier side vector breaks the pure int8 GEMM."
            )
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module.weight, Int8QTWeight):
            continue
        if filter_fn is not None and not filter_fn(module, name):
            continue
        w = module.weight
        module.weight = nn.Parameter(
            Int8QTWeight.from_float(
                w.data,
                group_size,
                outlier_clip_quantile=outlier_clip_quantile,
                sparse_ratio=sparse_ratio,
                convrot_group=convrot_group,
                w8a8_compute=w8a8,
            ),
            requires_grad=w.requires_grad,
        )
        replaced += 1
    return replaced


@torch.no_grad()
def load_int8_prequant_into_training(
    model: nn.Module,
    tensors: dict,
    *,
    filter_fn=None,
    group_size: int = 0,
    outlier_clip_quantile: float = 1.0,
    sparse_ratio: float = 0.0,
    w8a8: bool = False,
) -> int:
    """Reconstruct Int8QTWeight from pre-quantized tensors instead of quantizing from bf16.

    Bit-exact drop-in for convert_to_int8_training when ``tensors`` were produced by
    ltx2_export_int8_weights with the SAME grid (group_size / outlier_clip_quantile /
    sparse_ratio / convrot). Selects target Linears with the same ``filter_fn`` and, for
    each, rebuilds the Int8QTWeight from the stored int_data / scale / per-layer
    convrot_group (+ optional dense-sparse side vectors) — skipping the slow from_float
    Hadamard-rotation + outlier-scan that dominates int8-weight startup. ``tensors`` is a
    flat {name: Tensor} dict keyed by module FQN, e.g. ``"<fqn>.int_data"``.
    """

    # The exporter keys the sidecar by the checkpoint's base namespace (e.g.
    # "transformer_blocks.0.attn1.to_q"). At training time the DiT is wrapped, so
    # named_modules() FQNs carry a leading "model." that the sidecar keys lack.
    # Normalize both sides to the base namespace so lookup matches regardless.
    def _base_ns(k: str) -> str:
        return k[len("model.") :] if k.startswith("model.") else k

    tn = {_base_ns(k): v for k, v in tensors.items()}

    replaced = 0
    consumed = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module.weight, Int8QTWeight):
            continue
        if filter_fn is not None and not filter_fn(module, name):
            continue
        key = _base_ns(name)
        idk, sck, cgk = key + ".int_data", key + ".scale", key + ".convrot_group"
        if idk not in tn or sck not in tn:
            raise ValueError(
                f"pre-quantized int8-weights checkpoint is missing tensors for target layer {name!r} "
                f"(expected {idk!r} and {sck!r}). The file does not match this model / --int8_weights_targets."
            )
        dev = module.weight.device
        int_data = tn[idk].to(dev)
        scale = tn[sck].to(dev)
        convrot_group = int(tn[cgk].item()) if cgk in tn else 0
        sidx = tn.get(key + ".sparse_idx")
        sval = tn.get(key + ".sparse_val")
        if sidx is not None:
            sidx = sidx.to(dev)
        if sval is not None:
            sval = sval.to(dev)
        module.weight = nn.Parameter(
            Int8QTWeight(
                int_data,
                scale,
                group_size,
                outlier_clip_quantile,
                sidx,
                sval,
                sparse_ratio,
                convrot_group,
                w8a8,
            ),
            requires_grad=module.weight.requires_grad,
        )
        consumed.add(key)
        replaced += 1
    orphans = sorted({k[: -len(".int_data")] for k in tn if k.endswith(".int_data")} - consumed)
    if orphans:
        raise ValueError(
            f"pre-quantized int8-weights checkpoint has {len(orphans)} quantized layer(s) that matched no target "
            f"module in this model (e.g. {orphans[:3]}). The file's targets differ from --int8_weights_targets."
        )
    return replaced
