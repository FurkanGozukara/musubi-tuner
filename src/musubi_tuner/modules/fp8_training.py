"""FP8 full fine-tuning for LTX-2 via torch._scaled_mm (per-tensor or rowwise dynamic scaling).

Based on the FP8 training recipe from NVIDIA TransformerEngine and PyTorch torchao
(both BSD-3-Clause); independent re-implementation, no torchao runtime dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2

_FP8_DTYPES = {"e4m3": E4M3, "e5m2": E5M2}


def resolve_fp8_dtype(name: str) -> torch.dtype:
    key = str(name).lower().replace("float8_", "").replace("fn", "")
    if key not in _FP8_DTYPES:
        raise ValueError(f"Unknown fp8 dtype {name!r}; expected one of {sorted(_FP8_DTYPES)}")
    return _FP8_DTYPES[key]


def is_fp8_training_supported(device: torch.device | int | None = None) -> bool:
    """True if the CUDA device has FP8 tensor cores (sm_89+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def assert_fp8_training_supported(device: torch.device | int | None = None) -> None:
    if not is_fp8_training_supported(device):
        cap = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
        raise RuntimeError(
            f"--fp8_gemm requires FP8 tensor cores (compute capability >= 8.9 / Ada or Hopper); "
            f"got {cap}. torch._scaled_mm cannot run FP8 GEMMs on this GPU. "
            f"Use int8 full-FT (--qgalore_full_ft) on pre-Ada hardware instead."
        )


def _quantize_tensor(t: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic scaling. Returns (fp8 tensor, dequant scale fp32, quant scale).

    ``amax`` is a full reduction, so the returned quant scale is layout- and zero-pad-
    invariant: the same scalar quantizes any transpose/padding of ``t`` bit-for-bit.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = t.detach().abs().amax().clamp(min=1e-12)
    scale = fp8_max / amax
    q = (t.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv = (1.0 / scale).to(torch.float32)  # _scaled_mm multiplies operands by these
    return q, inv, scale


def _cast_with_scale(t: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast ``t`` to fp8 with a pre-computed quant scale (no amax reduction)."""
    fp8_max = torch.finfo(fp8_dtype).max
    q = (t.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv = (1.0 / scale).to(torch.float32)
    return q, inv


def _quantize_rowwise(t: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic scaling of a 2-D ``(R, C)`` tensor along its last (contraction) dim.

    Returns (fp8 tensor, dequant scale fp32 of shape ``(R, 1)``).
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = t.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (R,1)
    scale = fp8_max / amax
    q = (t.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv = (1.0 / scale).to(torch.float32)  # (R,1)
    return q, inv


def _scaled_mm_rowwise(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    try:
        return torch._scaled_mm(a, b, scale_a=scale_a.to(a.device), scale_b=scale_b.to(a.device), out_dtype=torch.bfloat16)
    except (RuntimeError, ValueError, TypeError) as e:
        raise RuntimeError(
            "--fp8_gemm_scaling rowwise requires a torch build whose torch._scaled_mm accepts per-row/"
            "per-column fp8 scale tensors (torch>=2.5; some builds need newer). "
            f"torch._scaled_mm rejected the rowwise scales: {e}"
        ) from e


def _fp8_forward(x: torch.Tensor, w: torch.Tensor, rowwise: bool):
    """y = x @ w.T. x:(M,K) w:(N,K) -> y:(M,N). K is a multiple of 16 (guaranteed by target dims).

    Returns (y, scale_x, scale_w); the quant scales feed the backward reuse (None in rowwise mode).
    """
    xc = x.contiguous()
    wc = w.contiguous()
    if rowwise:
        xq, inv_x = _quantize_rowwise(xc, E4M3)  # (M,K),(M,1)
        wq, inv_w = _quantize_rowwise(wc, E4M3)  # (N,K),(N,1)
        y = _scaled_mm_rowwise(xq, wq.t(), inv_x, inv_w.t())  # (M,N)
        return y, None, None
    xq, inv_x, scale_x = _quantize_tensor(xc, E4M3)  # (M,K) row-major
    wq, inv_w, scale_w = _quantize_tensor(wc, E4M3)  # (N,K) row-major
    y = torch._scaled_mm(
        xq, wq.t(), scale_a=inv_x.to(xq.device), scale_b=inv_w.to(xq.device), out_dtype=torch.bfloat16
    )  # wq.t() is (K,N) column-major
    return y, scale_x, scale_w


def _fp8_backward(
    gy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    scale_x: torch.Tensor | None,
    scale_w: torch.Tensor | None,
    grad_dtype: torch.dtype,
    rowwise: bool,
):
    """grad_x = gy @ w (M,K); grad_w = gy.T @ x (N,K). gy:(M,N).

    grad_w's contraction dim is the token count M, which is often not a multiple of 16, so it
    is zero-padded (padding leaves the matmul and per-row/per-tensor amax exact).
    """
    gyc = gy.contiguous()
    if rowwise:
        # grad_x = gy @ w
        gxq, inv_gx = _quantize_rowwise(gyc, grad_dtype)  # (M,N),(M,1)
        wtq, inv_wt = _quantize_rowwise(w.t().contiguous(), E4M3)  # (K,N),(K,1)
        grad_x = _scaled_mm_rowwise(gxq, wtq.t(), inv_gx, inv_wt.t())  # (M,K)
        # grad_w = gy.T @ x  (rowwise scale axis differs per layout, so gy is re-quantized here)
        gtq, inv_gt = _quantize_rowwise(gy.t().contiguous(), grad_dtype)  # (N,M),(N,1)
        xtq, inv_xt = _quantize_rowwise(x.t().contiguous(), E4M3)  # (K,M),(K,1)
        m = gtq.shape[-1]
        if m % 16 != 0:
            pad = 16 - (m % 16)
            gtq = torch.nn.functional.pad(gtq, (0, pad))
            xtq = torch.nn.functional.pad(xtq, (0, pad))
        grad_w = _scaled_mm_rowwise(gtq, xtq.t(), inv_gt, inv_xt.t())  # (N,K)
        return grad_x, grad_w

    gyq, inv_gy, _ = _quantize_tensor(gyc, grad_dtype)  # (M,N) row-major, quantized once
    # grad_x = gy @ w: reuse the forward weight scale, cast w.T in (K,N) row-major (no amax)
    wq_kn, inv_w = _cast_with_scale(w.t().contiguous(), scale_w, E4M3)  # (K,N)
    grad_x = torch._scaled_mm(
        gyq, wq_kn.t(), scale_a=inv_gy.to(gyq.device), scale_b=inv_w.to(gyq.device), out_dtype=torch.bfloat16
    )  # wq_kn.t() is (N,K) column-major
    # grad_w = gy.T @ x: transpose the existing fp8 gy (byte copy) and reuse the forward x scale
    gyq_nm = gyq.t().contiguous()  # (N,M) fp8; == quantize(gy.T) elementwise
    xq_km, inv_x = _cast_with_scale(x.t().contiguous(), scale_x, E4M3)  # (K,M)
    m = gyq_nm.shape[-1]
    if m % 16 != 0:
        pad = 16 - (m % 16)
        gyq_nm = torch.nn.functional.pad(gyq_nm, (0, pad))  # (N,M+pad)
        xq_km = torch.nn.functional.pad(xq_km, (0, pad))  # (K,M+pad)
    grad_w = torch._scaled_mm(
        gyq_nm, xq_km.t(), scale_a=inv_gy.to(gyq_nm.device), scale_b=inv_x.to(gyq_nm.device), out_dtype=torch.bfloat16
    )  # xq_km.t() is (M+pad,K) column-major
    return grad_x, grad_w


_fp8_forward_compiled = None
_fp8_backward_compiled = None
_compile_disabled = False


def _fp8_forward_fallback(*args):
    """Call the compiled forward; on any compile/runtime failure warn once and fall back
    to eager for the rest of the run."""
    global _compile_disabled
    try:
        return _fp8_forward_compiled(*args)
    except Exception as e:  # noqa: BLE001
        if not _compile_disabled:
            logger.warning("--fp8_gemm_compile: compiled FP8 GEMM failed (%s); using eager FP8 GEMM.", e)
            _compile_disabled = True
        return _fp8_forward(*args)


def _fp8_backward_fallback(*args):
    global _compile_disabled
    try:
        return _fp8_backward_compiled(*args)
    except Exception as e:  # noqa: BLE001
        if not _compile_disabled:
            logger.warning("--fp8_gemm_compile: compiled FP8 GEMM failed (%s); using eager FP8 GEMM.", e)
            _compile_disabled = True
        return _fp8_backward(*args)


def _get_fp8_ops(compile_gemm: bool) -> tuple[Callable, Callable]:
    """Return the (forward, backward) GEMM helpers, optionally torch.compiled.

    Region-compiling only these helpers (not the surrounding block) fuses the scaling
    (amax/cast/clamp) into the GEMM and survives the graph breaks block-level compile
    hits on LTX-2 (gradient checkpointing, flash attention, Python control flow).
    """
    global _fp8_forward_compiled, _fp8_backward_compiled, _compile_disabled
    if not compile_gemm or _compile_disabled:
        return _fp8_forward, _fp8_backward
    if _fp8_forward_compiled is None:
        try:
            # One compiled helper each serves every (attn/FFN) x (fwd/grad) GEMM shape; raise the
            # recompile cap above dynamo's default of 8 so all shapes compile instead of thrashing.
            import torch._dynamo as _dynamo

            _dynamo.config.recompile_limit = max(int(getattr(_dynamo.config, "recompile_limit", 8)), 64)
            _fp8_forward_compiled = torch.compile(_fp8_forward, dynamic=False)
            _fp8_backward_compiled = torch.compile(_fp8_backward, dynamic=False)
        except Exception as e:  # noqa: BLE001
            _compile_disabled = True
            logger.warning("--fp8_gemm_compile: torch.compile unavailable (%s); using eager FP8 GEMM.", e)
            return _fp8_forward, _fp8_backward
    return _fp8_forward_fallback, _fp8_backward_fallback


class _Float8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, grad_dtype: torch.dtype, rowwise: bool, fwd_fn, bwd_fn):
        y, scale_x, scale_w = fwd_fn(x, w, rowwise)
        ctx.save_for_backward(x, w)
        ctx.scale_x = scale_x  # 0-dim quant scales (None in rowwise mode); free to reuse in backward
        ctx.scale_w = scale_w
        ctx.grad_dtype = grad_dtype
        ctx.rowwise = rowwise
        ctx.bwd_fn = bwd_fn
        return y

    @staticmethod
    def backward(ctx, gy: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_x, grad_w = ctx.bwd_fn(gy, x, w, ctx.scale_x, ctx.scale_w, ctx.grad_dtype, ctx.rowwise)
        return grad_x, grad_w, None, None, None, None


class Float8TrainingLinear(nn.Linear):
    """Drop-in nn.Linear with FP8 fwd/bwd GEMMs; keeps a bf16 master weight + bias.

    The optimizer only sees the bf16 weight/grad, so it stays optimizer-agnostic.
    """

    grad_dtype: torch.dtype = E4M3
    compile_gemm: bool = False
    scaling: str = "tensor"  # "tensor" | "rowwise"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        rowwise = self.scaling == "rowwise"
        fwd_fn, bwd_fn = _get_fp8_ops(self.compile_gemm)
        y = _Float8LinearFn.apply(x.reshape(-1, shp[-1]), self.weight, self.grad_dtype, rowwise, fwd_fn, bwd_fn)
        y = y.reshape(*shp[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def _fp8_dims_ok(linear: nn.Linear) -> bool:
    # FP8 GEMMs (_scaled_mm) require both matmul dims to be multiples of 16.
    return (linear.in_features % 16 == 0) and (linear.out_features % 16 == 0)


@dataclass
class Fp8FftSummary:
    replaced: int = 0
    replaced_numel: int = 0
    skipped_not_target: int = 0
    skipped_dims: int = 0
    skipped_small: int = 0
    replaced_names: list[str] = field(default_factory=list)


def ltx2_fp8_filter(targets: str | Iterable[str] = "video", min_weight_numel: int = 16384) -> Callable[[nn.Linear, str], bool]:
    """Filter for the big LTX-2 attention/FFN GEMMs; excludes gates, norms, AdaLN.

    Reuses Q-GaLore's LTX-2 target matching so --fp8_gemm_targets behaves like
    --qgalore_targets (video / audio / attn / ff / blocks / all).
    """
    from musubi_tuner.optimizers.q_galore import _matches_ltx2_target, _parse_target_tokens

    tokens = _parse_target_tokens(targets)

    def _keep(mod: nn.Linear, fqn: str) -> bool:
        name = fqn.lower()
        if "gate" in name or "norm" in name:  # gated-attention logits / RMSNorm-adjacent
            return False
        return _matches_ltx2_target(fqn, tokens)

    _keep._tokens = tokens  # type: ignore[attr-defined]
    _keep._min_numel = int(min_weight_numel)  # type: ignore[attr-defined]
    return _keep


def convert_ltx2_to_fp8_training(
    model: nn.Module,
    *,
    targets: str | Iterable[str] = "video",
    grad_dtype: torch.dtype = E4M3,
    min_weight_numel: int = 16384,
    compile_gemm: bool = False,
    scaling: str = "tensor",
) -> Fp8FftSummary:
    """Swap eligible nn.Linear -> Float8TrainingLinear in place. Keeps weights."""
    keep = ltx2_fp8_filter(targets, min_weight_numel)
    summary = Fp8FftSummary()
    modules = dict(model.named_modules())
    for parent_name, parent in list(modules.items()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, Float8TrainingLinear):
                continue
            fqn = f"{parent_name}.{cname}" if parent_name else cname
            if not keep(child, fqn):
                summary.skipped_not_target += 1
                continue
            if child.weight.numel() < int(min_weight_numel):
                summary.skipped_small += 1
                continue
            if not _fp8_dims_ok(child):
                summary.skipped_dims += 1
                continue
            # In-place class reassignment preserves the exact weight/bias tensors and
            # consumes no RNG. (A fresh Float8TrainingLinear would run nn.Linear's random
            # init, desyncing the global RNG — hence data/noise sampling — vs a bf16 run.)
            child.__class__ = Float8TrainingLinear
            child.grad_dtype = grad_dtype
            child.compile_gemm = compile_gemm
            child.scaling = scaling
            summary.replaced += 1
            summary.replaced_numel += int(child.weight.numel())
            summary.replaced_names.append(fqn)
    return summary
