"""ConvRot INT8 Linear as torch.library custom ops (V39, kohya-ss/sd-scripts a4aaa20 / 881e578 / 5c2d263).

The patched Linear must be an opaque node for torch.compile (no graph break), so the trainers compile INT8 blocks with
their Linears (only block swap keeps Linears eager). The backward rotates the dequantized weight back to the original
basis (fused Triton kernel on CUDA) instead of rotating the [M, K] gradient.

CPU tests run the eager fallback and aot_eager; CUDA tests run the fused Triton kernels and inductor.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules import convrot_int8_kernels as kernels
from musubi_tuner.modules import convrot_int8_utils as utils

GS = utils.CONVROT_GROUPSIZE  # 256
K, N, M = 512, 96, 64

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_triton = pytest.mark.skipif(not kernels.HAS_TRITON, reason="triton required")


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _quantized(device, dtype, groupsize=GS, seed=0, bias=True):
    """(wq, scale, dequantized reference weight in the original basis, bias); groupsize 0 = plain (unrotated) INT8."""
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    if groupsize:
        wq, ws = kernels.quantize_int8_convrot_weight(w, groupsize)
        w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, groupsize).to(dtype)
    else:
        wq, ws = kernels.quantize_int8_rowwise(w.float())
        w_deq = (wq.float() * ws).to(dtype)
    b = torch.randn(N, device=device, dtype=dtype) * 0.1 if bias else None
    return wq, ws, w_deq, b


class _Block(nn.Module):
    """A frozen (patched) Linear with a trainable bias, a LoRA-like side branch and a norm: the shape of a DiT block."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(K, N, bias=True)
        self.lora = nn.Linear(K, N, bias=False)
        self.norm = nn.LayerNorm(N)

    def forward(self, x):
        return self.norm(self.lin(x) + self.lora(x)) * 2


def _patched_block(device, dtype, groupsize=GS, bwd_mode="bf16"):
    """(INT8-patched block, reference block holding the dequantized weight)."""
    wq, ws, w_deq, b = _quantized(device, dtype, groupsize)
    ref = _Block().to(device=device, dtype=dtype)
    with torch.no_grad():
        ref.lin.weight.copy_(w_deq)
        ref.lin.bias.copy_(b)
    block = _Block().to(device=device, dtype=dtype)
    block.load_state_dict(ref.state_dict())
    utils.apply_convrot_int8_monkey_patch(block, {"lin.weight": wq, "lin.scale_weight": ws}, bwd_mode=bwd_mode, groupsize_map={"lin": groupsize})
    block.lin.weight.requires_grad_(False)
    block.lin.load_state_dict({"weight": wq, "scale_weight": ws, "bias": b.clone()}, assign=True)
    block.lin.bias.requires_grad_(True)
    return block, ref


def _fwd_bwd(module, x):
    x = x.detach().clone().requires_grad_(True)
    out = module(x)
    # a linear loss: the squared norm of a LayerNorm output is constant, its input gradient would be rounding noise
    (out.float() * torch.linspace(-1.0, 1.0, out.shape[-1], device=out.device)).sum().backward()
    return out, x.grad, module.lin.bias.grad.clone(), module.lora.weight.grad.clone()


# ---------------------------------------------------------------------------
# eager (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groupsize", [GS, 64, 0])
def test_backward_rotates_the_weight_back_and_matches_the_dequantized_reference(groupsize):
    wq, ws, w_deq, _ = _quantized("cpu", torch.float32, groupsize, bias=False)
    g = torch.randn(3, M, N)
    out = utils.convrot_int8_linear_backward(g, wq, ws, groupsize, "bf16")
    assert out.shape == (3, M, K)
    assert _relerr(out, g @ w_deq) < 1e-5


def test_compat_apply_matches_the_custom_op_and_casts_under_autocast():
    wq, ws, w_deq, b = _quantized("cpu", torch.bfloat16)
    x = torch.randn(M, K)  # fp32 input, as after an fp32 modulation
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")
        ref = F.linear(x, w_deq, b)
    assert out.dtype == torch.bfloat16 == ref.dtype
    assert _relerr(out, ref) < 2e-2
    direct = utils.convrot_int8_linear(x.to(torch.bfloat16), wq, ws, b, GS, "bf16")
    assert torch.equal(out, direct)


def test_bias_gradient_flows_only_when_needed():
    wq, ws, _, b = _quantized("cpu", torch.float32)
    x = torch.randn(2, M, K, requires_grad=True)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")  # bias without grad
    out.sum().backward()
    assert x.grad is not None and b.grad is None
    b.requires_grad_(True)
    out = utils.ConvRotInt8LinearFn.apply(x.detach(), wq, ws, b, GS, "bf16")  # input without grad
    out.sum().backward()
    assert torch.allclose(b.grad, torch.full_like(b, 2 * M))


# ---------------------------------------------------------------------------
# torch.compile: opaque node, no graph break, same numbers (CPU: aot_eager)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groupsize", [GS, 0])
def test_compiled_block_with_patched_linear_has_no_graph_breaks(groupsize):
    torch._dynamo.reset()
    block, _ = _patched_block("cpu", torch.float32, groupsize)
    explanation = torch._dynamo.explain(block)(torch.randn(M, K, requires_grad=True))
    assert explanation.graph_break_count == 0, explanation.break_reasons


@pytest.mark.parametrize("groupsize", [GS, 0])
def test_compiled_block_aot_eager_matches_eager_forward_and_backward(groupsize):
    torch._dynamo.reset()
    block, ref = _patched_block("cpu", torch.float32, groupsize)
    compiled = torch.compile(block, backend="aot_eager", fullgraph=True)
    for m in (M, M + 8):  # second shape: recompile / dynamic shapes through the fake impl
        x = torch.randn(2, m, K)
        out_c, gx_c, gb_c, gl_c = _fwd_bwd(compiled, x)
        block.zero_grad()
        out_e, gx_e, gb_e, gl_e = _fwd_bwd(block, x)
        block.zero_grad()
        out_r, gx_r, gb_r, _ = _fwd_bwd(ref, x)
        ref.zero_grad()
        assert _relerr(out_c, out_e) < 1e-6 and _relerr(gx_c, gx_e) < 1e-6
        assert _relerr(gb_c, gb_e) < 1e-6 and _relerr(gl_c, gl_e) < 1e-6
        assert _relerr(out_c, out_r) < 1e-5 and _relerr(gx_c, gx_r) < 1e-5 and _relerr(gb_c, gb_r) < 1e-5


# ---------------------------------------------------------------------------
# CUDA: fused Triton kernels and inductor
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
@pytest.mark.parametrize(("groupsize", "rows"), [(256, N), (64, N), (256, 70), (16, 40)])
def test_triton_dequantize_rotate_weight_matches_eager(groupsize, rows):
    """Fused dequant + rotate-back (backward weight path): odd row counts exercise the row mask, 16 the smallest tl.dot."""
    torch.manual_seed(0)
    w = torch.randn(rows, 2 * groupsize, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, groupsize)
    ref = kernels.dequantize_int8_convrot_weight(wq, ws, groupsize)  # fp32
    out = kernels.triton_dequantize_rotate_weight(wq, ws, groupsize, torch.bfloat16)
    assert out.dtype == torch.bfloat16 and out.shape == w.shape
    assert _relerr(out, ref) < 5e-3  # bf16 rounding of the dequantized values before the Hadamard dot
    eager = kernels._rotate_weight(wq * ws.to(torch.bfloat16), kernels._build_hadamard(groupsize, "cuda", torch.bfloat16), groupsize)
    assert _relerr(eager, ref) < 5e-3 and _relerr(out, eager) < 5e-3


@requires_cuda
@requires_triton
@pytest.mark.parametrize(("bwd_mode", "tol"), [("bf16", 2e-2), ("int8", 5e-2)])
@pytest.mark.parametrize("groupsize", [GS, 0])
def test_grad_x_cuda_matches_dequant_reference(bwd_mode, tol, groupsize):
    wq, ws, w_deq, _ = _quantized("cuda", torch.bfloat16, groupsize, bias=False)
    x = torch.randn(2, M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, groupsize, bwd_mode)
    g = torch.randn_like(out)
    out.backward(g)
    x_ref = x.detach().clone().requires_grad_(True)
    F.linear(x_ref, w_deq).backward(g)
    assert _relerr(x.grad, x_ref.grad) < tol


@requires_cuda
@requires_triton
@pytest.mark.parametrize("bwd_mode", ["bf16", "int8"])
def test_compiled_block_inductor_cuda_matches_eager(bwd_mode):
    torch._dynamo.reset()
    block, ref = _patched_block("cuda", torch.bfloat16, bwd_mode=bwd_mode)
    compiled = torch.compile(block, fullgraph=True)
    for m in (M, M + 8):
        x = torch.randn(2, m, K, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_c, gx_c, gb_c, _ = _fwd_bwd(compiled, x)
            block.zero_grad()
            out_e, gx_e, gb_e, _ = _fwd_bwd(block, x)
            block.zero_grad()
            out_r, gx_r, gb_r, _ = _fwd_bwd(ref, x)
            ref.zero_grad()
        assert out_c.dtype == out_e.dtype
        # same kernels either side: only bf16 rounding of fused glue differs
        assert _relerr(out_c, out_e) < 5e-3 and _relerr(gx_c, gx_e) < 5e-3 and _relerr(gb_c, gb_e) < 5e-3
        assert _relerr(out_c, out_r) < 5e-2 and _relerr(gx_c, gx_r) < 5e-2  # int8 activation quantization noise
