"""ConvRot int8 storage for optimizer moment state (ported from ostris/ai-toolkit toolkit/util/convrot_quant.py).

Values are rotated with the block regular-Hadamard transform along the trailing dims (the rotation spreads outliers,
so one symmetric per-row scale is safe) and stored as int8 codes with one fp32 scale per row. Quantization uses a
fused Triton kernel when Triton and CUDA are available and plain torch ops otherwise (CPU, a GPU without Triton);
both round half to even and give the same codes.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_hadamard_cache: dict = {}


def regular_hadamard(rot_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Kronecker powers of the 4x4 regular Hadamard matrix, orthonormal: symmetric, so its own inverse."""
    key = (rot_size, str(device), dtype)
    matrix = _hadamard_cache.get(key)
    if matrix is None:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"rot_size {rot_size} is not a power of 4")
        matrix = (h / rot_size**0.5).to(device=device, dtype=dtype)
        _hadamard_cache[key] = matrix
    return matrix


def largest_pow4_divisor(d: int) -> int:
    h = 1
    while d % (h * 4) == 0:
        h *= 4
    return h


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """The block regular-Hadamard rotation along the last dim (self-inverse)."""
    if rot_size == 1:
        return x
    h = regular_hadamard(rot_size, x.device, x.dtype)
    shape = x.shape
    blocks = x.reshape(-1, shape[-1] // rot_size, rot_size)
    return torch.matmul(blocks, h).reshape(shape)


def quantize_int8_rows(x: torch.Tensor, qmax: int = 127) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization to [-qmax, qmax]: (int8 (rows, K), fp32 scales (rows,))."""
    xf = x.float()
    scales = xf.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-qmax, qmax).to(torch.int8)
    return q, scales


_triton_kernel = None
_triton_failed = False


def _get_kernel():
    global _triton_kernel
    if _triton_kernel is not None:
        return _triton_kernel
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def int8_rows_quant_kernel(x_ptr, q_ptr, s_ptr, K, QMAX: tl.constexpr, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        acc = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(x_ptr + base + offs, mask=offs < K, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        amax = tl.max(acc, axis=0)
        scale = tl.where(amax > 0, amax / QMAX, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            q = libdevice.rint(v / scale)  # round half to even, like torch.round
            q = tl.minimum(tl.maximum(q, -1.0 * QMAX), 1.0 * QMAX)
            tl.store(q_ptr + base + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    _triton_kernel = int8_rows_quant_kernel
    return _triton_kernel


def quantize_int8_rows_fused(x: torch.Tensor, qmax: int = 127) -> tuple[torch.Tensor, torch.Tensor]:
    """quantize_int8_rows in one Triton pass on CUDA; the torch ops anywhere else (or if Triton fails once)."""
    global _triton_failed
    if not x.is_cuda or _triton_failed:
        return quantize_int8_rows(x, qmax)
    try:
        kernel = _get_kernel()
        rows, K = x.shape
        x = x.contiguous()
        q = torch.empty(rows, K, device=x.device, dtype=torch.int8)
        scales = torch.empty(rows, device=x.device, dtype=torch.float32)
        block_k = min(2048, 1 << (K - 1).bit_length())
        kernel[(rows,)](x, q, scales, K, QMAX=qmax, BLOCK_K=block_k, num_warps=8)
        return q, scales
    except Exception as exc:  # no Triton / compiler: the torch path gives the same codes
        _triton_failed = True
        logger.warning("ConvRot int8 optimizer state: Triton kernel unavailable (%s); using torch ops", exc)
        return quantize_int8_rows(x, qmax)


class ConvRot8State:
    """One tensor of optimizer state as rotated int8 rows: ``shape[0]`` rows of ``numel // shape[0]`` values, one
    fp32 scale per row (the ConvRot weight layout). ``dequantize()`` returns fp32 in the original shape."""

    def __init__(self, source, rot_size: int = 256):
        if isinstance(source, dict):
            self._load(source)
            return
        self.shape = source.shape
        self.numel = source.numel()
        self.rows = source.shape[0]
        self.K = self.numel // self.rows
        rot = min(rot_size, largest_pow4_divisor(self.K))
        self.rot = rot if rot >= 16 else 1
        self.quantized = None
        self.scale = None
        self.quantize_(source)

    @property
    def device(self) -> torch.device:
        return self.quantized.device

    def quantize_(self, values_fp32: torch.Tensor) -> None:
        x = rotate(values_fp32.reshape(self.rows, self.K).to(torch.float32), self.rot)
        self.quantized, self.scale = quantize_int8_rows_fused(x)

    def dequantize(self) -> torch.Tensor:
        values = self.quantized.to(torch.float32).mul_(self.scale.unsqueeze(1))
        return rotate(values, self.rot).reshape(self.shape)

    def to_(self, device) -> "ConvRot8State":
        self.quantized = self.quantized.to(device)
        self.scale = self.scale.to(device)
        return self

    def state_dict(self) -> dict:
        return {"quantized": self.quantized, "scale": self.scale, "shape": tuple(self.shape), "numel": self.numel, "rot": self.rot}

    def _load(self, state: dict) -> None:
        self.quantized = state["quantized"]
        self.scale = state["scale"].to(torch.float32)
        self.shape = torch.Size(state["shape"])
        self.numel = int(state["numel"])
        self.rows = self.shape[0]
        self.K = self.numel // self.rows
        self.rot = int(state["rot"])
