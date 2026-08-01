"""NVFP4 W4A4G4 triple-branch training path for LTX-2.

This module provides a drop-in replacement for the frozen quantized ``F.linear`` of an
LTX-2 transformer Linear, structured for LoRA fine-tuning over an NVFP4-quantized base.
It mirrors the role of ``int4_convrot_utils`` but implements the NVFP4 scheme, which
has **no activation rotation**: the weight is factored ``W ~= R + L1 @ L2`` where the
low-rank stabilizer ``L1 @ L2`` (bf16, frozen) absorbs outliers and the residual ``R`` is
stored in NVFP4 (E2M1). Forward and backward run W4A4/G4 GEMMs on quantized activations
and gradients; the trainable LoRA A/B branch stays OUTSIDE this module (added by the
existing LoRA wrapper), so this Function covers only the frozen backbone + stabilizer.

Numerics (forward / backward)::

    Y      = GEMM4(Q(X), Q(R)) + X @ (L1 @ L2)^T                 [+ LoRA branch, external]
    grad_X = GEMM4(Q(G), Q(R)^T) + G @ (L1 @ L2)                 [+ LoRA branch, external]

Only the input gradient is produced here; the base residual ``R`` and the stabilizer
``L1, L2`` are non-trainable buffers (no weight grads).

NVFP4 storage / scale layout (this is the transposability contract)
--------------------------------------------------------------------------------------
E2M1 values use the LUT ``[0, .5, 1, 1.5, 2, 3, 4, 6]`` plus sign mirrors, two nibbles
packed per ``uint8`` (low nibble = even column, high nibble = odd column), matching
``nvfp4_utils``. Two-level scaling:

  * ``weight``          uint8  ``[N_pad, K_pad // 2]``  packed E2M1 residual nibbles
  * ``nvfp4_tile_scale`` float8_e4m3fn ``[N_pad // 16, K_pad // 16]`` — **one fp8 scale per
    16x16 weight tile** (this is the key departure from inference NVFP4, which uses a
    per-``1x16`` micro-block scale). Because a whole 16x16 tile shares a single scale, the
    hardware ``1x16`` micro-block scale vectors for **both** the forward (``W``, blocked
    along K) and the backward (``W^T``, blocked along N) layouts are obtained from the tile
    scales by pure repetition — no requantization — so ``dequant(Q(R))^T`` is bitwise equal
    to the dequant of the transposed layout. See ``forward_microscales_from_tiles`` /
    ``backward_microscales_from_tiles``.
  * ``nvfp4_tensor_scale`` float32 scalar — per-tensor scale ``weight_scale_2``.
  * ``nvfp4_shape``     int32 ``[N, K, N_pad, K_pad]`` — original and 16-padded dims.
  * optional ``nvfp4_stab_l1`` bf16 ``[N, r]`` / ``nvfp4_stab_l2`` bf16 ``[r, K]``.

Activations ``X`` and grad-outputs ``G`` are quantized **dynamically** with the standard
NVFP4 per-row ``1x16`` micro-block layout (per-16-element fp8 scale + per-tensor fp32
scale). The quantization math is a single source of truth shared by the emulate and Triton
providers (``_quantize_microblocks_core`` / ``quantize_nvfp4_weight_tiles``).

Backends (env ``LTX2_NVFP4_BACKEND`` = ``auto`` | ``emulate`` | ``triton``)
--------------------------------------------------------------------------
  * ``emulate`` (default): dequantizes ``R`` and round-trips ``X``/``G``
    through the exact NVFP4 quantization above, then runs a bf16/fp32 GEMM. The
    quantization matches what a Blackwell scaled-MMA would consume (accumulation order
    aside), so it runs on any CUDA GPU or CPU for correctness testing.
  * ``triton``: Blackwell (sm >= 10) scaled-MMA kernels using ``tl.dot_scaled`` over E2M1
    inputs with fp8 (e4m3) scales. **UNTESTED on hardware** — this code path has never run
    on a Blackwell device from this repo; it is gated so it imports everywhere and raises a
    clear ``RuntimeError`` if selected without support. Every Triton API use is feature
    detected (``hasattr(tl, "dot_scaled")``).
  * ``auto``: resolves to ``emulate``. The unvalidated Triton provider requires an explicit
    ``LTX2_NVFP4_BACKEND=triton`` selection.

Backward mode (env ``LTX2_NVFP4_BACKWARD`` = ``quantized`` | ``bf16``)
----------------------------------------------------------------------
  * ``quantized`` (default): preserve the W4A4G4 input-gradient path described above.
  * ``bf16``: reconstruct the frozen residual in bf16 and use a bf16 input-gradient GEMM.
    This avoids transposing and repacking the packed weight for a scaled-MMA backward,
    at the cost of a temporary dense bf16 residual and higher-precision gradients.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------
NVFP4_TRAINING_TILE = 16  # 16x16 weight scale tile
NVFP4_TRAINING_MICRO = 16  # 1x16 activation/grad micro-block
E2M1_MAX = 6.0
E4M3_MAX = 448.0

NVFP4_TILE_SCALE_SUFFIX = ".nvfp4_tile_scale"
NVFP4_TENSOR_SCALE_SUFFIX = ".nvfp4_tensor_scale"
NVFP4_SHAPE_SUFFIX = ".nvfp4_shape"
NVFP4_STAB_L1_SUFFIX = ".nvfp4_stab_l1"
NVFP4_STAB_L2_SUFFIX = ".nvfp4_stab_l2"

# Transformer-block Linear weights are the quantization targets; the offline converter and the
# on-the-fly load path share this pattern set and the exclusion tokens passed by the caller.
NVFP4_TARGET_PATTERNS = ("transformer_blocks",)

# Safetensors metadata written by the converter; read back for pre-quantized detection and rank.
NVFP4_TRAINING_METADATA_MARKER = "nvfp4_training_quantized"
NVFP4_TRAINING_STABILIZER_RANK_METADATA = "nvfp4_training_stabilizer_rank"

# E2M1 magnitude grid and nearest-rounding midpoints (ties round toward lower magnitude).
_E2M1_MAGNITUDE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
# Signed value LUT matching nvfp4_utils.FP4_E2M1_LUT ordering (0x0-0x7 positive, 0x8-0xF neg).
_E2M1_VALUE_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _ceil_to(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


# ---------------------------------------------------------------------------
# E2M1 nibble quantization (shared core)
# ---------------------------------------------------------------------------


def _e2m1_magnitude_index(mag: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude index 0..7 for a non-negative tensor (memory-light)."""
    boundaries = _E2M1_BOUNDARIES.to(device=mag.device, dtype=torch.float32)
    idx = torch.bucketize(mag.to(torch.float32).clamp(max=E2M1_MAX), boundaries)
    return idx.to(torch.uint8)


def quantize_to_e2m1_nibbles(x_scaled: torch.Tensor) -> torch.Tensor:
    """Round already-scaled values (grid ``[-6, 6]``) to signed E2M1 nibbles ``[0, 15]``."""
    neg = (x_scaled < 0).to(torch.uint8)
    idx = _e2m1_magnitude_index(x_scaled.abs())
    return idx | (neg << 3)


def e2m1_nibbles_to_values(nibbles: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    lut = _E2M1_VALUE_LUT.to(device=nibbles.device, dtype=dtype)
    return lut[nibbles.long()]


def pack_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack a ``[..., 2m]`` uint8 nibble tensor into ``[..., m]`` bytes (low=even, high=odd)."""
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack ``[..., m]`` bytes into ``[..., num_values]`` uint8 nibbles."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :num_values]


def _per_tensor_scale(absmax: torch.Tensor | float) -> torch.Tensor:
    """weight_scale_2 style per-tensor scale so per-block fp8 scales stay in e4m3 range."""
    if not isinstance(absmax, torch.Tensor):
        absmax = torch.tensor(float(absmax))
    return (absmax.to(torch.float32) / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)


# ---------------------------------------------------------------------------
# Weight quantization: 16x16 tile scales (transposable)
# ---------------------------------------------------------------------------


@dataclass
class NVFP4LayerQuality:
    key: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    mse: float
    mae: float
    max_abs_error: float
    cosine: float
    sqnr_db: float
    signal_mean_square: float
    stabilizer_rank: int = 0


@torch.no_grad()
def quantize_nvfp4_weight_tiles(
    weight: torch.Tensor,
    *,
    stabilizer: tuple[torch.Tensor, torch.Tensor] | None = None,
    calc_device: str | torch.device = "cpu",
    collect_quality: bool = False,
    key: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, NVFP4LayerQuality | None]:
    """Quantize a bf16/fp32 weight ``[N, K]`` to packed NVFP4 with 16x16 tile scales.

    Returns ``(packed_uint8[N_pad, K_pad//2], tile_scale_fp8[N_pad//16, K_pad//16],
    tensor_scale_fp32(scalar), shape_int32[N, K, N_pad, K_pad], quality)``. When a
    ``stabilizer=(L1, L2)`` is given, the residual ``R = W - L1 @ L2`` is what gets NVFP4
    quantized (the pair is cast to bf16 before the residual is formed so stored tensors
    reconstruct exactly what was subtracted).
    """
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 training expects a 2D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    n_pad = _ceil_to(out_features, NVFP4_TRAINING_TILE)
    k_pad = _ceil_to(in_features, NVFP4_TRAINING_TILE)
    device = torch.device(calc_device)

    w = weight.to(device=device, dtype=torch.float32)
    stab_full = None
    stabilizer_rank = 0
    if stabilizer is not None:
        l1 = stabilizer[0].to(device=device, dtype=torch.float32)
        l2 = stabilizer[1].to(device=device, dtype=torch.float32)
        if l1.shape[0] != out_features or l2.shape[1] != in_features or l1.shape[1] != l2.shape[0]:
            raise ValueError(
                f"NVFP4 stabilizer shapes {tuple(l1.shape)}/{tuple(l2.shape)} do not factor weight "
                f"[out={out_features}, in={in_features}]"
            )
        stab_full = l1 @ l2
        stabilizer_rank = int(l1.shape[1])
        residual = w - stab_full
    else:
        residual = w

    r_pad = F.pad(residual, (0, k_pad - in_features, 0, n_pad - out_features))

    amax = r_pad.abs().amax().clamp(min=1e-30)
    tensor_scale = _per_tensor_scale(amax)

    tiles = r_pad.reshape(n_pad // NVFP4_TRAINING_TILE, NVFP4_TRAINING_TILE, k_pad // NVFP4_TRAINING_TILE, NVFP4_TRAINING_TILE)
    tile_amax = tiles.abs().amax(dim=(1, 3))  # [N_pad/16, K_pad/16]
    tile_scale_raw = tile_amax / E2M1_MAX
    tile_scale_fp8 = (tile_scale_raw / tensor_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)

    effective = tile_scale_fp8.to(torch.float32) * tensor_scale  # [N_pad/16, K_pad/16]
    eff_full = effective.repeat_interleave(NVFP4_TRAINING_TILE, dim=0).repeat_interleave(NVFP4_TRAINING_TILE, dim=1)
    nibbles = quantize_to_e2m1_nibbles(r_pad / eff_full.clamp(min=1e-30))
    packed = pack_e2m1(nibbles)

    shape = torch.tensor([out_features, in_features, n_pad, k_pad], dtype=torch.int32)

    quality = None
    if collect_quality:
        r_deq = e2m1_nibbles_to_values(nibbles) * eff_full  # [N_pad, K_pad]
        deq = r_deq[:out_features, :in_features]
        if stab_full is not None:
            deq = deq + stab_full
        ref = w
        err = deq - ref
        numel = max(out_features * in_features, 1)
        ref_sq = float((ref * ref).sum().item())
        deq_sq = float((deq * deq).sum().item())
        dot = float((deq * ref).sum().item())
        sqerr = float((err * err).sum().item())
        denom = math.sqrt(max(ref_sq, 0.0) * max(deq_sq, 0.0))
        cosine = float(dot / denom) if denom > 0 else 1.0
        mse = sqerr / numel
        signal = ref_sq / numel
        sqnr = float(10.0 * math.log10(signal / mse)) if mse > 0 and signal > 0 else float("inf")
        quality = NVFP4LayerQuality(
            key=key,
            shape=(out_features, in_features),
            padded_shape=(n_pad, k_pad),
            mse=mse,
            mae=float(err.abs().sum().item() / numel),
            max_abs_error=float(err.abs().max().item()),
            cosine=cosine,
            sqnr_db=sqnr,
            signal_mean_square=signal,
            stabilizer_rank=stabilizer_rank,
        )

    return packed.to(torch.uint8), tile_scale_fp8, tensor_scale.to(torch.float32).reshape(()), shape, quality


def dequantize_nvfp4_weight_tiles(
    packed: torch.Tensor,
    tile_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    n_pad: int,
    k_pad: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize the packed residual to the padded ``[N_pad, K_pad]`` weight (no stabilizer)."""
    nibbles = unpack_e2m1(packed, k_pad)
    values = e2m1_nibbles_to_values(nibbles, dtype=torch.float32)
    eff = tile_scale.to(torch.float32) * tensor_scale.to(torch.float32)
    eff_full = eff.repeat_interleave(NVFP4_TRAINING_TILE, dim=0).repeat_interleave(NVFP4_TRAINING_TILE, dim=1)
    return (values * eff_full).to(dtype)


def forward_microscales_from_tiles(tile_scale: torch.Tensor) -> torch.Tensor:
    """Weight fp8 micro-scales for the forward layout ``W`` (blocked along K): ``[N_pad, K_pad/16]``.

    Each 16x16 tile scale is repeated 16x along the N (row) dimension. Exact, no requant.
    """
    return tile_scale.repeat_interleave(NVFP4_TRAINING_TILE, dim=0)


def backward_microscales_from_tiles(tile_scale: torch.Tensor) -> torch.Tensor:
    """Weight fp8 micro-scales for the backward layout ``W^T`` (blocked along N): ``[K_pad, N_pad/16]``.

    Transpose the tile-scale grid and repeat 16x along the K dimension. Exact, no requant —
    this is why ``Q(R)^T`` reuses the same tiles (transposability).
    """
    return tile_scale.t().contiguous().repeat_interleave(NVFP4_TRAINING_TILE, dim=0)


# ---------------------------------------------------------------------------
# Activation / gradient micro-block quantization (per-row 1x16, dynamic)
# ---------------------------------------------------------------------------


def _quantize_microblocks_core(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor ``[M, K]`` (K % 16 == 0) to per-row 1x16 NVFP4 micro-blocks.

    Single source of truth shared by the emulate and Triton providers. Returns
    ``(nibbles[M, K], effective_scale[M, K], micro_scale_fp8[M, K/16], tensor_scale scalar)``.
    """
    if x2d.ndim != 2:
        raise ValueError(f"micro-block quantization expects 2D input, got {tuple(x2d.shape)}")
    m, k = x2d.shape
    if k % NVFP4_TRAINING_MICRO != 0:
        raise ValueError(f"micro-block K={k} must be a multiple of {NVFP4_TRAINING_MICRO}")
    xf = x2d.to(torch.float32)
    amax = xf.abs().amax().clamp(min=1e-30)
    tensor_scale = _per_tensor_scale(amax)
    blocks = xf.reshape(m, k // NVFP4_TRAINING_MICRO, NVFP4_TRAINING_MICRO)
    micro_amax = blocks.abs().amax(dim=2)  # [M, K/16]
    micro_scale_raw = micro_amax / E2M1_MAX
    micro_scale_fp8 = (micro_scale_raw / tensor_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    eff = micro_scale_fp8.to(torch.float32) * tensor_scale  # [M, K/16]
    eff_full = eff.repeat_interleave(NVFP4_TRAINING_MICRO, dim=1)  # [M, K]
    nibbles = quantize_to_e2m1_nibbles(xf / eff_full.clamp(min=1e-30))
    return nibbles, eff_full, micro_scale_fp8, tensor_scale.reshape(())


def quantize_dequantize_nvfp4_microblocks(x2d: torch.Tensor) -> torch.Tensor:
    """Round-trip ``x2d`` through NVFP4 micro-block quantization; returns fp32 ``[M, K]``."""
    nibbles, eff_full, _, _ = _quantize_microblocks_core(x2d)
    return e2m1_nibbles_to_values(nibbles) * eff_full


def quantize_nvfp4_microblocks(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x2d`` to packed NVFP4 micro-blocks for the Triton scaled-MMA path.

    Returns ``(packed_uint8[M, K/2], micro_scale_fp8[M, K/16], tensor_scale scalar)``.
    """
    nibbles, _, micro_scale_fp8, tensor_scale = _quantize_microblocks_core(x2d)
    return pack_e2m1(nibbles), micro_scale_fp8, tensor_scale


# ---------------------------------------------------------------------------
# SVD stabilizer (low-rank SVD outlier branch, no rotation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_nvfp4_stabilizer(
    weight: torch.Tensor,
    *,
    rank: int,
    calc_device: str | torch.device = "cpu",
    niter: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank stabilizer of the weight: ``(L1[out, rank], L2[rank, in])`` bf16.

    ``W ~= L1 @ L2 + residual``; the residual is what gets NVFP4 quantized. Unlike the INT4
    ConvRot stabilizer this operates on the plain weight (no rotation).
    """
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 stabilizer expects a 2D weight, got shape {tuple(weight.shape)}")
    if rank <= 0:
        raise ValueError(f"NVFP4 stabilizer rank must be positive, got {rank}")
    out_features, in_features = weight.shape
    device = torch.device(calc_device)
    rank = min(int(rank), out_features, in_features)
    w = weight.to(device=device, dtype=torch.float32)
    q = min(rank + 8, out_features, in_features)
    u, s, v = torch.svd_lowrank(w, q=q, niter=int(niter))
    sqrt_s = s[:rank].clamp(min=0.0).sqrt()
    l1 = (u[:, :rank] * sqrt_s.reshape(1, -1)).to(torch.bfloat16).contiguous()
    l2 = (sqrt_s.reshape(-1, 1) * v[:, :rank].t()).to(torch.bfloat16).contiguous()
    return l1, l2


# ---------------------------------------------------------------------------
# Target selection + per-tensor quantization (shared by converter and on-the-fly load)
# ---------------------------------------------------------------------------


def is_nvfp4_target_key(
    model_key: str,
    value: torch.Tensor,
    *,
    exclude_tokens: Iterable[str],
    target_patterns: Iterable[str] = NVFP4_TARGET_PATTERNS,
) -> bool:
    """Whether a (renamed) weight key is an NVFP4 quantization target.

    Single source of truth for the offline converter and the on-the-fly load path: a 2D
    ``.weight`` with at least 8 output rows whose model key matches ``target_patterns`` and
    hits none of ``exclude_tokens``.
    """
    if not (model_key.endswith(".weight") and any(t in model_key for t in target_patterns)):
        return False
    if any(e in model_key for e in exclude_tokens):
        return False
    if value.ndim != 2 or value.shape[0] < 8:
        return False
    return True


@torch.no_grad()
def quantize_nvfp4_training_tensor(
    weight: torch.Tensor,
    *,
    stabilizer_rank: int,
    calc_device: str | torch.device = "cpu",
    collect_quality: bool = False,
    key: str = "",
) -> tuple[dict[str, torch.Tensor], NVFP4LayerQuality | None]:
    """Quantize one weight to the converter's NVFP4 state-dict entries (suffix -> tensor).

    Shared by ``ltx2_quantize_nvfp4`` and the on-the-fly load path. Matching inputs, options,
    device behavior, and RNG state traverse the same quantization routine. Returns the
    per-layer entries keyed by suffix (``.weight`` for the packed residual) plus optional quality.
    """
    stabilizer = None
    if stabilizer_rank > 0:
        stabilizer = compute_nvfp4_stabilizer(weight, rank=stabilizer_rank, calc_device=calc_device)
    packed, tile_scale, tensor_scale, shape, quality = quantize_nvfp4_weight_tiles(
        weight,
        stabilizer=stabilizer,
        calc_device=calc_device,
        collect_quality=collect_quality,
        key=key,
    )
    entries: dict[str, torch.Tensor] = {
        ".weight": packed,
        NVFP4_TILE_SCALE_SUFFIX: tile_scale,
        NVFP4_TENSOR_SCALE_SUFFIX: tensor_scale,
        NVFP4_SHAPE_SUFFIX: shape,
    }
    if stabilizer is not None:
        entries[NVFP4_STAB_L1_SUFFIX] = stabilizer[0]
        entries[NVFP4_STAB_L2_SUFFIX] = stabilizer[1]
    return entries, quality


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _have_triton_scaled_dot() -> bool:
    try:
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return hasattr(tl, "dot_scaled")


def _triton_supported(device: torch.device | None) -> bool:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability(device)[0] < 10:
            return False
    except Exception:
        return False
    return _have_triton_scaled_dot()


def resolve_nvfp4_backend(device: torch.device | None) -> str:
    """Resolve the active backend without automatically selecting an unvalidated provider."""
    requested = os.getenv("LTX2_NVFP4_BACKEND", "auto").strip().lower()
    if requested in ("", "auto"):
        return "emulate"
    if requested == "emulate":
        return "emulate"
    if requested == "triton":
        if not _triton_supported(device):
            raise RuntimeError(
                "LTX2_NVFP4_BACKEND=triton requires a Blackwell GPU (compute capability >= 10.0) "
                "and a Triton build exposing tl.dot_scaled. Use 'emulate' on other hardware."
            )
        return "triton"
    raise ValueError("LTX2_NVFP4_BACKEND must be one of auto, emulate, triton")


def resolve_nvfp4_backward_mode() -> str:
    """Resolve the frozen-backbone input-gradient implementation."""
    requested = os.getenv("LTX2_NVFP4_BACKWARD", "quantized").strip().lower()
    if requested in ("", "quantized", "nvfp4"):
        return "quantized"
    if requested == "bf16":
        return "bf16"
    raise ValueError("LTX2_NVFP4_BACKWARD must be one of quantized, bf16")


def _autocast_output_dtype(x: torch.Tensor) -> torch.dtype:
    if not x.is_floating_point():
        return x.dtype
    device_type = x.device.type
    try:
        autocast_enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        autocast_enabled = torch.is_autocast_enabled()
    if not autocast_enabled:
        return x.dtype
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, RuntimeError):
        if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
            return torch.get_autocast_gpu_dtype()
        if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
            return torch.get_autocast_cpu_dtype()
    return x.dtype


# ---------------------------------------------------------------------------
# Emulation provider (numerically matches Blackwell quantization, any CUDA/CPU)
# ---------------------------------------------------------------------------


def _forward_emulate_backbone(
    x2d: torch.Tensor,
    packed_weight: torch.Tensor,
    tile_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    n_pad: int,
    k_pad: int,
) -> torch.Tensor:
    """Y_backbone = GEMM4(Q(X), Q(R)) emulated in fp32. Returns ``[M, out_features]``."""
    x_pad = F.pad(x2d, (0, k_pad - in_features)) if k_pad > in_features else x2d
    x_qdq = quantize_dequantize_nvfp4_microblocks(x_pad)  # fp32 [M, K_pad]
    r_deq = dequantize_nvfp4_weight_tiles(packed_weight, tile_scale, tensor_scale, n_pad, k_pad, dtype=torch.float32)
    y = x_qdq @ r_deq.t()  # [M, N_pad]
    return y[:, :out_features]


def _backward_emulate_backbone(
    go2d: torch.Tensor,
    packed_weight: torch.Tensor,
    tile_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    n_pad: int,
    k_pad: int,
) -> torch.Tensor:
    """grad_X_backbone = GEMM4(Q(G), Q(R)^T) emulated in fp32. Returns ``[M, in_features]``."""
    go_pad = F.pad(go2d, (0, n_pad - out_features)) if n_pad > out_features else go2d
    go_qdq = quantize_dequantize_nvfp4_microblocks(go_pad)  # fp32 [M, N_pad]
    r_deq = dequantize_nvfp4_weight_tiles(packed_weight, tile_scale, tensor_scale, n_pad, k_pad, dtype=torch.float32)
    gx = go_qdq @ r_deq  # [M, K_pad]
    return gx[:, :in_features]


def _backward_bf16_backbone(
    go2d: torch.Tensor,
    packed_weight: torch.Tensor,
    tile_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    n_pad: int,
    k_pad: int,
) -> torch.Tensor:
    """Input gradient through a dense bf16 reconstruction of the frozen residual."""
    go_pad = F.pad(go2d, (0, n_pad - out_features)) if n_pad > out_features else go2d
    r_deq = dequantize_nvfp4_weight_tiles(
        packed_weight,
        tile_scale,
        tensor_scale,
        n_pad,
        k_pad,
        dtype=torch.bfloat16,
    )
    gx = go_pad.to(torch.bfloat16) @ r_deq
    return gx[:, :in_features]


# ---------------------------------------------------------------------------
# Triton provider (Blackwell scaled-MMA) — UNTESTED on hardware.
#
# These kernels are gated (see ``_triton_supported``) so this module imports on any
# machine and the path only runs on sm >= 10 with tl.dot_scaled. They have never executed
# on a Blackwell device from this repo; numerics are validated only indirectly (the shared
# torch quantizers below feed identical packed E2M1 + fp8 scales to the emulate path). The
# dynamic X/G quantization uses the shared ``quantize_nvfp4_microblocks`` so both providers
# share one source of truth; Triton performs only the scaled-dot GEMM.
# ---------------------------------------------------------------------------

_TRITON_KERNELS: dict[str, Any] | None = None


def _build_triton_kernels() -> dict[str, Any]:
    global _TRITON_KERNELS
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    import triton
    import triton.language as tl

    if not hasattr(tl, "dot_scaled"):  # pragma: no cover - Blackwell only
        raise RuntimeError("This Triton build does not expose tl.dot_scaled (required for NVFP4 scaled-MMA)")

    @triton.jit
    def _scaled_mm_kernel(  # pragma: no cover - Blackwell only
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # C[M, N] = dot_scaled(A[M, K] e2m1, B[N, K] e2m1) with per-1x16 e4m3 scales.
        # A/B hold packed e2m1 (two nibbles per byte along K); scales index K // 16.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_bytes = K // 2
        for k0 in range(0, k_bytes, BLOCK_K // 2):
            a_bytes = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[: BLOCK_K // 2][None, :]) * stride_ak)
            b_bytes = tl.load(b_ptr + offs_n[:, None] * stride_bn + (k0 + offs_k[: BLOCK_K // 2][None, :]) * stride_bk)
            ks = (k0 * 2) // 16
            a_sc = tl.load(a_scale_ptr + offs_m[:, None] * stride_asm + (ks + offs_k[: BLOCK_K // 16][None, :]) * stride_ask)
            b_sc = tl.load(b_scale_ptr + offs_n[:, None] * stride_bsn + (ks + offs_k[: BLOCK_K // 16][None, :]) * stride_bsk)
            acc = tl.dot_scaled(a_bytes, a_sc, "e2m1", b_bytes, b_sc, "e2m1", acc=acc)
        c = acc.to(tl.float32)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

    _TRITON_KERNELS = {"scaled_mm": _scaled_mm_kernel, "triton": triton, "tl": tl}
    return _TRITON_KERNELS


def _scaled_mm_triton(
    a_packed: torch.Tensor,
    a_scale_fp8: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale_fp8: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:  # pragma: no cover - Blackwell only
    """C[M, N] = dot_scaled(A e2m1[M, K], B e2m1[N, K]) with per-1x16 e4m3 scales, fp32 out.

    UNTESTED on hardware. Both operands are packed E2M1 ``[*, K/2]`` with e4m3 scales
    ``[*, K/16]``.
    """
    kernels = _build_triton_kernels()
    triton = kernels["triton"]
    c = torch.empty((m, n), device=a_packed.device, dtype=torch.float32)
    block_m, block_n, block_k = 64, 64, 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    kernels["scaled_mm"][grid](
        a_packed,
        a_scale_fp8,
        b_packed,
        b_scale_fp8,
        c,
        m,
        n,
        k,
        a_packed.stride(0),
        a_packed.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale_fp8.stride(0),
        a_scale_fp8.stride(1),
        b_scale_fp8.stride(0),
        b_scale_fp8.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


def _forward_triton_backbone(
    x2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad
):  # pragma: no cover - Blackwell only
    x_pad = F.pad(x2d, (0, k_pad - in_features)) if k_pad > in_features else x2d
    x_packed, x_scale, x_s2 = quantize_nvfp4_microblocks(x_pad)
    w_scale = forward_microscales_from_tiles(tile_scale)  # [N_pad, K_pad/16]
    acc = _scaled_mm_triton(x_packed, x_scale, packed_weight, w_scale, x_pad.shape[0], n_pad, k_pad)
    y = acc * (x_s2.to(torch.float32) * tensor_scale.to(torch.float32))
    return y[:, :out_features]


def _backward_triton_backbone(
    go2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad
):  # pragma: no cover - Blackwell only
    go_pad = F.pad(go2d, (0, n_pad - out_features)) if n_pad > out_features else go2d
    g_packed, g_scale, g_s2 = quantize_nvfp4_microblocks(go_pad)
    # Q(R)^T consumed with micro-blocks along N: transposed packed layout + backward tile scales.
    w_t_packed = pack_e2m1(unpack_e2m1(packed_weight, k_pad).t().contiguous())
    w_scale_t = backward_microscales_from_tiles(tile_scale)  # [K_pad, N_pad/16]
    acc = _scaled_mm_triton(g_packed, g_scale, w_t_packed, w_scale_t, go_pad.shape[0], k_pad, n_pad)
    gx = acc * (g_s2.to(torch.float32) * tensor_scale.to(torch.float32))
    return gx[:, :in_features]


# ---------------------------------------------------------------------------
# Autograd Function: frozen backbone + stabilizer (LoRA branch stays external)
# ---------------------------------------------------------------------------


class _NVFP4TrainingLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, tile_scale, tensor_scale, nvfp4_shape, bias, stab_l1, stab_l2):
        out_features, in_features, n_pad, k_pad = (int(v) for v in nvfp4_shape.detach().cpu().reshape(-1).tolist())
        original_shape = x.shape
        output_dtype = _autocast_output_dtype(x)
        has_stabilizer = stab_l1 is not None and stab_l2 is not None
        x2d = x.reshape(-1, in_features)

        backend = resolve_nvfp4_backend(x.device)
        active_backend = backend
        backbone = None
        if backend == "triton":
            try:
                backbone = _forward_triton_backbone(
                    x2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad
                )
            except Exception as exc:
                if os.getenv("LTX2_NVFP4_BACKEND", "auto").strip().lower() in ("", "auto"):
                    logger.debug("NVFP4 triton forward failed; falling back to emulate: %s", exc)
                    backbone = None
                    active_backend = "emulate"
                else:
                    raise
        if backbone is None:
            backbone = _forward_emulate_backbone(
                x2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad
            )

        y = backbone
        if has_stabilizer:
            y = y + (x2d.to(torch.float32) @ stab_l2.t().to(torch.float32)) @ stab_l1.t().to(torch.float32)
        if bias is not None:
            y = y + bias.to(torch.float32)

        ctx.save_for_backward(packed_weight, tile_scale, tensor_scale, nvfp4_shape, stab_l1, stab_l2)
        ctx.has_stabilizer = has_stabilizer
        ctx.backend = active_backend
        ctx.backward_mode = resolve_nvfp4_backward_mode()
        ctx.input_dtype = x.dtype
        ctx.original_shape = original_shape
        return y.to(output_dtype).reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        if any(ctx.needs_input_grad[i] for i in (1, 2, 3, 5, 6, 7)):
            raise RuntimeError("NVFP4 training base weights and stabilizer are frozen; only the input receives gradient.")
        packed_weight, tile_scale, tensor_scale, nvfp4_shape, stab_l1, stab_l2 = ctx.saved_tensors
        out_features, in_features, n_pad, k_pad = (int(v) for v in nvfp4_shape.detach().cpu().reshape(-1).tolist())
        go2d = grad_output.reshape(-1, out_features)

        gx = None
        if ctx.backward_mode == "bf16":
            gx = _backward_bf16_backbone(
                go2d,
                packed_weight,
                tile_scale,
                tensor_scale,
                out_features,
                in_features,
                n_pad,
                k_pad,
            )
        elif ctx.backend == "triton":
            try:
                gx = _backward_triton_backbone(
                    go2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad
                )
            except Exception as exc:
                if os.getenv("LTX2_NVFP4_BACKEND", "auto").strip().lower() in ("", "auto"):
                    logger.debug("NVFP4 triton backward failed; falling back to emulate: %s", exc)
                    gx = None
                else:
                    raise
        if gx is None:
            gx = _backward_emulate_backbone(go2d, packed_weight, tile_scale, tensor_scale, out_features, in_features, n_pad, k_pad)

        if ctx.has_stabilizer:
            gx = gx + (go2d.to(torch.float32) @ stab_l1.to(torch.float32)) @ stab_l2.to(torch.float32)

        grad_input = gx.to(ctx.input_dtype).reshape(ctx.original_shape)
        return grad_input, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Module patch / buffer registration (mirrors int4_convrot)
# ---------------------------------------------------------------------------


def _module_nvfp4_shape(module: nn.Module) -> tuple[int, int, int, int]:
    shape = getattr(module, "nvfp4_shape", None)
    if shape is None:
        raise ValueError("NVFP4 training module has no nvfp4_shape buffer")
    values = shape.detach().cpu().reshape(-1).tolist()
    if len(values) != 4:
        raise ValueError(f"Expected nvfp4_shape=[out,in,out_pad,in_pad], got {values}")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def _module_nvfp4_stabilizer(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    l1 = getattr(module, "nvfp4_stab_l1", None)
    l2 = getattr(module, "nvfp4_stab_l2", None)
    if not isinstance(l1, torch.Tensor) or not isinstance(l2, torch.Tensor):
        return None
    if l1.ndim != 2 or l2.ndim != 2 or l1.shape[1] != l2.shape[0]:
        raise ValueError(f"NVFP4 stabilizer shapes {tuple(l1.shape)}/{tuple(l2.shape)} are not a rank factorization")
    return l1, l2


def nvfp4_training_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    out_features, in_features, n_pad, k_pad = _module_nvfp4_shape(self)
    stabilizer = _module_nvfp4_stabilizer(self)
    stab_l1, stab_l2 = stabilizer if stabilizer is not None else (None, None)
    return _NVFP4TrainingLinearFunction.apply(
        x, self.weight, self.nvfp4_tile_scale, self.nvfp4_tensor_scale, self.nvfp4_shape, self.bias, stab_l1, stab_l2
    )


def register_nvfp4_training_buffers(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    """Swap target Linear weights for packed NVFP4 buffers so ``load_state_dict`` assigns them."""
    registered = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_key = name + ".weight"
        tile_key = name + NVFP4_TILE_SCALE_SUFFIX
        tensor_key = name + NVFP4_TENSOR_SCALE_SUFFIX
        shape_key = name + NVFP4_SHAPE_SUFFIX
        if not (weight_key in state_dict and tile_key in state_dict and tensor_key in state_dict and shape_key in state_dict):
            continue
        packed = state_dict[weight_key]
        if packed.dtype != torch.uint8:
            continue
        if hasattr(module, "weight"):
            del module.weight
        module.register_buffer("weight", torch.zeros_like(packed), persistent=True)
        module.register_buffer("nvfp4_tile_scale", torch.zeros_like(state_dict[tile_key]), persistent=True)
        module.register_buffer("nvfp4_tensor_scale", torch.zeros_like(state_dict[tensor_key].float()), persistent=True)
        module.register_buffer("nvfp4_shape", torch.zeros_like(state_dict[shape_key].to(torch.int32)), persistent=True)
        l1_key = name + NVFP4_STAB_L1_SUFFIX
        l2_key = name + NVFP4_STAB_L2_SUFFIX
        if l1_key in state_dict and l2_key in state_dict:
            module.register_buffer("nvfp4_stab_l1", torch.zeros_like(state_dict[l1_key]), persistent=True)
            module.register_buffer("nvfp4_stab_l2", torch.zeros_like(state_dict[l2_key]), persistent=True)
        registered += 1
    return registered


def apply_nvfp4_training_monkey_patch(model: nn.Module) -> nn.Module:
    """Replace the forward of every registered NVFP4 training Linear with the triple-branch path."""
    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (hasattr(module, "nvfp4_shape") and hasattr(module, "nvfp4_tile_scale") and module.weight.dtype == torch.uint8):
            continue
        out_features, in_features, n_pad, k_pad = _module_nvfp4_shape(module)
        if n_pad != _ceil_to(out_features, NVFP4_TRAINING_TILE) or k_pad != _ceil_to(in_features, NVFP4_TRAINING_TILE):
            raise ValueError(f"NVFP4 padded dims {(n_pad, k_pad)} inconsistent with {(out_features, in_features)}")
        if module.weight.shape != (n_pad, k_pad // 2):
            raise ValueError(f"NVFP4 packed weight shape {tuple(module.weight.shape)} incompatible with padded {(n_pad, k_pad)}")
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.nvfp4_tensor_scale = module.nvfp4_tensor_scale.to(device=module.weight.device, dtype=torch.float32).reshape(())
        stabilizer = _module_nvfp4_stabilizer(module)
        if stabilizer is not None:
            module.nvfp4_stab_l1 = stabilizer[0].to(device=module.weight.device).contiguous()
            module.nvfp4_stab_l2 = stabilizer[1].to(device=module.weight.device).contiguous()
        module._nvfp4_training_quantized = True
        module.forward = nvfp4_training_linear_forward.__get__(module, type(module))
        patched += 1
    backend = "triton" if _have_triton_scaled_dot() else "emulate-only"
    logger.info("NVFP4 training: patched %d linear layers (triton available: %s)", patched, backend)
    return model


def is_nvfp4_training_module(module: nn.Module) -> bool:
    return getattr(module, "_nvfp4_training_quantized", False)


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors load_comfy_int4_convrot_state_dict)
# ---------------------------------------------------------------------------


def load_nvfp4_training_state_dict(
    model_files: str | List[str],
    *,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict[str, torch.Tensor]:
    """Load an NVFP4 training checkpoint, keeping packed weights and scales in native dtype."""
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    if isinstance(model_files, str):
        model_files = [model_files]
    sd: dict[str, torch.Tensor] = {}
    quantized = 0
    passthrough = 0
    scale_bases = set()
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = list(f.keys())
            scale_bases |= {k[: -len(NVFP4_TILE_SCALE_SUFFIX)] for k in keys if k.endswith(NVFP4_TILE_SCALE_SUFFIX)}
            for key in keys:
                if key.endswith(NVFP4_TILE_SCALE_SUFFIX):
                    base = key[: -len(NVFP4_TILE_SCALE_SUFFIX)]
                    if key_filter is not None and not key_filter(base + ".weight"):
                        continue
                    sd[key] = f.get_tensor(key)
                    continue
                if key.endswith(NVFP4_TENSOR_SCALE_SUFFIX):
                    base = key[: -len(NVFP4_TENSOR_SCALE_SUFFIX)]
                    if key_filter is not None and not key_filter(base + ".weight"):
                        continue
                    sd[key] = f.get_tensor(key).float()
                    continue
                if key.endswith(NVFP4_SHAPE_SUFFIX):
                    base = key[: -len(NVFP4_SHAPE_SUFFIX)]
                    if key_filter is not None and not key_filter(base + ".weight"):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.int32)
                    continue
                if key.endswith(NVFP4_STAB_L1_SUFFIX) or key.endswith(NVFP4_STAB_L2_SUFFIX):
                    suffix = NVFP4_STAB_L1_SUFFIX if key.endswith(NVFP4_STAB_L1_SUFFIX) else NVFP4_STAB_L2_SUFFIX
                    base = key[: -len(suffix)]
                    if key_filter is not None and not key_filter(base + ".weight"):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.bfloat16)
                    continue
                if key_filter is not None and not key_filter(key):
                    continue
                value = f.get_tensor(key)
                if key.endswith(".weight") and key[: -len(".weight")] in scale_bases and value.dtype == torch.uint8:
                    sd[key] = value
                    quantized += 1
                    continue
                if value.dtype == torch.float32 and non_quant_dtype is not None:
                    value = value.to(non_quant_dtype)
                sd[key] = value
                passthrough += 1
    logger.info("NVFP4 training checkpoint: %d packed layers, %d passthrough tensors", quantized, passthrough)
    return sd


# ---------------------------------------------------------------------------
# Pre-quantized detection + on-the-fly streaming quantization
# ---------------------------------------------------------------------------


def detect_nvfp4_training_checkpoint(model_path: str | List[str]) -> bool:
    """Whether a safetensors file is a pre-quantized NVFP4 training checkpoint (converter output).

    Detected from the ``nvfp4_training_quantized`` metadata marker written by
    ``ltx2_quantize_nvfp4``, or a key-based fallback: a ``.nvfp4_tile_scale`` /
    ``.nvfp4_tensor_scale`` / ``.nvfp4_shape`` triple alongside a packed uint8 ``.weight``. A
    plain bf16/fp16 checkpoint has none of these, so it is quantized on the fly instead.
    """
    try:
        from safetensors import safe_open

        check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(check_path, framework="pt") as f:
            meta = f.metadata()
            if meta is not None and meta.get(NVFP4_TRAINING_METADATA_MARKER) == "true":
                return True
            keys = list(f.keys())
            has_tile = any(k.endswith(NVFP4_TILE_SCALE_SUFFIX) for k in keys)
            has_tensor = any(k.endswith(NVFP4_TENSOR_SCALE_SUFFIX) for k in keys)
            has_shape = any(k.endswith(NVFP4_SHAPE_SUFFIX) for k in keys)
            if not (has_tile and has_tensor and has_shape):
                return False
            for k in keys:
                base = k[: -len(NVFP4_TILE_SCALE_SUFFIX)] if k.endswith(NVFP4_TILE_SCALE_SUFFIX) else None
                if base is not None and f.get_slice(base + ".weight").get_dtype() == "U8":
                    return True
            return False
    except Exception:
        return False


def nvfp4_checkpoint_stabilizer_rank(model_path: str | List[str]) -> Optional[int]:
    """Stabilizer rank recorded in a pre-quantized NVFP4 training checkpoint's metadata, or None."""
    try:
        from safetensors import safe_open

        check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(check_path, framework="pt") as f:
            meta = f.metadata() or {}
            raw = meta.get(NVFP4_TRAINING_STABILIZER_RANK_METADATA)
            return int(raw) if raw is not None else None
    except Exception:
        return None


def nvfp4_stabilizer_rank_conflict_warning(requested_rank: int, checkpoint_rank: Optional[int]) -> Optional[str]:
    """Warning message when a requested rank differs from a pre-quantized checkpoint's rank, else None.

    The checkpoint carries its own stabilizer, so ``--w4a4g4_stabilizer_rank`` only affects the
    on-the-fly path; passing a different rank with a pre-quantized checkpoint has no effect.
    """
    if checkpoint_rank is None or int(requested_rank) == int(checkpoint_rank):
        return None
    return (
        f"--w4a4g4_stabilizer_rank={int(requested_rank)} is ignored for a pre-quantized checkpoint: the checkpoint "
        f"carries its own rank-{int(checkpoint_rank)} stabilizer, which is used as-is."
    )


def load_safetensors_dynamic_nvfp4_training(
    model_files: str | List[str],
    *,
    target_keys: Iterable[str] = NVFP4_TARGET_PATTERNS,
    exclude_keys: Iterable[str],
    stabilizer_rank: int = 32,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    calc_device: str | torch.device = "cpu",
    key_filter: Optional[Callable[[str], bool]] = None,
    quality_report: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Stream a bf16/fp16 checkpoint and quantize targeted Linear weights to packed NVFP4.

    Produces the same state-dict schema as the offline ``ltx2_quantize_nvfp4`` converter
    (packed E2M1 residual + tile/tensor scales + shape + optional stabilizer), keyed in the
    original checkpoint namespace. Matching tensor values also require matching inputs, options,
    device behavior, and RNG state. Weights are processed one tensor at a time (no full bf16 model in
    memory); FP8 source weights are dequantized with their ``.weight_scale`` first.
    """
    from tqdm import tqdm

    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
    from musubi_tuner.utils.device_utils import clean_memory_on_device
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    if isinstance(model_files, str):
        model_files = [model_files]
    calc_device = torch.device(calc_device)
    exclude_tokens = list(exclude_keys)
    target_patterns = tuple(target_keys)
    collect_quality = bool(quality_report)
    sd: dict[str, torch.Tensor] = {}
    quality_layers: list[NVFP4LayerQuality] = []
    quantized = 0
    passthrough = 0

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            all_keys = list(f.keys())
            fp8_scale_keys = {k for k in all_keys if k.endswith(".weight_scale") or k.endswith(".input_scale")}
            if fp8_scale_keys:
                logger.info(
                    "NVFP4 on-the-fly: detected %d FP8 scale tensors; FP8 weights will be dequantized before NVFP4",
                    len(fp8_scale_keys),
                )
            for key in tqdm(all_keys, desc=f"Quantizing {os.path.basename(model_file)} to NVFP4", unit="tensor"):
                if key in fp8_scale_keys:
                    continue
                renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
                model_key = renamed if renamed is not None else key
                if key_filter is not None and not key_filter(model_key):
                    continue
                value = f.get_tensor(key)
                if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                    scale_key = key.replace(".weight", ".weight_scale")
                    if scale_key not in fp8_scale_keys:
                        raise ValueError(
                            f"NVFP4 on-the-fly source has FP8 weight without weight_scale: {key}. "
                            "Use a bf16/fp16 checkpoint or a scaled FP8 checkpoint with matching scale tensors."
                        )
                    value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)

                if is_nvfp4_target_key(model_key, value, exclude_tokens=exclude_tokens, target_patterns=target_patterns):
                    entries, quality = quantize_nvfp4_training_tensor(
                        value,
                        stabilizer_rank=stabilizer_rank,
                        calc_device=calc_device,
                        collect_quality=collect_quality,
                        key=model_key,
                    )
                    base = key[: -len(".weight")]
                    for suffix, tensor in entries.items():
                        sd[base + suffix] = tensor
                    if quality is not None:
                        quality_layers.append(quality)
                    quantized += 1
                    if calc_device.type == "cuda" and quantized % 20 == 0:
                        clean_memory_on_device(calc_device)
                else:
                    if value.is_floating_point() and value.dtype == torch.float32 and non_quant_dtype is not None:
                        value = value.to(non_quant_dtype)
                    sd[key] = value
                    passthrough += 1

    logger.info("NVFP4 on-the-fly: quantized %d Linear weights (%d passthrough tensors)", quantized, passthrough)
    if quality_layers:
        summary = summarize_nvfp4_quality(quality_layers)
        logger.info(
            "NVFP4 on-the-fly quality: min_cosine=%.6f mean_cosine=%.6f weighted_sqnr=%.2f dB",
            summary.get("min_cosine", 0.0),
            summary.get("mean_cosine", 0.0),
            summary.get("weighted_sqnr_db", 0.0),
        )
        if quality_report:
            write_nvfp4_quality_report(
                quality_report,
                source=model_files[0],
                output=None,
                options={
                    "mode": "dynamic",
                    "target_keys": list(target_patterns),
                    "exclude_keys": list(exclude_tokens),
                    "calc_device": str(calc_device),
                    "storage": "packed_e2m1_tile16_scales",
                    "stabilizer_rank": int(stabilizer_rank),
                },
                layers=quality_layers,
            )
    return sd


# ---------------------------------------------------------------------------
# Quality report helpers (offline converter)
# ---------------------------------------------------------------------------


def summarize_nvfp4_quality(layers: Iterable[NVFP4LayerQuality]) -> dict[str, Any]:
    items = list(layers)
    if not items:
        return {"num_layers": 0}
    total = sum(l.shape[0] * l.shape[1] for l in items)
    weighted_mse = sum(l.mse * l.shape[0] * l.shape[1] for l in items) / max(total, 1)
    weighted_signal = sum(l.signal_mean_square * l.shape[0] * l.shape[1] for l in items) / max(total, 1)
    weighted_sqnr = (
        float(10.0 * math.log10(weighted_signal / weighted_mse)) if weighted_mse > 0 and weighted_signal > 0 else float("inf")
    )
    return {
        "num_layers": len(items),
        "numel": total,
        "min_cosine": min(l.cosine for l in items),
        "mean_cosine": sum(l.cosine for l in items) / len(items),
        "max_mse": max(l.mse for l in items),
        "weighted_mse": weighted_mse,
        "weighted_sqnr_db": weighted_sqnr,
        "max_abs_error": max(l.max_abs_error for l in items),
    }


def write_nvfp4_quality_report(
    path: str,
    *,
    source: str | None,
    output: str | None = None,
    options: dict[str, Any] | None = None,
    layers: Iterable[NVFP4LayerQuality],
) -> dict[str, Any]:
    import json

    layer_items = list(layers)
    report = {
        "format": "ltx2_nvfp4_training_quality_v1",
        "source": source,
        "output": output,
        "options": options or {},
        "summary": summarize_nvfp4_quality(layer_items),
        "layers": [asdict(l) for l in layer_items],
    }
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
