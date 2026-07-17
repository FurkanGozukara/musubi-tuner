"""INT4 ConvRot helpers for frozen-base LTX-2 LoRA training.

This is a clean-room companion to the existing INT8 ConvRot path.  We keep
resident base weights packed as signed int4 nibbles, rotate activations online,
and run dynamic INT4 ConvRot matmuls when the CUDA extension is available.  The fallback
path unpacks to int8-valued tensors and is intended for correctness tests and
unsupported devices.
"""

from __future__ import annotations

import json
import logging
import math
import os
import types
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules.int8_convrot_utils import build_hadamard, rotate_activation, rotate_weight
from musubi_tuner.modules.int4_convrot_awq import INT4_CONVROT_AWQ_SCALE_SUFFIX

DEFAULT_INT4_CONVROT_GROUP_SIZES = (256, 64, 16)
DEFAULT_INT4_CONVROT_CLIP_MIN = 0.35
DEFAULT_INT4_CONVROT_CLIP_STEPS = 80
DEFAULT_INT4_CONVROT_CHUNK_ELEMENTS = 4 * 1024 * 1024
INT4_CONVROT_STABILIZER_L1_SUFFIX = ".int4_stabilizer_l1"
INT4_CONVROT_STABILIZER_L2_SUFFIX = ".int4_stabilizer_l2"
INT4_CONVROT_METADATA_MARKER = "int4_convrot_quantized"

logger = logging.getLogger(__name__)
_CUTLASS_INT4 = "unset"
_CUTLASS_INT8_MM = "unset"
_CUTLASS_INT4_TRANSPOSE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_CUTLASS_INT4_TRANSPOSE_CACHE_BYTES = 0
_INT_MM_MIN_CUDA_CAPABILITY = (7, 5)

# W4A4G4 / W4A8 mode-flag gate overrides (set by configure_int4cr_training_defaults for
# --w4a4g4 / --w4a8). Both stay None unless a mode flag runs the setter, so the existing
# environment/default resolution remains active when no mode flag is passed. An explicitly-set
# environment variable always wins over these overrides.
_INT4CR_ACT_BITS_OVERRIDE: int | None = None
_INT4CR_BACKEND_OVERRIDE: str | None = None
# Backward gradient-quant bit-width, decoupled from forward activation bits so --w4a4g8 can run
# an a4 forward with a g8 backward. Stays None unless a mode flag sets it; when None the getter
# falls back to the activation bits.
_INT4CR_GRAD_BITS_OVERRIDE: int | None = None


@dataclass
class Int4ConvRotLayerQuality:
    key: str
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    group_size: int
    mse: float
    mae: float
    max_abs_error: float
    cosine: float
    sqnr_db: float
    signal_mean_square: float
    q_absmax: int
    scale_min: float
    scale_max: float
    stabilizer_rank: int = 0


def _is_power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value > 1 and value % 4 == 0:
        value //= 4
    return value == 1


def parse_int4_convrot_groupsizes(value: str | int | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_INT4_CONVROT_GROUP_SIZES
    if isinstance(value, int):
        groups = (value,)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "auto":
            return DEFAULT_INT4_CONVROT_GROUP_SIZES
        groups = tuple(int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip())
    else:
        groups = tuple(int(v) for v in value)
    if not groups:
        raise ValueError("INT4 ConvRot group size list is empty")
    invalid = [g for g in groups if not _is_power_of_four(g)]
    if invalid:
        raise ValueError(f"INT4 ConvRot group sizes must be powers of 4 >= 4, got {invalid}")
    return tuple(dict.fromkeys(groups))


def best_int4_convrot_groupsize(in_features: int, groupsizes: Iterable[int] | None = None) -> int:
    """Resolve a ConvRot group size.

    Unlike the INT8 path, INT4 ConvRot pads internally, so divisibility is not a
    requirement.  For auto lists, prefer the largest group not exceeding the
    input width; if all groups are larger, use the smallest provided group.
    """

    candidates = sorted(parse_int4_convrot_groupsizes(groupsizes), reverse=True)
    for group_size in candidates:
        if group_size <= in_features:
            return group_size
    return candidates[-1]


def padded_features_for_group(in_features: int, group_size: int) -> int:
    if group_size <= 0:
        return in_features
    return int(math.ceil(in_features / group_size) * group_size)


def pad_last_dim(x: torch.Tensor, padded_features: int) -> torch.Tensor:
    features = x.shape[-1]
    if features == padded_features:
        return x
    if features > padded_features:
        raise ValueError(f"features={features} exceeds padded_features={padded_features}")
    return F.pad(x, (0, padded_features - features))


def rotate_activation_padded(
    x: torch.Tensor,
    group_size: int,
    padded_features: int | None = None,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    features = x.shape[-1]
    if padded_features is None:
        padded_features = padded_features_for_group(features, group_size)
    padded = pad_last_dim(x, padded_features)
    h = build_hadamard(group_size, device=padded.device, dtype=padded.dtype)
    return rotate_activation(padded, h, group_size, inverse=inverse)


def rotate_weight_padded(weight: torch.Tensor, group_size: int, *, inverse: bool = False) -> torch.Tensor:
    out_features, in_features = weight.shape
    padded_features = padded_features_for_group(in_features, group_size)
    padded = pad_last_dim(weight, padded_features)
    h = build_hadamard(group_size, device=padded.device, dtype=padded.dtype)
    return rotate_weight(padded.reshape(out_features, padded_features), h, group_size, inverse=inverse)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values stored in int8 into uint8 nibbles.

    Even elements use the low nibble, odd elements use the high nibble.  Values
    are interpreted as two's-complement signed 4-bit numbers on unpack.
    """

    q = q.to(torch.int8)
    if q.numel() == 0:
        return torch.empty((*q.shape[:-1], 0), dtype=torch.uint8, device=q.device)
    q = q.clamp(-8, 7)
    features = q.shape[-1]
    if features % 2:
        q = F.pad(q, (0, 1))
        features += 1
    low = (q[..., 0::2].to(torch.int16) & 0x0F).to(torch.uint8)
    high = (q[..., 1::2].to(torch.int16) & 0x0F).to(torch.uint8)
    return low | (high << 4)


def unpack_int4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack uint8 nibbles into signed int8 values."""

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.int8, device=packed.device)
    out[..., 0::2] = ((low.to(torch.int16) ^ 0x08) - 0x08).to(torch.int8)
    out[..., 1::2] = ((high.to(torch.int16) ^ 0x08) - 0x08).to(torch.int8)
    return out[..., :num_values]


def quantize_int4_rowwise(
    x: torch.Tensor,
    *,
    mse_clip: bool = True,
    clip_min: float = DEFAULT_INT4_CONVROT_CLIP_MIN,
    clip_steps: int = DEFAULT_INT4_CONVROT_CLIP_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mse_clip or clip_steps <= 1:
        scale = (absmax / 7.0).clamp(min=1e-30)
        q = (x / scale).round().clamp(-7, 7).to(torch.int8)
        return pack_int4(q), scale.float()

    best_mse = torch.full_like(absmax, float("inf"), dtype=torch.float32)
    best_scale = (absmax / 7.0).float()
    best_q: torch.Tensor | None = None
    for ratio in torch.linspace(float(clip_min), 1.0, int(clip_steps), device=x.device, dtype=torch.float32):
        scale = (absmax.float() * ratio / 7.0).clamp(min=1e-30)
        q = (x.float() / scale).round().clamp(-7, 7).to(torch.int8)
        mse = ((q.float() * scale - x.float()) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better.expand_as(q), q, best_q)
    assert best_q is not None
    return pack_int4(best_q), best_scale.float()


def dequantize_int4_convrot_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    in_features: int,
    padded_features: int,
    *,
    dtype: torch.dtype = torch.float32,
    stabilizer: tuple[torch.Tensor, torch.Tensor] | None = None,
    rotate: bool = True,
) -> torch.Tensor:
    q = unpack_int4(packed, padded_features).float()
    deq_rot = q * scale.float()
    if stabilizer is not None:
        stab_l1, stab_l2 = stabilizer
        deq_rot = deq_rot + stab_l1.to(device=deq_rot.device, dtype=torch.float32) @ stab_l2.to(
            device=deq_rot.device, dtype=torch.float32
        )
    if rotate:
        h = build_hadamard(group_size, device=deq_rot.device, dtype=torch.float32)
        dense_padded = rotate_weight(deq_rot, h, group_size, inverse=True)
    else:
        dense_padded = deq_rot
    return dense_padded[:, :in_features].to(dtype)


@torch.no_grad()
def compute_int4_convrot_stabilizer(
    weight: torch.Tensor,
    *,
    group_size: int,
    rank: int,
    calc_device: str | torch.device = "cpu",
    niter: int = 4,
    max_chunk_elements: int | None = None,
    rotate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank stabilizer of the (optionally rotated) weight (low-rank SVD outlier branch).

    Returns ``(L1, L2)`` in bfloat16 with shapes ``[out, rank]`` and ``[rank, padded]``
    such that ``rotate(pad(weight)) ~= L1 @ L2 + residual`` (or ``pad(weight)`` when
    ``rotate`` is False); the residual is what gets INT4-quantized. The pair is cast to
    bfloat16 *before* the residual is formed so the stored tensors reconstruct exactly
    what was subtracted. In no-rotation mode the stabilizer is the only
    outlier-isolation mechanism, so it is computed on the raw padded weight.
    """

    if weight.ndim != 2:
        raise ValueError(f"INT4 ConvRot stabilizer expects a 2D weight, got shape {tuple(weight.shape)}")
    if rank <= 0:
        raise ValueError(f"INT4 ConvRot stabilizer rank must be positive, got {rank}")
    out_features, in_features = weight.shape
    padded_features = padded_features_for_group(in_features, group_size)
    if max_chunk_elements is None:
        max_chunk_elements = DEFAULT_INT4_CONVROT_CHUNK_ELEMENTS
    calc_device = torch.device(calc_device)
    rank = min(int(rank), out_features, padded_features)

    h = build_hadamard(group_size, device=calc_device, dtype=torch.float32) if rotate else None
    rotated = torch.empty((out_features, padded_features), device=calc_device, dtype=torch.float32)
    for row_slice in _row_slices(out_features, padded_features, int(max_chunk_elements)):
        w_chunk = weight[row_slice].to(device=calc_device, dtype=torch.float32)
        w_padded = pad_last_dim(w_chunk, padded_features)
        rotated[row_slice] = rotate_weight(w_padded, h, group_size) if rotate else w_padded

    q = min(rank + 8, out_features, padded_features)
    u, s, v = torch.svd_lowrank(rotated, q=q, niter=int(niter))
    sqrt_s = s[:rank].clamp(min=0.0).sqrt()
    stab_l1 = (u[:, :rank] * sqrt_s.reshape(1, -1)).to(torch.bfloat16).contiguous()
    stab_l2 = (sqrt_s.reshape(-1, 1) * v[:, :rank].t()).to(torch.bfloat16).contiguous()
    return stab_l1, stab_l2


def _row_slices(num_rows: int, in_features: int, max_chunk_elements: int) -> Iterable[slice]:
    rows_per_chunk = max(1, int(max_chunk_elements) // max(int(in_features), 1))
    for start in range(0, num_rows, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, num_rows))


def _quality_from_accumulators(
    *,
    key: str,
    shape: tuple[int, int],
    padded_shape: tuple[int, int],
    group_size: int,
    dot: float,
    ref_norm_sq: float,
    deq_norm_sq: float,
    sqerr_sum: float,
    abserr_sum: float,
    max_abs_error: float,
    q_absmax: int,
    scale_min: float,
    scale_max: float,
    stabilizer_rank: int = 0,
) -> Int4ConvRotLayerQuality:
    numel = max(shape[0] * shape[1], 1)
    denom = math.sqrt(max(ref_norm_sq, 0.0) * max(deq_norm_sq, 0.0))
    cosine = float(dot / denom) if denom > 0 else 1.0
    mse = float(sqerr_sum / numel)
    signal_mean_square = float(ref_norm_sq / numel)
    sqnr_db = float(10.0 * math.log10(signal_mean_square / mse)) if mse > 0 and signal_mean_square > 0 else float("inf")
    return Int4ConvRotLayerQuality(
        key=key,
        shape=shape,
        padded_shape=padded_shape,
        group_size=int(group_size),
        mse=mse,
        mae=float(abserr_sum / numel),
        max_abs_error=float(max_abs_error),
        cosine=cosine,
        sqnr_db=sqnr_db,
        signal_mean_square=signal_mean_square,
        q_absmax=int(q_absmax),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        stabilizer_rank=int(stabilizer_rank),
    )


@torch.no_grad()
def quantize_int4_convrot_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    calc_device: str | torch.device = "cpu",
    mse_clip: bool = True,
    collect_quality: bool = False,
    key: str = "",
    max_chunk_elements: int | None = None,
    stabilizer: tuple[torch.Tensor, torch.Tensor] | None = None,
    rotate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Int4ConvRotLayerQuality | None]:
    if weight.ndim != 2:
        raise ValueError(f"INT4 ConvRot expects a 2D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    padded_features = padded_features_for_group(in_features, group_size)
    if max_chunk_elements is None:
        max_chunk_elements = DEFAULT_INT4_CONVROT_CHUNK_ELEMENTS
    calc_device = torch.device(calc_device)

    packed_out = torch.empty((out_features, padded_features // 2), device=calc_device, dtype=torch.uint8)
    scale_out = torch.empty((out_features, 1), device=calc_device, dtype=torch.float32)
    shape = torch.tensor([out_features, in_features, padded_features], device=calc_device, dtype=torch.int32)
    h = build_hadamard(group_size, device=calc_device, dtype=torch.float32) if rotate else None

    stab_l1_f32 = stab_l2_f32 = None
    stabilizer_rank = 0
    if stabilizer is not None:
        stab_l1_f32 = stabilizer[0].to(device=calc_device, dtype=torch.float32)
        stab_l2_f32 = stabilizer[1].to(device=calc_device, dtype=torch.float32)
        if stab_l1_f32.shape[0] != out_features or stab_l2_f32.shape[1] != padded_features:
            raise ValueError(
                f"INT4 ConvRot stabilizer shapes {tuple(stab_l1_f32.shape)}/{tuple(stab_l2_f32.shape)} "
                f"do not match weight [out={out_features}, padded={padded_features}]"
            )
        if stab_l1_f32.shape[1] != stab_l2_f32.shape[0]:
            raise ValueError("INT4 ConvRot stabilizer L1/L2 inner ranks differ")
        stabilizer_rank = int(stab_l1_f32.shape[1])

    dot = ref_norm_sq = deq_norm_sq = sqerr_sum = abserr_sum = 0.0
    max_abs_error = 0.0
    q_absmax = 0
    scale_min = float("inf")
    scale_max = 0.0

    for row_slice in _row_slices(out_features, padded_features, int(max_chunk_elements)):
        w_chunk = weight[row_slice].to(device=calc_device, dtype=torch.float32)
        w_padded = pad_last_dim(w_chunk, padded_features)
        rotated = rotate_weight(w_padded, h, group_size) if rotate else w_padded
        stab_chunk = None
        if stab_l1_f32 is not None:
            stab_chunk = stab_l1_f32[row_slice] @ stab_l2_f32
            rotated = rotated - stab_chunk
        q_packed, scale_chunk = quantize_int4_rowwise(rotated, mse_clip=mse_clip)
        packed_out[row_slice].copy_(q_packed)
        scale_out[row_slice].copy_(scale_chunk)

        if collect_quality:
            q_unpacked = unpack_int4(q_packed, padded_features).float()
            deq_rot = q_unpacked * scale_chunk
            if stab_chunk is not None:
                deq_rot = deq_rot + stab_chunk
            deq = (rotate_weight(deq_rot, h, group_size, inverse=True) if rotate else deq_rot)[:, :in_features]
            ref = w_chunk
            err = deq - ref
            dot += float((deq * ref).sum().item())
            ref_norm_sq += float((ref * ref).sum().item())
            deq_norm_sq += float((deq * deq).sum().item())
            sqerr_sum += float((err * err).sum().item())
            abserr_sum += float(err.abs().sum().item())
            max_abs_error = max(max_abs_error, float(err.abs().max().item()))
            q_absmax = max(q_absmax, int(q_unpacked.abs().max().item()))
            scale_min = min(scale_min, float(scale_chunk.min().item()))
            scale_max = max(scale_max, float(scale_chunk.max().item()))

    quality = None
    if collect_quality:
        quality = _quality_from_accumulators(
            key=key,
            shape=(out_features, in_features),
            padded_shape=(out_features, padded_features),
            group_size=group_size,
            dot=dot,
            ref_norm_sq=ref_norm_sq,
            deq_norm_sq=deq_norm_sq,
            sqerr_sum=sqerr_sum,
            abserr_sum=abserr_sum,
            max_abs_error=max_abs_error,
            q_absmax=q_absmax,
            scale_min=scale_min if scale_min != float("inf") else 0.0,
            scale_max=scale_max,
            stabilizer_rank=stabilizer_rank,
        )
    return packed_out, scale_out, shape, quality


def comfy_quant_tensor(
    group_size: int,
    in_features: int,
    padded_features: int,
    *,
    convrot: bool = True,
    awq: bool = False,
    stabilizer_rank: int = 0,
) -> torch.Tensor:
    cfg = {
        "format": "int4_tensorwise",
        "bits": 4,
        "convrot": bool(convrot),
        "convrot_groupsize": int(group_size),
        "in_features": int(in_features),
        "padded_in_features": int(padded_features),
        "packing": "signed_low_high",
        "awq": bool(awq),
    }
    if stabilizer_rank > 0:
        cfg["stabilizer_rank"] = int(stabilizer_rank)
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)


def parse_comfy_quant_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    data = bytes(int(v) for v in tensor.detach().cpu().flatten().tolist()).decode("utf-8")
    return json.loads(data)


def summarize_quality(layers: Iterable[Int4ConvRotLayerQuality]) -> dict[str, Any]:
    items = list(layers)
    if not items:
        return {"num_layers": 0}
    total_numel = sum(layer.shape[0] * layer.shape[1] for layer in items)
    weighted_mse = sum(layer.mse * layer.shape[0] * layer.shape[1] for layer in items) / max(total_numel, 1)
    weighted_signal = sum(layer.signal_mean_square * layer.shape[0] * layer.shape[1] for layer in items) / max(total_numel, 1)
    weighted_sqnr_db = (
        float(10.0 * math.log10(weighted_signal / weighted_mse)) if weighted_mse > 0 and weighted_signal > 0 else float("inf")
    )
    return {
        "num_layers": len(items),
        "numel": total_numel,
        "min_cosine": min(layer.cosine for layer in items),
        "mean_cosine": sum(layer.cosine for layer in items) / len(items),
        "max_mse": max(layer.mse for layer in items),
        "weighted_mse": weighted_mse,
        "weighted_sqnr_db": weighted_sqnr_db,
        "max_abs_error": max(layer.max_abs_error for layer in items),
        "groupsizes": sorted({layer.group_size for layer in items}),
    }


def write_quality_report(
    path: str,
    *,
    source: str | None,
    output: str | None = None,
    options: dict[str, Any] | None = None,
    layers: Iterable[Int4ConvRotLayerQuality],
) -> dict[str, Any]:
    layer_items = list(layers)
    report = {
        "format": "ltx2_int4_convrot_quality_v1",
        "source": source,
        "output": output,
        "options": options or {},
        "summary": summarize_quality(layer_items),
        "layers": [asdict(layer) for layer in layer_items],
    }
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def _module_int4_shape(module: nn.Module) -> tuple[int, int, int]:
    shape = getattr(module, "int4_shape", None)
    if shape is None:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("INT4 module has no int4_shape or weight")
        return int(module.out_features), int(module.in_features), int(weight.shape[1] * 2)
    values = shape.detach().cpu().reshape(-1).tolist() if isinstance(shape, torch.Tensor) else list(shape)
    if len(values) != 3:
        raise ValueError(f"Expected int4_shape=[out,in,padded], got {values}")
    return int(values[0]), int(values[1]), int(values[2])


def _module_int4_group(module: nn.Module) -> int:
    value = getattr(module, "int4_convrot_groupsize", None)
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item()) if value.numel() else 0
    return int(value or 0)


def _module_int4_rotate(module: nn.Module) -> bool:
    """Whether this layer uses the online ConvRot Hadamard rotation.

    Legacy checkpoints (and all rotated exports) carry no ``int4_rotation`` buffer, so
    absence of the buffer selects the established rotated path.
    No-rotation exports register ``int4_rotation`` = 0.
    """

    value = getattr(module, "int4_rotation", None)
    if isinstance(value, torch.Tensor):
        return bool(int(value.detach().reshape(-1)[0].item())) if value.numel() else True
    if value is None:
        return True
    return bool(value)


def _module_int4_awq_scales(module: nn.Module, in_features: int) -> torch.Tensor | None:
    scales = getattr(module, "int4_awq_scales", None)
    if not isinstance(scales, torch.Tensor):
        return None
    scales = scales.reshape(-1)
    if scales.numel() != int(in_features):
        raise ValueError(f"INT4 ConvRot AWQ scale length {scales.numel()} does not match in_features={in_features}")
    if not torch.isfinite(scales.float()).all() or (scales.float() <= 0).any():
        raise ValueError("INT4 ConvRot AWQ scales must be positive finite values")
    return scales


def _module_int4_stabilizer(module: nn.Module, out_features: int, padded_features: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    stab_l1 = getattr(module, "int4_stabilizer_l1", None)
    stab_l2 = getattr(module, "int4_stabilizer_l2", None)
    if not isinstance(stab_l1, torch.Tensor) or not isinstance(stab_l2, torch.Tensor):
        return None
    if stab_l1.ndim != 2 or stab_l2.ndim != 2 or stab_l1.shape[1] != stab_l2.shape[0]:
        raise ValueError(
            f"INT4 ConvRot stabilizer shapes {tuple(stab_l1.shape)}/{tuple(stab_l2.shape)} are not a rank factorization"
        )
    if stab_l1.shape[0] != int(out_features) or stab_l2.shape[1] != int(padded_features):
        raise ValueError(
            f"INT4 ConvRot stabilizer shapes {tuple(stab_l1.shape)}/{tuple(stab_l2.shape)} "
            f"do not match layer [out={out_features}, padded={padded_features}]"
        )
    return stab_l1, stab_l2


def _apply_int4_awq_activation_scale(module: nn.Module, x: torch.Tensor, in_features: int) -> torch.Tensor:
    scales = _module_int4_awq_scales(module, in_features)
    if scales is None:
        return x
    scale_dtype = x.dtype if x.is_floating_point() and x.dtype != torch.float32 else torch.float32
    scales = scales.to(device=x.device, dtype=scale_dtype).reshape(*([1] * (x.ndim - 1)), in_features)
    return x / scales


def _get_cuda_int4():
    try:
        from musubi_tuner.modules import cuda_int4_convrot
    except Exception:
        return None
    return cuda_int4_convrot if cuda_int4_convrot.is_available() else None


def _int_mm_allow_small_m(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch._int_mm may reject tiny M on some CUDA builds; pad those cases."""

    if a.size(0) > 16:
        return torch._int_mm(a, b)
    pad_rows = 17 - a.size(0)
    padded = F.pad(a, (0, 0, 0, pad_rows))
    return torch._int_mm(padded, b)[: a.size(0)]


def _get_cutlass_int8_mm():
    global _CUTLASS_INT8_MM
    if _CUTLASS_INT8_MM == "unset":
        try:
            from musubi_tuner.modules.cutlass_int8 import int8_mm, is_available

            _CUTLASS_INT8_MM = int8_mm if is_available() else None
        except Exception as exc:
            logger.info("INT4 ConvRot CUTLASS bridge unavailable: %s", exc)
            _CUTLASS_INT8_MM = None
    return _CUTLASS_INT8_MM


def _get_cutlass_int4():
    global _CUTLASS_INT4
    if _CUTLASS_INT4 == "unset":
        try:
            from musubi_tuner.modules import cutlass_int4

            _CUTLASS_INT4 = cutlass_int4 if cutlass_int4.is_available() else None
        except Exception as exc:
            logger.info("INT4 ConvRot native CUTLASS unavailable: %s", exc)
            _CUTLASS_INT4 = None
    return _CUTLASS_INT4


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _use_weight_only_diagnostic() -> bool:
    return _env_flag_enabled("LTX2_INT4_CONVROT_WEIGHT_ONLY", default=False)


def _cutlass_int4_transpose_cache_limit_bytes() -> int:
    value = os.getenv("LTX2_INT4_CUTLASS_TRANSPOSE_CACHE_MAX_MB")
    if value is None or not value.strip():
        return 0
    try:
        return max(0, int(float(value.strip()) * 1024 * 1024))
    except ValueError:
        logger.warning("Ignoring invalid LTX2_INT4_CUTLASS_TRANSPOSE_CACHE_MAX_MB=%r", value)
        return 0


def _cutlass_int4_transposed_weight(cutlass_int4, packed_weight: torch.Tensor, padded_features: int) -> torch.Tensor:
    if not _env_flag_enabled("LTX2_INT4_CUTLASS_TRANSPOSE_CACHE", default=False):
        return cutlass_int4.transpose_packed(packed_weight, padded_features)

    global _CUTLASS_INT4_TRANSPOSE_CACHE_BYTES
    key = (
        packed_weight.untyped_storage().data_ptr(),
        int(packed_weight.storage_offset()),
        str(packed_weight.device),
        tuple(int(v) for v in packed_weight.shape),
        tuple(int(v) for v in packed_weight.stride()),
        int(padded_features),
    )
    cached = _CUTLASS_INT4_TRANSPOSE_CACHE.get(key)
    if cached is not None:
        return cached

    weight_t = cutlass_int4.transpose_packed(packed_weight, padded_features)
    limit = _cutlass_int4_transpose_cache_limit_bytes()
    cache_bytes = int(weight_t.numel() * weight_t.element_size())
    if limit <= 0 or _CUTLASS_INT4_TRANSPOSE_CACHE_BYTES + cache_bytes <= limit:
        _CUTLASS_INT4_TRANSPOSE_CACHE[key] = weight_t
        _CUTLASS_INT4_TRANSPOSE_CACHE_BYTES += cache_bytes
    return weight_t


def _int4_backend_request() -> str:
    raw = os.getenv("LTX2_INT4_CONVROT_BACKEND")
    if (raw is None or not raw.strip()) and _INT4CR_BACKEND_OVERRIDE is not None:
        return _INT4CR_BACKEND_OVERRIDE
    return (raw if raw is not None else "auto").strip().lower()


def _int4_activation_bits() -> int:
    raw = os.getenv("LTX2_INT4_CONVROT_ACT_BITS")
    if raw is None or not raw.strip():
        if _INT4CR_ACT_BITS_OVERRIDE is not None:
            return _INT4CR_ACT_BITS_OVERRIDE
        raw = "8"
    value = raw.strip().lower()
    aliases = {
        "": 8,
        "4": 4,
        "a4": 4,
        "w4a4": 4,
        "int4": 4,
        "8": 8,
        "a8": 8,
        "w4a8": 8,
        "int8": 8,
    }
    if value in aliases:
        return aliases[value]
    raise ValueError("LTX2_INT4_CONVROT_ACT_BITS must be 4 or 8")


def _int4_gradient_bits() -> int:
    """Bit-width used to quantize grad_output in the backward GEMM.

    An explicit ``LTX2_INT4_CONVROT_GRAD_BITS`` env var wins; otherwise the mode-flag override
    (set for --w4a4g8) is used; otherwise it falls back to the forward activation bits, so a run
    with no gradient override behaves exactly as before (w4a4g4 -> g4, w4a8 -> g8).
    """
    raw = os.getenv("LTX2_INT4_CONVROT_GRAD_BITS")
    if raw is not None and raw.strip():
        value = raw.strip().lower().lstrip("g")
        aliases = {"4": 4, "a4": 4, "w4a4": 4, "int4": 4, "8": 8, "a8": 8, "w4a8": 8, "int8": 8}
        if value in aliases:
            return aliases[value]
        raise ValueError("LTX2_INT4_CONVROT_GRAD_BITS must be 4 or 8")
    if _INT4CR_GRAD_BITS_OVERRIDE is not None:
        return _INT4CR_GRAD_BITS_OVERRIDE
    return _int4_activation_bits()


def _supports_torch_int_mm(device: torch.device) -> bool:
    if not hasattr(torch, "_int_mm"):
        return False
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= _INT_MM_MIN_CUDA_CAPABILITY


def _supports_wmma_int4_mm(a_packed: torch.Tensor, b_t_packed: torch.Tensor, k_values: int) -> bool:
    return (
        a_packed.dim() == 2
        and b_t_packed.dim() == 2
        and a_packed.size(0) % 8 == 0
        and b_t_packed.size(0) % 8 == 0
        and int(k_values) % 32 == 0
    )


def _wmma_hybrid_max_rows() -> int:
    value = os.getenv("LTX2_INT4_WMMA_HYBRID_MAX_ROWS")
    if value is None or not value.strip():
        return 128
    try:
        return max(0, int(value.strip()))
    except ValueError:
        logger.warning("Ignoring invalid LTX2_INT4_WMMA_HYBRID_MAX_ROWS=%r", value)
        return 128


def _resolve_tensorcore_backend(device: torch.device | None) -> str | None:
    if device is None or device.type != "cuda":
        return None
    requested = _int4_backend_request()
    if requested in {"", "auto"}:
        return "torch" if _supports_torch_int_mm(device) else None
    if requested in {"scalar", "cuda", "naive"}:
        return None
    if requested in {"torch", "int_mm", "tensorcore"}:
        if not _supports_torch_int_mm(device):
            raise RuntimeError("LTX2_INT4_CONVROT_BACKEND=torch requires CUDA torch._int_mm support on SM 7.5+")
        return "torch"
    if requested in {"wmma", "cuda_wmma", "native_wmma"}:
        if _get_cuda_int4() is None:
            raise RuntimeError("LTX2_INT4_CONVROT_BACKEND=wmma requires the CUDA INT4 ConvRot extension")
        return "wmma"
    if requested in {"wmma_hybrid", "hybrid_wmma"}:
        if _get_cuda_int4() is None:
            raise RuntimeError("LTX2_INT4_CONVROT_BACKEND=wmma_hybrid requires the CUDA INT4 ConvRot extension")
        if not _supports_torch_int_mm(device):
            raise RuntimeError("LTX2_INT4_CONVROT_BACKEND=wmma_hybrid requires CUDA torch._int_mm support on SM 7.5+")
        return "wmma_hybrid"
    if requested in {"cutlass", "cutlass_int4", "native_cutlass"}:
        if _get_cutlass_int4() is None:
            raise RuntimeError(
                "LTX2_INT4_CONVROT_BACKEND=cutlass requires the native CUTLASS int4 extension. "
                "Set LTX2_CUTLASS_INCLUDE_DIR to a CUTLASS include tree."
            )
        return "cutlass"
    if requested in {"cutlass_int8", "int8_cutlass", "cutlass_bridge"}:
        if _get_cutlass_int8_mm() is None:
            raise RuntimeError(
                "LTX2_INT4_CONVROT_BACKEND=cutlass_int8 requires the CUTLASS int8 bridge extension. "
                "Set LTX2_CUTLASS_INCLUDE_DIR to a CUTLASS include tree."
            )
        return "cutlass_int8"
    raise ValueError("LTX2_INT4_CONVROT_BACKEND must be one of auto, torch, wmma, wmma_hybrid, cutlass, cutlass_int8, scalar")


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


def _linear_int4_tensorcore(
    qx: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    k_values: int,
) -> torch.Tensor | None:
    cuda_int4 = _get_cuda_int4() if qx.is_cuda else None
    backend = _resolve_tensorcore_backend(qx.device) if cuda_int4 is not None else None
    if backend is None:
        return None
    try:
        if backend == "cutlass":
            cutlass_int4 = _get_cutlass_int4()
            if _env_flag_enabled("LTX2_INT4_CUTLASS_SINGLE_CALL", default=False):
                return cutlass_int4.linear(
                    qx,
                    packed_weight,
                    x_scale.reshape(-1),
                    weight_scale.reshape(-1),
                    bias.float() if bias is not None else None,
                    out_dtype,
                    k_values,
                )
            acc = cutlass_int4.int4_mm(qx.contiguous(), packed_weight.contiguous(), k_values)
            return cuda_int4.dequant_epilogue(
                acc,
                x_scale.reshape(-1),
                weight_scale.reshape(-1),
                bias.float() if bias is not None else None,
                out_dtype,
            )
        if backend in {"wmma", "wmma_hybrid"}:
            use_wmma = _supports_wmma_int4_mm(qx, packed_weight, k_values)
            if backend == "wmma_hybrid":
                use_wmma = use_wmma and qx.size(0) <= _wmma_hybrid_max_rows()
            if use_wmma:
                acc = cuda_int4.wmma_int4_mm(qx.contiguous(), packed_weight.contiguous(), k_values)
                return cuda_int4.dequant_epilogue(
                    acc,
                    x_scale.reshape(-1),
                    weight_scale.reshape(-1),
                    bias.float() if bias is not None else None,
                    out_dtype,
                )
            logger.debug(
                "INT4 ConvRot WMMA backend falling back to torch bridge for shape M=%d N=%d K=%d",
                qx.size(0),
                packed_weight.size(0),
                k_values,
            )
        x_int8 = cuda_int4.unpack_to_int8(qx, k_values)
        w_int8 = cuda_int4.unpack_to_int8(packed_weight, k_values)
        if backend == "cutlass_int8":
            acc = _get_cutlass_int8_mm()(x_int8.contiguous(), w_int8.contiguous())
        else:
            acc = _int_mm_allow_small_m(x_int8.contiguous(), w_int8.t())
        return cuda_int4.dequant_epilogue(
            acc,
            x_scale.reshape(-1),
            weight_scale.reshape(-1),
            bias.float() if bias is not None else None,
            out_dtype,
        )
    except Exception as exc:
        if _int4_backend_request() in {"", "auto"}:
            logger.debug("INT4 ConvRot tensor-core bridge failed; falling back to scalar CUDA path: %s", exc)
            return None
        raise


def _grad_input_int4_tensorcore(
    qg: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out_features: int,
    padded_features: int,
) -> torch.Tensor | None:
    cuda_int4 = _get_cuda_int4() if qg.is_cuda else None
    backend = _resolve_tensorcore_backend(qg.device) if cuda_int4 is not None else None
    if backend is None:
        return None
    try:
        if backend == "cutlass":
            cutlass_int4 = _get_cutlass_int4()
            if _env_flag_enabled("LTX2_INT4_CUTLASS_TRANSPOSE_CACHE", default=False):
                weight_t = _cutlass_int4_transposed_weight(cutlass_int4, packed_weight, padded_features)
                if _env_flag_enabled("LTX2_INT4_CUTLASS_SINGLE_CALL", default=False):
                    return cutlass_int4.linear(qg, weight_t, grad_scale.reshape(-1), None, None, out_dtype, out_features)
                acc = cutlass_int4.int4_mm(qg.contiguous(), weight_t.contiguous(), out_features)
                return cuda_int4.dequant_epilogue(acc, grad_scale.reshape(-1), None, None, out_dtype)
            if _env_flag_enabled("LTX2_INT4_CUTLASS_SINGLE_CALL", default=False):
                return cutlass_int4.linear_backward_input(
                    qg,
                    packed_weight,
                    grad_scale.reshape(-1),
                    out_dtype,
                    out_features,
                    padded_features,
                )
            weight_t = cutlass_int4.transpose_packed(packed_weight, padded_features)
            acc = cutlass_int4.int4_mm(qg.contiguous(), weight_t.contiguous(), out_features)
            return cuda_int4.dequant_epilogue(acc, grad_scale.reshape(-1), None, None, out_dtype)
        if backend in {"wmma", "wmma_hybrid"}:
            use_wmma = qg.dim() == 2 and qg.size(0) % 8 == 0 and padded_features % 8 == 0 and out_features % 32 == 0
            if backend == "wmma_hybrid":
                use_wmma = use_wmma and qg.size(0) <= _wmma_hybrid_max_rows()
            if use_wmma:
                weight_t = cuda_int4.transpose_packed(packed_weight, padded_features)
                acc = cuda_int4.wmma_int4_mm(qg.contiguous(), weight_t.contiguous(), out_features)
                return cuda_int4.dequant_epilogue(acc, grad_scale.reshape(-1), None, None, out_dtype)
            logger.debug(
                "INT4 ConvRot WMMA grad-input falling back to torch bridge for shape M=%d N=%d K=%d",
                qg.size(0),
                padded_features,
                out_features,
            )
        g_int8 = cuda_int4.unpack_to_int8(qg, out_features)
        w_int8 = cuda_int4.unpack_to_int8(packed_weight, padded_features)
        if backend == "cutlass_int8":
            acc = _get_cutlass_int8_mm()(g_int8.contiguous(), w_int8.t().contiguous())
        else:
            acc = _int_mm_allow_small_m(g_int8.contiguous(), w_int8)
        return cuda_int4.dequant_epilogue(acc, grad_scale.reshape(-1), None, None, out_dtype)
    except Exception as exc:
        if _int4_backend_request() in {"", "auto"}:
            logger.debug("INT4 ConvRot tensor-core grad-input bridge failed; falling back to scalar CUDA path: %s", exc)
            return None
        raise


def _quantize_activation_int4(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cuda_int4 = _get_cuda_int4() if x_2d.is_cuda else None
    if cuda_int4 is not None:
        try:
            q, scale = cuda_int4.quantize_rowwise(x_2d.contiguous())
            return q, scale.reshape(-1, 1)
        except Exception:
            pass
    return quantize_int4_rowwise(x_2d, mse_clip=False)


def _quantize_activation_int8(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    abs_max = x_2d.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30).float()
    q = (x_2d.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _linear_w4a8_tensorcore(
    qx: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    k_values: int,
) -> torch.Tensor | None:
    if not qx.is_cuda or not _supports_torch_int_mm(qx.device):
        return None
    cuda_int4 = _get_cuda_int4()
    if cuda_int4 is None:
        return None
    w_int8 = cuda_int4.unpack_to_int8(packed_weight, k_values)
    acc = _int_mm_allow_small_m(qx.contiguous(), w_int8.t())
    out = acc.float() * x_scale.float() * weight_scale.reshape(1, -1).float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _grad_input_w4a8_tensorcore(
    qg: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    padded_features: int,
) -> torch.Tensor | None:
    if not qg.is_cuda or not _supports_torch_int_mm(qg.device):
        return None
    cuda_int4 = _get_cuda_int4()
    if cuda_int4 is None:
        return None
    w_int8 = cuda_int4.unpack_to_int8(packed_weight, padded_features)
    acc = _int_mm_allow_small_m(qg.contiguous(), w_int8)
    return (acc.float() * grad_scale.float()).to(out_dtype)


def _linear_w4a8_fallback(
    qx: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    k_values: int,
) -> torch.Tensor:
    w_int = unpack_int4(packed_weight, k_values).to(torch.float32)
    acc = qx.to(torch.float32) @ w_int.t()
    out = acc.float() * x_scale.float() * weight_scale.reshape(1, -1).float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _grad_input_w4a8_fallback(
    qg: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    padded_features: int,
) -> torch.Tensor:
    w_int = unpack_int4(packed_weight, padded_features).to(torch.float32)
    acc = qg.to(torch.float32) @ w_int
    return (acc.float() * grad_scale.float()).to(out_dtype)


# ---------------------------------------------------------------------------
# Optional fused Triton activation path for the W4A8 default mode.
#
# On Hopper (no int4 tensor cores) the W4A8 GEMM already runs through the int8
# ``torch._int_mm`` bridge, so the activation-side elementwise passes -- online
# ConvRot rotation, per-token int8 quantization, and the int32->compute-dtype
# rescale -- are the only memory-bound overhead left around the matmul. They are
# the same tensor shapes as the INT8 ConvRot path, so this path reuses those fused
# Triton kernels. It is opt-in via ``LTX2_INT4_CONVROT_FUSE``; when it is off
# (default), unavailable, or the kernel fails, dispatch uses the eager implementation.
# ---------------------------------------------------------------------------

_INT4_FUSE: bool | None = None
_INT4_TRITON_FUSED_BROKEN = False
_INT4_FUSED_GROUP_SIZES = (4, 16, 64, 256)

# Independent CUDA fusion for the native-CUTLASS W4A4 path (LTX2_INT4_CONVROT_FUSE_CUDA).
# Distinct from the Triton W4A8 fusion above: it folds the ConvRot rotation into the
# activation-quantize kernel and the int32->bf16 dequant into the CUTLASS int4 GEMM's
# epilogue (an Sm80 Epilogue Visitor Tree), so the int32 accumulator never hits global
# memory. Off (unset) leaves this fusion branch disabled.
_INT4_FUSE_CUDA: bool | None = None
_INT4_CUDA_FUSED_BROKEN = False
_CUTLASS_INT4_FUSED = "unset"

# Triple-branch LoRA fusion for the frozen-base int4cr training path
# (LTX2_INT4_CONVROT_FUSE_LORA). Distinct from the two backbone fusions above: it fuses
# the trainable-LoRA and frozen-stabilizer *down* projections into a single GEMM that
# reads the activation once, and accumulates both *up* projections onto the int4 backbone
# output in one pass. The rotation is folded once into the frozen stabilizer down weight
# so all three branches share the raw activation. Off (unset) retains the eager down/up path.
_INT4_FUSE_LORA: bool | None = None


def _int4_fuse_enabled() -> bool:
    global _INT4_FUSE
    if _INT4_FUSE is None:
        _INT4_FUSE = os.getenv("LTX2_INT4_CONVROT_FUSE", "0").strip().lower() in ("1", "true", "yes", "on")
    return _INT4_FUSE


def _int4_fuse_cuda_enabled() -> bool:
    global _INT4_FUSE_CUDA
    if _INT4_FUSE_CUDA is None:
        _INT4_FUSE_CUDA = os.getenv("LTX2_INT4_CONVROT_FUSE_CUDA", "0").strip().lower() in ("1", "true", "yes", "on")
    return _INT4_FUSE_CUDA


def _int4_fuse_lora_enabled() -> bool:
    global _INT4_FUSE_LORA
    if _INT4_FUSE_LORA is None:
        _INT4_FUSE_LORA = os.getenv("LTX2_INT4_CONVROT_FUSE_LORA", "0").strip().lower() in ("1", "true", "yes", "on")
    return _INT4_FUSE_LORA


def configure_int4cr_training_defaults(
    *,
    mode_flag: str,
    act_bits: int,
    grad_bits: int | None = None,
    backend: str | None = None,
    fuse_cuda: bool | None = None,
    fuse_lora: bool | None = None,
) -> dict[str, Any]:
    """Resolve the four expert int4cr gates for an INT4 ConvRot mode flag (--w4a4g4 / --w4a8).

    Called once by the trainer, BEFORE model load / first gate read. For every gate the mode
    implies (a non-None argument), an explicitly-set environment variable WINS (expert
    override, warned once when it contradicts the implied value); an unset variable takes the
    implied default. Gates the mode does not imply (None argument) are left completely
    untouched. When no mode flag is passed the setter is not called.

    ``--w4a4g4`` -> act_bits=4, backend="cutlass", fuse_cuda=True, fuse_lora=True.
    ``--w4a4g8`` -> act_bits=4, grad_bits=8 (a4 forward, int8 grad backward) + the w4a4g4 backend/fusion.
    ``--w4a8``   -> act_bits=8 only (legacy default backend routing / no implied fusion).

    ``grad_bits`` is left None by --w4a4g4 / --w4a8 (the getter falls back to ``act_bits``);
    only --w4a4g8 passes it to decouple the backward gradient bit-width.
    """
    global _INT4CR_ACT_BITS_OVERRIDE, _INT4CR_BACKEND_OVERRIDE, _INT4CR_GRAD_BITS_OVERRIDE
    global _INT4_FUSE_CUDA, _INT4_FUSE_LORA
    resolved: dict[str, tuple[str, Any]] = {}
    warnings: list[str] = []

    act_env = os.getenv("LTX2_INT4_CONVROT_ACT_BITS")
    if act_env is not None and act_env.strip():
        env_bits = _int4_activation_bits()
        resolved["act_bits"] = ("env", env_bits)
        if env_bits != int(act_bits):
            warnings.append(f"LTX2_INT4_CONVROT_ACT_BITS={act_env.strip()} (W4A{env_bits}) overrides {mode_flag}")
    else:
        _INT4CR_ACT_BITS_OVERRIDE = int(act_bits)
        resolved["act_bits"] = ("flag", int(act_bits))

    if grad_bits is not None:
        grad_env = os.getenv("LTX2_INT4_CONVROT_GRAD_BITS")
        if grad_env is not None and grad_env.strip():
            env_g = _int4_gradient_bits()
            resolved["grad_bits"] = ("env", env_g)
            if env_g != int(grad_bits):
                warnings.append(f"LTX2_INT4_CONVROT_GRAD_BITS={grad_env.strip()} (G{env_g}) overrides {mode_flag}")
        else:
            _INT4CR_GRAD_BITS_OVERRIDE = int(grad_bits)
            resolved["grad_bits"] = ("flag", int(grad_bits))

    if backend is not None:
        be_env = os.getenv("LTX2_INT4_CONVROT_BACKEND")
        if be_env is not None and be_env.strip():
            resolved["backend"] = ("env", be_env.strip().lower())
            if be_env.strip().lower() not in ("auto", backend):
                warnings.append(f"LTX2_INT4_CONVROT_BACKEND={be_env.strip()} overrides {mode_flag} (backend={backend})")
        else:
            _INT4CR_BACKEND_OVERRIDE = backend
            resolved["backend"] = ("flag", backend)

    if fuse_cuda is not None:
        fc_env = os.getenv("LTX2_INT4_CONVROT_FUSE_CUDA")
        if fc_env is not None and fc_env.strip():
            resolved["fuse_cuda"] = ("env", _int4_fuse_cuda_enabled())
        else:
            _INT4_FUSE_CUDA = bool(fuse_cuda)
            resolved["fuse_cuda"] = ("flag", bool(fuse_cuda))

    if fuse_lora is not None:
        fl_env = os.getenv("LTX2_INT4_CONVROT_FUSE_LORA")
        if fl_env is not None and fl_env.strip():
            resolved["fuse_lora"] = ("env", _int4_fuse_lora_enabled())
        else:
            _INT4_FUSE_LORA = bool(fuse_lora)
            resolved["fuse_lora"] = ("flag", bool(fuse_lora))

    summary = " ".join(f"{k}={v[1]}[{v[0]}]" for k, v in resolved.items())
    logger.info("INT4 ConvRot %s resolved: %s", mode_flag, summary)
    for warning in warnings:
        logger.warning("INT4 ConvRot %s: %s", mode_flag, warning)
    return {"resolved": resolved, "warnings": warnings}


def detect_int4_convrot_checkpoint(model_path: str | list[str]) -> bool:
    """Whether a safetensors file is a pre-quantized INT4 ConvRot checkpoint (converter output).

    Detected from the ``int4_convrot_quantized`` file-metadata marker written by
    ``ltx2_quantize_int4_convrot``, or a key-based fallback: a packed uint8 ``.weight`` that
    carries a ``.comfy_quant`` or ``.int4_shape`` sidecar. A plain bf16/fp16 checkpoint has
    none of these, so a mode flag quantizes it on the fly (dynamic path) instead.
    """
    try:
        from safetensors import safe_open

        check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(check_path, framework="pt") as f:
            meta = f.metadata()
            if meta is not None and meta.get(INT4_CONVROT_METADATA_MARKER) == "true":
                return True
            keys = list(f.keys())
            shape_bases = {k[: -len(".int4_shape")] for k in keys if k.endswith(".int4_shape")}
            comfy_bases = {k[: -len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
            marked_bases = shape_bases | comfy_bases
            if not marked_bases:
                return False
            for base in marked_bases:
                weight_key = base + ".weight"
                if weight_key in keys and f.get_slice(weight_key).get_dtype() == "U8":
                    return True
            return False
    except Exception:
        return False


def _get_cutlass_int4_fused():
    global _CUTLASS_INT4_FUSED
    if _CUTLASS_INT4_FUSED == "unset":
        try:
            from musubi_tuner.modules import cutlass_int4_fused

            _CUTLASS_INT4_FUSED = cutlass_int4_fused if cutlass_int4_fused.is_available() else None
        except Exception as exc:
            logger.info("INT4 ConvRot fused CUTLASS epilogue unavailable: %s", exc)
            _CUTLASS_INT4_FUSED = None
    return _CUTLASS_INT4_FUSED


def _mark_int4_cuda_fused_broken() -> None:
    global _INT4_CUDA_FUSED_BROKEN
    if not _INT4_CUDA_FUSED_BROKEN:
        _INT4_CUDA_FUSED_BROKEN = True
        logger.warning("INT4 ConvRot CUDA-fused W4A4 path failed; using the unfused CUTLASS path for the rest of this run.")


def _mark_int4_triton_fused_broken() -> None:
    global _INT4_TRITON_FUSED_BROKEN
    if not _INT4_TRITON_FUSED_BROKEN:
        _INT4_TRITON_FUSED_BROKEN = True
        logger.warning("INT4 ConvRot fused Triton kernel failed; using the eager rotate+quant path for the rest of this run.")


def _get_int4_fused_quant():
    try:
        from musubi_tuner.modules.triton_int8_epilogue import have_triton, quantize_rowwise
    except Exception:
        return None
    return quantize_rowwise if have_triton() else None


def _get_int4_fused_epilogue():
    try:
        from musubi_tuner.modules.triton_int8_epilogue import dequant_epilogue, have_triton
    except Exception:
        return None
    return dequant_epilogue if have_triton() else None


def _get_int4_fused_convrot_epilogue():
    try:
        from musubi_tuner.modules.triton_int8_epilogue import dequant_epilogue_convrot, have_triton
    except Exception:
        return None
    return dequant_epilogue_convrot if have_triton() else None


def _unpack_weight_to_int8(packed_weight: torch.Tensor, k_values: int) -> torch.Tensor:
    cuda_int4 = _get_cuda_int4() if packed_weight.is_cuda else None
    if cuda_int4 is not None:
        try:
            return cuda_int4.unpack_to_int8(packed_weight, k_values)
        except Exception:
            pass
    return unpack_int4(packed_weight, k_values).to(torch.int8)


def _forward_w4a8_fused(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    group_size: int,
    padded_features: int,
    out_dtype: torch.dtype,
    need_rotated: bool,
    rotate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Fused W4A8 forward activation path.

    Returns ``(output_2d, x_rot_2d_or_None)`` on success, or ``None`` to tell the
    caller to fall back to the eager path. ``x_rot_2d`` is the rotated padded
    activation, materialized only when ``need_rotated`` (a stabilizer branch reads
    it); otherwise the online rotation is fused into the quant kernel and ``None`` is
    returned in its place. In no-rotation mode ``rotate`` is False: the activation is
    only zero-padded (identity rotation) and ``x_rot_2d`` holds the padded activation.
    """
    if not _int4_fuse_enabled() or _INT4_TRITON_FUSED_BROKEN:
        return None
    if not x.is_cuda or not _supports_torch_int_mm(x.device):
        return None
    quant = _get_int4_fused_quant()
    epilogue = _get_int4_fused_epilogue()
    if quant is None or epilogue is None:
        return None
    fuse_rotation = rotate and (not need_rotated) and group_size in _INT4_FUSED_GROUP_SIZES
    try:
        if fuse_rotation:
            x_2d = pad_last_dim(x, padded_features).reshape(-1, padded_features).contiguous()
            qx, x_scale = quant(x_2d, None, convrot_groupsize=group_size)
            x_rot_2d = None
        elif rotate:
            x_rot = rotate_activation_padded(x, group_size, padded_features)
            x_rot_2d = x_rot.reshape(-1, padded_features).contiguous()
            qx, x_scale = quant(x_rot_2d, None)
        else:
            x_rot_2d = pad_last_dim(x, padded_features).reshape(-1, padded_features).contiguous()
            qx, x_scale = quant(x_rot_2d, None)
        w_int8 = _unpack_weight_to_int8(packed_weight, padded_features)
        acc = _int_mm_allow_small_m(qx.contiguous(), w_int8.t())
        output = epilogue(acc, x_scale, weight_scale, bias.float() if bias is not None else None, out_dtype)
    except Exception as exc:  # pragma: no cover - defensive: fall back to eager
        _mark_int4_triton_fused_broken()
        logger.debug("INT4 ConvRot fused forward failed; reverting to eager path: %s", exc)
        return None
    return output, x_rot_2d


def _grad_input_w4a8_fused(
    go: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_size: int,
    padded_features: int,
    out_features: int,
    out_dtype: torch.dtype,
    fuse_inverse_rotation: bool,
    rotate: bool = True,
) -> tuple[torch.Tensor, bool] | None:
    """Fused W4A8 grad-input path.

    ``go`` is the 2D grad_output ``[M, out_features]`` in its original dtype. Folds
    the per-output-channel weight scale into grad_output and quantizes in a single
    pass, runs the int8-bridge matmul, then rescales. When ``fuse_inverse_rotation``
    the inverse ConvRot is folded into the rescale epilogue and the returned tensor
    is grad_input in the padded input space (second element ``True``); otherwise the
    returned tensor is still in the rotated space so a stabilizer term can be added
    before an eager inverse rotation (second element ``False``). Returns ``None`` to
    signal an eager fallback.
    """
    if not _int4_fuse_enabled() or _INT4_TRITON_FUSED_BROKEN:
        return None
    if not go.is_cuda or not _supports_torch_int_mm(go.device):
        return None
    if weight_scale.numel() != out_features:
        return None
    quant = _get_int4_fused_quant()
    epilogue = _get_int4_fused_epilogue()
    if quant is None or epilogue is None:
        return None
    convrot_epilogue = _get_int4_fused_convrot_epilogue()
    do_convrot = rotate and fuse_inverse_rotation and convrot_epilogue is not None and group_size in _INT4_FUSED_GROUP_SIZES
    try:
        qg, grad_scale = quant(go.contiguous(), weight_scale.reshape(out_features).contiguous())
        w_int8 = _unpack_weight_to_int8(packed_weight, padded_features)
        acc = _int_mm_allow_small_m(qg.contiguous(), w_int8)
        if do_convrot:
            grad_input_padded = convrot_epilogue(acc, grad_scale, group_size, out_dtype)
            return grad_input_padded, True
        grad_rot = epilogue(acc, grad_scale, None, None, out_dtype)
        return grad_rot, False
    except Exception as exc:  # pragma: no cover - defensive: fall back to eager
        _mark_int4_triton_fused_broken()
        logger.debug("INT4 ConvRot fused grad-input failed; reverting to eager path: %s", exc)
        return None


def _forward_w4a4_fused_cuda(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    group_size: int,
    padded_features: int,
    in_features: int,
    out_dtype: torch.dtype,
    rotate: bool,
    need_rotated: bool,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Fully-fused native-CUTLASS W4A4 forward.

    Folds the online ConvRot rotation into the activation-quantize kernel (when
    ``rotate`` and no stabilizer needs the rotated activation) and the dequant into the
    CUTLASS int4 GEMM epilogue. Returns ``(output_2d, x_2d_or_None)`` -- ``x_2d`` is the
    rotated/padded activation, materialized only when the stabilizer branch reads it --
    or ``None`` to signal the caller to use the unfused path.
    """

    if not _int4_fuse_cuda_enabled() or _INT4_CUDA_FUSED_BROKEN or not x.is_cuda:
        return None
    fused_ext = _get_cutlass_int4_fused()
    cuda_int4 = _get_cuda_int4()
    if fused_ext is None or cuda_int4 is None:
        return None
    try:
        backend = _resolve_tensorcore_backend(x.device)
    except Exception:
        return None
    if backend != "cutlass":
        return None
    fuse_rotation = rotate and (not need_rotated)
    try:
        if fuse_rotation:
            x_2d_in = x.reshape(-1, in_features).contiguous()
            qx, x_scale = cuda_int4.quantize_rowwise_convrot(x_2d_in, padded_features, group_size)
            x_2d = None
        elif rotate:
            x_rot = rotate_activation_padded(x, group_size, padded_features)
            x_2d = x_rot.reshape(-1, padded_features).contiguous()
            qx, x_scale = cuda_int4.quantize_rowwise(x_2d)
        else:
            x_2d = pad_last_dim(x, padded_features).reshape(-1, padded_features).contiguous()
            qx, x_scale = cuda_int4.quantize_rowwise(x_2d)
        output = fused_ext.linear_fused(
            qx, packed_weight, x_scale.reshape(-1), weight_scale.reshape(-1), bias, out_dtype, padded_features
        )
    except Exception as exc:  # pragma: no cover - defensive: fall back to unfused
        _mark_int4_cuda_fused_broken()
        logger.debug("INT4 ConvRot CUDA-fused forward failed; reverting to unfused path: %s", exc)
        return None
    return output, x_2d


def _grad_input_w4a4_fused_cuda(
    qg: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out_features: int,
    padded_features: int,
) -> torch.Tensor | None:
    """Fused native-CUTLASS W4A4 grad-input GEMM (transpose + row-scaled epilogue).

    Returns grad w.r.t. the rotated/padded activation, or ``None`` to fall back. The
    inverse ConvRot rotation (rotated mode) and the stabilizer term are applied by the
    caller, exactly as in the unfused path.
    """

    if not _int4_fuse_cuda_enabled() or _INT4_CUDA_FUSED_BROKEN or not qg.is_cuda:
        return None
    fused_ext = _get_cutlass_int4_fused()
    cutlass_int4 = _get_cutlass_int4()
    if fused_ext is None or cutlass_int4 is None:
        return None
    try:
        backend = _resolve_tensorcore_backend(qg.device)
    except Exception:
        return None
    if backend != "cutlass":
        return None
    try:
        weight_t = _cutlass_int4_transposed_weight(cutlass_int4, packed_weight, padded_features)
        output = fused_ext.linear_fused(qg, weight_t, grad_scale.reshape(-1), None, None, out_dtype, out_features)
    except Exception as exc:  # pragma: no cover - defensive: fall back to unfused
        _mark_int4_cuda_fused_broken()
        logger.debug("INT4 ConvRot CUDA-fused grad-input failed; reverting to unfused path: %s", exc)
        return None
    return output


def _linear_int4_fallback(
    qx: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    k_values: int,
) -> torch.Tensor:
    x_int = unpack_int4(qx, k_values).to(torch.float32)
    w_int = unpack_int4(packed_weight, k_values).to(torch.float32)
    acc = x_int @ w_int.t()
    out = acc.float() * x_scale.float() * weight_scale.reshape(1, -1).float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _grad_input_int4_fallback(
    qg: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out_features: int,
    padded_features: int,
) -> torch.Tensor:
    g_int = unpack_int4(qg, out_features).to(torch.float32)
    w_int = unpack_int4(packed_weight, padded_features).to(torch.float32)
    acc = g_int @ w_int
    return (acc.float() * grad_scale.float()).to(out_dtype)


class _Int4ConvRotLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, weight_scale, bias, int4_shape, convrot_groupsize, stab_l1=None, stab_l2=None, rotate=True):
        out_features, in_features, padded_features = (int(v) for v in int4_shape.detach().cpu().reshape(-1).tolist())
        group_size = (
            int(convrot_groupsize.detach().reshape(-1)[0].item())
            if isinstance(convrot_groupsize, torch.Tensor)
            else int(convrot_groupsize)
        )
        original_shape = x.shape
        output_dtype = _autocast_output_dtype(x)
        has_stabilizer = stab_l1 is not None and stab_l2 is not None

        activation_bits = _int4_activation_bits()
        # ``x_2d`` is the rotated, padded activation. The fused W4A8 path can fold the
        # rotation into the quant kernel and leave it unmaterialized; it is only needed
        # for the stabilizer branch, so materialize it lazily.
        x_2d = None
        output = None
        if activation_bits == 8:
            fused = _forward_w4a8_fused(
                x,
                packed_weight,
                weight_scale,
                bias,
                group_size=group_size,
                padded_features=padded_features,
                out_dtype=output_dtype,
                need_rotated=has_stabilizer,
                rotate=rotate,
            )
            if fused is not None:
                output, x_2d = fused
        elif activation_bits == 4:
            fused = _forward_w4a4_fused_cuda(
                x,
                packed_weight,
                weight_scale,
                bias,
                group_size=group_size,
                padded_features=padded_features,
                in_features=in_features,
                out_dtype=output_dtype,
                rotate=rotate,
                need_rotated=has_stabilizer,
            )
            if fused is not None:
                output, x_2d = fused

        if output is None:
            if rotate:
                x_rot = rotate_activation_padded(x, group_size, padded_features)
                x_2d = x_rot.reshape(-1, padded_features)
            else:
                x_2d = pad_last_dim(x, padded_features).reshape(-1, padded_features).contiguous()
            if activation_bits == 8:
                qx, x_scale = _quantize_activation_int8(x_2d)
                output = _linear_w4a8_tensorcore(
                    qx,
                    packed_weight,
                    x_scale,
                    weight_scale,
                    bias,
                    out_dtype=output_dtype,
                    k_values=padded_features,
                )
                if output is None:
                    output = _linear_w4a8_fallback(
                        qx,
                        packed_weight,
                        x_scale,
                        weight_scale,
                        bias,
                        out_dtype=output_dtype,
                        k_values=padded_features,
                    )
            else:
                qx, x_scale = _quantize_activation_int4(x_2d)
                output = _linear_int4_tensorcore(
                    qx,
                    packed_weight,
                    x_scale,
                    weight_scale,
                    bias,
                    out_dtype=output_dtype,
                    k_values=padded_features,
                )
                cuda_int4 = _get_cuda_int4() if output is None and qx.is_cuda else None
                if output is not None:
                    pass
                elif cuda_int4 is not None:
                    output = cuda_int4.linear_forward(
                        qx, packed_weight, x_scale.reshape(-1), weight_scale.reshape(-1), bias, output_dtype
                    )
                else:
                    output = _linear_int4_fallback(
                        qx,
                        packed_weight,
                        x_scale,
                        weight_scale,
                        bias,
                        out_dtype=output_dtype,
                        k_values=padded_features,
                    )
        if has_stabilizer:
            # Frozen high-precision outlier branch: shares the rotated/padded input with the
            # INT4 backbone, evaluated as two skinny GEMMs (never materializes L1 @ L2).
            compute_dtype = x_2d.dtype if x_2d.is_floating_point() else torch.float32
            y_stab = (x_2d.to(compute_dtype) @ stab_l2.t().to(compute_dtype)) @ stab_l1.t().to(compute_dtype)
            output = output + y_stab.to(output.dtype)
            ctx.save_for_backward(
                packed_weight,
                weight_scale,
                int4_shape,
                torch.tensor(group_size, device=x.device, dtype=torch.int32),
                stab_l1,
                stab_l2,
            )
        else:
            ctx.save_for_backward(
                packed_weight, weight_scale, int4_shape, torch.tensor(group_size, device=x.device, dtype=torch.int32)
            )
        ctx.has_stabilizer = has_stabilizer
        ctx.input_dtype = x.dtype
        ctx.activation_bits = activation_bits
        # Gradient-quant bits are resolved independently of the forward activation bits so
        # --w4a4g8 keeps the a4 forward above but takes the int8 grad path in backward.
        ctx.gradient_bits = _int4_gradient_bits()
        ctx.rotate = rotate
        ctx.original_shape = original_shape
        return output.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise RuntimeError("INT4 ConvRot base weights are frozen; full fine-tuning is not supported.")
        if ctx.has_stabilizer:
            packed_weight, weight_scale, int4_shape, group_tensor, stab_l1, stab_l2 = ctx.saved_tensors
        else:
            packed_weight, weight_scale, int4_shape, group_tensor = ctx.saved_tensors
            stab_l1 = stab_l2 = None
        out_features, in_features, padded_features = (int(v) for v in int4_shape.detach().cpu().reshape(-1).tolist())
        group_size = int(group_tensor.detach().reshape(-1)[0].item())
        go = grad_output.reshape(-1, out_features)
        has_stabilizer = stab_l1 is not None

        # ``grad_rot`` is the grad w.r.t. the rotated activation. The fused W4A8 path can
        # also fold the inverse rotation into its rescale epilogue -- but only when there
        # is no stabilizer term to add first (the stabilizer contributes in the rotated
        # space). ``grad_input_ready`` is set when the inverse rotation is already applied.
        grad_rot = None
        grad_input_ready = None
        if ctx.gradient_bits == 8:
            fused = _grad_input_w4a8_fused(
                go,
                packed_weight,
                weight_scale,
                group_size=group_size,
                padded_features=padded_features,
                out_features=out_features,
                out_dtype=ctx.input_dtype,
                fuse_inverse_rotation=not has_stabilizer,
                rotate=ctx.rotate,
            )
            if fused is not None:
                tensor, inverse_applied = fused
                if inverse_applied:
                    grad_input_ready = tensor
                else:
                    grad_rot = tensor

        if grad_input_ready is None and grad_rot is None:
            go_scaled = go.float() * weight_scale.reshape(1, out_features).float()
            if ctx.gradient_bits == 8:
                qg, grad_scale = _quantize_activation_int8(go_scaled.to(grad_output.dtype))
                grad_rot = _grad_input_w4a8_tensorcore(
                    qg,
                    packed_weight,
                    grad_scale,
                    out_dtype=ctx.input_dtype,
                    padded_features=padded_features,
                )
                if grad_rot is None:
                    grad_rot = _grad_input_w4a8_fallback(
                        qg,
                        packed_weight,
                        grad_scale,
                        out_dtype=ctx.input_dtype,
                        padded_features=padded_features,
                    )
            else:
                qg, grad_scale = _quantize_activation_int4(go_scaled.to(grad_output.dtype))
                grad_rot = _grad_input_w4a4_fused_cuda(
                    qg,
                    packed_weight,
                    grad_scale,
                    out_dtype=ctx.input_dtype,
                    out_features=out_features,
                    padded_features=padded_features,
                )
                if grad_rot is None:
                    grad_rot = _grad_input_int4_tensorcore(
                        qg,
                        packed_weight,
                        grad_scale,
                        out_dtype=ctx.input_dtype,
                        out_features=out_features,
                        padded_features=padded_features,
                    )
                cuda_int4 = _get_cuda_int4() if grad_rot is None and qg.is_cuda else None
                if grad_rot is not None:
                    pass
                elif cuda_int4 is not None:
                    grad_rot = cuda_int4.linear_backward_input(
                        qg, packed_weight, grad_scale.reshape(-1), padded_features, ctx.input_dtype
                    )
                else:
                    grad_rot = _grad_input_int4_fallback(
                        qg,
                        packed_weight,
                        grad_scale,
                        out_dtype=ctx.input_dtype,
                        out_features=out_features,
                        padded_features=padded_features,
                    )

        if grad_input_ready is not None:
            # Fused path already applied the rescale and the inverse rotation; no
            # stabilizer term is present in this branch (fuse_inverse_rotation was gated
            # on its absence).
            grad_input = grad_input_ready[..., :in_features]
        else:
            if stab_l1 is not None:
                # Stabilizer pathway: propagated from the unscaled high-precision grad_output,
                # not from the weight-scale-folded quantized gradient used by the backbone GEMM.
                compute_dtype = go.dtype if go.is_floating_point() else torch.float32
                grad_stab = (go.to(compute_dtype) @ stab_l1.to(compute_dtype)) @ stab_l2.to(compute_dtype)
                grad_rot = grad_rot + grad_stab.to(grad_rot.dtype)
            if ctx.rotate:
                grad_input = rotate_activation_padded(grad_rot, group_size, padded_features, inverse=True)[..., :in_features]
            else:
                grad_input = grad_rot[..., :in_features]
        grad_input = grad_input.reshape(*ctx.original_shape)
        return grad_input, None, None, None, None, None, None, None, None


def int4_convrot_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    out_features, in_features, padded_features = _module_int4_shape(self)
    x = _apply_int4_awq_activation_scale(self, x, in_features)
    stabilizer = _module_int4_stabilizer(self, out_features, padded_features)
    rotate = _module_int4_rotate(self)
    if _use_weight_only_diagnostic():
        weight = dequantize_int4_convrot_weight(
            self.weight,
            self.scale_weight,
            _module_int4_group(self),
            in_features,
            padded_features,
            dtype=x.dtype if x.is_floating_point() else torch.float32,
            stabilizer=stabilizer,
            rotate=rotate,
        )
        return F.linear(x, weight, self.bias)
    group_size = _module_int4_group(self)
    shape = getattr(self, "int4_shape")
    group = getattr(self, "int4_convrot_groupsize")
    stab_l1, stab_l2 = stabilizer if stabilizer is not None else (None, None)
    return _Int4ConvRotLinearFunction.apply(
        x,
        self.weight,
        self.scale_weight,
        self.bias,
        shape,
        group_size if group is None else group,
        stab_l1,
        stab_l2,
        rotate,
    )


def register_int4_convrot_buffers(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    registered = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_key = name + ".weight"
        shape_key = name + ".int4_shape"
        scale_key = name + ".scale_weight"
        group_key = name + ".int4_convrot_groupsize"
        awq_key = name + INT4_CONVROT_AWQ_SCALE_SUFFIX
        if weight_key not in state_dict or shape_key not in state_dict or scale_key not in state_dict:
            continue
        packed = state_dict[weight_key]
        if packed.dtype != torch.uint8:
            continue
        shape = state_dict[shape_key].to(torch.int32)
        if hasattr(module, "weight"):
            del module.weight
        module.register_buffer("weight", torch.zeros_like(packed), persistent=True)
        module.register_buffer("scale_weight", torch.zeros_like(state_dict[scale_key].float()), persistent=True)
        module.register_buffer("int4_shape", torch.zeros_like(shape), persistent=True)
        module.register_buffer(
            "int4_convrot_groupsize",
            torch.zeros_like(state_dict.get(group_key, torch.tensor(0, dtype=torch.int32)).to(torch.int32)),
            persistent=True,
        )
        if awq_key in state_dict:
            module.register_buffer("int4_awq_scales", torch.zeros_like(state_dict[awq_key].float()), persistent=True)
        rotation_key = name + ".int4_rotation"
        if rotation_key in state_dict:
            module.register_buffer("int4_rotation", torch.zeros_like(state_dict[rotation_key].to(torch.int32)), persistent=True)
        stab_l1_key = name + INT4_CONVROT_STABILIZER_L1_SUFFIX
        stab_l2_key = name + INT4_CONVROT_STABILIZER_L2_SUFFIX
        if stab_l1_key in state_dict and stab_l2_key in state_dict:
            module.register_buffer("int4_stabilizer_l1", torch.zeros_like(state_dict[stab_l1_key]), persistent=True)
            module.register_buffer("int4_stabilizer_l2", torch.zeros_like(state_dict[stab_l2_key]), persistent=True)
        registered += 1
    return registered


def apply_int4_convrot_monkey_patch(model: nn.Module) -> nn.Module:
    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (hasattr(module, "scale_weight") and hasattr(module, "int4_shape") and module.weight.dtype == torch.uint8):
            continue
        out_features, in_features, padded_features = _module_int4_shape(module)
        group_size = _module_int4_group(module)
        if group_size <= 0:
            raise ValueError("INT4 ConvRot module has no positive group size")
        if padded_features != padded_features_for_group(in_features, group_size):
            raise ValueError(
                f"INT4 ConvRot padded features mismatch: got {padded_features}, expected "
                f"{padded_features_for_group(in_features, group_size)}"
            )
        if module.weight.shape != (out_features, padded_features // 2):
            raise ValueError(
                f"INT4 packed weight shape {tuple(module.weight.shape)} incompatible with {tuple(_module_int4_shape(module))}"
            )
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.scale_weight = module.scale_weight.reshape(out_features, 1).to(device=module.weight.device, dtype=torch.float32)
        awq_scales = _module_int4_awq_scales(module, in_features)
        if awq_scales is not None:
            module.int4_awq_scales = awq_scales.to(device=module.weight.device, dtype=torch.float32)
        stabilizer = _module_int4_stabilizer(module, out_features, padded_features)
        if stabilizer is not None:
            module.int4_stabilizer_l1 = stabilizer[0].to(device=module.weight.device).contiguous()
            module.int4_stabilizer_l2 = stabilizer[1].to(device=module.weight.device).contiguous()
        module.forward = int4_convrot_linear_forward.__get__(module, type(module))
        patched += 1
    logger = __import__("logging").getLogger(__name__)
    backend = "cuda-lazy" if torch.cuda.is_available() else "torch-fallback"
    act_bits = _int4_activation_bits()
    logger.info("INT4 ConvRot (%s): patched %d linear layers, activation mode W4A%d", backend, patched, act_bits)
    return model


# ---------------------------------------------------------------------------
# Triple-branch LoRA fusion (LTX2_INT4_CONVROT_FUSE_LORA).
#
# The wrapped forward of an int4cr Linear under a plain LoRA adapter is
#   y = int4_backbone(x) + stabilizer(x) + scale * lora_up(lora_down(x))
# where stabilizer(x) = (rotate(x) @ L2^T) @ L1^T reads the *rotated* padded activation
# and the LoRA branch reads the raw activation. The eager path materializes the rotated
# activation and runs four skinny GEMMs (stab-down, stab-up, lora-down, lora-up) plus two
# elementwise adds, reading x from global memory several times.
#
# This fusion folds the (frozen, orthogonal) rotation once into the stabilizer down weight
#   S = rotate_basis @ L2^T          (precomputed at apply time, [in_features, r_stab])
# so stabilizer(x) = (x @ S) @ L1^T reads the same raw x as the LoRA branch. Both down
# projections then collapse into ONE GEMM over a concatenated weight, and both up
# projections into ONE GEMM whose result is added to the int4 backbone output. Only the
# LoRA weights carry gradients; S and L1 are frozen (no grad, not registered, not saved).
# ---------------------------------------------------------------------------

_FUSED_LORA_LOGGED_FALLBACKS: set[str] = set()


def _fused_lora_log_fallback(reason: str) -> None:
    if reason not in _FUSED_LORA_LOGGED_FALLBACKS:
        _FUSED_LORA_LOGGED_FALLBACKS.add(reason)
        logger.info("INT4 ConvRot fused-LoRA: %s; those modules use the unfused down/up path.", reason)


def _fused_stab_down_raw(
    stab_l2: torch.Tensor,
    *,
    group_size: int,
    in_features: int,
    padded_features: int,
    rotate: bool,
    device: torch.device,
    rot_cache: dict[Any, torch.Tensor],
) -> torch.Tensor:
    """Frozen stabilizer down weight in the raw (unrotated) activation space.

    Returns ``S`` with shape ``[in_features, r_stab]`` (float32) such that for every
    activation ``x`` the fused product ``x @ S`` equals the eager
    ``rotate_activation_padded(x) @ stab_l2.t()`` (or ``pad(x) @ stab_l2.t()`` when
    ``rotate`` is False). The rotation is orthogonal and frozen, so folding it into the
    frozen stabilizer factor is algebraically equivalent before floating-point rounding and is
    precomputed once per patched module.
    """
    l2 = stab_l2.to(device=device, dtype=torch.float32)  # [r_stab, padded]
    if not rotate:
        return l2.t()[:in_features].contiguous()  # pad(x) @ l2.t() drops the zero-pad columns
    key = (int(group_size), int(in_features), int(padded_features), str(device))
    rot_eye = rot_cache.get(key)
    if rot_eye is None:
        eye = torch.eye(in_features, device=device, dtype=torch.float32)
        rot_eye = rotate_activation_padded(eye, group_size, padded_features)  # [in_features, padded]
        rot_cache[key] = rot_eye
    return (rot_eye @ l2.t()).contiguous()  # [in_features, r_stab]


def _fused_int4_convrot_lora_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patched LoRAModule.forward for an int4cr-quantized base under a plain LoRA adapter.

    ``y = int4_backbone(x) + delta`` where ``delta`` is computed with the concat trick:
    a single down GEMM ``x @ [S | A]`` reads the activation once for both the frozen
    stabilizer and the trainable LoRA down projections, and a single up GEMM
    ``d @ [L1^T | s*B]`` produces both up projections in one pass. Standard autograd ops
    are used (not a custom Function), so the surrounding autocast context selects their
    dtypes. The fused operation order can differ numerically from the eager path. Gradients
    flow only to the LoRA down/up parameters (S and L1 are frozen).
    """
    if not self.enabled:
        return self.org_forward(x)

    org = self.org_forward.__self__
    shape = getattr(org, "int4_shape")
    group = getattr(org, "int4_convrot_groupsize", None)
    group_size = _module_int4_group(org)
    y_backbone = _Int4ConvRotLinearFunction.apply(
        x,
        org.weight,
        org.scale_weight,
        org.bias,
        shape,
        group_size if group is None else group,
        None,
        None,
        self._fused_int4_rotate,
    )

    lora_input = self._lora_input(x)
    down_w = self.lora_down.weight  # [r_lora, K]
    up_w = self.lora_up.weight  # [out, r_lora]
    scale = self.multiplier * self.scale

    stab_down_raw = self._fused_int4_stab_down_raw
    stab_l1 = self._fused_int4_stab_l1
    if stab_down_raw is not None:
        if stab_down_raw.device != lora_input.device:
            stab_down_raw = stab_down_raw.to(lora_input.device)
            stab_l1 = stab_l1.to(lora_input.device)
            self._fused_int4_stab_down_raw = stab_down_raw
            self._fused_int4_stab_l1 = stab_l1
        w_down = torch.cat([stab_down_raw.to(down_w.dtype), down_w.t()], dim=1)  # [K, r_stab + r_lora]
        w_up = torch.cat([stab_l1.to(up_w.dtype).t(), up_w.t() * scale], dim=0)  # [r_stab + r_lora, out]
    else:
        w_down = down_w.t()
        w_up = up_w.t() * scale

    d = lora_input @ w_down  # one read of the activation for both down projections
    delta = d @ w_up
    return self._match_org_dtype(y_backbone + delta, y_backbone)


def _fused_lora_eligible(lora_module) -> tuple[bool, str]:
    """Whether ``lora_module`` may use the fused triple-branch path; else a fallback reason."""
    if _int4_activation_bits() != 4:
        return False, "W4A8 activation mode uses the legacy eager down/up path (fused triple-branch is W4A4-only)"
    if lora_module.__class__.__name__ != "LoRAModule":
        return False, "DoRA/OFT/inference adapters are not fused (v1 supports plain LoRA only)"
    if getattr(lora_module, "split_dims", None) is not None:
        return False, "split-dim (fused-qkv) LoRA is not fused (v1)"
    if getattr(lora_module, "adaptive_rank", False):
        return False, "adaptive-rank LoRA is not fused (rank weights sit between down and up)"
    for attr in ("dropout", "rank_dropout", "module_dropout"):
        if getattr(lora_module, attr, None) is not None:
            return False, "dropout-configured LoRA is not fused (dropout breaks the down/up concat)"
    org_forward = getattr(lora_module, "org_forward", None)
    org = getattr(org_forward, "__self__", None)
    if org is None or not isinstance(org, nn.Linear):
        return False, "adapter base module is not an nn.Linear"
    if not (hasattr(org, "scale_weight") and hasattr(org, "int4_shape") and getattr(org, "weight", None) is not None):
        return False, "adapter base is not int4cr-quantized"
    if org.weight.dtype != torch.uint8:
        return False, "adapter base is not int4cr-quantized"
    if _use_weight_only_diagnostic():
        return False, "weight-only diagnostic mode routes the backbone through dequant"
    out_features, in_features, padded_features = _module_int4_shape(org)
    if _module_int4_awq_scales(org, in_features) is not None:
        return False, "AWQ-scaled int4cr layers are not fused (v1)"
    return True, ""


def apply_int4_convrot_fused_lora(network) -> int:
    """Route eligible int4cr-backed LoRA modules through the fused triple-branch forward.

    No-op (returns 0 without rebinding module forwards) unless ``LTX2_INT4_CONVROT_FUSE_LORA``
    is enabled. Precomputes the rotation-folded stabilizer down weight once per module and
    binds a fused ``forward``; ineligible modules keep their existing forward. The LoRA
    down/up tensors remain the module's own parameters, so the optimizer and checkpoint
    state dict are unchanged.
    """
    if not _int4_fuse_lora_enabled() or network is None:
        return 0
    loras = getattr(network, "unet_loras", None)
    if loras is None:
        return 0

    rot_cache: dict[Any, torch.Tensor] = {}
    patched = 0
    for lm in loras:
        eligible, reason = _fused_lora_eligible(lm)
        if not eligible:
            _fused_lora_log_fallback(reason)
            continue
        org = lm.org_forward.__self__
        out_features, in_features, padded_features = _module_int4_shape(org)
        group_size = _module_int4_group(org)
        rotate = _module_int4_rotate(org)
        stabilizer = _module_int4_stabilizer(org, out_features, padded_features)
        if stabilizer is not None:
            stab_l1, stab_l2 = stabilizer
            lm._fused_int4_stab_down_raw = _fused_stab_down_raw(
                stab_l2,
                group_size=group_size,
                in_features=in_features,
                padded_features=padded_features,
                rotate=rotate,
                device=org.weight.device,
                rot_cache=rot_cache,
            )
            lm._fused_int4_stab_l1 = stab_l1.to(device=org.weight.device).contiguous()
        else:
            lm._fused_int4_stab_down_raw = None
            lm._fused_int4_stab_l1 = None
        lm._fused_int4_rotate = rotate
        # ``apply_to`` rebound the base Linear's ``forward`` to the adapter's forward; that
        # bound method (not ``lm.forward``) is what the transformer invokes, so re-point it
        # to the fused forward. ``lm.org_forward`` still holds the original int4 forward, so
        # the fused method's backbone call does not recurse.
        org.forward = types.MethodType(_fused_int4_convrot_lora_forward, lm)
        patched += 1

    logger.info("INT4 ConvRot fused-LoRA: routed %d LoRA modules through the fused triple-branch path.", patched)
    return patched
