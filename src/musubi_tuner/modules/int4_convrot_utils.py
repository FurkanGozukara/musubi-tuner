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

logger = logging.getLogger(__name__)
_CUTLASS_INT4 = "unset"
_CUTLASS_INT8_MM = "unset"
_CUTLASS_INT4_TRANSPOSE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_CUTLASS_INT4_TRANSPOSE_CACHE_BYTES = 0
_INT_MM_MIN_CUDA_CAPABILITY = (7, 5)


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
) -> torch.Tensor:
    q = unpack_int4(packed, padded_features).float()
    deq_rot = q * scale.float()
    h = build_hadamard(group_size, device=deq_rot.device, dtype=torch.float32)
    dense_padded = rotate_weight(deq_rot, h, group_size, inverse=True)
    return dense_padded[:, :in_features].to(dtype)


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
    h = build_hadamard(group_size, device=calc_device, dtype=torch.float32)

    dot = ref_norm_sq = deq_norm_sq = sqerr_sum = abserr_sum = 0.0
    max_abs_error = 0.0
    q_absmax = 0
    scale_min = float("inf")
    scale_max = 0.0

    for row_slice in _row_slices(out_features, padded_features, int(max_chunk_elements)):
        w_chunk = weight[row_slice].to(device=calc_device, dtype=torch.float32)
        w_padded = pad_last_dim(w_chunk, padded_features)
        rotated = rotate_weight(w_padded, h, group_size)
        q_packed, scale_chunk = quantize_int4_rowwise(rotated, mse_clip=mse_clip)
        packed_out[row_slice].copy_(q_packed)
        scale_out[row_slice].copy_(scale_chunk)

        if collect_quality:
            q_unpacked = unpack_int4(q_packed, padded_features).float()
            deq = rotate_weight(q_unpacked * scale_chunk, h, group_size, inverse=True)[:, :in_features]
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
        )
    return packed_out, scale_out, shape, quality


def comfy_quant_tensor(
    group_size: int,
    in_features: int,
    padded_features: int,
    *,
    convrot: bool = True,
    awq: bool = False,
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
    return os.getenv("LTX2_INT4_CONVROT_BACKEND", "auto").strip().lower()


def _int4_activation_bits() -> int:
    value = os.getenv("LTX2_INT4_CONVROT_ACT_BITS", "8").strip().lower()
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
    def forward(ctx, x, packed_weight, weight_scale, bias, int4_shape, convrot_groupsize):
        out_features, in_features, padded_features = (int(v) for v in int4_shape.detach().cpu().reshape(-1).tolist())
        group_size = (
            int(convrot_groupsize.detach().reshape(-1)[0].item())
            if isinstance(convrot_groupsize, torch.Tensor)
            else int(convrot_groupsize)
        )
        original_shape = x.shape
        output_dtype = _autocast_output_dtype(x)
        x_rot = rotate_activation_padded(x, group_size, padded_features)
        x_2d = x_rot.reshape(-1, padded_features)

        activation_bits = _int4_activation_bits()
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
        ctx.save_for_backward(packed_weight, weight_scale, int4_shape, torch.tensor(group_size, device=x.device, dtype=torch.int32))
        ctx.input_dtype = x.dtype
        ctx.activation_bits = activation_bits
        ctx.original_shape = original_shape
        return output.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise RuntimeError("INT4 ConvRot base weights are frozen; full fine-tuning is not supported.")
        packed_weight, weight_scale, int4_shape, group_tensor = ctx.saved_tensors
        out_features, in_features, padded_features = (int(v) for v in int4_shape.detach().cpu().reshape(-1).tolist())
        group_size = int(group_tensor.detach().reshape(-1)[0].item())
        go = grad_output.reshape(-1, out_features)
        go_scaled = go.float() * weight_scale.reshape(1, out_features).float()

        if ctx.activation_bits == 8:
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
        grad_input = rotate_activation_padded(grad_rot, group_size, padded_features, inverse=True)[..., :in_features]
        grad_input = grad_input.reshape(*ctx.original_shape)
        return grad_input, None, None, None, None, None


def int4_convrot_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    out_features, in_features, padded_features = _module_int4_shape(self)
    x = _apply_int4_awq_activation_scale(self, x, in_features)
    if _use_weight_only_diagnostic():
        weight = dequantize_int4_convrot_weight(
            self.weight,
            self.scale_weight,
            _module_int4_group(self),
            in_features,
            padded_features,
            dtype=x.dtype if x.is_floating_point() else torch.float32,
        )
        return F.linear(x, weight, self.bias)
    group_size = _module_int4_group(self)
    shape = getattr(self, "int4_shape")
    group = getattr(self, "int4_convrot_groupsize")
    return _Int4ConvRotLinearFunction.apply(
        x, self.weight, self.scale_weight, self.bias, shape, group_size if group is None else group
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
        module.forward = int4_convrot_linear_forward.__get__(module, type(module))
        patched += 1
    logger = __import__("logging").getLogger(__name__)
    backend = "cuda-lazy" if torch.cuda.is_available() else "torch-fallback"
    act_bits = _int4_activation_bits()
    logger.info("INT4 ConvRot (%s): patched %d linear layers, activation mode W4A%d", backend, patched, act_bits)
    return model
