"""INT8 ConvRot helpers for LTX-2 quantized base loading.

ConvRot stores an offline-rotated int8 weight and rotates activations online:

    W_rot = W @ H.T
    x_rot = x @ H
    x_rot @ W_rot.T == x @ W.T

The base weight remains frozen for LoRA training.  The quality metrics here are
weight-reconstruction metrics, not generation-quality guarantees.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

DEFAULT_INT8_CONVROT_GROUP_SIZES = (256, 64, 16)
DEFAULT_INT8_CONVROT_CLIP_MIN = 0.55
DEFAULT_INT8_CONVROT_CLIP_STEPS = 80
DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS = 4 * 1024 * 1024

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


@dataclass
class Int8ConvRotLayerQuality:
    key: str
    shape: tuple[int, int]
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


def parse_int8_convrot_groupsizes(value: str | int | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_INT8_CONVROT_GROUP_SIZES
    if isinstance(value, int):
        groups = (value,)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "auto":
            return DEFAULT_INT8_CONVROT_GROUP_SIZES
        groups = tuple(int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip())
    else:
        groups = tuple(int(v) for v in value)
    if not groups:
        raise ValueError("INT8 ConvRot group size list is empty")
    invalid = [g for g in groups if not _is_power_of_four(g)]
    if invalid:
        raise ValueError(f"INT8 ConvRot group sizes must be powers of 4 >= 4, got {invalid}")
    return tuple(dict.fromkeys(groups))


def best_int8_convrot_groupsize(in_features: int, groupsizes: Iterable[int] | None = None) -> int | None:
    for group_size in sorted(parse_int8_convrot_groupsizes(groupsizes), reverse=True):
        if in_features % group_size == 0:
            return group_size
    return None


def build_hadamard(size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not _is_power_of_four(size):
        raise ValueError(f"Regular Hadamard size must be a power of 4 >= 4, got {size}")
    device = torch.device(device)
    cache_key = (int(size), str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h = h / math.sqrt(size)
    _HADAMARD_CACHE[cache_key] = h
    return h


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int, *, inverse: bool = False) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} is not divisible by ConvRot group size {group_size}")
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    rot = h if inverse else h.T
    rot = rot.to(device=weight.device, dtype=weight.dtype)
    return torch.matmul(grouped, rot).reshape(out_features, in_features)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int, *, inverse: bool = False) -> torch.Tensor:
    original_shape = x.shape
    features = original_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} is not divisible by ConvRot group size {group_size}")
    grouped = x.reshape(-1, features // group_size, group_size)
    rot = h.T if inverse else h
    rot = rot.to(device=x.device, dtype=x.dtype)
    return torch.matmul(grouped, rot).reshape(original_shape)


def rotate_activation_by_groupsize(x: torch.Tensor, group_size: int, *, inverse: bool = False) -> torch.Tensor:
    h = build_hadamard(group_size, device=x.device, dtype=x.dtype)
    return rotate_activation(x, h, group_size, inverse=inverse)


def quantize_int8_rowwise(
    x: torch.Tensor,
    *,
    mse_clip: bool = True,
    clip_min: float = DEFAULT_INT8_CONVROT_CLIP_MIN,
    clip_steps: int = DEFAULT_INT8_CONVROT_CLIP_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mse_clip or clip_steps <= 1:
        scale = (absmax / 127.0).clamp(min=1e-30)
        q = (x / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale.float()

    best_mse = torch.full_like(absmax, float("inf"), dtype=torch.float32)
    best_scale = (absmax / 127.0).float()
    best_q: torch.Tensor | None = None
    for ratio in torch.linspace(float(clip_min), 1.0, int(clip_steps), device=x.device, dtype=torch.float32):
        scale = (absmax.float() * ratio / 127.0).clamp(min=1e-30)
        q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
        mse = ((q.float() * scale - x.float()) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better.expand_as(q), q, best_q)
    assert best_q is not None
    return best_q, best_scale.float()


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    deq_rot = q.float() * scale.float()
    h = build_hadamard(group_size, device=deq_rot.device, dtype=torch.float32)
    return rotate_weight(deq_rot, h, group_size, inverse=True).to(dtype)


def _row_slices(num_rows: int, in_features: int, max_chunk_elements: int) -> Iterable[slice]:
    rows_per_chunk = max(1, int(max_chunk_elements) // max(int(in_features), 1))
    for start in range(0, num_rows, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, num_rows))


def _quality_from_accumulators(
    *,
    key: str,
    shape: tuple[int, int],
    group_size: int,
    numel: int,
    dot: float,
    ref_norm_sq: float,
    deq_norm_sq: float,
    sqerr_sum: float,
    abserr_sum: float,
    max_abs_error: float,
    q_absmax: int,
    scale_min: float,
    scale_max: float,
) -> Int8ConvRotLayerQuality:
    denom = math.sqrt(max(ref_norm_sq, 0.0) * max(deq_norm_sq, 0.0))
    cosine = float(dot / denom) if denom > 0 else 1.0
    mse = float(sqerr_sum / max(numel, 1))
    signal_mean_square = float(ref_norm_sq / max(numel, 1))
    sqnr_db = float(10.0 * math.log10(signal_mean_square / mse)) if mse > 0 and signal_mean_square > 0 else float("inf")
    return Int8ConvRotLayerQuality(
        key=key,
        shape=shape,
        group_size=int(group_size),
        mse=mse,
        mae=float(abserr_sum / max(numel, 1)),
        max_abs_error=float(max_abs_error),
        cosine=cosine,
        sqnr_db=sqnr_db,
        signal_mean_square=signal_mean_square,
        q_absmax=int(q_absmax),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
    )


@torch.no_grad()
def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    calc_device: str | torch.device = "cpu",
    mse_clip: bool = True,
    collect_quality: bool = False,
    key: str = "",
    max_chunk_elements: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Int8ConvRotLayerQuality | None]:
    if weight.ndim != 2:
        raise ValueError(f"INT8 ConvRot expects a 2D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} is not divisible by ConvRot group size {group_size}")

    if max_chunk_elements is None:
        max_chunk_elements = int(os.getenv("LTX2_INT8_CONVROT_CHUNK_ELEMENTS", DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS))
    calc_device = torch.device(calc_device)
    q_out = torch.empty((out_features, in_features), device=calc_device, dtype=torch.int8)
    scale_out = torch.empty((out_features, 1), device=calc_device, dtype=torch.float32)
    h = build_hadamard(group_size, device=calc_device, dtype=torch.float32)

    dot = ref_norm_sq = deq_norm_sq = sqerr_sum = abserr_sum = 0.0
    max_abs_error = 0.0
    q_absmax = 0
    scale_min = float("inf")
    scale_max = 0.0

    for row_slice in _row_slices(out_features, in_features, int(max_chunk_elements)):
        w_chunk = weight[row_slice].to(device=calc_device, dtype=torch.float32)
        rotated = rotate_weight(w_chunk, h, group_size)
        q_chunk, scale_chunk = quantize_int8_rowwise(rotated, mse_clip=mse_clip)
        q_out[row_slice].copy_(q_chunk)
        scale_out[row_slice].copy_(scale_chunk)

        if collect_quality:
            deq = rotate_weight(q_chunk.float() * scale_chunk, h, group_size, inverse=True)
            err = deq - w_chunk
            dot += float((deq * w_chunk).sum().item())
            ref_norm_sq += float((w_chunk * w_chunk).sum().item())
            deq_norm_sq += float((deq * deq).sum().item())
            sqerr_sum += float((err * err).sum().item())
            abserr_sum += float(err.abs().sum().item())
            max_abs_error = max(max_abs_error, float(err.abs().max().item()))
            q_absmax = max(q_absmax, int(q_chunk.to(torch.int16).abs().max().item()))
            scale_min = min(scale_min, float(scale_chunk.min().item()))
            scale_max = max(scale_max, float(scale_chunk.max().item()))

    quality = None
    if collect_quality:
        quality = _quality_from_accumulators(
            key=key,
            shape=(out_features, in_features),
            group_size=group_size,
            numel=out_features * in_features,
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
    return q_out, scale_out, quality


def comfy_quant_tensor(group_size: int, *, convrot: bool = True) -> torch.Tensor:
    cfg = {"format": "int8_tensorwise", "convrot": bool(convrot), "convrot_groupsize": int(group_size)}
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)


def parse_comfy_quant_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    data = bytes(int(v) for v in tensor.detach().cpu().flatten().tolist()).decode("utf-8")
    return json.loads(data)


def summarize_quality(layers: Iterable[Int8ConvRotLayerQuality]) -> dict[str, Any]:
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
    layers: Iterable[Int8ConvRotLayerQuality],
) -> dict[str, Any]:
    layer_items = list(layers)
    report = {
        "format": "ltx2_int8_convrot_quality_v1",
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
