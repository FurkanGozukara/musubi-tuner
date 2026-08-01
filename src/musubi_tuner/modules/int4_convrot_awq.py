"""Dataset-independent AWQ-style scaling for INT4 ConvRot weights."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

INT4_CONVROT_AWQ_SCALE_SUFFIX = ".int4_awq_scales"
DEFAULT_INT4_CONVROT_AWQ_ALPHA = 0.25
DEFAULT_INT4_CONVROT_AWQ_MIN_SCALE = 1.0e-4
DEFAULT_INT4_CONVROT_AWQ_MAX_SCALE = 1.0e4

logger = logging.getLogger(__name__)


def default_int4_convrot_awq_scales_path(model_path: str | list[str]) -> str:
    if isinstance(model_path, list):
        model_path = model_path[0]
    base, _ = os.path.splitext(str(model_path))
    return base + ".int4_convrot_awq_scales.safetensors"


def compute_int4_convrot_awq_scale(
    weight: torch.Tensor,
    *,
    alpha: float = DEFAULT_INT4_CONVROT_AWQ_ALPHA,
    activation_stat: torch.Tensor | None = None,
    min_scale: float = DEFAULT_INT4_CONVROT_AWQ_MIN_SCALE,
    max_scale: float = DEFAULT_INT4_CONVROT_AWQ_MAX_SCALE,
) -> torch.Tensor:
    """Compute per-input-channel AWQ scales for a 2D Linear weight.

    The default path is dataset-independent: activation statistics are treated
    as uniform, so scale importance comes from weight column magnitudes. Passing
    ``activation_stat`` enables the same formula with measured activation norms.
    """

    if weight.ndim != 2:
        raise ValueError(f"INT4 ConvRot AWQ expects a 2D weight, got shape {tuple(weight.shape)}")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"INT4 ConvRot AWQ alpha must be in [0, 1], got {alpha}")

    w = weight.detach().float()
    w_norm = w.abs().amax(dim=0).clamp(min=1.0e-12)
    if activation_stat is None:
        act_norm = torch.ones_like(w_norm)
    else:
        act_norm = activation_stat.detach().to(device=w_norm.device, dtype=torch.float32).reshape(-1)
        if act_norm.numel() != w_norm.numel():
            raise ValueError(f"activation_stat length {act_norm.numel()} does not match weight input width {w_norm.numel()}")
        act_norm = act_norm.abs().clamp(min=1.0e-12)

    importance = (w_norm * act_norm).clamp(min=1.0e-12)
    normalized = importance / importance.mean().clamp(min=1.0e-12)
    scale = normalized.pow(float(alpha)).clamp(min=float(min_scale), max=float(max_scale))
    return scale.cpu().to(torch.float32)


def apply_int4_convrot_awq_scale_to_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"INT4 ConvRot AWQ expects a 2D weight, got shape {tuple(weight.shape)}")
    scale = scale.to(device=weight.device, dtype=torch.float32).reshape(-1)
    if scale.numel() != weight.shape[1]:
        raise ValueError(f"INT4 ConvRot AWQ scale length {scale.numel()} does not match weight width {weight.shape[1]}")
    scaled = weight.float() * scale.reshape(1, -1)
    return scaled.to(dtype=weight.dtype if weight.is_floating_point() else torch.float32)


def load_int4_convrot_awq_scales(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    scales = load_file(path, device="cpu")
    logger.info("INT4 ConvRot AWQ: loaded %d scale tensors from %s", len(scales), path)
    return {str(k): v.detach().cpu().float() for k, v in scales.items()}


def save_int4_convrot_awq_scales(scales: dict[str, torch.Tensor], path: str) -> None:
    from safetensors.torch import save_file

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_file({str(k): v.detach().cpu().float() for k, v in scales.items()}, path)
    logger.info("INT4 ConvRot AWQ: saved %d scale tensors to %s", len(scales), path)


def summarize_int4_convrot_awq_scales(scales: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not scales:
        return {"num_layers": 0}
    flat = torch.cat([scale.detach().float().reshape(-1).cpu() for scale in scales.values()])
    return {
        "num_layers": len(scales),
        "num_channels": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
    }
