"""HFATO - High-Frequency Awareness Training Objective (ViBe, arxiv 2603.23326).

Destroys high-frequency details in clean latents via downsample-upsample before
noise addition, then supervises the model to reconstruct the ORIGINAL clean
latents.  Forces the model to learn high-frequency detail recovery from images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class HFATOConfig:
    scale_factor: float = 0.5  # ViBe Stage 2 uses 0.5 spatial downsample
    interpolation: str = "trilinear"  # ViBe Stage 2 uses 5D trilinear, align_corners=False
    probability: float = 1.0  # Probability of applying HFATO per step (1.0 = always)


def parse_hfato_args(raw_args: Optional[list[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict.  Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    out: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"HFATO arg must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def degrade_latents(
    latents: torch.Tensor,
    scale_factor: float = 0.5,
    interpolation: str = "trilinear",
) -> torch.Tensor:
    """Apply downsample-then-upsample degradation to 5D video latents.

    The default path matches the ViBe authors' Stage 2 HFATO implementation:
    5D trilinear interpolation with scale_factor=(1.0, 0.5, 0.5) and
    align_corners=False.  Legacy 2D spatial modes remain available only when
    requested explicitly.

    Args:
        latents: ``[B, C, T, H, W]`` clean video latents.
        scale_factor: Spatial downscale factor (0.5 = halve each side).
        interpolation: Interpolation mode for ``F.interpolate``.

    Returns:
        Degraded latents with same shape as input, high-frequency info destroyed.
    """
    if latents.dim() != 5:
        raise ValueError(f"HFATO expects 5D latents [B, C, T, H, W], got shape {tuple(latents.shape)}")

    B, C, T, H, W = latents.shape
    if interpolation in ("trilinear", "nearest"):
        kwargs = {}
        if interpolation == "trilinear":
            kwargs["align_corners"] = False
        x_small = F.interpolate(
            latents,
            scale_factor=(1.0, scale_factor, scale_factor),
            mode=interpolation,
            **kwargs,
        )
        return F.interpolate(
            x_small,
            size=(T, H, W),
            mode=interpolation,
            **kwargs,
        )

    if interpolation not in ("bilinear", "bicubic"):
        raise ValueError(f"HFATO interpolation must be trilinear|nearest|bilinear|bicubic, got: {interpolation}")

    x = latents.reshape(B * T, C, H, W)
    x_small = F.interpolate(
        x,
        scale_factor=scale_factor,
        mode=interpolation,
        align_corners=False,
    )
    x_restored = F.interpolate(
        x_small,
        size=(H, W),
        mode=interpolation,
        align_corners=False,
    )
    return x_restored.reshape(B, C, T, H, W)


def compute_hfato_noisy_input(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    config: HFATOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute HFATO-modified noisy model input.

    Instead of ``(1-σ)·x₀ + σ·ε``, uses ``(1-σ)·DU(x₀) + σ·ε`` where DU is the
    downsample-upsample degradation operator.

    Args:
        latents: ``[B, C, T, H, W]`` clean latents (x₀).
        noise: ``[B, C, T, H, W]`` noise tensor.
        sigma: ``[B]`` noise level in ``[0, 1]``.
        config: HFATO configuration.

    Returns:
        ``(noisy_input, degraded_latents)`` — the degraded latents are returned
        for informational purposes; the clean ``latents`` arg is the reconstruction
        target.
    """
    degraded = degrade_latents(latents, config.scale_factor, config.interpolation)

    sigma_expanded = sigma.view(-1, 1, 1, 1, 1)
    noisy = (1.0 - sigma_expanded) * degraded + sigma_expanded * noise
    return noisy, degraded


def hfato_x0_loss(
    velocity_pred: torch.Tensor,
    noisy_input: torch.Tensor,
    clean_latents: torch.Tensor,
    sigma: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute x₀-prediction loss from velocity prediction.

    ``x̂₀ = noisy - σ·v_pred``, loss = ``‖x̂₀ − x₀‖²``.

    Args:
        velocity_pred: Model's velocity prediction ``[B, C, T, H, W]``.
        noisy_input: HFATO-degraded noisy input ``[B, C, T, H, W]``.
        clean_latents: Original clean latents (reconstruction target) ``[B, C, T, H, W]``.
        sigma: ``[B]`` noise level in ``[0, 1]``.
        loss_mask: Optional per-element or per-sample mask.

    Returns:
        Scalar loss value.
    """
    sigma_expanded = sigma.view(-1, 1, 1, 1, 1).to(dtype=velocity_pred.dtype)
    x_hat_0 = noisy_input.to(dtype=velocity_pred.dtype) - sigma_expanded * velocity_pred
    per_elem = F.mse_loss(x_hat_0.float(), clean_latents.float(), reduction="none")

    if loss_mask is None:
        return per_elem.mean()

    mask = loss_mask.to(device=per_elem.device)
    # Broadcast mask to match per_elem shape (same patterns as _masked_loss)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()
