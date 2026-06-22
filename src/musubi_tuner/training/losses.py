from __future__ import annotations

from typing import Optional

import torch


def per_element_loss(pred: torch.Tensor, tgt: torch.Tensor, loss_type: str = "mse", huber_delta: float = 1.0) -> torch.Tensor:
    """Compute per-element unreduced loss based on loss_type."""
    if loss_type == "mae" or loss_type == "l1":
        return torch.nn.functional.l1_loss(pred.float(), tgt.float(), reduction="none")
    if loss_type == "huber" or loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(pred.float(), tgt.float(), reduction="none", beta=huber_delta)
    return torch.nn.functional.mse_loss(pred.float(), tgt.float(), reduction="none")


_per_element_loss = per_element_loss


def apply_loss_mask(
    per_elem: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a per-element loss to a scalar with optional mask weighting."""
    if mask is None:
        return per_elem.mean(), {}

    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    metrics: dict[str, float] = {
        "mask_active": float(denom.detach().float().item()),
        "loss_unmasked": float(per_elem.detach().float().mean().item()),
    }
    if denom.item() == 0:
        loss = per_elem.mean()
    else:
        loss = (per_elem * mask_f).div(denom).mean()
    metrics["loss_masked"] = float(loss.detach().float().item())
    return loss, metrics


def apply_loss_mask_per_sample(
    per_elem: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a per-element loss with PER-SAMPLE mask renormalization.

    Each batch element is weighted equally regardless of how much of it is masked in (mean over a
    sample's own active elements, then mean over the batch) — so a heavily-conditioned sample is not
    down-weighted relative to a lightly-conditioned sibling the way the batch-global denominator in
    ``apply_loss_mask`` does. A fully-masked sample (no active elements) contributes ~0 via the eps
    clamp, degrading safely instead of reverting to an unmasked mean.

    The default path keeps ``apply_loss_mask`` (batch-global) for byte-identity; this is used when
    ``ltx2_per_sample_loss`` is set, which is auto-enabled whenever LTX-2 conditioning is active and a
    no-op for a plain run. The mask-broadcast block is duplicated from ``apply_loss_mask`` on purpose,
    so that function stays byte-for-byte unchanged for the off-path.
    """
    if mask is None:
        return per_elem.mean(), {}

    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype).expand_as(per_elem)
    dims = tuple(range(1, per_elem.dim()))  # all non-batch dims
    active = mask_f.sum(dim=dims)  # [B] active-element count per sample
    num = (per_elem * mask_f).sum(dim=dims)  # [B]
    per_sample = num / active.clamp(min=1e-8)  # [B] mean over each sample's own active elements
    loss = per_sample.mean()
    metrics: dict[str, float] = {
        "mask_active": float(mask_f.mean().detach().float().item()),
        "loss_unmasked": float(per_elem.detach().float().mean().item()),
        "loss_masked": float(loss.detach().float().item()),
    }
    return loss, metrics


def reduce_masked_loss(
    per_elem: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    per_sample: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Route to the per-sample reducer when ``per_sample`` else the batch-global one.

    The single routing seam for LTX-2 conditioning: callers pass ``per_sample=True`` only when a
    conditioning recipe is active, so the default path (``per_sample=False``) is byte-identical to
    calling ``apply_loss_mask`` directly.
    """
    if per_sample:
        return apply_loss_mask_per_sample(per_elem, mask)
    return apply_loss_mask(per_elem, mask)
