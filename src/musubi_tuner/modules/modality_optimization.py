from __future__ import annotations

from typing import Any

import torch


MODALITIES = ("video", "audio", "cross_modal")


def has_modality_group_controls(args: Any) -> bool:
    return any(
        getattr(args, f"{modality}_{suffix}", None) is not None
        for modality in MODALITIES
        for suffix in ("max_grad_norm", "weight_decay")
    )


def has_modality_clip_overrides(args: Any) -> bool:
    return any(getattr(args, f"{modality}_max_grad_norm", None) is not None for modality in MODALITIES)


def clip_modality_optimizer_groups(
    args: Any,
    accelerator: Any,
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    """Clip disjoint optimizer groups with modality-specific thresholds.

    The optimizer groups are split and tagged during adapter parameter
    assembly. Unspecified modalities and untagged auxiliary parameters retain
    the global ``max_grad_norm`` threshold.
    """

    if not has_modality_clip_overrides(args):
        return {}

    accelerator.unscale_gradients(optimizer)
    inner_optimizer = getattr(optimizer, "optimizer", optimizer)
    global_max_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
    grouped_params: dict[str, list[torch.nn.Parameter]] = {modality: [] for modality in MODALITIES}
    grouped_params["other"] = []
    seen: set[int] = set()

    for group in inner_optimizer.param_groups:
        modality = str(group.get("modality", "other"))
        if modality not in grouped_params:
            modality = "other"
        for param in group["params"]:
            if param.grad is None or id(param) in seen:
                continue
            seen.add(id(param))
            grouped_params[modality].append(param)

    norms: dict[str, torch.Tensor] = {}
    for modality, params in grouped_params.items():
        if not params:
            continue
        max_norm = global_max_norm if modality == "other" else getattr(args, f"{modality}_max_grad_norm", None)
        if max_norm is None:
            max_norm = global_max_norm
        max_norm = float(max_norm)
        if max_norm > 0:
            norms[modality] = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return norms
