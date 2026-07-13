import logging
import os
from collections.abc import Iterable
from typing import Any

import torch

from .automagic import Automagic
from .automagic2 import Automagic2
from .automagic3 import Automagic3

logger = logging.getLogger(__name__)

AUTOMAGIC_OPTIMIZER_CLASSES = {
    "automagic": Automagic,
    "automagic2": Automagic2,
    "automagic3": Automagic3,
}

_ADAFACTOR_ONLY_KWARGS = {"relative_step", "scale_parameter", "warmup_init"}


def is_automagic_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type.strip().lower() in AUTOMAGIC_OPTIMIZER_CLASSES


def _world_size() -> int:
    """Return the active process count without initializing Accelerate state."""
    try:
        from accelerate.state import PartialState

        if PartialState._shared_state:
            return int(PartialState().num_processes)
    except Exception:
        pass

    for name in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
        try:
            return max(1, int(os.environ.get(name, "1")))
        except ValueError:
            continue
    return 1


def _fused_incompatibilities(args: Any) -> list[str]:
    problems = []
    if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
        problems.append("gradient_accumulation_steps must be 1")
    if float(getattr(args, "max_grad_norm", 0.0)) != 0.0:
        problems.append("max_grad_norm must be 0 because fused updates bypass trainer gradient clipping")
    if str(getattr(args, "mixed_precision", "no")).lower() == "fp16":
        problems.append("mixed_precision=fp16 is unsupported because fused updates run before GradScaler unscaling")
    if bool(getattr(args, "block_swap_optimizer_patch_params", False)):
        problems.append("block_swap_optimizer_patch_params must be disabled because fused updates do not retain gradients to patch")
    if _world_size() != 1:
        problems.append("fused Automagic optimizers currently require single-process training")
    return problems


def prepare_automagic_optimizer(
    optimizer_type: str,
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
    learning_rate: float,
    optimizer_kwargs: dict[str, Any],
    args: Any,
) -> tuple[type[torch.optim.Optimizer], torch.optim.Optimizer, dict[str, Any]]:
    """Construct an Automagic optimizer with safe trainer-aware fused behavior."""
    normalized_type = optimizer_type.strip().lower()
    optimizer_class = AUTOMAGIC_OPTIMIZER_CLASSES[normalized_type]
    resolved_kwargs = dict(optimizer_kwargs)

    ignored_kwargs = sorted(_ADAFACTOR_ONLY_KWARGS.intersection(resolved_kwargs))
    if ignored_kwargs:
        for name in ignored_kwargs:
            resolved_kwargs.pop(name)
        logger.warning(
            "Ignoring Adafactor-only optimizer arguments for %s: %s",
            optimizer_class.__name__,
            ", ".join(ignored_kwargs),
        )

    if normalized_type == "automagic2" and "fused" in resolved_kwargs:
        raise ValueError("Automagic2 is always fused and does not accept fused=")

    problems = _fused_incompatibilities(args)
    offload_block_swap_gradients = bool(getattr(args, "full_finetune", False)) and int(
        getattr(args, "blocks_to_swap", 0) or 0
    ) > 0
    if normalized_type == "automagic2":
        if problems:
            raise ValueError("Automagic2 uses fused backward updates and is incompatible with this configuration: " + "; ".join(problems))
        logger.info("use Automagic2 fused-backward optimizer | %s", resolved_kwargs)
    elif normalized_type == "automagic3":
        explicitly_configured = "fused" in resolved_kwargs
        if not explicitly_configured:
            resolved_kwargs["fused"] = not problems
            if problems:
                logger.info(
                    "Automagic3 selected safe non-fused mode because fused mode is incompatible: %s",
                    "; ".join(problems),
                )
            else:
                logger.info("Automagic3 selected fused mode for this compatible single-process configuration")
        elif bool(resolved_kwargs["fused"]) and problems:
            raise ValueError("Automagic3 fused=True is incompatible with this configuration: " + "; ".join(problems))
        if not resolved_kwargs["fused"] and offload_block_swap_gradients:
            resolved_kwargs.setdefault("offload_gradients", True)
        logger.info("use Automagic3 optimizer | %s", resolved_kwargs)
    else:
        if offload_block_swap_gradients:
            explicitly_configured = "fused" in resolved_kwargs
            if not explicitly_configured:
                resolved_kwargs["fused"] = not problems
            elif bool(resolved_kwargs["fused"]) and problems:
                raise ValueError("Automagic fused=True is incompatible with this configuration: " + "; ".join(problems))
            if resolved_kwargs["fused"]:
                resolved_kwargs.setdefault("offload_state", True)
                logger.info("Automagic selected fused block-swap mode with CPU optimizer-state offload")
            else:
                resolved_kwargs.setdefault("offload_gradients", True)
                logger.info("Automagic selected non-fused mode with CPU gradient offload")
        logger.info("use Automagic optimizer | %s", resolved_kwargs)

    optimizer = optimizer_class(params, lr=learning_rate, **resolved_kwargs)
    return optimizer_class, optimizer, resolved_kwargs


def get_adaptive_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    """Read optimizer-managed rates through optional Accelerate wrappers."""
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    getter = getattr(raw_optimizer, "get_learning_rates", None)
    if callable(getter):
        return getter()
    return [group["lr"] for group in raw_optimizer.param_groups]


def materialize_stochastic_gradients(optimizer: torch.optim.Optimizer) -> None:
    """Expose low-precision accumulated grads before reduction and clipping."""
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(raw_optimizer, (Automagic, Automagic3)):
        return
    if isinstance(raw_optimizer, (Automagic, Automagic3)) and raw_optimizer.fused:
        return

    for group in raw_optimizer.param_groups:
        for parameter in group["params"]:
            accumulated = getattr(parameter, "_accum_grad", None)
            if accumulated is None:
                continue
            target_device = parameter.device
            if accumulated.device != target_device:
                accumulated = accumulated.to(target_device, non_blocking=target_device.type == "cuda")
            if parameter.grad is None:
                parameter.grad = accumulated
            else:
                if parameter.grad.device != target_device:
                    parameter.grad = parameter.grad.to(target_device, non_blocking=target_device.type == "cuda")
                parameter.grad.add_(accumulated)
            del parameter._accum_grad


def uses_fused_backward(optimizer: torch.optim.Optimizer) -> bool:
    """Return whether an Automagic optimizer consumes gradients during backward."""
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    return isinstance(raw_optimizer, Automagic2) or (
        isinstance(raw_optimizer, (Automagic, Automagic3)) and raw_optimizer.fused
    )


def should_patch_block_swap_gradients(optimizer: torch.optim.Optimizer, requested: bool = False) -> bool:
    """Return whether gradients must follow parameters moved by block swapping."""
    if requested:
        return True
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    return isinstance(raw_optimizer, (Automagic, Automagic3)) and not uses_fused_backward(raw_optimizer)


def move_optimizer_gradients_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    """Move retained gradients to their block-swapped parameter devices."""
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    for group in raw_optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and parameter.device != parameter.grad.device:
                parameter.grad = parameter.grad.to(parameter.device, non_blocking=parameter.device.type == "cuda")
