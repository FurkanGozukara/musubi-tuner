"""Helpers for attaching frozen LoRA-style networks during training."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch

from musubi_tuner.training.model_helpers import load_network_state_dict

logger = logging.getLogger(__name__)


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _accelerator_print(accelerator: Any, message: str) -> None:
    print_fn = getattr(accelerator, "print", None)
    if callable(print_fn):
        print_fn(message)
    else:
        logger.info(message)


def frozen_network_specs(args: Any) -> list[tuple[str, float]]:
    """Return validated ``(path, multiplier)`` pairs for frozen network weights."""

    paths = [str(path) for path in _as_sequence(getattr(args, "frozen_network_weights", None)) if str(path)]
    multipliers = _as_sequence(getattr(args, "frozen_network_multiplier", None))

    if not paths:
        return []
    if len(multipliers) > len(paths):
        raise ValueError("--frozen_network_multiplier cannot contain more values than --frozen_network_weights")

    specs: list[tuple[str, float]] = []
    for i, path in enumerate(paths):
        multiplier = 1.0 if i >= len(multipliers) else float(multipliers[i])
        specs.append((path, multiplier))
    return specs


def _normalize_created_network(created: Any) -> torch.nn.Module:
    if isinstance(created, tuple):
        if not created:
            raise ValueError("create_arch_network_from_weights returned an empty tuple")
        created = created[0]
    if not isinstance(created, torch.nn.Module):
        raise TypeError("create_arch_network_from_weights must return a torch.nn.Module")
    return created


def apply_frozen_networks(
    args: Any,
    accelerator: Any,
    network_module: Any,
    transformer: torch.nn.Module,
    load_network_weights: Callable[[str, Any], dict[str, torch.Tensor]],
) -> list[torch.nn.Module]:
    """Load and attach frozen networks to the transformer.

    The returned modules are intentionally separate from the trainable network. Keep the
    references alive for checkpointing clarity and device/dtype placement.
    """

    specs = frozen_network_specs(args)
    if not specs:
        return []

    create_from_weights = getattr(network_module, "create_arch_network_from_weights", None)
    if not callable(create_from_weights):
        raise ValueError("--frozen_network_weights requires a network module with create_arch_network_from_weights")

    frozen_networks: list[torch.nn.Module] = []
    for weight_path, multiplier in specs:
        _accelerator_print(accelerator, f"attaching frozen network weights: {weight_path} with multiplier {multiplier:g}")

        weights_sd = load_network_weights(weight_path, network_module)
        network = _normalize_created_network(create_from_weights(multiplier, weights_sd, unet=transformer))
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        info = load_network_state_dict(network, weights_sd, strict=False)
        network.requires_grad_(False)
        network.eval()
        frozen_networks.append(network)

        _accelerator_print(accelerator, f"attached frozen network weights from {weight_path}: {info}")

    return frozen_networks


def prepare_frozen_networks_for_training(
    frozen_networks: Sequence[torch.nn.Module],
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_parallel: bool,
    place_network_for_model_parallel: Callable[..., Any] | None = None,
    args: Any = None,
    accelerator: Any = None,
    transformer: torch.nn.Module | None = None,
) -> None:
    """Place frozen networks on the devices used by training and keep them frozen."""

    for network in frozen_networks:
        network.requires_grad_(False)
        network.eval()
        network.to(dtype=dtype)

        if model_parallel and callable(place_network_for_model_parallel):
            place_network_for_model_parallel(args, accelerator, transformer, network)
        else:
            network.to(device=device)

        network.requires_grad_(False)
        network.eval()
