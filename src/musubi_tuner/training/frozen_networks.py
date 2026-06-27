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


def _module_prefix(key: str) -> str | None:
    if "." not in key:
        return None
    return key.split(".", 1)[0]


def _state_dict_module_names(state_dict: dict[str, torch.Tensor]) -> set[str]:
    return {prefix for key in state_dict.keys() if (prefix := _module_prefix(key)) is not None}


def split_warm_start_state_dict(
    trainable_network: torch.nn.Module,
    weights_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split warm-start weights into trainable-network keys and surplus module keys."""

    active_modules = _state_dict_module_names(trainable_network.state_dict())
    active_sd: dict[str, torch.Tensor] = {}
    surplus_sd: dict[str, torch.Tensor] = {}

    for key, value in weights_sd.items():
        prefix = _module_prefix(key)
        if prefix is None or prefix in active_modules:
            active_sd[key] = value
        else:
            surplus_sd[key] = value

    return active_sd, surplus_sd


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


def apply_warm_start_surplus_network(
    args: Any,
    accelerator: Any,
    network_module: Any,
    transformer: torch.nn.Module,
    trainable_network: torch.nn.Module,
    weights_sd: dict[str, torch.Tensor],
) -> torch.nn.Module | None:
    """Attach warm-start modules that are outside the active trainable network."""

    if not getattr(args, "network_freeze_surplus_modules", False):
        return None
    if not weights_sd:
        return None

    create_from_weights = getattr(network_module, "create_arch_network_from_weights", None)
    if not callable(create_from_weights):
        raise ValueError("--network_freeze_surplus_modules requires create_arch_network_from_weights support")

    _, surplus_sd = split_warm_start_state_dict(trainable_network, weights_sd)
    surplus_modules = sorted(_state_dict_module_names(surplus_sd))
    if not surplus_modules:
        _accelerator_print(accelerator, "warm-start surplus freeze: no surplus modules found")
        return None

    surplus_module_set = set(surplus_modules)
    if not surplus_sd:
        _accelerator_print(accelerator, "warm-start surplus freeze: no loadable surplus weights found")
        return None

    network = _normalize_created_network(create_from_weights(1.0, surplus_sd, unet=transformer))
    created_modules = _state_dict_module_names(network.state_dict())
    if not created_modules:
        _accelerator_print(
            accelerator,
            f"warm-start surplus freeze: skipped {len(surplus_modules)} surplus modules because none matched this model",
        )
        return None

    loadable_sd = {
        key: value for key, value in surplus_sd.items() if (prefix := _module_prefix(key)) is not None and prefix in created_modules
    }
    skipped_modules = sorted(surplus_module_set - created_modules)

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    info = load_network_state_dict(network, loadable_sd, strict=False)
    network.requires_grad_(False)
    network.eval()

    _accelerator_print(
        accelerator,
        "warm-start surplus freeze: "
        f"attached {len(created_modules)} frozen modules; "
        f"skipped {len(skipped_modules)} incompatible modules; "
        f"load info: {info}",
    )
    return network


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
