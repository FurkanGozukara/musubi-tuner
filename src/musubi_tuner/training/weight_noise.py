"""Optional post-step weight perturbation helpers."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any


WEIGHT_NOISE_MODES = ("none", "relative", "absolute")
DEFAULT_WEIGHT_NOISE_SCALE = 0.01


def add_weight_noise_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weight_noise_mode",
        type=str,
        default="none",
        choices=WEIGHT_NOISE_MODES,
        help=(
            "Post-optimizer weight perturbation mode. 'none' disables it. 'relative' scales noise by "
            "each tensor RMS; 'absolute' uses --weight_noise_scale directly."
        ),
    )
    parser.add_argument(
        "--weight_noise_scale",
        type=float,
        default=DEFAULT_WEIGHT_NOISE_SCALE,
        help="Noise scale for --weight_noise_mode. Used only when the mode is not 'none'.",
    )


def validate_weight_noise_args(args: Any) -> None:
    mode = str(getattr(args, "weight_noise_mode", "none") or "none").lower()
    if mode not in WEIGHT_NOISE_MODES:
        raise ValueError(f"--weight_noise_mode must be one of: {', '.join(WEIGHT_NOISE_MODES)}")
    setattr(args, "weight_noise_mode", mode)

    scale = float(getattr(args, "weight_noise_scale", DEFAULT_WEIGHT_NOISE_SCALE) or 0.0)
    if mode != "none" and scale <= 0.0:
        raise ValueError("--weight_noise_scale must be > 0 when --weight_noise_mode is enabled")
    setattr(args, "weight_noise_scale", scale)


def apply_weight_noise_to_optimizer(optimizer: Any, args: Any, *, global_step: int) -> int:
    mode = str(getattr(args, "weight_noise_mode", "none") or "none").lower()
    if mode == "none":
        return 0
    scale = float(getattr(args, "weight_noise_scale", DEFAULT_WEIGHT_NOISE_SCALE) or 0.0)
    if scale <= 0.0:
        raise ValueError("--weight_noise_scale must be > 0 when --weight_noise_mode is enabled")
    base_seed = int(getattr(args, "seed", 0) or 0)
    return apply_weight_noise_to_parameters(
        _iter_optimizer_parameters(optimizer),
        mode=mode,
        scale=scale,
        base_seed=base_seed,
        global_step=int(global_step),
    )


def _iter_optimizer_parameters(optimizer: Any) -> Iterable[Any]:
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for param in group.get("params", ()):
            if param is None or id(param) in seen:
                continue
            seen.add(id(param))
            yield param


def apply_weight_noise_to_parameters(
    parameters: Iterable[Any],
    *,
    mode: str,
    scale: float,
    base_seed: int,
    global_step: int,
) -> int:
    import torch

    mode = mode.lower()
    if mode not in {"relative", "absolute"}:
        return 0

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    touched = 0
    with torch.no_grad():
        for index, param in enumerate(parameters):
            if not getattr(param, "requires_grad", False):
                continue
            if not param.is_floating_point() or param.dtype not in supported_dtypes:
                continue
            if param.numel() == 0 or param.device.type == "meta":
                continue

            if mode == "relative":
                rms = param.detach().to(dtype=torch.float32).pow(2).mean().sqrt()
                rms = torch.where(torch.isfinite(rms), rms, torch.zeros_like(rms))
                std = rms.to(device=param.device, dtype=param.dtype) * float(scale)
            else:
                std = float(scale)

            generator = torch.Generator(device=param.device)
            generator.manual_seed(_weight_noise_seed(base_seed=base_seed, global_step=global_step, param_index=index))
            noise = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator)
            param.add_(noise * std)
            touched += 1

    return touched


def _weight_noise_seed(*, base_seed: int, global_step: int, param_index: int) -> int:
    seed = (int(base_seed) & 0xFFFFFFFF) + 1_000_003 * (int(global_step) + 1) + 9_176 * (int(param_index) + 1)
    return seed % (2**63 - 1)
