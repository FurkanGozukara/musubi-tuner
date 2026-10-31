"""Training-only base weights: merged into the DiT for training, left out of the training samples.

A de-distillation training adapter (ostris' Turbo training adapters for Qwen-Image 2.1 and Krea 2, as Z-Image Turbo's)
is merged into a step-distilled DiT with ``--base_weights`` so the LoRA learns only the subject, and it is removed at
inference: ai-toolkit samples without it (``assistant_lora_path``: live while training, off while sampling) and ComfyUI
loads the trained LoRA on the plain Turbo model. ``--base_weights_sample_multiplier`` gives each base-weights file the
strength it has in the training samples (default: its training strength; 0 leaves a training adapter out).

The base weights stay merged for training (fp8 / ConvRot INT8 bases quantize the merged weights while loading). For
samples a frozen network with multiplier ``sample - train`` is attached outside the trained network and enabled only
while sampling, so a sample computes ``trained(x) - adapter(x)``: with a LoRA exactly the Turbo model plus the LoRA,
with input-axis DoRA ``(W + BA)(s * x) + adapter((s - 1) * x)`` where ``s`` is DoRA's per-input-channel factor
(1 at the start, the magnitudes move it by a fraction of a percent), against ComfyUI's ``(W + BA)(s' * x)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def sample_multiplier_deltas(args: Any) -> list[tuple[str, float]]:
    """``(path, sample multiplier - training multiplier)`` for every base-weights file whose strength in samples differs."""
    paths = [str(path) for path in (getattr(args, "base_weights", None) or []) if str(path)]
    sample = getattr(args, "base_weights_sample_multiplier", None)
    if not paths or sample is None:
        return []
    if isinstance(sample, (int, float)):
        sample = [sample]
    sample = [float(value) for value in sample]
    if len(sample) > len(paths):
        raise ValueError("--base_weights_sample_multiplier has more values than --base_weights has files")
    train = [float(value) for value in (getattr(args, "base_weights_multiplier", None) or [])]
    deltas = []
    for index, path in enumerate(paths):
        train_multiplier = train[index] if index < len(train) else 1.0
        sample_multiplier = sample[index] if index < len(sample) else train_multiplier
        if sample_multiplier != train_multiplier:
            deltas.append((path, sample_multiplier - train_multiplier))
    return deltas


class SampleBaseWeights:
    """Frozen networks that turn the merged base weights into their sample strengths while samples are generated."""

    def __init__(self, deltas: list[tuple[str, float]]):
        self.deltas = list(deltas)
        self.networks: list[torch.nn.Module] = []
        self._built = False

    def __bool__(self) -> bool:
        return bool(self.deltas)

    def build(
        self,
        network_module: Any,
        transformer: torch.nn.Module,
        load_weights: Callable[[str], dict[str, torch.Tensor]],
    ) -> None:
        """Attach one disabled inference network per file (after the trained network: it wraps the trained forward)."""
        if self._built:
            return
        self._built = True
        for path, delta in self.deltas:
            weights_sd = load_weights(path)
            network = network_module.create_arch_network_from_weights(delta, weights_sd, unet=transformer, for_inference=True)
            if isinstance(network, tuple):
                network = network[0]
            if not getattr(network, "unet_loras", None):
                raise ValueError(f"base weights {path}: no LoRA module matches the model, cannot leave them out of samples")
            network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=True)
            network.requires_grad_(False)
            network.eval()
            network.set_enabled(False)
            network.to("cpu")
            self.networks.append(network)
            # printed: the trainers' INFO logging is not shown in the GUI
            print(f"training samples take {path} at {delta:+g} relative to training ({len(network.unet_loras)} modules, {info})")

    def before_sampling(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        for network in self.networks:
            network.to(device=device, dtype=dtype) if dtype is not None else network.to(device)
            network.set_enabled(True)

    def after_sampling(self) -> None:
        for network in self.networks:
            network.set_enabled(False)
            network.to("cpu")


def add_sample_multiplier_argument(parser) -> None:
    parser.add_argument(
        "--base_weights_sample_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="multiplier of each --base_weights file in training samples (default: its --base_weights_multiplier). "
        "0 leaves a training adapter (a Turbo de-distillation LoRA) out of the samples, as at inference.",
    )
