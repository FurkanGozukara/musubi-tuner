"""Default NetworkTrainer extension hooks."""

from __future__ import annotations

import argparse

import torch

from musubi_tuner.differential_guidance import (
    collect_differential_guidance_grad_norms,
    get_differential_guidance_controller,
)


def is_model_parallel_enabled(self, args) -> bool:
    return False


def validate_model_parallel_setup(self, args, accelerator) -> None:
    pass


def enable_model_parallel_transformer(self, args, accelerator, transformer) -> None:
    pass


def place_network_for_model_parallel(self, args, accelerator, transformer, network) -> None:
    pass


def clip_grad_norm_for_model_parallel(self, args, accelerator, params, optimizer):
    return accelerator.clip_grad_norm_(params, args.max_grad_norm)


def pre_train_hook(self, args, accelerator, transformer=None, network=None):
    pass


def compute_prior_divergence_addition(self, args, accelerator, transformer, network, video_pred, network_dtype):
    return None


def preservation_backward(self, args, accelerator, transformer, network, network_dtype):
    return {}


def compute_validation_extra_loss(
    self,
    args,
    accelerator,
    transformer,
    network,
    batch,
    global_step: int,
    network_dtype,
):
    return None, {}


def modify_video_loss_per_element(self, args, per_elem, out, network_dtype):
    return per_elem, {}


def modify_audio_loss_per_element(self, args, per_elem, out, network_dtype):
    return per_elem, {}


def compute_video_extra_loss(self, args, out, network_dtype):
    return None, {}


def apply_differential_guidance_target(
    self,
    args: argparse.Namespace,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sigmas: torch.Tensor | None = None,
    global_step: int | None = None,
) -> torch.Tensor:
    if not bool(getattr(args, "differential_guidance", False)):
        self._differential_guidance_step_metrics = {}
        return target
    controller = get_differential_guidance_controller(self, args)
    adjusted, metrics = controller.transform_target(
        pred,
        target,
        sigmas=sigmas,
        global_step=int(getattr(self, "_current_train_global_step", 0) if global_step is None else global_step),
    )
    self._differential_guidance_step_metrics = metrics
    return adjusted


def update_differential_guidance_gradient_feedback(
    self,
    args: argparse.Namespace,
    network: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if not bool(getattr(args, "differential_guidance", False)):
        return {}
    controller = get_differential_guidance_controller(self, args)
    if not controller.config.adaptive_enabled:
        return {}
    video_norm, audio_norm = collect_differential_guidance_grad_norms(network, device=device)
    return controller.update_gradient_feedback(video_norm, audio_norm)
