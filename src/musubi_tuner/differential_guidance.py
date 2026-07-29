"""Adaptive prediction-relative target scaling for training."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DifferentialGuidanceConfig:
    scale: float = 3.0
    schedule: str = "constant"
    start_scale: float = 1.0
    end_scale: float = 1.0
    warmup_steps: int = 0
    hold_steps: int = 0
    decay_steps: int = 0
    timestep_mode: str = "none"
    timestep_floor: float = 1.0
    normalize_residual: bool = False
    residual_clip: float = 0.0
    adaptive_target_norm: float = 0.0
    adaptive_target_ratio: float = 0.0
    adaptive_ema: float = 0.95
    adaptive_rate: float = 0.1
    adaptive_min: float = 0.25
    adaptive_max: float = 4.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DifferentialGuidanceConfig":
        config = cls(
            scale=float(getattr(args, "differential_guidance_scale", 3.0)),
            schedule=str(getattr(args, "differential_guidance_schedule", "constant") or "constant").lower(),
            start_scale=float(getattr(args, "differential_guidance_start_scale", 1.0)),
            end_scale=float(getattr(args, "differential_guidance_end_scale", 1.0)),
            warmup_steps=int(getattr(args, "differential_guidance_warmup_steps", 0)),
            hold_steps=int(getattr(args, "differential_guidance_hold_steps", 0)),
            decay_steps=int(getattr(args, "differential_guidance_decay_steps", 0)),
            timestep_mode=str(getattr(args, "differential_guidance_timestep_mode", "none") or "none").lower(),
            timestep_floor=float(getattr(args, "differential_guidance_timestep_floor", 1.0)),
            normalize_residual=bool(getattr(args, "differential_guidance_normalize_residual", False)),
            residual_clip=float(getattr(args, "differential_guidance_residual_clip", 0.0)),
            adaptive_target_norm=float(getattr(args, "differential_guidance_adaptive_target_norm", 0.0)),
            adaptive_target_ratio=float(getattr(args, "differential_guidance_adaptive_target_ratio", 0.0)),
            adaptive_ema=float(getattr(args, "differential_guidance_adaptive_ema", 0.95)),
            adaptive_rate=float(getattr(args, "differential_guidance_adaptive_rate", 0.1)),
            adaptive_min=float(getattr(args, "differential_guidance_adaptive_min", 0.25)),
            adaptive_max=float(getattr(args, "differential_guidance_adaptive_max", 4.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        finite_values = {
            "scale": self.scale,
            "start_scale": self.start_scale,
            "end_scale": self.end_scale,
            "timestep_floor": self.timestep_floor,
            "residual_clip": self.residual_clip,
            "adaptive_target_norm": self.adaptive_target_norm,
            "adaptive_target_ratio": self.adaptive_target_ratio,
            "adaptive_ema": self.adaptive_ema,
            "adaptive_rate": self.adaptive_rate,
            "adaptive_min": self.adaptive_min,
            "adaptive_max": self.adaptive_max,
        }
        invalid = [name for name, value in finite_values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Differential Guidance values must be finite: {', '.join(invalid)}")
        nonnegative = {
            "scale": self.scale,
            "start_scale": self.start_scale,
            "end_scale": self.end_scale,
            "timestep_floor": self.timestep_floor,
            "residual_clip": self.residual_clip,
            "adaptive_target_norm": self.adaptive_target_norm,
            "adaptive_target_ratio": self.adaptive_target_ratio,
            "adaptive_rate": self.adaptive_rate,
            "adaptive_min": self.adaptive_min,
            "adaptive_max": self.adaptive_max,
        }
        invalid = [name for name, value in nonnegative.items() if value < 0.0]
        if invalid:
            raise ValueError(f"Differential Guidance values must be non-negative: {', '.join(invalid)}")
        if self.schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("Differential Guidance schedule must be constant, linear, or cosine")
        if self.timestep_mode not in {"none", "mid", "snr", "inverse_snr"}:
            raise ValueError("Differential Guidance timestep mode must be none, mid, snr, or inverse_snr")
        if min(self.warmup_steps, self.hold_steps, self.decay_steps) < 0:
            raise ValueError("Differential Guidance schedule step counts must be non-negative")
        if not 0.0 <= self.adaptive_ema < 1.0:
            raise ValueError("Differential Guidance adaptive EMA must be in [0, 1)")
        if self.adaptive_min > self.adaptive_max:
            raise ValueError("Differential Guidance adaptive_min must be <= adaptive_max")
        if self.adaptive_target_norm > 0.0 and self.adaptive_target_ratio > 0.0:
            raise ValueError("Set only one Differential Guidance adaptive target: norm or ratio")

    @property
    def adaptive_enabled(self) -> bool:
        return self.adaptive_target_norm > 0.0 or self.adaptive_target_ratio > 0.0


def _interpolation_fraction(progress: float, schedule: str) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    return progress


def scheduled_differential_guidance_scale(config: DifferentialGuidanceConfig, global_step: int) -> float:
    """Return the scalar schedule value before timestep/adaptive modulation."""
    if config.schedule == "constant":
        return config.scale
    step = max(int(global_step), 0)
    if config.warmup_steps > 0 and step < config.warmup_steps:
        fraction = _interpolation_fraction(step / config.warmup_steps, config.schedule)
        return config.start_scale + fraction * (config.scale - config.start_scale)
    decay_start = config.warmup_steps + config.hold_steps
    if config.decay_steps > 0 and step >= decay_start:
        fraction = _interpolation_fraction((step - decay_start) / config.decay_steps, config.schedule)
        return config.scale + fraction * (config.end_scale - config.scale)
    return config.scale


def differential_guidance_timestep_weight(sigmas: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a bounded [0, 1] modulation weight for each sample."""
    sigma = sigmas.detach().float()
    if sigma.dim() == 0:
        sigma = sigma.reshape(1)
    elif sigma.dim() > 1:
        sigma = sigma.reshape(sigma.shape[0], -1).mean(dim=1)
    if sigma.numel() and float(sigma.detach().max().item()) > 1.5:
        sigma = sigma / 1000.0
    sigma = sigma.clamp(0.0, 1.0)
    if mode == "mid":
        return 4.0 * sigma * (1.0 - sigma)
    clean = (1.0 - sigma).square()
    noise = sigma.square()
    if mode == "snr":
        return clean / (clean + noise).clamp_min(1e-8)
    if mode == "inverse_snr":
        return noise / (clean + noise).clamp_min(1e-8)
    return torch.ones_like(sigma)


class DifferentialGuidanceController:
    """Stateful schedule, timestep shaping, and adaptive gradient feedback."""

    def __init__(self, config: DifferentialGuidanceConfig) -> None:
        self.config = config
        self.adaptive_multiplier = 1.0
        self.observed_ema: float | None = None

    def transform_target(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        sigmas: torch.Tensor | None,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        detached_pred = pred.detach().to(device=target.device, dtype=target.dtype)
        residual = target - detached_pred
        residual_f = residual.detach().float()
        reduce_dims = tuple(range(1, residual.dim()))
        if reduce_dims:
            rms = residual_f.square().mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(1e-8)
        else:
            rms = residual_f.abs().clamp_min(1e-8)
        clipped_fraction = 0.0
        if self.config.residual_clip > 0.0:
            limit = rms.to(device=residual.device, dtype=residual.dtype) * self.config.residual_clip
            clipped = residual.abs() > limit
            clipped_fraction = float(clipped.detach().float().mean().item())
            residual = torch.clamp(residual, min=-limit, max=limit)
        if self.config.normalize_residual:
            residual = residual / rms.to(device=residual.device, dtype=residual.dtype)

        base_scale = scheduled_differential_guidance_scale(self.config, global_step)
        guided_scale = base_scale * self.adaptive_multiplier
        if sigmas is None or self.config.timestep_mode == "none":
            sample_scale = target.new_full((target.shape[0],), guided_scale)
        else:
            weight = differential_guidance_timestep_weight(sigmas, self.config.timestep_mode).to(
                device=target.device,
                dtype=target.dtype,
            )
            if weight.numel() == 1 and target.shape[0] != 1:
                weight = weight.expand(target.shape[0])
            if weight.shape[0] != target.shape[0]:
                raise ValueError(
                    f"Differential Guidance timestep batch mismatch: sigmas={weight.shape[0]} predictions={target.shape[0]}"
                )
            sample_scale = self.config.timestep_floor + weight * (guided_scale - self.config.timestep_floor)
        broadcast_scale = sample_scale.reshape(sample_scale.shape[0], *([1] * (target.dim() - 1)))
        if not self.config.normalize_residual and self.config.residual_clip == 0.0 and bool(torch.all(sample_scale == 1.0).item()):
            adjusted = target
        else:
            adjusted = detached_pred + broadcast_scale * residual
        metrics = {
            "dg/base_scale": float(base_scale),
            "dg/adaptive_multiplier": float(self.adaptive_multiplier),
            "dg/effective_scale_mean": float(sample_scale.detach().float().mean().item()),
            "dg/effective_scale_min": float(sample_scale.detach().float().min().item()),
            "dg/effective_scale_max": float(sample_scale.detach().float().max().item()),
            "dg/residual_rms": float(residual_f.square().mean().sqrt().item()),
            "dg/clipped_fraction": clipped_fraction,
        }
        return adjusted, metrics

    def update_gradient_feedback(self, video_grad_norm: float, audio_grad_norm: float | None) -> dict[str, float]:
        metrics = {"dg/gradient_video": float(video_grad_norm)}
        if audio_grad_norm is not None:
            metrics["dg/gradient_audio"] = float(audio_grad_norm)
            if audio_grad_norm > 0.0:
                metrics["dg/gradient_video_audio_ratio"] = float(video_grad_norm / audio_grad_norm)
        if not self.config.adaptive_enabled:
            return metrics

        target = self.config.adaptive_target_norm
        observed = float(video_grad_norm)
        if self.config.adaptive_target_ratio > 0.0:
            if audio_grad_norm is None or audio_grad_norm <= 0.0:
                metrics["dg/adaptive_feedback_available"] = 0.0
                return metrics
            target = self.config.adaptive_target_ratio
            observed = float(video_grad_norm / audio_grad_norm)
        metrics["dg/adaptive_feedback_available"] = 1.0
        if self.observed_ema is None:
            self.observed_ema = observed
        else:
            decay = self.config.adaptive_ema
            self.observed_ema = decay * self.observed_ema + (1.0 - decay) * observed
        error = math.log(max(target, 1e-12) / max(self.observed_ema, 1e-12))
        response = math.exp(self.config.adaptive_rate * max(min(error, 4.0), -4.0))
        self.adaptive_multiplier = min(
            max(self.adaptive_multiplier * response, self.config.adaptive_min),
            self.config.adaptive_max,
        )
        metrics["dg/adaptive_observed_ema"] = float(self.observed_ema)
        metrics["dg/adaptive_multiplier_next"] = float(self.adaptive_multiplier)
        return metrics


def collect_differential_guidance_grad_norms(
    network: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[float, float | None]:
    """Collect video and audio LoRA gradient norms before auxiliary backwards."""
    lora_modules = getattr(network, "unet_loras", None)
    video_sq = torch.zeros((), device=device, dtype=torch.float32)
    audio_sq = torch.zeros((), device=device, dtype=torch.float32)
    found_audio = False
    if lora_modules:
        for lora in lora_modules:
            is_audio = "audio_" in str(getattr(lora, "lora_name", ""))
            found_audio = found_audio or is_audio
            destination = audio_sq if is_audio else video_sq
            for parameter in lora.parameters():
                if parameter.grad is not None:
                    destination.add_(parameter.grad.detach().float().square().sum().to(device=device))
    else:
        for parameter in network.parameters():
            if parameter.grad is not None:
                video_sq.add_(parameter.grad.detach().float().square().sum().to(device=device))
    video_norm = float(video_sq.sqrt().item())
    audio_norm = float(audio_sq.sqrt().item()) if found_audio else None
    return video_norm, audio_norm


def get_differential_guidance_controller(
    trainer: Any,
    args: argparse.Namespace,
) -> DifferentialGuidanceController:
    config = DifferentialGuidanceConfig.from_args(args)
    controller = getattr(trainer, "_differential_guidance_controller", None)
    if controller is None or controller.config != config:
        controller = DifferentialGuidanceController(config)
        trainer._differential_guidance_controller = controller
    return controller
