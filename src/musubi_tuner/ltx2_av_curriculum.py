"""Optimizer-step-based audio/video loss curricula for LTX-2 training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


AV_CURRICULUM_MODES = ("none", "alternating", "two_stage")
AV_CURRICULUM_POLICIES = ("video", "audio", "joint")
AV_CURRICULUM_START_MODALITIES = ("video", "audio")

_POLICY_MULTIPLIERS = {
    "video": (1.0, 0.0),
    "audio": (0.0, 1.0),
    "joint": (1.0, 1.0),
}
_POLICY_IDS = {"joint": 0.0, "video": 1.0, "audio": 2.0}


@dataclass(frozen=True)
class AVCurriculumState:
    policy: str
    phase_index: int
    phase_step: int
    video_multiplier: float
    audio_multiplier: float

    def metrics(self, *, base_video_weight: float, base_audio_weight: float) -> dict[str, float]:
        return {
            "av_curriculum/policy_id": _POLICY_IDS[self.policy],
            "av_curriculum/phase": float(self.phase_index),
            "av_curriculum/phase_step": float(self.phase_step),
            "av_curriculum/video_active": self.video_multiplier,
            "av_curriculum/audio_active": self.audio_multiplier,
            "av_curriculum/video_weight": base_video_weight * self.video_multiplier,
            "av_curriculum/audio_weight": base_audio_weight * self.audio_multiplier,
        }


@dataclass(frozen=True)
class AVCurriculumConfig:
    mode: str = "none"
    interval_steps: int = 1
    start_modality: str = "video"
    stage1_steps: int = 0
    stage1_policy: str = "video"
    stage2_policy: str = "joint"

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def state_at_step(self, global_step: int) -> AVCurriculumState:
        step = max(0, int(global_step))
        if self.mode == "alternating":
            phase_index = step // self.interval_steps
            phase_step = step % self.interval_steps
            first = self.start_modality
            second = "audio" if first == "video" else "video"
            policy = first if phase_index % 2 == 0 else second
        elif self.mode == "two_stage":
            in_stage1 = step < self.stage1_steps
            phase_index = 0 if in_stage1 else 1
            phase_step = step if in_stage1 else step - self.stage1_steps
            policy = self.stage1_policy if in_stage1 else self.stage2_policy
        else:
            phase_index = 0
            phase_step = step
            policy = "joint"
        video_multiplier, audio_multiplier = _POLICY_MULTIPLIERS[policy]
        return AVCurriculumState(
            policy=policy,
            phase_index=phase_index,
            phase_step=phase_step,
            video_multiplier=video_multiplier,
            audio_multiplier=audio_multiplier,
        )


def config_from_args(args: argparse.Namespace) -> AVCurriculumConfig:
    return AVCurriculumConfig(
        mode=str(getattr(args, "av_curriculum_mode", "none") or "none").strip().lower(),
        interval_steps=int(getattr(args, "av_curriculum_interval_steps", 1)),
        start_modality=str(getattr(args, "av_curriculum_start_modality", "video") or "video").strip().lower(),
        stage1_steps=int(getattr(args, "av_curriculum_stage1_steps", 0) or 0),
        stage1_policy=str(getattr(args, "av_curriculum_stage1_policy", "video") or "video").strip().lower(),
        stage2_policy=str(getattr(args, "av_curriculum_stage2_policy", "joint") or "joint").strip().lower(),
    )


def validate_av_curriculum_setup(args: argparse.Namespace) -> AVCurriculumConfig:
    config = config_from_args(args)
    if config.mode not in AV_CURRICULUM_MODES:
        raise ValueError(f"--av_curriculum_mode must be one of {AV_CURRICULUM_MODES}. Got: {config.mode!r}")
    if config.start_modality not in AV_CURRICULUM_START_MODALITIES:
        raise ValueError(
            f"--av_curriculum_start_modality must be one of {AV_CURRICULUM_START_MODALITIES}. Got: {config.start_modality!r}"
        )
    for flag, policy in (
        ("--av_curriculum_stage1_policy", config.stage1_policy),
        ("--av_curriculum_stage2_policy", config.stage2_policy),
    ):
        if policy not in AV_CURRICULUM_POLICIES:
            raise ValueError(f"{flag} must be one of {AV_CURRICULUM_POLICIES}. Got: {policy!r}")
    if config.interval_steps <= 0:
        raise ValueError("--av_curriculum_interval_steps must be greater than zero")
    if config.stage1_steps < 0:
        raise ValueError("--av_curriculum_stage1_steps must be non-negative")
    if not config.enabled:
        return config

    if str(getattr(args, "ltx_mode", "video") or "video").lower() != "av":
        raise ValueError("--av_curriculum_mode requires --ltx2_mode av")
    if bool(getattr(args, "ltx2_audio_only_model", False)):
        raise ValueError("--av_curriculum_mode requires a joint video+audio transformer")
    if str(getattr(args, "ic_lora_strategy", "none") or "none").lower() != "none":
        raise ValueError("--av_curriculum_mode is not supported with an IC-LoRA strategy")
    if str(getattr(args, "ltx2_train_direction", "joint") or "joint").lower() != "joint":
        raise ValueError("--av_curriculum_mode is not supported with --ltx2_train_direction")
    if str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower() != "none":
        raise ValueError("--av_curriculum_mode requires --audio_loss_balance_mode none")
    if int(getattr(args, "modality_freeze_check_interval", 0) or 0) > 0:
        raise ValueError("--av_curriculum_mode is not supported with the modality freezer")
    if bool(getattr(args, "audio_silence_regularizer", False)):
        raise ValueError("--av_curriculum_mode requires paired AV batches; disable --audio_silence_regularizer")
    if bool(getattr(args, "self_flow", False)):
        raise ValueError("--av_curriculum_mode is not supported with --self_flow")
    if bool(getattr(args, "crepa", False)):
        raise ValueError("--av_curriculum_mode is not supported with --crepa")
    if (
        float(getattr(args, "cts_lambda_video_driven", 0.0) or 0.0) > 0.0
        or float(getattr(args, "cts_lambda_audio_driven", 0.0) or 0.0) > 0.0
    ):
        raise ValueError("--av_curriculum_mode is not supported with Cross-Task Synergy losses")
    if float(getattr(args, "video_loss_weight", 1.0) or 0.0) <= 0.0:
        raise ValueError("--av_curriculum_mode requires --video_loss_weight greater than zero")
    if float(getattr(args, "audio_loss_weight", 1.0) or 0.0) <= 0.0:
        raise ValueError("--av_curriculum_mode requires --audio_loss_weight greater than zero")

    if config.mode == "two_stage":
        if config.stage1_steps <= 0:
            raise ValueError("--av_curriculum_stage1_steps must be greater than zero for two_stage")
        if config.stage1_policy == config.stage2_policy:
            raise ValueError("two_stage requires different stage-1 and stage-2 policies")
        covered = {config.stage1_policy, config.stage2_policy}
        has_video = "video" in covered or "joint" in covered
        has_audio = "audio" in covered or "joint" in covered
        if not (has_video and has_audio):
            raise ValueError("two_stage policies must collectively train both video and audio")

    args.av_curriculum_mode = config.mode
    args.av_curriculum_interval_steps = config.interval_steps
    args.av_curriculum_start_modality = config.start_modality
    args.av_curriculum_stage1_steps = config.stage1_steps
    args.av_curriculum_stage1_policy = config.stage1_policy
    args.av_curriculum_stage2_policy = config.stage2_policy
    return config


def apply_av_curriculum_weights(
    config: AVCurriculumConfig,
    *,
    global_step: int,
    video_weight: float,
    audio_weight: float,
) -> tuple[float, float, AVCurriculumState]:
    state = config.state_at_step(global_step)
    if config.enabled:
        if video_weight <= 0.0 or audio_weight <= 0.0:
            raise ValueError(
                "AV curriculum requires positive video and audio batch loss weights before curriculum routing; "
                f"got video={video_weight}, audio={audio_weight}"
            )
    return (
        video_weight * state.video_multiplier,
        audio_weight * state.audio_multiplier,
        state,
    )
