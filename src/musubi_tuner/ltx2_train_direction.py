"""LTX-2 directional training mode (A2V / V2A).

Off by default (``joint``); enable with --ltx2_train_direction {a2v, v2a}. Byte-identical when off.

Makes the MAIN (and only) training loss directional by freezing one whole modality as clean,
given conditioning while the other is generated:

  * ``a2v`` -> freeze audio (clean), generate video.
  * ``v2a`` -> freeze video (clean), generate audio (foley).
  * ``joint`` -> normal joint audio-video denoising (default; both modalities noised).

A freeze = clean latents (no noise), timesteps 0, sigma 0, and loss-weight 0.0 for the frozen
modality. The forward still runs BOTH streams, so the frozen-clean one is what the generated one
cross-attends to. Deterministic (whole batch, every step) -- this trains a DEDICATED directional
model, so there is no per-sample probability.

Relationship to existing AV features (these are distinct, hence the separate flag):
  * Cross-Task Synergy (--cts_lambda_audio_driven / --cts_lambda_video_driven) runs the same
    clean-one/noise-other forward, but as an AUXILIARY loss ADDED to the main joint loss (extra
    forwards/step) to regularize a model that stays joint-capable. This mode REPLACES the main
    loss with the directional one (one forward/step), for a dedicated A2V or V2A model.
  * --modality_freeze_* ("G2D") adaptively freezes a modality's LoRA PARAMETERS by loss balance;
    it does not change the inputs. Orthogonal.

This module owns only the flag + validation; the freeze operations (sigma/timestep zeroing, the
V2A whole-video clean-paste, the loss-weight override) live in the shared ``call_dit``, because
they touch per-modality tensors that exist only there.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TRAIN_DIRECTION_CHOICES = ("joint", "a2v", "v2a")

# direction -> the modality that is frozen (clean conditioning); the other is generated.
_FROZEN_FOR_DIRECTION = {"a2v": "audio", "v2a": "video"}

# IC-LoRA reference strategies own the modality streams (clean-ref tokens + their own loss masks)
# and return early before the freeze hooks, so directional training cannot compose with them.
_IC_LORA_STRATEGIES = ("v2v", "av_ic", "video_ref_only_av", "audio_ref_ic")


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_train_direction_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the directional-training CLI flag. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_train_direction"):
        return parser
    parser.add_argument(
        "--ltx2_train_direction",
        type=str,
        default="joint",
        choices=list(TRAIN_DIRECTION_CHOICES),
        help=(
            "Directional training mode (requires --ltx2_mode av). 'a2v' freezes audio as clean "
            "conditioning and generates video; 'v2a' freezes video and generates audio (foley); "
            "'joint' = normal joint AV denoising (default). Unlike --cts_lambda_* (auxiliary loss "
            "added to joint training), this REPLACES the main loss to train a dedicated directional model."
        ),
    )
    return parser


def resolve_train_direction(args: argparse.Namespace) -> str:
    """getattr-safe, normalized accessor. Returns one of TRAIN_DIRECTION_CHOICES.

    Anything unset/empty/unknown resolves to ``"joint"`` (validation reports the real error path;
    this accessor never raises so the runtime gate stays cheap and total)."""
    value = getattr(args, "ltx2_train_direction", "joint")
    if value is None:
        return "joint"
    value = str(value).strip().lower()
    if value in _FROZEN_FOR_DIRECTION:
        return value
    return "joint"


def frozen_modality_for_direction(direction: str) -> Optional[str]:
    """Map a direction to the modality that is frozen, or None for ``joint``."""
    return _FROZEN_FOR_DIRECTION.get(direction)


def is_directional_training_enabled(args: argparse.Namespace) -> bool:
    """Master predicate: True when training is directional (a2v or v2a)."""
    return resolve_train_direction(args) != "joint"


def validate_train_direction_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on an incompatible directional-training setup. No-op when disabled (``joint``).

    MUST run AFTER ltx_mode normalization and after ``args.ic_lora_strategy`` is resolved.

    v1 is a strict standalone mode: directional training must run on a plain joint-AV configuration.
    Hard errors on single-modality mode, IC-LoRA reference strategies, Self-Flow, TREAD, Cross-Task
    Synergy, the G2D modality freezer, and -- for v2a -- any intrinsic video conditioner that would
    contend over the video latents / loss. Composition with these is deferred to the capstone.
    """
    direction = resolve_train_direction(args)
    if direction == "joint":
        return

    raw = getattr(args, "ltx2_train_direction", "joint")
    if str(raw).strip().lower() not in TRAIN_DIRECTION_CHOICES:
        raise RuntimeError(f"--ltx2_train_direction must be one of {TRAIN_DIRECTION_CHOICES}. Got: {raw!r}")

    frozen = frozen_modality_for_direction(direction)

    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode != "av":
        raise RuntimeError(
            f"--ltx2_train_direction {direction} requires --ltx2_mode av (both audio and video streams "
            f"must be present). Got --ltx2_mode {mode}."
        )

    ic_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_strategy in _IC_LORA_STRATEGIES:
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with --ic_lora_strategy "
            f"{ic_strategy} (the IC-LoRA reference path owns the modality streams). Disable one."
        )

    if bool(getattr(args, "self_flow", False)):
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with --self_flow "
            "(Self-Flow distills the video stream). Disable one."
        )

    if bool(getattr(args, "tread", False)):
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with --tread "
            "(token routing would drop frozen-conditioning tokens). Disable one."
        )

    cts_v = float(getattr(args, "cts_lambda_video_driven", 0.0) or 0.0)
    cts_a = float(getattr(args, "cts_lambda_audio_driven", 0.0) or 0.0)
    if cts_v > 0.0 or cts_a > 0.0:
        raise RuntimeError(
            f"--ltx2_train_direction {direction} already trains directionally; it is redundant and "
            "unsupported together with Cross-Task Synergy (--cts_lambda_video_driven / "
            "--cts_lambda_audio_driven), which adds directional losses on top of joint training. Disable one."
        )

    if int(getattr(args, "modality_freeze_check_interval", 0) or 0) > 0:
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with the G2D modality "
            "freezer (--modality_freeze_check_interval > 0), which balances two active modality losses; "
            "in directional mode one modality loss is zero. Disable one."
        )

    balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
    if balance_mode != "none":
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with "
            f"--audio_loss_balance_mode {balance_mode} (dynamic audio/video loss balancing divides "
            "against / learns from a modality loss that is held at zero in directional training, which "
            "diverges). Use --audio_loss_balance_mode none."
        )

    if bool(getattr(args, "dcr", False)):
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with --dcr "
            "(detach-clean-reference keys on sigma==0, which is exactly the frozen conditioning stream, "
            "and would cut its cross-attention backward). Disable one."
        )

    if bool(getattr(args, "tarp", False)):
        raise RuntimeError(
            f"--ltx2_train_direction {direction} is not supported together with --tarp "
            "(windowed AV attention would restrict the cross-attention that carries the sole conditioning "
            "signal). Disable one."
        )

    if frozen == "audio":
        if bool(getattr(args, "independent_audio_timestep", False)):
            raise RuntimeError(
                "--ltx2_train_direction a2v freezes audio at sigma 0; --independent_audio_timestep samples "
                "an audio noise level that is immediately discarded. Disable --independent_audio_timestep."
            )

    if frozen == "video":
        # Any video-side conditioner / latent modifier contends with a frozen (whole-clean) video:
        # they write model_noisy_video, build conditioning/loss masks, or degrade the latents that v2a
        # then uses as the clean conditioning signal. Lazy import avoids import-order coupling.
        from musubi_tuner.ltx2_extend import is_extend_enabled
        from musubi_tuner.ltx2_inpaint_mask import is_inpaint_mask_enabled
        from musubi_tuner.ltx2_intrinsic_cond import is_spatial_crop_enabled

        # first_frame conditioning is ON BY DEFAULT (p=0.1), so a plain v2a invocation would otherwise
        # hard-error below. It is redundant under v2a (the whole video is already frozen-clean), so
        # auto-disable it with a warning instead of erroring -- `--ltx2_mode av --ltx2_train_direction
        # v2a` then runs out of the box. The explicitly opt-in conditioners below stay hard errors.
        first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0) or 0.0)
        if first_frame_p > 0.0:
            logger.warning(
                "--ltx2_train_direction v2a freezes the whole video; disabling first-frame conditioning "
                "(--ltx2_first_frame_conditioning_p %.3f -> 0.0), redundant under v2a.",
                first_frame_p,
            )
            args.ltx2_first_frame_conditioning_p = 0.0

        conflicting = []
        if is_spatial_crop_enabled(args):
            conflicting.append("--ltx2_spatial_crop")
        if is_inpaint_mask_enabled(args):
            conflicting.append("--ltx2_inpaint_mask")
        if is_extend_enabled(args):
            conflicting.append("--ltx2_extend_prefix_frames/_suffix_frames")
        if bool(getattr(args, "video_anchor_training", False)):
            conflicting.append("--video_anchor_training")
        if bool(getattr(args, "hfato", False)):
            # HFATO degrades the video latents before call_dit runs; v2a would then clean-paste the
            # degraded latents as the conditioning signal.
            conflicting.append("--hfato")
        if conflicting:
            raise RuntimeError(
                "--ltx2_train_direction v2a freezes the whole video and cannot combine with video-side "
                f"conditioning / latent modification ({', '.join(conflicting)}). Disable one."
            )
