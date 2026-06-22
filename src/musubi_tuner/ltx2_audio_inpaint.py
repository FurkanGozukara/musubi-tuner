"""Opt-in LTX-2 audio inpaint / mask conditioning.

Off by default; enable with --ltx2_audio_inpaint_mask. Byte-identical when off.

Audio analog of the video inpaint feature (ltx2_inpaint_mask.py). A per-item audio mask supplied through
the dataset's ``audio_cond_mask_directory`` (time intervals, cached to a per-timestep ``[T]`` tensor) is
binarized at a threshold and used as a CONDITIONING mask: masked-in audio latent timesteps receive the
clean ground-truth latents (timestep 0) and are excluded from the loss, so the model learns to generate
the complement (audio inpaint). Applied per sample with a Bernoulli probability.

Mask convention: 1 = conditioned (kept clean). Use 'invert' if the mask marks the region to GENERATE.
The audio conditioning mask is a SEPARATE channel from the audio loss mask (the loss mask is a weight,
1 = include in loss — the opposite convention — so it cannot be reused as a conditioning mask).
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_audio_inpaint_mask_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the audio inpaint-mask CLI flags. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_audio_inpaint_mask"):
        return parser
    parser.add_argument(
        "--ltx2_audio_inpaint_mask",
        action="store_true",
        help="opt-in: audio inpaint conditioning. Binarizes the per-item audio mask supplied via the "
        "dataset's audio_cond_mask_directory and uses it as a conditioning mask (masked audio latent timesteps "
        "kept clean at timestep 0, excluded from loss). 1 = conditioned. Off by default.",
    )
    parser.add_argument(
        "--ltx2_audio_inpaint_mask_p",
        type=float,
        default=0.0,
        help="per-sample Bernoulli probability of applying audio inpaint conditioning when a mask is present. Default 0.0 (never).",
    )
    parser.add_argument(
        "--ltx2_audio_inpaint_mask_invert",
        action="store_true",
        help="with --ltx2_audio_inpaint_mask: condition the COMPLEMENT of the mask (generate the masked "
        "region instead of keeping it clean).",
    )
    parser.add_argument(
        "--ltx2_audio_inpaint_mask_threshold",
        type=float,
        default=0.5,
        help="binarization threshold for the audio inpaint mask (strict '>'). Default 0.5.",
    )
    return parser


def is_audio_inpaint_mask_enabled(args: argparse.Namespace) -> bool:
    """getattr-safe master predicate for the audio inpaint-mask feature."""
    return bool(getattr(args, "ltx2_audio_inpaint_mask", False))


def validate_audio_inpaint_mask_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on incompatible audio inpaint-mask setup. No-op when disabled.

    MUST run AFTER ltx_mode normalization and after ``args.ic_lora_strategy`` is resolved.
    Hard errors: video-only mode (no audio latents to condition); out-of-range probability or threshold;
    IC-LoRA reference strategies (they build their own conditioning/loss masks).
    """
    if not is_audio_inpaint_mask_enabled(args):
        return
    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode == "video":
        raise RuntimeError("--ltx2_audio_inpaint_mask requires an audio-bearing mode (got --ltx2_mode video).")
    p = float(getattr(args, "ltx2_audio_inpaint_mask_p", 0.0) or 0.0)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"--ltx2_audio_inpaint_mask_p must be in [0, 1]. Got: {p}")
    if p == 0.0:
        logger.warning(
            "--ltx2_audio_inpaint_mask is enabled but --ltx2_audio_inpaint_mask_p is 0.0; the feature will "
            "never apply. Set --ltx2_audio_inpaint_mask_p > 0 to use audio inpaint conditioning."
        )
    threshold = float(getattr(args, "ltx2_audio_inpaint_mask_threshold", 0.5))
    if not (0.0 <= threshold <= 1.0):
        raise RuntimeError(f"--ltx2_audio_inpaint_mask_threshold must be in [0, 1]. Got: {threshold}")
    ic_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_strategy in ("v2v", "av_ic", "video_ref_only_av", "audio_ref_ic"):
        raise RuntimeError(
            f"--ltx2_audio_inpaint_mask is not supported together with --ic_lora_strategy {ic_strategy} "
            "(the IC-LoRA reference path builds its own conditioning/loss masks). Disable one of them."
        )


def build_audio_inpaint_token_mask(
    mask_t: torch.Tensor,
    *,
    frames: int,
    threshold: float,
    invert: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build a single-sample audio conditioning mask from a per-timestep audio mask.

    ``mask_t`` is the cached per-item audio mask at (or broadcastable to) ``[frames]`` audio latent
    timesteps — accepts ``[T]``, ``[1, T]``, or ``[T, 1]``. It is binarized at ``threshold`` (strict
    ``>``) and optionally inverted. Returns a bool ``[frames]`` tensor (one entry per audio latent
    timestep / token), matching ``build_audio_extend_token_mask``.
    """
    m = mask_t.to(device=device)
    m = m.reshape(-1)
    if m.shape[0] != frames:
        # Defensive resize onto the audio timestep grid (normally already exact).
        m = torch.nn.functional.interpolate(
            m.reshape(1, 1, -1).to(dtype=torch.float32),
            size=(frames,),
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    binary = m.to(dtype=torch.float32) > float(threshold)  # strict '>' binarization
    if invert:
        binary = ~binary
    return binary.contiguous()
