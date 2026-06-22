"""Opt-in LTX-2 audio extension conditioning (prefix / suffix).

Off by default; enable with --ltx2_audio_extend_prefix_frames > 0 and/or
--ltx2_audio_extend_suffix_frames > 0. Byte-identical when off.

Audio analog of the video-extension feature (ltx2_extend.py). Marks the first N (prefix) and/or
last N (suffix) AUDIO LATENT timesteps as clean conditioning (timestep 0, excluded from the loss),
auto-derived from the target audio latents (no dataset, no cache). The model learns to continue audio
forward (from a prefix) or backward (from a suffix). Applied per sample with a Bernoulli probability.

Audio has one token per latent timestep (no spatial grid), so the conditioning mask is a 1-D
``[T_audio]`` boolean over timesteps.
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_audio_extend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the audio-extension CLI flags. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_audio_extend_prefix_frames"):
        return parser
    parser.add_argument(
        "--ltx2_audio_extend_prefix_frames",
        type=int,
        default=0,
        help="opt-in: number of leading AUDIO LATENT timesteps kept clean as conditioning (forward audio extension). 0 = off.",
    )
    parser.add_argument(
        "--ltx2_audio_extend_suffix_frames",
        type=int,
        default=0,
        help="opt-in: number of trailing AUDIO LATENT timesteps kept clean as conditioning (backward audio extension). 0 = off.",
    )
    parser.add_argument(
        "--ltx2_audio_extend_p",
        type=float,
        default=1.0,
        help="per-sample Bernoulli probability of applying audio-extension conditioning when enabled. Default 1.0 (always). "
        "Shared by prefix and suffix in a single draw unless --ltx2_audio_extend_prefix_p / --ltx2_audio_extend_suffix_p are set.",
    )
    parser.add_argument(
        "--ltx2_audio_extend_prefix_p",
        type=float,
        default=None,
        help="opt-in: per-sample probability for the audio PREFIX span, drawn independently of the suffix. When either "
        "this or --ltx2_audio_extend_suffix_p is set, prefix and suffix use independent draws; the unset side falls "
        "back to --ltx2_audio_extend_p. Default unset (single shared draw).",
    )
    parser.add_argument(
        "--ltx2_audio_extend_suffix_p",
        type=float,
        default=None,
        help="opt-in: per-sample probability for the audio SUFFIX span, drawn independently of the prefix. See "
        "--ltx2_audio_extend_prefix_p. Default unset (single shared draw).",
    )
    return parser


def is_audio_extend_enabled(args: argparse.Namespace) -> bool:
    """getattr-safe master predicate: enabled when a prefix or suffix span is requested."""
    prefix = int(getattr(args, "ltx2_audio_extend_prefix_frames", 0) or 0)
    suffix = int(getattr(args, "ltx2_audio_extend_suffix_frames", 0) or 0)
    return prefix > 0 or suffix > 0


def validate_audio_extend_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on incompatible audio-extension setup. No-op when disabled.

    MUST run AFTER ltx_mode normalization. Hard errors: video-only mode (no audio latents to
    extend); negative spans; out-of-range probability. Composes with the audio-bearing IC-LoRA
    reference strategies (av_ic / video_ref_only_av / audio_ref_ic): the conditioned audio timesteps
    are clean-pasted into the generated audio target and excluded from the audio loss, with the audio
    reference (if any) concatenated in front.
    """
    if not is_audio_extend_enabled(args):
        return
    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode == "video":
        raise RuntimeError("--ltx2_audio_extend_* requires an audio-bearing mode (got --ltx2_mode video).")
    # v2v is the video-only reference strategy: it has no audio target, so an audio intrinsic would be
    # silently ignored. The audio-bearing reference strategies (av_ic / video_ref_only_av / audio_ref_ic)
    # do compose and are allowed. (Reachable only via raw CLI flags; a recipe never derives v2v with audio.)
    if str(getattr(args, "ic_lora_strategy", "none") or "none").lower() == "v2v":
        raise RuntimeError("--ltx2_audio_extend_* has no effect with --ic_lora_strategy v2v (video-only; no audio target).")
    prefix = int(getattr(args, "ltx2_audio_extend_prefix_frames", 0) or 0)
    suffix = int(getattr(args, "ltx2_audio_extend_suffix_frames", 0) or 0)
    if prefix < 0 or suffix < 0:
        raise RuntimeError(f"--ltx2_audio_extend_prefix_frames / _suffix_frames must be >= 0. Got: prefix={prefix} suffix={suffix}")
    p = float(getattr(args, "ltx2_audio_extend_p", 1.0) or 0.0)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"--ltx2_audio_extend_p must be in [0, 1]. Got: {p}")
    for _name in ("ltx2_audio_extend_prefix_p", "ltx2_audio_extend_suffix_p"):
        _v = getattr(args, _name, None)
        if _v is not None and not (0.0 <= float(_v) <= 1.0):
            raise RuntimeError(f"--{_name} must be in [0, 1]. Got: {_v}")
    # Audio extension composes with the audio-bearing IC-LoRA reference strategies (av_ic /
    # video_ref_only_av / audio_ref_ic): the prefix/suffix audio timesteps are clean-pasted into the
    # generated audio target and OR'd into the audio loss-exclusion, with the audio reference (if any)
    # concatenated in front. v2v is video-only and is already rejected by the audio-bearing-mode check
    # above, so no strategy restriction is needed here.


def build_audio_extend_token_mask(
    *,
    frames: int,
    prefix: int,
    suffix: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a single-sample audio conditioning mask from prefix/suffix latent-timestep spans.

    The first ``prefix`` and last ``suffix`` audio latent timesteps are marked. Spans are clamped to
    ``[0, frames]``; a prefix+suffix that overlap simply union (no double effect). Returns a bool
    ``[frames]`` tensor (one entry per audio latent timestep / token).
    """
    mask = torch.zeros((frames,), device=device, dtype=torch.bool)
    p = max(0, min(int(prefix), frames))
    s = max(0, min(int(suffix), frames))
    if p > 0:
        mask[:p] = True
    if s > 0:
        mask[frames - s :] = True  # negative tail slice (backward extension)
    return mask
