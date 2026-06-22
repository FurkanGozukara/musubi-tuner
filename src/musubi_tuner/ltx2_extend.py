"""Opt-in LTX-2 video extension conditioning (prefix / suffix).

Off by default; enable with --ltx2_extend_prefix_frames > 0 and/or --ltx2_extend_suffix_frames > 0.
Byte-identical when off.

Marks the first N (prefix) and/or last N (suffix) LATENT frames of the video as clean
conditioning (timestep 0, excluded from the loss), auto-derived from the target latents (no
dataset, no cache). The model learns to extend a clip forward (from a prefix) or backward (from
a suffix). Applied per sample with a Bernoulli probability.

Coverage note: a temporal boundary of N latent frames corresponds to 1 + 8*(N-1) pixel frames
because the causal VAE encodes the first frame on its own and then groups of 8.
"""

from __future__ import annotations

import argparse
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_extend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the video-extension CLI flags. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_extend_prefix_frames"):
        return parser
    parser.add_argument(
        "--ltx2_extend_prefix_frames",
        type=int,
        default=0,
        help="opt-in: number of leading LATENT frames kept clean as conditioning (forward video extension). 0 = off.",
    )
    parser.add_argument(
        "--ltx2_extend_suffix_frames",
        type=int,
        default=0,
        help="opt-in: number of trailing LATENT frames kept clean as conditioning (backward video extension). 0 = off.",
    )
    parser.add_argument(
        "--ltx2_extend_p",
        type=float,
        default=1.0,
        help="per-sample Bernoulli probability of applying extension conditioning when enabled. Default 1.0 (always). "
        "Shared by prefix and suffix in a single draw unless --ltx2_extend_prefix_p / --ltx2_extend_suffix_p are set.",
    )
    parser.add_argument(
        "--ltx2_extend_prefix_p",
        type=float,
        default=None,
        help="opt-in: per-sample probability for the PREFIX span, drawn independently of the suffix. When either "
        "this or --ltx2_extend_suffix_p is set, prefix and suffix use independent draws (a sample can get prefix-only, "
        "suffix-only, both, or neither); the unset side falls back to --ltx2_extend_p. Default unset (single shared draw).",
    )
    parser.add_argument(
        "--ltx2_extend_suffix_p",
        type=float,
        default=None,
        help="opt-in: per-sample probability for the SUFFIX span, drawn independently of the prefix. See "
        "--ltx2_extend_prefix_p. Default unset (single shared draw).",
    )
    return parser


def is_extend_enabled(args: argparse.Namespace) -> bool:
    """getattr-safe master predicate: enabled when a prefix or suffix span is requested."""
    prefix = int(getattr(args, "ltx2_extend_prefix_frames", 0) or 0)
    suffix = int(getattr(args, "ltx2_extend_suffix_frames", 0) or 0)
    return prefix > 0 or suffix > 0


def validate_extend_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on incompatible video-extension setup. No-op when disabled.

    MUST run AFTER ltx_mode normalization. Hard errors: audio-only mode (no video latents to
    extend); negative spans; out-of-range probability. Composes with the IC-LoRA reference
    strategies (the extension mask is OR-merged into the reference target conditioning mask).
    """
    if not is_extend_enabled(args):
        return
    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode == "audio":
        raise RuntimeError("--ltx2_extend_* requires a video-bearing mode (got --ltx2_mode audio).")
    prefix = int(getattr(args, "ltx2_extend_prefix_frames", 0) or 0)
    suffix = int(getattr(args, "ltx2_extend_suffix_frames", 0) or 0)
    if prefix < 0 or suffix < 0:
        raise RuntimeError(f"--ltx2_extend_prefix_frames / _suffix_frames must be >= 0. Got: prefix={prefix} suffix={suffix}")
    p = float(getattr(args, "ltx2_extend_p", 1.0) or 0.0)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"--ltx2_extend_p must be in [0, 1]. Got: {p}")
    for _name in ("ltx2_extend_prefix_p", "ltx2_extend_suffix_p"):
        _v = getattr(args, _name, None)
        if _v is not None and not (0.0 <= float(_v) <= 1.0):
            raise RuntimeError(f"--{_name} must be in [0, 1]. Got: {_v}")
    # Video extension composes with the IC-LoRA reference strategies: the clean prefix/suffix frames
    # ride the target video tokens and the per-token mask is OR-merged into each reference branch's
    # target conditioning mask (timestep 0 + loss exclusion). No strategy restriction.


def build_extend_token_mask(
    *,
    frames: int,
    height: int,  # LATENT-grid height (latents.shape[3])
    width: int,  # LATENT-grid width  (latents.shape[4])
    patch_size: int,  # wrapper.patch_size; train loop uses 1
    prefix: int,
    suffix: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a single-sample temporal conditioning mask from prefix/suffix latent-frame spans.

    The first ``prefix`` and last ``suffix`` latent frames are marked (whole-frame), laid out in
    patchify ``(f h w)`` order. Spans are clamped to ``[0, frames]``; a prefix+suffix that overlap
    simply union (no double effect).

    Returns ``(token_mask, region_5d)``:
      * token_mask: bool ``[seq_len]``, ``seq_len == frames * H_tok * W_tok``.
      * region_5d: bool ``[1, frames, H_tok, W_tok]`` for clean-paste + loss.
    """
    H_tok = height // patch_size
    W_tok = width // patch_size

    grid = torch.zeros((frames, H_tok, W_tok), device=device, dtype=torch.bool)
    p = max(0, min(int(prefix), frames))
    s = max(0, min(int(suffix), frames))
    if p > 0:
        grid[:p] = True
    if s > 0:
        grid[frames - s :] = True  # negative tail slice (backward extension)

    token_mask = grid.reshape(-1).contiguous()  # (f h w), w fastest
    seq_len = frames * H_tok * W_tok
    assert token_mask.shape == (seq_len,), f"build_extend_token_mask token shape {tuple(token_mask.shape)} != {(seq_len,)}"

    region_5d = grid.unsqueeze(0).contiguous()  # [1, F, H_tok, W_tok]
    return token_mask, region_5d
