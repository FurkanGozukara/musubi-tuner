"""Opt-in LTX-2 on-disk mask conditioning (video inpaint / outpaint).

Off by default; enable with --ltx2_inpaint_mask. Byte-identical when off.

A per-item mask supplied through the dataset's ``loss_mask_directory`` is already
bucket-cropped in lockstep with the video content, resized to latent geometry, and present
in every batch as ``video_loss_mask``. When this feature is on, that mask is binarized at a
threshold and used as a CONDITIONING mask: masked-in tokens receive the clean ground-truth
latents (timestep 0) and are excluded from loss, so the model learns to generate the
complement (inpaint) or the surround (outpaint, via 'invert'). Applied per sample with a
Bernoulli probability.

Mask convention: white / 1 = conditioned (kept clean). Use 'invert' if the mask marks the
region to GENERATE instead.

Note: when this feature is on the mask is consumed as conditioning, NOT as a loss weight
(the two conventions are opposite); the train loop suppresses the loss-weight consumption of
the same mask while this is enabled.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# No new dataset columns: the mask is supplied via the existing loss_mask_directory channel.
INPAINT_MASK_DATASET_COLUMNS: Sequence[str] = ()


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    # idiom: ltx2_fsdp.py (private copy keeps this module self-contained / detect-and-degrade).
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_inpaint_mask_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the inpaint-mask CLI flags. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_inpaint_mask"):
        return parser
    parser.add_argument(
        "--ltx2_inpaint_mask",
        action="store_true",
        help="opt-in: on-disk mask conditioning (video inpaint/outpaint). Binarizes the mask "
        "supplied via the dataset's loss_mask_directory and uses it as a conditioning mask "
        "(masked tokens kept clean at timestep 0, excluded from loss). White/1 = conditioned. "
        "Off by default.",
    )
    parser.add_argument(
        "--ltx2_inpaint_mask_p",
        type=float,
        default=0.0,
        help="per-sample Bernoulli probability of applying inpaint-mask conditioning when a mask is present. Default 0.0 (never).",
    )
    parser.add_argument(
        "--ltx2_inpaint_mask_invert",
        action="store_true",
        help="with --ltx2_inpaint_mask: condition the COMPLEMENT of the mask (generate the masked "
        "region instead of keeping it clean).",
    )
    parser.add_argument(
        "--ltx2_inpaint_mask_threshold",
        type=float,
        default=0.5,
        help="binarization threshold for the inpaint mask (strict '>'). Default 0.5.",
    )
    return parser


def is_inpaint_mask_enabled(args: argparse.Namespace) -> bool:
    """getattr-safe master predicate for the inpaint-mask feature."""
    return bool(getattr(args, "ltx2_inpaint_mask", False))


def validate_inpaint_mask_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on incompatible inpaint-mask setup. No-op when disabled.

    MUST run AFTER ltx_mode normalization and after ``args.ic_lora_strategy`` is resolved.
    Hard errors: audio-only mode (no video latents to condition); out-of-range probability or
    threshold; IC-LoRA reference strategies (they build their own conditioning/loss masks and
    return before the inpaint mask is applied).
    """
    if not is_inpaint_mask_enabled(args):
        return
    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode == "audio":
        raise RuntimeError("--ltx2_inpaint_mask requires a video-bearing mode (got --ltx2_mode audio).")
    p = float(getattr(args, "ltx2_inpaint_mask_p", 0.0) or 0.0)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"--ltx2_inpaint_mask_p must be in [0, 1]. Got: {p}")
    if p == 0.0:
        logger.warning(
            "--ltx2_inpaint_mask is enabled but --ltx2_inpaint_mask_p is 0.0; the feature will never "
            "apply (and the loss_mask_directory mask, if any, is treated as a standard loss weight). "
            "Set --ltx2_inpaint_mask_p > 0 to use inpaint conditioning."
        )
    threshold = float(getattr(args, "ltx2_inpaint_mask_threshold", 0.5))
    if not (0.0 <= threshold <= 1.0):
        raise RuntimeError(f"--ltx2_inpaint_mask_threshold must be in [0, 1]. Got: {threshold}")
    ic_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_strategy in ("v2v", "av_ic", "video_ref_only_av", "audio_ref_ic"):
        raise RuntimeError(
            f"--ltx2_inpaint_mask is not supported together with --ic_lora_strategy {ic_strategy} "
            "(the IC-LoRA reference path builds its own conditioning/loss masks). Disable one of them."
        )


def build_inpaint_token_mask(
    mask_fhw: torch.Tensor,
    *,
    frames: int,
    height: int,  # LATENT-grid height (latents.shape[3])
    width: int,  # LATENT-grid width  (latents.shape[4])
    patch_size: int,  # wrapper.patch_size; train loop uses 1
    invert: bool,
    threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a single-sample conditioning mask from a latent-geometry mask.

    ``mask_fhw`` is the per-item mask already at (or broadcastable to) latent geometry —
    accepts ``[F, H, W]``, ``[1, F, H, W]`` (channel), or ``[F, 1, 1]`` (frame-only). It is
    binarized at ``threshold`` (strict ``>``), optionally inverted, and laid out in patchify
    ``(f h w)`` order (w fastest), matching ``build_spatial_token_mask``.

    Returns ``(token_mask, region_5d)``:
      * token_mask: bool ``[seq_len]``, ``seq_len == frames * H_tok * W_tok`` where
        ``H_tok = height // patch_size``, ``W_tok = width // patch_size``;
        ``t = ((f * H_tok) + h) * W_tok + w``.
      * region_5d: bool ``[1, frames, H_tok, W_tok]`` for clean-paste + loss. With
        patch_size==1 (the train-loop default) the token grid equals the latent grid.
    """
    H_tok = height // patch_size
    W_tok = width // patch_size

    m = mask_fhw.to(device=device)
    if m.dim() == 4:  # [1, F, H, W] -> [F, H, W]
        m = m[0]
    if m.dim() != 3:
        raise ValueError(f"inpaint mask must be [F,H,W], [1,F,H,W] or [F,1,1]; got shape {tuple(mask_fhw.shape)}")
    # The cached loss-mask is resized to the latent grid, so this normally matches exactly.
    # Defensive resize (e.g. a frame-only [F,1,1] broadcast) onto the token grid.
    if tuple(m.shape) != (frames, H_tok, W_tok):
        m = torch.nn.functional.interpolate(
            m.reshape(1, 1, *m.shape).to(dtype=torch.float32),
            size=(frames, H_tok, W_tok),
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    binary = m.to(dtype=torch.float32) > float(threshold)  # strict '>' binarization
    if invert:
        binary = ~binary

    token_mask = binary.reshape(-1).contiguous()  # (f h w), w fastest
    seq_len = frames * H_tok * W_tok
    assert token_mask.shape == (seq_len,), f"build_inpaint_token_mask token shape {tuple(token_mask.shape)} != {(seq_len,)}"

    # binary is already [F, H_tok, W_tok] (per-frame); one leading dim -> [1, F, H_tok, W_tok].
    region_5d = binary.unsqueeze(0).contiguous()  # [1, F, H_tok, W_tok]
    return token_mask, region_5d
