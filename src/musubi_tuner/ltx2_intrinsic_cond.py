"""Opt-in LTX-2 intrinsic (region) conditioning sources.

Off by default; enable with --ltx2_spatial_crop. Byte-identical when off.

Spatial-crop: marks a rectangular spatial region of the video latents as clean
conditioning (timestep=0, excluded from loss), so the model learns to generate the
surrounding content (outpaint). Region is given in PIXEL coords [y1, x1, y2, x2] (set
per dataset via the ``spatial_crop_region`` dataset column); applied per sample with a
Bernoulli probability; 'invert' conditions OUTSIDE the rect.

A single per-sample selection drives the clean-latent paste, the per-token conditioning
mask (timestep 0) and the loss-mask exclusion together.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence, Tuple

import torch

from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors

logger = logging.getLogger(__name__)

# Dataset columns this feature reads (gated; legal-but-inert for non-LTX-2).
SPATIAL_CROP_DATASET_COLUMNS = ("spatial_crop_region",)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    # idiom: ltx2_fsdp.py
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_intrinsic_cond_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the spatial-crop CLI flags. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_spatial_crop"):
        return parser
    parser.add_argument(
        "--ltx2_spatial_crop",
        action="store_true",
        help="opt-in: spatial-crop region conditioning (video outpaint). Marks a rectangular "
        "spatial region of the video latents as clean conditioning (timestep 0, excluded from "
        "loss) so the model learns to generate the surrounding content. Off by default.",
    )
    parser.add_argument(
        "--ltx2_spatial_crop_p",
        type=float,
        default=0.0,
        help="per-sample Bernoulli probability of applying spatial-crop conditioning when a "
        "region is present. Default 0.0 (never).",
    )
    parser.add_argument(
        "--ltx2_spatial_crop_invert",
        action="store_true",
        help="with --ltx2_spatial_crop: condition the region OUTSIDE the rect instead of inside.",
    )
    return parser


def is_spatial_crop_enabled(args: argparse.Namespace) -> bool:
    """getattr-safe master predicate for the spatial-crop feature."""
    return bool(getattr(args, "ltx2_spatial_crop", False))


def validate_intrinsic_cond_setup(args: argparse.Namespace, accelerator=None) -> None:
    """Raise on incompatible spatial-crop setup. No-op when disabled.

    MUST run AFTER ltx_mode normalization (reads the normalized ``args.ltx_mode``).
    Hard errors: (a) column-present while flag off (friendly; the dataset construction
    flag is the load-bearing gate); (b) audio-only mode (no video latents to crop).
    """
    enabled = is_spatial_crop_enabled(args)
    if not enabled:
        present = [col for col in SPATIAL_CROP_DATASET_COLUMNS if bool(getattr(args, f"_dataset_has_{col}", False))]
        if present:
            raise RuntimeError(
                f"dataset declares spatial-crop column(s) {present} but --ltx2_spatial_crop is "
                f"off. Enable --ltx2_spatial_crop or remove the column(s) from the dataset config."
            )
        return
    mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    if mode == "audio":
        raise RuntimeError("--ltx2_spatial_crop requires a video-bearing mode (got --ltx2_mode audio).")
    p = float(getattr(args, "ltx2_spatial_crop_p", 0.0) or 0.0)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"--ltx2_spatial_crop_p must be in [0, 1]. Got: {p}")
    # Spatial-crop targets the standard (non-IC-LoRA) training path. The IC-LoRA
    # reference branches (v2v / av_ic / video_ref_only_av / audio_ref_ic) build their own
    # conditioning + loss masks and return before the spatial-crop mask is applied, so the
    # region would be fed as clean latents without being excluded from loss. Block the
    # combination. (args.ic_lora_strategy is already resolved by handle_model_specific_args.)
    ic_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_strategy in ("v2v", "av_ic", "video_ref_only_av", "audio_ref_ic"):
        raise RuntimeError(
            f"--ltx2_spatial_crop is not supported together with --ic_lora_strategy {ic_strategy} "
            "(the IC-LoRA reference path builds its own conditioning/loss masks). Disable one of them."
        )


def build_spatial_token_mask(
    region_px: Sequence[int],
    *,
    frames: int,
    height: int,  # LATENT-grid height (latents.shape[3])
    width: int,  # LATENT-grid width  (latents.shape[4])
    patch_size: int,  # wrapper.patch_size; train loop uses 1
    invert: bool,
    device: torch.device,
    scale_factors: Optional[SpatioTemporalScaleFactors] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a single-sample spatial conditioning mask from a PIXEL rect.

    Returns ``(token_mask, region_5d)``:
      * token_mask: bool ``[seq_len]``, ``seq_len == frames * H_tok * W_tok`` where
        ``H_tok = height // patch_size``, ``W_tok = width // patch_size`` (PATCH grid).
        Patchify order is ``(f h w)`` with w fastest, then h, then f:
        ``t = ((f * H_tok) + h) * W_tok + w``.
      * region_5d: bool ``[1, frames, H_tok, W_tok]`` for clean-paste + loss. With
        patch_size==1 (the train-loop default) the token grid equals the latent grid,
        so it is directly usable as a ``[1, F, H, W]`` latent-grid mask. If patch_size
        != 1 ever becomes live, the clean-paste must upsample region_5d to the latent
        grid (repeat_interleave by patch_size on H/W).

    Pixel->latent: floor-divide pixel coords by the REAL per-axis VAE factors
    (``scale_factors.height`` for y, ``.width`` for x — read by NAME; the NamedTuple
    field order is (time, width, height)), NOT reference_downscale. Latent->token:
    floor-divide by patch_size. 'invert' conditions OUTSIDE the rect. An empty rect
    (sub-cell) logs a WARNING and returns all-False (or all-True when inverted).
    """
    region_px = tuple(region_px)
    if len(region_px) != 4:
        raise ValueError(f"spatial_crop_region must have exactly 4 ints [y1, x1, y2, x2]; got {list(region_px)}")

    sf = scale_factors or SpatioTemporalScaleFactors.default()
    vae_h = int(sf.height)  # 32
    vae_w = int(sf.width)  # 32

    H_tok = height // patch_size
    W_tok = width // patch_size

    y1, x1, y2, x2 = (int(v) for v in region_px)
    # PIXEL -> LATENT (real per-axis VAE factors) -> TOKEN (patch_size).
    ly1 = (y1 // vae_h) // patch_size
    lx1 = (x1 // vae_w) // patch_size
    ly2 = (y2 // vae_h) // patch_size
    lx2 = (x2 // vae_w) // patch_size
    # exact clamp (kept for weight interoperability); normalize ordering.
    ly1, ly2 = max(0, min(ly1, H_tok)), max(0, min(ly2, H_tok))
    lx1, lx2 = max(0, min(lx1, W_tok)), max(0, min(lx2, W_tok))
    if ly1 > ly2:
        ly1, ly2 = ly2, ly1
    if lx1 > lx2:
        lx1, lx2 = lx2, lx1

    if ly1 == ly2 or lx1 == lx2:
        # sub-cell rect floor-divides to an empty region: WARN, do not raise.
        logger.warning(
            "spatial_crop region %s floor-divides to an empty token region "
            "(ly=[%d,%d) lx=[%d,%d) on %dx%d token grid); mask is all-%s.",
            list(region_px),
            ly1,
            ly2,
            lx1,
            lx2,
            H_tok,
            W_tok,
            "True" if invert else "False",
        )

    grid = torch.zeros((H_tok, W_tok), device=device, dtype=torch.bool)  # True = inside rect
    grid[ly1:ly2, lx1:lx2] = True
    if invert:
        grid = ~grid

    # token mask: replicate spatial grid across frames in patchify order (f h w).
    token_mask = grid.reshape(-1).unsqueeze(0).expand(frames, -1).reshape(-1).contiguous()
    seq_len = frames * H_tok * W_tok
    assert token_mask.shape == (seq_len,), f"build_spatial_token_mask token shape {tuple(token_mask.shape)} != {(seq_len,)}"

    # 5D region mask [1, F, H_tok, W_tok] for clean-paste + loss.
    region_5d = grid.unsqueeze(0).unsqueeze(0).expand(1, frames, H_tok, W_tok).contiguous()
    return token_mask, region_5d
