"""Inference-side composable conditioning for LTX-2.

Adds the ``ConditioningItem`` types the inference package lacks but the training loop composes: a
region/inpaint mask (:class:`VideoConditionByMask`) and a reference-video append
(:class:`VideoConditionByReference`). Both follow the ``apply_to(latent_state, latent_tools)``
protocol and operate on a patchified :class:`~musubi_tuner.ltx_2.types.LatentState`.

The inference package ships ``VideoConditionByLatentIndex`` (I2V / replace-at-frame) and
``VideoConditionByKeyframeIndex`` (append). The training loop (``ltx2_train_network.call_dit``)
additionally composes ``inpaint`` / ``spatial_crop`` / ``extend`` intrinsics with a reference
strategy by OR-merging their per-token masks into the target conditioning mask
(``_or_intrinsic_video_token_masks``). To sample a LoRA trained that way we need the same
composition at inference — hence :class:`VideoConditionByMask` here.

OR-merge semantics. The training mask union is boolean ("a token is clean if ANY condition marks
it clean"). In the continuous ``denoise_mask`` (1 = full denoise, 0 = clean) the union is ``min``
(most-clean wins). :class:`VideoConditionByMask` therefore takes ``min(current, 1 - strength)`` on
its tokens so it composes correctly with a first-frame condition applied via
``VideoConditionByLatentIndex`` (whose overwrite is fine on the disjoint frame-0 tokens; on any
frame0 ∩ mask overlap the later ``min`` still resolves to clean).

Mask convention matches training (``ltx2_inpaint_mask.build_inpaint_token_mask``): white / True =
conditioned (kept clean); pass an already-inverted mask to condition the complement.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from musubi_tuner.ltx_2.conditioning.item import ConditioningItem
from musubi_tuner.ltx_2.tools import LatentTools
from musubi_tuner.ltx_2.types import LatentState


class VideoConditionByMask(ConditioningItem):
    """Region/inpaint conditioning over the target tokens.

    Keeps the masked (``token_mask == True``) tokens pinned to a clean source latent and OR-merges
    them into the denoise mask via ``min`` (clean-if-any), mirroring the training-side
    ``_or_intrinsic_video_token_masks`` + clean-paste. No-op on the un-masked tokens, so it composes
    with any other condition already applied to the state.

    Args:
        clean_latent: ``[B, C, F, H, W]`` clean source latents pasted into the conditioned region.
            For video inpaint this is the SOURCE video encoded at the target shape (edits are local,
            so the kept region equals the source). Spatial/temporal shape must match the target.
        token_mask: bool mask, ``[B, seq_len]`` or ``[seq_len]``. ``True`` = conditioned (kept clean).
            Built with ``ltx2_inpaint_mask.build_inpaint_token_mask`` (layout ``(f h w)``, w fastest)
            so it lines up token-for-token with the patchified target (``patch_size == 1``).
        strength: ``1.0`` => fully clean (``denoise_mask`` driven to 0 on the masked tokens). Lower
            values leave partial denoising, matching the ``1 - strength`` denoise-mask convention.
    """

    def __init__(self, clean_latent: torch.Tensor, token_mask: torch.Tensor, strength: float = 1.0):
        if clean_latent.dim() != 5:
            raise ValueError(f"clean_latent must be 5D [B,C,F,H,W], got shape {tuple(clean_latent.shape)}")
        self.clean_latent = clean_latent
        self.token_mask = token_mask
        self.strength = float(strength)

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.clean_latent)  # [B, seq, D]
        b, seq, _d = tokens.shape

        m = self.token_mask.to(device=tokens.device, dtype=torch.bool)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(b, -1)
        if m.shape != (b, seq):
            raise ValueError(
                f"token_mask shape {tuple(self.token_mask.shape)} does not match patchified target "
                f"[B={b}, seq={seq}]. Build it with build_inpaint_token_mask at the target latent geometry."
            )
        m3 = m.unsqueeze(-1)  # [B, seq, 1] for broadcast against latent [B,seq,D] and denoise_mask [B,seq,1]

        tokens = tokens.to(device=latent_state.latent.device, dtype=latent_state.latent.dtype)

        # Pin both the current latent and the stored clean latent at the conditioned tokens, and
        # OR-merge into the denoise mask (clean-if-any == min over conditions). LatentState is a
        # frozen dataclass, so build new tensors and return a replaced copy (no in-place field set).
        new_latent = torch.where(m3, tokens, latent_state.latent)
        new_clean = torch.where(m3, tokens, latent_state.clean_latent)
        target = torch.full_like(latent_state.denoise_mask, 1.0 - self.strength)
        new_denoise = torch.where(m3, torch.minimum(latent_state.denoise_mask, target), latent_state.denoise_mask)

        return replace(latent_state, latent=new_latent, denoise_mask=new_denoise, clean_latent=new_clean)


class VideoConditionByReference(ConditioningItem):
    """Reference-video (V2V) conditioning by appending clean reference tokens.

    Appends pre-patchified reference tokens (with their own positions) to the END of the sequence
    at ``denoise_mask == 0`` (fully clean) — the placement ``LatentTools.clear_conditioning`` relies
    on to strip them afterwards. The model is position-based (RoPE), so an appended reference is
    attention-equivalent to a prepended one.

    The V2V path in ``ltx2_sampling._do_v2v_denoising`` keeps the reference as a ref-FIRST prefix to
    stay byte-identical to ``call_dit``; this item provides the append-convention form for a
    standalone inference V2V path. Positions must be supplied already computed (the caller owns the
    temporal-scale / downscale / fps rewrite) to match training exactly.

    Args:
        ref_tokens: ``[B, ref_seq, D]`` already-patchified clean reference latents.
        ref_positions: pixel coordinates for the reference tokens, with the same layout as
            ``LatentState.positions`` (``[B, 3, ref_seq, ...]`` — 3 coordinate channels on dim 1, the
            token/sequence axis on dim 2) so they concatenate onto the existing positions.
    """

    def __init__(self, ref_tokens: torch.Tensor, ref_positions: torch.Tensor):
        if ref_tokens.dim() != 3:
            raise ValueError(f"ref_tokens must be [B, ref_seq, D], got {tuple(ref_tokens.shape)}")
        if ref_positions.dim() < 3 or ref_positions.shape[1] != 3:
            raise ValueError(
                f"ref_positions must be [B, 3, ref_seq, ...] (3 coord channels on dim 1), got {tuple(ref_positions.shape)}"
            )
        if ref_positions.shape[2] != ref_tokens.shape[1]:
            raise ValueError(f"ref_positions seq (dim 2) {ref_positions.shape[2]} != ref_tokens seq {ref_tokens.shape[1]}")
        self.ref_tokens = ref_tokens
        self.ref_positions = ref_positions

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        device = latent_state.latent.device
        dtype = latent_state.latent.dtype
        ref_tokens = self.ref_tokens.to(device=device, dtype=dtype)
        ref_positions = self.ref_positions.to(device=device, dtype=latent_state.positions.dtype)
        if ref_positions.dim() != latent_state.positions.dim():
            raise ValueError(
                f"ref_positions ndim {ref_positions.dim()} != LatentState.positions ndim "
                f"{latent_state.positions.dim()}; build them with the same patchifier/get_pixel_coords layout."
            )

        ref_denoise = torch.zeros(
            (ref_tokens.shape[0], ref_tokens.shape[1], 1), device=device, dtype=latent_state.denoise_mask.dtype
        )
        return LatentState(
            latent=torch.cat([latent_state.latent, ref_tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, ref_denoise], dim=1),
            positions=torch.cat([latent_state.positions, ref_positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, ref_tokens], dim=1),
        )
