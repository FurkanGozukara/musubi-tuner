"""Multi-reference (R2AV) conditioning for Kandinsky 6.

Reference images are VAE-encoded on the target canvas (letterboxed, never cropped) and
appended after the generated frames as clean, masked frames, the same mechanism the
released TI2AV mode uses for its first-frame reference. Each reference carries:

* token type ``REFERENCE_TOKEN_TYPE`` (2) instead of the first-frame type (1);
* a learned slot embedding (``visual_ref_slot_embeddings[k]`` for reference ``k``), which
  the DiT binds to the prompt mention ``@Image {k+1}``;
* its own RoPE time index after the video, ``T + REF_ROPE_GAP + k``, so it is never read
  as a frame of the clip.

Everything here is shared by latent caching, training and sampling so the three cannot
drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .model import REFERENCE_TOKEN_TYPE

# Gap (in latent frames) between the last generated frame and the first reference on the
# RoPE time axis. Fixed so training and inference always agree.
REF_ROPE_GAP = 4
DEFAULT_REFERENCE_SLOTS = 8
# Letterbox fill for references (white suits the segmented subject cut-outs of R2V data).
DEFAULT_REF_PAD_VALUE = 255
IMAGE_TAG_FORMAT = "@Image {n}"


# ----------------------------------------------------------------------------------------
# Pixels -> latents
# ----------------------------------------------------------------------------------------


def letterbox_reference(
    image: Image.Image | np.ndarray, width: int, height: int, pad_value: int = DEFAULT_REF_PAD_VALUE
) -> np.ndarray:
    """Fit an image inside ``width x height`` without cropping; returns uint8 [H,W,3]."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode == "RGBA":
        # composite transparent cut-outs onto the pad colour instead of black
        background = Image.new("RGBA", image.size, (pad_value, pad_value, pad_value, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    src_w, src_h = image.size
    if src_w < 1 or src_h < 1:
        raise ValueError("reference image is empty")
    scale = min(width / src_w, height / src_h)
    new_w = max(1, min(width, round(src_w * scale)))
    new_h = max(1, min(height, round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (pad_value, pad_value, pad_value))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def load_reference_image(path: str) -> Image.Image:
    with Image.open(path) as handle:
        handle.load()
        return handle.copy()


@torch.no_grad()
def encode_reference_images(video_vae, images: Sequence[np.ndarray]) -> Tensor:
    """VAE-encode letterboxed uint8 [H,W,3] images to ``[N, C, h, w]`` scaled latents."""
    if not images:
        raise ValueError("encode_reference_images needs at least one image")
    device = next(video_vae.parameters()).device
    dtype = next(video_vae.parameters()).dtype
    latents = []
    for image in images:
        pixels = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        pixels = pixels[:, None].unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,1,H,W]
        latent = video_vae.encode(pixels).latent_dist.sample()[0] * video_vae.config.scaling_factor  # [C,1,h,w]
        if latent.shape[1] != 1:
            raise ValueError(f"a reference image must encode to one latent frame, got {latent.shape[1]}")
        latents.append(latent[:, 0].float().cpu())
    return torch.stack(latents)


def encode_reference_paths(
    video_vae, paths: Sequence[str], width: int, height: int, pad_value: int = DEFAULT_REF_PAD_VALUE
) -> Tensor:
    """Load, letterbox to ``width x height`` and encode reference images -> ``[N,C,h,w]``."""
    images = [letterbox_reference(load_reference_image(path), width, height, pad_value) for path in paths]
    return encode_reference_images(video_vae, images)


# ----------------------------------------------------------------------------------------
# Model inputs
# ----------------------------------------------------------------------------------------


def reference_temporal_positions(frames: int, num_references: int, device: torch.device | None = None) -> Tensor:
    """RoPE time indices: ``0..T-1`` for the clip, then ``T+gap+k`` for reference ``k``."""
    positions = torch.arange(frames, device=device)
    if num_references:
        references = frames + REF_ROPE_GAP + torch.arange(num_references, device=device)
        positions = torch.cat([positions, references])
    return positions


def reference_visual_rope(transformer, frames: int, height: int, width: int, num_references: int, scale_factor, device=None):
    """3-D RoPE for ``frames`` generated latent frames plus ``num_references`` references.

    ``height``/``width`` are latent sizes; the transformer's patch size is applied here.
    """
    patch_t, patch_h, patch_w = transformer.patch_size
    if patch_t != 1:
        raise ValueError("Kandinsky 6 references assume a temporal patch size of 1")
    grid_h, grid_w = height // patch_h, width // patch_w
    if device is None:
        device = next(transformer.visual_rope.buffers()).device
    time_positions = reference_temporal_positions(frames, num_references, device)
    max_time = int(transformer.visual_rope.max_pos[0])
    if int(time_positions[-1]) >= max_time:
        raise ValueError(
            f"{frames} latent frames plus {num_references} references need RoPE time index {int(time_positions[-1])}, "
            f"but the table holds {max_time}"
        )
    shape = (int(time_positions.numel()), grid_h, grid_w)
    positions = [time_positions, torch.arange(grid_h, device=device), torch.arange(grid_w, device=device)]
    return transformer.visual_rope(shape, positions, tuple(float(value) for value in scale_factor))


def build_reference_video_input(
    noisy_video: Tensor, references: Tensor | None, visual_cond: bool
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Append clean references to the generated frames.

    ``noisy_video`` is ``[B,T,H,W,C]`` and ``references`` ``[B,N,H,W,C]`` (or None for the
    reference-free branch). Returns the model input (with the [cond, mask] channels when
    ``visual_cond``), token-type ids ``[B,T+N]`` and slot ids ``[B,T+N]`` (-1 on generated
    frames; None without references).
    """
    batch, frames, height, width, channels = noisy_video.shape
    num_references = 0
    if references is not None:
        if references.ndim != 5 or references.shape[0] != batch or references.shape[2:] != (height, width, channels):
            raise ValueError(f"references must be [{batch}, N, {height}, {width}, {channels}], got {tuple(references.shape)}")
        num_references = references.shape[1]
        noisy_video = torch.cat([noisy_video, references.to(device=noisy_video.device, dtype=noisy_video.dtype)], dim=1)
    device = noisy_video.device
    token_types = torch.zeros((batch, frames + num_references), device=device, dtype=torch.long)
    slot_ids = None
    if num_references:
        token_types[:, frames:] = REFERENCE_TOKEN_TYPE
        slot_ids = torch.full((batch, frames + num_references), -1, device=device, dtype=torch.long)
        slot_ids[:, frames:] = torch.arange(num_references, device=device)
    if not visual_cond:
        return noisy_video, token_types, slot_ids
    condition = torch.zeros_like(noisy_video)
    mask = torch.zeros((*noisy_video.shape[:-1], 1), device=device, dtype=noisy_video.dtype)
    if num_references:
        mask[:, frames:] = 1
    return torch.cat([noisy_video, condition, mask], dim=-1), token_types, slot_ids


def cached_references_to_batch(references: Sequence[Tensor], device, dtype) -> Tensor:
    """Collated cache rows (list of ``[N,C,h,w]``) -> ``[B,N,h,w,C]``; all items need the same N."""
    counts = {int(item.shape[0]) for item in references}
    if len(counts) != 1:
        raise ValueError(
            f"a Kandinsky 6 R2AV batch mixes reference counts {sorted(counts)}; use batch_size = 1 "
            "or a dataset whose items all have the same number of references"
        )
    return torch.stack([item.to(device=device, dtype=dtype).permute(0, 2, 3, 1) for item in references])


# ----------------------------------------------------------------------------------------
# Prompt tags
# ----------------------------------------------------------------------------------------

_FIGURE_TAG = re.compile(r"<\s*Figure\s+(\d+)\s*>", re.IGNORECASE)
_VIDEO_TAG = re.compile(r"<\s*Video\s+(\d+)\s*>", re.IGNORECASE)


def mentioned_figures(prompt: str) -> set[int]:
    return {int(match) for match in _FIGURE_TAG.findall(prompt)}


def mentions_video(prompt: str) -> bool:
    return _VIDEO_TAG.search(prompt) is not None


def rewrite_figure_tags(prompt: str, mapping: Mapping[int, int], tag_format: str = IMAGE_TAG_FORMAT) -> str:
    """Replace ``<Figure N>`` by ``tag_format`` with the remapped (1-based) index."""

    def replace(match: re.Match) -> str:
        old = int(match.group(1))
        if old not in mapping:
            raise KeyError(f"prompt mentions <Figure {old}>, which has no reference")
        return tag_format.format(n=mapping[old])

    return _FIGURE_TAG.sub(replace, prompt)


# ----------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------


@torch.no_grad()
def reference_denoise_loop(
    dit,
    video: Tensor,
    audio: Tensor,
    references: Tensor,
    text: Mapping[str, Tensor],
    null_text: Mapping[str, Tensor],
    *,
    text_rope,
    null_text_rope,
    audio_rope: Tensor,
    visual_rope_with_refs: Tensor,
    visual_rope_without_refs: Tensor,
    attention_mask: Tensor | None,
    null_attention_mask: Tensor | None,
    num_steps: int,
    guidance_scale: float,
    reference_guidance_scale: float,
    scheduler_scale: float,
    progress_callback=None,
) -> tuple[Tensor, Tensor]:
    """Euler sampling of ``[1,T,H,W,C]`` video and ``[1,A,C]`` audio with references.

    ``v = v(neg, refs) + g * (v(pos, refs) - v(neg, refs)) + r * (v(pos, refs) - v(pos, no refs))``.
    The text term guides with the references present; the optional ``r`` term strengthens
    reference adherence (it costs one extra forward per step).
    """
    from .runtime.core.algo.flow_matching import flow_match_timesteps

    frames = video.shape[1]
    batch = video.shape[0]
    timesteps = flow_match_timesteps(num_steps, scheduler_scale, video.device)

    def forward(t_step, text_embeds, rope, mask, with_refs: bool):
        model_video, token_types, slot_ids = build_reference_video_input(video, references if with_refs else None, dit.visual_cond)
        if not hasattr(dit, "visual_token_type_embeddings"):
            token_types = None
        video_vel, audio_vel = dit(
            x_video=model_video,
            x_audio=audio,
            text_embed=text_embeds["text_embeds"],
            pooled_text_embed=text_embeds["pooled_embed"],
            time=t_step,
            visual_rope=visual_rope_with_refs if with_refs else visual_rope_without_refs,
            audio_rope=audio_rope,
            text_rope=rope,
            attention_mask=mask,
            visual_token_type_ids=token_types,
            visual_ref_slot_ids=slot_ids,
        )
        return video_vel[:, :frames], audio_vel

    for t, dt in zip(timesteps[:-1], torch.diff(timesteps)):
        t_step = t.unsqueeze(0).expand(batch) * 1000
        cond_v, cond_a = forward(t_step, text, text_rope, attention_mask, True)
        if abs(guidance_scale - 1.0) > 1e-6:
            neg_v, neg_a = forward(t_step, null_text, null_text_rope, null_attention_mask, True)
            vel_v = neg_v + guidance_scale * (cond_v - neg_v)
            vel_a = neg_a + guidance_scale * (cond_a - neg_a)
        else:
            vel_v, vel_a = cond_v, cond_a
        if reference_guidance_scale:
            free_v, free_a = forward(t_step, text, text_rope, attention_mask, False)
            vel_v = vel_v + reference_guidance_scale * (cond_v - free_v)
            vel_a = vel_a + reference_guidance_scale * (cond_a - free_a)
        video = video + dt * vel_v.to(video.dtype)
        audio = audio + dt * vel_a.to(audio.dtype)
        if progress_callback is not None:
            progress_callback()
    return video, audio


@torch.no_grad()
def sample_reference_latents(
    dit,
    references: Tensor | None,
    text: Mapping[str, Tensor],
    null_text: Mapping[str, Tensor],
    *,
    attention_mask: Tensor | None,
    null_attention_mask: Tensor | None,
    video_frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    device: torch.device,
    num_steps: int,
    guidance_scale: float,
    reference_guidance_scale: float,
    scheduler_scale: float,
    visual_rope_scale,
    progress_callback=None,
):
    """Draw noise, build RoPE and run :func:`reference_denoise_loop` for one sample.

    ``references`` is ``[1,N,h,w,C]`` (or None for a reference-free R2AV sample). Returns a
    runtime ``LatentBundle`` with video ``[1,T,h,w,C]`` and audio ``[1,A,C]`` ready for
    ``postprocess_video`` / ``postprocess_audio``.
    """
    from .runtime.core.algo.prepare_latents import audio_latent_duration, prepare_audio_latents, prepare_video_latents
    from .runtime.core.algo.prepare_ropes import compute_rope1d
    from .runtime.core.types import LatentBundle

    if references is not None and references.shape[1] == 0:
        references = None
    if references is None and reference_guidance_scale:
        raise ValueError("reference guidance needs at least one reference image")
    num_references = 0 if references is None else int(references.shape[1])
    bundle = prepare_video_latents(1, video_frames, latent_height, latent_width, dit.in_visual_dim, seed, device)
    audio_frames = audio_latent_duration(video_frames, fps=24, audio_fps=44100, downsample_factor=1024)
    bundle = prepare_audio_latents(bundle, audio_frames, dit.in_audio_dim, seed, device)
    video = bundle.video.reshape(1, video_frames, latent_height, latent_width, dit.in_visual_dim)
    audio = bundle.audio.reshape(1, audio_frames, dit.in_audio_dim)
    if references is not None:
        references = references.to(device=device, dtype=video.dtype)
    text_length = text["text_embeds"].shape[-2]
    null_length = null_text["text_embeds"].shape[-2]
    rope_kwargs = dict(scale_factor=visual_rope_scale, device=device)
    with_refs = reference_visual_rope(dit, video_frames, latent_height, latent_width, num_references, **rope_kwargs)
    without_refs = (
        reference_visual_rope(dit, video_frames, latent_height, latent_width, 0, **rope_kwargs) if num_references else with_refs
    )
    video, audio = reference_denoise_loop(
        dit,
        video,
        audio,
        references,
        text,
        null_text,
        text_rope=[compute_rope1d(dit.video_text_rope, text_length), compute_rope1d(dit.audio_text_rope, text_length)],
        null_text_rope=[compute_rope1d(dit.video_text_rope, null_length), compute_rope1d(dit.audio_text_rope, null_length)],
        audio_rope=compute_rope1d(dit.audio_rope, audio_frames),
        visual_rope_with_refs=with_refs,
        visual_rope_without_refs=without_refs,
        attention_mask=attention_mask,
        null_attention_mask=null_attention_mask,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        reference_guidance_scale=reference_guidance_scale,
        scheduler_scale=scheduler_scale,
        progress_callback=progress_callback,
    )
    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=torch.tensor([0, video_frames], device=device, dtype=torch.int32),
        audio_cu_seqlens=bundle.audio_cu_seqlens,
    )
