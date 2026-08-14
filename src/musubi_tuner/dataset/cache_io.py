from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_HIDREAM_O1_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_IDEOGRAM4_FULL,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_KREA2_FULL,
    ARCHITECTURE_MINIMAX_H3_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_WAN_FULL,
    ARCHITECTURE_Z_IMAGE_FULL,
)

# LTX-2 fork addition — separate import so upstream's inserts into the sorted block above don't conflict
from musubi_tuner.dataset.architectures import ARCHITECTURE_LTX2_FULL
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)

SOURCE_PATH_METADATA_KEY = "source_path"
SOURCE_PATH_FORMAT_METADATA_KEY = "source_path_format"
SOURCE_PATH_FORMAT_RELATIVE_TO_CACHE = "relative_to_cache"
SOURCE_PATH_FORMAT_FILENAME_ONLY = "filename_only"
SOURCE_SIZE_METADATA_KEY = "source_size"
SOURCE_MTIME_NS_METADATA_KEY = "source_mtime_ns"


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def build_source_freshness_metadata(item_info: ItemInfo, source_path: Optional[str] = None) -> dict[str, str]:
    """Return source file identity for cache freshness checks, when available."""
    source_path = source_path or getattr(item_info, "source_item_key", None) or item_info.item_key
    if not source_path:
        return {}
    source_path = os.path.abspath(os.path.expanduser(str(source_path)))
    try:
        source_stat = os.stat(source_path)
    except OSError:
        return {}
    if not os.path.isfile(source_path):
        return {}

    cache_path = getattr(item_info, "latent_cache_path", None)
    source_path_format = SOURCE_PATH_FORMAT_FILENAME_ONLY
    portable_source_path = os.path.basename(source_path)
    if cache_path:
        cache_directory = os.path.dirname(os.path.abspath(os.path.expanduser(str(cache_path))))
        try:
            portable_source_path = os.path.relpath(source_path, cache_directory).replace("\\", "/")
            source_path_format = SOURCE_PATH_FORMAT_RELATIVE_TO_CACHE
        except ValueError:
            # Windows paths on different drives have no meaningful relative
            # representation. Keep a non-identifying filename for diagnostics
            # instead of embedding a machine-specific absolute path.
            pass

    metadata = {
        SOURCE_PATH_METADATA_KEY: portable_source_path,
        SOURCE_PATH_FORMAT_METADATA_KEY: source_path_format,
        SOURCE_SIZE_METADATA_KEY: str(source_stat.st_size),
        SOURCE_MTIME_NS_METADATA_KEY: str(source_stat.st_mtime_ns),
    }
    target_fps = getattr(item_info, "target_fps", None)
    if isinstance(target_fps, (int, float)) and target_fps > 0:
        metadata["target_fps"] = str(float(target_fps))
        if isinstance(item_info.frame_count, int) and item_info.frame_count > 0:
            metadata["duration_seconds"] = str(float(item_info.frame_count) / float(target_fps))
    return metadata


# Common audio conventions (audio-capable architectures):
# - the target audio latent is stored as `latents_audio_<shape>_<dtype>` (shape layout is
#   architecture-specific) in the same latent cache file
# - AUDIO_PRESENT_KEY holds a scalar 0/1 float32 tensor recording whether the item had real
#   audio (0: silence placeholder was encoded). This is a fact about the data; supervision
#   policy (loss weights, video-only training) is decided at training time.
AUDIO_PRESENT_KEY = "audio_present_float32"


def append_audio_present_entry(sd: dict[str, torch.Tensor], audio_present: bool):
    sd[AUDIO_PRESENT_KEY] = torch.tensor(1.0 if audio_present else 0.0, dtype=torch.float32)


def validate_audio_present_entry(sd: dict[str, torch.Tensor]) -> float:
    """Validates the audio_present entry of a latent cache dict and returns its value."""
    tensor = sd.get(AUDIO_PRESENT_KEY)
    if not isinstance(tensor, torch.Tensor) or tensor.shape != torch.Size([]) or tensor.dtype != torch.float32:
        raise ValueError(f"Audio latent cache requires a scalar float32 {AUDIO_PRESENT_KEY} tensor")
    value = tensor.item()
    if value not in (0.0, 1.0):
        raise ValueError(f"{AUDIO_PRESENT_KEY} must be exactly 0.0 or 1.0, got {value}")
    return value


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
):
    """Wan architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_ltx2(
    item_info: ItemInfo,
    latent: torch.Tensor,
    extra_tensors: Optional[dict[str, torch.Tensor]] = None,
    *,
    atomic: bool = False,
):
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}
    if extra_tensors:
        for key, value in extra_tensors.items():
            sd[key] = value.detach().cpu().contiguous()

    save_latent_cache_common(
        item_info,
        sd,
        ARCHITECTURE_LTX2_FULL,
        atomic=atomic,
        extra_metadata=build_source_freshness_metadata(item_info),
    )


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
):
    """FramePack architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()

    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_flux_kontext(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: torch.Tensor,
):
    """FLUX.1 Kontext architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    _, H, W = control_latent.shape
    F = 1
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_latent_cache_flux_2(
    item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]], arch_full: str
):
    """Flux 2 architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, arch_full)


def save_latent_cache_qwen_image(item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]]):
    """Qwen-Image architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"
    assert control_latent is None or all(cl.dim() == 4 for cl in control_latent), (
        "control_latent should be 4D tensor (frame, channel, height, width) or None"
    )

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, F, H, W = cl.shape
            sd[f"latents_control_{i}_{F}x{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_latent_cache_krea2(item_info: ItemInfo, latent: torch.Tensor):
    """Krea 2 (K2) architecture. Single image (F=1), Qwen-Image VAE latents (normalized).

    The latent uses the *same* normalization as the Qwen-Image VAE
    (`(raw - mean) / std`), which is exactly what K2's decoder inverts, so the
    Qwen-Image latent caching is reused as-is. No control latent for plain t2i.
    """
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


def save_latent_cache_kandinsky5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor] = None,
    control_latent: Optional[torch.Tensor] = None,
    scaling_factor: Optional[float] = None,
):
    """Kandinsky 5 architecture (image/video), with optional source/control latents for i2v/control."""
    assert latent.dim() == 3 or latent.dim() == 4, "latent should be 3D (C,H,W) or 4D (F,C,H,W) tensor"

    if latent.dim() == 4:
        _, F, H, W = latent.shape
    else:
        F, H, W = 1, latent.shape[1], latent.shape[2]
        latent = latent.unsqueeze(0)
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous().clone()}

    if image_latent is not None:
        _, F_img, H_img, W_img = image_latent.shape
        sd[f"latents_image_{F_img}x{H_img}x{W_img}_{dtype_str}"] = image_latent.detach().cpu().contiguous().clone()

    if control_latent is not None:
        _, F_ctrl, H_ctrl, W_ctrl = control_latent.shape
        sd[f"latents_control_{F_ctrl}x{H_ctrl}x{W_ctrl}_{dtype_str}"] = control_latent.detach().cpu().contiguous().clone()

    if scaling_factor is not None:
        sd["vae_scaling_factor"] = torch.tensor(float(scaling_factor))

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_latent_cache_hunyuan_video_1_5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor],
    vision_feature: Optional[torch.Tensor],
):
    """HunyuanVideo 1.5 architecture"""
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd: dict[str, torch.Tensor] = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if image_latent is not None:
        dtype_str = dtype_to_str(image_latent.dtype)
        _, F, H, W = image_latent.shape
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if vision_feature is not None:
        dtype_str = dtype_to_str(vision_feature.dtype)
        sd[f"siglip_{dtype_str}"] = vision_feature.detach().cpu()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_latent_cache_z_image(item_info: ItemInfo, latent: torch.Tensor):
    """Z-Image architecture. No control latent is supported."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    C, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_pixel_cache_hidream_o1(
    item_info: ItemInfo,
    pixel_tokens: torch.Tensor,
    control_pixel_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    """HiDream-O1 pixel-token cache."""
    height_patches, width_patches, _ = pixel_tokens.shape
    dtype_str = dtype_to_str(pixel_tokens.dtype)
    sd = {f"latents_1x{height_patches}x{width_patches}_{dtype_str}": pixel_tokens.detach().cpu().contiguous()}

    if control_pixel_tokens is not None:
        if torch.is_tensor(control_pixel_tokens):
            assert control_pixel_tokens.dim() == 4, (
                "control_pixel_tokens should be 4D tensor (num_controls, height_patches, width_patches, patch_dim)"
            )
            control_iter = list(control_pixel_tokens)
        else:
            control_iter = control_pixel_tokens
        for i, cl in enumerate(control_iter):
            control_height_patches, control_width_patches, _ = cl.shape
            control_dtype_str = dtype_to_str(cl.dtype)
            sd[f"latents_control_{i}_{control_height_patches}x{control_width_patches}_{control_dtype_str}"] = (
                cl.detach().cpu().contiguous()
            )

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL)


def save_latent_cache_ideogram4(item_info: ItemInfo, latent: torch.Tensor):
    """Ideogram 4 architecture."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def _merge_cache_metadata(required: dict[str, str], additional: Optional[dict[str, str]]) -> dict[str, str]:
    metadata = dict(additional or {})
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("Safetensors metadata keys and values must be strings")
    metadata.update(required)
    return metadata


def save_latent_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    additional_metadata: Optional[dict[str, str]] = None,
    *,
    atomic: bool = False,
    extra_metadata: Optional[dict[str, str]] = None,
):
    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "width": f"{item_info.original_size[0]}",
            "height": f"{item_info.original_size[1]}",
            "format_version": "1.0.1",
        },
        additional_metadata,
    )
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"
    if extra_metadata:
        metadata.update(extra_metadata)

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    if atomic:
        safetensors_utils.save_file_atomic(sd, item_info.latent_cache_path, metadata=metadata)
    else:
        save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture"""
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_ltx2(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor]):
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"text_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd["text_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_LTX2_FULL)


def save_text_encoder_output_cache_ltx2_gemma(
    item_info: ItemInfo,
    *,
    video_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    audio_prompt_embeds: Optional[torch.Tensor] = None,
    video_features: Optional[torch.Tensor] = None,
    audio_features: Optional[torch.Tensor] = None,
    atomic: bool = False,
):
    assert video_prompt_embeds.dim() == 1 or video_prompt_embeds.dim() == 2, (
        f"video_prompt_embeds should be 2D tensor (feature, hidden_size) or (hidden_size,), got {video_prompt_embeds.shape}"
    )
    assert prompt_attention_mask is None or prompt_attention_mask.dim() == 1, (
        f"prompt_attention_mask should be 1D tensor (feature), got {prompt_attention_mask.shape}"
    )
    if audio_prompt_embeds is not None:
        assert audio_prompt_embeds.dim() == 1 or audio_prompt_embeds.dim() == 2, (
            f"audio_prompt_embeds should be 2D tensor (feature, hidden_size) or (hidden_size,), got {audio_prompt_embeds.shape}"
        )

    sd = {}
    dtype_str = dtype_to_str(video_prompt_embeds.dtype)

    sd[f"video_prompt_embeds_{dtype_str}"] = video_prompt_embeds.detach().cpu()
    if audio_prompt_embeds is not None:
        sd[f"audio_prompt_embeds_{dtype_str}"] = audio_prompt_embeds.detach().cpu()
    if prompt_attention_mask is not None:
        sd["prompt_attention_mask"] = prompt_attention_mask.detach().cpu()

    if video_features is not None:
        sd[f"video_features_{dtype_str}"] = video_features.detach().cpu()
    if audio_features is not None:
        sd[f"audio_features_{dtype_str}"] = audio_features.detach().cpu()

    text = video_prompt_embeds
    if audio_prompt_embeds is not None:
        text = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
    sd[f"text_{dtype_str}"] = text.detach().cpu()
    if prompt_attention_mask is not None:
        sd["text_mask"] = prompt_attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_LTX2_FULL, atomic=atomic)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd["llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_flux_kontext(item_info: ItemInfo, t5_vec: torch.Tensor, clip_l_pooler: torch.Tensor):
    """Flux Kontext architecture."""

    sd = {}
    dtype_str = dtype_to_str(t5_vec.dtype)
    sd[f"t5_vec_{dtype_str}"] = t5_vec.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str):
    """Flux 2 architecture."""

    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_qwen_image(item_info: ItemInfo, embed: torch.Tensor):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_text_encoder_output_cache_krea2(item_info: ItemInfo, embed: torch.Tensor):
    """Krea 2 (K2) architecture.

    `embed` is the per-item stack of *selected* Qwen3-VL hidden-state layers for the
    valid (non-padding) tokens only: shape (valid_len, num_select_layers, hidden).
    Stored varlen (no padding, no mask): K2 gives text tokens zero RoPE position and
    masks padding in attention, so dropping padding is lossless for the image outputs.
    The layerwise fusion (TextFusionTransformer) is trainable and lives in the DiT, so
    the raw selected-layer stack is what gets cached.
    """
    assert embed.dim() == 3, "embed should be 3D tensor (valid_len, num_select_layers, hidden)"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_krea2_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


def save_text_encoder_output_cache_kandinsky5(
    item_info: ItemInfo, text_embeds: torch.Tensor, pooled_embed: torch.Tensor, attention_mask: torch.Tensor
):
    """Kandinsky 5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(text_embeds.dtype)
    sd[f"text_embeds_{dtype_str}"] = text_embeds.detach().cpu()
    dtype_str = dtype_to_str(pooled_embed.dtype)
    sd[f"pooled_embed_{dtype_str}"] = pooled_embed.detach().cpu()
    sd["attention_mask"] = attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_text_encoder_output_cache_hunyuan_video_1_5(item_info: ItemInfo, embed: torch.Tensor, byt5_embed: torch.Tensor):
    """Hunyuan-Video 1.5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()
    dtype_str = dtype_to_str(byt5_embed.dtype)
    sd[f"varlen_byt5_embed_{dtype_str}"] = byt5_embed.detach().cpu()
    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_text_encoder_output_cache_z_image(item_info: ItemInfo, embed: torch.Tensor):
    """Z-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_llm_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_text_encoder_output_cache_ideogram4(item_info: ItemInfo, features: torch.Tensor):
    """Ideogram 4 architecture."""
    sd = {}
    dtype_str = dtype_to_str(features.dtype)
    sd[f"varlen_i4_llm_features_{dtype_str}"] = features.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def save_text_encoder_output_cache_hidream_o1(
    item_info: ItemInfo,
    input_ids: torch.Tensor,
    input_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_types: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
):
    """HiDream-O1 architecture."""
    tensors = {
        "varlen_input_ids": input_ids,
        "varlen_input_embeds": input_embeds,
        "varlen_position_ids": position_ids,
        "varlen_token_types": token_types,
        "varlen_image_grid_thw": image_grid_thw,
    }
    sd = {f"{name}_{dtype_to_str(t.dtype)}": t.detach().cpu() for name, t in tensors.items() if t is not None}

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL, merge_existing=False)


def _h3_dtype_matches(tensor: torch.Tensor, dtype_name: str) -> bool:
    return dtype_to_str(tensor.dtype) == dtype_name


def save_latent_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    import re

    target_pattern = re.compile(r"^latents_(\d+)x(\d+)x(\d+)_(.+)$")
    audio_pattern = re.compile(r"^latents_audio_32x2x(\d+)_(.+)$")
    visual_condition_pattern = re.compile(r"^latents_(?:first|last|ref_\d{3}_(?:image|video))_(\d+)x(\d+)x(\d+)_(.+)$")
    audio_condition_pattern = re.compile(r"^latents_ref_\d{3}_audio_32x2x(\d+)_(.+)$")

    target_count = 0
    audio_count = 0
    normalized = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"MiniMax-H3 cache value must be a tensor: {key}")
        if key == AUDIO_PRESENT_KEY:
            normalized[key] = tensor.detach().cpu().contiguous()
            continue

        match = target_pattern.fullmatch(key)
        if match is not None:
            frames, height, width = (int(match.group(index)) for index in range(1, 4))
            if tensor.shape != (24, frames, height, width):
                raise ValueError(f"MiniMax-H3 target video latent must be [24,F,H,W], got {tuple(tensor.shape)}")
            if not _h3_dtype_matches(tensor, match.group(4)):
                raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
            target_count += 1
        else:
            match = audio_pattern.fullmatch(key)
            if match is not None:
                audio_frames = int(match.group(1))
                if tensor.shape != (32, 2, audio_frames):
                    raise ValueError(f"MiniMax-H3 audio latent [32,2,A] required, got {tuple(tensor.shape)}")
                if not _h3_dtype_matches(tensor, match.group(2)):
                    raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
                audio_count += 1
            else:
                match = visual_condition_pattern.fullmatch(key)
                if match is not None:
                    frames, height, width = (int(match.group(index)) for index in range(1, 4))
                    if tensor.shape != (24, frames, height, width):
                        raise ValueError(f"MiniMax-H3 visual condition latent must be [24,F,H,W], got {tuple(tensor.shape)}")
                    if not _h3_dtype_matches(tensor, match.group(4)):
                        raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
                else:
                    match = audio_condition_pattern.fullmatch(key)
                    if match is None:
                        raise ValueError(f"Unsupported MiniMax-H3 latent cache key: {key}")
                    audio_frames = int(match.group(1))
                    if tensor.shape != (32, 2, audio_frames):
                        raise ValueError(f"MiniMax-H3 audio latent [32,2,A] required, got {tuple(tensor.shape)}")
                    if not _h3_dtype_matches(tensor, match.group(2)):
                        raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
        normalized[key] = tensor.detach().cpu().contiguous()

    if target_count != 1:
        raise ValueError(f"MiniMax-H3 cache requires exactly one target video latent, found {target_count}")
    if audio_count != 1:
        raise ValueError(f"MiniMax-H3 cache requires exactly one target audio latent, found {audio_count}")
    validate_audio_present_entry(normalized)
    save_latent_cache_common(item_info, normalized, ARCHITECTURE_MINIMAX_H3_FULL, metadata)


def save_text_encoder_output_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    # the teacher prefix must be split off before matching the student prefix, because
    # "varlen_mmh3_teacher_hidden_states_*" does not share the student prefix
    student_hidden_keys = [key for key in tensors if key.startswith("varlen_mmh3_hidden_states_")]
    teacher_hidden_keys = [key for key in tensors if key.startswith("varlen_mmh3_teacher_hidden_states_")]
    if len(student_hidden_keys) != 1:
        raise ValueError(f"MiniMax-H3 text cache requires exactly one hidden-state tensor, found {len(student_hidden_keys)}")
    tags_key = "varlen_mmh3_token_tags_int64"
    teacher_tags_key = "varlen_mmh3_teacher_token_tags_int64"

    pairs = [(student_hidden_keys[0], "varlen_mmh3_hidden_states_", tags_key)]
    expected_keys = {student_hidden_keys[0], tags_key}
    if teacher_hidden_keys or teacher_tags_key in tensors:
        if len(teacher_hidden_keys) != 1 or teacher_tags_key not in tensors:
            raise ValueError("MiniMax-H3 teacher text rows require exactly one hidden-state tensor and its token tags")
        pairs.append((teacher_hidden_keys[0], "varlen_mmh3_teacher_hidden_states_", teacher_tags_key))
        expected_keys |= {teacher_hidden_keys[0], teacher_tags_key}
    if set(tensors) != expected_keys:
        raise ValueError(f"MiniMax-H3 text cache requires exactly the keys {sorted(expected_keys)}")

    normalized = {}
    for hidden_key, hidden_prefix, pair_tags_key in pairs:
        hidden_states = tensors[hidden_key]
        token_tags = tensors[pair_tags_key]
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 5120:
            raise ValueError(f"MiniMax-H3 hidden states must be [L,5120], got {tuple(hidden_states.shape)}")
        if not _h3_dtype_matches(hidden_states, hidden_key.removeprefix(hidden_prefix)):
            raise ValueError(f"MiniMax-H3 hidden-state key dtype does not match tensor: {hidden_key}")
        if hidden_states.shape[0] > 32768:
            raise ValueError(f"MiniMax-H3 text cache exceeds 32768 rows: {hidden_states.shape[0]}")
        if token_tags.dtype != torch.int64 or token_tags.shape != (hidden_states.shape[0],):
            raise ValueError("MiniMax-H3 token tags must be int64 [L]")
        if not torch.all((token_tags == 0) | (token_tags == 1)):
            raise ValueError("MiniMax-H3 token tags may contain only 0 and 1")
        normalized[hidden_key] = hidden_states.detach().cpu().contiguous()
        normalized[pair_tags_key] = token_tags.detach().cpu().contiguous()
    save_text_encoder_output_cache_common(
        item_info,
        normalized,
        ARCHITECTURE_MINIMAX_H3_FULL,
        merge_existing=False,
        additional_metadata=metadata,
    )


def save_text_encoder_output_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    *,
    atomic: bool = False,
    merge_existing: bool = True,
    additional_metadata: Optional[dict[str, str]] = None,
):
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "caption1": item_info.caption,
            "format_version": "1.0.1",
        },
        additional_metadata,
    )
    if merge_existing and os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        new_key_bases = {remove_dtype_suffix(key) for key in sd}
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if remove_dtype_suffix(key) not in new_key_bases:
                    sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata, atomic=atomic)
