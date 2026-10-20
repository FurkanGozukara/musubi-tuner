"""MOSS-Image colour / depth control adapters, in the format ComfyUI loads ("MOSS-Image Color/Depth Control", ComfyUI
e3bee7f; the published example is darask0/MOSS-Image-color-depth-control, the spec in dont_upload/moss_control_adapter_spec.md).

The DiT's patch embedder ``video_embedder`` (Linear 128 -> 3072 over the FLUX.2 VAE's packed latent channels) is widened
to 384 inputs, ``[noisy latent | colour-region map latent | depth map latent]``: the maps are encoded by the same VAE at
the target's size and concatenated channel-wise, clean (not noised); a missing map is a zero latent, which the zero
initialised new columns read as "no control". The released columns and the bias train at the LoRA rate, the new
control columns at a multiple of it; a LoRA / DoRA on the image blocks and on the text refiner's two blocks trains with
them. Each step keeps the colour map only, the depth map only, both or neither (35 / 35 / 20 / 10 %, the published
recipe), so either map alone works at inference.

The file ComfyUI detects holds unprefixed diffusers-style keys: ``layers.N.proj.q.lora_A.weight`` / ``lora_B.weight``
(q / k / v separate; ComfyUI applies them to row slices of its fused ``proj.qkv``), the text refiner's
``text_embedder.blocks.N.qkv`` ..., the LoRA scale (alpha / rank) folded into ``lora_B``, DoRA magnitudes as
``<module>.dora_scale``, and the widened ``video_embedder.weight`` / ``video_embedder.bias``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

CONTROL_MAPS = ("colour", "depth")  # dataset control images <name>_0 and <name>_1, in this order
LATENT_CHANNELS = 128
CONTROL_CHANNELS = LATENT_CHANNELS * len(CONTROL_MAPS)
# shares of steps that keep: the colour map only, the depth map only, both, neither (the published adapter's recipe)
DEFAULT_CONTROL_DROPOUT = (0.35, 0.35, 0.2, 0.1)
# the empty caption's text encoder output, written beside a control dataset's text encoder caches (not an item cache,
# whose names end in _mi_te) for the caption dropout
EMPTY_CAPTION_FILE = "moss_image_empty_caption_te.safetensors"
# sample prompts: a --ci entry that leaves its map out (--ci none --ci depth.png: depth only)
NO_MAP = ("", "-", "none", "zero", "zeros")
_LORA_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha", "dora_scale")


class MossControlEmbedder(nn.Module):
    """``video_embedder`` widened to ``[noisy | colour | depth]`` inputs; the control columns start at zero. The
    parameters are FP32 masters (they train with an optimizer); the product runs in the input's dtype."""

    def __init__(self, base: nn.Linear, control_channels: Optional[int] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        device = base.weight.device
        control_channels = control_channels or base.in_features * len(CONTROL_MAPS)  # 256 on the released model
        self.base_weight = nn.Parameter(base.weight.detach().to(dtype=dtype).clone())
        self.control_weight = nn.Parameter(torch.zeros(base.weight.shape[0], control_channels, dtype=dtype, device=device))
        self.bias = nn.Parameter(base.bias.detach().to(dtype=dtype).clone()) if base.bias is not None else None
        self.in_features = base.in_features + control_channels
        self.out_features = base.out_features

    @property
    def weight(self) -> Tensor:
        return torch.cat([self.base_weight, self.control_weight], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)

    def load_widened(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        """Load a saved widened embedder (``video_embedder.weight`` [out, base + control] and its bias)."""
        base = self.base_weight.shape[1]
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"control adapter embedder {tuple(weight.shape)} does not fit this model's {(self.out_features, self.in_features)}")
        with torch.no_grad():
            self.base_weight.copy_(weight[:, :base])
            self.control_weight.copy_(weight[:, base:])
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias)

    def control_share(self) -> dict:
        """Norms of the colour and depth columns relative to the noisy-latent columns (the published adapter: about
        0.37 / 0.41 after 2,500 steps, 0.69 / 0.85 after 10,000)."""
        with torch.no_grad():
            base = float(self.base_weight.float().norm())
            parts = self.control_weight.float().split(self.base_weight.shape[1], dim=1)
            return {name: float(part.norm()) / base for name, part in zip(CONTROL_MAPS, parts)}


def widen_video_embedder(model: nn.Module) -> MossControlEmbedder:
    """Replace the model's ``video_embedder`` by a :class:`MossControlEmbedder` (in place) and return it."""
    embedder = model.video_embedder
    if isinstance(embedder, MossControlEmbedder):
        return embedder
    widened = MossControlEmbedder(embedder)
    model.video_embedder = widened
    return widened


def parse_control_dropout(text: Optional[str]) -> tuple[float, float, float, float]:
    """``"colour_only,depth_only,both,none"`` shares (normalised to sum 1)."""
    if text is None or not str(text).strip():
        return DEFAULT_CONTROL_DROPOUT
    try:
        values = [float(part) for part in str(text).replace(";", ",").split(",") if part.strip()]
    except ValueError:
        values = []
    if len(values) != 4 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(f"--control_dropout needs four non-negative shares (colour only, depth only, both, none), got {text!r}")
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def control_keep(batch_size: int, shares: Sequence[float], device=None, generator: Optional[torch.Generator] = None) -> tuple[Tensor, Tensor]:
    """Per sample, whether to keep the colour map and the depth map: (B,) float masks drawn from ``shares``."""
    draw = torch.rand(batch_size, generator=generator)
    edges = torch.tensor(list(shares), dtype=torch.float32).cumsum(0)
    case = torch.bucketize(draw, edges[:-1], right=True)  # 0 colour only, 1 depth only, 2 both, 3 none
    keep_colour = ((case == 0) | (case == 2)).float()
    keep_depth = ((case == 1) | (case == 2)).float()
    return keep_colour.to(device), keep_depth.to(device)


def control_latents(like: Tensor, maps: Sequence[Optional[Tensor]], keep: Optional[Sequence[Tensor]] = None) -> Tensor:
    """``[colour | depth]`` (B, 256, H, W) shaped and typed like ``like`` (B, 128, H, W); a missing or dropped map is zeros."""
    parts = []
    for index in range(len(CONTROL_MAPS)):
        latent = maps[index] if index < len(maps) else None
        if latent is None:
            parts.append(torch.zeros_like(like))
            continue
        latent = latent.to(device=like.device, dtype=like.dtype)
        if latent.shape[0] == 1 and like.shape[0] > 1:
            latent = latent.expand(like.shape[0], *latent.shape[1:])
        if latent.shape != like.shape:
            raise ValueError(f"control map latent {tuple(latent.shape)} does not match the target latent {tuple(like.shape)}")
        if keep is not None:
            latent = latent * keep[index].to(device=like.device, dtype=like.dtype).view(-1, 1, 1, 1)
        parts.append(latent)
    return torch.cat(parts, dim=1)


def control_input(noisy: Tensor, maps: Sequence[Optional[Tensor]], keep: Optional[Sequence[Tensor]] = None) -> Tensor:
    """``[noisy | colour | depth]`` along channels (B, 384, H, W); a missing or dropped map is zeros."""
    return torch.cat([noisy, control_latents(noisy, maps, keep)], dim=1)


class ControlledDiT(nn.Module):
    """A sampling view of the DiT that appends the control channels to every call's latent input."""

    def __init__(self, model: nn.Module, control: Tensor):
        super().__init__()
        self.model = model
        self.control = control  # (B, 256, H, W)

    def forward(self, latents: Tensor, *args, **kwargs):
        control = self.control.to(device=latents.device, dtype=latents.dtype)
        if control.shape[0] != latents.shape[0]:
            control = control.expand(latents.shape[0], *control.shape[1:])
        return self.model(torch.cat([latents, control], dim=1), *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# --------------------------------------------------------------------------------------------------------- the maps
def fit_to_size(image: Image.Image, width: int, height: int, resample) -> Image.Image:
    """Scale to cover ``width`` x ``height`` and crop the centre: the training images' bucket fit
    (dataset.media_utils.resize_image_to_bucket)."""
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    scaled = (int(image.width * scale + 0.5), int(image.height * scale + 0.5))
    image = image.resize(scaled, resample)
    left, top = (scaled[0] - width) // 2, (scaled[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def load_control_map(path: str, width: int, height: int, kind: str) -> Tensor:
    """A colour-region or depth map as (3, height, width) pixels in [-1, 1]. Colour maps scale with nearest neighbour
    (the colours are region IDs), depth maps with Lanczos."""
    if kind not in CONTROL_MAPS:
        raise ValueError(f"unknown control map kind {kind!r}")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image = fit_to_size(image, width, height, Image.NEAREST if kind == "colour" else Image.LANCZOS)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
    return pixels.permute(2, 0, 1).contiguous()


def is_no_map(path) -> bool:
    return path is None or str(path).strip().lower() in NO_MAP


@torch.no_grad()
def sample_control(vae, paths: Optional[Sequence[str]], width: int, height: int, device) -> Tensor:
    """The control channels of one sample (1, 256, height / 16, width / 16): sample prompt ``--ci`` entries in map order
    (the colour map, then the depth map; ``--ci none`` leaves a map out), encoded by the FLUX.2 VAE (zeros for a missing
    map, as ComfyUI's node). The VAE goes back to the CPU afterwards."""
    paths = list(paths or [])
    if len(paths) > len(CONTROL_MAPS):
        raise ValueError(f"MOSS-Image control samples take at most {len(CONTROL_MAPS)} --ci maps (colour, depth), got {len(paths)}")
    latents: list[Optional[Tensor]] = []
    if any(not is_no_map(path) for path in paths):
        vae.to(device)
        try:
            for index, kind in enumerate(CONTROL_MAPS):
                path = paths[index] if index < len(paths) else None
                if is_no_map(path):
                    latents.append(None)
                    continue
                pixels = load_control_map(str(path).strip(), width, height, kind).unsqueeze(0)
                latents.append(vae.encode(pixels.to(device=device, dtype=vae.dtype)).float())
        finally:
            vae.to("cpu")
    like = torch.zeros(1, LATENT_CHANNELS, height // 16, width // 16, device=device, dtype=torch.float32)
    return control_latents(like, latents)


# ----------------------------------------------------------------------------------------------------------- caching
def latent_cache_is_current(item, base_test) -> bool:
    """--skip_existing for MOSS-Image: ``base_test`` (the bucket test) and the cached control maps match the item's
    (two latents when it has maps, none when it has none), so maps added to or removed from a dataset are re-encoded."""
    if not base_test(item):
        return False
    from musubi_tuner.cache_latents import read_cache_header

    header = read_cache_header(item.latent_cache_path)
    if header is None:
        return True
    cached = sum(1 for key in header if key.startswith("latents_control_"))
    controls = getattr(item, "control_content", None)
    wanted = 0 if controls is None else len(controls) if isinstance(controls, list) else 1
    if cached != wanted:
        logger.info(f"Re-encoding {os.path.basename(item.latent_cache_path)}: {cached} cached control maps, {wanted} now")
        return False
    return True


def empty_caption_path(cache_directory: str) -> str:
    return os.path.join(cache_directory, EMPTY_CAPTION_FILE)


def save_empty_caption(path: str, ctx: Tensor, architecture: str) -> None:
    from safetensors.torch import save_file

    from musubi_tuner.utils.model_utils import dtype_to_str

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tensor = ctx.detach().to("cpu").contiguous()
    save_file({f"varlen_ctx_vec_{dtype_to_str(tensor.dtype)}": tensor}, path, metadata={"architecture": architecture, "caption1": ""})


def load_empty_caption(cache_directories: Sequence[str]) -> Tensor:
    """The empty caption's (L, 4096) context from the first cache directory that holds it."""
    from safetensors.torch import load_file

    for directory in cache_directories:
        if not directory:
            continue
        path = empty_caption_path(directory)
        if os.path.isfile(path):
            tensors = load_file(path)
            for key, value in tensors.items():
                if key.startswith("varlen_ctx_vec_"):
                    return value
    raise ValueError(
        f"MOSS-Image control adapter caption dropout needs the empty caption's text encoder output ({EMPTY_CAPTION_FILE}), "
        "which moss_image_cache_text_encoder_outputs.py writes beside the caches of a dataset with control maps: run the "
        "text encoder caching again, or set --control_caption_dropout 0"
    )


# ------------------------------------------------------------------------------------------------------------ export
def comfy_control_state_dict(lora_sd: dict, module_names: dict, embedder: MossControlEmbedder, dtype: Optional[torch.dtype] = None) -> dict:
    """The trainer's LoRA / DoRA weights (``lora_unet_<module>.lora_down / lora_up / alpha / dora_scale``) and the widened
    embedder as the control-adapter file ComfyUI loads. ``module_names`` maps ``lora_unet_<path with _>`` to the dotted
    module path (``lora_unet_layers_0_proj_ffn_up_gate`` -> ``layers.0.proj.ffn_up_gate``). ``dtype`` defaults to the
    LoRA weights' own; DoRA scales keep theirs (the trainer saves them in FP32: they scale whole columns of the base
    weight, which BF16 would round by up to 0.4%)."""
    groups: dict[str, dict[str, Tensor]] = {}
    for key, value in lora_sd.items():
        lora_name, _, suffix = key.partition(".")
        if lora_name not in module_names:
            raise KeyError(f"no MOSS-Image module for LoRA key {key!r}")
        if suffix not in _LORA_SUFFIXES:
            raise ValueError(f"{key!r}: the control adapter format holds LoRA / input-axis DoRA weights only")
        groups.setdefault(lora_name, {})[suffix] = value
    out: dict[str, Tensor] = {}
    embedder_dtype = dtype
    for lora_name, parts in groups.items():
        module = module_names[lora_name]
        down, up = parts["lora_down.weight"], parts["lora_up.weight"]
        out_dtype = dtype or down.dtype
        embedder_dtype = embedder_dtype or out_dtype
        rank = down.shape[0]
        alpha = float(parts["alpha"]) if "alpha" in parts else float(rank)
        out[f"{module}.lora_A.weight"] = down.to(out_dtype).contiguous()
        out[f"{module}.lora_B.weight"] = (up.float() * (alpha / rank)).to(out_dtype).contiguous()
        if "dora_scale" in parts:
            out[f"{module}.dora_scale"] = parts["dora_scale"].contiguous()
    embedder_dtype = embedder_dtype or torch.bfloat16
    out["video_embedder.weight"] = embedder.weight.detach().to(device="cpu", dtype=embedder_dtype).contiguous()
    if embedder.bias is not None:
        out["video_embedder.bias"] = embedder.bias.detach().to(device="cpu", dtype=embedder_dtype).contiguous()
    return out


def lora_module_names(transformer: nn.Module) -> dict[str, str]:
    """``lora_unet_<path with _>`` -> dotted module path, for every module of the DiT."""
    return {"lora_unet_" + name.replace(".", "_"): name for name, _ in transformer.named_modules() if name}


def control_metadata(shares: Sequence[float], rank: Optional[int], base_model: str) -> dict:
    """Metadata of the exported file (ComfyUI ignores it; the published adapter's keys where they apply). ``alpha``
    equals the rank: the LoRA scale is folded into ``lora_B``."""
    names = ("color", "depth", "both", "none")
    metadata = {
        "control_channels": "color,depth",
        "control_preprocess": "vae",
        "control_dropout": json.dumps({name: round(float(share), 6) for name, share in zip(names, shares)}),
        "base_model": base_model,
        "moss_control_format": "comfyui",
    }
    if rank:
        metadata["rank"] = str(int(rank))
        metadata["alpha"] = str(float(rank))
    return metadata


def write_comfy_control_file(path: str, embedder: MossControlEmbedder, module_names: dict, extra_metadata: Optional[dict] = None) -> int:
    """Rewrite the trainer's LoRA checkpoint at ``path`` as the ComfyUI control adapter (in place, atomically); returns
    the number of tensors written."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, "pt") as handle:
        metadata = dict(handle.metadata() or {})
        lora_sd = {key: handle.get_tensor(key) for key in handle.keys()}
    out = comfy_control_state_dict(lora_sd, module_names, embedder)
    ranks = {int(value.shape[0]) for key, value in out.items() if key.endswith(".lora_A.weight")}
    metadata.update(extra_metadata or {})
    if len(ranks) == 1:
        rank = ranks.pop()
        metadata["rank"] = str(rank)
        metadata["alpha"] = str(float(rank))
    temporary = path + ".tmp"
    save_file(out, temporary, metadata={key: str(value) for key, value in metadata.items()})
    os.replace(temporary, path)
    return len(out)
