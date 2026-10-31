"""FIBO 2.0 loading, Bria's text conditioning, edit references and the training samples' sampler.

* DiT: the ComfyUI single file of Comfy-Org/Fibo-2.0 (``fibo2_dev_bf16.safetensors``; Bria's diffusers shards merged,
  q / k / v and k / v fused, w1 / w3 fused into w13 with the hidden width padded to 9728). Editing is not a separate model:
  ai-toolkit trains text-to-image and editing on the Dev file, references in the dataset make it an edit. The
  step-distilled Edit-Turbo file is an inference model; the LoRA trained on Dev loads on it.
* VAE: the FLUX.2 VAE (``flux2_utils.load_ae``, byte-identical to FLUX.3 Klein's and MOSS-Image's file): its packed,
  BatchNorm-normalized 128 channels at 1/16 are FIBO 2.0's latents.
* Text: Qwen3-VL-4B-Instruct with its vision tower (the ComfyUI file, byte-identical to FLUX.3 Klein's). As Bria's
  pipeline (ai-toolkit's src/text_encoder.py, ComfyUI's ``Fibo2Tokenizer``): the prompt in the user turn of the chat
  template; a JSON caption is cleaned and compacted, plain text stays as it is, an empty prompt is the null caption.
  Hidden states 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33 and 35 (0 = the embeddings), each scaled to unit RMS in
  FP32 and stored in BF16, make six bundles of three (Perceiver, then one per injection block): (L, 6 * 7680) per prompt,
  cached as the 15 distinct taps (L, 15 * 2560) and expanded for the DiT. Text-only prompts share one right-padded
  forward without a mask and are cut back to their own length: every layer is causal, so the pads never reach a real
  token, whose rotary positions stay 0..n-1 (Guardrail 19: no pad state reaches the DiT). An edit prompt is one JSON
  object with a vision marker per reference, the result's caption when the prompt has one, then the instruction, one
  forward per prompt with its references through the vision tower.
* References (the dataset loader): a single reference is cropped to the target's size (the edit lines up with it pixel for
  pixel), several each keep their own aspect ratio at the target's pixel count (Bria's sizes, ``fibo2_sizes``); Qwen3-VL
  sees each crop thumbnailed to at most 768 px a side on its 32 px grid; the DiT sees their latents clean.
* Flow: sigma 1 = noise, the DiT predicts ``noise - clean`` at timestep ``1000 * sigma``. Training samples use
  ComfyUI's sampler: ModelSamplingFlux with shift 1.15 at every size (alpha e^1.15 = 3.158; Bria shifts by image size,
  1.15 at 1 MP), the ``simple`` scheduler's sigmas, Euler, CFG 5 against the null caption (a non-empty negative is
  written into the chat template as it is, not cleaned), references conditioning both CFG passes.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from musubi_tuner.fibo2 import fibo2_model
from musubi_tuner.fibo2.fibo2_sizes import TRAINED_AREA, trained_size  # noqa: F401  (Bria's sizes)
from musubi_tuner.flux_2.flux2_utils import load_ae  # noqa: F401  (the FLUX.2 VAE is FIBO 2.0's VAE)
from musubi_tuner.flux_3.flux3_utils import QWEN3_VL_4B_REPO, QWEN3_VL_4B_REVISION
from musubi_tuner.modules.comfy_quant_utils import COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------------------------------------- constants
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PAD = "<|image_pad|>"
VISION_MARKER = f"<|vision_start|>{IMAGE_PAD}<|vision_end|>"
# the null caption Fibo 2.0 was trained with for classifier-free guidance (and caption dropout)
NULL_CAPTION = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)
# Qwen3-VL hidden_states indices (0 = embeddings) behind the six bundles
BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = tuple(sorted({layer for bundle in BUNDLE_LAYERS for layer in bundle}))
BUNDLE_INDEX = tuple(TAP_LAYERS.index(layer) for bundle in BUNDLE_LAYERS for layer in bundle)
TEXT_HIDDEN = 2560
MAX_TEXT_TOKENS = 4096
TEXT_BATCH_SIZE = 8  # text-only prompts per padded forward
SCORES = ("aesthetic_score", "preference_score")

DEFAULT_SHIFT_MU = 1.15  # ComfyUI's ModelSamplingFlux shift for Fibo 2.0 (supported_models.Fibo2)
DEFAULT_SHIFT = math.exp(DEFAULT_SHIFT_MU)  # the same as musubi's discrete flow shift: 3.158
DEFAULT_GUIDANCE = 5.0  # Bria's default and the ComfyUI template's: CFG 5 against the null caption, 30 steps
DEFAULT_SAMPLE_STEPS = 30

# reference images (ai-toolkit src/references.py, Bria's BriaFibo2ImageProcessor and ComfyUI's port)
VLM_MAX_SIDE = 768
VLM_GRID = 32  # Qwen3-VL processor grid for Fibo 2.0
VLM_MIN_PIXELS = 3136
VLM_MAX_PIXELS = 1605632


# ---------------------------------------------------------------------------------------------------------------- DiT
COMFY_ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"  # a uint8 JSON tensor per block in Comfy-Org's INT8 files


def checkpoint_shapes(dit_path: str) -> dict[str, tuple]:
    with safe_open(dit_path, framework="pt") as handle:
        return {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}


def is_int8_file(shapes: dict) -> bool:
    """A ComfyUI pre-quantized file (Comfy-Org's ``fibo2_*_int8_convrot``): ``.comfy_quant`` specs beside the weights."""
    return any(k.endswith(COMFY_QUANT_SUFFIX) for k in shapes)


def check_dit_checkpoint(dit_path: str, allow_int8: bool = False) -> dict[str, tuple]:
    """Refuse files that cannot train as the FIBO 2.0 DiT, with the reason; returns the tensor shapes. ``allow_int8`` (LoRA
    training) takes Comfy-Org's INT8 ConvRot files, whose INT8 layers are used as they are."""
    shapes = checkpoint_shapes(dit_path)
    if "perceiver.latent_queries" not in shapes or "layers.0.attention.qkv.weight" not in shapes:
        if any(k.startswith(("transformer_blocks.", "layers.0.attention.to_q.")) for k in shapes):
            raise ValueError(
                f"{dit_path} is a diffusers-layout FIBO 2.0 transformer: train on Comfy-Org/Fibo-2.0's "
                "diffusion_models/fibo2_dev_bf16.safetensors (the same weights in ComfyUI's layout, the one LoRAs load on)"
            )
        raise ValueError(f"{dit_path} is not a FIBO 2.0 DiT checkpoint (no perceiver.latent_queries / layers.N.attention.qkv)")
    if "layers.0.feed_forward.w13.weight" not in shapes:
        raise ValueError(f"{dit_path}: a FIBO 2.0 file with unfused w1 / w3; use Comfy-Org/Fibo-2.0's fibo2_dev_bf16.safetensors")
    if any(k.endswith((COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX, ".input_scale", ".scale_weight")) for k in shapes):
        if not allow_int8:
            raise ValueError(
                f"{dit_path} is a pre-quantized FIBO 2.0 file (INT8 ConvRot): full fine-tuning trains the BF16 Dev file "
                "(fibo2_dev_bf16.safetensors). LoRA training takes both."
            )
        if any(k.endswith((".input_scale", ".scale_weight")) for k in shapes) or not is_int8_file(shapes):
            raise ValueError(f"{dit_path} is a pre-quantized FIBO 2.0 file of another format: only INT8 ConvRot files train")
    if "turbo" in dit_path.lower().replace("\\", "/").rsplit("/", 1)[-1]:
        logger.warning(
            f"{dit_path} looks like FIBO 2.0 Edit-Turbo, a step-distilled inference model. LoRAs (also editing ones) train on "
            "the Dev model (fibo2_dev_bf16.safetensors) and load on Edit-Turbo in ComfyUI."
        )
    return shapes


def load_dit(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    disable_numpy_memmap: bool = False,
    lora_weights_list: Optional[list[dict[str, Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    allow_int8_file: bool = False,
) -> fibo2_model.Fibo2:
    """The FIBO 2.0 DiT. ``convrot_int8`` quantizes the block linears (all but the adaLN modulations, the layers Comfy-Org's
    INT8 file holds in INT8) to ConvRot INT8 while loading: Hadamard groups of 256 with row-wise scales. ``allow_int8_file``
    (LoRA training) takes Comfy-Org's ``fibo2_dev_int8_convrot`` as it is. The INT8 layers keep their nn.Linear modules
    with a patched forward (fused Triton INT8 GEMM, a dequantizing fallback without Triton) and the model is frozen: INT8
    weights cannot require gradients."""
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)
    shapes = check_dit_checkpoint(dit_path, allow_int8=allow_int8_file)
    int8_file = is_int8_file(shapes)
    use_int8 = convrot_int8 or int8_file
    if use_int8 and (fp8_scaled or (dit_weight_dtype is not None and dit_weight_dtype.itemsize == 1)):
        raise ValueError("FIBO 2.0: ConvRot INT8 and FP8 are two quantizations of the same weights; choose one (FP8 Base / Scaled off)")
    if use_int8 and lora_weights_list:
        raise ValueError("FIBO 2.0: LoRA weights cannot be merged into a ConvRot INT8 DiT; apply them at runtime")
    params = fibo2_model.params_from_shapes(shapes)
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = fibo2_model.Fibo2(params, attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading FIBO 2.0 DiT from {dit_path}, device={loading_device}" + (" (ConvRot INT8)" if use_int8 else ""))
    quantizer = None
    if use_int8:
        # an INT8 file's layers are used as they are; only --convrot_int8 quantizes BF16 block linears
        targets = fibo2_model.FP8_OPTIMIZATION_TARGET_KEYS if convrot_int8 else []
        quantizer = ConvRotInt8Quantizer(targets, fibo2_model.FP8_OPTIMIZATION_EXCLUDE_KEYS)
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=None if use_int8 else dit_weight_dtype,
        target_keys=None if use_int8 else fibo2_model.FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=None if use_int8 else fibo2_model.FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
        quantizer=quantizer,
    )
    for key in [k for k in sd if k.endswith(COMFY_ATTENTION_CONFIG_SUFFIX)]:
        del sd[key]  # Comfy-Org's INT8 file: ComfyUI's attention backend per block ({"attention": "comfy_kitchen_int8"}), not a weight
    if use_int8:
        if dit_weight_dtype is not None:  # the unquantized layers (adaLN, norms, embedders) in the training dtype
            for key, value in sd.items():
                if value.is_floating_point() and not key.endswith(".scale_weight"):
                    sd[key] = value.to(dit_weight_dtype)
        apply_convrot_int8_monkey_patch(model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes)
        # load_state_dict(assign=True) gives the loaded tensors the placeholders' requires_grad, and INT8 cannot have it
        model.requires_grad_(False)
        print(f"FIBO 2.0: {len(quantizer.module_groupsizes)} block linears in ConvRot INT8 (backward {convrot_int8_bwd})")
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded FIBO 2.0: {info}")
    return model


# ------------------------------------------------------------------------------------------------------- text encoder
def load_text_encoder(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu", disable_mmap: bool = True):
    """Qwen3-VL-4B-Instruct (language and vision tower) from a local safetensors (ComfyUI or official layout), its
    pinned tokenizer and image processor: (model, tokenizer, image_processor)."""
    from transformers import AutoImageProcessor, AutoTokenizer

    from musubi_tuner.krea2.krea2_encoder import _load_qwen3_vl_model

    model = _load_qwen3_vl_model(path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_4B_REPO, revision=QWEN3_VL_4B_REVISION)
    image_processor = AutoImageProcessor.from_pretrained(QWEN3_VL_4B_REPO, revision=QWEN3_VL_4B_REVISION)
    return model.eval().requires_grad_(False), tokenizer, image_processor


def _dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _empty(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean(value):
    if isinstance(value, dict):
        value = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean, value) if not _empty(v)]
    return value


def parse_prompt(prompt: str):
    """(caption, edit instruction) as diffusers' BriaFibo2Pipeline parses them; a prompt that isn't a JSON object comes
    back as it is."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction
    keys = set(caption)
    caption = {
        key: {k: v for k, v in value.items() if k not in SCORES} if isinstance(value, dict) else value
        for key, value in _clean(caption).items()
        if key not in SCORES
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    # exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "", "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def generation_text(prompt: str) -> str:
    """The text a text-to-image prompt is encoded as: the null caption for an empty prompt, a JSON caption cleaned and
    compacted, plain text as it is."""
    if prompt.strip() == "":
        return NULL_CAPTION
    caption, _ = parse_prompt(prompt)
    return prompt if isinstance(caption, str) else _dumps(caption or {})


def edit_text(prompt: str, num_images: int) -> str:
    """An edit prompt: one JSON object with a vision marker per reference, the result's caption when the prompt has one,
    then the instruction (plain text is the instruction)."""
    caption, instruction = parse_prompt(prompt)
    if instruction is None:
        caption, instruction = None, prompt
    if num_images == 1:
        edit = {"image": VISION_MARKER}
    else:
        edit = {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return _dumps(edit)


def template_text(prompt: str, as_written: bool = False) -> str:
    """The chat turn a text-only prompt is encoded as. ``as_written`` (a sample's negative, as ComfyUI's TextEncodeFibo2
    encodes it): a non-empty text goes into the template as it is, not cleaned like a caption; empty is the null caption
    either way."""
    if as_written and prompt.strip():
        return PROMPT_TEMPLATE.format(prompt)
    return PROMPT_TEMPLATE.format(generation_text(prompt))


def _normalized_taps(hidden_states, dtype: torch.dtype) -> Tensor:
    """(B, L, 15 * 2560): the tapped hidden states, each scaled to unit RMS in FP32, in ``dtype``."""
    taps = []
    for layer in TAP_LAYERS:
        states = hidden_states[layer].float()
        taps.append((states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype))
    return torch.cat(taps, -1)


def expand_taps(taps: Tensor) -> Tensor:
    """(..., L, 15 * 2560) cached taps -> (..., L, 6 * 7680) bundles, the DiT's (and ComfyUI's) context layout."""
    shape = taps.shape[:-1]
    return taps.reshape(*shape, len(TAP_LAYERS), TEXT_HIDDEN)[..., list(BUNDLE_INDEX), :].reshape(*shape, -1)


@torch.no_grad()
def encode_prompts(
    model,
    tokenizer,
    image_processor,
    prompts: Sequence[str],
    images: Optional[Sequence[Optional[list[Image.Image]]]] = None,
    max_length: int = MAX_TEXT_TOKENS,
    batch_size: int = TEXT_BATCH_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    as_written: Optional[Sequence[bool]] = None,
) -> list[Tensor]:
    """Prompts (and per-prompt reference images, already cropped) -> one (L_i, 15 * 2560) tensor of normalized taps per
    prompt, on the text encoder's device. Text-only prompts share right-padded forwards without a mask, each cut back to
    its own length; a prompt with references runs alone. An empty prompt is the null caption, without images.
    ``as_written[i]``: prompt i is a sample negative, text only and not cleaned (``template_text``)."""
    device = next(model.parameters()).device
    images = list(images) if images is not None else [None] * len(prompts)
    as_written = list(as_written) if as_written is not None else [False] * len(prompts)
    out: list[Optional[Tensor]] = [None] * len(prompts)
    backbone = model.model

    text_only = [i for i, imgs in enumerate(images) if not imgs or prompts[i].strip() == "" or as_written[i]]
    for start in range(0, len(text_only), max(1, batch_size)):
        rows = text_only[start : start + max(1, batch_size)]
        enc = tokenizer(
            [template_text(prompts[i], as_written[i]) for i in rows],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=max_length,
        )
        lengths = enc["attention_mask"].sum(1).tolist()
        hidden_states = backbone(input_ids=enc["input_ids"].to(device), output_hidden_states=True, use_cache=False).hidden_states
        taps = _normalized_taps(hidden_states, dtype)
        del hidden_states
        for row, (i, n) in enumerate(zip(rows, lengths)):
            out[i] = taps[row, :n].clone()
        del taps

    image_token_id = model.config.image_token_id
    merge = image_processor.merge_size**2
    for i, imgs in enumerate(images):
        if out[i] is not None:
            continue
        pixels = image_processor(images=list(imgs), do_resize=False, return_tensors="pt")
        grid = pixels["image_grid_thw"]
        text = PROMPT_TEMPLATE.format(edit_text(prompts[i], len(imgs)))
        # each marker's single pad expands to its image's token count; no truncation, the pad runs must stay whole
        parts = text.split(IMAGE_PAD)
        counts = (grid.prod(-1) // merge).tolist()
        text = parts[0] + "".join(IMAGE_PAD * n + part for n, part in zip(counts, parts[1:]))
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        mm_token_type_ids = (input_ids == image_token_id).int()
        if int(mm_token_type_ids.sum()) != sum(counts):
            raise ValueError("FIBO 2.0: the image placeholder count does not match the image grids")
        hidden_states = backbone(
            input_ids=input_ids,
            pixel_values=pixels["pixel_values"].to(device, next(model.parameters()).dtype),
            image_grid_thw=grid.to(device),
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        out[i] = _normalized_taps(hidden_states, dtype)[0].clone()
        del hidden_states
    return out


def stack_contexts(contexts: list[Tensor], device, dtype) -> tuple[Tensor, Optional[Tensor]]:
    """Per-item (L_i, 15 * 2560) taps -> right-padded (B, S, 46080) bundles and the (B, S) bool mask of real tokens,
    None when nothing is padded."""
    length = max(c.shape[0] for c in contexts)
    stacked = torch.stack([F.pad(expand_taps(c.to(device=device, dtype=dtype)), (0, 0, 0, length - c.shape[0])) for c in contexts])
    if all(c.shape[0] == length for c in contexts):
        return stacked, None
    mask = torch.stack([torch.arange(length, device=device) < c.shape[0] for c in contexts])
    return stacked, mask


# ---------------------------------------------------------------------------------------------------------- references
def sample_reference_sizes(images: Sequence[Image.Image], width: int, height: int) -> list[tuple[int, int]]:
    """ComfyUI's TextEncodeFibo2 (and Bria's pipeline): a single reference is cropped to the result's size, several
    references each to the ~1 MP size Fibo 2.0 was trained at for their own aspect ratio."""
    if len(images) == 1:
        return [(width, height)]
    return [trained_size(img.width / img.height) for img in images]


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scaled to cover (width, height) with Lanczos (sizes rounded up) and center cropped."""
    if (image.width, image.height) == (width, height):
        return image
    scale = max(height / image.height, width / image.width)
    resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
    image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)
    left, top = (resized_width - width) // 2, (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def vlm_image(crop: Image.Image) -> Image.Image:
    """Qwen3-VL's copy of a reference crop: thumbnailed to at most 768 px a side, then on the 32 px grid (pixel budget
    clamped like the processor's smart resize)."""
    image = crop.copy()
    image.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
    h, w = image.height, image.width
    height, width = round(h / VLM_GRID) * VLM_GRID, round(w / VLM_GRID) * VLM_GRID
    if height * width > VLM_MAX_PIXELS:
        beta = math.sqrt(h * w / VLM_MAX_PIXELS)
        height = max(VLM_GRID, math.floor(h / beta / VLM_GRID) * VLM_GRID)
        width = max(VLM_GRID, math.floor(w / beta / VLM_GRID) * VLM_GRID)
    elif height * width < VLM_MIN_PIXELS:
        beta = math.sqrt(VLM_MIN_PIXELS / (h * w))
        height = math.ceil(h * beta / VLM_GRID) * VLM_GRID
        width = math.ceil(w * beta / VLM_GRID) * VLM_GRID
    if (width, height) != (w, h):
        image = image.resize((width, height), resample=Image.BICUBIC)
    return image


def to_pil(image) -> Image.Image:
    """An (H, W, C) uint8 array (a cached dataset image) or a PIL image -> RGB PIL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def pil_to_tensor(image: Image.Image) -> Tensor:
    """RGB PIL -> (1, 3, H, W) in [-1, 1], the VAE's input."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


# ----------------------------------------------------------------------------------------------------------- sampling
def timeshift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def comfy_sigmas(num_steps: int, shift: float = DEFAULT_SHIFT, total: int = 1000) -> Tensor:
    """ComfyUI's ``simple`` scheduler over ModelSamplingFlux's table, the terminal 0 appended: step x at the table's
    entry t = (total - int(x * total / steps)) / total, sigma = timeshift(shift, t) (shift = e^mu, ComfyUI's mu 1.15)."""
    ss = total / num_steps
    t = torch.tensor([(total - int(x * ss)) / total for x in range(num_steps)], dtype=torch.float64)
    sigmas = timeshift(float(shift), t).float()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def denoise(
    model: fibo2_model.Fibo2,
    noise: Tensor,
    ctx: Tensor,
    ctx_mask: Optional[Tensor],
    neg_ctx: Optional[Tensor],
    neg_mask: Optional[Tensor],
    num_steps: int,
    guidance: float,
    shift: float = DEFAULT_SHIFT,
    ref_latents: Optional[list[Tensor]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Euler from fp32 noise (B, 128, h, w) to clean latents; CFG runs the negative as its own forward (cond and uncond
    of different lengths run separately in ComfyUI too, the gist count follows the text length). References condition
    both passes."""
    latents = noise.float()
    do_cfg = neg_ctx is not None and guidance != 1.0
    sigmas = comfy_sigmas(num_steps, shift).to(latents.device)
    for index in tqdm(range(num_steps), desc="Sampling"):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        timestep = (sigma * fibo2_model.NUM_TRAIN_TIMESTEPS).reshape(1).expand(latents.shape[0])
        x = latents.to(dtype)
        pred = model(x, timestep, ctx, ctx_mask, ref_latents).float()
        if do_cfg:
            pred_uncond = model(x, timestep, neg_ctx, neg_mask, ref_latents).float()
            pred = pred_uncond + guidance * (pred - pred_uncond)
        latents = latents + (sigma_next - sigma) * pred
    return latents
