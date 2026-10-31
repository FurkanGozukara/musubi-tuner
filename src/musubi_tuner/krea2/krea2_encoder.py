"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.
"""

import logging
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL-4B from the vendored config and load weights from a local safetensors."""
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)

    info = model.load_state_dict(sd, strict=False, assign=True)
    # Qwen3-VL-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so the
    # checkpoint omits lm_head.weight; re-tie after loading to materialize it.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)
    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    conditioner.tokenizer_repo = tokenizer_repo  # the image processor of edit prompts comes from the same repo
    return conditioner.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5
        self.tokenizer_repo = QWEN3_VL_4B_INSTRUCT_REPO_ID

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            # The suffix follows the padding, so its rotary positions must continue from the last prompt token.
            # transformers 4.x derived the positions from the attention mask for text-only input; 5.x uses the
            # sequence index. Pass them explicitly (padded positions repeat the previous one, keeping them monotonic).
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    @torch.no_grad()
    def encode_edit(self, prompt: str, images: list) -> Tensor:
        """One edit prompt with its reference images, as ComfyUI's Text Encode Qwen Image Edit Plus node feeds the Krea 2
        text encoder (the Krea-2 Turbo "Image Style Reference" template): Qwen-Image-Edit's system prompt, then
        ``Picture N: <vision>`` per reference before the prompt, each reference area-resized to about 384 x 384 for the
        vision tower (``edit_vision_pixels``), and Krea2TEModel's cut after ``<|im_start|>user\\n``. Returns the selected
        hidden states of the remaining tokens, vision tokens included: (L, num_select_layers, hidden)."""
        if not images:
            raise ValueError("encode_edit needs at least one reference image")
        processor = self._vl_processor()
        pictures = [edit_vision_pixels(image) for image in images]
        text = EDIT_TEMPLATE.format(
            "".join(f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>" for index in range(len(pictures))) + prompt
        )
        inputs = processor(text=[text], images=[picture.permute(2, 0, 1) for picture in pictures], return_tensors="pt", do_rescale=False)
        # the vision tower gets ComfyUI's own patches (bilinear to the 32-pixel grid, mean / std 0.5) for each reference
        patches, grids = zip(*(comfy_qwen3vl_patches(picture) for picture in pictures))
        grids = torch.cat(grids, dim=0)
        if not torch.equal(grids, inputs["image_grid_thw"].cpu()):
            raise RuntimeError(f"Qwen3-VL processor grid {inputs['image_grid_thw'].tolist()} differs from ComfyUI's {grids.tolist()}")
        device = self.qwen.device
        dtype = next(self.qwen.parameters()).dtype
        extra = {key: value.to(device) for key, value in inputs.items() if key not in ("input_ids", "attention_mask", "pixel_values")}
        input_ids = inputs["input_ids"].to(device)
        # DeepStack adds visual features at the image positions after each of the first decoder layers. ComfyUI taps a
        # layer's input (after that addition); transformers records the layer outputs before it. Keep ComfyUI's taps.
        captured = {}

        def keep_deepstack(_module, _args, kwargs):
            captured["masks"] = kwargs.get("visual_pos_masks")
            captured["embeds"] = kwargs.get("deepstack_visual_embeds")

        language_model = self.qwen.model.language_model
        handle = language_model.register_forward_pre_hook(keep_deepstack, with_kwargs=True)
        try:
            states = self.qwen(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=torch.cat(patches, dim=0).to(device, dtype=dtype),
                output_hidden_states=True,
                **extra,
            )
        finally:
            handle.remove()
        layers = list(states.hidden_states)
        deepstack, masks = captured.get("embeds") or [], captured.get("masks")
        for index in set(self.select_layers):
            if 1 <= index <= len(deepstack) and masks is not None:
                layers[index] = layers[index].clone()
                layers[index][masks] = layers[index][masks] + deepstack[index - 1].to(layers[index])
        hiddens = torch.stack([layers[i] for i in self.select_layers], dim=2)[0]
        starts = (input_ids[0] == IM_START_TOKEN_ID).nonzero().flatten().tolist()
        if len(starts) < 2:
            raise RuntimeError("edit prompt template lost its user turn")
        return hiddens[starts[1] + 3 :]  # after "<|im_start|>", "user", "\n" (Krea2TEModel.encode_token_weights)

    def _vl_processor(self):
        if getattr(self, "_processor_vl", None) is None:
            from transformers import AutoProcessor

            self._processor_vl = AutoProcessor.from_pretrained(self.tokenizer_repo)
        return self._processor_vl


# ComfyUI comfy_extras/nodes_qwen.py TextEncodeQwenImageEditPlus: the template it passes for every text encoder it gets
EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the "
    "user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
IM_START_TOKEN_ID = 151644
EDIT_VISION_PIXELS = 384 * 384


def edit_vision_pixels(image) -> Tensor:
    """A reference image (PIL, or HWC uint8 / float array) as ComfyUI's edit node sends it to the vision tower:
    float32 (H, W, 3) in [0, 1], area-resized (adaptive average) to about 384 x 384 pixels, aspect kept."""
    import math

    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"reference image must be (H, W, C), got {array.shape}")
    tensor = torch.from_numpy(np.array(array[..., :3], copy=True)).float()  # a writable copy (PIL arrays are read-only)
    if array.dtype == np.uint8:
        tensor = tensor / 255.0
    height, width = tensor.shape[:2]
    scale = math.sqrt(EDIT_VISION_PIXELS / (width * height))
    size = (round(height * scale), round(width * scale))
    resized = torch.nn.functional.interpolate(tensor.permute(2, 0, 1)[None], size=size, mode="area")
    return resized[0].permute(1, 2, 0).contiguous()


def comfy_qwen3vl_patches(image: Tensor, patch_size: int = 16, merge_size: int = 2, temporal_patch_size: int = 2):
    """comfy.text_encoders.qwen_vl.process_qwen2vl_images as ComfyUI's Qwen3-VL calls it (patch 16, mean / std 0.5):
    (H, W, 3) in [0, 1] -> flattened patches and the (1, 3) grid."""
    import math

    factor = patch_size * merge_size
    height, width = image.shape[:2]
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    min_pixels, max_pixels = 3136, 12845056
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    pixels = torch.nn.functional.interpolate(image.permute(2, 0, 1)[None].float(), size=(h_bar, w_bar), mode="bilinear", align_corners=False)[0]
    pixels = (pixels - 0.5) / 0.5
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    channels = pixels.shape[0]
    pixels = pixels[None].repeat(temporal_patch_size, 1, 1, 1)
    patches = pixels.reshape(
        1, temporal_patch_size, channels, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
    return flat, torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
