"""FLUX.3 [klein] loading, text conditioning, latent packing and sampling.

* DiT: BFL's single-file base checkpoint (``flux3-image-klein-base-bf16.safetensors`` of
  ``black-forest-labs/flux-3-image-klein-comfy``, the same bytes as ``model.safetensors`` of
  ``black-forest-labs/flux-3-image-klein-base``). The step-distilled release (``flux3-image-klein-*`` without ``base``,
  ``ref_time_in`` keys) is an inference model: LoRAs are trained on the base and applied to either.
* VAE: the FLUX.2 VAE, byte-identical to the file FLUX.2 / Klein training already uses (``flux2_utils.load_ae``); its
  packed, BatchNorm-normalized 128-channel latents at 1/16 are FLUX.3's latents, so FLUX.2's latent caching is reused.
* Text: Qwen3-VL-4B-Instruct (the ComfyUI single file shared with Krea 2). The prompt goes through the instruct chat
  template without a think block, is right-padded with ``<|endoftext|>`` to a multiple of 80 tokens, and the pad tokens
  sit at rotary position 1 (transformers 4.57, which BFL's reference pins); hidden layers 4, 8, ..., 32 are stacked
  per token into a (L, 20480) context. The DiT has no text mask, so the pad tokens' states are part of the condition.
  This matches ComfyUI's ``Flux3Tokenizer`` / ``Flux3Qwen3VL`` exactly, where the trained LoRAs are used.
* Flow: t = 1 is noise, the DiT predicts ``noise - clean``; the base sampler shifts a uniform grid with
  ``timeshift(6.93, t)`` (50 steps, guidance 3.5), and training samples timesteps with the same shift.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from musubi_tuner.flux_2.flux2_utils import load_ae  # noqa: F401  (the FLUX.2 VAE is FLUX.3's VAE)
from musubi_tuner.flux_3 import flux3_model
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Qwen3-VL-4B-Instruct tokenizer assets, pinned (Guardrail 6; the revision Mage-Flow uses)
QWEN3_VL_4B_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_4B_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TEXT_PAD_TOKEN_ID = 151643  # <|endoftext|>
TEXT_PAD_MULTIPLE = 80
TEXT_MAX_TOKENS = 8192
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_SHIFT = 6.93  # base sampler alpha; also the training timestep shift
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_GUIDANCE = 3.5
# the step-distilled inference model (where trained LoRAs are mostly used): 16 steps, CFG 1, shift 1.93586


# ---------------------------------------------------------------------------------------------------------------- DiT
def check_base_checkpoint(dit_path: str) -> None:
    """Refuse checkpoints that cannot be trained as FLUX.3 [klein] base, with the reason."""
    with safe_open(dit_path, framework="pt") as handle:
        keys = list(handle.keys())
    if not any(k.startswith("content_mode_blocks.") for k in keys):  # FLUX.1 / FLUX.2 files have single_blocks too
        raise ValueError(f"{dit_path} is not a FLUX.3 [klein] DiT checkpoint (no content_mode_blocks)")
    if any(k.startswith("ref_time_in.") for k in keys):
        raise ValueError(
            f"{dit_path} is the step-distilled FLUX.3 [klein] (flow map, 'ref_time_in' weights), an inference model. "
            "Train on the base model (flux3-image-klein-base-bf16.safetensors); the LoRA works with both."
        )
    if any(k.endswith((".comfy_quant", ".weight_scale", ".input_scale")) or ".qkv." in k for k in keys):
        raise ValueError(
            f"{dit_path} is a pre-quantized FLUX.3 repack (INT8 / NVFP4). Train on the BF16 base checkpoint "
            "(flux3-image-klein-base-bf16.safetensors) and use --fp8_base / --fp8_scaled for lower memory."
        )


def load_flow_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    lora_weights_list: Optional[list[dict[str, Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    disable_numpy_memmap: bool = False,
) -> flux3_model.Flux3:
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)
    check_base_checkpoint(dit_path)
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = flux3_model.Flux3(flux3_model.Flux3Params(), attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading FLUX.3 DiT from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=flux3_model.FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=flux3_model.FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded FLUX.3: {info}")
    return model


# ------------------------------------------------------------------------------------------------------- text encoder
def load_text_encoder(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu", disable_mmap: bool = True):
    """Qwen3-VL-4B-Instruct from a local safetensors (ComfyUI or official layout) and the pinned tokenizer.
    Returns (model, tokenizer). The vision tower is dropped: FLUX.3 conditions on text only."""
    from transformers import AutoTokenizer

    from musubi_tuner.krea2.krea2_encoder import _load_qwen3_vl_model

    model = _load_qwen3_vl_model(path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    model.model.visual = None  # never called without pixel values
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_4B_REPO, revision=QWEN3_VL_4B_REVISION)
    return model.eval().requires_grad_(False), tokenizer


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int = TEXT_MAX_TOKENS, pad_multiple: int = TEXT_PAD_MULTIPLE):
    """(input_ids, attention_mask, position_ids) of one prompt, each (L,): the chat template, right padding with
    <|endoftext|> to a multiple of ``pad_multiple`` (capped at ``max_tokens``), pad tokens at rotary position 1."""
    ids = tokenizer(PROMPT_TEMPLATE.format(prompt), add_special_tokens=False)["input_ids"][:max_tokens]
    length = min(max(pad_multiple, math.ceil(len(ids) / pad_multiple) * pad_multiple), max_tokens)
    input_ids = torch.full((length,), TEXT_PAD_TOKEN_ID, dtype=torch.long)
    input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    mask = torch.zeros(length, dtype=torch.long)
    mask[: len(ids)] = 1
    position_ids = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    return input_ids, mask, position_ids


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts: list[str], max_tokens: int = TEXT_MAX_TOKENS) -> list[Tensor]:
    """One (L_i, 20480) context per prompt (L_i its own multiple of 80), in the text encoder's dtype, on its device.
    Prompts of equal padded length run as one batch."""
    device = next(model.parameters()).device
    rows = [tokenize_prompt(tokenizer, p, max_tokens) for p in prompts]
    out: list[Optional[Tensor]] = [None] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for i, (input_ids, _, _) in enumerate(rows):
        by_length.setdefault(input_ids.shape[0], []).append(i)
    for indices in by_length.values():
        input_ids = torch.stack([rows[i][0] for i in indices]).to(device)
        mask = torch.stack([rows[i][1] for i in indices]).to(device)
        position_ids = torch.stack([rows[i][2] for i in indices]).to(device)
        # the language model only: the LM head's logits (152k per token) are not needed
        states = model.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, output_hidden_states=True, use_cache=False)
        stacked = torch.stack([states.hidden_states[k] for k in TEXT_LAYERS], dim=2)  # (B, L, 8, 2560)
        ctx = stacked.flatten(2)  # (B, L, 20480): per token, layer 4's features first
        for row, i in enumerate(indices):
            out[i] = ctx[row]
    return out


def pad_contexts(contexts: list[Tensor], device, dtype) -> Tensor:
    """(B, L_max, 20480): contexts of different lengths zero-padded at the end (cached captions are encoded one by one;
    a batch of one, the usual case, is unchanged)."""
    length = max(c.shape[0] for c in contexts)
    out = torch.zeros(len(contexts), length, contexts[0].shape[-1], device=device, dtype=dtype)
    for i, c in enumerate(contexts):
        out[i, : c.shape[0]] = c.to(device=device, dtype=dtype)
    return out


# ------------------------------------------------------------------------------------------------------------ latents
def pack_latents(latents: Tensor) -> Tensor:
    """(B, 128, h, w) -> (B, h * w, 128) tokens, row-major."""
    return latents.flatten(2).transpose(1, 2)


def unpack_latents(tokens: Tensor, height: int, width: int) -> Tensor:
    """(B, h * w, 128) -> (B, 128, h, w)."""
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)


# ----------------------------------------------------------------------------------------------------------- sampling
def timeshift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def get_schedule(num_steps: int, alpha: float = DEFAULT_SHIFT) -> Tensor:
    """num_steps + 1 times from 1 (noise) to 0 (clean)."""
    return timeshift(alpha, torch.linspace(1.0, 0.0, num_steps + 1))


@torch.no_grad()
def denoise(
    model: flux3_model.Flux3,
    noise: Tensor,
    ctx: Tensor,
    neg_ctx: Optional[Tensor],
    num_steps: int,
    guidance: float,
    alpha: float = DEFAULT_SHIFT,
    image_cond: Optional[Tensor] = None,
    image_cond_ids: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Euler integration of the flow from noise (B, 128, h, w) to clean latents. Classifier-free guidance runs the
    unconditional context as its own forward (no zero padding between prompts of different lengths)."""
    batch, _, height, width = noise.shape
    device = noise.device
    x = pack_latents(noise.float())
    ids = flux3_model.image_ids(batch, height, width, 0, device)
    do_cfg = neg_ctx is not None and guidance > 1.0
    timesteps = get_schedule(num_steps, alpha).to(device)
    for t_curr, t_prev in zip(tqdm(timesteps[:-1], desc="Sampling"), timesteps[1:]):
        t_vec = torch.full((batch,), float(t_curr), device=device, dtype=torch.float32)
        v = model(ctx, x.to(dtype), ids, t_vec, image_cond=image_cond, image_cond_ids=image_cond_ids).float()
        if do_cfg:
            v_uncond = model(neg_ctx, x.to(dtype), ids, t_vec, image_cond=image_cond, image_cond_ids=image_cond_ids).float()
            v = v_uncond + guidance * (v - v_uncond)
        x = x + (t_prev - t_curr) * v
    return unpack_latents(x, height, width)
