"""MOSS-Image loading, text conditioning and sampling.

* DiT: the ComfyUI single files of Comfy-Org/MOSS-Image (``moss_image_base_bf16`` / ``moss_image_sft_bf16``, the
  vendor's keys with each block's q / k / v fused into ``proj.qkv``) or the vendor's split layout; both load into
  ``moss_model.MossImage`` with q / k / v split. The INT8 ConvRot repacks are inference files.
* VAE: the FLUX.2 VAE (``flux2_utils.load_ae``): MOSS-Image's latents are its packed, BatchNorm-normalized 128 channels
  at 1/16, so FLUX.2's latent caching is reused.
* Text: Qwen3.5-9B's language tower (``Qwen3_5TextModel``, the vendor's text encoder without the vision tower: text-only
  inputs give the same states), the final-normed last hidden state (L, 4096) of the chat template with thinking on and
  the generation prompt appended (an open ``<think>`` block), one unpadded forward per prompt as the vendor (padding
  would leak into the gated delta-net recurrence). Read from the ComfyUI file: Comfy-Org/MOSS-Image's
  ``qwen3.5_9b_int8_convrot`` (dequantized to BF16 on load, the file SwarmUI uses) or Comfy-Org/Qwen3.5's BF16 file.
  Matches ComfyUI's ``MOSSImageTokenizer`` / ``qwen35`` text model, where the trained LoRAs are used.
* Flow: sigma 1 = noise, the DiT predicts ``noise - clean``; the vendor sampler shifts a uniform grid by 5 and snaps it
  to the 1000 training sigmas (50 steps, CFG 3 whose negative branch skips block 22, empty negative prompt).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from musubi_tuner.flux_2.flux2_utils import load_ae  # noqa: F401  (the FLUX.2 VAE is MOSS-Image's VAE)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
from musubi_tuner.modules.int8_convrot_utils import dequantize_int8_convrot_weight, parse_comfy_quant_tensor
from musubi_tuner.moss_image import moss_model
from musubi_tuner.utils.model_utils import cpu_bf16_is_slow, keep_weights_in_cpu_ram
from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# pinned (Guardrail 6): the vendor tokenizer (its chat template equals Qwen3.5-9B's)
MOSS_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_TOKENIZER_REVISION = "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"

# Qwen/Qwen3.5-9B config.json "text_config" (revision c202236235762e1c871ad0ccb60c8ee5ba337b9a)
QWEN35_9B_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "use_cache": False,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}

MAX_TEXT_TOKENS = 4096
DEFAULT_SHIFT = 5.0  # the vendor's sampler shift (ComfyUI's model_sampling); ai-toolkit also trains at it, the GUI presets
# train at --discrete_flow_shift 3 (about the pre-training distribution, one save cleaner on two seeds)
DEFAULT_GUIDANCE = 3.0
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_UNCOND_SKIP_LAYERS = (22,)


# ---------------------------------------------------------------------------------------------------------------- DiT
def check_dit_checkpoint(dit_path: str) -> None:
    """Refuse files that cannot train as the MOSS-Image DiT, with the reason."""
    with safe_open(dit_path, framework="pt") as handle:
        keys = list(handle.keys())
    if "shared_block_modulation.weight" not in keys or not any(k.startswith("layers.0.proj.") for k in keys):
        raise ValueError(f"{dit_path} is not a MOSS-Image DiT checkpoint (no shared_block_modulation / layers.N.proj)")
    if any(k.endswith((".comfy_quant", ".weight_scale", ".input_scale")) for k in keys):
        raise ValueError(
            f"{dit_path} is a pre-quantized MOSS-Image file (INT8 ConvRot), an inference file. Train on the BF16 file "
            "(moss_image_base_bf16.safetensors or moss_image_sft_bf16.safetensors) and use FP8 Scaled for lower memory."
        )


def load_dit(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    disable_numpy_memmap: bool = False,
) -> moss_model.MossImage:
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)
    check_dit_checkpoint(dit_path)
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = moss_model.MossImage(moss_model.MossImageParams(), attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading MOSS-Image DiT from {dit_path}, device={loading_device}")
    # q / k / v are split before any FP8 scaling, so every projection gets its own scales
    sd = load_safetensors(dit_path, device="cpu", disable_mmap=True, dtype=dit_weight_dtype, disable_numpy_memmap=disable_numpy_memmap)
    moss_model.split_fused_qkv(sd)
    if fp8_scaled:
        sd = optimize_state_dict_with_fp8(
            sd,
            device,
            moss_model.FP8_OPTIMIZATION_TARGET_KEYS,
            moss_model.FP8_OPTIMIZATION_EXCLUDE_KEYS,
            move_to_device=(loading_device == device),
        )
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
    if loading_device.type != "cpu":
        for key in sd.keys():
            sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded MOSS-Image: {info}")
    return model


# ------------------------------------------------------------------------------------------------------- text encoder
_TE_PREFIXES = ("model.language_model.", "language_model.", "model.")
_TE_DROPPED = ("lm_head.", "mtp.", "model.visual.", "visual.")
# elements per dequantized chunk: about 1 GiB of FP32 temporaries; the 248320 x 4096 token table dequantized whole held
# 10.4 GiB on the GPU, so text caching for the 6 GB preset (GPU streaming on a CPU slow in BF16) peaked at 10.8 GiB
_DEQUANT_CHUNK_ELEMENTS = 64 * 2**20


def _dequantize_rows(q: Tensor, scale: Tensor, quant: dict, dtype: torch.dtype, device, chunk_elements: int = _DEQUANT_CHUNK_ELEMENTS) -> Tensor:
    """An INT8 (ConvRot) weight dequantized on ``device`` in row chunks into a CPU tensor of ``dtype``. Rows are
    independent: the scale is per row (or one for the tensor) and the ConvRot rotation groups each row's inputs."""
    if q.dim() != 2:
        raise ValueError(f"INT8 weight of shape {tuple(q.shape)}: expected 2-D")
    per_row = scale.numel() > 1
    if per_row and scale.shape[0] != q.shape[0]:
        raise ValueError(f"INT8 weight {tuple(q.shape)} with scale {tuple(scale.shape)}: expected one scale per row")
    out = torch.empty(q.shape, dtype=dtype)
    step = max(1, chunk_elements // q.shape[1])
    for start in range(0, q.shape[0], step):
        rows = slice(start, start + step)
        chunk_q = q[rows].to(device)
        chunk_scale = (scale[rows] if per_row else scale).to(device)
        if quant.get("convrot"):
            chunk = dequantize_int8_convrot_weight(chunk_q, chunk_scale, int(quant["convrot_groupsize"]), dtype=dtype)
        else:
            chunk = (chunk_q.float() * chunk_scale.float()).to(dtype)
        out[rows] = chunk.cpu()
    return out


def _text_encoder_state_dict(path: str, dtype: torch.dtype, device) -> dict[str, Tensor]:
    """The language tower's weights of a ComfyUI Qwen3.5 file in ``dtype``: INT8 ConvRot tensors (``comfy_quant``)
    dequantized, the LM head, MTP head and vision tower dropped."""
    out = {}
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        quantized = {k[: -len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
        for key in keys:
            if key.startswith(_TE_DROPPED) or key.endswith((".comfy_quant", ".weight_scale")):
                continue
            base = key[: -len(".weight")] if key.endswith(".weight") else None
            if base is not None and base in quantized:
                quant = parse_comfy_quant_tensor(handle.get_tensor(f"{base}.comfy_quant"))
                if quant.get("format") != "int8_tensorwise":
                    raise ValueError(f"{path}: unsupported quantization {quant} at {base} (use the BF16 or INT8 ConvRot file)")
                tensor = _dequantize_rows(handle.get_tensor(key), handle.get_tensor(f"{base}.weight_scale"), quant, dtype, device)
            else:
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)  # one dtype throughout, as the vendor's BF16 text encoder
            for prefix in _TE_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            out[key] = tensor
    return out


def load_text_encoder(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu", stream: bool = False):
    """Qwen3.5-9B's language tower from a ComfyUI file and the pinned vendor tokenizer: (model, tokenizer). ``stream``
    (CUDA only) keeps the tower in CPU RAM and runs it on ``device`` one decoder layer at a time (stream_text_encoder)."""
    from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

    device, dtype, stream = text_encoder_plan(device, dtype, stream)
    config = Qwen3_5TextConfig(**QWEN35_9B_TEXT_CONFIG)
    with init_empty_weights():
        model = Qwen3_5TextModel(config)
    # dequantizing on the GPU is ~10x faster than on the CPU; the result goes to the CPU first either way
    dequant_device = torch.device("cuda") if torch.cuda.is_available() and torch.device(device).type == "cuda" else torch.device("cpu")
    logger.info(f"Loading MOSS-Image text encoder (Qwen3.5-9B) from {path}")
    sd = _text_encoder_state_dict(path, dtype, dequant_device)
    info = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in info.missing_keys if "rotary_emb" not in k]
    if missing or info.unexpected_keys:
        raise RuntimeError(f"Qwen3.5 text encoder checkpoint did not match: missing={missing[:10]}, unexpected={info.unexpected_keys[:10]}")
    # buffers the checkpoint does not hold (rotary inv_freq) are still on the meta device
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            raise RuntimeError(f"Qwen3.5 text encoder buffer {name} was not initialized")
    if stream and torch.device(device).type == "cuda":
        stream_text_encoder(model, device)
    else:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MOSS_TOKENIZER_REPO, subfolder="tokenizer", revision=MOSS_TOKENIZER_REVISION)
    return model.eval().requires_grad_(False), tokenizer


def text_encoder_plan(device: Union[str, torch.device], dtype: torch.dtype, stream: bool) -> tuple[torch.device, torch.dtype, bool]:
    """Where and in which dtype Qwen3.5-9B runs. BF16 on a CPU without fast BF16 (AVX2-only: 16 minutes per caption on
    a Broadwell Xeon) is replaced by low-VRAM streaming on the GPU when there is one (12 s per caption there, under
    1 GB of VRAM), else by FP32 on the CPU (22 s, about 36 GB of RAM)."""
    device = torch.device(device)
    if device.type != "cpu" or dtype != torch.bfloat16 or stream or not cpu_bf16_is_slow():
        return device, dtype, stream
    if torch.cuda.is_available():
        logger.warning("This CPU multiplies BF16 far slower than FP32: Qwen3.5-9B runs on the GPU one layer at a time instead (--low_vram_text_encoding)")
        return torch.device("cuda"), dtype, True
    logger.warning("This CPU multiplies BF16 far slower than FP32: Qwen3.5-9B is encoded in float32 on the CPU (about 36 GB of RAM)")
    return device, torch.float32, False


def stream_text_encoder(model, device: Union[str, torch.device]) -> None:
    """Low-VRAM text encoding: the Qwen3.5 tower stays in (pinned) CPU RAM and each decoder layer gets a GPU copy only
    while it runs; the embedding table stays on the CPU, where encode_prompts looks the tokens up. Under 1 GB of VRAM
    instead of about 18 GB, and fast on any CPU (the CPU path in BF16 took 16 minutes per caption on a Broadwell Xeon
    without BF16 support); each prompt copies the 17 GB of layer weights to the GPU once."""
    device = torch.device(device)
    for layer in model.layers:
        for buffer in layer.buffers():
            buffer.data = buffer.data.to(device)
        keep_weights_in_cpu_ram(layer, list(layer.parameters()), device)
    model.norm.to(device)
    model.rotary_emb.to(device)
    model.embed_tokens.to("cpu")
    model._moss_stream_device = device
    if device.type == "cuda":
        torch.cuda.empty_cache()


def chat_text(tokenizer, prompt: str) -> str:
    """The vendor's conditioning text: the chat template with thinking on and the generation prompt (an open
    ``<think>`` block); the template trims the user turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def tokenize_prompt(tokenizer, prompt: str, max_tokens: int = MAX_TEXT_TOKENS) -> Tensor:
    return tokenizer(chat_text(tokenizer, prompt), return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"][0]


TEXT_BATCH_SIZE = 8  # prompts per padded forward


@torch.no_grad()
def encode_prompts(
    model, tokenizer, prompts: list[str], max_tokens: int = MAX_TEXT_TOKENS, batch_size: int = TEXT_BATCH_SIZE
) -> list[Tensor]:
    """One (L_i, 4096) final hidden state per prompt, on the model's device and dtype.

    Up to ``batch_size`` prompts run as one right-padded forward without an attention mask, as ai-toolkit (0473387)
    does: every Qwen3.5 layer is causal (full attention under the causal mask, Gated DeltaNet and its short
    convolution), so the pads after a prompt never reach its tokens, which keep their rotary positions. Each prompt's
    states are cut back to its own length. With --low_vram_text_encoding a forward copies the 17 GB of layer weights to
    the GPU once, so 8 captions took 8 s instead of 63 s on the real model (vram_lab/moss_te_batch_check.py).
    """
    stream_device = getattr(model, "_moss_stream_device", None)
    device = stream_device or next(model.parameters()).device
    pad = getattr(tokenizer, "pad_token_id", None) or 0
    out = []
    for start in range(0, len(prompts), max(1, batch_size)):
        ids = [tokenize_prompt(tokenizer, prompt, max_tokens) for prompt in prompts[start : start + max(1, batch_size)]]
        input_ids = torch.full((len(ids), max(len(tokens) for tokens in ids)), pad, dtype=torch.long)
        for row, tokens in enumerate(ids):
            input_ids[row, : len(tokens)] = tokens
        if stream_device is not None:  # the embedding table stays on the CPU (stream_text_encoder)
            embeds = model.embed_tokens(input_ids).to(stream_device)
            hidden = model(inputs_embeds=embeds, use_cache=False).last_hidden_state
        else:
            hidden = model(input_ids=input_ids.to(device), use_cache=False).last_hidden_state
        out.extend(hidden[row, : len(tokens)].clone() for row, tokens in enumerate(ids))
        del hidden
    return out


def pad_contexts(contexts: list[Tensor], device, dtype) -> tuple[Tensor, Tensor]:
    """Per-item (L_i, D) contexts -> right-padded (B, S, D) and the (B, S) bool mask of real tokens."""
    length = max(c.shape[0] for c in contexts)
    stacked = torch.stack([F.pad(c.to(device=device, dtype=dtype), (0, 0, 0, length - c.shape[0])) for c in contexts])
    mask = torch.stack([torch.arange(length, device=device) < c.shape[0] for c in contexts])
    return stacked, mask


# ----------------------------------------------------------------------------------------------------------- sampling
def moss_sigmas(num_steps: int, shift: float = DEFAULT_SHIFT, training_shift: float = 1.0, total: int = 1000) -> Tensor:
    """The vendor scheduler's sigmas with the terminal 0 appended: a shifted grid snapped to the training grid,
    consecutive duplicates removed (so fewer than num_steps are possible)."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def denoise(
    model: moss_model.MossImage,
    noise: Tensor,
    ctx: Tensor,
    ctx_mask: Tensor,
    neg_ctx: Optional[Tensor],
    neg_mask: Optional[Tensor],
    num_steps: int,
    guidance: float,
    shift: float = DEFAULT_SHIFT,
    uncond_skip_layers: Sequence[int] = DEFAULT_UNCOND_SKIP_LAYERS,
) -> Tensor:
    """The vendor's sampler: fp32 Euler on fp32 noise (B, 128, h, w); CFG combines fp32 predictions, its negative
    branch skipping ``uncond_skip_layers``."""
    latents = noise.float()
    do_cfg = neg_ctx is not None and guidance != 1.0
    sigmas = moss_sigmas(num_steps, shift=shift).to(latents.device)
    for index in tqdm(range(len(sigmas) - 1), desc="Sampling"):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        timestep = (sigma * moss_model.NUM_TRAIN_TIMESTEPS).reshape(1).expand(latents.shape[0])
        pred = model(latents, timestep, ctx, ctx_mask).float()
        if do_cfg:
            pred_uncond = model(latents, timestep, neg_ctx, neg_mask, skip_layers=uncond_skip_layers).float()
            pred = pred_uncond + guidance * (pred - pred_uncond)
        latents = latents + (sigma_next - sigma) * pred
    return latents
