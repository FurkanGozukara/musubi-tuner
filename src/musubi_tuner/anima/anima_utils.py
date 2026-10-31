"""Anima loading, text conditioning, latents and sampling.

* DiT: ``anima-base-v1.0.safetensors`` of circlestone-labs/Anima (``split_files/diffusion_models``; keys with a ``net.``
  prefix, the LLM adapter inside). The model card: "LoRAs should be trained using this version"; they also load on the
  Aesthetic and Turbo files (``model.diffusion_model.`` prefix), which this loader reads as well.
* Text encoder: Qwen3-0.6B base, ``split_files/text_encoders/qwen_3_06b_base.safetensors`` (``model.*`` keys), run by
  transformers' ``Qwen3Model``: its last hidden state after the final norm.
* Tokenizers (no weights): the Qwen tokenizer and the T5 tokenizer of circlestone-labs/Anima-Base-v1.0-Diffusers at a
  pinned revision (Guardrail 6). A caption becomes its Qwen ids without special tokens (an empty caption: one
  ``<|endoftext|>``, as ComfyUI pads it to its minimum length 1) and its T5 ids with the closing ``</s>`` (an empty
  caption: ``</s>`` alone). Nothing is padded per caption; ComfyUI's ``AnimaTokenizer`` / ``Qwen3_06BModel`` give the
  same ids and states (Guardrail 19: ``dont_upload/vram_lab/anima_te_check.py``). Both are cut at 512 tokens like
  kohya's and ai-toolkit's trainers (ComfyUI does not cut; captions that long are rare).
* INT8 ConvRot (LoRA only): ``--convrot_int8`` quantizes the BF16 file's block projections while loading (Hadamard groups
  of 256, row-wise scales), and the comfy_quant INT8 ConvRot files (ruwwww/Anima-ConvRot) are used as they are; both
  keep blocks 0, 1 and 27, the modulations, embedders, final layer and norms in BF16, as that file does. The patched
  Linears run the fused Triton INT8 GEMM (a dequantizing fallback without Triton), and DoRA reads their columns in the
  original basis (Guardrail 18).
* VAE: the Qwen-Image VAE (``split_files/vae/qwen_image_vae.safetensors``, Wan 2.1's latent statistics = ComfyUI's
  latent format ``Wan21``), 16 channels at 1/8; musubi's Qwen-Image VAE loader and its normalized latents.
* Flow: sigma 1 = noise, the DiT takes sigma (0..1) as its timestep and predicts ``noise - clean``.
* Samples: ComfyUI's default Anima workflow (the one in the model card's example.png): ``er_sde`` with the ``simple``
  scheduler on the model's shift 3, 30 steps, CFG 4, the initial noise drawn on the CPU from the seed and the solver's
  noise from a generator on the device seeded the same way, as ComfyUI's KSampler does.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from musubi_tuner.anima import anima_model
from musubi_tuner.modules.comfy_quant_utils import COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX
from musubi_tuner.modules.convrot_int8_utils import (
    CONVROT_GROUPSIZE,
    apply_convrot_int8_monkey_patch,
    canonicalize_convrot_int8_key,
    parse_comfy_quant_spec,
    quantize_weight_convrot,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOKENIZER_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
TOKENIZER_REVISION = "073c3a9db359c31ad0e8aa268d15775473c2176c"
QWEN_PAD_TOKEN_ID = 151643  # <|endoftext|>
MAX_QWEN_TOKENS = 512
MAX_T5_TOKENS = 512

DEFAULT_SHIFT = 3.0  # ComfyUI's model sampling for Anima (shift 3, multiplier 1)
DEFAULT_SAMPLE_STEPS = 30
DEFAULT_GUIDANCE = 4.0
DEFAULT_SAMPLER = "er_sde"
SAMPLERS = ("er_sde", "euler")
DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"

# Qwen3-0.6B base (the text_encoder config of the Diffusers release)
QWEN3_06B_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


# ---------------------------------------------------------------------------------------------------------------- DiT
def check_dit_checkpoint(dit_path: str, allow_int8: bool = False) -> bool:
    """Refuse files that are not an Anima DiT; name the recommended file for the inference fine-tunes. True for a
    comfy_quant INT8 ConvRot file, which only LoRA training takes (``allow_int8``: its INT8 layers are used as they are)."""
    with safe_open(dit_path, framework="pt") as handle:
        keys = list(handle.keys())
    bare = {k.split("net.", 1)[-1] if k.startswith("net.") else k.replace("model.diffusion_model.", "", 1) for k in keys}
    if "llm_adapter.out_proj.weight" not in bare or "blocks.0.self_attn.q_proj.weight" not in bare:
        raise ValueError(f"{dit_path} is not an Anima DiT checkpoint (no LLM adapter / DiT blocks)")
    int8 = False
    if any(k.endswith((COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX, ".scale_weight", ".input_scale")) for k in keys):
        if not allow_int8:
            raise ValueError(
                f"{dit_path} is a quantized Anima file. Fine-tuning trains the BF16 base (anima-base-v1.0.safetensors); "
                "LoRA training takes the INT8 ConvRot files too."
            )
        if any(k.endswith((".input_scale", ".scale_weight")) for k in keys) or not any(k.endswith(COMFY_QUANT_SUFFIX) for k in keys):
            raise ValueError(f"{dit_path} is a quantized Anima file of another format: only comfy_quant INT8 ConvRot files train")
        int8 = True
    name = os.path.basename(dit_path).lower()
    if "turbo" in name or "aesthetic" in name:
        logger.warning(
            "Anima: %s is an inference fine-tune (Turbo is distilled for CFG 1). The model card recommends training on "
            "Anima-Base (anima-base-v1.0.safetensors); a LoRA trained there works on Aesthetic and Turbo too.",
            os.path.basename(dit_path),
        )
    return int8


def is_convrot_int8_target(key: str, num_blocks: int = 28, all_blocks: bool = False) -> bool:
    """The Linears ``--convrot_int8`` quantizes, as ruwwww/Anima-ConvRot did: the attention and MLP projections of
    blocks 2 to num_blocks - 2; blocks 0, 1 and the last, the modulations, the adapter and the rest stay BF16.
    ``all_blocks`` (block swap): every block's projections, so all blocks share one layout (the offloaders swap
    tensors between blocks: mixed BF16 / INT8 blocks failed with "Both operands must be same dtype" and "swap-tensor
    shape/dtype differs from the streaming template")."""
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not key.endswith(".weight") or not parts[1].isdigit():
        return False
    index = int(parts[1])
    if not all_blocks and (index < 2 or index > num_blocks - 2):
        return False
    return parts[2] in ("self_attn", "cross_attn", "mlp") and parts[3] in ("q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2")


def convrot_int8_state_dict(
    sd: dict,
    device: torch.device,
    dtype: Optional[torch.dtype],
    quantize: bool,
    keep_on_device: bool,
    num_blocks: int = 28,
    all_blocks: bool = False,
) -> dict:
    """In place: ``sd`` (bare keys) in musubi's ConvRot INT8 layout, each INT8 ``.weight`` in the rotated basis with its
    FP32 [N, 1] ``.scale_weight``; returns each INT8 layer's group size for ``apply_convrot_int8_monkey_patch``. An INT8
    ConvRot file's layers are kept as they are (``.weight_scale`` renamed, the group size from ``.comfy_quant``);
    ``quantize`` turns the BF16 targets (``is_convrot_int8_target``) into ConvRot INT8 on ``device``. Every other
    floating-point tensor gets ``dtype``; ``keep_on_device`` keeps the quantized layers on ``device`` (no block swap)."""
    groupsizes = {}
    for key in [k for k in sd if k.endswith(COMFY_QUANT_SUFFIX)]:
        spec = parse_comfy_quant_spec(key, sd.pop(key))
        groupsizes[key[: -len(COMFY_QUANT_SUFFIX)]] = spec["convrot_groupsize"]
    for key in [k for k in sd if k.endswith(COMFY_WEIGHT_SCALE_SUFFIX)]:
        sd[canonicalize_convrot_int8_key(key)] = sd.pop(key).float()
    for module, groupsize in groupsizes.items():
        weight, scale = sd.get(module + ".weight"), sd.get(module + ".scale_weight")
        if weight is None or scale is None or weight.dtype != torch.int8 or tuple(scale.shape) != (weight.shape[0], 1):
            raise ValueError(f"Anima INT8 layer {module}: expected an int8 weight with an FP32 [rows, 1] weight_scale")
        if weight.shape[1] % groupsize:
            raise ValueError(f"Anima INT8 layer {module}: {weight.shape[1]} inputs do not divide into ConvRot groups of {groupsize}")
    target = device if keep_on_device else torch.device("cpu")
    for key in list(sd):
        value = sd[key]
        module = key[: -len(".weight")] if key.endswith(".weight") else None
        if module in groupsizes or key.endswith(".scale_weight"):
            continue
        if module is not None and value.dtype.itemsize == 1:
            raise ValueError(f"Anima DiT weight {key} is {value.dtype} without an INT8 ConvRot spec")
        if quantize and is_convrot_int8_target(key, num_blocks, all_blocks):
            result = quantize_weight_convrot(key, value.to(device), (CONVROT_GROUPSIZE,))
            if result is not None:
                weight, scale, groupsize = result
                sd[key], sd[module + ".scale_weight"] = weight.to(target), scale.to(target)
                groupsizes[module] = groupsize
                continue
        if dtype is not None and value.is_floating_point():
            sd[key] = value.to(dtype)
    return groupsizes


def load_dit(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[list[dict[str, Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    disable_numpy_memmap: bool = False,
    params: Optional[anima_model.AnimaParams] = None,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    allow_int8_file: bool = False,
    convrot_int8_all_blocks: bool = False,
) -> anima_model.Anima:
    """The Anima DiT. ``convrot_int8`` quantizes the block projections to ConvRot INT8 while loading; ``allow_int8_file``
    (LoRA training) takes a comfy_quant INT8 ConvRot file, whose INT8 layers are used as they are. INT8 models are
    frozen: INT8 weights cannot require gradients. ``convrot_int8_all_blocks`` (block swap) quantizes every block's
    projections, a file's BF16 blocks too, so the swapped and resident blocks share one layout."""
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)
    int8_file = check_dit_checkpoint(dit_path, allow_int8=allow_int8_file)
    use_int8 = convrot_int8 or int8_file
    if use_int8 and (fp8_scaled or dit_weight_dtype.itemsize == 1):
        raise ValueError("Anima: ConvRot INT8 and FP8 are two quantizations of the same weights; choose one (FP8 Base / Scaled off)")
    device = torch.device(device)
    loading_device = torch.device(loading_device)
    params = params or anima_model.AnimaParams()

    with init_empty_weights():
        model = anima_model.Anima(params, attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    if use_int8:
        logger.info(f"Loading Anima DiT from {dit_path}, device={loading_device} (ConvRot INT8)")
        sd = load_safetensors(dit_path, device="cpu", disable_mmap=True, dtype=None, disable_numpy_memmap=disable_numpy_memmap)
        anima_model.strip_checkpoint_prefix(sd)
        groupsizes = convrot_int8_state_dict(
            sd,
            device,
            dit_weight_dtype,
            quantize=convrot_int8 or convrot_int8_all_blocks,
            keep_on_device=loading_device == device,
            num_blocks=params.num_blocks,
            all_blocks=convrot_int8_all_blocks,
        )
        apply_convrot_int8_monkey_patch(model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=groupsizes)
        # load_state_dict(assign=True) gives the loaded tensors the placeholders' requires_grad, and INT8 cannot have it
        model.requires_grad_(False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        info = model.load_state_dict(sd, strict=True, assign=True)
        model.is_convrot_int8 = True
        logger.info(f"Loaded Anima DiT: {info}; {len(groupsizes)} projections in ConvRot INT8 (backward {convrot_int8_bwd})")
        return model

    logger.info(f"Loading Anima DiT from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=anima_model.FP8_OPTIMIZATION_TARGET_KEYS,  # substrings: the released files' prefixes need no entry
        exclude_keys=anima_model.FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    anima_model.strip_checkpoint_prefix(sd)
    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Anima DiT: {info}")
    return model


# ------------------------------------------------------------------------------------------------------- text encoder
def load_tokenizers():
    """(Qwen tokenizer, T5 tokenizer) of the Diffusers release, pinned (Guardrail 6)."""
    from transformers import AutoTokenizer

    qwen = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="tokenizer", revision=TOKENIZER_REVISION)
    t5 = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="t5_tokenizer", revision=TOKENIZER_REVISION)
    return qwen, t5


def qwen3_config():
    from transformers import Qwen3Config

    config = Qwen3Config(**QWEN3_06B_CONFIG)
    config._attn_implementation = "sdpa"
    return config


def load_text_encoder(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu"):
    """Qwen3-0.6B base from the ComfyUI single file (``model.*`` keys; a bare layout loads too). Returns the frozen
    ``Qwen3Model`` (no LM head) in ``dtype`` on ``device``."""
    from transformers import Qwen3Model

    with init_empty_weights():
        model = Qwen3Model(qwen3_config())
    with safe_open(path, framework="pt", device="cpu") as handle:
        sd = {}
        for key in handle.keys():
            name = key[len("model.") :] if key.startswith("model.") else key
            if name.startswith("lm_head."):
                continue
            sd[name] = handle.get_tensor(key).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"{path} is not a Qwen3-0.6B text encoder (missing {missing[:5]}, unexpected {unexpected[:5]})")
    model.to(device)  # init_empty_weights leaves the buffers (the rotary inv_freq) real, in fp32
    return model.eval().requires_grad_(False)


def tokenize(qwen_tokenizer, t5_tokenizer, text: str) -> tuple[list[int], list[int]]:
    """(Qwen ids, T5 ids) of one caption, as ComfyUI's AnimaTokenizer gives them for text without weight syntax."""
    qwen_ids = qwen_tokenizer(text, add_special_tokens=False)["input_ids"][:MAX_QWEN_TOKENS]
    if not qwen_ids:
        qwen_ids = [QWEN_PAD_TOKEN_ID]
    t5_ids = t5_tokenizer(text)["input_ids"]  # ends with </s>
    if len(t5_ids) > MAX_T5_TOKENS:
        t5_ids = t5_ids[: MAX_T5_TOKENS - 1] + [anima_model.T5_EOS_TOKEN_ID]
    return qwen_ids, t5_ids


@torch.no_grad()
def encode_prompts(text_encoder, qwen_tokenizer, t5_tokenizer, prompts: list[str], batch_size: int = 8) -> list[tuple[Tensor, Tensor]]:
    """Per prompt (Qwen3 states (L, 1024) in the encoder's dtype on its device, T5 ids (T,) int64 on the CPU). Up to
    ``batch_size`` prompts run as one right-padded forward: Qwen3 is causal, so a prompt's states equal its own
    unpadded forward's."""
    device = next(text_encoder.parameters()).device
    rows = [tokenize(qwen_tokenizer, t5_tokenizer, p) for p in prompts]
    out: list[Optional[tuple[Tensor, Tensor]]] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(rows[i][0]))  # similar lengths together: less padding
    for start in range(0, len(order), max(1, batch_size)):
        indices = order[start : start + max(1, batch_size)]
        length = max(len(rows[i][0]) for i in indices)
        input_ids = torch.full((len(indices), length), QWEN_PAD_TOKEN_ID, dtype=torch.long)
        mask = torch.zeros((len(indices), length), dtype=torch.long)
        for row, i in enumerate(indices):
            ids = rows[i][0]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, : len(ids)] = 1
        states = text_encoder(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
        for row, i in enumerate(indices):
            out[i] = (states[row, : len(rows[i][0])], torch.tensor(rows[i][1], dtype=torch.long))
    return out


def batch_text(qwen_states: list[Tensor], t5_ids: list[Tensor], device, dtype):
    """Right-pad a batch of cached captions: (states (B, S, 1024), states_mask, t5 ids (B, T), t5_mask). The masks are
    None when every caption has the same lengths (one caption, the usual case: exactly ComfyUI's unmasked forward)."""
    s_len = max(s.shape[0] for s in qwen_states)
    t_len = max(t.shape[0] for t in t5_ids)
    batch = len(qwen_states)
    states = torch.zeros(batch, s_len, qwen_states[0].shape[-1], device=device, dtype=dtype)
    ids = torch.full((batch, t_len), anima_model.T5_PAD_TOKEN_ID, device=device, dtype=torch.long)
    states_mask = torch.zeros(batch, s_len, device=device, dtype=torch.bool)
    ids_mask = torch.zeros(batch, t_len, device=device, dtype=torch.bool)
    for i, (s, t) in enumerate(zip(qwen_states, t5_ids)):
        states[i, : s.shape[0]] = s.to(device=device, dtype=dtype)
        ids[i, : t.shape[0]] = t.to(device=device, dtype=torch.long)
        states_mask[i, : s.shape[0]] = True
        ids_mask[i, : t.shape[0]] = True
    uniform = all(s.shape[0] == s_len for s in qwen_states) and all(t.shape[0] == t_len for t in t5_ids)
    if uniform:
        return states, None, ids, None
    return states, states_mask, ids, ids_mask


# ----------------------------------------------------------------------------------------------------------------- VAE
def load_vae(path: str, device: Union[str, torch.device] = "cpu", dtype: Optional[torch.dtype] = None):
    """The Qwen-Image VAE (musubi's loader: ComfyUI or Diffusers keys); ``dtype`` None keeps the file's (BF16)."""
    from musubi_tuner.qwen_image import qwen_image_utils

    vae = qwen_image_utils.load_vae(path, 3, device=device, disable_mmap=True)
    if dtype is not None:
        vae.to(dtype)
    return vae.eval().requires_grad_(False)


# ------------------------------------------------------------------------------------------------------------ sampling
def time_snr_shift(alpha: float, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def model_sigmas(shift: float = DEFAULT_SHIFT, timesteps: int = 1000) -> Tensor:
    """ComfyUI's ModelSamplingDiscreteFlow grid (multiplier 1): sigma(i / 1000) for i = 1..1000, ascending."""
    return time_snr_shift(shift, torch.arange(1, timesteps + 1, 1) / timesteps)


def simple_schedule(steps: int, shift: float = DEFAULT_SHIFT) -> Tensor:
    """ComfyUI's ``simple`` scheduler: ``steps`` sigmas taken from the top of the model grid at even index strides,
    then 0 (float32)."""
    sigmas = model_sigmas(shift)
    stride = len(sigmas) / steps
    values = [float(sigmas[-(1 + int(x * stride))]) for x in range(steps)] + [0.0]
    return torch.FloatTensor(values)


def _er_sde_noise_scaler(x: Tensor) -> Tensor:
    return x * ((x**0.3).exp() + 10.0)


def _noise_sampler(x: Tensor, seed: Optional[int], generator: Optional[torch.Generator]):
    """ComfyUI's default_noise_sampler: a generator on x's device seeded with the seed (seed + 1 on the CPU)."""
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed + 1 if x.device.type == "cpu" else seed)
    return lambda: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


@torch.no_grad()
def sample_er_sde(
    denoise_fn, x: Tensor, sigmas: Tensor, shift: float = DEFAULT_SHIFT, seed: Optional[int] = None, generator=None, max_stage: int = 3
) -> Tensor:
    """ComfyUI's ``sample_er_sde`` (VP ER-SDE-Solver-3, arXiv 2309.06169) for a flow (CONST) model, s_noise 1: the first
    sigma 1 is moved to the model's sigma at 1e-4 below it (``offset_first_sigma_for_snr``)."""
    noise = _noise_sampler(x, seed, generator)
    num_points = 200.0
    point_indice = torch.arange(0, num_points, dtype=torch.float32, device=x.device)
    sigmas = sigmas.clone()
    if len(sigmas) > 1 and sigmas[0] >= 1:
        sigmas[0] = time_snr_shift(shift, 1.0 - 1e-4)
    half_log_snrs = sigmas.logit().neg()
    er_lambdas = half_log_snrs.neg().exp()  # sigma_t / alpha_t
    old_denoised = old_denoised_d = None
    for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
        denoised = denoise_fn(x, sigmas[i])
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = _er_sde_noise_scaler(er_lambda_t) / _er_sde_noise_scaler(er_lambda_s)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised  # stage 1: Euler
            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = _er_sde_noise_scaler(lambda_pos)
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * _er_sde_noise_scaler(er_lambda_t)) * denoised_d
                if stage_used >= 3:
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt**2) / 2 + s_u * _er_sde_noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d
            x = x + alpha_t * noise() * (er_lambda_t**2 - er_lambda_s**2 * r**2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_euler(denoise_fn, x: Tensor, sigmas: Tensor) -> Tensor:
    """ComfyUI's ``sample_euler`` (no churn) on x0 predictions."""
    for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
        denoised = denoise_fn(x, sigmas[i])
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


class _Denoiser:
    """x0 prediction with classifier-free guidance, as ComfyUI's CFGGuider for a flow model: the DiT gets x in its dtype
    and sigma as the timestep; denoised = x - sigma * v; cfg = uncond + scale * (cond - uncond). Both branches run as
    one batch (the DiT has no cross-item interaction)."""

    def __init__(self, model, context: Tensor, neg_context: Optional[Tensor], guidance: float):
        self.model = model
        self.guidance = guidance
        self.do_cfg = neg_context is not None and guidance != 1.0
        self.context = torch.cat((context, neg_context), 0) if self.do_cfg else context

    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        batch = x.shape[0]
        xin = torch.cat((x, x), 0) if self.do_cfg else x
        t = sigma.reshape(1).float().expand(xin.shape[0]).to(x.device)
        v = self.model(xin.to(self.model.dtype), t, self.context.to(self.model.dtype)).float()
        denoised = xin - v * sigma.to(x.device)
        if not self.do_cfg:
            return denoised
        cond, uncond = denoised[:batch], denoised[batch:]
        return uncond + (cond - uncond) * self.guidance


def initial_noise(shape: tuple, seed: Optional[int], device, generator: Optional[torch.Generator] = None) -> Tensor:
    """ComfyUI's prepare_noise: float32 noise drawn on the CPU from torch.manual_seed(seed) (latents (B, 16, 1, h, w))."""
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, dtype=torch.float32, generator=cpu_generator, device="cpu").to(device)
    return torch.randn(shape, dtype=torch.float32, generator=generator, device=generator.device if generator else device).to(device)


@torch.no_grad()
def denoise(
    model,
    context: Tensor,
    neg_context: Optional[Tensor],
    height: int,
    width: int,
    steps: int = DEFAULT_SAMPLE_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    shift: float = DEFAULT_SHIFT,
    sampler: str = DEFAULT_SAMPLER,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample (1, 16, 1, height / 8, width / 8) latents from noise. ``context`` / ``neg_context``: (1, N, 1024) DiT
    contexts (``Anima.encode_context``)."""
    if sampler not in SAMPLERS:
        raise ValueError(f"Anima: unknown sampler {sampler!r}; choose one of {SAMPLERS}")
    device = context.device
    x = initial_noise((1, anima_model.AnimaParams().in_channels, 1, height // 8, width // 8), seed, device, generator)
    sigmas = simple_schedule(steps, shift).to(device)
    x = x * sigmas[0]  # CONST noise scaling at the first sigma (1.0)
    fn = _Denoiser(model, context, neg_context, guidance)
    if sampler == "er_sde":
        return sample_er_sde(fn, x, sigmas, shift=shift, seed=seed, generator=generator)
    return sample_euler(fn, x, sigmas)
