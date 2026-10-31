"""MOSS-Image loading, text conditioning and sampling.

* DiT: the ComfyUI single files of Comfy-Org/MOSS-Image (``moss_image_base_bf16`` / ``moss_image_sft_bf16``, the
  vendor's keys with each block's q / k / v fused into ``proj.qkv``) or the vendor's split layout; both load into
  ``moss_model.MossImage`` with q / k / v split. LoRA training can keep the blocks' projections in ConvRot INT8: the
  INT8 ConvRot files as they are, or the BF16 file quantized while loading (``--convrot_int8``, as kohya's sd-scripts).
* VAE: the FLUX.2 VAE (``flux2_utils.load_ae``): MOSS-Image's latents are its packed, BatchNorm-normalized 128 channels
  at 1/16, so FLUX.2's latent caching is reused.
* Text: Qwen3.5-9B's language tower (``Qwen3_5TextModel``, the vendor's text encoder without the vision tower: text-only
  inputs give the same states), the final-normed last hidden state (L, 4096) of the chat template with thinking on and
  the generation prompt appended (an open ``<think>`` block), one unpadded forward per prompt as the vendor (padding
  would leak into the gated delta-net recurrence). Read from the ComfyUI file: Comfy-Org/MOSS-Image's
  ``qwen3.5_9b_int8_convrot`` (dequantized to BF16 on load, the file SwarmUI uses) or Comfy-Org/Qwen3.5's BF16 file.
  Matches ComfyUI's ``MOSSImageTokenizer`` / ``qwen35`` text model, where the trained LoRAs are used.
* Edit LoRAs (ai-toolkit's layout, ComfyUI's ``TextEncodeMOSSImageEdit`` edit_format "ai-toolkit"): the text encoder is
  the whole ``Qwen3_5Model`` (language and vision towers, the same files), and each prompt is preceded by its reference
  images, ``Picture N: <|vision_start|><|image_pad|>...<|vision_end|>``, each scaled down by area to at most 384 x 384
  pixels, then to Qwen's 32 px grid (bilinear), mean / std 0.5, as ComfyUI prepares them (``vlm_pixels``).
* Flow: sigma 1 = noise, the DiT predicts ``noise - clean``; the vendor sampler shifts a uniform grid by 5 and snaps it
  to the 1000 training sigmas (50 steps, CFG 3 whose negative branch skips block 22, empty negative prompt).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import logging
import math
from typing import Optional, Sequence, Union

import numpy as np

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from musubi_tuner.flux_2.flux2_utils import load_ae  # noqa: F401  (the FLUX.2 VAE is MOSS-Image's VAE)
from musubi_tuner.modules.comfy_quant_utils import COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX
from musubi_tuner.modules.convrot_int8_utils import (
    CONVROT_GROUPSIZE,
    apply_convrot_int8_monkey_patch,
    canonicalize_convrot_int8_key,
    parse_comfy_quant_spec,
    quantize_weight_convrot,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
from musubi_tuner.modules.int8_convrot_utils import dequantize_int8_convrot_weight, parse_comfy_quant_tensor
from musubi_tuner.moss_image import moss_model
from musubi_tuner.utils.model_utils import cpu_bf16_is_slow, keep_weights_in_cpu_ram
from musubi_tuner.utils.safetensors_utils import load_safetensors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# a uint8 JSON tensor per block in Comfy-Org's INT8 ConvRot DiT files: ComfyUI's attention backend for the block
COMFY_ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"

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

# Qwen/Qwen3.5-9B config.json (the same revision): the vision tower, for edit LoRAs (the INT8 ConvRot file holds it)
QWEN35_9B_VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
QWEN35_9B_TOKEN_IDS = {"image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053, "vision_end_token_id": 248054}
# the reference images the text encoder sees: at most 384 x 384 pixels (144 tokens), ComfyUI's TextEncodeMOSSImageEdit
VLM_MAX_PIXELS = 384 * 384
# a caption file can hold several captions separated by lines starting with this (kohya sd-scripts caab8bb's default)
CAPTION_SEPARATOR = "---===---"

MAX_TEXT_TOKENS = 4096
DEFAULT_SHIFT = 5.0  # the vendor's sampler shift (ComfyUI's model_sampling); ai-toolkit also trains at it, the GUI presets
# train at --discrete_flow_shift 3 (about the pre-training distribution, one save cleaner on two seeds)
DEFAULT_GUIDANCE = 3.0
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_UNCOND_SKIP_LAYERS = (22,)


# ---------------------------------------------------------------------------------------------------------------- DiT
def check_dit_checkpoint(dit_path: str, allow_int8: bool = False) -> bool:
    """Refuse files that cannot train as the MOSS-Image DiT, with the reason. True for a ComfyUI INT8 ConvRot file,
    which only LoRA training takes (``allow_int8``: its INT8 layers are used as they are)."""
    with safe_open(dit_path, framework="pt") as handle:
        keys = list(handle.keys())
    if "shared_block_modulation.weight" not in keys or not any(k.startswith("layers.0.proj.") for k in keys):
        raise ValueError(f"{dit_path} is not a MOSS-Image DiT checkpoint (no shared_block_modulation / layers.N.proj)")
    if not any(k.endswith((COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX, ".input_scale")) for k in keys):
        return False
    if not allow_int8:
        raise ValueError(
            f"{dit_path} is a pre-quantized MOSS-Image file (INT8 ConvRot): full fine-tuning trains the BF16 file "
            "(moss_image_base_bf16.safetensors or moss_image_sft_bf16.safetensors). LoRA training takes both."
        )
    if any(k.endswith(".input_scale") for k in keys) or not any(k.endswith(COMFY_QUANT_SUFFIX) for k in keys):
        raise ValueError(f"{dit_path} is a pre-quantized MOSS-Image file of another format: only INT8 ConvRot files train")
    return True


def is_convrot_int8_target(key: str) -> bool:
    """The layers ConvRot INT8 quantizes: the six projections of the 42 joint blocks (q / k / v / o, ffn_up_gate,
    ffn_down), the layers Comfy-Org's INT8 files hold in INT8 (their q / k / v fused) and kohya's sd-scripts quantizes.
    The text refiner, the embedders, the modulations and the head stay BF16."""
    return key.startswith("layers.") and ".proj." in key and key.endswith(".weight")


def convrot_int8_state_dict(sd: dict, device: torch.device, dtype: Optional[torch.dtype], quantize: bool, keep_on_device: bool) -> dict:
    """In place: ``sd`` (q / k / v split) in musubi's ConvRot INT8 layout, the ``.weight`` INT8 in the rotated basis
    with its FP32 [N, 1] ``.scale_weight``; returns each INT8 layer's group size for ``apply_convrot_int8_monkey_patch``.

    An INT8 ConvRot file's layers are kept as they are (``.weight_scale`` renamed, the group size from ``.comfy_quant``).
    ``quantize`` turns the other BF16 targets into ConvRot INT8 on ``device`` (Hadamard groups of 256, row-wise scales:
    Comfy-Org's files hold the same values for the same BF16 weights). Every other floating-point tensor gets ``dtype``;
    ``keep_on_device`` keeps the quantized layers on ``device`` (no block swap)."""
    groupsizes = {}
    for key in [k for k in sd if k.endswith(COMFY_ATTENTION_CONFIG_SUFFIX)]:
        del sd[key]  # ComfyUI's attention backend for the block ({"attention": "comfy_kitchen_int8"}), not a weight
    for key in [k for k in sd if k.endswith(COMFY_QUANT_SUFFIX)]:
        spec = parse_comfy_quant_spec(key, sd.pop(key))
        groupsizes[key[: -len(COMFY_QUANT_SUFFIX)]] = spec["convrot_groupsize"]
    for key in [k for k in sd if k.endswith(COMFY_WEIGHT_SCALE_SUFFIX)]:
        sd[canonicalize_convrot_int8_key(key)] = sd.pop(key).float()
    for module, groupsize in groupsizes.items():
        weight, scale = sd.get(module + ".weight"), sd.get(module + ".scale_weight")
        if weight is None or scale is None or weight.dtype != torch.int8 or tuple(scale.shape) != (weight.shape[0], 1):
            raise ValueError(f"MOSS-Image INT8 layer {module}: expected an int8 weight with an FP32 [rows, 1] weight_scale")
        if weight.shape[1] % groupsize:
            raise ValueError(f"MOSS-Image INT8 layer {module}: {weight.shape[1]} inputs do not divide into ConvRot groups of {groupsize}")
    target = device if keep_on_device else torch.device("cpu")
    for key in list(sd):
        value = sd[key]
        module = key[: -len(".weight")] if key.endswith(".weight") else None
        if module in groupsizes or key.endswith(".scale_weight"):
            continue
        if module is not None and value.dtype.itemsize == 1:
            raise ValueError(f"MOSS-Image DiT weight {key} is {value.dtype} without an INT8 ConvRot spec")
        if quantize and is_convrot_int8_target(key):
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
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    disable_numpy_memmap: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    allow_int8_file: bool = False,
) -> moss_model.MossImage:
    """The MOSS-Image DiT. ``convrot_int8`` quantizes the joint blocks' projections to ConvRot INT8 while loading;
    ``allow_int8_file`` (LoRA training) takes a ComfyUI INT8 ConvRot file, whose INT8 layers are used as they are. The
    INT8 layers keep their nn.Linear modules with a patched forward (fused Triton INT8 GEMM, a dequantizing fallback
    without Triton) and the model is frozen: INT8 weights cannot require gradients."""
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)
    int8_file = check_dit_checkpoint(dit_path, allow_int8=allow_int8_file)
    use_int8 = convrot_int8 or int8_file
    if use_int8 and (fp8_scaled or dit_weight_dtype.itemsize == 1):
        raise ValueError("MOSS-Image: ConvRot INT8 and FP8 are two quantizations of the same weights; choose one (FP8 Base / Scaled off)")
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = moss_model.MossImage(moss_model.MossImageParams(), attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading MOSS-Image DiT from {dit_path}, device={loading_device}" + (" (ConvRot INT8)" if use_int8 else ""))
    # q / k / v are split before any FP8 scaling, so every projection gets its own scales; an INT8 file is read as stored
    load_dtype = None if use_int8 else dit_weight_dtype
    sd = load_safetensors(dit_path, device="cpu", disable_mmap=True, dtype=load_dtype, disable_numpy_memmap=disable_numpy_memmap)
    moss_model.split_fused_qkv(sd)
    if use_int8:
        groupsizes = convrot_int8_state_dict(sd, device, dit_weight_dtype, quantize=convrot_int8, keep_on_device=loading_device == device)
        apply_convrot_int8_monkey_patch(model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=groupsizes)
        # load_state_dict(assign=True) gives the loaded tensors the placeholders' requires_grad, and INT8 cannot have it
        model.requires_grad_(False)
        print(f"MOSS-Image: {len(groupsizes)} block projections in ConvRot INT8 (backward {convrot_int8_bwd})")
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


def _text_encoder_state_dict(path: str, dtype: torch.dtype, device, vision: bool = False) -> dict[str, Tensor]:
    """The language tower's weights of a ComfyUI Qwen3.5 file in ``dtype``: INT8 ConvRot tensors (``comfy_quant``)
    dequantized, the LM head, MTP head and vision tower dropped. ``vision`` keeps the vision tower too, in
    ``Qwen3_5Model``'s names (``visual.*`` and ``language_model.*``)."""
    out = {}
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        quantized = {k[: -len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
        if vision and not any(k.startswith(("model.visual.", "visual.")) for k in keys):
            raise ValueError(f"{path} holds no Qwen3.5 vision tower (model.visual.*), which MOSS-Image edit training needs")
        for key in keys:
            is_visual = key.startswith(("model.visual.", "visual."))
            if (key.startswith(_TE_DROPPED) and not (vision and is_visual)) or key.endswith((".comfy_quant", ".weight_scale")):
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
            if is_visual:
                key = "visual." + key.split("visual.", 1)[1]
            else:
                for prefix in _TE_PREFIXES:
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                if vision:
                    key = "language_model." + key
            out[key] = tensor
    return out


# Qwen3.5's gated-delta layers call these module functions; transformers binds them at import to the
# flash-linear-attention (fla) / causal-conv1d kernels when those packages are installed (package per function)
_QWEN35_KERNELS = {
    "torch_chunk_gated_delta_rule": "fla",
    "torch_recurrent_gated_delta_rule": "fla",
    "causal_conv1d_fn": "causal_conv1d",
    "causal_conv1d_update": "causal_conv1d",
}


def _first_tensor(args, kwargs) -> Optional[Tensor]:
    return next((value for value in itertools.chain(args, kwargs.values()) if isinstance(value, Tensor)), None)


def _on_gpu(tensor: Optional[Tensor]) -> bool:
    return tensor is not None and tensor.device.type == "cuda"


def install_qwen35_kernel_dispatch(module=None) -> list[str]:
    """Route Qwen3.5's gated-delta kernels by device; returns the functions routed.

    With flash-linear-attention / causal-conv1d installed, transformers binds these functions to their Triton / CUDA
    kernels for every call. The kernels cannot run on CPU tensors, and MOSS text encoding also runs on the CPU (FP32
    where the CPU is slow in BF16, the low-VRAM presets' CPU caching): there they stopped with "Pointer argument cannot
    be accessed from Triton". Each call now goes to the kernel for CUDA tensors and to the reference otherwise; a
    kernel that raises (a Triton build problem, say) is reported once and the reference serves from then on. fla is
    not a requirement: on 64 captions (A6000, vram_lab/moss_te_fla_bench.py) it took 0.21 s instead of 0.27 s per
    batch of 8 with the encoder resident, after about 1.6 s of Triton compiling per process, and the same 7.3 s
    layer-streamed (bound by the weight copies); outputs within BF16 noise of the reference (2.8%).
    """
    if module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    routed = []
    for name, package in _QWEN35_KERNELS.items():
        fast = getattr(module, name, None)
        if fast is None or getattr(fast, "_moss_kernel_dispatch", False) or importlib.util.find_spec(package) is None:
            continue
        reference = inspect.unwrap(fast)
        if reference is fast:
            continue
        state = {"failed": False}

        def dispatch(*args, _fast=fast, _reference=reference, _name=name, _state=state, **kwargs):
            if _state["failed"] or not _on_gpu(_first_tensor(args, kwargs)):
                return _reference(*args, **kwargs)
            try:
                return _fast(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - the reference computes the same values
                _state["failed"] = True
                logger.warning(f"Qwen3.5 {_name}: the fast kernel failed ({type(exc).__name__}: {exc}); using the PyTorch reference")
                return _reference(*args, **kwargs)

        dispatch._moss_kernel_dispatch = True
        setattr(module, name, dispatch)
        routed.append(name)
    if routed:
        logger.info(f"Qwen3.5 text encoder: {', '.join(routed)} use their fast kernels on the GPU")
    return routed


def qwen35_vl_config(text_config: Optional[dict] = None, vision_config: Optional[dict] = None):
    """``Qwen3_5Config`` of Qwen3.5-9B with both towers (the edit LoRAs' text encoder)."""
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(text_config or QWEN35_9B_TEXT_CONFIG),
        vision_config=dict(vision_config or QWEN35_9B_VISION_CONFIG),
        tie_word_embeddings=False,
        **QWEN35_9B_TOKEN_IDS,
    )


def load_text_encoder(
    path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu", stream: bool = False, vision: bool = False
):
    """Qwen3.5-9B's language tower from a ComfyUI file and the pinned vendor tokenizer: (model, tokenizer). ``stream``
    (CUDA only) keeps the tower in CPU RAM and runs it on ``device`` one decoder layer at a time (stream_text_encoder).
    ``vision`` (edit LoRAs) loads ``Qwen3_5Model`` with the vision tower as well; its ``language_model`` is the text tower
    (streamed when ``stream``), the vision tower (0.4B parameters) runs on ``device``."""
    from transformers import AutoTokenizer, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5TextModel

    install_qwen35_kernel_dispatch()
    device, dtype, stream = text_encoder_plan(device, dtype, stream)
    with init_empty_weights():
        model = Qwen3_5Model(qwen35_vl_config()) if vision else Qwen3_5TextModel(Qwen3_5TextConfig(**QWEN35_9B_TEXT_CONFIG))
    # dequantizing on the GPU is ~10x faster than on the CPU; the result goes to the CPU first either way
    dequant_device = torch.device("cuda") if torch.cuda.is_available() and torch.device(device).type == "cuda" else torch.device("cpu")
    logger.info(f"Loading MOSS-Image text encoder (Qwen3.5-9B{' with its vision tower' if vision else ''}) from {path}")
    sd = _text_encoder_state_dict(path, dtype, dequant_device, vision=vision)
    info = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in info.missing_keys if "rotary_emb" not in k and "rotary_pos_emb" not in k]
    if missing or info.unexpected_keys:
        raise RuntimeError(f"Qwen3.5 text encoder checkpoint did not match: missing={missing[:10]}, unexpected={info.unexpected_keys[:10]}")
    # buffers the checkpoint does not hold (rotary inv_freq) are built by the constructors, never on the meta device
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            raise RuntimeError(f"Qwen3.5 text encoder buffer {name} was not initialized")
    text = model.language_model if vision else model
    if stream and torch.device(device).type == "cuda":
        stream_text_encoder(text, device)
        if vision:
            model.visual.to(device)
    else:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MOSS_TOKENIZER_REPO, subfolder="tokenizer", revision=MOSS_TOKENIZER_REVISION)
    return model.eval().requires_grad_(False), tokenizer


def text_tower(model):
    """The language tower of a text encoder from load_text_encoder (``Qwen3_5Model.language_model`` or the model)."""
    return getattr(model, "language_model", model)


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
    model = text_tower(model)  # text-only prompts give the same states with or without the vision tower
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


def split_caption_candidates(caption: str, separator: Optional[str] = CAPTION_SEPARATOR) -> list[str]:
    """The captions of one caption file: lines whose stripped text starts with ``separator`` separate them (text after
    ``#`` on such a line is a comment, other text there is refused); each candidate keeps its inner line breaks, loses
    trailing spaces and leading / trailing blank lines; empty candidates are dropped (kohya sd-scripts caab8bb). Without a
    separator, or none in the text, the whole caption is the one candidate."""
    if not separator or separator not in str(caption or ""):
        return [str(caption or "")]
    candidates, current = [], []
    for line in str(caption).split("\n"):
        stripped = line.strip()
        if stripped.startswith(separator):
            reserved = stripped[len(separator) :].split("#", 1)[0].strip()
            if reserved:
                raise ValueError(f"unsupported text after the caption separator (only # comments are allowed): {line!r}")
            candidates.append("\n".join(current))
            current = []
        else:
            current.append(line)
    candidates.append("\n".join(current))
    cleaned = []
    for candidate in candidates:
        lines = [line.rstrip() for line in candidate.split("\n")]
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        if lines:
            cleaned.append("\n".join(lines))
    return cleaned or [""]


def _image_tensor(image) -> Tensor:
    """(1, 3, H, W) float32 in [0, 1] from an (H, W, C) uint8 array, a PIL image or a (3, H, W) / (1, 3, H, W) tensor in
    [0, 1]."""
    if isinstance(image, Tensor):
        tensor = image.detach().float()
        return (tensor if tensor.dim() == 4 else tensor.unsqueeze(0))[:, :3].cpu()
    if not isinstance(image, np.ndarray):
        image = np.asarray(image.convert("RGB"))
    array = np.ascontiguousarray(image[..., :3])
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def vlm_pixels(image, max_pixels: int = VLM_MAX_PIXELS, patch: int = 16, merge: int = 2, temporal: int = 2) -> tuple[Tensor, Tensor]:
    """A reference image for Qwen3.5's vision tower, prepared as ComfyUI's ``TextEncodeMOSSImageEdit`` and
    ``qwen_vl.process_qwen2vl_images`` do: scaled down (never up) by area to at most ``max_pixels``, resized bilinearly to
    the 32 px grid, normalized with mean / std 0.5 and cut into 2 x 16 x 16 patches in Qwen's order.
    Returns (patches (N, 3 * 2 * 16 * 16) float32, grid_thw (1, 3))."""
    x = _image_tensor(image)
    _, _, height, width = x.shape
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    height, width = round(height * scale), round(width * scale)
    x = F.interpolate(x, size=(height, width), mode="area")
    factor = patch * merge
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar < 3136:  # Qwen's minimum pixel count (tiny references only)
        beta = math.sqrt(3136 / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    h_bar, w_bar = max(factor, h_bar), max(factor, w_bar)
    x = F.interpolate(x, size=(h_bar, w_bar), mode="bilinear", align_corners=False)[0]
    x = (x - 0.5) / 0.5
    grid_h, grid_w = h_bar // patch, w_bar // patch
    patches = x.unsqueeze(0).repeat(temporal, 1, 1, 1).reshape(
        1, temporal, 3, grid_h // merge, merge, patch, grid_w // merge, merge, patch
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_h * grid_w, 3 * temporal * patch * patch)
    return patches.contiguous(), torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)


def edit_prompt_text(prompt: str, image_tokens: Sequence[int]) -> str:
    """The user turn of an edit prompt: ``Picture N: <|vision_start|><|image_pad|> x tokens<|vision_end|>`` per reference,
    then the instruction (the Qwen-Image-Edit-Plus layout ai-toolkit trains and ComfyUI's TextEncodeMOSSImageEdit builds)."""
    pictures = "".join(f"Picture {index + 1}: <|vision_start|>{'<|image_pad|>' * count}<|vision_end|>" for index, count in enumerate(image_tokens))
    return pictures + str(prompt or "")


@torch.no_grad()
def encode_prompts_with_images(model, tokenizer, prompts: list[str], images: list[Optional[Sequence]]) -> list[Tensor]:
    """One (L_i, 4096) final hidden state per prompt, each prompt preceded by its reference images (edit LoRAs; ``model``
    from load_text_encoder(vision=True)). A prompt without images is a text-only prompt. One forward per prompt with
    images (their image-pad runs differ); the vision tower's tokens replace the pads, positions are Qwen's M-RoPE
    (``get_rope_index``), as ComfyUI's qwen35 model computes them."""
    language = text_tower(model)
    stream_device = getattr(language, "_moss_stream_device", None)
    device = stream_device or next(language.parameters()).device
    image_token = int(model.config.image_token_id)
    merge = int(model.config.vision_config.spatial_merge_size)
    out: list[Optional[Tensor]] = [None] * len(prompts)
    text_only = [index for index, refs in enumerate(images) if not refs]
    if text_only:
        for index, state in zip(text_only, encode_prompts(language, tokenizer, [prompts[i] for i in text_only])):
            out[index] = state
    visual_dtype = next(model.visual.parameters()).dtype
    for index, (prompt, refs) in enumerate(zip(prompts, images)):
        if not refs:
            continue
        patches, grids = zip(*(vlm_pixels(image, patch=int(model.config.vision_config.patch_size), merge=merge) for image in refs))
        grid = torch.cat(grids)
        counts = (grid.prod(-1) // merge**2).tolist()
        input_ids = tokenizer(chat_text(tokenizer, edit_prompt_text(prompt, counts)), return_tensors="pt")["input_ids"]
        mm_types = (input_ids == image_token).int()
        if int(mm_types.sum()) != sum(counts):
            raise ValueError(f"the prompt holds {int(mm_types.sum())} image tokens for {sum(counts)} image features")
        embeds = language.embed_tokens(input_ids if stream_device is not None else input_ids.to(device)).to(device)
        features = model.visual(torch.cat(patches).to(device=device, dtype=visual_dtype), grid_thw=grid.to(device)).pooler_output
        slots = (input_ids == image_token).to(device).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(slots, features.to(embeds.dtype))
        position_ids, _ = model.get_rope_index(input_ids.to(device), mm_types.to(device), image_grid_thw=grid.to(device))
        hidden = language(inputs_embeds=embeds, position_ids=position_ids, use_cache=False).last_hidden_state
        out[index] = hidden[0].clone()
        del hidden, embeds, features
    return out  # type: ignore[return-value]


def reference_signature(images: Optional[Sequence]) -> str:
    """A short hash of reference images as the text encoder sees them (shape and pixels), stored with a text encoder
    cache so --skip_existing re-encodes a caption whose references changed."""
    if not images:
        return "none"
    digest = hashlib.sha1()
    for image in images:
        tensor = _image_tensor(image)
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update((tensor * 255).round().to(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def fit_reference(image, width: int, height: int):
    """A sample's reference image as training saw its references: aspect kept at about ``width`` x ``height`` pixels,
    sides on the 16 px grid (the dataset's control_resolution bucket). Returns (H, W, 3) uint8."""
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.dataset.bucket import BucketSelector
    from musubi_tuner.dataset.media_utils import resize_image_to_bucket

    if not isinstance(image, np.ndarray):
        image = np.asarray(image.convert("RGB"))
    image = np.ascontiguousarray(image[..., :3])
    size = (image.shape[1], image.shape[0])
    reso = BucketSelector.calculate_bucket_resolution(size, (int(width), int(height)), architecture=ARCHITECTURE_MOSS_IMAGE)
    return resize_image_to_bucket(image, reso)


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
    ref_latents=None,
    isolate_refs: bool = False,
) -> Tensor:
    """The vendor's sampler: fp32 Euler on fp32 noise (B, 128, h, w); CFG combines fp32 predictions, its negative
    branch skipping ``uncond_skip_layers``. Edit LoRAs: ``ref_latents`` (per sample a list of (128, h, w) reference
    latents) ride in both branches; isolated references (kv_cache) are computed once, before the first step, as ComfyUI
    does (``precompute_ref_kv``), and serve both branches."""
    latents = noise.float()
    do_cfg = neg_ctx is not None and guidance != 1.0
    ref_kwargs = {}
    if ref_latents is not None and any(len(refs or ()) > 0 for refs in ref_latents):
        if isolate_refs:
            ref_kwargs = {"isolate_refs": True, "ref_kv_cache": model.precompute_ref_kv(ref_latents)}
        else:
            ref_kwargs = {"ref_latents": ref_latents}
    sigmas = moss_sigmas(num_steps, shift=shift).to(latents.device)
    for index in tqdm(range(len(sigmas) - 1), desc="Sampling"):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        timestep = (sigma * moss_model.NUM_TRAIN_TIMESTEPS).reshape(1).expand(latents.shape[0])
        pred = model(latents, timestep, ctx, ctx_mask, **ref_kwargs).float()
        if do_cfg:
            pred_uncond = model(latents, timestep, neg_ctx, neg_mask, skip_layers=uncond_skip_layers, **ref_kwargs).float()
            pred = pred_uncond + guidance * (pred - pred_uncond)
        latents = latents + (sigma_next - sigma) * pred
    return latents
