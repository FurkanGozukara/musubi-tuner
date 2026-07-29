#!/usr/bin/env python3
"""Pre-quantize LTX-2 transformer weights to packed INT4 ConvRot."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time

import safetensors
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner.ltx2_model_loading import KEEP_FP8_HIGH_PRECISION_TOKENS
from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from musubi_tuner.modules.convrot_policy import load_convrot_policy, resolve_int4_policy_parameters
from musubi_tuner.modules.int4_convrot_utils import (
    INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX,
    INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX,
    INT4_CONVROT_METADATA_MARKER,
    INT4_CONVROT_STABILIZER_L1_SUFFIX,
    INT4_CONVROT_STABILIZER_L2_SUFFIX,
    best_int4_convrot_groupsize,
    comfy_quant_tensor,
    compare_int4_convrot_group_scales,
    compute_int4_convrot_stabilizer,
    parse_int4_convrot_groupsizes,
    parse_int4_convrot_scale_group_candidates,
    quantize_int4_convrot_weight,
    quantize_int4_convrot_weight_grouped,
    summarize_quality,
    validate_int4_convrot_scale_group_size,
    write_quality_report,
)
from musubi_tuner.modules.int4_convrot_awq import (
    INT4_CONVROT_AWQ_SCALE_SUFFIX,
    apply_int4_convrot_awq_scale_to_weight,
    compute_int4_convrot_awq_scale,
    default_int4_convrot_awq_scales_path,
    load_int4_convrot_awq_scales,
    save_int4_convrot_awq_scales,
    summarize_int4_convrot_awq_scales,
)
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

_INT4_CONVROT_TARGET_PATTERNS = ("transformer_blocks",)


def _is_quantizable(key: str, value: torch.Tensor, groupsizes: tuple[int, ...]) -> tuple[bool, int | None, str]:
    renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
    model_key = renamed if renamed is not None else key
    is_target = model_key.endswith(".weight") and any(t in model_key for t in _INT4_CONVROT_TARGET_PATTERNS)
    is_excluded = any(e in model_key for e in KEEP_FP8_HIGH_PRECISION_TOKENS)
    if not is_target or is_excluded or value.ndim != 2 or value.shape[0] < 8:
        return False, None, model_key
    return True, best_int4_convrot_groupsize(value.shape[1], groupsizes), model_key


def default_quality_report_path(output_model: str) -> str:
    base, _ = os.path.splitext(output_model)
    return f"{base}.quality.json"


def quantize_model(
    input_model: str,
    output_model: str,
    *,
    calc_device: str,
    groupsize: str,
    mse_clip: bool,
    quality_report: str | None,
    awq_calibration: bool,
    awq_alpha: float,
    awq_scales: str | None,
    stabilizer_rank: int = 0,
    no_rotation: bool = False,
    policy_path: str | None = None,
    scale_refine_steps: int = 0,
    group_scales: int = 0,
    group_ratio_q8: bool = False,
    compare_group_scales: str | tuple[int, ...] | None = None,
) -> None:
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    groupsizes = parse_int4_convrot_groupsizes(groupsize)
    group_scales = validate_int4_convrot_scale_group_size(group_scales)
    if group_ratio_q8 and not group_scales:
        raise ValueError("group_ratio_q8 requires group_scales")
    comparison_candidates = parse_int4_convrot_scale_group_candidates(compare_group_scales)
    if comparison_candidates and quality_report is None:
        raise ValueError("compare_group_scales requires a quality report")
    device = torch.device(calc_device)
    rotate = not no_rotation
    policy = load_convrot_policy(policy_path)
    logger.info("INT4 ConvRot quantization device: %s", device)
    logger.info("INT4 ConvRot group candidates: %s", ", ".join(str(g) for g in groupsizes))
    logger.info("INT4 ConvRot MSE clipping: %s", "on" if mse_clip else "off")
    logger.info("INT4 ConvRot least-squares scale refinement steps: %d", int(scale_refine_steps))
    logger.info("INT4 ConvRot group scales: %s", int(group_scales) if group_scales else "off")
    logger.info("INT4 ConvRot group-ratio storage: %s", "Q8.8 int16" if group_ratio_q8 else "float32")
    if comparison_candidates:
        logger.info(
            "INT4 ConvRot report-only group-scale candidates: %s (no selection is performed)",
            ", ".join(str(candidate) for candidate in comparison_candidates),
        )
    logger.info("INT4 ConvRot rotation: %s", "hadamard" if rotate else "none (stabilizer-only)")
    if stabilizer_rank < 0:
        raise ValueError(f"INT4 ConvRot stabilizer rank must be >= 0, got {stabilizer_rank}")
    if no_rotation and stabilizer_rank < 1:
        raise ValueError(
            "INT4 ConvRot --no_rotation requires --stabilizer_rank >= 1: without the online "
            "Hadamard rotation the low-rank SVD stabilizer is the only outlier-isolation mechanism, so INT4 "
            "quantization of the raw weight would be catastrophic. Re-run with e.g. --stabilizer_rank 32."
        )
    if stabilizer_rank > 0:
        logger.info(
            "INT4 ConvRot stabilizer: rank %d low-rank branch (SVD of %s weights)",
            stabilizer_rank,
            "rotated" if rotate else "unrotated",
        )
    if not (0.0 <= float(awq_alpha) <= 1.0):
        raise ValueError(f"INT4 ConvRot AWQ alpha must be in [0, 1], got {awq_alpha}")
    loaded_awq_scales = None
    generated_awq_scales: dict[str, torch.Tensor] = {}
    applied_awq_scales: dict[str, torch.Tensor] = {}
    awq_save_path = None
    if awq_calibration:
        awq_save_path = awq_scales or default_int4_convrot_awq_scales_path(output_model)
        logger.info("INT4 ConvRot AWQ: computing dataset-independent scales (alpha=%.3f)", float(awq_alpha))
    elif awq_scales:
        loaded_awq_scales = load_int4_convrot_awq_scales(awq_scales)

    state_dict: dict[str, torch.Tensor] = {}
    quality_layers = []
    group_scale_comparisons = []
    applied_int4_parameters = []
    quantized_count = 0
    skipped_count = 0
    passthrough_count = 0
    t0 = time.time()

    with MemoryEfficientSafeOpen(input_model) as f:
        keys = list(f.keys())
        fp8_scale_keys = {key for key in keys if key.endswith(".weight_scale") or key.endswith(".input_scale")}
        if fp8_scale_keys:
            logger.info(
                "Detected %d FP8 scale tensors; FP8 weights will be dequantized before INT4 ConvRot quantization",
                len(fp8_scale_keys),
            )
        for key in tqdm(keys, desc="Quantizing INT4 ConvRot", unit="tensor"):
            if key in fp8_scale_keys:
                continue
            value = f.get_tensor(key)
            if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in fp8_scale_keys:
                    raise ValueError(
                        f"INT4 ConvRot source has FP8 weight without weight_scale: {key}. "
                        "Use a bf16/fp16 checkpoint or a scaled FP8 checkpoint with matching scale tensors."
                    )
                value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)
            quantizable, group_size, model_key = _is_quantizable(key, value, groupsizes)
            decision = policy.resolve(model_key) if policy is not None and quantizable else None
            if decision is not None and not decision.quantize:
                quantizable = False
            if not quantizable:
                if key.endswith(".weight") and value.ndim == 2 and any(t in model_key for t in _INT4_CONVROT_TARGET_PATTERNS):
                    skipped_count += 1
                else:
                    passthrough_count += 1
                state_dict[key] = value
                continue

            assert group_size is not None
            layer_parameters = resolve_int4_policy_parameters(
                decision,
                group_scales=group_scales,
                group_ratio_q8=group_ratio_q8,
                scale_refine_steps=scale_refine_steps,
                name=model_key,
            )
            awq_scale = None
            if awq_calibration:
                awq_scale = compute_int4_convrot_awq_scale(value, alpha=float(awq_alpha))
                generated_awq_scales[model_key] = awq_scale
            elif loaded_awq_scales is not None:
                awq_scale = loaded_awq_scales.get(model_key)
                if awq_scale is None:
                    awq_scale = loaded_awq_scales.get(key)
                if awq_scale is None:
                    raise ValueError(f"INT4 ConvRot AWQ scales missing required key for {model_key}")
            if awq_scale is not None:
                value = apply_int4_convrot_awq_scale_to_weight(value, awq_scale)
                applied_awq_scales[model_key] = awq_scale

            stabilizer = None
            if stabilizer_rank > 0:
                stabilizer = compute_int4_convrot_stabilizer(
                    value,
                    group_size=group_size,
                    rank=stabilizer_rank,
                    calc_device=device,
                    rotate=rotate,
                )
            quant_kwargs = {
                "group_size": group_size,
                "calc_device": device,
                "mse_clip": mse_clip,
                "collect_quality": quality_report is not None,
                "key": model_key,
                "stabilizer": stabilizer,
                "rotate": rotate,
                "scale_refine_steps": layer_parameters.scale_refine_steps,
            }
            group_scale_state = None
            if layer_parameters.group_scales:
                q, scale, shape, quality, group_scale_state = quantize_int4_convrot_weight_grouped(
                    value,
                    scale_group_size=layer_parameters.group_scales,
                    ratio_q8=layer_parameters.group_ratio_q8,
                    **quant_kwargs,
                )
            else:
                q, scale, shape, quality = quantize_int4_convrot_weight(value, **quant_kwargs)
            if quality_report is not None:
                applied_int4_parameters.append(
                    {
                        "key": model_key,
                        "group_scales_requested": layer_parameters.group_scales,
                        "group_scales_resolved": (
                            int(group_scale_state.group_size.detach().reshape(-1)[0].item()) if group_scale_state is not None else 0
                        ),
                        "group_ratio_q8": bool(layer_parameters.group_ratio_q8 and group_scale_state is not None),
                        "scale_refine_steps": layer_parameters.scale_refine_steps,
                    }
                )
            if comparison_candidates:
                group_scale_comparisons.append(
                    compare_int4_convrot_group_scales(
                        value,
                        candidates=comparison_candidates,
                        selected_group_scales=layer_parameters.group_scales,
                        selected_quality=quality,
                        selected_group_ratio=group_scale_state.ratio if group_scale_state is not None else None,
                        group_size=group_size,
                        calc_device=device,
                        mse_clip=mse_clip,
                        key=model_key,
                        stabilizer=stabilizer,
                        rotate=rotate,
                        scale_refine_steps=layer_parameters.scale_refine_steps,
                    )
                )
            base = key[: -len(".weight")]
            in_features = int(shape.detach().cpu().reshape(-1)[1].item())
            padded_features = int(shape.detach().cpu().reshape(-1)[2].item())
            state_dict[key] = q.cpu()
            state_dict[base + ".weight_scale"] = scale.cpu()
            state_dict[base + ".int4_shape"] = shape.cpu()
            if group_scale_state is not None:
                state_dict[base + INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX] = group_scale_state.ratio.cpu()
                state_dict[base + INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX] = group_scale_state.group_size.cpu()
                state_dict[base + ".int4_convrot_groupsize"] = torch.tensor(group_size, dtype=torch.int32)
                if not rotate:
                    state_dict[base + ".int4_rotation"] = torch.tensor(0, dtype=torch.int32)
            else:
                state_dict[base + ".comfy_quant"] = comfy_quant_tensor(
                    group_size,
                    in_features,
                    padded_features,
                    convrot=rotate,
                    awq=awq_scale is not None,
                    stabilizer_rank=int(stabilizer[0].shape[1]) if stabilizer is not None else 0,
                )
            if awq_scale is not None:
                state_dict[base + INT4_CONVROT_AWQ_SCALE_SUFFIX] = awq_scale.cpu().float()
            if stabilizer is not None:
                state_dict[base + INT4_CONVROT_STABILIZER_L1_SUFFIX] = stabilizer[0].cpu()
                state_dict[base + INT4_CONVROT_STABILIZER_L2_SUFFIX] = stabilizer[1].cpu()
            if quality is not None:
                quality_layers.append(quality)
            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)

    output_metadata = dict(original_metadata)
    output_metadata[INT4_CONVROT_METADATA_MARKER] = "true"
    output_metadata["int4_convrot_groupsizes"] = ",".join(str(g) for g in groupsizes)
    output_metadata["int4_convrot_mse_clip"] = "true" if mse_clip else "false"
    output_metadata["int4_convrot_scale_refine_steps"] = str(int(scale_refine_steps))
    output_metadata["int4_convrot_group_scales"] = str(int(group_scales))
    output_metadata["int4_convrot_group_ratio_q8"] = "true" if group_ratio_q8 else "false"
    if policy is not None and policy.has_int4_quantization_parameters():
        output_metadata["int4_convrot_per_layer_quantization"] = "true"
    output_metadata["int4_convrot_storage"] = "packed_signed_int4_low_high"
    output_metadata["int4_convrot_rotation"] = "hadamard" if rotate else "none"
    output_metadata["int4_convrot_awq"] = "true" if (awq_calibration or loaded_awq_scales is not None) else "false"
    output_metadata["int4_convrot_awq_alpha"] = str(float(awq_alpha))
    if stabilizer_rank > 0:
        output_metadata["int4_convrot_stabilizer_rank"] = str(int(stabilizer_rank))

    if generated_awq_scales and awq_save_path:
        save_int4_convrot_awq_scales(generated_awq_scales, awq_save_path)
    if applied_awq_scales:
        summary = summarize_int4_convrot_awq_scales(applied_awq_scales)
        logger.info(
            "INT4 ConvRot AWQ scales: layers=%s channels=%s min=%.4g max=%.4g mean=%.4g",
            summary.get("num_layers", 0),
            summary.get("num_channels", 0),
            summary.get("min", 0.0),
            summary.get("max", 0.0),
            summary.get("mean", 0.0),
        )

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Saving INT4 ConvRot checkpoint to %s", output_model)
    save_file(state_dict, output_model, metadata=output_metadata)

    elapsed = time.time() - t0
    input_size = os.path.getsize(input_model) / (1024**3)
    output_size = os.path.getsize(output_model) / (1024**3)
    logger.info(
        "INT4 ConvRot complete in %.1fs: quantized=%d skipped=%d passthrough=%d size=%.2fGB -> %.2fGB",
        elapsed,
        quantized_count,
        skipped_count,
        passthrough_count,
        input_size,
        output_size,
    )

    if quality_report is not None:
        report = write_quality_report(
            quality_report,
            source=input_model,
            output=output_model,
            options={
                "mode": "prequantize",
                "groupsizes": list(groupsizes),
                "mse_clip": mse_clip,
                "target_keys": list(_INT4_CONVROT_TARGET_PATTERNS),
                "exclude_keys": list(KEEP_FP8_HIGH_PRECISION_TOKENS),
                "calc_device": str(device),
                "storage": "packed_signed_int4",
                "awq_calibration": bool(awq_calibration),
                "awq_alpha": float(awq_alpha),
                "awq_scales": awq_save_path or awq_scales,
                "stabilizer_rank": int(stabilizer_rank),
                "scale_refine_steps": int(scale_refine_steps),
                "group_scales": int(group_scales),
                "group_ratio_q8": bool(group_ratio_q8),
                "policy_int4_parameters": bool(policy is not None and policy.has_int4_quantization_parameters()),
                "compare_group_scales": list(comparison_candidates),
            },
            layers=quality_layers,
            group_scale_comparisons=group_scale_comparisons,
            applied_parameters=applied_int4_parameters,
        )
        summary = report["summary"]
        if summary.get("num_layers", 0):
            logger.info(
                "Quality report: %s (min_cosine=%.6f mean_cosine=%.6f weighted_sqnr=%.2f dB)",
                quality_report,
                summary["min_cosine"],
                summary["mean_cosine"],
                summary["weighted_sqnr_db"],
            )
        else:
            logger.warning("Quality report written to %s, but no layers were quantized.", quality_report)
    elif quality_layers:
        summary = summarize_quality(quality_layers)
        logger.info("INT4 ConvRot quality summary: %s", summary)


_COMFY_CONVROT_GROUPSIZE = 256
_COMFY_QUANT_GROUPSIZE = 64


def _is_comfy_convrot_target(key: str, value: torch.Tensor) -> tuple[bool, str]:
    """Same target set as the int4cr path (transformer-block Linear weights, precision-sensitive excluded)."""
    renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
    model_key = renamed if renamed is not None else key
    is_target = model_key.endswith(".weight") and any(t in model_key for t in _INT4_CONVROT_TARGET_PATTERNS)
    is_excluded = any(e in model_key for e in KEEP_FP8_HIGH_PRECISION_TOKENS)
    if not is_target or is_excluded or value.ndim != 2 or value.shape[0] < 8:
        return False, model_key
    return True, model_key


def _comfy_quant_conf_tensor(convrot_groupsize: int, linear_dtype: str) -> torch.Tensor:
    """The per-weight ``<base>.comfy_quant`` blob ComfyUI's loader reads (UTF-8 JSON in a uint8 tensor)."""
    conf = {"format": "convrot_w4a4", "convrot_groupsize": int(convrot_groupsize), "linear_dtype": linear_dtype}
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


# --- Local writer for the ComfyUI convrot_w4a4 checkpoint layout -------------------------------
# This code writes the layout without a runtime dependency on comfy-kitchen. Consumer compatibility
# depends on keeping its packing, scale, and metadata conventions synchronized with that format.
_COMFY_INT4_MAX = 7  # symmetric absmax quantizer range [-7, 7], scale = absmax/7


def _comfy_regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Regular (power-of-4) Hadamard built from H4 Kronecker products, normalized by 1/sqrt(size)."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    return h / (size**0.5)


def _comfy_rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = weight.shape
    n_groups = in_f // group_size
    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h_t).reshape(out_f, in_f)


def _comfy_pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _quantize_comfy_convrot_w4a4_weight(weight: torch.Tensor, convrot_groupsize: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate + symmetric per-row int4 quantize + pack, matching the convrot_w4a4 format (deterministic round).

    Returns (packed_int8 [out, in//2], scale_fp32 [out]).
    """
    h = _comfy_regular_hadamard(convrot_groupsize, weight.device, weight.dtype)
    w_rot = _comfy_rotate_weight(weight, h, convrot_groupsize)
    rows = w_rot.shape[0]
    absmax = w_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _COMFY_INT4_MAX
    q = (w_rot / scales).round_().clamp_(-_COMFY_INT4_MAX, _COMFY_INT4_MAX).to(torch.int8)
    return _comfy_pack_int4_row_major(q), scales.reshape(rows).to(torch.float32)


def export_comfy_convrot_w4a4(
    input_model: str,
    output_model: str,
    *,
    calc_device: str,
    linear_dtype: str = "int4",
) -> None:
    """Export a ComfyUI-loadable ``convrot_w4a4`` checkpoint.

    This is a one-way *publish for inference* artifact, NOT a trainer input: it writes the
    buffers/metadata used by the supported ComfyUI loader (packed int8 ``<base>.weight`` +
    ``<base>.weight_scale`` + ``<base>.comfy_quant`` config). The int4 ConvRot quantization is a
    self-contained implementation of the convrot_w4a4 math (no runtime dependency on any external
    package). Weights whose in_features are not
    divisible by the ConvRot group size (256) are kept unquantized (16-bit).
    No stabilizer / AWQ (convrot_w4a4 has no slot for them).
    """
    if linear_dtype not in ("int4", "int8"):
        raise ValueError(f"--comfy_linear_dtype must be int4 or int8, got {linear_dtype!r}")
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    device = torch.device(calc_device)
    logger.info("comfy_convrot_w4a4 export device: %s, linear_dtype: %s", device, linear_dtype)

    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    state_dict: dict[str, torch.Tensor] = {}
    quantized_count = 0
    passthrough_count = 0
    fallback_layers: list[tuple[str, int]] = []
    t0 = time.time()

    with MemoryEfficientSafeOpen(input_model) as f:
        keys = list(f.keys())
        fp8_scale_keys = {key for key in keys if key.endswith(".weight_scale") or key.endswith(".input_scale")}
        for key in tqdm(keys, desc="Exporting convrot_w4a4", unit="tensor"):
            if key in fp8_scale_keys:
                continue
            value = f.get_tensor(key)
            if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in fp8_scale_keys:
                    raise ValueError(
                        f"comfy_convrot_w4a4 source has FP8 weight without weight_scale: {key}. Use a bf16/fp16 checkpoint."
                    )
                value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)

            is_target, _model_key = _is_comfy_convrot_target(key, value)
            if not is_target:
                state_dict[key] = value
                passthrough_count += 1
                continue

            in_features = int(value.shape[1])
            if in_features % _COMFY_CONVROT_GROUPSIZE != 0:
                # 16-bit fallback tier: in_features not divisible by the ConvRot group size (256).
                state_dict[key] = value
                fallback_layers.append((key, in_features))
                continue

            qdata, scale = _quantize_comfy_convrot_w4a4_weight(
                value.to(device=device, dtype=torch.float32), _COMFY_CONVROT_GROUPSIZE
            )
            base = key[: -len(".weight")]
            state_dict[key] = qdata.cpu().contiguous()
            state_dict[base + ".weight_scale"] = scale.cpu().contiguous()
            state_dict[base + ".comfy_quant"] = _comfy_quant_conf_tensor(_COMFY_CONVROT_GROUPSIZE, linear_dtype)
            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Preserve the source metadata unchanged; ComfyUI keys off the per-weight .comfy_quant blobs.
    logger.info("Saving comfy_convrot_w4a4 checkpoint to %s", output_model)
    save_file(state_dict, output_model, metadata=dict(original_metadata))

    if fallback_layers:
        logger.warning(
            "comfy_convrot_w4a4: %d target Linear(s) kept 16-bit (in_features not divisible by %d): %s",
            len(fallback_layers),
            _COMFY_CONVROT_GROUPSIZE,
            ", ".join(f"{k}(in={n})" for k, n in fallback_layers[:8]) + (" ..." if len(fallback_layers) > 8 else ""),
        )
    elapsed = time.time() - t0
    input_size = os.path.getsize(input_model) / (1024**3)
    output_size = os.path.getsize(output_model) / (1024**3)
    logger.info(
        "comfy_convrot_w4a4 export complete in %.1fs: quantized=%d fallback_16bit=%d passthrough=%d size=%.2fGB -> %.2fGB",
        elapsed,
        quantized_count,
        len(fallback_layers),
        passthrough_count,
        input_size,
        output_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-quantize LTX-2 model weights to packed INT4 ConvRot")
    parser.add_argument("--input_model", required=True, help="Path to original .safetensors checkpoint")
    parser.add_argument("--output_model", required=True, help="Path for INT4 ConvRot output .safetensors")
    parser.add_argument(
        "--export_format",
        default="int4cr",
        choices=["int4cr", "comfy_convrot_w4a4"],
        help=(
            "int4cr (default): this trainer's reusable INT4 ConvRot prepack (load with --w4a4g4/--w4a4g8/--w4a8). "
            "comfy_convrot_w4a4: a one-way ComfyUI-loadable convrot_w4a4 inference checkpoint; "
            "requires a ComfyUI build with convrot_w4a4 support; ignores stabilizer/AWQ/rotation options."
        ),
    )
    parser.add_argument(
        "--comfy_linear_dtype",
        default="int4",
        choices=["int4", "int8"],
        help="Matrix-mult precision recorded per layer for --export_format comfy_convrot_w4a4 (default int4).",
    )
    parser.add_argument(
        "--calc_device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for rotation/quantization math (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--groupsize",
        default="auto",
        help="ConvRot group size or comma list. Default: auto (256,64,16); tensors are padded when needed.",
    )
    parser.add_argument("--no_mse_clip", action="store_true", help="Use plain absmax scales instead of MSE clipping")
    parser.add_argument(
        "--scale_refine_steps",
        type=int,
        default=0,
        help="Alternating least-squares row-scale refinement steps after clipping search (default: 0)",
    )
    parser.add_argument(
        "--int4_convrot_group_scales",
        type=int,
        default=0,
        metavar="SIZE",
        help="Enable per-group INT4 weight scales with this maximum K-group size (for example 128; default: 0/off)",
    )
    parser.add_argument(
        "--int4_convrot_group_ratio_q8",
        action="store_true",
        help="Store grouped INT4 scale ratios as exact-mapping int16 Q8.8 instead of float32 (requires group scales).",
    )
    parser.add_argument(
        "--int4_convrot_compare_group_scales",
        default="",
        metavar="SIZES",
        help=(
            "Comma-separated group-scale sizes to measure in the quality report, for example 0,128,64. "
            "This is report-only and never changes the selected quantization parameters."
        ),
    )
    parser.add_argument(
        "--convrot_policy",
        default=None,
        help="Optional ltx2_convrot_policy_v1 JSON; quantize=false rules keep matching weights in floating point",
    )
    parser.add_argument(
        "--no_rotation",
        action="store_true",
        help=(
            "No-rotation mode: skip the online ConvRot Hadamard rotation entirely and INT4-quantize the raw "
            "(unrotated) weight residual. The low-rank SVD stabilizer then becomes the sole outlier-isolation "
            "mechanism, so --stabilizer_rank >= 1 is required. Removes the runtime rotation/inverse-rotation passes."
        ),
    )
    parser.add_argument(
        "--stabilizer_rank",
        type=int,
        default=0,
        help=(
            "Rank of the frozen low-rank stabilizer branch split off each rotated weight before INT4 "
            "quantization (low-rank SVD outlier isolation). 0 disables it; no nonzero rank is selected "
            "automatically by this standalone converter. "
            "The stabilizer tensors are stored in the output checkpoint and applied automatically at load."
        ),
    )
    parser.add_argument(
        "--int4_convrot_awq_calibration",
        action="store_true",
        help=(
            "Compute dataset-independent AWQ-style per-input-channel scales before INT4 ConvRot quantization. "
            "If --int4_convrot_awq_scales is omitted, writes <output>.int4_convrot_awq_scales.safetensors."
        ),
    )
    parser.add_argument(
        "--int4_convrot_awq_scales",
        default=None,
        help=(
            "Path to reusable AWQ scales. With --int4_convrot_awq_calibration this is the output path; "
            "without it, scales are loaded and applied."
        ),
    )
    parser.add_argument(
        "--int4_convrot_awq_alpha",
        type=float,
        default=0.25,
        help="INT4 ConvRot AWQ scaling strength (0=no effect, 1=full column-importance scaling; default 0.25).",
    )
    parser.add_argument(
        "--quality_report",
        default=None,
        help="Quality JSON path. Defaults to <output_model_without_ext>.quality.json unless --no_quality_report is set.",
    )
    parser.add_argument("--no_quality_report", action="store_true", help="Skip quality metric report generation")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.scale_refine_steps < 0:
        raise ValueError("--scale_refine_steps must be >= 0")
    validate_int4_convrot_scale_group_size(args.int4_convrot_group_scales)

    if args.export_format == "comfy_convrot_w4a4":
        for opt, flag in (
            (args.stabilizer_rank, "--stabilizer_rank"),
            (bool(args.int4_convrot_awq_calibration), "--int4_convrot_awq_calibration"),
            (bool(args.int4_convrot_awq_scales), "--int4_convrot_awq_scales"),
            (bool(args.no_rotation), "--no_rotation"),
            (bool(args.convrot_policy), "--convrot_policy"),
            (int(args.scale_refine_steps), "--scale_refine_steps"),
            (int(args.int4_convrot_group_scales), "--int4_convrot_group_scales"),
            (bool(args.int4_convrot_group_ratio_q8), "--int4_convrot_group_ratio_q8"),
            (bool(args.int4_convrot_compare_group_scales), "--int4_convrot_compare_group_scales"),
        ):
            if opt:
                raise ValueError(
                    f"{flag} is not supported with --export_format comfy_convrot_w4a4 "
                    "(the convrot_w4a4 format has no stabilizer/AWQ slot and always uses Hadamard rotation)"
                )
        export_comfy_convrot_w4a4(
            input_model=args.input_model,
            output_model=args.output_model,
            calc_device=args.calc_device,
            linear_dtype=args.comfy_linear_dtype,
        )
        return

    quality_report = None if args.no_quality_report else (args.quality_report or default_quality_report_path(args.output_model))
    if args.int4_convrot_group_ratio_q8 and not args.int4_convrot_group_scales:
        raise ValueError("--int4_convrot_group_ratio_q8 requires --int4_convrot_group_scales")
    quantize_model(
        input_model=args.input_model,
        output_model=args.output_model,
        calc_device=args.calc_device,
        groupsize=args.groupsize,
        mse_clip=not args.no_mse_clip,
        quality_report=quality_report,
        awq_calibration=bool(args.int4_convrot_awq_calibration),
        awq_alpha=float(args.int4_convrot_awq_alpha),
        awq_scales=args.int4_convrot_awq_scales,
        stabilizer_rank=int(args.stabilizer_rank),
        no_rotation=bool(args.no_rotation),
        policy_path=args.convrot_policy,
        scale_refine_steps=int(args.scale_refine_steps),
        group_scales=int(args.int4_convrot_group_scales),
        group_ratio_q8=bool(args.int4_convrot_group_ratio_q8),
        compare_group_scales=args.int4_convrot_compare_group_scales,
    )


if __name__ == "__main__":
    main()
