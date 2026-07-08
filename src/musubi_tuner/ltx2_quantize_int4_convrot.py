#!/usr/bin/env python3
"""Pre-quantize LTX-2 transformer weights to packed INT4 ConvRot."""

from __future__ import annotations

import argparse
import logging
import os
import time

import safetensors
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner.ltx2_model_loading import KEEP_FP8_HIGH_PRECISION_TOKENS
from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from musubi_tuner.modules.int4_convrot_utils import (
    best_int4_convrot_groupsize,
    comfy_quant_tensor,
    parse_int4_convrot_groupsizes,
    quantize_int4_convrot_weight,
    summarize_quality,
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
) -> None:
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    groupsizes = parse_int4_convrot_groupsizes(groupsize)
    device = torch.device(calc_device)
    logger.info("INT4 ConvRot quantization device: %s", device)
    logger.info("INT4 ConvRot group candidates: %s", ", ".join(str(g) for g in groupsizes))
    logger.info("INT4 ConvRot MSE clipping: %s", "on" if mse_clip else "off")
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
            if not quantizable:
                if key.endswith(".weight") and value.ndim == 2 and any(t in model_key for t in _INT4_CONVROT_TARGET_PATTERNS):
                    skipped_count += 1
                else:
                    passthrough_count += 1
                state_dict[key] = value
                continue

            assert group_size is not None
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

            q, scale, shape, quality = quantize_int4_convrot_weight(
                value,
                group_size=group_size,
                calc_device=device,
                mse_clip=mse_clip,
                collect_quality=quality_report is not None,
                key=model_key,
            )
            base = key[: -len(".weight")]
            in_features = int(shape.detach().cpu().reshape(-1)[1].item())
            padded_features = int(shape.detach().cpu().reshape(-1)[2].item())
            state_dict[key] = q.cpu()
            state_dict[base + ".weight_scale"] = scale.cpu()
            state_dict[base + ".int4_shape"] = shape.cpu()
            state_dict[base + ".comfy_quant"] = comfy_quant_tensor(
                group_size, in_features, padded_features, awq=awq_scale is not None
            )
            if awq_scale is not None:
                state_dict[base + INT4_CONVROT_AWQ_SCALE_SUFFIX] = awq_scale.cpu().float()
            if quality is not None:
                quality_layers.append(quality)
            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)

    output_metadata = dict(original_metadata)
    output_metadata["int4_convrot_quantized"] = "true"
    output_metadata["int4_convrot_groupsizes"] = ",".join(str(g) for g in groupsizes)
    output_metadata["int4_convrot_mse_clip"] = "true" if mse_clip else "false"
    output_metadata["int4_convrot_storage"] = "packed_signed_int4_low_high"
    output_metadata["int4_convrot_awq"] = "true" if (awq_calibration or loaded_awq_scales is not None) else "false"
    output_metadata["int4_convrot_awq_alpha"] = str(float(awq_alpha))

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
            },
            layers=quality_layers,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-quantize LTX-2 model weights to packed INT4 ConvRot")
    parser.add_argument("--input_model", required=True, help="Path to original .safetensors checkpoint")
    parser.add_argument("--output_model", required=True, help="Path for INT4 ConvRot output .safetensors")
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
    quality_report = None if args.no_quality_report else (args.quality_report or default_quality_report_path(args.output_model))
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
    )


if __name__ == "__main__":
    main()
