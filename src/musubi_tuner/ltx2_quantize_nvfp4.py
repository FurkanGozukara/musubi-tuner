#!/usr/bin/env python3
"""Pre-quantize LTX-2 transformer weights to packed NVFP4 for W4A4G4 LoRA training."""

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
from musubi_tuner.modules.nvfp4_training import (
    NVFP4_TARGET_PATTERNS,
    NVFP4_TRAINING_METADATA_MARKER,
    NVFP4_TRAINING_STABILIZER_RANK_METADATA,
    is_nvfp4_target_key,
    quantize_nvfp4_training_tensor,
    summarize_nvfp4_quality,
    write_nvfp4_quality_report,
)
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)


def _is_quantizable(key: str, value: torch.Tensor) -> tuple[bool, str]:
    renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
    model_key = renamed if renamed is not None else key
    quantizable = is_nvfp4_target_key(model_key, value, exclude_tokens=KEEP_FP8_HIGH_PRECISION_TOKENS)
    return quantizable, model_key


def default_quality_report_path(output_model: str) -> str:
    base, _ = os.path.splitext(output_model)
    return f"{base}.quality.json"


def quantize_model(
    input_model: str,
    output_model: str,
    *,
    calc_device: str,
    quality_report: str | None,
    stabilizer_rank: int = 32,
) -> None:
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")
    if stabilizer_rank < 0:
        raise ValueError(f"NVFP4 stabilizer rank must be >= 0, got {stabilizer_rank}")

    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    device = torch.device(calc_device)
    logger.info("NVFP4 training quantization device: %s", device)
    if stabilizer_rank > 0:
        logger.info("NVFP4 stabilizer: rank %d low-rank branch (SVD of the weight)", stabilizer_rank)

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
            logger.info("Detected %d FP8 scale tensors; FP8 weights will be dequantized before NVFP4", len(fp8_scale_keys))
        for key in tqdm(keys, desc="Quantizing NVFP4", unit="tensor"):
            if key in fp8_scale_keys:
                continue
            value = f.get_tensor(key)
            if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in fp8_scale_keys:
                    raise ValueError(
                        f"NVFP4 source has FP8 weight without weight_scale: {key}. "
                        "Use a bf16/fp16 checkpoint or a scaled FP8 checkpoint with matching scale tensors."
                    )
                value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)

            quantizable, model_key = _is_quantizable(key, value)
            if not quantizable:
                if key.endswith(".weight") and value.ndim == 2 and any(t in model_key for t in NVFP4_TARGET_PATTERNS):
                    skipped_count += 1
                else:
                    passthrough_count += 1
                state_dict[key] = value
                continue

            entries, quality = quantize_nvfp4_training_tensor(
                value,
                stabilizer_rank=stabilizer_rank,
                calc_device=device,
                collect_quality=quality_report is not None,
                key=model_key,
            )
            base = key[: -len(".weight")]
            for suffix, tensor in entries.items():
                state_dict[base + suffix] = tensor.cpu()
            if quality is not None:
                quality_layers.append(quality)
            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)

    output_metadata = dict(original_metadata)
    output_metadata[NVFP4_TRAINING_METADATA_MARKER] = "true"
    output_metadata["nvfp4_training_storage"] = "packed_e2m1_tile16_scales"
    if stabilizer_rank > 0:
        output_metadata[NVFP4_TRAINING_STABILIZER_RANK_METADATA] = str(int(stabilizer_rank))

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Saving NVFP4 training checkpoint to %s", output_model)
    save_file(state_dict, output_model, metadata=output_metadata)

    elapsed = time.time() - t0
    input_size = os.path.getsize(input_model) / (1024**3)
    output_size = os.path.getsize(output_model) / (1024**3)
    logger.info(
        "NVFP4 quantization complete in %.1fs: quantized=%d skipped=%d passthrough=%d size=%.2fGB -> %.2fGB",
        elapsed,
        quantized_count,
        skipped_count,
        passthrough_count,
        input_size,
        output_size,
    )

    if quality_report is not None:
        report = write_nvfp4_quality_report(
            quality_report,
            source=input_model,
            output=output_model,
            options={
                "mode": "prequantize",
                "target_keys": list(NVFP4_TARGET_PATTERNS),
                "exclude_keys": list(KEEP_FP8_HIGH_PRECISION_TOKENS),
                "calc_device": str(device),
                "storage": "packed_e2m1_tile16_scales",
                "stabilizer_rank": int(stabilizer_rank),
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
        logger.info("NVFP4 quality summary: %s", summarize_nvfp4_quality(quality_layers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-quantize LTX-2 model weights to packed NVFP4 (W4A4G4 training)")
    parser.add_argument("--input_model", required=True, help="Path to original .safetensors checkpoint")
    parser.add_argument(
        "--output_model",
        default=None,
        help="Path for NVFP4 output .safetensors (default: <input>.nvfp4t.safetensors)",
    )
    parser.add_argument(
        "--calc_device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for SVD/quantization math (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--stabilizer_rank",
        type=int,
        default=32,
        help=(
            "Rank of the frozen low-rank stabilizer branch split off each weight before NVFP4 quantization "
            "(low-rank SVD outlier isolation). 0 disables it; 32 is the default. Stored in the checkpoint "
            "and applied automatically at load."
        ),
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
    output_model = args.output_model
    if output_model is None:
        base, _ = os.path.splitext(args.input_model)
        output_model = f"{base}.nvfp4t.safetensors"
    quality_report = None if args.no_quality_report else (args.quality_report or default_quality_report_path(output_model))
    quantize_model(
        input_model=args.input_model,
        output_model=output_model,
        calc_device=args.calc_device,
        quality_report=quality_report,
        stabilizer_rank=int(args.stabilizer_rank),
    )


if __name__ == "__main__":
    main()
