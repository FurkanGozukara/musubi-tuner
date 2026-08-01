#!/usr/bin/env python3
"""Pre-quantize LTX-2 transformer weights to packed NVFP4 for W4A4G4 LoRA training."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import logging
import os
import time

import safetensors
import torch
from tqdm import tqdm

from musubi_tuner.ltx2_model_loading import KEEP_FP8_HIGH_PRECISION_TOKENS
from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from musubi_tuner.modules.nvfp4_training import (
    NVFP4LayerQuality,
    NVFP4_TARGET_PATTERNS,
    NVFP4_TRAINING_METADATA_MARKER,
    NVFP4_TRAINING_STABILIZER_RANK_METADATA,
    is_nvfp4_target_key,
    quantize_nvfp4_training_tensor,
    summarize_nvfp4_quality,
    write_nvfp4_quality_report,
)
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, StreamingSafetensorsWriter, safetensors_resume_id

logger = logging.getLogger(__name__)


def _is_quantizable(key: str, value: torch.Tensor) -> tuple[bool, str]:
    renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
    model_key = renamed if renamed is not None else key
    quantizable = is_nvfp4_target_key(model_key, value, exclude_tokens=KEEP_FP8_HIGH_PRECISION_TOKENS)
    return quantizable, model_key


def default_quality_report_path(output_model: str) -> str:
    base, _ = os.path.splitext(output_model)
    return f"{base}.quality.json"


def _layer_seed_context(model_key: str, seed: int, device: torch.device):
    digest = hashlib.sha256(f"{int(seed)}:{model_key}".encode("utf-8")).digest()
    layer_seed = int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    context = torch.random.fork_rng(devices=devices)

    class _SeedContext:
        def __enter__(self):
            context.__enter__()
            torch.manual_seed(layer_seed)

        def __exit__(self, exc_type, exc_val, exc_tb):
            return context.__exit__(exc_type, exc_val, exc_tb)

    return _SeedContext()


def quantize_model(
    input_model: str,
    output_model: str,
    *,
    calc_device: str,
    quality_report: str | None,
    stabilizer_rank: int = 32,
    resume: bool = False,
    resume_seed: int = 0,
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

    output_metadata = dict(original_metadata)
    output_metadata[NVFP4_TRAINING_METADATA_MARKER] = "true"
    output_metadata["nvfp4_training_storage"] = "packed_e2m1_tile16_scales"
    if stabilizer_rank > 0:
        output_metadata[NVFP4_TRAINING_STABILIZER_RANK_METADATA] = str(int(stabilizer_rank))

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    resume_fingerprint = safetensors_resume_id(
        input_model,
        {
            "format": "ltx2_nvfp4_training",
            "calc_device": str(device),
            "stabilizer_rank": int(stabilizer_rank),
            "collect_quality": quality_report is not None,
            "resume_seed": int(resume_seed),
        },
    )
    t0 = time.time()

    with (
        MemoryEfficientSafeOpen(input_model) as f,
        StreamingSafetensorsWriter(
            output_model,
            metadata=output_metadata,
            resume=resume,
            resume_id=resume_fingerprint,
        ) as writer,
    ):
        progress = writer.progress
        quality_by_key = dict(progress.get("quality_by_key", {}))
        quantized_count = int(progress.get("quantized_count", 0))
        skipped_count = int(progress.get("skipped_count", 0))
        passthrough_count = int(progress.get("passthrough_count", 0))
        groups_since_checkpoint = 0

        def progress_state() -> dict:
            return {
                "quality_by_key": quality_by_key,
                "quantized_count": quantized_count,
                "skipped_count": skipped_count,
                "passthrough_count": passthrough_count,
            }

        keys = list(f.keys())
        fp8_scale_keys = {key for key in keys if key.endswith(".weight_scale") or key.endswith(".input_scale")}
        if fp8_scale_keys:
            logger.info("Detected %d FP8 scale tensors; FP8 weights will be dequantized before NVFP4", len(fp8_scale_keys))
        for key in tqdm(keys, desc="Quantizing NVFP4", unit="tensor"):
            if key in fp8_scale_keys:
                continue
            if writer.is_group_complete(key):
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
                writer.write_tensor(key, value)
                writer.mark_group_complete(key)
                groups_since_checkpoint += 1
                if groups_since_checkpoint >= 32:
                    writer.checkpoint(progress=progress_state())
                    groups_since_checkpoint = 0
                continue

            seed_context = _layer_seed_context(model_key, resume_seed, device) if resume and stabilizer_rank > 0 else nullcontext()
            with seed_context:
                entries, quality = quantize_nvfp4_training_tensor(
                    value,
                    stabilizer_rank=stabilizer_rank,
                    calc_device=device,
                    collect_quality=quality_report is not None,
                    key=model_key,
                )
            base = key[: -len(".weight")]
            for suffix, tensor in entries.items():
                writer.write_tensor(base + suffix, tensor.cpu())
            if quality is not None:
                quality_by_key[model_key] = asdict(quality)
            quantized_count += 1
            writer.mark_group_complete(key)
            writer.checkpoint(progress=progress_state())
            groups_since_checkpoint = 0
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)
        logger.info("Finalizing streamed NVFP4 training checkpoint at %s", output_model)
        writer.finalize(progress=progress_state())

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

    quality_layers = [NVFP4LayerQuality(**quality_by_key[key]) for key in sorted(quality_by_key)]
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep an incomplete output plus a journal and resume from the last durable tensor group. "
            "The source file and output-affecting options must remain unchanged."
        ),
    )
    parser.add_argument(
        "--resume_seed",
        type=int,
        default=0,
        help="Base seed for deterministic per-layer stabilizer SVD when --resume is enabled (default: 0)",
    )
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
        resume=bool(args.resume),
        resume_seed=int(args.resume_seed),
    )


if __name__ == "__main__":
    main()
