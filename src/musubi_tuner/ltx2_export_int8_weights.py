#!/usr/bin/env python3
"""Pre-quantize LTX-2 transformer weights to the INT8 weight-only (Int8QTWeight) FFT grid.

The output is a compact sidecar checkpoint that carries only the quantized target layers
(``<fqn>.int_data`` / ``<fqn>.scale`` / ``<fqn>.convrot_group`` + optional dense-sparse
side vectors). Point a full fine-tuning run at the original bf16 checkpoint as usual and
add ``--int8_weights_prequant <sidecar>`` so the trainer rebuilds each Int8QTWeight from
these tensors instead of running the slow startup quantization (Hadamard rotation +
outlier scan over every target matrix).

The grid flags here MUST match the training flags (``--int8_weights_group_size``,
``--int8_weights_targets``, ``--int8_weights_convrot``, ``--int8_weights_outlier_quantile``,
``--int8_weights_sparse_ratio``, ``--int8_weights_min_numel``) and the weight dtype; the
trainer validates this and refuses a mismatched file. The quantization math is the same
deterministic ``Int8QTWeight.from_float`` the trainer uses, so a matched sidecar is
bit-identical to on-the-fly quantization.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import safetensors
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from musubi_tuner.modules.fp8_training import ltx2_fp8_filter
from musubi_tuner.modules.int8_training import Int8QTWeight
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def quantize_model(
    input_model: str,
    output_model: str,
    *,
    targets: str,
    min_numel: int,
    group_size: int,
    outlier_quantile: float,
    sparse_ratio: float,
    convrot: str | int,
    dtype: str,
    calc_device: str,
) -> None:
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    if group_size < 0:
        raise ValueError(f"--group_size must be >= 0, got {group_size}")
    # mirror the trainer's w8a8-independent guards so a sidecar can never encode an invalid grid
    if sparse_ratio and sparse_ratio > 0.0 and group_size and group_size > 0:
        logger.warning("Both group_size>0 and sparse_ratio>0 set; this matches convert_to_int8_training but is unusual.")

    convrot_spec: str | int = convrot if (convrot not in ("", None)) else 0
    torch_dtype = _DTYPES[dtype]
    device = torch.device(calc_device)
    # _keep(mod, fqn) ignores mod (name-based), so a dummy module reference is fine here.
    _target = ltx2_fp8_filter(targets, min_numel)

    def _is_target(fqn: str, numel: int) -> bool:
        return bool(_target(None, fqn)) and numel >= min_numel

    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    logger.info("INT8 weight-only pre-quantization device=%s dtype=%s", device, dtype)
    logger.info(
        "grid: targets=%s min_numel=%d group_size=%d outlier_quantile=%g sparse_ratio=%g convrot=%s",
        targets,
        min_numel,
        group_size,
        outlier_quantile,
        sparse_ratio,
        convrot_spec,
    )

    state_dict: dict[str, torch.Tensor] = {}
    quantized_count = 0
    skipped_count = 0
    t0 = time.time()

    with MemoryEfficientSafeOpen(input_model) as f:
        keys = list(f.keys())
        fp8_scale_keys = {k for k in keys if k.endswith(".weight_scale") or k.endswith(".input_scale")}
        for key in tqdm(keys, desc="Pre-quantizing INT8 weights", unit="tensor"):
            if key in fp8_scale_keys:
                continue
            if not key.endswith(".weight"):
                continue
            value = f.get_tensor(key)
            # dequantize an FP8-scaled source weight before quantizing, so a scaled-FP8 checkpoint works too
            if value.is_floating_point() and value.dtype.itemsize == 1:
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in fp8_scale_keys:
                    raise ValueError(f"FP8 weight without weight_scale: {key}. Use a bf16/fp16/fp32 or scaled-FP8 checkpoint.")
                value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)
            if value.ndim != 2:
                continue
            renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
            model_key = renamed if renamed is not None else key
            base = model_key[: -len(".weight")]
            if not _is_target(base, value.numel()):
                continue

            # cast to the training weight dtype BEFORE from_float: the trainer quantizes the
            # already-cast module.weight, so matching the dtype here keeps the grid bit-identical.
            w = value.to(device=device, dtype=torch_dtype)
            qt = Int8QTWeight.from_float(
                w,
                group_size,
                outlier_clip_quantile=outlier_quantile,
                sparse_ratio=sparse_ratio,
                convrot_group=convrot_spec,
            )
            state_dict[base + ".int_data"] = qt.int_data.cpu().contiguous()
            state_dict[base + ".scale"] = qt.scale.cpu().contiguous()
            state_dict[base + ".convrot_group"] = torch.tensor(int(qt.convrot_group), dtype=torch.int32)
            if qt.sparse_val is not None and qt.sparse_val.numel() > 0:
                state_dict[base + ".sparse_idx"] = qt.sparse_idx.cpu().contiguous()
                state_dict[base + ".sparse_val"] = qt.sparse_val.cpu().contiguous()
            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)
            del w, qt

    if quantized_count == 0:
        raise ValueError("No target Linear weights matched; check --targets / --min_numel against the checkpoint.")

    output_metadata = dict(original_metadata)
    output_metadata["int8_weights_prequant"] = "true"
    output_metadata["group_size"] = str(int(group_size))
    output_metadata["outlier_clip_quantile"] = str(float(outlier_quantile))
    output_metadata["sparse_ratio"] = str(float(sparse_ratio))
    output_metadata["convrot"] = str(convrot_spec)
    output_metadata["w8a8_compute"] = "runtime"  # compute mode is chosen at train time, not baked into the grid
    output_metadata["targets"] = str(targets)
    output_metadata["min_numel"] = str(int(min_numel))
    output_metadata["dtype"] = dtype

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Saving pre-quantized INT8 weights sidecar to %s", output_model)
    save_file(state_dict, output_model, metadata=output_metadata)

    elapsed = time.time() - t0
    input_size = os.path.getsize(input_model) / (1024**3)
    output_size = os.path.getsize(output_model) / (1024**3)
    logger.info(
        "INT8 weight-only pre-quant complete in %.1fs: quantized=%d skipped=%d size=%.2fGB -> %.2fGB",
        elapsed,
        quantized_count,
        skipped_count,
        input_size,
        output_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-quantize LTX-2 weights to the INT8 weight-only FFT grid")
    parser.add_argument("--input_model", required=True, help="Path to original .safetensors checkpoint")
    parser.add_argument("--output_model", required=True, help="Path for the INT8 pre-quantized sidecar .safetensors")
    parser.add_argument("--targets", default="video", help="Target selection (matches --int8_weights_targets)")
    parser.add_argument(
        "--min_numel", type=int, default=16384, help="Min weight numel to quantize (matches --int8_weights_min_numel)"
    )
    parser.add_argument("--group_size", type=int, default=0, help="Weight-scale group size (matches --int8_weights_group_size)")
    parser.add_argument(
        "--outlier_quantile", type=float, default=1.0, help="Outlier clip quantile (matches --int8_weights_outlier_quantile)"
    )
    parser.add_argument(
        "--sparse_ratio", type=float, default=0.0, help="Dense-sparse outlier ratio (matches --int8_weights_sparse_ratio)"
    )
    parser.add_argument("--convrot", default="auto", help="ConvRot group spec: auto / int / 0 (matches --int8_weights_convrot)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=list(_DTYPES.keys()),
        help="Weight dtype to quantize at; MUST match the training weight dtype",
    )
    parser.add_argument(
        "--calc_device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the quantization math (default: cuda if available else cpu). Result is device-independent.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    quantize_model(
        input_model=args.input_model,
        output_model=args.output_model,
        targets=args.targets,
        min_numel=args.min_numel,
        group_size=args.group_size,
        outlier_quantile=args.outlier_quantile,
        sparse_ratio=args.sparse_ratio,
        convrot=args.convrot,
        dtype=args.dtype,
        calc_device=args.calc_device,
    )


if __name__ == "__main__":
    main()
