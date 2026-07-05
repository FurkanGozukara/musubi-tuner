#!/usr/bin/env python3
"""Merge an ltx2_train.py full-finetune checkpoint into a base LTX-2 bundle.

The full-parameter trainer saves a transformer-only checkpoint with internal key
naming (``model.<...>``) and without the ``config`` safetensors metadata. The
LTX-2 loaders (``load_ltx2_model`` in generation, training and caching scripts)
expect a self-contained bundle: comfy naming (``model.diffusion_model.<...>``),
VAE / vocoder / audio VAE / text-projection weights, and the ``config`` metadata
entry. This utility rebuilds such a bundle by overlaying the trained transformer
weights onto a copy of the base checkpoint:

    python ltx2_merge_fft_to_model.py \
        --fft_checkpoint output/full_ft/ltx23_fft.safetensors \
        --base_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
        --output output/full_ft/ltx23_fft_full.safetensors

The resulting file loads anywhere the base checkpoint does (``--ltx2_checkpoint``
for generation, caching, LoRA training on top of the finetune, etc.).
"""

import argparse
import logging
import os

from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRAINER_PREFIX = "model."
BUNDLE_PREFIX = "model.diffusion_model."


def merge_fft_to_model(fft_checkpoint: str, base_checkpoint: str, output: str) -> None:
    with safe_open(fft_checkpoint, framework="pt") as f:
        fft = {}
        for key in f.keys():
            if not key.startswith(TRAINER_PREFIX):
                raise ValueError(f"Unexpected non-transformer key in FFT checkpoint: {key}")
            fft[key.replace(TRAINER_PREFIX, BUNDLE_PREFIX, 1)] = f.get_tensor(key)
    logger.info("FFT checkpoint: %d transformer tensors", len(fft))

    with safe_open(base_checkpoint, framework="pt") as f:
        metadata = f.metadata() or {}
        merged = {}
        replaced = 0
        for key in f.keys():
            if key in fft:
                merged[key] = fft.pop(key)
                replaced += 1
            else:
                merged[key] = f.get_tensor(key)

    if fft:
        raise ValueError(f"{len(fft)} FFT keys not found in the base bundle (naming mismatch?); examples: {list(fft)[:5]}")
    logger.info("Replaced %d / %d bundle tensors with finetuned weights", replaced, len(merged))

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    save_file(merged, output, metadata=metadata)
    logger.info("Saved merged bundle: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fft_checkpoint", required=True, help="Transformer-only checkpoint saved by ltx2_train.py")
    parser.add_argument("--base_checkpoint", required=True, help="Base LTX-2 bundle (e.g. ltx-2.3-22b-dev.safetensors)")
    parser.add_argument("--output", required=True, help="Output path for the merged self-contained bundle")
    args = parser.parse_args()
    merge_fft_to_model(args.fft_checkpoint, args.base_checkpoint, args.output)


if __name__ == "__main__":
    main()
