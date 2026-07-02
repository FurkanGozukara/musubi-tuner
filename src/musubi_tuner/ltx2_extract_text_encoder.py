"""Extract a full-fine-tuned Gemma text encoder from an LTX-2 training checkpoint.

`--full_ft_train_text_encoder` saves the trained Gemma, the caption projection and the
embeddings connector(s) inside the DiT checkpoint under a ``text_encoder.`` prefix. This
tool splits those back out into:

  1. a standalone Gemma safetensors file (with the key convention of the original Gemma
     weights), which no longer carries the ``text_encoder.`` prefix and is not merged with
     the DiT, and
  2. a DiT-side patch that puts the trained caption projection and connector(s) back under
     the keys the inference text-encoder loader reads from the DiT checkpoint
     (``text_embedding_projection.*`` / ``model.diffusion_model.*_embeddings_connector.*``).

The projection and connector are trained together with Gemma under
``--full_ft_train_text_encoder`` but live in the DiT-checkpoint namespace at inference, so a
Gemma-only file is not enough on its own; the DiT patch carries those weights.
"""

import argparse
import logging
import os
from typing import Callable, Dict, Optional

import torch

from musubi_tuner.utils.safetensors_utils import LazyTensorForSave, MemoryEfficientSafeOpen, mem_eff_save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# safetensors header dtype string -> torch dtype
_STR2DT: Dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _name, _label in (("float8_e5m2", "F8_E5M2"), ("float8_e4m3fn", "F8_E4M3")):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _STR2DT[_label] = _dt

# How the trainer nests the text-encoder object inside the saved state dict.
# text_encoder.model.<Gemma3ForConditionalGeneration internal key> -> the raw Gemma weights
GEMMA_STRIP = "text_encoder.model."
PROJ_STRIP = "text_encoder.feature_extractor_linear."
VCONN_STRIP = "text_encoder.embeddings_connector."
ACONN_STRIP = "text_encoder.audio_embeddings_connector."

# DiT-checkpoint key prefixes read by the inference text-encoder loader (SDOps in the encoders).
PROJ_DIT_PREFIX = "text_embedding_projection."
VCONN_DIT_PREFIX = "model.diffusion_model.video_embeddings_connector."
ACONN_DIT_PREFIX = "model.diffusion_model.audio_embeddings_connector."


def gemma_key_comfyui(internal: str) -> str:
    """Map a Gemma3ForConditionalGeneration internal key to the ComfyUI/flattened convention.

    Inverse of the remapping in gemma/encoders/base_encoder.py::module_ops_from_gemma_root.
    """
    if internal.startswith("model.language_model."):
        return "model." + internal[len("model.language_model.") :]
    if internal.startswith("model.vision_tower."):
        # loader maps file "vision_model.X" -> "model.vision_tower.vision_model.X"
        return internal[len("model.vision_tower.") :]
    if internal.startswith("model.multi_modal_projector."):
        return "multi_modal_projector." + internal[len("model.multi_modal_projector.") :]
    return internal  # e.g. a tied lm_head; kept as-is


def _dtype_of(header: dict, key: str) -> torch.dtype:
    label = header[key]["dtype"]
    if label not in _STR2DT:
        raise ValueError(f"Unsupported safetensors dtype {label!r} for key {key!r}")
    return _STR2DT[label]


def _shape_of(header: dict, key: str) -> tuple:
    return tuple(header[key]["shape"])


def _write_subset(input_path: str, key_map: Dict[str, str], out_path: str, metadata: Optional[Dict[str, str]]) -> None:
    """Stream a {out_key: in_key} subset from input_path into out_path, one tensor at a time."""
    with MemoryEfficientSafeOpen(input_path) as reader:
        header = reader.header
        tensors: Dict[str, LazyTensorForSave] = {}
        for out_key, in_key in key_map.items():

            def _materialize(k: str = in_key) -> torch.Tensor:
                return reader.get_tensor(k)

            tensors[out_key] = LazyTensorForSave(
                shape=_shape_of(header, in_key),
                dtype=_dtype_of(header, in_key),
                materialize_fn=_materialize,
            )
        mem_eff_save_file(tensors, out_path, metadata=metadata)
    logger.info("Wrote %d tensors -> %s", len(key_map), out_path)


def _write_patched_dit(
    input_path: str, dit_path: str, overlay: Dict[str, str], out_path: str, metadata: Optional[Dict[str, str]]
) -> None:
    """Write a copy of dit_path with the overlay {dit_key: fft_key} tensors replaced/added."""
    with MemoryEfficientSafeOpen(dit_path) as dit_reader, MemoryEfficientSafeOpen(input_path) as fft_reader:
        dit_header = dit_reader.header
        fft_header = fft_reader.header
        tensors: Dict[str, LazyTensorForSave] = {}
        # base DiT tensors (skip any keys the overlay will replace)
        for key in dit_reader.keys():
            if key in overlay:
                continue

            def _mat_base(k: str = key) -> torch.Tensor:
                return dit_reader.get_tensor(k)

            tensors[key] = LazyTensorForSave(
                shape=_shape_of(dit_header, key), dtype=_dtype_of(dit_header, key), materialize_fn=_mat_base
            )
        # overlay trained projection/connector tensors from the FFT checkpoint
        replaced = 0
        added = 0
        base_keys = set(dit_reader.keys())
        for dit_key, fft_key in overlay.items():
            if dit_key in base_keys:
                replaced += 1
            else:
                added += 1

            def _mat_overlay(k: str = fft_key) -> torch.Tensor:
                return fft_reader.get_tensor(k)

            tensors[dit_key] = LazyTensorForSave(
                shape=_shape_of(fft_header, fft_key), dtype=_dtype_of(fft_header, fft_key), materialize_fn=_mat_overlay
            )
        if metadata is None:
            metadata = dit_reader.metadata() or None
        mem_eff_save_file(tensors, out_path, metadata=metadata)
    logger.info("Wrote patched DiT (%d replaced, %d added) -> %s", replaced, added, out_path)


def _partition(keys) -> dict:
    """Split checkpoint keys into gemma / projection / connector / other buckets."""
    gemma, proj, vconn, aconn, other_te, dit = {}, {}, {}, {}, [], []
    for key in keys:
        if key.startswith(GEMMA_STRIP):
            gemma[key] = key[len(GEMMA_STRIP) :]
        elif key.startswith(PROJ_STRIP):
            proj[key] = PROJ_DIT_PREFIX + key[len(PROJ_STRIP) :]
        elif key.startswith(ACONN_STRIP):
            aconn[key] = ACONN_DIT_PREFIX + key[len(ACONN_STRIP) :]
        elif key.startswith(VCONN_STRIP):
            vconn[key] = VCONN_DIT_PREFIX + key[len(VCONN_STRIP) :]
        elif key.startswith("text_encoder."):
            other_te.append(key)
        else:
            dit.append(key)
    return {"gemma": gemma, "proj": proj, "vconn": vconn, "aconn": aconn, "other_te": other_te, "dit": dit}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Full-FT checkpoint (.safetensors) with text_encoder.* keys.")
    p.add_argument("--output_gemma", default=None, help="Output path for the standalone Gemma file.")
    p.add_argument(
        "--key_format",
        choices=["comfyui", "hf", "both"],
        default="both",
        help="Key convention for the Gemma file. 'hf' = Gemma3ForConditionalGeneration internal keys "
        "(loadable via --gemma_safetensors); 'comfyui' = flattened model.layers.* / vision_model.* keys; "
        "'both' writes two files with .hf/.comfyui suffixes.",
    )
    p.add_argument("--dit_checkpoint", default=None, help="Base DiT checkpoint to overlay the trained proj/connector onto.")
    p.add_argument(
        "--output_dit",
        default=None,
        help="Output path. With --dit_checkpoint: full patched DiT. Without: a patch-only file containing "
        "just the trained projection/connector under DiT-inference keys.",
    )
    p.add_argument("--reference_gemma", default=None, help="Copy __metadata__ (e.g. tokenizer) from this Gemma file.")
    p.add_argument("--dry_run", action="store_true", help="Report the key partition and remapping without writing.")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    with MemoryEfficientSafeOpen(args.input) as reader:
        keys = reader.keys()
        input_metadata = reader.metadata() or {}
    part = _partition(keys)
    is_av = len(part["aconn"]) > 0

    logger.info("Checkpoint: %s", args.input)
    logger.info(
        "Partition: gemma=%d projection=%d video_connector=%d audio_connector=%d other_text_encoder=%d dit=%d",
        len(part["gemma"]),
        len(part["proj"]),
        len(part["vconn"]),
        len(part["aconn"]),
        len(part["other_te"]),
        len(part["dit"]),
    )
    logger.info("Detected mode: %s", "audio-video (av)" if is_av else "video-only")
    if not part["gemma"]:
        raise ValueError(
            "No 'text_encoder.model.*' keys found. This checkpoint was not trained with "
            "--full_ft_train_text_encoder, or uses a different layout."
        )
    if part["other_te"]:
        logger.warning("Unrecognized text_encoder.* keys (skipped): %s", part["other_te"][:8])

    # metadata for the standalone gemma file
    gemma_meta: Optional[Dict[str, str]] = None
    if args.reference_gemma:
        with MemoryEfficientSafeOpen(args.reference_gemma) as ref:
            gemma_meta = ref.metadata() or None
        logger.info("Copied metadata from reference: %s", args.reference_gemma)
    elif input_metadata:
        gemma_meta = {k: v for k, v in input_metadata.items()}

    def _sample(m: Dict[str, str], fn: Callable[[str], str] = lambda x: x, n: int = 3):
        return [f"{fn(v)}  <-  {k}" for k, v in list(m.items())[:n]]

    # Build the gemma key maps for each requested format.
    formats = ["hf", "comfyui"] if args.key_format == "both" else [args.key_format]
    gemma_maps: Dict[str, Dict[str, str]] = {}
    for fmt in formats:
        remap = (lambda x: x) if fmt == "hf" else gemma_key_comfyui
        gemma_maps[fmt] = {remap(internal): in_key for in_key, internal in part["gemma"].items()}
        logger.info("Gemma[%s] sample keys:\n  %s", fmt, "\n  ".join(_sample(part["gemma"], remap)))

    # Build the DiT overlay map (projection + connectors -> DiT-inference keys).
    overlay: Dict[str, str] = {}
    for bucket in ("proj", "vconn", "aconn"):
        for in_key, dit_key in part[bucket].items():
            overlay[dit_key] = in_key
    if overlay:
        logger.info(
            "DiT overlay sample keys:\n  %s",
            "\n  ".join(f"{part_v}  <-  {k}" for k, part_v in [(k, k) for k in list(overlay.keys())[:4]]),
        )

    if args.dry_run:
        logger.info("Dry run: no files written.")
        return

    # ---- write standalone Gemma file(s) ----
    if args.output_gemma:
        base, ext = os.path.splitext(args.output_gemma)
        ext = ext or ".safetensors"
        for fmt, kmap in gemma_maps.items():
            out = args.output_gemma if len(gemma_maps) == 1 else f"{base}.{fmt}{ext}"
            _write_subset(args.input, kmap, out, gemma_meta)
    else:
        logger.info("No --output_gemma given; skipping Gemma export.")

    # ---- write DiT patch ----
    if args.output_dit:
        if not overlay:
            logger.warning("No projection/connector tensors found; nothing to patch into the DiT.")
        elif args.dit_checkpoint:
            _write_patched_dit(args.input, args.dit_checkpoint, overlay, args.output_dit, None)
        else:
            _write_subset(args.input, overlay, args.output_dit, None)
            logger.info("Wrote patch-only file (merge these keys into your DiT checkpoint before inference).")
    else:
        logger.info("No --output_dit given; skipping DiT patch.")


if __name__ == "__main__":
    main()
