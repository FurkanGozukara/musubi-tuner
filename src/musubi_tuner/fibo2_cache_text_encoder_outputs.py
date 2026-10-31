"""Cache FIBO 2.0 text encoder outputs: per caption the (L, 15 * 2560) RMS-normalized Qwen3-VL-4B taps behind Bria's
six bundles, L the caption's own token count (see fibo2/fibo2_utils.py). Editing datasets (reference images) encode each
caption as Bria's edit prompt with its references through Qwen3-VL's vision tower: the references are the dataset's
crops (the ones latent caching encodes), so a cache records their sizes and ``--skip_existing`` re-encodes a caption
whose references changed (another resolution or count, references added or removed)."""

import argparse
import logging
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_FIBO2
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_fibo2
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.fibo2 import fibo2_utils
from musubi_tuner.utils.model_utils import cast_for_cache, cpu_text_encoder_dtype, mark_cpu_fp32_fallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def item_references(item: ItemInfo) -> list[np.ndarray]:
    """The item's reference crops ((H, W, C) arrays), in order; none for a text-to-image item."""
    content = item.control_content
    if content is None:
        return []
    if isinstance(content, np.ndarray) and content.ndim == 3:
        return [content]
    return list(content)


def reference_sizes_text(references) -> str:
    return ";".join(f"{ref.shape[1]}x{ref.shape[0]}" for ref in references)


def cache_is_current(item: ItemInfo) -> bool:
    """--skip_existing: the cache exists, was written by this recipe and with the same reference crops."""
    try:
        with safe_open(item.text_encoder_output_cache_path, framework="pt") as handle:
            metadata = handle.metadata() or {}
    except Exception:  # noqa: BLE001 - missing or unreadable: encode again
        return False
    if metadata.get("fibo2_text") != "taps15_v1":
        return False
    return metadata.get("fibo2_references", "") == reference_sizes_text(item_references(item))


def encode_and_save_batch(text_encoder, tokenizer, image_processor, batch: list[ItemInfo], batch_size: Optional[int] = None):
    prompts = [item.caption for item in batch]
    references = [item_references(item) for item in batch]
    images = [[fibo2_utils.vlm_image(fibo2_utils.to_pil(ref)) for ref in refs] or None for refs in references]
    taps = fibo2_utils.encode_prompts(
        text_encoder, tokenizer, image_processor, prompts, images, batch_size=batch_size or fibo2_utils.TEXT_BATCH_SIZE
    )
    for item, item_taps, refs in zip(batch, taps, references):
        sizes = [(ref.shape[1], ref.shape[0]) for ref in refs]
        save_text_encoder_output_cache_fibo2(item, cast_for_cache(text_encoder, item_taps.cpu()).to(torch.bfloat16), sizes)


def fibo2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL-4B safetensors (ComfyUI or official layout)")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="dtype of the text encoder (default bfloat16)")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = fibo2_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FIBO2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets
    # editing datasets: the captions are encoded with their references, which come with the images
    with_references = any(getattr(dataset, "has_control", False) for dataset in datasets)

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    requested_dtype = te_dtype
    te_dtype = cpu_text_encoder_dtype(device, te_dtype, args.text_encoder, "Qwen3-VL text encoder")  # saved as bf16 below
    text_encoder, tokenizer, image_processor = fibo2_utils.load_text_encoder(args.text_encoder, dtype=te_dtype, device=device)
    mark_cpu_fp32_fallback(text_encoder, requested_dtype, te_dtype)
    logger.info("Encoding with Qwen3-VL" + (" (edit prompts with their reference images)" if with_references else ""))

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, tokenizer, image_processor, batch, args.batch_size)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=with_references,
        cache_is_current=cache_is_current,
    )
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


if __name__ == "__main__":
    main()
