"""Cache MOSS-Image text encoder outputs: per caption the (L, 4096) final hidden state of Qwen3.5-9B on the chat template,
L the caption's own token count, as the vendor encodes. The captions of a batch (--batch_size) run as one right-padded
forward whose causal layers keep each caption's states (moss_utils.encode_prompts). See moss_image/moss_utils.py.

A caption file may hold several captions, separated by lines that start with ``---===---`` (--caption_separator, kohya
sd-scripts caab8bb): all are cached, and training draws one per step. ``--moss_edit`` (edit LoRAs) encodes each caption
after its reference images (the dataset's control images, through Qwen3.5's vision tower) and also the references with
an empty caption, the context a dropped caption trains with. The empty caption's context is written beside every
dataset's caches (caption dropout of text-to-image runs and control adapters). With --skip_existing a cache is
re-encoded when its caption text, its caption count or (edit) its references changed.
"""

import argparse
import logging
import os
from typing import Optional

import torch
from safetensors import safe_open

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_moss_image
from musubi_tuner.moss_image import moss_control, moss_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# metadata of the caches written here: the caption count and, for edit LoRAs, the references they were encoded with
CACHE_FORMAT_KEY = "moss_te_format"
CACHE_FORMAT = "2"
CAPTIONS_KEY = "moss_captions"
REFERENCES_KEY = "moss_edit_refs"


def item_references(item: ItemInfo, edit: bool) -> list:
    if not edit or item.control_content is None:
        return []
    return [content for content in item.control_content if content is not None]


def encode_and_save_batch(text_encoder, tokenizer, batch: list[ItemInfo], separator: Optional[str] = None, edit: bool = False):
    """Every caption of every item in ``batch`` (and, for edit LoRAs, each item's references-only context) in one pass,
    then one cache file per item."""
    prompts, images, owners = [], [], []
    for index, item in enumerate(batch):
        references = item_references(item, edit)
        for candidate in moss_utils.split_caption_candidates(item.caption, separator):
            prompts.append(candidate)
            images.append(references)
            owners.append((index, "caption"))
        if edit:
            prompts.append("")
            images.append(references)
            owners.append((index, "uncond"))
    if edit:
        contexts = moss_utils.encode_prompts_with_images(text_encoder, tokenizer, prompts, images)
    else:
        contexts = moss_utils.encode_prompts(text_encoder, tokenizer, prompts)
    captions: dict[int, list] = {}
    unconds: dict[int, torch.Tensor] = {}
    for (index, kind), ctx in zip(owners, contexts):
        ctx = ctx.to(torch.bfloat16)
        if kind == "caption":
            captions.setdefault(index, []).append(ctx)
        else:
            unconds[index] = ctx
    for index, item in enumerate(batch):
        candidates = captions[index]
        metadata = {CACHE_FORMAT_KEY: CACHE_FORMAT, CAPTIONS_KEY: str(len(candidates))}
        if edit:
            references = item_references(item, edit)
            if not references:
                logger.warning(f"{item.item_key}: no reference image for the edit caption; cached as a text-only caption")
            metadata[REFERENCES_KEY] = moss_utils.reference_signature(references)
        save_text_encoder_output_cache_moss_image(item, candidates[0], candidates[1:], unconds.get(index), metadata)


def cache_is_current(item: ItemInfo, separator: Optional[str], edit: bool) -> bool:
    """--skip_existing for MOSS-Image: the cache exists and was encoded from this caption text, as many captions as the
    file holds now and (edit) the same references; a text-to-image cache never serves an edit run or the other way."""
    path = item.text_encoder_output_cache_path
    if not path or not os.path.isfile(path):
        return False
    try:
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
    except Exception:  # noqa: BLE001 - an unreadable cache is re-encoded
        return False
    if metadata.get("caption1") != item.caption:
        return False
    cached = 1 + sum(1 for key in keys if key.startswith("varlen_ctx_vec_cand"))
    if cached != len(moss_utils.split_caption_candidates(item.caption, separator)):
        return False
    has_uncond = any(key.startswith("varlen_ctx_vec_uncond_") for key in keys)
    if edit:
        return has_uncond and metadata.get(REFERENCES_KEY) == moss_utils.reference_signature(item_references(item, edit))
    return not has_uncond and metadata.get(REFERENCES_KEY, "none") == "none"


def save_empty_caption_for_datasets(text_encoder, tokenizer, datasets, control_only: bool = False) -> None:
    """Caption dropout (text-to-image runs, control adapters) trains with the empty caption: its context goes beside the
    caches of every dataset (moss_control.EMPTY_CAPTION_FILE), encoded by this run's text encoder. ``control_only``
    writes it beside the caches of datasets with control maps only (control adapters)."""
    directories = sorted(
        {
            dataset.cache_directory
            for dataset in datasets
            if getattr(dataset, "cache_directory", None) and (not control_only or getattr(dataset, "has_control", False))
        }
    )
    if not directories:
        return
    empty = moss_utils.encode_prompts(text_encoder, tokenizer, [""])[0].to(torch.bfloat16)
    for directory in directories:
        path = moss_control.empty_caption_path(directory)
        moss_control.save_empty_caption(path, empty, ARCHITECTURE_MOSS_IMAGE_FULL)
        logger.info(f"Saved the empty caption's context for caption dropout: {path}")


def save_empty_caption_for_control_datasets(text_encoder, tokenizer, datasets) -> None:
    """A control adapter's caption dropout: the empty caption's context beside the caches of datasets with control maps."""
    save_empty_caption_for_datasets(text_encoder, tokenizer, datasets, control_only=True)


def moss_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3.5-9B ComfyUI file (BF16 or INT8 ConvRot)")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="dtype of the text encoder (default bfloat16)")
    parser.add_argument(
        "--low_vram_text_encoding",
        action="store_true",
        help="Keep Qwen3.5-9B in CPU RAM and run it on the GPU one decoder layer at a time: under 1 GB of VRAM instead "
             "of about 18 GB, fast on any CPU (17 GB of host-to-device copies per prompt)",
    )
    parser.add_argument(
        "--moss_edit",
        action="store_true",
        help="edit LoRAs: encode each caption after its reference images (the dataset's control images) through Qwen3.5's "
             "vision tower, and the references alone for caption dropout",
    )
    parser.add_argument(
        "--caption_separator",
        type=str,
        default=moss_utils.CAPTION_SEPARATOR,
        help=f"lines starting with this separate the captions of a caption file (default {moss_utils.CAPTION_SEPARATOR}; "
             "empty: a caption file is one caption)",
    )
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = moss_image_setup_parser(parser)
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
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MOSS_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets
    edit = bool(args.moss_edit)
    separator = args.caption_separator or None
    if edit and not any(getattr(dataset, "has_control", False) for dataset in datasets):
        raise ValueError("--moss_edit encodes each caption with its reference images: the dataset config needs a control_directory")

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder, tokenizer = moss_utils.load_text_encoder(
        args.text_encoder, dtype=te_dtype, device=device, stream=bool(args.low_vram_text_encoding), vision=edit
    )
    logger.info("Encoding with Qwen3.5-9B" + (" and its vision tower (edit references)" if edit else ""))

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, tokenizer, batch, separator=separator, edit=edit)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=edit,  # the references come with the images
        cache_is_current=lambda item: cache_is_current(item, separator, edit),
    )
    if not edit:
        save_empty_caption_for_datasets(text_encoder, tokenizer, datasets)
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


if __name__ == "__main__":
    main()
