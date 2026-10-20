"""Cache MOSS-Image text encoder outputs: per caption the (L, 4096) final hidden state of Qwen3.5-9B on the chat template,
L the caption's own token count, as the vendor encodes. The captions of a batch (--batch_size) run as one right-padded
forward whose causal layers keep each caption's states (moss_utils.encode_prompts). See moss_image/moss_utils.py."""

import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_moss_image
from musubi_tuner.moss_image import moss_control, moss_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_encoder, tokenizer, batch: list[ItemInfo]):
    prompts = [item.caption for item in batch]
    contexts = moss_utils.encode_prompts(text_encoder, tokenizer, prompts)
    for item, ctx in zip(batch, contexts):
        save_text_encoder_output_cache_moss_image(item, ctx.to(torch.bfloat16))


def save_empty_caption_for_control_datasets(text_encoder, tokenizer, datasets) -> None:
    """A control adapter's caption dropout trains with the empty caption: its context goes beside the caches of every
    dataset with control maps (moss_control.EMPTY_CAPTION_FILE), encoded by this run's text encoder."""
    directories = sorted({dataset.cache_directory for dataset in datasets if getattr(dataset, "has_control", False) and dataset.cache_directory})
    if not directories:
        return
    empty = moss_utils.encode_prompts(text_encoder, tokenizer, [""])[0].to(torch.bfloat16)
    for directory in directories:
        path = moss_control.empty_caption_path(directory)
        moss_control.save_empty_caption(path, empty, ARCHITECTURE_MOSS_IMAGE_FULL)
        logger.info(f"Saved the empty caption's context for the control adapter's caption dropout: {path}")


def moss_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3.5-9B ComfyUI file (BF16 or INT8 ConvRot)")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="dtype of the text encoder (default bfloat16)")
    parser.add_argument(
        "--low_vram_text_encoding",
        action="store_true",
        help="Keep Qwen3.5-9B in CPU RAM and run it on the GPU one decoder layer at a time: under 1 GB of VRAM instead "
             "of about 18 GB, fast on any CPU (17 GB of host-to-device copies per prompt)",
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

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder, tokenizer = moss_utils.load_text_encoder(
        args.text_encoder, dtype=te_dtype, device=device, stream=bool(args.low_vram_text_encoding)
    )
    logger.info("Encoding with Qwen3.5-9B")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, tokenizer, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    save_empty_caption_for_control_datasets(text_encoder, tokenizer, datasets)
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


if __name__ == "__main__":
    main()
