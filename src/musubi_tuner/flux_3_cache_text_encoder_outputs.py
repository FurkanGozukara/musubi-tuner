"""Cache FLUX.3 [klein] text encoder outputs: per caption the (L, 20480) stack of Qwen3-VL-4B hidden layers 4..32,
L the caption's own multiple of 80 tokens (pad tokens included: the DiT has no text mask). See flux_3/flux3_utils.py."""

import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_flux_3
from musubi_tuner.flux_3 import flux3_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_encoder, tokenizer, batch: list[ItemInfo]):
    prompts = [item.caption for item in batch]
    contexts = flux3_utils.encode_prompts(text_encoder, tokenizer, prompts)
    for item, ctx in zip(batch, contexts):
        save_text_encoder_output_cache_flux_3(item, ctx.to(torch.bfloat16))


def flux_3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL-4B safetensors (ComfyUI or official layout)")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="dtype of the text encoder (default bfloat16)")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = flux_3_setup_parser(parser)
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
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_3)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder, tokenizer = flux3_utils.load_text_encoder(args.text_encoder, dtype=te_dtype, device=device)
    logger.info("Encoding with Qwen3-VL")

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
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


if __name__ == "__main__":
    main()
