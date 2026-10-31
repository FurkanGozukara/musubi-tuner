"""Cache Anima text encoder outputs: per caption the Qwen3-0.6B last hidden state (L, 1024) of its Qwen tokens (no
special tokens, unpadded) and its T5 token ids with the closing </s> (anima/anima_utils.py). The LLM adapter that turns
them into the DiT's 512-token context is part of the DiT and runs in training."""

import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.anima import anima_utils
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_ANIMA
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_anima
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.utils.model_utils import cpu_text_encoder_dtype, str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_encoder, qwen_tokenizer, t5_tokenizer, batch: list[ItemInfo], save_dtype: torch.dtype):
    prompts = [item.caption for item in batch]
    encoded = anima_utils.encode_prompts(text_encoder, qwen_tokenizer, t5_tokenizer, prompts, batch_size=len(prompts))
    for item, (states, t5_ids) in zip(batch, encoded):
        save_text_encoder_output_cache_anima(item, states.to(save_dtype), t5_ids)


def anima_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-0.6B base (qwen_3_06b_base.safetensors)")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="dtype of the text encoder (default bfloat16)")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = anima_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    te_dtype = torch.bfloat16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ANIMA)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    save_dtype = te_dtype  # the caches keep the requested dtype when a slow-BF16 CPU encodes in FP32
    run_dtype = cpu_text_encoder_dtype(device, te_dtype, args.text_encoder, "Qwen3-0.6B text encoder")
    text_encoder = anima_utils.load_text_encoder(args.text_encoder, dtype=run_dtype, device=device)
    qwen_tokenizer, t5_tokenizer = anima_utils.load_tokenizers()
    logger.info("Encoding with Qwen3-0.6B")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, qwen_tokenizer, t5_tokenizer, batch, save_dtype)

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
