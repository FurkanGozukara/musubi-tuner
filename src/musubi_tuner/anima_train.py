"""Full-parameter (dense) BF16 fine-tuning for Anima (circlestone-labs/Anima Base).

Reuses the LoRA trainer's data path, loss, sampling and saving lifecycle (``anima_train_network``) through
``training.dense_trainer``: every DiT weight trains in BF16 with fused per-tensor Adafactor updates during backward,
block swap with fused updates (``--block_swap_trainable_ring`` for the fastest swap) and dense resume. The LLM adapter
stays frozen unless ``--train_llm_adapter`` (the model card: "Don't train the LLM adapter"; diffusion-pipe's example
config sets its learning rate to 0). The checkpoint is written in the released layout (``net.`` prefix, the 685 tensors
of anima-base-v1.0, BF16), so it loads as the DiT here, in ComfyUI and in SwarmUI. It runs trainer_base's training loop,
whose network / accelerator / resume hooks the dense trainer overrides.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

import torch

from musubi_tuner.anima import anima_model
from musubi_tuner.anima_train_network import AnimaTrainerMixin, anima_setup_parser
from musubi_tuner.training.dense_trainer import DenseTrainerMixin, add_dense_arguments
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer

logger = logging.getLogger(__name__)


class AnimaTrainer(DenseTrainerMixin, AnimaTrainerMixin, NetworkTrainer):
    """Every Anima DiT weight in BF16 (the LLM adapter only with --train_llm_adapter); checkpoints in the released layout."""

    dense_model_name = "Anima"

    def dense_swap_blocks(self, transformer: torch.nn.Module):
        return transformer.blocks

    def dense_state_dict_for_save(self, transformer: torch.nn.Module) -> dict:
        return anima_model.comfy_state_dict(transformer.state_dict())

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        network = super()._build_network(args, accelerator, transformer, vae, weight_dtype)
        if not getattr(args, "train_llm_adapter", False):
            transformer.llm_adapter.requires_grad_(False)
            trainable = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
            logger.info("Anima: the LLM adapter stays frozen (--train_llm_adapter trains it); %.3fB parameters train", trainable / 1e9)
        return network

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = super().extra_metadata(args)
        metadata["ss_train_llm_adapter"] = str(bool(getattr(args, "train_llm_adapter", False)))
        return metadata


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = anima_setup_parser(parser)
    parser.description = "Full-parameter BF16 training for Anima"
    parser.add_argument(
        "--train_llm_adapter",
        action="store_true",
        help="also train the LLM adapter between the text encoder and the DiT (the model card advises against it)",
    )
    return add_dense_arguments(parser, "Anima", learning_rate=1e-5, output_name="anima_full_finetune")


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "bfloat16"  # the Qwen-Image VAE file's dtype
    AnimaTrainer().train(args)


if __name__ == "__main__":
    main()
