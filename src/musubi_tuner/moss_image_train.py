"""Full-parameter (dense) BF16 fine-tuning for MOSS-Image (Base or SFT).

Reuses the LoRA trainer's data path, loss, sampling and saving lifecycle (``moss_image_train_network``) through
``training.dense_trainer``: every DiT weight trains in BF16 with fused per-tensor Adafactor updates during backward,
block swap with fused updates (``--block_swap_trainable_ring`` for the fastest swap) and dense resume. The checkpoint is
written in the vendor / ComfyUI layout (fused ``proj.qkv``, ``rope.bands``), so it loads as the DiT here, in ComfyUI and
in SwarmUI.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch

from musubi_tuner.moss_image import moss_model
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_image_setup_parser
from musubi_tuner.training.dense_trainer import DenseTrainerMixin, add_dense_arguments
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common


class MossImageTrainer(DenseTrainerMixin, MossImageNetworkTrainer):
    """Every MOSS-Image DiT weight in BF16; checkpoints in the layout ComfyUI and ``load_dit`` read."""

    dense_model_name = "MOSS-Image"

    def dense_swap_blocks(self, transformer: torch.nn.Module):
        return transformer.layers

    def dense_state_dict_for_save(self, transformer: torch.nn.Module) -> dict:
        return moss_model.fuse_qkv(transformer.state_dict(), transformer.params)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = moss_image_setup_parser(parser)
    parser.description = "Full-parameter BF16 training for MOSS-Image"
    return add_dense_arguments(parser, "MOSS-Image", learning_rate=1e-5, output_name="moss_image_full_finetune")


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"
    MossImageTrainer().train(args)


if __name__ == "__main__":
    main()
