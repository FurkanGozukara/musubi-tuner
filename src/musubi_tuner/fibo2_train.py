"""Full-parameter (dense) BF16 fine-tuning for FIBO 2.0 Dev.

Reuses the LoRA trainer's data path, loss, sampling and saving lifecycle (``fibo2_train_network``) through
``training.dense_trainer``: every DiT weight trains in BF16 with fused per-tensor Adafactor updates during backward,
block swap with fused updates (``--block_swap_trainable_ring`` for the fastest swap; the blocks' recompute waits for its
weights through ``recompute_wait``) and dense resume. The model's own state dict is Comfy-Org's layout (fused qkv / to_kv /
w13), so the checkpoint is the DiT file this trainer, ComfyUI and SwarmUI load, with the released file's names, shapes and
dtypes. It runs trainer_base's training loop, whose network / accelerator / resume hooks the dense trainer overrides: the
extended loop of the LoRA trainer builds its LoRA network inline (no --network_module here). Text-to-image and editing
datasets both train (references enter as in the LoRA trainer). The INT8 ConvRot file and FP8 bases are refused: they are
frozen-weight LoRA paths.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch

from musubi_tuner.fibo2_train_network import Fibo2TrainerMixin, fibo2_setup_parser
from musubi_tuner.training.dense_trainer import DenseTrainerMixin, add_dense_arguments
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer


class Fibo2Trainer(DenseTrainerMixin, Fibo2TrainerMixin, NetworkTrainer):
    """Every FIBO 2.0 DiT weight in BF16; checkpoints in the layout ComfyUI and ``load_dit`` read."""

    dense_model_name = "FIBO 2.0"

    def dense_swap_blocks(self, transformer: torch.nn.Module):
        return transformer.swap_units()  # what the offloader swaps: the blocks without an injection block's cross-attention


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = fibo2_setup_parser(parser)
    parser.description = "Full-parameter BF16 training for FIBO 2.0"
    return add_dense_arguments(parser, "FIBO 2.0", learning_rate=1e-5, output_name="fibo2_full_finetune")


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"
    Fibo2Trainer().train(args)


if __name__ == "__main__":
    main()
