import argparse
from dataclasses import asdict, dataclass

import torch

from musubi_tuner.utils import model_utils, train_utils


_LORA_INERT_DEFAULTS = {
    "network_weights": None,
    "network_module": None,
    "network_dim": None,
    "network_alpha": 1,
    "network_dropout": None,
    "network_args": None,
    "dim_from_weights": False,
    "scale_weight_norms": None,
    "base_weights": None,
    "base_weights_multiplier": None,
}


@dataclass(eq=True)
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    next_batch: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.next_batch = int(state["next_batch"])


def add_full_finetune_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--full_bf16", action="store_true", help="enable full bfloat16 training")
    parser.add_argument("--fused_backward_pass", action="store_true", help="use fused backward pass for Adafactor")
    parser.add_argument("--mem_eff_save", action="store_true", help="enable memory-efficient model saving")
    parser.add_argument(
        "--block_swap_optimizer_patch_params",
        action="store_true",
        help="patch optimizer parameters when block swap is enabled",
    )
    return parser


def resolve_trainable_dtype(args: argparse.Namespace) -> torch.dtype:
    return model_utils.str_to_dtype("bf16" if args.full_bf16 else "fp32")


def validate_full_finetune_args(args: argparse.Namespace, num_processes: int) -> None:
    specified_lora_args = [
        f"--{name}" for name, inert_default in _LORA_INERT_DEFAULTS.items() if getattr(args, name) != inert_default
    ]
    if specified_lora_args:
        raise ValueError("full finetuning does not use LoRA/network arguments: " + ", ".join(specified_lora_args))

    specified_fp8_args = [f"--{name}" for name in ("fp8_base", "fp8_scaled") if getattr(args, name)]
    if specified_fp8_args:
        raise ValueError("full finetuning does not support FP8 options: " + ", ".join(specified_fp8_args))

    if args.block_swap_h2d_only:
        raise ValueError("--block_swap_h2d_only is only supported for frozen-base training")

    if (args.blocks_to_swap or 0) > 0 and num_processes > 1:
        raise ValueError("--blocks_to_swap is not supported for multi-process full finetuning")

    if args.fused_backward_pass:
        if args.optimizer_type.lower() != "adafactor":
            raise ValueError("--fused_backward_pass requires --optimizer_type Adafactor")
        if num_processes > 1:
            raise ValueError("--fused_backward_pass is not supported for multi-process full finetuning")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("--fused_backward_pass requires --gradient_accumulation_steps 1")

    if args.full_bf16 and args.mixed_precision != "bf16":
        raise ValueError("--full_bf16 requires --mixed_precision bf16")

    trainable_dtype = resolve_trainable_dtype(args)
    save_dtype = train_utils.resolve_save_dtype(args.save_precision, full_bf16=args.full_bf16)
    if save_dtype != trainable_dtype:
        raise ValueError(f"--save_precision must match the trainable dtype ({trainable_dtype}); got {save_dtype}")


def normalize_compiled_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.").replace("._orig_mod.", ".")
        if key in normalized:
            raise ValueError(f"compiled state_dict key collision: {key}")
        normalized[key] = value
    return normalized
