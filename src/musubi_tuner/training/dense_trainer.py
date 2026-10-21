"""Full-parameter (dense) BF16 fine-tuning on top of a model's LoRA trainer.

The model's LoRA trainer keeps its data path, noise schedule, loss, sampling and saving lifecycle; the "network" becomes a
thin module that hands every transformer parameter to the optimizer. This is Kandinsky 6's ``kandinsky6_train.py``
(AkaneTendo25) generalised so other models can reuse it: fused per-tensor Adafactor updates during backward (optionally
the Triton BF16 fast path), block swap with fused updates (``--block_swap_trainable_ring`` streams the swapped blocks
through reusable GPU buffers from pinned CPU masters), dense resume (optimizer, scheduler, RNG, next batch).

A model plugs in with ``class XTrainer(DenseTrainerMixin, XNetworkTrainer)`` and overrides ``dense_model_name``,
``dense_swap_blocks`` (the swapped block list, for parking the ring around Accelerate's state save / load) and
``dense_state_dict_for_save`` (the checkpoint layout the model's loader and ComfyUI read).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import time
import types
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from safetensors.torch import save_file

from musubi_tuner.training.trainer_base import DenseResumeCursor
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file

logger = logging.getLogger(__name__)


def drain_block_swap(transformer: torch.nn.Module) -> None:
    """Wait for in-flight block-swap copies so a state_dict read sees settled weights."""
    offloader = getattr(transformer, "offloader", None)
    if offloader is None:
        return
    commit = getattr(offloader, "commit", None)
    if callable(commit):
        commit()
    synchronize = getattr(offloader, "synchronize", None)
    if callable(synchronize):
        synchronize()
    copier = getattr(offloader, "copier", None)
    if copier is not None:
        copier.sync()
    futures = getattr(offloader, "futures", None)
    if futures:
        for block_idx in list(futures.keys()):
            offloader._wait_blocks_move(block_idx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def triton_adafactor_usable(device: torch.device) -> bool:
    """--adafactor_triton: whether Triton builds and runs its Adafactor kernels here, checked once before training, so a
    machine without a working Triton trains with the standard fused update instead of stopping at the first backward."""
    try:
        from musubi_tuner.modules.adafactor_triton import check_triton_adafactor

        check_triton_adafactor(device)
        return True
    except Exception as error:  # no Triton here, or its compiler cannot build the kernels for this GPU
        logger.warning(
            f"Triton Adafactor is not available ({type(error).__name__}: {error}); using the standard fused Adafactor "
            "update, which needs more VRAM for its temporaries"
        )
        return False


def load_ring_checkpoint_state(model: torch.nn.Module, checkpoint: Path) -> None:
    """Stream a packed-ring checkpoint with strict keys and tensor shapes."""
    current = model.state_dict()
    with MemoryEfficientSafeOpen(str(checkpoint)) as reader, torch.no_grad():
        saved_keys = set(reader.keys())
        current_keys = set(current)
        if saved_keys != current_keys:
            missing = sorted(current_keys - saved_keys)
            unexpected = sorted(saved_keys - current_keys)
            raise RuntimeError(f"ring checkpoint keys differ from the model; missing={missing[:20]}, unexpected={unexpected[:20]}")
        for key, target in current.items():
            value = reader.get_tensor(key, device=torch.device("cpu"))
            if value.shape != target.shape:
                raise RuntimeError(f"ring checkpoint tensor {key!r} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            target.copy_(value)


class DenseFinetuneModule(torch.nn.Module):
    """Adapter between the shared trainer lifecycle (the LoRA-network API) and dense transformer parameters."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        state_dict_for_save: Callable[[torch.nn.Module], dict],
        *,
        model_name: str = "model",
        save_weights: bool = True,
        mem_eff_save: bool = True,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self._state_dict_for_save = state_dict_for_save
        self._model_name = model_name
        self._save_weights = save_weights
        self._mem_eff_save = mem_eff_save

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def prepare_optimizer_params(self, unet_lr: float, **_kwargs):
        params = [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError(f"no trainable {self._model_name} transformer parameters were found")
        return [{"params": params, "lr": unet_lr}], ["transformer"]

    def prepare_grad_etc(self, _transformer) -> None:
        self.train()

    def on_epoch_start(self, _transformer) -> None:
        self.train()

    def on_step_start(self) -> None:
        pass

    def enable_gradient_checkpointing(self) -> None:
        pass  # the transformer itself is switched in DenseTrainerMixin._build_network

    def get_trainable_params(self):
        return (parameter for parameter in self.transformer.parameters() if parameter.requires_grad)

    def apply_max_norm_regularization(self, *_args, **_kwargs):
        raise ValueError(f"--scale_weight_norms is not supported by {self._model_name} full fine-tuning")

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[dict]) -> None:
        if not self._save_weights:
            logger.warning(f"Skipping the {self._model_name} checkpoint write because --debug_no_save_weights is set")
            return
        if dtype not in (None, torch.bfloat16):
            raise ValueError(f"{self._model_name} full fine-tuning checkpoints can only be saved in BF16")
        drain_block_swap(self.transformer)
        state_dict = self._state_dict_for_save(self.transformer)
        if any("_orig_mod." in key for key in state_dict):
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        if self._mem_eff_save:
            mem_eff_save_file(state_dict, file, metadata)
        else:
            save_file({key: value.detach().cpu().contiguous() for key, value in state_dict.items()}, file, metadata)


class DenseTrainerMixin:
    """Full-parameter BF16 training with optional fused Adafactor updates (see the module docstring)."""

    trains_full_transformer = True
    dense_model_name = "model"

    # ------------------------------------------------------------------ model hooks
    def dense_swap_blocks(self, transformer: torch.nn.Module):
        """The block list the offloader swaps (the trainable ring is parked over it around state save / load)."""
        raise NotImplementedError

    def dense_state_dict_for_save(self, transformer: torch.nn.Module) -> dict:
        """The checkpoint's keys and tensors: by default the transformer's own state dict."""
        return transformer.state_dict()

    def dense_validate_model_args(self, args: argparse.Namespace) -> None:
        """Model-specific refusals (quantized bases and the like)."""

    # ------------------------------------------------------------------ validation
    def _validate_full_finetune_args(self, args: argparse.Namespace) -> None:
        name = self.dense_model_name
        if args.mixed_precision != "bf16" or not args.full_bf16:
            raise ValueError(f"{name} full fine-tuning requires --mixed_precision bf16 and --full_bf16")
        if args.full_fp16:
            raise ValueError(f"{name} full fine-tuning does not support --full_fp16")
        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False) or getattr(args, "convrot_int8", False):
            raise ValueError(f"{name} full fine-tuning trains BF16 weights; FP8 / ConvRot INT8 bases are frozen-weight LoRA paths")
        if getattr(args, "block_swap_h2d_only", False):
            raise ValueError(f"{name} full fine-tuning cannot use frozen-weight --block_swap_h2d_only")
        if args.base_weights or args.network_weights or args.dim_from_weights:
            raise ValueError(f"{name} full fine-tuning does not accept LoRA initialization or merge weights")
        if args.network_args or args.network_dropout is not None or args.scale_weight_norms:
            raise ValueError(f"{name} full fine-tuning does not accept LoRA network / dropout / max-norm options")
        if args.save_precision not in (None, "bf16"):
            raise ValueError(f"{name} full fine-tuning checkpoints are BF16 (omit --save_precision or use bf16)")
        if args.fused_backward_pass:
            if args.optimizer_type.lower() != "adafactor":
                raise ValueError("--fused_backward_pass currently requires --optimizer_type Adafactor")
            if args.max_grad_norm != 0.0:
                raise ValueError("--fused_backward_pass requires --max_grad_norm 0 so each gradient can be stepped and freed immediately")
        if args.adafactor_triton and not args.fused_backward_pass:
            raise ValueError("--adafactor_triton requires --fused_backward_pass")
        if args.adafactor_triton:
            optimizer_args = {}
            for item in args.optimizer_args or []:
                key, separator, value = item.partition("=")
                if not separator:
                    raise ValueError(f"invalid --optimizer_args entry {item!r}; expected key=value")
                optimizer_args[key] = value.lower()
            if optimizer_args.get("scale_parameter") != "false" or optimizer_args.get("relative_step") != "false":
                raise ValueError(
                    "--adafactor_triton requires manual-LR Adafactor arguments: "
                    "--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
                )
        if args.block_swap_trainable_ring and not (args.blocks_to_swap or 0):
            raise ValueError("--block_swap_trainable_ring requires --blocks_to_swap")
        if args.blocks_to_swap and not args.fused_backward_pass:
            # block swap moves parameter storage between CPU and GPU during backward; a deferred optimizer step would
            # meet CPU weights with GPU gradients, fused updates step each tensor while its block is resident
            raise ValueError(f"{name} full fine-tuning with --blocks_to_swap requires --fused_backward_pass")
        self.dense_validate_model_args(args)

    def _validate_args_and_init(self, args) -> bool:
        proceed = super()._validate_args_and_init(args)
        self._validate_full_finetune_args(args)
        return proceed

    def _prepare_accelerator_and_dtypes(self, args):
        result = super()._prepare_accelerator_and_dtypes(args)
        if result[0].num_processes != 1:
            raise ValueError(f"{self.dense_model_name} full fine-tuning currently supports one process / GPU")
        return result

    # ------------------------------------------------------------------ network
    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        del accelerator, vae, weight_dtype
        transformer.requires_grad_(True)
        transformer.train()
        trainable = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
        logger.info("%s dense training updates %.3fB parameters", self.dense_model_name, trainable / 1e9)
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        return DenseFinetuneModule(
            transformer,
            self.dense_state_dict_for_save,
            model_name=self.dense_model_name,
            save_weights=not args.debug_no_save_weights,
            mem_eff_save=bool(args.mem_eff_save),
        )

    def on_train_start(self, args, accelerator, network, transformer, optimizer) -> None:
        del args, network, transformer, optimizer
        self._dense_step_started_at = time.perf_counter()
        self._dense_step_metrics: dict[str, float] = {}
        if accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(accelerator.device)

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients: bool, global_step: int) -> None:
        del args, network, transformer, global_step
        if not sync_gradients:
            return
        metrics = {"dense_step_seconds": time.perf_counter() - self._dense_step_started_at}
        if accelerator.device.type == "cuda":
            metrics["dense_peak_allocated_gib"] = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
            metrics["dense_peak_reserved_gib"] = torch.cuda.max_memory_reserved(accelerator.device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        self._dense_step_metrics = metrics
        self._dense_step_started_at = time.perf_counter()

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        del args, logs
        return getattr(self, "_dense_step_metrics", {})

    def _prepare_with_accelerator(
        self, args, accelerator, transformer, network, optimizer, train_dataloader, lr_scheduler, weight_dtype, dit_dtype, dit_weight_dtype
    ):
        del weight_dtype, dit_weight_dtype
        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            network.transformer = transformer
        if (self.blocks_to_swap or 0) > 0:
            network = accelerator.prepare(network, device_placement=[False])
            unwrapped = accelerator.unwrap_model(network)
            unwrapped.transformer.move_to_device_except_swap_blocks(accelerator.device)
            unwrapped.transformer.prepare_block_swap_before_forward()
            unwrapped.transformer.switch_block_swap_for_training()
        else:
            network = accelerator.prepare(network)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        transformer = accelerator.unwrap_model(network).transformer
        transformer.train()
        self._install_fused_optimizer(args, accelerator, optimizer)
        # the weights are the trained tensors, so the model runs in BF16 end to end
        return transformer, network, optimizer, train_dataloader, lr_scheduler, network, dit_dtype

    def _register_hooks_and_resume(self, args, accelerator, network):
        """Dense resume (cursor, RNG) and, for a trainable ring, parking it around Accelerate's model-state save / load:
        Accelerate loads into each Parameter's current storage, and a prepared ring has Parameters bound to GPU slots
        whose CPU masters still hold pre-resume values (evicting such a slot without writeback would lose the restore)."""
        unwrapped_network = accelerator.unwrap_model(network)
        signature_rows = [
            (name, tuple(parameter.shape), str(parameter.dtype)) for name, parameter in unwrapped_network.named_parameters() if parameter.requires_grad
        ]
        optimizer_signature = hashlib.sha256(repr(signature_rows).encode("utf-8")).hexdigest()
        current_ga = int(args.gradient_accumulation_steps)
        current_seed = int(args.seed) if args.seed is not None else None
        self._dense_resume_cursor = DenseResumeCursor(
            optimizer_signature=optimizer_signature, gradient_accumulation_steps=current_ga, data_seed=current_seed
        )
        cursor_registered = True
        if args.resume and not args.resume_from_huggingface:
            cursor_path = Path(args.resume) / "custom_checkpoint_0.pkl"
            if cursor_path.exists():
                saved_cursor = torch.load(cursor_path, map_location="cpu", weights_only=True)
                saved_signature = str(saved_cursor.get("optimizer_signature", ""))
                if saved_signature and saved_signature != optimizer_signature:
                    raise RuntimeError("dense resume optimizer parameter mapping differs from the checkpoint")
                if int(saved_cursor.get("gradient_accumulation_steps", 1)) != current_ga:
                    raise RuntimeError("dense resume gradient_accumulation_steps differs from the checkpoint")
                if saved_cursor.get("data_seed") != current_seed:
                    raise RuntimeError("dense resume data seed differs from the checkpoint")
            else:
                cursor_registered = False
        if cursor_registered:
            accelerator.register_for_checkpointing(self._dense_resume_cursor)
        transformer = accelerator.unwrap_model(network).transformer
        offloader = getattr(transformer, "offloader", None)
        is_trainable_ring = bool(args.block_swap_trainable_ring and offloader is not None)

        if is_trainable_ring:
            blocks = self.dense_swap_blocks(transformer)

            def park_ring_for_save(models, weights, output_dir):
                offloader.commit()
                # Accelerate gathers aliased tensor views before save prehooks and safetensors deduplicates every view
                # sharing a packed ring storage: stream the complete state here and drop this model entry, so Accelerate
                # still writes optimizer / scheduler / RNG state without overwriting model.safetensors
                for index in reversed(range(len(models))):
                    model = models[index]
                    if accelerator.unwrap_model(model) is accelerator.unwrap_model(network):
                        suffix = f"_{index}" if index else ""
                        state_dict = accelerator.get_state_dict(model, unwrap=False)
                        mem_eff_save_file(state_dict, str(Path(output_dir) / f"model{suffix}.safetensors"))
                        if index < len(weights):
                            weights.pop(index)

            def park_ring_for_load(models, input_dir):
                offloader.offload_to_cpu(blocks)
                for index in reversed(range(len(models))):
                    model = models[index]
                    if accelerator.unwrap_model(model) is not accelerator.unwrap_model(network):
                        continue
                    suffix = f"_{index}" if index else ""
                    load_ring_checkpoint_state(accelerator.unwrap_model(model), Path(input_dir) / f"model{suffix}.safetensors")
                    models.pop(index)

            accelerator.register_save_state_pre_hook(park_ring_for_save)
            accelerator.register_load_state_pre_hook(park_ring_for_load)

        super()._register_hooks_and_resume(args, accelerator, network)

        if args.resume:
            self._dense_resume_rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        if not cursor_registered:
            match = re.search(r"-(\d+)-state$", Path(args.resume).name)
            restored_step = int(match.group(1)) if match else None
            if restored_step is None:
                scheduler_path = Path(args.resume) / "scheduler.bin"
                if scheduler_path.exists():
                    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
                    restored_step = int(scheduler_state.get("last_epoch", 0))
            if restored_step is None or restored_step <= 0:
                raise RuntimeError("legacy dense resume state has no cursor and its completed optimizer step cannot be inferred")
            self._dense_resume_cursor.global_step = restored_step
            self._dense_resume_cursor.step_in_epoch = restored_step * int(getattr(args, "gradient_accumulation_steps", 1))
            accelerator.register_for_checkpointing(self._dense_resume_cursor)

        if is_trainable_ring and args.resume:
            transformer.move_to_device_except_swap_blocks(accelerator.device)
            transformer.prepare_block_swap_before_forward()
            transformer.switch_block_swap_for_training()

    def _install_fused_optimizer(self, args, accelerator, optimizer) -> None:
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)

        if args.optimizer_type.lower() == "adafactor":
            # torch.optim casts floating optimizer tensors to the parameter dtype on load; dense BF16 Adafactor keeps
            # its moments in FP32
            original_load_state_dict = raw_optimizer.load_state_dict

            def load_state_dict_preserving_fp32(self, state_dict):
                result = original_load_state_dict(state_dict)
                for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
                    for parameter, saved_id in zip(group["params"], saved_group["params"]):
                        saved_state = state_dict["state"].get(saved_id, {})
                        current_state = self.state.get(parameter, {})
                        for key in ("exp_avg", "exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col"):
                            value = saved_state.get(key)
                            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype == torch.float32:
                                current_state[key] = value.detach().to(device=parameter.device, dtype=torch.float32).clone()
                return result

            raw_optimizer.load_state_dict = types.MethodType(load_state_dict_preserving_fp32, raw_optimizer)

        if not args.fused_backward_pass:
            return
        if accelerator.num_processes != 1:
            raise ValueError("--fused_backward_pass requires a single process; hooks step before distributed gradient reduction")
        # patch only the underlying optimizer: the Accelerate wrapper keeps its gradient-accumulation guard, and after
        # the hooks stepped and freed every gradient its step() sees grad=None everywhere and does nothing
        if args.adafactor_triton and triton_adafactor_usable(accelerator.device):
            from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

            patch_adafactor_triton(raw_optimizer)
            logger.info(f"{self.dense_model_name} dense Adafactor uses the Triton 2D BF16 fast path")
        else:
            from musubi_tuner.modules.adafactor_fused import patch_adafactor_fused

            patch_adafactor_fused(raw_optimizer)
            logger.info(f"{self.dense_model_name} dense Adafactor uses fused per-parameter backward updates")
        optimizer.step_param = raw_optimizer.step_param

        def make_hook(parameter: torch.nn.Parameter, param_group: dict):
            def grad_hook(_tensor: torch.Tensor) -> None:
                if not accelerator.sync_gradients:
                    return
                optimizer.step_param(parameter, param_group)
                parameter.grad = None

            return grad_hook

        for param_group in raw_optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:
                    parameter.register_post_accumulate_grad_hook(make_hook(parameter, param_group))

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = super().extra_metadata(args)
        metadata.update(
            {
                "ss_training_type": "full_finetune",
                "ss_fused_backward_pass": str(args.fused_backward_pass),
                "ss_adafactor_triton": str(args.adafactor_triton),
                "ss_block_swap_trainable_ring": str(args.block_swap_trainable_ring),
            }
        )
        return metadata


def add_dense_arguments(parser: argparse.ArgumentParser, model_name: str, *, learning_rate: float, output_name: str) -> argparse.ArgumentParser:
    """The dense options and defaults (fused Adafactor, BF16, no clipping, gradient checkpointing)."""
    # the fork's common parser defines --full_fp16 / --full_bf16 (store_true): add only the negative form, BF16 the default
    for flag in ("--full_fp16", "--full_bf16"):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-full_bf16", dest="full_bf16", action="store_false", help=f"train {model_name} weights in BF16 (required, the default)")
    parser.add_argument(
        "--fused_backward_pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="step Adafactor per tensor during backward and free each gradient immediately "
        "(requires --optimizer_type Adafactor and --max_grad_norm 0; required with --blocks_to_swap)",
    )
    if "--adafactor_triton" not in parser._option_string_actions:
        parser.add_argument(
            "--adafactor_triton",
            action="store_true",
            help="use the Triton BF16 matrix fast path for fused Adafactor (unsupported tensors use the reference path)",
        )
    if "--block_swap_trainable_ring" not in parser._option_string_actions:
        parser.add_argument(
            "--block_swap_trainable_ring",
            action="store_true",
            help="coalesce trainable block swap through pinned CPU masters and reusable GPU ring buffers",
        )
    parser.add_argument(
        "--mem_eff_save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream the full checkpoint tensor by tensor instead of copying the model to host memory first",
    )
    parser.add_argument("--debug_no_save_weights", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(
        full_bf16=True,
        network_module=None,
        mixed_precision="bf16",
        optimizer_type="Adafactor",
        optimizer_args=["scale_parameter=False", "relative_step=False", "warmup_init=False"],
        learning_rate=learning_rate,
        lr_scheduler="constant",
        max_grad_norm=0.0,
        gradient_checkpointing=True,
        output_name=output_name,
    )
    return parser
