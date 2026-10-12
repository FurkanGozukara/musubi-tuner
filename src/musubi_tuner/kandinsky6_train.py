"""Full-parameter (dense) BF16 fine-tuning for Kandinsky 6 Lite/Pro.

Reuses the LoRA trainer's data path, noise schedule, joint video/audio loss and
sampling (``Kandinsky6NetworkTrainer``) and swaps the LoRA network for a thin
wrapper that exposes every transformer parameter to the optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
import types
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file

from musubi_tuner.kandinsky6_train_network import Kandinsky6NetworkTrainer, kandinsky6_setup_parser
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DenseResumeCursor
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file

logger = logging.getLogger(__name__)

# ``named_parameters()`` prefix of the swappable/freezable Kandinsky 6 DiT blocks.
VISUAL_BLOCK_PREFIX = "visual_transformer_blocks"


def parse_block_spec(spec: str) -> set[int]:
    """Parse ``"0-19,45"`` into block indices; bounds are checked against the model later."""
    indices: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty entry in block list {spec!r}")
        start, separator, end = item.partition("-")
        try:
            first = int(start)
            last = int(end) if separator else first
        except ValueError:
            raise ValueError(f"invalid block list entry {item!r}; expected N or N-M") from None
        if first < 0 or last < first:
            raise ValueError(f"invalid block range {item!r}")
        indices.update(range(first, last + 1))
    return indices


def compile_freeze_patterns(patterns: Sequence[str] | None) -> list[re.Pattern]:
    compiled = []
    for pattern in patterns or []:
        try:
            compiled.append(re.compile(pattern))
        except re.error as error:
            raise ValueError(f"invalid --k6_freeze_params pattern {pattern!r}: {error}") from None
    return compiled


def freeze_transformer_parameters(
    transformer: torch.nn.Module, freeze_blocks: str | None, freeze_params: Sequence[str] | None
) -> tuple[int, int]:
    """Turn off gradients for the selected visual blocks and parameter-name patterns.

    Block indices refer to ``transformer.visual_transformer_blocks`` (the fused video/audio
    decoder blocks: 32 for Lite, 60 for Pro). Patterns use ``re.search`` over
    ``transformer.named_parameters()`` names, e.g. ``video_text_transformer_blocks`` or
    ``visual_transformer_blocks\\.\\d+\\.audio_``. A block index outside the model, or a
    pattern that matches nothing, is an error: a typo would otherwise train the whole
    model silently. Returns (frozen tensors, frozen elements).
    """
    blocks: set[int] = set()
    if freeze_blocks:
        blocks = parse_block_spec(freeze_blocks)
        num_blocks = len(getattr(transformer, VISUAL_BLOCK_PREFIX))
        outside = sorted(index for index in blocks if index >= num_blocks)
        if outside:
            raise ValueError(f"--k6_freeze_blocks names blocks {outside}, but the model has {num_blocks} (0-{num_blocks - 1})")
    patterns = compile_freeze_patterns(freeze_params)
    matched = [False] * len(patterns)
    block_name = re.compile(rf"^{VISUAL_BLOCK_PREFIX}\.(\d+)\.")
    frozen_tensors = frozen_elements = 0
    for name, parameter in transformer.named_parameters():
        match = block_name.match(name)
        freeze = match is not None and int(match.group(1)) in blocks
        for index, pattern in enumerate(patterns):
            if pattern.search(name):
                matched[index] = True
                freeze = True
        if freeze and parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen_tensors += 1
            frozen_elements += parameter.numel()
    unmatched = [pattern.pattern for pattern, hit in zip(patterns, matched) if not hit]
    if unmatched:
        raise ValueError(f"--k6_freeze_params patterns match no transformer parameter: {unmatched}")
    return frozen_tensors, frozen_elements


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


class Kandinsky6FullFinetuneModule(torch.nn.Module):
    """Adapter between the shared trainer lifecycle (LoRA-network API) and dense K6 parameters."""

    def __init__(self, transformer: torch.nn.Module, *, save_weights: bool = True, mem_eff_save: bool = True) -> None:
        super().__init__()
        self.transformer = transformer
        self._save_weights = save_weights
        self._mem_eff_save = mem_eff_save

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def prepare_optimizer_params(self, unet_lr: float, **_kwargs):
        params = [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("no trainable Kandinsky 6 transformer parameters were found")
        return [{"params": params, "lr": unet_lr}], ["transformer"]

    def prepare_grad_etc(self, _transformer) -> None:
        self.train()

    def on_epoch_start(self, _transformer) -> None:
        self.train()

    def on_step_start(self) -> None:
        pass

    def enable_gradient_checkpointing(self) -> None:
        pass  # the transformer itself is switched in Kandinsky6Trainer._build_network

    def get_trainable_params(self):
        return (parameter for parameter in self.transformer.parameters() if parameter.requires_grad)

    def apply_max_norm_regularization(self, *_args, **_kwargs):
        raise ValueError("--scale_weight_norms is not supported by Kandinsky 6 full fine-tuning")

    def save_weights(self, file: str, dtype: torch.dtype | None, metadata: dict | None) -> None:
        if not self._save_weights:
            logger.warning("Skipping Kandinsky 6 dense checkpoint write because --debug_no_save_weights is set")
            return
        if dtype not in (None, torch.bfloat16):
            raise ValueError("Kandinsky 6 dense checkpoints can only be saved in native BF16 precision")
        drain_block_swap(self.transformer)
        # Keys are the native DiffusionTransformer3D names, which is exactly what
        # ``kandinsky6.model.load_dit`` reads (strict), so the file is a drop-in --dit.
        state_dict = self.transformer.state_dict()
        if any("_orig_mod." in key for key in state_dict):
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        if self._mem_eff_save:
            mem_eff_save_file(state_dict, file, metadata)
        else:
            # ``save_file`` needs the whole checkpoint contiguous in host memory,
            # but it is the reference writer: keep it available for users who hit
            # a problem with the streaming path.
            save_file({key: value.detach().cpu().contiguous() for key, value in state_dict.items()}, file, metadata)


class Kandinsky6Trainer(Kandinsky6NetworkTrainer):
    """Full-parameter BF16 Kandinsky 6 trainer with optional fused Adafactor updates."""

    trains_full_transformer = True

    def _validate_full_finetune_args(self, args: argparse.Namespace) -> None:
        if args.mixed_precision != "bf16" or not args.full_bf16:
            raise ValueError("Kandinsky 6 full fine-tuning requires --mixed_precision bf16 and --full_bf16")
        if args.full_fp16:
            raise ValueError("Kandinsky 6 full fine-tuning does not support --full_fp16")
        if args.fp8_base or args.convrot_int8:
            raise ValueError(
                "Kandinsky 6 full fine-tuning requires ordinary BF16 weights; FP8 / ConvRot INT8 bases "
                "(including pre-quantized INT8 checkpoints) are frozen-weight LoRA paths"
            )
        if args.block_swap_h2d_only:
            raise ValueError("Kandinsky 6 full fine-tuning cannot use frozen-weight --block_swap_h2d_only")
        if args.base_weights or args.network_weights or args.dim_from_weights:
            raise ValueError("Kandinsky 6 full fine-tuning does not accept LoRA initialization or merge weights")
        if args.network_args or args.network_dropout is not None or args.scale_weight_norms:
            raise ValueError("Kandinsky 6 full fine-tuning does not accept LoRA network/dropout/max-norm options")
        if args.save_precision not in (None, "bf16"):
            raise ValueError("Kandinsky 6 dense checkpoints must be saved as BF16 (omit --save_precision or use bf16)")
        if args.fused_backward_pass:
            if args.optimizer_type.lower() != "adafactor":
                raise ValueError("--fused_backward_pass currently requires --optimizer_type Adafactor")
            if args.max_grad_norm != 0.0:
                raise ValueError(
                    "--fused_backward_pass requires --max_grad_norm 0 so each gradient can be stepped and freed immediately"
                )
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
            # Block swap moves parameter storage between CPU and GPU during backward.
            # A deferred optimizer step would then meet CPU weights with GPU gradients;
            # fused updates step each tensor while its block is still resident.
            raise ValueError("Kandinsky 6 full fine-tuning with --blocks_to_swap requires --fused_backward_pass")
        # Syntax only; block bounds and pattern matches need the loaded model.
        if args.k6_freeze_blocks:
            parse_block_spec(args.k6_freeze_blocks)
        compile_freeze_patterns(args.k6_freeze_params)

    def _validate_args_and_init(self, args) -> bool:
        # The parent runs handle_model_specific_args, which may switch on --convrot_int8
        # after detecting a pre-quantized checkpoint; validate after it so that is caught.
        proceed = super()._validate_args_and_init(args)
        self._validate_full_finetune_args(args)
        return proceed

    def _prepare_accelerator_and_dtypes(self, args):
        result = super()._prepare_accelerator_and_dtypes(args)
        accelerator = result[0]
        if accelerator.num_processes != 1:
            raise ValueError("Kandinsky 6 full fine-tuning currently supports one process/GPU")
        return result

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        del accelerator, vae, weight_dtype
        transformer.requires_grad_(True)
        transformer.train()
        if args.k6_freeze_blocks or args.k6_freeze_params:
            frozen_tensors, frozen_elements = freeze_transformer_parameters(
                transformer, args.k6_freeze_blocks, args.k6_freeze_params
            )
            logger.info("Kandinsky 6 dense training freezes %d tensors (%.3fB parameters)", frozen_tensors, frozen_elements / 1e9)
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        if trainable == 0:
            raise ValueError("--k6_freeze_blocks/--k6_freeze_params freeze every Kandinsky 6 transformer parameter")
        if getattr(transformer, "reference_extension_is_fresh", False):
            frozen_new = [
                name
                for name in ("visual_ref_slot_embeddings", "visual_token_type_embeddings")
                if hasattr(transformer, name) and not getattr(transformer, name).weight.requires_grad
            ]
            if frozen_new:
                raise ValueError(f"--k6_freeze_params freezes the untrained R2AV embeddings {frozen_new}; keep them trainable")
        logger.info("Kandinsky 6 dense training updates %.3fB parameters", trainable / 1e9)
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        return Kandinsky6FullFinetuneModule(
            transformer,
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
        del args, network, transformer
        if not sync_gradients:
            return
        elapsed = time.perf_counter() - self._dense_step_started_at
        metrics = {"dense_step_seconds": elapsed}
        if accelerator.device.type == "cuda":
            metrics.update(
                {
                    "dense_peak_allocated_gib": torch.cuda.max_memory_allocated(accelerator.device) / (1024**3),
                    "dense_peak_reserved_gib": torch.cuda.max_memory_reserved(accelerator.device) / (1024**3),
                }
            )
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        self._dense_step_metrics = metrics
        logger.debug(
            "Kandinsky 6 dense step %d: %.3fs, peak allocated %.3f GiB, peak reserved %.3f GiB",
            global_step + 1,
            elapsed,
            metrics.get("dense_peak_allocated_gib", 0.0),
            metrics.get("dense_peak_reserved_gib", 0.0),
        )
        self._dense_step_started_at = time.perf_counter()

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        del args, logs
        return getattr(self, "_dense_step_metrics", {})

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        transformer,
        network,
        optimizer,
        train_dataloader,
        lr_scheduler,
        weight_dtype,
        dit_dtype,
        dit_weight_dtype,
    ):
        del weight_dtype, dit_weight_dtype
        blocks_to_swap = self.blocks_to_swap or 0
        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            network.transformer = transformer

        if blocks_to_swap > 0:
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
        # The weights are the trained tensors, so the model runs in BF16 end to end.
        return transformer, network, optimizer, train_dataloader, lr_scheduler, network, dit_dtype

    def _register_hooks_and_resume(self, args, accelerator, network):
        """Park a trainable ring around Accelerate model-state serialization.

        Accelerate loads into each Parameter's current storage. A prepared ring
        has some Parameters bound to GPU slots while their persistent CPU masters
        still contain pre-resume values; evicting such a slot without writeback
        would discard the restored tensor. Parking first makes every Parameter
        point at its unique CPU master for both save and load.
        """
        unwrapped_network = accelerator.unwrap_model(network)
        signature_rows = [
            (name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in unwrapped_network.named_parameters()
            if parameter.requires_grad
        ]
        optimizer_signature = hashlib.sha256(repr(signature_rows).encode("utf-8")).hexdigest()
        current_ga = int(args.gradient_accumulation_steps)
        current_seed = int(args.seed) if args.seed is not None else None
        self._dense_resume_cursor = DenseResumeCursor(
            optimizer_signature=optimizer_signature,
            gradient_accumulation_steps=current_ga,
            data_seed=current_seed,
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
            blocks = transformer.visual_transformer_blocks

            def park_ring_for_save(models, weights, output_dir):
                offloader.commit()
                # Accelerate gathers aliased tensor views before save prehooks,
                # and safetensors deduplicates every view sharing a packed ring
                # storage. Stream the complete state ourselves, then remove this
                # model entry so Accelerate still writes optimizer/scheduler/RNG
                # state without overwriting model.safetensors.
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
                    checkpoint = Path(input_dir) / f"model{suffix}.safetensors"
                    load_ring_checkpoint_state(accelerator.unwrap_model(model), checkpoint)
                    models.pop(index)

            accelerator.register_save_state_pre_hook(park_ring_for_save)
            accelerator.register_load_state_pre_hook(park_ring_for_load)

        super()._register_hooks_and_resume(args, accelerator, network)

        if args.resume:
            self._dense_resume_rng_state = {
                "python": __import__("random").getstate(),
                "numpy": __import__("numpy").random.get_state(),
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

    @staticmethod
    def _install_fused_optimizer(args, accelerator, optimizer) -> None:
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)

        if args.optimizer_type.lower() == "adafactor":
            # torch.optim casts floating optimizer tensors to the parameter dtype
            # on load. Dense BF16 Adafactor deliberately keeps its moments in
            # FP32, including the supported non-fused/no-swap recipe.
            original_load_state_dict = raw_optimizer.load_state_dict

            def load_state_dict_preserving_fp32(self, state_dict):
                result = original_load_state_dict(state_dict)
                saved_groups = state_dict["param_groups"]
                for group, saved_group in zip(self.param_groups, saved_groups):
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
        # ``accelerator.prepare`` wraps the optimizer. Patching that wrapper's
        # ``step`` bypasses AcceleratedOptimizer's gradient-accumulation guard:
        # trainer_base calls optimizer.step() on every microstep, and the fused
        # replacement would update early. Patch only the underlying optimizer so
        # the wrapper retains its step/zero_grad semantics. After the hooks have
        # stepped and freed every gradient, the wrapper's step() sees grad=None
        # everywhere and is a no-op.
        if args.adafactor_triton:
            from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

            patch_adafactor_triton(raw_optimizer)
            logger.info("Kandinsky 6 dense Adafactor uses the Triton 2D BF16 fast path")
        else:
            from musubi_tuner.modules.adafactor_fused import patch_adafactor_fused

            patch_adafactor_fused(raw_optimizer)
            logger.info("Kandinsky 6 dense Adafactor uses fused per-parameter backward updates")

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
                "ss_k6_task": str(args.task),
            }
        )
        if args.k6_freeze_blocks:
            metadata["ss_k6_freeze_blocks"] = args.k6_freeze_blocks
        if args.k6_freeze_params:
            metadata["ss_k6_freeze_params"] = " ".join(args.k6_freeze_params)
        return metadata


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = kandinsky6_setup_parser(parser)
    parser.description = "Full-parameter BF16 training for Kandinsky 6"
    parser.add_argument("--full_fp16", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--full_bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train Kandinsky 6 weights directly in BF16 (required)",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="step Adafactor per tensor during backward and free each gradient immediately "
        "(requires --optimizer_type Adafactor and --max_grad_norm 0; required with --blocks_to_swap)",
    )
    parser.add_argument(
        "--adafactor_triton",
        action="store_true",
        help="use the Triton BF16 matrix fast path for fused Adafactor (unsupported tensors use the reference path)",
    )
    parser.add_argument(
        "--block_swap_trainable_ring",
        action="store_true",
        help="coalesce trainable block swap through pinned CPU masters and reusable GPU ring buffers",
    )
    parser.add_argument(
        "--k6_freeze_blocks",
        type=str,
        default=None,
        help="visual_transformer_blocks kept frozen, e.g. 0-9,31; they are saved unchanged",
    )
    parser.add_argument(
        "--k6_freeze_params",
        type=str,
        nargs="+",
        default=None,
        help="regexes (re.search) over transformer parameter names to keep frozen, e.g. audio_text_transformer_blocks",
    )
    parser.add_argument(
        "--mem_eff_save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream the full checkpoint tensor-by-tensor instead of copying the model to host memory first",
    )
    parser.add_argument("--debug_no_save_weights", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(
        network_module=None,
        mixed_precision="bf16",
        optimizer_type="Adafactor",
        optimizer_args=["scale_parameter=False", "relative_step=False", "warmup_init=False"],
        learning_rate=1e-6,
        lr_scheduler="constant_with_warmup",
        max_grad_norm=0.0,
        gradient_checkpointing=True,
        output_name="kandinsky6_full_finetune",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    Kandinsky6Trainer().train(args)


if __name__ == "__main__":
    main()
