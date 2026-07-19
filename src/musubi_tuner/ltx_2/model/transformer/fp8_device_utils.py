import os
import threading
import weakref
from functools import wraps
from typing import Callable, TypeVar

import torch

_LOGGED_MISMATCH = False
# Global flag to disable auto-moves when block swap is handling all moves
_BLOCK_SWAP_ACTIVE = False
# Track which block index is currently being processed by swap logic
# Only disable auto-moves for THIS block (others need auto-load for gradient checkpointing)
_CURRENT_SWAP_BLOCK_IDX = -1
_FP8_PLACEMENT_SCOPE_ENABLED: bool | None = None
_PLACEMENT_SCOPE_STATE = threading.local()

_Forward = TypeVar("_Forward", bound=Callable)
_PlacementScope = tuple[tuple[str, int | None], frozenset[int], bool, bool]
_PendingScope = tuple[weakref.ReferenceType[torch.nn.Module], _PlacementScope]


def fp8_placement_scope_enabled() -> bool:
    enabled = _FP8_PLACEMENT_SCOPE_ENABLED
    if enabled is None:
        enabled = os.getenv("LTX2_FP8_PLACEMENT_SCOPE", "0") == "1"
    return enabled and not torch.compiler.is_compiling()


def _device_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    index = device.index
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return device.type, index


def _scope_stack() -> list[_PlacementScope]:
    stack = getattr(_PLACEMENT_SCOPE_STATE, "stack", None)
    if stack is None:
        stack = []
        _PLACEMENT_SCOPE_STATE.stack = stack
    return stack


def _pending_scopes() -> dict[int, _PendingScope]:
    pending = getattr(_PLACEMENT_SCOPE_STATE, "pending", None)
    if pending is None:
        pending = {}
        _PLACEMENT_SCOPE_STATE.pending = pending
    return pending


def _scope_covers(
    scope: _PlacementScope,
    module: torch.nn.Module,
    target_device: torch.device,
    only_lora: bool,
    skip_trainable: bool,
) -> bool:
    device_key, module_ids, scope_only_lora, scope_skip_trainable = scope
    if device_key != _device_key(target_device) or id(module) not in module_ids:
        return False
    if scope_only_lora and not only_lora:
        return False
    return not (device_key[0] == "cpu" and scope_skip_trainable and not skip_trainable)


def prepare_fp8_placement_scope(
    module: torch.nn.Module,
    target_device: torch.device,
    verified_module_ids: frozenset[int] | None,
    *,
    only_lora: bool = False,
    skip_trainable: bool = True,
) -> None:
    if not fp8_placement_scope_enabled() or not verified_module_ids or id(module) not in verified_module_ids:
        return
    _pending_scopes()[id(module)] = (
        weakref.ref(module),
        (_device_key(target_device), verified_module_ids, only_lora, skip_trainable),
    )


def _take_prepared_scope(module: torch.nn.Module):
    record = _pending_scopes().pop(id(module), None)
    if record is None or record[0]() is not module:
        return None
    return record[1]


def _find_tensor_device(values) -> torch.device | None:
    stack = list(values)
    seen: set[int] = set()
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            return value.device
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        else:
            for name in ("x", "latent"):
                tensor = getattr(value, name, None)
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
    return None


def fp8_placement_scoped_forward(*, prepared_only: bool = False):
    def decorate(forward: _Forward) -> _Forward:
        @wraps(forward)
        def wrapped(module: torch.nn.Module, *args, **kwargs):
            if not fp8_placement_scope_enabled():
                return forward(module, *args, **kwargs)

            active = _scope_stack()
            if active and id(module) in active[-1][1]:
                return forward(module, *args, **kwargs)

            scope = _take_prepared_scope(module)
            if scope is None and not prepared_only:
                target_device = _find_tensor_device((args, kwargs))
                if target_device is not None:
                    verified = ensure_fp8_modules_on_device(module, target_device)
                    if verified:
                        scope = (_device_key(target_device), verified, False, True)
            if scope is None:
                return forward(module, *args, **kwargs)

            active.append(scope)
            try:
                return forward(module, *args, **kwargs)
            finally:
                active.pop()

        return wrapped

    return decorate


def set_block_swap_active(active: bool) -> None:
    """Set global flag to disable ensure_fp8_modules_on_device auto-moves."""
    global _BLOCK_SWAP_ACTIVE
    _BLOCK_SWAP_ACTIVE = active


def set_current_swap_block(block_idx: int) -> None:
    """Track which block is currently being processed by swap logic."""
    global _CURRENT_SWAP_BLOCK_IDX
    _CURRENT_SWAP_BLOCK_IDX = block_idx


def get_current_swap_block() -> int:
    """Get the block index currently being processed by swap logic."""
    return _CURRENT_SWAP_BLOCK_IDX


def _is_norm_module(module: torch.nn.Module) -> bool:
    if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
        return True
    name = module.__class__.__name__
    return name.endswith("RMSNorm") or name.endswith("LayerNorm")


def ensure_fp8_modules_on_device(
    module: torch.nn.Module, target_device: torch.device, only_lora: bool = False, skip_trainable: bool = True
) -> frozenset[int] | None:
    """Move FP8 module components to target device.

    Args:
        module: Module to process
        target_device: Target device
        only_lora: If True, only move LoRA modules
        skip_trainable: If True AND target is CPU, skip parameters with requires_grad=True

    NOTE: Classic block swap may auto-load a CPU weight for gradient-checkpoint recomputation.
    H2D-only ring-managed weights are the exception: their offloader must bind a ring view,
    because moving their CPU-master Parameter here would create an untracked device allocation.
    """
    global _LOGGED_MISMATCH

    if fp8_placement_scope_enabled():
        active = _scope_stack()
        if active and _scope_covers(active[-1], module, target_device, only_lora, skip_trainable):
            return active[-1][1]
        verified_module_ids: set[int] | None = set()
    else:
        verified_module_ids = None

    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and target_device.type == "cpu"

    for submodule in module.modules():
        if verified_module_ids is not None:
            verified_module_ids.add(id(submodule))
        if only_lora:
            allow_weight_move = False
            allow_norm_move = False
        else:
            allow_weight_move = isinstance(submodule, torch.nn.Linear) or submodule.__class__.__name__.endswith("Linear")
            allow_norm_move = _is_norm_module(submodule)

        if not only_lora:
            weight = getattr(submodule, "weight", None)

            # Compute avoid_weight_move per-submodule based on actual weight location
            # Classic swap avoids a redundant move only when the weight is already on target.
            # Its CPU weights may still auto-load for gradient-checkpoint recomputation.
            submodule_swap_attr = getattr(submodule, "swap_weight_offload", False)
            h2d_stream_managed = getattr(submodule, "_h2d_stream_managed", False)
            weight_already_on_target = weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
            # H2D-only weights must only ever be rebound to the offloader's CPU masters or GPU ring views.
            # Calling module.to() here would mutate the CPU-master Parameter into a standalone CUDA allocation
            # and permanently grow residency beyond the configured ring.
            avoid_weight_move = bool(h2d_stream_managed) or (
                bool(submodule_swap_attr) and target_device.type == "cuda" and weight_already_on_target
            )

            if (
                hasattr(submodule, "weight")
                and weight is not None
                and isinstance(weight, torch.Tensor)
                and weight.device != target_device
            ):
                # Skip trainable parameters only when offloading to CPU
                if should_skip_trainable and hasattr(weight, "requires_grad") and weight.requires_grad:
                    pass  # Skip
                elif not _LOGGED_MISMATCH:
                    _LOGGED_MISMATCH = True
                    print(f"[LTX-2 fp8] weight on {weight.device}, target {target_device}: {submodule.__class__.__name__}")
                if (allow_weight_move or allow_norm_move) and not avoid_weight_move:
                    if not (should_skip_trainable and hasattr(weight, "requires_grad") and weight.requires_grad):
                        submodule.to(target_device)
                        weight = submodule.weight
            scale_weight = getattr(submodule, "scale_weight", None)
            if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                if scale_weight.device != weight.device:
                    if not (should_skip_trainable and hasattr(scale_weight, "requires_grad") and scale_weight.requires_grad):
                        submodule.scale_weight = scale_weight.to(device=weight.device)
            org_forward = getattr(submodule, "org_forward", None)
            if callable(org_forward):
                orig_module = getattr(org_forward, "__self__", None)
                if isinstance(orig_module, torch.nn.Module):
                    allow_norm_move = _is_norm_module(orig_module)
                    weight = getattr(orig_module, "weight", None)
                    # Compute avoid_weight_move for orig_module
                    orig_swap_attr = getattr(orig_module, "swap_weight_offload", False)
                    orig_h2d_stream_managed = getattr(orig_module, "_h2d_stream_managed", False)
                    orig_weight_on_target = (
                        weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
                    )
                    orig_avoid_weight_move = bool(orig_h2d_stream_managed) or (
                        bool(orig_swap_attr) and target_device.type == "cuda" and orig_weight_on_target
                    )

                    if isinstance(weight, torch.Tensor) and weight.device != target_device and not _LOGGED_MISMATCH:
                        if not (should_skip_trainable and hasattr(weight, "requires_grad") and weight.requires_grad):
                            _LOGGED_MISMATCH = True
                            print(
                                f"[LTX-2 fp8] org_forward weight on {weight.device}, target {target_device}: {orig_module.__class__.__name__}"
                            )
                    if (
                        (allow_weight_move or allow_norm_move)
                        and isinstance(weight, torch.Tensor)
                        and weight.device != target_device
                        and not orig_avoid_weight_move
                    ):
                        if not (should_skip_trainable and hasattr(weight, "requires_grad") and weight.requires_grad):
                            orig_module.weight.data = weight.data.to(device=target_device)
                            weight = orig_module.weight
                    bias = getattr(orig_module, "bias", None)
                    if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                        if bias.device != weight.device:
                            if not (should_skip_trainable and hasattr(bias, "requires_grad") and bias.requires_grad):
                                bias.data = bias.data.to(device=weight.device)
                    scale_weight = getattr(orig_module, "scale_weight", None)
                    if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                        if scale_weight.device != weight.device:
                            if not (
                                should_skip_trainable and hasattr(scale_weight, "requires_grad") and scale_weight.requires_grad
                            ):
                                orig_module.scale_weight = scale_weight.to(device=weight.device)

        # LoRA replacement stores a bound forward on the original Linear.
        forward_self = getattr(getattr(submodule, "forward", None), "__self__", None)
        if forward_self is not None and forward_self is not submodule:
            if not only_lora:
                org_forward = getattr(forward_self, "org_forward", None)
                if callable(org_forward):
                    orig_module = getattr(org_forward, "__self__", None)
                    if isinstance(orig_module, torch.nn.Module):
                        allow_norm_move = _is_norm_module(orig_module)
                        weight = getattr(orig_module, "weight", None)
                        # Compute avoid_weight_move for orig_module in LoRA context
                        lora_orig_swap_attr = getattr(orig_module, "swap_weight_offload", False)
                        lora_orig_h2d_stream_managed = getattr(orig_module, "_h2d_stream_managed", False)
                        lora_orig_weight_on_target = (
                            weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
                        )
                        lora_orig_avoid_weight_move = bool(lora_orig_h2d_stream_managed) or (
                            bool(lora_orig_swap_attr) and target_device.type == "cuda" and lora_orig_weight_on_target
                        )

                        if (
                            (allow_weight_move or allow_norm_move)
                            and isinstance(weight, torch.Tensor)
                            and weight.device != target_device
                            and not lora_orig_avoid_weight_move
                        ):
                            if not (should_skip_trainable and hasattr(weight, "requires_grad") and weight.requires_grad):
                                orig_module.weight.data = weight.data.to(device=target_device)
                                weight = orig_module.weight
                        bias = getattr(orig_module, "bias", None)
                        if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                            if bias.device != weight.device:
                                if not (should_skip_trainable and hasattr(bias, "requires_grad") and bias.requires_grad):
                                    bias.data = bias.data.to(device=weight.device)
                        scale_weight = getattr(orig_module, "scale_weight", None)
                        if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                            if scale_weight.device != weight.device:
                                if not (
                                    should_skip_trainable and hasattr(scale_weight, "requires_grad") and scale_weight.requires_grad
                                ):
                                    orig_module.scale_weight = scale_weight.to(device=weight.device)
            # Move LoRA module weights (lora_down, lora_up) to target device
            # Skip if should_skip_trainable is True (LoRA weights are trainable, keep on GPU)
            if not should_skip_trainable:
                lora_down = getattr(forward_self, "lora_down", None)
                lora_up = getattr(forward_self, "lora_up", None)

                # Handle both single Linear and ModuleList (split_dims case)
                if isinstance(lora_down, torch.nn.ModuleList):
                    for ld in lora_down:
                        if hasattr(ld, "weight") and ld.weight.device != target_device:
                            ld.to(target_device)
                elif isinstance(lora_down, torch.nn.Module):
                    lora_down_weight = getattr(lora_down, "weight", None)
                    if isinstance(lora_down_weight, torch.Tensor) and lora_down_weight.device != target_device:
                        lora_down.to(target_device)

                if isinstance(lora_up, torch.nn.ModuleList):
                    for lu in lora_up:
                        if hasattr(lu, "weight") and lu.weight.device != target_device:
                            lu.to(target_device)
                elif isinstance(lora_up, torch.nn.Module):
                    lora_up_weight = getattr(lora_up, "weight", None)
                    if isinstance(lora_up_weight, torch.Tensor) and lora_up_weight.device != target_device:
                        lora_up.to(target_device)

    return frozenset(verified_module_ids) if verified_module_ids is not None else None


def move_fp8_scale_weights(module: torch.nn.Module, target_device: torch.device) -> None:
    non_blocking = target_device.type != "cpu"
    for submodule in module.modules():
        scale_weight = getattr(submodule, "scale_weight", None)
        if isinstance(scale_weight, torch.Tensor) and scale_weight.device != target_device:
            submodule.scale_weight = scale_weight.to(device=target_device, non_blocking=non_blocking)
        org_forward = getattr(submodule, "org_forward", None)
        if callable(org_forward):
            orig_module = getattr(org_forward, "__self__", None)
            if isinstance(orig_module, torch.nn.Module):
                scale_weight = getattr(orig_module, "scale_weight", None)
                if isinstance(scale_weight, torch.Tensor) and scale_weight.device != target_device:
                    orig_module.scale_weight = scale_weight.to(device=target_device, non_blocking=non_blocking)
