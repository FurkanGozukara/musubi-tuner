"""Bounded, selective activation offload for LTX gradient checkpoints."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def activation_view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify the exact live tensor view saved by autograd."""
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _flat_storage_view(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return a contiguous view over a dense tensor's storage span."""
    if tensor.numel() == 0 or any(stride < 0 for stride in tensor.stride()):
        return None
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    if span != tensor.numel():
        return None
    return torch.as_strided(
        tensor,
        (tensor.numel(),),
        (1,),
        tensor.storage_offset(),
    )


@dataclass(frozen=True)
class LTX2ActivationOffloadConfig:
    max_inflight: int = 2
    keep_trailing: int = 2
    min_bytes: int = 1024 * 1024


@dataclass
class _ActivationHandle:
    cpu: torch.Tensor | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    dense: bool
    gpu: torch.Tensor | None = None
    reload_event: torch.cuda.Event | None = None


@dataclass
class _ForwardState:
    handles: dict[int, list[_ActivationHandle]] = field(default_factory=dict)


class BoundedActivationOffloader:
    """Offload only checkpoint boundary activations with bounded CUDA run-ahead."""

    def __init__(
        self,
        config: LTX2ActivationOffloadConfig,
        *,
        total_blocks: int,
    ) -> None:
        self.config = config
        self.total_blocks = int(total_blocks)
        self.current_state: _ForwardState | None = None
        self._device: torch.device | None = None
        self._transfer_stream: torch.cuda.Stream | None = None
        self._compute_events: deque[torch.cuda.Event] = deque()
        self._transfer_events: deque[torch.cuda.Event] = deque()
        self._warned_non_dense = False

    @property
    def first_trailing_block(self) -> int:
        return max(0, self.total_blocks - self.config.keep_trailing)

    def _ensure_stream(self, device: torch.device) -> torch.cuda.Stream:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("LTX bounded activation offload requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._device is None:
            self._device = device
            self._transfer_stream = torch.cuda.Stream(device=device)
        elif self._device != device:
            raise RuntimeError(
                "LTX bounded activation offload supports one CUDA device per transformer; "
                f"configured for {self._device}, received {device}"
            )
        assert self._transfer_stream is not None
        return self._transfer_stream

    def start_forward(self, device: torch.device) -> _ForwardState:
        self._ensure_stream(device)
        self._discard_completed_events(self._compute_events)
        self._discard_completed_events(self._transfer_events)
        state = _ForwardState()
        self.current_state = state
        return state

    @staticmethod
    def _discard_completed_events(events: deque[torch.cuda.Event]) -> None:
        while events and events[0].query():
            events.popleft()

    def before_block(self) -> None:
        limit = self.config.max_inflight
        for events in (self._compute_events, self._transfer_events):
            self._discard_completed_events(events)
            while len(events) >= limit:
                events.popleft().synchronize()

    def after_block(self, device: torch.device) -> None:
        transfer_stream = self._ensure_stream(device)
        current_stream = torch.cuda.current_stream(device)
        self._compute_events.append(current_stream.record_event())
        self._transfer_events.append(transfer_stream.record_event())

    def _should_offload(self, block_idx: int, tensor: torch.Tensor) -> bool:
        return (
            block_idx < self.first_trailing_block
            and tensor.device.type == "cuda"
            and tensor.numel() * tensor.element_size() >= self.config.min_bytes
        )

    def pack_activation(
        self,
        state: _ForwardState,
        block_idx: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor | _ActivationHandle:
        if not self._should_offload(block_idx, tensor):
            return tensor

        transfer_stream = self._ensure_stream(tensor.device)
        current_stream = torch.cuda.current_stream(tensor.device)
        source = _flat_storage_view(tensor)
        dense = source is not None
        if dense:
            cpu = torch.empty(
                tensor.numel(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            destination = cpu
        else:
            cpu = torch.empty_strided(
                tensor.shape,
                tensor.stride(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            source = tensor
            destination = cpu
            if not self._warned_non_dense:
                logger.warning(
                    "LTX bounded activation offload encountered a non-dense boundary "
                    "view at block %s (shape=%s stride=%s); using the logical copy path",
                    block_idx,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                self._warned_non_dense = True

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_stream(current_stream)
            destination.copy_(source, non_blocking=True)
            tensor.record_stream(transfer_stream)

        handle = _ActivationHandle(
            cpu=cpu,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            dense=dense,
        )
        state.handles.setdefault(int(block_idx), []).append(handle)
        return handle

    def _reload(self, handle: _ActivationHandle) -> None:
        if handle.gpu is not None:
            return
        if handle.cpu is None or self._device is None:
            raise RuntimeError("LTX bounded activation handle was released before reload")

        current_stream = torch.cuda.current_stream(self._device)
        transfer_stream = self._ensure_stream(self._device)
        with torch.cuda.stream(current_stream):
            if handle.dense:
                flat = torch.empty(
                    handle.cpu.numel(),
                    dtype=handle.dtype,
                    device=self._device,
                )
                gpu = torch.as_strided(flat, handle.shape, handle.stride)
            else:
                gpu = torch.empty_strided(
                    handle.shape,
                    handle.stride,
                    dtype=handle.dtype,
                    device=self._device,
                )
            allocation_event = current_stream.record_event()

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(allocation_event)
            if handle.dense:
                destination = _flat_storage_view(gpu)
                if destination is None:
                    raise RuntimeError("Dense activation reload did not preserve dense storage")
                destination.copy_(handle.cpu.view(-1), non_blocking=True)
            else:
                gpu.copy_(handle.cpu, non_blocking=True)
            gpu.record_stream(transfer_stream)
            handle.reload_event = transfer_stream.record_event()
        handle.gpu = gpu

    def prefetch(self, state: _ForwardState, block_idx: int) -> None:
        if block_idx < 0:
            return
        for handle in state.handles.get(int(block_idx), ()):
            self._reload(handle)

    def unpack_activation(
        self,
        value: torch.Tensor | _ActivationHandle,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        self._reload(value)
        if value.gpu is None or value.reload_event is None or self._device is None:
            raise RuntimeError("LTX bounded activation reload did not produce a CUDA tensor")

        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_event(value.reload_event)
        tensor = value.gpu
        tensor.record_stream(current_stream)
        value.gpu = None
        value.reload_event = None
        return tensor

    def wrap_outputs(
        self,
        state: _ForwardState,
        block_idx: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        prefetch_idx = int(block_idx) - 1

        def prefetch_hook(grad: torch.Tensor) -> torch.Tensor:
            self.prefetch(state, prefetch_idx)
            return grad

        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(prefetch_hook)
        return tensors


def configure_ltx2_bounded_activation_offload(
    transformer: torch.nn.Module,
    args: argparse.Namespace,
) -> BoundedActivationOffloader | None:
    """Validate and attach the opt-in LTX bounded activation offloader."""
    if not bool(getattr(args, "ltx2_bounded_activation_offload", False)):
        return None
    if not bool(getattr(args, "gradient_checkpointing", False)):
        raise ValueError("--ltx2_bounded_activation_offload requires --gradient_checkpointing")

    conflicts = {
        "--gradient_checkpointing_cpu_offload": bool(getattr(args, "gradient_checkpointing_cpu_offload", False)),
        "--blockwise_checkpointing": bool(getattr(args, "blockwise_checkpointing", False)),
        "--ltx2_partial_gradient_checkpointing": bool(getattr(args, "ltx2_partial_gradient_checkpointing", False)),
        "--compile": bool(getattr(args, "compile", False)),
        "--ltx2_model_parallel": bool(getattr(args, "ltx2_model_parallel", False)),
        "--ltx2_remote_stage": bool(getattr(args, "ltx2_remote_stage", False)),
    }
    active_conflicts = [name for name, active in conflicts.items() if active]
    if active_conflicts:
        raise ValueError("--ltx2_bounded_activation_offload cannot be combined with: " + ", ".join(active_conflicts))

    raw_max_inflight = getattr(args, "ltx2_activation_offload_max_inflight", None)
    raw_keep_trailing = getattr(
        args,
        "ltx2_activation_offload_keep_trailing",
        None,
    )
    raw_min_mb = getattr(args, "ltx2_activation_offload_min_mb", None)
    max_inflight = 2 if raw_max_inflight is None else int(raw_max_inflight)
    keep_trailing = 2 if raw_keep_trailing is None else int(raw_keep_trailing)
    min_mb = 1.0 if raw_min_mb is None else float(raw_min_mb)
    if max_inflight < 1:
        raise ValueError("--ltx2_activation_offload_max_inflight must be >= 1")
    if keep_trailing < 0:
        raise ValueError("--ltx2_activation_offload_keep_trailing must be >= 0")
    if min_mb < 0:
        raise ValueError("--ltx2_activation_offload_min_mb must be >= 0")

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("LTX bounded activation offload requires transformer_blocks")
    if keep_trailing >= len(blocks):
        raise ValueError(f"--ltx2_activation_offload_keep_trailing must be smaller than the transformer depth ({len(blocks)})")

    config = LTX2ActivationOffloadConfig(
        max_inflight=max_inflight,
        keep_trailing=keep_trailing,
        min_bytes=int(min_mb * 1024 * 1024),
    )
    controller = BoundedActivationOffloader(config, total_blocks=len(blocks))
    transformer._ltx2_bounded_activation_offloader = controller
    core_model = getattr(transformer, "model", None)
    if isinstance(core_model, torch.nn.Module):
        core_model._ltx2_bounded_activation_offloader = controller
    for block in blocks:
        block._ltx2_bounded_activation_offloader = controller

    logger.warning(
        "LTX bounded activation offload enabled: max_inflight=%s keep_trailing=%s min_mb=%s",
        max_inflight,
        keep_trailing,
        min_mb,
    )
    return controller
