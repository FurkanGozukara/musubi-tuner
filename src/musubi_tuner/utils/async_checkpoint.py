from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


def validate_async_checkpoint_save_options(args, *, fsdp_enabled: bool, remote_stage_enabled: bool) -> None:
    if not getattr(args, "async_checkpoint_save", False):
        return

    conflicts = []
    if fsdp_enabled:
        conflicts.append("--ltx2_fsdp")
    if remote_stage_enabled:
        conflicts.append("--ltx2_remote_stage")
    if getattr(args, "use_ema", False):
        conflicts.append("--use_ema")
    if getattr(args, "huggingface_repo_id", None) is not None:
        conflicts.append("--huggingface_repo_id")
    if getattr(args, "qgalore_streaming_dequantize_save", False):
        conflicts.append("--qgalore_streaming_dequantize_save")
    if getattr(args, "int8_weights", False):
        conflicts.append("--int8_weights")
    if getattr(args, "save_merged_checkpoint", False):
        conflicts.append("--save_merged_checkpoint")
    if conflicts:
        raise ValueError(f"--async_checkpoint_save is incompatible with: {', '.join(conflicts)}.")


@dataclass(frozen=True)
class AsyncCheckpointStats:
    snapshot_seconds: float
    snapshot_bytes: int


class AsyncCheckpointSaver:
    def __init__(
        self,
        save_fn: Callable[[dict[str, torch.Tensor], str, dict[str, Any]], None],
        on_write_complete: Callable[[str, float], None] | None = None,
    ) -> None:
        self._save_fn = save_fn
        self._on_write_complete = on_write_complete
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._thread is not None

    def wait(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None
        error = self._error
        self._error = None
        if error is not None:
            raise RuntimeError("asynchronous checkpoint write failed") from error

    def start(
        self,
        tensors: Mapping[str, torch.Tensor],
        filename: str,
        metadata: Mapping[str, Any],
    ) -> AsyncCheckpointStats:
        self.wait()
        started_at = time.perf_counter()
        snapshot: dict[str, torch.Tensor] = {}
        snapshot_bytes = 0
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"asynchronous checkpoint saving requires tensor values, got {type(value)!r} for {name!r}")
            copied = value.detach().to(device="cpu", copy=True).contiguous()
            snapshot[name] = copied
            snapshot_bytes += copied.numel() * copied.element_size()

        metadata_snapshot = dict(metadata)
        stats = AsyncCheckpointStats(
            snapshot_seconds=time.perf_counter() - started_at,
            snapshot_bytes=snapshot_bytes,
        )

        def writer() -> None:
            write_started_at = time.perf_counter()
            try:
                self._save_fn(snapshot, filename, metadata_snapshot)
            except BaseException as exc:
                self._error = exc
                return
            if self._on_write_complete is not None:
                self._on_write_complete(filename, time.perf_counter() - write_started_at)

        self._thread = threading.Thread(target=writer, name="async-checkpoint-writer")
        self._thread.start()
        return stats
