from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def combine_full_ft_loss_weight(
    resolved_weight: float,
    batch_weight: object,
    cli_weight: float,
) -> float:
    """Combine dataset and CLI weights without applying a CLI fallback twice.

    ``resolved_weight`` is produced by LTX-2 ``call_dit``: it is the dataset
    value when one exists and otherwise is already the CLI value.
    """
    if batch_weight is None:
        return float(resolved_weight)
    return float(resolved_weight) * float(cli_weight)


def distributed_any(accelerator, local_flag: bool, *, device: torch.device | None = None) -> bool:
    """Return the rank-wide logical OR of a local boolean flag."""
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return bool(local_flag)

    flag = torch.tensor(
        1 if local_flag else 0,
        device=device if device is not None else accelerator.device,
        dtype=torch.int32,
    )
    flag = accelerator.reduce(flag, reduction="sum")
    return bool(flag.item() > 0)


def optimizer_step_succeeded(accelerator) -> bool:
    """Whether the current synchronization boundary performed an update."""
    return bool(accelerator.sync_gradients) and not bool(accelerator.optimizer_step_was_skipped)


def synchronize_parameter_gradients(accelerator, parameters: Iterable[torch.nn.Parameter]) -> None:
    """Mean-reduce gradients for optimizer-only parameters not owned by DDP."""
    if not bool(accelerator.sync_gradients) or int(getattr(accelerator, "num_processes", 1)) <= 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = accelerator.reduce(parameter.grad, reduction="mean")


@dataclass
class AccumulationSkipGuard:
    """Discard an entire accumulation window after a bad microbatch.

    Accelerate advances its accumulation phase when entering ``accumulate``.
    Once one microbatch is unusable, subsequent microbatches are ignored until
    the next synchronization boundary so an under-filled update is never made.
    """

    dropping_window: bool = False

    def mark_bad_microbatch(self, *, sync_gradients: bool) -> None:
        self.dropping_window = not bool(sync_gradients)

    def should_drop_current_microbatch(self, *, sync_gradients: bool) -> bool:
        if not self.dropping_window:
            return False
        if sync_gradients:
            self.dropping_window = False
        return True
