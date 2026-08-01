"""Target-frame attention routing for existing LTX-2 reference streams."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch


def normalize_reference_target_frame_ranges(
    raw_ranges: Optional[Sequence[Sequence[int]]],
    *,
    reference_count: int,
) -> tuple[tuple[int, int], ...]:
    """Validate half-open pixel-frame ranges aligned with reference order."""

    if raw_ranges is None:
        return ()
    ranges = list(raw_ranges)
    if len(ranges) != reference_count:
        raise ValueError(
            "reference_target_frame_ranges must have one [start, end] entry per reference directory; "
            f"got {len(ranges)} ranges for {reference_count} references"
        )

    normalized: list[tuple[int, int]] = []
    for index, value in enumerate(ranges):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"reference_target_frame_ranges[{index}] must be [start, end]")
        start, end = value
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"reference_target_frame_ranges[{index}] must satisfy integer 0 <= start < end, got {value!r}")
        normalized.append((start, end))
    return tuple(normalized)


def build_reference_target_range_attention_mask(
    *,
    ranges: Sequence[Sequence[int]],
    positions: torch.Tensor,
    frame_rate: float,
    target_token_start: int,
    target_token_count: int,
    reference_token_spans: Sequence[tuple[int, int]],
    base_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Restrict target queries to each existing reference's pixel-frame range."""

    normalized = normalize_reference_target_frame_ranges(ranges, reference_count=len(reference_token_spans))
    if not normalized:
        return base_mask
    if positions.ndim != 4 or positions.shape[1] < 1 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions shaped [B, axes, tokens, 2], got {tuple(positions.shape)}")
    if not isinstance(frame_rate, (int, float)) or isinstance(frame_rate, bool) or float(frame_rate) <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate!r}")

    total_tokens = int(positions.shape[2])
    target_start = int(target_token_start)
    target_end = target_start + int(target_token_count)
    if target_start < 0 or target_end < target_start or target_end > total_tokens:
        raise ValueError(f"Target token span [{target_start}, {target_end}) is outside sequence length {total_tokens}")

    if base_mask is None:
        mask = torch.ones((positions.shape[0], total_tokens, total_tokens), device=positions.device, dtype=torch.bool)
        restore_head_axis = False
    else:
        if base_mask.ndim == 4 and base_mask.shape[1] == 1:
            mask = base_mask[:, 0].to(device=positions.device, dtype=torch.bool).clone()
            restore_head_axis = True
        elif base_mask.ndim == 3:
            mask = base_mask.to(device=positions.device, dtype=torch.bool).clone()
            restore_head_axis = False
        else:
            raise ValueError(f"base_mask must be [B,T,T] or [B,1,T,T], got {tuple(base_mask.shape)}")
        if mask.shape != (positions.shape[0], total_tokens, total_tokens):
            raise ValueError(
                f"base_mask shape {tuple(mask.shape)} does not match {(positions.shape[0], total_tokens, total_tokens)}"
            )

    target_frames = positions[:, 0, target_start:target_end, 0].to(torch.float32) * float(frame_rate)
    for index, ((range_start, range_end), (stream_start, stream_end)) in enumerate(zip(normalized, reference_token_spans)):
        stream_start, stream_end = int(stream_start), int(stream_end)
        if stream_start < 0 or stream_end <= stream_start or stream_end > total_tokens:
            raise ValueError(f"Reference {index} span [{stream_start}, {stream_end}) is outside sequence length {total_tokens}")
        visible = (target_frames >= float(range_start)) & (target_frames < float(range_end))
        mask[:, target_start:target_end, stream_start:stream_end] &= visible.unsqueeze(-1)

    return mask.unsqueeze(1) if restore_head_axis else mask
