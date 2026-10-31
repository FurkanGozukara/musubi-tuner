"""FIBO 2.0 image sizes (Bria's BriaFibo2ImageProcessor, ai-toolkit's src/references.py, ComfyUI's TextEncodeFibo2), with no
model imports: the dataset loader uses them for several reference images."""

from __future__ import annotations

import math
from functools import lru_cache

TRAINED_AREA = 1024 * 1024


@lru_cache(maxsize=256)
def trained_size(aspect_ratio: float, area: int = TRAINED_AREA) -> tuple[int, int]:
    """(width, height): of the multiples of 16 (>= 64 a side) within 10% of area, the least cropping for this aspect,
    then the closest to area (Bria's sizes)."""
    sizes = [
        (w, h)
        for w in range(64, 2 * area // 64, 16)
        for h in range(max(64, 16 * math.ceil(0.9 * area / w / 16)), 16 * math.floor(1.1 * area / w / 16) + 1, 16)
    ]
    return min(
        sizes,
        key=lambda s: (round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9), abs(s[0] * s[1] - area), -s[0]),
    )
