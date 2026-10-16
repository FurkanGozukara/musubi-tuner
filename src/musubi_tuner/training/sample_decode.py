"""A training sample's VAE decode that falls back to the CPU when the GPU runs out of memory.

Training-time sampling decodes with the DiT still on the GPU, so a preset that fits training can overflow in the
decode: one-to-three-second spikes of +5 to +11 GB at 1024 px (measured on 48 GB cards: Z-Image Turbo 8 GB preset
4.8 -> 9.8 GB, Qwen-Image 2.1 12 GB preset 7.2 -> 18.5 GB). On Linux that is an out-of-memory error that ends the run
(Windows spills into shared memory instead). The decode is then redone with the VAE and the latents on the CPU in
float32, and the following samples of the run decode there directly.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_ON_CPU = "_sample_decode_on_cpu"  # set on a VAE whose decode ran out of GPU memory once


def _out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc)
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text.lower() or "CUDNN_STATUS_ALLOC_FAILED" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text
    )


def _dtype_of(vae) -> torch.dtype | None:
    dtype = getattr(vae, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    parameters = getattr(vae, "parameters", None)
    try:
        return next(iter(parameters())).dtype if callable(parameters) else None
    except (StopIteration, TypeError, AttributeError):
        return None


def decode_sample(vae, decode, latents: torch.Tensor):
    """``decode(latents)`` with ``vae`` where the caller put it; when that runs out of GPU memory, ``decode`` again with
    ``vae`` and the latents on the CPU in float32. ``vae`` stays on the CPU afterwards (callers move it there after
    decoding anyway); ``vae`` is an ``nn.Module`` or anything with the same ``to()`` (Wan's VAE wrapper)."""
    if getattr(vae, _ON_CPU, False) is not True:  # `is`: a mock's attributes are all truthy
        try:
            return decode(latents)
        except Exception as exc:  # noqa: BLE001 - only out-of-memory is retried
            if not _out_of_memory(exc):
                raise
            logger.warning(f"The sample's VAE decode ran out of GPU memory ({type(exc).__name__}): decoding on the CPU from now on (slower)")
        # outside the except block, so the failed attempt's GPU tensors are released with its frames
    setattr(vae, _ON_CPU, True)
    dtype = _dtype_of(vae)
    vae.to("cpu", torch.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return decode(latents.detach().to("cpu", torch.float32))
    finally:
        if dtype is not None:
            vae.to(dtype)
