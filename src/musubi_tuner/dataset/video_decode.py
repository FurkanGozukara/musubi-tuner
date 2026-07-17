"""Opt-in alternate video-decode backends (decord/torchcodec) for latent caching.

Selected via LTX2_VIDEO_DECODE_BACKEND / --video_decode_backend; default 'pyav' keeps the original
path. See the Caching Latents section of docs/ltx_2.md for backends, activation and threading.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np

from musubi_tuner.dataset.media_utils import resize_image_to_bucket

logger = logging.getLogger(__name__)

ALT_BACKENDS = ("decord", "torchcodec")


def get_video_decode_backend() -> str:
    """Active backend from the environment; 'pyav' (default) selects the original code path."""
    return os.environ.get("LTX2_VIDEO_DECODE_BACKEND", "pyav").strip().lower()


def _select_source_indices(
    num_frames: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
    source_fps: Optional[float],
    target_fps: Optional[float],
    fps_threshold: float,
) -> list[int]:
    """Source indices to decode, matching image_video_dataset.load_video's selection.

    Resample only when abs(ceil(source_fps) - target_fps) > fps_threshold; otherwise take frames
    [start_frame, end_frame). In resample mode drop duplicate frames and apply start/end in the
    target index space, as the PyAV loop does.
    """
    needs_resampling = source_fps is not None and target_fps is not None and abs(math.ceil(source_fps) - target_fps) > fps_threshold
    if not needs_resampling:
        s = start_frame if start_frame is not None else 0
        e = end_frame if end_frame is not None else num_frames
        return list(range(max(0, s), min(num_frames, e)))

    delta = target_fps / source_fps
    kept: list[int] = []
    frac = 0.0
    prev = -1
    for i in range(num_frames):
        tgt = int(frac)
        frac += delta
        if tgt == prev:
            continue
        prev = tgt
        if start_frame is not None and tgt < start_frame:
            continue
        if end_frame is not None and tgt >= end_frame:
            break
        kept.append(i)
    return kept


def _open_decord(video_path: str):
    import decord

    decord.bridge.set_bridge("native")
    # Default 1: a reader is opened per dataloader worker, so nested per-reader threads would explode
    # the thread count and livelock the host. Parallelism must come from the caller's worker pool.
    nthreads = int(os.environ.get("LTX2_DECORD_NUM_THREADS", "1"))
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=nthreads)
    try:
        fps = float(vr.get_avg_fps())
        fps = fps if fps > 0 else None
    except Exception:
        fps = None
    return vr, fps, len(vr)


def _frames_decord(vr, indices: list[int]) -> list[np.ndarray]:
    batch = vr.get_batch(indices).asnumpy()  # (K, H, W, 3) uint8 RGB
    return [np.ascontiguousarray(batch[i]) for i in range(batch.shape[0])]


def _open_torchcodec(video_path: str):
    from torchcodec.decoders import VideoDecoder

    device = os.environ.get("LTX2_VIDEO_DECODE_DEVICE", "cpu")  # "cuda" = NVDEC hardware decode
    dec = VideoDecoder(video_path, device=device)
    md = dec.metadata
    fps = getattr(md, "average_fps", None)
    return dec, (float(fps) if fps else None), int(md.num_frames)


def _frames_torchcodec(dec, indices: list[int]) -> list[np.ndarray]:
    fb = dec.get_frames_at(indices=indices)  # FrameBatch, .data is (K, C, H, W) uint8
    data = fb.data.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (K, H, W, C) uint8 RGB
    return [np.ascontiguousarray(data[i]) for i in range(data.shape[0])]


def load_video_alt(
    video_path: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
    bucket_selector,
    bucket_reso,
    source_fps: Optional[float],
    target_fps: Optional[float],
    fps_threshold: float,
    backend: Optional[str] = None,
) -> Optional[list[np.ndarray]]:
    """Opt-in alternate decode -> list of HWC RGB uint8 frames resized to bucket.

    Returns None to signal the caller to use the PyAV path (backend disabled, or a non-file path).
    Raises on decode error so the caller can catch and fall back to PyAV.
    """
    backend = backend or get_video_decode_backend()
    if backend not in ALT_BACKENDS:
        return None
    if not os.path.isfile(video_path):
        return None  # image-directory inputs: let the original branch handle them

    if backend == "decord":
        handle, det_fps, n = _open_decord(video_path)
    else:
        handle, det_fps, n = _open_torchcodec(video_path)

    sfps = source_fps if source_fps is not None else det_fps
    indices = _select_source_indices(n, start_frame, end_frame, sfps, target_fps, fps_threshold)
    if not indices:
        return []

    frames = _frames_decord(handle, indices) if backend == "decord" else _frames_torchcodec(handle, indices)

    out: list[np.ndarray] = []
    for fr in frames:
        if bucket_selector is not None and bucket_reso is None:
            h, w = fr.shape[0], fr.shape[1]
            bucket_reso = bucket_selector.get_bucket_resolution((w, h))  # (width, height) from first frame
        if bucket_reso is not None:
            fr = resize_image_to_bucket(fr, bucket_reso)
        out.append(fr)
    return out
