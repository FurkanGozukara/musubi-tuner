#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    VocoderConfigurator,
)
from musubi_tuner.ltx_2.model.video_vae import VideoDecoderConfigurator, decode_video
from musubi_tuner.ltx_2.model.video_vae.model_configurator import VAE_DECODER_COMFY_KEYS_FILTER
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VIDEO_KEY_RE = re.compile(r"^latents_(?P<frames>\d+)x(?P<height>\d+)x(?P<width>\d+)_(?P<dtype>.+)$")
AUDIO_KEY_RE = re.compile(r"^audio_latents_(?P<steps>\d+)x(?P<mel_bins>\d+)x(?P<channels>\d+)_(?P<dtype>.+)$")


@dataclass
class TensorSummary:
    key: str
    shape: list[int]
    dtype: str
    role: str
    has_nan: bool | None = None
    has_inf: bool | None = None
    min: float | None = None
    max: float | None = None


@dataclass
class FileSummary:
    path: str
    relative_path: str
    kind: str = "unknown"
    status: str = "ok"
    output: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    tensors: list[TensorSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _checkpoint_has_fp8(path: str) -> bool:
    try:
        with MemoryEfficientSafeOpen(path) as handle:
            for key in handle.keys():
                dtype = handle.get_tensor(key).dtype
                if str(dtype).startswith("torch.float8"):
                    return True
    except Exception:
        return False
    return False


def _resolve_device(device_arg: str, checkpoint: str | None, dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    device_choice = device_arg
    if device_choice == "auto":
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint and _checkpoint_has_fp8(checkpoint) and device_choice == "cpu" and torch.cuda.is_available():
            logger.warning("Detected fp8 weights; using CUDA for preview.")
            device_choice = "cuda"

    device = torch.device(device_choice)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        logger.warning("CPU preview does not support %s reliably; using float32", dtype)
        dtype = torch.float32
    return device, dtype


def _iter_cache_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for suffix in ("*.safetensors", "*.pt", "*.pth"):
        files.extend(input_path.rglob(suffix))
    return sorted(files)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root if root.is_dir() else root.parent))
    except ValueError:
        return path.name


def _safe_output_stem(relative_path: str) -> Path:
    rel = Path(relative_path)
    return rel.with_suffix("")


def _tensor_stats(tensor: torch.Tensor, *, stats: bool) -> tuple[bool | None, bool | None, float | None, float | None]:
    if not torch.is_floating_point(tensor):
        return None, None, None, None
    finite_tensor = tensor.detach().float()
    has_nan = bool(torch.isnan(finite_tensor).any().item())
    has_inf = bool(torch.isinf(finite_tensor).any().item())
    if not stats:
        return has_nan, has_inf, None, None
    valid = finite_tensor[torch.isfinite(finite_tensor)]
    if valid.numel() == 0:
        return has_nan, has_inf, None, None
    return has_nan, has_inf, float(valid.min().item()), float(valid.max().item())


def _summarize_tensor(key: str, tensor: torch.Tensor, role: str, *, stats: bool) -> TensorSummary:
    has_nan, has_inf, min_value, max_value = _tensor_stats(tensor, stats=stats)
    return TensorSummary(
        key=key,
        shape=[int(v) for v in tensor.shape],
        dtype=str(tensor.dtype).replace("torch.", ""),
        role=role,
        has_nan=has_nan,
        has_inf=has_inf,
        min=min_value,
        max=max_value,
    )


def _load_safetensors_summary(path: Path, *, stats: bool) -> tuple[dict[str, str], list[TensorSummary], dict[str, torch.Tensor]]:
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())

    tensors_for_decode: dict[str, torch.Tensor] = {}
    summaries: list[TensorSummary] = []
    if not keys:
        return metadata, summaries, tensors_for_decode

    loaded = load_file(str(path), device="cpu")
    for key, tensor in loaded.items():
        role = "other"
        if VIDEO_KEY_RE.match(key):
            role = "video"
            tensors_for_decode.setdefault("video", tensor)
        elif AUDIO_KEY_RE.match(key):
            role = "audio"
            tensors_for_decode.setdefault("audio", tensor)
        elif key.endswith("_mask") or "mask" in key:
            role = "mask"
        elif key.endswith("_int32") or key.startswith("audio_lengths_"):
            role = "metadata"
        summaries.append(_summarize_tensor(key, tensor, role, stats=stats))
    return metadata, summaries, tensors_for_decode


def _load_torch_summary(path: Path, *, stats: bool) -> tuple[dict[str, str], list[TensorSummary], dict[str, torch.Tensor]]:
    payload = torch.load(str(path), map_location="cpu")
    summaries: list[TensorSummary] = []
    tensors_for_decode: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    if isinstance(payload, torch.Tensor):
        summaries.append(_summarize_tensor("latents", payload, "unknown", stats=stats))
        tensors_for_decode["video" if payload.dim() in (4, 5) else "audio"] = payload
        return metadata, summaries, tensors_for_decode

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported torch cache payload type: {type(payload).__name__}")

    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            role = "other"
            if key == "latents" or VIDEO_KEY_RE.match(key):
                role = "video" if value.dim() in (4, 5) else "audio"
                tensors_for_decode.setdefault(role, value)
            elif key.startswith("audio_latents"):
                role = "audio"
                tensors_for_decode.setdefault("audio", value)
            summaries.append(_summarize_tensor(str(key), value, role, stats=stats))
        elif isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = str(value)
    return metadata, summaries, tensors_for_decode


def _validate_summary(summary: FileSummary) -> None:
    if not summary.tensors:
        summary.status = "skipped"
        summary.warnings.append("No tensor payloads found.")
        return

    latent_tensors = [t for t in summary.tensors if t.role in {"video", "audio"}]
    if not latent_tensors:
        summary.status = "skipped"
        summary.warnings.append("No video/audio latent tensors found.")
        return

    for tensor in latent_tensors:
        if tensor.role == "video" and len(tensor.shape) not in (4, 5):
            summary.errors.append(f"{tensor.key}: expected video latent rank 4 or 5, got {tensor.shape}")
        if tensor.role == "audio" and len(tensor.shape) not in (3, 4):
            summary.errors.append(f"{tensor.key}: expected audio latent rank 3 or 4, got {tensor.shape}")
        if tensor.has_nan:
            summary.errors.append(f"{tensor.key}: contains NaN values")
        if tensor.has_inf:
            summary.errors.append(f"{tensor.key}: contains Inf values")

    if summary.errors:
        summary.status = "error"


def _save_audio_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.detach().cpu().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] > 2:
        audio = audio[:2, :]
    audio_int16 = (audio.clamp(-1, 1) * 32767.0).to(torch.int16)
    interleaved = audio_int16.t().contiguous().numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(audio_int16.shape[0]))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved)


def _save_video_mp4(path: Path, frame_chunks: list[torch.Tensor], fps: float) -> None:
    import av

    first = frame_chunks[0]
    height, width = int(first.shape[1]), int(first.shape[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 1_000_000
        for chunk in frame_chunks:
            for frame_img in chunk.detach().cpu().numpy():
                frame = av.VideoFrame.from_ndarray(frame_img, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _save_png(path: Path, frames: torch.Tensor) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[0].detach().cpu().numpy()).save(path)


class PreviewModels:
    def __init__(self, checkpoint: str, device: torch.device, dtype: torch.dtype) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        self._video_decoder = None
        self._audio_decoder = None
        self._vocoder = None

    def video_decoder(self):
        if self._video_decoder is None:
            logger.info("Loading LTX-2 video decoder on %s", self.device)
            self._video_decoder = SingleGPUModelBuilder(
                model_path=self.checkpoint,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            ).build(device=self.device, dtype=self.dtype)
            self._video_decoder.eval()
        return self._video_decoder

    def audio_models(self):
        if self._audio_decoder is None or self._vocoder is None:
            logger.info("Loading LTX-2 audio decoder/vocoder on %s", self.device)
            self._audio_decoder = SingleGPUModelBuilder(
                model_path=self.checkpoint,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            ).build(device=self.device, dtype=self.dtype)
            self._vocoder = SingleGPUModelBuilder(
                model_path=self.checkpoint,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            ).build(device=self.device, dtype=self.dtype)
            self._audio_decoder.eval()
            self._vocoder.eval()
        return self._audio_decoder, self._vocoder


def _decode_video_preview(
    latent: torch.Tensor,
    models: PreviewModels,
    output_path: Path,
    *,
    fps: float,
) -> str:
    decoder = models.video_decoder()
    first_param = next(decoder.parameters(), None)
    decode_dtype = first_param.dtype if first_param is not None else models.dtype
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(device=models.device, dtype=decode_dtype)
    with torch.no_grad():
        chunks = [chunk for chunk in decode_video(latent, decoder)]
    if not chunks:
        raise ValueError("Video decoder produced no frames.")
    frame_count = sum(int(chunk.shape[0]) for chunk in chunks)
    if frame_count <= 1:
        png_path = output_path.with_suffix(".png")
        _save_png(png_path, chunks[0])
        return str(png_path)
    mp4_path = output_path.with_suffix(".mp4")
    _save_video_mp4(mp4_path, chunks, fps=fps)
    return str(mp4_path)


def _decode_audio_preview(latent: torch.Tensor, models: PreviewModels, output_path: Path) -> str:
    audio_decoder, vocoder = models.audio_models()
    first_param = next(audio_decoder.parameters(), None)
    decode_dtype = first_param.dtype if first_param is not None else models.dtype
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    latent = latent.to(device=models.device, dtype=decode_dtype)
    with torch.no_grad():
        decoded_audio = audio_decoder(latent)
        audio_waveform = vocoder(decoded_audio).squeeze(0)
    sample_rate = int(getattr(vocoder, "output_sample_rate", 24000))
    wav_path = output_path.with_suffix(".wav")
    _save_audio_wav(wav_path, audio_waveform, sample_rate)
    return str(wav_path)


def _inspect_file(path: Path, root: Path, *, stats: bool) -> tuple[FileSummary, dict[str, torch.Tensor]]:
    summary = FileSummary(path=str(path), relative_path=_relative_path(path, root))
    try:
        if path.suffix.lower() == ".safetensors":
            metadata, tensors, decode_tensors = _load_safetensors_summary(path, stats=stats)
        else:
            metadata, tensors, decode_tensors = _load_torch_summary(path, stats=stats)
        summary.metadata = metadata
        summary.tensors = tensors
        if any(t.role == "video" for t in tensors):
            summary.kind = "video"
        if any(t.role == "audio" for t in tensors):
            summary.kind = "audio" if summary.kind == "unknown" else "audio_video"
        _validate_summary(summary)
        return summary, decode_tensors
    except Exception as exc:
        summary.status = "error"
        summary.errors.append(str(exc))
        return summary, {}


def _write_summary(path: Path, summaries: list[FileSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "files": [asdict(summary) for summary in summaries],
        "counts": {
            "ok": sum(1 for item in summaries if item.status == "ok"),
            "error": sum(1 for item in summaries if item.status == "error"),
            "skipped": sum(1 for item in summaries if item.status == "skipped"),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally decode LTX-2 latent cache files.")
    parser.add_argument("--input", type=str, required=True, help="Cache file or directory to inspect.")
    parser.add_argument("--output", type=str, required=True, help="Directory for preview media and summary.json.")
    parser.add_argument("--checkpoint", type=str, default=None, help="LTX-2 checkpoint for decoding previews.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Decode device.")
    parser.add_argument("--dtype", type=str, default=None, help="Decode dtype. Auto: bf16 on CUDA, fp32 on CPU.")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for decoded video previews.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of latent cache files to process.")
    parser.add_argument("--stats", action="store_true", help="Compute finite min/max values in addition to NaN/Inf checks.")
    parser.add_argument("--no_decode", action="store_true", help="Inspect only, even when --checkpoint is supplied.")
    parser.add_argument("--fail_on_error", action="store_true", help="Exit non-zero if any cache has errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = _iter_cache_files(input_path)
    if args.limit is not None:
        files = files[: max(int(args.limit), 0)]
    if not files:
        raise FileNotFoundError(f"No .safetensors/.pt/.pth cache files found under: {input_path}")

    dtype_default = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dtype = str_to_dtype(args.dtype, dtype_default)
    device, dtype = _resolve_device(args.device, args.checkpoint, dtype)
    models = PreviewModels(args.checkpoint, device, dtype) if args.checkpoint and not args.no_decode else None

    summaries: list[FileSummary] = []
    for path in files:
        summary, decode_tensors = _inspect_file(path, input_path, stats=bool(args.stats))
        if models is not None and summary.status == "ok" and decode_tensors:
            output_stem = output_dir / _safe_output_stem(summary.relative_path)
            try:
                if "video" in decode_tensors:
                    summary.output = _decode_video_preview(decode_tensors["video"], models, output_stem, fps=float(args.fps))
                if "audio" in decode_tensors:
                    audio_output = _decode_audio_preview(
                        decode_tensors["audio"], models, output_stem.with_name(output_stem.name + "_audio")
                    )
                    summary.output = audio_output if summary.output is None else f"{summary.output}; {audio_output}"
            except Exception as exc:
                summary.status = "error"
                summary.errors.append(f"decode failed: {exc}")
        summaries.append(summary)
        logger.info("%s: %s (%s)", summary.status, summary.relative_path, summary.kind)

    summary_path = output_dir / "summary.json"
    _write_summary(summary_path, summaries)
    logger.info("Wrote cache preview summary to %s", summary_path)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.fail_on_error and any(summary.status == "error" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
