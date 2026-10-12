from __future__ import annotations

import hashlib
import json
import logging
import math
import os

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.audio_utils import AudioSpec, add_audio_tolerance_arguments, apply_audio_tolerance_arguments
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.kandinsky6.reference import DEFAULT_REF_PAD_VALUE, DEFAULT_REFERENCE_SLOTS, encode_reference_paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

K6_SAMPLE_RATE = 44100
K6_AUDIO_DOWNSAMPLE = 1024
K6_VIDEO_FPS = 24
K6_REFERENCE_CACHE_FINGERPRINT = "kandinsky6_reference_fingerprint"
K6_REFERENCE_CACHE_FINGERPRINT_VERSION = "1"


def audio_samples_for_frames(frame_count: int) -> int:
    """Return the waveform window whose encoded grid spans the video duration."""
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    latent_frames = math.ceil(frame_count / K6_VIDEO_FPS * K6_SAMPLE_RATE / K6_AUDIO_DOWNSAMPLE)
    return latent_frames * K6_AUDIO_DOWNSAMPLE


def audio_latent_frames_for_video(frame_count: int) -> int:
    return audio_samples_for_frames(frame_count) // K6_AUDIO_DOWNSAMPLE


K6_AUDIO_SPEC = AudioSpec(
    sample_rate=K6_SAMPLE_RATE,
    channels=1,
    samples_per_crop=audio_samples_for_frames,
)


def _encode_pixels(video_vae, pixels: torch.Tensor) -> torch.Tensor:
    device = next(video_vae.parameters()).device
    dtype = next(video_vae.parameters()).dtype
    posterior = video_vae.encode(pixels.unsqueeze(0).to(device, dtype)).latent_dist
    return posterior.sample()[0] * video_vae.config.scaling_factor


def prepare_video_pixels(content: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    data = np.stack(content) if isinstance(content, list) else np.asarray(content)
    if data.ndim == 3:
        data = data[None]
    if data.ndim != 4 or data.shape[-1] not in (3, 4):
        raise ValueError(f"Kandinsky6 content must be [F,H,W,C], got {data.shape}")
    data = data[..., :3]
    return torch.from_numpy(data).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def first_control_frame(content: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Return one external condition frame from directory/JSONL dataset layouts."""
    if isinstance(content, list):
        if not content:
            raise ValueError("Kandinsky6 image condition list is empty")
        content = content[0]
    frame = np.asarray(content)
    if frame.ndim == 4:
        if frame.shape[0] == 0:
            raise ValueError("Kandinsky6 image condition has no frames")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Kandinsky6 image condition must be [H,W,C] or [F,H,W,C], got {frame.shape}")
    return frame


def reference_paths_for_item(dataset, item: ItemInfo) -> list[str]:
    """Absolute image paths of the record's R2AV ``references`` field ([] when absent).

    Each entry is ``{"type": "image", "path": ...}`` or a bare path string; relative paths
    resolve from the JSONL's directory. Video references are not supported yet.
    """
    datasource = getattr(dataset, "datasource", None)
    if datasource is None or item.datasource_index is None or not hasattr(datasource, "get_item_extras"):
        return []
    extras = datasource.get_item_extras(item.datasource_index)
    entries = extras.fields.get("references")
    if not entries:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{extras.label}: 'references' must be a list")
    paths = []
    for entry in entries:
        if isinstance(entry, str):
            kind, path = "image", entry
        elif isinstance(entry, dict):
            kind, path = str(entry.get("type", "image")).lower(), entry.get("path")
        else:
            raise ValueError(f"{extras.label}: invalid reference entry {entry!r}")
        if kind != "image":
            raise ValueError(f"{extras.label}: Kandinsky 6 R2AV supports image references only, got type {kind!r}")
        if not path:
            raise ValueError(f"{extras.label}: reference entry without a path")
        if not os.path.isabs(path):
            path = os.path.join(extras.base_directory, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{extras.label}: reference image not found: {path}")
        paths.append(path)
    return paths


def reference_cache_fingerprint(reference_paths: list[str], width: int, height: int, ref_pad_value: int) -> str:
    """Fingerprint the ordered reference inputs and every setting that affects their latents."""
    references = []
    for path in reference_paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        references.append({"path": os.path.normcase(os.path.abspath(path)), "sha256": digest.hexdigest()})
    payload = {
        "version": K6_REFERENCE_CACHE_FINGERPRINT_VERSION,
        "references": references,
        "canvas": [width, height],
        "ref_pad_value": ref_pad_value,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_canvas_for_item(item: ItemInfo) -> tuple[int, int]:
    if item.content is not None:
        data = item.content[0] if isinstance(item.content, list) else item.content
        if data.ndim == 3:
            height, width = data.shape[:2]
        elif data.ndim == 4:
            height, width = data.shape[1:3]
        else:
            raise ValueError(f"Kandinsky6 content must be [F,H,W,C], got {data.shape}")
        return int(width), int(height)
    if item.bucket_size is None:
        raise ValueError(f"{item.item_key}: cannot determine the reference cache canvas")
    width, height = item.bucket_size
    return int(width), int(height)


@torch.no_grad()
def encode_batch(
    video_vae,
    audio_vae,
    batch: list[ItemInfo],
    dataset=None,
    ref_pad_value: int = DEFAULT_REF_PAD_VALUE,
    max_references: int | None = None,
) -> None:
    from musubi_tuner.dataset.cache_io import save_latent_cache_kandinsky6

    for item in batch:
        if item.content is None:
            raise ValueError(f"Content not loaded for item {item.item_key}")
        pixels = prepare_video_pixels(item.content)
        latent = _encode_pixels(video_vae, pixels)  # [C,T,H,W]

        audio_latent = None
        audio_present = False
        if audio_vae is not None:
            if item.frame_count is None:
                # ImageDataset does not receive AudioSpec. Supply the one-frame silence
                # target locally so joint K6 batches retain a valid audio tensor which
                # audio_present=False masks out of the loss.
                waveform = torch.zeros(1, audio_samples_for_frames(1), dtype=torch.float32)
            else:
                if item.audio_content is None or item.audio_present is None:
                    raise ValueError(f"Audio was not loaded for audio-enabled video item {item.item_key}")
                waveform = item.audio_content
                audio_present = bool(item.audio_present)
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(f"Kandinsky6 waveform must be mono [1,L], got {tuple(waveform.shape)}")
            waveform = waveform.to(device=audio_vae.device, dtype=audio_vae.dtype)
            # Training targets use the deterministic posterior mean, matching upstream
            # FeaturesUtils.wrapped_encode. Keep [C_audio,A]; the trainer transposes to [A,C].
            audio_latent = audio_vae.encode_audio(waveform).mean[0] * audio_vae.scaling_factor
            expected_audio_frames = audio_latent_frames_for_video(int(item.frame_count or 1))
            if audio_latent.ndim != 2 or audio_latent.shape[-1] != expected_audio_frames:
                raise ValueError(
                    "Kandinsky6 audio VAE returned an unexpected latent grid: expected "
                    f"[C,{expected_audio_frames}], got {tuple(audio_latent.shape)}"
                )

        image_latent = None
        if item.control_content is not None:
            control = first_control_frame(item.control_content)
            control_pixels = prepare_video_pixels(control)
            image_latent = _encode_pixels(video_vae, control_pixels)
            if image_latent.shape[1] != 1:
                raise ValueError(f"Kandinsky6 image condition must encode to one latent frame, got {image_latent.shape[1]}")
            if image_latent.shape[2:] != latent.shape[2:]:
                raise ValueError(
                    "Kandinsky6 image condition must match the target latent canvas, got "
                    f"{tuple(image_latent.shape[2:])} versus {tuple(latent.shape[2:])}"
                )

        reference_latents = None
        reference_paths = reference_paths_for_item(dataset, item) if dataset is not None else []
        height, width = int(pixels.shape[2]), int(pixels.shape[3])
        reference_fingerprint = reference_cache_fingerprint(reference_paths, width, height, ref_pad_value)
        if reference_paths:
            if image_latent is not None:
                raise ValueError(f"{item.item_key}: a Kandinsky 6 item cannot carry both a TI2AV control image and references")
            if max_references is not None and len(reference_paths) > max_references:
                raise ValueError(f"{item.item_key}: {len(reference_paths)} references exceed --max_references {max_references}")
            # letterbox onto the (bucketed) target canvas so references share its latent grid
            reference_latents = encode_reference_paths(video_vae, reference_paths, width, height, ref_pad_value)
            if reference_latents.shape[2:] != latent.shape[2:]:
                raise ValueError(
                    f"{item.item_key}: reference latents {tuple(reference_latents.shape[2:])} do not match the "
                    f"target canvas {tuple(latent.shape[2:])}"
                )
            reference_latents = reference_latents.to(latent.dtype)
        save_latent_cache_kandinsky6(
            item,
            video_latent=latent.cpu(),
            audio_latent=audio_latent.cpu(),
            audio_present=audio_present,
            image_latent=None if image_latent is None else image_latent.cpu(),
            metadata={
                "video_vae_scaling_factor": str(float(video_vae.config.scaling_factor)),
                "audio_vae_scaling_factor": str(float(audio_vae.scaling_factor)),
                "audio_sample_rate": str(K6_SAMPLE_RATE),
                "video_fps": str(K6_VIDEO_FPS),
                K6_REFERENCE_CACHE_FINGERPRINT: reference_fingerprint,
            },
            reference_latents=None if reference_latents is None else reference_latents.cpu(),
        )


def reference_cache_is_current(dataset, item: ItemInfo, ref_pad_value: int = DEFAULT_REF_PAD_VALUE) -> bool:
    """Return whether the cache matches the item's ordered reference inputs and encoding settings."""
    if not os.path.exists(item.latent_cache_path):
        return False
    reference_paths = reference_paths_for_item(dataset, item)
    from safetensors import safe_open

    with safe_open(item.latent_cache_path, framework="pt") as handle:
        keys = [key for key in handle.keys() if key.startswith("varlen_latents_refs_")]
        cached_fingerprint = (handle.metadata() or {}).get(K6_REFERENCE_CACHE_FINGERPRINT)

    if not reference_paths and cached_fingerprint is None:
        # Pre-fingerprint caches are safe only for ordinary items that never held references.
        return not keys
    if len(keys) != (1 if reference_paths else 0):
        return False
    width, height = reference_canvas_for_item(item)
    expected_fingerprint = reference_cache_fingerprint(reference_paths, width, height, ref_pad_value)
    return cached_fingerprint == expected_fingerprint


def main() -> None:
    parser = cache_latents.setup_parser_common()
    vae_action = next(action for action in parser._actions if action.dest == "vae")
    vae_action.required = True
    vae_action.help = "Kandinsky6 HunyuanVideo VAE directory"
    parser.add_argument("--audio_vae", required=True, help="Kandinsky6 MMAudio VAE checkpoint")
    parser.add_argument("--audio_vae_scaling_factor", type=float, default=0.5302)
    parser.add_argument(
        "--ref_pad_value",
        type=int,
        default=DEFAULT_REF_PAD_VALUE,
        help="R2AV: grey level (0-255) of the letterbox around reference images; keep it equal at inference",
    )
    parser.add_argument(
        "--max_references",
        type=int,
        default=DEFAULT_REFERENCE_SLOTS,
        help="R2AV: reject records with more reference images than the model's reference slots",
    )
    parser.add_argument("--upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    parser.add_argument("--sr_upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    add_audio_tolerance_arguments(parser)
    args = parser.parse_args()
    if not 0 <= args.ref_pad_value <= 255:
        raise ValueError("--ref_pad_value must be in [0, 255]")

    try:
        from musubi_tuner.kandinsky6.runtime.core.components.vae_audio import build_audio_vae
        from musubi_tuner.kandinsky6.runtime.core.components.vae_video import build_vae
    except ImportError as exc:
        raise RuntimeError(
            "Kandinsky6 caching requires the optional Kandinsky6 dependencies; install `musubi-tuner[kandinsky6]`."
        ) from exc
    from musubi_tuner.dataset.architectures import ARCHITECTURE_KANDINSKY6

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_KANDINSKY6
    )
    audio_spec = apply_audio_tolerance_arguments(K6_AUDIO_SPEC, args)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, audio_spec=audio_spec).datasets
    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    video_vae = build_vae(args.vae, device=device)
    audio_vae = build_audio_vae(
        tod_vae_ckpt=args.audio_vae,
        device=device,
        need_vae_decoder=False,
        scaling_factor=args.audio_vae_scaling_factor,
    )
    # One dataset at a time so encode_batch can read each record's JSONL `references`.
    for dataset in datasets:
        cache_latents.encode_datasets(
            [dataset],
            lambda batch, dataset=dataset: encode_batch(
                video_vae, audio_vae, batch, dataset, args.ref_pad_value, args.max_references
            ),
            args,
            cache_is_current=lambda item, dataset=dataset: reference_cache_is_current(dataset, item, args.ref_pad_value),
        )


if __name__ == "__main__":
    main()
