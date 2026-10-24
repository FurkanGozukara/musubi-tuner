"""Which LTX generation a cache was made for, so a cache of one generation is never reused for another.

LTX-2.5 encodes captions with Gemma 4 (LTX-2 / 2.3: Gemma 3) and video with its own VAE, so its text, latent and audio
caches differ from LTX-2.3's even where the shapes agree. Every LTX cache file therefore records the generation in its
safetensors metadata (``ltx_version``); a cache without it was written before the field existed and counts as LTX-2.3.

* Caching with ``--skip_existing`` re-encodes a cache recorded for another version (exact match, so a switch between
  LTX-2.0 and 2.3 re-encodes too, now that the version is recorded).
* Training refuses caches of the other side of the 2.5 boundary (an LTX-2.5 run with LTX-2 / 2.3 caches or the other
  way round) with an instruction to re-run caching. LTX-2.0 and 2.3 caches stay interchangeable at training time, as
  they were: a caching run that was not told ``--ltx_version 2.0`` records 2.3.

The caching scripts record the version of the model files they read: ``--ltx_version`` normally, and LTX-2.5 whenever
the files themselves are LTX-2.5 (a 2.5 transformer, a Gemma 4 text encoder, the 2.5 video VAE).
"""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

CACHE_VERSION_METADATA_KEY = "ltx_version"
LEGACY_CACHE_VERSION = "2.3"  # caches written before the version was recorded
KNOWN_VERSIONS = ("2.0", "2.3", "2.5")


def normalize_ltx_version(value: object, default: str = LEGACY_CACHE_VERSION) -> str:
    """'2.0' / '2.3' / '2.5' for the spellings an argument or metadata may use ('2', '2.3.0', 2.5, ...)."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    parts = text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        return default
    if (major, minor) >= (2, 4):
        return "2.5"
    if (major, minor) >= (2, 3):
        return "2.3"
    return "2.0"


def is_ltx25(version: object) -> bool:
    return normalize_ltx_version(version) == "2.5"


def _read_header(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < length <= 256 * 1024 * 1024:
                return None
            header = json.loads(handle.read(length))
    except (OSError, ValueError, struct.error, MemoryError):
        return None
    return header if isinstance(header, dict) else None


def read_cache_ltx_version(path: str) -> Optional[str]:
    """The LTX version a cache file records ('2.3' for a cache from before the field), or None when unreadable."""
    header = _read_header(path)
    if header is None:
        return None
    metadata = header.get("__metadata__") or {}
    return normalize_ltx_version(metadata.get(CACHE_VERSION_METADATA_KEY), LEGACY_CACHE_VERSION)


def cache_metadata(version: Optional[str]) -> dict[str, str]:
    """The metadata entry a cache writer adds (empty without a version: old call sites write what they wrote)."""
    return {CACHE_VERSION_METADATA_KEY: normalize_ltx_version(version)} if version else {}


def cache_matches_version(path: str, version: Optional[str], *, log: bool = True) -> bool:
    """--skip_existing test: the cache at ``path`` exists and records ``version`` (no version: any cache matches)."""
    if not os.path.exists(path):
        return False
    if not version:
        return True
    cached = read_cache_ltx_version(path)
    if cached is None:
        return True  # unreadable header: keep the plain exists test, the loader reports a broken file
    wanted = normalize_ltx_version(version)
    if cached != wanted:
        if log:
            logger.info(f"Re-encoding {os.path.basename(path)}: cached for LTX-{cached}, this run caches for LTX-{wanted}")
        return False
    return True


def same_side_of_ltx25(cached: str, version: str) -> bool:
    """Training-time compatibility: LTX-2.5 caches only with LTX-2.5 runs; LTX-2.0 / 2.3 with each other (as before)."""
    return is_ltx25(cached) == is_ltx25(version)


# --- the version a caching run writes ------------------------------------------------------------------------


def _safetensors_metadata(path: str) -> dict:
    header = _read_header(path)
    return dict(header.get("__metadata__") or {}) if header else {}


def _tensor_names(path: str) -> list[str]:
    header = _read_header(path)
    return [key for key in header if key != "__metadata__"] if header else []


def file_is_ltx25(path: Optional[str]) -> bool:
    """Whether a model file or directory belongs to LTX-2.5: an official model_version >= 2.4, the 2.5 transformer's
    keyframe embedding, a packed Gemma 4 text encoder, a Gemma 4 model directory, or the 2.5 video VAE layout."""
    if not path:
        return False
    if os.path.isdir(path):
        config_file = os.path.join(path, "config.json")
        try:
            with open(config_file, encoding="utf-8") as handle:
                model_type = str(json.load(handle).get("model_type") or "")
        except (OSError, ValueError):
            return False
        return bool(model_type) and model_type.startswith("gemma") and not model_type.startswith("gemma3")
    if not os.path.isfile(path):
        return False
    metadata = _safetensors_metadata(path)
    if metadata.get("model_version") is not None and is_ltx25(metadata["model_version"]):
        return True
    raw_gemma = metadata.get("gemma_config")
    if raw_gemma:
        try:
            model_type = str(json.loads(raw_gemma).get("model_type") or "")
        except ValueError:
            model_type = ""
        if model_type.startswith("gemma") and not model_type.startswith("gemma3"):
            return True
    raw_config = metadata.get("config")
    if raw_config:
        try:
            vae = (json.loads(raw_config) or {}).get("vae") or {}
        except ValueError:
            vae = {}
        if isinstance(vae, dict) and (vae.get("_class_name", "CausalVideoAutoencoder") != "CausalVideoAutoencoder"):
            return True
    return any(name.endswith("keyframes_abs_pos_embedding") for name in _tensor_names(path))


def resolve_caching_version(args, model_paths: Iterable[Optional[str]]) -> str:
    """The version a caching run records: LTX-2.5 when --ltx_version says so or the files it reads are LTX-2.5,
    else --ltx_version (default 2.3)."""
    requested = normalize_ltx_version(getattr(args, "ltx_version", None))
    if requested == "2.5":
        return "2.5"
    for path in model_paths:
        if file_is_ltx25(path):
            logger.warning(
                "LTX caching: %s is an LTX-2.5 file but --ltx_version is %s; the caches are recorded as LTX-2.5 "
                "(pass --ltx_version 2.5)",
                path,
                requested,
            )
            return "2.5"
    return requested


def find_mismatched_caches(paths: Iterable[Optional[str]], version: str, *, limit: int = 5) -> tuple[int, list[str]]:
    """(count, first ``limit`` paths) of existing caches recorded for the other side of the LTX-2.5 boundary."""
    count = 0
    examples: list[str] = []
    seen = set()
    for path in paths:
        if not path or path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        cached = read_cache_ltx_version(path)
        if cached is None or same_side_of_ltx25(cached, version):
            continue
        count += 1
        if len(examples) < limit:
            examples.append(f"{path} (LTX-{cached})")
    return count, examples


def dataset_cache_paths(datasets) -> list[str]:
    """Latent, text-encoder and audio cache paths of the items the training datasets hold."""
    paths: list[str] = []
    for dataset in datasets:
        manager = getattr(dataset, "batch_manager", None)
        buckets = getattr(manager, "buckets", None) or {}
        for items in buckets.values():
            for item in items:
                for attribute in ("latent_cache_path", "text_encoder_output_cache_path", "audio_latent_cache_path"):
                    path = getattr(item, attribute, None)
                    if path:
                        paths.append(path)
    return paths


def check_training_caches(datasets, version: str) -> None:
    """Refuse to train an LTX-2.5 model on LTX-2 / 2.3 caches or the other way round."""
    count, examples = find_mismatched_caches(dataset_cache_paths(datasets), version)
    if count:
        raise ValueError(
            f"{count} LTX cache file(s) were made for another LTX generation than this LTX-{normalize_ltx_version(version)} "
            "run (LTX-2.5 uses its own text encoder and VAE), e.g. " + "; ".join(examples) + ". Re-run latent and text "
            "encoder caching for this model (with --skip_existing they re-encode only those files)."
        )
