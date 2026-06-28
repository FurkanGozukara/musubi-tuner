"""Cache readiness scan API."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from musubi_tuner.dataset.media_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig

router = APIRouter(prefix="/api/cache", tags=["cache"])

ARCH = "ltx2"
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"]


class CacheBucketStatus(BaseModel):
    bucket: str
    count: int


class CacheDatasetStatus(BaseModel):
    index: int
    group: str
    type: str
    directory: str
    cache_directory: str
    source_count: int = 0
    latent_count: int = 0
    text_count: int = 0
    audio_count: int = 0
    dino_count: int = 0
    missing_latent: int = 0
    missing_text: int = 0
    missing_audio: int = 0
    stale_latent: int = 0
    stale_text: int = 0
    stale_audio: int = 0
    orphan_latent: int = 0
    buckets: list[CacheBucketStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CacheStatusResponse(BaseModel):
    generated_at: str
    rows: list[CacheDatasetStatus]
    totals: dict[str, int]


def _project_config(request: Request) -> ProjectConfig:
    config = getattr(request.app.state, "project_config", None)
    if config is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return config


def _media_extensions(dataset_type: str) -> set[str]:
    if dataset_type == "image":
        return {ext.lower() for ext in IMAGE_EXTENSIONS}
    if dataset_type == "audio":
        return {ext.lower() for ext in AUDIO_EXTENSIONS}
    return {ext.lower() for ext in VIDEO_EXTENSIONS}


def _iter_media_files(path: Path, dataset_type: str) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    exts = _media_extensions(dataset_type)
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def _jsonl_source_keys(dataset_type: str) -> tuple[str, ...]:
    return {
        "video": ("video_path", "video", "video_key", "item_key", "path", "file"),
        "image": ("image_path", "image", "image_key", "item_key", "path", "file"),
        "audio": ("audio_path", "audio", "audio_key", "item_key", "path", "file"),
    }.get(dataset_type, ("item_key", "path", "file"))


def _iter_jsonl_sources(path: Path, dataset_type: str) -> list[Path]:
    if not path.exists() or not path.is_file():
        return []
    sources: list[Path] = []
    keys = _jsonl_source_keys(dataset_type)
    try:
        base = path.parent
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            raw = next((row.get(key) for key in keys if row.get(key)), None)
            if not raw:
                continue
            source = Path(str(raw))
            sources.append(source if source.is_absolute() else base / source)
    except Exception:
        return []
    return sources


def _source_files(entry: DatasetEntry) -> list[Path]:
    jsonl = Path(entry.jsonl_file) if entry.jsonl_file else None
    if jsonl:
        sources = _iter_jsonl_sources(jsonl, entry.type)
        if sources:
            return sources
    directory = Path(entry.directory) if entry.directory else None
    if not directory:
        return []
    return _iter_media_files(directory, entry.type)


def _default_cache_dir(entry: DatasetEntry) -> str:
    if entry.cache_directory:
        return entry.cache_directory
    if entry.directory:
        return str(Path(entry.directory) / "cache")
    if entry.jsonl_file:
        return str(Path(entry.jsonl_file).parent / "cache")
    return ""


def _stem_set(paths: list[Path]) -> set[str]:
    return {path.stem for path in paths}


def _source_key_set(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        keys.add(path.stem)
        keys.add(path.name)
    return keys


def _cache_files(cache_dir: Path, suffix: str) -> list[Path]:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob(f"*{suffix}"))


def _cache_stems(files: list[Path], suffix: str) -> set[str]:
    stems: set[str] = set()
    dim_re = re.compile(r"_(\d{4}x\d{4})$")
    for path in files:
        name = path.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        stems.add(dim_re.sub("", stem))
    return stems


def _matches_source(cache_stem: str, source_keys: set[str]) -> bool:
    if cache_stem in source_keys:
        return True
    return any(cache_stem.startswith(f"{source_key}_") for source_key in source_keys)


def _matched_source_count(cache_stems: set[str], source_keys: set[str]) -> int:
    if not source_keys:
        return len(cache_stems)
    return sum(
        1
        for source_key in source_keys
        if source_key in cache_stems or any(stem.startswith(f"{source_key}_") for stem in cache_stems)
    )


def _bucket_counts(latent_files: list[Path]) -> list[CacheBucketStatus]:
    counts: Counter[str] = Counter()
    pattern = re.compile(r"_(\d{4}x\d{4})_ltx2\.safetensors$")
    for path in latent_files:
        match = pattern.search(path.name)
        counts[match.group(1) if match else "unknown"] += 1
    return [CacheBucketStatus(bucket=bucket, count=count) for bucket, count in sorted(counts.items())]


def _stale_count(cache_files: list[Path], source_by_stem: dict[str, Path], suffix: str) -> int:
    stale = 0
    dim_re = re.compile(r"_(\d{4}x\d{4})$")
    for cache_file in cache_files:
        stem = cache_file.name[: -len(suffix)]
        stem = dim_re.sub("", stem)
        source = source_by_stem.get(stem)
        if source is None:
            source = next((path for key, path in source_by_stem.items() if stem.startswith(f"{key}_")), None)
        if source is None:
            continue
        try:
            if cache_file.stat().st_mtime < source.stat().st_mtime:
                stale += 1
        except OSError:
            continue
    return stale


def _scan_entry(entry: DatasetEntry, *, index: int, group: str, require_audio: bool) -> CacheDatasetStatus:
    cache_dir_text = _default_cache_dir(entry)
    row = CacheDatasetStatus(
        index=index, group=group, type=entry.type, directory=entry.directory or entry.jsonl_file, cache_directory=cache_dir_text
    )

    sources = _source_files(entry)
    source_stems = _stem_set(sources)
    source_keys = _source_key_set(sources)
    source_by_stem = {key: path for path in sources for key in (path.stem, path.name)}
    row.source_count = len(source_stems)

    if not row.directory:
        row.warnings.append("No source path configured")
    elif not sources:
        row.warnings.append("No source media found")

    cache_dir = Path(cache_dir_text) if cache_dir_text else None
    if cache_dir is None or not cache_dir_text:
        row.warnings.append("No cache directory configured")
        return row
    if not cache_dir.exists():
        row.warnings.append("Cache directory not found")
        row.missing_latent = row.source_count
        row.missing_text = row.source_count
        if require_audio:
            row.missing_audio = row.source_count
        return row

    latent_files = _cache_files(cache_dir, f"_{ARCH}.safetensors")
    text_files = _cache_files(cache_dir, f"_{ARCH}_te.safetensors")
    audio_files = _cache_files(cache_dir, f"_{ARCH}_audio.safetensors")
    dino_files = _cache_files(cache_dir, f"_{ARCH}_dino.safetensors")

    latent_stems = _cache_stems(latent_files, f"_{ARCH}.safetensors")
    text_stems = _cache_stems(text_files, f"_{ARCH}_te.safetensors")
    audio_stems = _cache_stems(audio_files, f"_{ARCH}_audio.safetensors")

    row.latent_count = _matched_source_count(latent_stems, source_keys)
    row.text_count = _matched_source_count(text_stems, source_keys)
    row.audio_count = _matched_source_count(audio_stems, source_keys)
    row.dino_count = len(dino_files)

    row.missing_latent = max(row.source_count - row.latent_count, 0)
    row.missing_text = max(row.source_count - row.text_count, 0)
    if require_audio:
        row.missing_audio = max(row.source_count - row.audio_count, 0)

    row.stale_latent = _stale_count(latent_files, source_by_stem, f"_{ARCH}.safetensors")
    row.stale_text = _stale_count(text_files, source_by_stem, f"_{ARCH}_te.safetensors")
    row.stale_audio = _stale_count(audio_files, source_by_stem, f"_{ARCH}_audio.safetensors")
    row.orphan_latent = sum(1 for stem in latent_stems if not _matches_source(stem, source_keys)) if source_keys else 0
    row.buckets = _bucket_counts(latent_files)
    return row


@router.get("/status", response_model=CacheStatusResponse)
async def cache_status(request: Request):
    config = _project_config(request)
    rows: list[CacheDatasetStatus] = []
    mode = str(config.caching.ltx2_mode or "video").lower()
    for index, entry in enumerate(config.dataset.datasets):
        rows.append(_scan_entry(entry, index=index, group="train", require_audio=entry.type == "audio" or mode == "av"))
    for index, entry in enumerate(config.dataset.validation_datasets):
        rows.append(_scan_entry(entry, index=index, group="validation", require_audio=entry.type == "audio" or mode == "av"))

    total_keys = (
        "source_count",
        "latent_count",
        "text_count",
        "audio_count",
        "dino_count",
        "missing_latent",
        "missing_text",
        "missing_audio",
        "stale_latent",
        "stale_text",
        "stale_audio",
        "orphan_latent",
    )
    totals = {key: sum(int(getattr(row, key)) for row in rows) for key in total_keys}
    return CacheStatusResponse(generated_at=datetime.now().isoformat(timespec="seconds"), rows=rows, totals=totals)
