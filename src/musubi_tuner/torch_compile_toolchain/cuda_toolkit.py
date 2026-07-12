"""Portable CUDA Toolkit selection helpers for optional ``torch.compile``."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterable, MutableMapping


_CUDA_PATH_VERSION_PATTERN = re.compile(
    r"^(?:cuda[-_]?|v)?(\d{1,2})(?:\.(\d{1,2}))?$",
    re.IGNORECASE,
)
_CUDA_VERSION_PATTERN = re.compile(r"(?<!\d)(\d{1,2})(?:\.(\d{1,2}))?")


def select_cuda_toolkit_root(env: MutableMapping[str, str], platform_name: str) -> Path | None:
    """Return a CUDA Toolkit root, preferring the installed torch CUDA version.

    Args:
        env: Environment mapping used for CUDA and PATH discovery.
        platform_name: Current platform name, usually ``sys.platform``.

    Returns:
        The selected CUDA Toolkit root, or ``None`` when no candidate exists.
    """

    candidates = [
        path.resolve() for path in _cuda_root_candidates(env, platform_name) if path and _looks_like_cuda_root(path, platform_name)
    ]
    if not candidates:
        return None

    torch_version = torch_cuda_version()
    if torch_version:
        matching = _best_version_match(candidates, torch_version)
        if matching is not None:
            return matching
    return candidates[0]


def torch_cuda_version() -> str:
    """Return the CUDA version reported by the installed torch package."""

    try:
        import torch
    except (ImportError, OSError):
        return ""

    version_module = getattr(torch, "version", None)
    cuda_version = getattr(version_module, "cuda", None)
    return str(cuda_version).strip() if cuda_version else ""


def _cuda_root_candidates(env: MutableMapping[str, str], platform_name: str) -> list[Path]:
    candidates: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CUDA_ROOT"):
        value = env.get(key)
        if value:
            candidates.append(_path_from_environment(value))

    versioned_keys = re.compile(r"^CUDA_(?:PATH|HOME)(?:_V?\d+(?:_\d+)?)$", re.IGNORECASE)
    for key, value in env.items():
        if value and versioned_keys.fullmatch(key):
            candidates.append(_path_from_environment(value))

    for key in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CUDA_ROOT"):
        if env.get(key):
            continue
        value = os.environ.get(key)
        if value:
            candidates.append(_path_from_environment(value))
    for key, value in os.environ.items():
        if key not in env and value and versioned_keys.fullmatch(key):
            candidates.append(_path_from_environment(value))

    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_root = _path_from_environment(conda_prefix)
        candidates.append(conda_root)
        candidates.extend(conda_root / "targets" / target for target in ("x86_64-linux", "sbsa-linux", "aarch64-linux"))

    if platform_name == "win32":
        program_files = [
            env.get("ProgramW6432"),
            env.get("ProgramFiles"),
            os.environ.get("ProgramW6432"),
            os.environ.get("ProgramFiles"),
            "C:/Program Files",
        ]
        for value in (item for item in program_files if item):
            default_root = _path_from_environment(value) / ("NVIDIA GPU Computing Toolkit/CUDA")
            if default_root.exists():
                candidates.extend(_sort_cuda_paths(default_root.glob("v*")))
    else:
        candidates.extend(
            [
                Path("/usr/local/cuda"),
                Path("/opt/cuda"),
                Path("/usr/lib/cuda"),
                Path("/usr"),
            ]
        )
        candidates.extend(_sort_cuda_paths(Path("/usr/local").glob("cuda-*")))
        candidates.extend(_sort_cuda_paths(Path("/opt").glob("cuda-*")))

    nvcc_root = _nvcc_root(env)
    if nvcc_root is not None:
        candidates.append(nvcc_root)

    return _unique_paths(candidates)


def _looks_like_cuda_root(path: Path, platform_name: str) -> bool:
    if (path / "bin" / _exe_name("nvcc", platform_name)).is_file() or (path / "include" / "cuda.h").is_file():
        return True
    library_patterns = (
        "lib64/libcudart.so*",
        "lib/libcudart.so*",
        "lib/x64/cudart.lib",
        "lib/x64/cuda.lib",
    )
    return any(any(path.glob(pattern)) for pattern in library_patterns)


def _best_version_match(candidates: Iterable[Path], torch_version: str) -> Path | None:
    target = _parse_cuda_version(torch_version)
    if target is None:
        return None

    best_rank = (0, 0, 0)
    best_path: Path | None = None
    for path in candidates:
        version = _cuda_root_version(path)
        if version is None:
            continue
        rank = _version_match_rank(version, target)
        if rank > best_rank:
            best_rank = rank
            best_path = path
    return best_path


def _cuda_root_version(path: Path) -> tuple[int, int | None] | None:
    version = _parse_cuda_path_version(path.name)
    if version is not None:
        return version

    for metadata in (path / "version.txt", path / "version.json"):
        if not metadata.is_file():
            continue
        version = _parse_cuda_version(metadata.read_text(encoding="utf-8", errors="ignore"))
        if version is not None:
            return version
    return None


def _parse_cuda_path_version(value: str) -> tuple[int, int | None] | None:
    match = _CUDA_PATH_VERSION_PATTERN.search(value)
    if not match:
        return None
    minor = match.group(2)
    return int(match.group(1)), int(minor) if minor is not None else None


def _parse_cuda_version(value: str) -> tuple[int, int | None] | None:
    match = _CUDA_VERSION_PATTERN.search(value)
    if not match:
        return None
    minor = match.group(2)
    return int(match.group(1)), int(minor) if minor is not None else None


def _version_match_rank(
    candidate: tuple[int, int | None],
    target: tuple[int, int | None],
) -> tuple[int, int, int]:
    candidate_major, candidate_minor = candidate
    target_major, target_minor = target
    if candidate_major != target_major:
        return (0, 0, 0)
    candidate_minor_value = candidate_minor if candidate_minor is not None else -1
    if target_minor is None:
        return (1, 0, candidate_minor_value)
    if candidate_minor is None:
        return (1, -1000, -1)
    distance = abs(candidate_minor - target_minor)
    return (2 if distance == 0 else 1, -distance, candidate_minor)


def _nvcc_root(env: MutableMapping[str, str]) -> Path | None:
    for key in ("CUDACXX", "NVCC"):
        configured = str(env.get(key, "") or "").strip().strip('"')
        if not configured:
            continue
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            parents = candidate.resolve().parents
            return parents[1] if len(parents) > 1 else None
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if not nvcc:
        return None
    parents = Path(nvcc).resolve().parents
    return parents[1] if len(parents) > 1 else None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(os.path.normpath(str(path)))
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def _sort_cuda_paths(paths: Iterable[Path]) -> list[Path]:
    """Sort versioned CUDA folders numerically, newest first."""

    def sort_key(path: Path) -> tuple[int, int]:
        version = _cuda_root_version(path)
        if version is None:
            return (0, 0)
        major, minor = version
        return major, minor if minor is not None else 0

    return sorted(paths, key=sort_key, reverse=True)


def _path_from_environment(value: str) -> Path:
    return Path(str(value).strip().strip('"')).expanduser()


def _exe_name(name: str, platform_name: str) -> str:
    return f"{name}.exe" if platform_name == "win32" else name
