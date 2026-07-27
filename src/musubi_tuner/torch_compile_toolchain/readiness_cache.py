"""Fingerprint and reuse successful native toolchain verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from .cuda_toolkit import torch_cuda_version


_CACHE_SCHEMA = 1
_CACHE_FILENAME = "toolchain_probe_v1.json"
_DEFAULT_TTL_HOURS = 24 * 30
_FALSE_VALUES = {"", "0", "false", "no", "none", "off"}

_VERIFIED_KEY = "MUSUBI_TORCH_COMPILE_VERIFIED"
_TOKEN_KEY = "MUSUBI_TORCH_COMPILE_VERIFICATION_TOKEN"
_COMPILER_KEY = "MUSUBI_TORCH_COMPILE_COMPILER_PATH"
_CUDA_ROOT_KEY = "MUSUBI_TORCH_COMPILE_CUDA_ROOT"
_NINJA_KEY = "MUSUBI_TORCH_COMPILE_NINJA_PATH"

# Only compiler-related values are persisted. In particular, never serialize the
# complete process environment because it may contain credentials.
_TOOLCHAIN_ENV_KEYS = (
    "PATH",
    "CC",
    "CXX",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDACXX",
    "CUDAHOSTCXX",
    "NVCC",
    "NVCC_CCBIN",
    "NINJA",
    "CMAKE_MAKE_PROGRAM",
    "DevEnvDir",
    "ExtensionSdkDir",
    "FrameworkDir",
    "FrameworkDir32",
    "FrameworkVersion",
    "FrameworkVersion32",
    "FrameworkVersion64",
    "NETFXSDKDir",
    "UCRTVersion",
    "UniversalCRTSdkDir",
    "VCIDEInstallDir",
    "VCINSTALLDIR",
    "VCToolsInstallDir",
    "VCToolsRedistDir",
    "VCToolsVersion",
    "VisualStudioVersion",
    "VSINSTALLDIR",
    "VSCMD_ARG_HOST_ARCH",
    "VSCMD_ARG_TGT_ARCH",
    "VSCMD_VER",
    "WindowsLibPath",
    "WindowsSdkBinPath",
    "WindowsSdkDir",
    "WindowsSDKLibVersion",
    "WindowsSDKVersion",
)

_BASE_ENV_KEYS = (
    *_TOOLCHAIN_ENV_KEYS,
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "MUSUBI_VS_INSTALLDIR",
    "MUSUBI_VS_DEV_CMD",
    "MUSUBI_VSWHERE",
)


@dataclass(frozen=True)
class VerifiedToolchain:
    """Minimal status needed to reconstruct ``CompileToolchainStatus``."""

    detail: str
    platform: str
    cuda_root: str
    compiler_path: str
    ninja_path: str


def probe_cache_enabled(env: Mapping[str, str]) -> bool:
    """Return whether successful probe results may be read and persisted."""

    value = _env_value(env, "MUSUBI_TORCH_COMPILE_CACHE_PROBE")
    return not value or value.strip().casefold() not in _FALSE_VALUES


def force_probe_requested(env: Mapping[str, str]) -> bool:
    """Return whether the caller explicitly requested fresh native probes."""

    value = _env_value(env, "MUSUBI_TORCH_COMPILE_FORCE_PROBE")
    return bool(value) and value.strip().casefold() not in _FALSE_VALUES


def restore_inherited_verification(
    env: Mapping[str, str],
    requirements: Mapping[str, bool],
) -> VerifiedToolchain | None:
    """Trust a verified parent only while its exact environment still matches."""

    if _env_value(env, _VERIFIED_KEY) != "1":
        return None
    status = _status_from_environment(env)
    expected = _env_value(env, _TOKEN_KEY)
    actual = _verification_token(env, status, requirements)
    if not expected or not actual or not hmac.compare_digest(expected, actual):
        return None
    return status


def restore_persisted_verification(
    base_env: Mapping[str, str],
    env: MutableMapping[str, str],
    cache_root: str | Path,
    requirements: Mapping[str, bool],
) -> VerifiedToolchain | None:
    """Restore a successful probe when its environment and files are unchanged."""

    if not str(cache_root):
        return None
    cache_path = _cache_path(cache_root)
    try:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("schema") != _CACHE_SCHEMA:
        return None
    if record.get("platform") != sys.platform or record.get("machine") != platform.machine():
        return None
    if record.get("requirements") != dict(requirements):
        return None
    if record.get("base_environment") != _environment_signature(base_env, _BASE_ENV_KEYS):
        return None
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return None
    if expires_at > 0 and time.time() >= expires_at:
        return None

    updates = record.get("environment_updates")
    raw_status = record.get("status")
    expected = record.get("verification_token")
    if not isinstance(updates, dict) or not isinstance(raw_status, dict) or not isinstance(expected, str):
        return None
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in updates.items()):
        return None
    try:
        status = VerifiedToolchain(
            detail=str(raw_status["detail"]),
            platform=str(raw_status["platform"]),
            cuda_root=str(raw_status.get("cuda_root", "")),
            compiler_path=str(raw_status["compiler_path"]),
            ninja_path=str(raw_status.get("ninja_path", "")),
        )
    except KeyError:
        return None

    candidate = dict(env)
    _apply_environment_updates(candidate, updates)
    actual = _verification_token(candidate, status, requirements)
    if not actual or not hmac.compare_digest(expected, actual):
        return None

    _apply_environment_updates(env, updates)
    mark_inherited_verification(env, status, requirements)
    return status


def record_successful_verification(
    base_env: Mapping[str, str],
    env: MutableMapping[str, str],
    cache_root: str | Path,
    status: VerifiedToolchain,
    requirements: Mapping[str, bool],
    *,
    persist: bool,
) -> None:
    """Mark a prepared child environment and atomically persist a safe subset."""

    token = mark_inherited_verification(env, status, requirements)
    if not persist or not token or not str(cache_root):
        return

    updates = _environment_updates(base_env, env)
    now = time.time()
    ttl_hours = _cache_ttl_hours(env)
    record = {
        "schema": _CACHE_SCHEMA,
        "created_at": now,
        "expires_at": now + ttl_hours * 3600 if ttl_hours > 0 else 0,
        "platform": sys.platform,
        "machine": platform.machine(),
        "requirements": dict(requirements),
        "base_environment": _environment_signature(base_env, _BASE_ENV_KEYS),
        "environment_updates": updates,
        "status": {
            "detail": status.detail,
            "platform": status.platform,
            "cuda_root": status.cuda_root,
            "compiler_path": status.compiler_path,
            "ninja_path": status.ninja_path,
        },
        "verification_token": token,
    }
    cache_path = _cache_path(cache_root)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def mark_inherited_verification(
    env: MutableMapping[str, str],
    status: VerifiedToolchain,
    requirements: Mapping[str, bool],
) -> str:
    """Stamp the exact prepared environment for child-process reuse."""

    token = _verification_token(env, status, requirements)
    if not token:
        return ""
    _set_env_value(env, _VERIFIED_KEY, "1")
    _set_env_value(env, _TOKEN_KEY, token)
    _set_env_value(env, _COMPILER_KEY, status.compiler_path)
    _set_env_value(env, _CUDA_ROOT_KEY, status.cuda_root)
    _set_env_value(env, _NINJA_KEY, status.ninja_path)
    return token


def clear_verification_markers(env: MutableMapping[str, str]) -> None:
    """Remove stale inherited markers before performing a fresh probe."""

    for key in (_VERIFIED_KEY, _TOKEN_KEY, _COMPILER_KEY, _CUDA_ROOT_KEY, _NINJA_KEY):
        _delete_env_value(env, key)


def _status_from_environment(env: Mapping[str, str]) -> VerifiedToolchain:
    return VerifiedToolchain(
        detail=_env_value(env, "MUSUBI_TORCH_COMPILE_DETAIL") or "toolchain verified by parent process",
        platform=sys.platform,
        cuda_root=_env_value(env, _CUDA_ROOT_KEY),
        compiler_path=_env_value(env, _COMPILER_KEY),
        ninja_path=_env_value(env, _NINJA_KEY),
    )


def _verification_token(
    env: Mapping[str, str],
    status: VerifiedToolchain,
    requirements: Mapping[str, bool],
) -> str:
    compiler = _path_snapshot(status.compiler_path, require_file=True)
    if compiler is None:
        return ""
    files: dict[str, object] = {"compiler": compiler}
    for label, value, require_file in (
        ("cuda_root", status.cuda_root, False),
        ("ninja", status.ninja_path, True),
        ("nvcc", _env_value(env, "CUDACXX") or _env_value(env, "NVCC"), True),
    ):
        if not value:
            continue
        snapshot = _path_snapshot(value, require_file=require_file)
        if snapshot is None:
            return ""
        files[label] = snapshot

    search_directories: dict[str, object] = {}
    for key in ("INCLUDE", "LIB", "LIBPATH"):
        for index, value in enumerate(_split_search_path(_env_value(env, key))):
            snapshot = _path_snapshot(value, require_file=False)
            if snapshot is not None:
                search_directories[f"{key}:{index}"] = snapshot

    payload = {
        "schema": _CACHE_SCHEMA,
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": list(sys.version_info[:3]),
        },
        "torch": _torch_version(),
        "torch_cuda": torch_cuda_version(),
        "requirements": dict(requirements),
        "status": {
            "platform": status.platform,
            "cuda_root": _normalize_path(status.cuda_root),
            "compiler_path": _normalize_path(status.compiler_path),
            "ninja_path": _normalize_path(status.ninja_path),
        },
        "environment": {key: _env_value(env, key) for key in _TOOLCHAIN_ENV_KEYS if _env_value(env, key)},
        "files": files,
        "search_directories": search_directories,
    }
    return _digest(payload)


def _path_snapshot(value: str, *, require_file: bool) -> dict[str, object] | None:
    if not value:
        return None
    path = Path(str(value).strip().strip('"')).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    if require_file and not resolved.is_file():
        return None
    if not require_file and not resolved.exists():
        return None
    return {
        "path": os.path.normcase(os.path.normpath(str(resolved))),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "directory": resolved.is_dir(),
    }


def _environment_updates(
    base_env: Mapping[str, str],
    prepared_env: Mapping[str, str],
) -> dict[str, str]:
    updates: dict[str, str] = {}
    for key in _TOOLCHAIN_ENV_KEYS:
        value = _env_value(prepared_env, key)
        if value and value != _env_value(base_env, key):
            updates[key] = value
    return updates


def _apply_environment_updates(
    env: MutableMapping[str, str],
    updates: Mapping[str, str],
) -> None:
    for key, value in updates.items():
        if key in _TOOLCHAIN_ENV_KEYS:
            _set_env_value(env, key, value)


def _environment_signature(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    values = {key: _env_value(env, key) for key in keys if _env_value(env, key)}
    return _digest(values)


def _cache_ttl_hours(env: Mapping[str, str]) -> float:
    value = _env_value(env, "MUSUBI_TORCH_COMPILE_PROBE_CACHE_TTL_HOURS")
    if not value:
        return float(_DEFAULT_TTL_HOURS)
    try:
        return max(0.0, float(value))
    except ValueError:
        return float(_DEFAULT_TTL_HOURS)


def _torch_version() -> str:
    try:
        import torch
    except (ImportError, OSError):
        return ""
    return str(getattr(torch, "__version__", "") or "")


def _cache_path(cache_root: str | Path) -> Path:
    return Path(cache_root).expanduser().resolve() / _CACHE_FILENAME


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    path = Path(str(value).strip().strip('"')).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return os.path.normcase(os.path.normpath(str(path)))


def _split_search_path(value: str) -> list[str]:
    return [part.strip().strip('"') for part in value.split(os.pathsep) if part.strip().strip('"')]


def _env_value(env: Mapping[str, str], key: str) -> str:
    exact = env.get(key)
    if exact is not None:
        return str(exact)
    folded = key.casefold()
    for candidate, value in env.items():
        if candidate.casefold() == folded:
            return str(value)
    return ""


def _set_env_value(env: MutableMapping[str, str], key: str, value: str) -> None:
    _delete_env_value(env, key)
    env[key] = str(value)


def _delete_env_value(env: MutableMapping[str, str], key: str) -> None:
    folded = key.casefold()
    for candidate in list(env):
        if candidate.casefold() == folded:
            del env[candidate]
