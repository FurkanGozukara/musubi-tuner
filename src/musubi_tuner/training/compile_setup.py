"""Shared torch.compile environment setup for training entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from musubi_tuner.torch_compile_toolchain import (
    CompileToolchainStatus,
    ensure_compile_environment,
)


logger = logging.getLogger(__name__)

_DISABLED_BACKENDS = {"", "0", "false", "no", "none", "off"}
_NATIVE_CODEGEN_BACKENDS = {"inductor"}
_ready_status: CompileToolchainStatus | None = None


class TorchCompileToolchainError(RuntimeError):
    """Raised before model loading when a requested compiler toolchain is unusable."""


def compile_requested(args: Any | None = None) -> bool:
    """Detect direct compile and Accelerate Dynamo requests."""

    direct = getattr(args, "compile", False) if args is not None else False
    if isinstance(direct, str):
        direct_enabled = direct.strip().casefold() not in _DISABLED_BACKENDS
    else:
        direct_enabled = bool(direct)

    backends = [
        getattr(args, "dynamo_backend", "") if args is not None else "",
        os.environ.get("ACCELERATE_DYNAMO_BACKEND", ""),
    ]
    dynamo_enabled = any(str(value or "").strip().casefold() not in _DISABLED_BACKENDS for value in backends)
    return direct_enabled or dynamo_enabled


def native_compile_toolchain_requested(args: Any | None = None) -> bool:
    """Return whether the selected compile backend emits native host code."""

    if args is not None and _flag_enabled(getattr(args, "compile", False)):
        backend = str(getattr(args, "compile_backend", "inductor") or "inductor").strip().casefold()
        if backend in _NATIVE_CODEGEN_BACKENDS:
            return True

    backends = [
        getattr(args, "dynamo_backend", "") if args is not None else "",
        os.environ.get("ACCELERATE_DYNAMO_BACKEND", ""),
    ]
    return any(str(value or "").strip().casefold() in _NATIVE_CODEGEN_BACKENDS for value in backends)


def ensure_training_compile_environment(
    *,
    required: bool = True,
    force: bool = False,
) -> CompileToolchainStatus:
    """Prepare the current process before the first compilation attempt."""

    global _ready_status
    if _ready_status is not None and not force:
        return _ready_status

    project_root = Path(__file__).resolve().parents[3]
    status = ensure_compile_environment(
        os.environ,
        project_root=project_root,
        cache_dir=project_root / ".cache" / "torch_compile",
        require_cuda_toolkit=_env_flag("MUSUBI_TORCH_COMPILE_REQUIRE_CUDA", False),
        require_ninja=_env_flag("MUSUBI_TORCH_COMPILE_REQUIRE_NINJA", False),
        require_openmp=_env_flag("MUSUBI_TORCH_COMPILE_REQUIRE_OPENMP", True),
    )
    os.environ["MUSUBI_TORCH_COMPILE_READY"] = "1" if status.ok else "0"
    os.environ["MUSUBI_TORCH_COMPILE_DETAIL"] = status.detail
    if status.ok:
        _ready_status = status
        logger.info("torch.compile toolchain ready: %s", status.detail)
        return status

    message = f"torch.compile was requested but its native toolchain is unavailable: {status.detail}"
    if required:
        raise TorchCompileToolchainError(message)
    logger.warning(message)
    return status


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _flag_enabled(value)


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() not in _DISABLED_BACKENDS
    return bool(value)
