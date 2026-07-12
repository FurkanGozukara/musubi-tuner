"""Public environment preparation API for portable ``torch.compile`` setup."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from .compiler_probe import probe_cuda_compiler
from .discovery import (
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from .host_compiler_probe import probe_host_cpp_compiler
from .msvc import (
    describe_msvc_environment,
    has_cl_exe,
    load_msvc_environment,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileToolchainStatus:
    """Summarize the runtime environment prepared for ``torch.compile``."""

    ok: bool
    detail: str
    changed: bool = False
    platform: str = ""
    cuda_root: str = ""
    compiler_path: str = ""
    ninja_path: str = ""
    cache_root: str = ""

    def as_dict(self) -> dict[str, str | bool]:
        """Return a JSON-serializable report suitable for diagnostics and APIs."""

        return {
            "ok": self.ok,
            "detail": self.detail,
            "changed": self.changed,
            "platform": self.platform,
            "cuda_root": self.cuda_root,
            "compiler_path": self.compiler_path,
            "ninja_path": self.ninja_path,
            "cache_root": self.cache_root,
        }


def ensure_compile_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
    require_openmp: bool = False,
) -> CompileToolchainStatus:
    """Prepare the current or supplied environment for PyTorch Inductor.

    Args:
        env: Environment mapping to update. Defaults to ``os.environ``.
        project_root: Optional project root used for stable compile cache folders.
        cache_dir: Optional explicit parent for Inductor and Triton cache folders.
        require_cuda_toolkit: Fail readiness when no CUDA Toolkit root is found.
            PyTorch distributions with bundled CUDA tools may leave this disabled.
        require_ninja: Fail readiness when Ninja is not found. Some Inductor paths
            do not need Ninja, while C++/CUDA extension builds generally do.
        require_openmp: Require the host C++ probe to compile and link an OpenMP
            program. This is recommended for PyTorch Inductor training workloads.

    Returns:
        Toolchain status with a human-readable detail string.
    """

    target_env = os.environ if env is None else env
    before = dict(target_env)
    cache_root = _ensure_compile_cache_dirs(target_env, project_root, cache_dir)
    cuda_status = discover_cuda_environment(target_env)
    if sys.platform != "win32":
        compiler_status = discover_posix_compiler(
            target_env,
            require_openmp=require_openmp,
        )
        ninja_status = discover_ninja(target_env)
        detail = _join_details(cuda_status.detail, compiler_status.detail, ninja_status.detail)
        return CompileToolchainStatus(
            _requirements_satisfied(
                compiler_status.ok,
                cuda_status.ok,
                ninja_status.ok,
                require_cuda_toolkit=require_cuda_toolkit,
                require_ninja=require_ninja,
            ),
            detail,
            target_env != before,
            platform=sys.platform,
            cuda_root=_status_value(cuda_status, "root"),
            compiler_path=_status_value(compiler_status, "path"),
            ninja_path=_status_value(ninja_status, "path"),
            cache_root=cache_root,
        )

    if has_cl_exe(target_env):
        host_probe = probe_host_cpp_compiler(
            target_env,
            platform_name=sys.platform,
            require_openmp=require_openmp,
        )
        probe = probe_cuda_compiler(target_env, platform_name=sys.platform)
        if host_probe.ok and probe.ok:
            ninja_status = discover_ninja(target_env)
            detail = _join_details(
                cuda_status.detail,
                describe_msvc_environment(target_env),
                host_probe.detail,
                probe.detail,
                ninja_status.detail,
            )
            return CompileToolchainStatus(
                _requirements_satisfied(
                    True,
                    cuda_status.ok,
                    ninja_status.ok,
                    require_cuda_toolkit=require_cuda_toolkit,
                    require_ninja=require_ninja,
                ),
                detail,
                target_env != before,
                platform=sys.platform,
                cuda_root=_status_value(cuda_status, "root"),
                compiler_path=_compiler_on_path(target_env),
                ninja_path=_status_value(ninja_status, "path"),
                cache_root=cache_root,
            )
        if not host_probe.ok:
            logger.warning(
                "Current MSVC environment is incomplete: %s. Activating a Visual Studio developer environment.",
                host_probe.detail,
            )
        elif not probe.ok:
            logger.warning("Current MSVC compiler rejected by CUDA: %s", probe.detail)

    status = load_msvc_environment(
        target_env,
        require_openmp=require_openmp,
    )
    ninja_status = discover_ninja(target_env)
    if status.ok:
        cuda_status = discover_cuda_environment(target_env)
        detail = _join_details(cuda_status.detail, status.detail, ninja_status.detail)
        return CompileToolchainStatus(
            _requirements_satisfied(
                True,
                cuda_status.ok,
                ninja_status.ok,
                require_cuda_toolkit=require_cuda_toolkit,
                require_ninja=require_ninja,
            ),
            detail,
            target_env != before,
            platform=sys.platform,
            cuda_root=_status_value(cuda_status, "root"),
            compiler_path=_compiler_on_path(target_env),
            ninja_path=_status_value(ninja_status, "path"),
            cache_root=cache_root,
        )
    detail = _join_details(cuda_status.detail, status.detail, ninja_status.detail)
    return CompileToolchainStatus(
        False,
        detail,
        target_env != before,
        platform=sys.platform,
        cuda_root=_status_value(cuda_status, "root"),
        compiler_path=_compiler_on_path(target_env),
        ninja_path=_status_value(ninja_status, "path"),
        cache_root=cache_root,
    )


def prepare_compile_subprocess_env(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    compile_requested: bool = True,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
    require_openmp: bool = False,
) -> dict[str, str]:
    """Return a subprocess environment prepared only when compile is requested."""

    child_env = dict(os.environ if env is None else env)
    if compile_requested:
        status = ensure_compile_environment(
            child_env,
            project_root=project_root,
            cache_dir=cache_dir,
            require_cuda_toolkit=require_cuda_toolkit,
            require_ninja=require_ninja,
            require_openmp=require_openmp,
        )
        if not status.ok:
            logger.warning("torch.compile toolchain preparation: %s", status.detail)
    return child_env


def compile_environment_report(
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
    require_openmp: bool = False,
) -> CompileToolchainStatus:
    """Return the current process toolchain status for diagnostics."""

    return ensure_compile_environment(
        os.environ,
        project_root=project_root,
        cache_dir=cache_dir,
        require_cuda_toolkit=require_cuda_toolkit,
        require_ninja=require_ninja,
        require_openmp=require_openmp,
    )


def _ensure_compile_cache_dirs(
    env: MutableMapping[str, str],
    project_root: str | Path | None,
    cache_dir: str | Path | None,
) -> str:
    """Set stable local cache folders used by Inductor and Triton."""

    configured_inductor = str(env.get("TORCHINDUCTOR_CACHE_DIR") or "").strip()
    configured_triton = str(env.get("TRITON_CACHE_DIR") or "").strip()
    if configured_inductor and configured_triton:
        return str(Path(configured_inductor).expanduser().parent)

    root_text = str(project_root or env.get("TORCH_COMPILE_PROJECT_ROOT") or "").strip()
    root = Path(root_text).expanduser().resolve() if root_text else Path.cwd()
    preferred = Path(cache_dir).expanduser().resolve() if cache_dir is not None else root / ".cache" / "torch_compile_toolchain"
    fallback = Path(tempfile.gettempdir()) / "torch_compile_toolchain"
    for cache_root in _unique_cache_roots(preferred, fallback):
        inductor = Path(configured_inductor).expanduser() if configured_inductor else cache_root / "inductor"
        triton = Path(configured_triton).expanduser() if configured_triton else cache_root / "triton"
        try:
            inductor.mkdir(parents=True, exist_ok=True)
            triton.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("Unable to create torch.compile cache at %s: %s", cache_root, exc)
            continue
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor))
        env.setdefault("TRITON_CACHE_DIR", str(triton))
        return str(cache_root)
    return ""


def _requirements_satisfied(
    compiler_ok: bool,
    cuda_ok: bool,
    ninja_ok: bool,
    *,
    require_cuda_toolkit: bool,
    require_ninja: bool,
) -> bool:
    return compiler_ok and (cuda_ok or not require_cuda_toolkit) and (ninja_ok or not require_ninja)


def _compiler_on_path(env: MutableMapping[str, str]) -> str:
    return shutil.which("cl.exe", path=env.get("PATH")) or ""


def _status_value(status: object, name: str) -> str:
    return str(getattr(status, name, "") or "")


def _unique_cache_roots(*roots: Path) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = os.path.normcase(os.path.normpath(str(root)))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _join_details(*details: str) -> str:
    """Join diagnostic detail fragments without empty entries."""

    return "; ".join(detail for detail in details if detail)
