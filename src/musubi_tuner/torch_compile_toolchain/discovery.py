"""Portable cross-platform discovery for optional ``torch.compile``."""

from __future__ import annotations

import importlib
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableMapping

from .compiler_probe import probe_cuda_compiler
from .cuda_toolkit import select_cuda_toolkit_root
from .host_compiler_probe import probe_host_cpp_compiler
from .msvc import visual_studio_install_roots


@dataclass(frozen=True)
class ExecutableStatus:
    """Describe one discovered compiler or toolkit executable."""

    ok: bool
    detail: str
    path: str = ""


@dataclass(frozen=True)
class CudaDiscoveryStatus:
    """Describe CUDA toolkit discovery and environment changes."""

    ok: bool
    detail: str
    root: str = ""
    changed: bool = False


def discover_cuda_environment(env: MutableMapping[str, str]) -> CudaDiscoveryStatus:
    """Discover CUDA Toolkit folders, update ``env``, and return status."""

    before_path = env.get("PATH", "")
    before_cuda_home = env.get("CUDA_HOME")
    before_cuda_path = env.get("CUDA_PATH")
    before_cudacxx = env.get("CUDACXX")
    root = select_cuda_toolkit_root(env, sys.platform)
    if root is None:
        return CudaDiscoveryStatus(False, "CUDA Toolkit not found")

    env["CUDA_HOME"] = str(root)
    env["CUDA_PATH"] = str(root)
    _prepend_existing_paths(env, _cuda_path_additions(root), prioritize=True)
    changed = (
        env.get("PATH", "") != before_path or env.get("CUDA_HOME") != before_cuda_home or env.get("CUDA_PATH") != before_cuda_path
    )
    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path:
        env["CUDACXX"] = nvcc_path
        changed = changed or env.get("CUDACXX") != before_cudacxx
        return CudaDiscoveryStatus(True, f"CUDA Toolkit found at {root}", str(root), changed)
    return CudaDiscoveryStatus(
        True,
        f"CUDA root found at {root}, nvcc not present",
        str(root),
        changed,
    )


def discover_posix_compiler(
    env: MutableMapping[str, str],
    *,
    require_openmp: bool = False,
) -> ExecutableStatus:
    """Find a Linux/macOS C++ compiler, update ``CC``/``CXX``, and return status."""

    if sys.platform == "win32":
        return ExecutableStatus(True, "Windows compiler discovery handled by MSVC")

    failures: list[str] = []
    attempted: set[str] = set()
    for env_key in ("NVCC_CCBIN", "CUDAHOSTCXX", "CXX"):
        configured_cxx = env.get(env_key)
        cxx_path = _resolve_compiler_executable(configured_cxx, env) if configured_cxx else ""
        if not cxx_path:
            if configured_cxx:
                failures.append(f"{env_key}={configured_cxx}: executable not found")
            continue
        attempted.add(os.path.normcase(cxx_path))
        cc_path = _resolve_executable(env.get("CC", ""), env)
        if not cc_path:
            cc_path = _paired_c_compiler(cxx_path, env)
        candidate_env = dict(env)
        candidate_env["CXX"] = cxx_path
        if cc_path:
            candidate_env["CC"] = cc_path
        host_probe = probe_host_cpp_compiler(
            candidate_env,
            platform_name=sys.platform,
            compiler_path=cxx_path,
            require_openmp=require_openmp,
        )
        if not host_probe.ok:
            failures.append(f"{env_key}={cxx_path}: {host_probe.detail}")
            continue
        probe = probe_cuda_compiler(
            candidate_env,
            platform_name=sys.platform,
            compiler_path=cxx_path,
        )
        if probe.ok:
            _set_posix_compiler_environment(env, cxx_path, cc_path)
            return ExecutableStatus(
                True,
                f"{env_key} compiler found: {cxx_path}; {host_probe.detail}; {probe.detail}",
                cxx_path,
            )
        failures.append(f"{env_key}={cxx_path}: {probe.detail}")

    _prepend_existing_paths(env, _posix_compiler_path_candidates(env))
    for cxx, cc in _posix_compiler_candidates(env):
        cxx_path = shutil.which(cxx, path=env.get("PATH"))
        if not cxx_path:
            continue
        normalized = os.path.normcase(cxx_path)
        if normalized in attempted:
            continue
        attempted.add(normalized)
        cc_path = shutil.which(cc, path=env.get("PATH")) if cc else ""
        candidate_env = dict(env)
        candidate_env["CXX"] = cxx_path
        if cc_path:
            candidate_env["CC"] = cc_path
        host_probe = probe_host_cpp_compiler(
            candidate_env,
            platform_name=sys.platform,
            compiler_path=cxx_path,
            require_openmp=require_openmp,
        )
        if not host_probe.ok:
            failures.append(f"{cxx_path}: {host_probe.detail}")
            continue
        probe = probe_cuda_compiler(
            candidate_env,
            platform_name=sys.platform,
            compiler_path=cxx_path,
        )
        if probe.ok:
            _set_posix_compiler_environment(env, cxx_path, cc_path)
            return ExecutableStatus(
                True,
                f"C++ compiler found: {cxx_path}; {host_probe.detail}; {probe.detail}",
                cxx_path,
            )
        failures.append(f"{cxx_path}: {probe.detail}")

    detail = "No compatible C++ compiler found. Install gcc/g++ or clang/clang++."
    if failures:
        detail += f" Last probe: {failures[-1]}"
    return ExecutableStatus(False, detail)


def discover_ninja(env: MutableMapping[str, str]) -> ExecutableStatus:
    """Find Ninja from PATH, environment, Python, Conda, or Visual Studio."""

    for env_key in ("NINJA", "CMAKE_MAKE_PROGRAM"):
        configured = env.get(env_key)
        ninja = _resolve_executable(configured, env) if configured else ""
        if ninja:
            return _record_ninja(env, ninja, f"Ninja from {env_key}")

    for name in ("ninja", "ninja-build"):
        ninja = shutil.which(name, path=env.get("PATH"))
        if ninja:
            return _record_ninja(env, ninja, "ninja found")

    for candidate in _ninja_file_candidates(env):
        if _is_executable_file(candidate):
            return _record_ninja(env, str(candidate), "ninja found")
    return ExecutableStatus(False, "ninja not found")


def _cuda_path_additions(root: Path) -> list[Path]:
    additions = [root / "bin"]
    if sys.platform == "win32":
        additions.append(root / "libnvvp")
    return additions


def _posix_compiler_path_candidates(env: MutableMapping[str, str]) -> list[Path]:
    candidates = [
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
        Path.home() / ".local" / "bin",
    ]
    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.insert(0, Path(conda_prefix) / "bin")
    virtual_env = env.get("VIRTUAL_ENV") or os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.insert(0, Path(virtual_env) / "bin")
    for pattern in (
        "/opt/rh/gcc-toolset-*/root/usr/bin",
        "/opt/rh/devtoolset-*/root/usr/bin",
        "/usr/local/gcc-*/bin",
        "/opt/gcc-*/bin",
        "/usr/lib/llvm-*/bin",
    ):
        candidates.extend(sorted(Path("/").glob(pattern.lstrip("/")), reverse=True))
    return _unique_paths(candidates)


def _posix_compiler_candidates(
    env: MutableMapping[str, str],
) -> Iterable[tuple[str, str]]:
    discovered = _versioned_compilers_on_path(env)
    versioned_gcc = [(f"g++-{version}", f"gcc-{version}") for version in range(30, 3, -1)]
    versioned_clang = [(f"clang++-{version}", f"clang-{version}") for version in range(30, 3, -1)]
    candidates = (
        ("g++", "gcc"),
        *discovered,
        *versioned_gcc,
        ("clang++", "clang"),
        *versioned_clang,
        ("nvc++", "nvc"),
        ("icpx", "icx"),
        ("armclang++", "armclang"),
        ("c++", "cc"),
    )
    return _unique_compiler_pairs(candidates)


def _prepend_existing_paths(
    env: MutableMapping[str, str],
    paths: Iterable[Path],
    *,
    prioritize: bool = False,
) -> None:
    existing = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    preferred: list[str] = []
    preferred_keys: set[str] = set()
    for path in paths:
        if not path.is_dir():
            continue
        text = str(path)
        key = _normalized_path_key(path)
        if key in preferred_keys:
            continue
        preferred.append(text)
        preferred_keys.add(key)
    if not preferred:
        return
    if not prioritize:
        existing_keys = {_normalized_path_key(Path(part)) for part in existing}
        additions = [part for part in preferred if _normalized_path_key(Path(part)) not in existing_keys]
        if additions:
            env["PATH"] = os.pathsep.join(additions + existing)
        return
    remaining = [part for part in existing if _normalized_path_key(Path(part)) not in preferred_keys]
    env["PATH"] = os.pathsep.join(preferred + remaining)


def _resolve_executable(executable: str, env: MutableMapping[str, str]) -> str:
    text = str(executable or "").strip().strip('"')
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(text, path=env.get("PATH")) or ""


def _resolve_compiler_executable(
    command: str,
    env: MutableMapping[str, str],
) -> str:
    configured_path = Path(str(command or "").strip().strip('"')).expanduser()
    if configured_path.is_dir():
        for name in ("g++", "clang++", "nvc++", "icpx", "armclang++", "c++"):
            candidate = configured_path / name
            if candidate.is_file():
                return str(candidate.resolve())
    resolved = _resolve_executable(command, env)
    if resolved:
        return resolved
    try:
        tokens = shlex.split(str(command or ""), posix=True)
    except ValueError:
        return ""
    for token in reversed(tokens):
        if not token or token.startswith("-") or "=" in token:
            continue
        resolved = _resolve_executable(token, env)
        if resolved:
            return resolved
    return ""


def _set_posix_compiler_environment(
    env: MutableMapping[str, str],
    cxx_path: str,
    cc_path: str,
) -> None:
    """Expose one probed host compiler to PyTorch, CMake, and modern nvcc."""

    env["CXX"] = cxx_path
    env["CUDAHOSTCXX"] = cxx_path
    env["NVCC_CCBIN"] = cxx_path
    if cc_path:
        env["CC"] = cc_path


def _paired_c_compiler(
    cxx_path: str,
    env: MutableMapping[str, str],
) -> str:
    path = Path(cxx_path)
    name = path.name
    suffix = ".exe" if name.casefold().endswith(".exe") else ""
    stem = name[: -len(suffix)] if suffix else name
    if "g++" in stem:
        cc_name = stem.replace("g++", "gcc", 1) + suffix
    elif "clang++" in stem:
        cc_name = stem.replace("clang++", "clang", 1) + suffix
    elif stem == "c++":
        cc_name = "cc" + suffix
    elif stem == "nvc++":
        cc_name = "nvc" + suffix
    elif stem == "icpx":
        cc_name = "icx" + suffix
    elif stem == "armclang++":
        cc_name = "armclang" + suffix
    else:
        return ""
    sibling = path.with_name(cc_name)
    if sibling.is_file():
        return str(sibling.resolve())
    return shutil.which(cc_name, path=env.get("PATH")) or ""


def _versioned_compilers_on_path(
    env: MutableMapping[str, str],
) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"^(?P<prefix>.*?)(?P<family>clang\+\+|g\+\+|c\+\+)"
        r"(?:-(?P<version>\d+))?(?:\.exe)?$",
        re.IGNORECASE,
    )
    found: list[tuple[int, str, str]] = []
    directories = [Path(part) for part in env.get("PATH", "").split(os.pathsep) if part and Path(part).is_dir()]
    for directory in directories:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            match = pattern.fullmatch(entry.name)
            if not match:
                continue
            prefix = match.group("prefix") or ""
            version_text = match.group("version") or ""
            if not prefix and not version_text:
                continue
            version = int(version_text) if version_text else 0
            cxx = entry.name[:-4] if entry.name.casefold().endswith(".exe") else entry.name
            family = match.group("family")
            replacement = {
                "g++": "gcc",
                "clang++": "clang",
                "c++": "cc",
            }[family.casefold()]
            start, end = match.span("family")
            cc = cxx[:start] + replacement + cxx[end:]
            found.append((version, cxx, cc))
    found.sort(key=lambda item: item[0], reverse=True)
    return [(cxx, cc) for _, cxx, cc in found]


def _unique_compiler_pairs(
    candidates: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for cxx, cc in candidates:
        key = cxx.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append((cxx, cc))
    return tuple(unique)


def _ninja_file_candidates(env: MutableMapping[str, str]) -> list[Path]:
    executable = "ninja.exe" if sys.platform == "win32" else "ninja"
    roots: list[Path] = [Path(sys.executable).resolve().parent]
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        value = env.get(key) or os.environ.get(key)
        if not value:
            continue
        prefix = Path(value)
        roots.extend(
            [
                prefix / "bin",
                prefix / "Scripts",
                prefix / "Library" / "bin",
            ]
        )

    try:
        ninja_module = importlib.import_module("ninja")
    except (ImportError, OSError):
        ninja_module = None
    bin_dir = getattr(ninja_module, "BIN_DIR", "") if ninja_module is not None else ""
    if bin_dir:
        roots.append(Path(bin_dir))

    candidates = [root / executable for root in roots]
    if sys.platform == "win32":
        for install_root in visual_studio_install_roots(env):
            candidates.append(install_root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe")
        program_files = [
            env.get("ProgramW6432"),
            env.get("ProgramFiles"),
            env.get("ProgramFiles(x86)"),
            os.environ.get("ProgramW6432"),
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ]
        for value in (item for item in program_files if item):
            vs_root = Path(value) / "Microsoft Visual Studio"
            if not vs_root.is_dir():
                continue
            candidates.extend(vs_root.glob("*/*/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"))
    else:
        candidates.extend(
            [
                Path("/usr/bin/ninja"),
                Path("/usr/bin/ninja-build"),
                Path("/usr/local/bin/ninja"),
                Path.home() / ".local" / "bin" / "ninja",
            ]
        )
    return _unique_paths(candidates)


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return sys.platform == "win32" or os.access(path, os.X_OK)


def _record_ninja(
    env: MutableMapping[str, str],
    ninja: str,
    label: str,
) -> ExecutableStatus:
    path = str(Path(ninja).resolve())
    _prepend_existing_paths(env, [Path(path).parent], prioritize=True)
    env["NINJA"] = path
    return ExecutableStatus(True, f"{label}: {path}", path)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = _normalized_path_key(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _normalized_path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))
