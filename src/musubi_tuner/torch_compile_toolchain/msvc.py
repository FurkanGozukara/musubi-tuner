"""Portable Visual Studio C++ discovery for ``torch.compile`` on Windows."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from .compiler_probe import probe_cuda_compiler
from .host_compiler_probe import probe_host_cpp_compiler


logger = logging.getLogger(__name__)


_MSVC_TOOLSET_PATTERN = re.compile(r"^\d+(?:\.\d+){1,3}$")


@dataclass(frozen=True)
class MsvcLoadStatus:
    """Summarize the detected MSVC compiler environment."""

    ok: bool
    detail: str
    changed: bool = False


def has_cl_exe(env: MutableMapping[str, str]) -> bool:
    """Return whether ``cl.exe`` is resolvable from the supplied PATH."""

    return shutil.which("cl.exe", path=_environment_value(env, "PATH")) is not None


def describe_msvc_environment(env: MutableMapping[str, str]) -> str:
    """Return a diagnostic string for the active MSVC compiler."""

    cl_path = shutil.which("cl.exe", path=_environment_value(env, "PATH")) or "cl.exe"
    version = _environment_value(env, "VCToolsVersion") or _toolset_from_cl_path(cl_path)
    if version:
        return f"MSVC compiler already on PATH ({version}, {cl_path})"
    return f"MSVC compiler already on PATH ({cl_path})"


def load_msvc_environment(
    env: MutableMapping[str, str],
    *,
    require_openmp: bool = False,
) -> MsvcLoadStatus:
    """Load the first Visual Studio compiler environment accepted by CUDA."""

    failures: list[str] = []
    rejected_environments: set[tuple[str, ...]] = set()
    candidates = _candidate_msvc_scripts(env)
    for script, args in candidates:
        loaded_env = _environment_from_vc_script(script, args, base_env=env)
        if not loaded_env:
            failures.append(f"{script}: developer environment setup failed")
            continue
        candidate_env = dict(env)
        _merge_windows_environment(candidate_env, loaded_env)
        if not has_cl_exe(candidate_env):
            failures.append(f"{script}: setup completed but cl.exe was not added to PATH")
            continue
        cl_path = shutil.which(
            "cl.exe",
            path=_environment_value(candidate_env, "PATH"),
        )
        environment_key = _compiler_environment_key(candidate_env, cl_path or "")
        if environment_key in rejected_environments:
            continue
        host_probe = probe_host_cpp_compiler(
            candidate_env,
            platform_name="win32",
            compiler_path=cl_path or "",
            require_openmp=require_openmp,
        )
        if not host_probe.ok:
            label = _environment_value(loaded_env, "VCToolsVersion") or str(script)
            failures.append(f"{label}: {host_probe.detail}")
            rejected_environments.add(environment_key)
            logger.debug("Skipping incomplete MSVC environment: %s", failures[-1])
            continue
        probe = probe_cuda_compiler(candidate_env, platform_name="win32")
        version = _environment_value(loaded_env, "VCToolsVersion")
        if probe.ok:
            _merge_windows_environment(env, loaded_env)
            detail = (
                f"MSVC {version} loaded from {script}; {host_probe.detail}; {probe.detail}"
                if version
                else f"MSVC loaded from {script}; {host_probe.detail}; {probe.detail}"
            )
            logger.info("torch.compile %s", detail)
            return MsvcLoadStatus(True, detail, True)
        label = version or str(script)
        failures.append(f"{label}: {probe.detail}")
        rejected_environments.add(environment_key)
        logger.debug("Skipping MSVC candidate for torch.compile: %s", failures[-1])
    detail = (
        "MSVC cl.exe was not found after checking PATH, vswhere, Visual Studio, "
        "and Visual Studio Build Tools installations. Install the Desktop development "
        "with C++ workload."
    )
    if failures:
        detail = f"No usable CUDA-compatible MSVC toolset found. Tried: {_summarize_failures(failures)}"
    elif not candidates:
        detail += " No Visual Studio developer command scripts were discovered."
    return MsvcLoadStatus(
        False,
        detail,
        False,
    )


def _candidate_msvc_scripts(
    env: Mapping[str, str] | None = None,
) -> list[tuple[Path, str]]:
    configured = _existing_unique_scripts(_configured_dev_scripts(env))
    candidates: list[tuple[Path, str]] = []
    for root in visual_studio_install_roots(env):
        candidates.extend(_scripts_for_vs_root(root))
    discovered = _sort_msvc_candidates(_existing_unique_scripts(candidates))
    configured_keys = {(str(script).casefold(), args) for script, args in configured}
    return configured + [
        candidate for candidate in discovered if (str(candidate[0]).casefold(), candidate[1]) not in configured_keys
    ]


def visual_studio_install_roots(
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Discover existing Visual Studio roots, including custom install locations.

    The same root discovery is shared by MSVC and Ninja selection so a standalone
    Build Tools installation on a non-default drive is treated exactly like a full
    Visual Studio installation.
    """

    source_env = os.environ if env is None else env
    roots: list[Path] = []
    vswhere = _vswhere_path(source_env)
    if vswhere is not None:
        roots.extend(Path(path) for path in _vswhere_install_paths(vswhere))

    for env_key in (
        "MUSUBI_VS_INSTALLDIR",
        "VSINSTALLDIR",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "VS170COMNTOOLS",
        "VS160COMNTOOLS",
        "VS150COMNTOOLS",
        "VS140COMNTOOLS",
        "VS120COMNTOOLS",
    ):
        value = _environment_value(source_env, env_key) or os.environ.get(env_key)
        if value:
            root = _find_visual_studio_root(_environment_path(value))
            if root is not None:
                roots.append(root)
    roots.extend(_registry_visual_studio_roots())
    roots.extend(_standard_visual_studio_roots(source_env))
    return [root for root in _unique_paths(roots) if root.is_dir()]


def _existing_unique_scripts(candidates: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Path, str]] = []
    for script, args in candidates:
        key = (str(script).casefold(), args)
        if script.is_file() and key not in seen:
            unique.append((script, args))
            seen.add(key)
    return unique


def _sort_msvc_candidates(
    candidates: list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    """Try the newest installed toolsets first across every VS installation."""

    def sort_key(candidate: tuple[Path, str]) -> tuple[int, ...]:
        match = re.search(r"(?:^|\s)-vcvars_ver=(\d+(?:\.\d+)*)", candidate[1])
        return _version_key(match.group(1)) if match else (0,)

    return sorted(candidates, key=sort_key, reverse=True)


def _scripts_for_vs_root(root: Path) -> list[tuple[Path, str]]:
    build_root = root / "VC" / "Auxiliary" / "Build"
    scripts = [
        (root / "Common7" / "Tools" / "VsDevCmd.bat", "-arch=amd64 -host_arch=amd64"),
        (build_root / "vcvars64.bat", ""),
        (build_root / "vcvarsall.bat", "amd64"),
        (build_root / "vcvarsall.bat", "x64"),
        # Visual Studio 2015 and older use this pre-Auxiliary layout. Keeping it
        # as a fallback supports matching legacy CUDA/PyTorch distributions.
        (root / "VC" / "vcvarsall.bat", "amd64"),
        (root / "VC" / "vcvarsall.bat", "x64"),
    ]
    candidates: list[tuple[Path, str]] = []
    for version in _msvc_toolset_versions(root):
        selectors = [version]
        family = ".".join(version.split(".")[:2])
        if family and family != version:
            selectors.append(family)
        for selector in selectors:
            for script, args in scripts:
                candidates.append((script, f"{args} -vcvars_ver={selector}".strip()))
    candidates.extend(scripts)
    candidates.append((root / "Common7" / "Tools" / "VsDevCmd.bat", ""))
    return candidates


def _configured_dev_scripts(
    env: Mapping[str, str] | None,
) -> list[tuple[Path, str]]:
    source_env = os.environ if env is None else env
    candidates: list[tuple[Path, str]] = []
    for key in ("MUSUBI_VS_DEV_CMD", "VS_DEV_CMD"):
        value = _environment_value(source_env, key) or os.environ.get(key, "")
        if not value:
            continue
        script = _environment_path(value)
        name = script.name.casefold()
        if name == "vsdevcmd.bat":
            args = "-arch=amd64 -host_arch=amd64"
        elif name == "vcvarsall.bat":
            args = "amd64"
        else:
            args = ""
        candidates.append((script, args))
    return candidates


def _vswhere_path(env: Mapping[str, str] | None = None) -> Path | None:
    source_env = os.environ if env is None else env
    for key in ("MUSUBI_VSWHERE", "VSWHERE"):
        configured = _environment_value(source_env, key)
        if configured and _environment_path(configured).is_file():
            return _environment_path(configured)
    found = shutil.which("vswhere.exe", path=_environment_value(source_env, "PATH"))
    if found:
        return Path(found)
    for base in _program_files_roots(source_env):
        path = base / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if path.is_file():
            return path
    return None


def _vswhere_install_paths(vswhere: Path) -> list[str]:
    prefix = [
        str(vswhere),
        "-products",
        "*",
        "-all",
    ]
    suffix = [
        "-sort",
        "-property",
        "installationPath",
        "-utf8",
    ]
    commands = [
        [
            *prefix,
            "-prerelease",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            *suffix,
        ],
        [
            *prefix,
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            *suffix,
        ],
        [*prefix, "-prerelease", *suffix],
        [*prefix, *suffix],
    ]
    discovered: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8-sig",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        paths = [line.strip().lstrip("\ufeff") for line in completed.stdout.splitlines() if line.strip().lstrip("\ufeff")]
        if completed.returncode == 0 and paths:
            discovered.extend(paths)
            break

    legacy_command = [
        str(vswhere),
        "-legacy",
        "-all",
        "-property",
        "installationPath",
        "-utf8",
    ]
    try:
        legacy = subprocess.run(
            legacy_command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        legacy = None
    if legacy is not None and legacy.returncode == 0:
        discovered.extend(line.strip().lstrip("\ufeff") for line in legacy.stdout.splitlines() if line.strip().lstrip("\ufeff"))
    return list(dict.fromkeys(discovered))


def _environment_from_vc_script(
    script: Path,
    args: str,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Run a Visual Studio developer script and capture its environment.

    ``cmd.exe`` has quoting rules that are incompatible with Python's Windows
    sequence-to-command-line escaping when the ``/c`` payload itself contains
    quoted paths.  Pass one fully formed command line so installations under
    ``Program Files`` and custom paths containing spaces remain executable.
    """

    run_env = dict(os.environ)
    if base_env is not None:
        _merge_windows_environment(run_env, base_env)
    comspec = _resolve_comspec(run_env)
    setup = f'call "{script}"'
    if args:
        setup += f" {args}"
    setup += " >nul && set"
    command_line = f'"{comspec}" /d /u /s /c "{setup}"'
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command_line,
            check=False,
            capture_output=True,
            timeout=45,
            env=run_env,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        error = _last_nonempty_line(_decode_cmd_output(completed.stderr)) or (f"exit code {completed.returncode}")
        logger.debug("Visual Studio environment setup failed for %s: %s", script, error)
        return {}
    return _parse_set_output(_decode_cmd_output(completed.stdout))


def _parse_set_output(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line or line.startswith("=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def _msvc_toolset_versions(root: Path) -> list[str]:
    tools_root = root / "VC" / "Tools" / "MSVC"
    if not tools_root.is_dir():
        return []
    versions = [
        path.name for path in tools_root.iterdir() if _MSVC_TOOLSET_PATTERN.fullmatch(path.name) and _toolset_has_x64_cl(path)
    ]
    return sorted(versions, key=_version_key, reverse=True)


def _version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.replace("-", ".").split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _toolset_has_x64_cl(toolset: Path) -> bool:
    return any((toolset / "bin" / host / "x64" / "cl.exe").is_file() for host in ("Hostx64", "Hostx86"))


def _toolset_from_cl_path(cl_path: str) -> str:
    parts = Path(cl_path).parts
    for index, part in enumerate(parts):
        if part.casefold() == "msvc" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _environment_value(env: Mapping[str, str], key: str) -> str:
    """Read a Windows environment variable without depending on key casing."""

    exact = env.get(key)
    if exact is not None:
        return str(exact)
    folded = key.casefold()
    for candidate, value in env.items():
        if candidate.casefold() == folded:
            return str(value)
    return ""


def _merge_windows_environment(
    target: MutableMapping[str, str],
    updates: Mapping[str, str],
) -> None:
    """Merge case-insensitive Windows environment keys without duplicates."""

    existing = {key.casefold(): key for key in target}
    for key, value in updates.items():
        folded = key.casefold()
        destination = existing.get(folded, key)
        target[destination] = str(value)
        existing[folded] = destination


def _program_files_roots(env: Mapping[str, str]) -> list[Path]:
    values = [
        _environment_value(env, "ProgramW6432"),
        _environment_value(env, "ProgramFiles(x86)"),
        _environment_value(env, "ProgramFiles"),
        os.environ.get("ProgramW6432", ""),
        os.environ.get("ProgramFiles(x86)", ""),
        os.environ.get("ProgramFiles", ""),
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        path = _environment_path(value)
        key = str(path).casefold()
        if key not in seen:
            roots.append(path)
            seen.add(key)
    return roots


def _standard_visual_studio_roots(env: Mapping[str, str]) -> list[Path]:
    """Return installed and conventional Visual Studio edition roots."""

    roots: list[Path] = []
    editions = ("BuildTools", "Community", "Professional", "Enterprise")
    for base in _program_files_roots(env):
        visual_studio = base / "Microsoft Visual Studio"
        if visual_studio.is_dir():
            versions = sorted(
                (path for path in visual_studio.iterdir() if path.is_dir()),
                key=lambda path: _version_key(path.name),
                reverse=True,
            )
            for version in versions:
                installed_editions = sorted(
                    (
                        path
                        for path in version.iterdir()
                        if path.is_dir() and ((path / "Common7").is_dir() or (path / "VC").is_dir())
                    ),
                    key=lambda path: (
                        editions.index(path.name) if path.name in editions else len(editions),
                        path.name.casefold(),
                    ),
                )
                roots.extend(installed_editions)
        for year in ("2022", "2019", "2017"):
            roots.extend(visual_studio / year / edition for edition in editions)
        roots.append(base / "Microsoft Visual Studio 14.0")
    return roots


def _registry_visual_studio_roots() -> list[Path]:
    """Find custom Visual Studio locations when vswhere is unavailable."""

    try:
        import winreg
    except ImportError:
        return []

    roots: list[Path] = []
    key_names = (
        r"SOFTWARE\Microsoft\VisualStudio\SxS\VS7",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7",
    )
    hives = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
    access_modes = (
        winreg.KEY_READ,
        winreg.KEY_READ | getattr(winreg, "KEY_WOW64_32KEY", 0),
        winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
    )
    for hive in hives:
        for key_name in key_names:
            for access in access_modes:
                try:
                    key = winreg.OpenKey(hive, key_name, 0, access)
                except OSError:
                    continue
                try:
                    index = 0
                    while True:
                        try:
                            _, value, _ = winreg.EnumValue(key, index)
                        except OSError:
                            break
                        if isinstance(value, str) and value.strip():
                            roots.append(Path(value.strip()))
                        index += 1
                finally:
                    winreg.CloseKey(key)
    return _unique_paths(roots)


def _find_visual_studio_root(path: Path) -> Path | None:
    """Walk upward from VS/VC tool variables to the owning installation root."""

    path = path.expanduser()
    candidates = [path, *path.parents]
    for candidate in candidates:
        if (candidate / "VC" / "Auxiliary" / "Build").is_dir() or (candidate / "Common7" / "Tools").is_dir():
            return candidate
    return path if path.is_dir() else None


def _last_nonempty_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _environment_path(value: str) -> Path:
    return Path(str(value).strip().strip('"')).expanduser()


def _resolve_comspec(env: Mapping[str, str]) -> str:
    configured = _environment_value(env, "COMSPEC")
    if configured:
        path = _environment_path(configured)
        if path.is_file():
            return str(path)
        resolved = shutil.which(configured, path=_environment_value(env, "PATH"))
        if resolved:
            return resolved
    return shutil.which("cmd.exe", path=_environment_value(env, "PATH")) or "cmd.exe"


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _decode_cmd_output(output: bytes | str) -> str:
    """Decode ``cmd /u`` output while keeping mocked/string callers convenient."""

    if isinstance(output, str):
        return output
    return output.decode("utf-16le", errors="replace")


def _compiler_environment_key(
    env: Mapping[str, str],
    cl_path: str,
) -> tuple[str, ...]:
    """Identify duplicate setup-script results without hiding distinct SDK setups."""

    resolved_cl = str(Path(cl_path).resolve()).casefold() if cl_path else ""
    return (
        resolved_cl,
        _environment_value(env, "VCToolsVersion").casefold(),
        _environment_value(env, "WindowsSdkDir").casefold(),
        _environment_value(env, "WindowsSDKVersion").casefold(),
        _environment_value(env, "INCLUDE").casefold(),
        _environment_value(env, "LIB").casefold(),
    )


def _summarize_failures(failures: list[str], limit: int = 4) -> str:
    unique = list(dict.fromkeys(failures))
    shown = unique[:limit]
    summary = " | ".join(shown)
    if len(unique) > limit:
        summary += f" | +{len(unique) - limit} more candidates"
    return summary
