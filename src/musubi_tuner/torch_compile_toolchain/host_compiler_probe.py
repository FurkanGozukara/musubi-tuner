"""Native C++ compiler readiness checks for :func:`torch.compile`."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class HostCompilerProbeStatus:
    """Describe whether a host C++ compiler can compile, link, and run code."""

    ok: bool
    detail: str
    path: str = ""


def probe_host_cpp_compiler(
    env: Mapping[str, str],
    *,
    platform_name: str | None = None,
    compiler_path: str = "",
    require_openmp: bool = False,
    timeout: int = 45,
) -> HostCompilerProbeStatus:
    """Compile and run a small C++17 program using standard-library headers.

    Merely finding ``cl.exe`` or ``g++`` is insufficient. On Windows in
    particular, users often add the compiler binary to PATH without activating
    the Visual Studio developer environment, leaving INCLUDE, LIB, and the
    Windows SDK unavailable. This probe catches that partial configuration.
    """

    platform_name = platform_name or sys.platform
    compiler = _resolve_compiler(env, platform_name, compiler_path)
    if not compiler:
        label = "cl.exe" if platform_name == "win32" else "a C++ compiler"
        return HostCompilerProbeStatus(False, f"{label} was not found")

    with tempfile.TemporaryDirectory(prefix="torch_compile_host_probe_") as temp_dir:
        root = Path(temp_dir)
        source = root / "probe.cpp"
        output = root / ("probe.exe" if platform_name == "win32" else "probe")
        source.write_text(_probe_source(require_openmp), encoding="utf-8")
        command = _compile_command(
            compiler,
            source,
            output,
            platform_name=platform_name,
            require_openmp=require_openmp,
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(env),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HostCompilerProbeStatus(
                False,
                f"host C++ compiler probe failed: {exc}",
                compiler,
            )

        if completed.returncode != 0 or not output.is_file():
            return HostCompilerProbeStatus(
                False,
                f"host C++ compiler probe failed: {_probe_error(completed)}",
                compiler,
            )

        try:
            executed = subprocess.run(
                [str(output)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(env),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HostCompilerProbeStatus(
                False,
                f"compiled host probe could not run: {exc}",
                compiler,
            )

    if executed.returncode != 0:
        return HostCompilerProbeStatus(
            False,
            f"compiled host probe exited with code {executed.returncode}",
            compiler,
        )
    suffix = " with OpenMP" if require_openmp else ""
    return HostCompilerProbeStatus(
        True,
        f"host C++17 compiler probe passed{suffix}",
        compiler,
    )


def _resolve_compiler(
    env: Mapping[str, str],
    platform_name: str,
    configured: str,
) -> str:
    if configured:
        resolved = _resolve_executable(configured, env)
        if resolved:
            return resolved
    if platform_name == "win32":
        return shutil.which("cl.exe", path=_env_value(env, "PATH")) or ""
    for key in ("CXX", "CUDAHOSTCXX", "NVCC_CCBIN"):
        value = _env_value(env, key)
        if value:
            resolved = _resolve_executable(value, env)
            if resolved:
                return resolved
    for name in ("c++", "g++", "clang++", "icpx", "icpc"):
        resolved = shutil.which(name, path=_env_value(env, "PATH"))
        if resolved:
            return resolved
    return ""


def _resolve_executable(value: str, env: Mapping[str, str]) -> str:
    text = str(value or "").strip().strip('"')
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(text, path=_env_value(env, "PATH")) or ""


def _compile_command(
    compiler: str,
    source: Path,
    output: Path,
    *,
    platform_name: str,
    require_openmp: bool,
) -> list[str]:
    if platform_name == "win32":
        command = [
            compiler,
            "/nologo",
            "/EHsc",
            "/std:c++17",
            str(source),
            f"/Fe:{output}",
            f"/Fo:{output.with_suffix('.obj')}",
        ]
        if require_openmp:
            command.insert(4, "/openmp")
        return command
    command = [compiler, "-std=c++17", str(source), "-o", str(output)]
    if require_openmp:
        command.append("-fopenmp")
    return command


def _probe_source(require_openmp: bool) -> str:
    openmp = "#include <omp.h>\n" if require_openmp else ""
    openmp_use = " + (omp_get_max_threads() > 0 ? 0 : 1)" if require_openmp else ""
    return (
        "#include <algorithm>\n"
        "#include <cmath>\n"
        "#include <vector>\n"
        f"{openmp}"
        "int main() {\n"
        "  std::vector<double> values{1.0, 2.0, 3.0};\n"
        "  const auto maximum = *std::max_element(values.begin(), values.end());\n"
        f"  return std::abs(maximum - 3.0) < 0.001 ? 0{openmp_use} : 2;\n"
        "}\n"
    )


def _probe_error(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (completed.stderr, completed.stdout) if part and part.strip())
    if not output:
        return f"compiler exited with code {completed.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.casefold()
        if any(token in lowered for token in ("fatal", "error", "unsupported", "failed")):
            return line
    return lines[-1]


def _env_value(env: Mapping[str, str], key: str) -> str:
    exact = env.get(key)
    if exact is not None:
        return str(exact)
    folded = key.casefold()
    for candidate, value in env.items():
        if candidate.casefold() == folded:
            return str(value)
    return ""
