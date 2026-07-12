"""Portable CUDA host-compiler compatibility checks for ``torch.compile``."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping


@dataclass(frozen=True)
class CompilerProbeStatus:
    """Describe whether CUDA accepted the selected host compiler."""

    ok: bool
    detail: str
    skipped: bool = False


def probe_cuda_compiler(
    env: MutableMapping[str, str],
    *,
    platform_name: str,
    compiler_path: str = "",
    timeout: int = 45,
) -> CompilerProbeStatus:
    """Compile a tiny CUDA translation unit to verify host compiler compatibility.

    Args:
        env: Environment containing CUDA, PATH, and host compiler variables.
        platform_name: Runtime platform name, usually ``sys.platform``.
        compiler_path: Optional POSIX compiler executable to pass to ``nvcc -ccbin``.
        timeout: Probe timeout in seconds.

    Returns:
        Probe status. Missing ``nvcc`` is treated as a skipped compatibility check.
    """

    nvcc = _nvcc_path(env, platform_name)
    if not nvcc:
        return CompilerProbeStatus(True, "nvcc not present; host compiler probe skipped", True)

    with tempfile.TemporaryDirectory(prefix="torch_compile_probe_") as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "probe.cu"
        output = temp_path / ("probe.obj" if platform_name == "win32" else "probe.o")
        source.write_text(
            'extern "C" __global__ void torch_compile_toolchain_probe() {}\n',
            encoding="utf-8",
        )
        command = [nvcc, "-c", str(source), "-o", str(output)]
        if compiler_path and platform_name != "win32":
            command[1:1] = ["-ccbin", compiler_path]
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
            return CompilerProbeStatus(False, f"CUDA host compiler probe failed: {exc}")

    if completed.returncode == 0:
        return CompilerProbeStatus(True, "CUDA host compiler probe passed")
    return CompilerProbeStatus(
        False,
        f"CUDA host compiler probe failed: {_probe_error(completed)}",
    )


def _nvcc_path(env: MutableMapping[str, str], platform_name: str) -> str:
    for key in ("CUDACXX", "NVCC"):
        configured = str(env.get(key, "") or "").strip().strip('"')
        if not configured:
            continue
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(configured, path=env.get("PATH"))
        if resolved:
            return resolved
    candidates = ["nvcc.exe", "nvcc"] if platform_name == "win32" else ["nvcc"]
    for name in candidates:
        nvcc = shutil.which(name, path=env.get("PATH"))
        if nvcc:
            return nvcc
    return ""


def _probe_error(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (completed.stderr, completed.stdout) if part and part.strip())
    if not output:
        return f"nvcc exited with code {completed.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.casefold()
        if any(token in lowered for token in ("fatal", "error", "unsupported", "failed")):
            return line
    return lines[-1] if lines else output
