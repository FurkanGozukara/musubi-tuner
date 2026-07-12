"""Portable automatic toolchain setup for :func:`torch.compile`.

The package is dependency-light and can prepare Windows and Linux build tools for
both in-process compilation and worker subprocesses.
"""

from .compiler_probe import CompilerProbeStatus, probe_cuda_compiler
from .discovery import (
    CudaDiscoveryStatus,
    ExecutableStatus,
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from .environment import (
    CompileToolchainStatus,
    compile_environment_report,
    ensure_compile_environment,
    prepare_compile_subprocess_env,
)
from .host_compiler_probe import HostCompilerProbeStatus, probe_host_cpp_compiler
from .msvc import (
    MsvcLoadStatus,
    describe_msvc_environment,
    has_cl_exe,
    load_msvc_environment,
    visual_studio_install_roots,
)
from .runtime import SafeCompileResult, compile_callable, compile_module_callable

__version__ = "2.0.0-musubi"

__all__ = [
    "CompileToolchainStatus",
    "CompilerProbeStatus",
    "CudaDiscoveryStatus",
    "ExecutableStatus",
    "HostCompilerProbeStatus",
    "MsvcLoadStatus",
    "SafeCompileResult",
    "compile_callable",
    "compile_environment_report",
    "compile_module_callable",
    "describe_msvc_environment",
    "discover_cuda_environment",
    "discover_ninja",
    "discover_posix_compiler",
    "ensure_compile_environment",
    "has_cl_exe",
    "load_msvc_environment",
    "prepare_compile_subprocess_env",
    "probe_cuda_compiler",
    "probe_host_cpp_compiler",
    "visual_studio_install_roots",
]
