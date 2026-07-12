# Musubi torch.compile Toolchain

This package prepares the native Windows or Linux build environment used by
PyTorch Inductor and Musubi's training-time `torch.compile` support. It uses only
the Python standard library until PyTorch's CUDA version must be inspected.

## Public API

```python
from musubi_tuner.torch_compile_toolchain import ensure_compile_environment

status = ensure_compile_environment(
    project_root="/path/to/musubi-tuner",
    require_openmp=True,
)
if not status.ok:
    raise RuntimeError(status.detail)
```

For a child process:

```python
from musubi_tuner.torch_compile_toolchain import prepare_compile_subprocess_env

child_env = prepare_compile_subprocess_env(
    project_root="/path/to/musubi-tuner",
    compile_requested=True,
    require_openmp=True,
)
```

The returned `CompileToolchainStatus` includes the selected compiler, CUDA root,
Ninja executable, cache root, whether the environment changed, and a diagnostic
summary.

## Windows

Discovery covers:

- An already active Visual Studio developer environment.
- Full Visual Studio and standalone Build Tools installations.
- `vswhere`, registry, standard locations, custom drives, prerelease, future,
  and legacy layouts.
- `MUSUBI_VS_INSTALLDIR`, `MUSUBI_VS_DEV_CMD`, `MUSUBI_VSWHERE`, and standard
  Visual Studio environment overrides.
- Every installed x64 MSVC toolset, ordered by the actual toolset version.
- Exact and family `-vcvars_ver` selection with fallback to older toolsets.

An existing `cl.exe` is not accepted merely because it is on `PATH`. The selected
environment must compile, link, and run a C++17 program using standard-library
headers. With `require_openmp=True`, it must also compile and link OpenMP. If
`nvcc` is installed, a real CUDA source is compiled so CUDA-incompatible MSVC
versions are skipped.

## Linux

Discovery honors `NVCC_CCBIN`, `CUDAHOSTCXX`, `CXX`, and `CC`, then checks common
GCC, Clang, NVIDIA HPC, Intel, ARM, Conda, and versioned compiler locations. Each
candidate must pass the native C++ probe before it can pass the optional
`nvcc -ccbin` compatibility probe.

## CUDA, Ninja, And Caches

CUDA Toolkit candidates come from explicit environment variables, Conda, standard
installation paths, and `nvcc` on `PATH`. An exact `torch.version.cuda` match is
preferred, followed by the nearest installed minor version in the same major
release. The chosen toolkit is placed first on `PATH`.

Ninja is discovered from overrides, `PATH`, virtual environments, Conda, the
Python `ninja` package, and Visual Studio's CMake installation.

Default cache directories are:

```text
<project>/.cache/torch_compile_toolchain/inductor
<project>/.cache/torch_compile_toolchain/triton
```

Explicit `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` values are preserved.

## Diagnostic

```text
python -m musubi_tuner.torch_compile_toolchain --require-openmp --smoke-test
python -m musubi_tuner.torch_compile_toolchain --json --verbose
```

The smoke test executes a real compiled forward and backward pass on CUDA when
available, otherwise on CPU.
