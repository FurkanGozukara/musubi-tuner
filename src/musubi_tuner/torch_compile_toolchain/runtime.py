"""Small dependency-light helpers for applying ``torch.compile`` safely."""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .environment import CompileToolchainStatus, ensure_compile_environment


logger = logging.getLogger(__name__)
StatusCallback = Callable[[str, str], None]
_NATIVE_CODEGEN_BACKENDS = {"inductor"}


@dataclass
class SafeCompileResult:
    """Mutable state for a callable guarded by eager fallback.

    ``callable`` is always usable. Before its first invocation, ``compiled`` means
    that ``torch.compile`` created a wrapper. Afterward, ``verified`` confirms a
    successful compiled call; if the cold compiled call failed, ``compiled`` is
    changed to ``False`` and later calls use the original eager callable.
    """

    callable: Callable[..., Any]
    requested: bool
    compiled: bool
    verified: bool = False
    detail: str = ""
    attempts: int = 0
    toolchain: CompileToolchainStatus | None = None


def compile_callable(
    target: Callable[..., Any],
    *,
    enabled: bool = True,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
    require_openmp: bool = False,
    fallback_on_first_error: bool = True,
    on_status: StatusCallback | None = None,
) -> SafeCompileResult:
    """Compile one callable and guard its cold call with optional eager fallback.

    PyTorch is imported lazily, so importing the package remains safe in installer
    and launcher processes that do not import the ML stack themselves.
    """

    if not callable(target):
        raise TypeError("target must be callable")
    if not enabled:
        return _result(target, False, False, "disabled", on_status)

    backend_name = str(backend or "").strip().casefold()
    if backend_name in _NATIVE_CODEGEN_BACKENDS:
        toolchain = ensure_compile_environment(
            os.environ,
            project_root=project_root,
            cache_dir=cache_dir,
            require_cuda_toolkit=require_cuda_toolkit,
            require_ninja=require_ninja,
            require_openmp=require_openmp,
        )
    else:
        toolchain = CompileToolchainStatus(
            True,
            f"backend {backend!r} does not require the Inductor host toolchain",
        )
    if not toolchain.ok:
        return _result(
            target,
            True,
            False,
            f"toolchain unavailable: {toolchain.detail}",
            on_status,
            toolchain=toolchain,
        )

    try:
        import torch
    except (ImportError, OSError) as exc:
        return _result(
            target,
            True,
            False,
            f"PyTorch import failed: {exc}",
            on_status,
            toolchain=toolchain,
        )
    torch_compile = getattr(torch, "compile", None)
    if not callable(torch_compile):
        return _result(
            target,
            True,
            False,
            "torch.compile is unavailable",
            on_status,
            toolchain=toolchain,
        )

    compile_kwargs: dict[str, Any] = {"backend": backend, "fullgraph": fullgraph}
    if mode is not None:
        compile_kwargs["mode"] = mode
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    try:
        compiled_target = torch_compile(target, **compile_kwargs)
    except Exception as exc:
        return _result(
            target,
            True,
            False,
            f"torch.compile setup failed: {exc}",
            on_status,
            attempts=1,
            toolchain=toolchain,
        )

    result = SafeCompileResult(
        callable=target,
        requested=True,
        compiled=True,
        detail=f"compiled wrapper created; {toolchain.detail}",
        attempts=1,
        toolchain=toolchain,
    )
    active = True

    @functools.wraps(target)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        nonlocal active
        if not active:
            return target(*args, **kwargs)
        if result.verified:
            return compiled_target(*args, **kwargs)
        try:
            output = compiled_target(*args, **kwargs)
        except Exception as exc:
            if not fallback_on_first_error:
                raise
            active = False
            result.compiled = False
            result.detail = f"first compiled call failed; using eager fallback: {exc}"
            _notify("fallback_eager", result.detail, on_status, warning=True)
            return target(*args, **kwargs)
        result.verified = True
        result.detail = f"first compiled call succeeded; {toolchain.detail}"
        _notify("first_call_ok", result.detail, on_status)
        return output

    result.callable = guarded
    _notify("setup_ready", result.detail, on_status)
    return result


def compile_module_callable(
    module: Any,
    attribute_name: str = "forward",
    *,
    enabled: bool = True,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
    require_openmp: bool = False,
    fallback_on_first_error: bool = True,
    on_status: StatusCallback | None = None,
) -> SafeCompileResult:
    """Compile and replace a module callable such as ``forward`` or ``decode``."""

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise TypeError(f"module.{attribute_name} must be callable")
    result = compile_callable(
        target,
        enabled=enabled,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        project_root=project_root,
        cache_dir=cache_dir,
        require_cuda_toolkit=require_cuda_toolkit,
        require_ninja=require_ninja,
        require_openmp=require_openmp,
        fallback_on_first_error=fallback_on_first_error,
        on_status=on_status,
    )
    setattr(module, attribute_name, result.callable)
    return result


def _result(
    target: Callable[..., Any],
    requested: bool,
    compiled: bool,
    detail: str,
    callback: StatusCallback | None,
    *,
    attempts: int = 0,
    toolchain: CompileToolchainStatus | None = None,
) -> SafeCompileResult:
    result = SafeCompileResult(
        callable=target,
        requested=requested,
        compiled=compiled,
        detail=detail,
        attempts=attempts,
        toolchain=toolchain,
    )
    status = "setup_ready" if compiled else ("unavailable" if requested else "disabled")
    _notify(
        status,
        detail,
        callback,
        warning=requested and not compiled,
    )
    return result


def _notify(
    status: str,
    detail: str,
    callback: StatusCallback | None,
    *,
    warning: bool = False,
) -> None:
    if callback is not None:
        try:
            callback(status, detail)
        except Exception:
            logger.exception("torch.compile status callback failed")
    log = logger.warning if warning else logger.info
    log("torch.compile status=%s detail=%s", status, detail)
