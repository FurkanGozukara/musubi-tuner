import argparse
import atexit
import hashlib
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional
import logging
import safetensors.torch
import torch

from musubi_tuner.training.compile_setup import ensure_training_compile_environment

logger = logging.getLogger(__name__)

_COMPILE_DIAGNOSTIC_STATES: dict[Path, list[dict[str, Any]]] = {}


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def dtype_to_str(dtype: torch.dtype) -> str:
    # get name of the dtype
    dtype_name = str(dtype).split(".")[-1]
    return dtype_name


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


@lru_cache(maxsize=1)
def _known_dtype_strs() -> tuple[str, ...]:
    """All dtype strings dtype_to_str() can emit, longest first.

    Sorting by length keeps multi-underscore names (e.g. "float8_e4m3fn") ahead of their
    prefixes ("float8...") so the longest matching suffix wins.
    """
    names = set()
    for attr in dir(torch):
        try:
            obj = getattr(torch, attr)
        except Exception:
            continue
        if isinstance(obj, torch.dtype):
            names.add(dtype_to_str(obj))
    return tuple(sorted(names, key=len, reverse=True))


def remove_dtype_suffix(name: str) -> str:
    """Remove a trailing ``_<dtype>`` suffix (as written by dtype_to_str) from a cache key.

    Robust to dtype names that contain underscores such as ``float8_e4m3fn``; a plain
    ``rsplit("_", 1)`` would only drop the final ``fn`` segment. Returns ``name`` unchanged
    if it does not end with a known dtype suffix.
    """
    for dtype_str in _known_dtype_strs():
        suffix = "_" + dtype_str
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


def disable_linear_from_compile(module: torch.nn.Module):
    """Monkey-patch to disable torch.compile for all Linear layers (if the class name ends with 'Linear') in the given module."""
    for sub_module in module.modules():
        # if isinstance(sub_module, torch.nn.Linear):
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
    disable_linear: bool,
    offloaders: Optional[list[Any]] = None,
) -> torch.nn.Module:
    fallback_on_first_error = _compile_fallback_enabled()
    compile_state = {
        "enabled": True,
        "verified": set(),
        "fallback_reason": "",
        "reported_success": False,
        "policy": "all_blocks",
        "plans": [],
    }
    if str(args.compile_backend).strip().casefold() == "inductor":
        toolchain = ensure_training_compile_environment(required=not fallback_on_first_error)
        if not toolchain.ok:
            return _use_eager_compile_fallback(
                transformer,
                compile_state,
                f"native toolchain unavailable: {toolchain.detail}",
            )
    resident_only_requested = bool(getattr(args, "compile_resident_blocks_only", False))
    resident_only = resident_only_requested and disable_linear
    compile_state["policy"] = "resident_only" if resident_only else "all_blocks"

    if resident_only:
        selector_error = None
        if offloaders is None or len(offloaders) != len(target_blocks):
            selector_error = "no residency selector was supplied for every block group"
        elif any(
            offloader is not None and not callable(getattr(offloader, "compile_safe_block_indices", None))
            for offloader in offloaders
        ):
            selector_error = "an offloader does not expose compile_safe_block_indices()"

        if selector_error is None:
            try:
                for group_index, (blocks, offloader) in enumerate(zip(target_blocks, offloaders)):
                    compile_indices = (
                        list(range(len(blocks)))
                        if offloader is None
                        else sorted(set(offloader.compile_safe_block_indices()))
                    )
                    if any(index < 0 or index >= len(blocks) for index in compile_indices):
                        raise ValueError(
                            f"Invalid resident compile index in block group {group_index}: {compile_indices}"
                        )
                    compile_set = set(compile_indices)
                    eager_indices = [index for index in range(len(blocks)) if index not in compile_set]
                    compile_state["plans"].append(
                        _compile_group_plan(blocks, compile_indices, eager_indices, disabled_linear_indices=[])
                    )
            except Exception as exc:
                selector_error = f"the residency selector failed: {type(exc).__name__}: {exc}"

        if selector_error is not None:
            resident_only = False
            compile_state["plans"].clear()
            compile_state["policy"] = "all_blocks_fallback_missing_residency_selector"
            compile_state["policy_fallback_reason"] = selector_error
            logger.warning(
                "Resident-only compile is unavailable because %s; using the established all-block compile policy "
                "with moving Linear modules kept eager.",
                selector_error,
            )

    if not resident_only:
        for blocks in target_blocks:
            all_indices = list(range(len(blocks)))
            compile_state["plans"].append(
                _compile_group_plan(
                    blocks,
                    all_indices,
                    [],
                    disabled_linear_indices=all_indices if disable_linear else [],
                )
            )

    if disable_linear and not resident_only:
        logger.info("Disable linear from torch.compile for swap blocks...")
        for blocks in target_blocks:
            for block in blocks:
                disable_linear_from_compile(block)

    compile_dynamic = None
    if args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

    logger.info(
        f"Compiling DiT model with torch.compile: backend={args.compile_backend}, mode={args.compile_mode}, dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )

    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit

    installed_blocks = []
    try:
        for group_index, blocks in enumerate(target_blocks):
            compile_indices = compile_state["plans"][group_index]["compile_indices"]
            for i in compile_indices:
                block = blocks[i]
                original_block = block
                compiled_block = torch.compile(
                    block,
                    backend=args.compile_backend,
                    mode=args.compile_mode,
                    dynamic=compile_dynamic,
                    fullgraph=args.compile_fullgraph,
                )
                _install_first_call_fallback(
                    compiled_block,
                    original_block,
                    compile_state,
                    label=f"{original_block.__class__.__name__}[{group_index}:{i}]",
                    fallback_enabled=fallback_on_first_error,
                )
                blocks[i] = compiled_block
                installed_blocks.append((blocks, i, original_block))
    except Exception as exc:
        if not fallback_on_first_error:
            raise
        for blocks, index, original_block in installed_blocks:
            blocks[index] = original_block
        return _use_eager_compile_fallback(
            transformer,
            compile_state,
            f"torch.compile setup failed: {type(exc).__name__}: {exc}",
        )
    compile_state["expected_compiled_blocks"] = sum(len(plan["compile_indices"]) for plan in compile_state["plans"])
    compile_state["enabled"] = compile_state["expected_compiled_blocks"] > 0
    if not compile_state["enabled"]:
        os.environ["MUSUBI_TORCH_COMPILE_ACTIVE"] = "0"
        logger.info("Resident-only compile selected no safe blocks; transformer remains eager.")
    else:
        logger.info(
            "Compile block plan: %s compiled, %s eager; %s Linear modules compile-eligible, %s eager-disabled",
            compile_state["expected_compiled_blocks"],
            sum(len(plan["eager_indices"]) for plan in compile_state["plans"]),
            sum(plan["compile_eligible_linears"] for plan in compile_state["plans"]),
            sum(plan["eager_linears"] for plan in compile_state["plans"]),
        )
    transformer.__dict__["_musubi_compile_state"] = compile_state
    diagnostics_path = os.environ.get("MUSUBI_COMPILE_DIAGNOSTICS_PATH")
    if diagnostics_path:
        _register_compile_diagnostics(Path(diagnostics_path), compile_state)
    return transformer


def _use_eager_compile_fallback(
    transformer: torch.nn.Module,
    compile_state: dict[str, Any],
    reason: str,
) -> torch.nn.Module:
    compile_state["enabled"] = False
    compile_state["fallback_reason"] = reason
    compile_state["expected_compiled_blocks"] = 0
    os.environ["MUSUBI_TORCH_COMPILE_ACTIVE"] = "0"
    transformer.__dict__["_musubi_compile_state"] = compile_state
    diagnostics_path = os.environ.get("MUSUBI_COMPILE_DIAGNOSTICS_PATH")
    if diagnostics_path:
        _register_compile_diagnostics(Path(diagnostics_path), compile_state)
    logger.warning("torch.compile is unavailable; continuing with eager training: %s", reason)
    return transformer


def _linear_count(module: torch.nn.Module) -> int:
    return sum(1 for item in module.modules() if item.__class__.__name__.endswith("Linear"))


def _compile_group_plan(
    blocks: torch.nn.ModuleList | list[torch.nn.Module],
    compile_indices: list[int],
    eager_indices: list[int],
    disabled_linear_indices: list[int],
) -> dict[str, Any]:
    compile_set = set(compile_indices)
    disabled_set = set(disabled_linear_indices)
    linear_counts = [_linear_count(block) for block in blocks]
    compile_eligible = sum(
        count for index, count in enumerate(linear_counts) if index in compile_set and index not in disabled_set
    )
    eager_linears = sum(linear_counts) - compile_eligible
    return {
        "total_blocks": len(blocks),
        "compile_indices": compile_indices,
        "eager_indices": eager_indices,
        "linear_modules": sum(linear_counts),
        "compile_eligible_linears": compile_eligible,
        "eager_linears": eager_linears,
    }


def _register_compile_diagnostics(path: Path, compile_state: dict[str, Any]) -> None:
    states = _COMPILE_DIAGNOSTIC_STATES.setdefault(path.resolve(), [])
    states.append(compile_state)
    if len(states) == 1:
        atexit.register(_write_compile_diagnostics, path.resolve())


def _write_compile_diagnostics(path: Path) -> None:
    try:
        from torch._dynamo import guard_failures
        from torch._dynamo.utils import counters

        counter_payload = {
            str(category): {str(key): int(value) for key, value in counter.items()}
            for category, counter in counters.items()
        }
        failures = []
        for code, reasons in guard_failures.items():
            failures.append(
                {
                    "function": getattr(code, "co_name", "unknown"),
                    "file": getattr(code, "co_filename", "unknown"),
                    "line": getattr(code, "co_firstlineno", None),
                    "reasons": [str(reason) for reason in reasons],
                }
            )

        inductor_metrics = {}
        try:
            from torch._inductor import metrics

            for name in (
                "generated_kernel_count",
                "generated_cpp_vec_kernel_count",
                "ir_nodes_pre_fusion",
                "cpp_outer_loop_fused_inner_counts",
                "num_bytes_accessed",
                "nodes_num_elem",
            ):
                value = getattr(metrics, name, None)
                if isinstance(value, (bool, int, float, str)) or value is None:
                    inductor_metrics[name] = value
        except Exception as exc:
            inductor_metrics["error"] = f"{type(exc).__name__}: {exc}"

        states = []
        for state in _COMPILE_DIAGNOSTIC_STATES.get(path, []):
            states.append(
                {
                    **{key: value for key, value in state.items() if key != "verified"},
                    "verified": sorted(state["verified"]),
                    "verified_count": len(state["verified"]),
                }
            )
        payload = {
            "schema_version": 1,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "compile_states": states,
            "counter_scope": "process",
            "counters": counter_payload,
            "guard_failure_scope": "process",
            "guard_failure_count": sum(len(item["reasons"]) for item in failures),
            "guard_failures": failures,
            "inductor_metrics": inductor_metrics,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write compile diagnostics to %s: %s", path, exc)


def _install_first_call_fallback(
    compiled_block,
    original_block,
    state: dict,
    *,
    label: str,
    fallback_enabled: bool,
) -> None:
    compiled_forward = compiled_block.forward

    def guarded_forward(*args, **kwargs):
        if not state["enabled"]:
            return original_block(*args, **kwargs)
        try:
            output = compiled_forward(*args, **kwargs)
        except Exception as exc:
            if label in state["verified"]:
                raise
            if not fallback_enabled:
                state["fallback_reason"] = f"{label}: {exc}"
                raise
            state["enabled"] = False
            state["fallback_reason"] = f"{label}: {exc}"
            os.environ["MUSUBI_TORCH_COMPILE_ACTIVE"] = "0"
            logger.warning(
                "First compiled call failed for %s; continuing all compiled blocks in eager mode: %s",
                label,
                exc,
            )
            return original_block(*args, **kwargs)
        state["verified"].add(label)
        os.environ["MUSUBI_TORCH_COMPILE_ACTIVE"] = "1"
        if not state["reported_success"]:
            state["reported_success"] = True
            logger.info("First compiled transformer block executed successfully: %s", label)
        return output

    compiled_block.forward = guarded_forward


def _compile_fallback_enabled() -> bool:
    value = os.environ.get("MUSUBI_TORCH_COMPILE_FALLBACK", "1")
    return value.strip().casefold() not in {"", "0", "false", "no", "off"}


def compile_dynamic_arg(args: argparse.Namespace) -> bool | None:
    compile_dynamic = getattr(args, "compile_dynamic", None)
    if compile_dynamic is None:
        return None
    if isinstance(compile_dynamic, bool):
        return compile_dynamic
    return {"true": True, "false": False, "auto": None}[compile_dynamic.lower()]


def unwrap_compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap common compile/distributed wrappers without requiring an Accelerator instance."""
    unwrapped = module
    seen: set[int] = set()
    for _ in range(8):
        if id(unwrapped) in seen:
            break
        seen.add(id(unwrapped))

        next_module = getattr(unwrapped, "module", None)
        if isinstance(next_module, torch.nn.Module) and next_module is not unwrapped:
            unwrapped = next_module
            continue

        next_module = getattr(unwrapped, "_orig_mod", None)
        if isinstance(next_module, torch.nn.Module) and next_module is not unwrapped:
            unwrapped = next_module
            continue

        break
    return unwrapped


def resolve_compile_block_lists(
    module: torch.nn.Module,
    block_attr_names: tuple[str, ...] = ("transformer_blocks",),
) -> list[torch.nn.ModuleList | list[torch.nn.Module]]:
    """Find block lists on a model or a single-GPU wrapper's `.model` child."""
    roots = [unwrap_compile_module(module)]
    wrapped_model = getattr(roots[0], "model", None)
    if isinstance(wrapped_model, torch.nn.Module) and wrapped_model is not roots[0]:
        roots.append(wrapped_model)

    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]] = []
    seen: set[int] = set()
    for root in roots:
        for attr_name in block_attr_names:
            value: Any = root
            for part in attr_name.split("."):
                value = getattr(value, part, None)
                if value is None:
                    break
            if value is None or id(value) in seen:
                continue
            if isinstance(value, (torch.nn.ModuleList, list)):
                target_blocks.append(value)
                seen.add(id(value))
    return target_blocks


def _collect_compile_targets(
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
) -> list[tuple[torch.nn.ModuleList | list[torch.nn.Module], int, torch.nn.Module]]:
    targets: list[tuple[torch.nn.ModuleList | list[torch.nn.Module], int, torch.nn.Module]] = []
    for blocks in target_blocks:
        for i, block in enumerate(blocks):
            if not isinstance(block, torch.nn.Module):
                logger.debug("Skipping non-module torch.compile target at index %s: %r", i, block)
                continue
            if hasattr(block, "_hf_hook"):
                logger.info("Skipping torch.compile target at index %s because an HF offload hook is attached", i)
                continue
            targets.append((blocks, i, block))
    return targets


def _configure_compile_cache(args: argparse.Namespace, target_count: int) -> None:
    cache_size_limit = getattr(args, "compile_cache_size_limit", None)
    if getattr(args, "compile_auto_cache_size_limit", False) and target_count > 0:
        auto_limit = target_count * 2
        if cache_size_limit is None:
            current_limit = getattr(torch._dynamo.config, "cache_size_limit", 0)
            cache_size_limit = max(int(current_limit or 0), auto_limit)
        else:
            cache_size_limit = max(cache_size_limit, auto_limit)

    if cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = cache_size_limit
        logger.info("Set torch._dynamo.config.cache_size_limit to %s", cache_size_limit)


def _parse_inductor_config_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_inductor_config(args: argparse.Namespace) -> None:
    overrides = getattr(args, "inductor_config", None)
    if not overrides:
        return

    import torch._inductor.config as _ind
    import torch._dynamo.config as _dyn

    applied: list[str] = []
    for token in overrides:
        if "=" not in token:
            raise ValueError(f"--inductor_config expects KEY=VALUE tokens, got {token!r}")
        key, raw_value = token.split("=", 1)
        key = key.strip()
        value = _parse_inductor_config_value(raw_value)

        parts = key.split(".")
        root = _ind if hasattr(_ind, parts[0]) else _dyn
        target = root
        missing = False
        for part in parts[:-1]:
            if not hasattr(target, part):
                missing = True
                break
            target = getattr(target, part)
        if missing or not hasattr(target, parts[-1]):
            logger.warning("--inductor_config: skipping unknown key %s", key)
            continue
        setattr(target, parts[-1], value)
        applied.append(f"{key}={value}")

    if applied:
        logger.info("Applied %d inductor/dynamo config overrides: %s", len(applied), " ".join(applied))


def compile_module(args: argparse.Namespace, module: torch.nn.Module) -> torch.nn.Module:
    _apply_inductor_config(args)
    compile_dynamic = compile_dynamic_arg(args)
    logger.info(
        "Compiling model with torch.compile: backend=%s, mode=%s, dynamic=%s, fullgraph=%s",
        args.compile_backend,
        args.compile_mode,
        compile_dynamic,
        args.compile_fullgraph,
    )
    _configure_compile_cache(args, 1)
    try:
        return torch.compile(
            module,
            backend=args.compile_backend,
            mode=args.compile_mode,
            dynamic=compile_dynamic,
            fullgraph=args.compile_fullgraph,
        )
    except Exception as e:
        if getattr(args, "compile_fallback_to_eager", False):
            logger.warning("torch.compile setup failed; continuing with eager model: %s", e)
            return module
        raise


def compile_transformer_inner_forwards(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
    disable_linear: bool,
) -> torch.nn.Module:
    targets = _collect_compile_targets(target_blocks)
    if len(targets) == 0:
        logger.warning("Inner-block torch.compile was requested, but no compile target blocks were found")
        return transformer

    if disable_linear:
        logger.info("Disable linear from torch.compile for swap blocks...")
        for _, _, block in targets:
            disable_linear_from_compile(block)

    inner_forwards: list[tuple[torch.nn.Module, Callable]] = []
    for _, i, block in targets:
        inner_forward = getattr(block, "_forward", None)
        if not callable(inner_forward):
            raise TypeError(f"torch.compile target block at index {i} has no callable _forward")
        inner_forwards.append((block, inner_forward))

    _apply_inductor_config(args)
    compile_dynamic = compile_dynamic_arg(args)
    logger.info(
        "Compiling DiT block compute with eager wrappers: blocks=%d, backend=%s, mode=%s, dynamic=%s, fullgraph=%s",
        len(inner_forwards),
        args.compile_backend,
        args.compile_mode,
        compile_dynamic,
        args.compile_fullgraph,
    )
    _configure_compile_cache(args, len(inner_forwards))

    compiled_refs: list[tuple[torch.nn.Module, Callable]] = []
    try:
        for block, inner_forward in inner_forwards:
            block._forward = torch.compile(
                inner_forward,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
            compiled_refs.append((block, inner_forward))
    except Exception as e:
        for block, inner_forward in reversed(compiled_refs):
            block._forward = inner_forward
        if getattr(args, "compile_fallback_to_eager", False):
            logger.warning("Inner-block torch.compile setup failed; restored eager compute and continuing: %s", e)
            return transformer
        raise

    return transformer
