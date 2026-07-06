"""
W8A8 activation quantization for reducing VRAM during LoRA training.

Uses custom autograd.Functions that save only references to quantized weight
buffers instead of full-size dequantized weights, eliminating ~32MB per linear
layer from the autograd graph.

Two modes:
  - int8 (default): Converts FP8 weights to row-wise int8, uses torch._int_mm.
    Requires SM 7.5+ (Turing: RTX 2080+, T4, A100, etc.)
  - fp8: Keeps FP8 weights as-is, uses transient dequantization.
    Works on any GPU that supports FP8 (same as --fp8_scaled).
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules.int8_convrot_utils import rotate_activation_by_groupsize

logger = logging.getLogger(__name__)

SMALL_BATCH_THRESHOLD = 16
_INT8_CONVERSION_CHUNK_ELEMENTS = 4 * 1024 * 1024
_INT_MM_MIN_CUDA_CAPABILITY = (7, 5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dequantize_weight(weight, scale_weight):
    """Dequantize quantized weight to float32, handling all scale shapes."""
    if scale_weight.ndim < 3:
        # per-tensor [1] or per-channel [out, 1]
        return weight.float() * scale_weight.float()
    else:
        # block-wise [out, num_blocks, 1]
        out_features, num_blocks, _ = scale_weight.shape
        w = weight.float().contiguous().view(out_features, num_blocks, -1)
        w = w * scale_weight.float()
        return w.view(weight.shape)


def _iter_weight_row_slices(weight, *, max_chunk_elements: int | None = None):
    """Yield row slices capped by element count to bound transient fp32 memory."""
    if max_chunk_elements is None:
        max_chunk_elements = _INT8_CONVERSION_CHUNK_ELEMENTS
    if weight.ndim != 2:
        yield slice(0, weight.shape[0])
        return

    out_features, in_features = weight.shape
    rows_per_chunk = max(1, max_chunk_elements // max(int(in_features), 1))
    for start in range(0, out_features, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, out_features))


def _slice_scale_weight_for_rows(scale_weight, row_slice):
    if scale_weight.ndim == 0:
        return scale_weight
    if scale_weight.ndim == 1:
        return scale_weight
    if scale_weight.shape[0] == 1:
        return scale_weight
    return scale_weight[row_slice]


def _dequantize_weight_chunk(weight_chunk, scale_weight_chunk):
    """Dequantize one row chunk to float32 without materializing the whole weight."""
    chunk = weight_chunk.to(dtype=torch.float32, copy=True)
    if scale_weight_chunk.ndim < 3:
        chunk.mul_(scale_weight_chunk.float())
        return chunk

    out_features, num_blocks, _ = scale_weight_chunk.shape
    if not chunk.is_contiguous():
        chunk = chunk.contiguous()
    chunk = chunk.view(out_features, num_blocks, -1)
    chunk.mul_(scale_weight_chunk.float())
    return chunk.view(weight_chunk.shape)


def _dequantized_weight_abs_max_chunked(weight, scale_weight):
    abs_max = torch.zeros((), device=weight.device, dtype=torch.float32)
    for row_slice in _iter_weight_row_slices(weight):
        scale_chunk = _slice_scale_weight_for_rows(scale_weight, row_slice)
        chunk = _dequantize_weight_chunk(weight[row_slice], scale_chunk)
        if chunk.numel() == 0:
            continue
        abs_max = torch.maximum(abs_max, chunk.abs_().max())
    return abs_max


def _dequantized_weight_row_abs_max_chunked(weight, scale_weight):
    if weight.ndim != 2:
        return _dequantized_weight_abs_max_chunked(weight, scale_weight).reshape(1)

    row_abs_max = torch.empty((weight.shape[0], 1), device=weight.device, dtype=torch.float32)
    for row_slice in _iter_weight_row_slices(weight):
        scale_chunk = _slice_scale_weight_for_rows(scale_weight, row_slice)
        chunk = _dequantize_weight_chunk(weight[row_slice], scale_chunk)
        if chunk.numel() == 0:
            continue
        row_abs_max[row_slice] = chunk.abs_().amax(dim=1, keepdim=True)
    return row_abs_max


def _output_scale_view(weight_scale):
    scale = weight_scale.float()
    if scale.numel() == 1:
        return scale.reshape(1, 1)
    if scale.ndim == 1:
        return scale.reshape(1, -1)
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale.transpose(0, 1)
    raise ValueError(f"Expected scalar or row-wise W8A8 int8 weight scale, got shape {tuple(weight_scale.shape)}")


def _slice_int8_scale_for_rows(int8_scale, row_slice):
    if int8_scale.numel() == 1:
        return int8_scale
    if int8_scale.ndim == 1:
        return int8_scale[row_slice].reshape(-1, 1)
    if int8_scale.shape[0] == 1:
        return int8_scale
    return int8_scale[row_slice]


def _quantize_dequantized_weight_to_int8_chunked(weight, scale_weight, int8_scale):
    weight_int8 = torch.empty(weight.shape, device=weight.device, dtype=torch.int8)
    for row_slice in _iter_weight_row_slices(weight):
        scale_chunk = _slice_scale_weight_for_rows(scale_weight, row_slice)
        int8_scale_chunk = _slice_int8_scale_for_rows(int8_scale, row_slice)
        chunk = _dequantize_weight_chunk(weight[row_slice], scale_chunk)
        chunk.div_(int8_scale_chunk)
        chunk.round_()
        chunk.clamp_(min=-127, max=127)
        weight_int8[row_slice].copy_(chunk.to(torch.int8))
    return weight_int8


def _quantize_int8_per_token(x):
    """Per-token int8 quantization.

    Args:
        x: float tensor [M, K]
    Returns:
        (int8 tensor [M, K], float32 scale [M, 1])
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30).float()
    quantized = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _int8_convrot_groupsize_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _module_int8_convrot_groupsize(module) -> int:
    return _int8_convrot_groupsize_value(getattr(module, "int8_convrot_groupsize", None))


def _has_prequantized_int8_convrot_linears(model) -> bool:
    for module in model.modules():
        if (
            isinstance(module, nn.Linear)
            and hasattr(module, "scale_weight")
            and module.weight.dtype == torch.int8
            and _module_int8_convrot_groupsize(module) > 0
        ):
            return True
    return False


def _linear_dequant_fallback(module, x, convrot_groupsize: int = 0):
    w = _dequantize_weight(module.weight.detach(), module.scale_weight).to(x.dtype)
    x_linear = rotate_activation_by_groupsize(x, convrot_groupsize) if convrot_groupsize > 0 else x
    return F.linear(x_linear, w, module.bias)


def _int_mm_allow_small_m(a, b):
    """torch._int_mm requires M > 16 on CUDA; pad tiny batches instead of dequantizing weights."""
    m = a.shape[0]
    if m > SMALL_BATCH_THRESHOLD:
        return torch._int_mm(a, b)

    padded_m = SMALL_BATCH_THRESHOLD + 1
    padding = torch.zeros((padded_m - m, a.shape[1]), device=a.device, dtype=a.dtype)
    padded = torch.cat((a, padding), dim=0)
    return torch._int_mm(padded, b)[:m]


def _supports_int_mm(device):
    """Return whether torch._int_mm can be used for tensors on the requested device."""
    if not hasattr(torch, "_int_mm"):
        return False
    device = torch.device(device)
    if device.type == "cpu":
        return True
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= _INT_MM_MIN_CUDA_CAPABILITY


def _get_triton_mm_8bit():
    """Import the Triton row-major int8 matmul lazily."""
    try:
        from musubi_tuner.modules.triton_mm_8bit import mm_8bit
    except ImportError:
        return None
    return mm_8bit


def _supports_triton_mm(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.is_available()


_INT8_EPILOGUE = "unset"
_INT8_CUDA_QUANTIZE = "unset"
_INT8_CUDA_EPILOGUE = "unset"
_INT8_CUDA_CONVROT_EPILOGUE = "unset"
_CUTLASS_MM_8BIT = "unset"
_INT8_BACKENDS = {"torch", "triton", "cutlass"}


def _get_int8_epilogue():
    """Lazily import the fused Triton dequant epilogue (None if Triton absent)."""
    global _INT8_EPILOGUE
    if _INT8_EPILOGUE == "unset":
        try:
            from musubi_tuner.modules.triton_int8_epilogue import dequant_epilogue, have_triton

            _INT8_EPILOGUE = dequant_epilogue if have_triton() else None
        except ImportError:
            _INT8_EPILOGUE = None
    return _INT8_EPILOGUE


def _get_int8_cuda_quantize():
    """Lazily import native CUDA row-wise quantization (None if unavailable)."""
    global _INT8_CUDA_QUANTIZE
    if _INT8_CUDA_QUANTIZE == "unset":
        try:
            from musubi_tuner.modules.cuda_int8_convrot import is_available, quantize_rowwise

            _INT8_CUDA_QUANTIZE = quantize_rowwise if is_available() else None
        except Exception:
            _INT8_CUDA_QUANTIZE = None
    return _INT8_CUDA_QUANTIZE


def _get_int8_cuda_epilogue():
    """Lazily import native CUDA dequant epilogue (None if unavailable)."""
    global _INT8_CUDA_EPILOGUE
    if _INT8_CUDA_EPILOGUE == "unset":
        try:
            from musubi_tuner.modules.cuda_int8_convrot import dequant_epilogue, is_available

            _INT8_CUDA_EPILOGUE = dequant_epilogue if is_available() else None
        except Exception:
            _INT8_CUDA_EPILOGUE = None
    return _INT8_CUDA_EPILOGUE


def _get_int8_cuda_convrot_epilogue():
    """Lazily import native CUDA ConvRot dequant epilogue (None if unavailable)."""
    global _INT8_CUDA_CONVROT_EPILOGUE
    if _INT8_CUDA_CONVROT_EPILOGUE == "unset":
        try:
            from musubi_tuner.modules.cuda_int8_convrot import dequant_epilogue_convrot, is_available

            _INT8_CUDA_CONVROT_EPILOGUE = dequant_epilogue_convrot if is_available() else None
        except Exception:
            _INT8_CUDA_CONVROT_EPILOGUE = None
    return _INT8_CUDA_CONVROT_EPILOGUE


def _get_cutlass_mm_8bit():
    """Import and build the optional CUTLASS int8 matmul lazily."""
    global _CUTLASS_MM_8BIT
    if _CUTLASS_MM_8BIT == "unset":
        try:
            from musubi_tuner.modules.cutlass_int8 import int8_mm, is_available

            _CUTLASS_MM_8BIT = int8_mm if is_available() else None
        except Exception as exc:
            logger.info("W8A8 CUTLASS int8 backend unavailable: %s", exc)
            _CUTLASS_MM_8BIT = None
    return _CUTLASS_MM_8BIT


def _torch_dtype_code_supported(dtype: torch.dtype) -> bool:
    return dtype in {torch.float16, torch.bfloat16, torch.float32}


def _autocast_output_dtype(x: torch.Tensor) -> torch.dtype:
    if not x.is_floating_point():
        return x.dtype
    device_type = x.device.type
    try:
        autocast_enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        autocast_enabled = torch.is_autocast_enabled()
    if not autocast_enabled:
        return x.dtype
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, RuntimeError):
        if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
            return torch.get_autocast_gpu_dtype()
        if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
            return torch.get_autocast_cpu_dtype()
    return x.dtype


def _resolve_int8_backend(w8a8_backend: str, *, use_int_mm: bool):
    """Resolve an explicitly selected int8 matmul backend.

    Acceleration backends are opt-in. If the selected backend is not usable,
    fail early instead of silently picking a different implementation.
    """
    if w8a8_backend not in _INT8_BACKENDS:
        expected = "', '".join(sorted(_INT8_BACKENDS))
        raise ValueError(f"Unsupported W8A8 backend: {w8a8_backend!r}. Expected '{expected}'.")
    if not use_int_mm:
        if w8a8_backend != "torch":
            raise RuntimeError(f"W8A8 backend {w8a8_backend!r} requires torch._int_mm support.")
        return None, None
    if w8a8_backend == "torch":
        return None, None
    if w8a8_backend == "triton":
        triton_mm = _get_triton_mm_8bit()
        if triton_mm is None:
            raise RuntimeError("W8A8 backend 'triton' was selected, but the Triton int8 matmul is unavailable.")
        return triton_mm, None
    cutlass_mm = _get_cutlass_mm_8bit()
    if cutlass_mm is None:
        raise RuntimeError(
            "W8A8 backend 'cutlass' was selected, but the CUTLASS extension could not be built or loaded. "
            "Set LTX2_CUTLASS_INCLUDE_DIR to a CUTLASS include tree."
        )
    return None, cutlass_mm


def _resolve_legacy_int8_backend(*, use_int_mm: bool):
    """Default W8A8 int8 backend used when no --w8a8_backend selector is set."""
    if not use_int_mm:
        return None, None
    return _get_triton_mm_8bit(), None


_INT8_QUANTIZE = "unset"
_INT8_CONVROT_QUANTIZE = "unset"
_INT8_CONVROT_EPILOGUE = "unset"
_USE_FUSED_QUANT = None
_USE_CUDA_CONVROT_QUANT = None
_TRITON_CONVROT_FUSED_BROKEN = False


def _mark_triton_convrot_broken():
    """Disable the Triton ConvRot fused kernels for the rest of the run after a compile failure.

    Some Triton builds fail to compile the ConvRot butterfly for bf16 inputs; when that happens we
    fall back to the eager rotate+quant path (identical math) instead of crashing training.
    """
    global _TRITON_CONVROT_FUSED_BROKEN
    if not _TRITON_CONVROT_FUSED_BROKEN:
        _TRITON_CONVROT_FUSED_BROKEN = True
        logger.warning(
            "Triton ConvRot fused kernel failed to compile; using the eager rotate+quant path for the rest of this run "
            "(build the CUDA ConvRot extension to keep the fused path)."
        )


def _get_int8_quantize():
    """Lazily import the fused Triton per-token quant (None if Triton absent)."""
    global _INT8_QUANTIZE
    if _INT8_QUANTIZE == "unset":
        try:
            from musubi_tuner.modules.triton_int8_epilogue import have_triton, quantize_rowwise

            _INT8_QUANTIZE = quantize_rowwise if have_triton() else None
        except ImportError:
            _INT8_QUANTIZE = None
    return _INT8_QUANTIZE


def _get_int8_convrot_quantize():
    """Lazily import the CUDA ConvRot quantizer (None if it cannot be built)."""
    global _INT8_CONVROT_QUANTIZE
    if _INT8_CONVROT_QUANTIZE == "unset":
        try:
            from musubi_tuner.modules.cuda_int8_convrot import is_available, quantize_rowwise_convrot

            _INT8_CONVROT_QUANTIZE = quantize_rowwise_convrot if is_available() else None
        except Exception:
            _INT8_CONVROT_QUANTIZE = None
    return _INT8_CONVROT_QUANTIZE


def _get_int8_convrot_epilogue():
    """Lazily import the fused Triton ConvRot grad-input epilogue."""
    global _INT8_CONVROT_EPILOGUE
    if _INT8_CONVROT_EPILOGUE == "unset":
        try:
            from musubi_tuner.modules.triton_int8_epilogue import dequant_epilogue_convrot, have_triton

            _INT8_CONVROT_EPILOGUE = dequant_epilogue_convrot if have_triton() else None
        except ImportError:
            _INT8_CONVROT_EPILOGUE = None
    return _INT8_CONVROT_EPILOGUE


def _use_fused_quant():
    """Fuse the per-token activation quant, the backward weight-scale fold, and the ConvRot
    rotation into single kernels (Triton when available, else eager fallback). On by default;
    set LTX2_INT8_FUSED_QUANT=0 to force the eager multi-pass path."""
    global _USE_FUSED_QUANT
    if _USE_FUSED_QUANT is None:
        _USE_FUSED_QUANT = os.getenv("LTX2_INT8_FUSED_QUANT", "1").strip().lower() not in ("0", "false", "no", "off")
    return _USE_FUSED_QUANT


def _use_cuda_convrot_quant():
    """Use the native CUDA ConvRot extension for the fused rotate+quant / epilogue instead of Triton.

    Off by default: the fused path is Triton-first, which avoids a first-run nvcc build and needs no
    CUDA toolkit. Set LTX2_INT8_CONVROT_CUDA_QUANT=1 to build and use the CUDA extension, which is
    the best bf16 path (it has no Triton bf16 group-16 compile gap). Falls back cleanly if unbuildable.
    """
    global _USE_CUDA_CONVROT_QUANT
    if _USE_CUDA_CONVROT_QUANT is None:
        _USE_CUDA_CONVROT_QUANT = os.getenv("LTX2_INT8_CONVROT_CUDA_QUANT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return _USE_CUDA_CONVROT_QUANT


# ---------------------------------------------------------------------------
# Custom autograd Functions
# ---------------------------------------------------------------------------


class _W8A8Int8Function(torch.autograd.Function):
    """int8 W8A8: per-token input quantization + torch._int_mm.

    Saves only int8 weight and float32 row-wise scale in the autograd graph
    (both are existing module buffers, so 0 extra bytes).
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight_int8,
        weight_int8_t,
        weight_scale,
        bias,
        triton_mm,
        cutlass_mm,
        use_triton_scaled_forward,
        convrot_groupsize,
    ):
        # weight_int8: [out, in] int8
        # weight_int8_t: optional [in, out] contiguous int8 fallback for fast backward RHS layout
        # weight_scale: [out, 1] float32
        original_shape = x.shape
        output_dtype = _autocast_output_dtype(x)
        convrot_groupsize = int(convrot_groupsize or 0)
        quantize = _get_int8_quantize()
        convrot_quantize = (
            _get_int8_convrot_quantize()
            if _use_fused_quant() and _use_cuda_convrot_quant() and x.is_cuda and convrot_groupsize in (16, 64, 256)
            else None
        )

        use_fused_quant = quantize is not None and _use_fused_quant() and x.is_cuda
        use_triton_convrot = (
            use_fused_quant and convrot_groupsize <= 256 and not (convrot_groupsize > 0 and _TRITON_CONVROT_FUSED_BROKEN)
        )
        if convrot_quantize is not None and convrot_groupsize > 0:
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            x_int8, x_scale = convrot_quantize(x_2d, convrot_groupsize)
        elif use_triton_convrot:
            x_2d = x.reshape(-1, x.shape[-1])
            try:
                x_int8, x_scale = quantize(x_2d, None, convrot_groupsize=convrot_groupsize)
            except Exception:
                if convrot_groupsize <= 0:
                    raise  # plain fused quant failure is not a ConvRot fallback case
                _mark_triton_convrot_broken()
                x_for_quant = rotate_activation_by_groupsize(x, convrot_groupsize)
                x_int8, x_scale = _quantize_int8_per_token(x_for_quant.reshape(-1, x_for_quant.shape[-1]))
        else:
            x_for_quant = rotate_activation_by_groupsize(x, convrot_groupsize) if convrot_groupsize > 0 else x
            x_2d = x_for_quant.reshape(-1, x_for_quant.shape[-1])
            x_int8, x_scale = _quantize_int8_per_token(x_2d)

        # [M, K] @ [K, N] -> [M, N] int32   (N = out_features)
        # _int_mm handles transposed second operand (column-major) natively via cuBLAS;
        # no .contiguous() needed on weight_int8.t(), which avoids a 16MB transient copy.
        out_int32 = None
        triton_scaled = getattr(triton_mm, "scaled", None) if use_triton_scaled_forward and triton_mm is not None else None
        if triton_scaled is not None and _torch_dtype_code_supported(output_dtype):
            output = triton_scaled(
                x_int8.contiguous(),
                weight_int8.t(),
                x_scale.contiguous(),
                weight_scale.reshape(-1).contiguous(),
                bias.float().contiguous() if bias is not None else None,
                output_dtype=output_dtype,
            )
            output = output.reshape(*original_shape[:-1], weight_int8.shape[0])
        elif cutlass_mm is not None and weight_int8_t is not None:
            out_int32 = cutlass_mm(x_int8.contiguous(), weight_int8)
        elif triton_mm is not None:
            out_int32 = triton_mm(x_int8.contiguous(), weight_int8.t())
        else:
            out_int32 = _int_mm_allow_small_m(x_int8.contiguous(), weight_int8.t())

        if out_int32 is not None:
            # Rescale int32 -> compute dtype. Fused Triton epilogue (one pass over [M,N])
            # when available and weight scale is per-row; else eager multi-pass rescale.
            out_features = weight_int8.shape[0]
            epilogue = (
                _get_int8_cuda_epilogue()
                if _use_fused_quant()
                and _use_cuda_convrot_quant()
                and out_int32.is_cuda
                and _torch_dtype_code_supported(output_dtype)
                else None
            )
            if epilogue is None:
                epilogue = _get_int8_epilogue() if out_int32.is_cuda else None
            if epilogue is not None and weight_scale.numel() == out_features:
                output = epilogue(out_int32, x_scale, weight_scale, bias.float() if bias is not None else None, output_dtype)
                output = output.reshape(*original_shape[:-1], out_features)
            else:
                output = out_int32.float() * x_scale * _output_scale_view(weight_scale)
                if bias is not None:
                    output = output + bias.float()
                output = output.to(output_dtype).reshape(*original_shape[:-1], -1)

        # Save ONLY references to existing module buffers.
        if cutlass_mm is not None:
            ctx.save_for_backward(weight_int8_t, weight_scale)
        else:
            # torch._int_mm and Triton both take the [out, in] weight directly in the backward
            # (torch._int_mm accepts a row-major RHS), so no transposed copy is saved.
            ctx.save_for_backward(weight_int8, weight_scale)
        ctx.input_dtype = x.dtype
        ctx.triton_mm = triton_mm
        ctx.cutlass_mm = cutlass_mm
        ctx.convrot_groupsize = convrot_groupsize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Only activation gradients are expected (LoRA-only training).
        # needs_input_grad = (x=True, weight=False, weight_t=False, scale=False, bias=False, triton_mm=False)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise RuntimeError(
                "W8A8 int8 does not support weight gradients (full fine-tuning). Use --network_module for LoRA training."
            )

        weight_saved, weight_scale = ctx.saved_tensors
        go = grad_output.reshape(-1, grad_output.shape[-1])
        out_features = grad_output.shape[-1]

        # Row-wise weight scales live on the contraction dimension for grad_input,
        # so fold them into grad_output before the int8 matmul.
        quantize = _get_int8_quantize()
        cuda_quantize = (
            _get_int8_cuda_quantize()
            if _use_fused_quant() and _use_cuda_convrot_quant() and go.is_cuda and weight_scale.numel() == out_features
            else None
        )
        use_fused_quant = quantize is not None and _use_fused_quant() and go.is_cuda
        if cuda_quantize is not None:
            grad_int8, grad_scale = cuda_quantize(go.contiguous(), weight_scale.reshape(out_features).contiguous())
        elif use_fused_quant and weight_scale.numel() == out_features:
            grad_int8, grad_scale = quantize(go, weight_scale)  # fused fold + per-token quant
        else:
            go_2d = go.float() * _output_scale_view(weight_scale)
            grad_int8, grad_scale = _quantize_int8_per_token(go_2d)
        convrot_groupsize = int(ctx.convrot_groupsize or 0)
        convrot_epilogue = None
        if _use_fused_quant() and grad_output.is_cuda and convrot_groupsize in (16, 64, 256):
            convrot_epilogue = _get_int8_cuda_convrot_epilogue() if _use_cuda_convrot_quant() else None
            if convrot_epilogue is None and not _TRITON_CONVROT_FUSED_BROKEN:
                convrot_epilogue = _get_int8_convrot_epilogue()
        convrot_applied = False
        if ctx.cutlass_mm is not None:
            grad_int32 = ctx.cutlass_mm(grad_int8.contiguous(), weight_saved)
            in_features = weight_saved.shape[0]
            epilogue = (
                _get_int8_cuda_epilogue()
                if _use_fused_quant()
                and _use_cuda_convrot_quant()
                and grad_int32.is_cuda
                and _torch_dtype_code_supported(ctx.input_dtype)
                else None
            )
            if epilogue is None:
                epilogue = _get_int8_epilogue() if grad_int32.is_cuda else None
            if convrot_epilogue is not None:
                grad_input = convrot_epilogue(grad_int32, grad_scale, convrot_groupsize, ctx.input_dtype)
                convrot_applied = True
            elif epilogue is not None:
                grad_input = epilogue(grad_int32, grad_scale, None, None, ctx.input_dtype)
            else:
                grad_input = (grad_int32.float() * grad_scale).to(ctx.input_dtype)
        elif ctx.triton_mm is None:
            # weight_saved is the [out, in] int8 weight; torch._int_mm takes it as a row-major RHS.
            grad_int32 = _int_mm_allow_small_m(grad_int8.contiguous(), weight_saved)
            in_features = weight_saved.shape[1]
            epilogue = (
                _get_int8_cuda_epilogue()
                if _use_fused_quant()
                and _use_cuda_convrot_quant()
                and grad_int32.is_cuda
                and _torch_dtype_code_supported(ctx.input_dtype)
                else None
            )
            if epilogue is None:
                epilogue = _get_int8_epilogue() if grad_int32.is_cuda else None
            if convrot_epilogue is not None:
                grad_input = convrot_epilogue(grad_int32, grad_scale, convrot_groupsize, ctx.input_dtype)
                convrot_applied = True
            elif epilogue is not None:
                grad_input = epilogue(grad_int32, grad_scale, None, None, ctx.input_dtype)
            else:
                grad_input = (grad_int32.float() * grad_scale).to(ctx.input_dtype)
        else:
            grad_int32 = ctx.triton_mm(grad_int8.contiguous(), weight_saved)
            in_features = weight_saved.shape[1]
            epilogue = (
                _get_int8_cuda_epilogue()
                if _use_fused_quant()
                and _use_cuda_convrot_quant()
                and grad_int32.is_cuda
                and _torch_dtype_code_supported(ctx.input_dtype)
                else None
            )
            if epilogue is None:
                epilogue = _get_int8_epilogue() if grad_int32.is_cuda else None
            if convrot_epilogue is not None:
                grad_input = convrot_epilogue(grad_int32, grad_scale, convrot_groupsize, ctx.input_dtype)
                convrot_applied = True
            elif epilogue is not None:
                grad_input = epilogue(grad_int32, grad_scale, None, None, ctx.input_dtype)
            else:
                grad_input = (grad_int32.float() * grad_scale).to(ctx.input_dtype)
        if convrot_groupsize > 0 and not convrot_applied:
            grad_input = rotate_activation_by_groupsize(grad_input, convrot_groupsize, inverse=True)
        grad_input = grad_input.reshape(*grad_output.shape[:-1], in_features)
        return grad_input, None, None, None, None, None, None, None, None


class _W8A8TransientDequantFunction(torch.autograd.Function):
    """Generic W8A8: dequantize transiently, don't save dequantized weight.

    Works with any quantized format (int8 per-channel, FP8 per-tensor/channel/block).
    Uses standard F.linear for the forward matmul.
    """

    @staticmethod
    def forward(ctx, x, weight, scale_weight, bias, convrot_groupsize=0):
        convrot_groupsize = int(convrot_groupsize or 0)
        weight_deq = _dequantize_weight(weight, scale_weight).to(x.dtype)
        x_linear = rotate_activation_by_groupsize(x, convrot_groupsize) if convrot_groupsize > 0 else x
        output = F.linear(x_linear, weight_deq, bias)
        # weight_deq is NOT saved; it is freed when this scope ends.
        ctx.save_for_backward(weight, scale_weight)
        ctx.input_dtype = x.dtype
        ctx.convrot_groupsize = convrot_groupsize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1]:
            raise RuntimeError("W8A8 does not support weight gradients (full fine-tuning). Use --network_module for LoRA training.")

        weight, scale_weight = ctx.saved_tensors
        weight_deq = _dequantize_weight(weight, scale_weight)  # transient

        go_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_input = go_2d @ weight_deq  # [M, out] @ [out, in]
        if ctx.convrot_groupsize > 0:
            grad_input = rotate_activation_by_groupsize(grad_input, ctx.convrot_groupsize, inverse=True)
        grad_input = grad_input.to(ctx.input_dtype).reshape(*grad_output.shape[:-1], weight_deq.shape[1])
        return grad_input, None, None, None, None


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def _convert_fp8_to_int8_weights(module, *, keep_transpose_buffer: bool):
    """One-time conversion: FP8 weight + scale -> int8 weight + row-wise scale.

    After conversion:
      module.weight.data  = int8 [out, in]
      module.scale_weight = float32 [out, 1]
    """
    with torch.no_grad():
        weight = module.weight.detach()
        scale_weight = module.scale_weight.detach()
        abs_max = _dequantized_weight_row_abs_max_chunked(weight, scale_weight)
        scale = (abs_max / 127.0).clamp(min=1e-30).float()
        weight_int8 = _quantize_dequantized_weight_to_int8_chunked(weight, scale_weight, scale)

        module.weight.requires_grad_(False)
        module.weight = torch.nn.Parameter(weight_int8, requires_grad=False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.scale_weight = scale.reshape(weight_int8.shape[0], 1).to(device=weight_int8.device, dtype=torch.float32)
        if keep_transpose_buffer:
            weight_int8_t = weight_int8.t().contiguous()
            if "weight_int8_t" in module._buffers:
                module.weight_int8_t = weight_int8_t
            else:
                module.register_buffer("weight_int8_t", weight_int8_t, persistent=False)
        elif "weight_int8_t" in module._buffers:
            del module._buffers["weight_int8_t"]


# ---------------------------------------------------------------------------
# Monkey-patched forward functions
# ---------------------------------------------------------------------------


def _make_w8a8_int8_forward(use_int_mm, triton_mm, cutlass_mm=None, use_triton_scaled_forward=False):
    """Create W8A8 int8 forward with small-batch fallback."""
    if use_int_mm:

        def forward(self, x):
            convrot_groupsize = _module_int8_convrot_groupsize(self)
            if not _supports_int_mm(x.device):
                return _W8A8TransientDequantFunction.apply(x, self.weight, self.scale_weight, self.bias, convrot_groupsize)
            if cutlass_mm is not None and hasattr(self, "weight_int8_t"):
                weight_int8_t = self.weight_int8_t
                triton_mm_for_device = None
                cutlass_mm_for_device = cutlass_mm
            elif triton_mm is not None and _supports_triton_mm(x.device):
                weight_int8_t = None
                triton_mm_for_device = triton_mm
                cutlass_mm_for_device = None
            else:
                # torch._int_mm: forward uses weight.t(), backward uses the [out, in] weight directly;
                # no transposed buffer is needed.
                weight_int8_t = None
                triton_mm_for_device = None
                cutlass_mm_for_device = None
            return _W8A8Int8Function.apply(
                x,
                self.weight,
                weight_int8_t,
                self.scale_weight,
                self.bias,
                triton_mm_for_device,
                cutlass_mm_for_device,
                use_triton_scaled_forward,
                convrot_groupsize,
            )
    else:
        # _int_mm unavailable: still save VRAM via custom autograd
        def forward(self, x):
            convrot_groupsize = _module_int8_convrot_groupsize(self)
            num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
            if num_tokens <= SMALL_BATCH_THRESHOLD:
                return _linear_dequant_fallback(self, x, convrot_groupsize)
            return _W8A8TransientDequantFunction.apply(x, self.weight, self.scale_weight, self.bias, convrot_groupsize)

    return forward


def _make_w8a8_fp8_forward():
    """Create W8A8 fp8 forward with small-batch fallback."""

    def forward(self, x):
        num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
        if num_tokens <= SMALL_BATCH_THRESHOLD:
            w = _dequantize_weight(self.weight.detach(), self.scale_weight).to(x.dtype)
            return F.linear(x, w, self.bias)
        return _W8A8TransientDequantFunction.apply(x, self.weight, self.scale_weight, self.bias, 0)

    return forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_w8a8_monkey_patch(
    model, w8a8_mode="int8", state_dict: dict[str, torch.Tensor] | None = None, w8a8_backend: str | None = None
):
    """Replace forward methods on FP8-patched linears with W8A8 versions.

    Must be called AFTER apply_fp8_monkey_patch + load_state_dict
    (needs scale_weight buffers and loaded FP8 weights).

    For int8 mode: also converts FP8 weights to row-wise int8.

    Args:
        model: nn.Module with FP8-patched linear layers
        w8a8_mode: "int8" (default, SM 7.5+) or "fp8" (any GPU)
        state_dict: Optional loaded state dict to update in-place after int8
            conversion. With load_state_dict(assign=True), this releases the
            old FP8 weight entries instead of keeping both FP8 and int8 copies.
        w8a8_backend: optional int8 matmul backend: "torch", "triton", or
            "cutlass". None uses the default backend selection. Non-None
            selections fail early when unavailable.
    """
    if w8a8_mode not in {"int8", "fp8"}:
        raise ValueError(f"Unsupported W8A8 mode: {w8a8_mode!r}. Expected 'int8' or 'fp8'.")

    use_int_mm = hasattr(torch, "_int_mm")
    triton_mm = None
    cutlass_mm = None
    use_triton_scaled_forward = False
    if w8a8_mode == "int8":
        if w8a8_backend is None:
            triton_mm, cutlass_mm = _resolve_legacy_int8_backend(use_int_mm=use_int_mm)
        else:
            triton_mm, cutlass_mm = _resolve_int8_backend(w8a8_backend, use_int_mm=use_int_mm)
            use_triton_scaled_forward = w8a8_backend == "triton"
    elif w8a8_backend is not None and w8a8_backend not in _INT8_BACKENDS:
        expected = "', '".join(sorted(_INT8_BACKENDS))
        raise ValueError(f"Unsupported W8A8 backend: {w8a8_backend!r}. Expected '{expected}'.")
    if w8a8_mode == "int8" and not use_int_mm:
        logger.warning(
            "torch._int_mm not available (requires PyTorch 2.1+); W8A8 int8 will use dequantize+matmul fallback (still saves VRAM)."
        )
    if w8a8_mode == "int8" and use_int_mm and triton_mm is None and cutlass_mm is None:
        logger.info("W8A8 int8: Triton row-major matmul unavailable; keeping transpose buffers for fast backward.")

    # Build the forward function once (shared by all patched layers)
    if w8a8_mode == "int8":
        new_forward = _make_w8a8_int8_forward(
            use_int_mm, triton_mm, cutlass_mm, use_triton_scaled_forward=use_triton_scaled_forward
        )
    else:
        new_forward = _make_w8a8_fp8_forward()

    patched_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "scale_weight"):
            continue

        if w8a8_mode == "int8":
            _convert_fp8_to_int8_weights(module, keep_transpose_buffer=cutlass_mm is not None)
            if state_dict is not None:
                weight_key = f"{name}.weight"
                scale_key = f"{name}.scale_weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = module.weight.detach()
                if scale_key in state_dict:
                    state_dict[scale_key] = module.scale_weight.detach()

        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    backend = "cutlass" if cutlass_mm is not None else "triton" if triton_mm is not None else "torch"
    logger.info("W8A8 %s (%s): patched %d linear layers", w8a8_mode, backend, patched_count)
    return model


# ---------------------------------------------------------------------------
# Pre-quantized Optimum-Quanto int8 checkpoints
# ---------------------------------------------------------------------------


def register_quanto_int8_scale_buffers(model, state_dict):
    """Register scale_weight buffers on the Linear layers that carry a
    pre-quantized int8 weight in ``state_dict``.

    Call BEFORE ``load_state_dict(assign=True)`` so both the int8 weight and its
    per-row scale get assigned (mirrors apply_fp8_monkey_patch's registration).
    """
    scale_shapes = {k.rsplit(".scale_weight", 1)[0]: state_dict[k].shape for k in state_dict if k.endswith(".scale_weight")}
    convrot_groups = {
        k.rsplit(".int8_convrot_groupsize", 1)[0]: _int8_convrot_groupsize_value(state_dict[k])
        for k in state_dict
        if k.endswith(".int8_convrot_groupsize")
    }
    registered = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in scale_shapes:
            module.register_buffer("scale_weight", torch.ones(scale_shapes[name], dtype=torch.float32))
            if name in convrot_groups:
                module.register_buffer("int8_convrot_groupsize", torch.tensor(convrot_groups[name], dtype=torch.int32))
            # int8 weights can't be leaf parameters that require grad; clear the flag
            # so load_state_dict(assign=True) can assign the int8 tensor in place.
            module.weight.requires_grad_(False)
            registered += 1
    logger.info("Quanto/int8: registered scale_weight buffers on %d linear layers", registered)
    return registered


def apply_quanto_int8_monkey_patch(model, w8a8_backend: str | None = None):
    """Bind the int8 W8A8 forward to Linear layers that already hold an int8
    weight + scale_weight buffer (Optimum-Quanto qint8 checkpoints).

    Unlike apply_w8a8_monkey_patch, the weights are already int8, so there is no
    fp8->int8 conversion. When the Triton matmul is unavailable, a transposed
    int8 buffer is built for the fast backward. Call AFTER load_state_dict.
    """
    use_int_mm = hasattr(torch, "_int_mm")
    triton_mm = None
    cutlass_mm = None
    use_triton_scaled_forward = False
    has_convrot = _has_prequantized_int8_convrot_linears(model)
    if w8a8_backend is None and has_convrot and use_int_mm:
        logger.info(
            "Quanto int8 ConvRot: defaulting to torch._int_mm backend for the matmul; pass --w8a8_backend triton "
            "to force the Triton matmul backend."
        )
    elif w8a8_backend is None:
        triton_mm, cutlass_mm = _resolve_legacy_int8_backend(use_int_mm=use_int_mm)
    else:
        triton_mm, cutlass_mm = _resolve_int8_backend(w8a8_backend, use_int_mm=use_int_mm)
        use_triton_scaled_forward = w8a8_backend == "triton"
    if not use_int_mm:
        logger.warning(
            "torch._int_mm unavailable (PyTorch < 2.1); quanto int8 will dequantize+matmul (VRAM saved, no int8 speedup)."
        )
    new_forward = _make_w8a8_int8_forward(use_int_mm, triton_mm, cutlass_mm, use_triton_scaled_forward=use_triton_scaled_forward)

    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or not hasattr(module, "scale_weight"):
            continue
        if module.weight.dtype != torch.int8:
            continue
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.scale_weight = module.scale_weight.reshape(module.weight.shape[0], 1).to(
            device=module.weight.device, dtype=torch.float32
        )
        convrot_groupsize = _module_int8_convrot_groupsize(module)
        if convrot_groupsize > 0 and module.weight.shape[1] % convrot_groupsize != 0:
            raise ValueError(f"INT8 ConvRot group size {convrot_groupsize} does not divide in_features={module.weight.shape[1]}")
        if cutlass_mm is not None and use_int_mm:
            weight_int8_t = module.weight.t().contiguous()
            if "weight_int8_t" in module._buffers:
                module.weight_int8_t = weight_int8_t
            else:
                module.register_buffer("weight_int8_t", weight_int8_t, persistent=False)
        module.forward = new_forward.__get__(module, type(module))
        patched += 1

    backend = (
        "cutlass"
        if cutlass_mm is not None
        else "triton"
        if triton_mm is not None
        else ("torch._int_mm" if use_int_mm else "dequant-fallback")
    )
    logger.info("Quanto int8 (%s): patched %d linear layers", backend, patched)
    return model
