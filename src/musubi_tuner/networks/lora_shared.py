# Shared tensor helpers for LoRA-family network modules.

import math
from typing import Any, Dict, List, Optional, Union

import torch


def _metadata_tensor_to_int(value: Optional[torch.Tensor], default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, torch.Tensor):
        return int(round(float(value.detach().cpu().item())))
    return int(round(float(value)))


def _metadata_tensor_to_bool(value: Optional[torch.Tensor], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        return bool(int(round(float(value.detach().cpu().item()))))
    return bool(value)


def _metadata_tensor_to_float(value: Optional[torch.Tensor], default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _solve_oft_block_size(n_elements: int) -> int:
    return int(round((1.0 + math.sqrt(1.0 + 8.0 * float(n_elements))) / 2.0))


def _parse_bool_network_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_optional_float_network_arg(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    return float(value)


def _effective_weight_dtype(
    weight: torch.Tensor,
    scale_weight: Optional[torch.Tensor],
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    if dtype is not None:
        return dtype
    if isinstance(scale_weight, torch.Tensor) and scale_weight.is_floating_point() and scale_weight.dtype.itemsize >= 2:
        return scale_weight.dtype
    if weight.is_floating_point() and weight.dtype.itemsize >= 2:
        return weight.dtype
    return torch.float32


def _get_scaled_weight(
    weight: torch.Tensor,
    scale_weight: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    detach: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    if detach:
        weight = weight.detach()
        scale_weight = scale_weight.detach()
    target_dtype = _effective_weight_dtype(weight, scale_weight, dtype)
    target_device = device if device is not None else weight.device
    dense_weight = weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    scale = scale_weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    if scale.ndim < 3:
        return dense_weight * scale

    out_features, num_blocks, _ = scale.shape
    dense_weight = dense_weight.contiguous().view(out_features, num_blocks, -1)
    dense_weight = dense_weight * scale
    return dense_weight.view(weight.shape)


def _get_effective_module_weight(
    module: torch.nn.Module,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    detach: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise AttributeError(f"{module.__class__.__name__} has no tensor weight")

    scale_weight = getattr(module, "scale_weight", None)
    if isinstance(scale_weight, torch.Tensor):
        if all(hasattr(module, attr) for attr in ("nf4_out_features", "nf4_in_features", "nf4_block_size")):
            from musubi_tuner.modules.nf4_optimization_utils import dequantize_nf4_block

            if detach:
                weight = weight.detach()
                scale_weight = scale_weight.detach()
            target_dtype = _effective_weight_dtype(weight, scale_weight, dtype)
            target_device = device if device is not None else weight.device
            dense_weight = dequantize_nf4_block(
                weight.to(device=target_device, non_blocking=non_blocking),
                scale_weight.to(device=target_device, non_blocking=non_blocking),
                int(getattr(module, "nf4_out_features")),
                int(getattr(module, "nf4_in_features")),
                int(getattr(module, "nf4_block_size")),
                scale_weight.dtype,
            )
            awq_scales = getattr(module, "awq_scales", None)
            if isinstance(awq_scales, torch.Tensor):
                awq_scales = awq_scales.to(device=dense_weight.device, dtype=dense_weight.dtype, non_blocking=non_blocking)
                dense_weight = dense_weight / awq_scales.unsqueeze(0)
            return dense_weight.to(dtype=target_dtype)

        return _get_scaled_weight(
            weight,
            scale_weight,
            dtype=dtype,
            device=device,
            detach=detach,
            non_blocking=non_blocking,
        )

    if detach:
        weight = weight.detach()
    target_dtype = dtype
    if target_dtype is None and (not weight.is_floating_point() or weight.dtype.itemsize < 2):
        target_dtype = torch.float32
    if target_dtype is None and device is None:
        return weight
    to_kwargs: Dict[str, Any] = {"non_blocking": non_blocking}
    if target_dtype is not None:
        to_kwargs["dtype"] = target_dtype
    if device is not None:
        to_kwargs["device"] = device
    return weight.to(**to_kwargs)


def _get_dora_compute_dtype(weight: torch.Tensor) -> torch.dtype:
    if not weight.is_floating_point() or weight.dtype.itemsize < 2:
        return torch.float32
    if weight.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return weight.dtype


def _get_lora_weight_from_tensors(down_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    if len(down_weight.size()) == 2:
        return up_weight @ down_weight
    if down_weight.size()[2:4] == (1, 1):
        return (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    return torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)


def _get_split_lora_weight(split_dims: List[int], down_weights: List[torch.Tensor], up_weights: List[torch.Tensor]) -> torch.Tensor:
    total_dims = sum(split_dims)
    in_dim = down_weights[0].size(1)
    device = down_weights[0].device
    dtype = down_weights[0].dtype
    weight = torch.zeros((total_dims, in_dim), device=device, dtype=dtype)

    offset = 0
    for split_dim, down_weight, up_weight in zip(split_dims, down_weights, up_weights):
        weight[offset : offset + split_dim] = up_weight @ down_weight
        offset += split_dim

    return weight


def _get_linear_weight_norm_factored(
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    compute_dtype = _get_dora_compute_dtype(base_weight)
    base_weight = base_weight.to(dtype=compute_dtype)
    down_weight = down_weight.to(device=base_weight.device, dtype=compute_dtype)
    up_weight = up_weight.to(device=base_weight.device, dtype=compute_dtype)

    w_norm_sq = (base_weight * base_weight).sum(dim=1)
    if float(scaling) == 0.0:
        return torch.sqrt(w_norm_sq.clamp_min_(0))

    u_term = base_weight @ down_weight.transpose(0, 1)
    gram = down_weight @ down_weight.transpose(0, 1)
    cross_term = (up_weight * u_term).sum(dim=1)
    ba_norm_sq = ((up_weight @ gram) * up_weight).sum(dim=1)
    norm_sq = w_norm_sq + (2.0 * float(scaling)) * cross_term + (float(scaling) * float(scaling)) * ba_norm_sq
    return torch.sqrt(norm_sq.clamp_min_(0))


def _get_conv_weight_norm_factored(
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    out_channels = base_weight.shape[0]
    flat_weight = base_weight.reshape(out_channels, -1)
    flat_down = down_weight.reshape(down_weight.shape[0], -1)
    flat_up = up_weight.reshape(out_channels, -1)
    norms = _get_linear_weight_norm_factored(flat_weight, flat_down, flat_up, scaling)
    return norms.view((1, out_channels) + (1,) * max(0, base_weight.dim() - 2))


def _get_dense_weight_norm(base_weight: torch.Tensor, lora_weight: torch.Tensor, scaling: float) -> torch.Tensor:
    compute_dtype = _get_dora_compute_dtype(base_weight)
    total = base_weight.to(dtype=compute_dtype) + float(scaling) * lora_weight.to(device=base_weight.device, dtype=compute_dtype)
    if total.dim() == 2:
        return torch.linalg.vector_norm(total, dim=1)
    dims = tuple(range(1, total.dim()))
    return total.norm(p=2, dim=dims, keepdim=True).transpose(1, 0)


def _get_dora_weight_norm(
    base_weight: torch.Tensor,
    down_weight: Union[torch.Tensor, List[torch.Tensor]],
    up_weight: Union[torch.Tensor, List[torch.Tensor]],
    scaling: float,
    split_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    if split_dims is not None:
        lora_weight = _get_split_lora_weight(split_dims, down_weight, up_weight)
        return _get_dense_weight_norm(base_weight, lora_weight, scaling)

    if base_weight.dim() == 2:
        return _get_linear_weight_norm_factored(base_weight, down_weight, up_weight, scaling)
    return _get_conv_weight_norm_factored(base_weight, down_weight, up_weight, scaling)


# ------------------------------------------------------------------------------------------------ input-axis DoRA
# DoRA with the magnitude on the INPUT axis (the DoRA paper's column-wise norm; LyCORIS / kohya ``dora_scale``):
#     W' = (W + s*B@A) * m / ||W + s*B@A||_col,   m and the norm of shape (1, in)
# ComfyUI, SwarmUI, Forge and A1111 apply this form exactly from a ``<module>.dora_scale`` key. Training never builds
# W + s*B@A: the forward scales the layer's input channels by f = m / ||.||_col, since W'x = (W + s*B@A)(f * x), and
#     ||col_k||^2 = ||W_k||^2 + 2s * (W^T B)_k . A_k + s^2 * A_k^T (B^T B) A_k
# with ||W_k||^2 computed once (W is frozen) and W^T B built over row chunks of W.

DORA_CHUNK_ELEMENTS = 1 << 24  # a base weight that needs converting is read 16M values at a time
DORA_EPS = float(torch.finfo(torch.float32).eps)  # what ComfyUI adds to the column norm (it patches in fp32)
DORA_CHECK_TOLERANCE = 0.1


def _rotation_group_size(module: torch.nn.Module) -> int:
    """Group size of a ConvRot INT8 layer whose weight is stored in the Hadamard-rotated basis (0: not rotated)."""
    for attr in ("_convrot_groupsize", "int8_convrot_groupsize"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            value = int(value.detach().reshape(-1)[0].item()) if value.numel() else 0
        if value:
            return int(value)
    return 0


class DoRABaseWeight:
    """The frozen weight of a DoRA target, read as (out, in * kernel) rows in its original basis.

    fp8 and INT8 weights are dequantized with their ``scale_weight`` (per tensor, row or block), ConvRot rotations are
    undone and NF4 is dequantized whole. Every read goes through the module: block swap moves blocks by reusing GPU
    storage, so a tensor kept from the forward pass may hold another block's weight by the time backward runs."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        if module.__class__.__name__ in ("Conv2d", "Conv3d"):
            if getattr(module, "groups", 1) != 1:
                raise ValueError("DoRA does not support grouped convolutions")
            self.out_features = int(module.out_channels)
            self.in_features = int(module.in_channels)
            self.kernel = int(math.prod(module.kernel_size))
        else:
            self.out_features = int(module.out_features)
            self.in_features = int(module.in_features)
            self.kernel = 1
        self.columns = self.in_features * self.kernel
        self.dense_only = all(hasattr(module, attr) for attr in ("nf4_out_features", "nf4_in_features", "nf4_block_size"))
        self.group_size = 0 if self.dense_only else _rotation_group_size(module)
        weight = module.weight
        if not self.dense_only and (weight.shape[0] != self.out_features or weight.numel() != self.out_features * self.columns):
            raise ValueError(f"DoRA cannot read a base weight stored as {tuple(weight.shape)} {weight.dtype}")
        # A weight the layer multiplies as stored needs no check; a dequantized one is compared once with the layer.
        self.needs_check = (
            self.dense_only
            or isinstance(getattr(module, "scale_weight", None), torch.Tensor)
            or not (weight.is_floating_point() and weight.dtype.itemsize >= 2)
        )
        # stored as it is multiplied (no scale, rotation or packing): the per-step products read it directly
        self.plain = not self.needs_check and not self.group_size

    def describe(self) -> str:
        weight = self.module.weight
        detail = f"{weight.dtype}, {tuple(weight.shape)}"
        if self.group_size:
            detail += f", ConvRot group {self.group_size}"
        return detail

    def format_key(self) -> tuple:
        """Layers with the same key are read by the same code path (dtype, scale layout, rotation, NF4)."""
        scale = getattr(self.module, "scale_weight", None)
        scale_kind = None
        if isinstance(scale, torch.Tensor):
            scale_kind = "tensor" if scale.numel() == 1 else ("block" if scale.dim() >= 3 else "row")
        return (type(self.module).__name__, str(self.module.weight.dtype), scale_kind, self.group_size, self.dense_only, self.kernel)

    def compute_dtype(self) -> torch.dtype:
        """fp32 weights are multiplied in fp32, everything else in bf16 (the products only perturb the column norm)."""
        return torch.float32 if self.module.weight.dtype == torch.float32 else torch.bfloat16

    def chunks(
        self,
        row_start: int = 0,
        row_end: Optional[int] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        max_elements: int = DORA_CHUNK_ELEMENTS,
    ):
        """Yield (first row, rows) for rows [row_start, row_end), each rows tensor (n, in * kernel) in ``dtype``."""
        module = self.module
        row_end = self.out_features if row_end is None else row_end
        step = max(1, max_elements // max(1, self.columns))
        if self.dense_only:
            dense = _get_effective_module_weight(module, dtype=torch.float32, device=device, detach=True)
            dense = dense.reshape(self.out_features, self.columns)
            for start in range(row_start, row_end, step):
                yield start, dense[start : min(start + step, row_end)].to(dtype)
            return

        weight = module.weight.detach()
        device = weight.device if device is None else torch.device(device)
        scale = getattr(module, "scale_weight", None)
        scale = scale.detach().to(device=device, dtype=torch.float32) if isinstance(scale, torch.Tensor) else None
        hadamard = None
        if self.group_size:
            from musubi_tuner.modules.convrot_int8_kernels import _build_hadamard

            hadamard = _build_hadamard(self.group_size, device=device, dtype=torch.float32)
        for start in range(row_start, row_end, step):
            end = min(start + step, row_end)
            rows = weight[start:end].to(device=device).reshape(end - start, self.columns)
            if scale is None and hadamard is None:
                yield start, rows.to(dtype)
                continue
            rows = rows.to(torch.float32)
            if scale is not None:
                if scale.numel() == 1:
                    rows = rows * scale.reshape(())
                elif scale.dim() >= 3:  # block-wise (out, blocks, 1)
                    part = scale[start:end]
                    rows = (rows.reshape(end - start, part.shape[1], -1) * part).reshape(end - start, self.columns)
                else:  # per row: (out,) or (out, 1)
                    rows = rows * scale.reshape(-1, 1)[start:end]
            if hadamard is not None:  # the regular Hadamard is symmetric and orthonormal: rotating again undoes it
                rows = (rows.reshape(end - start, -1, self.group_size) @ hadamard).reshape(end - start, self.columns)
            yield start, rows.to(dtype)

    def stored_chunks(
        self,
        row_start: int = 0,
        row_end: Optional[int] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_elements: int = DORA_CHUNK_ELEMENTS,
    ):
        """Yield (first row, rows, row_scale) with the weight read as stored: still in the ConvRot basis
        (``rotate_columns`` moves the small products between the bases) and, for a per-tensor or per-row scale, not
        yet scaled (``row_scale``, (n, 1) fp32, multiplies the rows). Block-wise scales and NF4 are dequantized here
        (row_scale None). The per-step DoRA products then read the weight once in its own precision."""
        if self.dense_only:
            for start, rows in self.chunks(row_start, row_end, device=device, dtype=dtype, max_elements=max_elements):
                yield start, rows, None
            return
        weight = self.module.weight.detach()
        device = weight.device if device is None else torch.device(device)
        row_end = self.out_features if row_end is None else row_end
        step = max(1, max_elements // max(1, self.columns))
        scale = getattr(self.module, "scale_weight", None)
        scale = scale.detach().to(device=device, dtype=torch.float32) if isinstance(scale, torch.Tensor) else None
        per_row = scale is not None and (scale.numel() == 1 or scale.dim() < 3)
        if (scale is None or per_row) and weight.dtype == dtype and weight.device == device:
            step = row_end - row_start  # usable as stored: one product over all rows, no copy
        for start in range(row_start, row_end, max(1, step)):
            end = min(start + step, row_end)
            rows = weight[start:end].to(device=device).reshape(end - start, self.columns)
            if scale is None:
                yield start, rows.to(dtype), None
            elif per_row:
                row_scale = scale.reshape(1, 1).expand(end - start, 1) if scale.numel() == 1 else scale.reshape(-1, 1)[start:end]
                yield start, rows.to(dtype), row_scale
            else:  # block-wise (out, blocks, 1)
                part = scale[start:end]
                rows = (rows.to(torch.float32).reshape(end - start, part.shape[1], -1) * part).reshape(end - start, self.columns)
                yield start, rows.to(dtype), None

    def rotate_columns(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the ConvRot rotation along dim 0 of an (in, r) tensor: W^T B = H (W_rot^T B) and W v = W_rot (H v)
        (the regular Hadamard is symmetric and orthonormal). The identity for weights stored unrotated."""
        if not self.group_size:
            return tensor
        from musubi_tuner.modules.convrot_int8_kernels import _build_hadamard

        hadamard = _build_hadamard(self.group_size, device=tensor.device, dtype=tensor.dtype)
        return (hadamard @ tensor.reshape(-1, self.group_size, tensor.shape[1])).reshape(tensor.shape)

    def dense(self, device=None) -> torch.Tensor:
        """The whole weight as (out, in * kernel) fp32 (tests, merges and the one-time check)."""
        return torch.cat([rows for _, rows in self.chunks(device=device)], dim=0)


def dora_column_norm_sq_of(base: DoRABaseWeight, device=None) -> torch.Tensor:
    """||W_k||^2 for every flattened column k of the frozen weight (fp32)."""
    total = None
    for _, rows in base.chunks(device=device):
        part = (rows * rows).sum(dim=0)
        total = part if total is None else total + part
    return total


def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``tensor.to(dtype)`` without dispatching an op when it already has that dtype (eager DoRA is dispatch-bound)."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b as fp32: one kernel on CUDA for 16-bit inputs (cuBLAS accumulates in fp32 anyway), a cast elsewhere."""
    if a.is_cuda and a.dtype in (torch.bfloat16, torch.float16) and b.dtype == a.dtype:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a, b).float()


def _frozen_products(base, up, row_offset, compute_dtype, weighted=None):
    """One pass over the frozen weight rows [row_offset, row_offset + n): returns (W^T B, W v) in fp32, the second only
    when ``weighted`` (v, (in, r), original basis) is given. A per-row scale goes onto B's rows / the output rows and a
    ConvRot rotation onto the small (in, r) side, so the weight is read once as stored."""
    weight = base.module.weight
    if base.plain and weight.dtype == compute_dtype and weight.device == up.device:
        # stored as multiplied (bf16 / fp32): no chunks, scales or rotations, and no op that is not needed (eager
        # training is bound by op dispatch; these products run inside autograd Functions, so nothing is recorded)
        rows = weight if weight.dim() == 2 else weight.reshape(base.out_features, base.columns)
        if row_offset or up.shape[0] != base.out_features:
            rows = rows[row_offset : row_offset + up.shape[0]]
        product = _mm_fp32(rows.t(), up if up.dtype == compute_dtype else up.to(compute_dtype))
        out_rows = None
        if weighted is not None:
            out_rows = _mm_fp32(rows, weighted if weighted.dtype == compute_dtype else weighted.to(compute_dtype))
        return product, out_rows
    product = None
    out_rows = None
    if weighted is not None:
        v = base.rotate_columns(weighted).to(compute_dtype)
        out_rows = torch.empty(up.shape[0], weighted.shape[1], dtype=torch.float32, device=up.device)
    for start, rows, row_scale in base.stored_chunks(row_offset, row_offset + up.shape[0], device=up.device, dtype=compute_dtype):
        local = start - row_offset
        b = up[local : local + rows.shape[0]]
        if row_scale is not None:
            b = b * row_scale
        part = _mm_fp32(rows.t(), b.to(compute_dtype))
        product = part if product is None else product + part
        if out_rows is not None:
            block = _mm_fp32(rows, v)
            if row_scale is not None:
                block = block * row_scale
            out_rows[local : local + rows.shape[0]] = block
    return base.rotate_columns(product), out_rows


def _contribution(base, down, up, row_offset, compute_dtype, scaling):
    """sum_r A[r, k] (2s (W^T B)[k, r] + s^2 ((B^T B) A)[r, k]) for every column k: what one LoRA block adds to
    ||W_k||^2. 2s (W^T B)^T + s^2 (B^T B) A is one addmm: eager training launches every op separately."""
    product, _ = _frozen_products(base, up, row_offset, compute_dtype)
    inner = torch.addmm(product.t(), up.t() @ up, down, beta=2.0 * scaling, alpha=scaling * scaling)
    return (down * inner).sum(dim=0)


def _contribution_grads(base, down, up, row_offset, compute_dtype, scaling, grad, need_down, need_up):
    """dL/dA and dL/dB of ``_contribution`` for dL/d(contribution) = ``grad`` (fp32 (in,)). W^T B is recomputed in
    the same pass over W that gives W (A diag(g))^T: only A and B are kept for backward, so layers trained without
    gradient checkpointing keep no extra (r, in) tensors."""
    weighted = down * grad[None, :]  # A diag(g), (r, in)
    product, w_weighted = _frozen_products(base, up, row_offset, compute_dtype, weighted.t() if need_up else None)
    grad_down = grad_up = None
    if need_down:
        inner = torch.addmm(product.t(), up.t() @ up, down, beta=2.0 * scaling, alpha=2.0 * scaling * scaling)
        grad_down = inner * grad[None, :]
    if need_up:
        grad_up = torch.addmm(w_weighted, up, weighted @ down.t(), beta=2.0 * scaling, alpha=2.0 * scaling * scaling)
    return grad_down, grad_up


class _DoRANormSqContribution(torch.autograd.Function):
    """``_contribution`` as an autograd node that keeps only A and B (``dora_column_norm``)."""

    @staticmethod
    def forward(ctx, down, up, base, row_offset, compute_dtype, scaling):
        ctx.save_for_backward(down, up)
        ctx.base, ctx.row_offset, ctx.compute_dtype, ctx.scaling = base, row_offset, compute_dtype, scaling
        return _contribution(base, down, up, row_offset, compute_dtype, scaling)

    @staticmethod
    def backward(ctx, grad):
        down, up = ctx.saved_tensors
        grad_down, grad_up = _contribution_grads(
            ctx.base, down, up, ctx.row_offset, ctx.compute_dtype, ctx.scaling, grad.to(torch.float32),
            ctx.needs_input_grad[0], ctx.needs_input_grad[1],
        )
        return grad_down, grad_up, None, None, None, None


class _DoRAFactor(torch.autograd.Function):
    """dora_scale / (||W + s*B@A||_col + eps) for every input channel as ONE autograd node. Built from separate ops
    it is a dozen nodes per layer, and eager DoRA training was bound by their dispatch (CPU 190 ms vs GPU 131 ms per
    step on a 24-block stack). Inputs: the magnitude (in,), ||W_k||^2 (in * kernel,), then down_0, up_0, down_1, ...
    (one pair per output block of a fused module, top to bottom; fp32 2-D)."""

    @staticmethod
    def forward(ctx, magnitude, base_col_sq, base, scaling, compute_dtype, *pairs):
        norm = _input_column_norm(base, base_col_sq, pairs, scaling, compute_dtype)
        denom = norm + DORA_EPS
        # (1, in, ...) -> (in,) fp32 inside the node: no reshape / cast nodes per layer
        flat = magnitude.reshape(-1).to(device=denom.device, dtype=torch.float32)
        ctx.save_for_backward(flat, norm, denom, *pairs)
        ctx.base, ctx.scaling, ctx.compute_dtype = base, scaling, compute_dtype
        ctx.magnitude_meta = (magnitude.shape, magnitude.dtype, magnitude.device)
        return flat / denom

    @staticmethod
    def backward(ctx, grad):
        magnitude, norm, denom, *pairs = ctx.saved_tensors
        base, scaling = ctx.base, ctx.scaling
        grad = grad.to(torch.float32)
        grad_magnitude = None
        if ctx.needs_input_grad[0]:
            shape, dtype, device = ctx.magnitude_meta
            grad_magnitude = (grad / denom).reshape(shape).to(device=device, dtype=dtype)
        # factor = m / (sqrt(n) + eps): d/dn = -m / (denom^2 * 2 sqrt(n)); clamp_min(0) passes nothing at n <= 0
        grad_norm_sq = torch.where(norm > 0, -grad * magnitude / (denom * denom * (2.0 * norm).clamp_min(1e-30)), 0.0)
        if base.kernel > 1:
            grad_norm_sq = grad_norm_sq.repeat_interleave(base.kernel)
        grads = []
        offset = 0
        for index, (down, up) in enumerate(zip(pairs[0::2], pairs[1::2])):
            need_down, need_up = ctx.needs_input_grad[5 + 2 * index], ctx.needs_input_grad[6 + 2 * index]
            if need_down or need_up:
                grads += _contribution_grads(base, down, up, offset, ctx.compute_dtype, scaling, grad_norm_sq, need_down, need_up)
            else:
                grads += [None, None]
            offset += up.shape[0]
        return (grad_magnitude, None, None, None, None, *grads)


def _input_column_norm(base, base_col_sq, pairs, scaling, compute_dtype) -> torch.Tensor:
    """||W + s*B@A|| per input channel from ||W_k||^2 and the (down, up) pairs (fp32 (in,), not differentiable)."""
    norm_sq = base_col_sq
    offset = 0
    for down, up in zip(pairs[0::2], pairs[1::2]):
        norm_sq = norm_sq + _contribution(base, down, up, offset, compute_dtype, scaling)
        offset += up.shape[0]
    if base.kernel > 1:
        norm_sq = norm_sq.view(-1, base.kernel).sum(dim=1)
    return norm_sq.clamp_min(0).sqrt()


def dora_input_factor(
    base: DoRABaseWeight,
    down_weights: List[torch.Tensor],
    up_weights: List[torch.Tensor],
    scaling: float,
    base_col_norm_sq: torch.Tensor,
    magnitude: torch.Tensor,
    compute_dtype: Optional[torch.dtype] = None,
    detach_norm: bool = False,
) -> torch.Tensor:
    """dora_scale / (||W + s*B@A||_col + eps) for every input channel (fp32 (in,), differentiable in the magnitude, A
    and B): what the input channels of the base layer and of the LoRA down projection are multiplied by.

    ``detach_norm`` treats the norm as a constant in backward (DoRA paper section 4.3, PEFT's default): A and B then
    get their gradient through the LoRA path only, and backward reads no base weight."""
    device = base_col_norm_sq.device
    scaling = float(scaling)
    compute_dtype = compute_dtype or base.compute_dtype()
    pairs = []
    if scaling != 0.0:
        for down, up in zip(down_weights, up_weights):
            for weight in (down, up):
                if weight.dim() != 2:
                    weight = weight.reshape(weight.shape[0], -1)
                if weight.device != device or weight.dtype != torch.float32:
                    weight = weight.to(device=device, dtype=torch.float32)
                pairs.append(weight)
    if detach_norm or not pairs:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            denom = _input_column_norm(base, base_col_norm_sq, pairs, scaling, compute_dtype) + DORA_EPS
        return magnitude.reshape(-1).to(device=device, dtype=torch.float32) / denom
    with torch.autocast(device_type=device.type, enabled=False):
        return _DoRAFactor.apply(magnitude, base_col_norm_sq, base, scaling, compute_dtype, *pairs)


def dora_column_norm(
    base: DoRABaseWeight,
    down_weights: List[torch.Tensor],
    up_weights: List[torch.Tensor],
    scaling: float,
    base_col_norm_sq: torch.Tensor,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """||W + s*B@A|| for every INPUT channel (fp32 (in,), differentiable in A and B). One (down, up) pair per output
    block: one for a plain module, one per split of a fused module, top to bottom.

    ||W_k + s (B A)_k||^2 = ||W_k||^2 + sum_r A[r, k] (2s (W^T B)[k, r] + s^2 ((B^T B) A)[r, k])"""
    device = base_col_norm_sq.device
    norm_sq = base_col_norm_sq
    scaling = float(scaling)
    if scaling != 0.0:
        compute_dtype = compute_dtype or base.compute_dtype()
        with torch.autocast(device_type=device.type, enabled=False):
            offset = 0
            for down, up in zip(down_weights, up_weights):
                a = down.reshape(down.shape[0], -1).to(device=device, dtype=torch.float32)
                b = up.reshape(up.shape[0], -1).to(device=device, dtype=torch.float32)
                norm_sq = norm_sq + _DoRANormSqContribution.apply(a, b, base, offset, compute_dtype, scaling)
                offset += b.shape[0]
    if base.kernel > 1:
        norm_sq = norm_sq.view(-1, base.kernel).sum(dim=1)
    return norm_sq.clamp_min(0).sqrt()


class _ScaledDownProjection(torch.autograd.Function):
    """x @ (A diag(f))^T for a Linear LoRA down projection. It keeps x (which the base layer's LoRA path keeps anyway),
    A and f, and rebuilds A diag(f) in backward instead of keeping a scaled copy of A per layer."""

    @staticmethod
    def forward(ctx, x, weight, factor):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        ctx.save_for_backward(x, weight, factor)
        ctx.dtype = dtype
        return torch.nn.functional.linear(_cast(x, dtype), _cast(weight * factor, dtype))  # (r, in) * (in,)

    @staticmethod
    def backward(ctx, grad):
        x, weight, factor = ctx.saved_tensors
        dtype = ctx.dtype
        grad = _cast(grad, dtype)
        grad_x = grad_weight = grad_factor = None
        if ctx.needs_input_grad[0]:
            grad_x = _cast(grad @ _cast(weight * factor, dtype), x.dtype)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            # dL/d(A diag(f)) = grad^T x, (r, in)
            outer = _mm_fp32(grad.reshape(-1, grad.shape[-1]).t(), _cast(x, dtype).reshape(-1, x.shape[-1]))
            if ctx.needs_input_grad[1]:
                grad_weight = _cast(outer * factor, weight.dtype)
            if ctx.needs_input_grad[2]:
                grad_factor = (outer * weight).sum(dim=0)
        return grad_x, grad_weight, grad_factor


def scaled_down_projection(x: torch.Tensor, weight: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return _ScaledDownProjection.apply(x, weight, factor)


def dora_input_axis_merge(
    base_weight: torch.Tensor, delta_weight: torch.Tensor, dora_scale: torch.Tensor, eps: float = DORA_EPS
) -> torch.Tensor:
    """(W + dW) * dora_scale / (||W + dW||_col + eps): ComfyUI's weight_decompose for an input-axis ``dora_scale``."""
    merged = base_weight + delta_weight
    in_channels = merged.shape[1]
    norm = merged.transpose(0, 1).reshape(in_channels, -1).norm(dim=1, keepdim=True)
    norm = norm.reshape(in_channels, *([1] * (merged.dim() - 1))).transpose(0, 1)
    return merged * (dora_scale.to(device=merged.device, dtype=merged.dtype).reshape(norm.shape) / (norm + eps))


def _channel_shape(tensor: torch.Tensor, channel_dim: int) -> list:
    shape = [1] * tensor.dim()
    shape[channel_dim % tensor.dim()] = -1
    return shape


def _scale_channels(tensor: torch.Tensor, factor: torch.Tensor, channel_dim: int, offset=None) -> torch.Tensor:
    """tensor * factor along ``channel_dim``. A 16-bit tensor gets tensor + tensor * (factor - 1) in one kernel: the
    small factor - 1 keeps its precision in bf16 where the factor itself (near 1) would round in steps of 0.4-0.8%.
    ``offset`` is factor - 1 already in the tensor's dtype."""
    last = channel_dim % tensor.dim() == tensor.dim() - 1  # (in,) broadcasts over the trailing dim as it is
    if tensor.dtype in (torch.float32, torch.float64):
        factor = _cast(factor, tensor.dtype)
        return tensor * (factor if last else factor.view(_channel_shape(tensor, channel_dim)))
    if offset is None or offset.dtype != tensor.dtype:
        offset = (factor - 1.0).to(tensor.dtype)
    return torch.addcmul(tensor, tensor, offset if last else offset.view(_channel_shape(tensor, channel_dim)))


def _channel_sums(grad: torch.Tensor, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """sum(grad * x) for every channel, accumulated in fp32."""
    dim = channel_dim % grad.dim()
    return (grad * x).sum(dim=[d for d in range(grad.dim()) if d != dim], dtype=torch.float32)


class _ScaleInputChannels(torch.autograd.Function):
    """x * f along the channel dim. It keeps x for backward, the tensor the LoRA path keeps anyway (layers sharing one
    input keep it once): the scaled copy only lives while the frozen base layer runs. factor - 1 in x's dtype is
    computed once and reused for the gradient."""

    @staticmethod
    def forward(ctx, x, factor, channel_dim):
        offset = None if x.dtype in (torch.float32, torch.float64) else (factor - 1.0).to(x.dtype)
        ctx.save_for_backward(x, factor, offset)
        ctx.channel_dim = channel_dim
        return _scale_channels(x, factor, channel_dim, offset)

    @staticmethod
    def backward(ctx, grad):
        x, factor, offset = ctx.saved_tensors
        grad_x = grad_factor = None
        if ctx.needs_input_grad[0]:
            grad_x = _scale_channels(grad, factor, ctx.channel_dim, offset)
        if ctx.needs_input_grad[1]:
            grad_factor = _cast(_channel_sums(grad, x, ctx.channel_dim), factor.dtype)
        return grad_x, grad_factor, None


def scale_input_channels(x: torch.Tensor, factor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    return _ScaleInputChannels.apply(x, factor, channel_dim)


def _reshape_dora_factor_for_weight(factor: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
    return factor.reshape(-1).view(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))


def _reshape_dora_factor_for_output(factor: torch.Tensor, output: torch.Tensor, is_conv2d: bool) -> torch.Tensor:
    flat_factor = factor.reshape(-1)
    if is_conv2d:
        return flat_factor.view(1, -1, *([1] * (output.dim() - 2)))
    return flat_factor.view(*([1] * (output.dim() - 1)), -1)


def _remove_module_bias(base_output: torch.Tensor, bias: Optional[torch.Tensor], is_conv2d: bool) -> torch.Tensor:
    if bias is None:
        return base_output
    if bias.device != base_output.device or bias.dtype != base_output.dtype:
        bias = bias.to(device=base_output.device, dtype=base_output.dtype)
    if is_conv2d:
        return base_output - bias.view(1, -1, *([1] * (base_output.dim() - 2)))
    return base_output - bias.view(*([1] * (base_output.dim() - 1)), -1)


def _apply_dora_weight_merge(
    base_weight: torch.Tensor, delta_weight: torch.Tensor, magnitude: torch.Tensor, weight_norm: torch.Tensor
) -> torch.Tensor:
    if weight_norm.is_floating_point():
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
    factor = magnitude.to(device=weight_norm.device, dtype=weight_norm.dtype) / weight_norm
    return _reshape_dora_factor_for_weight(factor, base_weight) * (base_weight + delta_weight)


def _convert_absolute_dora_oft_scales_to_ratios(state_dict: Dict[str, torch.Tensor]) -> None:
    for key in list(state_dict.keys()):
        if not key.endswith(".dora_scale"):
            continue
        module_name = key[: -len(".dora_scale")]
        initial_norm_key = f"{module_name}.initial_norm"
        if initial_norm_key not in state_dict:
            continue
        scale = state_dict[key]
        initial_norm = state_dict[initial_norm_key].to(dtype=scale.dtype)
        eps = 1e-12 if scale.dtype in (torch.float32, torch.float64) else 1e-6
        state_dict[key] = scale / initial_norm.clamp_min(eps)
