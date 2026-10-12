"""Input-axis DoRA (``<module>.dora_scale``, the ComfyUI / LyCORIS layout) on every LoRA target.

Training never builds W + s*B@A, so these tests hold the module to the dense definition
    W' = (W + s*B@A) * m / (||W + s*B@A||_col + eps)
for outputs and gradients, the saved file to ComfyUI's own ``weight_decompose`` (read from a local ComfyUI when
present), and the reader to the quantized bases the presets use (fp8 per row / per block, ConvRot INT8)."""

import ast
import math
import os
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from musubi_tuner.convert_lora import convert_from_diffusers, convert_to_diffusers
from musubi_tuner.modules import convrot_int8_kernels as kernels
from musubi_tuner.modules.convrot_int8_utils import _patch_convrot_int8_linear
from musubi_tuner.modules.fp8_optimization_utils import fp8_linear_forward_patch
from musubi_tuner.networks.lora import (
    DoRABaseCheckError,
    DoRAInputInfModule,
    DoRAInputModule,
    DoRAModule,
    LoRAModule,
    create_network,
    create_network_from_weights,
    merge_weights_to_tensor,
)
from musubi_tuner.networks.lora_shared import DORA_EPS

COMFY_BASE = os.environ.get("COMFYUI_WEIGHT_ADAPTER_BASE", r"G:\ComfyUI_V92\ComfyUI\comfy\weight_adapter\base.py")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _dense_merged(weight, down, up, magnitude, scale):
    """The DoRA definition written out densely, independent of the module's helpers."""
    delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)).reshape(weight.shape)
    merged = weight + scale * delta
    in_channels = weight.shape[1]
    norm = merged.transpose(0, 1).reshape(in_channels, -1).norm(dim=1)
    factor = magnitude.reshape(-1) / (norm + DORA_EPS)
    return merged * factor.view(1, in_channels, *([1] * (weight.dim() - 2)))


def _wrap(base, rank=4, alpha=2.0, seed=0, perturb=True, **kwargs):
    torch.manual_seed(seed)
    dora = DoRAInputModule("lora_unet_layer", base, multiplier=1.0, lora_dim=rank, alpha=alpha, **kwargs)
    if perturb:
        with torch.no_grad():
            ups = dora.lora_up if isinstance(dora.lora_up, nn.ModuleList) else [dora.lora_up]
            for up in ups:
                up.weight.normal_(std=0.05)
            dora.dora_scale.mul_(1.0 + 0.05 * torch.randn_like(dora.dora_scale))
    dora.apply_to()
    return dora


def _down_up(dora):
    if isinstance(dora.lora_down, nn.ModuleList):
        down = torch.cat([m.weight for m in dora.lora_down], dim=0)
        rank = dora.lora_dim
        up = torch.zeros(sum(m.weight.shape[0] for m in dora.lora_up), rank * len(dora.lora_up))
        row = 0
        for i, m in enumerate(dora.lora_up):
            up[row : row + m.weight.shape[0], i * rank : (i + 1) * rank] = m.weight
            row += m.weight.shape[0]
        return down, up
    return dora.lora_down.weight, dora.lora_up.weight


# --------------------------------------------------------------------------------------------- exact math (CPU)


def test_init_is_the_identity_and_dora_scale_holds_column_norms():
    base = nn.Linear(24, 16, bias=True)
    x = torch.randn(3, 5, 24)
    expected = base(x)
    dora = _wrap(base, perturb=False)
    assert dora.dora_scale.shape == (1, 24)
    assert torch.allclose(dora.dora_scale.reshape(-1), base.weight.detach().norm(dim=0), rtol=1e-6)
    assert torch.allclose(base(x), expected, atol=1e-6)


@pytest.mark.parametrize("bias", [False, True])
def test_linear_forward_and_gradients_match_the_dense_definition(bias):
    base = nn.Linear(32, 20, bias=bias)
    dora = _wrap(base, rank=4, alpha=3.0)
    x = torch.randn(2, 7, 32, requires_grad=True)
    out = base(x)
    weight = base.weight.detach()
    down, up = dora.lora_down.weight, dora.lora_up.weight
    ref_down = down.detach().clone().requires_grad_(True)
    ref_up = up.detach().clone().requires_grad_(True)
    ref_mag = dora.dora_scale.detach().clone().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    ref = F.linear(ref_x, _dense_merged(weight, ref_down, ref_up, ref_mag, dora.scale), base.bias)
    assert _relerr(out, ref) < 1e-5

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert _relerr(down.grad, ref_down.grad) < 1e-4
    assert _relerr(up.grad, ref_up.grad) < 1e-4
    assert _relerr(dora.dora_scale.grad, ref_mag.grad) < 1e-4
    assert _relerr(x.grad, ref_x.grad) < 1e-5


def test_detached_norm_matches_the_dense_definition_with_a_constant_norm():
    """dora_detach_norm (DoRA paper section 4.3, PEFT's default): same forward, the norm is a constant in backward."""
    base = nn.Linear(32, 20)
    dora = _wrap(base, rank=4, alpha=3.0, dora_detach_norm=True)
    x = torch.randn(2, 7, 32, requires_grad=True)
    out = base(x)
    weight = base.weight.detach()
    ref_down = dora.lora_down.weight.detach().clone().requires_grad_(True)
    ref_up = dora.lora_up.weight.detach().clone().requires_grad_(True)
    ref_mag = dora.dora_scale.detach().clone().requires_grad_(True)
    merged = weight + dora.scale * (ref_up @ ref_down)
    norm = merged.detach().norm(dim=0)
    ref = F.linear(x, merged * (ref_mag.reshape(-1) / (norm + DORA_EPS)), base.bias)
    assert _relerr(out, ref) < 1e-5
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert _relerr(dora.lora_down.weight.grad, ref_down.grad) < 1e-4
    assert _relerr(dora.lora_up.weight.grad, ref_up.grad) < 1e-4
    assert _relerr(dora.dora_scale.grad, ref_mag.grad) < 1e-4


def test_conv2d_forward_and_gradients_match_the_dense_definition():
    base = nn.Conv2d(6, 8, kernel_size=3, padding=1, bias=True)
    dora = _wrap(base, rank=3, alpha=3.0)
    assert dora.dora_scale.shape == (1, 6, 1, 1)
    x = torch.randn(2, 6, 9, 9)
    out = base(x)
    ref_down = dora.lora_down.weight.detach().clone().requires_grad_(True)
    ref_up = dora.lora_up.weight.detach().clone().requires_grad_(True)
    ref_mag = dora.dora_scale.detach().clone().requires_grad_(True)
    merged = _dense_merged(base.weight.detach(), ref_down, ref_up, ref_mag, dora.scale)
    ref = F.conv2d(x, merged, base.bias, padding=1)
    assert _relerr(out, ref) < 1e-5

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert _relerr(dora.lora_down.weight.grad, ref_down.grad) < 1e-4
    assert _relerr(dora.lora_up.weight.grad, ref_up.grad) < 1e-4
    assert _relerr(dora.dora_scale.grad, ref_mag.grad) < 1e-4


def test_split_dims_forward_and_gradients_match_the_dense_definition():
    base = nn.Linear(16, 30, bias=False)
    dora = _wrap(base, rank=2, alpha=2.0, split_dims=[10, 10, 10])
    x = torch.randn(4, 16)
    out = base(x)
    down, up = _down_up(dora)
    ref_mag = dora.dora_scale.detach().clone().requires_grad_(True)
    ref = F.linear(x, _dense_merged(base.weight.detach(), down.detach(), up.detach(), ref_mag, dora.scale))
    assert _relerr(out, ref) < 1e-5
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert _relerr(dora.dora_scale.grad, ref_mag.grad) < 1e-4
    assert all(m.weight.grad is not None for m in dora.lora_up)


def test_matches_comfyui_weight_decompose():
    """The exported file, applied the way ComfyUI applies a LoRA with dora_scale, gives the trained layer."""
    if not os.path.isfile(COMFY_BASE):
        pytest.skip("ComfyUI weight_adapter/base.py not found")
    tree = ast.parse(open(COMFY_BASE, encoding="utf-8").read())
    source = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "weight_decompose")
    comfy = types.SimpleNamespace(
        model_management=types.SimpleNamespace(cast_to_device=lambda t, device, dtype, copy=False: t.to(device=device, dtype=dtype))
    )
    namespace = {"torch": torch, "comfy": comfy}
    exec(compile(ast.Module(body=[source], type_ignores=[]), COMFY_BASE, "exec"), namespace)
    weight_decompose = namespace["weight_decompose"]

    base = nn.Linear(40, 24, bias=True)
    dora = _wrap(base, rank=4, alpha=8.0)
    exported = dora.export_state_dict()
    down = exported["lora_unet_layer.lora_down.weight"].float()
    up = exported["lora_unet_layer.lora_up.weight"].float()
    alpha = float(exported["lora_unet_layer.alpha"])
    lora_diff = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(base.weight.shape)
    comfy_weight = weight_decompose(
        exported["lora_unet_layer.dora_scale"], base.weight.detach().clone(), lora_diff, alpha / down.shape[0], 1.0,
        torch.float32, lambda a: a,
    )
    x = torch.randn(5, 40)
    assert _relerr(F.linear(x, comfy_weight, base.bias), base(x)) < 1e-5
    assert _relerr(comfy_weight, dora.get_merged_weight()) < 1e-6


# --------------------------------------------------------------------------------------------- quantized bases


def _fp8_linear(weight, block=None):
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    if block is None:
        scale = weight.abs().amax(dim=1, keepdim=True) / 448.0
        quantized = (weight / scale).to(torch.float8_e4m3fn)
    else:
        grouped = weight.view(weight.shape[0], -1, block)
        scale = grouped.abs().amax(dim=2, keepdim=True) / 448.0
        quantized = (grouped / scale).to(torch.float8_e4m3fn).view(weight.shape)
    module.weight = nn.Parameter(quantized, requires_grad=False)
    module.register_buffer("scale_weight", scale.to(torch.float32))
    module.forward = fp8_linear_forward_patch.__get__(module, type(module))
    if block is None:
        dense = quantized.float() * scale
    else:
        dense = (quantized.float().view(weight.shape[0], -1, block) * scale).view(weight.shape)
    return module, dense


@pytest.mark.parametrize("block", [None, 16])
def test_fp8_scaled_base_is_dequantized(block):
    torch.manual_seed(1)
    module, dense = _fp8_linear(torch.randn(24, 64) * 0.05, block=block)
    dora = _wrap(module, rank=4, alpha=4.0)
    x = torch.randn(3, 64)
    out = module(x)
    assert dora._dora_checked  # the one-time comparison with the layer passed
    ref = F.linear(x, _dense_merged(dense, dora.lora_down.weight.detach(), dora.lora_up.weight.detach(), dora.dora_scale.detach(), dora.scale))
    assert _relerr(out, ref) < 1e-4


def _convrot_linear(weight, group_size=256):
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=True, device=weight.device)
    quantized, scale = kernels.quantize_int8_convrot_weight(weight, group_size)
    module.weight = nn.Parameter(quantized, requires_grad=False)
    module.register_buffer("scale_weight", scale)
    _patch_convrot_int8_linear(module, group_size, "bf16")
    return module, kernels.dequantize_int8_convrot_weight(quantized, scale, group_size)


def test_convrot_int8_base_is_read_in_the_original_basis():
    torch.manual_seed(2)
    module, dense = _convrot_linear(torch.randn(48, 512) * 0.03)
    dora = _wrap(module, rank=4, alpha=4.0)
    x = torch.randn(6, 512)
    out = module(x)
    ref = F.linear(x, _dense_merged(dense, dora.lora_down.weight.detach(), dora.lora_up.weight.detach(), dora.dora_scale.detach(), dora.scale), module.bias)
    assert _relerr(out, ref) < 1e-4


def test_unreadable_base_is_refused_at_the_first_forward():
    torch.manual_seed(3)
    module, _ = _convrot_linear(torch.randn(16, 256) * 0.03)
    dora = _wrap(module)
    dora.dora_base.group_size = 0  # pretend the rotation is unknown: the weight read no longer matches the layer
    with pytest.raises(RuntimeError, match="DoRA cannot read the base weight"):
        module(torch.randn(2, 256))


class CheckpointedFp8Block(nn.Module):
    """A block as the trainers build it: activation checkpointing inside, compiled by the trainer."""

    def __init__(self):
        super().__init__()
        self.to_q = _fp8_linear(torch.randn(64, 64) * 0.05)[0]
        self.proj = _fp8_linear(torch.randn(64, 64) * 0.05, block=16)[0]

    def _forward(self, x):
        return self.proj(torch.relu(self.to_q(x)))

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)


class ConvRotBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = _convrot_linear(torch.randn(16, 256) * 0.03)[0]

    def forward(self, x):
        return self.to_q(x)


class Blocks(nn.Module):
    def __init__(self, block_class):
        super().__init__()
        self.blocks = nn.ModuleList([block_class(), block_class()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_compiled_checkpointed_blocks_trace_whole_with_quantized_bases():
    """Dynamo stops compiling a block for good once its checkpointed forward has a side effect: the DoRA forward sets
    no attributes and the one-time base check runs in a pre-hook of the model, before any compiled block."""
    torch.manual_seed(7)
    model = Blocks(CheckpointedFp8Block)
    network = create_network(["CheckpointedFp8Block"], "lora_unet", 1.0, 4, 2.0, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    loras = [lora for lora in network.unet_loras if isinstance(lora, DoRAInputModule)]
    assert len(loras) == 4 and not any(lora._dora_checked for lora in loras)
    for index, block in enumerate(model.blocks):
        model.blocks[index] = torch.compile(block, fullgraph=True, backend="eager")  # a graph break raises
    for _ in range(2):
        model(torch.randn(3, 64, requires_grad=True)).sum().backward()
    assert all(lora._dora_checked for lora in loras)
    assert not model._forward_pre_hooks  # the checks ran once
    assert all(lora.lora_down.weight.grad is not None and lora.dora_scale.grad is not None for lora in loras)


def test_unreadable_base_is_refused_before_the_first_block_runs():
    torch.manual_seed(8)
    model = Blocks(ConvRotBlock)
    network = create_network(["ConvRotBlock"], "lora_unet", 1.0, 4, 2.0, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    for lora in network.unet_loras:
        lora.dora_base.group_size = 0  # pretend the rotation is unknown: the weight read no longer matches the layer
    with pytest.raises(DoRABaseCheckError, match="DoRA cannot read the base weight"):
        model(torch.randn(2, 256))


def test_backward_reads_the_current_weight_after_a_block_swap():
    """Block swap reuses GPU storage between blocks: backward must read through the module, not a stale tensor."""
    base = nn.Linear(16, 12, bias=False)
    dora = _wrap(base)
    x = torch.randn(3, 16)
    original = base.weight.data
    out = base(x)
    base.weight.data = torch.zeros_like(original)  # the storage now holds another block
    base.weight.data = original.clone()  # and the block comes back before its backward
    out.sum().backward()
    ref_up = dora.lora_up.weight.detach().clone().requires_grad_(True)
    F.linear(x, _dense_merged(original, dora.lora_down.weight.detach(), ref_up, dora.dora_scale.detach(), dora.scale)).sum().backward()
    assert _relerr(dora.lora_up.weight.grad, ref_up.grad) < 1e-4


@pytest.mark.parametrize("module_class", [LoRAModule, DoRAInputModule])
def test_layers_sharing_one_input_keep_it_once_for_backward(module_class):
    """q, k and v read the same input: DoRA must not keep a scaled copy per layer (LoRA keeps the input once)."""
    torch.manual_seed(8)
    layers = [nn.Linear(64, 64).requires_grad_(False) for _ in range(3)]  # frozen base, as in training
    for index, layer in enumerate(layers):
        module_class(f"lora_unet_l{index}", layer, 1.0, 4, 4.0).apply_to()
    x = torch.randn(32, 64, requires_grad=True)
    kept = set()

    def pack(tensor):
        if tensor.numel() == x.numel():
            kept.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = sum(layer(x) for layer in layers)
    out.sum().backward()
    assert len(kept) == 1


def test_without_checkpointing_a_layer_keeps_only_per_channel_vectors():
    """Presets without gradient checkpointing keep every layer's saved tensors until backward: DoRA may add only
    (in,)-sized vectors, never (rank, in) or (out, rank) intermediates."""
    torch.manual_seed(9)
    layer = nn.Linear(256, 512).requires_grad_(False)
    dora = DoRAInputModule("lora_unet_l", layer, 1.0, 32, 32.0)
    dora.apply_to()
    params = {p.untyped_storage().data_ptr() for p in list(dora.parameters()) + list(layer.parameters())}
    x = torch.randn(8, 256, requires_grad=True)
    extra = []

    def pack(tensor):
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in params and pointer != x.untyped_storage().data_ptr():
            extra.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = layer(x)
    out.sum().backward()
    assert max(extra) <= 256, extra  # (in,) vectors only
    assert dora.lora_up.weight.grad is not None and dora.dora_scale.grad is not None


def test_a_plain_lora_continued_as_dora_starts_where_the_lora_was():
    torch.manual_seed(10)
    lora_layer = nn.Linear(48, 32).requires_grad_(False)
    dora_layer = nn.Linear(48, 32).requires_grad_(False)
    dora_layer.load_state_dict(lora_layer.state_dict())
    lora = LoRAModule("lora_unet_l", lora_layer, 1.0, 4, 4.0)
    with torch.no_grad():
        lora.lora_up.weight.normal_(std=0.2)
    lora.apply_to()
    dora = DoRAInputModule("lora_unet_l", dora_layer, 1.0, 4, 4.0)
    dora.apply_to()
    state = {f"lora_unet_l.{key}": value for key, value in lora.state_dict().items()}
    incompatible = dora.load_state_dict({key.split(".", 1)[1]: value for key, value in state.items()}, strict=True)
    assert not incompatible.missing_keys  # the magnitudes were derived from the LoRA, not left missing
    x = torch.randn(3, 48)
    with torch.no_grad():
        assert _relerr(dora_layer(x), lora_layer(x)) < 1e-5


def test_a_replaced_base_weight_is_read_again():
    """Wan 2.2 expert switching loads the other DiT's weights with assign=True: the base norms must follow."""
    torch.manual_seed(11)
    layer = nn.Linear(32, 24, bias=False).requires_grad_(False)
    dora = _wrap(layer)
    other = torch.randn(24, 32) * 0.5
    layer.load_state_dict({"weight": other}, assign=True)
    x = torch.randn(4, 32)
    ref = F.linear(x, _dense_merged(other, dora.lora_down.weight.detach(), dora.lora_up.weight.detach(), dora.dora_scale.detach(), dora.scale))
    with torch.no_grad():
        assert _relerr(layer(x), ref) < 1e-5


def test_no_grad_forwards_reuse_the_factor_until_a_parameter_changes():
    base = nn.Linear(16, 12)
    dora = _wrap(base)
    x = torch.randn(2, 16)
    with torch.no_grad():
        first = base(x)
        cached = dora._dora_cache[1]
        assert torch.equal(base(x), first)
        assert dora._dora_cache[1] is cached
        dora.lora_up.weight.add_(0.1)
        changed = base(x)
    assert dora._dora_cache[1] is not cached
    assert not torch.allclose(changed, first)


# --------------------------------------------------------------------------------------------- network, files


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 32)
        self.proj = nn.Linear(32, 48)
        self.gate = nn.Linear(32, 1)

    def forward(self, x):
        return self.proj(self.to_q(x)) + self.gate(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


def _network(model, **kwargs):
    return create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model, **kwargs)


def test_use_dora_trains_input_axis_by_default_and_output_axis_on_request():
    network = _network(Model(), use_dora="True")
    classes = {type(lora) for lora in network.unet_loras}
    assert DoRAInputModule in classes
    assert all(type(lora) is LoRAModule for lora in network.unet_loras if lora.lora_name.endswith("_gate"))
    output = _network(Model(), use_dora="True", dora_axis="output")
    assert all(type(lora) is DoRAModule for lora in output.unet_loras)
    with pytest.raises(ValueError, match="dora_axis"):
        _network(Model(), use_dora="True", dora_axis="sideways")


def test_saved_file_loads_back_and_merges_like_the_trained_network(tmp_path):
    torch.manual_seed(4)
    model = Model()
    network = _network(model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.normal_(std=0.05)
            if hasattr(lora, "dora_scale"):
                lora.dora_scale.mul_(1.02)
    x = torch.randn(3, 32)
    with torch.no_grad():
        trained = model(x)

    path = str(tmp_path / "dora.safetensors")
    network.save_weights(path, torch.bfloat16, {})
    saved = load_file(path)
    scale_key = "lora_unet_blocks_0_to_q.dora_scale"
    assert saved[scale_key].dtype == torch.float32 and saved[scale_key].shape == (1, 32)
    assert saved["lora_unet_blocks_0_to_q.lora_down.weight"].dtype == torch.bfloat16
    assert "lora_unet_blocks_0_gate.dora_scale" not in saved

    fresh = Model()
    fresh.load_state_dict({k: v for k, v in model.state_dict().items()})
    inference = create_network_from_weights(["Block"], 1.0, saved, unet=fresh, for_inference=True)
    assert any(isinstance(lora, DoRAInputInfModule) for lora in inference.unet_loras)
    inference.apply_to(None, fresh, apply_text_encoder=False, apply_unet=True)
    inference.load_state_dict(saved, strict=True)
    with torch.no_grad():
        assert _relerr(fresh(x), trained) < 1e-2  # bf16-saved down / up

    lora = next(l for l in inference.unet_loras if l.lora_name == "lora_unet_blocks_0_to_q")
    keys = {k for k in saved if k.startswith("lora_unet_blocks_0_to_q.")}
    merged = merge_weights_to_tensor(
        lora.org_module_ref[0].weight.detach().clone(), "lora_unet_blocks_0_to_q", saved, keys, 1.0, torch.device("cpu")
    )
    assert not keys  # every key of the module was used
    assert _relerr(merged, lora.get_merged_weight()) < 1e-5


def test_generation_scripts_merge_dora_files_with_their_magnitudes(tmp_path, caplog):
    """The *_generate_* scripts merge LoRA files while loading the model (lora_utils): a DoRA file must not be merged
    as a plain LoRA, which drops dora_scale."""
    from safetensors.torch import save_file

    from musubi_tuner.utils.lora_utils import detect_network_type, load_safetensors_with_lora_and_fp8

    torch.manual_seed(6)
    model = Model()
    network = _network(model, use_dora="True")
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.normal_(std=0.05)
            if hasattr(lora, "dora_scale"):
                lora.dora_scale.mul_(1.03)
    path = str(tmp_path / "dora.safetensors")
    network.save_weights(path, torch.float32, {})
    saved = load_file(path)
    assert detect_network_type(saved) == "dora"

    model_file = str(tmp_path / "model.safetensors")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, model_file)
    with caplog.at_level("WARNING"):
        merged = load_safetensors_with_lora_and_fp8(model_file, [saved], [1.0], False, torch.device("cpu"))
    assert "not all LoRA keys are used" not in caplog.text

    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    lora = next(l for l in network.unet_loras if l.lora_name == "lora_unet_blocks_0_to_q")
    with torch.no_grad():
        assert _relerr(merged["blocks.0.to_q.weight"], lora.get_merged_weight()) < 1e-5
        plain = model.blocks[0].to_q.weight + lora.lora_up.weight @ lora.lora_down.weight * lora.scale
    assert _relerr(merged["blocks.0.to_q.weight"], plain) > 1e-3  # the magnitude was applied


def test_diffusers_conversion_keeps_dora_scale():
    torch.manual_seed(5)
    network = _network(Model(), use_dora="True")
    exported = network.build_export_state_dict()
    other = convert_to_diffusers("lora_unet_", "diffusion_model", exported)
    modules = [key[: -len(".lora_A.weight")] for key in other if key.endswith(".lora_A.weight")]
    scaled = [module for module in modules if f"{module}.dora_scale" in other]
    assert len(scaled) == 4 and len(modules) == 6  # every DoRA layer keeps its scale next to lora_A / lora_B
    back = convert_from_diffusers("lora_unet_", other)
    assert {k for k in back if k.endswith(".dora_scale")} == {k for k in exported if k.endswith(".dora_scale")}


# --------------------------------------------------------------------------------------------- CUDA, bf16


@requires_cuda
def test_bf16_autocast_training_matches_fp32_on_cuda():
    torch.manual_seed(6)
    base = nn.Linear(256, 192, bias=True).cuda()
    dora = _wrap(base, rank=8, alpha=8.0)
    dora.cuda()
    x = torch.randn(4, 64, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = base(x)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref_up = dora.lora_up.weight.detach().clone().requires_grad_(True)
    ref_mag = dora.dora_scale.detach().clone().requires_grad_(True)
    ref = F.linear(x, _dense_merged(base.weight.detach(), dora.lora_down.weight.detach(), ref_up, ref_mag, dora.scale), base.bias)
    ref.backward(grad.float())
    assert _relerr(out, ref) < 1e-2
    assert _relerr(dora.lora_up.weight.grad, ref_up.grad) < 3e-2
    assert _relerr(dora.dora_scale.grad, ref_mag.grad) < 3e-2


@requires_cuda
@pytest.mark.skipif(not kernels.HAS_TRITON, reason="triton required")
def test_convrot_int8_triton_base_on_cuda():
    torch.manual_seed(7)
    module, dense = _convrot_linear(torch.randn(256, 512, device="cuda") * 0.03)
    dora = _wrap(module, rank=8, alpha=8.0)
    dora.cuda()
    x = torch.randn(2, 96, 512, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(x)
    ref = F.linear(x.float(), _dense_merged(dense, dora.lora_down.weight.detach(), dora.lora_up.weight.detach(), dora.dora_scale.detach(), dora.scale), module.bias)
    assert dora._dora_checked
    assert _relerr(out, ref) < 3e-2  # INT8 activation quantization
