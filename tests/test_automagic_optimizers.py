from copy import deepcopy
from types import SimpleNamespace
import gc

import pytest
import torch
import torch.nn as nn
from transformers import Adafactor

from musubi_tuner.modules import convrot_int8_kernels
from musubi_tuner.modules.adafactor_fused import patch_adafactor_fused
from musubi_tuner.modules.convrot_int8_utils import CONVROT_GROUPSIZE, ConvRotInt8LinearFn
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.optimizers import Automagic, Automagic2, Automagic3
from musubi_tuner.optimizers.optimizer_utils import stochastic_grad_accummulation
from musubi_tuner.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from musubi_tuner.ideogram4.ideogram4_model import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.optimizers.factory import (
    get_adaptive_learning_rates,
    is_automagic_optimizer_type,
    materialize_stochastic_gradients,
    move_optimizer_gradients_to_parameters,
    prepare_automagic_optimizer,
    should_patch_block_swap_gradients,
    uses_fused_backward,
)
from musubi_tuner.zimage.zimage_model import ZImageTransformer2DModel


def make_args(**overrides):
    values = {
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 0.0,
        "mixed_precision": "bf16",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [
        (Automagic, {}),
        (Automagic2, {}),
        (Automagic3, {"fused": True}),
        (Automagic3, {"fused": False}),
    ],
)
def test_automagic_optimizer_updates_float32_parameter(optimizer_class, kwargs):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = optimizer_class([parameter], lr=1e-4, **kwargs)
    before = parameter.detach().clone()

    parameter.square().sum().backward()
    optimizer.step()

    assert not torch.equal(parameter, before)
    assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    assert all(torch.isfinite(torch.as_tensor(lr)) for lr in optimizer.get_learning_rates())


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [
        (Automagic, {}),
        (Automagic2, {}),
        (Automagic3, {"fused": True}),
        (Automagic3, {"fused": False}),
    ],
)
def test_automagic_optimizer_state_round_trip(optimizer_class, kwargs):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = optimizer_class([parameter], lr=1e-4, **kwargs)
    parameter.square().sum().backward()
    optimizer.step()
    saved_state = deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = optimizer_class([restored_parameter], lr=2e-4, **kwargs)
    restored.load_state_dict(saved_state)
    restored_parameter.square().sum().backward()
    restored.step()

    assert restored.state[restored_parameter]
    assert restored.state[restored_parameter]["step"] == 2
    assert all(torch.isfinite(torch.as_tensor(lr)) for lr in restored.get_learning_rates())


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [
        (Automagic, {}),
        (Automagic, {"fused": True}),
        (Automagic2, {}),
        (Automagic3, {"fused": True}),
        (Automagic3, {"fused": False}),
    ],
)
def test_late_param_groups_are_hooked_and_updated(optimizer_class, kwargs):
    initial = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
    late = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
    optimizer = optimizer_class([initial], lr=1e-3, **kwargs)
    optimizer.add_param_group({"params": [late], "lr": 1e-3})
    initial_before = initial.detach().clone()
    late_before = late.detach().clone()

    (initial.square().sum() + late.square().sum()).backward()
    optimizer.step()

    assert not torch.equal(initial, initial_before)
    assert not torch.equal(late, late_before)
    assert optimizer.state[initial]["step"] == 1
    assert optimizer.state[late]["step"] == 1


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [
        (Automagic, {"fused": True}),
        (Automagic2, {}),
        (Automagic3, {"fused": True}),
    ],
)
def test_param_group_rebuild_rebinds_hooks_without_double_updates(optimizer_class, kwargs):
    parameter = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
    optimizer = optimizer_class([parameter], lr=1e-3, **kwargs)
    optimizer.param_groups = []
    optimizer.add_param_group({"params": [parameter], "lr": 1e-3})

    parameter.square().sum().backward()
    optimizer.step()

    assert optimizer.state[parameter]["step"] == 1


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs"),
    [(Automagic, {}), (Automagic3, {"fused": False})],
)
def test_zero_grad_discards_stochastic_accumulation_buffer(optimizer_class, kwargs):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = optimizer_class([parameter], lr=1e-4, **kwargs)
    parameter.float().square().sum().backward()
    assert hasattr(parameter, "_accum_grad")

    optimizer.zero_grad(set_to_none=True)

    assert parameter.grad is None
    assert not hasattr(parameter, "_accum_grad")


def test_automagic3_honors_latest_upstream_lr_bounds():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = Automagic3([parameter], lr=1e-3, min_lr=1e-5, max_lr=1e-4, fused=False)
    parameter.square().sum().backward()
    optimizer.step()

    assert optimizer.get_learning_rates() == pytest.approx([1e-4])
    with pytest.raises(ValueError, match="min_lr"):
        Automagic3([torch.nn.Parameter(torch.ones(1))], min_lr=1e-3, max_lr=1e-4)


@pytest.mark.parametrize("optimizer_class", [Automagic, Automagic3])
def test_materialize_stochastic_gradients_for_low_precision(optimizer_class):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    kwargs = {"fused": False} if optimizer_class is Automagic3 else {}
    optimizer = optimizer_class([parameter], lr=1e-4, **kwargs)

    parameter.float().square().sum().backward()
    parameter.float().square().sum().backward()
    assert parameter.grad is None
    assert hasattr(parameter, "_accum_grad")

    materialize_stochastic_gradients(SimpleNamespace(optimizer=optimizer))

    assert parameter.grad is not None
    assert not hasattr(parameter, "_accum_grad")


def test_automagic_accumulation_hook_ignores_checkpoint_recompute_without_gradient():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    stochastic_grad_accummulation(parameter)

    assert parameter.grad is None
    assert not hasattr(parameter, "_accum_grad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to emulate a block-swapped parameter")
def test_materialize_stochastic_gradients_moves_to_swapped_parameter_device():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = Automagic([parameter], lr=1e-4)
    parameter._accum_grad = torch.ones(2, dtype=torch.bfloat16, device="cuda")

    materialize_stochastic_gradients(optimizer)

    assert parameter.grad is not None
    assert parameter.grad.device.type == "cpu"


@pytest.mark.parametrize(
    ("optimizer_class", "kwargs", "expected"),
    [
        (Automagic, {}, True),
        (Automagic2, {}, False),
        (Automagic3, {"fused": True}, False),
        (Automagic3, {"fused": False}, True),
        (torch.optim.AdamW, {}, True),
    ],
)
def test_block_swap_gradient_patch_is_automatic_for_non_fused_optimizers(optimizer_class, kwargs, expected):
    optimizer = optimizer_class([torch.nn.Parameter(torch.ones(2))], lr=1e-4, **kwargs)

    assert should_patch_block_swap_gradients(SimpleNamespace(optimizer=optimizer)) is expected
    assert should_patch_block_swap_gradients(optimizer, requested=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to emulate block-swapped gradients")
def test_move_optimizer_gradients_to_parameters_handles_accelerate_wrapper():
    parameter = torch.nn.Parameter(torch.ones(2, device="cuda"))
    parameter.grad = torch.ones_like(parameter)
    parameter.data = parameter.data.cpu()
    optimizer = Automagic3([parameter], lr=1e-4, fused=False)

    move_optimizer_gradients_to_parameters(SimpleNamespace(optimizer=optimizer))

    assert parameter.grad.device == parameter.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to emulate block-swapped gradients")
def test_adafactor_step_repairs_block_swapped_gradient_device():
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    optimizer = Adafactor(
        [parameter],
        lr=1e-3,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    parameter.float().sum().backward()
    parameter.data = parameter.data.cpu()

    assert parameter.grad.device.type == "cuda"
    assert should_patch_block_swap_gradients(optimizer)

    move_optimizer_gradients_to_parameters(optimizer)
    optimizer.step()

    assert parameter.grad.device == parameter.device
    assert optimizer.state[parameter]["step"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to emulate block-swapped gradients")
def test_fused_adafactor_step_repairs_block_swapped_gradient_device():
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    optimizer = Adafactor(
        [parameter],
        lr=1e-3,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    patch_adafactor_fused(optimizer)
    parameter.float().sum().backward()
    parameter.data = parameter.data.cpu()

    assert uses_fused_backward(optimizer)
    assert not should_patch_block_swap_gradients(optimizer)

    optimizer.step_param(parameter, optimizer.param_groups[0])

    assert optimizer.state[parameter]["step"] == 1


def test_automagic_type_detection_is_case_insensitive():
    assert is_automagic_optimizer_type("Automagic")
    assert is_automagic_optimizer_type("AUTOMAGIC2")
    assert is_automagic_optimizer_type(" automagic3 ")
    assert not is_automagic_optimizer_type("AdamW")


def test_automagic3_automatically_uses_fused_mode_when_safe(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))

    _, optimizer, kwargs = prepare_automagic_optimizer("Automagic3", [parameter], 1e-4, {}, make_args())

    assert optimizer.fused is True
    assert kwargs["fused"] is True


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_automagic_ignores_adafactor_preset_arguments(monkeypatch, caplog, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))
    preset_kwargs = {
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "weight_decay": 0.01,
    }

    _, optimizer, kwargs = prepare_automagic_optimizer(optimizer_type, [parameter], 1e-4, preset_kwargs, make_args())

    assert set(kwargs).isdisjoint({"scale_parameter", "relative_step", "warmup_init"})
    assert kwargs["weight_decay"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert "Ignoring Adafactor-only optimizer arguments" in caplog.text


def test_automagic3_automatically_falls_back_to_non_fused(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))

    _, optimizer, kwargs = prepare_automagic_optimizer(
        "Automagic3",
        [parameter],
        1e-4,
        {},
        make_args(gradient_accumulation_steps=2, max_grad_norm=1.0),
    )

    assert optimizer.fused is False
    assert kwargs["fused"] is False


def test_automagic3_block_swap_optimizer_patch_selects_non_fused(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))

    _, optimizer, kwargs = prepare_automagic_optimizer(
        "Automagic3",
        [parameter],
        1e-4,
        {},
        make_args(block_swap_optimizer_patch_params=True),
    )

    assert optimizer.fused is False
    assert kwargs["fused"] is False
    assert not uses_fused_backward(optimizer)


def test_non_fused_full_finetune_offloads_block_swap_gradients(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16, device=device))

    _, optimizer, resolved = prepare_automagic_optimizer(
        "Automagic3",
        [parameter],
        1e-4,
        {"fused": False},
        make_args(full_finetune=True, blocks_to_swap=1),
    )

    assert resolved["offload_gradients"] is True
    assert optimizer.offload_gradients is True

    parameter.float().square().sum().backward()
    assert parameter._accum_grad.device.type == "cpu"


def test_automagic_full_finetune_fuses_updates_and_offloads_state(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter = torch.nn.Parameter(torch.ones(9, dtype=torch.float32, device=device))
    before = parameter.detach().clone()

    _, optimizer, resolved = prepare_automagic_optimizer(
        "Automagic",
        [parameter],
        1e-4,
        {},
        make_args(full_finetune=True, blocks_to_swap=1),
    )

    assert resolved["fused"] is True
    assert resolved["offload_state"] is True
    assert optimizer.fused is True
    assert optimizer.offload_state is True

    parameter.float().square().sum().backward()
    state = optimizer.state[parameter]
    assert not torch.equal(parameter, before)
    assert parameter.grad is None
    assert state["lr_mask"].quantized.device.type == "cpu"
    assert state["last_polarity_packed"].device.type == "cpu"
    assert state["last_polarity_packed"].numel() == 2


def test_automagic_fused_offloaded_state_round_trip():
    parameter = torch.nn.Parameter(torch.ones(9))
    optimizer = Automagic([parameter], lr=1e-4, fused=True, offload_state=True)
    parameter.square().sum().backward()
    saved_state = deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = Automagic([restored_parameter], lr=1e-4, fused=True, offload_state=True)
    restored.load_state_dict(saved_state)
    restored_parameter.square().sum().backward()

    restored_state = restored.state[restored_parameter]
    assert restored_state["step"] == 2
    assert restored_state["lr_mask"].quantized.device.type == "cpu"
    assert restored_state["last_polarity_packed"].device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires a CUDA device with bfloat16 support",
)
@pytest.mark.filterwarnings("ignore:The argument 'device' of Tensor.pin_memory.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:The argument 'device' of Tensor.is_pinned.*:DeprecationWarning")
def test_all_automagic_modes_update_cpu_and_cuda_parameters_with_real_block_swap(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda")
    cases = (
        ("Automagic", {}, False),
        ("Automagic2", {}, False),
        ("Automagic3", {}, False),
        ("Automagic3", {}, True),
    )

    for optimizer_type, optimizer_kwargs, patch_optimizer in cases:
        torch.manual_seed(321)
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=4,
            dim=32,
            n_layers=4,
            n_refiner_layers=1,
            n_heads=4,
            n_kv_heads=4,
            cap_feat_dim=24,
            axes_dims=[2, 2, 4],
            axes_lens=[32, 32, 32],
            attn_mode="torch",
        ).to(dtype=torch.bfloat16)
        model.requires_grad_(True)
        model.enable_gradient_checkpointing(False)
        model.enable_block_swap(2, BlockSwapConfig(device, supports_backward=True, use_pinned_memory=False))
        model.move_to_device_except_swap_blocks(device)

        _, optimizer, _ = prepare_automagic_optimizer(
            optimizer_type,
            model.parameters(),
            6e-5,
            optimizer_kwargs,
            make_args(block_swap_optimizer_patch_params=patch_optimizer),
        )
        model.prepare_block_swap_before_forward()
        tracked = [layer.attention.to_q.weight for layer in model.layers]
        before = [parameter.detach().float().cpu().clone() for parameter in tracked]

        for _ in range(2):
            x = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
            timestep = torch.rand(1, device=device, dtype=torch.bfloat16)
            caption = torch.randn(1, 2, 24, device=device, dtype=torch.bfloat16)
            loss = model(x, timestep, caption, None).float().square().mean()
            loss.backward()
            materialize_stochastic_gradients(optimizer)
            move_optimizer_gradients_to_parameters(optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        final_devices = {parameter.device.type for parameter in tracked}
        assert final_devices == {"cpu", "cuda"}
        assert all(not torch.equal(old, parameter.detach().float().cpu()) for old, parameter in zip(before, tracked))
        assert uses_fused_backward(optimizer) is (optimizer_type != "Automagic" and not patch_optimizer)

        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires a CUDA device with bfloat16 support",
)
@pytest.mark.filterwarnings("ignore:The argument 'device' of Tensor.pin_memory.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:The argument 'device' of Tensor.is_pinned.*:DeprecationWarning")
@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_ideogram4_full_finetune_automagic_updates_with_real_block_swap(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda")
    model = Ideogram4Transformer(
        Ideogram4Config(
            emb_dim=32,
            num_layers=4,
            num_heads=4,
            intermediate_size=64,
            adanln_dim=16,
            in_channels=4,
            llm_features_dim=8,
            mrope_section=(2, 1, 1),
        ),
        attn_mode="torch",
    ).to(dtype=torch.bfloat16)
    model.requires_grad_(True)
    model.enable_block_swap(2, BlockSwapConfig(device, supports_backward=True, use_pinned_memory=False))
    model.move_to_device_except_swap_blocks(device)

    _, optimizer, _ = prepare_automagic_optimizer(
        optimizer_type,
        model.parameters(),
        1e-3,
        {},
        make_args(full_finetune=True, blocks_to_swap=2),
    )
    tracked = [layer.attention.qkv.weight for layer in model.layers]
    before = [parameter.detach().float().cpu().clone() for parameter in tracked]
    model.prepare_block_swap_before_forward()

    indicator = torch.tensor(
        [[OUTPUT_IMAGE_INDICATOR, OUTPUT_IMAGE_INDICATOR, LLM_TOKEN_INDICATOR, LLM_TOKEN_INDICATOR]],
        device=device,
    )
    output = model(
        llm_features=torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16),
        x=torch.randn(1, 4, 4, device=device, dtype=torch.bfloat16),
        t=torch.rand(1, device=device),
        position_ids=torch.arange(4, device=device).view(1, 4, 1).expand(-1, -1, 3),
        attention_mask=torch.ones(1, 2, device=device, dtype=torch.bool),
        indicator=indicator,
    )
    output.square().mean().backward()
    materialize_stochastic_gradients(optimizer)
    move_optimizer_gradients_to_parameters(optimizer)
    optimizer.step()

    assert {parameter.device.type for parameter in tracked} == {"cpu", "cuda"}
    assert all(optimizer.state[parameter].get("step") == 1 for parameter in tracked)
    assert any(not torch.equal(old, parameter.detach().float().cpu()) for old, parameter in zip(before, tracked))

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or not convrot_int8_kernels.HAS_TRITON,
    reason="requires CUDA, bfloat16, and Triton ConvRot kernels",
)
@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
@pytest.mark.parametrize("backward_mode", ["bf16", "int8"])
def test_automagic_updates_lora_on_convrot_int8_base(monkeypatch, optimizer_type, backward_mode):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda")
    in_features, out_features, tokens, rank = 512, 96, 16, 8
    torch.manual_seed(73)
    base_weight = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16) * 0.02
    quantized_weight, weight_scale = convrot_int8_kernels.quantize_int8_convrot_weight(
        base_weight, CONVROT_GROUPSIZE
    )
    lora_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=torch.bfloat16)
    lora_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=torch.bfloat16)
    parameters = list(lora_a.parameters()) + list(lora_b.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    _, optimizer, _ = prepare_automagic_optimizer(optimizer_type, parameters, 1e-3, {}, make_args())

    inputs = torch.randn(tokens, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(tokens, out_features, device=device, dtype=torch.bfloat16)
    base_output = ConvRotInt8LinearFn.apply(
        inputs,
        quantized_weight,
        weight_scale,
        None,
        CONVROT_GROUPSIZE,
        backward_mode,
    )
    prediction = base_output + lora_b(lora_a(inputs))
    prediction.float().sub(target.float()).square().mean().backward()
    materialize_stochastic_gradients(optimizer)
    optimizer.step()

    assert quantized_weight.dtype == torch.int8
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(optimizer.state[parameter].get("step") == 1 for parameter in parameters)
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, parameters))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or not hasattr(torch, "float8_e4m3fn"),
    reason="requires CUDA with bfloat16 and FP8 storage support",
)
@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_automagic_updates_lora_on_scaled_fp8_base(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    device = torch.device("cuda")
    in_features, out_features, tokens, rank = 64, 32, 16, 8
    torch.manual_seed(91)

    class TinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)

    base = TinyBase()
    source_weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32) * 0.02
    scale = source_weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).div(448.0)
    quantized_weight = source_weight.div(scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    optimized_state = {
        "proj.weight": quantized_weight,
        "proj.scale_weight": scale.to(torch.bfloat16),
    }
    apply_fp8_monkey_patch(base, optimized_state, use_scaled_mm=False)
    base.requires_grad_(False)
    base.load_state_dict(optimized_state, strict=True, assign=True)

    lora_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=torch.bfloat16)
    lora_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=torch.bfloat16)
    parameters = list(lora_a.parameters()) + list(lora_b.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    _, optimizer, _ = prepare_automagic_optimizer(optimizer_type, parameters, 1e-3, {}, make_args())

    inputs = torch.randn(tokens, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(tokens, out_features, device=device, dtype=torch.bfloat16)
    prediction = base.proj(inputs) + lora_b(lora_a(inputs))
    prediction.float().sub(target.float()).square().mean().backward()
    materialize_stochastic_gradients(optimizer)
    optimizer.step()

    assert base.proj.weight.dtype == torch.float8_e4m3fn
    assert not base.proj.weight.requires_grad
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(optimizer.state[parameter].get("step") == 1 for parameter in parameters)
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, parameters))


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_explicit_fused_optimizer_rejects_unsafe_configuration(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))
    kwargs = {"fused": True} if optimizer_type in {"Automagic", "Automagic3"} else {}

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        prepare_automagic_optimizer(
            optimizer_type,
            [parameter],
            1e-4,
            kwargs,
            make_args(gradient_accumulation_steps=2),
        )


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_separate_adafactor_fused_backward_mode_is_rejected(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)

    with pytest.raises(ValueError, match="Adafactor-only"):
        prepare_automagic_optimizer(
            optimizer_type,
            [torch.nn.Parameter(torch.ones(2))],
            1e-4,
            {},
            make_args(fused_backward_pass=True),
        )


@pytest.mark.parametrize("feature", ["blank_preservation", "dop", "audio_dop", "motion_preservation_separate_backward"])
def test_automagic2_rejects_features_with_additional_backward_passes(monkeypatch, feature):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)

    with pytest.raises(ValueError, match="additional backward passes"):
        prepare_automagic_optimizer(
            "Automagic2",
            [torch.nn.Parameter(torch.ones(2))],
            1e-4,
            {},
            make_args(**{feature: True}),
        )


def test_automagic3_uses_non_fused_mode_for_additional_backward_passes(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)

    _, optimizer, resolved = prepare_automagic_optimizer(
        "Automagic3",
        [torch.nn.Parameter(torch.ones(2))],
        1e-4,
        {},
        make_args(blank_preservation=True),
    )

    assert optimizer.fused is False
    assert resolved["fused"] is False


def test_adaptive_learning_rates_unwrap_accelerate_optimizer():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = Automagic3([parameter], lr=1e-4, fused=False)

    assert get_adaptive_learning_rates(SimpleNamespace(optimizer=optimizer)) == [1e-4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires mixed CPU/CUDA optimizer state")
@pytest.mark.parametrize("optimizer_class", [Automagic, Automagic3])
def test_adaptive_learning_rate_reporting_handles_block_swapped_state(optimizer_class):
    cpu_parameter = torch.nn.Parameter(torch.ones(2))
    cuda_parameter = torch.nn.Parameter(torch.ones(2, device="cuda"))
    kwargs = {"fused": False} if optimizer_class is Automagic3 else {}
    optimizer = optimizer_class([cpu_parameter, cuda_parameter], lr=1e-4, **kwargs)

    if optimizer_class is Automagic:
        optimizer.state[cpu_parameter]["avg_lr"] = torch.tensor(1e-4)
        optimizer.state[cuda_parameter]["avg_lr"] = torch.tensor(2e-4, device="cuda")
        expected = 1.5e-4
    else:
        optimizer.state[cpu_parameter]["lr"] = torch.tensor(1e-4)
        optimizer.state[cuda_parameter]["lr"] = torch.tensor(1e-4, device="cuda")
        expected = 1e-4

    assert get_adaptive_learning_rates(optimizer) == pytest.approx([expected])


@pytest.mark.parametrize(
    "trainer_factory",
    [
        pytest.param(lambda: __import__("musubi_tuner.training.trainer_base", fromlist=["NetworkTrainer"]).NetworkTrainer(), id="shared"),
        pytest.param(lambda: __import__("musubi_tuner.hv_train", fromlist=["FineTuningTrainer"]).FineTuningTrainer(), id="legacy-hv"),
    ],
)
def test_all_trainer_factories_construct_automagic3(monkeypatch, trainer_factory):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    trainer = trainer_factory()
    args = make_args(
        optimizer_type="Automagic3",
        optimizer_args=[],
        learning_rate=1e-4,
        max_grad_norm=1.0,
    )

    name, serialized_args, optimizer, _, _ = trainer.get_optimizer(args, [torch.nn.Parameter(torch.ones(2))])
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1) if hasattr(trainer, "get_lr_scheduler") else trainer.get_scheduler(args, optimizer, 1)

    assert name.endswith(".Automagic3")
    assert "fused=False" in serialized_args
    assert optimizer.fused is False
    assert trainer.is_schedulefree_optimizer(optimizer, args)
    assert scheduler.get_last_lr() == [1e-4]


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_modular_trainer_constructs_every_automagic_variant(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    trainer = __import__("musubi_tuner.training.trainer_ext", fromlist=["NetworkTrainer"]).NetworkTrainer()
    args = make_args(
        optimizer_type=optimizer_type,
        optimizer_args=[],
        learning_rate=1e-4,
        max_grad_norm=0.0,
    )

    name, _, optimizer, _, _ = trainer.get_optimizer(args, [torch.nn.Parameter(torch.ones(2))])
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)

    assert name.endswith(f".{optimizer_type}")
    assert trainer.is_schedulefree_optimizer(optimizer, args)
    assert scheduler.get_last_lr() == pytest.approx(get_adaptive_learning_rates(optimizer))


@pytest.mark.parametrize(
    "trainer_factory",
    [
        pytest.param(
            lambda: __import__("musubi_tuner.qwen_image_train_network", fromlist=["QwenImageNetworkTrainer"]).QwenImageNetworkTrainer(),
            id="qwen-image",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.wan_train_network", fromlist=["WanNetworkTrainer"]).WanNetworkTrainer(),
            id="wan",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.flux_2_train_network", fromlist=["Flux2NetworkTrainer"]).Flux2NetworkTrainer(),
            id="flux",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.zimage_train_network", fromlist=["ZImageNetworkTrainer"]).ZImageNetworkTrainer(),
            id="z-image",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.ideogram4_train_network", fromlist=["Ideogram4NetworkTrainer"]).Ideogram4NetworkTrainer(),
            id="ideogram-4",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.krea2_train_network", fromlist=["Krea2NetworkTrainer"]).Krea2NetworkTrainer(),
            id="krea-2",
        ),
        pytest.param(
            lambda: __import__("musubi_tuner.ltx2_train_network", fromlist=["LTX2NetworkTrainer"]).LTX2NetworkTrainer(),
            id="ltx-2.3",
        ),
    ],
)
@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic2", "Automagic3"])
def test_every_model_family_constructs_and_steps_every_automagic_variant(monkeypatch, trainer_factory, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    trainer = trainer_factory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter = torch.nn.Parameter(torch.tensor([0.75, -0.5], device=device, dtype=torch.float32))
    before = parameter.detach().clone()
    args = make_args(
        optimizer_type=optimizer_type,
        optimizer_args=[],
        learning_rate=1e-4,
        lr_scheduler="constant",
        full_finetune=False,
        blocks_to_swap=0,
        fused_backward_pass=False,
    )

    name, _, optimizer, _, _ = trainer.get_optimizer(args, [parameter])
    parameter.float().square().sum().backward()
    optimizer.step()

    assert name.endswith(f".{optimizer_type}")
    assert not torch.equal(parameter, before)
    assert all(torch.isfinite(torch.as_tensor(lr)) for lr in get_adaptive_learning_rates(optimizer))
