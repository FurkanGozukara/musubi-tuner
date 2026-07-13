from copy import deepcopy
from types import SimpleNamespace
import gc

import pytest
import torch

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
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

    assert restored.state[restored_parameter]
    assert all(torch.isfinite(torch.as_tensor(lr)) for lr in restored.get_learning_rates())


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
    ],
)
def test_block_swap_gradient_patch_is_automatic_only_for_non_fused_automagic(optimizer_class, kwargs, expected):
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


@pytest.mark.parametrize("optimizer_type", ["Automagic2", "Automagic3"])
def test_explicit_fused_optimizer_rejects_unsafe_configuration(monkeypatch, optimizer_type):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.ones(2))
    kwargs = {"fused": True} if optimizer_type == "Automagic3" else {}

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        prepare_automagic_optimizer(
            optimizer_type,
            [parameter],
            1e-4,
            kwargs,
            make_args(gradient_accumulation_steps=2),
        )


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
