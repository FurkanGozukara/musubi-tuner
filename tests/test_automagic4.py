"""Automagic4 (ai-toolkit 534d462, V39): Automagic3 with sensitivity-balanced updates and a re-centered flip-rate vote.

The first step only measures (second moments and the sensitivity reference); from the second step each element's
clipped v3 update is scaled by min(1, reference / sensitivity), and the group lr moves toward the flip rate of a
weight whose effective_lr * curvature equals ``step_target``.
"""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from musubi_tuner.optimizers import Automagic3, Automagic4
from musubi_tuner.optimizers.factory import (
    get_adaptive_learning_rates,
    materialize_stochastic_gradients,
    prepare_automagic_optimizer,
    uses_fused_backward,
)


def make_args(**overrides):
    values = {"gradient_accumulation_steps": 1, "max_grad_norm": 0.0, "mixed_precision": "bf16"}
    values.update(overrides)
    return SimpleNamespace(**values)


def _step(optimizer, params, loss_fn):
    loss_fn().backward()
    materialize_stochastic_gradients(optimizer)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_first_step_only_measures_then_the_parameters_move(fused, dtype):
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(8, 16, dtype=dtype))
    bias = nn.Parameter(torch.randn(8, dtype=dtype))
    optimizer = Automagic4([weight, bias], lr=1e-3, fused=fused)
    before = [weight.detach().clone(), bias.detach().clone()]
    x = torch.randn(4, 16)

    def loss():
        return (x @ weight.float().t() + bias.float()).square().mean()

    _step(optimizer, [weight, bias], loss)
    assert torch.equal(weight, before[0]) and torch.equal(bias, before[1])  # measurement only
    assert optimizer.state[weight]["step"] == 1 and "sens_ref" in optimizer.state[weight]
    assert optimizer.state[weight]["sens_ref"].dtype == torch.float32

    _step(optimizer, [weight, bias], loss)
    assert not torch.equal(weight, before[0]) and not torch.equal(bias, before[1])
    assert optimizer.state[weight]["step"] == 2
    assert all(math.isfinite(lr) for lr in optimizer.get_learning_rates())


def test_sensitive_tensors_take_smaller_steps_than_with_automagic3():
    """A tensor whose gradient power is far above the group's reference moves less than v3's normalized step; the bulk
    (at or below the reference) keeps v3's step."""
    torch.manual_seed(1)
    init_bulk = torch.randn(64, 64) * 0.02
    init_hot = torch.randn(4, 64) * 0.02  # 6% of the elements: the 90% (by element count) reference is the bulk's
    x = torch.randn(32, 64)

    def run(cls):
        bulk = nn.Parameter(init_bulk.clone())
        hot = nn.Parameter(init_hot.clone())
        optimizer = cls([bulk, hot], lr=1e-3, fused=False)
        moves = []
        for _ in range(2):
            before = (bulk.detach().clone(), hot.detach().clone())
            loss = (x @ bulk.t()).square().mean() + 1e4 * (x @ hot.t()).square().mean()  # hot: ~1e4x the gradient
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            moves.append(((bulk - before[0]).abs().mean().item(), (hot - before[1]).abs().mean().item()))
        return moves

    v3 = run(Automagic3)
    v4 = run(Automagic4)
    # v3 normalizes every element to a step of about lr, whatever its gradient scale
    assert v3[0][1] == pytest.approx(v3[0][0], rel=0.5)
    # v4: step 1 measured only, step 2 balanced: the hot tensor moves far less than v3's, the bulk about as much
    assert v4[0] == (0.0, 0.0)
    assert v4[1][1] < 0.05 * v3[1][1]
    assert v4[1][0] == pytest.approx(v3[1][0], rel=0.5)


def test_reference_is_the_numel_weighted_quantile_of_tensor_sensitivities():
    torch.manual_seed(2)
    small = nn.Parameter(torch.zeros(4))  # 4 elements, large gradient
    large = nn.Parameter(torch.zeros(96))  # 96 elements, small gradient
    optimizer = Automagic4([small, large], lr=1e-3, fused=False, sensitivity_quantile=0.9, beta2=0.0, eps=1e-30)
    small.grad = torch.full((4,), 10.0)
    large.grad = torch.full((96,), 0.1)
    optimizer.step()
    # beta2 = 0: v = g^2 exactly, so each tensor's sensitivity is |g|; 90% of the elements lie at or below 0.1
    ref = optimizer.state[small]["sens_ref"]
    assert ref.item() == pytest.approx(0.1, rel=1e-4)
    assert torch.equal(optimizer.state[large]["sens_ref"], ref)
    assert optimizer._group_ref[0].item() == pytest.approx(0.1, rel=1e-4)


def test_flip_rate_controller_raises_the_lr_on_a_trend_and_lowers_it_on_alternation():
    def final_lr(sign_of_step):
        parameter = nn.Parameter(torch.zeros(256))
        optimizer = Automagic4([parameter], lr=1e-4, fused=False, polarity_history=4)
        for step in range(12):
            parameter.grad = torch.full((256,), sign_of_step(step))
            optimizer.step()
        return optimizer.get_learning_rates()[0]

    assert final_lr(lambda step: 1.0) > 1e-4  # no flips: below the neutral rate, votes up
    assert final_lr(lambda step: 1.0 if step % 2 else -1.0) < 1e-4  # every step flips: above it, votes down


def test_neutral_flip_rate_and_argument_validation():
    parameter = nn.Parameter(torch.zeros(2))
    optimizer = Automagic4([parameter], step_target=0.1)
    assert optimizer._neutral_flip == pytest.approx(0.5 + math.asin(0.05) / math.pi)
    with pytest.raises(ValueError, match="sensitivity_quantile"):
        Automagic4([nn.Parameter(torch.zeros(2))], sensitivity_quantile=0.0)
    with pytest.raises(ValueError, match="step_target"):
        Automagic4([nn.Parameter(torch.zeros(2))], step_target=2.0)


def test_state_round_trip_restores_the_fp32_reference():
    torch.manual_seed(3)
    weight = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16))
    optimizer = Automagic4([weight], lr=1e-3, fused=False)
    x = torch.randn(4, 16)
    for _ in range(3):
        (x @ weight.float().t()).square().mean().backward()
        materialize_stochastic_gradients(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    saved = deepcopy(optimizer.state_dict())

    restored_weight = nn.Parameter(weight.detach().clone())
    restored = Automagic4([restored_weight], lr=5e-4, fused=False)
    restored.load_state_dict(saved)
    state = restored.state[restored_weight]
    assert state["sens_ref"].dtype == torch.float32  # the parent load casts it to the bf16 parameter dtype
    assert restored._group_ref[0] is not None
    assert state["step"] == 3
    before = restored_weight.detach().clone()
    (x @ restored_weight.float().t()).square().mean().backward()
    materialize_stochastic_gradients(restored)
    restored.step()
    assert not torch.equal(restored_weight, before)  # no second measurement step after a resume


def test_late_param_groups_are_measured_then_updated():
    first = nn.Parameter(torch.tensor([0.1, -0.2]))
    late = nn.Parameter(torch.tensor([0.3, -0.4]))
    optimizer = Automagic4([first], lr=1e-3, fused=True)
    optimizer.add_param_group({"params": [late], "lr": 1e-3})
    befores = [first.detach().clone(), late.detach().clone()]
    for _ in range(2):
        (first.square().sum() + late.square().sum()).backward()
        optimizer.step()
    assert optimizer.state[first]["step"] == 2 and optimizer.state[late]["step"] == 2
    assert not torch.equal(first, befores[0]) and not torch.equal(late, befores[1])


def test_factory_builds_automagic4_with_automagic3_fused_rules(monkeypatch):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    parameter = nn.Parameter(torch.ones(2))
    _, optimizer, kwargs = prepare_automagic_optimizer("Automagic4", [parameter], 1e-4, {"step_target": 0.2}, make_args())
    assert isinstance(optimizer, Automagic4) and optimizer.fused is True and kwargs["fused"] is True
    assert optimizer.step_target == 0.2 and uses_fused_backward(optimizer)
    _, optimizer, kwargs = prepare_automagic_optimizer(
        "automagic4", [nn.Parameter(torch.ones(2))], 1e-4, {}, make_args(max_grad_norm=1.0)
    )
    assert optimizer.fused is False and kwargs["fused"] is False  # clipping: Automagic3's safe non-fused mode
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        prepare_automagic_optimizer("Automagic4", [nn.Parameter(torch.ones(2))], 1e-4, {"fused": True}, make_args(gradient_accumulation_steps=2))


@pytest.mark.parametrize(
    "trainer_factory",
    [
        pytest.param(lambda: __import__("musubi_tuner.zimage_train_network", fromlist=["ZImageNetworkTrainer"]).ZImageNetworkTrainer(), id="z-image"),
        pytest.param(lambda: __import__("musubi_tuner.krea2_train_network", fromlist=["Krea2NetworkTrainer"]).Krea2NetworkTrainer(), id="krea-2"),
        pytest.param(lambda: __import__("musubi_tuner.ltx2_train_network", fromlist=["LTX2NetworkTrainer"]).LTX2NetworkTrainer(), id="ltx-2.3"),
    ],
)
def test_trainers_construct_and_step_automagic4(monkeypatch, trainer_factory):
    monkeypatch.setattr("musubi_tuner.optimizers.factory._world_size", lambda: 1)
    trainer = trainer_factory()
    parameter = nn.Parameter(torch.tensor([0.75, -0.5]))
    before = parameter.detach().clone()
    args = make_args(
        optimizer_type="Automagic4",
        optimizer_args=[],
        learning_rate=1e-4,
        lr_scheduler="constant",
        full_finetune=False,
        blocks_to_swap=0,
        fused_backward_pass=False,
    )
    name, _, optimizer, _, _ = trainer.get_optimizer(args, [parameter])
    for _ in range(2):
        parameter.float().square().sum().backward()
        optimizer.step()
    assert name.endswith(".Automagic4")
    assert trainer.is_schedulefree_optimizer(optimizer, args)
    assert not torch.equal(parameter, before)
    assert all(math.isfinite(lr) for lr in get_adaptive_learning_rates(optimizer))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_state_follows_a_parameter_that_block_swap_moved():
    parameter = nn.Parameter(torch.randn(4, 8, device="cuda"))
    optimizer = Automagic4([parameter], lr=1e-3, fused=False)
    for device in ("cuda", "cpu", "cuda"):
        parameter.data = parameter.data.to(device)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        state = optimizer.state[parameter]
        assert all(state[key].device == parameter.device for key in ("lr", "sign_history", "exp_avg_sq_row", "sens_ref"))
