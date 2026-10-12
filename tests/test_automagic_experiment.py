"""AutomagicEXPERIMENT (optimizers/automagic_experiment.py): the int8 ConvRot second moment of 2D tensors, beta1 = 0,
bias correction, checkpoints and the Triton quantizer (convrot_state.py)."""

from copy import deepcopy

import pytest
import torch

from musubi_tuner.optimizers import Automagic3, AutomagicEXPERIMENT
from musubi_tuner.optimizers.convrot_state import ConvRot8State, quantize_int8_rows, quantize_int8_rows_fused, rotate


def _quadratic_run(optimizer_class, steps=60, dtype=torch.float32, **kwargs):
    torch.manual_seed(0)
    target = torch.randn(16, 64)
    weight = torch.nn.Parameter(torch.zeros(16, 64, dtype=dtype))
    optimizer = optimizer_class([weight], lr=1e-2, **kwargs)
    losses = []
    for _ in range(steps):
        loss = (weight.float() - target).square().mean()
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return losses, optimizer, weight


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_converges_on_a_2d_tensor_with_int8_state(fused, dtype):
    losses, optimizer, weight = _quadratic_run(AutomagicEXPERIMENT, fused=fused, dtype=dtype)
    assert losses[-1] < losses[0] * 0.5
    state = optimizer.state[weight]
    assert isinstance(state["exp_avg_sq"], ConvRot8State) and state["exp_avg_sq"].quantized.dtype == torch.int8
    assert state["step"] == 60 and all(torch.isfinite(torch.as_tensor(lr)) for lr in optimizer.get_learning_rates())


def test_one_dimensional_tensors_keep_an_fp32_moment():
    bias = torch.nn.Parameter(torch.ones(8))
    optimizer = AutomagicEXPERIMENT([bias], lr=1e-3, fused=False)
    bias.square().sum().backward()
    optimizer.step()
    moment = optimizer.state[bias]["exp_avg_sq"]
    assert isinstance(moment, torch.Tensor) and moment.dtype == torch.float32


def test_momentum_is_refused():
    with pytest.raises(ValueError, match="beta1 = 0"):
        AutomagicEXPERIMENT([torch.nn.Parameter(torch.ones(2, 2))], betas=(0.9, 0.999))
    optimizer = AutomagicEXPERIMENT([torch.nn.Parameter(torch.ones(2, 2))], beta2=0.99)  # Automagic3 spelling
    assert optimizer.param_groups[0]["beta2"] == 0.99


def test_checkpoint_round_trip_keeps_the_int8_moment_and_never_touches_live_state():
    _losses, optimizer, weight = _quadratic_run(AutomagicEXPERIMENT, steps=12, fused=False)
    live = optimizer.state[weight]["exp_avg_sq"]
    saved = deepcopy(optimizer.state_dict())
    assert optimizer.state[weight]["exp_avg_sq"] is live and isinstance(live, ConvRot8State)
    assert isinstance(saved["state"][0]["exp_avg_sq"], dict)

    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored = AutomagicEXPERIMENT([restored_weight], lr=5e-2, fused=False)
    restored.load_state_dict(saved)
    moment = restored.state[restored_weight]["exp_avg_sq"]
    assert isinstance(moment, ConvRot8State)
    assert torch.equal(moment.quantized, live.quantized) and moment.scale.dtype == torch.float32
    assert torch.allclose(moment.scale, live.scale)
    assert restored.get_learning_rates() == pytest.approx(optimizer.get_learning_rates())
    (restored_weight - 1).square().mean().backward()
    restored.step()
    assert restored.state[restored_weight]["step"] == 13


def test_lr_controller_moves_like_automagic3():
    """Same controller: on the same problem both adapt the lr away from a too-small start."""
    _, experiment, _ = _quadratic_run(AutomagicEXPERIMENT, steps=80, fused=False)
    _, automagic3, _ = _quadratic_run(Automagic3, steps=80, fused=False)
    assert experiment.get_learning_rates()[0] != pytest.approx(1e-2)
    assert automagic3.get_learning_rates()[0] != pytest.approx(1e-2)


def test_int8_state_reconstructs_within_quantization_error():
    torch.manual_seed(1)
    values = torch.rand(32, 256)
    state = ConvRot8State(values)
    error = (state.dequantize() - values).abs().max()
    assert error < values.abs().max() / 64
    assert torch.allclose(rotate(rotate(values, 64), 64), values, atol=1e-5)  # self-inverse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton kernel)")
def test_triton_quantizer_matches_torch():
    torch.manual_seed(2)
    x = torch.randn(48, 1000, device="cuda")
    q_fused, s_fused = quantize_int8_rows_fused(x)
    q_ref, s_ref = quantize_int8_rows(x)
    assert torch.allclose(s_fused, s_ref)
    assert (q_fused.int() - q_ref.int()).abs().max() <= 1  # one-code ties (division rounding) at most


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_state_follows_a_block_swapped_parameter():
    weight = torch.nn.Parameter(torch.ones(8, 16, device="cuda"))
    optimizer = AutomagicEXPERIMENT([weight], lr=1e-3, fused=False)
    weight.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    weight.data = weight.data.cpu()  # what block swap does between steps
    weight.square().sum().backward()
    optimizer.step()
    assert optimizer.state[weight]["exp_avg_sq"].device.type == "cpu" and optimizer.state[weight]["step"] == 2
