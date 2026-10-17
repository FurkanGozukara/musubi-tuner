"""Wan 2.2 high + low noise training with DoRA: the experts share the LoRA matrices and get one magnitude each.

A DoRA magnitude belongs to the column norms of one base weight, and ComfyUI applies a file's dora_scale as the
absolute magnitudes of the model it is loaded on. So the high-noise expert's magnitudes start from its own weights,
train on its steps only, and are saved to <name>_high_noise.safetensors next to the low-noise <name>.safetensors.
A tiny two-block model stands in for the DiT; its weights are swapped with load_state_dict(assign=True), as
wan_train_network.swap_high_low_weights does. CPU only."""

import argparse
import ast
import os
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.networks.lora import DoRAInputModule, create_network
from musubi_tuner.networks.lora_shared import DORA_EPS

COMFY_BASE = os.environ.get("COMFYUI_WEIGHT_ADAPTER_BASE", r"G:\ComfyUI_V92\ComfyUI\comfy\weight_adapter\base.py")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 32)
        self.proj = nn.Linear(32, 48)
        self.gate = nn.Linear(32, 1)  # one output: stays plain LoRA under use_dora

    def forward(self, x):
        return self.proj(self.to_q(x)) + self.gate(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


def _state(seed: int) -> dict:
    torch.manual_seed(seed)
    return {key: value.detach().clone() for key, value in Model().state_dict().items()}


def _load(model: nn.Module, state: dict) -> None:
    """The expert swap: the other DiT's tensors replace the weights (Wan uses assign=True)."""
    model.load_state_dict({key: value.clone() for key, value in state.items()}, strict=True, assign=True)


def _column_norms(weight: torch.Tensor) -> torch.Tensor:
    return weight.float().norm(dim=0)


def _dense(weight, down, up, magnitude, scale):
    merged = weight.float() + scale * (up.float() @ down.float())
    return merged * (magnitude.reshape(-1).float() / (merged.norm(dim=0) + DORA_EPS)).view(1, -1)


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _setup(perturb=True):
    """A DoRA network over the low-noise weights with the high-noise expert's magnitudes from its own weights."""
    low, high = _state(0), _state(1)
    model = Model()
    _load(model, low)
    torch.manual_seed(2)
    network = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    if perturb:
        with torch.no_grad():
            for lora in network.unet_loras:
                lora.lora_up.weight.normal_(std=0.05)
    assert network.enable_dora_experts(2) == 4  # to_q and proj of both blocks; the one-output gate is plain LoRA
    _load(model, high)
    network.init_dora_expert_from_current_weights(1)
    _load(model, low)
    network.set_dora_expert(0)
    return model, network, low, high


def _doras(network):
    return [lora for lora in network.unet_loras if isinstance(lora, DoRAInputModule)]


def _block_layer(lora_name: str) -> tuple[int, str]:
    parts = lora_name.split("_")  # lora_unet_blocks_<i>_<layer>
    return int(parts[3]), "_".join(parts[4:])


def _weight_key(lora_name: str) -> str:
    block, layer = _block_layer(lora_name)
    return f"blocks.{block}.{layer}.weight"


def test_each_expert_magnitude_starts_from_its_own_weights():
    _, network, low, high = _setup(perturb=False)
    for lora in _doras(network):
        key = _weight_key(lora.lora_name)
        assert torch.allclose(lora.dora_expert_scale(0).reshape(-1), _column_norms(low[key]), rtol=1e-5)
        assert torch.allclose(lora.dora_expert_scale(1).reshape(-1), _column_norms(high[key]), rtol=1e-5)
        assert lora.dora_expert_scale(1)._is_dora_scale and lora.dora_expert_scale(1).requires_grad
    # the network's parameters (what the optimizer sees) hold both experts' magnitudes
    names = {name for name, _ in network.named_parameters()}
    assert {"lora_unet_blocks_0_to_q.dora_scale", "lora_unet_blocks_0_to_q.dora_scale_expert1"} <= names


def test_the_forward_uses_the_selected_experts_magnitude_on_that_experts_weights():
    model, network, low, high = _setup()
    x = torch.randn(3, 32)
    for expert, state in ((1, high), (0, low), (1, high)):
        _load(model, state)
        network.set_dora_expert(expert)
        with torch.no_grad():
            for lora in _doras(network):
                block, layer = _block_layer(lora.lora_name)
                module = getattr(model.blocks[block], layer)
                expected = F.linear(
                    x,
                    _dense(state[_weight_key(lora.lora_name)], lora.lora_down.weight, lora.lora_up.weight,
                           lora.dora_expert_scale(expert), lora.scale),
                    module.bias,
                )
                assert _relerr(module(x), expected) < 1e-5, (expert, lora.lora_name)


def test_a_step_trains_the_shared_matrices_and_only_the_active_experts_magnitude():
    model, network, low, high = _setup()
    network.train()
    x = torch.randn(4, 32)
    for expert, state in ((1, high), (0, low)):
        network.zero_grad(set_to_none=True)
        _load(model, state)
        network.set_dora_expert(expert)
        model(x).square().mean().backward()
        for lora in _doras(network):
            assert lora.lora_up.weight.grad is not None and lora.lora_down.weight.grad is not None
            assert lora.dora_expert_scale(expert).grad is not None
            assert lora.dora_expert_scale(1 - expert).grad is None


def _comfy_weight_decompose():
    if not os.path.isfile(COMFY_BASE):
        return None
    tree = ast.parse(open(COMFY_BASE, encoding="utf-8").read())
    source = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "weight_decompose")
    comfy = types.SimpleNamespace(
        model_management=types.SimpleNamespace(cast_to_device=lambda t, device, dtype, copy=False: t.to(device=device, dtype=dtype))
    )
    namespace = {"torch": torch, "comfy": comfy}
    exec(compile(ast.Module(body=[source], type_ignores=[]), COMFY_BASE, "exec"), namespace)
    return namespace["weight_decompose"]


def test_each_experts_file_holds_its_magnitudes_and_merges_like_comfyui(tmp_path):
    model, network, low, high = _setup()
    path = str(tmp_path / "wan22.safetensors")
    assert network.dora_expert_files(path) == [str(tmp_path / "wan22_high_noise.safetensors")]
    network.save_weights(path, torch.bfloat16, {"ss_network_dim": "4"})
    companion = network.dora_expert_files(path)[0]
    files = {0: load_file(path), 1: load_file(companion)}
    for index, file in ((0, path), (1, companion)):
        with safe_open(file, "pt") as handle:
            metadata = handle.metadata()
        assert metadata["ss_dora_expert"] == ("low_noise", "high_noise")[index] and metadata["ss_network_dim"] == "4"
        assert "sshs_model_hash" in metadata
    assert files[0].keys() == files[1].keys()
    assert not any("expert1" in key for key in files[0])  # the standard layout: one dora_scale per module
    weight_decompose = _comfy_weight_decompose()
    for lora in _doras(network):
        key = f"{lora.lora_name}.dora_scale"
        assert torch.equal(files[0][f"{lora.lora_name}.lora_up.weight"], files[1][f"{lora.lora_name}.lora_up.weight"])
        for expert, state in ((0, low), (1, high)):
            saved = files[expert]
            assert saved[key].dtype == torch.float32
            assert torch.equal(saved[key], lora.dora_expert_scale(expert).detach().float())
            base = state[_weight_key(lora.lora_name)]
            down = saved[f"{lora.lora_name}.lora_down.weight"].float()
            up = saved[f"{lora.lora_name}.lora_up.weight"].float()
            alpha = float(saved[f"{lora.lora_name}.alpha"])
            _load(model, state)
            network.set_dora_expert(expert)
            trained = lora.get_merged_weight()
            assert _relerr(_dense(base, down, up, saved[key], alpha / down.shape[0]), trained) < 1e-2  # bf16 down / up
            if weight_decompose is not None:
                comfy = weight_decompose(saved[key], base.clone(), up @ down, alpha / down.shape[0], 1.0, torch.float32, lambda a: a)
                assert _relerr(comfy, trained) < 1e-2


def test_the_pair_loads_back_for_continued_training(tmp_path):
    _, network, low, _ = _setup()
    path = str(tmp_path / "wan22.safetensors")
    network.save_weights(path, None, {})
    model = Model()
    _load(model, low)
    fresh = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model, use_dora="True")
    fresh.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    fresh.load_weights(path)
    fresh.enable_dora_experts(2)
    assert fresh.load_dora_expert_weights(1, load_file(fresh.dora_expert_files(path)[0])) == 4
    trained = {lora.lora_name: lora for lora in _doras(network)}
    for lora in _doras(fresh):
        for expert in (0, 1):
            assert torch.equal(lora.dora_expert_scale(expert), trained[lora.lora_name].dora_expert_scale(expert))


def test_without_experts_nothing_changes(tmp_path):
    model = Model()
    lora_network = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model)
    lora_network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    assert lora_network.enable_dora_experts(2) == 0  # plain LoRA: one file for both experts
    assert lora_network.dora_expert_files(str(tmp_path / "a.safetensors")) == []
    dora_network = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], Model(), use_dora="True")
    assert dora_network.dora_expert_files(str(tmp_path / "a.safetensors")) == []


def test_output_axis_dora_is_refused_for_experts():
    network = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], Model(), use_dora="True", dora_axis="output")
    with pytest.raises(ValueError, match="input-axis"):
        network.enable_dora_experts(2)


def test_the_wan_trainer_switches_magnitudes_with_the_expert(tmp_path):
    """wan_train_network: on_network_applied adds the high-noise magnitudes from the high-noise weights, every expert
    swap selects that expert's magnitudes, and continued training reads the companion file."""
    from musubi_tuner.wan_train_network import WanNetworkTrainer

    low, high = _state(0), _state(1)
    model = Model()
    _load(model, low)
    network = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)

    trainer = WanNetworkTrainer()
    trainer.high_low_training = True
    trainer._dora_experts_expected = True
    trainer.blocks_to_swap = 0
    trainer.dit_inactive_state_dict = {key: value.clone() for key, value in high.items()}
    trainer.current_model_is_high_noise = trainer.next_model_is_high_noise = False
    args = argparse.Namespace(compile=False, offload_inactive_dit=False, network_weights=None)
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))

    trainer.on_network_applied(args, accelerator, model, network)
    assert not trainer.current_model_is_high_noise and trainer._dora_expert_network is network
    lora = next(l for l in _doras(network) if l.lora_name == "lora_unet_blocks_0_to_q")
    assert torch.allclose(lora.dora_expert_scale(1).reshape(-1), _column_norms(high["blocks.0.to_q.weight"]), rtol=1e-5)
    assert torch.equal(model.blocks[0].to_q.weight, low["blocks.0.to_q.weight"]) and lora._dora_expert == 0

    trainer.next_model_is_high_noise = True
    trainer.swap_high_low_weights(args, accelerator, model)
    assert torch.equal(model.blocks[0].to_q.weight, high["blocks.0.to_q.weight"]) and lora._dora_expert == 1
    trainer.next_model_is_high_noise = False
    trainer.swap_high_low_weights(args, accelerator, model)
    assert lora._dora_expert == 0

    # continued training: <network_weights>_high_noise.safetensors gives the high-noise magnitudes back
    with torch.no_grad():
        lora.dora_expert_scale(1).mul_(1.5)
    path = str(tmp_path / "run.safetensors")
    network.save_weights(path, None, {})
    model2 = Model()
    _load(model2, low)
    network2 = create_network(["Block"], "lora_unet", 1.0, 4, 2.0, None, [], model2, use_dora="True")
    network2.apply_to(None, model2, apply_text_encoder=False, apply_unet=True)
    network2.load_weights(path)
    trainer2 = WanNetworkTrainer()
    trainer2.high_low_training = True
    trainer2._dora_experts_expected = True
    trainer2.blocks_to_swap = 0
    trainer2.dit_inactive_state_dict = {key: value.clone() for key, value in high.items()}
    trainer2.current_model_is_high_noise = trainer2.next_model_is_high_noise = False
    trainer2.on_network_applied(argparse.Namespace(compile=False, offload_inactive_dit=False, network_weights=path), accelerator, model2, network2)
    lora2 = next(l for l in _doras(network2) if l.lora_name == "lora_unet_blocks_0_to_q")
    assert torch.equal(lora2.dora_expert_scale(1), lora.dora_expert_scale(1))
    assert torch.equal(lora2.dora_expert_scale(0), lora.dora_expert_scale(0))
