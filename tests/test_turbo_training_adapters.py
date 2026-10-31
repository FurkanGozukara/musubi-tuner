"""Turbo training adapters (Qwen-Image 2.1 Turbo, Krea 2 Turbo): key conversion, and samples without the merged adapter."""

import argparse

import pytest
import torch
from torch import nn

from musubi_tuner.networks import lora, lora_krea2, lora_qwen_image_21
from musubi_tuner.training.base_weights_sampling import SampleBaseWeights, sample_multiplier_deltas


def _aitk_lora(modules: dict[str, tuple[int, int]], rank: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    """An ai-toolkit LoRA file's tensors: diffusion_model.<module>.lora_A / lora_B, no alpha (scale 1)."""
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for name, (out_features, in_features) in modules.items():
        weights[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, in_features, generator=generator)
        weights[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(out_features, rank, generator=generator)
    return weights


def test_qwen21_adapter_conversion_splits_the_fused_mlp_exactly():
    adapter = _aitk_lora({"transformer_blocks.0.img_mlp.gate_up": (12, 8), "transformer_blocks.0.attn.to_q": (8, 8)})
    converted = lora_qwen_image_21.convert_weight_keys(adapter)
    stem = "lora_unet_transformer_blocks_0_img_mlp_"
    assert {key.split(".", 1)[0] for key in converted} == {stem + "gate_layer", stem + "proj", "lora_unet_transformer_blocks_0_attn_to_q"}
    fused = adapter["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"] @ adapter[
        "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"
    ]
    for part, rows in (("gate_layer", slice(0, 6)), ("proj", slice(6, 12))):
        delta = converted[f"{stem}{part}.lora_up.weight"] @ converted[f"{stem}{part}.lora_down.weight"]
        assert torch.equal(delta, fused[rows])
        assert float(converted[f"{stem}{part}.alpha"]) == 4  # no alpha in the file: alpha = rank, scale 1
    assert lora_qwen_image_21.convert_weight_keys(converted) is None  # already in the trainer's names


def test_qwen21_adapter_conversion_refuses_a_fused_dora_magnitude():
    adapter = _aitk_lora({"transformer_blocks.0.img_mlp.gate_up": (12, 8)})
    adapter["diffusion_model.transformer_blocks.0.img_mlp.gate_up.dora_scale"] = torch.ones(1, 8)
    with pytest.raises(ValueError, match="cannot be split"):
        lora_qwen_image_21.convert_weight_keys(adapter)


@pytest.mark.parametrize(
    "base_weights,train,sample,expected",
    [
        (None, None, [0.0], []),
        (["a"], None, None, []),
        (["a"], None, [0.0], [("a", -1.0)]),
        (["a", "b"], [0.5], [0.5, 0.0], [("b", -1.0)]),
        (["a", "b"], [0.5, 2.0], [1.0], [("a", 0.5)]),
    ],
)
def test_sample_multiplier_deltas(base_weights, train, sample, expected):
    args = argparse.Namespace(base_weights=base_weights, base_weights_multiplier=train, base_weights_sample_multiplier=sample)
    assert sample_multiplier_deltas(args) == expected


def test_sample_multiplier_deltas_refuses_extra_values():
    args = argparse.Namespace(base_weights=["a"], base_weights_multiplier=None, base_weights_sample_multiplier=[0.0, 1.0])
    with pytest.raises(ValueError):
        sample_multiplier_deltas(args)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(16, 16, bias=False)
        self.up = nn.Linear(16, 24, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for block in self.blocks:
            x = x + block.up(block.wq(x))[..., :16]
        return x


def _merge(model: nn.Module, weights: dict[str, torch.Tensor], multiplier: float = 1.0) -> None:
    network = lora_krea2.create_arch_network_from_weights(multiplier, weights, unet=model, for_inference=True)
    network.merge_to(None, model, weights, torch.float32, "cpu")


@pytest.mark.parametrize("use_dora", [False, True])
def test_samples_leave_a_merged_training_adapter_out(use_dora):
    torch.manual_seed(0)
    from musubi_tuner.convert_lora import convert_from_diffusers

    adapter = convert_from_diffusers(
        "lora_unet_", _aitk_lora({f"blocks.{i}.{name}": shape for i in range(2) for name, shape in (("wq", (16, 16)), ("up", (24, 16)))}, seed=1)
    )
    turbo = _Model()
    layers = dict(turbo.named_modules())
    for key in list(adapter):
        if key.endswith("lora_up.weight"):
            # an adapter-sized delta: ostris' Turbo adapters move the weights by 0.1-0.3% (Frobenius), the column norms by
            # 0.003% on average (dont_upload/vram_lab/turbo_adapter_colnorm.py)
            name = key.split(".", 1)[0]
            weight = layers[name.removeprefix("lora_unet_").replace("blocks_", "blocks.").replace("_", ".", 1)].weight
            delta = adapter[key] @ adapter[name + ".lora_down.weight"]
            adapter[key] = adapter[key] * (0.003 * weight.norm() / delta.norm())
    merged = _Model()
    merged.load_state_dict(turbo.state_dict())
    _merge(merged, adapter)  # --base_weights: merged before the trained network exists

    # the trained network (LoRA / input-axis DoRA) on the merged model, its up matrices trained to something
    trained = lora_krea2.create_arch_network(1.0, 4, 4, None, [], merged, **({"use_dora": "True"} if use_dora else {}))
    trained.apply_to(None, merged, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for module in trained.unet_loras:
            module.lora_up.weight.normal_(std=0.05)

    samples = SampleBaseWeights([("adapter.safetensors", -1.0)])
    samples.build(lora_krea2, merged, lambda path: dict(adapter))
    x = torch.randn(3, 16)
    with torch.no_grad():
        training_output = merged(x)
        samples.before_sampling(torch.device("cpu"))
        sample_output = merged(x)
        samples.after_sampling()
        assert torch.equal(merged(x), training_output)  # off again after sampling

        # what ComfyUI computes: the trained file applied to the plain Turbo weights
        reference = _Model()
        reference.load_state_dict(turbo.state_dict())
        saved = {key: value.clone() for key, value in trained.state_dict().items()}
        reloaded = lora_krea2.create_arch_network_from_weights(1.0, saved, unet=reference, for_inference=True)
        reloaded.apply_to(None, reference, apply_text_encoder=False, apply_unet=True)
        reloaded.load_state_dict(saved, strict=False)
        expected = reference(x)
    assert not torch.allclose(training_output, expected, rtol=0, atol=1e-6)  # the adapter matters while training
    if use_dora:
        # DoRA learned its magnitudes on the merged weights: the samples differ from ComfyUI's by the adapter's share of
        # the column norms, far below the trained change itself
        trained_change = (training_output - x).norm()
        assert (sample_output - expected).norm() < 1e-3 * trained_change
    else:
        assert torch.allclose(sample_output, expected, rtol=0, atol=1e-5)


def test_sample_base_weights_refuses_an_adapter_without_matching_layers():
    model = _Model()
    samples = SampleBaseWeights([("other.safetensors", -1.0)])
    weights = {"lora_unet_nothing_here.lora_down.weight": torch.zeros(4, 16), "lora_unet_nothing_here.lora_up.weight": torch.zeros(16, 4)}
    with pytest.raises((ValueError, RuntimeError)):
        samples.build(lora_krea2, model, lambda path: weights)
    assert isinstance(lora.LoRANetwork, type)
