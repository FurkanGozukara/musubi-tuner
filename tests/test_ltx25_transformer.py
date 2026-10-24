"""LTX-2.5 transformer on the CPU with tiny configs: capability detection, checkpoint loading through the trainer's
loader, version detection, and LoRA / DoRA on the bias-free LTX-2.5 feed-forward layers."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.ltx2_model_loading import (
    detect_ltx_checkpoint_version,
    infer_ltx_version_from_checkpoint_config,
    load_ltx2_model,
)
from musubi_tuner.ltx_2.model.transformer.ltx25_compat import (
    apply_generated_keyframe_embedding,
    build_generated_keyframe_mask,
    enrich_config_from_state_dict,
)
from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
from musubi_tuner.networks import lora_ltx2
from musubi_tuner.networks.lora_ltx2 import LTX2Wrapper

# the architecture constants the configurators assert
_CONSTANTS = {
    "dropout": 0.0,
    "attention_bias": True,
    "num_vector_embeds": None,
    "activation_fn": "gelu-approximate",
    "num_embeds_ada_norm": 1000,
    "use_linear_projection": False,
    "only_cross_attention": False,
    "cross_attention_norm": True,
    "double_self_attention": False,
    "upcast_attention": False,
    "standardization_norm": "rms_norm",
    "norm_elementwise_affine": False,
    "qk_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "use_audio_video_cross_attention": True,
    "share_ff": False,
    "av_cross_ada_norm": True,
    "use_middle_indices_grid": True,
}


def _transformer_config(**overrides) -> dict:
    config = dict(
        _CONSTANTS,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=8,
        out_channels=8,
        num_layers=2,
        cross_attention_dim=16,
        caption_channels=16,
        attention_type="pytorch",
        # LTX-2.3 / 2.5 architecture switches
        cross_attention_adaln=True,
        caption_proj_before_connector=True,
        apply_gated_attention=True,
        rope_type="split",
        frequencies_precision="float64",
        positional_embedding_max_pos=[20, 2048, 2048],
    )
    config.update(overrides)
    return config


def _build(transformer_config: dict) -> torch.nn.Module:
    torch.manual_seed(0)
    model = LTXVideoOnlyModelConfigurator.from_config({"transformer": transformer_config})
    with torch.no_grad():
        for _name, param in model.named_parameters():
            param.normal_(0.0, 0.05)
    return model.float()


def _ltx25_model() -> torch.nn.Module:
    return _build(_transformer_config(ff_bias=False, use_keyframes_abs_pos_embedding=True))


def _write_checkpoint(path, model: torch.nn.Module, config: dict, metadata_extra: dict | None = None) -> None:
    tensors = {f"model.diffusion_model.{k}": v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    metadata = {"config": json.dumps({"transformer": config})}
    metadata.update(metadata_extra or {})
    save_file(tensors, str(path), metadata=metadata)


def _inner(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", model)


def test_capabilities_come_from_the_tensor_names():
    ltx25_keys = [
        "model.diffusion_model.keyframes_abs_pos_embedding",
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
        "model.diffusion_model.transformer_blocks.0.ff.net.2.weight",
        "model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.weight",
        "model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.bias",
    ]
    enriched = enrich_config_from_state_dict({"transformer": {}}, ltx25_keys)["transformer"]
    assert enriched["ff_bias"] is False
    assert enriched["audio_ff_bias"] is True
    assert enriched["use_keyframes_abs_pos_embedding"] is True
    assert enriched["native_dtype_adaln"] is True

    ltx23_keys = [
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias",
    ]
    enriched = enrich_config_from_state_dict({"transformer": {}}, ltx23_keys)["transformer"]
    assert enriched == {
        "ff_bias": True,
        "audio_ff_bias": True,
        "use_prompt_adaln_single": True,
        "use_keyframes_abs_pos_embedding": False,
        "native_dtype_adaln": False,
    }
    # a value the config states wins over the tensor names
    stated = enrich_config_from_state_dict({"transformer": {"ff_bias": True}}, ltx25_keys)["transformer"]
    assert stated["ff_bias"] is True


def test_ltx25_checkpoint_loads_through_the_trainer_loader(tmp_path):
    """The official 2.5 metadata need not state the 2.5 switches: load_ltx2_model reads them off the tensor names."""
    reference = _ltx25_model()
    path = tmp_path / "ltx-2.5-tiny-transformer.safetensors"
    _write_checkpoint(path, reference, _transformer_config(), {"model_version": "2.5.0"})

    loaded = _inner(load_ltx2_model(str(path), device="cpu", load_device="cpu", torch_dtype=torch.float32))
    assert loaded.keyframes_abs_pos_embedding is not None
    assert loaded.native_dtype_adaln is True
    for block in loaded.transformer_blocks:
        assert block.ff.net[0].proj.bias is None and block.ff.net[2].bias is None
    expected = reference.state_dict()
    actual = loaded.state_dict()
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key].float(), value.float()), key

    assert detect_ltx_checkpoint_version(str(path))[0] == "2.5"
    # the config alone reads as 2.3 (cross-attention AdaLN): the tensor / model_version markers make it 2.5
    assert infer_ltx_version_from_checkpoint_config({"transformer": _transformer_config()})[0] == "2.3"


def test_ltx23_checkpoint_loads_as_before(tmp_path):
    reference = _build(_transformer_config())
    path = tmp_path / "ltx-2.3-tiny.safetensors"
    _write_checkpoint(path, reference, _transformer_config())
    loaded = _inner(load_ltx2_model(str(path), device="cpu", load_device="cpu", torch_dtype=torch.float32))
    assert loaded.keyframes_abs_pos_embedding is None
    assert loaded.native_dtype_adaln is False
    assert all(block.ff.net[0].proj.bias is not None for block in loaded.transformer_blocks)
    assert detect_ltx_checkpoint_version(str(path))[0] == "2.3"


def test_generated_keyframe_embedding_marks_only_the_suffix():
    mask = build_generated_keyframe_mask(2, 5, 2, torch.device("cpu"))
    assert mask.tolist() == [[False, False, False, True, True]] * 2
    tokens = torch.zeros(2, 5, 4)
    out = apply_generated_keyframe_embedding(tokens, mask, torch.ones(1, 4))
    assert torch.equal(out[:, :3], torch.zeros(2, 3, 4)) and torch.equal(out[:, 3:], torch.ones(2, 2, 4))


def _inputs():
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 8, 2, 2, 2, generator=generator)
    context = torch.randn(1, 5, 16, generator=generator)
    mask = torch.ones(1, 5, dtype=torch.long)
    return latents, context, mask


def _forward(wrapper, latents, context, mask):
    return wrapper(latents, timestep=torch.tensor([0.6]), context=context, attention_mask=mask, frame_rate=24)


@pytest.mark.parametrize(
    "network_args",
    [
        {},  # LoRA
        {"use_dora": True, "dora_axis": "input"},  # input-axis DoRA (the fork's default DoRA, ComfyUI dora_scale)
        {"use_dora": True, "dora_axis": "output"},  # output-axis DoRA (PEFT magnitude)
    ],
    ids=["lora", "dora_input", "dora_output"],
)
def test_lora_and_dora_train_the_bias_free_ltx25_feed_forward(network_args):
    model = _ltx25_model()
    model.requires_grad_(False)
    wrapper = LTX2Wrapper(model)
    latents, context, mask = _inputs()
    with torch.no_grad():
        base_out = _forward(wrapper, latents, context, mask)

    network = lora_ltx2.create_arch_network(
        1.0, 4, 4, None, [], wrapper, lora_target_preset="video_sa_ca_ff", **network_args
    )
    network.apply_to(None, wrapper, apply_text_encoder=False, apply_unet=True)
    ff_modules = [lora for lora in network.unet_loras if ".ff.net" in lora.lora_name.replace("_", ".") or "ff_net" in lora.lora_name]
    assert ff_modules, [lora.lora_name for lora in network.unet_loras]
    for lora in ff_modules:
        org = lora.org_module_ref[0] if hasattr(lora, "org_module_ref") else None
        if org is not None:
            assert org.bias is None
    expected_type = {"input": "DoRAInputModule", "output": "DoRAModule"}.get(network_args.get("dora_axis"), "LoRAModule")
    assert {type(lora).__name__ for lora in ff_modules} == {expected_type}

    # B starts at zero: the adapted model reproduces the base model exactly at initialisation
    with torch.no_grad():
        adapted_out = _forward(wrapper, latents, context, mask)
    assert torch.allclose(adapted_out, base_out, atol=1e-6, rtol=1e-5)

    network.train()
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1.0)
    for _step in range(2):
        optimizer.zero_grad()
        loss = _forward(wrapper, latents, context, mask).float().pow(2).mean()
        loss.backward()
        for lora in ff_modules:
            grads = [p.grad for p in lora.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), lora.lora_name
        optimizer.step()
    with torch.no_grad():
        trained_out = _forward(wrapper, latents, context, mask)
    assert torch.isfinite(trained_out).all()
    assert not torch.allclose(trained_out, base_out)

    # the saved weights carry the DoRA magnitude of every bias-free FF layer
    state = network.state_dict()
    if network_args.get("dora_axis") == "input":
        assert all(f"{lora.lora_name}.dora_scale" in state for lora in ff_modules)


def test_input_axis_dora_on_every_linear_of_the_ltx25_av_blocks():
    """The ``full`` preset on an audio-video 2.5 model: bias-free video FF, the audio FF (with biases), self /
    cross / audio-video attention and the attention gates all train with input-axis DoRA."""
    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXModelConfigurator

    config = _transformer_config(
        ff_bias=False,
        use_keyframes_abs_pos_embedding=True,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_cross_attention_dim=16,
        av_ca_timestep_scale_multiplier=1000.0,
        audio_positional_embedding_max_pos=[20],
    )
    torch.manual_seed(0)
    model = LTXModelConfigurator.from_config({"transformer": config})
    with torch.no_grad():
        for _name, param in model.named_parameters():
            param.normal_(0.0, 0.05)
    model = model.float().requires_grad_(False)
    block = model.transformer_blocks[0]
    assert block.ff.net[0].proj.bias is None and block.audio_ff.net[0].proj.bias is not None

    wrapper = LTX2Wrapper(model)
    generator = torch.Generator().manual_seed(2)
    video = torch.randn(1, 8, 2, 2, 2, generator=generator)
    audio = torch.randn(1, 4, 4, 2, generator=generator)  # channels x mel = audio_in_channels
    context = torch.randn(1, 5, 32, generator=generator)  # video + audio context
    mask = torch.ones(1, 5, dtype=torch.long)

    def forward():
        out = wrapper([video, audio], timestep=torch.tensor([0.6]), context=context, attention_mask=mask, frame_rate=24)
        return out[0].float().pow(2).mean() + out[1].float().pow(2).mean()

    network = lora_ltx2.create_arch_network(
        1.0, 4, 4, None, [], wrapper, audio_video=True, lora_target_preset="full", use_dora=True, dora_axis="input"
    )
    network.apply_to(None, wrapper, apply_text_encoder=False, apply_unet=True)
    names = [lora.lora_name for lora in network.unet_loras]
    for fragment in ("ff_net_0_proj", "audio_ff_net_0_proj", "attn1_to_q", "audio_attn1_to_q", "to_gate_logits"):
        assert any(fragment in name for name in names), (fragment, names)
    assert any("audio_to_video_attn" in name or "video_to_audio_attn" in name for name in names), names

    network.train()
    loss = forward()
    loss.backward()
    missing = [
        lora.lora_name
        for lora in network.unet_loras
        if lora.dora_scale.grad is None or not torch.isfinite(lora.dora_scale.grad).all()
    ]
    assert not missing, missing
