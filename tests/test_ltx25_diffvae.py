"""LTX-2.5 DiffVAE (diffusion video decoder) on the CPU with a tiny config: decode, tiling, loading a checkpoint in the
release key layout through the trainer's VAE loader, and the tiling layout training samples hand each VAE."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.ltx_2.model.video_vae import DiffusionVideoDecoder, TilingConfig
from musubi_tuner.ltx_2.model.video_vae.diffusion_model_configurator import (
    _build_diffusion_video_decoder,
    is_diffusion_video_vae,
    read_vae_checkpoint_config,
)
from musubi_tuner.ltx_2.model.video_vae.transformer import DiffVAEMode, apply_diffvae_mode
from musubi_tuner.ltx_2.tiling import DimensionSizeConfig, TileSizeConfig

# the release layout of ComfyUI's LTX_24_VAE_CONFIG, scaled down (head_dim 16: the RoPE split needs it)
TINY_VAE_CONFIG = {
    "_class_name": "CausalDiffusionVAE",
    "dims": 3,
    "model_output_type": "x0",
    "decoder": {
        "in_channels": 8,
        "out_channels": 3,
        "patch_size": 4,
        "head_dim": 16,
        "t_emb_dim": 16,
        "stage_channels": [128, 64, 32, 32, 16],
        "stage_depths": [1, 1, 1, 1, 1],
        "stage_kernels": [[3, 3, 3]] * 5,
        "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
        "stage5_kernel": [3, 3, 3],
        "timestep_scale_multiplier": 1000.0,
        "default_num_inference_steps": 1,
    },
}


def _decoder() -> DiffusionVideoDecoder:
    torch.manual_seed(0)
    decoder = _build_diffusion_video_decoder(TINY_VAE_CONFIG)
    with torch.no_grad():
        for _name, param in decoder.named_parameters():
            param.normal_(0.0, 0.05)
        # the latent statistics are torch.empty buffers until a checkpoint fills them
        for name, buffer in decoder.named_buffers():
            if name.startswith("per_channel_statistics."):
                buffer.copy_(torch.linspace(0.5, 1.5, buffer.numel()).reshape(buffer.shape).to(buffer.dtype))
    return apply_diffvae_mode(decoder, DiffVAEMode.CHUNKED_EAGER).eval()


def _latent() -> torch.Tensor:
    return torch.randn(1, 8, 2, 2, 3, generator=torch.Generator().manual_seed(1))


def _decode(decoder, latent, tiling=None):
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        if tiling is None:
            return decoder(latent, generator=generator)
        return torch.cat(list(decoder.tiled_decode(latent, tiling, generator=generator)), dim=2)


def test_tiny_diffvae_decodes_and_tiles():
    """The DiffVAE decodes by denoising pixel noise, so a tiled decode draws other noise than a full one: the layouts
    are checked for shape, determinism under one generator and closeness, not equality."""
    decoder = _decoder()
    latent = _latent()
    full = _decode(decoder, latent)
    assert full.shape == (1, 3, 9, 64, 96)  # 8x temporal (causal first frame), 32x spatial
    assert torch.isfinite(full).all()
    assert torch.equal(_decode(decoder, latent), full)

    explicit = TileSizeConfig(
        frames=DimensionSizeConfig(),
        height=DimensionSizeConfig(tile_size=64, overlap=32),
        width=DimensionSizeConfig(tile_size=64, overlap=32),
    )
    auto = decoder.recommended_tiling_config(height=64, width=96, num_frames=9, free_bytes=64 << 30)
    assert isinstance(auto, TileSizeConfig)
    for layout in (explicit, auto):
        tiled = _decode(decoder, latent, layout)
        assert tiled.shape == full.shape and torch.isfinite(tiled).all()
        assert torch.equal(_decode(decoder, latent, layout), tiled)
        assert float((tiled - full).norm() / full.norm()) < 0.5


def _release_state_dict(decoder: DiffusionVideoDecoder, gate: torch.Tensor) -> dict:
    """The decoder's tensors the way a released DiffVAE file stores them (bare ``decoder.`` keys, fused qkv, the
    t-embedder MLP names, a static gate the loader folds into the Linear before it)."""
    out = {}
    state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    fused = {}
    for key, value in state.items():
        if key.startswith("per_channel_statistics."):
            out[key] = value
            continue
        if ".qkv.to_" in key:
            prefix, rest = key.split(".qkv.to_", 1)
            part, leaf = rest.split(".", 1)
            fused.setdefault((prefix, leaf), {})[part] = value
            continue
        name = key.replace("t_embedder.timestep_embedder.linear_1.", "t_embedder.mlp.0.").replace(
            "t_embedder.timestep_embedder.linear_2.", "t_embedder.mlp.2."
        )
        out["decoder." + name] = value
    for (prefix, leaf), parts in fused.items():
        out[f"decoder.{prefix}.qkv.{leaf}"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
    # a gated block: the file holds proj / gate separately, the model the product
    out["decoder.diff_blocks.0.gate_msa"] = gate
    return {k: v.contiguous() for k, v in out.items()}


def test_trainer_loads_a_released_diffvae_file(tmp_path):
    from musubi_tuner.ltx2_sampling import _is_diffusion_vae, _vae_tiling_config
    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    reference = _decoder()
    gate = torch.linspace(0.5, 1.5, reference.diff_blocks[0].attn.proj.weight.shape[0])
    path = tmp_path / "ltx-2.5-tiny-video-vae.safetensors"
    save_file(_release_state_dict(reference, gate), str(path), metadata={"config": json.dumps({"vae": TINY_VAE_CONFIG})})
    assert is_diffusion_video_vae(str(path))
    assert read_vae_checkpoint_config(str(path))["vae"]["_class_name"] == "CausalDiffusionVAE"

    trainer = LTX2NetworkTrainer.__new__(LTX2NetworkTrainer)
    trainer._latent_norm_cache = {}
    vae = trainer._load_vae_impl(SimpleNamespace(), torch.float32, str(path))
    assert _is_diffusion_vae(vae) and isinstance(vae.decoder, DiffusionVideoDecoder)

    # the loader folded the gate into the projection, as the release format requires
    with torch.no_grad():
        reference.diff_blocks[0].attn.proj.weight.mul_(gate[:, None])
        reference.diff_blocks[0].attn.proj.bias.mul_(gate)
    loaded = vae.decoder.state_dict()
    for key, value in reference.state_dict().items():
        assert torch.allclose(loaded[key].float(), value.float()), key

    latent = _latent()
    expected = _decode(reference, latent)
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    decoded = vae.decode([latent[0]])[0]
    torch.random.set_rng_state(generator_state)
    assert decoded.shape == expected.shape[1:]

    # training samples: the DiffVAE gets its own tile layout, the convolutional VAE keeps TilingConfig
    explicit = _vae_tiling_config(vae, 64, 32, 0, 0, latents=None)
    assert isinstance(explicit, TileSizeConfig) and not explicit.frames.is_tiled()
    temporal = _vae_tiling_config(vae, 64, 32, 16, 8)
    assert isinstance(temporal, TileSizeConfig) and temporal.frames.tile_size == 16
    # the auto layout budgets free CUDA memory; on the CPU it falls back to the configured tiles
    fallback = _vae_tiling_config(vae, 64, 32, 0, 0, latents=latent)
    assert isinstance(fallback, TileSizeConfig) and fallback.height.tile_size == 64
    tiled = vae.tiled_decode(latent[0], fallback)
    assert tiled.shape == (1, 3, 9, 64, 96) and torch.isfinite(tiled).all()


def test_convolutional_vae_keeps_its_tiling_config():
    from musubi_tuner.ltx2_sampling import _is_diffusion_vae, _vae_tiling_config

    conv_vae = SimpleNamespace(decoder=torch.nn.Linear(2, 2), recommended_tiling_config=lambda **_: pytest.fail("unused"))
    assert not _is_diffusion_vae(conv_vae)
    config = _vae_tiling_config(conv_vae, 512, 64, 0, 8, latents=torch.zeros(1, 128, 3, 4, 4))
    assert isinstance(config, TilingConfig)
    assert config.spatial_config.tile_size_in_pixels == 512 and config.spatial_config.tile_overlap_in_pixels == 64
    # no temporal tiling: one 8192-frame tile without overlap, exactly as before the DiffVAE port
    assert config.temporal_config.tile_size_in_frames == 8192 and config.temporal_config.tile_overlap_in_frames == 0
    config = _vae_tiling_config(conv_vae, 512, 64, 24, 8)
    assert config.temporal_config.tile_size_in_frames == 24 and config.temporal_config.tile_overlap_in_frames == 8


def test_ltx25_video_encoder_config_layout_builds_the_conv_encoder(tmp_path):
    """The 2.5 VAE nests its encoder config under ``vae.encoder`` (latent caching reads it, d631633c)."""
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.loader.sft_loader import in_process_checkpoint_loader
    from musubi_tuner.ltx_2.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    encoder_config = {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 8,
        "blocks": [["compress_space_res", {"multiplier": 2}], ["compress_time_res", {"multiplier": 2}]],
        "patch_size": 4,
        "latent_log_var": "constant",
        "norm_layer": "pixel_norm",
        "spatial_padding_mode": "zeros",
    }
    config = {"vae": {"_class_name": "CausalDiffusionVAE", "encoder": encoder_config}}
    torch.manual_seed(0)
    reference = VideoEncoderConfigurator.from_config(config)
    with torch.no_grad():
        for _name, param in reference.named_parameters():
            param.normal_(0.0, 0.05)
        for name, buffer in reference.named_buffers():
            if "per_channel_statistics" in name:
                buffer.uniform_(0.5, 1.5)
    tensors = {}
    for key, value in reference.state_dict().items():
        prefix = "" if key.startswith("per_channel_statistics.") else "encoder."
        tensors[prefix + key] = value.detach().clone().contiguous()
    path = tmp_path / "ltx-2.5-tiny-video-vae-encoder.safetensors"
    save_file(tensors, str(path), metadata={"config": json.dumps(config)})

    encoder = SingleGPUModelBuilder(
        model_path=str(path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        model_loader=in_process_checkpoint_loader(),
    ).build(device=torch.device("cpu"), dtype=torch.float32)
    video = torch.rand(1, 3, 9, 32, 32) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(encoder(video), reference(video), atol=1e-6)


def test_diffvae_defaults_to_bfloat16_without_an_explicit_vae_dtype(tmp_path):
    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    reference = _decoder()
    path = tmp_path / "ltx-2.5-tiny-video-vae.safetensors"
    save_file(
        _release_state_dict(reference, torch.ones(reference.diff_blocks[0].attn.proj.weight.shape[0])),
        str(path),
        metadata={"config": json.dumps({"vae": TINY_VAE_CONFIG})},
    )
    trainer = LTX2NetworkTrainer.__new__(LTX2NetworkTrainer)
    trainer._latent_norm_cache = {}
    # the trainers' default (no --vae_dtype) is FP16, meant for the convolutional VAE
    vae = trainer._load_vae_impl(SimpleNamespace(vae_dtype=None), torch.float16, str(path))
    assert vae.dtype == torch.bfloat16
    vae = trainer._load_vae_impl(SimpleNamespace(vae_dtype="float16"), torch.float16, str(path))
    assert vae.dtype == torch.float16
