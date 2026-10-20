"""MOSS-Image colour / depth control adapters (moss_image/moss_control.py, moss_image_train_network.py --control_adapter):
the widened embedder starts as the released one, the per-sample map dropout follows the recipe, and the exported file
carries the keys ComfyUI's MOSS-Image Color/Depth Control loads, reproducing the trained adapter when patched the way
ComfyUI patches (LoRA folded scale, input-axis DoRA, the embedder replaced)."""

import argparse
import copy
import os
import types

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner import moss_image_control_maps as maps_tool
from musubi_tuner.moss_image import moss_control, moss_utils
from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams
from musubi_tuner.networks import lora_moss_image

PARAMS = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=48, freq_dim=16, latent_channels=8)


def _tiny() -> MossImage:
    torch.manual_seed(0)
    model = MossImage(MossImageParams(**PARAMS))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    return model.eval()


def _inputs(batch=2, channels=8):
    torch.manual_seed(1)
    x = torch.randn(batch, channels, 3, 4)
    t = torch.tensor([300.0, 700.0][:batch])
    ctx = torch.randn(batch, 5, 48)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    return x, t, ctx, mask


# ------------------------------------------------------------------------------------------------------- embedder
def test_the_widened_embedder_starts_as_the_released_one():
    model = _tiny()
    x, t, ctx, mask = _inputs()
    with torch.no_grad():
        expected = model(x, t, ctx, mask)
    embedder = moss_control.widen_video_embedder(model)
    assert embedder.in_features == 24 and embedder.control_weight.shape == (64, 16)
    assert embedder.base_weight.dtype == torch.float32 and not embedder.control_weight.any()
    assert moss_control.widen_video_embedder(model) is embedder  # idempotent
    control = torch.randn(2, 16, 3, 4)
    with torch.no_grad():
        got = model(torch.cat([x, control], dim=1), t, ctx, mask)  # zero control columns: maps change nothing yet
    assert torch.allclose(got, expected, atol=1e-6)
    with torch.no_grad():
        embedder.control_weight.normal_(0, 0.1)
        changed = model(torch.cat([x, control], dim=1), t, ctx, mask)
    assert not torch.allclose(changed, expected, atol=1e-3)


def test_load_widened_round_trips_and_checks_the_shape():
    embedder = moss_control.MossControlEmbedder(torch.nn.Linear(8, 64))
    weight, bias = torch.randn(64, 24), torch.randn(64)
    embedder.load_widened(weight, bias)
    assert torch.equal(embedder.weight, weight) and torch.equal(embedder.bias, bias)
    with pytest.raises(ValueError, match="does not fit"):
        embedder.load_widened(torch.randn(64, 16), bias)


# -------------------------------------------------------------------------------------------------------- dropout
def test_control_dropout_follows_the_recipe_shares():
    torch.manual_seed(0)
    colour, depth = moss_control.control_keep(200_000, moss_control.DEFAULT_CONTROL_DROPOUT)
    colour_only = ((colour == 1) & (depth == 0)).float().mean()
    depth_only = ((colour == 0) & (depth == 1)).float().mean()
    both = ((colour == 1) & (depth == 1)).float().mean()
    neither = ((colour == 0) & (depth == 0)).float().mean()
    for got, want in zip((colour_only, depth_only, both, neither), (0.35, 0.35, 0.2, 0.1)):
        assert abs(float(got) - want) < 0.005
    always_colour = moss_control.control_keep(64, (1, 0, 0, 0))
    assert always_colour[0].all() and not always_colour[1].any()


@pytest.mark.parametrize(
    "text, expected",
    [(None, (0.35, 0.35, 0.2, 0.1)), ("", (0.35, 0.35, 0.2, 0.1)), ("1,1,1,1", (0.25,) * 4), ("2;2;0;0", (0.5, 0.5, 0, 0))],
)
def test_parse_control_dropout(text, expected):
    assert moss_control.parse_control_dropout(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["0.5,0.5,0", "a,b,c,d", "-1,1,1,1", "0,0,0,0"])
def test_parse_control_dropout_rejects_bad_shares(text):
    with pytest.raises(ValueError, match="four non-negative shares"):
        moss_control.parse_control_dropout(text)


def test_control_input_zeroes_missing_and_dropped_maps():
    noisy = torch.randn(2, 8, 3, 4)
    colour, depth = torch.randn(2, 8, 3, 4), torch.randn(2, 8, 3, 4)
    keep = (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))
    out = moss_control.control_input(noisy, [colour, depth], keep)
    assert out.shape == (2, 24, 3, 4)
    assert torch.equal(out[:, :8], noisy) and torch.equal(out[0, 8:16], colour[0]) and not out[1, 8:16].any()
    assert torch.equal(out[:, 16:], depth)
    missing = moss_control.control_input(noisy, [None, depth])
    assert not missing[:, 8:16].any() and torch.equal(missing[:, 16:], depth)
    with pytest.raises(ValueError, match="does not match"):
        moss_control.control_input(noisy, [torch.randn(2, 8, 4, 4), None])


def test_controlled_dit_appends_the_control_on_every_call():
    seen = []

    class _Model(torch.nn.Module):
        dtype = torch.float32

        def forward(self, latents, timestep, ctx, mask, skip_layers=None):
            seen.append((latents.shape, skip_layers))
            return latents[:, :8]

    control = torch.randn(1, 16, 3, 4)
    wrapped = moss_control.ControlledDiT(_Model(), control)
    out = wrapped(torch.randn(2, 8, 3, 4), torch.tensor([1.0, 1.0]), None, None, skip_layers=[22])
    assert out.shape == (2, 8, 3, 4) and seen == [(torch.Size([2, 24, 3, 4]), [22])]
    assert wrapped.dtype == torch.float32  # attributes the sampler reads come from the DiT


# --------------------------------------------------------------------------------------------------------- export
def _comfy_patch(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, dora_scale=None) -> torch.Tensor:
    """ComfyUI's LoRA patch (weight_adapter/lora.py + base.weight_decompose) at strength 1 for a file without alpha keys."""
    diff = lora_b.float() @ lora_a.float()
    if dora_scale is None:
        return weight + diff
    calc = weight + diff
    if dora_scale.shape[0] == calc.shape[0]:  # output axis (not what the trainer writes)
        norm = weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True)
    else:
        norm = calc.transpose(0, 1).reshape(calc.shape[1], -1).norm(dim=1, keepdim=True).transpose(0, 1)
    return calc * (dora_scale.float() / (norm + torch.finfo(weight.dtype).eps))


@pytest.mark.parametrize("use_dora", [True, False])
def test_the_exported_adapter_reproduces_the_trained_model_when_patched_as_comfyui(tmp_path, use_dora):
    model = _tiny()
    embedder = moss_control.widen_video_embedder(model)
    torch.manual_seed(5)
    with torch.no_grad():
        embedder.control_weight.normal_(0, 0.1)
        embedder.base_weight.add_(torch.randn_like(embedder.base_weight) * 0.01)
    reference = copy.deepcopy(model)
    names = moss_control.lora_module_names(model)

    kwargs = {"include_text_embedder": "True"}
    if use_dora:
        kwargs["use_dora"] = "True"
    network = lora_moss_image.create_arch_network(1.0, 4, 2, None, [], model, **kwargs)  # alpha 2 / rank 4: scale 0.5
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.normal_(0, 0.05)
            if use_dora:
                lora.dora_scale.mul_(1 + 0.05 * torch.randn_like(lora.dora_scale))
    assert any(name.startswith("lora_unet_text_embedder_blocks_") for name in (lora.lora_name for lora in network.unet_loras))

    x, t, ctx, mask = _inputs(channels=24)
    with torch.no_grad():
        trained = model(x, t, ctx, mask)

    path = str(tmp_path / "adapter.safetensors")
    network.save_weights(path, None, {"ss_network_dim": "4"})
    count = moss_control.write_comfy_control_file(path, embedder, names, moss_control.control_metadata((0.35, 0.35, 0.2, 0.1), None, "moss-base"))
    exported = load_file(path)
    assert count == len(exported)
    assert "layers.0.proj.q.lora_A.weight" in exported and "video_embedder.weight" in exported  # ComfyUI's detection pair
    assert not any(key.startswith("lora_unet_") or key.endswith(".alpha") for key in exported)
    assert "text_embedder.blocks.0.qkv.lora_B.weight" in exported and exported["text_embedder.blocks.0.qkv.lora_B.weight"].shape == (192, 4)
    assert exported["video_embedder.weight"].shape == (64, 24) and exported["video_embedder.bias"].shape == (64,)
    assert any(key.endswith(".dora_scale") for key in exported) == use_dora
    with safe_open(path, "pt") as handle:
        metadata = handle.metadata()
    assert metadata["rank"] == "4" and metadata["alpha"] == "4.0" and metadata["control_channels"] == "color,depth"
    assert metadata["ss_network_dim"] == "4"  # the trainer's own metadata stays

    modules = dict(reference.named_modules())
    with torch.no_grad():
        for key in [k for k in exported if k.endswith(".lora_A.weight")]:
            module_name = key[: -len(".lora_A.weight")]
            module = modules[module_name]
            module.weight.copy_(
                _comfy_patch(module.weight, exported[key], exported[f"{module_name}.lora_B.weight"], exported.get(f"{module_name}.dora_scale"))
            )
        widened = torch.nn.Linear(24, 64)
        widened.weight.copy_(exported["video_embedder.weight"])
        widened.bias.copy_(exported["video_embedder.bias"])
        reference.video_embedder = widened
        patched = reference(x, t, ctx, mask)
    assert torch.allclose(patched, trained, rtol=1e-4, atol=1e-5), float((patched - trained).abs().max())


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_rewritten_checkpoint_keeps_its_file_mode(tmp_path):
    """safetensors creates its file owner-only (0600): the rewritten checkpoint keeps the trainer's file mode."""
    path = str(tmp_path / "adapter.safetensors")
    save_file({"lora_unet_layers_0_proj_q.lora_down.weight": torch.zeros(2, 8), "lora_unet_layers_0_proj_q.lora_up.weight": torch.zeros(8, 2)}, path)
    os.chmod(path, 0o664)
    moss_control.write_comfy_control_file(path, moss_control.MossControlEmbedder(torch.nn.Linear(8, 16)), {"lora_unet_layers_0_proj_q": "layers.0.proj.q"})
    assert os.stat(path).st_mode & 0o777 == 0o664


def test_export_keeps_dora_scales_in_fp32_and_folds_alpha_into_lora_b():
    embedder = moss_control.MossControlEmbedder(torch.nn.Linear(8, 16))
    down = torch.randn(4, 16, dtype=torch.bfloat16)
    up = torch.randn(32, 4, dtype=torch.bfloat16)
    scale = torch.rand(1, 16, dtype=torch.float32) + 0.5
    sd = {
        "lora_unet_layers_0_proj_q.lora_down.weight": down,
        "lora_unet_layers_0_proj_q.lora_up.weight": up,
        "lora_unet_layers_0_proj_q.alpha": torch.tensor(8.0),
        "lora_unet_layers_0_proj_q.dora_scale": scale,
    }
    out = moss_control.comfy_control_state_dict(sd, {"lora_unet_layers_0_proj_q": "layers.0.proj.q"}, embedder)
    assert out["layers.0.proj.q.lora_B.weight"].dtype == torch.bfloat16
    assert torch.equal(out["layers.0.proj.q.lora_B.weight"], (up.float() * 2.0).to(torch.bfloat16))
    assert out["layers.0.proj.q.dora_scale"].dtype == torch.float32 and torch.equal(out["layers.0.proj.q.dora_scale"], scale)
    assert out["video_embedder.weight"].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="LoRA / input-axis DoRA"):
        moss_control.comfy_control_state_dict(
            {"lora_unet_layers_0_proj_q.lora_magnitude_vector.weight": scale}, {"lora_unet_layers_0_proj_q": "layers.0.proj.q"}, embedder
        )
    with pytest.raises(KeyError, match="no MOSS-Image module"):
        moss_control.comfy_control_state_dict({"lora_unet_nope.lora_down.weight": down}, {}, embedder)


# ------------------------------------------------------------------------------------------------------ the maps
def test_colour_maps_scale_with_nearest_neighbour_and_depth_smoothly(tmp_path):
    colour = np.zeros((40, 60, 3), dtype=np.uint8)
    colour[:, :30] = (200, 40, 90)
    colour[:, 30:] = (20, 180, 250)
    Image.fromarray(colour).save(tmp_path / "c.png")
    pixels = moss_control.load_control_map(str(tmp_path / "c.png"), 48, 32, "colour")
    assert pixels.shape == (3, 32, 48)
    values = {tuple(v) for v in ((pixels.permute(1, 2, 0).reshape(-1, 3) + 1) * 127.5).round().int().tolist()}
    assert values == {(200, 40, 90), (20, 180, 250)}  # region IDs, never blended
    depth = np.tile(np.linspace(0, 255, 60, dtype=np.uint8), (40, 1))
    Image.fromarray(depth, "L").save(tmp_path / "d.png")
    depth_pixels = moss_control.load_control_map(str(tmp_path / "d.png"), 48, 32, "depth")
    assert depth_pixels.shape == (3, 32, 48) and torch.equal(depth_pixels[0], depth_pixels[2])
    # a different aspect is the training images' fit: cover, then crop the centre
    assert moss_control.fit_to_size(Image.new("RGB", (100, 50)), 32, 32, Image.NEAREST).size == (32, 32)


class _FakeVae:
    dtype = torch.float32

    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def encode(self, pixels):
        pooled = torch.nn.functional.avg_pool2d(pixels, 16)  # (1, 3, h, w)
        return pooled.repeat(1, 43, 1, 1)[:, :128] + 1.0


def test_sample_control_encodes_the_given_maps_and_zeroes_the_others(tmp_path):
    Image.new("RGB", (64, 32), (255, 0, 0)).save(tmp_path / "colour.png")
    Image.new("L", (64, 32), 128).save(tmp_path / "depth.png")
    vae = _FakeVae()
    control = moss_control.sample_control(vae, [str(tmp_path / "colour.png"), "none"], 64, 32, "cpu")
    assert control.shape == (1, 256, 2, 4) and control[:, :128].abs().sum() > 0 and not control[:, 128:].any()
    assert vae.devices[-1] == "cpu"  # back off the GPU for the sampling
    both = moss_control.sample_control(vae, [str(tmp_path / "colour.png"), str(tmp_path / "depth.png")], 64, 32, "cpu")
    assert both[:, 128:].abs().sum() > 0
    nothing = moss_control.sample_control(_FakeVae(), None, 64, 32, "cpu")
    assert nothing.shape == (1, 256, 2, 4) and not nothing.any()
    with pytest.raises(ValueError, match="at most 2"):
        moss_control.sample_control(vae, ["a", "b", "c"], 64, 32, "cpu")


# ------------------------------------------------------------------------------------------------------- caching
def _cache(path, keys):
    save_file({key: torch.zeros(1) for key in keys}, str(path))
    return str(path)


def test_latent_cache_is_current_needs_the_items_maps(tmp_path):
    with_maps = _cache(tmp_path / "a_mi.safetensors", ["latents_2x2_float32", "latents_control_0_2x2_float32", "latents_control_1_2x2_float32"])
    without = _cache(tmp_path / "b_mi.safetensors", ["latents_2x2_float32"])
    item = lambda path, controls: types.SimpleNamespace(latent_cache_path=path, control_content=controls)  # noqa: E731
    both = [np.zeros((32, 32, 3)), np.zeros((32, 32, 3))]
    assert moss_control.latent_cache_is_current(item(with_maps, both), lambda _: True)
    assert not moss_control.latent_cache_is_current(item(without, both), lambda _: True)  # maps added later
    assert not moss_control.latent_cache_is_current(item(with_maps, None), lambda _: True)  # maps removed
    assert moss_control.latent_cache_is_current(item(without, None), lambda _: True)
    assert not moss_control.latent_cache_is_current(item(with_maps, both), lambda _: False)


def test_the_empty_caption_round_trips_and_its_absence_names_the_fix(tmp_path):
    ctx = torch.randn(6, 48).to(torch.bfloat16)
    moss_control.save_empty_caption(moss_control.empty_caption_path(str(tmp_path / "cache")), ctx, "moss_image")
    assert torch.equal(moss_control.load_empty_caption([None, str(tmp_path / "missing"), str(tmp_path / "cache")]), ctx)
    with pytest.raises(ValueError, match="text encoder caching again"):
        moss_control.load_empty_caption([str(tmp_path / "missing")])
    assert not moss_control.EMPTY_CAPTION_FILE.endswith("_mi_te.safetensors")  # never read as an item's cache


def test_text_encoder_caching_writes_the_empty_caption_only_for_control_datasets(tmp_path, monkeypatch):
    from musubi_tuner import moss_image_cache_text_encoder_outputs as te_cache

    monkeypatch.setattr(moss_utils, "encode_prompts", lambda model, tokenizer, prompts: [torch.ones(3, 48) for _ in prompts])
    plain = types.SimpleNamespace(has_control=False, cache_directory=str(tmp_path / "plain"))
    control = types.SimpleNamespace(has_control=True, cache_directory=str(tmp_path / "control"))
    te_cache.save_empty_caption_for_control_datasets(None, None, [plain, control])
    assert os.path.isfile(moss_control.empty_caption_path(control.cache_directory))
    assert not os.path.exists(moss_control.empty_caption_path(plain.cache_directory))


def test_a_moss_dataset_pairs_two_maps_per_image(tmp_path):
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.dataset.image_video_dataset import ImageDataset

    images, controls = tmp_path / "images", tmp_path / "control"
    images.mkdir()
    controls.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (32, 32)).save(images / f"{name}.png")
        (images / f"{name}.txt").write_text("caption", encoding="utf-8")
        Image.new("RGB", (32, 32), (9, 9, 9)).save(controls / f"{name}_0.png")
        Image.new("L", (32, 32), 99).save(controls / f"{name}_1.png")

    def dataset():
        return ImageDataset(
            resolution=(32, 32), caption_extension=".txt", batch_size=1, num_repeats=1, enable_bucket=True, bucket_no_upscale=False,
            image_directory=str(images), control_directory=str(controls), cache_directory=str(tmp_path / "cache"),
            architecture=ARCHITECTURE_MOSS_IMAGE,
        )  # fmt: skip

    built = dataset()
    assert built.has_control and built.datasource.control_count_per_image == 2
    assert [os.path.basename(p) for p in built.datasource.control_paths[str(images / "a.png")]] == ["a_0.png", "a_1.png"]
    os.remove(controls / "b_1.png")
    with pytest.raises(ValueError, match="Not enough control images"):
        dataset()


# ------------------------------------------------------------------------------------------------------- trainer
def _args(**extra):
    base = dict(
        mixed_precision="bf16", sample_prompts=None, text_encoder=None, control_adapter=True, network_module="networks.lora_moss_image",
        network_weights=None, dim_from_weights=False, fp8_base=False, fp8_scaled=False, control_lr_ratio=10.0, control_dropout=None,
        control_caption_dropout=0.1, network_args=["use_dora=True"],
    )  # fmt: skip
    base.update(extra)
    return argparse.Namespace(**base)


def test_control_adapter_arguments_add_the_text_refiner_and_parse_the_shares():
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    args = _args(control_dropout="1,1,1,1")
    trainer.handle_model_specific_args(args)
    assert trainer.control_adapter and trainer._control_shares == pytest.approx((0.25,) * 4) and trainer._caption_dropout == 0.1
    assert args.network_args == ["use_dora=True", "include_text_embedder=True"]
    chosen = _args(network_args=["include_text_embedder=False"])
    trainer.handle_model_specific_args(chosen)
    assert chosen.network_args == ["include_text_embedder=False"]


@pytest.mark.parametrize(
    "extra, message",
    [
        (dict(network_module="networks.lora"), "networks.lora_moss_image"),
        (dict(network_weights="x.safetensors"), "--resume"),
        (dict(fp8_base=True), "--fp8_scaled"),
        (dict(control_lr_ratio=0.0), "control_lr_ratio"),
        (dict(control_caption_dropout=1.0), "control_caption_dropout"),
        (dict(network_args=["use_dora=True", "dora_axis=output"]), "input-axis"),
    ],
)
def test_control_adapter_refuses_what_the_export_cannot_carry(extra, message):
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    with pytest.raises(ValueError, match=message):
        MossImageNetworkTrainer().handle_model_specific_args(_args(**extra))


def test_full_fine_tuning_refuses_a_control_adapter():
    from musubi_tuner.moss_image_train import MossImageTrainer

    with pytest.raises(ValueError, match="moss_image_train_network.py"):
        MossImageTrainer()._validate_control_adapter_args(_args())


def test_training_datasets_must_carry_maps_and_the_empty_caption(tmp_path):
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    trainer.handle_model_specific_args(_args())
    with pytest.raises(ValueError, match="needs colour and depth maps"):
        trainer.on_train_datasets(None, [types.SimpleNamespace(has_control=False, cache_only=False, cache_directory=str(tmp_path))])
    with pytest.raises(ValueError, match="text encoder caching again"):
        trainer.on_train_datasets(None, [types.SimpleNamespace(has_control=True, cache_only=False, cache_directory=str(tmp_path))])
    moss_control.save_empty_caption(moss_control.empty_caption_path(str(tmp_path)), torch.ones(3, 48), "moss_image")
    trainer.on_train_datasets(None, [types.SimpleNamespace(has_control=True, cache_only=False, cache_directory=str(tmp_path))])
    assert trainer._empty_ctx.shape == (3, 48)


def test_call_dit_feeds_the_maps_beside_the_noisy_latent_and_drops_captions():
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    trainer.handle_model_specific_args(_args(control_dropout="0,0,1,0", control_caption_dropout=0.999))
    trainer._empty_ctx = torch.zeros(2, 48)
    model = _tiny()
    moss_control.widen_video_embedder(model)
    seen = {}
    original = model.forward

    def spy(hidden, t, ctx, mask, *args, **kwargs):
        seen["hidden"], seen["mask"] = hidden, mask
        return original(hidden, t, ctx, mask, *args, **kwargs)

    model.forward = spy
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    torch.manual_seed(3)
    latents = torch.randn(2, 8, 3, 4)
    noise, colour, depth = torch.randn_like(latents), torch.randn_like(latents), torch.randn_like(latents)
    batch = {"ctx_vec": [torch.randn(7, 48), torch.randn(4, 48)], "latents_control_0": colour, "latents_control_1": depth}
    with torch.no_grad():
        out = trainer.call_dit(argparse.Namespace(gradient_checkpointing=False), accelerator, model, latents, batch, noise, noise, torch.tensor([500.0, 500.0]), torch.float32)
    assert seen["hidden"].shape == (2, 24, 3, 4)
    assert torch.equal(seen["hidden"][:, 8:16], colour) and torch.equal(seen["hidden"][:, 16:], depth)  # "both" share 1
    assert seen["mask"].sum(1).tolist() == [2, 2]  # every caption dropped to the empty one (2 tokens)
    assert torch.equal(out.target, noise - latents)


def test_the_optimizer_groups_train_the_embedder_at_the_recipe_rates():
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    trainer.handle_model_specific_args(_args())
    model = _tiny()
    trainer._control_embedder = moss_control.widen_video_embedder(model)
    trainer._control_embedder.requires_grad_(False)
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    args = argparse.Namespace(learning_rate=1e-4, control_lr_ratio=10.0, network_module="networks.lora_moss_image")
    groups, descriptions = trainer._prepare_network_optimizer_params(args, network)
    released, control = groups[-2], groups[-1]
    assert released["lr"] == 1e-4 and released["weight_decay"] == 0.0 and control["lr"] == pytest.approx(1e-3)
    assert control["params"] == [trainer._control_embedder.control_weight] and trainer._control_embedder.control_weight.requires_grad
    assert descriptions[-2:] == ["control embedder", "control maps"]
    assert set(map(id, trainer.extra_grad_params())) == {id(p) for p in trainer._control_embedder.parameters()}


# ---------------------------------------------------------------------------------------------- map making tool
def test_region_colours_are_distinct_and_never_black():
    colours = maps_tool.region_colours(40, seed=1)
    assert colours.shape == (40, 3) and (colours.max(axis=1) >= 40).all()
    assert len({tuple(c) for c in colours.tolist()}) == 40
    assert np.array_equal(maps_tool.region_colours(5, seed=7), maps_tool.region_colours(5, seed=7))


def test_colour_regions_paint_segments_and_leave_void_black():
    segmentation = np.array([[1, 1, 0], [2, 2, 0]])
    image = np.asarray(maps_tool.colour_regions(segmentation, [1, 2], seed=0))
    assert (image[0, 0] == image[0, 1]).all() and not (image[0, 0] == image[1, 0]).all()
    assert not image[:, 2].any()


def test_panoptic_regions_leave_pixels_no_kept_mask_covers_void():
    """A confident segment owns only where its own mask reaches 0.5: the rest is void (black), so a subject outside the
    classes keeps its silhouette instead of joining the stuff around it (transformers' post-processing gives every pixel
    to some kept segment); unconfident queries and "no object" are dropped."""
    size = (4, 6)
    big = torch.full(size, -6.0)
    big[:, :3] = 6.0  # query 0: left half
    small = torch.full(size, -6.0)
    small[0, 4:] = 6.0  # query 1: a corner
    unsure = torch.full(size, 6.0)  # query 2: everywhere, but its class is unsure
    masks = torch.stack([big, small, unsure])
    classes = torch.full((3, 5), -10.0)
    classes[0, 1] = 10.0
    classes[1, 2] = 10.0
    classes[2, 1], classes[2, 4] = 0.0, 0.0  # half "no object" (the last class): score 0.5 < 0.8
    segmentation, count = maps_tool.panoptic_regions(classes, masks, size)
    assert count == 2
    assert (segmentation[:, :3] == 1).all() and (segmentation[0, 4:] == 2).all()
    assert (segmentation[1:, 3:] == 0).all() and (segmentation[0, 3] == 0)
    nothing, none = maps_tool.panoptic_regions(torch.full((2, 5), -10.0).index_fill(1, torch.tensor([4]), 10.0), masks[:2], size)
    assert none == 0 and not nothing.any()


def test_depth_is_normalised_per_image_with_near_white():
    predicted = torch.tensor([[[1.0, 3.0], [5.0, 9.0]]])  # relative inverse depth: larger = nearer
    image = np.asarray(maps_tool.normalised_depth(predicted, 2, 2))
    assert image.dtype == np.uint8 and image.min() == 0 and image.max() == 255
    assert image[1, 1] == 255 and image[0, 0] == 0


def test_map_paths_follow_the_control_directory_layout(tmp_path):
    assert maps_tool.map_paths("out", "x/photo.jpg") == (os.path.join("out", "photo_0.png"), os.path.join("out", "photo_1.png"))
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="must not be the image folder"):
        maps_tool.make_maps([(str(tmp_path), str(tmp_path))])


def test_dataset_config_names_the_folders_that_get_maps(tmp_path):
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[general]\nresolution = [1024, 1024]\n\n"
        '[[datasets]]\nimage_directory = "/data/a"\ncontrol_directory = "/data/a/control_images"\n\n'
        '[[datasets]]\nimage_directory = "/data/b"\n',
        encoding="utf-8",
    )
    assert maps_tool.dataset_folders(str(config)) == [("/data/a", "/data/a/control_images")]
