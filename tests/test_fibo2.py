"""FIBO 2.0: the DiT's wiring, Bria's prompt formatting and text taps, references, the sampler, the LoRA targets and the
trainer's DiT call.

All on the CPU with tiny models. The real-size DiT matched ai-toolkit's fibo2 transformer (b69b983) bit for bit in FP32 on
random weights, text-only, with two references and in padded batches (dont_upload/fibo2_reference/parity_check.py).
"""

import argparse
import json
import math
import types

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from musubi_tuner.fibo2 import fibo2_model, fibo2_utils
from musubi_tuner.fibo2.fibo2_model import Fibo2, Fibo2Params

TINY = dict(dim=256, n_heads=2, n_layers=6, injection_layers=(1, 3, 5), cap_feat_dim=16, ffn_dim=64)
CTX_DIM = (1 + len(TINY["injection_layers"])) * TINY["cap_feat_dim"]


def _tiny(**kwargs) -> Fibo2:
    torch.manual_seed(0)
    model = Fibo2(Fibo2Params(**TINY), **kwargs)
    with torch.no_grad():  # default inits (zero gates, zero queries) could hide wiring mistakes
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.2 if p.ndim == 1 else 1.0 / p.shape[-1] ** 0.5)
    return model.eval()


def _inputs(batch=2, height=2, width=3, text=5):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 128, height, width, generator=g)
    t = torch.linspace(300.0, 900.0, batch)
    ctx = torch.randn(batch, text, CTX_DIM, generator=g)
    return x, t, ctx


@torch.no_grad()
def test_padded_batch_matches_each_items_own_forward():
    """Captions of different lengths: the padding is masked (and with different gist counts the gist goes last), so
    every item gets its own forward's velocity."""
    model = _tiny()
    x, t, _ = _inputs()
    g = torch.Generator().manual_seed(2)
    ctx = torch.randn(2, 1000, CTX_DIM, generator=g)
    mask = torch.arange(1000)[None] < torch.tensor([30, 1000])[:, None]  # 256 and 384 gist tokens
    refs = [torch.randn(2, 128, 2, 2, generator=g)]
    batched = model(x, t, ctx, mask, refs)
    for i, n in enumerate((30, 1000)):
        single = model(x[i : i + 1], t[i : i + 1], ctx[i : i + 1, :n], None, [refs[0][i : i + 1]])
        assert torch.allclose(batched[i : i + 1], single, atol=1e-4), (i, float((batched[i] - single[0]).abs().max()))


@torch.no_grad()
def test_flash_auto_attention_matches_torch():
    """Guardrail 3: --sdpa resolves to "flash_auto" where native flash SDPA is missing; on tensors flash cannot take it
    falls back to PyTorch SDPA."""
    x, t, ctx = _inputs()
    expected = _tiny(attn_mode="torch")(x, t, ctx)
    out = _tiny(attn_mode="flash_auto")(x, t, ctx)
    assert torch.allclose(out, expected, atol=1e-5)


def test_block_swap_keeps_two_blocks_resident():
    """Guardrail 9: at most n_layers - 2 blocks swap (42 of the real model's 44)."""
    with pytest.raises(AssertionError, match=f"1 to {TINY['n_layers'] - 2}"):
        _tiny().enable_block_swap(TINY["n_layers"] - 1, None)


@torch.no_grad()
def test_references_condition_the_image_and_keep_its_shape():
    model = _tiny()
    x, t, ctx = _inputs(batch=1)
    plain = model(x, t, ctx)
    ref = torch.randn(1, 128, 3, 2)
    edited = model(x, t, ctx, None, [ref])
    assert edited.shape == x.shape == plain.shape
    assert not torch.allclose(plain, edited)


def test_gist_counts_follow_the_text_length():
    p = Fibo2Params()
    assert fibo2_model.gist_counts([1, 256, 257, 1000, 2048, 5000], p) == [256, 256, 320, 384, 512, 512]


def test_gradient_checkpointing_trains_the_blocks():
    model = _tiny().train()
    model.enable_gradient_checkpointing()
    x, t, ctx = _inputs(batch=1)
    model(x, t, ctx, None, [torch.randn(1, 128, 2, 3)]).square().mean().backward()
    for name in ("layers.1.cross_attention.to_kv.weight", "noise_refiner.0.attention.qkv.weight", "perceiver.layers.0.ffn.w13.weight"):
        assert model.get_parameter(name).grad is not None, name


# ------------------------------------------------------------------------------------------------------------- text
def test_prompts_are_formatted_as_bria_trains_them():
    assert fibo2_utils.generation_text("") == fibo2_utils.NULL_CAPTION
    assert fibo2_utils.generation_text("a red fox") == "a red fox"
    caption = {"short_description": "a fox", "objects": [], "lighting": {"type": "soft", "aesthetic_score": 7}, "preference_score": 3}
    assert fibo2_utils.generation_text(json.dumps(caption, indent=2)) == '{"short_description":"a fox","lighting":{"type":"soft"}}'
    assert fibo2_utils.edit_text("make it blue", 1) == (
        '{"image":"<|vision_start|><|image_pad|><|vision_end|>","edit_instruction":"make it blue"}'
    )
    edit = json.dumps({"edit_instruction": "swap", "structured_caption": {"short_description": "two cats"}})
    assert fibo2_utils.edit_text(edit, 2) == (
        '{"image_1":"<|vision_start|><|image_pad|><|vision_end|>","image_2":"<|vision_start|><|image_pad|><|vision_end|>",'
        '"structured_caption":{"short_description":"two cats"},"edit_instruction":"swap"}'
    )
    # a sample's negative is written into the chat turn as it is (ComfyUI's TextEncodeFibo2), a caption is cleaned
    negative = '{"style_medium": "blurry", "objects": []}'
    assert fibo2_utils.template_text(negative, as_written=True) == fibo2_utils.PROMPT_TEMPLATE.format(negative)
    assert fibo2_utils.template_text(negative) == fibo2_utils.PROMPT_TEMPLATE.format('{"style_medium":"blurry"}')
    assert fibo2_utils.template_text(" ", as_written=True) == fibo2_utils.PROMPT_TEMPLATE.format(fibo2_utils.NULL_CAPTION)


def test_taps_expand_to_the_six_bundles_in_order():
    assert fibo2_utils.TAP_LAYERS == (5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35)
    hidden = fibo2_utils.TEXT_HIDDEN
    taps = torch.cat([torch.full((2, hidden), float(layer)) for layer in fibo2_utils.TAP_LAYERS], -1)
    bundles = fibo2_utils.expand_taps(taps)
    assert bundles.shape == (2, 6 * 3 * hidden)
    order = [float(v) for v in bundles[0].reshape(18, hidden)[:, 0]]
    assert order == [float(layer) for bundle in fibo2_utils.BUNDLE_LAYERS for layer in bundle]


def test_contexts_of_a_batch_are_right_padded_and_masked():
    taps = [torch.ones(3, 15 * 2560), torch.ones(5, 15 * 2560)]
    ctx, mask = fibo2_utils.stack_contexts(taps, "cpu", torch.float32)
    assert ctx.shape == (2, 5, 46080) and mask.tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert ctx[0, 3:].eq(0).all()
    _, none = fibo2_utils.stack_contexts(taps[:1], "cpu", torch.float32)
    assert none is None


def _tiny_qwen3_vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from musubi_tuner.krea2.krea2_encoder import QWEN3_VL_4B_INSTRUCT_CONFIG

    config = json.loads(json.dumps(QWEN3_VL_4B_INSTRUCT_CONFIG))
    config["text_config"].update(
        hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
    )
    config["vision_config"].update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32, deepstack_visual_indexes=[0, 1])
    torch.manual_seed(0)
    # the config's BF16 reaches some submodules: FP32 throughout, so a padded batch differs only by summation order
    return Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(config)).float().eval()


def _processors():
    from transformers import AutoImageProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(fibo2_utils.QWEN3_VL_4B_REPO, revision=fibo2_utils.QWEN3_VL_4B_REVISION)
        image_processor = AutoImageProcessor.from_pretrained(fibo2_utils.QWEN3_VL_4B_REPO, revision=fibo2_utils.QWEN3_VL_4B_REVISION)
    except OSError as exc:  # offline without the pinned files
        pytest.skip(f"Qwen3-VL-4B tokenizer / processor unavailable: {exc}")
    return tokenizer, image_processor


@torch.no_grad()
def test_batched_text_encoding_equals_one_forward_per_caption_and_edit_prompts_see_their_images():
    """Guardrail 19: text-only captions share a right-padded forward without a mask and are cut back to their length; no
    pad state reaches the DiT and the real tokens keep positions 0..n-1, so each equals its own unpadded forward."""
    model = _tiny_qwen3_vl()
    tokenizer, image_processor = _processors()
    prompts = ["a red fox in the snow", "", '{"short_description": "a cat", "objects": []}']
    batched = fibo2_utils.encode_prompts(model, tokenizer, image_processor, prompts, dtype=torch.float32)
    for prompt, taps in zip(prompts, batched):
        (single,) = fibo2_utils.encode_prompts(model, tokenizer, image_processor, [prompt], dtype=torch.float32)
        assert taps.shape == single.shape and taps.shape[1] == 15 * 32  # tiny model: 15 taps of 32
        assert torch.allclose(taps, single, atol=1e-5)
        assert torch.allclose(taps.reshape(taps.shape[0], 15, -1).pow(2).mean(-1), torch.ones(()), atol=5e-3)  # unit RMS (eps 1e-6)

    crop = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    (edit,) = fibo2_utils.encode_prompts(model, tokenizer, image_processor, ["make it blue"], [[fibo2_utils.vlm_image(crop)]], dtype=torch.float32)
    (plain,) = fibo2_utils.encode_prompts(model, tokenizer, image_processor, ["make it blue"], dtype=torch.float32)
    image_tokens = (64 // 32) * (96 // 32)  # 16 px patches merged 2 x 2
    text = fibo2_utils.PROMPT_TEMPLATE.format(fibo2_utils.edit_text("make it blue", 1))
    assert edit.shape[0] == len(tokenizer(text)["input_ids"]) - 1 + image_tokens != plain.shape[0]


# --------------------------------------------------------------------------------------------------------- references
def test_reference_sizes_follow_bria_and_comfyui():
    assert fibo2_utils.trained_size(1.0) == (1024, 1024)
    w, h = fibo2_utils.trained_size(16 / 9)
    assert w % 16 == 0 and h % 16 == 0 and abs(w * h - 1024 * 1024) <= 0.1 * 1024 * 1024 and abs(w / h - 16 / 9) < 0.02
    one = [Image.new("RGB", (300, 200))]
    assert fibo2_utils.sample_reference_sizes(one, 512, 768) == [(512, 768)]  # a single reference: the result's size
    two = one + [Image.new("RGB", (200, 400))]
    assert fibo2_utils.sample_reference_sizes(two, 512, 768) == [fibo2_utils.trained_size(1.5), fibo2_utils.trained_size(0.5)]
    crop = fibo2_utils.resize_and_crop(Image.new("RGB", (300, 200)), 512, 768)
    assert crop.size == (512, 768)
    small = fibo2_utils.vlm_image(Image.new("RGB", (1000, 600)))
    assert max(small.size) <= 768 + 16 and small.width % 32 == 0 and small.height % 32 == 0


# ----------------------------------------------------------------------------------------------------------- sampling
def test_sigmas_are_comfyuis_simple_schedule_at_shift_1_15():
    sigmas = fibo2_utils.comfy_sigmas(30)
    assert len(sigmas) == 31 and sigmas[0].item() == 1.0 and sigmas[-1].item() == 0.0
    for x in (1, 7, 29):
        t = (1000 - int(x * 1000 / 30)) / 1000
        expected = math.exp(1.15) / (math.exp(1.15) + (1 / t - 1))  # ComfyUI's flux_time_shift(1.15, 1.0, t)
        assert abs(sigmas[x].item() - expected) < 1e-6
    assert torch.all(sigmas[1:] < sigmas[:-1])


def test_denoise_runs_the_negative_as_its_own_forward_with_the_references():
    calls = []

    def model(x, timestep, ctx, mask, refs):
        calls.append((ctx.shape[1], refs))
        return torch.ones_like(x) * (2.0 if ctx.shape[1] == 3 else 1.0)

    noise = torch.zeros(1, 128, 2, 2)
    refs = [torch.zeros(1, 128, 2, 2)]
    out = fibo2_utils.denoise(model, noise, torch.zeros(1, 3, 8), None, torch.zeros(1, 5, 8), None, 4, 5.0, ref_latents=refs, dtype=torch.float32)
    assert [length for length, _ in calls] == [3, 5] * 4 and all(r is refs for _, r in calls)
    # CFG velocity 1 + 5 * (2 - 1) = 6, integrated from sigma 1 to 0
    assert torch.allclose(out, torch.full_like(out, -6.0), atol=1e-5)


# ------------------------------------------------------------------------------------------------- checkpoints, LoRA
def _write(path, keys):
    save_file({k: torch.zeros(1) for k in keys}, str(path))
    return str(path)


def test_checkpoints_that_cannot_train_are_refused(tmp_path):
    with pytest.raises(ValueError, match="diffusers-layout"):
        fibo2_utils.check_dit_checkpoint(_write(tmp_path / "d.safetensors", ["transformer_blocks.0.attn.to_q.weight"]))
    with pytest.raises(ValueError, match="not a FIBO 2.0 DiT"):
        fibo2_utils.check_dit_checkpoint(_write(tmp_path / "x.safetensors", ["model.diffusion_model.x.weight"]))
    base = ["perceiver.latent_queries", "layers.0.attention.qkv.weight", "layers.0.feed_forward.w13.weight"]
    with pytest.raises(ValueError, match="INT8 ConvRot"):
        fibo2_utils.check_dit_checkpoint(_write(tmp_path / "q.safetensors", base + ["layers.0.attention.qkv.comfy_quant"]))
    assert "perceiver.latent_queries" in fibo2_utils.check_dit_checkpoint(_write(tmp_path / "ok.safetensors", base))


def test_params_come_from_the_checkpoint_shapes():
    model = _tiny()
    shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    params = fibo2_model.params_from_shapes(shapes)
    for key, value in TINY.items():
        assert getattr(params, key) == value, key
    assert params.max_gist_tokens == 512 and params.n_refiner_layers == 2 and params.perceiver_layers == 2


def test_lora_targets_every_block_linear_under_comfyui_names():
    from musubi_tuner.networks import lora_fibo2

    model = _tiny()
    network = lora_fibo2.create_arch_network(1.0, 4, 1.0, None, [], model)
    names = {lora.lora_name for lora in network.unet_loras}
    expected = {
        "lora_unet_" + name.replace(".", "_")
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith(("layers.", "noise_refiner.", "perceiver.layers."))
    }
    assert names == expected
    assert "lora_unet_layers_1_cross_attention_to_kv" in names and "lora_unet_layers_0_attention_qkv" in names
    assert "lora_unet_perceiver_layers_0_self_attention_to_q" in names and "lora_unet_noise_refiner_0_feed_forward_w13" in names
    assert not any(part in name for name in names for part in ("x_embedder", "t_embedder", "final_layer", "norm"))


# ------------------------------------------------------------------------------------------------------------ trainer
def test_call_dit_feeds_references_and_targets_noise_minus_latents():
    from musubi_tuner.fibo2_train_network import Fibo2NetworkTrainer

    seen = {}

    def model(hidden, t, ctx, mask, refs):
        seen.update(hidden=hidden, t=t, ctx=ctx, mask=mask, refs=refs)
        return torch.zeros_like(hidden)

    latents, noise = torch.randn(2, 128, 2, 2), torch.randn(2, 128, 2, 2)
    batch = {"ctx_vec": [torch.ones(3, 15 * 2560), torch.ones(4, 15 * 2560)], "latents_control_0": torch.randn(2, 128, 2, 2)}
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    out = Fibo2NetworkTrainer().call_dit(argparse.Namespace(), accelerator, model, latents, batch, noise, latents, torch.tensor([500.0, 10.0]), torch.float32)
    assert torch.equal(out.target, noise - latents)
    assert seen["ctx"].shape == (2, 4, 46080) and seen["mask"][0].tolist() == [True, True, True, False]
    assert len(seen["refs"]) == 1 and torch.equal(seen["refs"][0], batch["latents_control_0"])
    assert seen["t"].tolist() == [500.0, 10.0]


def test_direct_timesteps_reach_the_dit_as_1000_sigma():
    from musubi_tuner.fibo2_train_network import Fibo2NetworkTrainer

    trainer = Fibo2NetworkTrainer()
    args = argparse.Namespace(
        timestep_sampling="flux2_shift", sigmoid_scale=1.0, discrete_flow_shift=1.0, min_timestep=None, max_timestep=None,
        preserve_distribution_shape=False, weighting_scheme="none", logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
    )
    trainer.num_timestep_buckets = None
    latents, noise = torch.zeros(4, 128, 64, 64), torch.ones(4, 128, 64, 64)
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(args, noise, latents, None, None, "cpu", torch.float32)
    # noise 1 and latents 0: the noisy input is sigma itself
    assert torch.allclose(noisy[:, 0, 0, 0] * 1000, timesteps, atol=1e-3)


def test_fibo2_is_registered_with_the_shared_dataset_and_metadata_code():
    from musubi_tuner.dataset import bucket
    from musubi_tuner.dataset.architectures import ARCHITECTURE_FIBO2, ARCHITECTURE_FIBO2_FULL
    from musubi_tuner.utils import sai_model_spec
    from musubi_tuner.verify_cache_files import resolve_architecture

    assert resolve_architecture(ARCHITECTURE_FIBO2_FULL) == ARCHITECTURE_FIBO2
    assert bucket.BucketSelector.resolve_resolution_steps(ARCHITECTURE_FIBO2) == 16
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_FIBO2, 0.0, title="fibo2")
    assert metadata["modelspec.architecture"].startswith(sai_model_spec.ARCH_FIBO2)


def test_text_cache_records_its_references_and_skip_existing_reencodes_changes(tmp_path):
    from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_fibo2
    from musubi_tuner.dataset.image_video_dataset import ItemInfo
    from musubi_tuner.fibo2_cache_text_encoder_outputs import cache_is_current

    item = ItemInfo("a", "make it blue", (64, 64), (64, 64))
    item.text_encoder_output_cache_path = str(tmp_path / "a_fb2_te.safetensors")
    assert not cache_is_current(item)
    item.control_content = [np.zeros((48, 32, 3), np.uint8)]
    save_text_encoder_output_cache_fibo2(item, torch.zeros(3, 15 * 2560, dtype=torch.bfloat16), [(32, 48)])
    assert cache_is_current(item)
    item.control_content = [np.zeros((64, 64, 3), np.uint8)]  # another resolution
    assert not cache_is_current(item)
    item.control_content = None  # the same caption trained text-to-image
    assert not cache_is_current(item)


# ------------------------------------------------------------------------------------------------------- ConvRot INT8
# ConvRot groups are 256 inputs wide: the smallest DiT whose block linears all quantize
INT8_TINY = dict(dim=256, n_heads=2, n_layers=3, injection_layers=(1,), cap_feat_dim=256, ffn_dim=256)
INT8_CTX = 2 * 256
INT8_SPEC = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'


def _released_dit(tmp_path):
    torch.manual_seed(0)
    model = Fibo2(Fibo2Params(**INT8_TINY))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.2 if p.ndim == 1 else 1.0 / p.shape[-1] ** 0.5)
    path = str(tmp_path / "fibo2_dev_bf16.safetensors")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)
    return model.eval(), path


def _int8_targets(model) -> set:
    """The layers Comfy-Org's INT8 file holds in INT8 (215 of the real model's 261 block linears): every linear of the 44
    blocks, the noise refiner and the Perceiver's layers but the adaLN modulations."""
    return {
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith(("layers.", "noise_refiner.", "perceiver.layers.")) and "adaLN" not in name
    }


def _int8_inputs():
    g = torch.Generator().manual_seed(3)
    return torch.randn(1, 128, 2, 3, generator=g), torch.tensor([700.0]), torch.randn(1, 6, INT8_CTX, generator=g)


def test_convrot_int8_quantizes_the_block_linears_while_loading(tmp_path):
    reference, path = _released_dit(tmp_path)
    model = fibo2_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == _int8_targets(reference)
    assert model.is_convrot_int8 and not any(p.requires_grad for p in model.parameters())
    assert model.layers[0].adaLN_modulation[0].weight.dtype == torch.float32  # the modulations stay unquantized
    x, t, ctx = _int8_inputs()
    with torch.no_grad():
        expected, out = reference(x, t, ctx), model(x, t, ctx)
    assert ((out - expected).norm() / expected.norm()).item() < 0.03


def test_an_int8_convrot_file_trains_lora_as_it_is_and_holds_the_bf16_files_values(tmp_path):
    """Comfy-Org's fibo2_dev_int8_convrot layout: int8 weights, FP32 [rows, 1] weight_scale, a comfy_quant spec per layer and
    a ComfyUI attention hint per block. Full fine-tuning refuses it."""
    from safetensors.torch import load_file

    from musubi_tuner.modules.convrot_int8_utils import quantize_weight_convrot

    reference, path = _released_dit(tmp_path)
    quantized_on_load = fibo2_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True).state_dict()
    released = {}
    targets = _int8_targets(reference)
    for key, value in load_file(path).items():
        if key in targets:
            module = key[: -len(".weight")]
            released[key], released[module + ".weight_scale"], _ = quantize_weight_convrot(key, value, (256,))
            released[module + ".comfy_quant"] = torch.tensor(list(INT8_SPEC), dtype=torch.uint8)
        else:
            released[key] = value
    released["layers.0.attention.comfy_attention.config"] = torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)
    int8_path = str(tmp_path / "fibo2_dev_int8_convrot.safetensors")
    save_file(released, int8_path)
    with pytest.raises(ValueError, match="full fine-tuning"):
        fibo2_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32)
    model = fibo2_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32, allow_int8_file=True)
    state = model.state_dict()
    assert state.keys() == quantized_on_load.keys()
    for name, tensor in state.items():
        expected = quantized_on_load[name]
        if tensor.dtype == torch.int8:
            assert torch.equal(tensor, expected), name
        else:
            assert torch.allclose(tensor, expected, rtol=1e-5, atol=0), name


def test_dora_trains_on_the_int8_convrot_dit(tmp_path):
    """DoRA reads the INT8 base in its original basis; with the up matrices at zero the adapted DiT is the base, and the
    adapter gets gradients through the INT8 layers' backward."""
    from musubi_tuner.networks import lora_fibo2

    _, path = _released_dit(tmp_path)
    model = fibo2_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True)
    x, t, ctx = _int8_inputs()
    with torch.no_grad():
        base = model(x, t, ctx)
    network = lora_fibo2.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    out = model(x, t, ctx)
    assert torch.allclose(out, base, atol=1e-5)
    out.square().mean().backward()
    assert all(p.grad is not None for p in network.parameters() if p.requires_grad)


def test_the_trainer_refuses_int8_with_fp8():
    from musubi_tuner.fibo2_train_network import Fibo2NetworkTrainer

    args = argparse.Namespace(mixed_precision="bf16", sample_prompts=None, text_encoder=None, convrot_int8=True, fp8_base=True, fp8_scaled=True)
    with pytest.raises(ValueError, match="choose one"):
        Fibo2NetworkTrainer().handle_model_specific_args(args)


# ------------------------------------------------------------------------------------------------- full fine-tuning
FT_REQUIRED = ["--dit", "dit.safetensors", "--dataset_config", "d.toml", "--output_dir", "out", "--output_name", "o"]


def _ft_parse(*extra):
    from musubi_tuner import fibo2_train as ft
    from musubi_tuner.training.parser_common import read_config_from_file

    parser = ft.create_parser()
    return read_config_from_file(parser.parse_args([*FT_REQUIRED, *extra]), parser)


def test_full_fine_tuning_parser_defaults_form_a_valid_dense_recipe():
    from musubi_tuner import fibo2_train as ft

    args = _ft_parse()
    assert args.full_bf16 and args.fused_backward_pass and args.mem_eff_save and args.gradient_checkpointing
    assert args.network_module is None and args.optimizer_type == "Adafactor" and args.max_grad_norm == 0.0
    assert args.learning_rate == 1e-5
    ft.Fibo2Trainer()._validate_full_finetune_args(args)
    for extra, message in ((["--convrot_int8"], "BF16 weights"), (["--fp8_scaled"], "BF16 weights"), (["--block_swap_h2d_only"], "h2d_only")):
        with pytest.raises(ValueError, match=message):
            ft.Fibo2Trainer()._validate_full_finetune_args(_ft_parse(*extra))


def test_full_fine_tuning_runs_the_loop_whose_hooks_the_dense_trainer_overrides():
    from musubi_tuner import fibo2_train as ft
    from musubi_tuner.fibo2_train_network import Fibo2NetworkTrainer, Fibo2TrainerMixin
    from musubi_tuner.training import trainer_base, training_loop

    assert ft.Fibo2Trainer.train is trainer_base.NetworkTrainer.train
    assert Fibo2NetworkTrainer.train is training_loop.train
    assert issubclass(ft.Fibo2Trainer, Fibo2TrainerMixin) and not issubclass(ft.Fibo2Trainer, Fibo2NetworkTrainer)


def test_the_full_fine_tuning_checkpoint_has_the_released_layout_and_loads_back(tmp_path):
    """The model's state dict is Comfy-Org's layout: the saved DiT loads back through load_dit weight for weight (and in
    ComfyUI); the released file's 634 keys are the full-size model's state dict (checked on the server)."""
    from safetensors import safe_open

    from musubi_tuner import fibo2_train as ft
    from musubi_tuner.training.dense_trainer import DenseFinetuneModule

    model = _tiny().to(torch.bfloat16)
    path = tmp_path / "fibo2_ft.safetensors"
    module = DenseFinetuneModule(model, ft.Fibo2Trainer().dense_state_dict_for_save, model_name="FIBO 2.0")
    module.save_weights(str(path), torch.bfloat16, {"ss_training_type": "full_finetune"})
    with safe_open(str(path), "pt") as handle:
        assert set(handle.keys()) == set(model.state_dict()) and handle.metadata()["ss_training_type"] == "full_finetune"
    reloaded = fibo2_utils.load_dit("cpu", str(path), "torch", False, "cpu", torch.bfloat16)
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name
    with pytest.raises(ValueError, match="full fine-tuning"):
        released = {k: v for k, v in model.state_dict().items()}
        released["layers.0.attention.qkv.comfy_quant"] = torch.tensor(list(INT8_SPEC), dtype=torch.uint8)
        save_file({k: v.contiguous() for k, v in released.items()}, str(tmp_path / "int8.safetensors"))
        fibo2_utils.load_dit("cpu", str(tmp_path / "int8.safetensors"), "torch", False, "cpu", torch.bfloat16)


# ------------------------------------------------------------------------------------------------------- references
@pytest.mark.parametrize("count", [1, 2])
def test_edit_datasets_frame_references_as_bria(tmp_path, count):
    """A single reference is cropped to its target's bucket (the edit lines up with it pixel for pixel); several each keep
    their own aspect ratio at the target's pixel count (ai-toolkit's match_target_res, Bria's sizes)."""
    from musubi_tuner.dataset.architectures import ARCHITECTURE_FIBO2
    from musubi_tuner.dataset.image_video_dataset import ImageDataset
    from musubi_tuner.fibo2.fibo2_sizes import trained_size

    targets, controls = tmp_path / "targets", tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    Image.new("RGB", (512, 512), (90, 90, 90)).save(targets / "sample.png")
    (targets / "sample.txt").write_text("make it blue", encoding="utf-8")
    shapes = [(640, 360), (300, 600)][:count]
    for index, size in enumerate(shapes):
        Image.new("RGB", size, (index * 50, 0, 0)).save(controls / f"sample_{index}.png")
    dataset = ImageDataset(
        resolution=(512, 512), caption_extension=".txt", batch_size=1, num_repeats=1, enable_bucket=True, bucket_no_upscale=False,
        image_directory=str(targets), control_directory=str(controls), cache_directory=str(tmp_path / "cache"),
        architecture=ARCHITECTURE_FIBO2,
    )
    (_key, batch), = list(dataset.retrieve_latent_cache_batches(1))
    sizes = [(control.shape[1], control.shape[0]) for control in batch[0].control_content]
    if count == 1:
        assert sizes == [(512, 512)]
    else:
        assert sizes == [trained_size(640 / 360, 512 * 512), trained_size(300 / 600, 512 * 512)]
        assert sizes[0][0] > sizes[0][1] and sizes[1][0] < sizes[1][1]


def test_block_swap_moves_units_of_one_layout():
    """The trainable ring and the H2D-only stream copy every swapped block through one buffer layout: block swap sees
    each block without an injection block's cross-attention (which stays on the GPU), so all 44 units match."""
    model = _tiny()
    units = model.swap_units()
    layouts = {tuple((name, tuple(p.shape), p.dtype) for name, p in unit.named_parameters()) for unit in units}
    assert len(units) == TINY["n_layers"] and len(layouts) == 1
    injection = model.layers[TINY["injection_layers"][0]]
    assert units[TINY["injection_layers"][0]].attention is injection.attention and not hasattr(units[1], "cross_attention")
    assert set(model.state_dict()) == set(_tiny().state_dict())  # the units are not part of the model's state


def test_a_swap_unit_runs_its_block_so_backward_hooks_fire():
    """The forward runs a swapped block through its unit (the same computation, the block's cross-attention included), so
    the offloaders' backward hooks on the units fire: hooks that never fired left a swapped block's weights on the CPU for
    its backward recompute ("mat2 is on cpu", the first server run with block swap)."""
    model = _tiny().train()
    index = TINY["injection_layers"][0]
    unit = model.swap_units()[index]
    fired = []
    unit.register_full_backward_hook(lambda *args: fired.append(True))
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 262, TINY["dim"], generator=g, requires_grad=True)
    rope = model._rope(torch.zeros(262, 3, dtype=torch.long))
    adaln = model.t_embedder(torch.tensor([500.0]))
    context = torch.randn(1, 5, TINY["cap_feat_dim"], generator=g)
    out = unit(x, rope, adaln, None, None, context, None)
    assert torch.equal(out, model.layers[index](x, rope, adaln, None, None, context, None))
    out.square().mean().backward()
    assert fired
