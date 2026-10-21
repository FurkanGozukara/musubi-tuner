"""MOSS-Image: the DiT's wiring, checkpoint checks, the sampler's sigmas, the LoRA targets and the trainer's DiT call.

All on the CPU with a tiny model; the port was compared with ai-toolkit's transformer (bit-identical, padded batches and
the negative branch's layer skip included) and the vendor's per-item forward (1.8e-7 in fp32) before these were written.
"""

import argparse
import types

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.moss_image import moss_model, moss_utils
from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams

PARAMS = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=48, freq_dim=16, latent_channels=8)


def _tiny() -> MossImage:
    torch.manual_seed(0)
    model = MossImage(MossImageParams(**PARAMS))
    with torch.no_grad():  # default inits could hide wiring mistakes
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    return model.eval()


def _inputs():
    torch.manual_seed(1)
    x = torch.randn(2, 8, 3, 4)
    t = torch.tensor([300.0, 900.0])
    ctx = torch.randn(2, 7, 48)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 4:] = False
    return x, t, ctx, mask


def test_a_padded_batch_equals_one_forward_per_item():
    """The vendor runs every item alone at its own text length; the masked batch must match it."""
    model = _tiny()
    x, t, ctx, mask = _inputs()
    with torch.no_grad():
        batched = model(x, t, ctx, mask)
        single = model(x[1:], t[1:], ctx[1:, :4], None)
    assert batched.shape == x.shape
    assert torch.allclose(batched[1:], single, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="activation CPU offloading moves block outputs between CUDA and the CPU")
def test_activation_cpu_offloading_matches_and_trains():
    """The 6 GB preset candidates (activation CPU offloading) failed at the first step: the text refiner's last output
    came back on the CPU and was joined with the image tokens on the GPU."""
    model = _tiny().cuda().train()
    x, t, ctx, mask = (value.cuda() for value in _inputs())
    expected = model(x, t, ctx, mask).detach()
    model.enable_gradient_checkpointing(activation_cpu_offloading=True)
    out = model(x, t, ctx, mask)
    out.square().mean().backward()
    assert out.device.type == "cuda" and torch.allclose(out, expected, atol=1e-5)
    assert all(p.grad is not None for p in model.layers[0].parameters())


def test_skipping_a_layer_changes_the_prediction():
    model = _tiny()
    x, t, ctx, mask = _inputs()
    with torch.no_grad():
        assert not torch.allclose(model(x, t, ctx, mask), model(x, t, ctx, mask, skip_layers=(1,)))


def test_fused_qkv_splits_in_row_order():
    weight = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    sd = {"layers.0.proj.qkv.weight": weight.clone(), "text_embedder.blocks.0.qkv.weight": weight.clone(), "rope.bands": torch.ones(2)}
    moss_model.split_fused_qkv(sd)
    assert torch.equal(sd["layers.0.proj.q.weight"], weight[:2]) and torch.equal(sd["layers.0.proj.v.weight"], weight[4:])
    assert "text_embedder.blocks.0.qkv.weight" in sd  # the refiner stays fused
    assert "rope.bands" not in sd


def test_block_swap_keeps_two_blocks_resident():
    """Guardrail 9: at most num_layers - 2 blocks swap (40 of the real model's 42)."""
    with pytest.raises(AssertionError, match="1 to 2"):
        _tiny().enable_block_swap(3, None)


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        (["shared_block_modulation.weight", "layers.0.proj.qkv.weight", "layers.0.proj.qkv.comfy_quant"], "pre-quantized"),
        (["double_blocks.0.img_attn.qkv.weight"], "not a MOSS-Image"),
    ],
)
def test_checkpoints_that_cannot_train_are_refused(tmp_path, keys, message):
    path = str(tmp_path / "dit.safetensors")
    save_file({k: torch.zeros(1) for k in keys}, path)
    with pytest.raises(ValueError, match=message):
        moss_utils.check_dit_checkpoint(path)


def test_the_bf16_checkpoint_passes(tmp_path):
    path = str(tmp_path / "dit.safetensors")
    save_file({k: torch.zeros(1) for k in ("shared_block_modulation.weight", "layers.0.proj.qkv.weight")}, path)
    moss_utils.check_dit_checkpoint(path)


def test_lora_training_takes_int8_convrot_files_only(tmp_path):
    released = {"shared_block_modulation.weight": torch.zeros(1), "layers.0.proj.qkv.weight": torch.zeros(1)}
    int8 = str(tmp_path / "int8.safetensors")
    save_file({**released, "layers.0.proj.qkv.comfy_quant": torch.zeros(1, dtype=torch.uint8)}, int8)
    assert moss_utils.check_dit_checkpoint(int8, allow_int8=True) is True
    fp8 = str(tmp_path / "fp8.safetensors")
    save_file({**released, "layers.0.proj.qkv.weight_scale": torch.zeros(1), "layers.0.proj.qkv.input_scale": torch.zeros(1)}, fp8)
    with pytest.raises(ValueError, match="another format"):
        moss_utils.check_dit_checkpoint(fp8, allow_int8=True)


def test_fused_qkv_of_an_int8_file_splits_its_scales_and_specs():
    weight = torch.arange(12, dtype=torch.int8).reshape(6, 2)
    scale = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    spec = torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    sd = {"layers.0.proj.qkv.weight": weight, "layers.0.proj.qkv.weight_scale": scale, "layers.0.proj.qkv.comfy_quant": spec}
    moss_model.split_fused_qkv(sd)
    assert torch.equal(sd["layers.0.proj.k.weight"], weight[2:4]) and torch.equal(sd["layers.0.proj.k.weight_scale"], scale[2:4])
    assert all(torch.equal(sd[f"layers.0.proj.{name}.comfy_quant"], spec) for name in "qkv")
    assert not any("qkv" in key for key in sd)


# ConvRot groups are 256 inputs wide: the smallest DiT whose block projections all quantize
INT8_PARAMS = dict(dim=256, ffn_dim=256, num_heads=4, num_layers=2, text_dim=48, freq_dim=16, latent_channels=8)
INT8_SPEC = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
INT8_LAYERS = {f"layers.{i}.proj.{name}.weight" for i in range(2) for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")}


def _released_dit(tmp_path, monkeypatch):
    """A tiny DiT saved in the released files' layout (fused q / k / v); load_dit builds that size."""
    torch.manual_seed(0)
    params = MossImageParams(**INT8_PARAMS)
    model = MossImage(params)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    monkeypatch.setattr(moss_utils.moss_model, "MossImageParams", lambda: params)
    path = str(tmp_path / "moss_image_base_bf16.safetensors")
    save_file({k: v.contiguous() for k, v in moss_model.fuse_qkv(model.state_dict(), params).items()}, path)
    return model.eval(), path


def test_convrot_int8_quantizes_the_block_projections_while_loading(tmp_path, monkeypatch):
    reference, path = _released_dit(tmp_path, monkeypatch)
    model = moss_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == INT8_LAYERS
    assert model.is_convrot_int8 and not any(p.requires_grad for p in model.parameters())
    assert model.text_embedder.blocks[0].qkv.weight.dtype == torch.float32  # the refiner stays unquantized
    x, t, ctx, mask = _inputs()
    with torch.no_grad():
        expected, out = reference(x, t, ctx, mask), model(x, t, ctx, mask)
    assert ((out - expected).norm() / expected.norm()).item() < 0.02


def test_an_int8_convrot_file_trains_as_it_is_and_holds_the_bf16_files_values(tmp_path, monkeypatch):
    """Comfy-Org's INT8 files quantize the fused qkv (and carry a ComfyUI attention hint per block); the rotation mixes
    input columns only, so its rows hold the split projections' values quantized while loading. The rotation's float
    sums may round differently for the fused and the split matrix (an INT8 step at a rounding boundary on some CPUs).
    Full fine-tuning refuses the file."""
    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight

    _, path = _released_dit(tmp_path, monkeypatch)
    quantized_on_load = moss_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True).state_dict()
    released = {}
    for key, value in moss_utils.load_safetensors(path, device="cpu").items():
        if moss_utils.is_convrot_int8_target(key):
            module = key[: -len(".weight")]
            released[key], released[module + ".weight_scale"] = quantize_int8_convrot_weight(value, 256)
            released[module + ".comfy_quant"] = torch.tensor(list(INT8_SPEC), dtype=torch.uint8)
            released[module.rsplit(".proj.", 1)[0] + ".comfy_attention.config"] = torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)
        else:
            released[key] = value
    int8_path = str(tmp_path / "moss_image_base_int8_convrot.safetensors")
    save_file(released, int8_path)
    with pytest.raises(ValueError, match="full fine-tuning"):
        moss_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32)
    model = moss_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32, allow_int8_file=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == INT8_LAYERS
    state = model.state_dict()
    assert state.keys() == quantized_on_load.keys()
    for name, tensor in state.items():
        expected = quantized_on_load[name]
        if tensor.dtype == torch.int8:
            steps = (tensor.int() - expected.int()).abs()
            assert steps.max().item() <= 1 and steps.count_nonzero().item() <= tensor.numel() // 1000, name
        else:
            assert torch.allclose(tensor, expected, rtol=1e-5, atol=0), name


def test_dora_trains_on_the_int8_convrot_dit(tmp_path, monkeypatch):
    """DoRA reads the INT8 base in its original basis; with the up matrices at zero the adapted DiT is the base, and the
    adapter gets gradients through the INT8 layers' backward."""
    from musubi_tuner.networks import lora_moss_image

    _, path = _released_dit(tmp_path, monkeypatch)
    model = moss_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True)
    x, t, ctx, mask = _inputs()
    with torch.no_grad():
        base = model(x, t, ctx, mask)
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    out = model(x, t, ctx, mask)
    assert torch.allclose(out, base, atol=1e-5)
    out.square().mean().backward()
    assert all(p.grad is not None for p in network.parameters() if p.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused ConvRot INT8 kernels run on CUDA")
@pytest.mark.parametrize("bwd", ["bf16", "int8"])
def test_convrot_int8_runs_the_fused_kernels_on_the_gpu(tmp_path, monkeypatch, bwd):
    """With Triton the INT8 layers also quantize their inputs row-wise: the DiT stays within a few percent of BF16, and
    DoRA trains through either backward."""
    from musubi_tuner.modules.convrot_int8_kernels import HAS_TRITON
    from musubi_tuner.networks import lora_moss_image

    if not HAS_TRITON:
        pytest.skip("no Triton")
    reference, path = _released_dit(tmp_path, monkeypatch)
    model = moss_utils.load_dit("cuda", path, "torch", False, "cuda", torch.bfloat16, convrot_int8=True, convrot_int8_bwd=bwd)
    x, t, ctx, mask = (value.cuda() for value in _inputs())
    reference = reference.cuda().to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        expected, out = reference(x, t, ctx, mask).float(), model(x, t, ctx, mask).float()
    assert ((out - expected).norm() / expected.norm()).item() < 0.05
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.cuda()
    with torch.autocast("cuda", torch.bfloat16):
        model(x, t, ctx, mask).float().square().mean().backward()
    trained = [p for p in network.parameters() if p.requires_grad]
    assert trained and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trained)


def test_the_trainer_refuses_int8_with_fp8():
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    args = argparse.Namespace(mixed_precision="bf16", sample_prompts=None, text_encoder=None, convrot_int8=True, fp8_base=True, fp8_scaled=True)
    with pytest.raises(ValueError, match="choose one"):
        MossImageNetworkTrainer().handle_model_specific_args(args)


def test_sigmas_follow_the_vendor_grid():
    sigmas = moss_utils.moss_sigmas(50)
    assert sigmas[0].item() == 1.0 and sigmas[-1].item() == 0.0
    assert torch.all(sigmas[1:] < sigmas[:-1])
    # shift 5 snapped to the 1000 uniform training sigmas: every inner value is k / 1000
    inner = sigmas[1:-1] * 1000
    assert torch.allclose(inner, inner.round(), atol=1e-3)


def test_padded_contexts_carry_their_mask():
    ctx, mask = moss_utils.pad_contexts([torch.ones(3, 4), torch.ones(5, 4)], "cpu", torch.float32)
    assert ctx.shape == (2, 5, 4) and ctx[0, 3:].eq(0).all()
    assert mask.tolist() == [[True] * 3 + [False] * 2, [True] * 5]


class _Template:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking
        return "<|im_start|>user\n" + messages[0]["content"].strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_conditioning_text_is_comfyuis():
    assert moss_utils.chat_text(_Template(), " a cat ") == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"


class _CharTokenizer(_Template):
    def __call__(self, text, return_tensors, truncation, max_length):
        return {"input_ids": torch.tensor([[ord(c) % 97 for c in text][:max_length]])}


def _tiny_text_encoder():
    """Qwen3.5's layout (three linear-attention layers, then full attention) at toy sizes, random weights. Routed by
    device as load_text_encoder routes the real one: with flash-linear-attention installed, transformers otherwise hands
    CPU tensors to its Triton kernels."""
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    moss_utils.install_qwen35_kernel_dispatch()
    config = dict(moss_utils.QWEN35_9B_TEXT_CONFIG)
    config.update(hidden_size=64, intermediate_size=96, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                  head_dim=32, num_attention_heads=2, num_key_value_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
                  linear_num_key_heads=2, linear_num_value_heads=4, vocab_size=97, eos_token_id=1)
    config["rope_parameters"] = dict(config["rope_parameters"], mrope_section=[2, 1, 1])
    torch.manual_seed(3)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**config)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    return model


def _streamed_matches_resident(device: str):
    prompts = ["photo of ohwx man", "a red bicycle leaning on a wall at sunset"]
    resident = _tiny_text_encoder().to(device)
    expected = moss_utils.encode_prompts(resident, _CharTokenizer(), prompts)
    streamed = _tiny_text_encoder()  # same seed, same weights
    moss_utils.stream_text_encoder(streamed, device)
    got = moss_utils.encode_prompts(streamed, _CharTokenizer(), prompts)
    for a, b in zip(got, expected):
        assert a.device.type == torch.device(device).type and torch.allclose(a, b, atol=1e-5)
    # between forwards the layer weights live in CPU RAM; the embedding table never leaves it
    assert all(p.device.type == "cpu" for p in streamed.layers.parameters()) and streamed.embed_tokens.weight.device.type == "cpu"


def test_a_padded_batch_of_prompts_matches_one_forward_per_prompt():
    """encode_prompts runs up to TEXT_BATCH_SIZE prompts as one right-padded forward without a mask: every Qwen3.5 layer
    is causal, so each prompt's states equal its own unpadded forward's (ai-toolkit 0473387)."""
    model = _tiny_text_encoder()
    prompts = ["ohwx man", "a red bicycle leaning on a wall at sunset, long shadows on the cobblestones", "", "photo of ohwx dog"]
    singles = moss_utils.encode_prompts(model, _CharTokenizer(), prompts, batch_size=1)
    batched = moss_utils.encode_prompts(model, _CharTokenizer(), prompts)
    assert [b.shape for b in batched] == [s.shape for s in singles] and len({s.shape[0] for s in singles}) == 4
    # FP32 matrix products of another row count round differently (3e-5 on an AVX2 Xeon); a pad reaching a prompt would be O(1)
    for a, b in zip(batched, singles):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    pairs = moss_utils.encode_prompts(model, _CharTokenizer(), prompts, batch_size=3)  # a batch, then a single prompt
    for a, b in zip(pairs, singles):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_low_vram_text_encoding_matches_the_resident_text_encoder():
    """--low_vram_text_encoding (stream_text_encoder) looks the tokens up on the CPU and passes the embeddings in; the
    outputs must not change."""
    _streamed_matches_resident("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="streams the layers to the GPU")
def test_low_vram_text_encoding_on_the_gpu_matches_the_resident_text_encoder():
    _streamed_matches_resident("cuda")


@pytest.mark.parametrize(
    ("slow_cpu", "cuda", "asked", "expected"),
    [
        # BF16 on a CPU without fast BF16 (a Broadwell Xeon: 16 minutes per caption): stream on the GPU, else FP32
        (True, True, ("cpu", torch.bfloat16, False), ("cuda", torch.bfloat16, True)),
        (True, False, ("cpu", torch.bfloat16, False), ("cpu", torch.float32, False)),
        # a CPU with fast BF16, an explicit choice, or the GPU: unchanged
        (False, True, ("cpu", torch.bfloat16, False), ("cpu", torch.bfloat16, False)),
        (True, True, ("cpu", torch.float32, False), ("cpu", torch.float32, False)),
        (True, True, ("cuda", torch.bfloat16, False), ("cuda", torch.bfloat16, False)),
        (True, True, ("cuda", torch.bfloat16, True), ("cuda", torch.bfloat16, True)),
    ],
)
def test_text_encoder_plan_avoids_slow_cpu_bf16(monkeypatch, slow_cpu, cuda, asked, expected):
    monkeypatch.setattr(moss_utils, "cpu_bf16_is_slow", lambda: slow_cpu)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    device, dtype, stream = moss_utils.text_encoder_plan(*asked)
    assert (device.type, dtype, stream) == (torch.device(expected[0]).type, expected[1], expected[2])


def test_cpu_bf16_check_returns_a_verdict():
    from musubi_tuner.utils.model_utils import cpu_bf16_is_slow

    assert isinstance(cpu_bf16_is_slow(), bool)


def test_lora_targets_the_block_linears_under_comfyui_names():
    from musubi_tuner.networks import lora_moss_image

    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], _tiny())
    names = {m.lora_name for m in network.unet_loras}
    expected = {f"lora_unet_layers_{i}_proj_{p}" for i in range(PARAMS["num_layers"]) for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")}
    assert names == expected


def test_call_dit_targets_noise_minus_latents_on_the_vendor_timestep_scale():
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    model = _tiny()
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    args = argparse.Namespace(gradient_checkpointing=False)
    torch.manual_seed(2)
    latents = torch.randn(2, 8, 3, 4)
    noise = torch.randn_like(latents)
    noisy = 0.5 * latents + 0.5 * noise
    contexts = [torch.randn(7, 48), torch.randn(4, 48)]
    with torch.no_grad():
        out = trainer.call_dit(args, accelerator, model, latents, {"ctx_vec": contexts}, noise, noisy, torch.tensor([500.0, 500.0]), torch.float32)
        ctx, mask = moss_utils.pad_contexts(contexts, "cpu", torch.float32)
        expected = model(noisy, torch.tensor([500.0, 500.0]), ctx, mask)
    assert torch.allclose(out.pred, expected)
    assert torch.equal(out.target, noise - latents)


def test_the_target_stays_fp32_when_the_network_runs_in_bf16():
    """Full fine-tuning trains in BF16; the target is computed in FP32 like both reference trainers (it rounded the
    latents to BF16 before)."""
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    model = _tiny().to(torch.bfloat16)
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    torch.manual_seed(2)
    latents = torch.randn(2, 8, 3, 4)
    noise = torch.randn_like(latents)
    contexts = [torch.randn(7, 48), torch.randn(4, 48)]
    with torch.no_grad():
        out = trainer.call_dit(
            argparse.Namespace(gradient_checkpointing=False), accelerator, model, latents, {"ctx_vec": contexts}, noise,
            0.5 * latents + 0.5 * noise, torch.tensor([500.0, 500.0]), torch.bfloat16,
        )
    assert out.target.dtype == torch.float32 and torch.equal(out.target, noise - latents)


@pytest.mark.parametrize("trainer_kind", ["lora", "full"])
def test_the_dit_gets_the_vendor_timestep_of_the_noise_it_sees(trainer_kind):
    """The shared direct samplers return 1000 * sigma + 1 (HunyuanVideo's 1..1000 index); MOSS's DiT is conditioned on
    1000 * sigma, as the vendor's sampler, kohya's sd-scripts and sdbds's port pass it. LoRA runs the extended loop's
    sampler, full fine-tuning trainer_base's; uniform sampling of given bucket values makes both deterministic."""
    if trainer_kind == "lora":
        from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer as Trainer
    else:
        from musubi_tuner.moss_image_train import MossImageTrainer as Trainer

    args = argparse.Namespace(
        timestep_sampling="uniform", min_timestep=None, max_timestep=None, preserve_distribution_shape=False,
        sigmoid_scale=1.0, discrete_flow_shift=3.0,
    )
    torch.manual_seed(3)
    latents = torch.randn(2, 8, 3, 4)
    noise = torch.randn_like(latents)
    sigmas = [0.25, 0.8]
    noisy, timesteps = Trainer().get_noisy_model_input_and_timesteps(
        args, noise, latents, sigmas, None, torch.device("cpu"), torch.float32
    )
    assert torch.allclose(timesteps, torch.tensor(sigmas) * 1000.0, rtol=0, atol=1e-4)
    mix = torch.tensor(sigmas).view(-1, 1, 1, 1)
    assert torch.allclose(noisy, (1 - mix) * latents + mix * noise)


@pytest.mark.parametrize("precision, warned", [("fp16", True), ("bf16", False)])
def test_fp16_training_warns_about_the_residual_stream(caplog, precision, warned):
    """KBlueLeaf/MOSS-Image-FP16-fixed measured the residual stream near 62,000 at block 26 (FP16 ends at 65,504)."""
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    trainer.handle_model_specific_args(argparse.Namespace(mixed_precision=precision, sample_prompts=None, text_encoder=None))
    assert trainer.dit_dtype == (torch.float16 if precision == "fp16" else torch.bfloat16)
    assert ("65,504" in caplog.text) == warned


class _ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skips = []

    def forward(self, latents, timestep, ctx, mask, skip_layers=None):
        self.skips.append(tuple(skip_layers or ()))
        return torch.full_like(latents, float(ctx[0, 0, 0]))


def test_denoise_runs_cfg_with_the_negative_branch_skipping_block_22():
    model = _ConstantVelocity()
    noise = torch.randn(1, 8, 2, 3)
    ctx, mask = moss_utils.pad_contexts([torch.ones(3, 4)], "cpu", torch.float32)
    neg, neg_mask = moss_utils.pad_contexts([torch.zeros(5, 4)], "cpu", torch.float32)
    out = moss_utils.denoise(model, noise, ctx, mask, neg, neg_mask, 4, 3.0)
    assert torch.allclose(out, noise - 3.0, atol=1e-5)  # v = 0 + 3 * (1 - 0), integrated from sigma 1 to 0
    assert set(model.skips) == {(), (22,)}


def test_moss_image_is_registered_with_the_shared_code():
    from musubi_tuner.dataset import bucket
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
    from musubi_tuner.utils import sai_model_spec
    from musubi_tuner.verify_cache_files import resolve_architecture

    assert resolve_architecture(ARCHITECTURE_MOSS_IMAGE_FULL) == ARCHITECTURE_MOSS_IMAGE
    assert bucket.BucketSelector.RESOLUTION_STEPS_MOSS_IMAGE == 16
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_MOSS_IMAGE, 0.0, title="moss")
    assert metadata["modelspec.architecture"].startswith(sai_model_spec.ARCH_MOSS_IMAGE)


@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("per_row", [True, False])
def test_text_encoder_dequantizes_in_row_chunks_like_the_whole_tensor(convrot, per_row):
    """The 248320 x 4096 token table dequantized whole held 10.4 GiB on the GPU; row chunks give the same weights."""
    from musubi_tuner.modules.int8_convrot_utils import dequantize_int8_convrot_weight

    torch.manual_seed(0)
    q = torch.randint(-127, 128, (37, 256), dtype=torch.int8)
    scale = torch.rand(37, 1) * 0.01 if per_row else torch.tensor(0.004)
    quant = {"format": "int8_tensorwise", "convrot": convrot, "convrot_groupsize": 64}  # a power of 4
    if convrot:
        whole = dequantize_int8_convrot_weight(q, scale, 64, dtype=torch.bfloat16)
    else:
        whole = (q.float() * scale.float()).to(torch.bfloat16)
    chunked = moss_utils._dequantize_rows(q, scale, quant, torch.bfloat16, torch.device("cpu"), chunk_elements=256 * 5)
    assert chunked.device.type == "cpu"
    if convrot:  # the inverse rotation is a matrix product: another row count may round differently (an AVX2 Xeon did, 1e-8)
        torch.testing.assert_close(chunked.float(), whole.float(), atol=1e-6, rtol=2**-7)
    else:
        assert torch.equal(chunked, whole)


def _kernel_module(fast_impl):
    """A stand-in for transformers' modeling_qwen3_5: a reference function wrapped (functools.wraps) by a kernel path."""
    import functools

    def torch_chunk_gated_delta_rule(query, key, value, g=None, beta=None):
        return ("reference", query)

    @functools.wraps(torch_chunk_gated_delta_rule)
    def wrapped(*args, **kwargs):
        return fast_impl(*args, **kwargs)

    return types.SimpleNamespace(torch_chunk_gated_delta_rule=wrapped)


def test_qwen35_kernels_run_on_the_gpu_only_and_fall_back_when_they_fail(monkeypatch, caplog):
    """fla's Triton kernels cannot take CPU tensors (CPU text caching); a failing kernel hands over to the reference."""
    monkeypatch.setattr(moss_utils.importlib.util, "find_spec", lambda name: object())
    calls = []

    def fast(query, key, value, g=None, beta=None):
        calls.append("fast")
        return ("fast", query)

    module = _kernel_module(fast)
    assert moss_utils.install_qwen35_kernel_dispatch(module) == ["torch_chunk_gated_delta_rule"]
    assert moss_utils.install_qwen35_kernel_dispatch(module) == []  # installed once
    x = torch.zeros(2)
    assert module.torch_chunk_gated_delta_rule(x, x, x, g=x)[0] == "reference" and not calls  # CPU tensors
    monkeypatch.setattr(moss_utils, "_on_gpu", lambda tensor: True)
    assert module.torch_chunk_gated_delta_rule(x, x, x, g=x)[0] == "fast" and calls == ["fast"]

    def broken(*args, **kwargs):
        raise RuntimeError("triton build failed")

    module = _kernel_module(broken)
    moss_utils.install_qwen35_kernel_dispatch(module)
    assert module.torch_chunk_gated_delta_rule(x, x, x)[0] == "reference" and "triton build failed" in caplog.text
    assert module.torch_chunk_gated_delta_rule(x, x, x)[0] == "reference"  # the reference from then on


def test_qwen35_kernels_are_left_alone_without_the_packages(monkeypatch):
    monkeypatch.setattr(moss_utils.importlib.util, "find_spec", lambda name: None)
    module = _kernel_module(lambda *a, **k: ("fast", None))
    assert moss_utils.install_qwen35_kernel_dispatch(module) == []
