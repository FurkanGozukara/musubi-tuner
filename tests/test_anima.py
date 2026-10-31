"""Anima: the DiT's wiring, text batching, checkpoint loading, the ComfyUI sampler, the LoRA targets and the trainer's
DiT call.

All on the CPU with a tiny model. The port itself was compared in fp32 on random weights with kohya-ss sd-scripts'
anima_models, ComfyUI's comfy/ldm/anima and diffusers' CosmosTransformer3DModel + AnimaTextConditioner
(dont_upload/vram_lab/anima_ref_check.py) before these were written.
"""

import argparse
import types

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.anima import anima_model, anima_utils
from musubi_tuner.anima.anima_model import Anima, AnimaParams

PARAMS = dict(
    model_channels=64,
    num_blocks=3,
    num_heads=2,
    crossattn_emb_channels=32,
    adaln_lora_dim=16,
    adapter_dim=32,
    adapter_layers=2,
    adapter_heads=2,
    adapter_vocab_size=50,
)


def _tiny(attn_mode: str = "torch") -> Anima:
    torch.manual_seed(0)
    model = Anima(AnimaParams(**PARAMS), attn_mode=attn_mode)
    with torch.no_grad():  # default inits could hide wiring mistakes
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    return model.eval()


def _text(lengths=(7, 4), t5_lengths=(5, 9)):
    torch.manual_seed(1)
    states = [torch.randn(n, 32) for n in lengths]
    ids = [torch.randint(2, 50, (n,)) for n in t5_lengths]
    for t in ids:
        t[-1] = anima_model.T5_EOS_TOKEN_ID
    return states, ids


def test_a_masked_caption_batch_equals_one_caption_at_a_time():
    """ComfyUI runs each caption's adapter alone, unpadded; a training batch pads and masks. The real positions must
    match, the padded ones be zero, and the context reach 512 tokens."""
    model = _tiny()
    states, ids = _text()
    batch_states, states_mask, batch_ids, ids_mask = anima_utils.batch_text(states, ids, "cpu", torch.float32)
    assert states_mask is not None and ids_mask is not None
    with torch.no_grad():
        batched = model.encode_context(batch_states, batch_ids, ids_mask, states_mask)
        for i, (s, t) in enumerate(zip(states, ids)):
            single = model.encode_context(s[None], t[None])
            assert single.shape == (1, anima_model.CONTEXT_TOKENS, 32)
            torch.testing.assert_close(batched[i, : t.shape[0]], single[0, : t.shape[0]], atol=1e-5, rtol=1e-5)
            assert batched[i, t.shape[0] :].abs().max() == 0 and single[0, t.shape[0] :].abs().max() == 0


def test_one_caption_runs_without_masks():
    states, ids = _text(lengths=(6,), t5_lengths=(3,))
    _, states_mask, _, ids_mask = anima_utils.batch_text(states, ids, "cpu", torch.float32)
    assert states_mask is None and ids_mask is None  # exactly ComfyUI's unmasked forward


def test_forward_takes_4d_and_5d_latents_and_the_adapter_inside():
    model = _tiny()
    states, ids = _text(lengths=(6,), t5_lengths=(3,))
    torch.manual_seed(2)
    x = torch.randn(1, 16, 8, 12)
    t = torch.tensor([0.7])
    with torch.no_grad():
        ctx = model.encode_context(states[0][None], ids[0][None])
        out4 = model(x, t, ctx)
        out5 = model(x.unsqueeze(2), t, ctx)
        inside = model(x, t, states[0][None], t5_ids=ids[0][None])
    assert out4.shape == x.shape and out5.shape == (1, 16, 1, 8, 12)
    torch.testing.assert_close(out5.squeeze(2), out4)
    torch.testing.assert_close(inside, out4)


def test_the_resolved_flash_auto_mode_is_accepted():
    """Guardrail 3: --sdpa resolves to "flash_auto" where PyTorch has no native flash kernel; on the CPU it falls back to
    PyTorch SDPA with the same result."""
    reference, auto = _tiny("torch"), _tiny("flash_auto")
    states, ids = _text(lengths=(5,), t5_lengths=(4,))
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        ctx = reference.encode_context(states[0][None], ids[0][None])
        torch.testing.assert_close(auto(x, torch.tensor([0.3]), ctx), reference(x, torch.tensor([0.3]), ctx))


def test_block_swap_keeps_two_blocks_resident():
    """Guardrail 9: at most num_blocks - 2 (26 of 28) blocks swap."""
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    model = _tiny()
    with pytest.raises(AssertionError):
        model.enable_block_swap(PARAMS["num_blocks"] - 1, BlockSwapConfig(device=torch.device("cpu"), supports_backward=True))


def _checkpoint(tmp_path, prefix: str):
    model = _tiny()
    sd = {f"{prefix}{k}": v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    path = tmp_path / f"anima_{prefix.strip('.') or 'bare'}.safetensors"
    save_file(sd, str(path))
    return model, str(path)


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
def test_the_released_key_layouts_load(tmp_path, prefix):
    """Base / preview files carry net., the Aesthetic and Turbo files model.diffusion_model."""
    reference, path = _checkpoint(tmp_path, prefix)
    model = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.bfloat16, params=AnimaParams(**PARAMS))
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value.to(torch.bfloat16)), key


def test_files_that_cannot_train_are_refused(tmp_path):
    save_file({"net.blocks.0.self_attn.q_proj.weight": torch.zeros(2, 2)}, str(tmp_path / "x.safetensors"))
    with pytest.raises(ValueError, match="not an Anima DiT"):
        anima_utils.check_dit_checkpoint(str(tmp_path / "x.safetensors"))
    _, path = _checkpoint(tmp_path, "net.")
    from safetensors.torch import load_file

    sd = load_file(path)
    sd["net.blocks.1.mlp.layer1.comfy_quant"] = torch.zeros(1, dtype=torch.uint8)
    save_file(sd, str(tmp_path / "int8.safetensors"))
    with pytest.raises(ValueError, match="quantized"):
        anima_utils.check_dit_checkpoint(str(tmp_path / "int8.safetensors"))


def test_the_saved_layout_is_the_released_one():
    sd = anima_model.comfy_state_dict(_tiny().state_dict())
    assert all(k.startswith("net.") for k in sd) and "net.llm_adapter.out_proj.weight" in sd
    assert "net.x_embedder.proj.1.weight" in sd and "net.t_embedder.1.linear_2.weight" in sd


class _CharTokenizer:
    """Stand-in tokenizer: one id per character (Qwen: no special tokens; T5: closing </s> = 1)."""

    def __init__(self, eos=None):
        self.eos = eos

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + ord(c) % 40 for c in text]
        if self.eos is not None and add_special_tokens:
            ids = ids + [self.eos]
        return {"input_ids": ids}


def test_tokenization_follows_comfyui():
    qwen, t5 = _CharTokenizer(), _CharTokenizer(eos=1)
    assert anima_utils.tokenize(qwen, t5, "") == ([anima_utils.QWEN_PAD_TOKEN_ID], [1])  # min length 1, a pad; </s> alone
    q, t = anima_utils.tokenize(qwen, t5, "a cat")
    assert len(q) == 5 and t[-1] == 1 and len(t) == 6
    q, t = anima_utils.tokenize(qwen, t5, "x" * 600)
    assert len(q) == anima_utils.MAX_QWEN_TOKENS and len(t) == anima_utils.MAX_T5_TOKENS and t[-1] == 1


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3Model

    config = dict(anima_utils.QWEN3_06B_CONFIG)
    config.update(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=200000)
    torch.manual_seed(3)
    model = Qwen3Model(Qwen3Config(**config)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    return model


def test_a_padded_batch_of_prompts_matches_one_forward_per_prompt():
    """Qwen3 is causal: right-padded prompts keep their own states."""
    model = _tiny_qwen3()
    prompts = ["ohwx girl", "a red bicycle leaning on a wall at sunset", "", "1girl, solo"]
    qwen, t5 = _CharTokenizer(), _CharTokenizer(eos=1)
    singles = anima_utils.encode_prompts(model, qwen, t5, prompts, batch_size=1)
    batched = anima_utils.encode_prompts(model, qwen, t5, prompts, batch_size=4)
    for (a, ia), (b, ib) in zip(batched, singles):
        assert a.shape == b.shape and torch.equal(ia, ib)
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_simple_schedule_is_comfyuis():
    sigmas = anima_utils.simple_schedule(30, 3.0)
    assert len(sigmas) == 31 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    # ComfyUI: model_sampling.sigmas[-(1 + int(x * 1000 / 30))], the grid sigma(i / 1000), i = 1..1000
    assert torch.isclose(sigmas[1], torch.tensor(anima_utils.time_snr_shift(3.0, 967 / 1000)))
    assert torch.all(sigmas[:-1] > sigmas[1:])


class _ToTarget(torch.nn.Module):
    """A 'DiT' whose x0 prediction is a fixed target: v = (x - target) / sigma."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    @property
    def dtype(self):
        return torch.float32

    def forward(self, x, t, ctx):
        self.calls += 1
        return (x - self.target.expand_as(x)) / t.reshape(-1, 1, 1, 1, 1)


@pytest.mark.parametrize("sampler", ["er_sde", "euler"])
def test_samplers_land_on_the_prediction_and_run_cfg_as_one_batch(sampler):
    target = torch.full((1, 16, 1, 2, 3), 0.25)
    model = _ToTarget(target)
    ctx = torch.zeros(1, 4, 32)
    out = anima_utils.denoise(model, ctx, ctx, 16, 24, steps=6, guidance=4.0, sampler=sampler, seed=7)
    torch.testing.assert_close(out, target)  # the last step is the x0 prediction (CFG of equal branches)
    assert model.calls == 6  # cond and uncond in one batch per step


def test_initial_noise_is_comfyuis_cpu_noise():
    expected = torch.randn((1, 16, 1, 4, 4), generator=torch.manual_seed(1234), dtype=torch.float32)
    torch.testing.assert_close(anima_utils.initial_noise((1, 16, 1, 4, 4), 1234, "cpu"), expected)


def test_lora_targets_the_block_linears_under_comfyui_names():
    from musubi_tuner.networks import lora_anima

    network = lora_anima.create_arch_network(1.0, 4, 4, None, [], _tiny())
    names = {m.lora_name for m in network.unet_loras}
    expected = {
        f"lora_unet_blocks_{i}_{part}"
        for i in range(PARAMS["num_blocks"])
        for part in (
            "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_output_proj",
            "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj", "cross_attn_output_proj",
            "mlp_layer1", "mlp_layer2",
        )
    }
    assert names == expected
    with_adapter = lora_anima.create_arch_network(1.0, 4, 4, None, [], _tiny(), train_llm_adapter="True")
    adapter = {m.lora_name for m in with_adapter.unet_loras} - expected
    assert "lora_unet_llm_adapter_blocks_0_self_attn_q_proj" in adapter and "lora_unet_llm_adapter_blocks_1_mlp_2" in adapter
    assert not any("embed" in name or "out_proj" in name for name in adapter)


def test_call_dit_targets_noise_minus_latents_at_sigma():
    from musubi_tuner.anima_train_network import AnimaNetworkTrainer

    trainer = AnimaNetworkTrainer()
    model = _tiny()
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    args = argparse.Namespace(gradient_checkpointing=False)
    torch.manual_seed(2)
    latents = torch.randn(2, 16, 1, 4, 6)
    noise = torch.randn_like(latents)
    noisy = 0.5 * latents + 0.5 * noise
    states, ids = _text()
    with torch.no_grad():
        out = trainer.call_dit(args, accelerator, model, latents, {"ctx_vec": states, "t5_ids": ids}, noise, noisy, torch.tensor([500.0, 500.0]), torch.float32)
        s, sm, t, tm = anima_utils.batch_text(states, ids, "cpu", torch.float32)
        expected = model(noisy, torch.tensor([0.5, 0.5]), model.encode_context(s, t, tm, sm))
    torch.testing.assert_close(out.pred, expected)
    assert out.target.dtype == torch.float32 and torch.equal(out.target, noise - latents)


def test_the_dit_gets_the_sigma_of_the_noise_it_sees():
    """The shared direct samplers return 1000 * sigma + 1; Anima's DiT takes sigma itself."""
    from musubi_tuner.anima_train_network import AnimaNetworkTrainer

    args = argparse.Namespace(
        timestep_sampling="uniform", min_timestep=None, max_timestep=None, preserve_distribution_shape=False,
        sigmoid_scale=1.0, discrete_flow_shift=3.0,
    )
    torch.manual_seed(3)
    latents = torch.randn(2, 16, 1, 4, 6)
    noise = torch.randn_like(latents)
    sigmas = [0.25, 0.8]
    noisy, timesteps = AnimaNetworkTrainer().get_noisy_model_input_and_timesteps(args, noise, latents, sigmas, None, torch.device("cpu"), torch.float32)
    assert torch.allclose(timesteps, torch.tensor(sigmas) * 1000.0, rtol=0, atol=1e-4)
    mix = torch.tensor(sigmas).view(-1, 1, 1, 1, 1)
    assert torch.allclose(noisy, (1 - mix) * latents + mix * noise)


def test_sampling_without_a_negative_uses_comfyuis_default_negative():
    from musubi_tuner.anima_train_network import AnimaNetworkTrainer

    assert AnimaNetworkTrainer._negative({}) == anima_utils.DEFAULT_NEGATIVE_PROMPT
    assert AnimaNetworkTrainer._negative({"negative_prompt": " "}) == ""
    assert AnimaNetworkTrainer._negative({"negative_prompt": "bad hands"}) == "bad hands"


def test_anima_is_registered_with_the_shared_code():
    from musubi_tuner.dataset import bucket
    from musubi_tuner.dataset.architectures import ARCHITECTURE_ANIMA, ARCHITECTURE_ANIMA_FULL
    from musubi_tuner.networks.network_arch import detect_arch_config
    from musubi_tuner.utils import sai_model_spec
    from musubi_tuner.verify_cache_files import resolve_architecture

    assert resolve_architecture(ARCHITECTURE_ANIMA_FULL) == ARCHITECTURE_ANIMA
    assert bucket.BucketSelector.ARCHITECTURE_STEPS_MAP[ARCHITECTURE_ANIMA] == 16
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_ANIMA, 0.0, title="anima")
    assert metadata["modelspec.architecture"].startswith(sai_model_spec.ARCH_ANIMA)
    targets, _ = detect_arch_config(_tiny())
    assert targets == ["AnimaBlock"]


def test_text_caches_round_trip_through_the_writer(tmp_path):
    from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_anima
    from safetensors.torch import load_file

    item = types.SimpleNamespace(text_encoder_output_cache_path=str(tmp_path / "x_an_te.safetensors"), caption="a cat", item_key="x")
    states, ids = _text(lengths=(3,), t5_lengths=(4,))
    save_text_encoder_output_cache_anima(item, states[0].to(torch.bfloat16), ids[0])
    sd = load_file(item.text_encoder_output_cache_path)
    assert set(sd) == {"varlen_ctx_vec_bfloat16", "varlen_t5_ids_int64"}
    assert torch.equal(sd["varlen_t5_ids_int64"], ids[0])


def test_fp8_scaled_quantizes_only_the_block_linears(tmp_path):
    """FP8 Scaled: the 28 blocks' ten Linears per block in FP8 with their scales (the released files' net. prefix stripped
    after quantizing); the adapter, modulations, embedders and final layer keep the file's precision."""
    _, path = _checkpoint(tmp_path, "net.")
    model = anima_utils.load_dit("cpu", path, "torch", False, "cpu", None, fp8_scaled=True, params=AnimaParams(**PARAMS))
    fp8 = {name for name, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn}
    assert len(fp8) == 10 * PARAMS["num_blocks"] and all(name.startswith("blocks.") for name in fp8)
    assert not any("adaln_modulation" in name or "norm" in name for name in fp8)
    reference = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.bfloat16, params=AnimaParams(**PARAMS))
    states, ids = _text(lengths=(5,), t5_lengths=(4,))
    x = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        a = reference(x, torch.tensor([0.4]), states[0][None], t5_ids=ids[0][None]).float()
        b = model(x, torch.tensor([0.4]), states[0][None], t5_ids=ids[0][None]).float()
    assert torch.isfinite(b).all() and ((a - b).norm() / a.norm()) < 0.1


# ConvRot groups are 256 inputs wide: block 2 of a 4-block DiT with 256 channels has eight quantizable projections (the
# cross-attention k / v read the 32-wide context and stay unquantized, as quantize_weight_convrot leaves them)
INT8_PARAMS = dict(PARAMS, model_channels=256, num_blocks=4, num_heads=2)
INT8_SPEC = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
INT8_LAYERS = {
    f"blocks.2.{name}.weight"
    for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "cross_attn.q_proj",
        "cross_attn.output_proj", "mlp.layer1", "mlp.layer2",
    )
}


def _int8_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = Anima(AnimaParams(**INT8_PARAMS))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    path = str(tmp_path / "anima-base-v1.0.safetensors")
    save_file({f"net.{k}": v.contiguous() for k, v in model.state_dict().items()}, path)
    return model.eval(), path


def test_convrot_int8_quantizes_the_middle_blocks_while_loading(tmp_path):
    """As ruwwww/Anima-ConvRot: blocks 0, 1 and the last stay BF16, so do the modulations, embedders and the adapter."""
    reference, path = _int8_checkpoint(tmp_path)
    model = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, params=AnimaParams(**INT8_PARAMS), convrot_int8=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == INT8_LAYERS
    assert model.is_convrot_int8 and not any(p.requires_grad for p in model.parameters())
    states, ids = _text(lengths=(5,), t5_lengths=(4,))
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        expected = reference(x, torch.tensor([0.5]), states[0][None], t5_ids=ids[0][None])
        out = model(x, torch.tensor([0.5]), states[0][None], t5_ids=ids[0][None])
    assert ((out - expected).norm() / expected.norm()).item() < 0.02


def test_an_int8_convrot_file_is_used_as_it_is_and_only_by_lora(tmp_path):
    """A comfy_quant INT8 ConvRot file (ruwwww/Anima-ConvRot's layout: net. keys, .weight_scale, .comfy_quant) loads its
    INT8 layers as stored, equal to the BF16 file quantized while loading; fine-tuning refuses it."""
    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
    from safetensors.torch import load_file

    _, path = _int8_checkpoint(tmp_path)
    params = AnimaParams(**INT8_PARAMS)
    quantized_on_load = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, params=params, convrot_int8=True).state_dict()
    released = {}
    for key, value in load_file(path).items():
        bare = key[len("net."):]
        if bare in INT8_LAYERS:
            module = key[: -len(".weight")]
            released[key], released[module + ".weight_scale"] = quantize_int8_convrot_weight(value, 256)
            released[module + ".comfy_quant"] = torch.tensor(list(INT8_SPEC), dtype=torch.uint8)
        else:
            released[key] = value
    int8_path = str(tmp_path / "anima-base-v1.0-convrot-int8.safetensors")
    save_file(released, int8_path)
    with pytest.raises(ValueError, match="Fine-tuning trains the BF16 base"):
        anima_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32, params=params)
    model = anima_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32, params=params, allow_int8_file=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == INT8_LAYERS
    state = model.state_dict()
    assert state.keys() == quantized_on_load.keys()
    for name, tensor in state.items():
        expected = quantized_on_load[name]
        if tensor.dtype == torch.int8:
            assert (tensor.int() - expected.int()).abs().max().item() <= 1, name
        else:
            assert torch.allclose(tensor, expected, rtol=1e-5, atol=0), name


def test_dora_trains_on_the_int8_convrot_dit(tmp_path):
    """DoRA reads the INT8 base in its original basis: with the up matrices at zero the adapted DiT is the base, and the
    adapter gets gradients through the INT8 layers' backward."""
    from musubi_tuner.networks import lora_anima

    _, path = _int8_checkpoint(tmp_path)
    model = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, params=AnimaParams(**INT8_PARAMS), convrot_int8=True)
    states, ids = _text(lengths=(5,), t5_lengths=(4,))
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        base = model(x, torch.tensor([0.5]), states[0][None], t5_ids=ids[0][None])
    network = lora_anima.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        adapted = model(x, torch.tensor([0.5]), states[0][None], t5_ids=ids[0][None])
    torch.testing.assert_close(adapted, base, rtol=1e-4, atol=1e-4)
    for lora in network.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.01)
    out = model(x, torch.tensor([0.5]), states[0][None], t5_ids=ids[0][None])
    out.square().mean().backward()
    int8_modules = {m.lora_name for m in network.unet_loras if m.lora_name.startswith("lora_unet_blocks_2_") and "cross_attn_k" not in m.lora_name and "cross_attn_v" not in m.lora_name}
    assert int8_modules and all(m.lora_down.weight.grad is not None for m in network.unet_loras if m.lora_name in int8_modules)


def test_the_trainer_refuses_int8_with_fp8():
    from musubi_tuner.anima_train_network import AnimaNetworkTrainer

    args = argparse.Namespace(mixed_precision="bf16", sample_prompts=None, text_encoder=None, fp8_base=True, fp8_scaled=True, convrot_int8=True, sample_solver="er_sde")
    with pytest.raises(ValueError, match="choose one"):
        AnimaNetworkTrainer().handle_model_specific_args(args)


def test_block_swap_gets_one_int8_layout_for_every_block(tmp_path):
    """The offloaders swap tensors between blocks: with block swap every block's projections are INT8 (an INT8 file's
    BF16 blocks 0, 1 and the last are quantized while loading), so no BF16 / INT8 pair meets in a swap."""
    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
    from safetensors.torch import load_file

    _, path = _int8_checkpoint(tmp_path)
    params = AnimaParams(**INT8_PARAMS)
    every_block = {name.replace("blocks.2.", f"blocks.{i}.") for name in INT8_LAYERS for i in range(INT8_PARAMS["num_blocks"])}
    model = anima_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, params=params, convrot_int8=True, convrot_int8_all_blocks=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == every_block
    released = {}
    for key, value in load_file(path).items():
        if key[len("net."):] in INT8_LAYERS:
            module = key[: -len(".weight")]
            released[key], released[module + ".weight_scale"] = quantize_int8_convrot_weight(value, 256)
            released[module + ".comfy_quant"] = torch.tensor(list(INT8_SPEC), dtype=torch.uint8)
        else:
            released[key] = value
    int8_path = str(tmp_path / "anima-int8.safetensors")
    save_file(released, int8_path)
    model = anima_utils.load_dit("cpu", int8_path, "torch", False, "cpu", torch.float32, params=params, allow_int8_file=True, convrot_int8_all_blocks=True)
    assert {name for name, p in model.named_parameters() if p.dtype == torch.int8} == every_block
