"""FLUX.3 [klein]: the DiT's wiring, text tokenization, checkpoint checks, the LoRA targets and the trainer's DiT call.

All on the CPU with a tiny model; the real model was compared with ai-toolkit's FLUX.3 transformer in fp32 (max abs
difference 1.5e-5 with references) and trained end to end on GPU before these were written.
"""

import argparse
import types

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.flux_3 import flux3_model, flux3_utils
from musubi_tuner.flux_3.flux3_model import Flux3, Flux3Params

DEPTH, DEPTH_SINGLE = 1, 4


def _tiny(**kwargs) -> Flux3:
    params = Flux3Params(
        in_channels=8, context_in_dim=16, hidden_size=64, num_heads=2, depth=DEPTH, depth_single_blocks=DEPTH_SINGLE, axes_dim=[8, 8, 8, 8]
    )
    torch.manual_seed(0)
    model = Flux3(params, **kwargs)
    with torch.no_grad():  # default inits (zero-like output layers) could hide wiring mistakes
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    return model.eval()


def _inputs(batch=2, height=2, width=3, text=5):
    torch.manual_seed(1)
    ctx = torch.randn(batch, text, 16)
    image = torch.randn(batch, height * width, 8)
    ids = flux3_model.image_ids(batch, height, width, 0, "cpu")
    t = torch.linspace(0.3, 0.9, batch)
    return ctx, image, ids, t


def test_forward_returns_the_image_tokens_and_references_condition_them():
    model = _tiny()
    ctx, image, ids, t = _inputs()
    with torch.no_grad():
        out = model(ctx, image, ids, t)
        ref = torch.randn(2, 4, 8)
        out_ref = model(ctx, image, ids, t, image_cond=ref, image_cond_ids=flux3_model.image_ids(2, 2, 2, -1, "cpu"))
    assert out.shape == image.shape and out_ref.shape == image.shape
    assert not torch.allclose(out, out_ref)


def test_references_are_refused_while_their_stream_stays_on_the_cpu():
    model = _tiny()
    model.keep_image_cond_on_cpu(True)
    ctx, image, ids, t = _inputs()
    with pytest.raises(RuntimeError, match="reference stream"):
        model(ctx, image, ids, t, image_cond=torch.randn(2, 4, 8), image_cond_ids=flux3_model.image_ids(2, 2, 2, -1, "cpu"))


def test_flash_auto_attention_matches_torch():
    """Guardrail 3: --sdpa resolves to "flash_auto" on machines without native flash SDPA."""
    model = _tiny(attn_mode="torch")
    ctx, image, ids, t = _inputs()
    with torch.no_grad():
        expected = model(ctx, image, ids, t)
        model.attn_mode = "flash_auto"
        out = model(ctx, image, ids, t)
    assert torch.allclose(out, expected, atol=1e-5)


def test_block_swap_keeps_two_joint_blocks_resident():
    """Guardrail 9: at most num_single_blocks - 2 joint blocks swap (26 of the real model's 28)."""
    model = _tiny()
    with pytest.raises(AssertionError, match=f"1 to {DEPTH_SINGLE - 2}"):
        model.enable_block_swap(DEPTH_SINGLE - 1, None)


def test_image_ids_follow_the_packed_token_order():
    ids = flux3_model.image_ids(1, 2, 3, -2, "cpu")[0]
    assert ids.shape == (6, 4)
    assert ids[:, 0].eq(0).all() and ids[:, 3].eq(-2).all()
    assert ids[4, 1:3].tolist() == [0.5, 0.5]  # token 4 = row 1, column 1, at RoPE scale 0.5
    latents = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    tokens = flux3_utils.pack_latents(latents)
    assert tokens[0, 4].tolist() == [4.0, 10.0]  # same row-major order
    assert torch.equal(flux3_utils.unpack_latents(tokens, 2, 3), latents)
    text = flux3_model.text_ids(1, 3, "cpu")[0]
    assert text[:, 3].tolist() == [0.0, 1.0, 2.0] and text[:, :3].eq(0).all()


class _Tokenizer:
    """Counts words; ids 1000 + i."""

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [1000 + i for i in range(len(text.split()))]}


def test_tokenization_pads_to_multiples_of_80_with_pads_at_position_one():
    prompt = "w " * 9  # the template adds "<|im_start|>user\n" ... "assistant\n" words
    real = len(flux3_utils.PROMPT_TEMPLATE.format(prompt).split())
    input_ids, mask, positions = flux3_utils.tokenize_prompt(_Tokenizer(), prompt)
    assert input_ids.shape == mask.shape == positions.shape == (80,)
    assert mask.sum().item() == real and input_ids[real:].eq(flux3_utils.TEXT_PAD_TOKEN_ID).all()
    assert positions[:real].tolist() == list(range(real)) and positions[real:].eq(1).all()

    long_prompt = "w " * 100
    input_ids, mask, _ = flux3_utils.tokenize_prompt(_Tokenizer(), long_prompt)
    assert input_ids.shape == (160,)
    input_ids, mask, _ = flux3_utils.tokenize_prompt(_Tokenizer(), long_prompt, max_tokens=96)
    assert input_ids.shape == (96,) and mask.all()  # truncated to the cap


def test_contexts_of_a_batch_are_zero_padded_to_the_longest():
    ctx = flux3_utils.pad_contexts([torch.ones(80, 4), torch.ones(160, 4)], "cpu", torch.float32)
    assert ctx.shape == (2, 160, 4)
    assert ctx[0, 80:].eq(0).all() and ctx[1].eq(1).all()


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        (["content_mode_blocks.image.0.q_proj.weight", "ref_time_in.in_layer.weight"], "step-distilled"),
        (["content_mode_blocks.image.0.q_proj.weight", "single_blocks.0.q_proj.weight_scale"], "pre-quantized"),
        (["content_mode_blocks.image.0.qkv.weight", "single_blocks.0.qkv.weight"], "pre-quantized"),  # fused repack
        (["double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight"], "not a FLUX.3"),  # FLUX.1 / FLUX.2
    ],
)
def test_checkpoints_that_cannot_train_are_refused(tmp_path, keys, message):
    path = str(tmp_path / "dit.safetensors")
    save_file({k: torch.zeros(1) for k in keys}, path)
    with pytest.raises(ValueError, match=message):
        flux3_utils.check_base_checkpoint(path)


def test_the_base_checkpoint_passes(tmp_path):
    path = str(tmp_path / "dit.safetensors")
    save_file({k: torch.zeros(1) for k in ("content_mode_blocks.image.0.q_proj.weight", "single_blocks.0.q_proj.weight")}, path)
    flux3_utils.check_base_checkpoint(path)


def _lora_names(model) -> set:
    from musubi_tuner.networks import lora_flux_3

    network = lora_flux_3.create_arch_network(1.0, 4, 4, None, [], model)
    return {m.lora_name for m in network.unet_loras}


def test_lora_targets_the_block_linears_under_comfyui_names():
    model = _tiny()
    model.trains_references = True
    names = _lora_names(model)
    per_block = ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out")
    blocks = [f"content_mode_blocks_{m}_0" for m in ("image", "image_cond")] + ["txt_mode_blocks_0"]
    blocks += [f"single_blocks_{i}" for i in range(DEPTH_SINGLE)]
    assert names == {f"lora_unet_{b}_{p}" for b in blocks for p in per_block}


def test_text_to_image_lora_leaves_out_the_reference_stream():
    model = _tiny()
    model.trains_references = False
    names = _lora_names(model)
    assert names and not any("image_cond" in n for n in names)
    assert "lora_unet_content_mode_blocks_image_0_q_proj" in names


def test_the_datasets_decide_whether_references_train():
    from musubi_tuner.flux_3_train_network import Flux3NetworkTrainer

    trainer = Flux3NetworkTrainer()
    t2i = types.SimpleNamespace(has_control=False, control_directory=None)
    trainer.configure_reference_target_ranges_from_datasets([t2i])
    assert trainer._trains_references is False
    trainer.configure_reference_target_ranges_from_datasets([t2i, types.SimpleNamespace(has_control=True)])
    assert trainer._trains_references is True


def test_call_dit_feeds_references_and_targets_noise_minus_latents():
    from musubi_tuner.flux_3_train_network import Flux3NetworkTrainer

    trainer = Flux3NetworkTrainer()
    model = _tiny()
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    args = argparse.Namespace(gradient_checkpointing=False)
    torch.manual_seed(2)
    latents = torch.randn(1, 8, 2, 3)
    noise = torch.randn_like(latents)
    noisy = 0.5 * latents + 0.5 * noise
    ctx = torch.randn(5, 16)
    reference = torch.randn(1, 8, 2, 2)
    batch = {"ctx_vec": [ctx], "latents_control_0": reference}
    with torch.no_grad():
        out = trainer.call_dit(args, accelerator, model, latents, batch, noise, noisy, torch.tensor([500.0]), torch.float32)
        expected = model(
            ctx[None],
            flux3_utils.pack_latents(noisy),
            flux3_model.image_ids(1, 2, 3, 0, "cpu"),
            torch.tensor([0.5]),
            image_cond=flux3_utils.pack_latents(reference),
            image_cond_ids=flux3_model.image_ids(1, 2, 2, -1, "cpu"),
        )
    assert torch.allclose(out.pred, flux3_utils.unpack_latents(expected, 2, 3))
    assert torch.equal(out.target, noise - latents)


class _ConstantVelocity(torch.nn.Module):
    """v = 1 for the prompt, 0 for the negative (the context's first value tells them apart)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, ctx, image, image_ids, timesteps, image_cond=None, image_cond_ids=None):
        self.calls += 1
        return torch.full_like(image, float(ctx[0, 0, 0]))


@pytest.mark.parametrize(("guidance", "calls_per_step", "velocity"), [(1.0, 1, 1.0), (3.5, 2, 3.5)])
def test_denoise_integrates_from_noise_to_zero_with_cfg(guidance, calls_per_step, velocity):
    model = _ConstantVelocity()
    noise = torch.randn(1, 8, 2, 3)
    ctx, neg = torch.ones(1, 3, 16), torch.zeros(1, 5, 16)  # different lengths run as separate forwards
    out = flux3_utils.denoise(model, noise, ctx, neg, 4, guidance, dtype=torch.float32)
    assert model.calls == 4 * calls_per_step
    assert torch.allclose(out, noise - velocity, atol=1e-6)  # the schedule runs from t = 1 to t = 0
    schedule = flux3_utils.get_schedule(4)
    assert schedule[0].item() == 1.0 and schedule[-1].item() == 0.0
    assert abs(flux3_utils.timeshift(6.93, torch.tensor(0.5)).item() - 6.93 * 0.5 / (1 + 5.93 * 0.5)) < 1e-7


def test_flux3_is_registered_with_the_shared_dataset_and_metadata_code():
    from musubi_tuner.dataset import bucket
    from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_3, ARCHITECTURE_FLUX_3_FULL
    from musubi_tuner.utils import sai_model_spec
    from musubi_tuner.verify_cache_files import resolve_architecture

    assert resolve_architecture(ARCHITECTURE_FLUX_3_FULL) == ARCHITECTURE_FLUX_3
    assert bucket.BucketSelector.RESOLUTION_STEPS_FLUX_3 == 16
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_FLUX_3, 0.0, title="flux3")
    assert metadata["modelspec.architecture"].startswith(sai_model_spec.ARCH_FLUX_3_KLEIN)
