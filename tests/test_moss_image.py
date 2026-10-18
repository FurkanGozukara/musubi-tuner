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
