"""V39: a real DiT block with ConvRot INT8 projections (and a DoRA network on them) compiles without graph breaks.

The MOSS-Image, MiniMax H3, Kandinsky 6 and Krea 2 trainers no longer keep INT8 Linears out of torch.compile (only block
swap does), so a compiled block must trace through the patched Linears (kohya-ss/sd-scripts a4aaa20). CPU, tiny model.
"""

import torch
from safetensors.torch import save_file

from musubi_tuner.moss_image import moss_model, moss_utils
from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams

# ConvRot groups are 256 inputs wide: the smallest DiT whose block projections all quantize
INT8_PARAMS = dict(dim=256, ffn_dim=256, num_heads=4, num_layers=2, text_dim=48, freq_dim=16, latent_channels=8)


def _int8_dit(tmp_path, monkeypatch):
    torch.manual_seed(0)
    params = MossImageParams(**INT8_PARAMS)
    model = MossImage(params)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    monkeypatch.setattr(moss_utils.moss_model, "MossImageParams", lambda: params)
    path = str(tmp_path / "moss_image_base_bf16.safetensors")
    save_file({k: v.contiguous() for k, v in moss_model.fuse_qkv(model.state_dict(), params).items()}, path)
    return moss_utils.load_dit("cpu", path, "torch", False, "cpu", torch.float32, convrot_int8=True)


def _inputs():
    torch.manual_seed(1)
    x = torch.randn(2, 8, 3, 4)
    t = torch.tensor([300.0, 900.0])
    ctx = torch.randn(2, 7, 48)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 4:] = False
    return x, t, ctx, mask


def _block_inputs(model, x, t, ctx, mask):
    captured = []
    handle = model.layers[0].register_forward_pre_hook(lambda module, args, kwargs: captured.append((args, kwargs)), with_kwargs=True)
    model(x, t, ctx, mask)  # also runs the one-time DoRA base checks before any compiled block
    handle.remove()
    return captured[0]


def test_int8_dora_block_compiles_without_graph_breaks(tmp_path, monkeypatch):
    from musubi_tuner.networks import lora_moss_image

    model = _int8_dit(tmp_path, monkeypatch)
    assert model.is_convrot_int8
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    args, kwargs = _block_inputs(model, *_inputs())
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model.layers[0])(*args, **kwargs)
    assert explanation.graph_break_count == 0, explanation.break_reasons


def test_compiled_int8_dora_block_matches_eager(tmp_path, monkeypatch):
    from musubi_tuner.networks import lora_moss_image

    model = _int8_dit(tmp_path, monkeypatch)
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for p in network.parameters():
            if p.requires_grad and p.dim() == 2 and p.abs().sum() == 0:
                p.normal_(0.0, 0.05)  # move the zero-initialized up matrices so the adapter contributes
    args, kwargs = _block_inputs(model, *_inputs())
    block = model.layers[0]
    torch._dynamo.reset()
    compiled = torch.compile(block, backend="aot_eager", fullgraph=True)

    def run(module):
        hidden = args[0].detach().clone().requires_grad_(True)
        out = module(hidden, *args[1:], **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        out.float().square().mean().backward()
        grads = [p.grad.clone() for p in network.parameters() if p.requires_grad and p.grad is not None]
        network.zero_grad(set_to_none=True)
        return out.detach(), hidden.grad.detach(), grads

    out_c, gx_c, g_c = run(compiled)
    out_e, gx_e, g_e = run(block)
    assert torch.allclose(out_c, out_e, rtol=1e-5, atol=1e-5)
    assert torch.allclose(gx_c, gx_e, rtol=1e-4, atol=1e-6)
    assert len(g_c) == len(g_e) > 0
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(g_c, g_e))
