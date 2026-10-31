"""Krea 2 Edit: reference tokens, their t = 0 modulation, cache fingerprints and the edit prompt's image patches.

The DiT is also matched against ComfyUI's own Krea 2 model (index_timestep_zero) by dont_upload/vram_lab/krea2_edit_comfy_equiv.py
(max difference 1.8e-7 in float32), and the edit text encoding against ComfyUI's Text Encode Qwen Image Edit Plus node by
krea2_edit_te_check.py.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from musubi_tuner.krea2 import krea2_mmdit, krea2_utils
from musubi_tuner.krea2.krea2_encoder import comfy_qwen3vl_patches, edit_vision_pixels
from musubi_tuner.krea2_train_network import reference_latents_from_batch

CONFIG = dict(features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=2, patch=2, channels=4, txtlayers=3, txtheads=4,
              txtkvheads=4)


def _model():
    torch.manual_seed(0)
    model = krea2_mmdit.SingleStreamDiT(krea2_mmdit.SingleMMDiTConfig(**CONFIG), attn_mode="torch")
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * (0.05 if param.dim() > 1 else 0.2))
    return model.eval()


def _tokens(latent):
    batch, _, height, width = latent.shape
    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] = torch.arange(height // 2)[:, None]
    ids[..., 2] = torch.arange(width // 2)[None, :]
    return rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2), ids.reshape(1, -1, 3).expand(batch, -1, -1)


def test_reference_tokens_take_their_index_on_the_first_axis():
    refs = [torch.randn(2, 4, 4, 6), torch.randn(2, 4, 2, 2)]
    tokens, pos = krea2_utils.reference_tokens(refs, 2, torch.device("cpu"), torch.float32)
    assert tokens.shape == (2, 6 + 1, 16) and pos.shape == (2, 7, 3)
    assert pos[0, :6, 0].tolist() == [1.0] * 6 and pos[0, 6, 0].item() == 2.0
    assert pos[0, :6, 1:].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    with pytest.raises(ValueError):
        krea2_utils.reference_tokens([torch.randn(1, 4, 3, 4)], 2, torch.device("cpu"), torch.float32)


def _reference_forward(model, x, context, t, refs):
    """ComfyUI's Krea 2 order ([text | image | references]), references modulated with t = 0, without masks (batch 1)."""
    img, imgpos = _tokens(x)
    imglen = img.shape[1]
    ref_tokens, ref_pos = krea2_utils.reference_tokens(refs, 2, torch.device("cpu"), torch.float32)
    tokens = model.first(torch.cat((img, ref_tokens), 1))
    temb = model.tmlp(krea2_mmdit.temb(t, CONFIG["tdim"], dtype=tokens.dtype))
    vec, vec0 = model.tproj(temb), model.tproj(model.tmlp(krea2_mmdit.temb(torch.zeros_like(t), CONFIG["tdim"], dtype=tokens.dtype)))
    text = model.txtmlp(model.txtfusion(context))
    split = text.shape[1] + imglen
    x_all = torch.cat((text, tokens), 1)
    freqs = model.posemb(torch.cat((torch.zeros(1, text.shape[1], 3), imgpos, ref_pos), 1))
    for block in model.blocks:
        m, r = block.mod(vec), block.mod(vec0)

        def modulate(h, scale, shift):
            return torch.cat(((1 + m[scale]) * h[:, :split] + m[shift], (1 + r[scale]) * h[:, split:] + r[shift]), 1)

        def gate(h, index):
            return torch.cat((m[index] * h[:, :split], r[index] * h[:, split:]), 1)

        attn = block.attn
        q_in = modulate(block.prenorm(x_all), 0, 1)
        q, k, v = attn.wq(q_in), attn.wk(q_in), attn.wv(q_in)
        q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads).repeat_interleave(attn.heads // attn.kvheads, 1)
        v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads).repeat_interleave(attn.heads // attn.kvheads, 1)
        q, k = attn.qknorm.qnorm(q), attn.qknorm.knorm(k)
        q, k = krea2_mmdit.ropeapply(q, k, freqs)
        out = rearrange(F.scaled_dot_product_attention(q, k, v), "B H L D -> B L (H D)")
        x_all = x_all + gate(attn.wo(out * torch.sigmoid(attn.gate(q_in))), 2)
        x_all = x_all + gate(block.mlp(modulate(block.postnorm(x_all), 3, 4)), 5)
    final = model.last(x_all, temb)
    return final[:, text.shape[1] : text.shape[1] + imglen]


@pytest.mark.parametrize("count", [1, 2])
def test_references_lead_the_sequence_and_match_comfyuis_order(count):
    model = _model()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 8, 12)
    context = torch.randn(1, 5, CONFIG["txtlayers"], CONFIG["txtdim"])
    t = torch.tensor([0.4])
    refs = [torch.randn(1, 4, 6, 10), torch.randn(1, 4, 8, 8)][:count]
    img, imgpos = _tokens(x)
    ref_tokens, ref_pos = krea2_utils.reference_tokens(refs, 2, torch.device("cpu"), torch.float32)
    tokens, pos = torch.cat((ref_tokens, img), 1), torch.cat((ref_pos, imgpos), 1)
    mask = torch.ones(1, tokens.shape[1] + 5, dtype=torch.bool)
    with torch.no_grad():
        ours = model(img=tokens, context=context, t=t, pos=torch.cat((pos, torch.zeros(1, 5, 3)), 1), mask=mask, ref_len=ref_tokens.shape[1])
        expected = _reference_forward(model, x, context, t, refs)
        plain = model(img=img, context=context, t=t, pos=torch.cat((imgpos, torch.zeros(1, 5, 3)), 1),
                      mask=torch.ones(1, img.shape[1] + 5, dtype=torch.bool))
    assert ours.shape == img.shape
    assert torch.allclose(ours, expected, atol=2e-5, rtol=0)
    assert not torch.allclose(ours, plain, atol=1e-4)  # the references change the prediction


def test_references_keep_their_t0_modulation_under_gradient_checkpointing():
    model = _model()
    model.enable_gradient_checkpointing()
    model.train()
    x = torch.randn(1, 4, 8, 8)
    img, imgpos = _tokens(x)
    ref_tokens, ref_pos = krea2_utils.reference_tokens([torch.randn(1, 4, 8, 8)], 2, torch.device("cpu"), torch.float32)
    tokens = torch.cat((ref_tokens, img), 1).requires_grad_(True)
    pos = torch.cat((ref_pos, imgpos, torch.zeros(1, 3, 3)), 1)
    args = dict(context=torch.randn(1, 3, CONFIG["txtlayers"], CONFIG["txtdim"]), t=torch.tensor([0.7]), pos=pos,
                mask=torch.ones(1, tokens.shape[1] + 3, dtype=torch.bool), ref_len=ref_tokens.shape[1])
    out = model(img=tokens, **args)
    out.square().mean().backward()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(out, model(img=tokens, **args), atol=1e-6)
    assert tokens.grad is not None and tokens.grad[:, : ref_tokens.shape[1]].abs().sum() > 0  # references feed the target


def _hash(seed):
    return torch.randint(0, 255, (1, 32), generator=torch.Generator().manual_seed(seed), dtype=torch.int64).to(torch.uint8)


def test_training_batch_references_need_matching_fingerprints():
    latent = torch.randn(1, 16, 1, 8, 8)
    batch = {"latents_control_0": latent, "reference_hashes": [_hash(0)], "vlm_reference_hashes": [_hash(0)]}
    refs = reference_latents_from_batch(batch)
    assert len(refs) == 1 and refs[0].shape == (1, 16, 8, 8)
    assert reference_latents_from_batch({"latents": latent}) == []
    with pytest.raises(ValueError, match="text encoder"):
        reference_latents_from_batch({"latents_control_0": latent, "reference_hashes": [_hash(0)]})
    with pytest.raises(ValueError, match="differ"):
        reference_latents_from_batch({"latents_control_0": latent, "reference_hashes": [_hash(0)], "vlm_reference_hashes": [_hash(1)]})
    with pytest.raises(ValueError, match="contiguous"):
        reference_latents_from_batch({"latents_control_1": latent, "reference_hashes": [_hash(0)], "vlm_reference_hashes": [_hash(0)]})


def test_edit_vision_pixels_follow_comfyuis_384_pixel_budget():
    import numpy as np

    image = np.zeros((1024, 768, 3), dtype=np.uint8)
    pixels = edit_vision_pixels(image)
    scale = math.sqrt(384 * 384 / (1024 * 768))
    assert pixels.shape == (round(1024 * scale), round(768 * scale), 3)
    patches, grid = comfy_qwen3vl_patches(pixels)
    height, width = pixels.shape[:2]
    h_bar, w_bar = round(height / 32) * 32, round(width / 32) * 32
    assert grid.tolist() == [[1, h_bar // 16, w_bar // 16]]
    assert patches.shape == (h_bar // 16 * (w_bar // 16), 3 * 2 * 16 * 16)


def test_comfy_patches_match_the_qwen3vl_processor_layout():
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001 - not in the local Hugging Face cache
        pytest.skip("Qwen/Qwen3-VL-4B-Instruct processor not cached")
    torch.manual_seed(0)
    image = torch.rand(384, 448, 3)  # already on the 32-pixel grid: no resize on either side
    ours, grid = comfy_qwen3vl_patches(image)
    inputs = processor.image_processor(images=[image.permute(2, 0, 1)], return_tensors="pt", do_rescale=False)
    assert inputs["image_grid_thw"].tolist() == grid.tolist()
    assert torch.allclose(inputs["pixel_values"].float(), ours, atol=1e-4)
