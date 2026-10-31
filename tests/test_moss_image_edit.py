"""MOSS-Image edit LoRAs (ai-toolkit's layout, ComfyUI's Text Encode MOSS-Image Edit "ai-toolkit" formats), several captions
per caption file and caption dropout: the DiT's reference path, the text encoder with reference images, the caches and
the trainer's DiT call. All on the CPU with tiny random models.

The DiT's edit paths were compared with ComfyUI's own MOSS-Image model on random weights (dont_upload/vram_lab/
comfy_moss_edit_check.py: "index_timestep_zero" and "index_timestep_zero_isolated" within FP32 rounding of the plain
text-to-image forward); these tests pin the properties that comparison relies on.
"""

import argparse
import os
import math
import random
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner.moss_image import moss_utils
from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams

PARAMS = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=48, freq_dim=16, latent_channels=8)


def _tiny() -> MossImage:
    torch.manual_seed(0)
    model = MossImage(MossImageParams(**PARAMS))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    return model.eval()


def _inputs(batch=1, text=7):
    torch.manual_seed(1)
    x = torch.randn(batch, 8, 3, 4)
    t = torch.full((batch,), 420.0)
    ctx = torch.randn(batch, text, 48)
    refs = [[torch.randn(8, 3, 4), torch.randn(8, 2, 5)] for _ in range(batch)]
    return x, t, ctx, refs


# ---------------------------------------------------------------------------------------------------------------- DiT
def test_references_change_the_prediction_and_isolation_changes_it_again():
    model = _tiny()
    x, t, ctx, refs = _inputs()
    with torch.no_grad():
        plain = model(x, t, ctx)
        shared = model(x, t, ctx, ref_latents=refs, isolate_refs=False)
        isolated = model(x, t, ctx, ref_latents=refs, isolate_refs=True)
    assert plain.shape == shared.shape == isolated.shape == x.shape
    assert (plain - shared).abs().max() > 1e-3 and (shared - isolated).abs().max() > 1e-4


def test_isolation_as_two_attention_calls_equals_the_masked_form():
    """Without padding the isolation runs as two unmasked SDPA calls; a padded batch takes the boolean-mask form. An item
    of the padded batch must equal its own forward (both forms compute the same attention)."""
    model = _tiny()
    x, t, ctx, refs = _inputs(batch=2)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    with torch.no_grad():
        batched = model(x, t, ctx, mask, ref_latents=refs, isolate_refs=True)
        single = model(x[1:], t[1:], ctx[1:, :5], None, ref_latents=refs[1:], isolate_refs=True)
        shared_batched = model(x, t, ctx, mask, ref_latents=refs, isolate_refs=False)
        shared_single = model(x[1:], t[1:], ctx[1:, :5], None, ref_latents=refs[1:], isolate_refs=False)
    torch.testing.assert_close(batched[1:], single, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(shared_batched[1:], shared_single, atol=2e-5, rtol=1e-4)


def test_samples_with_different_reference_counts_share_a_padded_batch():
    model = _tiny()
    x, t, ctx, refs = _inputs(batch=2)
    refs[0] = refs[0][:1]  # one reference for the first sample, two for the second
    with torch.no_grad():
        batched = model(x, t, ctx, ref_latents=refs, isolate_refs=True)
        first = model(x[:1], t[:1], ctx[:1], ref_latents=refs[:1], isolate_refs=True)
    torch.testing.assert_close(batched[:1], first, atol=2e-5, rtol=1e-4)


def test_precomputed_reference_keys_give_the_training_prediction():
    """The sampler computes isolated references once (references alone at f = 0, the image shifted to f = 0 too, as
    ComfyUI's preprocess_refs); attention sees position differences only, so it equals the training forward."""
    model = _tiny()
    x, t, ctx, refs = _inputs()
    with torch.no_grad():
        trained = model(x, t, ctx, ref_latents=refs, isolate_refs=True)
        cache = model.precompute_ref_kv(refs)
        cached = model(x, t, ctx, ref_kv_cache=cache)
        cached_skip = model(x, t, ctx, ref_kv_cache=cache, skip_layers=(1,))
    torch.testing.assert_close(cached, trained, atol=2e-5, rtol=1e-4)
    assert len(cache["kv"]) == PARAMS["num_layers"] and (cached_skip - cached).abs().max() > 1e-4


def test_the_reference_frame_follows_the_prompt_length():
    """References sit at the image's frame coordinate f = the prompt's token count (ai-toolkit): one more text token moves
    the image and the references together, so swapping the text for a longer one must not leave the references at the
    old position. Checked through the sampler's cache, whose references sit at f = 0 with the image shifted to match."""
    model = _tiny()
    x, t, _, refs = _inputs()
    for text in (3, 9):
        ctx = torch.randn(1, text, 48)
        with torch.no_grad():
            trained = model(x, t, ctx, ref_latents=refs, isolate_refs=True)
            cached = model(x, t, ctx, ref_kv_cache=model.precompute_ref_kv(refs))
        torch.testing.assert_close(cached, trained, atol=2e-5, rtol=1e-4)


def test_references_train_through_checkpointed_blocks():
    model = _tiny().train()
    x, t, ctx, refs = _inputs()
    expected = model(x, t, ctx, ref_latents=refs, isolate_refs=True).detach()
    model.enable_gradient_checkpointing()
    out = model(x.requires_grad_(True), t, ctx.requires_grad_(True), ref_latents=refs, isolate_refs=True)
    out.square().mean().backward()
    torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-4)
    assert all(p.grad is not None for p in model.layers[0].proj.parameters())


def test_a_widened_control_embedder_takes_the_references_with_zero_maps():
    from musubi_tuner.moss_image import moss_control

    model = _tiny()
    x, t, ctx, refs = _inputs()
    with torch.no_grad():
        expected = model(x, t, ctx, ref_latents=refs, isolate_refs=True)
        moss_control.widen_video_embedder(model)  # zero control columns: no maps is the plain model
        widened = moss_control.ControlledDiT(model, torch.zeros(1, 16, 3, 4))(x, t, ctx, ref_latents=refs, isolate_refs=True)
    torch.testing.assert_close(widened, expected, atol=1e-5, rtol=1e-4)


class _ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def precompute_ref_kv(self, refs):
        self.calls.append("precompute")
        return {"kv": {}, "mask": torch.ones(1, 1, dtype=torch.bool)}

    def forward(self, latents, timestep, ctx, mask, skip_layers=None, **kwargs):
        self.calls.append(tuple(sorted(kwargs)))
        return torch.full_like(latents, float(ctx[0, 0, 0]))


@pytest.mark.parametrize("isolate, expected", [(True, ("isolate_refs", "ref_kv_cache")), (False, ("ref_latents",))])
def test_denoise_runs_the_references_in_both_branches(isolate, expected):
    model = _ConstantVelocity()
    ctx, mask = moss_utils.pad_contexts([torch.ones(3, 4)], "cpu", torch.float32)
    neg, neg_mask = moss_utils.pad_contexts([torch.zeros(5, 4)], "cpu", torch.float32)
    moss_utils.denoise(model, torch.randn(1, 8, 2, 3), ctx, mask, neg, neg_mask, 3, 3.0, ref_latents=[[torch.zeros(8, 2, 3)]], isolate_refs=isolate)
    forwards = [call for call in model.calls if call != "precompute"]
    assert set(forwards) == {expected} and model.calls.count("precompute") == (1 if isolate else 0)


# ----------------------------------------------------------------------------------------------------- text encoder
def _comfyui_area_then_process(image_bhwc: torch.Tensor):
    """ComfyUI c7bd15c2: TextEncodeMOSSImageEdit's 384 x 384 area downscale, then comfy/text_encoders/qwen_vl.py
    process_qwen2vl_images(patch_size=16, image_mean=std=0.5), copied as the reference."""
    scale = min(1.0, math.sqrt(384 * 384 / (image_bhwc.shape[1] * image_bhwc.shape[2])))
    width, height = round(image_bhwc.shape[2] * scale), round(image_bhwc.shape[1] * scale)
    images = F.interpolate(image_bhwc.movedim(-1, 1), size=(height, width), mode="area").movedim(1, -1)
    patch_size, temporal_patch_size, merge_size = 16, 2, 2
    images = images[..., :3].permute(0, 3, 1, 2)
    img = images[0]
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    img_resized = F.interpolate(img.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False).squeeze(0)
    normalized = (img_resized - 0.5) / 0.5
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = normalized.unsqueeze(0).repeat(2, 1, 1, 1).reshape(
        1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size), torch.tensor([[1, grid_h, grid_w]])


@pytest.mark.parametrize("size", [(1024, 768), (300, 200), (96, 160), (2000, 3000)])
def test_reference_pixels_are_comfyuis(size):
    rng = np.random.default_rng(size[0])
    image = rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
    patches, grid = moss_utils.vlm_pixels(image)
    expected_patches, expected_grid = _comfyui_area_then_process(torch.from_numpy(image).float().unsqueeze(0) / 255.0)
    assert torch.equal(grid, expected_grid)
    torch.testing.assert_close(patches, expected_patches, atol=1e-6, rtol=0)
    assert grid[0, 1] * grid[0, 2] * 16 * 16 <= 384 * 384 * 1.3


class _SpecialTokenizer:
    """Characters as ids below 80, Qwen's special tokens as single ids, Qwen3.5's chat template."""

    SPECIAL = {"<|image_pad|>": 90, "<|vision_start|>": 91, "<|vision_end|>": 92, "<|im_start|>": 93, "<|im_end|>": 94, "<think>": 95}
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "<|im_start|>user\n" + messages[0]["content"].strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        import re

        ids = []
        for part in re.split(r"(<\|[a-z_]+\|>|<think>)", text):
            if part in self.SPECIAL:
                ids.append(self.SPECIAL[part])
            else:
                ids.extend(1 + ord(c) % 79 for c in part)
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


def _tiny_vl_encoder():
    from transformers import Qwen3_5Model

    moss_utils.install_qwen35_kernel_dispatch()
    text = dict(moss_utils.QWEN35_9B_TEXT_CONFIG)
    text.update(hidden_size=64, intermediate_size=96, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                head_dim=32, num_attention_heads=2, num_key_value_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4, vocab_size=97, eos_token_id=94)
    text["rope_parameters"] = dict(text["rope_parameters"], mrope_section=[2, 1, 1])
    vision = dict(moss_utils.QWEN35_9B_VISION_CONFIG)
    vision.update(depth=2, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64, num_position_embeddings=16)
    config = moss_utils.qwen35_vl_config(text, vision)
    config.image_token_id, config.video_token_id, config.vision_start_token_id, config.vision_end_token_id = 90, 96, 91, 92
    torch.manual_seed(5)
    model = Qwen3_5Model(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    return model


def _images():
    rng = np.random.default_rng(3)
    return [rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8), rng.integers(0, 256, size=(96, 64, 3), dtype=np.uint8)]


def test_prompts_with_references_match_the_multimodal_forward():
    """encode_prompts_with_images builds the embeddings itself (so the language tower can stream); it must equal
    Qwen3_5Model's own forward on the same tokens, image patches and M-RoPE positions."""
    model, tokenizer, images = _tiny_vl_encoder(), _SpecialTokenizer(), _images()
    got = moss_utils.encode_prompts_with_images(model, tokenizer, ["make it winter", "a sign"], [images, images[:1]])
    for prompt, refs, state in zip(["make it winter", "a sign"], [images, images[:1]], got):
        patches, grids = zip(*(moss_utils.vlm_pixels(image) for image in refs))
        grid = torch.cat(grids)
        counts = (grid.prod(-1) // 4).tolist()
        ids = tokenizer(moss_utils.chat_text(tokenizer, moss_utils.edit_prompt_text(prompt, counts)))["input_ids"]
        with torch.no_grad():
            expected = model(input_ids=ids, pixel_values=torch.cat(patches), image_grid_thw=grid, mm_token_type_ids=(ids == 90).int()).last_hidden_state[0]
        assert state.shape == expected.shape and state.shape[0] == ids.shape[1]
        torch.testing.assert_close(state, expected, atol=1e-5, rtol=1e-4)
    # the references change the context; a prompt without references is the text tower's
    plain = moss_utils.encode_prompts_with_images(model, tokenizer, ["make it winter"], [None])[0]
    torch.testing.assert_close(plain, moss_utils.encode_prompts(model.language_model, tokenizer, ["make it winter"])[0])


def _comfyui_mrope_position_ids(embeds_info, seq_len):
    """ComfyUI c7bd15c2 comfy/text_encoders/qwen_vl.py qwen2vl_mrope_position_ids, copied as the reference."""
    position_ids = None
    offset = 0
    for e in embeds_info:
        if e.get("type") == "image":
            grid = e["extra"]
            start = e.get("index")
            if position_ids is None:
                position_ids = torch.zeros((3, seq_len))
                position_ids[:, :start] = torch.arange(0, start)
            end = e.get("size") + start
            len_max = int(grid.max()) // 2
            start_next = len_max + start
            position_ids[:, end:] = torch.arange(start_next + offset, start_next + (seq_len - end) + offset)
            position_ids[0, start:end] = start + offset
            max_d = int(grid[0][1]) // 2
            position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
            max_d = int(grid[0][2]) // 2
            position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
            offset += len_max - (end - start)
    return position_ids


def test_reference_token_positions_are_comfyuis():
    """The positions encode_prompts_with_images gives the language tower (transformers' get_rope_index) equal ComfyUI's
    for a prompt with two references of different shapes."""
    model, tokenizer, images = _tiny_vl_encoder(), _SpecialTokenizer(), _images()
    grids = [moss_utils.vlm_pixels(image)[1] for image in images]
    counts = [int(grid.prod() // 4) for grid in grids]
    ids = tokenizer(moss_utils.chat_text(tokenizer, moss_utils.edit_prompt_text("make it winter", counts)))["input_ids"]
    ours, _ = model.get_rope_index(ids, (ids == 90).int(), image_grid_thw=torch.cat(grids))
    starts = [int(i) for i in (ids[0] == 91).nonzero().flatten() + 1]  # each image's pads follow its vision_start
    # ComfyUI holds one placeholder per image until the embeddings expand: rebuild its info on the expanded sequence
    info = [{"type": "image", "index": start, "size": count, "extra": grid} for start, count, grid in zip(starts, counts, grids)]
    theirs = _comfyui_mrope_position_ids(info, ids.shape[1])
    assert torch.equal(ours[:, 0].float(), theirs)


def test_the_edit_prompt_puts_numbered_pictures_before_the_instruction():
    text = moss_utils.edit_prompt_text("swap the hat", [2, 3])
    assert text == ("Picture 1: <|vision_start|>" + "<|image_pad|>" * 2 + "<|vision_end|>"
                    "Picture 2: <|vision_start|>" + "<|image_pad|>" * 3 + "<|vision_end|>swap the hat")


def test_streamed_language_tower_with_references_matches_the_resident_one():
    model, tokenizer, images = _tiny_vl_encoder(), _SpecialTokenizer(), _images()
    expected = moss_utils.encode_prompts_with_images(model, tokenizer, ["make it winter"], [images])[0]
    streamed = _tiny_vl_encoder()
    moss_utils.stream_text_encoder(streamed.language_model, "cpu")
    got = moss_utils.encode_prompts_with_images(streamed, tokenizer, ["make it winter"], [images])[0]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_the_vision_tower_loads_from_a_comfyui_file(tmp_path):
    """The ComfyUI Qwen3.5 file keeps the vision tower under model.visual (INT8 ConvRot in the SwarmUI file): vision=True
    keeps it under Qwen3_5Model's names; the text-only load drops it as before."""
    sd = {
        "model.language_model.norm.weight": torch.ones(4),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(6, 4),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(9, 3),
        "model.visual.merger.linear_fc1.bias": torch.randn(3),
        "lm_head.weight": torch.randn(5, 4),
        "mtp.fc.weight": torch.randn(4, 8),
    }
    path = tmp_path / "te.safetensors"
    save_file(sd, str(path))
    vision = moss_utils._text_encoder_state_dict(str(path), torch.float32, "cpu", vision=True)
    assert sorted(vision) == sorted(
        ["language_model.norm.weight", "language_model.layers.0.mlp.up_proj.weight", "visual.blocks.0.attn.qkv.weight", "visual.merger.linear_fc1.bias"]
    )
    text = moss_utils._text_encoder_state_dict(str(path), torch.float32, "cpu")
    assert sorted(text) == ["layers.0.mlp.up_proj.weight", "norm.weight"]
    save_file({k: v for k, v in sd.items() if "visual" not in k}, str(tmp_path / "text_only.safetensors"))
    with pytest.raises(ValueError, match="vision tower"):
        moss_utils._text_encoder_state_dict(str(tmp_path / "text_only.safetensors"), torch.float32, "cpu", vision=True)


# ------------------------------------------------------------------------------------------------ captions and caches
@pytest.mark.parametrize(
    "caption, expected",
    [
        ("a photo of ohwx man", ["a photo of ohwx man"]),
        ("first\n---===---\nsecond line one\nsecond line two\n", ["first", "second line one\nsecond line two"]),
        ("first\n---===--- # a comment\n\n---===---\nthird  \n", ["first", "third"]),
        ("---===---\n---===---", [""]),
    ],
)
def test_caption_files_split_into_candidates(caption, expected):
    assert moss_utils.split_caption_candidates(caption) == expected


def test_a_separator_line_with_other_text_is_refused():
    with pytest.raises(ValueError, match="separator"):
        moss_utils.split_caption_candidates("a\n---===--- weight=2\nb")


def test_a_cache_with_several_captions_yields_one_of_them_per_load():
    from musubi_tuner.dataset.cache_io import pick_caption_candidate

    sd = {
        "varlen_ctx_vec_bfloat16": torch.zeros(3, 2),
        "varlen_ctx_vec_cand1_bfloat16": torch.ones(4, 2),
        "varlen_ctx_vec_cand2_bfloat16": torch.full((5, 2), 2.0),
        "varlen_ctx_vec_uncond_bfloat16": torch.full((2, 2), 7.0),
    }
    seen = set()
    rng = random.Random(0)
    for _ in range(60):
        picked = pick_caption_candidate(sd, rng)
        assert sorted(picked) == ["varlen_ctx_vec_bfloat16", "varlen_ctx_vec_uncond_bfloat16"]
        seen.add(int(picked["varlen_ctx_vec_bfloat16"][0, 0]))
    assert seen == {0, 1, 2}
    plain = {"varlen_ctx_vec_bfloat16": torch.zeros(3, 2)}
    assert pick_caption_candidate(plain) is plain


def _item(tmp_path, caption, references=None):
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    item = ItemInfo("img", caption, (64, 64), (64, 64))
    item.text_encoder_output_cache_path = str(tmp_path / "img_mi_te.safetensors")
    item.control_content = references
    return item


def test_text_caches_hold_every_caption_and_know_when_they_are_stale(tmp_path):
    from musubi_tuner import moss_image_cache_text_encoder_outputs as te_cache
    from musubi_tuner.dataset.cache_io import pick_caption_candidate
    from safetensors.torch import load_file

    encoder, tokenizer = _tiny_vl_encoder(), _SpecialTokenizer()
    caption = "a red car\n---===---\na car painted red"
    item = _item(tmp_path, caption)
    assert not te_cache.cache_is_current(item, moss_utils.CAPTION_SEPARATOR, edit=False)
    te_cache.encode_and_save_batch(encoder.language_model, tokenizer, [item], separator=moss_utils.CAPTION_SEPARATOR)
    sd = load_file(item.text_encoder_output_cache_path)
    assert sorted(sd) == ["varlen_ctx_vec_bfloat16", "varlen_ctx_vec_cand1_bfloat16"]
    expected = moss_utils.encode_prompts(encoder.language_model, tokenizer, ["a red car", "a car painted red"])
    torch.testing.assert_close(sd["varlen_ctx_vec_cand1_bfloat16"].float(), expected[1].to(torch.bfloat16).float())
    assert te_cache.cache_is_current(item, moss_utils.CAPTION_SEPARATOR, edit=False)
    assert not te_cache.cache_is_current(_item(tmp_path, "a red car"), moss_utils.CAPTION_SEPARATOR, edit=False)  # caption changed
    assert not te_cache.cache_is_current(item, None, edit=False)  # one caption without the separator
    assert not te_cache.cache_is_current(item, moss_utils.CAPTION_SEPARATOR, edit=True)  # an edit run needs references
    assert {tuple(pick_caption_candidate(sd, random.Random(seed))["varlen_ctx_vec_bfloat16"].shape) for seed in range(20)} == {
        tuple(expected[0].shape), tuple(expected[1].shape)
    }


def test_edit_text_caches_carry_the_references_and_the_dropped_caption(tmp_path):
    from musubi_tuner import moss_image_cache_text_encoder_outputs as te_cache
    from safetensors.torch import load_file

    encoder, tokenizer, images = _tiny_vl_encoder(), _SpecialTokenizer(), _images()
    item = _item(tmp_path, "make it winter", images)
    te_cache.encode_and_save_batch(encoder, tokenizer, [item], separator=moss_utils.CAPTION_SEPARATOR, edit=True)
    sd = load_file(item.text_encoder_output_cache_path)
    caption, uncond = moss_utils.encode_prompts_with_images(encoder, tokenizer, ["make it winter", ""], [images, images])
    torch.testing.assert_close(sd["varlen_ctx_vec_bfloat16"].float(), caption.to(torch.bfloat16).float())
    torch.testing.assert_close(sd["varlen_ctx_vec_uncond_bfloat16"].float(), uncond.to(torch.bfloat16).float())
    assert te_cache.cache_is_current(item, moss_utils.CAPTION_SEPARATOR, edit=True)
    assert not te_cache.cache_is_current(item, moss_utils.CAPTION_SEPARATOR, edit=False)  # a text-to-image run re-encodes
    changed = _item(tmp_path, "make it winter", [images[0][::-1].copy(), images[1]])
    assert not te_cache.cache_is_current(changed, moss_utils.CAPTION_SEPARATOR, edit=True)


def test_an_edit_dataset_takes_any_number_of_references(tmp_path):
    """Control adapter maps are exactly two per image at the target's bucket; edit references keep their framing
    (control_resolution, written for MOSS-Image Edit) and come one or more per image."""
    from PIL import Image

    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.dataset.image_video_dataset import ImageDataset

    images, controls = tmp_path / "images", tmp_path / "control"
    images.mkdir()
    controls.mkdir()
    Image.new("RGB", (32, 32)).save(images / "a.png")
    (images / "a.txt").write_text("make it winter", encoding="utf-8")
    Image.new("RGB", (48, 32), (9, 9, 9)).save(controls / "a_0.png")
    edit = ImageDataset(
        resolution=(32, 32), caption_extension=".txt", batch_size=1, num_repeats=1, enable_bucket=True, bucket_no_upscale=False,
        image_directory=str(images), control_directory=str(controls), cache_directory=str(tmp_path / "cache"),
        architecture=ARCHITECTURE_MOSS_IMAGE, control_resolution=(32, 32),
    )  # fmt: skip
    assert edit.has_control and edit.datasource.control_count_per_image is None
    assert [os.path.basename(p) for p in edit.datasource.control_paths[str(images / "a.png")]] == ["a_0.png"]


# ------------------------------------------------------------------------------------------------------------ trainer
def _trainer(**options):
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    trainer = MossImageNetworkTrainer()
    defaults = dict(mixed_precision="bf16", sample_prompts=None, text_encoder=None, fp8_base=False, fp8_scaled=False, network_args=None)
    trainer.handle_model_specific_args(argparse.Namespace(**{**defaults, **options}))
    return trainer


def test_the_edit_trainer_feeds_the_cached_references_to_the_dit():
    trainer = _trainer(moss_edit=True)
    model = _tiny()
    accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m)
    torch.manual_seed(2)
    latents = torch.randn(1, 8, 3, 4)
    noise = torch.randn_like(latents)
    noisy = 0.5 * latents + 0.5 * noise
    batch = {"ctx_vec": [torch.randn(7, 48)], "latents_control_0": torch.randn(1, 8, 3, 4), "latents_control_1": torch.randn(1, 8, 2, 5)}
    args = argparse.Namespace(gradient_checkpointing=False)
    with torch.no_grad():
        out = trainer.call_dit(args, accelerator, model, latents, batch, noise, noisy, torch.tensor([500.0]), torch.float32)
        ctx, mask = moss_utils.pad_contexts(batch["ctx_vec"], "cpu", torch.float32)
        refs = [[batch["latents_control_0"][0], batch["latents_control_1"][0]]]
        expected = model(noisy, torch.tensor([500.0]), ctx, mask, ref_latents=refs, isolate_refs=True)
        unisolated = _trainer(moss_edit=True, moss_edit_kv_cache=False).call_dit(
            args, accelerator, model, latents, batch, noise, noisy, torch.tensor([500.0]), torch.float32
        )
    assert torch.allclose(out.pred, expected) and not torch.allclose(unisolated.pred, expected)
    assert torch.equal(out.target, noise - latents)


def test_an_edit_batch_needs_its_references():
    from musubi_tuner.moss_image_train_network import batch_reference_latents

    assert batch_reference_latents({"ctx_vec": []}, 1) is None
    with pytest.raises(ValueError, match="same number of references"):
        batch_reference_latents({"latents_control_0": torch.zeros(1, 8, 2, 2)}, 2)
    trainer = _trainer(moss_edit=True)
    with pytest.raises(ValueError, match="reference latents"):
        trainer.prepare_dit_inputs(argparse.Namespace(), "cpu", torch.zeros(1, 8, 2, 2), {"ctx_vec": [torch.zeros(3, 48)]},
                                   torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2), torch.tensor([1.0]), torch.float32)


@pytest.mark.parametrize("edit", [False, True])
def test_caption_dropout_trains_with_the_empty_or_the_references_only_context(edit):
    trainer = _trainer(moss_edit=edit, caption_dropout_rate=0.5)
    trainer._empty_ctx = torch.full((2, 48), -1.0)
    contexts = [torch.full((3, 48), float(index)) for index in range(64)]
    batch = {"ctx_vec": contexts, "ctx_vec_uncond": [torch.full((4, 48), 100.0 + index) for index in range(64)]}
    if edit:
        batch["latents_control_0"] = torch.zeros(64, 8, 2, 2)
    torch.manual_seed(0)
    inputs = trainer.prepare_dit_inputs(argparse.Namespace(), "cpu", torch.zeros(64, 8, 2, 2), batch, torch.zeros(64, 8, 2, 2),
                                        torch.zeros(64, 8, 2, 2), torch.ones(64), torch.float32)
    dropped = [index for index, ctx in enumerate(inputs["contexts"]) if not torch.equal(ctx, contexts[index])]
    assert 16 < len(dropped) < 48
    for index in dropped:
        expected = batch["ctx_vec_uncond"][index] if edit else trainer._empty_ctx
        assert torch.equal(inputs["contexts"][index], expected)


def test_caption_dropout_replaces_the_control_adapters_own_rate():
    assert _trainer(caption_dropout_rate=0.25)._caption_dropout == 0.25
    with pytest.raises(ValueError, match="caption_dropout_rate"):
        _trainer(caption_dropout_rate=1.0)
    with pytest.raises(ValueError, match="choose one"):
        _trainer(moss_edit=True, control_adapter=True)


def test_sample_noise_is_comfyuis_cpu_noise_for_the_seed():
    """ComfyUI's prepare_noise draws float32 normal noise of the latent's shape from torch.manual_seed(seed) on the CPU."""
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    generator = torch.Generator().manual_seed(1234)
    noise = MossImageNetworkTrainer.sample_noise(generator, 64, 96, "cpu")
    expected = torch.randn((1, 128, 4, 6), generator=torch.manual_seed(1234), dtype=torch.float32, device="cpu")
    assert torch.equal(noise, expected)


def test_text_embedder_projection_lora_targets():
    from musubi_tuner.networks import lora_moss_image

    model = _tiny()
    blocks = {f"lora_unet_layers_{i}_proj_{p}" for i in range(PARAMS["num_layers"]) for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")}
    refiner = {f"lora_unet_text_embedder_blocks_{b}_{n}" for b in range(2) for n in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")}
    for options, extra in (
        ({}, set()),
        ({"include_text_proj": "True"}, {"lora_unet_text_embedder_proj"}),
        ({"include_text_embedder": "True"}, refiner),
        ({"include_text_proj": "True", "include_text_embedder": "True"}, refiner | {"lora_unet_text_embedder_proj"}),
        ({"train_text_embedder": "True"}, refiner | {"lora_unet_text_embedder_proj"}),
    ):
        network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model, **options)
        names = [m.lora_name for m in network.unet_loras]
        assert len(names) == len(set(names)) and set(names) == blocks | extra
        sd = {}
        for module in network.unet_loras:
            sd[f"{module.lora_name}.lora_down.weight"] = module.lora_down.weight.detach()
            sd[f"{module.lora_name}.lora_up.weight"] = module.lora_up.weight.detach()
            sd[f"{module.lora_name}.alpha"] = torch.tensor(4.0)
        loaded = lora_moss_image.create_arch_network_from_weights(1.0, sd, unet=model)
        assert {m.lora_name for m in loaded.unet_loras} == blocks | extra


def test_the_trainer_adds_the_projection_to_the_network_args():
    args = argparse.Namespace(mixed_precision="bf16", sample_prompts=None, text_encoder=None, fp8_base=False, fp8_scaled=False,
                              network_args=["use_dora=True"], train_text_embedder_proj=True)
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    MossImageNetworkTrainer().handle_model_specific_args(args)
    assert args.network_args == ["use_dora=True", "include_text_proj=True"]
