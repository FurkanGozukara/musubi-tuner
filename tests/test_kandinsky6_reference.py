import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner import kandinsky6_train as ft
from musubi_tuner import kandinsky6_prepare_omni_r2v as omni
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import save_latent_cache_kandinsky6
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.kandinsky6 import reference as ref
from musubi_tuner.kandinsky6.dit import DiffusionTransformer3D
from musubi_tuner.kandinsky6.model import (
    REF_SLOT_KEY,
    TOKEN_TYPE_KEY,
    _reference_config,
    _stream_assign_safetensors,
    inspect_reference_layout,
)
from musubi_tuner.kandinsky6_train_network import Kandinsky6NetworkTrainer
from musubi_tuner.training.parser_common import read_config_from_file


def tiny_dit(*, reference_slots: int = 0, num_visual_blocks: int = 1) -> DiffusionTransformer3D:
    return DiffusionTransformer3D(
        in_visual_dim=16,
        out_visual_dim=16,
        in_text_dim=8,
        in_text_dim2=6,
        time_dim=8,
        patch_size=(1, 1, 1),
        model_dim=12,
        ff_dim=20,
        num_text_blocks=1,
        num_visual_blocks=num_visual_blocks,
        axes_dims=(4, 4, 4),
        visual_cond=True,
        is_multimodal=True,
        in_audio_dim=40,
        out_audio_dim=40,
        model_dim_a=12,
        time_dim_a=8,
        ff_dim_a=20,
        axes_dims_a=(4, 4, 4),
        attention_engine="sdpa",
        text_token_padding=True,
        ca_rope=True,
        cross_gates=True,
        fix_modulation=True,
        visual_token_type_num_embeddings=3 if reference_slots else 2,
        reference_slots=reference_slots,
    )


def randomize(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    # Fresh K6 blocks are zero-gated; randomize so every path carries signal and gradient.
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model


def forward(model, references, *, slot_ids_override=None, seed=1):
    gen = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 2, 2, 2, 16, generator=gen)
    audio = torch.randn(1, 3, 40, generator=gen)
    model_video, token_types, slot_ids = ref.build_reference_video_input(video, references, model.visual_cond)
    if slot_ids_override is not None:
        slot_ids = slot_ids_override
    n = 0 if references is None else references.shape[1]
    rope = ref.reference_visual_rope(model, 2, 2, 2, n, (1.0, 2.0, 2.0), device=torch.device("cpu"))
    text_pos = torch.arange(2)
    video_out, audio_out = model(
        x_video=model_video,
        x_audio=audio,
        text_embed=torch.randn(1, 2, 8, generator=gen),
        pooled_text_embed=torch.randn(1, 6, generator=gen),
        time=torch.tensor([500.0]),
        visual_rope=rope,
        audio_rope=model.audio_rope(torch.arange(3)),
        text_rope=[model.video_text_rope(text_pos), model.audio_text_rope(text_pos)],
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        visual_token_type_ids=token_types,
        visual_ref_slot_ids=slot_ids,
    )
    return video_out[:, :2], audio_out


# ----------------------------------------------------------------------------------------
# DiT and checkpoint extension
# ----------------------------------------------------------------------------------------


def test_reference_slots_require_three_token_types():
    with pytest.raises(ValueError, match=">= 3"):
        tiny_dit_bad()


def tiny_dit_bad():
    # reference slots with only the two released token types -> invalid
    return DiffusionTransformer3D(
        in_visual_dim=16,
        out_visual_dim=16,
        in_text_dim=8,
        in_text_dim2=6,
        time_dim=8,
        patch_size=(1, 1, 1),
        model_dim=12,
        ff_dim=20,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(4, 4, 4),
        visual_cond=True,
        is_multimodal=True,
        in_audio_dim=40,
        out_audio_dim=40,
        model_dim_a=12,
        time_dim_a=8,
        ff_dim_a=20,
        axes_dims_a=(4, 4, 4),
        attention_engine="sdpa",
        visual_token_type_num_embeddings=2,
        reference_slots=2,
    )


def test_slot_embeddings_bind_references_to_slots():
    model = randomize(tiny_dit(reference_slots=4)).eval()
    references = torch.randn(1, 2, 2, 2, 16)
    with torch.no_grad():
        model.visual_ref_slot_embeddings.weight.zero_()
        base, _ = forward(model, references)
        # zero slot table: swapping slot ids changes nothing
        swapped, _ = forward(model, references, slot_ids_override=torch.tensor([[-1, -1, 1, 0]]))
        assert torch.allclose(base, swapped, atol=1e-6)
        model.visual_ref_slot_embeddings.weight.normal_(0.0, 0.5)
        bound, _ = forward(model, references)
        swapped, _ = forward(model, references, slot_ids_override=torch.tensor([[-1, -1, 1, 0]]))
    assert not torch.allclose(bound, base)
    assert not torch.allclose(bound, swapped)


def test_slot_id_out_of_range_is_rejected():
    model = randomize(tiny_dit(reference_slots=1)).eval()
    with pytest.raises(ValueError, match="exceeds"):
        forward(model, torch.randn(1, 2, 2, 2, 16))


def test_released_checkpoint_extends_with_copied_type_row_and_zero_slots(tmp_path):
    released = randomize(tiny_dit())
    path = tmp_path / "released.safetensors"
    save_file({key: value.contiguous() for key, value in released.state_dict().items()}, str(path))
    assert inspect_reference_layout(path) == (2, 0)
    overrides, extend = _reference_config(path, 4)
    assert extend and overrides == {"visual_token_type_num_embeddings": 3, "reference_slots": 4}
    assert _reference_config(path, None) == ({}, False)

    extended = tiny_dit(reference_slots=4)
    _stream_assign_safetensors(extended, path, dtype=torch.float32, device="cpu", strict=True, extend_reference=True)
    rows = extended.visual_token_type_embeddings.weight
    assert torch.equal(rows[:2], released.visual_token_type_embeddings.weight)
    assert torch.equal(rows[2], rows[1])
    assert torch.count_nonzero(extended.visual_ref_slot_embeddings.weight) == 0
    with pytest.raises(RuntimeError, match="missing"):
        _stream_assign_safetensors(tiny_dit(reference_slots=4), path, dtype=None, device="cpu", strict=True)

    trained = tmp_path / "trained.safetensors"
    save_file({key: value.contiguous() for key, value in extended.state_dict().items()}, str(trained))
    assert inspect_reference_layout(trained) == (3, 4)
    assert _reference_config(trained, None) == ({"visual_token_type_num_embeddings": 3, "reference_slots": 4}, False)
    restored = tiny_dit(reference_slots=4)
    _stream_assign_safetensors(restored, trained, dtype=torch.float32, device="cpu", strict=True)
    for key, value in extended.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key
    with pytest.raises(ValueError, match="4 reference slots"):
        _reference_config(trained, 8)
    assert TOKEN_TYPE_KEY in extended.state_dict() and REF_SLOT_KEY in extended.state_dict()


# ----------------------------------------------------------------------------------------
# Input packing
# ----------------------------------------------------------------------------------------


def test_build_reference_video_input_layout():
    video = torch.randn(2, 3, 2, 2, 16)
    references = torch.randn(2, 2, 2, 2, 16)
    packed, token_types, slot_ids = ref.build_reference_video_input(video, references, visual_cond=True)
    assert packed.shape == (2, 5, 2, 2, 33)
    assert torch.equal(packed[:, 3:, ..., :16], references)
    assert torch.count_nonzero(packed[..., 16:32]) == 0
    assert torch.equal(packed[:, :3, ..., 32], torch.zeros(2, 3, 2, 2))
    assert torch.equal(packed[:, 3:, ..., 32], torch.ones(2, 2, 2, 2))
    assert token_types.tolist() == [[0, 0, 0, 2, 2]] * 2
    assert slot_ids.tolist() == [[-1, -1, -1, 0, 1]] * 2

    bare, bare_types, bare_slots = ref.build_reference_video_input(video, None, visual_cond=True)
    assert bare.shape == (2, 3, 2, 2, 33) and bare_slots is None
    assert torch.count_nonzero(bare_types) == 0


def test_reference_rope_places_references_after_the_clip():
    positions = ref.reference_temporal_positions(5, 3)
    assert positions.tolist() == [0, 1, 2, 3, 4, 5 + ref.REF_ROPE_GAP, 6 + ref.REF_ROPE_GAP, 7 + ref.REF_ROPE_GAP]
    model = tiny_dit(reference_slots=2)
    rope = ref.reference_visual_rope(model, 3, 2, 2, 2, (1.0, 2.0, 2.0), device=torch.device("cpu"))
    plain = model.visual_rope((3, 2, 2), [torch.arange(3), torch.arange(2), torch.arange(2)], (1.0, 2.0, 2.0))
    assert rope.shape[0] == 5
    assert torch.equal(rope[:3], plain)
    with pytest.raises(ValueError, match="RoPE time index"):
        ref.reference_visual_rope(model, 200, 2, 2, 2, (1.0, 2.0, 2.0), device=torch.device("cpu"))


def test_letterbox_keeps_aspect_and_composites_alpha():
    image = np.zeros((10, 40, 4), dtype=np.uint8)  # transparent, wide
    image[..., 0] = 200
    image[:, :20, 3] = 255  # left half opaque red
    boxed = ref.letterbox_reference(image, 32, 32, pad_value=255)
    assert boxed.shape == (32, 32, 3)
    assert (boxed[0] == 255).all()  # top bar is padding
    middle = boxed[16]
    assert middle[2, 0] == 200 and middle[2, 1] == 0  # opaque red survives
    assert (middle[-2] == 255).all()  # transparent part composited onto the pad colour


def test_cached_references_to_batch_rejects_mixed_counts():
    rows = [torch.randn(2, 16, 3, 4), torch.randn(2, 16, 3, 4)]
    batch = ref.cached_references_to_batch(rows, torch.device("cpu"), torch.float32)
    assert batch.shape == (2, 2, 3, 4, 16)
    with pytest.raises(ValueError, match="batch_size = 1"):
        ref.cached_references_to_batch([torch.randn(1, 16, 3, 4), torch.randn(2, 16, 3, 4)], "cpu", torch.float32)


def test_figure_tags_rewrite():
    prompt = "<Subject 1> is the boy in <Figure 1> and < figure 2 >, holding <Figure 3>."
    assert ref.mentioned_figures(prompt) == {1, 2, 3}
    assert not ref.mentions_video(prompt) and ref.mentions_video("as in <Video 1>")
    out = ref.rewrite_figure_tags(prompt, {1: 3, 2: 1, 3: 2})
    assert out == "<Subject 1> is the boy in @Image 3 and @Image 1, holding @Image 2."
    with pytest.raises(KeyError):
        ref.rewrite_figure_tags(prompt, {1: 1})


# ----------------------------------------------------------------------------------------
# Cache and collation
# ----------------------------------------------------------------------------------------


def test_reference_latents_round_trip_through_the_bucket_collator(tmp_path):
    items = []
    for index, count in enumerate((2, 2)):
        item = ItemInfo(f"k{index}", "c", (32, 24), (32, 24), frame_count=5)
        item.latent_cache_path = str(tmp_path / f"k{index}_latent.safetensors")
        item.text_encoder_output_cache_path = str(tmp_path / f"k{index}_te.safetensors")
        save_latent_cache_kandinsky6(
            item,
            video_latent=torch.randn(16, 2, 3, 4),
            audio_latent=torch.randn(40, 6),
            audio_present=True,
            reference_latents=torch.full((count, 16, 3, 4), float(index)),
        )
        save_file(
            {
                "varlen_text_embeds_float32": torch.randn(2, 8),
                "pooled_embed_float32": torch.randn(6),
                "varlen_attention_mask_bool": torch.ones(2, dtype=torch.bool),
            },
            item.text_encoder_output_cache_path,
        )
        items.append(item)
    batch = BucketBatchManager({(32, 24, 5): items}, batch_size=2)[0]
    assert isinstance(batch["latents_refs"], list) and len(batch["latents_refs"]) == 2
    assert batch["latents_refs"][1].shape == (2, 16, 3, 4) and float(batch["latents_refs"][1].mean()) == 1.0
    assert batch["latents"].shape == (2, 16, 2, 3, 4)

    with pytest.raises(ValueError, match="reference latents must be"):
        save_latent_cache_kandinsky6(
            items[0],
            video_latent=torch.randn(16, 2, 3, 4),
            audio_latent=torch.randn(40, 6),
            audio_present=True,
            reference_latents=torch.randn(1, 16, 4, 4),
        )


def test_reference_paths_resolve_from_jsonl_extras(tmp_path):
    from musubi_tuner.dataset.datasources import ItemExtras
    from musubi_tuner.kandinsky6_cache_latents import reference_paths_for_item

    (tmp_path / "a.png").write_bytes(b"x")
    fields = {"references": [{"type": "image", "path": "a.png"}, str(tmp_path / "a.png")]}
    datasource = SimpleNamespace(get_item_extras=lambda idx: ItemExtras(fields, str(tmp_path), "rec"))
    item = ItemInfo("k", "c", (8, 8))
    item.datasource_index = 0
    paths = reference_paths_for_item(SimpleNamespace(datasource=datasource), item)
    assert paths == [str(tmp_path / "a.png")] * 2
    fields["references"] = [{"type": "video", "path": "a.png"}]
    with pytest.raises(ValueError, match="image references only"):
        reference_paths_for_item(SimpleNamespace(datasource=datasource), item)


# ----------------------------------------------------------------------------------------
# Omni-R2V converter
# ----------------------------------------------------------------------------------------


def write_omni_root(root):
    (root / "meta" / "content").mkdir(parents=True)
    assets = root / "assets"
    assets.mkdir()
    for name in ("t0.mp4", "a.jpg", "b.jpg", "c.jpg", "t1.mp4", "t2.mp4"):
        (assets / name).write_bytes(b"x")
    records = [
        {  # two figures, one unmentioned image
            "sample_id": "s0",
            "prompt_en": "<Subject 1> is <Figure 1>; <Subject 2> is <Figure 3>.",
            "inputs": [
                {"role": "subject", "assets": [{"modality": "image", "rel_path": "assets/a.jpg"}]},
                {
                    "role": "subject",
                    "assets": [
                        {"modality": "image", "rel_path": "assets/b.jpg"},
                        {"modality": "image", "rel_path": "assets/c.jpg"},
                    ],
                },
            ],
            "target": "assets/t0.mp4",
        },
        {  # video reference
            "sample_id": "s1",
            "prompt_en": "<Figure 1> moves like <Video 1>",
            "inputs": [{"role": "subject", "assets": [{"modality": "image", "rel_path": "assets/a.jpg"}]}],
            "target": "assets/t1.mp4",
        },
        {  # missing media (partial extraction)
            "sample_id": "s2",
            "prompt_en": "<Figure 1> walks",
            "inputs": [{"role": "subject", "assets": [{"modality": "image", "rel_path": "assets/missing.jpg"}]}],
            "target": "assets/t2.mp4",
        },
        {  # tag beyond the inputs
            "sample_id": "s3",
            "prompt_en": "<Figure 2> walks",
            "inputs": [{"role": "subject", "assets": [{"modality": "image", "rel_path": "assets/a.jpg"}]}],
            "target": "assets/t2.mp4",
        },
    ]
    with (root / "meta" / "content" / "content.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_omni_r2v_converter(tmp_path):
    root = tmp_path / "omni"
    write_omni_root(root)
    output = tmp_path / "out.jsonl"
    omni.main(["--root", str(root), "--output", str(output), "--subtasks", "content", "--seed", "3"])
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert line["sample_id"] == "s0" and line["video_path"].endswith("t0.mp4")
    paths = [entry["path"] for entry in line["references"]]
    assert sorted(p.rsplit("assets", 1)[1][1:] for p in paths) == ["a.jpg", "c.jpg"]  # b.jpg unmentioned
    slot_of = {p.rsplit("assets", 1)[1][1:]: index + 1 for index, p in enumerate(paths)}
    assert line["caption"] == f"<Subject 1> is @Image {slot_of['a.jpg']}; <Subject 2> is @Image {slot_of['c.jpg']}."


def test_omni_r2v_convert_record_reasons(tmp_path):
    root = tmp_path / "omni"
    write_omni_root(root)
    records = [json.loads(line) for line in (root / "meta" / "content" / "content.jsonl").read_text().splitlines()]
    kwargs = dict(
        max_refs=8, tag_format="@Image {n}", prompt_key="prompt_en", drop_unmentioned=True, shuffle=True, check_files=True
    )
    reasons = [omni.convert_record(record, root, random.Random(0), **kwargs)[1] for record in records]
    assert reasons == [None, "video_reference", "missing_file", "missing_figure"]
    assert omni.convert_record(records[0], root, random.Random(0), **{**kwargs, "max_refs": 1})[1] == "too_many_refs"


# ----------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------


def make_batch(references=None):
    batch = {
        "latents_audio": torch.randn(1, 40, 6),
        "audio_present": torch.ones(1),
        "text_embeds": [torch.randn(2, 8)],
        "attention_mask": [torch.ones(2, dtype=torch.bool)],
        "pooled_embed": torch.randn(1, 6),
        "timesteps": torch.tensor([0.5]),
    }
    if references is not None:
        batch["latents_refs"] = references
    return batch


def run_process_batch(trainer, model, batch, network=None):
    accelerator = SimpleNamespace(sync_gradients=True, num_processes=1, device=torch.device("cpu"), autocast=nullcontext)
    run_args = SimpleNamespace(video_only=False, audio_loss_weight=1.0, independent_time=False, gradient_checkpointing=False)
    loss, _ = trainer.process_batch(
        run_args,
        accelerator,
        model,
        network,
        batch,
        torch.randn(1, 16, 2, 2, 2),
        torch.randn(1, 16, 2, 2, 2),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return loss


def r2av_trainer(cls=Kandinsky6NetworkTrainer, slots=4):
    trainer = cls()
    trainer._task = "r2av"
    trainer._scheduler_scale = 1.0
    trainer._reference_slots = slots
    trainer._ref_drop_prob = 0.0
    return trainer


def test_r2av_process_batch_trains_slot_embeddings():
    model = randomize(tiny_dit(reference_slots=4)).train()
    trainer = r2av_trainer()
    loss = run_process_batch(trainer, model, make_batch([torch.randn(3, 16, 2, 2)]))
    loss.backward()
    slot_grad = model.visual_ref_slot_embeddings.weight.grad
    assert slot_grad is not None
    assert (slot_grad[:3].abs().sum(dim=1) > 0).all()
    assert torch.count_nonzero(slot_grad[3]) == 0
    assert model.visual_token_type_embeddings.weight.grad[0].abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.visual_embeddings.parameters())
    assert torch.count_nonzero(model.visual_token_type_embeddings.weight.grad[2]) > 0


def test_r2av_builds_visual_rope_once(monkeypatch):
    model = randomize(tiny_dit(reference_slots=4)).train()
    trainer = r2av_trainer()
    calls = []
    original = model.visual_rope.forward

    def record_visual_rope(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model.visual_rope, "forward", record_visual_rope)
    run_process_batch(trainer, model, make_batch([torch.randn(2, 16, 2, 2)])).backward()
    assert len(calls) == 1


def test_r2av_reference_dropout_and_task_guards(monkeypatch):
    model = randomize(tiny_dit(reference_slots=4)).train()
    trainer = r2av_trainer()
    trainer._ref_drop_prob = 0.999999
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    run_process_batch(trainer, model, make_batch([torch.randn(2, 16, 2, 2)])).backward()
    assert model.visual_ref_slot_embeddings.weight.grad is None or not model.visual_ref_slot_embeddings.weight.grad.any()

    # Validation must retain references even when the training dropout draw would drop them.
    inputs = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: inputs.append(kwargs["x_video"].shape[1]), with_kwargs=True
    )
    try:
        with torch.no_grad():
            run_process_batch(trainer, model, make_batch([torch.randn(2, 16, 2, 2)]))
    finally:
        handle.remove()
    assert inputs == [4]

    with pytest.raises(ValueError, match="slots"):
        run_process_batch(r2av_trainer(slots=1), model, make_batch([torch.randn(2, 16, 2, 2)]))
    t2av = r2av_trainer()
    t2av._task = "t2av"
    with pytest.raises(ValueError, match="--task r2av"):
        run_process_batch(t2av, model, make_batch([torch.randn(2, 16, 2, 2)]))


def parse_train(*extra):
    required = ["--dit", "dit.safetensors", "--dataset_config", "d.toml", "--output_dir", "out", "--output_name", "o"]
    parser = ft.create_parser()
    args = read_config_from_file(parser.parse_args([*required, *extra]), parser)
    args.dit_dtype = None
    return args


def test_lora_cannot_extend_a_released_checkpoint(tmp_path):
    path = tmp_path / "released.safetensors"
    save_file({key: value.contiguous() for key, value in tiny_dit().state_dict().items()}, str(path))
    args = parse_train("--task", "r2av")
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="full fine-tuning"):
        Kandinsky6NetworkTrainer().load_transformer(accelerator, args, str(path), "sdpa", False, "cpu", None)
    args.fp8_base = True
    with pytest.raises(ValueError, match="BF16 base"):
        ft.Kandinsky6Trainer().load_transformer(accelerator, args, str(path), "sdpa", False, "cpu", None)
    assert ft.Kandinsky6Trainer.trains_full_transformer and not Kandinsky6NetworkTrainer.trains_full_transformer


def test_full_finetune_fused_step_updates_reference_tables():
    trainer = r2av_trainer(ft.Kandinsky6Trainer)
    model = randomize(tiny_dit(reference_slots=4))
    model.reference_extension_is_fresh = True
    args = parse_train("--task", "r2av")
    args.gradient_checkpointing = False
    network = trainer._build_network(args, None, model, None, None)
    params, _ = network.prepare_optimizer_params(unet_lr=1e-2)
    _, _, optimizer, _, _ = trainer.get_optimizer(args, params)
    accelerator = SimpleNamespace(sync_gradients=True, num_processes=1, device=torch.device("cpu"), autocast=nullcontext)
    trainer._install_fused_optimizer(args, accelerator, optimizer)
    slots_before = model.visual_ref_slot_embeddings.weight.detach().clone()
    run_process_batch(trainer, model, make_batch([torch.randn(2, 16, 2, 2)]), network).backward()
    changed = (model.visual_ref_slot_embeddings.weight != slots_before).any(dim=1)
    assert changed[:2].all() and not changed[2:].any()

    frozen = tiny_dit(reference_slots=4)
    frozen.reference_extension_is_fresh = True
    args.k6_freeze_params = ["visual_ref_slot_embeddings"]
    with pytest.raises(ValueError, match="untrained R2AV"):
        trainer._build_network(args, None, frozen, None, None)


def test_sample_reference_latents_runs_end_to_end():
    model = randomize(tiny_dit(reference_slots=2)).eval()
    text = {"text_embeds": torch.randn(1, 2, 8), "pooled_embed": torch.randn(1, 6)}
    null = {"text_embeds": torch.randn(1, 3, 8), "pooled_embed": torch.randn(1, 6)}
    common = dict(
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        null_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        video_frames=2,
        latent_height=2,
        latent_width=2,
        seed=0,
        device=torch.device("cpu"),
        num_steps=2,
        guidance_scale=3.0,
        scheduler_scale=1.0,
        visual_rope_scale=(1.0, 2.0, 2.0),
    )
    model = model.to(torch.bfloat16)
    references = torch.randn(1, 2, 2, 2, 16)
    with_refs = ref.sample_reference_latents(model, references, text, null, reference_guidance_scale=1.5, **common)
    assert with_refs.video.shape == (1, 2, 2, 2, 16) and with_refs.audio.shape[0] == 1
    assert torch.isfinite(with_refs.video.float()).all()
    without = ref.sample_reference_latents(model, None, text, null, reference_guidance_scale=0.0, **common)
    assert not torch.equal(with_refs.video, without.video)
    with pytest.raises(ValueError, match="at least one reference"):
        ref.sample_reference_latents(model, None, text, null, reference_guidance_scale=1.0, **common)


@pytest.mark.parametrize("guidance,reference_guidance,expected_calls", [(3.0, 1.5, 6), (1.0, 0.0, 2)])
def test_reference_guidance_equation_keeps_clean_refs_and_slices_tail(guidance, reference_guidance, expected_calls):
    references = torch.full((1, 2, 1, 1, 16), 7.0)
    original = references.clone()
    calls = []

    class RecordingDiT:
        visual_cond = True
        visual_token_type_embeddings = True

        def __call__(self, **kwargs):
            x = kwargs["x_video"]
            with_refs = x.shape[1] == 4
            positive = bool(kwargs["text_embed"].item())
            calls.append((positive, with_refs, kwargs))
            if with_refs:
                assert torch.equal(x[:, 2:, ..., :16], original)
                assert torch.count_nonzero(x[..., 16:32]) == 0
                assert x[:, 2:, ..., 32].eq(1).all()
                assert kwargs["visual_ref_slot_ids"].tolist() == [[-1, -1, 0, 1]]
            else:
                assert kwargs["visual_ref_slot_ids"] is None
            velocity = 5.0 if positive and with_refs else 2.0 if with_refs else 1.0
            video = torch.full_like(x[..., :16], velocity)
            video[:, 2:] = 10000.0  # a reference-output error must never enter the solver
            return video, torch.full_like(kwargs["x_audio"], velocity * 2)

    text = {"text_embeds": torch.ones(1), "pooled_embed": torch.zeros(1)}
    null = {"text_embeds": torch.zeros(1), "pooled_embed": torch.zeros(1)}
    video, audio = ref.reference_denoise_loop(
        RecordingDiT(),
        torch.zeros(1, 2, 1, 1, 16),
        torch.zeros(1, 3, 40),
        references,
        text,
        null,
        text_rope="positive",
        null_text_rope="negative",
        audio_rope=None,
        visual_rope_with_refs="with_refs",
        visual_rope_without_refs="without_refs",
        attention_mask=torch.tensor([True]),
        null_attention_mask=torch.tensor([False]),
        num_steps=2,
        guidance_scale=guidance,
        reference_guidance_scale=reference_guidance,
        scheduler_scale=1.0,
    )
    expected = 2 + guidance * (5 - 2) + reference_guidance * (5 - 1)
    assert video.eq(-expected).all() and audio.eq(-2 * expected).all()
    assert len(calls) == expected_calls
    assert torch.equal(references, original)
    for positive, with_refs, kwargs in calls:
        assert kwargs["text_rope"] == ("positive" if positive else "negative")
        assert kwargs["visual_rope"] == ("with_refs" if with_refs else "without_refs")
        assert bool(kwargs["attention_mask"].item()) == positive


def test_training_reference_tail_is_clean_and_unsupervised():
    trainer = r2av_trainer()
    model = randomize(tiny_dit(reference_slots=2)).train()
    captured = []

    def capture(module, args, kwargs, output):
        captured.append(kwargs["x_video"].detach().clone())
        # Tail predictions are deliberately huge; training loss only covers target frames.
        return torch.cat([output[0][:, :2], torch.full_like(output[0][:, 2:], 1e6)], dim=1), output[1]

    references = torch.full((2, 16, 2, 2), 7.0)
    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        loss = run_process_batch(trainer, model, make_batch([references]))
    finally:
        handle.remove()
    assert torch.isfinite(loss) and loss < 1e4
    assert torch.equal(captured[0][:, 2:, ..., :16], references.permute(0, 2, 3, 1).unsqueeze(0))
    assert captured[0][:, :2, ..., 32].eq(0).all()
    assert captured[0][:, 2:, ..., 32].eq(1).all()
    loss.backward()
    assert model.visual_ref_slot_embeddings.weight.grad.abs().sum(dim=1).gt(0).all()
