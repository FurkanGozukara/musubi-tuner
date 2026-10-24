"""LTX caches record the LTX generation they were made for: --skip_existing re-encodes a cache of another version, an
old cache without the field counts as LTX-2.3, and training refuses caches from the other side of the LTX-2.5
boundary (text encoder Gemma 3 vs Gemma 4, another VAE)."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner import ltx2_cache_version as cv
from musubi_tuner.dataset.cache_io import save_latent_cache_ltx2, save_text_encoder_output_cache_ltx2_gemma
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.ltx2_cache_version import (
    cache_matches_version,
    check_training_caches,
    normalize_ltx_version,
    read_cache_ltx_version,
    resolve_caching_version,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def _item(tmp_path, name: str, bucket=(64, 64)) -> ItemInfo:
    item = ItemInfo(name, "a caption", (64, 64), bucket_size=bucket, frame_count=1)
    item.latent_cache_path = str(tmp_path / f"{name}_0001-001_0064x0064_ltx2.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / f"{name}_ltx2_te.safetensors")
    return item


def _latent() -> torch.Tensor:
    return torch.zeros(128, 1, 2, 2)


def _metadata(path) -> dict:
    with MemoryEfficientSafeOpen(str(path)) as handle:
        return dict(handle.metadata())


@pytest.mark.parametrize(
    "value,expected",
    [(None, "2.3"), ("", "2.3"), ("2", "2.0"), ("2.0", "2.0"), ("2.3", "2.3"), ("2.3.0", "2.3"), ("2.5", "2.5"), (2.5, "2.5"), ("2.4.1", "2.5"), ("x", "2.3")],
)
def test_version_spellings(value, expected):
    assert normalize_ltx_version(value) == expected


def test_latent_cache_records_its_version_and_old_caches_count_as_ltx23(tmp_path):
    new = _item(tmp_path, "new")
    save_latent_cache_ltx2(new, _latent(), ltx_version="2.5")
    assert _metadata(new.latent_cache_path)["ltx_version"] == "2.5"
    assert read_cache_ltx_version(new.latent_cache_path) == "2.5"

    old = _item(tmp_path, "old")
    save_latent_cache_ltx2(old, _latent())  # a writer that passes no version (and every cache from before)
    assert "ltx_version" not in _metadata(old.latent_cache_path)
    assert read_cache_ltx_version(old.latent_cache_path) == "2.3"

    assert cache_matches_version(new.latent_cache_path, "2.5")
    assert not cache_matches_version(new.latent_cache_path, "2.3")
    assert cache_matches_version(old.latent_cache_path, "2.3")
    assert not cache_matches_version(old.latent_cache_path, "2.5")
    assert not cache_matches_version(old.latent_cache_path, "2.0")  # exact: a 2.0 run re-encodes (once) too
    assert not cache_matches_version(str(tmp_path / "missing.safetensors"), "2.3")


class _FakeLatentDataset:
    def __init__(self, items):
        self.items = items

    def retrieve_latent_cache_batches(self, num_workers, skip_broken=False):
        yield (64, 64, 1), list(self.items)

    def get_all_latent_cache_files(self):
        return [item.latent_cache_path for item in self.items]


def _cache_args(**overrides):
    args = argparse.Namespace(
        num_workers=1, skip_broken=False, skip_existing=True, batch_size=None, keep_cache=True, _cache_shard=(0, 1)
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("run_version,re_encoded", [("2.5", {"old", "v23"}), ("2.3", {"v25"})])
def test_skip_existing_re_encodes_latent_caches_of_another_version(tmp_path, run_version, re_encoded):
    items = {name: _item(tmp_path, name) for name in ("old", "v23", "v25")}
    save_latent_cache_ltx2(items["old"], _latent())
    save_latent_cache_ltx2(items["v23"], _latent(), ltx_version="2.3")
    save_latent_cache_ltx2(items["v25"], _latent(), ltx_version="2.5")
    encoded = []

    def is_current(item):  # as ltx2_cache_latents.main builds it
        return cache_latents.latent_cache_matches_bucket(item) and cache_matches_version(item.latent_cache_path, run_version)

    cache_latents.encode_datasets(
        [_FakeLatentDataset(items.values())],
        lambda batch: encoded.extend(item.item_key for item in batch),
        _cache_args(),
        cache_is_current=is_current,
    )
    assert set(encoded) == re_encoded


class _FakeTextDataset:
    def __init__(self, items):
        self.items = items

    def retrieve_text_encoder_output_cache_batches(self, num_workers):
        yield list(self.items)

    def get_all_text_encoder_output_cache_files(self):
        return [item.text_encoder_output_cache_path for item in self.items]


def _save_text(item, version, *, audio=True):
    save_text_encoder_output_cache_ltx2_gemma(
        item,
        video_prompt_embeds=torch.zeros(4, 8),
        prompt_attention_mask=torch.ones(4, dtype=torch.long),
        audio_prompt_embeds=torch.zeros(4, 6) if audio else None,
        ltx_version=version,
    )


def test_text_cache_of_another_version_is_replaced_not_merged(tmp_path):
    item = _item(tmp_path, "clip")
    _save_text(item, "2.3", audio=True)  # an LTX-2.3 AV cache
    _save_text(item, "2.5", audio=False)  # an LTX-2.5 video-only run over it
    with MemoryEfficientSafeOpen(item.text_encoder_output_cache_path) as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
    assert metadata["ltx_version"] == "2.5"
    assert not any(key.startswith("audio_prompt_embeds") for key in keys)  # LTX-2.3 Gemma 3 tensors are gone

    # the same version still merges (e.g. video-only then AV for the same model)
    _save_text(item, "2.5", audio=True)
    _save_text(item, "2.5", audio=False)
    with MemoryEfficientSafeOpen(item.text_encoder_output_cache_path) as handle:
        assert any(key.startswith("audio_prompt_embeds") for key in handle.keys())


@pytest.mark.parametrize("run_version,re_encoded", [("2.5", {"old", "v23"}), ("2.3", {"v25"})])
def test_skip_existing_re_encodes_text_caches_of_another_version(tmp_path, run_version, re_encoded):
    items = {name: _item(tmp_path, name) for name in ("old", "v23", "v25")}
    save_file(  # a cache from before the version field, as the previous writer left it
        {"video_prompt_embeds_float32": torch.zeros(4, 8), "text_float32": torch.zeros(4, 8)},
        items["old"].text_encoder_output_cache_path,
        metadata={"architecture": "ltx2_v1", "caption1": "a caption", "format_version": "1.0.1"},
    )
    _save_text(items["v23"], "2.3")
    _save_text(items["v25"], "2.5")
    datasets = [_FakeTextDataset(items.values())]
    files, paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    encoded = []
    cache_text_encoder_outputs.process_text_encoder_batches(
        1,
        True,
        None,
        datasets,
        files,
        paths,
        lambda batch: encoded.extend(item.item_key for item in batch),
        cache_is_current=lambda item: cache_matches_version(item.text_encoder_output_cache_path, run_version),
    )
    assert set(encoded) == re_encoded


def _training_dataset(items):
    return SimpleNamespace(batch_manager=SimpleNamespace(buckets={(64, 64, 1): list(items)}))


def test_training_refuses_caches_from_the_other_side_of_ltx25(tmp_path):
    old = _item(tmp_path, "old")
    save_latent_cache_ltx2(old, _latent())
    _save_text(old, None)  # no version: LTX-2.3
    datasets = [_training_dataset([old])]

    check_training_caches(datasets, "2.3")
    check_training_caches(datasets, "2.0")  # LTX-2.0 and 2.3 caches stay interchangeable at training time
    with pytest.raises(ValueError, match="another LTX generation"):
        check_training_caches(datasets, "2.5")

    new = _item(tmp_path, "new")
    save_latent_cache_ltx2(new, _latent(), ltx_version="2.5")
    _save_text(new, "2.5")
    check_training_caches([_training_dataset([new])], "2.5")
    with pytest.raises(ValueError, match="Re-run latent and text encoder caching"):
        check_training_caches([_training_dataset([new])], "2.3")


def _safetensors(path, tensors=None, metadata=None):
    save_file(tensors or {"x": torch.zeros(1)}, str(path), metadata=metadata or {})
    return str(path)


def test_caching_runs_record_the_version_of_the_model_files(tmp_path):
    args = argparse.Namespace(ltx_version="2.3")
    ltx23 = _safetensors(tmp_path / "ltx-2.3.safetensors", metadata={"config": json.dumps({"vae": {"_class_name": "CausalVideoAutoencoder"}})})
    assert resolve_caching_version(args, [ltx23]) == "2.3"
    assert resolve_caching_version(argparse.Namespace(ltx_version="2.0"), [ltx23]) == "2.0"
    assert resolve_caching_version(argparse.Namespace(ltx_version="2.5"), [ltx23]) == "2.5"

    diff_vae = _safetensors(tmp_path / "ltx-2.5-video-vae.safetensors", metadata={"config": json.dumps({"vae": {"_class_name": "CausalDiffusionVAE"}})})
    transformer25 = _safetensors(tmp_path / "ltx-2.5-transformer.safetensors", tensors={"model.diffusion_model.keyframes_abs_pos_embedding": torch.zeros(1, 4)})
    versioned = _safetensors(tmp_path / "ltx-2.5-versioned.safetensors", metadata={"model_version": "2.5.0"})
    packed_gemma = _safetensors(tmp_path / "gemma4-with-proj.safetensors", metadata={"gemma_config": json.dumps({"model_type": "gemma4_unified"})})
    gemma3_file = _safetensors(tmp_path / "gemma3.safetensors", metadata={"gemma_config": json.dumps({"model_type": "gemma3"})})
    gemma4_dir = tmp_path / "gemma4-dir"
    gemma4_dir.mkdir()
    (gemma4_dir / "config.json").write_text(json.dumps({"model_type": "gemma4_unified"}), encoding="utf-8")
    gemma3_dir = tmp_path / "gemma3-dir"
    gemma3_dir.mkdir()
    (gemma3_dir / "config.json").write_text(json.dumps({"model_type": "gemma3"}), encoding="utf-8")

    for path in (diff_vae, transformer25, versioned, packed_gemma, str(gemma4_dir)):
        assert resolve_caching_version(args, [ltx23, path]) == "2.5", path
    for path in (gemma3_file, str(gemma3_dir), None, str(tmp_path / "missing.safetensors")):
        assert resolve_caching_version(args, [ltx23, path]) == "2.3", path


def test_precached_sample_files_must_match_the_ltx25_side(tmp_path):
    from musubi_tuner.ltx2_sampling import _check_sample_cache_version

    _check_sample_cache_version({"prompt_cache": []}, SimpleNamespace(ltx_version="2.3"), "c.pt", "sample prompt")
    _check_sample_cache_version({"ltx_version": "2.5"}, SimpleNamespace(ltx_version="2.5"), "c.pt", "sample prompt")
    with pytest.raises(ValueError, match="made for LTX-2.3"):
        _check_sample_cache_version({"prompt_cache": []}, SimpleNamespace(ltx_version="2.5"), "c.pt", "sample prompt")
    with pytest.raises(ValueError, match="made for LTX-2.5"):
        _check_sample_cache_version({"ltx_version": "2.5"}, SimpleNamespace(ltx_version="2.3"), "c.pt", "sample latent")


def test_ltx_trainer_checks_its_dataset_caches(tmp_path):
    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    item = _item(tmp_path, "clip")
    save_latent_cache_ltx2(item, _latent(), ltx_version="2.3")
    trainer = LTX2NetworkTrainer.__new__(LTX2NetworkTrainer)
    trainer._ltx_version = "2.5"
    with pytest.raises(ValueError, match="another LTX generation"):
        trainer.check_dataset_caches(argparse.Namespace(ltx_version="2.5"), [_training_dataset([item])])
    trainer.check_dataset_caches(argparse.Namespace(ltx_version="2.3"), [_training_dataset([item])])
    assert cv.CACHE_VERSION_METADATA_KEY == "ltx_version"
