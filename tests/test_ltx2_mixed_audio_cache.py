"""Audio clips mixed into an LTX-2 audio+video run get a virtual video latent cache (akane 9dc9abc2).

Training takes an audio clip only when its ``*_ltx2.safetensors`` video latent sits next to its
``*_ltx2_audio.safetensors``. Latent caching wrote that virtual video latent in audio mode only, so in an AV
run with an audio dataset next to image/video datasets every clip was dropped at training start.
"""

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_LTX2
from musubi_tuner.dataset.audio_quota_sampler import split_concat_indices_by_audio
from musubi_tuner.dataset.image_video_dataset import AudioDataset, DatasetGroup
from musubi_tuner.ltx2_cache_latents import _should_cache_audio_dataset_video_latents


def test_mixed_av_audio_dataset_gets_virtual_video_cache():
    assert _should_cache_audio_dataset_video_latents(audio_only=False, audio_video=True, audio_datasets=[object()])
    assert _should_cache_audio_dataset_video_latents(audio_only=True, audio_video=False, audio_datasets=[object()])


def test_video_only_cache_does_not_create_audio_virtual_video_cache():
    assert not _should_cache_audio_dataset_video_latents(audio_only=False, audio_video=False, audio_datasets=[object()])
    assert not _should_cache_audio_dataset_video_latents(audio_only=False, audio_video=True, audio_datasets=[])


def _write_clip_caches(cache_dir, *, with_video_latent: bool) -> None:
    if with_video_latent:
        save_file({"latents_1x2x2_float32": torch.zeros(4, 1, 2, 2)}, str(cache_dir / "voice_1024x1024_ltx2.safetensors"))
    save_file({"audio_latents_10x4x2_float32": torch.zeros(4, 10, 2)}, str(cache_dir / "voice_1024x1024_ltx2_audio.safetensors"))
    save_file({"gemma_embeds_float32": torch.zeros(1, 2)}, str(cache_dir / "voice_ltx2_te.safetensors"))


def _cached_audio_dataset(tmp_path, cache_dir) -> AudioDataset:
    dataset = AudioDataset(
        resolution=(1024, 1024),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        audio_directory=str(tmp_path),
        cache_directory=str(cache_dir),
        cache_only=True,
        architecture=ARCHITECTURE_LTX2,
    )
    dataset.prepare_for_training()
    return dataset


def test_virtual_video_and_audio_cache_reach_audio_sampler(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_clip_caches(cache_dir, with_video_latent=True)

    dataset = _cached_audio_dataset(tmp_path, cache_dir)

    assert dataset.num_train_items == 1
    audio_indices, non_audio_indices = split_concat_indices_by_audio(DatasetGroup([dataset]))
    assert audio_indices == [0]
    assert non_audio_indices == []


def test_audio_cache_without_video_latent_is_dropped(tmp_path):
    # what an AV run cached before the fix: the clip never reaches training
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_clip_caches(cache_dir, with_video_latent=False)

    dataset = _cached_audio_dataset(tmp_path, cache_dir)

    assert dataset.num_train_items == 0

def test_audio_dataset_builds_from_a_dataset_config(tmp_path):
    """Every dataset of a config gets skip_corrupted_cache (BaseDatasetParams, the corrupted-cache handling); the audio
    dataset refused it, so each LTX audio-only run stopped at latent caching with a TypeError."""
    import argparse

    from musubi_tuner.dataset import config_utils

    config = {"general": {"caption_extension": ".txt", "batch_size": 1},
              "datasets": [{"audio_directory": str(tmp_path), "cache_directory": str(tmp_path / "cache"), "num_repeats": 1}]}
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config, argparse.Namespace(), architecture=ARCHITECTURE_LTX2
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    assert isinstance(group.datasets[0], AudioDataset)

def test_every_dataset_class_takes_every_field_of_its_config_params():
    """generate_dataset_group_by_blueprint calls each dataset class with asdict(params): a field added to the shared
    BaseDatasetParams (an upstream merge) must reach every class, the fork-only AudioDataset included."""
    import dataclasses
    import inspect

    from musubi_tuner.dataset import config_utils
    from musubi_tuner.dataset.image_video_dataset import ImageDataset, VideoDataset

    pairs = ((config_utils.ImageDatasetParams, ImageDataset), (config_utils.VideoDatasetParams, VideoDataset),
             (config_utils.AudioDatasetParams, AudioDataset))
    for params, dataset_class in pairs:
        accepted = set(inspect.signature(dataset_class.__init__).parameters)
        missing = [field.name for field in dataclasses.fields(params) if field.name not in accepted]
        assert not missing, f"{dataset_class.__name__} does not take {missing}"
