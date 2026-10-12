"""Wan --skip_existing: a text-to-video latent cache is re-encoded for image-to-video training (same file name; I2V
training reads latents_image_* and, with CLIP, clip_*), and an I2V cache serves both."""

import argparse
import types

import torch
from safetensors.torch import save_file

from musubi_tuner.wan_cache_latents import wan_cache_is_current


def _cache(path, keys, bucket="512x512"):
    save_file({key: torch.zeros(1) for key in keys}, path, metadata={"bucket_size": bucket})
    return types.SimpleNamespace(latent_cache_path=str(path), bucket_size=(512, 512))


def _args(i2v=False, clip=None, one_frame=False):
    return argparse.Namespace(i2v=i2v, clip=clip, one_frame=one_frame)


def test_text_to_video_caches_are_re_encoded_for_image_to_video(tmp_path):
    t2v = _cache(tmp_path / "a_0512x0512_wan.safetensors", ["latents_21x64x64_bf16"])
    i2v = _cache(tmp_path / "b_0512x0512_wan.safetensors", ["latents_21x64x64_bf16", "latents_image_21x64x64_bf16"])
    i2v_clip = _cache(tmp_path / "c_0512x0512_wan.safetensors", ["latents_21x64x64_bf16", "latents_image_21x64x64_bf16", "clip_fp16"])
    assert wan_cache_is_current(_args())(t2v) and wan_cache_is_current(_args())(i2v)
    assert not wan_cache_is_current(_args(i2v=True))(t2v)
    assert wan_cache_is_current(_args(i2v=True))(i2v)
    assert not wan_cache_is_current(_args(i2v=True, clip="clip.pth"))(i2v)
    assert wan_cache_is_current(_args(i2v=True, clip="clip.pth"))(i2v_clip)
    other_bucket = _cache(tmp_path / "d_0512x0512_wan.safetensors", ["latents_21x64x64_bf16", "latents_image_21x64x64_bf16"], "768x768")
    assert not wan_cache_is_current(_args(i2v=True))(other_bucket)
