"""LTX-2.5 command-line surface: the new flags of the trainer, the caching scripts and the generate script, the
sampling presets / sampler / sigma resolution for 2.5, the generate script's prompt-file seeds, the Euler ancestral
step, and the audio/video mux (FFmpeg stream copy with the PyAV fallback)."""

from __future__ import annotations

import argparse
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _train_parser():
    from musubi_tuner.ltx2_args import ltx2_setup_parser
    from musubi_tuner.training.parser_common import setup_parser_common

    return ltx2_setup_parser(setup_parser_common())


def test_trainer_flags():
    parser = _train_parser()
    args = parser.parse_args([])
    assert args.ltx_version == "2.3"
    assert args.ltx2_video_vae is None and args.ltx2_audio_vae is None
    assert args.independent_audio_timestep is None  # resolved per version: on for LTX-2.5 AV
    assert args.ltx_version_check_mode is None  # resolved per version: error for LTX-2.5
    assert args.sample_sampling_preset == "defaults"

    args = parser.parse_args(
        [
            "--ltx_version",
            "2.5",
            "--ltx2_video_vae",
            "vae.safetensors",
            "--ltx2_audio_vae",
            "audio.safetensors",
            "--ltx2_text_encoder_checkpoint",
            "gemma4.safetensors",
            "--sample_sampling_preset",
            "ltx25_full",
            "--no-independent_audio_timestep",
            "--ltx_version_check_mode",
            "warn",
        ]
    )
    assert args.ltx_version == "2.5"
    assert (args.ltx2_video_vae, args.ltx2_audio_vae) == ("vae.safetensors", "audio.safetensors")
    assert args.ltx2_text_encoder_checkpoint == "gemma4.safetensors"
    assert args.sample_sampling_preset == "ltx25_full"
    assert args.independent_audio_timestep is False
    assert args.ltx_version_check_mode == "warn"
    assert parser.parse_args(["--independent_audio_timestep"]).independent_audio_timestep is True
    assert parser.parse_args(["--sampling_preset", "ltx25"]).sample_sampling_preset == "ltx25"
    with pytest.raises(SystemExit):
        parser.parse_args(["--ltx_version", "2.4"])


def test_caching_script_flags():
    from musubi_tuner import cache_latents, cache_text_encoder_outputs
    from musubi_tuner import ltx2_cache_latents, ltx2_cache_text_encoder_outputs

    latents = ltx2_cache_latents.ltx2_setup_parser(cache_latents.setup_parser_common()).parse_args(
        ["--dataset_config", "d.toml", "--ltx_version", "2.5", "--ltx2_audio_vae", "a.safetensors", "--sampling_preset", "ltx25_full"]
    )
    assert (latents.ltx_version, latents.ltx2_audio_vae, latents.sample_sampling_preset) == ("2.5", "a.safetensors", "ltx25_full")
    text = ltx2_cache_text_encoder_outputs.ltx2_setup_parser(cache_text_encoder_outputs.setup_parser_common()).parse_args(
        ["--dataset_config", "d.toml", "--ltx_version", "2.5", "--ltx2_text_encoder_checkpoint", "g.safetensors", "--sampling_preset", "ltx25"]
    )
    assert (text.ltx_version, text.ltx2_text_encoder_checkpoint, text.sample_sampling_preset) == ("2.5", "g.safetensors", "ltx25")


def test_generate_script_flags(monkeypatch):
    from musubi_tuner import ltx2_generate_video

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ltx2_generate_video.py",
            "--prompt",
            "a fox",
            "--ltx_version",
            "2.5",
            "--ltx2_text_encoder_checkpoint",
            "gemma4.safetensors",
            "--ltx2_audio_vae",
            "audio.safetensors",
        ],
    )
    args = ltx2_generate_video.parse_args()
    assert args.sampling_preset == "ltx25" and args.sample_two_stage  # the official distilled two-stage recipe
    assert (args.width, args.height, args.frame_count, args.sample_steps) == (1536, 1024, 121, 8)
    assert args.ltx2_text_encoder_checkpoint == "gemma4.safetensors" and args.ltx2_audio_vae == "audio.safetensors"

    monkeypatch.setattr(sys, "argv", ["g.py", "--prompt", "a fox", "--ltx_version", "2.5", "--sampling_preset", "ltx25_full"])
    args = ltx2_generate_video.parse_args()
    assert not getattr(args, "sample_two_stage", False)
    assert (args.width, args.height, args.sample_steps, args.video_cfg_scale, args.stg_scale) == (768, 512, 30, 3.0, 1.0)


def test_ltx25_presets_sampler_and_sigmas():
    from musubi_tuner.ltx2_defaults import get_ltx2_sampling_preset
    from musubi_tuner.ltx2_samplers import resolve_ltx2_sampler
    from musubi_tuner.ltx_2.components.schedulers import LTX23_DISTILLED_SIGMAS, build_ltx2_sigmas

    assert get_ltx2_sampling_preset("defaults", ltx_version="2.5") is get_ltx2_sampling_preset("ltx25")
    assert get_ltx2_sampling_preset("defaults", ltx_version="2.3") is get_ltx2_sampling_preset("ltx23")
    full = get_ltx2_sampling_preset("ltx25_full")
    ltx23 = get_ltx2_sampling_preset("ltx23")
    for field in ("sample_steps", "width", "height", "frame_count", "video_cfg_scale", "audio_cfg_scale", "stg_scale", "stg_blocks"):
        assert getattr(full, field) == getattr(ltx23, field), field
    assert resolve_ltx2_sampler("auto", "ltx25") == "euler"
    assert resolve_ltx2_sampler("auto", "ltx25_full") == "res_2s"
    assert resolve_ltx2_sampler("auto", "ltx23") == "res_2s"
    assert torch.equal(build_ltx2_sigmas(8, sampling_preset="ltx25"), torch.tensor(LTX23_DISTILLED_SIGMAS))
    assert not torch.equal(build_ltx2_sigmas(8, torch.zeros(1, 128, 2, 4, 4), sampling_preset="ltx25_full"), torch.tensor(LTX23_DISTILLED_SIGMAS))


def test_generate_prompt_file_seeds(monkeypatch):
    from musubi_tuner.ltx2_generate_video import _build_prompt_list

    def _prompts(*_args):
        return [{"prompt": "a"}, {"prompt": "b", "seed": 7}]

    trainer = SimpleNamespace(process_sample_prompts=_prompts)
    args = argparse.Namespace(sample_prompts="p.txt", frame_count=9, frame_rate=24.0, sample_steps=8, seed=None, guidance_scale=None, height=None, width=None)
    prompts = _build_prompt_list(trainer, args, None)  # no --seed: no TypeError, every prompt keeps its own seed
    assert [p.get("seed") for p in prompts] == [None, 7]
    args.seed = 100
    prompts = _build_prompt_list(trainer, args, None)
    assert [p.get("seed") for p in prompts] == [100, 7]


def test_euler_ancestral_step():
    from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerAncestralDiffusionStep

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    sample, denoised, noise = torch.full((4,), 2.0), torch.zeros(4), torch.ones(4)
    step = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
    assert torch.equal(step.step(sample, denoised, sigmas, 1, noise), denoised)  # last step returns x0
    out = step.step(sample, denoised, sigmas, 0, noise)
    # eta 1: sigma_down = sigma_next^2 / sigma, renoise to sigma_next in the rectified-flow parameterisation
    sigma, sigma_next = 1.0, 0.5
    sigma_down = sigma_next * (1.0 + (sigma_next / sigma - 1.0))
    ratio = sigma_down / sigma
    alpha_next, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    expected = (alpha_next / alpha_down) * (ratio * 2.0) + (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2) ** 0.5
    assert torch.allclose(out, torch.full((4,), expected))


def _write_media(tmp_path):
    av = pytest.importorskip("av")
    video_path = tmp_path / "sample.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((np.sin(np.arange(8000) / 10.0) * 8000).astype(np.int16).tobytes())
    return video_path, wav_path


def _streams(path):
    import av

    with av.open(str(path)) as container:
        return sorted(stream.type for stream in container.streams)


def test_mux_without_ffmpeg_falls_back_to_pyav(tmp_path, monkeypatch):
    from musubi_tuner import ltx2_inference

    video_path, wav_path = _write_media(tmp_path)
    monkeypatch.setattr(ltx2_inference, "find_ffmpeg_executable", lambda: None)
    out = tmp_path / "pyav_av.mp4"
    ltx2_inference.mux_video_audio(str(video_path), str(wav_path), str(out))
    assert _streams(out) == ["audio", "video"]

    # the training sampler keeps its own PyAV muxer as the fallback
    from musubi_tuner.ltx2_sampling import LTX2SamplingMixin

    out2 = tmp_path / "sampler_av.mp4"
    LTX2SamplingMixin._mux_video_audio(str(video_path), str(wav_path), str(out2))
    assert _streams(out2) == ["audio", "video"]


def test_mux_with_ffmpeg_copies_the_video_stream(tmp_path):
    from musubi_tuner import ltx2_inference

    ffmpeg = ltx2_inference.find_ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("no FFmpeg executable")
    video_path, wav_path = _write_media(tmp_path)
    out = tmp_path / "copy_av.mp4"
    assert ltx2_inference.mux_video_audio_ffmpeg(ffmpeg, str(video_path), str(wav_path), str(out))
    assert _streams(out) == ["audio", "video"]

    import av

    def _video_packets(path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return [bytes(packet) for packet in container.demux(stream) if packet.size]

    assert _video_packets(out) == _video_packets(video_path)  # stream copy: the encoded frames are untouched
