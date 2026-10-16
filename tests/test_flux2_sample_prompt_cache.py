"""FLUX.2 sample prompts encoded by the text encoder caching step (--sample_prompts_cache): training samples from them
without loading the text encoder on the GPU (Mistral 3 in BF16 peaked at 45.5 GB with the 32 GB FLUX.2 Dev preset)."""

import argparse
import types

import pytest
import torch

from musubi_tuner import flux_2_cache_text_encoder_outputs as cache_script
from musubi_tuner import flux_2_train_network
from musubi_tuner.flux_2 import flux2_utils


class _FakeEmbedder:
    dtype = torch.bfloat16

    def __init__(self):
        self.encoded = []

    def __call__(self, texts):
        self.encoded += list(texts)
        return torch.full((len(texts), 2, 3), float(len(self.encoded)))


def _args(tmp_path, prompts=None, model_version="dev"):
    encoder = tmp_path / "text_encoder.safetensors"
    if not encoder.exists():
        encoder.write_bytes(b"weights")
    return argparse.Namespace(model_version=model_version, text_encoder=str(encoder), fp8_text_encoder=False,
                              sample_prompts=str(prompts) if prompts else None, sample_prompts_cache=str(tmp_path / "prompts_cache.pt"))


def test_the_cache_keeps_one_model_and_text_encoder(tmp_path):
    args = _args(tmp_path)
    key = flux2_utils.sample_prompts_cache_key(args.model_version, args.text_encoder)
    flux2_utils.save_sample_prompts_cache(args.sample_prompts_cache, key, {"a": torch.ones(1, 2, 3), " ": torch.zeros(1, 2, 3)})
    assert set(flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, key)) == {"a", " "}
    assert flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, flux2_utils.sample_prompts_cache_key("klein-base-4b", args.text_encoder)) == {}
    other_encoder = tmp_path / "other.safetensors"
    other_encoder.write_bytes(b"other weights")
    assert flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, flux2_utils.sample_prompts_cache_key("dev", str(other_encoder))) == {}
    (tmp_path / "prompts_cache.pt").write_bytes(b"not a cache")
    assert flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, key) == {}
    assert flux2_utils.load_sample_prompts_cache(str(tmp_path / "missing.pt"), key) == {}


def test_the_caching_step_encodes_each_prompt_and_negative_once_and_keeps_earlier_ones(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a photo --d 1\na photo --d 2\nb photo --n blurry --d 3\n", encoding="utf-8")
    args = _args(tmp_path, prompts)
    embedder = _FakeEmbedder()
    cache_script.precache_sample_prompts(embedder, args, torch.device("cpu"))
    assert embedder.encoded == ["a photo", " ", "b photo", "blurry"]
    key = flux2_utils.sample_prompts_cache_key("dev", args.text_encoder)
    first = flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, key)
    assert set(first) == {"a photo", " ", "b photo", "blurry"}
    prompts.write_text("a photo --d 1\nc photo --d 1\n", encoding="utf-8")
    cache_script.precache_sample_prompts(embedder, args, torch.device("cpu"))
    assert embedder.encoded[4:] == ["c photo"]  # only the new text is encoded
    second = flux2_utils.load_sample_prompts_cache(args.sample_prompts_cache, key)
    assert set(second) == {"a photo", " ", "c photo"} and torch.equal(second["a photo"], first["a photo"])


def test_training_samples_from_the_cache_without_loading_the_text_encoder(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a photo --d 1\nb photo --n blurry --d 2\n", encoding="utf-8")
    args = _args(tmp_path)
    key = flux2_utils.sample_prompts_cache_key("dev", args.text_encoder)
    texts = ["a photo", " ", "b photo", "blurry"]
    flux2_utils.save_sample_prompts_cache(args.sample_prompts_cache, key, {t: torch.full((1, 2, 3), float(i)) for i, t in enumerate(texts)})

    def no_text_encoder(*_args, **_kwargs):
        raise AssertionError("text encoder loaded")

    monkeypatch.setattr(flux2_utils, "load_text_embedder", no_text_encoder)
    trainer = flux_2_train_network.Flux2NetworkTrainer()
    trainer.model_version_info = flux2_utils.FLUX2_MODEL_INFO["dev"]
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    params = trainer.process_sample_prompts(args, accelerator, str(prompts))
    assert [p["prompt"] for p in params] == ["a photo", "b photo"]
    assert torch.equal(params[0]["negative_ctx_vec"], torch.full((1, 2, 3), 1.0))
    assert torch.equal(params[1]["ctx_vec"], torch.full((1, 2, 3), 2.0)) and torch.equal(params[1]["negative_ctx_vec"], torch.full((1, 2, 3), 3.0))
    # a prompt the cache lacks: the text encoder is loaded as before
    prompts.write_text("c photo --d 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="text encoder loaded"):
        trainer.process_sample_prompts(args, accelerator, str(prompts))
    # no cache configured: as before
    args.sample_prompts_cache = None
    with pytest.raises(AssertionError, match="text encoder loaded"):
        trainer.process_sample_prompts(args, accelerator, str(tmp_path / "prompts.txt"))


def test_both_scripts_take_the_cache_argument():
    parser = flux_2_train_network.flux2_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args(["--sample_prompts_cache", "x.pt"]).sample_prompts_cache == "x.pt"
    parser = cache_script.flux_2_setup_parser(argparse.ArgumentParser())
    parsed = parser.parse_args(["--text_encoder", "te", "--sample_prompts", "p.txt", "--sample_prompts_cache", "x.pt"])
    assert (parsed.sample_prompts, parsed.sample_prompts_cache) == ("p.txt", "x.pt")
