"""Krea 2 training samples use CFG as the official sampler does: an empty negative prompt when none is given and the
CFG scale is above 1 (musubi only runs CFG with a negative prompt; RAW samples came out without it)."""

from types import SimpleNamespace

import torch

from musubi_tuner import krea2_train_network as krea2


def test_sample_prompts_get_an_empty_negative_when_cfg_is_on(monkeypatch, tmp_path):
    encoded = []

    def fake_embeds(encoder, prompts):
        encoded.extend(prompts)
        return torch.zeros(1, 3, 2, 4), torch.ones(1, 3, dtype=torch.bool)

    monkeypatch.setattr(krea2.krea2_utils, "load_krea2_text_encoder", lambda *a, **k: object())
    monkeypatch.setattr(krea2.krea2_utils, "get_krea2_prompt_embeds", fake_embeds)
    path = tmp_path / "prompts.txt"
    path.write_text("raw default --w 512\nturbo --l 1.0\ncfg with negative --l 5.5 --n blurry\n", encoding="utf-8")
    trainer = krea2.Krea2NetworkTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    parameters = trainer.process_sample_prompts(SimpleNamespace(text_encoder="te.safetensors"), accelerator, str(path))

    raw, turbo, explicit = parameters
    assert raw["negative_prompt"] == "" and "negative_krea2_vl_embed" in raw  # default CFG 5.5 -> CFG on
    assert "negative_prompt" not in turbo and "negative_krea2_vl_embed" not in turbo  # CFG 1: none needed
    assert explicit["negative_prompt"] == "blurry"
    assert "" in encoded and "blurry" in encoded
