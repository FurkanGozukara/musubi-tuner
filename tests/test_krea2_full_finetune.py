import argparse
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser


class _FakeVAE:
    dtype = torch.bfloat16

    def to(self, _device):
        return self

    def eval(self):
        return self

    def decode_to_pixels(self, latent):
        return torch.zeros(
            (latent.shape[0], 3, latent.shape[-2] * 8, latent.shape[-1] * 8),
            device=latent.device,
            dtype=self.dtype,
        )


class _PreparedTransformer:
    def __init__(self, fill_value=0.0):
        self.calls = []
        self.fill_value = fill_value

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return torch.full_like(inputs["img"], self.fill_value)


class _WrapperAwareAccelerator:
    device = torch.device("cpu")

    def __init__(self, prepared_transformer, raw_transformer):
        self.prepared_transformer = prepared_transformer
        self.raw_transformer = raw_transformer
        self.unwrap_calls = []

    def unwrap_model(self, model, keep_fp32_wrapper=False):
        self.unwrap_calls.append((model, keep_fp32_wrapper))
        assert model is self.prepared_transformer
        return self.raw_transformer

    @staticmethod
    def autocast():
        return nullcontext()


def _run_sample(monkeypatch, *, dit_variant, turbo_dit=None):
    network_module = importlib.import_module("musubi_tuner.krea2_train_network")
    observed_schedule = {}

    def fake_timesteps(seqlen, steps, x1, x2, y1, y2, mu=None):
        observed_schedule.update(
            seqlen=seqlen,
            steps=steps,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            mu=mu,
        )
        return [1.0, 0.0]

    monkeypatch.setattr(network_module.krea2_sampling, "timesteps", fake_timesteps)
    monkeypatch.setattr(network_module, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(network_module, "tqdm", lambda values, **_kwargs: values)

    prepared = _PreparedTransformer()
    raw = SimpleNamespace(config=SimpleNamespace(patch=2, channels=1))
    accelerator = _WrapperAwareAccelerator(prepared, raw)
    trainer = Krea2NetworkTrainer()

    pixels = trainer.do_inference(
        accelerator,
        SimpleNamespace(dit_variant=dit_variant, turbo_dit=turbo_dit, mixed_precision="no"),
        {"krea2_vl_embed": torch.ones((2, 1, 4), dtype=torch.bfloat16)},
        _FakeVAE(),
        torch.float32,
        prepared,
        discrete_flow_shift=0,
        sample_steps=1,
        width=16,
        height=16,
        frame_count=1,
        generator=torch.Generator().manual_seed(1),
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        cfg_scale=1.0,
    )

    return observed_schedule, accelerator, prepared, pixels


def test_dit_variant_parser_defaults_to_raw_and_accepts_turbo():
    parser = krea2_setup_parser(argparse.ArgumentParser())

    assert parser.parse_args([]).dit_variant == "raw"
    assert parser.parse_args(["--dit_variant", "turbo"]).dit_variant == "turbo"


def test_raw_primary_uses_resolution_aware_sampling_schedule(monkeypatch):
    schedule, _, _, _ = _run_sample(monkeypatch, dit_variant="raw")

    assert schedule == {
        "seqlen": 1,
        "steps": 1,
        "x1": 256,
        "x2": 6400,
        "y1": 0.5,
        "y2": 1.15,
        "mu": None,
    }


def test_turbo_primary_uses_fixed_sampling_mu(monkeypatch):
    schedule, _, _, _ = _run_sample(monkeypatch, dit_variant="turbo")

    assert schedule["mu"] == 1.15


def test_lora_raw_to_turbo_sample_path_keeps_fixed_sampling_mu(monkeypatch):
    schedule, _, _, _ = _run_sample(monkeypatch, dit_variant="raw", turbo_dit="turbo.safetensors")

    assert schedule["mu"] == 1.15


def test_fp32_sampling_reads_raw_config_but_forwards_through_prepared_wrapper(monkeypatch):
    _, accelerator, prepared, pixels = _run_sample(monkeypatch, dit_variant="raw")

    assert accelerator.unwrap_calls == [(prepared, False)]
    assert len(prepared.calls) == 1
    assert prepared.calls[0]["img"].dtype is torch.float32
    assert prepared.calls[0]["context"].dtype is torch.float32
    assert pixels.dtype is torch.float32


def test_call_dit_reads_raw_config_but_forwards_through_prepared_wrapper():
    trainer = Krea2NetworkTrainer()
    prepared = _PreparedTransformer(fill_value=3.0)
    raw = SimpleNamespace(config=SimpleNamespace(patch=2))
    accelerator = _WrapperAwareAccelerator(prepared, raw)
    latents = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    noise = torch.ones_like(latents)

    output = trainer.call_dit(
        SimpleNamespace(gradient_checkpointing=False),
        accelerator,
        prepared,
        latents,
        {"latents": latents, "krea2_vl_embed": [torch.zeros((1, 1, 4), dtype=torch.float32)]},
        noise,
        torch.zeros_like(latents),
        torch.tensor([500.0]),
        torch.float32,
    )

    assert accelerator.unwrap_calls == [(prepared, False)]
    assert len(prepared.calls) == 1
    assert torch.equal(output.pred, torch.full_like(latents, 3.0))
    assert torch.equal(output.target, noise - latents)


@pytest.mark.parametrize(
    ("turbo_dit", "turbo_dit_cache", "rejected_option"),
    [
        ("turbo.safetensors", False, "--turbo_dit"),
        (None, True, "--turbo_dit_cache"),
    ],
)
def test_full_trainer_rejects_lora_only_turbo_swapping(turbo_dit, turbo_dit_cache, rejected_option):
    full_train = importlib.import_module("musubi_tuner.krea2_train")

    with pytest.raises(ValueError, match=rejected_option):
        full_train.Krea2Trainer().validate_full_finetune_model_args(
            SimpleNamespace(
                dit_variant="raw",
                turbo_dit=turbo_dit,
                turbo_dit_cache=turbo_dit_cache,
            )
        )


def test_full_loader_passes_trainable_dtype_directly(monkeypatch):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    loaded_model = object()
    loader_call = {}

    def fake_load_krea2_dit(path, **kwargs):
        loader_call.update(path=path, **kwargs)
        return loaded_model

    monkeypatch.setattr(full_train.krea2_utils, "load_krea2_dit", fake_load_krea2_dit)

    result = full_train.Krea2Trainer().load_full_finetune_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(fp8_scaled=False),
        "turbo.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        torch.float32,
    )

    assert result is loaded_model
    assert loader_call == {
        "path": "turbo.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "fp8_scaled": False,
        "loading_device": torch.device("cpu"),
        "attn_mode": "torch",
        "split_attn": False,
    }


@pytest.mark.parametrize("dit_variant", ["raw", "turbo"])
def test_full_primary_variant_is_accepted_and_recorded_as_string_metadata(dit_variant):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    trainer = full_train.Krea2Trainer()
    args = SimpleNamespace(dit_variant=dit_variant, turbo_dit=None, turbo_dit_cache=False)

    trainer.validate_full_finetune_model_args(args)

    assert trainer.full_finetune_metadata(args) == {"ss_krea2_dit_variant": dit_variant}


def test_root_entrypoint_is_exact_thin_main_shim():
    root_entrypoint = Path(__file__).parents[1] / "krea2_train.py"

    assert root_entrypoint.read_text(encoding="utf-8") == (
        "from musubi_tuner.krea2_train import main\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
