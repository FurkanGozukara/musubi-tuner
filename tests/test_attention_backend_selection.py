from pathlib import Path

import torch

from musubi_tuner.modules import attention
from musubi_tuner import hv_train
from musubi_tuner.training.parser_common import setup_parser_common


def _mock_external_flash_ready(monkeypatch, *, native_available=False, capability=(12, 0)):
    monkeypatch.delenv("MUSUBI_DISABLE_EXTERNAL_FLASH_SDPA", raising=False)
    monkeypatch.setattr(attention, "flash_attn_func", object())
    monkeypatch.setattr(attention, "flash_attn_varlen_func", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: native_available)
    monkeypatch.setattr(attention, "_external_flash_supports_training", lambda device=None: True)


def test_external_flash_replaces_missing_native_sdpa_flash(monkeypatch):
    _mock_external_flash_ready(monkeypatch)

    assert attention.should_use_external_flash_for_sdpa()


def test_external_flash_does_not_replace_native_sdpa_flash(monkeypatch):
    _mock_external_flash_ready(monkeypatch, native_available=True)

    assert not attention.should_use_external_flash_for_sdpa()


def test_external_flash_sdpa_can_be_disabled(monkeypatch):
    _mock_external_flash_ready(monkeypatch)
    monkeypatch.setenv("MUSUBI_DISABLE_EXTERNAL_FLASH_SDPA", "1")

    assert not attention.should_use_external_flash_for_sdpa()


def test_external_flash_falls_back_when_training_probe_fails(monkeypatch):
    _mock_external_flash_ready(monkeypatch)
    monkeypatch.setattr(attention, "_external_flash_supports_training", lambda device=None: False)

    assert not attention.should_use_external_flash_for_sdpa()


def test_external_flash_handles_broken_native_availability_probe(monkeypatch):
    _mock_external_flash_ready(monkeypatch)
    monkeypatch.setattr(
        torch.backends.cuda,
        "is_flash_attention_available",
        lambda: (_ for _ in ()).throw(RuntimeError("backend probe failed")),
    )

    assert attention.should_use_external_flash_for_sdpa()


def test_sdpa_resolver_routes_to_verified_external_flash(monkeypatch):
    monkeypatch.setattr(attention, "should_use_external_flash_for_sdpa", lambda device=None: True)

    assert attention.resolve_sdpa_backend() == attention.AUTO_FLASH_ATTENTION_MODE


def test_sdpa_resolver_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr(attention, "should_use_external_flash_for_sdpa", lambda device=None: False)

    assert attention.resolve_sdpa_backend() == "torch"


def test_legacy_sdpa_bypasses_external_probe(monkeypatch):
    def unexpected_probe(device=None):
        raise AssertionError("legacy SDPA must not probe the external backend")

    monkeypatch.setattr(attention, "should_use_external_flash_for_sdpa", unexpected_probe)

    assert attention.resolve_sdpa_backend(use_legacy_sdpa=True) == "torch"


def test_automatic_flash_runtime_falls_back_for_unsupported_tensors(monkeypatch):
    def unexpected_flash(*args, **kwargs):
        raise AssertionError("unsupported tensors must not reach external FlashAttention")

    monkeypatch.setattr(attention, "flash_attn_func", unexpected_flash)
    query = torch.randn(1, 4, 1, 8, dtype=torch.float32)
    params = attention.AttentionParams.create_attention_params(attention.AUTO_FLASH_ATTENTION_MODE, False)

    output = attention.attention(query, query.clone(), query.clone(), params)

    assert output.shape == (1, 4, 8)
    assert output.dtype == torch.float32


def test_automatic_flash_runtime_fallback_preserves_attention_mask():
    query = torch.randn(1, 4, 1, 8, dtype=torch.float32)
    text_mask = torch.tensor([[1, 0]], dtype=torch.int64)
    automatic = attention.AttentionParams.create_attention_params_from_mask(
        attention.AUTO_FLASH_ATTENTION_MODE, False, 2, text_mask
    )
    legacy = attention.AttentionParams.create_attention_params_from_mask("torch", False, 2, text_mask)

    automatic_output = attention.attention(query, query.clone(), query.clone(), automatic)
    legacy_output = attention.attention(query, query.clone(), query.clone(), legacy)

    torch.testing.assert_close(automatic_output, legacy_output)


def test_automatic_flash_runtime_routes_compatible_tensors(monkeypatch):
    monkeypatch.setattr(attention, "external_flash_supports_inputs", lambda q, k, v: True)
    monkeypatch.setattr(attention, "flash_attn_func", lambda q, k, v, dropout_p: q + k + v)
    query = torch.ones(1, 4, 1, 8, dtype=torch.bfloat16)
    params = attention.AttentionParams.create_attention_params(attention.AUTO_FLASH_ATTENTION_MODE, False)

    output = attention.attention(query, query.clone(), query.clone(), params)

    torch.testing.assert_close(output, torch.full((1, 4, 8), 3, dtype=torch.bfloat16))


def test_common_parser_accepts_legacy_sdpa_flag():
    args = setup_parser_common().parse_args(["--sdpa", "--use_legacy_sdpa"])

    assert args.sdpa
    assert args.use_legacy_sdpa


def test_hunyuan_parser_accepts_legacy_sdpa_flag():
    args = hv_train.setup_parser().parse_args(["--sdpa", "--use_legacy_sdpa"])

    assert args.sdpa
    assert args.use_legacy_sdpa


def test_every_training_selector_delegates_sdpa_resolution():
    source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
    selector_files = [
        source_root / "training" / "trainer_base.py",
        source_root / "qwen_image_train.py",
        source_root / "zimage_train.py",
        source_root / "hv_train.py",
    ]

    for path in selector_files:
        source = path.read_text(encoding="utf-8")
        assert 'resolve_sdpa_backend(getattr(args, "use_legacy_sdpa", False)' in source, path.name
