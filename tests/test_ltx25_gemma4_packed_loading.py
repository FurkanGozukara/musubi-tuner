"""LTX-2.5 Gemma 4 text encoder: the packed safetensors file and a Gemma 4 model directory load into the same model
Transformers builds itself (tiny random gemma4_unified config, CPU). Run this after a Transformers upgrade: it covers
every internal gemma4_compat relies on (config mapping, module tree, rotary / embed_scale buffers, forward contract)."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402
from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: E402

if "gemma4_unified" not in CONFIG_MAPPING:  # pragma: no cover - older Transformers
    pytest.skip("this Transformers release has no gemma4_unified", allow_module_level=True)

from musubi_tuner.ltx_2.text_encoders.gemma import gemma4_compat  # noqa: E402
from musubi_tuner.ltx_2.text_encoders.gemma.encoders import base_encoder  # noqa: E402
from musubi_tuner.ltx_2.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer  # noqa: E402

VOCAB = ["<pad>", "<eos>", "<bos>", "<unk>", "a", "red", "fox", "runs", "in", "the", "snow"]


def _tiny_config(multimodal: bool = False):
    config_class = CONFIG_MAPPING["gemma4_unified"]
    text = dict(
        vocab_size=len(VOCAB),
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=8,
        global_head_dim=32,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    vision = audio = None
    if multimodal:  # the released 12B has both encoder-free branches; tiny ones here
        vision = dict(patch_size=2, pooling_kernel_size=2, mm_embed_dim=32, mm_posemb_size=8, output_proj_dims=32)
        audio = dict(audio_embed_dim=16)
    # round-trip through to_dict / from_dict like the packed file's gemma_config metadata
    return config_class.from_dict(config_class(text_config=text, vision_config=vision, audio_config=audio).to_dict())


def _reference_model(config):
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).to(torch.float32).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
        for layer in model.model.language_model.layers:
            layer.layer_scalar.fill_(0.75)  # a persistent buffer the checkpoint must carry
    return model


def _tokenizer_json() -> bytes:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(VOCAB)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer.to_str().encode("utf-8")


def _bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _write_packed_file(path, config, reference) -> None:
    """The LTX-2.5 packed layout: ComfyUI-style language-model keys, the config in metadata, the tokenizer as tensors."""
    tensors = {}
    for key, value in reference.state_dict().items():
        if not key.startswith("model.language_model."):
            continue  # lm_head is tied to the embeddings and left out, as in the release
        tensors["model." + key[len("model.language_model.") :]] = value.detach().clone().contiguous()
    tensors["tokenizer_json"] = _bytes_tensor(_tokenizer_json())
    tokenizer_config = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}
    tensors["hf_asset__tokenizer_config.json"] = _bytes_tensor(json.dumps(tokenizer_config).encode("utf-8"))
    # the LTX text projection lives in the same file and is not a Gemma tensor
    tensors["text_embedding_projection.video_aggregate_embed.weight"] = torch.zeros(8, 8)
    metadata = {"gemma_config": json.dumps(config.to_dict()), "format": "pt"}
    save_file(tensors, str(path), metadata=metadata)


class _Encoder(base_encoder.GemmaTextEncoderModelBase):
    def __init__(self):
        super().__init__(feature_extractor_linear=torch.nn.Identity(), dtype=torch.float32)


def _load(ops) -> _Encoder:
    module = _Encoder()
    for op in ops:
        if op.matcher(module):
            module = op.mutator(module)
    return module


def _hidden_states(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, logits_to_keep=1)
    return out.hidden_states


def _assert_same_forward(loaded, reference, num_layers):
    input_ids = torch.tensor([[0, 0, 2, 4, 5, 6, 7], [2, 4, 5, 6, 7, 8, 9]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    expected = _hidden_states(reference, input_ids, attention_mask)
    actual = _hidden_states(loaded, input_ids, attention_mask)
    assert len(actual) == len(expected) == num_layers + 1
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert torch.allclose(a, e, atol=1e-5, rtol=1e-5), f"hidden state {index} differs: {(a - e).abs().max()}"


@pytest.mark.parametrize("multimodal", [False, True])
def test_packed_ltx25_text_encoder_loads_like_transformers(tmp_path, multimodal):
    config = _tiny_config(multimodal)
    reference = _reference_model(config)
    packed = tmp_path / "gemma4-tiny-with-proj-ltx-2.5.safetensors"
    _write_packed_file(packed, config, reference)

    packed_config, packed_tokenizer = gemma4_compat.read_packed_gemma_assets(str(packed))
    assert packed_config.model_type == "gemma4_unified"
    assert packed_tokenizer.tokenizer_config["bos_token"] == "<bos>"

    ops = base_encoder.module_ops_from_gemma_root(
        None, gemma_safetensors=str(packed), torch_dtype=torch.float32, device=torch.device("cpu")
    )
    encoder = _load(ops)
    assert type(encoder.model).__name__ == type(reference).__name__
    assert not [name for name, p in encoder.model.named_parameters() if p.device.type == "meta"]
    assert not [name for name, b in encoder.model.named_buffers() if b.device.type == "meta"]
    rotary = encoder.model.model.language_model.rotary_emb
    for name, buffer in reference.model.language_model.rotary_emb.named_buffers():
        assert getattr(rotary, name).dtype == torch.float32
        assert torch.equal(getattr(rotary, name).cpu(), buffer.cpu()), name
    _assert_same_forward(encoder.model, reference, config.text_config.num_hidden_layers)

    # the packed tokenizer puts Gemma's BOS in front, as the official pipeline and ComfyUI do
    assert isinstance(encoder.tokenizer, LTXVGemmaTokenizer) and encoder.tokenizer.ensure_bos
    pairs = encoder.tokenizer.tokenize_with_weights("a red fox")["gemma"]
    active = [token for token, weight in pairs if weight]
    assert [int(t) for t in active] == [2, 4, 5, 6]


def test_the_superseded_gemma4_release_is_refused(tmp_path, monkeypatch):
    """Lightricks replaced the Gemma 4 file on 2026-08-17 (490 of 686 tensors changed); copies made before, which mirrors
    and downloaders still serve, carry the old embedded tokenizer config, and their embeddings do not match the
    LTX-2.5 transformer."""
    import hashlib

    from safetensors import safe_open

    config = _tiny_config()
    packed = tmp_path / "gemma4-tiny-with-proj-ltx-2.5.safetensors"
    _write_packed_file(packed, config, _reference_model(config))
    with safe_open(str(packed), "pt") as handle:
        raw = handle.get_tensor(gemma4_compat.PACKED_TOKENIZER_CONFIG_KEY).numpy().tobytes()
    monkeypatch.setattr(gemma4_compat, "SUPERSEDED_TOKENIZER_CONFIG_SHA256", frozenset({hashlib.sha256(raw).hexdigest()}))
    monkeypatch.delenv(gemma4_compat.ALLOW_SUPERSEDED_ENV, raising=False)
    with pytest.raises(ValueError, match="2026-08-17"):
        gemma4_compat.read_packed_gemma_assets(str(packed))
    monkeypatch.setenv(gemma4_compat.ALLOW_SUPERSEDED_ENV, "1")
    packed_config, _ = gemma4_compat.read_packed_gemma_assets(str(packed))
    assert packed_config.model_type == "gemma4_unified"


def test_low_vram_loads_the_gemma_into_cpu_ram_and_streams_its_decoder_layers(tmp_path):
    """--low_vram_text_encoding: an unquantized Gemma 4 loads into CPU RAM even when the run's device is the GPU (the
    BF16 file held 23.4 GB of VRAM while loading, before the layers moved back), and its decoder layers stream through
    the GPU one at a time with the same hidden states. Needs no GPU: nothing may go to the requested one."""
    config = _tiny_config()
    reference = _reference_model(config)
    packed = tmp_path / "gemma4-tiny-with-proj-ltx-2.5.safetensors"
    _write_packed_file(packed, config, reference)
    ops = base_encoder.module_ops_from_gemma_root(
        None, gemma_safetensors=str(packed), torch_dtype=torch.float32, device=torch.device("cuda"), low_vram=True
    )
    encoder = _load(ops)
    assert {param.device.type for param in encoder.model.parameters()} == {"cpu"}
    managed = base_encoder.enable_low_vram_text_encoding(encoder, "cpu")
    assert managed[:2] == ["Gemma embeddings", f"{config.text_config.num_hidden_layers} Gemma decoder layers"]
    _assert_same_forward(encoder.model, reference, config.text_config.num_hidden_layers)


@pytest.mark.parametrize("multimodal", [False, True])
def test_gemma4_directory_streams_like_transformers(tmp_path, multimodal):
    config = _tiny_config(multimodal)
    reference = _reference_model(config)
    directory = tmp_path / "ltx-2.5-tiny-gemma4"
    reference.save_pretrained(str(directory), safe_serialization=True)
    (directory / "tokenizer.json").write_bytes(_tokenizer_json())
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}),
        encoding="utf-8",
    )
    assert base_encoder.resolve_gemma4_directory(str(directory)) == str(directory)

    ops = base_encoder.module_ops_from_gemma_root(str(directory), torch_dtype=torch.float32, device=torch.device("cpu"))
    encoder = _load(ops)
    _assert_same_forward(encoder.model, reference, config.text_config.num_hidden_layers)
    assert encoder.tokenizer.ensure_bos


def test_runtime_buffers_fall_back_to_per_layer_computation(monkeypatch):
    """When the installed release's rotary class cannot be rebuilt from its config, the per-layer-type computation
    (AkaneTendo25's 5.15 path) gives the same buffers."""
    config = _tiny_config()
    reference = _reference_model(config)
    with torch.device("meta"):
        meta_model = type(reference)(config)
    rotary = meta_model.model.language_model.rotary_emb
    monkeypatch.setattr(gemma4_compat, "_rotary_from_fresh_instance", lambda _rotary: None)
    rebuilt = gemma4_compat.initialize_runtime_buffers(meta_model)
    assert "embed_tokens.embed_scale" in rebuilt
    for name, buffer in reference.model.language_model.rotary_emb.named_buffers():
        assert torch.allclose(getattr(rotary, name), buffer), name
    expected_scale = reference.model.language_model.embed_tokens.embed_scale.float()
    assert torch.allclose(meta_model.model.language_model.embed_tokens.embed_scale.float(), expected_scale)


def test_gemma3_single_file_tokenizer_keeps_encoding_without_bos():
    """LTX-2 / 2.3 single-file Gemma (spiece bytes) keeps its previous tokens: no BOS is added."""
    spm = pytest.importorskip("sentencepiece")
    import io

    corpus = io.BytesIO()
    model_proto = io.BytesIO()
    corpus.write("\n".join(["a red fox runs in the snow"] * 20).encode("utf-8"))
    corpus.seek(0)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus.read().decode("utf-8").splitlines()),
        model_writer=model_proto,
        vocab_size=11,
        hard_vocab_limit=False,
        model_type="word",
        minloglevel=2,
        bos_id=2,
        eos_id=1,
        pad_id=0,
        unk_id=3,
    )
    tokenizer = LTXVGemmaTokenizer(model_proto.getvalue(), max_length=8)
    assert not tokenizer.ensure_bos
    active = [int(t) for t, w in tokenizer.tokenize_with_weights("a red fox")["gemma"] if w]
    assert active and active[0] != 2


def test_unknown_packed_model_type_names_the_requirement():
    with pytest.raises(ValueError, match="does not know the packed Gemma model_type"):
        gemma4_compat.config_from_dict({"model_type": "gemma99_imaginary"}, source="x.safetensors")
