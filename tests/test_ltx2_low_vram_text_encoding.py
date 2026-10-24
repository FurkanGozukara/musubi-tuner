"""Low-VRAM LTX-2 text encoding and the leaner per-token RMS feature extraction.

LTX-2.3 text caching with the single-file FP8 Gemma peaked at 12.8 GB (11.5 after loading the FP8 weights straight
to the CPU): the Gemma embedding table, the text projection and the video connector (1.9 / 2.2 / 3.0 GB) stayed on the
GPU together, and the feature extractor made float32 copies of all 49 hidden-state layers. Low-VRAM text encoding
moves each module to the GPU only while it runs; on an RTX A6000 the peak fell to 4.0 GB with bit-identical caches.
"""

import pytest
import torch

from musubi_tuner.ltx_2.text_encoders.gemma import feature_extractor
from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import enable_low_vram_text_encoding


def _reference_per_token_rms(encoded_text, attention_mask):
    # the previous implementation
    b, _t, d, l = encoded_text.shape  # noqa: E741
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(b, attention_mask.shape[1], d * l)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_per_token_rms_matches_the_previous_implementation(dtype):
    torch.manual_seed(0)
    encoded = torch.randn(2, 7, 16, 5, dtype=dtype)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    out = feature_extractor._norm_and_concat_per_token_rms(encoded, mask)
    assert torch.equal(out, _reference_per_token_rms(encoded, mask))


class _TinyGemma(torch.nn.Module):
    """Embedding table tied to the LM head, like Gemma 3."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 11, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids):
        hidden = self.norm(self.embed(input_ids))
        return hidden, self.lm_head(hidden[:, -1:])


class _DualProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_aggregate_embed = torch.nn.Linear(8, 6)
        self.audio_aggregate_embed = torch.nn.Linear(8, 4)

    def forward(self, hidden):
        audio = self.audio_aggregate_embed(hidden) if self.audio_aggregate_embed is not None else None
        return self.video_aggregate_embed(hidden), audio


class _TinyVideoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyGemma()
        self.feature_extractor_linear = _DualProjection()
        self.embeddings_connector = torch.nn.Linear(6, 6)

    def forward(self, input_ids):
        hidden, _logits = self.model(input_ids)
        video, _audio = self.feature_extractor_linear(hidden)
        return self.embeddings_connector(video)


def test_low_vram_encoding_keeps_outputs_and_drops_the_unused_audio_projection():
    torch.manual_seed(0)
    encoder = _TinyVideoEncoder()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = encoder(input_ids)
    managed = enable_low_vram_text_encoding(encoder, "cpu")
    assert managed == ["Gemma embeddings", "feature_extractor_linear", "embeddings_connector"]
    assert encoder.feature_extractor_linear.audio_aggregate_embed is None  # video-only: output never used
    assert encoder._execution_device == torch.device("cpu")
    assert encoder.model.lm_head.weight is encoder.model.embed.weight  # tied weights stay tied
    assert torch.equal(encoder(input_ids), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_low_vram_encoding_keeps_weights_in_cpu_ram_between_forwards():
    torch.manual_seed(0)
    encoder = _TinyVideoEncoder().cuda()
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    expected = encoder(input_ids)
    enable_low_vram_text_encoding(encoder, "cuda")
    for module in (encoder.model.embed, encoder.feature_extractor_linear, encoder.embeddings_connector):
        assert all(param.device.type == "cpu" for param in module.parameters())
    assert torch.equal(encoder(input_ids), expected)
    assert encoder.model.norm.weight.device.type == "cuda"  # unmanaged Gemma weights stay where they run
    assert all(param.device.type == "cpu" for param in encoder.embeddings_connector.parameters())


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.tensor(0.5), persistent=False)

    def forward(self, hidden):
        return hidden + self.scale * torch.tanh(self.proj(hidden))


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 8)
        self.layers = torch.nn.ModuleList([_TinyLayer() for _ in range(3)])
        self.norm = torch.nn.LayerNorm(8)


class _TinyGemma4(torch.nn.Module):
    """Gemma 3 / 4 as transformers 5 builds them: decoder layers under model.language_model, the embedding table tied
    to the LM head."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = _TinyLanguageModel()
        self.lm_head = torch.nn.Linear(8, 11, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, input_ids):
        language = self.model.language_model
        hidden = language.embed_tokens(input_ids)
        for layer in language.layers:
            hidden = layer(hidden)
        hidden = language.norm(hidden)
        return hidden, self.lm_head(hidden[:, -1:])


class _TinyLayeredVideoEncoder(_TinyVideoEncoder):
    def __init__(self):
        super().__init__()
        self.model = _TinyGemma4()


def test_low_vram_encoding_streams_the_decoder_layers_of_an_unquantized_gemma():
    """LTX-2.5's BF16 Gemma 4 (20.3 GB of decoder layers) on the GPU as a whole peaked at 30.8 GB in text caching:
    an unquantized Gemma keeps its decoder layers in CPU RAM too, each moved to the GPU only while it runs."""
    torch.manual_seed(0)
    encoder = _TinyLayeredVideoEncoder()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = encoder(input_ids)
    managed = enable_low_vram_text_encoding(encoder, "cpu")
    assert managed == ["Gemma embeddings", "3 Gemma decoder layers", "feature_extractor_linear", "embeddings_connector"]
    gemma = encoder.model
    assert gemma.lm_head.weight is gemma.model.language_model.embed_tokens.weight  # tied weights stay tied
    assert torch.equal(encoder(input_ids), expected)


def test_low_vram_encoding_leaves_the_layers_of_a_quantized_gemma_alone():
    """FP8 / bitsandbytes Gemma weights have their own offload; their decoder layers are not streamed."""
    torch.manual_seed(0)
    encoder = _TinyLayeredVideoEncoder()
    encoder._has_fp8_model = True
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = encoder(input_ids)
    managed = enable_low_vram_text_encoding(encoder, "cpu")
    assert managed == ["Gemma embeddings", "feature_extractor_linear", "embeddings_connector"]
    assert torch.equal(encoder(input_ids), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_low_vram_encoding_streams_decoder_layers_through_the_gpu():
    torch.manual_seed(0)
    encoder = _TinyLayeredVideoEncoder().cuda()
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    expected = encoder(input_ids)
    encoder.cpu()  # low-VRAM encoding builds the bundle in CPU RAM
    enable_low_vram_text_encoding(encoder, "cuda")
    language = encoder.model.model.language_model
    for layer in language.layers:
        assert all(param.device.type == "cpu" and param.is_pinned() for param in layer.parameters())
        assert layer.scale.device.type == "cuda"
    assert language.norm.weight.device.type == "cuda"
    assert torch.equal(encoder(input_ids), expected)
    assert all(param.device.type == "cpu" for layer in language.layers for param in layer.parameters())


class _RecordingBuilder:
    """Stands in for SingleGPUModelBuilder: records where the text encoder bundle is built."""

    built_on: list = []

    def __init__(self, **kwargs):
        pass

    def build(self, device, dtype):
        _RecordingBuilder.built_on.append(torch.device(device))
        return _TinyVideoEncoder()


@pytest.mark.parametrize("low_vram", [False, True])
def test_training_sample_prompts_encode_like_low_vram_text_caching(monkeypatch, low_vram):
    """--low_vram_text_encoding in training: the sample prompts' text encoder is built in CPU RAM and streamed per
    module, as in text caching (the 10 / 12 GB LTX-2.3 LoRA presets drew their samples at 11.6 GB without it)."""
    from types import SimpleNamespace

    from musubi_tuner import ltx2_sampling
    from musubi_tuner.ltx_2.loader import single_gpu_model_builder, sft_loader
    from musubi_tuner.ltx_2.text_encoders.gemma.encoders import base_encoder

    _RecordingBuilder.built_on = []
    streamed = []
    monkeypatch.setattr(single_gpu_model_builder, "SingleGPUModelBuilder", _RecordingBuilder)
    monkeypatch.setattr(sft_loader, "in_process_checkpoint_loader", lambda: None)
    monkeypatch.setattr(base_encoder, "module_ops_from_gemma_root", lambda *args, **kwargs: ())
    monkeypatch.setattr(base_encoder, "apply_text_encoder_checkpoint_overrides", lambda *args: None)
    monkeypatch.setattr(base_encoder, "enable_low_vram_text_encoding", lambda encoder, device: streamed.append(torch.device(device)) or ["Gemma embeddings"])

    class _Sampler(ltx2_sampling.LTX2SamplingMixin):
        _audio_video = False

        def __init__(self):
            self._text_encoder = None

    sampler = _Sampler()
    sampler._text_encoder = None
    args = SimpleNamespace(gemma_safetensors="gemma_fp8.safetensors", gemma_root=None, ltx2_checkpoint="ltx.safetensors", low_vram_text_encoding=low_vram)
    accelerator = SimpleNamespace(device=torch.device("cuda"), mixed_precision="bf16")  # read as CUDA, never used as one
    monkeypatch.setattr(_TinyVideoEncoder, "to", lambda self, device: self)  # the plain path moves the bundle to the GPU
    assert sampler._build_text_encoder(args, accelerator) == torch.bfloat16
    assert _RecordingBuilder.built_on == [torch.device("cpu") if low_vram else torch.device("cuda")]
    assert streamed == ([torch.device("cuda")] if low_vram else [])
