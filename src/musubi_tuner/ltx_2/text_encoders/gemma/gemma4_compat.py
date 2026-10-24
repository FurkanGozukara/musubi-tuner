"""Everything the LTX-2.5 Gemma 4 text encoder loader takes from Transformers' internals, in one place.

LTX-2.5 ships its text encoder either as a packed safetensors file (``gemma4-12b-with-proj-ltx-2.5-bf16.safetensors``:
the Gemma 4 weights in ComfyUI-style keys, the LTX projection, the Hugging Face config in the ``gemma_config``
metadata entry and the tokenizer as ``tokenizer_json`` / ``hf_asset__tokenizer_config.json`` uint8 tensors) or as a
Transformers model directory (LTX-2.5 Pre-Trained's ``ltx-2.5-22b-gemma4-12b/``). The model is built from that config
on the meta device and the weights are streamed in, so the loader depends on these Transformers internals (verified on
transformers 5.19.0; AkaneTendo25 wrote the port against 5.14 / 5.15):

* ``transformers.models.auto.configuration_auto.CONFIG_MAPPING[model_type].from_dict`` for the packed config, and
  ``AutoModelForImageTextToText.from_config`` mapping it (Gemma 4 12B: ``gemma4_unified``) to its
  ``*ForConditionalGeneration`` class. Gemma 4 needs a Transformers release that registers the model type; the
  heterogeneous per-layer configs Gemma 4 uses (``per_layer_config``: a wider full-attention head) arrived in 5.15.
* The module tree ``model.language_model.{embed_tokens, layers.<i>, norm, rotary_emb}`` + ``lm_head``, the vision
  branch under ``model.embed_vision`` (a ``multimodal_embedder`` level since 5.14; 5.10 wrote
  ``model.vision_embedder`` without it: ``GEMMA_KEY_ALIASES``) and the audio projector under ``model.embed_audio``.
* Non-persistent buffers no checkpoint stores, rebuilt here after the stream (``initialize_runtime_buffers``):
  the rotary embedding's ``<layer_type>_inv_freq`` / ``<layer_type>_original_inv_freq`` buffers with their
  ``<layer_type>_attention_scaling`` attributes, and ``embed_tokens.embed_scale``. They are rebuilt by constructing
  the installed release's own rotary class from the model's config (whatever its per-layer rules are); only when
  that constructor is unusable are they computed from ``rope_init_fns`` / ``ROPE_INIT_FUNCTIONS`` /
  ``compute_default_rope_parameters`` with ``config.per_layer_config[layer_type]`` (the 5.15+ per-layer view).
  The persistent ``layers.<i>.layer_scalar`` buffer comes from the checkpoint.
* The forward contract the encoder uses: ``output_hidden_states=True`` returns every decoder layer's hidden state
  with ``hidden_states[-1]`` tied to the post-norm ``last_hidden_state`` (``capture_outputs``' default, as for
  Gemma 3), and ``logits_to_keep`` is accepted.
* ``BitsAndBytesConfig`` (``quantization_method()``, ``llm_int8_*``, ``bnb_4bit_*``) for 8/4-bit weight files
  (``quantize_linear_modules``), and the ``is_loaded_in_8bit`` / ``is_loaded_in_4bit`` markers its quantizers set.

``tests/test_ltx25_gemma4_packed_loading.py`` builds a tiny random Gemma 4 config, packs it the LTX-2.5 way and checks
the streamed model against Transformers' own fully initialised one; run it after a Transformers upgrade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

PACKED_CONFIG_METADATA_KEY = "gemma_config"
PACKED_TOKENIZER_KEY = "tokenizer_json"
PACKED_TOKENIZER_CONFIG_KEY = "hf_asset__tokenizer_config.json"

# hf_asset__tokenizer_config.json of the Gemma 4 text encoder Lightricks first published with LTX-2.5. They replaced the
# file on 2026-08-17 ("Aligns the published encoders with the LTX-2.5 model checkpoints": 490 of its 686 tensors changed,
# the text projection by up to 26.7%, with the chat template and this tokenizer config); copies made before, in mirrors
# and downloaders, still carry it
SUPERSEDED_TOKENIZER_CONFIG_SHA256 = frozenset({"794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8"})
ALLOW_SUPERSEDED_ENV = "LTX2_ALLOW_SUPERSEDED_GEMMA4"


@dataclass(frozen=True)
class PackedTokenizerAssets:
    """Tokenizer payload embedded in an LTX-2.5 text-encoder safetensors."""

    tokenizer_json: bytes
    tokenizer_config: dict


def is_gemma3_config(config: object) -> bool:
    """Whether a Gemma config describes the Gemma 3 family (LTX-2 / LTX-2.3)."""
    return str(getattr(config, "model_type", "") or "").startswith("gemma3")


def is_non_weight_key(key: str) -> bool:
    """Whether a packed checkpoint key carries packed assets rather than weights."""
    return key == PACKED_TOKENIZER_KEY or key.startswith("hf_asset__")


def check_packed_gemma_release(path: str, tokenizer_config: bytes) -> None:
    """Refuse the Gemma 4 release Lightricks replaced on 2026-08-17 (its embedded tokenizer config identifies it): its
    embeddings do not match the LTX-2.5 transformer. ``LTX2_ALLOW_SUPERSEDED_GEMMA4=1`` only warns."""
    import hashlib
    import os

    if hashlib.sha256(tokenizer_config).hexdigest() not in SUPERSEDED_TOKENIZER_CONFIG_SHA256:
        return
    message = (
        f"{path} is the Gemma 4 text encoder Lightricks replaced on 2026-08-17: 490 of its 686 tensors differ from the file "
        "that matches the LTX-2.5 transformer. Download text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors again "
        "from Lightricks/LTX-2.5 (or from a mirror updated after 2026-08-17)."
    )
    if os.environ.get(ALLOW_SUPERSEDED_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        logger.warning("%s Continuing because %s is set.", message, ALLOW_SUPERSEDED_ENV)
        return
    raise ValueError(f"{message} Set {ALLOW_SUPERSEDED_ENV}=1 to use it anyway.")


def _uint8_tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.cpu().contiguous().numpy().astype("uint8", copy=False).tobytes()


def config_from_dict(config_dict: dict, *, source: str = "") -> Any:
    """A Transformers config for a packed ``gemma_config`` (``CONFIG_MAPPING[model_type].from_dict``)."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Packed text encoder {source} has no model_type in gemma_config")
    try:
        config_class = CONFIG_MAPPING[model_type]
    except KeyError as exc:
        import transformers

        raise ValueError(
            f"transformers {transformers.__version__} does not know the packed Gemma model_type={model_type!r} of "
            f"{source}; LTX-2.5's Gemma 4 text encoder needs a release with Gemma 4 and per-layer configs (5.15 or newer; "
            "verified on 5.19)"
        ) from exc
    return config_class.from_dict(config_dict)


def read_packed_gemma_assets(path: str) -> tuple[Any | None, PackedTokenizerAssets | None]:
    """The Gemma 4 config and tokenizer packed into an LTX-2.5 text-encoder file, or (None, None) for any other file.

    Plain file reads (no memory map, only the two small tokenizer tensors are loaded): a Gemma 3 weight file passed
    with ``--gemma_safetensors`` is only inspected here.
    """
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    with MemoryEfficientSafeOpen(path, disable_numpy_memmap=True) as handle:
        metadata = handle.metadata() or {}
        raw_config = metadata.get(PACKED_CONFIG_METADATA_KEY)
        keys = set(handle.keys())
        if raw_config is None or PACKED_TOKENIZER_KEY not in keys:
            return None, None
        config = config_from_dict(json.loads(raw_config), source=path)
        tokenizer_json = _uint8_tensor_bytes(handle.get_tensor(PACKED_TOKENIZER_KEY))
        tokenizer_config = {}
        if PACKED_TOKENIZER_CONFIG_KEY in keys:
            raw_tokenizer_config = _uint8_tensor_bytes(handle.get_tensor(PACKED_TOKENIZER_CONFIG_KEY))
            check_packed_gemma_release(path, raw_tokenizer_config)
            tokenizer_config = json.loads(raw_tokenizer_config)
    return config, PackedTokenizerAssets(tokenizer_json, tokenizer_config)


def build_model_from_config(config: Any, dtype: torch.dtype) -> torch.nn.Module:
    """The ``*ForConditionalGeneration`` model of a non-Gemma-3 config on the meta device."""
    from transformers import AutoModelForImageTextToText

    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(config).to(dtype=dtype)


# Gemma 4 exports disagree on the vision-branch names across Transformers releases:
# 5.10 wrote ``model.vision_embedder.*`` and a bare ``embedding_projection``, while
# 5.14 expects ``model.embed_vision.*`` with a ``multimodal_embedder`` level. Checkpoints
# are therefore matched against the live module tree, trying each spelling in turn.
GEMMA_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    ("model.embed_vision.embedding_projection.", "model.embed_vision.multimodal_embedder.embedding_projection."),
    ("model.embed_vision.multimodal_embedder.embedding_projection.", "model.embed_vision.embedding_projection."),
    ("model.vision_embedder.", "model.embed_vision."),
    ("model.embed_vision.", "model.vision_embedder."),
)


def key_alias_candidates(key: str) -> list[str]:
    """Alternative spellings for a checkpoint key, most specific first."""
    candidates = []
    for source, target in GEMMA_KEY_ALIASES:
        if key.startswith(source):
            candidate = target + key[len(source) :]
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _text_config(model: torch.nn.Module):
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return getattr(config, "text_config", config)


def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def _rotary_from_fresh_instance(rotary: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    """Buffers and scaling attributes of a rotary embedding built anew from its config by the installed release."""
    try:
        with torch.device("cpu"):
            fresh = type(rotary)(config=rotary.config)
    except Exception as exc:  # the constructor's signature or rules changed: computed by hand instead
        logger.info("Gemma rotary embedding: rebuilding %s from its config failed (%s)", type(rotary).__name__, exc)
        return None
    buffers = {name: buffer for name, buffer in fresh.named_buffers(recurse=False)}
    scalings = {name: value for name, value in vars(fresh).items() if name.endswith("attention_scaling")}
    if not buffers:
        return None
    return buffers, scalings


def _layer_rope_config(config, layer_type: str):
    per_layer = getattr(config, "per_layer_config", None)
    if per_layer is not None:
        try:
            return per_layer[layer_type]
        except (KeyError, ValueError, TypeError, IndexError):
            pass
    return config


def _rotary_computed(rotary: torch.nn.Module, config) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The rotary buffers computed per layer type (the path AkaneTendo25 wrote for 5.15)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    buffers: dict[str, torch.Tensor] = {}
    scalings: dict[str, Any] = {}
    init_fns = getattr(rotary, "rope_init_fns", None) or {}
    for layer_type in dict.fromkeys(config.layer_types):
        rope_parameters = config.rope_parameters[layer_type]
        if rope_parameters is None:
            continue
        rope_type = rope_parameters["rope_type"]
        init_fn = init_fns.get(layer_type)
        if init_fn is None:
            init_fn = rotary.compute_default_rope_parameters if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = init_fn(_layer_rope_config(config, layer_type), None, layer_type=layer_type)
        buffers[f"{layer_type}_inv_freq"] = inv_freq
        buffers[f"{layer_type}_original_inv_freq"] = inv_freq.clone()
        scalings[f"{layer_type}_attention_scaling"] = attention_scaling
    return buffers, scalings


def initialize_runtime_buffers(model: torch.nn.Module, device: torch.device | str = "cpu") -> list[str]:
    """Materialise the deterministic Gemma 4 buffers no checkpoint stores (rotary frequencies, embedding scale).

    Returns the buffer names it set. The rotary frequencies stay in float32 as ``from_pretrained`` leaves them (the
    forward casts them to float32 anyway), on ``device``.
    """
    device = torch.device(device)
    language_model = _language_model(model)
    config = _text_config(model)
    rotary = getattr(language_model, "rotary_emb", None)
    rebuilt: list[str] = []
    if rotary is not None:
        result = _rotary_from_fresh_instance(rotary)
        if result is None:
            result = _rotary_computed(rotary, config)
        buffers, scalings = result
        for name, buffer in buffers.items():
            rotary.register_buffer(name, buffer.to(device=device), persistent=False)
            rebuilt.append(f"rotary_emb.{name}")
        for name, value in scalings.items():
            setattr(rotary, name, value)

    embed_tokens = getattr(language_model, "embed_tokens", None)
    if embed_tokens is not None and "embed_scale" in getattr(embed_tokens, "_buffers", {}):
        scale = getattr(embed_tokens, "scalar_embed_scale", None)
        if scale is None:
            scale = float(config.hidden_size) ** 0.5
        embed_tokens.register_buffer("embed_scale", torch.tensor(float(scale), device=device), persistent=False)
        rebuilt.append("embed_tokens.embed_scale")
    return rebuilt


def quantize_linear_modules(
    model: torch.nn.Module,
    quantization_config,
    *,
    skip: Iterable[str] = ("lm_head",),
) -> int:
    """Swap every ``nn.Linear`` (except ``skip``) for its bitsandbytes counterpart, carrying the loaded weights over.

    Transformers' own helper builds the replacement modules on the meta device because it runs before the checkpoint
    is read. Weights that are already materialized need the opposite order, so the tensors are handed to the bnb
    parameter types here and quantized when the module is moved to the accelerator (AkaneTendo25 e194f1fb).
    """
    import bitsandbytes as bnb

    skip = tuple(skip)
    eight_bit = quantization_config.quantization_method() == "llm_int8"
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if type(child) is not torch.nn.Linear or any(part in full_name.split(".") for part in skip):
                continue
            if eight_bit:
                replacement = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                    threshold=quantization_config.llm_int8_threshold,
                )
                replacement.weight = bnb.nn.Int8Params(
                    child.weight.data,
                    requires_grad=False,
                    has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                )
            else:
                replacement = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                    compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                    quant_type=quantization_config.bnb_4bit_quant_type,
                )
                replacement.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                    quant_type=quantization_config.bnb_4bit_quant_type,
                )
            if child.bias is not None:
                replacement.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(parent, child_name, replacement)
            replaced += 1
    return replaced
