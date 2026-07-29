"""Preservation & regularization techniques for LTX-2 LoRA training.

Implements three optional techniques (from ai-toolkit via diffusion-pipe):
1. Blank Prompt Preservation  -- regularise LoRA to not change blank-prompt output
2. Differential Output Preservation (DOP) -- regularise LoRA to not change class-prompt output
3. Prior Divergence -- encourage LoRA output to diverge from base model on training prompts
"""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file

from musubi_tuner.ltx2_text_conditioning import select_video_text_embeds_for_video_mode
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arg parsing helper (same format as --optimizer_args: key=value pairs)
# ---------------------------------------------------------------------------

DEFAULT_PRESERVATION_CACHE = "ltx2_preservation_cache.pt"
DOP_CACHE_VERSION = 2
DOP_REWRITE_VERSION = 1


def parse_preservation_args(raw_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict.  Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    result: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _split_prompt_variants(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_dop_replacements(value: str) -> dict[str, list[str]]:
    """Parse ``trigger=>class1|class2;trigger2=>class`` mappings."""
    replacements: dict[str, list[str]] = {}
    if not str(value).strip():
        return replacements
    for raw_mapping in str(value).split(";"):
        raw_mapping = raw_mapping.strip()
        if not raw_mapping:
            continue
        separator = "=>" if "=>" in raw_mapping else ":"
        if separator not in raw_mapping:
            raise ValueError(f"Invalid DOP replacement {raw_mapping!r}; expected trigger=>class or trigger=>class1|class2")
        trigger, classes = raw_mapping.split(separator, 1)
        trigger = trigger.strip()
        variants = _split_prompt_variants(classes)
        if not trigger or not variants:
            raise ValueError(f"Invalid DOP replacement {raw_mapping!r}; trigger and class must be non-empty")
        replacements[trigger] = variants
    return replacements


def _replace_trigger(text: str, trigger: str, replacement: str, known_classes: Sequence[str]) -> str:
    pattern = rf"(?<!\w){re.escape(trigger)}(?!\w)"
    for known_class in sorted(known_classes, key=len, reverse=True):
        combined = rf"{pattern}\s+(?<!\w){re.escape(known_class)}(?!\w)"
        text = re.sub(combined, replacement, text)
    rewritten = re.sub(pattern, replacement, text)
    return " ".join(rewritten.split())


def build_dop_prompt_variants(
    caption: str,
    *,
    mode: str,
    class_prompts: Sequence[str],
    replacements: Mapping[str, Sequence[str]],
    prompt_bank: Sequence[str],
) -> list[str]:
    """Build stable, deduplicated preservation prompts for one caption."""
    prompts: list[str] = []
    if mode == "fixed":
        prompts.extend(class_prompts)
    elif mode == "caption_replace":
        variant_count = max((len(values) for values in replacements.values()), default=1)
        for variant_index in range(variant_count):
            rewritten = str(caption)
            for trigger, values in replacements.items():
                if not values:
                    continue
                rewritten = _replace_trigger(
                    rewritten,
                    trigger,
                    values[variant_index % len(values)],
                    values,
                )
            prompts.append(rewritten)
    else:
        raise ValueError(f"Unsupported DOP mode: {mode!r}")
    prompts.extend(prompt_bank)
    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        prompt = " ".join(str(prompt).split())
        if prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    return unique


def dop_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def dop_prompt_config_hash(
    *,
    mode: str,
    class_prompts: Sequence[str],
    replacements: Mapping[str, Sequence[str]],
    prompt_bank: Sequence[str],
) -> str:
    payload = {
        "rewrite_version": DOP_REWRITE_VERSION,
        "mode": mode,
        "class_prompts": list(class_prompts),
        "replacements": {key: list(values) for key, values in sorted(replacements.items())},
        "prompt_bank": list(prompt_bank),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_text_encoder_identity(args: argparse.Namespace) -> str:
    """Build a portable identity from the configured encoder/checkpoint files."""

    def _sampled_file_hash(path: str, size: int) -> str:
        digest = hashlib.sha256()
        chunk_size = 1024 * 1024
        with open(path, "rb") as handle:
            offsets = [0]
            if size > chunk_size:
                offsets.append(max(0, size // 2 - chunk_size // 2))
                offsets.append(max(0, size - chunk_size))
            for offset in sorted(set(offsets)):
                handle.seek(offset)
                digest.update(offset.to_bytes(8, "little", signed=False))
                digest.update(handle.read(chunk_size))
        return digest.hexdigest()

    def _path_identity(value: Any) -> dict[str, Any] | None:
        if not value:
            return None
        path = os.path.abspath(os.fspath(value))
        identity: dict[str, Any] = {"name": os.path.basename(path)}
        try:
            stat = os.stat(path)
            is_directory = os.path.isdir(path)
            identity["is_dir"] = is_directory
            if not is_directory:
                identity["size"] = int(stat.st_size)
                identity["sampled_sha256"] = _sampled_file_hash(path, int(stat.st_size))
        except OSError:
            identity["missing"] = True
        if os.path.isdir(path):
            relevant_files: dict[str, dict[str, Any]] = {}
            metadata_names = {
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            }
            for root, _dirs, filenames in os.walk(path):
                for filename in sorted(filenames):
                    if filename not in metadata_names and not filename.endswith(".safetensors") and not filename.endswith(".bin"):
                        continue
                    child = os.path.join(root, filename)
                    relative_path = os.path.relpath(child, path).replace("\\", "/")
                    try:
                        child_stat = os.stat(child)
                    except OSError:
                        continue
                    relevant_files[relative_path] = {
                        "size": int(child_stat.st_size),
                        "sampled_sha256": _sampled_file_hash(child, int(child_stat.st_size)),
                    }
            identity["files"] = relevant_files
        return identity

    payload = {
        "ltx2_checkpoint": _path_identity(getattr(args, "ltx2_checkpoint", None)),
        "ltx2_text_encoder_checkpoint": _path_identity(getattr(args, "ltx2_text_encoder_checkpoint", None)),
        "gemma_root": _path_identity(getattr(args, "gemma_root", None)),
        "gemma_safetensors": _path_identity(getattr(args, "gemma_safetensors", None)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolve_default_preservation_cache(args: argparse.Namespace) -> str:
    """Resolve default cache path from dataset config (same pattern as sample prompts)."""
    from musubi_tuner.dataset import config_utils
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

    if not getattr(args, "dataset_config", None):
        raise ValueError("--dataset_config is required to resolve the preservation cache directory")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    if not datasets:
        raise ValueError("No datasets available to resolve preservation cache directory")
    cache_dir = getattr(datasets[0], "cache_directory", None)
    if not cache_dir:
        raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
    return os.path.join(cache_dir, DEFAULT_PRESERVATION_CACHE)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreservationConfig:
    blank_preservation: bool = False
    blank_multiplier: float = 1.0
    blank_embed: Optional[torch.Tensor] = field(default=None, repr=False)
    blank_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    dop: bool = False
    dop_multiplier: float = 1.0
    dop_class_prompt: str = ""
    dop_mode: str = "fixed"
    dop_class_prompts: list[str] = field(default_factory=list)
    dop_replacements: dict[str, list[str]] = field(default_factory=dict)
    dop_prompt_bank_prompts: list[str] = field(default_factory=list)
    dop_prompt_config_hash: str = ""
    dop_text_encoder_identity: str = ""
    dop_embed: Optional[torch.Tensor] = field(default=None, repr=False)
    dop_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    dop_prompt_bank: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    dop_caption_index: dict[str, list[str]] = field(default_factory=dict, repr=False)
    dop_cache_base_dir: str = field(default="", repr=False)
    dop_strict_cache: bool = True
    dop_loss_type: str = "mse"
    dop_huber_delta: float = 1.0
    dop_relative_eps: float = 1e-6
    dop_temporal_weight: float = 0.0
    dop_inside_weight: float = 1.0
    dop_outside_weight: float = 1.0
    dop_timestep_bins: int = 0
    dop_timestep_min: float = 0.0
    dop_timestep_max: float = 1.0
    dop_timestep_weight: str = "none"
    dop_adaptive_target: float = 0.0
    dop_adaptive_ema_decay: float = 0.95
    dop_adaptive_rate: float = 0.1
    dop_adaptive_min: float = 0.0
    dop_adaptive_max: float = 100.0
    dop_adaptive_warmup: int = 0
    dop_microbatch: int = 0
    dop_anchor_size: int = 0
    dop_anchor_interval: int = 0
    dop_anchor_weight: float = 1.0
    dop_anchor_path: str = ""

    prior_divergence: bool = False
    prior_divergence_multiplier: float = 0.1

    audio_dop: bool = False
    audio_dop_multiplier: float = 1.0

    @property
    def any_active(self) -> bool:
        return self.blank_preservation or self.dop or self.prior_divergence or self.audio_dop

    @property
    def needs_text_encoding(self) -> bool:
        return self.blank_preservation or self.dop


def configure_dop(cfg: PreservationConfig, options: Mapping[str, str]) -> PreservationConfig:
    """Apply and validate DOP key/value options shared by caching and training."""
    aliases = {
        "class_prompt": "class",
        "loss_type": "loss",
        "ema_decay": "adaptive_ema",
        "anchor_replay_interval": "anchor_interval",
        "anchor_bank_size": "anchor_size",
    }
    normalized = {aliases.get(key, key): value for key, value in options.items()}
    known = {
        "mode",
        "class",
        "trigger",
        "replace",
        "bank",
        "multiplier",
        "loss",
        "huber_delta",
        "relative_eps",
        "temporal_weight",
        "inside_weight",
        "outside_weight",
        "timestep_bins",
        "timestep_min",
        "timestep_max",
        "timestep_weight",
        "adaptive_target",
        "adaptive_ema",
        "adaptive_rate",
        "adaptive_min",
        "adaptive_max",
        "adaptive_warmup",
        "microbatch",
        "anchor_size",
        "anchor_interval",
        "anchor_weight",
        "anchor_path",
        "strict_cache",
    }
    unknown = sorted(set(normalized) - known)
    if unknown:
        raise ValueError(f"Unknown DOP arguments: {', '.join(unknown)}")

    cfg.dop_mode = normalized.get("mode", cfg.dop_mode).strip().lower()
    if cfg.dop_mode not in {"fixed", "caption_replace"}:
        raise ValueError("DOP mode must be 'fixed' or 'caption_replace'")
    class_value = normalized.get("class", cfg.dop_class_prompt)
    cfg.dop_class_prompts = _split_prompt_variants(class_value)
    cfg.dop_class_prompt = cfg.dop_class_prompts[0] if cfg.dop_class_prompts else ""
    cfg.dop_prompt_bank_prompts = [prompt.strip() for prompt in normalized.get("bank", "").split(";") if prompt.strip()]
    replacements = parse_dop_replacements(normalized.get("replace", ""))
    trigger = normalized.get("trigger", "").strip()
    if trigger:
        if not cfg.dop_class_prompts:
            raise ValueError("DOP trigger replacement requires class=<prompt>")
        if replacements:
            raise ValueError("Use either trigger=<token> with class=<prompt> or replace=<mappings>, not both")
        replacements = {trigger: list(cfg.dop_class_prompts)}
    cfg.dop_replacements = replacements
    if cfg.dop_mode == "caption_replace" and not cfg.dop_replacements:
        raise ValueError("DOP caption_replace mode requires trigger=<token> or replace=<mappings>")
    if cfg.dop_mode == "fixed" and not cfg.dop_class_prompts and not cfg.dop_prompt_bank_prompts:
        cfg.dop_class_prompts = [""]

    float_fields = {
        "multiplier": "dop_multiplier",
        "huber_delta": "dop_huber_delta",
        "relative_eps": "dop_relative_eps",
        "temporal_weight": "dop_temporal_weight",
        "inside_weight": "dop_inside_weight",
        "outside_weight": "dop_outside_weight",
        "timestep_min": "dop_timestep_min",
        "timestep_max": "dop_timestep_max",
        "adaptive_target": "dop_adaptive_target",
        "adaptive_ema": "dop_adaptive_ema_decay",
        "adaptive_rate": "dop_adaptive_rate",
        "adaptive_min": "dop_adaptive_min",
        "adaptive_max": "dop_adaptive_max",
        "anchor_weight": "dop_anchor_weight",
    }
    int_fields = {
        "timestep_bins": "dop_timestep_bins",
        "adaptive_warmup": "dop_adaptive_warmup",
        "microbatch": "dop_microbatch",
        "anchor_size": "dop_anchor_size",
        "anchor_interval": "dop_anchor_interval",
    }
    for key, attr in float_fields.items():
        if key in normalized:
            setattr(cfg, attr, float(normalized[key]))
    for key, attr in int_fields.items():
        if key in normalized:
            setattr(cfg, attr, int(normalized[key]))
    cfg.dop_loss_type = normalized.get("loss", cfg.dop_loss_type).strip().lower()
    cfg.dop_timestep_weight = normalized.get("timestep_weight", cfg.dop_timestep_weight).strip().lower()
    cfg.dop_anchor_path = normalized.get("anchor_path", cfg.dop_anchor_path).strip()
    if "strict_cache" in normalized:
        cfg.dop_strict_cache = _parse_bool(normalized["strict_cache"], name="DOP strict_cache")

    if cfg.dop_loss_type not in {"mse", "relative_mse", "relative_huber"}:
        raise ValueError("DOP loss must be mse, relative_mse, or relative_huber")
    if cfg.dop_timestep_weight not in {"none", "snr", "inverse_snr", "mid"}:
        raise ValueError("DOP timestep_weight must be none, snr, inverse_snr, or mid")
    finite_values = {
        "multiplier": cfg.dop_multiplier,
        "huber_delta": cfg.dop_huber_delta,
        "relative_eps": cfg.dop_relative_eps,
        "temporal_weight": cfg.dop_temporal_weight,
        "inside_weight": cfg.dop_inside_weight,
        "outside_weight": cfg.dop_outside_weight,
        "timestep_min": cfg.dop_timestep_min,
        "timestep_max": cfg.dop_timestep_max,
        "adaptive_target": cfg.dop_adaptive_target,
        "adaptive_ema": cfg.dop_adaptive_ema_decay,
        "adaptive_rate": cfg.dop_adaptive_rate,
        "adaptive_min": cfg.dop_adaptive_min,
        "adaptive_max": cfg.dop_adaptive_max,
        "anchor_weight": cfg.dop_anchor_weight,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"DOP {name} must be finite")
    if cfg.dop_multiplier < 0 or cfg.dop_temporal_weight < 0:
        raise ValueError("DOP multiplier and temporal_weight must be non-negative")
    if cfg.dop_adaptive_target < 0 or cfg.dop_adaptive_rate < 0 or cfg.dop_anchor_weight < 0:
        raise ValueError("DOP adaptive target/rate and anchor_weight must be non-negative")
    if cfg.dop_huber_delta <= 0 or cfg.dop_relative_eps <= 0:
        raise ValueError("DOP huber_delta and relative_eps must be greater than zero")
    if cfg.dop_inside_weight < 0 or cfg.dop_outside_weight < 0:
        raise ValueError("DOP spatial weights must be non-negative")
    if not 0.0 <= cfg.dop_timestep_min <= cfg.dop_timestep_max <= 1.0:
        raise ValueError("DOP timestep range must satisfy 0 <= timestep_min <= timestep_max <= 1")
    if cfg.dop_timestep_bins < 0 or cfg.dop_microbatch < 0:
        raise ValueError("DOP timestep_bins and microbatch must be non-negative")
    if not 0.0 <= cfg.dop_adaptive_ema_decay < 1.0:
        raise ValueError("DOP adaptive_ema must be in [0, 1)")
    if cfg.dop_adaptive_min < 0 or cfg.dop_adaptive_max < cfg.dop_adaptive_min:
        raise ValueError("DOP adaptive multiplier bounds are invalid")
    if cfg.dop_adaptive_target > 0.0 and cfg.dop_multiplier <= 0.0:
        raise ValueError("DOP adaptive control requires multiplier > 0")
    if cfg.dop_adaptive_warmup < 0 or cfg.dop_anchor_size < 0 or cfg.dop_anchor_interval < 0:
        raise ValueError("DOP warmup and anchor controls must be non-negative")
    if cfg.dop_anchor_interval > 0 and cfg.dop_anchor_size == 0:
        raise ValueError("DOP anchor_interval requires anchor_size > 0")
    if cfg.dop_anchor_path and cfg.dop_anchor_size == 0:
        raise ValueError("DOP anchor_path requires anchor_size > 0")

    cfg.dop_prompt_config_hash = dop_prompt_config_hash(
        mode=cfg.dop_mode,
        class_prompts=cfg.dop_class_prompts,
        replacements=cfg.dop_replacements,
        prompt_bank=cfg.dop_prompt_bank_prompts,
    )
    return cfg


def _expand_spatial_weight(
    pred: torch.Tensor,
    video_loss_mask: Optional[torch.Tensor],
    *,
    model_input_shape: Optional[Sequence[int]],
    inside_weight: float,
    outside_weight: float,
) -> Optional[torch.Tensor]:
    if video_loss_mask is None or inside_weight == outside_weight:
        return None
    mask = video_loss_mask.to(device=pred.device, dtype=torch.float32).clamp(0.0, 1.0)
    if pred.dim() == 3:
        batch, seq_len, _hidden = pred.shape
        if mask.dim() == 5:
            mask = mask.mean(dim=1).reshape(mask.shape[0], -1)
        elif mask.dim() == 4:
            mask = mask.reshape(mask.shape[0], -1)
        elif mask.dim() == 2 and mask.shape[1] != seq_len and model_input_shape is not None:
            frames = int(model_input_shape[2])
            if mask.shape[1] == frames and seq_len % max(frames, 1) == 0:
                mask = mask.repeat_interleave(seq_len // frames, dim=1)
        if mask.dim() != 2 or mask.shape != (batch, seq_len):
            logger.warning(
                "DOP spatial mask shape %s cannot be aligned to prediction %s; using uniform weighting.",
                tuple(mask.shape),
                tuple(pred.shape),
            )
            return None
        mask = mask.unsqueeze(-1)
    elif pred.dim() == 5:
        if mask.dim() == 2:
            mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
        elif mask.dim() == 4:
            mask = mask.unsqueeze(1)
        if mask.dim() != 5:
            return None
        if tuple(mask.shape[2:]) != tuple(pred.shape[2:]):
            mask = F.interpolate(mask, size=pred.shape[2:], mode="trilinear", align_corners=False)
    else:
        return None
    return outside_weight + (inside_weight - outside_weight) * mask


def _weighted_per_sample_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    reduce_dims = tuple(range(1, values.dim()))
    if weights is None:
        return values.mean(dim=reduce_dims)
    weights = weights.to(device=values.device, dtype=values.dtype)
    weighted = values * weights
    denominator = weights.expand_as(values).sum(dim=reduce_dims).clamp_min(1e-12)
    return weighted.sum(dim=reduce_dims) / denominator


def _timestep_weights(sigmas: Optional[torch.Tensor], mode: str, eps: float) -> Optional[torch.Tensor]:
    if sigmas is None or mode == "none":
        return None
    sigma = sigmas.float()
    if sigma.dim() > 1:
        sigma = sigma.reshape(sigma.shape[0], -1).mean(dim=1)
    sigma = sigma.clamp(0.0, 1.0)
    if mode == "snr":
        weight = (1.0 - sigma).square() / (sigma.square() + eps)
    elif mode == "inverse_snr":
        weight = sigma.square() / ((1.0 - sigma).square() + eps)
    elif mode == "mid":
        weight = 4.0 * sigma * (1.0 - sigma)
    else:
        raise ValueError(f"Unsupported DOP timestep weight: {mode}")
    weight = weight.clamp(max=100.0)
    return weight / weight.mean().clamp_min(eps)


def _temporal_view(tensor: torch.Tensor, model_input_shape: Optional[Sequence[int]]) -> Optional[torch.Tensor]:
    if tensor.dim() == 5:
        if tensor.shape[2] < 2:
            return None
        return tensor[:, :, 1:] - tensor[:, :, :-1]
    if tensor.dim() != 3 or model_input_shape is None:
        return None
    frames = int(model_input_shape[2])
    if frames < 2 or tensor.shape[1] % frames != 0:
        return None
    tokens_per_frame = tensor.shape[1] // frames
    viewed = tensor.reshape(tensor.shape[0], frames, tokens_per_frame, tensor.shape[2])
    return viewed[:, 1:] - viewed[:, :-1]


def compute_dop_loss(
    pred: torch.Tensor,
    prior: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
    relative_eps: float,
    spatial_weight: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    timestep_weight: str = "none",
    temporal_weight: float = 0.0,
    model_input_shape: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unscaled total loss, relative output drift, and temporal component."""
    pred_f = pred.float()
    prior_f = prior.float()
    error = pred_f - prior_f
    squared = error.square()
    mse_per_sample = _weighted_per_sample_mean(squared, spatial_weight)
    energy_per_sample = _weighted_per_sample_mean(prior_f.square(), spatial_weight).clamp_min(relative_eps)
    relative_per_sample = mse_per_sample / energy_per_sample
    if loss_type == "mse":
        primary_per_sample = mse_per_sample
    elif loss_type == "relative_mse":
        primary_per_sample = relative_per_sample
    elif loss_type == "relative_huber":
        huber = F.huber_loss(pred_f, prior_f, reduction="none", delta=huber_delta)
        primary_per_sample = _weighted_per_sample_mean(huber, spatial_weight) / energy_per_sample
    else:
        raise ValueError(f"Unsupported DOP loss type: {loss_type}")

    sample_weights = _timestep_weights(sigmas, timestep_weight, relative_eps)
    if sample_weights is not None:
        primary = (primary_per_sample * sample_weights).mean()
        relative_drift = (relative_per_sample * sample_weights).mean()
    else:
        primary = primary_per_sample.mean()
        relative_drift = relative_per_sample.mean()

    temporal = primary.new_zeros(())
    if temporal_weight > 0.0:
        pred_delta = _temporal_view(pred_f, model_input_shape)
        prior_delta = _temporal_view(prior_f, model_input_shape)
        if pred_delta is not None and prior_delta is not None:
            delta_mse = (pred_delta - prior_delta).square().flatten(1).mean(dim=1)
            delta_energy = prior_delta.square().flatten(1).mean(dim=1).clamp_min(relative_eps)
            if loss_type == "mse":
                temporal_per_sample = delta_mse
            elif loss_type == "relative_mse":
                temporal_per_sample = delta_mse / delta_energy
            else:
                temporal_huber = F.huber_loss(pred_delta, prior_delta, reduction="none", delta=huber_delta)
                temporal_per_sample = temporal_huber.flatten(1).mean(dim=1) / delta_energy
            if sample_weights is not None:
                temporal = (temporal_per_sample * sample_weights).mean()
            else:
                temporal = temporal_per_sample.mean()
            primary = primary + temporal_weight * temporal
    return primary, relative_drift.detach(), temporal.detach()


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------


class PreservationHelper:
    def __init__(self, config: PreservationConfig) -> None:
        self.config = config
        self._step = 0
        self._timestep_cursor = 0
        self._dop_drift_ema: Optional[float] = None
        self._dop_effective_multiplier = max(
            config.dop_adaptive_min,
            min(config.dop_adaptive_max, float(config.dop_multiplier)),
        )
        self._anchor_bank: list[dict[str, Any]] = []
        self._anchor_cursor = 0
        self._warned_anchor_unsupported = False
        self._prompt_tensor_cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._prompt_tensor_cache_limit = 64
        if config.dop_anchor_path and os.path.isfile(config.dop_anchor_path):
            self._load_anchor_bank(config.dop_anchor_path)

    # -- block swap compatibility ----------------------------------------

    @staticmethod
    def _prepare_block_swap(transformer: torch.nn.Module, accelerator: Accelerator) -> None:
        """Reset block-swap device placement before an extra forward pass.

        When ``--blocks_to_swap`` is active the offloader tracks which
        transformer blocks sit on GPU vs CPU.  After the main fwd/bwd the
        placement may be stale, so we must reset it before each preservation
        forward to avoid device-mismatch errors.
        """
        unwrapped = accelerator.unwrap_model(transformer)
        if hasattr(unwrapped, "prepare_block_swap_before_forward"):
            # Suppress verbose offloader logs during preservation forwards
            offload_logger = logging.getLogger("musubi_tuner.ltx_2.model.transformer.offloading_utils")
            prev_level = offload_logger.level
            offload_logger.setLevel(logging.WARNING)
            try:
                unwrapped.prepare_block_swap_before_forward()
            finally:
                offload_logger.setLevel(prev_level)

    # -- encode prompts --------------------------------------------------

    def encode_prompts(
        self,
        trainer: Any,  # LTX2NetworkTrainer
        args: argparse.Namespace,
        accelerator: Accelerator,
    ) -> None:
        """Load embeddings from cache or Gemma.  No-op if nothing to encode."""
        cfg = self.config
        if not cfg.needs_text_encoding:
            return
        cfg.dop_text_encoder_identity = build_text_encoder_identity(args)

        # Try loading from precached file first
        if getattr(args, "use_precached_preservation", False):
            cache_path = getattr(args, "preservation_prompts_cache", None)
            if not cache_path:
                cache_path = _resolve_default_preservation_cache(args)
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"Precached preservation embeddings not found: {cache_path}\n"
                    "Run ltx2_cache_text_encoder_outputs.py with --precache_preservation_prompts first."
                )
            self._load_from_cache(cache_path)
            return

        if cfg.dop and cfg.dop_mode == "caption_replace":
            raise ValueError(
                "DOP mode=caption_replace requires --use_precached_preservation. "
                "Run ltx2_cache_text_encoder_outputs.py with matching --dop_args first."
            )

        # Fall back to loading Gemma and encoding live
        text_encoder_dtype = trainer._build_text_encoder(args, accelerator)

        # In AV mode, _encode_prompt_text returns concatenated video+audio embeddings.
        # Preservation only regularises the video branch.
        av_mode = getattr(trainer, "_audio_video", False)
        expected_video_dim = 0
        expected_audio_dim = 0
        if av_mode and hasattr(trainer, "_load_ltx2_checkpoint_config"):
            try:
                checkpoint_cfg = trainer._load_ltx2_checkpoint_config(args)
                transformer_cfg = checkpoint_cfg.get("transformer", {}) if isinstance(checkpoint_cfg, dict) else {}
                expected_video_dim = int(transformer_cfg.get("cross_attention_dim", 0) or 0)
                expected_audio_dim = int(transformer_cfg.get("audio_cross_attention_dim", 0) or 0)
            except Exception:
                logger.warning("Preservation: failed to read checkpoint dims; falling back to legacy embed splitting.")

        if cfg.blank_preservation:
            embed, mask = trainer._encode_prompt_text(accelerator, "", text_encoder_dtype)
            if av_mode:
                embed = select_video_text_embeds_for_video_mode(
                    embed,
                    expected_video_dim=expected_video_dim,
                    expected_audio_dim=expected_audio_dim,
                )
            cfg.blank_embed = embed
            cfg.blank_mask = mask
            logger.info("Preservation: encoded blank prompt  embed=%s (av_mode=%s)", tuple(embed.shape), av_mode)

        if cfg.dop:
            embeds: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            prompts = build_dop_prompt_variants(
                "",
                mode=cfg.dop_mode,
                class_prompts=cfg.dop_class_prompts,
                replacements=cfg.dop_replacements,
                prompt_bank=cfg.dop_prompt_bank_prompts,
            )
            for prompt in prompts:
                embed, mask = trainer._encode_prompt_text(accelerator, prompt, text_encoder_dtype)
                if av_mode:
                    embed = select_video_text_embeds_for_video_mode(
                        embed,
                        expected_video_dim=expected_video_dim,
                        expected_audio_dim=expected_audio_dim,
                    )
                embeds.append(embed)
                masks.append(mask)
            cfg.dop_embed = torch.stack(embeds)
            cfg.dop_mask = torch.stack(masks)
            logger.info(
                "Preservation: encoded %d fixed DOP prompts  embed=%s (av_mode=%s)",
                len(prompts),
                tuple(cfg.dop_embed.shape),
                av_mode,
            )

        # unload text encoder
        trainer._text_encoder = None
        clean_memory_on_device(accelerator.device)
        gc.collect()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Preservation: text encoder unloaded")

    def _load_from_cache(self, cache_path: str) -> None:
        """Load preservation embeddings from a precached .pt file."""
        cfg = self.config
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        version = int(payload.get("version", 1))
        logger.info("Preservation: loading precached embeddings from %s (version=%s)", cache_path, version)

        if cfg.blank_preservation:
            if "blank_embed" in payload and "blank_mask" in payload:
                cfg.blank_embed = payload["blank_embed"]
                cfg.blank_mask = payload["blank_mask"]
                logger.info("Preservation: loaded blank prompt  embed=%s", tuple(cfg.blank_embed.shape))
            else:
                raise ValueError("Preservation cache is missing blank embeddings required by --blank_preservation")

        if cfg.dop:
            if version >= DOP_CACHE_VERSION:
                cached_hash = str(payload.get("dop_prompt_config_hash", ""))
                if cached_hash != cfg.dop_prompt_config_hash:
                    raise ValueError(
                        "DOP cache configuration mismatch. Re-run preservation caching with exactly the same "
                        "mode, class, trigger/replacement mappings, and prompt bank."
                    )
                cached_identity = str(payload.get("dop_text_encoder_identity", ""))
                if cfg.dop_strict_cache and cached_identity != cfg.dop_text_encoder_identity:
                    raise ValueError(
                        "DOP cache text-encoder identity mismatch. Re-run preservation caching with the checkpoint "
                        "and Gemma/tokenizer configuration used for this training run, or set strict_cache=false."
                    )
                prompt_bank = payload.get("dop_prompt_bank")
                if not isinstance(prompt_bank, dict) or not prompt_bank:
                    raise ValueError("DOP cache is missing the deduplicated prompt bank")
                cfg.dop_prompt_bank = prompt_bank
                cfg.dop_cache_base_dir = os.path.dirname(os.path.abspath(cache_path))
                caption_index = payload.get("dop_caption_index", {})
                if cfg.dop_mode == "caption_replace" and not isinstance(caption_index, dict):
                    raise ValueError("Contextual DOP cache is missing its source-caption index")
                cfg.dop_caption_index = caption_index
                cached_dimensions = payload.get("dop_cache_dimensions")
                first_prompt_hash = next(iter(prompt_bank))
                first_entry = self._load_prompt_entry(first_prompt_hash, prompt_bank[first_prompt_hash])
                actual_dimensions = {
                    "embed": list(first_entry["embed"].shape),
                    "mask": list(first_entry["mask"].shape),
                }
                if cached_dimensions and cached_dimensions != actual_dimensions:
                    raise ValueError("DOP cache dimension metadata does not match its tensors")
                if cfg.dop_mode == "fixed":
                    prompts = build_dop_prompt_variants(
                        "",
                        mode=cfg.dop_mode,
                        class_prompts=cfg.dop_class_prompts,
                        replacements=cfg.dop_replacements,
                        prompt_bank=cfg.dop_prompt_bank_prompts,
                    )
                    entries = [self._lookup_prompt_entry(prompt) for prompt in prompts]
                    cfg.dop_embed = torch.stack([entry["embed"] for entry in entries])
                    cfg.dop_mask = torch.stack([entry["mask"] for entry in entries])
                logger.info("Preservation: loaded %d deduplicated DOP prompts", len(cfg.dop_prompt_bank))
            elif cfg.dop_mode != "fixed":
                raise ValueError("Legacy preservation caches cannot be used with DOP mode=caption_replace")
            elif "dop_embed" in payload and "dop_mask" in payload:
                cached_class = str(payload.get("dop_class_prompt", ""))
                if cached_class != cfg.dop_class_prompt:
                    raise ValueError(
                        f"DOP class prompt mismatch: cache has {cached_class!r}, training uses {cfg.dop_class_prompt!r}"
                    )
                cfg.dop_embed = payload["dop_embed"].unsqueeze(0)
                cfg.dop_mask = payload["dop_mask"].unsqueeze(0)
                logger.info("Preservation: loaded legacy DOP class prompt %r", cached_class)
            else:
                raise ValueError("Preservation cache is missing DOP embeddings required by --dop")

    def _lookup_prompt_entry(self, prompt: str) -> dict[str, Any]:
        prompt_hash = dop_prompt_hash(prompt)
        entry = self.config.dop_prompt_bank.get(prompt_hash)
        if not isinstance(entry, dict):
            raise ValueError(
                f"DOP prompt {prompt!r} is absent from the preservation cache. "
                "The dataset captions or DOP rewriting configuration changed; re-run text-encoder caching."
            )
        if entry.get("prompt_hash", prompt_hash) != prompt_hash:
            raise ValueError("DOP prompt-bank hash mismatch; the preservation cache is corrupted")
        return self._load_prompt_entry(prompt_hash, entry)

    def _load_prompt_entry(self, prompt_hash: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        cached = self._prompt_tensor_cache.get(prompt_hash)
        if cached is not None:
            self._prompt_tensor_cache.move_to_end(prompt_hash)
            return {"prompt_hash": prompt_hash, **cached}
        embed = entry.get("embed")
        mask = entry.get("mask")
        if not isinstance(embed, torch.Tensor) or not isinstance(mask, torch.Tensor):
            relative_path = entry.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError("DOP prompt-bank entry is missing tensors and its lazy shard path")
            base_dir = os.path.abspath(self.config.dop_cache_base_dir)
            shard_path = os.path.abspath(os.path.join(base_dir, relative_path))
            try:
                if os.path.commonpath([base_dir, shard_path]) != base_dir:
                    raise ValueError("DOP prompt-bank shard path escapes the preservation cache directory")
            except ValueError as exc:
                raise ValueError("Invalid DOP prompt-bank shard path") from exc
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"DOP prompt-bank shard not found: {shard_path}")
            tensors = load_file(shard_path, device="cpu")
            embed = tensors.get("embed")
            mask = tensors.get("mask")
        if not isinstance(embed, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("DOP prompt-bank entry is missing its embedding or mask tensor")
        loaded = {"embed": embed, "mask": mask}
        self._prompt_tensor_cache[prompt_hash] = loaded
        self._prompt_tensor_cache.move_to_end(prompt_hash)
        while len(self._prompt_tensor_cache) > self._prompt_tensor_cache_limit:
            self._prompt_tensor_cache.popitem(last=False)
        return {"prompt_hash": prompt_hash, **loaded}

    def _resolve_dop_conditioning(
        self,
        captions: Sequence[str],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if cfg.dop_mode == "fixed":
            if cfg.dop_embed is None or cfg.dop_mask is None:
                raise ValueError("Fixed DOP conditioning was not encoded")
            embeds = cfg.dop_embed
            masks = cfg.dop_mask
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            if masks.dim() == 1:
                masks = masks.unsqueeze(0)
            variant = step % embeds.shape[0]
            embed = embeds[variant].unsqueeze(0).expand(batch_size, -1, -1)
            mask = masks[variant].unsqueeze(0).expand(batch_size, -1)
            return embed.to(device=device, dtype=dtype), mask.to(device=device)

        if len(captions) != batch_size:
            raise ValueError(f"Contextual DOP requires one caption per sample, got {len(captions)} captions for batch {batch_size}")
        selected_embeds: list[torch.Tensor] = []
        selected_masks: list[torch.Tensor] = []
        for sample_index, caption in enumerate(captions):
            prompts = build_dop_prompt_variants(
                caption,
                mode=cfg.dop_mode,
                class_prompts=cfg.dop_class_prompts,
                replacements=cfg.dop_replacements,
                prompt_bank=cfg.dop_prompt_bank_prompts,
            )
            if not prompts:
                raise ValueError(f"Contextual DOP produced no preservation prompt for caption {caption!r}")
            prompt = prompts[(step + sample_index) % len(prompts)]
            caption_hash = dop_prompt_hash(caption)
            indexed_prompts = cfg.dop_caption_index.get(caption_hash)
            if cfg.dop_strict_cache and (not isinstance(indexed_prompts, list) or dop_prompt_hash(prompt) not in indexed_prompts):
                raise ValueError(
                    "The current caption or its rewritten DOP prompt is absent from the cache index. "
                    "Re-run preservation caching after changing dataset captions."
                )
            entry = self._lookup_prompt_entry(prompt)
            selected_embeds.append(entry["embed"])
            selected_masks.append(entry["mask"])
        return (
            torch.stack(selected_embeds).to(device=device, dtype=dtype),
            torch.stack(selected_masks).to(device=device),
        )

    @staticmethod
    def _tree_batch_map(value: Any, fn, *, batch_size: int) -> Any:
        if isinstance(value, torch.Tensor):
            if value.dim() > 0 and value.shape[0] == batch_size:
                return fn(value)
            return value
        if isinstance(value, dict):
            return {key: PreservationHelper._tree_batch_map(child, fn, batch_size=batch_size) for key, child in value.items()}
        if isinstance(value, list):
            return [PreservationHelper._tree_batch_map(child, fn, batch_size=batch_size) for child in value]
        if isinstance(value, tuple):
            return tuple(PreservationHelper._tree_batch_map(child, fn, batch_size=batch_size) for child in value)
        return value

    @staticmethod
    def _tree_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: PreservationHelper._tree_to_cpu(child) for key, child in value.items()}
        if isinstance(value, list):
            return [PreservationHelper._tree_to_cpu(child) for child in value]
        if isinstance(value, tuple):
            return tuple(PreservationHelper._tree_to_cpu(child) for child in value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return None

    @staticmethod
    def _tree_anchor_safe(value: Any) -> bool:
        if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, dict):
            return all(PreservationHelper._tree_anchor_safe(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return all(PreservationHelper._tree_anchor_safe(child) for child in value)
        return False

    @staticmethod
    def _tree_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: PreservationHelper._tree_to_device(child, device) for key, child in value.items()}
        if isinstance(value, list):
            return [PreservationHelper._tree_to_device(child, device) for child in value]
        if isinstance(value, tuple):
            return tuple(PreservationHelper._tree_to_device(child, device) for child in value)
        return value

    def _stratified_video_inputs(
        self,
        dit_inputs: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        model_input = dit_inputs["model_input"]
        if isinstance(model_input, (list, tuple)):
            model_input = model_input[0]
        model_timesteps = dit_inputs["model_timesteps"]
        transformer_options = dict(dit_inputs["transformer_options"])
        if cfg.dop_timestep_bins <= 0:
            return model_input, model_timesteps, transformer_options
        clean = dit_inputs.get("clean_video")
        noise = dit_inputs.get("video_noise")
        if not isinstance(clean, torch.Tensor) or not isinstance(noise, torch.Tensor):
            raise ValueError("DOP timestep stratification requires clean latents and video noise in the training context")
        batch_size = clean.shape[0]
        span = cfg.dop_timestep_max - cfg.dop_timestep_min
        indices = (torch.arange(batch_size, device=clean.device) + self._timestep_cursor) % cfg.dop_timestep_bins
        sigmas = cfg.dop_timestep_min + (indices.float() + 0.5) * span / cfg.dop_timestep_bins
        self._timestep_cursor = int((self._timestep_cursor + batch_size) % cfg.dop_timestep_bins)
        sigma_5d = sigmas.to(device=clean.device, dtype=clean.dtype).view(batch_size, 1, 1, 1, 1)
        model_input = (1.0 - sigma_5d) * clean + sigma_5d * noise
        model_timesteps = sigmas.to(device=clean.device, dtype=clean.dtype).view(batch_size, 1)

        conditioning_mask = transformer_options.get("video_conditioning_mask")
        if isinstance(conditioning_mask, torch.Tensor):
            flat_size = int(clean.shape[2] * clean.shape[3] * clean.shape[4])
            if conditioning_mask.shape == (batch_size, flat_size):
                mask_5d = conditioning_mask.reshape(batch_size, 1, clean.shape[2], clean.shape[3], clean.shape[4])
                model_input = torch.where(mask_5d, clean, model_input)

        old_override = transformer_options.get("video_timestep_override")
        if isinstance(old_override, torch.Tensor):
            old_base = dit_inputs["model_timesteps"]
            if old_base.dim() == 1:
                old_base = old_base.unsqueeze(1)
            old_base = old_base[:, :1].to(device=old_override.device, dtype=old_override.dtype)
            ratio = torch.where(
                old_base.abs() > 1e-8,
                old_override / old_base,
                torch.zeros_like(old_override),
            )
            transformer_options["video_timestep_override"] = ratio * model_timesteps.to(device=ratio.device, dtype=ratio.dtype)
        elif isinstance(conditioning_mask, torch.Tensor):
            expanded = model_timesteps.expand(batch_size, conditioning_mask.shape[1])
            transformer_options["video_timestep_override"] = torch.where(
                conditioning_mask,
                torch.zeros_like(expanded),
                expanded,
            )
        return model_input, model_timesteps, transformer_options

    def _effective_dop_multiplier(self, relative_drift: float, *, update: bool) -> float:
        cfg = self.config
        if cfg.dop_adaptive_target <= 0.0:
            return float(cfg.dop_multiplier)
        if self._dop_drift_ema is None:
            self._dop_drift_ema = relative_drift
        elif update:
            decay = cfg.dop_adaptive_ema_decay
            self._dop_drift_ema = decay * self._dop_drift_ema + (1.0 - decay) * relative_drift
        if update and self._step >= cfg.dop_adaptive_warmup:
            normalized_error = (self._dop_drift_ema - cfg.dop_adaptive_target) / max(
                cfg.dop_adaptive_target,
                cfg.dop_relative_eps,
            )
            factor = math.exp(max(-2.0, min(2.0, cfg.dop_adaptive_rate * normalized_error)))
            self._dop_effective_multiplier = max(
                cfg.dop_adaptive_min,
                min(cfg.dop_adaptive_max, self._dop_effective_multiplier * factor),
            )
        return float(self._dop_effective_multiplier)

    def _load_anchor_bank(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("dop_prompt_config_hash") != self.config.dop_prompt_config_hash:
            raise ValueError("DOP anchor bank configuration does not match the current prompt rewriting configuration")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("DOP anchor bank is missing its entries")
        self._anchor_bank = entries[: self.config.dop_anchor_size or len(entries)]
        logger.info("Preservation: loaded %d deterministic DOP anchors from %s", len(self._anchor_bank), path)

    def _save_anchor_bank(self) -> None:
        path = self.config.dop_anchor_path
        if not path:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        temp_path = path + ".tmp"
        torch.save(
            {
                "version": 1,
                "dop_prompt_config_hash": self.config.dop_prompt_config_hash,
                "entries": self._anchor_bank,
            },
            temp_path,
        )
        os.replace(temp_path, path)

    def _capture_anchor(
        self,
        *,
        pres_inputs: Dict[str, Any],
        prior_pred: torch.Tensor,
        dop_slice: slice,
        video_loss_mask: Optional[torch.Tensor],
    ) -> None:
        cfg = self.config
        if cfg.dop_anchor_size <= 0 or len(self._anchor_bank) >= cfg.dop_anchor_size:
            return
        if not self._tree_anchor_safe(pres_inputs):
            if not self._warned_anchor_unsupported:
                logger.warning(
                    "DOP anchor capture is disabled for this path because transformer options contain "
                    "non-serializable runtime objects."
                )
                self._warned_anchor_unsupported = True
            return
        index = int(dop_slice.start or 0)
        full_batch = int(pres_inputs["model_timesteps"].shape[0])
        entry_inputs = self._tree_batch_map(
            pres_inputs,
            lambda tensor: tensor[index : index + 1],
            batch_size=full_batch,
        )
        entry_mask = None
        if isinstance(video_loss_mask, torch.Tensor):
            entry_mask = video_loss_mask[index : index + 1]
        entry = {
            "inputs": self._tree_to_cpu(entry_inputs),
            "prior": prior_pred[index : index + 1].detach().cpu(),
            "video_loss_mask": self._tree_to_cpu(entry_mask),
        }
        self._anchor_bank.append(entry)
        self._save_anchor_bank()

    def _replay_anchor_backward(
        self,
        *,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
    ) -> Optional[float]:
        cfg = self.config
        if (
            cfg.dop_anchor_size <= 0
            or cfg.dop_anchor_interval <= 0
            or not self._anchor_bank
            or self._step == 0
            or self._step % cfg.dop_anchor_interval != 0
        ):
            return None
        entry = self._anchor_bank[self._anchor_cursor % len(self._anchor_bank)]
        self._anchor_cursor += 1
        inputs = self._tree_to_device(entry["inputs"], accelerator.device)
        prior = entry["prior"].to(device=accelerator.device)
        self._prepare_block_swap(transformer, accelerator)
        with accelerator.autocast():
            pred = transformer(
                inputs["model_input"],
                timestep=inputs["model_timesteps"],
                context=inputs["text_embeds"],
                attention_mask=inputs["text_mask"],
                frame_rate=inputs["frame_rate"],
                transformer_options=inputs["transformer_options"],
            )
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        model_shape = tuple(inputs["model_input"].shape)
        spatial_weight = _expand_spatial_weight(
            pred,
            self._tree_to_device(entry.get("video_loss_mask"), accelerator.device),
            model_input_shape=model_shape,
            inside_weight=cfg.dop_inside_weight,
            outside_weight=cfg.dop_outside_weight,
        )
        raw_loss, _drift, _temporal = compute_dop_loss(
            pred,
            prior,
            loss_type=cfg.dop_loss_type,
            huber_delta=cfg.dop_huber_delta,
            relative_eps=cfg.dop_relative_eps,
            spatial_weight=spatial_weight,
            sigmas=inputs["model_timesteps"],
            timestep_weight=cfg.dop_timestep_weight,
            temporal_weight=cfg.dop_temporal_weight,
            model_input_shape=model_shape,
        )
        anchor_multiplier = self._dop_effective_multiplier if cfg.dop_adaptive_target > 0.0 else float(cfg.dop_multiplier)
        loss = raw_loss * anchor_multiplier * cfg.dop_anchor_weight
        if not torch.isfinite(loss):
            logger.warning("DOP anchor replay loss is non-finite; skipping backward")
            return float("nan")
        accelerator.backward(loss)
        return float(loss.detach().item())

    # -- prior divergence (no-grad fwd, LoRA OFF) -------------------------

    def compute_prior_divergence(
        self,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> torch.Tensor:
        """No-grad forward with LoRA OFF using training batch text embeddings.
        Returns prior_pred tensor (detached)."""
        self._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    dit_inputs["model_input"],
                    timestep=dit_inputs["model_timesteps"],
                    context=dit_inputs["text_embeds"],
                    attention_mask=dit_inputs["text_mask"],
                    frame_rate=dit_inputs["frame_rate"],
                    transformer_options=dit_inputs["transformer_options"],
                )
            if isinstance(prior_pred, (list, tuple)):
                prior_pred = prior_pred[0]  # video only
            return prior_pred.detach()
        finally:
            network.set_multiplier(1.0)

    # -- preservation backward (blank / DOP) ------------------------------

    def compute_preservation_losses(
        self,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
        *,
        backward: bool,
        update_state: bool,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Run blank and DOP in shared OFF/ON forwards and optionally backpropagate."""
        cfg = self.config
        device = accelerator.device
        batch_size = int(dit_inputs["model_timesteps"].shape[0])
        timestep_cursor_before = self._timestep_cursor
        model_input, model_timesteps, transformer_options = self._stratified_video_inputs(dit_inputs)
        if not update_state:
            self._timestep_cursor = timestep_cursor_before
        conditions: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        if cfg.blank_preservation:
            if cfg.blank_embed is None or cfg.blank_mask is None:
                raise ValueError("Blank preservation conditioning was not encoded")
            conditions.append(
                (
                    "blank",
                    cfg.blank_embed.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=network_dtype),
                    cfg.blank_mask.unsqueeze(0).expand(batch_size, -1).to(device=device),
                )
            )
        if cfg.dop:
            captions = dit_inputs.get("captions") or []
            dop_embed, dop_mask = self._resolve_dop_conditioning(
                captions,
                batch_size=batch_size,
                device=device,
                dtype=network_dtype,
                step=self._step,
            )
            conditions.append(("dop", dop_embed, dop_mask))
        if not conditions:
            return None, {}

        combined_embed = torch.cat([embed for _name, embed, _mask in conditions], dim=0)
        combined_mask = torch.cat([mask for _name, _embed, mask in conditions], dim=0)
        condition_count = len(conditions)
        combined_model_input = torch.cat([model_input] * condition_count, dim=0)
        combined_timesteps = torch.cat([model_timesteps] * condition_count, dim=0)
        combined_options = self._tree_batch_map(
            transformer_options,
            lambda tensor: torch.cat([tensor] * condition_count, dim=0),
            batch_size=batch_size,
        )
        frame_rate = dit_inputs["frame_rate"]
        if isinstance(frame_rate, torch.Tensor) and frame_rate.dim() > 0 and frame_rate.shape[0] == batch_size:
            frame_rate = torch.cat([frame_rate] * condition_count, dim=0)
        pres_inputs = {
            "model_input": combined_model_input,
            "model_timesteps": combined_timesteps,
            "text_embeds": combined_embed,
            "text_mask": combined_mask,
            "frame_rate": frame_rate,
            "transformer_options": combined_options,
        }
        total_batch = combined_timesteps.shape[0]
        microbatch = cfg.dop_microbatch if cfg.dop_microbatch > 0 else total_batch
        prior_chunks: list[torch.Tensor] = []
        pred_chunks: list[torch.Tensor] = []
        for start in range(0, total_batch, microbatch):
            end = min(total_batch, start + microbatch)
            chunk = self._tree_batch_map(
                pres_inputs,
                lambda tensor: tensor[start:end],
                batch_size=total_batch,
            )
            self._prepare_block_swap(transformer, accelerator)
            network.set_multiplier(0.0)
            try:
                with torch.no_grad(), accelerator.autocast():
                    prior = transformer(
                        chunk["model_input"],
                        timestep=chunk["model_timesteps"],
                        context=chunk["text_embeds"],
                        attention_mask=chunk["text_mask"],
                        frame_rate=chunk["frame_rate"],
                        transformer_options=chunk["transformer_options"],
                    )
                if isinstance(prior, (list, tuple)):
                    prior = prior[0]
                prior_chunks.append(prior.detach())
            finally:
                network.set_multiplier(1.0)

            self._prepare_block_swap(transformer, accelerator)
            grad_context = torch.enable_grad() if backward else torch.no_grad()
            with grad_context, accelerator.autocast():
                pred = transformer(
                    chunk["model_input"],
                    timestep=chunk["model_timesteps"],
                    context=chunk["text_embeds"],
                    attention_mask=chunk["text_mask"],
                    frame_rate=chunk["frame_rate"],
                    transformer_options=chunk["transformer_options"],
                )
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred_chunks.append(pred)

        prior_pred = torch.cat(prior_chunks, dim=0)
        pres_pred = torch.cat(pred_chunks, dim=0)
        metrics: Dict[str, float] = {}
        total_loss: Optional[torch.Tensor] = None
        dop_slice: Optional[slice] = None
        repeated_video_mask = dit_inputs.get("video_loss_mask")
        if isinstance(repeated_video_mask, torch.Tensor):
            repeated_video_mask = torch.cat([repeated_video_mask] * condition_count, dim=0)

        for condition_index, (name, _embed, _mask) in enumerate(conditions):
            condition_slice = slice(condition_index * batch_size, (condition_index + 1) * batch_size)
            condition_pred = pres_pred[condition_slice]
            condition_prior = prior_pred[condition_slice]
            if name == "blank":
                condition_loss = F.mse_loss(condition_pred.float(), condition_prior.float()) * cfg.blank_multiplier
                metrics["loss/blank_pres"] = float(condition_loss.detach().item())
            else:
                dop_slice = condition_slice
                condition_video_mask = (
                    repeated_video_mask[condition_slice] if isinstance(repeated_video_mask, torch.Tensor) else None
                )
                model_shape = tuple(model_input.shape)
                spatial_weight = _expand_spatial_weight(
                    condition_pred,
                    condition_video_mask,
                    model_input_shape=model_shape,
                    inside_weight=cfg.dop_inside_weight,
                    outside_weight=cfg.dop_outside_weight,
                )
                raw_loss, relative_drift, temporal = compute_dop_loss(
                    condition_pred,
                    condition_prior,
                    loss_type=cfg.dop_loss_type,
                    huber_delta=cfg.dop_huber_delta,
                    relative_eps=cfg.dop_relative_eps,
                    spatial_weight=spatial_weight,
                    sigmas=model_timesteps,
                    timestep_weight=cfg.dop_timestep_weight,
                    temporal_weight=cfg.dop_temporal_weight,
                    model_input_shape=model_shape,
                )
                effective_multiplier = self._effective_dop_multiplier(
                    float(relative_drift.item()),
                    update=update_state,
                )
                condition_loss = raw_loss * effective_multiplier
                metrics["loss/dop"] = float(condition_loss.detach().item())
                metrics["dop/relative_drift"] = float(relative_drift.item())
                metrics["dop/preservation_score"] = 1.0 / (1.0 + float(relative_drift.item()))
                metrics["dop/temporal"] = float(temporal.item())
                metrics["dop/multiplier"] = effective_multiplier
                if self._dop_drift_ema is not None:
                    metrics["dop/drift_ema"] = float(self._dop_drift_ema)
            total_loss = condition_loss if total_loss is None else total_loss + condition_loss

        if total_loss is None or not torch.isfinite(total_loss):
            logger.warning("Combined preservation loss is non-finite; skipping backward")
            return total_loss, {**metrics, "loss/preservation": float("nan")}
        metrics["loss/preservation"] = float(total_loss.detach().item())
        if backward:
            accelerator.backward(total_loss)
            if dop_slice is not None:
                self._capture_anchor(
                    pres_inputs=pres_inputs,
                    prior_pred=prior_pred,
                    dop_slice=dop_slice,
                    video_loss_mask=repeated_video_mask,
                )
                anchor_loss = self._replay_anchor_backward(
                    transformer=transformer,
                    network=network,
                    accelerator=accelerator,
                )
                if anchor_loss is not None:
                    metrics["loss/dop_anchor"] = anchor_loss
        if update_state:
            self._step += 1
        clean_memory_on_device(device)
        return total_loss, metrics

    def compute_preservation_backward(
        self,
        technique: str,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> float:
        """Compatibility wrapper for callers that request one preservation technique."""
        original_blank = self.config.blank_preservation
        original_dop = self.config.dop
        try:
            self.config.blank_preservation = technique == "blank"
            self.config.dop = technique == "dop"
            _loss, metrics = self.compute_preservation_losses(
                trainer,
                transformer,
                network,
                accelerator,
                dit_inputs,
                network_dtype,
                backward=True,
                update_state=True,
            )
            return metrics.get("loss/blank_pres" if technique == "blank" else "loss/dop", 0.0)
        finally:
            self.config.blank_preservation = original_blank
            self.config.dop = original_dop

    # -- audio DOP (preserve audio predictions on non-audio steps) ----------

    def compute_audio_dop_backward(
        self,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        av_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> float:
        """Two-forward audio DOP: run transformer with AV inputs, compare audio predictions
        between LoRA OFF (prior) and LoRA ON.  MSE on audio branch only × multiplier.
        Returns loss float."""
        mult = self.config.audio_dop_multiplier
        device = accelerator.device

        # (a) no-grad forward, LoRA OFF -> extract audio prior
        self._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    av_inputs["model_input"],
                    timestep=av_inputs["model_timesteps"],
                    audio_timestep=av_inputs["audio_timestep"],
                    context=av_inputs["text_embeds"],
                    attention_mask=av_inputs["text_mask"],
                    frame_rate=av_inputs["frame_rate"],
                    transformer_options=av_inputs["transformer_options"],
                )
            if not isinstance(prior_pred, (list, tuple)) or len(prior_pred) < 2:
                logger.warning("Audio DOP: transformer did not return [video, audio] — skipping.")
                return 0.0
            audio_prior = prior_pred[1].detach()
            del prior_pred
        finally:
            network.set_multiplier(1.0)

        # (b) with-grad forward, LoRA ON -> extract audio prediction
        self._prepare_block_swap(transformer, accelerator)
        with accelerator.autocast():
            lora_pred = transformer(
                av_inputs["model_input"],
                timestep=av_inputs["model_timesteps"],
                audio_timestep=av_inputs["audio_timestep"],
                context=av_inputs["text_embeds"],
                attention_mask=av_inputs["text_mask"],
                frame_rate=av_inputs["frame_rate"],
                transformer_options=av_inputs["transformer_options"],
            )
        if not isinstance(lora_pred, (list, tuple)) or len(lora_pred) < 2:
            logger.warning("Audio DOP: transformer did not return [video, audio] — skipping.")
            del audio_prior
            clean_memory_on_device(device)
            return 0.0
        audio_lora = lora_pred[1]
        del lora_pred

        # (c) MSE on audio predictions × multiplier
        adop_loss = F.mse_loss(audio_lora.float(), audio_prior.float()) * mult

        # (d) NaN guard
        if not torch.isfinite(adop_loss):
            logger.warning("Audio DOP loss is non-finite (%.4g), skipping backward.", adop_loss.item())
            del audio_prior, audio_lora, adop_loss
            clean_memory_on_device(device)
            return float("nan")

        # (e) separate backward
        accelerator.backward(adop_loss)

        loss_val = adop_loss.detach().item()

        # cleanup
        del audio_prior, audio_lora, adop_loss
        clean_memory_on_device(device)

        return loss_val
