"""Self-Flow helper for LTX-2 training.

Implements the Self-Flow training regularizer:
- Dual-timestep noising is performed by the trainer.
- This module handles feature alignment loss with a teacher model.

Teacher modes:
  "ema" (default): teacher = EMA-smoothed copy of trainable weights.
  "partial_ema": teacher = EMA-smoothed copy of the selected teacher block.
  "base": teacher = frozen pretrained base model with adapters disabled. This is
      a base-consistency regularizer, not canonical Self-Flow.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SelfFlowAudioView(TypedDict):
    student_noisy_audio: torch.Tensor
    student_audio_timesteps: torch.Tensor
    teacher_noisy_audio: torch.Tensor
    teacher_audio_timesteps: torch.Tensor
    audio_mask: torch.Tensor
    audio_masked_token_ratio: torch.Tensor
    audio_tau_mean: torch.Tensor
    audio_tau_min_mean: torch.Tensor


def build_self_flow_video_context(
    *,
    base_sigmas: torch.Tensor,
    alt_sigmas: torch.Tensor,
    teacher_noisy_model_input: torch.Tensor,
    teacher_model_timesteps: torch.Tensor,
    dual_timestep_mask: torch.Tensor,
    tau_tokens: torch.Tensor,
    tau_min: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> Dict[str, Any]:
    """Build the per-step video Self-Flow context captured by the trainer."""
    # Tokens where the student is genuinely noisier than the cleaner teacher view (masked AND the
    # alt sigma exceeds the base sigma). mask_focus_loss uses this so the focus lands on higher-noise
    # tokens, not merely "masked" ones — the alt sigma is an independent unordered draw.
    focus_mask = dual_timestep_mask & (alt_sigmas.view(-1, 1) > base_sigmas.view(-1, 1))
    return {
        "base_sigmas": base_sigmas.detach(),
        "focus_mask": focus_mask.detach(),
        "alt_sigmas": alt_sigmas.detach(),
        "teacher_noisy_model_input": teacher_noisy_model_input.detach(),
        "teacher_model_timesteps": teacher_model_timesteps.detach(),
        "dual_timestep_mask": dual_timestep_mask.detach(),
        "masked_token_ratio": float(dual_timestep_mask.float().mean().item()),
        "tau_mean": float(tau_tokens.mean().item()),
        "tau_min_mean": float(tau_min.mean().item()),
        "num_latent_frames": int(num_latent_frames),
        "latent_height": int(latent_height),
        "latent_width": int(latent_width),
    }


def prepare_self_flow_audio_view(
    *,
    audio_latents: torch.Tensor,
    audio_noise: torch.Tensor,
    base_audio_sigmas: torch.Tensor,
    alt_audio_sigmas: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> SelfFlowAudioView:
    """Build tokenwise student audio and cleaner teacher audio views for Self-Flow."""
    if audio_latents.dim() != 4:
        raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got {tuple(audio_latents.shape)}")
    batch_size = int(audio_latents.shape[0])
    audio_tokens = int(audio_latents.shape[2])
    if audio_tokens <= 0:
        raise ValueError("Self-Flow audio view requires at least one audio token")

    base = base_audio_sigmas.to(device=device, dtype=torch.float32).view(batch_size, -1)[:, :1]
    alt = alt_audio_sigmas.to(device=device, dtype=torch.float32).view(batch_size, -1)[:, :1]
    mask_ratio = max(0.0, min(0.5, float(mask_ratio)))
    token_mask = torch.rand((batch_size, audio_tokens), device=device, dtype=torch.float32) < mask_ratio

    student_token_sigmas = torch.where(token_mask, alt.expand(-1, audio_tokens), base.expand(-1, audio_tokens))
    teacher_sigmas = torch.minimum(base, alt)
    student_sigma_grid = student_token_sigmas[:, None, :, None].to(device=device, dtype=dtype)
    teacher_sigma_grid = teacher_sigmas[:, :, None, None].to(device=device, dtype=dtype)

    audio_latents_f = audio_latents.to(device=device, dtype=dtype)
    audio_noise_f = audio_noise.to(device=device, dtype=dtype)
    student_noisy_audio = (1.0 - student_sigma_grid) * audio_latents_f + student_sigma_grid * audio_noise_f
    teacher_noisy_audio = (1.0 - teacher_sigma_grid) * audio_latents_f + teacher_sigma_grid * audio_noise_f

    return {
        "student_noisy_audio": student_noisy_audio,
        "student_audio_timesteps": student_token_sigmas.to(device=device, dtype=dtype),
        "teacher_noisy_audio": teacher_noisy_audio,
        "teacher_audio_timesteps": teacher_sigmas.to(device=device, dtype=dtype),
        "audio_mask": token_mask,
        "audio_masked_token_ratio": token_mask.to(dtype=torch.float32).mean(),
        "audio_tau_mean": student_token_sigmas.mean(),
        "audio_tau_min_mean": teacher_sigmas.mean(),
    }


@dataclass
class SelfFlowConfig:
    student_block_idx: int = 16
    teacher_block_idx: int = 32
    student_block_ratio: Optional[float] = None
    teacher_block_ratio: Optional[float] = None
    lambda_self_flow: float = 0.1
    temporal_mode: str = "off"  # "off" | "frame" | "delta" | "hybrid"
    lambda_temporal: float = 0.0
    lambda_delta: float = 0.0
    temporal_tau: float = 1.0
    num_neighbors: int = 2
    temporal_granularity: str = "frame"  # "frame" | "patch"
    patch_spatial_radius: int = 0
    patch_match_mode: str = "hard"  # "hard" | "soft"
    patch_match_temperature: float = 0.1
    delta_num_steps: int = 1
    motion_weighting: str = "none"  # "none" | "teacher_delta"
    motion_weight_strength: float = 0.0
    temporal_schedule: str = "constant"  # "constant" | "linear" | "cosine" | "polynomial"
    temporal_warmup_steps: int = 0
    temporal_max_steps: int = 0
    schedule_end_weight: float = 0.0
    schedule_power: float = 1.0
    schedule_cutoff_step: int = 0
    similarity_cutoff: Optional[float] = None
    similarity_ema_decay: float = 0.99
    similarity_cutoff_mode: str = "permanent"  # "permanent" | "recoverable"
    mask_ratio: float = 0.10
    image_mask_ratio: Optional[float] = None  # defaults to mask_ratio when unset
    audio_mask_ratio: Optional[float] = None  # defaults to mask_ratio when unset
    frame_level_mask: bool = False  # mask whole frames instead of individual tokens
    teacher_mode: str = "ema"  # "ema" | "partial_ema" | "base"
    teacher_momentum: float = 0.999
    teacher_update_interval: int = 1
    projector_hidden_multiplier: int = 1
    projector_activation: str = "silu"  # "silu" | "gelu"
    loss_type: str = "one_minus_cosine"  # "negative_cosine" | "one_minus_cosine"
    dual_timestep: bool = True
    tokenwise_timestep: bool = True
    mask_focus_loss: bool = False  # focus rep loss on masked (higher-noise) tokens only
    max_loss: float = (
        0.0  # cap Self-Flow loss magnitude by rescaling (0 = disabled); caps the summed scalar loss value, not a gradient norm
    )
    student_block_stochastic_range: int = 0  # randomly vary student block ± this many blocks each step
    offload_teacher_features: bool = False
    offload_teacher_params: bool = False
    projector_lr: Optional[float] = None
    lambda_audio: float = 0.0  # audio representation alignment weight (0 = disabled)


class SelfFlowScheduler:
    """Stateful weight scheduler shared by every Self-Flow loss term."""

    def __init__(self, config: SelfFlowConfig):
        self.config = config
        self.similarity_ema: Optional[float] = None
        self.cutoff_latched = False
        self.cutoff_active = False
        self.last_scale = 1.0

    def update_similarity(self, similarity: float) -> None:
        value = float(similarity)
        if not math.isfinite(value):
            raise RuntimeError(f"Self-Flow cosine similarity is non-finite: {value}")
        if self.similarity_ema is None:
            self.similarity_ema = value
        else:
            decay = float(self.config.similarity_ema_decay)
            self.similarity_ema = decay * self.similarity_ema + (1.0 - decay) * value

        threshold = self.config.similarity_cutoff
        if threshold is None:
            return
        reached = self.similarity_ema >= float(threshold)
        if str(self.config.similarity_cutoff_mode).lower() == "permanent":
            self.cutoff_latched = self.cutoff_latched or reached
        else:
            self.cutoff_latched = reached
        self.cutoff_active = self.cutoff_latched

    def scale(self, global_step: int) -> float:
        step = max(0, int(global_step))
        hard_cutoff = max(0, int(self.config.schedule_cutoff_step))
        if (hard_cutoff > 0 and step >= hard_cutoff) or self.cutoff_latched:
            self.cutoff_active = True
            self.last_scale = 0.0
            return 0.0
        self.cutoff_active = False

        warmup_steps = max(0, int(self.config.temporal_warmup_steps))
        if warmup_steps > 0 and step < warmup_steps:
            self.last_scale = float(step) / float(warmup_steps)
            return self.last_scale

        schedule = str(self.config.temporal_schedule).lower()
        max_steps = max(0, int(self.config.temporal_max_steps))
        if schedule == "constant" or max_steps <= 0:
            self.last_scale = 1.0
            return 1.0

        progress = min(
            max(float(step - warmup_steps), 0.0) / max(float(max_steps - warmup_steps), 1.0),
            1.0,
        )
        if schedule == "linear":
            decay = 1.0 - progress
        elif schedule == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule == "polynomial":
            decay = (1.0 - progress) ** float(self.config.schedule_power)
        else:
            raise ValueError(f"Unsupported Self-Flow schedule: {schedule!r}")

        end_weight = float(self.config.schedule_end_weight)
        self.last_scale = end_weight + (1.0 - end_weight) * decay
        return self.last_scale


def parse_self_flow_args(raw_args: Optional[list[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict. Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    out: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"Self-Flow arg must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


class SelfFlowModule:
    """Training helper for Self-Flow feature alignment.

    Student and teacher features are returned explicitly by the LTX transformer.
    The teacher pass uses EMA weights unless base-consistency mode is selected.
    """

    def __init__(self, config: SelfFlowConfig, transformer: nn.Module):
        self.config = config
        self.transformer = transformer

        self.projectors: Optional[nn.ModuleDict] = None
        self.projector: Optional[nn.Sequential] = None
        self.audio_projector: Optional[nn.Sequential] = None
        self._student_features: Optional[torch.Tensor] = None
        self._teacher_features: Optional[torch.Tensor] = None
        self._student_audio_features: Optional[torch.Tensor] = None
        self._teacher_audio_features: Optional[torch.Tensor] = None
        self._step_uses_audio = False

        self._shadow_params: Dict[str, torch.Tensor] = {}
        self._step_counter: int = 0
        self._last_cosine: Optional[float] = None
        self._last_audio_cosine: Optional[float] = None
        self._last_frame_cosine: Optional[float] = None
        self._last_delta_cosine: Optional[float] = None
        self._last_ema_drift: Optional[float] = None
        self._current_lambda_self_flow: float = float(config.lambda_self_flow)
        self._current_lambda_audio: float = float(config.lambda_audio)
        self._current_lambda_temporal: float = float(config.lambda_temporal)
        self._current_lambda_delta: float = float(config.lambda_delta)
        self._resolved_student_block_idx: Optional[int] = None
        self._resolved_teacher_block_idx: Optional[int] = None
        self._active_student_block_idx: Optional[int] = None  # may differ each step when stochastic
        self._stochastic_student_indices: list = []
        self._scheduler = SelfFlowScheduler(config)

    @property
    def last_cosine(self) -> Optional[float]:
        return self._last_cosine

    @property
    def last_ema_drift(self) -> Optional[float]:
        return self._last_ema_drift

    @property
    def last_audio_cosine(self) -> Optional[float]:
        return self._last_audio_cosine

    @property
    def last_frame_cosine(self) -> Optional[float]:
        return self._last_frame_cosine

    @property
    def last_delta_cosine(self) -> Optional[float]:
        return self._last_delta_cosine

    @property
    def current_lambda_temporal(self) -> float:
        return float(self._current_lambda_temporal)

    @property
    def current_lambda_delta(self) -> float:
        return float(self._current_lambda_delta)

    @property
    def current_lambda_self_flow(self) -> float:
        return float(self._current_lambda_self_flow)

    @property
    def has_active_loss(self) -> bool:
        return (
            self._current_lambda_self_flow > 0.0
            or self._current_lambda_audio > 0.0
            or (str(self.config.temporal_mode).lower() in {"frame", "hybrid"} and self.current_lambda_temporal > 0.0)
            or (str(self.config.temporal_mode).lower() in {"delta", "hybrid"} and self.current_lambda_delta > 0.0)
        )

    @property
    def should_capture(self) -> bool:
        return self.has_active_loss or (
            self.config.similarity_cutoff is not None and str(self.config.similarity_cutoff_mode).lower() == "recoverable"
        )

    @property
    def schedule_scale(self) -> float:
        return float(self._scheduler.last_scale)

    @property
    def similarity_ema(self) -> Optional[float]:
        return self._scheduler.similarity_ema

    @property
    def schedule_cutoff_active(self) -> bool:
        return bool(self._scheduler.cutoff_active)

    @property
    def has_ema_teacher(self) -> bool:
        return bool(self._shadow_params)

    @property
    def effective_audio_mask_ratio(self) -> float:
        value = self.config.audio_mask_ratio
        return float(self.config.mask_ratio if value is None else value)

    def effective_video_mask_ratio(self, num_latent_frames: int) -> float:
        if int(num_latent_frames) == 1 and self.config.image_mask_ratio is not None:
            return float(self.config.image_mask_ratio)
        return float(self.config.mask_ratio)

    @property
    def needs_video_features(self) -> bool:
        temporal_mode = str(self.config.temporal_mode).lower()
        return (
            self._current_lambda_self_flow > 0.0
            or (temporal_mode in {"frame", "hybrid"} and self.current_lambda_temporal > 0.0)
            or (temporal_mode in {"delta", "hybrid"} and self.current_lambda_delta > 0.0)
            or self.config.similarity_cutoff is not None
        )

    @property
    def student_hidden_state_layer(self) -> int:
        if self._active_student_block_idx is None:
            raise RuntimeError("Self-Flow student block has not been resolved")
        return int(self._active_student_block_idx)

    @property
    def teacher_hidden_state_layer(self) -> int:
        if self._resolved_teacher_block_idx is None:
            raise RuntimeError("Self-Flow teacher block has not been resolved")
        return int(self._resolved_teacher_block_idx)

    def _make_activation(self) -> nn.Module:
        act = str(self.config.projector_activation).lower()
        if act == "gelu":
            return nn.GELU()
        return nn.SiLU()

    def _get_blocks(self) -> tuple[list[nn.Module], int]:
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Self-Flow requires transformer.transformer_blocks")
        block_list = list(blocks)
        return block_list, len(block_list)

    @staticmethod
    def _matches_block(param_name: str, block_idx: int) -> bool:
        """Return True if a parameter name belongs to the given transformer block index.

        Handles both dot notation (``transformer_blocks.32.``) and kohya-style
        underscore notation (``transformer_blocks_32_``).
        """
        return f"transformer_blocks.{block_idx}." in param_name or f"transformer_blocks_{block_idx}_" in param_name

    @staticmethod
    def _resolve_shadow_name(param_name: str, shadow_params: Dict[str, torch.Tensor]) -> Optional[str]:
        if param_name in shadow_params:
            return param_name
        if param_name.startswith("module."):
            stripped = param_name[len("module.") :]
            if stripped in shadow_params:
                return stripped
        prefixed = f"module.{param_name}"
        if prefixed in shadow_params:
            return prefixed
        return None

    @staticmethod
    def _resolve_ratio_index(ratio: float, depth: int, *, mode: str) -> int:
        if not (0.0 < float(ratio) < 1.0):
            raise ValueError(f"Self-Flow block ratio must be in (0, 1), got {ratio!r}")
        position = float(ratio) * float(depth)
        if mode == "floor":
            resolved = int(math.floor(position))
        elif mode == "ceil":
            resolved = int(math.ceil(position))
        else:
            raise ValueError(f"Unsupported Self-Flow ratio resolution mode: {mode!r}")
        return max(0, min(depth - 1, resolved))

    def resolve_block_indices(self, depth: int) -> tuple[int, int]:
        student_idx = int(self.config.student_block_idx)
        teacher_idx = int(self.config.teacher_block_idx)
        if self.config.student_block_ratio is not None:
            student_idx = self._resolve_ratio_index(self.config.student_block_ratio, depth, mode="floor")
        if self.config.teacher_block_ratio is not None:
            teacher_idx = self._resolve_ratio_index(self.config.teacher_block_ratio, depth, mode="ceil")
        return student_idx, teacher_idx

    def setup(self, device: torch.device, dtype: torch.dtype, registration_target: nn.Module) -> None:
        blocks, depth = self._get_blocks()
        student_idx, teacher_idx = self.resolve_block_indices(depth)
        if not (0 <= student_idx < depth):
            raise ValueError(f"student_block_idx={student_idx} out of range (model has {depth} blocks)")
        if not (0 <= teacher_idx < depth):
            raise ValueError(f"teacher_block_idx={teacher_idx} out of range (model has {depth} blocks)")
        if teacher_idx <= student_idx:
            raise ValueError("teacher_block_idx must be > student_block_idx for Self-Flow")
        _valid_teacher_modes = {"base", "ema", "partial_ema"}
        if str(self.config.teacher_mode).lower() not in _valid_teacher_modes:
            raise ValueError(f"Unknown teacher_mode={self.config.teacher_mode!r}. Must be one of: {sorted(_valid_teacher_modes)}")
        self._resolved_student_block_idx = student_idx
        self._resolved_teacher_block_idx = teacher_idx
        self._active_student_block_idx = student_idx

        stochastic_range = max(0, int(self.config.student_block_stochastic_range))
        lo = max(0, student_idx - stochastic_range)
        hi = min(teacher_idx - 1, student_idx + stochastic_range)  # must stay below teacher
        self._stochastic_student_indices = list(range(lo, hi + 1))
        if stochastic_range > 0 and len(self._stochastic_student_indices) > 1:
            logger.warning(
                "Self-Flow: student_block_stochastic_range=%d selects among %d student blocks [%d..%d], "
                "but a single projector MLP is shared across all depths. "
                "The projector will be a compromise; consider range=0 for best alignment accuracy.",
                stochastic_range,
                len(self._stochastic_student_indices),
                lo,
                hi,
            )
        if self._stochastic_student_indices and max(self._stochastic_student_indices) >= teacher_idx - 1:
            logger.warning(
                "Self-Flow: stochastic student range reaches block %d which is adjacent to teacher block %d. "
                "A 1-block student-teacher gap may produce a trivially satisfied loss. "
                "Consider reducing student_block_stochastic_range.",
                max(self._stochastic_student_indices),
                teacher_idx,
            )

        inner_dim = int(getattr(self.transformer, "inner_dim", 0))
        if inner_dim <= 0:
            raise ValueError("Self-Flow could not resolve transformer.inner_dim")
        hidden_dim = inner_dim * int(self.config.projector_hidden_multiplier)
        activation = self._make_activation()
        video_projector = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, inner_dim),
        ).to(device=device, dtype=dtype)

        projectors = nn.ModuleDict({"video": video_projector})
        if float(self.config.lambda_audio) > 0.0:
            audio_inner_dim = int(getattr(self.transformer, "audio_inner_dim", 0))
            if audio_inner_dim <= 0:
                raise ValueError(f"Self-Flow lambda_audio={self.config.lambda_audio:.4f} requires a transformer audio branch")
            audio_hidden_dim = audio_inner_dim * int(self.config.projector_hidden_multiplier)
            projectors["audio"] = nn.Sequential(
                nn.Linear(audio_inner_dim, audio_hidden_dim),
                self._make_activation(),
                nn.Linear(audio_hidden_dim, audio_inner_dim),
            ).to(device=device, dtype=dtype)
            logger.info("Self-Flow audio projector created: audio_inner_dim=%d", audio_inner_dim)

        if "_self_flow_projectors" in registration_target._modules:
            raise RuntimeError("Self-Flow projectors are already registered on the training model")
        registration_target.add_module("_self_flow_projectors", projectors)
        self.projectors = projectors
        self.projector = projectors["video"]
        self.audio_projector = projectors["audio"] if "audio" in projectors else None

        logger.info(
            "Self-Flow ready: student_block=%d teacher_block=%d student_ratio=%s teacher_ratio=%s "
            "mask_ratio=%.3f image_mask_ratio=%s audio_mask_ratio=%.3f frame_level_mask=%s teacher_mode=%s lambda=%.4f "
            "temporal_mode=%s lambda_temporal=%.4f lambda_delta=%.4f "
            "temporal_tau=%.3f num_neighbors=%d temporal_granularity=%s patch_spatial_radius=%d "
            "patch_match_mode=%s patch_match_temperature=%.4f delta_num_steps=%d motion_weighting=%s "
            "motion_weight_strength=%.4f temporal_schedule=%s "
            "temporal_warmup_steps=%d temporal_max_steps=%d schedule_end_weight=%.4f "
            "schedule_power=%.3f schedule_cutoff_step=%d similarity_cutoff=%s "
            "max_loss=%.4f student_block_stochastic_range=%d "
            "momentum=%.4f dual_timestep=%s tokenwise_timestep=%s "
            "offload_teacher_params=%s projector_lr=%s",
            student_idx,
            teacher_idx,
            self.config.student_block_ratio,
            self.config.teacher_block_ratio,
            self.config.mask_ratio,
            self.config.image_mask_ratio,
            self.effective_audio_mask_ratio,
            str(self.config.frame_level_mask).lower(),
            self.config.teacher_mode,
            self.config.lambda_self_flow,
            self.config.temporal_mode,
            self.config.lambda_temporal,
            self.config.lambda_delta,
            self.config.temporal_tau,
            self.config.num_neighbors,
            self.config.temporal_granularity,
            self.config.patch_spatial_radius,
            self.config.patch_match_mode,
            self.config.patch_match_temperature,
            self.config.delta_num_steps,
            self.config.motion_weighting,
            self.config.motion_weight_strength,
            self.config.temporal_schedule,
            self.config.temporal_warmup_steps,
            self.config.temporal_max_steps,
            self.config.schedule_end_weight,
            self.config.schedule_power,
            self.config.schedule_cutoff_step,
            self.config.similarity_cutoff,
            self.config.max_loss,
            int(self.config.student_block_stochastic_range),
            self.config.teacher_momentum,
            str(self.config.dual_timestep).lower(),
            str(self.config.tokenwise_timestep).lower(),
            str(self.config.offload_teacher_params).lower(),
            self.config.projector_lr,
        )

    @staticmethod
    def _extract_native_hidden_states(output: Any) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise RuntimeError(
                "Self-Flow requested native hidden states but the LTX wrapper did not return "
                "(video_pred, audio_pred, video_hidden, audio_hidden)"
            )
        video_hidden, audio_hidden = output[2], output[3]
        if video_hidden is not None and not isinstance(video_hidden, torch.Tensor):
            raise RuntimeError("Self-Flow video hidden state has an invalid type")
        if audio_hidden is not None and not isinstance(audio_hidden, torch.Tensor):
            raise RuntimeError("Self-Flow audio hidden state has an invalid type")
        return video_hidden, audio_hidden

    @staticmethod
    def _is_adapter_module(m: nn.Module) -> bool:
        """Return True if *m* is a kohya-style adapter wrapper (LoRA/DoRA/LoHa/LoKr/DoKr/OFT/DoRA-OFT).

        Detected by the universal wrapper contract: an ``org_forward`` callable plus a
        ``multiplier``. This also covers OFT / DoRA-OFT (rotation) and DoKr, which the old
        lora_down/hada/lokr_w1 checks missed. Disabling these via ``enabled=False`` makes the
        teacher forward return the pristine frozen base, correctly neutralizing the DoRA
        magnitude and OFT rotation that zeroing ``multiplier`` alone would leave active.
        """
        return callable(getattr(m, "org_forward", None)) and hasattr(m, "multiplier")

    @staticmethod
    def _collect_lora_modules(network: nn.Module) -> list:
        """Return all adapter modules (LoRA, LoHa, LoKr) that have a multiplier attribute."""
        return [m for m in network.modules() if SelfFlowModule._is_adapter_module(m)]

    @staticmethod
    def _disable_adapters(modules: list) -> list:
        """Disable every adapter for the frozen-base teacher pass; return saved enabled-states.

        Setting ``enabled=False`` makes each adapter forward return ``org_forward`` (the pristine
        base), neutralizing the DoRA magnitude vector and OFT rotation — unlike zeroing
        ``multiplier``, which leaves those base-rescaling terms active.
        """
        saved = [bool(getattr(m, "enabled", True)) for m in modules]
        for m in modules:
            m.enabled = False
        return saved

    @staticmethod
    def _restore_adapters(modules: list, saved: list) -> None:
        for m, v in zip(modules, saved):
            m.enabled = v

    def init_teacher(self, network: nn.Module) -> None:
        mode = str(self.config.teacher_mode).lower()
        if mode == "base":
            adapters = self._collect_lora_modules(network)
            if not adapters:
                raise ValueError(
                    "Self-Flow teacher_mode=base needs an adapter network (LoRA/DoRA/LoKr/OFT) to disable for "
                    "the frozen-base teacher pass, but none were found. For full fine-tuning use "
                    "teacher_mode=ema or partial_ema instead."
                )
            logger.info(
                "Self-Flow teacher_mode=base: frozen-base teacher via disabling %d adapter module(s) for the "
                "teacher pass (no EMA shadow needed)",
                len(adapters),
            )
            return
        teacher_block = self._resolved_teacher_block_idx
        self._shadow_params.clear()
        for name, param in network.named_parameters():
            if name.startswith("_self_flow_projectors.") or "._self_flow_projectors." in name:
                continue
            if not param.requires_grad:
                continue
            if mode == "partial_ema" and teacher_block is not None:
                if not self._matches_block(name, teacher_block):
                    continue
            if bool(self.config.offload_teacher_params):
                self._shadow_params[name] = param.detach().to(device="cpu").clone()
            else:
                self._shadow_params[name] = param.detach().clone()
        # Warn if EMA target is the full transformer (full fine-tuning mode).
        is_full_transformer = hasattr(network, "transformer_blocks")
        if mode in ("ema", "partial_ema") and is_full_transformer and self._shadow_params:
            param_count = sum(p.numel() for p in self._shadow_params.values())
            param_mb = param_count * 4 / (1024 * 1024)  # fp32 estimate
            logger.warning(
                "Self-Flow: EMA shadow params cover the full transformer (%.0f MB). "
                "Use teacher_mode=partial_ema to limit shadow params to one block.",
                param_mb,
            )

        if mode == "partial_ema":
            if not self._shadow_params:
                raise ValueError(
                    "Self-Flow teacher_mode=partial_ema: no trainable parameters matched block %s. "
                    "Ensure that block's transformer layers are included in the training target. "
                    "Use teacher_mode=ema or include the selected teacher block in the training target." % teacher_block
                )
            logger.info(
                "Self-Flow teacher_mode=partial_ema: EMA for %d tensors in block %s",
                len(self._shadow_params),
                teacher_block,
            )
        else:
            if not self._shadow_params:
                raise ValueError("Self-Flow teacher_mode=ema found no trainable parameters for the teacher")
            logger.info("Self-Flow: initialized EMA teacher with %d tensors", len(self._shadow_params))

    def update_teacher(self, network: nn.Module) -> None:
        if not self._shadow_params:
            return
        self._step_counter += 1
        if self._step_counter % max(1, int(self.config.teacher_update_interval)) != 0:
            return
        momentum = float(self.config.teacher_momentum)
        drift_sum = 0.0
        drift_count = 0
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                shadow_target_dtype = shadow.dtype
                source = param.detach()
                if source.device != shadow.device or source.dtype != shadow_target_dtype:
                    source = source.to(device=shadow.device, dtype=shadow_target_dtype)
                # Compute drift before EMA update
                drift_sum += (shadow - source).norm().item()
                drift_count += 1
                shadow.mul_(momentum).add_(source, alpha=1.0 - momentum)
        if drift_count > 0:
            self._last_ema_drift = drift_sum / drift_count

    def _swap_in_teacher(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        """Temporarily rebind tracked parameters to EMA storage without cloning student weights."""
        backups: Dict[str, torch.Tensor] = {}
        if not self._shadow_params:
            return backups
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                backups[name] = param.data
                if shadow.device != param.device or shadow.dtype != param.dtype:
                    param.data = shadow.to(device=param.device, dtype=param.dtype)
                else:
                    param.data = shadow
        return backups

    @staticmethod
    def _restore_from_backups(network: nn.Module, backups: Dict[str, torch.Tensor]) -> None:
        if not backups:
            return
        with torch.no_grad():
            for name, param in network.named_parameters():
                backup = backups.get(name)
                if backup is None:
                    continue
                param.data = backup

    def apply_teacher_weights(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        """Activate the EMA teacher for evaluation, sampling, or adapter export."""
        return self._swap_in_teacher(network)

    def restore_student_weights(self, network: nn.Module, backups: Dict[str, torch.Tensor]) -> None:
        """Restore weights returned by :meth:`apply_teacher_weights`."""
        self._restore_from_backups(network, backups)

    @staticmethod
    def is_auxiliary_state_name(name: str) -> bool:
        return name.startswith("_self_flow_projectors.") or "._self_flow_projectors." in name

    def build_teacher_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Build a deployable model state dict with EMA parameters and ordinary model buffers."""
        if not self._shadow_params:
            raise RuntimeError("Self-Flow has no EMA teacher weights to export")
        state: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if self.is_auxiliary_state_name(name):
                continue
            shadow_name = self._resolve_shadow_name(name, self._shadow_params)
            state[name] = self._shadow_params[shadow_name] if shadow_name is not None else param.data
        for name, buffer in model.named_buffers():
            if not self.is_auxiliary_state_name(name):
                state[name] = buffer
        return state

    def mark_student_forward(self) -> None:
        self._student_features = None
        self._student_audio_features = None
        if len(self._stochastic_student_indices) > 1:
            self._active_student_block_idx = random.choice(self._stochastic_student_indices)
        else:
            self._active_student_block_idx = self._resolved_student_block_idx

    def cache_student_output(self, output: Any) -> tuple[Any, Any]:
        video_hidden, audio_hidden = self._extract_native_hidden_states(output)
        self._student_features = video_hidden
        self._student_audio_features = audio_hidden
        if self.needs_video_features and video_hidden is None:
            raise RuntimeError("Self-Flow video loss is active but the student video hidden state is missing")
        if self._step_uses_audio and self.audio_projector is not None and audio_hidden is None:
            raise RuntimeError("Self-Flow audio loss is active but the student audio hidden state is missing")
        return output[0], output[1]

    def cleanup_step(self) -> None:
        self._student_features = None
        self._teacher_features = None
        self._student_audio_features = None
        self._teacher_audio_features = None
        self._last_cosine = None
        self._last_audio_cosine = None
        self._last_frame_cosine = None
        self._last_delta_cosine = None
        self._step_uses_audio = False
        # Note: _last_ema_drift is NOT cleared here — it's updated in update_teacher
        # which runs after optimizer.step, not during compute_loss

    def on_step(self, global_step: int) -> None:
        scale = self._scheduler.scale(global_step)
        # Apply the schedule to every Self-Flow regularization term so the
        # logged lambdas match the documented effective weights.
        self._current_lambda_self_flow = float(self.config.lambda_self_flow) * scale
        self._current_lambda_audio = float(self.config.lambda_audio) * scale
        self._current_lambda_temporal = float(self.config.lambda_temporal) * scale
        self._current_lambda_delta = float(self.config.lambda_delta) * scale

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = []
        if self.projector is not None:
            params.extend(self.projector.parameters())
        if self.audio_projector is not None:
            params.extend(self.audio_projector.parameters())
        return params

    def prepare_teacher_features(
        self,
        *,
        accelerator,
        transformer: nn.Module,
        network: nn.Module,
        teacher_model_input: Any,
        teacher_timesteps: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        frame_rate: int | float,
        transformer_options: Dict[str, Any],
        extra_forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.projector is None or not self.should_capture:
            return
        self._teacher_features = None
        self._teacher_audio_features = None
        self._step_uses_audio = audio_timestep is not None and self._current_lambda_audio > 0.0
        prev_training = bool(getattr(transformer, "training", False))
        teacher_forward_kwargs = {
            "timestep": teacher_timesteps,
            "audio_timestep": audio_timestep,
            "context": text_embeds,
            "attention_mask": text_mask,
            "frame_rate": frame_rate,
            "transformer_options": transformer_options,
            "output_hidden_states": True,
            "hidden_state_layer": self.teacher_hidden_state_layer,
        }
        if extra_forward_kwargs:
            teacher_forward_kwargs.update(extra_forward_kwargs)

        if str(self.config.teacher_mode).lower() == "base":
            # Teacher = frozen pretrained base: disable all adapters for this pass so each forward
            # returns org_forward (the pristine base). This correctly neutralizes the DoRA magnitude /
            # OFT rotation that zeroing `multiplier` alone would leave active.
            adapter_mods = self._collect_lora_modules(network)
            if not adapter_mods:
                raise RuntimeError(
                    "Self-Flow teacher_mode=base found no adapter modules to disable; the teacher would equal "
                    "the student (zero distillation signal). Use teacher_mode=ema/partial_ema."
                )
            saved_states = self._disable_adapters(adapter_mods)
            try:
                if prev_training:
                    transformer.eval()
                with torch.no_grad(), accelerator.autocast():
                    output = transformer(teacher_model_input, **teacher_forward_kwargs)
            finally:
                self._restore_adapters(adapter_mods, saved_states)
                if prev_training:
                    transformer.train()
        else:
            # Teacher = EMA-smoothed LoRA weights ("ema" or "partial_ema").
            # For partial_ema, shadow_params is already scoped to the teacher block only,
            # so _swap_in_teacher naturally only touches those params.
            backups = self._swap_in_teacher(network)
            try:
                if prev_training:
                    transformer.eval()
                with torch.no_grad(), accelerator.autocast():
                    output = transformer(teacher_model_input, **teacher_forward_kwargs)
            finally:
                self._restore_from_backups(network, backups)
                if prev_training:
                    transformer.train()

        teacher_hidden, teacher_audio_hidden = self._extract_native_hidden_states(output)
        self._teacher_features = teacher_hidden.detach() if teacher_hidden is not None else None
        self._teacher_audio_features = teacher_audio_hidden.detach() if teacher_audio_hidden is not None else None
        if self.needs_video_features and self._teacher_features is None:
            raise RuntimeError("Self-Flow video loss is active but the teacher video hidden state is missing")
        if self._step_uses_audio and self.audio_projector is not None and self._teacher_audio_features is None:
            raise RuntimeError("Self-Flow audio loss is active but the teacher audio hidden state is missing")

        if bool(self.config.offload_teacher_features):
            if self._teacher_features is not None and self._teacher_features.device.type != "cpu":
                self._teacher_features = self._teacher_features.to(device="cpu", non_blocking=False)
            if self._teacher_audio_features is not None and self._teacher_audio_features.device.type != "cpu":
                self._teacher_audio_features = self._teacher_audio_features.to(device="cpu", non_blocking=False)

    def _loss_from_cosine(self, cosine: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "one_minus_cosine":
            return 1.0 - cosine
        return -cosine  # negative cosine: gradient pushes cosine toward +1 (same direction as 1-cosine but different magnitude)

    def _reshape_temporal_features(self, features: torch.Tensor, num_latent_frames: Optional[int]) -> Optional[torch.Tensor]:
        if num_latent_frames is None:
            return None
        total_tokens = int(features.shape[1])
        num_frames = int(num_latent_frames)
        if num_frames <= 1 or total_tokens < num_frames:
            return None
        usable_tokens = (total_tokens // num_frames) * num_frames
        if usable_tokens <= 0:
            return None
        if usable_tokens != total_tokens:
            features = features[:, :usable_tokens]
        spatial_tokens = usable_tokens // num_frames
        return features.reshape(features.shape[0], num_frames, spatial_tokens, features.shape[-1])

    @staticmethod
    def _reshape_temporal_grid(
        features: torch.Tensor,
        *,
        num_latent_frames: Optional[int],
        latent_height: Optional[int],
        latent_width: Optional[int],
    ) -> Optional[torch.Tensor]:
        if num_latent_frames is None or latent_height is None or latent_width is None:
            return None
        num_frames = int(num_latent_frames)
        height = int(latent_height)
        width = int(latent_width)
        if num_frames <= 1 or height <= 0 or width <= 0:
            return None
        expected_tokens = num_frames * height * width
        total_tokens = int(features.shape[1])
        if total_tokens < expected_tokens:
            return None
        if total_tokens != expected_tokens:
            features = features[:, :expected_tokens]
        return features.reshape(features.shape[0], num_frames, height, width, features.shape[-1])

    @staticmethod
    def _neighbor_weighted_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sim = torch.bmm(student_frames, teacher_frames.transpose(1, 2))
        num_frames = sim.shape[1]
        tau = max(float(temporal_tau), 1e-6)
        total = sim.new_zeros(())
        normalizer = sim.new_zeros(())

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                diag = sim.diagonal(dim1=1, dim2=2)
                if motion_weights is None:
                    total = total + diag.sum()
                    normalizer = normalizer + sim.new_tensor(diag.numel() * weight)
                else:
                    cast_weights = motion_weights.to(device=diag.device, dtype=diag.dtype)
                    total = total + weight * (diag * cast_weights).sum()
                    normalizer = normalizer + weight * cast_weights.sum()
                continue

            forward = sim.diagonal(offset=delta, dim1=1, dim2=2)
            backward = sim.diagonal(offset=-delta, dim1=1, dim2=2)
            if motion_weights is None:
                total = total + weight * (forward.sum() + backward.sum())
                normalizer = normalizer + sim.new_tensor(sim.shape[0] * (forward.shape[-1] + backward.shape[-1]) * weight)
            else:
                forward_weights = motion_weights[:, :-delta].to(device=forward.device, dtype=forward.dtype)
                backward_weights = motion_weights[:, delta:].to(device=backward.device, dtype=backward.dtype)
                total = total + weight * (forward * forward_weights).sum()
                total = total + weight * (backward * backward_weights).sum()
                normalizer = normalizer + weight * (forward_weights.sum() + backward_weights.sum())

        if normalizer.item() <= 0.0:
            return sim.diagonal(dim1=1, dim2=2).mean()
        return total / normalizer

    @staticmethod
    def _normalize_motion_weights(motion: torch.Tensor, strength: float) -> torch.Tensor:
        if float(strength) <= 0.0:
            return torch.ones_like(motion)
        motion = motion.to(dtype=torch.float32)
        baseline = motion.mean()
        if not torch.isfinite(baseline) or float(baseline.item()) <= 1e-8:
            return torch.ones_like(motion)
        normalized = motion / baseline.clamp_min(1e-8)
        weights = 1.0 + float(strength) * normalized
        return weights.to(dtype=motion.dtype)

    @staticmethod
    def _teacher_delta_motion_weights(
        teacher_frames: torch.Tensor,
        *,
        strength: float,
    ) -> torch.Tensor:
        if float(strength) <= 0.0:
            return torch.ones(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)
        if teacher_frames.shape[1] <= 1:
            return torch.ones(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)

        forward = (teacher_frames[:, 1:] - teacher_frames[:, :-1]).pow(2).mean(dim=-1)
        motion = torch.zeros(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)
        motion[:, :-1] = motion[:, :-1] + forward
        motion[:, 1:] = motion[:, 1:] + forward
        return SelfFlowModule._normalize_motion_weights(motion, strength)

    @staticmethod
    def _neighbor_weighted_local_patch_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        spatial_radius: int,
        patch_match_mode: str,
        patch_match_temperature: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_radius <= 0:
            if motion_weights is not None and motion_weights.dim() > 2:
                # Grid path supplies per-patch [B, F, H, W] weights; collapse to per-frame [B, F]
                # since this branch pools spatial patches (local_values is [B, F]).
                motion_weights = motion_weights.flatten(2).mean(dim=2)
            flat_student = student_frames.reshape(
                student_frames.shape[0],
                student_frames.shape[1],
                student_frames.shape[2] * student_frames.shape[3],
                student_frames.shape[4],
            )
            flat_teacher = teacher_frames.reshape(
                teacher_frames.shape[0],
                teacher_frames.shape[1],
                teacher_frames.shape[2] * teacher_frames.shape[3],
                teacher_frames.shape[4],
            )
            sim = torch.einsum("btnd,bsnd->btsn", flat_student, flat_teacher)
            tau = max(float(temporal_tau), 1e-6)
            total = sim.new_zeros(())
            normalizer = sim.new_zeros(())

            def _reduce(
                values: torch.Tensor, weight: float, weights_slice: Optional[torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                local_values = values.mean(dim=-1)
                if weights_slice is None:
                    return weight * local_values.sum(), sim.new_tensor(local_values.numel() * weight)
                weights_view = weights_slice.to(device=local_values.device, dtype=local_values.dtype).reshape_as(local_values)
                return (
                    weight * (local_values * weights_view).sum(),
                    weight * weights_view.sum(),
                )

            for delta in range(0, max(0, int(num_neighbors)) + 1):
                weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
                if delta == 0:
                    diag = sim.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
                    part_total, part_norm = _reduce(diag, weight, motion_weights)
                    total = total + part_total
                    normalizer = normalizer + part_norm
                    continue

                forward = sim.diagonal(offset=delta, dim1=1, dim2=2).permute(0, 2, 1)
                backward = sim.diagonal(offset=-delta, dim1=1, dim2=2).permute(0, 2, 1)
                forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
                backward_weights = None if motion_weights is None else motion_weights[:, delta:]
                part_total, part_norm = _reduce(forward, weight, forward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm
                part_total, part_norm = _reduce(backward, weight, backward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm

            if normalizer.item() <= 0.0:
                return sim.diagonal(dim1=1, dim2=2).mean()
            return total / normalizer

        batch_size, num_frames, height, width, channels = teacher_frames.shape
        kernel_size = 2 * int(spatial_radius) + 1
        teacher_bt = teacher_frames.permute(0, 1, 4, 2, 3).reshape(batch_size * num_frames, channels, height, width)
        teacher_neighborhoods = F.unfold(teacher_bt, kernel_size=kernel_size, padding=int(spatial_radius))
        neighborhood_size = kernel_size * kernel_size
        teacher_neighborhoods = teacher_neighborhoods.reshape(
            batch_size, num_frames, channels, neighborhood_size, height, width
        ).permute(0, 1, 4, 5, 3, 2)

        tau = max(float(temporal_tau), 1e-6)
        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())

        def _accumulate(
            student_slice: torch.Tensor,
            teacher_slice: torch.Tensor,
            weight: float,
            weights_slice: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            similarities = (student_slice.unsqueeze(-2) * teacher_slice).sum(dim=-1)
            if patch_match_mode == "soft":
                temperature = max(float(patch_match_temperature), 1e-6)
                attn = torch.softmax(similarities / temperature, dim=-1)
                matched = (attn * similarities).sum(dim=-1)
            else:
                matched = similarities.max(dim=-1).values
            if weights_slice is None:
                return weight * matched.sum(), student_frames.new_tensor(matched.numel() * weight)
            local_weights = weights_slice.to(device=matched.device, dtype=matched.dtype)
            return weight * (matched * local_weights).sum(), weight * local_weights.sum()

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                delta_total, delta_norm = _accumulate(student_frames, teacher_neighborhoods, weight, motion_weights)
                total = total + delta_total
                normalizer = normalizer + delta_norm
                continue

            forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
            backward_weights = None if motion_weights is None else motion_weights[:, delta:]
            forward_total, forward_norm = _accumulate(
                student_frames[:, :-delta], teacher_neighborhoods[:, delta:], weight, forward_weights
            )
            backward_total, backward_norm = _accumulate(
                student_frames[:, delta:], teacher_neighborhoods[:, :-delta], weight, backward_weights
            )
            total = total + forward_total + backward_total
            normalizer = normalizer + forward_norm + backward_norm

        if normalizer.item() <= 0.0:
            return student_frames.new_zeros(())
        return total / normalizer

    @staticmethod
    def _multi_step_delta_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        delta_num_steps: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        max_step = min(max(1, int(delta_num_steps)), max(student_frames.shape[1] - 1, 0), max(teacher_frames.shape[1] - 1, 0))
        if max_step <= 0:
            return None

        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())
        tau = max(float(temporal_tau), 1e-6)

        for step in range(1, max_step + 1):
            weight = math.exp(-float(step - 1) / tau)
            student_delta = F.normalize(student_frames[:, step:] - student_frames[:, :-step], dim=-1)
            teacher_delta = F.normalize(teacher_frames[:, step:] - teacher_frames[:, :-step], dim=-1)
            cosine = F.cosine_similarity(student_delta, teacher_delta, dim=-1)
            step_weights = None if motion_weights is None else motion_weights[:, step:]
            if step_weights is None:
                total = total + weight * cosine.sum()
                normalizer = normalizer + student_frames.new_tensor(cosine.numel() * weight)
            else:
                cast_weights = step_weights.to(device=cosine.device, dtype=cosine.dtype)
                total = total + weight * (cosine * cast_weights).sum()
                normalizer = normalizer + weight * cast_weights.sum()

        if normalizer.item() <= 0.0:
            return None
        return total / normalizer

    def compute_loss_from_cached_features(
        self,
        *,
        num_latent_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.projector is None:
            raise RuntimeError("Self-Flow projector is not initialized")
        if not self.should_capture:
            return next(self.projector.parameters()).new_zeros(())

        student_feat = self._student_features
        teacher_feat = self._teacher_features
        cosine: Optional[torch.Tensor] = None
        if self.needs_video_features and (student_feat is None or teacher_feat is None):
            missing = []
            if student_feat is None:
                missing.append("student")
            if teacher_feat is None:
                missing.append("teacher")
            raise RuntimeError(f"Self-Flow native hidden-state capture is missing: {', '.join(missing)}")

        if student_feat is not None and teacher_feat is not None:
            projector_param = next(self.projector.parameters())
            if student_feat.device != projector_param.device or student_feat.dtype != projector_param.dtype:
                student_feat = student_feat.to(device=projector_param.device, dtype=projector_param.dtype)
            if teacher_feat.device != student_feat.device or teacher_feat.dtype != student_feat.dtype:
                teacher_feat = teacher_feat.to(device=student_feat.device, dtype=student_feat.dtype, non_blocking=True)
            if student_feat.shape[1] != teacher_feat.shape[1]:
                min_tokens = min(student_feat.shape[1], teacher_feat.shape[1])
                student_feat = student_feat[:, :min_tokens]
                teacher_feat = teacher_feat[:, :min_tokens]
                if token_mask is not None and token_mask.shape[1] > min_tokens:
                    token_mask = token_mask[:, :min_tokens]

            student_proj = self.projector(student_feat)
            teacher_feat = teacher_feat.detach()

            student_proj_norm = F.normalize(student_proj, dim=-1)
            teacher_norm = F.normalize(teacher_feat, dim=-1)

            # Optionally focus the rep loss on masked (higher-noise) tokens only.
            if self.config.mask_focus_loss and token_mask is not None:
                valid = token_mask.to(device=student_proj_norm.device)  # [B, T]
                feat_tokens = student_proj_norm.shape[1]
                mask_tokens = valid.shape[1]
                if mask_tokens != feat_tokens:
                    if mask_tokens > feat_tokens:
                        valid = valid[:, :feat_tokens]
                    else:
                        # mask shorter than features — only score tokens that are masked
                        student_proj_norm = student_proj_norm[:, :mask_tokens]
                        teacher_norm = teacher_norm[:, :mask_tokens]
                if valid.any():
                    cosine = F.cosine_similarity(student_proj_norm[valid], teacher_norm[valid], dim=-1).mean()
                else:
                    cosine = F.cosine_similarity(student_proj_norm, teacher_norm, dim=-1).mean()
            else:
                cosine = F.cosine_similarity(student_proj_norm, teacher_norm, dim=-1).mean()

        self._last_cosine = float(cosine.detach().item()) if cosine is not None else None
        step_has_active_loss = self.needs_video_features or (self._step_uses_audio and self._current_lambda_audio > 0.0)
        if not step_has_active_loss:
            similarity = self._last_cosine if self._last_cosine is not None else self._last_audio_cosine
            if similarity is not None:
                self._scheduler.update_similarity(similarity)
            return next(self.projector.parameters()).new_zeros(())

        loss = next(self.projector.parameters()).new_zeros(())
        applied_terms = 0
        if self._current_lambda_self_flow > 0.0:
            if cosine is None:
                raise RuntimeError("Self-Flow video loss is active but video cosine similarity is unavailable")
            loss = loss + self._loss_from_cosine(cosine) * self._current_lambda_self_flow
            applied_terms += 1

        # Audio representation alignment loss
        if self._step_uses_audio and self._current_lambda_audio > 0.0:
            if self.audio_projector is None:
                raise RuntimeError("Self-Flow audio loss is active but the audio projector is unavailable")
            if self._student_audio_features is None or self._teacher_audio_features is None:
                raise RuntimeError("Self-Flow audio loss is active but student or teacher audio features are missing")
            audio_student = self._student_audio_features
            audio_teacher = self._teacher_audio_features
            audio_projector_param = next(self.audio_projector.parameters())
            if audio_student.device != audio_projector_param.device or audio_student.dtype != audio_projector_param.dtype:
                audio_student = audio_student.to(
                    device=audio_projector_param.device,
                    dtype=audio_projector_param.dtype,
                )
            if audio_teacher.device != audio_student.device or audio_teacher.dtype != audio_student.dtype:
                audio_teacher = audio_teacher.to(device=audio_student.device, dtype=audio_student.dtype, non_blocking=True)
            if audio_student.shape[1] != audio_teacher.shape[1]:
                min_t = min(audio_student.shape[1], audio_teacher.shape[1])
                audio_student = audio_student[:, :min_t]
                audio_teacher = audio_teacher[:, :min_t]
            audio_proj = self.audio_projector(audio_student)
            audio_teacher = audio_teacher.detach()
            audio_proj_norm = F.normalize(audio_proj, dim=-1)
            audio_teacher_norm = F.normalize(audio_teacher, dim=-1)
            audio_cosine = F.cosine_similarity(audio_proj_norm, audio_teacher_norm, dim=-1).mean()
            self._last_audio_cosine = float(audio_cosine.detach().item())
            loss = loss + self._loss_from_cosine(audio_cosine) * self._current_lambda_audio
            applied_terms += 1

        temporal_mode = str(self.config.temporal_mode).lower()
        temporal_granularity = str(self.config.temporal_granularity).lower()
        motion_weighting = str(self.config.motion_weighting).lower()
        # Temporal losses operate in the original feature space (not projected) so that
        # frame-to-frame deltas reflect the transformer's actual spatiotemporal representations
        # rather than the learned projection.
        temporal_student = self._reshape_temporal_features(student_feat, num_latent_frames) if student_feat is not None else None
        temporal_teacher = self._reshape_temporal_features(teacher_feat, num_latent_frames) if teacher_feat is not None else None
        temporal_student_grid = (
            self._reshape_temporal_grid(
                student_feat,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            if student_feat is not None
            else None
        )
        temporal_teacher_grid = (
            self._reshape_temporal_grid(
                teacher_feat,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            if teacher_feat is not None
            else None
        )
        temporal_motion_weights = None
        temporal_motion_grid_weights = None
        if motion_weighting == "teacher_delta":
            if temporal_teacher is not None:
                temporal_motion_weights = self._teacher_delta_motion_weights(
                    temporal_teacher,
                    strength=self.config.motion_weight_strength,
                )
            if temporal_teacher_grid is not None:
                temporal_motion_grid_weights = self._teacher_delta_motion_weights(
                    temporal_teacher_grid,
                    strength=self.config.motion_weight_strength,
                )

        if temporal_mode in {"frame", "hybrid"} and self.current_lambda_temporal > 0.0:
            if temporal_student is None or temporal_teacher is None:
                raise RuntimeError("Self-Flow temporal frame loss could not reshape the captured video features")
            if temporal_granularity == "patch":
                if temporal_student_grid is not None and temporal_teacher_grid is not None:
                    student_frames = F.normalize(temporal_student_grid, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher_grid, dim=-1)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames,
                        teacher_frames,
                        num_neighbors=self.config.num_neighbors,
                        temporal_tau=self.config.temporal_tau,
                        spatial_radius=self.config.patch_spatial_radius,
                        patch_match_mode=self.config.patch_match_mode,
                        patch_match_temperature=self.config.patch_match_temperature,
                        motion_weights=temporal_motion_grid_weights,
                    )
                else:
                    student_frames = F.normalize(temporal_student, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher, dim=-1)
                    flat_motion_weights = None
                    if temporal_motion_weights is not None:
                        flat_motion_weights = temporal_motion_weights.mean(dim=-1).unsqueeze(2)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames.reshape(
                            student_frames.shape[0], student_frames.shape[1], 1, student_frames.shape[2], student_frames.shape[3]
                        ),
                        teacher_frames.reshape(
                            teacher_frames.shape[0], teacher_frames.shape[1], 1, teacher_frames.shape[2], teacher_frames.shape[3]
                        ),
                        num_neighbors=self.config.num_neighbors,
                        temporal_tau=self.config.temporal_tau,
                        spatial_radius=0,
                        patch_match_mode=self.config.patch_match_mode,
                        patch_match_temperature=self.config.patch_match_temperature,
                        motion_weights=flat_motion_weights,
                    )
            else:
                student_frames = F.normalize(temporal_student.mean(dim=2), dim=-1)
                teacher_frames = F.normalize(temporal_teacher.mean(dim=2), dim=-1)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                frame_cosine = self._neighbor_weighted_cosine(
                    student_frames,
                    teacher_frames,
                    num_neighbors=self.config.num_neighbors,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            self._last_frame_cosine = float(frame_cosine.detach().item())
            loss = loss + self._loss_from_cosine(frame_cosine) * self.current_lambda_temporal
            applied_terms += 1

        if temporal_mode in {"delta", "hybrid"} and self.current_lambda_delta > 0.0:
            if temporal_student is None or temporal_teacher is None:
                raise RuntimeError("Self-Flow temporal delta loss could not reshape the captured video features")
            if temporal_student.shape[1] <= 1 or temporal_teacher.shape[1] <= 1:
                raise RuntimeError("Self-Flow temporal delta loss requires at least two latent frames")
            if temporal_granularity == "patch":
                delta_cosine = self._multi_step_delta_cosine(
                    temporal_student,
                    temporal_teacher,
                    delta_num_steps=self.config.delta_num_steps,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=temporal_motion_weights,
                )
            else:
                student_frames = temporal_student.mean(dim=2)
                teacher_frames = temporal_teacher.mean(dim=2)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                delta_cosine = self._multi_step_delta_cosine(
                    student_frames,
                    teacher_frames,
                    delta_num_steps=self.config.delta_num_steps,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            if delta_cosine is None or not torch.isfinite(delta_cosine):
                raise RuntimeError("Self-Flow temporal delta cosine is unavailable or non-finite")
            self._last_delta_cosine = float(delta_cosine.detach().item())
            loss = loss + self._loss_from_cosine(delta_cosine) * self.current_lambda_delta
            applied_terms += 1

        if applied_terms == 0:
            raise RuntimeError("Self-Flow has positive scheduled weights but produced no loss terms")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Self-Flow loss is non-finite: {loss.detach().item():.4g}")

        max_loss = float(self.config.max_loss)
        if max_loss > 0.0:
            loss_abs = float(loss.detach().abs().item())
            if loss_abs > max_loss:
                loss = loss * (max_loss / loss_abs)

        similarity = self._last_cosine if self._last_cosine is not None else self._last_audio_cosine
        if similarity is not None:
            self._scheduler.update_similarity(similarity)
        return loss

    def compute_loss(self, **_kwargs) -> Optional[torch.Tensor]:
        # Backward-compatible shim: loss is now computed from already-cached student/teacher features.
        return self.compute_loss_from_cached_features(
            num_latent_frames=_kwargs.get("num_latent_frames"),
            latent_height=_kwargs.get("latent_height"),
            latent_width=_kwargs.get("latent_width"),
            token_mask=_kwargs.get("token_mask"),
        )

    def remove_hooks(self) -> None:
        self.cleanup_step()

    def state_dict(self) -> Dict[str, Any]:
        if self.projector is None:
            return {}
        sd = self.projector.state_dict()
        if self.audio_projector is not None:
            for k, v in self.audio_projector.state_dict().items():
                sd[f"audio.{k}"] = v
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if self.projector is not None and sd:
            # Split video and audio projector weights
            video_sd = {k: v for k, v in sd.items() if not k.startswith("audio.")}
            audio_sd = {k[len("audio.") :]: v for k, v in sd.items() if k.startswith("audio.")}
            if video_sd:
                self.projector.load_state_dict(video_sd)
                logger.info("Self-Flow: loaded video projector weights (%d tensors)", len(video_sd))
            if audio_sd and self.audio_projector is not None:
                self.audio_projector.load_state_dict(audio_sd)
                logger.info("Self-Flow: loaded audio projector weights (%d tensors)", len(audio_sd))
            elif audio_sd and self.audio_projector is None:
                logger.warning(
                    "Self-Flow: checkpoint contains audio projector weights (%d tensors) but the "
                    "current run has no audio projector (lambda_audio=0 or model lacks audio_inner_dim); "
                    "these weights are being ignored.",
                    len(audio_sd),
                )
            elif not audio_sd and self.audio_projector is not None:
                logger.warning(
                    "Self-Flow: audio projector is active (lambda_audio>0) but the checkpoint contains "
                    "no audio projector weights; the audio projector is starting from random init "
                    "(optimizer state may not correspond)."
                )

    def teacher_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {"__self_flow_step_counter__": torch.tensor([int(self._step_counter)], dtype=torch.int64)}
        if self._scheduler.similarity_ema is not None:
            out["__self_flow_similarity_ema__"] = torch.tensor(
                [float(self._scheduler.similarity_ema)],
                dtype=torch.float64,
            )
        out["__self_flow_cutoff_latched__"] = torch.tensor(
            [int(self._scheduler.cutoff_latched)],
            dtype=torch.int64,
        )
        for name, tensor in self._shadow_params.items():
            out[f"shadow::{name}"] = tensor.detach().clone().to(device="cpu")
        return out

    def load_teacher_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        carries_ema_teacher = any(str(key).startswith("shadow::") for key in sd)
        if str(self.config.teacher_mode).lower() == "base" and carries_ema_teacher:
            raise ValueError(
                "Self-Flow: this checkpoint carries an EMA teacher (trained with teacher_mode=ema/"
                "partial_ema), but the current run uses teacher_mode=base. Resuming as base "
                "would silently change the training objective and discard the EMA teacher. Pass "
                "teacher_mode=ema (or partial_ema) to continue the canonical run, or delete "
                "self_flow_teacher_ema.safetensors from the resume directory to intentionally switch to base."
            )
        step_tensor = sd.get("__self_flow_step_counter__")
        if isinstance(step_tensor, torch.Tensor) and step_tensor.numel() > 0:
            self._step_counter = int(step_tensor.flatten()[0].item())
        similarity_tensor = sd.get("__self_flow_similarity_ema__")
        if isinstance(similarity_tensor, torch.Tensor) and similarity_tensor.numel() > 0:
            self._scheduler.similarity_ema = float(similarity_tensor.flatten()[0].item())
        cutoff_tensor = sd.get("__self_flow_cutoff_latched__")
        if isinstance(cutoff_tensor, torch.Tensor) and cutoff_tensor.numel() > 0:
            self._scheduler.cutoff_latched = bool(cutoff_tensor.flatten()[0].item())

        restored: Dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not key.startswith("shadow::"):
                continue
            restored[key[len("shadow::") :]] = value.detach().clone()

        if restored:
            self._shadow_params = restored
            logger.info("Self-Flow: loaded EMA teacher state (%d tensors)", len(restored))
