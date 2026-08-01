from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.networks import lora_ltx2
from musubi_tuner.training.metadata import (
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_MODULE,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

RankMode = Literal["fixed", "fro", "quantile", "knee", "relative_drop"]
ExtractMode = Literal["lora", "dora"]
UnsupportedMode = Literal["report", "skip", "error", "sidecar"]


@dataclass
class ExtractedModuleReport:
    checkpoint_key: str
    module_path: str
    lora_name: str
    shape: list[int]
    rank: int
    retained_energy: float
    delta_norm: float
    residual_norm: float
    relative_error: float


@dataclass
class SkippedTensorReport:
    checkpoint_key: str
    shape: list[int]
    reason: str
    max_abs_diff: Optional[float] = None


@dataclass
class ExtractionSummary:
    base_model: str
    finetuned_model: str
    save_to: str
    extract_mode: str
    rank_mode: str
    target_preset: str
    scanned_common_tensors: int = 0
    extracted_modules: int = 0
    skipped_tensors: int = 0
    unsupported_changed_tensors: int = 0
    potential_unsupported_tensors: int = 0
    dry_run_extractable_tensors: int = 0
    sidecar_delta_tensors: int = 0
    max_rank_used: int = 0
    reports: list[ExtractedModuleReport] = field(default_factory=list)
    skipped: list[SkippedTensorReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionConfig:
    base_model: str
    finetuned_model: str
    save_to: str
    target_preset: str = "full"
    include_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    connector_lora: bool = False
    extract_mode: ExtractMode = "lora"
    rank_mode: RankMode = "fro"
    dim: int = 64
    min_rank: int = 1
    max_rank: int = 128
    fro_target: float = 0.98
    quantile: float = 0.98
    relative_drop_threshold: float = 0.25
    clamp_quantile: float = 0.0
    min_diff: float = 0.0
    svd_backend: str = "full"
    lowrank_oversample: int = 8
    lowrank_niter: int = 2
    device: Optional[str] = None
    save_precision: Optional[str] = "bf16"
    mem_eff_safe_open: bool = False
    unsupported_tensors: UnsupportedMode = "report"
    dry_run: bool = False
    report_json: str = ""
    max_modules: int = 0
    no_metadata: bool = False


def dtype_from_name(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None or name == "":
        return None
    normalized = str(name).lower()
    if normalized in {"float", "fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {name}")


def _parse_pattern_arg(value: Optional[str]) -> Optional[list[str]]:
    if value is None or str(value).strip() == "":
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Pattern arguments must evaluate to a list of strings")
    return parsed


def _compile_patterns(patterns: Optional[Iterable[str]]) -> Optional[list[re.Pattern[str]]]:
    if patterns is None:
        return None
    return [re.compile(pattern) for pattern in patterns]


def _default_exclude_patterns(connector_lora: bool) -> list[str]:
    patterns = [
        r".*text_embedding_projection\.aggregate_embed.*",
        r".*text_embedding_projection\.video_aggregate_embed.*",
        r".*text_embedding_projection\.audio_aggregate_embed.*",
    ]
    if not connector_lora:
        patterns.extend(
            [
                r".*embeddings_connector\..*",
                r".*audio_embeddings_connector\..*",
            ]
        )
    return patterns


def _patterns_for_config(config: ExtractionConfig) -> tuple[Optional[list[re.Pattern[str]]], list[re.Pattern[str]]]:
    if config.include_patterns is not None:
        include_patterns = config.include_patterns
    elif config.target_preset == "custom":
        include_patterns = None
    else:
        if config.target_preset not in lora_ltx2.LTX2_LORA_TARGET_PRESETS:
            valid = ", ".join(sorted(lora_ltx2.LTX2_LORA_TARGET_PRESETS.keys()))
            raise ValueError(f"Unknown target preset {config.target_preset!r}. Valid presets: {valid}, custom")
        include_patterns = lora_ltx2.LTX2_LORA_TARGET_PRESETS[config.target_preset]

    exclude_patterns = _default_exclude_patterns(config.connector_lora)
    if config.exclude_patterns:
        exclude_patterns.extend(config.exclude_patterns)

    return _compile_patterns(include_patterns), _compile_patterns(exclude_patterns) or []


def _normalize_checkpoint_key(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    if key.startswith("model."):
        tail = key[len("model.") :]
        if tail.startswith(("transformer_blocks.", "video_embeddings_connector.", "audio_embeddings_connector.")):
            return tail
    return key


def _strip_tensor_suffix(key: str) -> Optional[tuple[str, str]]:
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix[1:]
    return None


def module_path_from_unified_key(unified_key: str) -> Optional[str]:
    stripped = _strip_tensor_suffix(unified_key)
    if stripped is None:
        return None
    base, _kind = stripped
    if base.startswith("model.transformer_blocks."):
        return base
    if base.startswith("transformer_blocks."):
        return f"model.{base}"
    if base.startswith("video_embeddings_connector."):
        return "embeddings_connector." + base[len("video_embeddings_connector.") :]
    if base.startswith("embeddings_connector."):
        return base
    if base.startswith("audio_embeddings_connector."):
        return base
    return None


def lora_name_from_module_path(module_path: str) -> str:
    return "lora_unet_" + module_path.replace(".", "_")


def _module_allowed(
    module_path: str,
    include_patterns: Optional[list[re.Pattern[str]]],
    exclude_patterns: list[re.Pattern[str]],
) -> bool:
    included = include_patterns is None or any(pattern.fullmatch(module_path) for pattern in include_patterns)
    excluded = any(pattern.fullmatch(module_path) for pattern in exclude_patterns)
    if excluded and not included:
        return False
    if include_patterns is not None and not included:
        return False
    return True


def _open_safetensors(path: str, mem_eff: bool):
    if mem_eff:
        return MemoryEfficientSafeOpen(path)
    return safe_open(path, framework="pt")


def _reader_shape(reader, key: str) -> tuple[int, ...]:
    get_slice = getattr(reader, "get_slice", None)
    if callable(get_slice):
        try:
            return tuple(get_slice(key).get_shape())
        except Exception:
            pass
    return tuple(reader.get_tensor(key).shape)


def _build_key_map(reader) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in reader.keys():
        mapping[_normalize_checkpoint_key(str(key))] = str(key)
    return mapping


def _rank_from_energy(values: torch.Tensor, target: float, squared: bool) -> int:
    if values.numel() == 0:
        return 0
    masses = values.square() if squared else values
    total = masses.sum()
    if float(total.item()) <= 0.0:
        return 1
    cumulative = torch.cumsum(masses, dim=0) / total
    idx = int(torch.searchsorted(cumulative, torch.tensor(float(target), device=cumulative.device)).item())
    return idx + 1


def _rank_from_knee(values: torch.Tensor) -> int:
    n = int(values.numel())
    if n <= 2:
        return n
    y = values.detach().float()
    if float(y.max().item()) <= 0.0:
        return 1
    x = torch.linspace(0.0, 1.0, n, device=y.device)
    y = (y - y[-1]) / (y[0] - y[-1]).clamp_min(1e-12)
    line = 1.0 - x
    distances = torch.abs(y - line)
    return int(torch.argmax(distances).item()) + 1


def select_rank(singular_values: torch.Tensor, config: ExtractionConfig) -> int:
    max_possible = int(singular_values.numel())
    if max_possible <= 0:
        return 0

    if config.rank_mode == "fixed":
        rank = int(config.dim)
    elif config.rank_mode == "fro":
        rank = _rank_from_energy(singular_values, config.fro_target, squared=True)
    elif config.rank_mode == "quantile":
        rank = _rank_from_energy(singular_values, config.quantile, squared=False)
    elif config.rank_mode == "knee":
        rank = _rank_from_knee(singular_values)
    elif config.rank_mode == "relative_drop":
        if singular_values.numel() <= 1:
            rank = 1
        else:
            head = singular_values[:-1].clamp_min(1e-12)
            drops = (singular_values[:-1] - singular_values[1:]) / head
            matches = torch.nonzero(drops >= float(config.relative_drop_threshold), as_tuple=False)
            rank = int(matches[0].item()) + 1 if matches.numel() else max_possible
    else:
        raise ValueError(f"Unsupported rank mode: {config.rank_mode}")

    rank = max(int(config.min_rank), rank)
    rank = min(int(config.max_rank), rank, max_possible)
    return max(1, rank)


def _flatten_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError("LoRA extraction requires tensors with at least 2 dimensions")
    return weight.reshape(weight.shape[0], -1)


def _dora_magnitude_from_norm(norm: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    if len(original_shape) == 2:
        return norm
    return norm.view((1, int(original_shape[0])) + (1,) * (len(original_shape) - 2))


def _factor_shapes(up_flat: torch.Tensor, down_flat: torch.Tensor, original_shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
    if len(original_shape) == 2:
        return up_flat.contiguous(), down_flat.contiguous()
    out_dim = int(original_shape[0])
    in_dim = int(original_shape[1])
    kernel_shape = tuple(int(dim) for dim in original_shape[2:])
    rank = int(down_flat.shape[0])
    up = up_flat.reshape(out_dim, rank, *([1] * len(kernel_shape))).contiguous()
    down = down_flat.reshape(rank, in_dim, *kernel_shape).contiguous()
    return up, down


def _compute_svd(
    matrix: torch.Tensor, config: ExtractionConfig, rank_hint: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.svd_backend == "lowrank":
        if rank_hint is None:
            rank_hint = min(int(config.dim), int(config.max_rank), min(matrix.shape))
        q = min(min(matrix.shape), int(rank_hint) + max(0, int(config.lowrank_oversample)))
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=int(config.lowrank_niter))
        return u, s, v.transpose(0, 1)
    if config.svd_backend != "full":
        raise ValueError("--svd_backend must be full or lowrank")
    return torch.linalg.svd(matrix, full_matrices=False)


def _clamp_factors(up: torch.Tensor, down: torch.Tensor, clamp_quantile: float) -> tuple[torch.Tensor, torch.Tensor]:
    if clamp_quantile <= 0.0 or clamp_quantile >= 1.0:
        return up, down
    dist = torch.cat([up.flatten(), down.flatten()])
    hi = torch.quantile(dist.abs(), float(clamp_quantile))
    if float(hi.item()) <= 0.0:
        return up, down
    return up.clamp(-hi, hi), down.clamp(-hi, hi)


def _extract_factors(
    base_weight: torch.Tensor,
    tuned_weight: torch.Tensor,
    config: ExtractionConfig,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, float, float, float, float]:
    base_flat = _flatten_weight(base_weight).to(torch.float32)
    tuned_flat = _flatten_weight(tuned_weight).to(torch.float32)
    dora_magnitude = None

    if config.extract_mode == "dora":
        base_norm = torch.linalg.vector_norm(base_flat, dim=1)
        tuned_norm = torch.linalg.vector_norm(tuned_flat, dim=1)
        eps = 1e-12
        direction_target = tuned_flat * (base_norm / tuned_norm.clamp_min(eps)).unsqueeze(1)
        delta_matrix = direction_target - base_flat
        dora_magnitude = _dora_magnitude_from_norm(tuned_norm, tuned_weight.shape)
    elif config.extract_mode == "lora":
        delta_matrix = tuned_flat - base_flat
    else:
        raise ValueError(f"Unsupported extract mode: {config.extract_mode}")

    rank_hint = int(config.dim) if config.rank_mode == "fixed" else int(config.max_rank)
    u, s, vh = _compute_svd(delta_matrix, config, rank_hint=rank_hint)
    rank = select_rank(s, config)

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    up_flat = u_r @ torch.diag(s_r)
    down_flat = vh_r
    up_flat, down_flat = _clamp_factors(up_flat, down_flat, config.clamp_quantile)

    approx = up_flat @ down_flat
    delta_norm = float(torch.linalg.vector_norm(delta_matrix).item())
    residual_norm = float(torch.linalg.vector_norm(delta_matrix - approx).item())
    relative_error = residual_norm / max(delta_norm, 1e-12)
    total_energy = torch.sum(s.square())
    retained_energy = float(torch.sum(s[:rank].square()).div(total_energy.clamp_min(1e-30)).item())

    up, down = _factor_shapes(up_flat, down_flat, tuned_weight.shape)
    return up, down, dora_magnitude, rank, retained_energy, delta_norm, residual_norm, relative_error


def _tensor_max_abs_diff(base: torch.Tensor, tuned: torch.Tensor) -> float:
    if base.shape != tuned.shape:
        return math.inf
    if not base.is_floating_point():
        base = base.to(torch.float32)
    if not tuned.is_floating_point():
        tuned = tuned.to(torch.float32)
    return float((tuned.to(torch.float32) - base.to(torch.float32)).abs().max().item())


def _save_report(path: str, summary: ExtractionSummary) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _metadata_for_output(config: ExtractionConfig, summary: ExtractionSummary) -> Optional[dict[str, str]]:
    if config.no_metadata:
        return None
    network_args: dict[str, str] = {}
    if config.extract_mode == "dora":
        network_args["use_dora"] = "true"
    if config.connector_lora:
        network_args["connector_lora"] = "true"
    if config.target_preset and config.target_preset != "custom":
        network_args["lora_target_preset"] = config.target_preset

    return {
        "title": Path(config.save_to).stem,
        "created_at": str(int(time.time())),
        SS_METADATA_KEY_NETWORK_MODULE: "networks.lora_ltx2",
        SS_METADATA_KEY_NETWORK_DIM: str(summary.max_rank_used or config.max_rank),
        SS_METADATA_KEY_NETWORK_ALPHA: "rank",
        SS_METADATA_KEY_NETWORK_ARGS: json.dumps(network_args),
        "musubi_ltx2_extractor": "svd",
        "musubi_ltx2_extract_mode": config.extract_mode,
        "musubi_ltx2_rank_mode": config.rank_mode,
        "musubi_ltx2_target_preset": config.target_preset,
    }


def extract_lora(config: ExtractionConfig) -> ExtractionSummary:
    save_dtype = dtype_from_name(config.save_precision)
    device = torch.device(config.device) if config.device else torch.device("cpu")
    include_patterns, exclude_patterns = _patterns_for_config(config)
    summary = ExtractionSummary(
        base_model=config.base_model,
        finetuned_model=config.finetuned_model,
        save_to=config.save_to,
        extract_mode=config.extract_mode,
        rank_mode=config.rank_mode,
        target_preset=config.target_preset,
    )

    lora_state: dict[str, torch.Tensor] = {}

    with (
        _open_safetensors(config.base_model, config.mem_eff_safe_open) as base_reader,
        _open_safetensors(config.finetuned_model, config.mem_eff_safe_open) as tuned_reader,
    ):
        base_map = _build_key_map(base_reader)
        tuned_map = _build_key_map(tuned_reader)
        common_keys = sorted(set(base_map).intersection(tuned_map))
        summary.scanned_common_tensors = len(common_keys)

        for unified_key in common_keys:
            module_path = module_path_from_unified_key(unified_key)
            if module_path is None:
                continue

            base_key = base_map[unified_key]
            tuned_key = tuned_map[unified_key]
            shape = _reader_shape(tuned_reader, tuned_key)
            stripped = _strip_tensor_suffix(unified_key)
            tensor_kind = stripped[1] if stripped is not None else ""

            allowed = _module_allowed(module_path, include_patterns, exclude_patterns)
            is_extractable = tensor_kind == "weight" and len(shape) >= 2 and allowed

            if not is_extractable:
                if tensor_kind in {"weight", "bias"} and allowed:
                    if config.dry_run:
                        summary.skipped.append(SkippedTensorReport(unified_key, list(shape), "unsupported_shape_or_kind"))
                        summary.potential_unsupported_tensors += 1
                    elif config.unsupported_tensors != "skip":
                        base_tensor = base_reader.get_tensor(base_key)
                        tuned_tensor = tuned_reader.get_tensor(tuned_key)
                        max_abs = _tensor_max_abs_diff(base_tensor, tuned_tensor)
                        if max_abs > float(config.min_diff):
                            summary.unsupported_changed_tensors += 1
                            summary.skipped.append(
                                SkippedTensorReport(unified_key, list(shape), "unsupported_changed_tensor", max_abs)
                            )
                            if config.unsupported_tensors == "sidecar":
                                sidecar_key = "musubi_extracted_delta." + unified_key
                                lora_state[sidecar_key] = (tuned_tensor.to(torch.float32) - base_tensor.to(torch.float32)).cpu()
                                summary.sidecar_delta_tensors += 1
                            elif config.unsupported_tensors == "error":
                                raise ValueError(f"Unsupported changed tensor cannot be represented as pure LoRA: {unified_key}")
                continue

            if config.max_modules and summary.extracted_modules >= int(config.max_modules):
                continue

            if config.dry_run:
                summary.skipped.append(SkippedTensorReport(unified_key, list(shape), "dry_run_extractable"))
                summary.dry_run_extractable_tensors += 1
                continue

            base_weight = base_reader.get_tensor(base_key)
            tuned_weight = tuned_reader.get_tensor(tuned_key)
            if base_weight.shape != tuned_weight.shape:
                summary.skipped.append(SkippedTensorReport(unified_key, list(shape), "shape_mismatch"))
                continue

            max_abs = _tensor_max_abs_diff(base_weight, tuned_weight)
            if max_abs <= float(config.min_diff):
                summary.skipped.append(SkippedTensorReport(unified_key, list(shape), "unchanged_or_below_min_diff", max_abs))
                continue

            up, down, dora_magnitude, rank, retained_energy, delta_norm, residual_norm, relative_error = _extract_factors(
                base_weight.to(device), tuned_weight.to(device), config
            )

            lora_name = lora_name_from_module_path(module_path)
            if save_dtype is not None:
                up = up.to(save_dtype)
                down = down.to(save_dtype)
                if dora_magnitude is not None:
                    dora_magnitude = dora_magnitude.to(save_dtype)

            lora_state[f"{lora_name}.lora_down.weight"] = down.detach().cpu().contiguous()
            lora_state[f"{lora_name}.lora_up.weight"] = up.detach().cpu().contiguous()
            lora_state[f"{lora_name}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
            if dora_magnitude is not None:
                lora_state[f"{lora_name}.lora_magnitude_vector.weight"] = dora_magnitude.detach().cpu().contiguous()

            summary.max_rank_used = max(summary.max_rank_used, rank)
            summary.extracted_modules += 1
            summary.reports.append(
                ExtractedModuleReport(
                    checkpoint_key=unified_key,
                    module_path=module_path,
                    lora_name=lora_name,
                    shape=list(shape),
                    rank=rank,
                    retained_energy=retained_energy,
                    delta_norm=delta_norm,
                    residual_norm=residual_norm,
                    relative_error=relative_error,
                )
            )

    summary.skipped_tensors = len(summary.skipped)
    _save_report(config.report_json, summary)

    if config.dry_run:
        logger.info("Dry run complete: %d extractable tensors identified", summary.dry_run_extractable_tensors)
        return summary

    if not lora_state:
        raise ValueError("No LoRA weights were extracted")

    save_path = Path(config.save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state, str(save_path), metadata=_metadata_for_output(config, summary))
    logger.info("Saved %d extracted modules to %s", summary.extracted_modules, save_path)
    return summary


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract an LTX-2 LoRA/DoRA adapter from a full fine-tuned checkpoint.")
    parser.add_argument("--base_model", required=True, help="Base LTX-2 safetensors checkpoint")
    parser.add_argument("--finetuned_model", required=True, help="Fine-tuned LTX-2 safetensors checkpoint")
    parser.add_argument("--save_to", required=True, help="Output native Musubi LoRA safetensors path")
    parser.add_argument("--target_preset", default="full", help="LTX-2 target preset or custom")
    parser.add_argument("--include_patterns", default=None, help="Python list of regex patterns, e.g. \"['.*to_q$']\"")
    parser.add_argument("--exclude_patterns", default=None, help="Python list of regex patterns")
    parser.add_argument("--connector_lora", action="store_true", help="Allow video/audio embedding connector LoRA extraction")
    parser.add_argument("--extract_mode", default="lora", choices=["lora", "dora"])
    parser.add_argument("--rank_mode", default="fro", choices=["fixed", "fro", "quantile", "knee", "relative_drop"])
    parser.add_argument("--dim", type=int, default=64, help="Fixed rank, or low-rank SVD hint")
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=128)
    parser.add_argument("--fro_target", type=float, default=0.98)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--relative_drop_threshold", type=float, default=0.25)
    parser.add_argument("--clamp_quantile", type=float, default=0.0, help="0 disables factor clamping")
    parser.add_argument("--min_diff", type=float, default=0.0)
    parser.add_argument("--svd_backend", default="full", choices=["full", "lowrank"])
    parser.add_argument("--lowrank_oversample", type=int, default=8)
    parser.add_argument("--lowrank_niter", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_precision", default="bf16", choices=["float", "fp32", "fp16", "bf16", None])
    parser.add_argument("--mem_eff_safe_open", action="store_true")
    parser.add_argument("--unsupported_tensors", default="report", choices=["report", "skip", "error", "sidecar"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--report_json", default="")
    parser.add_argument(
        "--max_modules",
        type=int,
        default=0,
        help="Limit the number of selected modules to extract; 0 extracts all selected modules",
    )
    parser.add_argument("--no_metadata", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExtractionConfig:
    return ExtractionConfig(
        base_model=args.base_model,
        finetuned_model=args.finetuned_model,
        save_to=args.save_to,
        target_preset=args.target_preset,
        include_patterns=_parse_pattern_arg(args.include_patterns),
        exclude_patterns=_parse_pattern_arg(args.exclude_patterns),
        connector_lora=bool(args.connector_lora),
        extract_mode=args.extract_mode,
        rank_mode=args.rank_mode,
        dim=int(args.dim),
        min_rank=int(args.min_rank),
        max_rank=int(args.max_rank),
        fro_target=float(args.fro_target),
        quantile=float(args.quantile),
        relative_drop_threshold=float(args.relative_drop_threshold),
        clamp_quantile=float(args.clamp_quantile),
        min_diff=float(args.min_diff),
        svd_backend=args.svd_backend,
        lowrank_oversample=int(args.lowrank_oversample),
        lowrank_niter=int(args.lowrank_niter),
        device=args.device,
        save_precision=args.save_precision,
        mem_eff_safe_open=bool(args.mem_eff_safe_open),
        unsupported_tensors=args.unsupported_tensors,
        dry_run=bool(args.dry_run),
        report_json=args.report_json,
        max_modules=int(args.max_modules),
        no_metadata=bool(args.no_metadata),
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = setup_parser()
    args = parser.parse_args(argv)
    summary = extract_lora(config_from_args(args))
    payload = summary.to_dict()
    payload["reports"] = f"{len(summary.reports)} entries; use --report_json for details"
    payload["skipped"] = f"{len(summary.skipped)} entries; use --report_json for details"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
