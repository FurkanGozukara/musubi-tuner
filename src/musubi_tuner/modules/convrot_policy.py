"""Per-layer storage and compute policy for LTX-2 ConvRot weights."""

from __future__ import annotations

import fnmatch
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONVROT_POLICY_FORMAT = "ltx2_convrot_policy_v1"
CONVROT_COMPUTE_MODES = frozenset({"quantized", "dequantize"})
CONVROT_INT4_POLICY_KEYS = frozenset({"group_scales", "group_ratio_q8", "scale_refine_steps"})


def _module_name(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def _validate_compute(value: Any, *, where: str) -> str:
    compute = str(value).strip().lower()
    if compute not in CONVROT_COMPUTE_MODES:
        expected = ", ".join(sorted(CONVROT_COMPUTE_MODES))
        raise ValueError(f"{where}.compute must be one of {expected}, got {value!r}")
    return compute


def _validate_optional_group_scales(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}.group_scales must be an integer")
    if value < 0 or (value and (value < 16 or value & (value - 1))):
        raise ValueError(f"{where}.group_scales must be 0 or a power of two >= 16")
    return int(value)


def _validate_optional_nonnegative_int(value: Any, *, field: str, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}.{field} must be an integer")
    if value < 0:
        raise ValueError(f"{where}.{field} must be >= 0")
    return int(value)


def _validate_optional_bool(value: Any, *, field: str, where: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{where}.{field} must be a boolean")
    return value


@dataclass(frozen=True)
class ConvRotPolicyDecision:
    quantize: bool = True
    compute: str = "quantized"
    group_scales: int | None = None
    group_ratio_q8: bool | None = None
    scale_refine_steps: int | None = None
    matched_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConvRotInt4Parameters:
    group_scales: int = 0
    group_ratio_q8: bool = False
    scale_refine_steps: int = 0


@dataclass(frozen=True)
class ConvRotPolicyRule:
    pattern: str
    quantize: bool | None = None
    compute: str | None = None
    group_scales: int | None = None
    group_ratio_q8: bool | None = None
    scale_refine_steps: int | None = None

    def matches(self, module_name: str) -> bool:
        return fnmatch.fnmatchcase(module_name, self.pattern)


@dataclass(frozen=True)
class ConvRotPolicy:
    default_quantize: bool = True
    default_compute: str = "quantized"
    default_group_scales: int | None = None
    default_group_ratio_q8: bool | None = None
    default_scale_refine_steps: int | None = None
    rules: tuple[ConvRotPolicyRule, ...] = ()

    def resolve(self, name: str) -> ConvRotPolicyDecision:
        module_name = _module_name(name)
        quantize = self.default_quantize
        compute = self.default_compute
        group_scales = self.default_group_scales
        group_ratio_q8 = self.default_group_ratio_q8
        scale_refine_steps = self.default_scale_refine_steps
        matched: list[str] = []
        for rule in self.rules:
            if not rule.matches(module_name):
                continue
            matched.append(rule.pattern)
            if rule.quantize is not None:
                quantize = rule.quantize
            if rule.compute is not None:
                compute = rule.compute
            if rule.group_scales is not None:
                group_scales = rule.group_scales
            if rule.group_ratio_q8 is not None:
                group_ratio_q8 = rule.group_ratio_q8
            if rule.scale_refine_steps is not None:
                scale_refine_steps = rule.scale_refine_steps
        return ConvRotPolicyDecision(
            quantize=quantize,
            compute=compute,
            group_scales=group_scales,
            group_ratio_q8=group_ratio_q8,
            scale_refine_steps=scale_refine_steps,
            matched_patterns=tuple(matched),
        )

    def has_int4_quantization_parameters(self) -> bool:
        if (
            self.default_group_scales is not None
            or self.default_group_ratio_q8 is not None
            or self.default_scale_refine_steps is not None
        ):
            return True
        return any(
            rule.group_scales is not None or rule.group_ratio_q8 is not None or rule.scale_refine_steps is not None
            for rule in self.rules
        )


def resolve_int4_policy_parameters(
    decision: ConvRotPolicyDecision | None,
    *,
    group_scales: int,
    group_ratio_q8: bool,
    scale_refine_steps: int,
    name: str,
) -> ConvRotInt4Parameters:
    """Overlay explicit per-layer policy values on the caller's CLI values."""

    resolved_group_scales = int(group_scales if decision is None or decision.group_scales is None else decision.group_scales)
    resolved_group_ratio_q8 = bool(
        group_ratio_q8 if decision is None or decision.group_ratio_q8 is None else decision.group_ratio_q8
    )
    resolved_scale_refine_steps = int(
        scale_refine_steps if decision is None or decision.scale_refine_steps is None else decision.scale_refine_steps
    )
    _validate_optional_group_scales(resolved_group_scales, where=name)
    _validate_optional_nonnegative_int(resolved_scale_refine_steps, field="scale_refine_steps", where=name)
    if resolved_group_ratio_q8 and not resolved_group_scales:
        raise ValueError(f"{name}: group_ratio_q8=true requires group_scales > 0")
    return ConvRotInt4Parameters(
        group_scales=resolved_group_scales,
        group_ratio_q8=resolved_group_ratio_q8,
        scale_refine_steps=resolved_scale_refine_steps,
    )


def parse_convrot_policy(data: Mapping[str, Any]) -> ConvRotPolicy:
    if not isinstance(data, Mapping):
        raise TypeError("ConvRot policy must be a JSON object")
    allowed_top = {"format", "defaults", "rules"}
    unknown_top = sorted(set(data) - allowed_top)
    if unknown_top:
        raise ValueError(f"Unknown ConvRot policy keys: {', '.join(unknown_top)}")
    if data.get("format") != CONVROT_POLICY_FORMAT:
        raise ValueError(f"ConvRot policy format must be {CONVROT_POLICY_FORMAT!r}")

    defaults = data.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("ConvRot policy defaults must be a JSON object")
    unknown_defaults = sorted(set(defaults) - {"quantize", "compute", *CONVROT_INT4_POLICY_KEYS})
    if unknown_defaults:
        raise ValueError(f"Unknown ConvRot policy default keys: {', '.join(unknown_defaults)}")
    default_quantize = defaults.get("quantize", True)
    if not isinstance(default_quantize, bool):
        raise TypeError("ConvRot policy defaults.quantize must be a boolean")
    default_compute = _validate_compute(defaults.get("compute", "quantized"), where="defaults")
    default_group_scales = _validate_optional_group_scales(defaults.get("group_scales"), where="defaults")
    default_group_ratio_q8 = _validate_optional_bool(defaults.get("group_ratio_q8"), field="group_ratio_q8", where="defaults")
    default_scale_refine_steps = _validate_optional_nonnegative_int(
        defaults.get("scale_refine_steps"),
        field="scale_refine_steps",
        where="defaults",
    )

    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        raise TypeError("ConvRot policy rules must be a JSON array")
    rules: list[ConvRotPolicyRule] = []
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, Mapping):
            raise TypeError(f"ConvRot policy rule {index} must be a JSON object")
        unknown_rule = sorted(set(raw_rule) - {"pattern", "quantize", "compute", *CONVROT_INT4_POLICY_KEYS})
        if unknown_rule:
            raise ValueError(f"Unknown ConvRot policy rule {index} keys: {', '.join(unknown_rule)}")
        pattern = raw_rule.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"ConvRot policy rule {index}.pattern must be a non-empty string")
        quantize = raw_rule.get("quantize")
        if quantize is not None and not isinstance(quantize, bool):
            raise TypeError(f"ConvRot policy rule {index}.quantize must be a boolean")
        compute = raw_rule.get("compute")
        if compute is not None:
            compute = _validate_compute(compute, where=f"rules[{index}]")
        group_scales = _validate_optional_group_scales(raw_rule.get("group_scales"), where=f"rules[{index}]")
        group_ratio_q8 = _validate_optional_bool(
            raw_rule.get("group_ratio_q8"),
            field="group_ratio_q8",
            where=f"rules[{index}]",
        )
        scale_refine_steps = _validate_optional_nonnegative_int(
            raw_rule.get("scale_refine_steps"),
            field="scale_refine_steps",
            where=f"rules[{index}]",
        )
        if quantize is None and compute is None and group_scales is None and group_ratio_q8 is None and scale_refine_steps is None:
            raise ValueError(f"ConvRot policy rule {index} does not set any policy parameter")
        rules.append(
            ConvRotPolicyRule(
                pattern=pattern.strip(),
                quantize=quantize,
                compute=compute,
                group_scales=group_scales,
                group_ratio_q8=group_ratio_q8,
                scale_refine_steps=scale_refine_steps,
            )
        )

    return ConvRotPolicy(
        default_quantize=default_quantize,
        default_compute=default_compute,
        default_group_scales=default_group_scales,
        default_group_ratio_q8=default_group_ratio_q8,
        default_scale_refine_steps=default_scale_refine_steps,
        rules=tuple(rules),
    )


def load_convrot_policy(path: str | os.PathLike[str] | None) -> ConvRotPolicy | None:
    if path is None or not str(path).strip():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_convrot_policy(data)


def convrot_policy_to_dict(policy: ConvRotPolicy) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for rule in policy.rules:
        item: dict[str, Any] = {"pattern": rule.pattern}
        if rule.quantize is not None:
            item["quantize"] = rule.quantize
        if rule.compute is not None:
            item["compute"] = rule.compute
        if rule.group_scales is not None:
            item["group_scales"] = rule.group_scales
        if rule.group_ratio_q8 is not None:
            item["group_ratio_q8"] = rule.group_ratio_q8
        if rule.scale_refine_steps is not None:
            item["scale_refine_steps"] = rule.scale_refine_steps
        rules.append(item)
    defaults: dict[str, Any] = {"quantize": policy.default_quantize, "compute": policy.default_compute}
    if policy.default_group_scales is not None:
        defaults["group_scales"] = policy.default_group_scales
    if policy.default_group_ratio_q8 is not None:
        defaults["group_ratio_q8"] = policy.default_group_ratio_q8
    if policy.default_scale_refine_steps is not None:
        defaults["scale_refine_steps"] = policy.default_scale_refine_steps
    return {
        "format": CONVROT_POLICY_FORMAT,
        "defaults": defaults,
        "rules": rules,
    }


def build_convrot_policy_from_quality_report(
    report: Mapping[str, Any],
    *,
    min_cosine: float | None = None,
    min_sqnr_db: float | None = None,
    max_mse: float | None = None,
    action: str = "dequantize",
) -> ConvRotPolicy:
    """Build exact per-layer overrides for metrics outside the requested gates."""

    if min_cosine is None and min_sqnr_db is None and max_mse is None:
        raise ValueError("At least one quality threshold is required")
    if action not in {"dequantize", "keep_bf16"}:
        raise ValueError("action must be 'dequantize' or 'keep_bf16'")
    layers = report.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Quality report has no layers array")

    rules: list[ConvRotPolicyRule] = []
    seen: set[str] = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise TypeError(f"Quality report layer {index} must be an object")
        key = layer.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Quality report layer {index} has no key")
        failures = []
        if min_cosine is not None:
            cosine = float(layer.get("cosine", float("-inf")))
            failures.append(not math.isfinite(cosine) or cosine < min_cosine)
        if min_sqnr_db is not None:
            sqnr_db = float(layer.get("sqnr_db", float("-inf")))
            failures.append(not math.isfinite(sqnr_db) or sqnr_db < min_sqnr_db)
        if max_mse is not None:
            mse = float(layer.get("mse", float("inf")))
            failures.append(not math.isfinite(mse) or mse > max_mse)
        if not any(failures):
            continue
        pattern = _module_name(key)
        if pattern in seen:
            continue
        seen.add(pattern)
        rules.append(
            ConvRotPolicyRule(
                pattern=pattern,
                quantize=False if action == "keep_bf16" else None,
                compute="dequantize",
            )
        )
    return ConvRotPolicy(rules=tuple(rules))


def write_convrot_policy(path: str | os.PathLike[str], policy: ConvRotPolicy) -> None:
    output_path = os.fspath(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(convrot_policy_to_dict(policy), handle, indent=2)
        handle.write("\n")
