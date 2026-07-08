"""Activation-aware diagnostics for INT4 ConvRot Linear layers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules.int4_convrot_utils import (
    dequantize_int4_convrot_weight,
    padded_features_for_group,
    rotate_activation_padded,
)


@dataclass
class Int4ConvRotActivationLayerStats:
    key: str
    calls: int = 0
    rows: int = 0
    numel: int = 0
    dot: float = 0.0
    ref_norm_sq: float = 0.0
    quant_norm_sq: float = 0.0
    sqerr_sum: float = 0.0
    abserr_sum: float = 0.0
    max_abs_error: float = 0.0
    input_absmax_max: float = 0.0
    input_absmax_sum: float = 0.0
    input_rms_sum: float = 0.0
    activation_sqerr_sum: float = 0.0
    activation_ref_norm_sq: float = 0.0
    activation_max_abs_error: float = 0.0

    def to_report(self) -> dict[str, Any]:
        mse = self.sqerr_sum / max(self.numel, 1)
        mae = self.abserr_sum / max(self.numel, 1)
        signal_mean_square = self.ref_norm_sq / max(self.numel, 1)
        sqnr_db = _sqnr_db(self.ref_norm_sq, self.sqerr_sum)
        denom = math.sqrt(max(self.ref_norm_sq, 0.0) * max(self.quant_norm_sq, 0.0))
        cosine = self.dot / denom if denom > 0 else 1.0
        relative_l2 = math.sqrt(self.sqerr_sum / self.ref_norm_sq) if self.ref_norm_sq > 0 else 0.0
        activation_sqnr_db = _sqnr_db(self.activation_ref_norm_sq, self.activation_sqerr_sum)
        return {
            **asdict(self),
            "mse": float(mse),
            "mae": float(mae),
            "signal_mean_square": float(signal_mean_square),
            "sqnr_db": float(sqnr_db),
            "cosine": float(cosine),
            "relative_l2": float(relative_l2),
            "input_absmax_mean": float(self.input_absmax_sum / max(self.calls, 1)),
            "input_rms_mean": float(self.input_rms_sum / max(self.calls, 1)),
            "activation_sqnr_db": float(activation_sqnr_db),
        }


def _sqnr_db(signal_norm_sq: float, err_norm_sq: float) -> float:
    if signal_norm_sq <= 0:
        return float("inf")
    if err_norm_sq <= 0:
        return float("inf")
    return float(10.0 * math.log10(signal_norm_sq / err_norm_sq))


def _module_int4_shape(module: nn.Module) -> tuple[int, int, int]:
    shape = getattr(module, "int4_shape", None)
    if not isinstance(shape, torch.Tensor):
        raise ValueError("INT4 ConvRot activation calibration requires module.int4_shape")
    values = shape.detach().cpu().reshape(-1).tolist()
    if len(values) != 3:
        raise ValueError(f"Expected int4_shape=[out,in,padded], got {values}")
    return int(values[0]), int(values[1]), int(values[2])


def _module_int4_group(module: nn.Module) -> int:
    value = getattr(module, "int4_convrot_groupsize", None)
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item()) if value.numel() else 0
    return int(value or 0)


def _activation_bits() -> int:
    value = os.getenv("LTX2_INT4_CONVROT_ACT_BITS", "8").strip().lower()
    return 4 if value in {"4", "a4", "w4a4", "int4"} else 8


def _sample_rows(x_2d: torch.Tensor, y_2d: torch.Tensor, max_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_rows <= 0 or x_2d.shape[0] <= max_rows:
        return x_2d, y_2d
    indices = torch.linspace(0, x_2d.shape[0] - 1, steps=max_rows, device=x_2d.device).round().to(torch.long)
    return x_2d.index_select(0, indices), y_2d.index_select(0, indices)


def _quant_dequant_activation(x_2d: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 7.0 if bits == 4 else 127.0
    scale = (x_2d.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-30)
    q = (x_2d.float() / scale.float()).round().clamp(-int(qmax), int(qmax))
    return (q * scale.float()).to(dtype=x_2d.dtype)


class Int4ConvRotActivationCalibrator:
    """Collect output-error metrics for patched INT4 ConvRot Linear modules."""

    def __init__(
        self,
        model: nn.Module,
        *,
        module_regex: str | None = None,
        max_rows: int = 128,
        max_layers: int = 0,
    ) -> None:
        self.model = model
        self.module_regex = re.compile(module_regex) if module_regex else None
        self.max_rows = int(max_rows)
        self.max_layers = int(max_layers)
        self.activation_bits = _activation_bits()
        self.stats: dict[str, Int4ConvRotActivationLayerStats] = {}
        self._handles: list[Any] = []
        self._active = False
        self._step: int | None = None
        self._registered = 0
        self._register_hooks()

    @property
    def registered_layers(self) -> int:
        return self._registered

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def start_step(self, step: int) -> None:
        self._step = int(step)
        self._active = True

    def finish_step(self) -> None:
        self._active = False
        self._step = None

    def write_report(self, path: str, *, batches: int, options: dict[str, Any] | None = None) -> dict[str, Any]:
        layers = [stat.to_report() for stat in self.stats.values()]
        layers.sort(key=lambda item: (item["sqnr_db"], -item["relative_l2"]))
        total_numel = sum(int(item["numel"]) for item in layers)
        total_ref = sum(float(item["ref_norm_sq"]) for item in layers)
        total_err = sum(float(item["sqerr_sum"]) for item in layers)
        report = {
            "format": "ltx2_int4_convrot_activation_calibration_v1",
            "baseline": "same packed INT4 reconstructed weights with torch.nn.functional.linear",
            "activation_mode": f"W4A{self.activation_bits}",
            "options": options or {},
            "summary": {
                "registered_layers": self.registered_layers,
                "measured_layers": len(layers),
                "batches": int(batches),
                "numel": int(total_numel),
                "weighted_sqnr_db": _sqnr_db(total_ref, total_err),
                "worst_sqnr_db": layers[0]["sqnr_db"] if layers else None,
                "worst_relative_l2": max((float(item["relative_l2"]) for item in layers), default=0.0),
                "worst_activation_sqnr_db": min((float(item["activation_sqnr_db"]) for item in layers), default=None),
            },
            "layers": layers,
        }
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        return report

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if self.max_layers > 0 and self._registered >= self.max_layers:
                break
            if not isinstance(module, nn.Linear):
                continue
            if not (hasattr(module, "scale_weight") and hasattr(module, "int4_shape")):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
                continue
            if self.module_regex is not None and self.module_regex.search(name) is None:
                continue
            self.stats[name] = Int4ConvRotActivationLayerStats(key=name)
            self._handles.append(module.register_forward_hook(self._make_hook(name), always_call=False))
            self._registered += 1

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active or not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                return
            self._measure(name, module, inputs[0], output)

        return hook

    @torch.no_grad()
    def _measure(self, name: str, module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
        if not x.is_floating_point() or not output.is_floating_point():
            return
        out_features, in_features, padded_features = _module_int4_shape(module)
        group_size = _module_int4_group(module)
        if group_size <= 0 or padded_features != padded_features_for_group(in_features, group_size):
            return

        x_2d = x.detach().reshape(-1, in_features)
        y_2d = output.detach().reshape(-1, out_features)
        x_sample, y_sample = _sample_rows(x_2d, y_2d, self.max_rows)
        if x_sample.numel() == 0:
            return

        weight = dequantize_int4_convrot_weight(
            module.weight.detach(),
            module.scale_weight.detach(),
            group_size,
            in_features,
            padded_features,
            dtype=x_sample.dtype,
        )
        awq_scales = getattr(module, "int4_awq_scales", None)
        if isinstance(awq_scales, torch.Tensor):
            awq_scales = awq_scales.detach().to(device=x_sample.device, dtype=torch.float32).reshape(-1)
            if awq_scales.numel() != in_features:
                raise ValueError(
                    f"{name} INT4 ConvRot AWQ scale length {awq_scales.numel()} does not match in_features={in_features}"
                )
            x_sample = (x_sample.float() / awq_scales.reshape(1, -1)).to(dtype=x_sample.dtype)
        bias = module.bias.detach().to(dtype=x_sample.dtype, device=x_sample.device) if module.bias is not None else None
        y_ref = F.linear(x_sample, weight.to(device=x_sample.device), bias)
        y_quant = y_sample.to(device=y_ref.device, dtype=y_ref.dtype)
        err = y_quant.float() - y_ref.float()
        ref = y_ref.float()
        quant = y_quant.float()

        x_rot = rotate_activation_padded(x_sample, group_size, padded_features).reshape(-1, padded_features)
        x_deq = _quant_dequant_activation(x_rot, self.activation_bits)
        act_err = x_deq.float() - x_rot.float()

        stat = self.stats[name]
        stat.calls += 1
        stat.rows += int(x_sample.shape[0])
        stat.numel += int(ref.numel())
        stat.dot += float((ref * quant).sum().item())
        stat.ref_norm_sq += float((ref * ref).sum().item())
        stat.quant_norm_sq += float((quant * quant).sum().item())
        stat.sqerr_sum += float((err * err).sum().item())
        stat.abserr_sum += float(err.abs().sum().item())
        stat.max_abs_error = max(stat.max_abs_error, float(err.abs().max().item()))
        stat.input_absmax_max = max(stat.input_absmax_max, float(x_sample.float().abs().max().item()))
        stat.input_absmax_sum += float(x_sample.float().abs().amax(dim=-1).mean().item())
        stat.input_rms_sum += float(x_sample.float().pow(2).mean().sqrt().item())
        stat.activation_sqerr_sum += float((act_err * act_err).sum().item())
        stat.activation_ref_norm_sq += float((x_rot.float() * x_rot.float()).sum().item())
        stat.activation_max_abs_error = max(stat.activation_max_abs_error, float(act_err.abs().max().item()))
