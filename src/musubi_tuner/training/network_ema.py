from __future__ import annotations

from typing import Optional

import torch


def _ema_param_value(param: torch.nn.Parameter) -> torch.Tensor:
    data = param.data
    try:
        from musubi_tuner.modules.int8_training import Int8QTWeight

        if isinstance(data, Int8QTWeight):
            return data.dequantize()
    except Exception:
        pass
    return data


class NetworkEMAModel:
    """EMA shadow for trainable network parameters.

    This is intended for adapter/LoRA training. It tracks only parameters with
    ``requires_grad=True`` and can keep the shadow on CPU to avoid extra VRAM.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = max(1, int(update_every))
        self.step = 0
        self.device = device
        self.shadow_params: dict[str, torch.Tensor] = {}
        self._sync_shadow_keys(model, copy_current=True)

    def _target_device(self, value: torch.Tensor) -> torch.device:
        return self.device if self.device is not None else value.device

    def _sync_shadow_keys(self, model: torch.nn.Module, *, copy_current: bool = False) -> None:
        tracked = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                value = _ema_param_value(param).detach()
                tracked[name] = value
                shadow = self.shadow_params.get(name)
                if copy_current or shadow is None or tuple(shadow.shape) != tuple(value.shape):
                    self.shadow_params[name] = value.clone().to(self._target_device(value))
        stale = set(self.shadow_params) - set(tracked)
        for name in stale:
            del self.shadow_params[name]

    def _copy_to_shadow(self, model: torch.nn.Module) -> None:
        self._sync_shadow_keys(model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow_params.get(name)
                if shadow is not None and param.requires_grad:
                    value = _ema_param_value(param).detach().to(device=shadow.device, dtype=shadow.dtype)
                    shadow.copy_(value)

    def reset(self, model: torch.nn.Module, step: int = 0) -> None:
        self.step = int(step)
        self.shadow_params.clear()
        self._sync_shadow_keys(model, copy_current=True)

    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        self._sync_shadow_keys(model)

        if self.step <= self.update_after_step:
            if self.step == self.update_after_step or self.step % self.update_every == 0:
                self._copy_to_shadow(model)
            return

        if (self.step - self.update_after_step) % self.update_every != 0:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow_params.get(name)
                if shadow is None or not param.requires_grad:
                    continue
                value = _ema_param_value(param).detach().to(device=shadow.device, dtype=shadow.dtype)
                shadow.lerp_(value, 1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        self._sync_shadow_keys(model)
        original_params: dict[str, torch.Tensor] = {}
        copied_params: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            named_params = [(name, param) for name, param in model.named_parameters() if name in self.shadow_params]
            for name, param in named_params:
                original_params[name] = param.data.detach().clone()
            try:
                for name, param in named_params:
                    shadow = self.shadow_params[name]
                    copied_params[name] = original_params[name]
                    param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
            except Exception:
                self.restore(model, copied_params)
                raise
        return original_params

    def restore(self, model: torch.nn.Module, original_params: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                original = original_params.get(name)
                if original is not None:
                    param.data.copy_(original.to(device=param.device, dtype=param.dtype))

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "step": self.step,
            "shadow_params": {name: tensor.detach().cpu() for name, tensor in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_after_step = int(state_dict.get("update_after_step", self.update_after_step))
        self.update_every = max(1, int(state_dict.get("update_every", self.update_every)))
        self.step = int(state_dict.get("step", self.step))
        shadows = state_dict.get("shadow_params", {})
        if isinstance(shadows, dict):
            old_shadows = self.shadow_params
            loaded_shadows: dict[str, torch.Tensor] = {}
            for name, tensor in shadows.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                key = str(name)
                old_shadow = old_shadows.get(key)
                target_device = (
                    self.device if self.device is not None else (old_shadow.device if old_shadow is not None else tensor.device)
                )
                loaded_shadows[key] = tensor.detach().clone().to(target_device)
            self.shadow_params = loaded_shadows
