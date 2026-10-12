"""AutomagicEXPERIMENT (ported from ostris/ai-toolkit toolkit/optimizers/automagicEXPERIMENT.py).

Automagic3's learning-rate controller (one shared lr per param group, moved by the pooled vote of each element's
1-bit gradient-sign history) with a FULL per-element second moment, AdamW-style and bias-corrected, instead of
Adafactor's factored approximation. The second moment of every >=2D tensor is stored in the square-root domain as
ConvRot int8 rows (convrot_state.py): about 1 byte per element plus the H/8-byte sign history (2 bytes per element
at the default polarity_history=8). 1D tensors keep an fp32 second moment.

No momentum: beta1 must be 0. The controller detects overshoot as perfect period-2 gradient alternation, which only
exists while the applied update follows the instantaneous gradient; momentum turns a bounce into runs of same-sign
gradients that would vote the lr up.

Everything else is Automagic3's machinery in this fork: fused updates in the backward hooks (or the safe non-fused
mode with stochastic low-precision gradient accumulation and optional CPU gradient offload), the trust region
(update RMS <= clip_threshold, then each element clamped), decoupled weight decay, stochastic rounding of
low-precision weights, and lr reporting through get_learning_rates().
"""

from __future__ import annotations

import torch

from .automagic3 import Automagic3
from .convrot_state import ConvRot8State


class AutomagicEXPERIMENT(Automagic3):
    _QUANT_KEYS = ("exp_avg_sq",)

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-30,  # failsafes only, parked decades outside any operating range
        max_lr: float = 1e3,
        betas=(0.0, 0.999),
        beta2: float | None = None,  # Automagic3 spelling, accepted too
        eps: float = 1e-8,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history: int = 8,
        fused: bool = True,
        offload_gradients: bool = False,
    ):
        beta1, second = (float(betas[0]), float(betas[1])) if isinstance(betas, (list, tuple)) else (0.0, float(betas))
        if beta2 is not None:
            second = float(beta2)
        if beta1 != 0.0:
            raise ValueError(
                "AutomagicEXPERIMENT requires beta1 = 0: momentum turns an overshoot bounce into runs of same-sign "
                "gradients, which the sign-window controller reads as consistency and votes the lr up."
            )
        super().__init__(
            params,
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            beta2=second,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history=polarity_history,
            fused=fused,
            offload_gradients=offload_gradients,
        )

    # ------------------------------------------------------------------ state

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.tensor(
            min(max(float(group["lr"]), group["min_lr"]), group["max_lr"]), dtype=torch.float32, device=p.device
        )
        H = group["polarity_history"]
        state["sign_history"] = torch.zeros((H, (p.numel() + 7) // 8), dtype=torch.uint8, device=p.device)
        state["hist_idx"] = 0
        state["hist_fill"] = 0
        zeros = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        # >=2D: sqrt(second moment) as ConvRot int8 rows; 1D (biases, norms, DoRA magnitudes): plain fp32
        state["exp_avg_sq"] = ConvRot8State(zeros) if p.dim() >= 2 else zeros

    @staticmethod
    def _follow_param_device(state: dict, device: torch.device) -> None:
        """Block swap may move a parameter between the CPU and the GPU: its state follows it."""
        for key in ("lr", "sign_history"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device)
        moment = state.get("exp_avg_sq")
        if isinstance(moment, ConvRot8State):
            if moment.device != device:
                moment.to_(device)
        elif isinstance(moment, torch.Tensor) and moment.device != device:
            state["exp_avg_sq"] = moment.to(device)

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0 or "exp_avg_sq" not in state or "sign_history" not in state or "lr" not in state:
            self._init_state(p, group)
        self._follow_param_device(state, p.device)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("AutomagicEXPERIMENT does not support sparse gradients.")
        # an fp32 gradient is used in place (p.grad is released below)
        grad = grad.to(device=p.device, dtype=torch.float32)
        # fused mode runs inside backward, before the trainer's nan/inf checks: a non-finite gradient would poison
        # the second moment forever, so those elements contribute nothing this step
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        state["step"] += 1
        beta2 = group["beta2"]
        bias_correction2 = 1.0 - beta2 ** state["step"]

        moment = state["exp_avg_sq"]
        if isinstance(moment, ConvRot8State):
            # the store holds sqrt(v): square it back for the EMA (clamped first: rotation noise can leave tiny
            # negatives, and squaring them would bias v upward), then store the new sqrt(v)
            v = moment.dequantize().clamp_(min=0.0).square_()
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            v_sqrt = v.sqrt_()
            moment.quantize_(v_sqrt)
            denom = v_sqrt.div_(bias_correction2**0.5).add_(group["eps"])
        else:
            moment.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = (moment / bias_correction2).sqrt_().add_(group["eps"])
        update = grad.div_(denom)

        # Automagic3's trust region; not optional with 8-bit moments: an element whose sqrt(v) is below the int8
        # resolution dequantizes to zero and its raw update would explode
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        # the vote: Automagic3's rule on gradient signs (the update keeps the gradient's sign)
        hist = state["sign_history"]
        idx = state["hist_idx"]
        H = hist.shape[0]
        hist[idx].copy_(self._pack_bits(update.gt(0.0)))
        state["hist_idx"] = (idx + 1) % H
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill
        if fill == H:
            _, shifts = self._pack_consts(hist.device)
            chron = torch.roll(hist, -state["hist_idx"], dims=0)
            bits = (chron.unsqueeze(-1) >> shifts).bitwise_and_(1).view(H, -1)[:, : update.numel()]
            s1 = bits.sum(0, dtype=torch.int16)
            flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
            up = s1.eq(H).logical_or_(s1.eq(0))
            down = flips.eq(H - 1)
            w = update.abs().view(-1)
            num = (w * up).sum().sub_((w * down).sum())
            den = w.sum()
            gi = self._param_group_index.get(p)
            if gi is not None:
                if self._group_num[gi] is None:
                    self._group_num[gi] = num
                    self._group_den[gi] = den
                else:
                    acc = self._group_num[gi]
                    if num.device != acc.device:
                        num, den = num.to(acc.device), den.to(acc.device)
                    acc.add_(num)
                    self._group_den[gi].add_(den)

        lr_t = state["lr"]
        wd = group["weight_decay"]
        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-1.0)
        else:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-1.0)
            self._stochastic_copy_(p, new_p_fp32)
        p.grad = None

    # ------------------------------------------------------------ checkpoints

    def state_dict(self, *args, **kwargs):
        """The int8 second moments are saved through their own state dicts (they are not tensors)."""
        saved = super().state_dict(*args, **kwargs)
        # torch hands out the live per-parameter state dicts: copy each one before replacing its entries
        saved["state"] = {
            param_id: {
                key: (value.state_dict() if key in self._QUANT_KEYS and isinstance(value, ConvRot8State) else value)
                for key, value in state.items()
            }
            for param_id, state in saved["state"].items()
        }
        return saved

    def load_state_dict(self, state_dict):
        # the parent load would cast the fp32 row scales to the parameter dtype: load without them, restore after
        quantized = {}
        trimmed = {"param_groups": state_dict["param_groups"], "state": {}}
        for param_id, param_state in state_dict["state"].items():
            trimmed["state"][param_id] = {k: v for k, v in param_state.items() if k not in self._QUANT_KEYS}
            for key in self._QUANT_KEYS:
                if key in param_state:
                    quantized.setdefault(param_id, {})[key] = param_state[key]
        super().load_state_dict(trimmed)
        params = [p for group in self.param_groups for p in group["params"]]
        for param_id, values in quantized.items():
            if not isinstance(param_id, int) or param_id >= len(params):
                continue
            p = params[param_id]
            state = self.state.setdefault(p, {})
            for key, value in values.items():
                if isinstance(value, dict):
                    moment = ConvRot8State(value).to_(p.device)
                    if tuple(moment.shape) == tuple(p.shape):
                        state[key] = moment
                elif isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(p.shape):
                    state[key] = value.to(device=p.device, dtype=torch.float32)
