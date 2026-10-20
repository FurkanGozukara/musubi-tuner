from __future__ import annotations

import torch.nn.functional as F
import torch

from .runtime.runtime.kernels.flash_attention_2 import flash_attention_2 as _flash_attention_2

try:
    from flash_attn_3 import flash_attn_interface as _flash_attention_3_interface

    _flash_attention_3_import_error = None
except Exception as current_error:
    try:
        import flash_attn_interface as _flash_attention_3_interface

        _flash_attention_3_import_error = None
    except Exception as legacy_error:
        _flash_attention_3_interface = None
        _flash_attention_3_import_error = (current_error, legacy_error)


def _match_flash_dtypes(q, k, v):
    """Match normalized q/k to a half-precision value projection for FlashAttention."""
    if v.dtype in (torch.float16, torch.bfloat16):
        q = q.to(v.dtype)
        k = k.to(v.dtype)
    return q, k, v


def flash_attention_2(q, k, v):
    q, k, v = _match_flash_dtypes(q, k, v)
    return _flash_attention_2(q, k, v)


def flash_attention_3(q, k, v):
    """Run FA3 while normalizing its release-dependent return type."""
    q, k, v = _match_flash_dtypes(q, k, v)
    result = _flash_attention_3_interface.flash_attn_func(q, k, v)
    return result[0] if isinstance(result, tuple) else result


def _sdpa(q, k, v, attn_mask=None):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(
        1, 2
    )


class SelfAttentionEngine:
    """Small upstream-compatible attention dispatcher for training."""

    def __init__(self, engine: str = "auto"):
        if engine not in ("auto", "sdpa", "flash_attention_2", "flash_attention_3"):
            raise ValueError(
                "Vendored Kandinsky 6 supports attention_engine='auto', 'sdpa', 'flash_attention_2', "
                "or 'flash_attention_3', "
                f"got {engine!r}"
            )
        if engine == "flash_attention_2" and _flash_attention_2 is None:
            raise RuntimeError(
                "Kandinsky 6 --flash_attn requires the optional flash-attn package. Install FlashAttention 2 or use --sdpa."
            )
        if engine == "flash_attention_3" and _flash_attention_3_interface is None:
            detail = "; ".join(str(error) for error in (_flash_attention_3_import_error or ()) if str(error))
            raise RuntimeError(
                "Kandinsky 6 --flash3 requires FlashAttention 3. Install the Hopper package so "
                "`from flash_attn_3 import flash_attn_interface` succeeds, or use --sdpa."
                + (f" Import errors: {detail}" if detail else "")
            )
        self._fn = {
            "flash_attention_2": flash_attention_2,
            "flash_attention_3": flash_attention_3,
        }.get(engine, _sdpa)

    def get_attention(self):
        return self._fn


def require_attention_engine(engine: str) -> None:
    """Validate an explicitly requested engine before checkpoint loading."""
    SelfAttentionEngine(engine)
