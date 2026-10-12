"""``key=value`` items of --optimizer_args / --lr_scheduler_args / --network_args.

Every trainer reads these lists the same way: split each item at its first ``=``, strip the spaces around the
key and the value (a TOML written by hand or another tool may carry them), read optimizer / scheduler values as
Python literals and keep network values as text. A value that holds ``=`` (sd-scripts style
``network_reg_lrs=a_=1e-3``) stays whole; an item without a key fails with a clear message.
"""

from __future__ import annotations

import ast


def split_key_value(item, option: str = "--optimizer_args") -> tuple[str, str]:
    text = str(item).strip()
    key, separator, value = text.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"Invalid {option} entry (expected key=value): {item}")
    return key, value.strip()


def literal_or_text(value: str):
    """A Python literal (``0.01``, ``(0.9, 0.999)``, ``'triangular2'``, ``True``) or, if it is none, the text itself;
    lowercase true / false / none become the Python values (as text they would always be "true")."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        return value


def parse_literal_args(items, option: str = "--optimizer_args") -> dict:
    """{key: Python value} of an optimizer / scheduler argument list."""
    kwargs = {}
    for item in items or []:
        if not str(item).strip():
            continue
        key, value = split_key_value(item, option)
        kwargs[key] = literal_or_text(value)
    return kwargs


def parse_text_args(items, option: str = "--network_args") -> dict:
    """{key: text} of a network argument list (network modules parse their own values)."""
    kwargs = {}
    for item in items or []:
        if not str(item).strip():
            continue
        key, value = split_key_value(item, option)
        kwargs[key] = value
    return kwargs
