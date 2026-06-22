"""Opt-in LTX-2 declarative conditioning recipe (prototype).

Off by default; activate with --ltx2_conditioning_config <toml>. Byte-identical when off.

A minimal composable surface over the individual conditioning flags. A TOML file lists the
conditions to apply; each entry is lowered onto the corresponding --ltx2_* flags BEFORE the
per-feature validators run, so multiple conditions compose through the shared train-loop funnel.

Example recipe::

    [[conditions]]
    type = "spatial_crop"
    probability = 0.3
    invert = true

    [[conditions]]
    type = "inpaint"
    probability = 0.5
    threshold = 0.5

lowers to ``--ltx2_spatial_crop --ltx2_spatial_crop_p 0.3 --ltx2_spatial_crop_invert`` plus
``--ltx2_inpaint_mask --ltx2_inpaint_mask_p 0.5 --ltx2_inpaint_mask_threshold 0.5``. Condition
data that is inherently per-item (the spatial_crop region, the inpaint mask files) stays in the
dataset config; the recipe only selects which conditions are active and how often.

When a recipe is active it is the complete declarative set of intrinsic conditions: an intrinsic
that is not listed is off. ``first_frame`` is the only intrinsic that is otherwise on by default
(``--ltx2_first_frame_conditioning_p`` defaults to 0.1), so a recipe that omits it disables
first-frame conditioning; list a ``first_frame`` condition (optionally with a probability) to keep
it. Supported types: ``first_frame``, ``spatial_crop``, ``inpaint`` (alias ``mask``), ``extend``,
``audio_extend``, and ``audio_inpaint`` (alias ``audio_mask``). The audio types require an
audio-bearing mode (validated downstream).

The recipe is authoritative: when a condition is declared both in the recipe and via its CLI flag,
the recipe wins and the command-line setting is overridden (logged at WARNING), rather than raising.
"""

from __future__ import annotations

import argparse
import logging

import toml

logger = logging.getLogger(__name__)

_SENTINEL = "_ltx2_conditioning_applied"


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_conditioning_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the conditioning-recipe CLI flag. Idempotent (guarded). Off by default."""
    if _parser_has_option(parser, "--ltx2_conditioning_config"):
        return parser
    parser.add_argument(
        "--ltx2_conditioning_config",
        type=str,
        default=None,
        help="opt-in: path to a TOML conditioning recipe with a [[conditions]] list "
        "(type = first_frame | spatial_crop | inpaint | extend | audio_extend | audio_inpaint, plus "
        "probability/invert/threshold/prefix/suffix). Each entry is lowered onto the matching --ltx2_* flags so "
        "conditions compose. When a recipe is active it is the complete declarative set: an "
        "intrinsic not listed is disabled (this includes first_frame, which is otherwise on by "
        "default at 0.1). Off by default.",
    )
    parser.add_argument(
        "--ltx2_per_sample_loss",
        action="store_true",
        help="Renormalize the masked loss per sample (each batch element weighted equally regardless "
        "of how much of it is masked in) instead of using the batch-global mask denominator. Off by "
        "default; set on the command line, or via a recipe top-level 'per_sample_loss = true'.",
    )
    return parser


def _resolve_probability(cond: dict, ctype: str, default: float) -> float:
    """Validated probability from the recipe entry, or ``default`` when the entry omits it."""
    if "probability" in cond and "p" in cond:
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' sets both 'probability' and 'p'; use one.")
    value = cond.get("probability", cond.get("p"))
    if value is None:
        return default
    p = float(value)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' probability must be in [0, 1]. Got: {p}")
    return p


def _set_probability(args: argparse.Namespace, attr: str, cond: dict, ctype: str) -> None:
    # Used by first_frame: an omitted probability keeps the attribute's existing value (1d semantics).
    setattr(args, attr, _resolve_probability(cond, ctype, float(getattr(args, attr))))


def _reject_unknown_keys(cond: dict, allowed: set, ctype: str) -> None:
    extra = set(cond.keys()) - allowed
    if extra:
        raise RuntimeError(
            f"conditioning recipe: condition '{ctype}' has unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}."
        )


def _warn_cli_override(was_set_on_cli: bool, ctype: str, cli_label: str) -> None:
    # Recipe authoritative: the recipe wins. A condition also enabled on the command line is
    # overridden (warning), not a hard error. Byte-identical off-path: only reached for a recipe
    # entry whose condition was ALSO set on the CLI (an explicit opt-in collision).
    if was_set_on_cli:
        logger.warning(
            "conditioning recipe governs '%s'; %s set on the command line is overridden by the recipe.",
            ctype,
            cli_label,
        )


def apply_conditioning_config(args: argparse.Namespace) -> None:
    """Parse --ltx2_conditioning_config and lower its conditions onto the --ltx2_* flags.

    No-op when the flag is unset (byte-identical). Idempotent (safe to call from multiple
    setup sites). MUST run before the per-feature validators and before dataset construction.
    """
    if getattr(args, _SENTINEL, False):
        return
    path = getattr(args, "ltx2_conditioning_config", None)
    if not path:
        # Normalize a falsy-but-non-None path (e.g. --ltx2_conditioning_config "" from an unset shell
        # variable) to None, so every recipe-active gate downstream agrees this is a no-recipe run.
        # Without this the loss-reducer gate (`is not None`) disagrees with the parser/metadata
        # (truthiness) and silently switches to per-sample renorm with no recipe actually active.
        if path is not None:
            setattr(args, "ltx2_conditioning_config", None)
        setattr(args, _SENTINEL, True)
        return

    data = toml.load(path)
    conditions = data.get("conditions")
    if conditions is None:
        raise RuntimeError(f"--ltx2_conditioning_config {path}: no [[conditions]] entries found.")
    if not isinstance(conditions, list):
        raise RuntimeError(f"--ltx2_conditioning_config {path}: 'conditions' must be a list of [[conditions]] tables.")

    seen: set = set()
    for index, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            raise RuntimeError(f"--ltx2_conditioning_config {path}: condition #{index} is not a table.")
        raw_type = cond.get("type")
        if raw_type is None:
            raise RuntimeError(f"--ltx2_conditioning_config {path}: condition #{index} is missing 'type'.")
        ctype = str(raw_type).lower()
        if ctype in seen:
            raise RuntimeError(f"--ltx2_conditioning_config {path}: duplicate condition type '{ctype}'.")
        seen.add(ctype)

        # Recipe fully owns each listed condition (authoritative + declarative-complete): EVERY
        # parameter comes from the recipe, defaulting to the feature's own default when the entry
        # omits it — so a sub-setting the recipe leaves out resets to default rather than leaking
        # from the command line. Defaults below mirror the argparse defaults (byte-identical for a
        # recipe-only run, which never set them on the CLI).
        if ctype == "spatial_crop":
            _reject_unknown_keys(cond, {"type", "probability", "p", "invert"}, ctype)
            _warn_cli_override(bool(getattr(args, "ltx2_spatial_crop", False)), ctype, "--ltx2_spatial_crop")
            args.ltx2_spatial_crop = True
            args.ltx2_spatial_crop_p = _resolve_probability(cond, ctype, 0.0)
            args.ltx2_spatial_crop_invert = bool(cond.get("invert", False))
        elif ctype in ("inpaint", "mask"):
            _reject_unknown_keys(cond, {"type", "probability", "p", "invert", "threshold"}, ctype)
            _warn_cli_override(bool(getattr(args, "ltx2_inpaint_mask", False)), ctype, "--ltx2_inpaint_mask")
            args.ltx2_inpaint_mask = True
            args.ltx2_inpaint_mask_p = _resolve_probability(cond, ctype, 0.0)
            args.ltx2_inpaint_mask_invert = bool(cond.get("invert", False))
            threshold = float(cond.get("threshold", 0.5))
            if not (0.0 <= threshold <= 1.0):
                raise RuntimeError(f"conditioning recipe: 'inpaint' threshold must be in [0, 1]. Got: {threshold}")
            args.ltx2_inpaint_mask_threshold = threshold
        elif ctype == "extend":
            _reject_unknown_keys(cond, {"type", "probability", "p", "prefix", "suffix"}, ctype)
            _warn_cli_override(
                int(getattr(args, "ltx2_extend_prefix_frames", 0) or 0) > 0
                or int(getattr(args, "ltx2_extend_suffix_frames", 0) or 0) > 0,
                ctype,
                "--ltx2_extend_prefix_frames/--ltx2_extend_suffix_frames",
            )
            args.ltx2_extend_prefix_frames = int(cond.get("prefix", 0))
            args.ltx2_extend_suffix_frames = int(cond.get("suffix", 0))
            args.ltx2_extend_p = _resolve_probability(cond, ctype, 1.0)
            if args.ltx2_extend_prefix_frames <= 0 and args.ltx2_extend_suffix_frames <= 0:
                raise RuntimeError("conditioning recipe: condition 'extend' requires prefix > 0 or suffix > 0.")
        elif ctype == "first_frame":
            # first_frame has no off-default boolean flag (it is governed by a probability that
            # defaults to 0.1, i.e. on). Listing it activates first-frame conditioning; the
            # probability is optional (absent -> keep the existing default). There is nothing to
            # double-declare ambiguously, so no CLI-duplicate rejection: under the declarative
            # recipe (see below) the recipe always governs first_frame.
            _reject_unknown_keys(cond, {"type", "probability", "p"}, ctype)
            _set_probability(args, "ltx2_first_frame_conditioning_p", cond, ctype)
        elif ctype == "audio_extend":
            # Audio analog of 'extend' (lowers onto --ltx2_audio_extend_*). Requires an audio-bearing
            # mode; that is enforced downstream by validate_audio_extend_setup, not here.
            _reject_unknown_keys(cond, {"type", "probability", "p", "prefix", "suffix"}, ctype)
            _warn_cli_override(
                int(getattr(args, "ltx2_audio_extend_prefix_frames", 0) or 0) > 0
                or int(getattr(args, "ltx2_audio_extend_suffix_frames", 0) or 0) > 0,
                ctype,
                "--ltx2_audio_extend_prefix_frames/--ltx2_audio_extend_suffix_frames",
            )
            args.ltx2_audio_extend_prefix_frames = int(cond.get("prefix", 0))
            args.ltx2_audio_extend_suffix_frames = int(cond.get("suffix", 0))
            args.ltx2_audio_extend_p = _resolve_probability(cond, ctype, 1.0)
            if args.ltx2_audio_extend_prefix_frames <= 0 and args.ltx2_audio_extend_suffix_frames <= 0:
                raise RuntimeError("conditioning recipe: condition 'audio_extend' requires prefix > 0 or suffix > 0.")
        elif ctype in ("audio_inpaint", "audio_mask"):
            # Audio analog of 'inpaint' (lowers onto --ltx2_audio_inpaint_mask*). Audio-mode-only;
            # enforced downstream by validate_audio_inpaint_mask_setup.
            _reject_unknown_keys(cond, {"type", "probability", "p", "invert", "threshold"}, ctype)
            _warn_cli_override(bool(getattr(args, "ltx2_audio_inpaint_mask", False)), ctype, "--ltx2_audio_inpaint_mask")
            args.ltx2_audio_inpaint_mask = True
            args.ltx2_audio_inpaint_mask_p = _resolve_probability(cond, ctype, 0.0)
            args.ltx2_audio_inpaint_mask_invert = bool(cond.get("invert", False))
            threshold = float(cond.get("threshold", 0.5))
            if not (0.0 <= threshold <= 1.0):
                raise RuntimeError(f"conditioning recipe: 'audio_inpaint' threshold must be in [0, 1]. Got: {threshold}")
            args.ltx2_audio_inpaint_mask_threshold = threshold
        else:
            raise RuntimeError(
                f"--ltx2_conditioning_config {path}: unknown condition type '{ctype}'. "
                "Known types: first_frame, spatial_crop, inpaint (alias: mask), extend, "
                "audio_extend, audio_inpaint (alias: audio_mask)."
            )

    # Declarative-complete: an active recipe is the full set of intrinsic conditions, so a recipe
    # that does not list first_frame disables it (parity with the other intrinsics, which are off
    # unless listed). first_frame is the only intrinsic that is otherwise on by default, so this is
    # the only one that needs an explicit off when absent.
    if "first_frame" not in seen:
        prev = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
        if prev != 0.0:
            logger.info(
                "Conditioning recipe %s does not list 'first_frame'; disabling first-frame "
                "conditioning (was p=%.3f). Add a 'first_frame' condition to keep it.",
                path,
                prev,
            )
        args.ltx2_first_frame_conditioning_p = 0.0

    # Per-sample loss is a top-level loss policy (not a per-condition entry). The recipe owns it like
    # any other dial: set from the recipe top-level 'per_sample_loss' (default False when omitted),
    # overriding a CLI --ltx2_per_sample_loss with a warning. A recipe therefore never IMPLICITLY
    # changes the loss reduction; per-sample loss is opt-in here exactly as it is on the command line.
    _warn_cli_override(bool(getattr(args, "ltx2_per_sample_loss", False)), "per_sample_loss", "--ltx2_per_sample_loss")
    args.ltx2_per_sample_loss = bool(data.get("per_sample_loss", False))

    setattr(args, _SENTINEL, True)
    logger.info("Applied conditioning recipe from %s: %s", path, sorted(seen))
