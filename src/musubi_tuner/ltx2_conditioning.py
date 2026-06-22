"""Opt-in LTX-2 declarative conditioning recipe (prototype).

Off by default; activate with --ltx2_conditioning_config <toml>. Byte-identical when off.

A minimal composable surface over the individual conditioning flags. A TOML file lists the
conditions to apply; each entry is lowered onto the corresponding --ltx2_* flags BEFORE the
per-feature validators run, so multiple conditions compose through the shared train-loop funnel.

Example recipe::

    [video]

      [[video.conditions]]
      type = "spatial_crop"
      probability = 0.3
      invert = true

    [audio]

      [[audio.conditions]]
      type = "extend"
      suffix = 8

lowers to ``--ltx2_spatial_crop --ltx2_spatial_crop_p 0.3 --ltx2_spatial_crop_invert`` plus
``--ltx2_audio_extend_suffix_frames 8``. Condition data that is inherently per-item (the
spatial_crop region, the inpaint mask files) stays in the dataset config; the recipe only selects
which conditions are active and how often.

When a recipe is active it is the complete declarative set of intrinsic conditions: an intrinsic
that is not listed is off. ``first_frame`` is the only intrinsic that is otherwise on by default
(``--ltx2_first_frame_conditioning_p`` defaults to 0.1), so a recipe that omits a ``[video]``
``first_frame`` disables first-frame conditioning; list one (optionally with a probability) to keep
it (a reference recipe is the exception — see below). ``[video]`` types: ``first_frame``,
``spatial_crop``, ``inpaint`` (alias ``mask``), ``extend``, ``reference``. ``[audio]`` types:
``extend``, ``inpaint`` (alias ``mask``), ``reference`` — the intrinsic ones lower onto the
``--ltx2_audio_*`` flags and require an audio-bearing mode (validated downstream). The legacy flat
top-level ``[[conditions]]`` list is no longer supported.

A ``reference`` condition is a zero-data marker: its presence in a block selects a reference IC-LoRA
strategy, lowered onto ``--ic_lora_strategy`` from the recipe structure — a ``[video]`` reference
alone -> ``v2v``; ``[video]`` + ``[audio]`` references -> ``av_ic``; a ``[video]`` reference with an
``[audio]`` block but no audio reference -> ``video_ref_only_av``; an ``[audio]`` reference with no
video reference -> ``audio_ref_ic``. An optional ``probability`` on the ``[video]`` reference sets the
reference-dropout dial (``--ic_lora_ref_probability``); the audio reference takes no parameters. The
reference video/audio data itself stays in the dataset config; the recipe only selects the strategy.
Because a reference strategy bypasses the intrinsic funnel, a reference may not be combined with a
gated intrinsic (``spatial_crop`` / ``inpaint`` / ``extend``) in the same recipe; ``first_frame``
composes with IC-LoRA and is exempt, so it is left at its CLI/default value (not auto-disabled) in a
reference recipe. Each block also accepts ``is_generated`` (bool, default true). Setting exactly one
modality's ``is_generated = false`` freezes it as clean conditioning and trains directionally:
``[audio]`` ``is_generated = false`` -> ``a2v`` (freeze audio, generate video); ``[video]``
``is_generated = false`` -> ``v2a`` (freeze video, generate audio/foley). This lowers onto
``--ltx2_train_direction`` and requires ``--ltx2_mode av`` (enforced downstream); it cannot be
combined with a ``reference``.

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
        help="opt-in: path to a TOML conditioning recipe with [video]/[audio] blocks, each holding a "
        "[[video.conditions]] / [[audio.conditions]] list (type = first_frame | spatial_crop | inpaint | "
        "extend | reference in [video]; extend | inpaint | reference in [audio]; plus "
        "probability/invert/threshold/prefix/suffix). Intrinsic entries lower onto the matching --ltx2_* "
        "flags so conditions compose; a 'reference' marker selects an IC-LoRA strategy (--ic_lora_strategy). "
        "When a recipe is active it is the complete declarative set: an intrinsic not listed is disabled "
        "(this includes first_frame, which is otherwise on by default at 0.1). Off by default.",
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
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' probability must be a number. Got: {value!r}")
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' probability must be in [0, 1]. Got: {p}")
    return p


def _bool_key(cond: dict, key: str, ctype: str, default: bool = False) -> bool:
    """Boolean recipe value; rejects non-bool (e.g. the string "false", which bool() coerces to True)."""
    value = cond.get(key, default)
    if not isinstance(value, bool):
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' key '{key}' must be a boolean (true/false). Got: {value!r}")
    return value


def _int_key(cond: dict, key: str, ctype: str, default: int = 0) -> int:
    """Integer recipe value; rejects floats/strings (which would silently truncate or crash raw)."""
    value = cond.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"conditioning recipe: condition '{ctype}' key '{key}' must be an integer. Got: {value!r}")
    return value


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


def _lower_video_condition(args: argparse.Namespace, cond: dict, ctype: str, path: str) -> None:
    """Lower one [video] recipe condition onto the --ltx2_* video flags."""
    if ctype == "spatial_crop":
        _reject_unknown_keys(cond, {"type", "probability", "p", "invert"}, ctype)
        _warn_cli_override(bool(getattr(args, "ltx2_spatial_crop", False)), ctype, "--ltx2_spatial_crop")
        args.ltx2_spatial_crop = True
        args.ltx2_spatial_crop_p = _resolve_probability(cond, ctype, 0.0)
        args.ltx2_spatial_crop_invert = _bool_key(cond, "invert", ctype)
    elif ctype in ("inpaint", "mask"):
        _reject_unknown_keys(cond, {"type", "probability", "p", "invert", "threshold"}, ctype)
        _warn_cli_override(bool(getattr(args, "ltx2_inpaint_mask", False)), ctype, "--ltx2_inpaint_mask")
        args.ltx2_inpaint_mask = True
        args.ltx2_inpaint_mask_p = _resolve_probability(cond, ctype, 0.0)
        args.ltx2_inpaint_mask_invert = _bool_key(cond, "invert", ctype)
        threshold = float(cond.get("threshold", 0.5))
        if not (0.0 <= threshold <= 1.0):
            raise RuntimeError(f"conditioning recipe: [video] 'inpaint' threshold must be in [0, 1]. Got: {threshold}")
        args.ltx2_inpaint_mask_threshold = threshold
    elif ctype == "extend":
        _reject_unknown_keys(cond, {"type", "probability", "p", "prefix", "suffix"}, ctype)
        _warn_cli_override(
            int(getattr(args, "ltx2_extend_prefix_frames", 0) or 0) > 0
            or int(getattr(args, "ltx2_extend_suffix_frames", 0) or 0) > 0,
            ctype,
            "--ltx2_extend_prefix_frames/--ltx2_extend_suffix_frames",
        )
        args.ltx2_extend_prefix_frames = _int_key(cond, "prefix", ctype)
        args.ltx2_extend_suffix_frames = _int_key(cond, "suffix", ctype)
        args.ltx2_extend_p = _resolve_probability(cond, ctype, 1.0)
        if args.ltx2_extend_prefix_frames <= 0 and args.ltx2_extend_suffix_frames <= 0:
            raise RuntimeError("conditioning recipe: [video] 'extend' requires prefix > 0 or suffix > 0.")
    elif ctype == "first_frame":
        # first_frame has no off-default boolean flag (governed by a probability that defaults to 0.1).
        # Listing it activates first-frame conditioning; probability is optional (absent -> keep the
        # existing default). The recipe always governs first_frame (declarative-complete, below).
        _reject_unknown_keys(cond, {"type", "probability", "p"}, ctype)
        _set_probability(args, "ltx2_first_frame_conditioning_p", cond, ctype)
    elif ctype == "reference":
        # Zero-data marker: presence selects a reference IC-LoRA strategy (resolved from the full
        # recipe structure after the block is lowered). No --ltx2_* flag of its own. An optional
        # probability sets the reference-dropout dial (--ic_lora_ref_probability); when omitted the
        # dial keeps its CLI/default value (parity with a bare --ic_lora_strategy, which leaves it).
        _reject_unknown_keys(cond, {"type", "probability", "p"}, ctype)
        if "probability" in cond or "p" in cond:
            _warn_cli_override(float(getattr(args, "ic_lora_ref_probability", 1.0)) != 1.0, ctype, "--ic_lora_ref_probability")
            args.ic_lora_ref_probability = _resolve_probability(cond, ctype, 1.0)
    else:
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: unknown [video] condition type '{ctype}'. "
            "Known [video] types: first_frame, spatial_crop, inpaint (alias: mask), extend, reference."
        )


def _lower_audio_condition(args: argparse.Namespace, cond: dict, ctype: str, path: str) -> None:
    """Lower one [audio] recipe condition onto the --ltx2_audio_* flags."""
    if ctype == "extend":
        # Lowers onto --ltx2_audio_extend_*. Requires an audio-bearing mode; enforced downstream by
        # validate_audio_extend_setup, not here.
        _reject_unknown_keys(cond, {"type", "probability", "p", "prefix", "suffix"}, ctype)
        _warn_cli_override(
            int(getattr(args, "ltx2_audio_extend_prefix_frames", 0) or 0) > 0
            or int(getattr(args, "ltx2_audio_extend_suffix_frames", 0) or 0) > 0,
            ctype,
            "--ltx2_audio_extend_prefix_frames/--ltx2_audio_extend_suffix_frames",
        )
        args.ltx2_audio_extend_prefix_frames = _int_key(cond, "prefix", ctype)
        args.ltx2_audio_extend_suffix_frames = _int_key(cond, "suffix", ctype)
        args.ltx2_audio_extend_p = _resolve_probability(cond, ctype, 1.0)
        if args.ltx2_audio_extend_prefix_frames <= 0 and args.ltx2_audio_extend_suffix_frames <= 0:
            raise RuntimeError("conditioning recipe: [audio] 'extend' requires prefix > 0 or suffix > 0.")
    elif ctype in ("inpaint", "mask"):
        # Lowers onto --ltx2_audio_inpaint_mask*. Audio-mode-only; enforced downstream by
        # validate_audio_inpaint_mask_setup.
        _reject_unknown_keys(cond, {"type", "probability", "p", "invert", "threshold"}, ctype)
        _warn_cli_override(bool(getattr(args, "ltx2_audio_inpaint_mask", False)), ctype, "--ltx2_audio_inpaint_mask")
        args.ltx2_audio_inpaint_mask = True
        args.ltx2_audio_inpaint_mask_p = _resolve_probability(cond, ctype, 0.0)
        args.ltx2_audio_inpaint_mask_invert = _bool_key(cond, "invert", ctype)
        threshold = float(cond.get("threshold", 0.5))
        if not (0.0 <= threshold <= 1.0):
            raise RuntimeError(f"conditioning recipe: [audio] 'inpaint' threshold must be in [0, 1]. Got: {threshold}")
        args.ltx2_audio_inpaint_mask_threshold = threshold
    elif ctype == "reference":
        # Structural marker only: presence signals an audio reference for strategy resolution. The
        # reference-dropout dial is video-only, so an audio reference takes no parameters.
        _reject_unknown_keys(cond, {"type"}, ctype)
    elif ctype in ("first_frame", "spatial_crop"):
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: '{ctype}' is a video-only condition and is not valid in "
            "the [audio] block. Move it under [[video.conditions]], or remove it."
        )
    else:
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: unknown [audio] condition type '{ctype}'. "
            "Known [audio] types: extend, inpaint (alias: mask), reference."
        )


def derive_ic_lora_strategy(
    has_video: bool,
    has_audio: bool,
    video_generated: bool,
    audio_generated: bool,
    video_has_ref: bool,
    audio_has_ref: bool,
) -> str:
    """Map the recipe structure onto the ic_lora_strategy value the equivalent CLI flag would set.

    Returns ``"none"`` when no reference is declared, so an intrinsic-only (or empty) recipe leaves
    ``--ic_lora_strategy`` untouched. Directional training (a modality's ``is_generated = false``) is
    derived separately by ``derive_train_direction`` and is mutually exclusive with a reference, so the
    ``*_generated`` flags only gate the reference here (a frozen modality carries no reference).
    """
    v_ref = has_video and video_generated and video_has_ref
    a_ref = has_audio and audio_generated and audio_has_ref
    if v_ref and a_ref:
        return "av_ic"
    if v_ref and has_audio and not audio_has_ref:
        return "video_ref_only_av"
    if v_ref and not has_audio:
        return "v2v"
    if a_ref and not video_has_ref:
        return "audio_ref_ic"
    return "none"


def derive_train_direction(video_generated: bool, audio_generated: bool) -> str:
    """Map per-block is_generated onto the --ltx2_train_direction value the equivalent CLI flag would set.

    is_generated = false freezes a modality as clean conditioning; the other is generated. Returns
    ``"a2v"`` (audio frozen), ``"v2a"`` (video frozen), ``"joint"`` (nothing frozen — leaves the flag
    untouched), or ``"invalid"`` (both frozen, which has nothing to generate; the caller raises). The
    av-mode requirement and feature-conflict legality are enforced downstream by
    validate_train_direction_setup, exactly as for the equivalent --ltx2_train_direction flag.
    """
    if not video_generated and not audio_generated:
        return "invalid"
    if video_generated and not audio_generated:
        return "a2v"
    if not video_generated and audio_generated:
        return "v2a"
    return "joint"


def validate_recipe_cross_conditions(path: str, seen_video: set, seen_audio: set, has_reference: bool) -> None:
    """Reject a reference combined with a gated intrinsic in the same recipe (a recipe-structure error).

    A 'reference' selects an IC-LoRA strategy whose path builds its own conditioning/loss masks, so the
    intrinsic validators hard-error on a gated intrinsic + that strategy. Catching it here names the
    offending recipe entries instead of a CLI flag the recipe author never set. This fires only where
    ``validate_ltx2_conditioning_setup`` would also raise after lowering, so a recipe is never rejected
    for a combination the equivalent CLI flags accept. ``first_frame`` composes with IC-LoRA (absent
    from those validators) and is exempt. Mode legality (an audio intrinsic in --ltx2_mode video, a
    video intrinsic in --ltx2_mode audio) is left to those same validators, which run on both paths.
    """
    if not has_reference:
        return
    gated = sorted(
        (seen_video & {"spatial_crop", "inpaint", "extend"}) | {f"audio.{c}" for c in (seen_audio & {"extend", "inpaint"})}
    )
    if gated:
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: a 'reference' condition selects an IC-LoRA strategy "
            f"that cannot be combined with gated intrinsic(s) {gated} in the same recipe (the reference "
            "path builds its own conditioning/loss masks). Remove the reference or the intrinsic(s). "
            "first_frame composes with IC-LoRA and is exempt."
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

    # Per-modality schema: conditions live under [video]/[audio] blocks. The legacy flat top-level
    # [[conditions]] list is no longer supported.
    if "conditions" in data:
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: top-level [[conditions]] is no longer supported. Move "
            "each entry under [[video.conditions]] or [[audio.conditions]] (audio types use "
            "'extend'/'inpaint' inside [audio])."
        )
    if "video" not in data and "audio" not in data:
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: no [video] or [audio] block found. Add at least one "
            "(e.g. a [[video.conditions]] entry)."
        )

    def _block_conditions(block_name: str) -> list:
        block = data.get(block_name)
        if block is None:
            return []
        if not isinstance(block, dict):
            raise RuntimeError(f"--ltx2_conditioning_config {path}: [{block_name}] must be a table.")
        extra = set(block.keys()) - {"conditions", "is_generated"}
        if extra:
            raise RuntimeError(
                f"--ltx2_conditioning_config {path}: [{block_name}] has unknown key(s) {sorted(extra)}; "
                "allowed: ['conditions', 'is_generated'] (did you mistype 'conditions'?)."
            )
        conds = block.get("conditions", [])
        if not isinstance(conds, list):
            raise RuntimeError(
                f"--ltx2_conditioning_config {path}: [{block_name}].conditions must be a list of "
                f"[[{block_name}.conditions]] tables."
            )
        return conds

    def _block_is_generated(block_name: str) -> bool:
        # Per-block is_generated (bool, default true). is_generated=false freezes that modality; the
        # combination is lowered onto --ltx2_train_direction by derive_train_direction below.
        block = data.get(block_name)
        if not isinstance(block, dict):
            return True
        value = block.get("is_generated", True)
        if not isinstance(value, bool):
            raise RuntimeError(
                f"--ltx2_conditioning_config {path}: [{block_name}].is_generated must be a boolean (true/false). Got: {value!r}"
            )
        return value

    def _lower_block(block_name: str, lower_fn) -> set:
        # Recipe fully owns each listed condition (authoritative + declarative-complete): every
        # parameter comes from the recipe, defaulting to the feature's own default when omitted
        # (mirrors the argparse defaults, so a recipe-only run is byte-identical to the equivalent
        # CLI flags).
        seen: set = set()
        for index, cond in enumerate(_block_conditions(block_name)):
            if not isinstance(cond, dict):
                raise RuntimeError(f"--ltx2_conditioning_config {path}: [{block_name}] condition #{index} is not a table.")
            raw_type = cond.get("type")
            if raw_type is None:
                raise RuntimeError(f"--ltx2_conditioning_config {path}: [{block_name}] condition #{index} is missing 'type'.")
            ctype = str(raw_type).lower()
            # Canonicalize aliases so a type and its alias (inpaint == mask) count as one for dedup —
            # both lower onto the same flags, so a second would otherwise silently override the first.
            canon = "inpaint" if ctype in ("inpaint", "mask") else ctype
            if canon in seen:
                raise RuntimeError(f"--ltx2_conditioning_config {path}: duplicate [{block_name}] condition type '{ctype}'.")
            seen.add(canon)
            lower_fn(args, cond, ctype, path)
        return seen

    seen_video = _lower_block("video", _lower_video_condition)
    seen_audio = _lower_block("audio", _lower_audio_condition)

    video_generated = _block_is_generated("video")
    audio_generated = _block_is_generated("audio")
    video_has_ref = "reference" in seen_video
    audio_has_ref = "reference" in seen_audio
    has_reference = video_has_ref or audio_has_ref

    # Reference IC-LoRA strategy: a 'reference' marker selects a reference strategy, lowered onto the
    # raw args.ic_lora_strategy from the recipe structure. The strategy resolution that runs after this
    # normalizes/validates it and sets self._ic_lora_strategy (what the train loop reads), so a recipe-
    # derived strategy and the equivalent --ic_lora_strategy flag take the same path.
    derived_strategy = derive_ic_lora_strategy(
        "video" in data, "audio" in data, video_generated, audio_generated, video_has_ref, audio_has_ref
    )
    if derived_strategy != "none":
        current = str(getattr(args, "ic_lora_strategy", "auto") or "auto").lower()
        _warn_cli_override(current != "auto", "reference", "--ic_lora_strategy")
        args.ic_lora_strategy = derived_strategy  # raw; resolved/validated by the strategy resolution

    # Directional training: is_generated = false freezes a modality (clean conditioning) and generates
    # the other, lowered onto the raw args.ltx2_train_direction. resolve_train_direction normalizes it,
    # and validate_train_direction_setup enforces the av-mode requirement and feature conflicts, so a
    # recipe-derived direction and the equivalent --ltx2_train_direction flag take the same path.
    derived_direction = derive_train_direction(video_generated, audio_generated)
    if derived_direction == "invalid":
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: both [video] and [audio] set is_generated = false, so no "
            "modality is generated. Set exactly one to false (a2v freezes audio; v2a freezes video)."
        )
    # Reference + directional are mutually exclusive — keyed on the RAW reference marker, not the
    # derived strategy. A reference on a FROZEN modality derives strategy "none" (derive_ic_lora_strategy
    # gates each ref on its generated flag), but it is a contradictory declaration (a frozen modality is
    # clean conditioning, not an IC-LoRA reference) with no faithful CLI lowering. Rejecting it is better
    # than silently dropping the marker, so do NOT weaken this to `derived_strategy != "none"`.
    if has_reference and derived_direction != "joint":
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: a 'reference' condition (IC-LoRA) cannot be combined with "
            "directional training (is_generated = false); the IC-LoRA reference path owns the modality "
            "streams, and a reference on a frozen modality is contradictory. Use one or the other."
        )
    if derived_direction != "joint":
        _warn_cli_override(
            str(getattr(args, "ltx2_train_direction", "joint") or "joint").lower() != "joint",
            "is_generated",
            "--ltx2_train_direction",
        )
        args.ltx2_train_direction = derived_direction  # raw; resolved/validated by resolve_train_direction

    validate_recipe_cross_conditions(path, seen_video, seen_audio, has_reference)

    # Declarative-complete: a recipe that does not list a [video] 'first_frame' disables it (parity
    # with the other intrinsics, off unless listed) — UNLESS the recipe selects a reference strategy or
    # is directional. A bare --ic_lora_strategy / --ltx2_train_direction leaves first_frame at its
    # default (v2a's validator then auto-disables it; a2v keeps it), so a reference/directional recipe
    # must match that and does not auto-disable first_frame here.
    if "first_frame" not in seen_video and not has_reference and derived_direction == "joint":
        prev = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
        if prev != 0.0:
            logger.info(
                "Conditioning recipe %s does not list a [video] 'first_frame'; disabling first-frame "
                "conditioning (was p=%.3f). Add a [[video.conditions]] first_frame to keep it.",
                path,
                prev,
            )
        args.ltx2_first_frame_conditioning_p = 0.0

    # Per-sample loss is a top-level loss policy (not a per-condition entry). The recipe owns it like
    # any other dial: set from the recipe top-level 'per_sample_loss' (default False when omitted),
    # overriding a CLI --ltx2_per_sample_loss with a warning. A recipe therefore never IMPLICITLY
    # changes the loss reduction; per-sample loss is opt-in here exactly as it is on the command line.
    _warn_cli_override(bool(getattr(args, "ltx2_per_sample_loss", False)), "per_sample_loss", "--ltx2_per_sample_loss")
    ps_loss = data.get("per_sample_loss", False)
    if not isinstance(ps_loss, bool):
        raise RuntimeError(
            f"--ltx2_conditioning_config {path}: top-level 'per_sample_loss' must be a boolean (true/false). Got: {ps_loss!r}"
        )
    args.ltx2_per_sample_loss = ps_loss

    setattr(args, _SENTINEL, True)
    logger.info("Applied conditioning recipe from %s: video=%s audio=%s", path, sorted(seen_video), sorted(seen_audio))
