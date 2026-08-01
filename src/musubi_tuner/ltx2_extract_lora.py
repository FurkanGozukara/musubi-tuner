"""Standalone LTX-2 full-finetune to LoRA extraction entry point."""

from musubi_tuner.ltx_2.extract_lora import (
    ExtractionConfig,
    ExtractionSummary,
    config_from_args,
    extract_lora,
    main,
    module_path_from_unified_key,
    select_rank,
    setup_parser,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionSummary",
    "config_from_args",
    "extract_lora",
    "main",
    "module_path_from_unified_key",
    "select_rank",
    "setup_parser",
]


if __name__ == "__main__":
    raise SystemExit(main())
