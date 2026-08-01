from __future__ import annotations

import argparse
import json
import logging

from musubi_tuner.modules.convrot_policy import build_convrot_policy_from_quality_report, write_convrot_policy


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-layer ConvRot policy from an INT8/INT4 quality report")
    parser.add_argument("--quality_report", required=True, help="Input INT8/INT4 ConvRot quality JSON")
    parser.add_argument("--output", required=True, help="Output ltx2_convrot_policy_v1 JSON")
    parser.add_argument("--min_cosine", type=float, default=None, help="Flag layers with cosine below this value")
    parser.add_argument("--min_sqnr_db", type=float, default=None, help="Flag layers with SQNR below this value")
    parser.add_argument("--max_mse", type=float, default=None, help="Flag layers with MSE above this value")
    parser.add_argument(
        "--action",
        choices=["dequantize", "keep_bf16"],
        default="dequantize",
        help=(
            "dequantize keeps compressed storage but uses dense floating-point matmul; "
            "keep_bf16 also prevents on-the-fly quantization when the policy is used with a standard checkpoint"
        ),
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with open(args.quality_report, encoding="utf-8") as handle:
        report = json.load(handle)
    policy = build_convrot_policy_from_quality_report(
        report,
        min_cosine=args.min_cosine,
        min_sqnr_db=args.min_sqnr_db,
        max_mse=args.max_mse,
        action=args.action,
    )
    write_convrot_policy(args.output, policy)
    logger.info("Wrote ConvRot policy with %d layer overrides to %s", len(policy.rules), args.output)


if __name__ == "__main__":
    main()
