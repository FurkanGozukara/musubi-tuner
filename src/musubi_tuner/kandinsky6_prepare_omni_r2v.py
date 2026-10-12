"""Convert Omni-R2V annotations into a Kandinsky 6 R2AV video JSONL.

Omni-R2V (https://huggingface.co/datasets/wxli318/Omni-R2V, research-only licence) stores
records in ``meta/<aspect>/<subtask>.jsonl``::

    {"sample_id": ..., "prompt_en": "<Subject 1> is the boy in <Figure 1> ...",
     "inputs": [{"role": "subject", "assets": [{"modality": "image", "rel_path": ...}]}],
     "target": "assets/.../target.mp4"}

``<Figure N>`` is the N-th image across ``inputs`` (in order). The converter keeps
image-only records, shuffles the reference order once per record (so the slot a subject
lands in does not follow mention order), rewrites the tags to ``@Image k`` and writes::

    {"video_path": ..., "caption": ..., "references": [{"type": "image", "path": ...}], "sample_id": ...}

Records whose media are missing are skipped, so a partially extracted dataset works.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from musubi_tuner.kandinsky6.reference import (
    DEFAULT_REFERENCE_SLOTS,
    IMAGE_TAG_FORMAT,
    mentioned_figures,
    mentions_video,
    rewrite_figure_tags,
)

logger = logging.getLogger(__name__)

DEFAULT_SUBTASKS = ("content", "multi_content")


def find_annotation_files(root: Path, subtasks: list[str]) -> list[Path]:
    """``meta/**/<subtask>.jsonl`` for each subtask name (or an explicit ``aspect/subtask``)."""
    files: list[Path] = []
    for subtask in subtasks:
        name = subtask if subtask.endswith(".jsonl") else f"{subtask}.jsonl"
        direct = root / "meta" / name
        matches = [direct] if direct.is_file() else sorted((root / "meta").rglob(Path(name).name))
        if "/" in subtask or "\\" in subtask:
            matches = [match for match in matches if match.as_posix().endswith(Path(name).as_posix())]
        if not matches:
            raise FileNotFoundError(f"no annotation file for subtask {subtask!r} under {root / 'meta'}")
        files.extend(match for match in matches if match not in files)
    return files


def convert_record(
    record: dict,
    root: Path,
    rng: random.Random,
    *,
    max_refs: int,
    tag_format: str,
    prompt_key: str,
    drop_unmentioned: bool,
    shuffle: bool,
    check_files: bool,
) -> tuple[dict | None, str | None]:
    """Return (output line, None) or (None, skip reason)."""
    prompt = record.get(prompt_key)
    if not prompt or not isinstance(prompt, str):
        return None, "no_prompt"
    target = record.get("target")
    if not target:
        return None, "no_target"
    if mentions_video(prompt):
        return None, "video_reference"
    images: list[Path] = []
    for group in record.get("inputs") or []:
        for asset in group.get("assets") or []:
            modality = str(asset.get("modality", "")).lower()
            if modality != "image":
                return None, "video_reference"
            images.append(root / asset["rel_path"])
    if not images:
        return None, "no_images"
    mentioned = mentioned_figures(prompt)
    if any(index < 1 or index > len(images) for index in mentioned):
        return None, "missing_figure"
    if not mentioned:
        return None, "no_figure_tag"
    keep = sorted(mentioned) if drop_unmentioned else list(range(1, len(images) + 1))
    if len(keep) > max_refs:
        return None, "too_many_refs"
    target_path = root / target
    if check_files and (not target_path.is_file() or any(not images[index - 1].is_file() for index in keep)):
        return None, "missing_file"
    order = list(range(len(keep)))
    if shuffle:
        rng.shuffle(order)
    # keep[order[k]] is the original figure placed in slot k
    mapping = {keep[original]: slot + 1 for slot, original in enumerate(order)}
    caption = rewrite_figure_tags(prompt, mapping, tag_format)
    references = [{"type": "image", "path": str(images[keep[original] - 1])} for original in order]
    line = {
        "video_path": str(target_path),
        "caption": caption,
        "references": references,
        "sample_id": record.get("sample_id"),
    }
    return line, None


def convert(args: argparse.Namespace) -> Counter:
    root = Path(args.root).resolve()
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_key = f"prompt_{args.language}"
    with output.open("w", encoding="utf-8") as sink:
        for annotation in find_annotation_files(root, [item.strip() for item in args.subtasks.split(",") if item.strip()]):
            logger.info("reading %s", annotation)
            with annotation.open(encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    stats["records"] += 1
                    line, reason = convert_record(
                        json.loads(raw),
                        root,
                        rng,
                        max_refs=args.max_refs,
                        tag_format=args.tag_format,
                        prompt_key=prompt_key,
                        drop_unmentioned=args.drop_unmentioned,
                        shuffle=args.shuffle,
                        check_files=not args.no_file_check,
                    )
                    if line is None:
                        stats[f"skip_{reason}"] += 1
                        continue
                    sink.write(json.dumps(line, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats[f"refs_{len(line['references'])}"] += 1
                    if args.max_records and stats["written"] >= args.max_records:
                        return stats
    return stats


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Omni-R2V root (contains meta/ and the extracted assets/)")
    parser.add_argument("--output", required=True, help="output video JSONL for the Kandinsky 6 dataset config")
    parser.add_argument(
        "--subtasks",
        default=",".join(DEFAULT_SUBTASKS),
        help="comma-separated subtask names (meta/**/<name>.jsonl); image-reference subtasks only",
    )
    parser.add_argument("--max_refs", type=int, default=DEFAULT_REFERENCE_SLOTS, help="skip records needing more references")
    parser.add_argument("--tag_format", default=IMAGE_TAG_FORMAT, help="replacement for <Figure N>; must contain {n}")
    parser.add_argument("--language", choices=("en", "cn"), default="en", help="prompt_en or prompt_cn")
    parser.add_argument("--seed", type=int, default=0, help="seed of the per-record reference shuffle")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle reference order per record so slots do not follow mention order",
    )
    parser.add_argument(
        "--drop_unmentioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop images the prompt never mentions (they would have no tag to bind to)",
    )
    parser.add_argument("--max_records", type=int, default=0, help="stop after this many written records (0 = all)")
    parser.add_argument("--no_file_check", action="store_true", help="do not skip records whose media are missing")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = setup_parser().parse_args(argv)
    if "{n}" not in args.tag_format:
        raise ValueError("--tag_format must contain {n}")
    if args.max_refs < 1:
        raise ValueError("--max_refs must be positive")
    stats = convert(args)
    for key in sorted(stats):
        logger.info("%s: %d", key, stats[key])


if __name__ == "__main__":
    main()
