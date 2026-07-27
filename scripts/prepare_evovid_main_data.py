#!/usr/bin/env python3
"""Validate EvoVid preprocessed rows and build the Questioner training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


QUESTIONER_TASK = "Generate exactly one reasoning question based on this video."


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object.")
            yield line_number, row


def _prepare_row(
    source: dict[str, Any], *, index: int, line_number: int, video_dir: Path
) -> dict[str, Any]:
    preprocessed_video = source.get("preprocessed_video")
    videos = source.get("videos")
    if not isinstance(preprocessed_video, str) or not preprocessed_video:
        raise ValueError(f"Line {line_number} has no preprocessed_video reference.")
    artifact = Path(preprocessed_video)
    if not artifact.is_absolute():
        artifact = video_dir / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str):
        raise ValueError(f"Line {line_number} must contain exactly one source video path.")

    row = dict(source)
    row.update(
        {
            "problem_id": str(source.get("problem_id", source.get("id", index))),
            "problem": QUESTIONER_TASK,
            "answer": "unused",
            "data_type": "video",
            "problem_type": "question generation",
            "preprocessed_video": str(artifact),
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--preprocessed_video_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=0, help="0 means all rows")
    parser.add_argument("--expected_count", type=int, default=5778)
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be non-negative.")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate_only is used.")
    return args


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.preprocessed_video_dir.is_dir():
        raise NotADirectoryError(args.preprocessed_video_dir)

    prepared: list[dict[str, Any]] = []
    problem_ids: set[str] = set()
    for index, (line_number, source) in enumerate(_rows(args.input)):
        if args.count and index >= args.count:
            break
        row = _prepare_row(
            source,
            index=index,
            line_number=line_number,
            video_dir=args.preprocessed_video_dir,
        )
        problem_id = row["problem_id"]
        if problem_id in problem_ids:
            raise ValueError(f"Duplicate problem_id: {problem_id}")
        problem_ids.add(problem_id)
        prepared.append(row)

    expected = args.count or args.expected_count
    if expected > 0 and len(prepared) != expected:
        raise RuntimeError(f"Expected {expected} usable rows, found {len(prepared)}.")

    if not args.validate_only:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in prepared:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    print(
        json.dumps(
            {
                "rows": len(prepared),
                "artifacts_checked": len(prepared),
                "mode": "validate_only" if args.validate_only else "write",
                "output": str(args.output) if args.output else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
