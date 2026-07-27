#!/usr/bin/env python3
"""Curate evaluated EvoVid questions and optionally upload the Solver dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


_SEGMENT_TAG_PATTERN = re.compile(r"</?segment\b", re.IGNORECASE)


def _load_result_shards(
    generated_dir: Path,
    experiment_name: str,
    input_files: list[Path] | None = None,
) -> list[dict[str, Any]]:
    paths = [path.expanduser().resolve() for path in input_files] if input_files else sorted(
        generated_dir.glob(f"{experiment_name}_*_results.json")
    )
    if not paths:
        raise FileNotFoundError(
            f"No evaluation result shards matching {experiment_name}_*_results.json in {generated_dir}"
        )

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise TypeError(f"{path} must contain a JSON list of objects.")
        rows.extend(payload)
    return rows


def _training_row(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("problem") or row.get("question")
    answer = row.get("answer")
    videos = row.get("videos")
    preprocessed_video = row.get("preprocessed_video") or row.get("video_pt_path")
    target_segment = row.get("target_segment")
    if not isinstance(target_segment, (list, tuple)) or len(target_segment) != 2:
        target_segment = [row.get("segment_start_sec"), row.get("segment_end_sec")]

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Curated row has no question text.")
    if not isinstance(answer, str) or not answer.strip() or answer.strip().lower() == "none":
        raise ValueError("Curated row has no majority-voted answer.")
    if _SEGMENT_TAG_PATTERN.search(answer):
        raise ValueError("Majority-voted answer contains a <segment> tag.")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Curated video row must contain a non-empty videos list.")
    if not isinstance(preprocessed_video, str) or not preprocessed_video:
        raise ValueError("Curated video row has no preprocessed_video path.")
    try:
        start, end = float(target_segment[0]), float(target_segment[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Curated row has no numeric target segment.") from exc
    if start < 0 or end <= start:
        raise ValueError(f"Invalid target segment [{start}, {end}].")

    selected_keys = (
        "source_id",
        "sample_id",
        "manifest_index",
        "window_start_frame",
        "window_end_frame_exclusive",
        "window_source_frame_indices",
        "segment_start_sec",
        "segment_end_sec",
        "questioner_answer",
        "score",
    )
    result = {key: row[key] for key in selected_keys if key in row}
    result.update(
        {
            "problem_id": row.get("problem_id", row.get("sample_id", row.get("source_id"))),
            "problem": question.strip(),
            "answer": answer.strip(),
            "data_type": "video",
            "problem_type": row.get("problem_type") or row.get("question_type", ""),
            "videos": videos,
            "preprocessed_video": preprocessed_video,
            "target_segment": [start, end],
        }
    )
    return result


def curate_rows(
    rows: list[dict[str, Any]], *, min_score: float, max_score: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        try:
            score = float(row.get("score", -1))
            if not min_score <= score <= max_score:
                continue
            curated.append(_training_row(row))
        except Exception as exc:
            rejected.append(
                {
                    "problem_id": row.get("problem_id", row.get("sample_id")),
                    "error": str(exc),
                }
            )
    return curated, rejected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _push_to_hub(rows: list[dict[str, Any]], *, namespace: str, repo_name: str, config_name: str) -> None:
    if not rows:
        raise RuntimeError("Cannot push an empty curated Solver dataset.")
    from datasets import Dataset, DatasetDict

    dataset = DatasetDict({"train": Dataset.from_list(rows)})
    dataset.push_to_hub(f"{namespace}/{repo_name}", private=True, config_name=config_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", default="")
    parser.add_argument("--max_score", type=float, default=0.8)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--experiment_name", default="Qwen_Qwen3-4B-Base_all")
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--input_files", type=Path, nargs="+")
    parser.add_argument("--cleanup_result_shards", action="store_true")
    args = parser.parse_args()
    if args.min_score > args.max_score:
        parser.error("--min_score must be no greater than --max_score.")
    return args


def main(args: argparse.Namespace) -> dict[str, Any]:
    storage_path = os.getenv("STORAGE_PATH")
    if not storage_path:
        raise EnvironmentError("Set STORAGE_PATH first.")
    generated_dir = Path(storage_path) / "generated_question"
    rows = _load_result_shards(generated_dir, args.experiment_name, args.input_files)
    curated, rejected = curate_rows(rows, min_score=args.min_score, max_score=args.max_score)

    output_file = args.output_file or generated_dir / f"{args.experiment_name}_train.json"
    _write_json(output_file, curated)
    rejected_file = output_file.with_name(f"{output_file.stem}_rejected.json")
    _write_json(rejected_file, rejected)

    if args.repo_name:
        namespace = os.getenv("HUGGINGFACENAME")
        if not namespace:
            raise EnvironmentError("Set HUGGINGFACENAME before uploading the dataset.")
        _push_to_hub(
            curated,
            namespace=namespace,
            repo_name=args.repo_name,
            config_name=args.experiment_name,
        )

    if args.cleanup_result_shards:
        for path in generated_dir.glob(f"{args.experiment_name}_*_results.json"):
            path.unlink()

    summary = {
        "evaluated": len(rows),
        "in_score_band": len(curated) + len(rejected),
        "curated": len(curated),
        "rejected_metadata": len(rejected),
        "output": str(output_file),
        "uploaded": bool(args.repo_name),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main(parse_args())
