#!/usr/bin/env python3
"""Build globally balanced contiguous-frame windows for Questioner generation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            records = payload["data"]
        else:
            raise TypeError(f"{path} must contain a JSON list or a JSON object with a 'data' list.")

    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"Every record in {path} must be a JSON object.")
    return records


def _resolve_video_pt(
    record: dict[str, Any],
    *,
    dataset_path: Path,
    preprocessed_video_dir: Path | None,
) -> Path:
    raw_path = record.get("video_pt_path") or record.get("preprocessed_video")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("record has neither video_pt_path nor preprocessed_video")

    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute() and preprocessed_video_dir is not None:
        candidates.append(preprocessed_video_dir / path)
    if not path.is_absolute():
        candidates.append(dataset_path.parent / path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"preprocessed video artifact not found; searched: {searched}")


def _frame_count(video_pt_path: Path) -> int:
    artifact = torch.load(video_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "frames" not in artifact:
        raise KeyError(f"{video_pt_path} must contain a 'frames' field")
    frames = artifact["frames"]
    if isinstance(frames, torch.Tensor):
        return int(frames.shape[0])
    return len(frames)


def _parse_window_starts(raw: str) -> list[int]:
    starts = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not starts or any(start < 0 for start in starts):
        raise ValueError("--window_starts must contain non-negative comma-separated integers.")
    if len(set(starts)) != len(starts):
        raise ValueError("--window_starts must not contain duplicates.")
    return starts


def _balanced_starts(total_samples: int, starts: list[int], rng: random.Random) -> list[int]:
    base, remainder = divmod(total_samples, len(starts))
    assignments = [start for index, start in enumerate(starts) for _ in range(base + int(index < remainder))]
    rng.shuffle(assignments)
    return assignments


def _balanced_videos(
    videos: list[dict[str, Any]], total_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    while len(assignments) < total_samples:
        epoch = videos.copy()
        rng.shuffle(epoch)
        assignments.extend(epoch[: total_samples - len(assignments)])
    return assignments


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.total_samples < 1:
        raise ValueError("--total_samples must be positive.")
    if args.window_size < 1:
        raise ValueError("--window_size must be positive.")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive.")

    dataset_path = args.video_data.expanduser().resolve()
    preprocessed_video_dir = (
        args.preprocessed_video_dir.expanduser().resolve() if args.preprocessed_video_dir else None
    )
    window_starts = _parse_window_starts(args.window_starts)
    required_frames = max(window_starts) + args.window_size

    compatible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_index, record in enumerate(_load_records(dataset_path)):
        try:
            video_pt_path = _resolve_video_pt(
                record,
                dataset_path=dataset_path,
                preprocessed_video_dir=preprocessed_video_dir,
            )
            frame_count = _frame_count(video_pt_path)
            if frame_count < required_frames:
                raise ValueError(f"has {frame_count} frames; at least {required_frames} are required")
        except Exception as exc:
            skipped.append({"source_index": source_index, "error": str(exc)})
            continue

        compatible.append(
            {
                "source_index": source_index,
                "source_id": str(record.get("problem_id", record.get("id", source_index))),
                "video_pt_path": str(video_pt_path),
                "frame_count": frame_count,
                "videos": record.get("videos", []),
            }
        )

    if not compatible:
        raise RuntimeError(
            f"No compatible videos in {dataset_path}; every window requires at least {required_frames} frames."
        )

    rng = random.Random(args.seed)
    start_assignments = _balanced_starts(args.total_samples, window_starts, rng)
    video_assignments = _balanced_videos(compatible, args.total_samples, rng)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    video_usage: Counter[str] = Counter()

    for manifest_index, (video, window_start) in enumerate(zip(video_assignments, start_assignments)):
        window_end = window_start + args.window_size
        sample_id = f"{args.save_name}_{manifest_index:08d}"
        row = {
            **video,
            "sample_id": sample_id,
            "manifest_index": manifest_index,
            "window_start_frame": window_start,
            "window_end_frame_exclusive": window_end,
            "window_size": args.window_size,
        }
        shards[manifest_index % args.num_shards].append(row)
        video_usage[video["video_pt_path"]] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_index, rows in enumerate(shards):
        shard_path = args.output_dir / f"{args.save_name}_{shard_index}.jsonl"
        _write_jsonl(shard_path, rows)
        shard_paths.append(str(shard_path))

    summary = {
        "video_data": str(dataset_path),
        "total_samples": args.total_samples,
        "window_size": args.window_size,
        "window_starts": window_starts,
        "window_counts": dict(sorted(Counter(start_assignments).items())),
        "compatible_video_count": len(compatible),
        "skipped_video_count": len(skipped),
        "video_usage_min": min(video_usage.values()),
        "video_usage_max": max(video_usage.values()),
        "num_shards": args.num_shards,
        "shard_counts": [len(rows) for rows in shards],
        "shard_paths": shard_paths,
        "skipped": skipped,
    }
    summary_path = args.output_dir / f"{args.save_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exactly balanced global assignment of contiguous video windows."
    )
    parser.add_argument("--video_data", type=Path, required=True, help="JSON/JSONL records with .pt references.")
    parser.add_argument("--preprocessed_video_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--total_samples", type=int, default=9000)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_starts", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    report = build_manifest(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
