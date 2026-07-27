#!/usr/bin/env python3
"""Shuffle Questioner rows once and write disjoint per-iteration JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create disjoint Questioner JSONL chunks from one seeded shuffle."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_iterations", type=int, required=True)
    parser.add_argument("--samples_per_iteration", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.num_iterations <= 0:
        parser.error("--num_iterations must be positive.")
    if args.samples_per_iteration <= 0:
        parser.error("--samples_per_iteration must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    rows = _load_rows(args.input)
    required = args.num_iterations * args.samples_per_iteration
    if required > len(rows):
        raise ValueError(
            f"Requested {required} rows "
            f"({args.num_iterations} iterations x {args.samples_per_iteration}), "
            f"but {args.input} contains only {len(rows)}."
        )

    random.Random(args.seed).shuffle(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    for iteration in range(1, args.num_iterations + 1):
        start = (iteration - 1) * args.samples_per_iteration
        end = start + args.samples_per_iteration
        output = args.output_dir / f"iteration_{iteration:03d}.jsonl"
        _write_jsonl(output, rows[start:end])
        files.append(
            {
                "iteration": iteration,
                "start": start,
                "end_exclusive": end,
                "rows": args.samples_per_iteration,
                "path": str(output),
            }
        )

    manifest = {
        "input": str(args.input),
        "input_rows": len(rows),
        "seed": args.seed,
        "num_iterations": args.num_iterations,
        "samples_per_iteration": args.samples_per_iteration,
        "selected_rows": required,
        "files": files,
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
