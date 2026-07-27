#!/usr/bin/env python3
"""Generate Solver-training questions from assigned contiguous video windows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import vllm
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from prompt_role_utils import build_chat_messages


_RESPONSE_PATTERN = re.compile(
    r"^\s*<type>\s*(multiple choice|numerical|regression)\s*</type>\s*"
    r"<question>\s*(.*?)\s*</question>\s*<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL,
)
_VALID_TYPES = {"multiple choice", "numerical", "regression"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Every row in {path} must be a JSON object.")
    return rows


def _load_instruction(path: Path) -> str:
    instruction = path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"Questioner prompt is empty: {path}")
    return instruction


def _slice_frames(frames: Any, start: int, end: int) -> Any:
    selected = frames[start:end]
    count = int(selected.shape[0]) if hasattr(selected, "shape") else len(selected)
    if count != end - start:
        raise ValueError(f"Requested frames[{start}:{end}], but only received {count} frames.")
    return selected


def _window_metadata(metadata: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    result = dict(metadata)
    frame_indices = metadata.get("frames_indices")
    if frame_indices is not None:
        selected = list(frame_indices)[start:end]
        if len(selected) != end - start:
            raise ValueError("metadata.frames_indices is shorter than the requested window.")
        result["frames_indices"] = selected
    return result


def _metadata_to_vllm(metadata: dict[str, Any], frames: Any) -> VideoMetadata:
    frame_count = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)
    return VideoMetadata(
        total_num_frames=metadata.get("total_num_frames", frame_count),
        fps=metadata.get("fps"),
        frames_indices=metadata.get("frames_indices"),
        video_backend=metadata.get("video_backend"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        duration=metadata.get("duration"),
    )


def _segment_seconds(
    metadata: dict[str, Any], artifact: dict[str, Any], start: int, end: int
) -> tuple[float | None, float | None]:
    frame_indices = metadata.get("frames_indices")
    fps = metadata.get("fps")
    if frame_indices is not None and isinstance(fps, (int, float)) and float(fps) > 0:
        selected = list(frame_indices)[start:end]
        if selected:
            return float(selected[0]) / float(fps), float(selected[-1]) / float(fps)

    sample_fps = artifact.get("sample_fps")
    if isinstance(sample_fps, (int, float)) and float(sample_fps) > 0:
        return float(start) / float(sample_fps), float(end - 1) / float(sample_fps)
    return None, None


def _load_window(row: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    video_pt_path = Path(row["video_pt_path"])
    artifact = torch.load(video_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "frames" not in artifact or "metadata" not in artifact:
        raise KeyError(f"{video_pt_path} must contain 'frames' and 'metadata'.")

    start = int(row["window_start_frame"])
    end = int(row["window_end_frame_exclusive"])
    frames = _slice_frames(artifact["frames"], start, end)
    metadata = dict(artifact["metadata"])
    window_metadata = _window_metadata(metadata, start, end)
    start_sec, end_sec = _segment_seconds(metadata, artifact, start, end)
    source_indices = window_metadata.get("frames_indices")
    window_info = {
        "window_source_frame_indices": list(source_indices) if source_indices is not None else list(range(start, end)),
        "segment_start_sec": start_sec,
        "segment_end_sec": end_sec,
    }
    return frames, window_metadata, window_info


def _build_prompt(processor: Any, tokenizer: Any, instruction: str) -> list[int]:
    messages = build_chat_messages(instruction, modality="video")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(prompt, add_special_tokens=False)


def _parse_response(response: str) -> tuple[str, str, str] | None:
    match = _RESPONSE_PATTERN.fullmatch(response)
    if match is None:
        return None
    question_type = match.group(1).strip()
    question = match.group(2).strip()
    answer = match.group(3).strip()
    if question_type not in _VALID_TYPES or not question or not answer:
        return None
    return question_type, question, answer


def _error_result(row: dict[str, Any], error: Exception | str) -> dict[str, Any]:
    return {
        **row,
        "question_type": "",
        "question": "",
        "answer": "",
        "score": -1,
        "error": str(error),
    }


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _load_jsonl(args.manifest)
    instruction = _load_instruction(args.prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    raw_prompt_ids = _build_prompt(processor, tokenizer, instruction)

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"video": 1},
    )
    sample_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        valid_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        video_inputs: list[dict[str, Any]] = []
        for row in batch_rows:
            try:
                frames, metadata, window_info = _load_window(row)
                video_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": {"video": [(frames, _metadata_to_vllm(metadata, frames))]},
                        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
                    }
                )
                valid_rows.append((row, window_info))
            except Exception as exc:
                results.append(_error_result(row, exc))

        if not video_inputs:
            continue
        responses = model.generate(video_inputs, sampling_params=sample_params, use_tqdm=True)
        for (row, window_info), response in zip(valid_rows, responses):
            raw_response = response.outputs[0].text
            parsed = _parse_response(raw_response)
            if parsed is None:
                results.append(
                    {
                        **_error_result(row, "Questioner output did not match the required three-block format."),
                        **window_info,
                        "questioner_response": raw_response,
                    }
                )
                continue
            question_type, question, answer = parsed
            results.append(
                {
                    **row,
                    **window_info,
                    "question_type": question_type,
                    "question": question,
                    "answer": answer,
                    "score": 0,
                    "questioner_response": raw_response,
                }
            )

    results.sort(key=lambda item: int(item["manifest_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    temporary.replace(args.output)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one video question per assigned 8-frame window.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt_template",
        type=Path,
        default=Path("examples/format_prompt/questioner_strict.jinja"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    generated = generate(parse_args())
    valid_count = sum(item.get("score") == 0 for item in generated)
    print(json.dumps({"total": len(generated), "valid": valid_count}, ensure_ascii=False))
