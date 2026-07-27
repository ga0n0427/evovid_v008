#!/usr/bin/env python3
"""Build EvoVid Solver pseudo-labels from full-video Solver responses.

The Questioner sees one contiguous frame window when it creates a question, but
the frozen/current Solver must answer that question from the full preprocessed
video.  This module samples M Solver responses, obtains a majority-voted answer,
and keeps the window timestamps attached for the later temporal-IoU reward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import stopit
import torch
import vllm
from jinja2 import Template
from mathruler.grader import extract_boxed_content, grade_answer
from solver_answer_normalization import normalize_mc_option_token
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from prompt_role_utils import build_chat_messages


_SEGMENT_PATTERN = re.compile(
    r"<segment>\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*s?\s*[-\u2013\u2014]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*s?\s*</segment>",
    re.IGNORECASE,
)
_SEGMENT_TAG_PATTERN = re.compile(r"</?segment\b", re.IGNORECASE)


@stopit.threading_timeoutable(default="TIMED_OUT")
def grade_answer_with_timeout(answer_a: str, answer_b: str) -> bool:
    """Bound the symbolic answer-equivalence fallback."""
    return grade_answer(answer_a, answer_b)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError(f"{path} must contain a JSON list of objects.")
    return payload


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _load_prompt_template(path: Path) -> Template:
    source = path.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError(f"Solver prompt template is empty: {path}")
    return Template(source)


def _render_solver_instruction(template: Template, question: str) -> str:
    instruction = template.render(content=question, problem=question).strip()
    if not instruction:
        raise ValueError("Rendered Solver prompt is empty.")
    return instruction


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


def _target_segment(row: dict[str, Any]) -> list[float]:
    segment = row.get("target_segment")
    if isinstance(segment, (list, tuple)) and len(segment) == 2:
        start, end = segment
    else:
        start, end = row.get("segment_start_sec"), row.get("segment_end_sec")

    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("Target segment endpoints must be numeric seconds.")
    try:
        start_value, end_value = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("Generated question is missing numeric segment timestamps.") from exc
    if not math.isfinite(start_value) or not math.isfinite(end_value):
        raise ValueError("Target segment endpoints must be finite.")
    if start_value < 0 or end_value <= start_value:
        raise ValueError(f"Invalid target segment [{start_value}, {end_value}].")
    return [start_value, end_value]


def _load_full_video(row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    video_pt_path = row.get("video_pt_path") or row.get("preprocessed_video")
    if not isinstance(video_pt_path, str) or not video_pt_path:
        raise ValueError("Generated question has no video_pt_path or preprocessed_video.")
    path = Path(video_pt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessed video artifact not found: {path}")

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "frames" not in artifact or "metadata" not in artifact:
        raise KeyError(f"{path} must contain 'frames' and 'metadata'.")
    frames = artifact["frames"]
    frame_count = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)
    if frame_count < 1:
        raise ValueError(f"Preprocessed video contains no frames: {path}")
    return frames, dict(artifact["metadata"])


def _build_video_input(
    row: dict[str, Any],
    *,
    processor: Any,
    tokenizer: Any,
    prompt_template: Template,
) -> dict[str, Any]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question text is empty.")
    _target_segment(row)
    frames, metadata = _load_full_video(row)
    instruction = _render_solver_instruction(prompt_template, question.strip())
    messages = build_chat_messages(instruction, modality="video")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return {
        # Keep one raw video placeholder. vLLM performs the visual-token
        # expansion exactly once when it consumes multi_modal_data.
        "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
        "multi_modal_data": {"video": [(frames, _metadata_to_vllm(metadata, frames))]},
        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
    }


def _extract_candidate_answer(text: str) -> str:
    answer = extract_boxed_content(text)
    if answer is None:
        return ""
    normalized = str(answer).strip()
    if not normalized or normalized.lower() == "none":
        return ""
    if _SEGMENT_TAG_PATTERN.search(normalized):
        return ""
    return normalize_mc_option_token(normalized) or normalized


def _extract_candidate_segment(text: str) -> list[float] | None:
    matches = list(_SEGMENT_PATTERN.finditer(text))
    if not matches:
        return None
    start, end = (float(value) for value in matches[-1].groups())
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None
    return [start, end]


def _answers_equivalent(answer_a: str, answer_b: str) -> bool:
    if answer_a.strip().casefold() == answer_b.strip().casefold():
        return True
    for left, right in ((answer_a, answer_b), (answer_b, answer_a)):
        try:
            result = grade_answer_with_timeout(left, right, timeout=10)
        except Exception as exc:
            print(f"[grader] comparison failed: {exc}")
            continue
        if result != "TIMED_OUT" and bool(result):
            return True
    return False


def score_solver_responses(response_texts: list[str]) -> dict[str, Any]:
    """Majority-vote every response containing a parseable answer.

    The fixed denominator remains M (the total number of sampled Solver
    responses), so outputs without a parseable answer still lower confidence.
    Temporal segments are retained only as diagnostics and do not affect the
    pseudo-label or its confidence.
    """
    candidate_answers = [_extract_candidate_answer(text) for text in response_texts]
    candidate_segments = [_extract_candidate_segment(text) for text in response_texts]
    valid_pairs = [
        bool(answer) and segment is not None
        for answer, segment in zip(candidate_answers, candidate_segments)
    ]
    answer_counts: dict[str, int] = {}
    for candidate in candidate_answers:
        if not candidate:
            continue
        matched_answer = next(
            (existing for existing in answer_counts if _answers_equivalent(candidate, existing)),
            None,
        )
        if matched_answer is None:
            answer_counts[candidate] = 1
        else:
            answer_counts[matched_answer] += 1

    if answer_counts:
        majority_answer = max(answer_counts, key=answer_counts.get)
        majority_count = answer_counts[majority_answer]
    else:
        majority_answer = ""
        majority_count = 0
    confidence = majority_count / len(response_texts) if response_texts else 0.0
    return {
        "answer": majority_answer,
        "score": float(confidence),
        "majority_count": majority_count,
        "valid_answer_count": sum(bool(answer) for answer in candidate_answers),
        "valid_pair_count": sum(valid_pairs),
        "candidate_answers": candidate_answers,
        "candidate_segments": candidate_segments,
    }


def _common_result_fields(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    question = str(row.get("question", ""))
    video_pt_path = row.get("video_pt_path") or row.get("preprocessed_video")
    result.update(
        {
            "problem_id": row.get("sample_id", row.get("source_id", row.get("manifest_index"))),
            "problem": question,
            "data_type": "video",
            "problem_type": row.get("question_type", ""),
            "preprocessed_video": video_pt_path,
            "target_segment": _target_segment(row),
            "questioner_answer": row.get("answer", ""),
        }
    )
    return result


def _error_result(row: dict[str, Any], error: Exception | str) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "problem": row.get("question", ""),
            "questioner_answer": row.get("answer", ""),
            "answer": "",
            "score": -1,
            "majority_count": 0,
            "valid_answer_count": 0,
            "candidate_answers": [],
            "candidate_segments": [],
            "error": str(error),
        }
    )
    return result


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in _load_rows(args.input_file) if row.get("score") == 0]
    if not rows:
        _write_rows(args.output_file, [])
        return []

    prompt_template = _load_prompt_template(args.prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    seed = args.seed if args.seed is not None else int(args.suffix) if args.suffix.isdigit() else 0
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=seed,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"video": 1},
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_samples,
    )

    results: list[dict[str, Any] | None] = [None] * len(rows)
    for batch_start in range(0, len(rows), args.batch_size):
        batch_indices = range(batch_start, min(batch_start + args.batch_size, len(rows)))
        valid_indices: list[int] = []
        video_inputs: list[dict[str, Any]] = []
        for index in batch_indices:
            row = rows[index]
            try:
                video_inputs.append(
                    _build_video_input(
                        row,
                        processor=processor,
                        tokenizer=tokenizer,
                        prompt_template=prompt_template,
                    )
                )
                valid_indices.append(index)
            except Exception as exc:
                results[index] = _error_result(row, exc)

        if not video_inputs:
            continue
        responses = model.generate(video_inputs, sampling_params=sampling_params, use_tqdm=True)
        for index, response in zip(valid_indices, responses):
            row = rows[index]
            response_texts = [output.text for output in response.outputs]
            scored = score_solver_responses(response_texts)
            result = _common_result_fields(row)
            result.update(scored)
            result["num_candidates"] = len(response_texts)
            if args.save_raw_responses:
                result["solver_responses"] = response_texts
            results[index] = result

    finalized = [
        result if result is not None else _error_result(rows[index], "No evaluation result was produced.")
        for index, result in enumerate(results)
    ]
    _write_rows(args.output_file, finalized)
    return finalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated video questions with the full video and construct Solver pseudo-labels."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--suffix", default="0")
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--prompt_template", type=Path, default=Path("examples/format_prompt/solver.jinja"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_raw_responses", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num_samples and --batch_size must be positive.")
    storage_path = Path(os.getenv("STORAGE_PATH", "/apdcephfs_sh2/share_300000800/user/chengchuang"))
    generated_dir = storage_path / "generated_question"
    if args.input_file is None:
        args.input_file = generated_dir / f"{args.save_name}_{args.suffix}.json"
    if args.output_file is None:
        args.output_file = generated_dir / f"{args.save_name}_{args.suffix}_results.json"
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    evaluated = evaluate(parsed_args)
    retained = sum(0.3 <= float(row.get("score", -1)) <= 0.8 for row in evaluated)
    print(
        json.dumps(
            {"total": len(evaluated), "in_score_band": retained, "output": str(parsed_args.output_file)},
            ensure_ascii=False,
        )
    )
