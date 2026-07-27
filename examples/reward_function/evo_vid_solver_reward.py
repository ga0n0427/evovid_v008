"""EvoVid Solver reward with answer accuracy and temporal IoU.

The Questioner window used to create each question is carried in the training
row as ``target_segment=[start_seconds, end_seconds]``.  The Solver must return
its answer in ``\\boxed{}`` and the relevant interval as
``<segment>Xs-Ys</segment>``.
"""

from __future__ import annotations

import math
import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from solver_answer_normalization import normalize_mc_option_token


REWARD_NAME = "evovid_solver_temporal_iou"
REWARD_TYPE = "batch"

_SEGMENT_PATTERN = re.compile(
    r"<segment>\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*s?\s*[-\u2013\u2014]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*s?\s*</segment>",
    re.IGNORECASE,
)
_SEGMENT_TAG_PATTERN = re.compile(r"</?segment\b", re.IGNORECASE)


def _parse_interval(value: Any) -> tuple[float, float] | None:
    """Normalize a two-value interval and reject invalid timestamps."""
    if isinstance(value, (str, bytes)) or value is None:
        return None
    try:
        if len(value) != 2:
            return None
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def extract_predicted_segment(response: Any) -> tuple[float, float] | None:
    """Extract the last valid ``<segment>`` interval from a Solver response."""
    if not isinstance(response, str):
        return None
    matches = list(_SEGMENT_PATTERN.finditer(response))
    if not matches:
        return None
    return _parse_interval(matches[-1].groups())


def temporal_iou(predicted: Any, target: Any) -> float:
    """Return standard one-dimensional intersection-over-union."""
    pred_interval = _parse_interval(predicted)
    target_interval = _parse_interval(target)
    if pred_interval is None or target_interval is None:
        return 0.0

    pred_start, pred_end = pred_interval
    target_start, target_end = target_interval
    intersection = max(0.0, min(pred_end, target_end) - max(pred_start, target_start))
    union = (pred_end - pred_start) + (target_end - target_start) - intersection
    return intersection / union if union > 0 else 0.0


def _extract_boxed_answer(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    answer = extract_boxed_content(value)
    if answer is None:
        return ""
    normalized = str(answer).strip()
    if not normalized or normalized.casefold() == "none":
        return ""
    # A time span accidentally placed inside \boxed{} is not an answer.
    if _SEGMENT_TAG_PATTERN.search(normalized):
        return ""
    return normalized


def _extract_ground_truth(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    boxed = extract_boxed_content(text)
    boxed_text = "" if boxed is None else str(boxed).strip()
    # mathruler uses the literal string "None" as its no-box sentinel.
    # Solver pseudo labels are normally plain values such as "B", so keep the
    # original ground truth when no actual boxed value was found.
    if not boxed_text or boxed_text.casefold() == "none":
        return text
    return boxed_text


def _normalize_mc_answer(value: str) -> str:
    return normalize_mc_option_token(value) or ""


def answer_accuracy(response: Any, ground_truth: Any, problem_type: Any = None) -> float:
    """Score the boxed answer; a missing box always receives zero accuracy."""
    predicted = _extract_boxed_answer(response)
    target = _extract_ground_truth(ground_truth)
    if not predicted or not target:
        return 0.0

    normalized_type = str(problem_type or "").strip().casefold()
    if normalized_type == "multiple choice":
        predicted_mc = _normalize_mc_answer(predicted)
        target_mc = _normalize_mc_answer(target)
        return float(bool(predicted_mc and target_mc and predicted_mc == target_mc))

    if predicted.casefold() == target.casefold():
        return 1.0
    try:
        return float(bool(grade_answer(predicted, target)))
    except Exception:
        return 0.0


def format_reward(response: Any) -> float:
    """Require both a boxed answer and one valid temporal segment."""
    boxed_answer = _extract_boxed_answer(response)
    predicted_segment = extract_predicted_segment(response)
    return float(bool(boxed_answer and predicted_segment is not None))


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    temporal_weight: float = 0.3,
    **_: Any,
) -> list[dict[str, float]]:
    """Compute ``(1-w)*Acc + w*Fmt + lambda_s*(Acc*IoU)`` per sample."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError(f"format_weight must be in [0, 1], got {format_weight}.")
    if temporal_weight < 0.0:
        raise ValueError(f"temporal_weight must be non-negative, got {temporal_weight}.")

    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        accuracy = answer_accuracy(
            response,
            reward_input.get("ground_truth", ""),
            reward_input.get("problem_type"),
        )
        format_score = format_reward(response)
        predicted_segment = extract_predicted_segment(response)
        iou = temporal_iou(predicted_segment, reward_input.get("target_segment"))
        temporal = accuracy * iou
        overall = (1.0 - format_weight) * accuracy + format_weight * format_score + temporal_weight * temporal
        scores.append(
            {
                "overall": float(overall),
                "accuracy": float(accuracy),
                "format": float(format_score),
                "iou": float(iou),
                "temporal": float(temporal),
            }
        )
    return scores
