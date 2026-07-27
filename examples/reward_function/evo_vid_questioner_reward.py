"""EvoVid-style Questioner reward.

Configure EasyVideoR1 with:
``worker.reward.reward_function=examples/reward_function/evo_vid_questioner_reward.py:compute_score``.

It preserves R-Zero's existing file-based RPC pattern:

    reward worker writes task JSON -> GET /hello?name=<task-file>
    -> Solver server writes <task-file>_results.json -> reward worker reads it.

The video-aware Solver server must accept the task schema below.  In particular,
it must load ``video_pt_path`` itself, apply ``frames[permutation]`` when
``frame_order == "shuffle"``, and write one result per task containing at least
``score`` (the Solver's majority-agreement confidence).  The reward worker does
not modify or duplicate the original .pt artifact.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence


REWARD_NAME = "evovid_questioner_temporal"
REWARD_TYPE = "batch"

_QUESTIONER_RESPONSE_PATTERN = re.compile(
    r"^\s*<type>\s*(multiple choice|numerical|regression)\s*</type>\s*"
    r"<question>\s*(.*?)\s*</question>\s*"
    r"<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _empty_score(overall: float, *, format_score: float, solver_ok: float = 0.0) -> dict[str, float]:
    """Return a metric-complete score row for invalid or failed samples."""
    return {
        "overall": float(overall),
        "format": float(format_score),
        "r_base": 0.0,
        "r_difficulty": 0.0,
        "r_diversity": 0.0,
        "solver_conf_original": 0.0,
        "solver_conf_shuffle": 0.0,
        "r_temporal": 0.0,
        "solver_ok": float(solver_ok),
    }


def _extract_question(response: Any) -> str | None:
    if not isinstance(response, str):
        return None
    match = _QUESTIONER_RESPONSE_PATTERN.fullmatch(response)
    if match is None:
        return None
    question, answer = match.group(2).strip(), match.group(3).strip()
    return question if question and answer else None


def _extract_preprocessed_video_path(reward_input: dict[str, Any]) -> str:
    """Resolve the preprocessed-video path carried through DataProto.video_ref."""
    video_ref = reward_input.get("video_ref")
    if not isinstance(video_ref, dict):
        raise ValueError("Reward input does not contain video_ref.")

    # Kept for compatibility with the older EasyVideoR1 multi-modal contract.
    legacy_path = video_ref.get("preprocessed_video_path")
    if isinstance(legacy_path, str) and legacy_path:
        return legacy_path

    video = video_ref.get("video")
    if not isinstance(video, dict):
        raise ValueError("video_ref['video'] must be the preprocessed-video contract.")
    if video.get("source_type") != "preprocessed":
        raise ValueError(
            "EvoVid Questioner reward requires a preprocessed .pt artifact; "
            f"got source_type={video.get('source_type')!r}."
        )
    paths = video.get("paths")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0]:
        raise ValueError("EvoVid Questioner reward currently requires exactly one video .pt path per sample.")
    return paths[0]


def _parse_ports(ports: Sequence[int] | str | None) -> list[int]:
    if ports is None:
        ports = os.getenv("EVOVID_SOLVER_PORTS", "5000,5001,5002,5003")
    if isinstance(ports, str):
        ports = [part.strip() for part in ports.split(",") if part.strip()]
    parsed = [int(port) for port in ports]
    if not parsed:
        raise ValueError("solver_ports must contain at least one port.")
    return parsed


def _resolve_task_dir(task_dir: str | None) -> Path:
    # STORAGE_PATH is the shared location used by the original R-Zero caller.
    raw_path = task_dir or os.getenv("EVOVID_REWARD_TASK_DIR")
    if not raw_path:
        raw_path = os.path.join(os.getenv("STORAGE_PATH", "."), "temp_results")
    resolved = Path(raw_path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_reward_audit(
    reward_inputs: list[dict[str, Any]],
    scores: list[dict[str, float]],
    *,
    solver_host: str,
    solver_ports: Sequence[int] | str | None,
    error: str | None = None,
) -> None:
    """Persist one JSON document per reward call when smoke auditing is enabled."""
    raw_dir = os.getenv("EVOVID_REWARD_AUDIT_DIR")
    if not raw_dir:
        return
    try:
        audit_dir = Path(raw_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for index, (reward_input, score) in enumerate(zip(reward_inputs, scores)):
            response = reward_input.get("response")
            question = _extract_question(response)
            video_pt_path = None
            video_error = None
            try:
                video_pt_path = _extract_preprocessed_video_path(reward_input)
            except ValueError as exc:
                video_error = str(exc)

            if question is None:
                status = "invalid_question_format"
            elif video_error:
                status = "invalid_video_reference"
            elif score.get("solver_ok") == 1.0:
                status = "scored"
            elif error:
                status = "solver_error"
            else:
                status = "not_scored"
            rows.append(
                {
                    "index": index,
                    "problem_id": reward_input.get("problem_id"),
                    "response": response,
                    "extracted_question": question,
                    "video_pt_path": video_pt_path,
                    "video_error": video_error,
                    "status": status,
                    "score": score,
                }
            )
        payload = {
            "schema": "evovid_questioner_reward_audit_v1",
            "created_at_ns": time.time_ns(),
            "pid": os.getpid(),
            "solver_host": solver_host,
            "solver_ports": solver_ports,
            "error": error,
            "rows": rows,
        }
        path = audit_dir / f"reward_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        # Diagnostics must never change reward behavior.
        return

def _rzero_diversity_penalties(questions: list[str], distance_threshold: float) -> list[float]:
    """Use R-Zero's BLEU-distance agglomerative-clustering penalty.

    The original caller depends on numpy, nltk, and scikit-learn.  If a minimal
    reward environment lacks them, fall back to an exact-normalized-question
    cluster penalty rather than failing a whole training step.
    """
    if not questions:
        return []
    try:
        import numpy as np
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from sklearn.cluster import AgglomerativeClustering

        count = len(questions)
        distances = np.zeros((count, count))
        smoother = SmoothingFunction().method1
        for i in range(count):
            for j in range(i, count):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = sentence_bleu(
                        [questions[j].split()],
                        questions[i].split(),
                        smoothing_function=smoother,
                    )
                distances[i, j] = distances[j, i] = 1.0 - similarity

        try:
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(distances)
        except TypeError:  # scikit-learn before the ``metric`` rename
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                affinity="precomputed",
                linkage="average",
            ).fit_predict(distances)

        cluster_sizes = Counter(labels)
        return [float(cluster_sizes[label] / count) for label in labels]
    except ImportError:
        normalized = [" ".join(question.lower().split()) for question in questions]
        duplicate_counts = Counter(normalized)
        total = len(normalized)
        return [float(duplicate_counts[question] / total) for question in normalized]


def _task_file_path(task_dir: Path) -> Path:
    token = f"{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}"
    return task_dir / f"evovid_questioner_{token}.json"


def _write_task_file(task_dir: Path, tasks: list[dict[str, Any]]) -> Path:
    path = _task_file_path(task_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(tasks, handle, ensure_ascii=False)
    return path


def _read_result_file(path: Path, expected_tasks: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list) or len(results) != len(expected_tasks):
        raise RuntimeError(
            f"Solver result length mismatch for {path}: expected {len(expected_tasks)}, "
            f"got {len(results) if isinstance(results, list) else type(results).__name__}."
        )

    scores: dict[tuple[str, str], float] = {}
    for task, result in zip(expected_tasks, results):
        if not isinstance(result, dict):
            raise RuntimeError(f"Solver result for task {task['id']} is not a JSON object.")
        result_id = result.get("id", task["id"])
        result_order = result.get("frame_order", task["frame_order"])
        if result_id != task["id"] or result_order != task["frame_order"]:
            raise RuntimeError("Solver result id/frame_order does not match its submitted task.")
        try:
            score = float(result["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Solver result for task {task['id']} has no valid score.") from exc
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Solver score for task {task['id']} is outside [0, 1]: {score}.")
        scores[(task["id"], task["frame_order"])] = score
    return scores


def _call_solver_server(
    *,
    host: str,
    port: int,
    task_dir: Path,
    tasks: list[dict[str, Any]],
    request_timeout_s: float,
    cleanup_result_files: bool,
) -> dict[tuple[str, str], float]:
    """Perform one unchanged R-Zero-style file-path RPC to a Solver service."""
    import requests

    task_file = _write_task_file(task_dir, tasks)
    result_file = Path(str(task_file).replace(".json", "_results.json"))
    try:
        response = requests.get(
            f"http://{host}:{port}/hello",
            params={"name": str(task_file)},
            timeout=request_timeout_s,
        )
        response.raise_for_status()
        if not result_file.exists():
            raise RuntimeError(f"Solver {host}:{port} returned without creating {result_file}.")
        return _read_result_file(result_file, tasks)
    finally:
        # The legacy server deletes task_file itself.  The cleanup is therefore
        # deliberately tolerant of either old or new server behavior.
        if cleanup_result_files:
            for path in (task_file, result_file):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _dispatch_tasks(
    tasks: list[dict[str, Any]],
    *,
    solver_host: str,
    solver_ports: list[int],
    task_dir: Path,
    request_timeout_s: float,
    cleanup_result_files: bool,
) -> dict[tuple[str, str], float]:
    """Distribute tasks across the same 5000--5003-style Solver workers as R-Zero."""
    chunks = [tasks[index::len(solver_ports)] for index in range(len(solver_ports))]
    active = [(port, chunk) for port, chunk in zip(solver_ports, chunks) if chunk]
    scores: dict[tuple[str, str], float] = {}
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = {
            executor.submit(
                _call_solver_server,
                host=solver_host,
                port=port,
                task_dir=task_dir,
                tasks=chunk,
                request_timeout_s=request_timeout_s,
                cleanup_result_files=cleanup_result_files,
            ): port
            for port, chunk in active
        }
        for future in as_completed(futures):
            port = futures[future]
            try:
                scores.update(future.result())
            except Exception as exc:
                raise RuntimeError(f"Solver request to {solver_host}:{port} failed: {exc}") from exc
    return scores


def _build_tasks(prepared: Iterable[dict[str, Any]], solver_samples: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for item in prepared:
        common = {
            "schema": "evovid_questioner_v1",
            "id": item["id"],
            "reward_index": item["index"],
            "question": item["question"],
            "video_pt_path": item["video_pt_path"],
            "num_candidates": int(solver_samples),
        }
        tasks.append({**common, "frame_order": "original", "shuffle_seed": None})
        tasks.append(
            {
                **common,
                "frame_order": "shuffle",
                "shuffle_seed": item["shuffle_seed"],
            }
        )
    return tasks


def compute_score(
    reward_inputs: list[dict[str, Any]],
    *,
    solver_host: str | None = None,
    solver_ports: Sequence[int] | str | None = None,
    task_dir: str | None = None,
    solver_samples: int = 10,
    lambda_temporal: float = 0.1,
    diversity_distance_threshold: float = 0.5,
    diversity_weight: float = 1.0,
    clamp_base: bool = True,
    invalid_reward: float = 0.0,
    failure_reward: float = -1.0,
    request_timeout_s: float = 3600.0,
    cleanup_result_files: bool = True,
    **_: Any,
) -> list[dict[str, float]]:
    """Compute the R-Zero base reward plus EvoVid temporal sensitivity.

    ``solver_host`` defaults to ``EVOVID_SOLVER_HOST`` or ``127.0.0.1`` and
    ``solver_ports`` defaults to ``EVOVID_SOLVER_PORTS`` or ``5000,5001,5002,5003``.
    If reward workers and Solver services are on separate machines, ``task_dir``
    must be a shared filesystem path and ``solver_host`` must be the Solver host's
    real address (not ``0.0.0.0``).
    """
    if solver_samples < 1:
        raise ValueError("solver_samples must be positive.")
    if lambda_temporal < 0:
        raise ValueError("lambda_temporal must be non-negative.")
    if diversity_weight < 0:
        raise ValueError("diversity_weight must be non-negative.")

    scores = [_empty_score(invalid_reward, format_score=0.0) for _ in reward_inputs]
    resolved_solver_host = solver_host or os.getenv("EVOVID_SOLVER_HOST", "127.0.0.1")

    def finish(error: str | None = None) -> list[dict[str, float]]:
        _write_reward_audit(
            reward_inputs,
            scores,
            solver_host=resolved_solver_host,
            solver_ports=solver_ports or os.getenv("EVOVID_SOLVER_PORTS"),
            error=error,
        )
        return scores

    prepared: list[dict[str, Any]] = []
    for index, reward_input in enumerate(reward_inputs):
        question = _extract_question(reward_input.get("response"))
        if question is None:
            continue
        try:
            video_pt_path = _extract_preprocessed_video_path(reward_input)
        except ValueError:
            # A well-formed Questioner output without its corresponding video
            # must not receive a temporal reward.
            scores[index] = _empty_score(failure_reward, format_score=1.0)
            continue
        prepared.append(
            {
                "id": uuid.uuid4().hex,
                "index": index,
                "question": question,
                "video_pt_path": video_pt_path,
                "problem_id": reward_input.get("problem_id"),
                "shuffle_seed": secrets.randbits(63),
            }
        )

    if not prepared:
        return finish()

    try:
        solver_scores = _dispatch_tasks(
            _build_tasks(prepared, solver_samples),
            solver_host=resolved_solver_host,
            solver_ports=_parse_ports(solver_ports),
            task_dir=_resolve_task_dir(task_dir),
            request_timeout_s=float(request_timeout_s),
            cleanup_result_files=bool(cleanup_result_files),
        )
    except Exception as exc:
        # Do not accidentally reward a question when the frozen Solver was not
        # actually evaluated.  Preserve the rest of the batch's alignment.
        for item in prepared:
            scores[item["index"]] = _empty_score(failure_reward, format_score=1.0)
        return finish(str(exc))

    # Compute diversity within each prompt's G candidates, not across prompts.
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for item in prepared:
        key = item["problem_id"] if item["problem_id"] is not None else "__missing_problem_id__"
        grouped.setdefault(key, []).append(item)

    diversity_by_index: dict[int, float] = {}
    for items in grouped.values():
        penalties = _rzero_diversity_penalties(
            [item["question"] for item in items], float(diversity_distance_threshold)
        )
        diversity_by_index.update({item["index"]: penalty for item, penalty in zip(items, penalties)})

    for item in prepared:
        diversity_penalty = diversity_by_index[item["index"]]
        index = item["index"]
        try:
            original_confidence = solver_scores[(item["id"], "original")]
            shuffled_confidence = solver_scores[(item["id"], "shuffle")]
        except KeyError:
            scores[index] = _empty_score(failure_reward, format_score=1.0)
            continue

        difficulty = min(original_confidence, 1.0 - original_confidence)
        base = difficulty - diversity_weight * diversity_penalty
        if clamp_base:
            # EvoVid Eq. (4).  Set clamp_base=False for the literal legacy
            # caller_penalty.py behavior, which did not clamp this term.
            base = max(0.0, base)
        temporal = max(0.0, original_confidence - shuffled_confidence)
        scores[index] = {
            "overall": float(base + lambda_temporal * temporal),
            "format": 1.0,
            "r_base": float(base),
            "r_difficulty": float(difficulty),
            "r_diversity": float(diversity_penalty),
            "solver_conf_original": float(original_confidence),
            "solver_conf_shuffle": float(shuffled_confidence),
            "r_temporal": float(temporal),
            "solver_ok": 1.0,
        }
    return finish()

