"""Small, role-agnostic helpers for file-backed vLLM LoRA adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedLoraAdapter:
    """A validated local PEFT adapter that vLLM can load on demand."""

    path: str
    rank: int
    base_model_name_or_path: str | None


def resolve_lora_adapter(raw_path: str | None) -> ResolvedLoraAdapter | None:
    """Resolve either an adapter directory or an actor checkpoint containing it.

    ``raw_path`` is intentionally role-agnostic: Solver and Questioner callers
    each supply their own path.  An omitted path means normal base-model
    inference and therefore needs no special handling.
    """

    if raw_path is None or not str(raw_path).strip():
        return None

    root = Path(raw_path).expanduser()
    candidates = (root, root / "lora_adapter")
    adapter_dir: Path | None = None
    for candidate in candidates:
        config_path = candidate / "adapter_config.json"
        has_weights = any(
            (candidate / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if config_path.is_file() and has_weights:
            adapter_dir = candidate
            break

    if adapter_dir is None:
        raise FileNotFoundError(
            "LoRA path must be an adapter directory containing adapter_config.json "
            "and adapter weights, or an actor checkpoint containing lora_adapter/. "
            f"Got: {root}"
        )

    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"LoRA adapter {adapter_dir} has an invalid positive rank 'r': {rank!r}.")

    base_model = config.get("base_model_name_or_path")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError(
            f"LoRA adapter {adapter_dir} has a non-string base_model_name_or_path: {base_model!r}."
        )

    return ResolvedLoraAdapter(
        path=str(adapter_dir.resolve()),
        rank=rank,
        base_model_name_or_path=base_model,
    )
