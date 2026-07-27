"""Shared normalization for Solver multiple-choice answer tokens."""

from __future__ import annotations

import re
from typing import Any


_MC_WRAPPER_PATTERN = re.compile(
    r"^\\(?:text|mathrm|mathbf)\s*\{\s*([A-D])\s*\}$",
    re.IGNORECASE,
)


def normalize_mc_option_token(value: Any) -> str | None:
    """Normalize a paper-format MC answer: A-D or Yes/No."""
    if value is None:
        return None

    original = str(value).strip()
    if not original:
        return None

    candidate = original.rstrip(".:").strip()
    wrapped = _MC_WRAPPER_PATTERN.fullmatch(candidate)
    if wrapped:
        return wrapped.group(1).upper()

    candidate = candidate.strip("()[] ").strip()
    if re.fullmatch(r"[A-D]", candidate, re.IGNORECASE):
        return candidate.upper()
    if candidate.casefold() in {"yes", "no"}:
        return candidate.upper()
    return None
