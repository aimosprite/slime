"""Shared \\boxed{} integer extraction for rollout stopping and 0/1 rewards (consistent semantics)."""

from __future__ import annotations

import re

BOXED_INT_RE = re.compile(r"\\boxed\{(-?\d+)\}")


def extract_last_boxed_integer(response: str) -> str | None:
    """Return the string inside the last ``\\boxed{...}`` if it matches an integer token."""
    m = BOXED_INT_RE.findall(response)
    return m[-1] if m else None


def normalize_int_answer(s: str) -> str:
    """Normalize ground truth / extracted integers for comparison."""
    return str(int(float(str(s).strip())))


def boxed_answer_valid_for_stop(response: str) -> bool:
    """True if the last ``\\boxed{}`` contains a parseable integer (no fixed range; labels may be any int)."""
    last = extract_last_boxed_integer(response)
    if last is None:
        return False
    try:
        int(last)
        return True
    except (ValueError, OverflowError):
        return False
