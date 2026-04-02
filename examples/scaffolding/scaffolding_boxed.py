"""Shared integer-answer extraction for rollout stopping and 0/1 rewards."""

from __future__ import annotations

import re
from fractions import Fraction

BOXED_NUMBER_RE = re.compile(r"\\boxed\s*\{\s*(-?[\d,]+(?:\.\d+)?(?:/\d+)?)\s*\}")
JUDGMENT_RE = re.compile(r"\*{0,2}Judgment:\*{0,2}\s*\[?\s*(-?[\d,]+(?:\.\d+)?(?:/\d+)?)\s*\]?", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"final\s+answer\s+is\s*:?\s*(-?[\d,]+)", re.IGNORECASE)
NOTEBOOK_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([0-9,]+)\s*\}")
NOTEBOOK_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s+is\s*([0-9,]+)", re.IGNORECASE)
NAMED_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\n\r]|[.!?]\s+)\s*(?:hence|therefore|thus|so|we\s+(?:get|find)|it\s+follows\s+that)?\s*"
    r"(?:the\s+minimum\s+)?(?:n(?:_min)?|m(?:_min)?|answer)\s*(?:=|is)\s*"
    r"(-?[\d,]+(?:\.\d+)?(?:/\d+)?)\s*(?:[.)]|$)",
    re.IGNORECASE,
)
STREAM_FINAL_ANSWER_RE = re.compile(
    r"final\s+answer\s+is\s*:?\s*(-?[\d,]+)(?=(?:\s*[.)]|<\|end\|>|[\n\r]))",
    re.IGNORECASE,
)
STREAM_NAMED_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\n\r]|[.!?]\s+)\s*(?:hence|therefore|thus|so|we\s+(?:get|find)|it\s+follows\s+that)?\s*"
    r"(?:the\s+minimum\s+)?(?:n(?:_min)?|m(?:_min)?|answer)\s*(?:=|is)\s*"
    r"(-?[\d,]+(?:\.\d+)?(?:/\d+)?)(?=(?:\s*[.)]|<\|end\|>|[\n\r]))",
    re.IGNORECASE,
)
FINAL_CHANNEL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", re.DOTALL)


def _normalize_numeric_text(s: str) -> str:
    return str(s).strip().replace(",", "")


def normalize_int_answer(s: str) -> str:
    """Normalize ground truth / extracted answers that are mathematically integers."""
    value = Fraction(_normalize_numeric_text(s))
    if value.denominator != 1:
        raise ValueError(f"answer is not an integer: {s!r}")
    return str(value.numerator)


def _search_texts(response: str) -> list[str]:
    final_match = FINAL_CHANNEL_RE.search(response)
    texts = []
    if final_match:
        texts.append(final_match.group(1))
    texts.append(response)
    return texts


def extract_last_boxed_integer(response: str) -> str | None:
    """
    Return the last integer answer emitted by the model.

    Prefer the Harmony final channel when present. Accept the notebook's boxed
    and judge formats, and tolerate commas / integer-like decimals or fractions.
    """
    for text in _search_texts(response):
        for regex in (BOXED_NUMBER_RE, JUDGMENT_RE, FINAL_ANSWER_RE, NAMED_ASSIGNMENT_RE):
            matches = regex.findall(text)
            for match in reversed(matches):
                try:
                    return normalize_int_answer(match)
                except (ValueError, ZeroDivisionError):
                    continue
    return None


def boxed_answer_valid_for_stop(response: str) -> bool:
    """True if the response already contains a parseable final integer answer."""
    return extract_last_boxed_integer(response) is not None


def extract_last_notebook_solver_integer(response: str) -> str | None:
    """Notebook-parity solver extraction: boxed or 'final answer is', bounded to [0, 99999]."""
    for text in _search_texts(response):
        for regex in (NOTEBOOK_BOXED_RE, NOTEBOOK_FINAL_ANSWER_RE):
            matches = regex.findall(text)
            for match in reversed(matches):
                try:
                    value = int(match.replace(",", ""))
                except ValueError:
                    continue
                if 0 <= value <= 99999:
                    return str(value)
    return None


def notebook_solver_answer_valid_for_stop(response: str) -> bool:
    """Notebook-parity completed-turn solver answer check."""
    return extract_last_notebook_solver_integer(response) is not None


def stream_answer_valid_for_stop(response: str) -> bool:
    """
    Conservative online stop detector for partially streamed text.

    Unlike offline extraction, do not treat a bare end-of-buffer integer as final.
    This avoids stopping on prefixes like "answer = 238" when the model is still
    streaming "310156".
    """
    for text in _search_texts(response):
        if BOXED_NUMBER_RE.search(text) or JUDGMENT_RE.search(text):
            return True
        if STREAM_FINAL_ANSWER_RE.search(text) or STREAM_NAMED_ASSIGNMENT_RE.search(text):
            return True
    return False


def stream_notebook_solver_answer_valid_for_stop(response: str, latest_chunk: str) -> bool:
    """
    Notebook-parity online stop detector for solver streaming.

    The notebook only attempts an early answer scan once a streamed chunk contains
    a closing brace, which in practice means early stream stopping is effectively
    tied to completed boxed answers.
    """
    return "}" in latest_chunk and extract_last_notebook_solver_integer(response) is not None
