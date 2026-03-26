"""Reward helpers for scaffolding (solver + judge 0/1 tasks)."""

from __future__ import annotations

from slime.utils.types import Sample

from examples.scaffolding.scaffolding_boxed import extract_last_boxed_integer, normalize_int_answer


def scalar_correctness_reward(response: str, label: str) -> float:
    """1.0 if the last \\boxed{} integer matches ground truth (no fixed answer range), else 0.0."""
    pred = extract_last_boxed_integer(response)
    if pred is None:
        return 0.0
    try:
        int(pred)
        if normalize_int_answer(pred) != normalize_int_answer(label):
            return 0.0
    except (ValueError, OverflowError):
        return 0.0
    return 1.0


def judge_selection_reward(response: str, ground_truth: str, proposed_answers: set[str]) -> float:
    """
    1.0 iff the judge's last \\boxed{} integer equals ground truth *and* appears among solver proposals.

    ``proposed_answers`` should be normalized string forms (e.g. ``normalize_int_answer`` of each solver extract).
    """
    pred = extract_last_boxed_integer(response)
    if pred is None:
        return 0.0
    try:
        pred_norm = normalize_int_answer(pred)
        gt_norm = normalize_int_answer(ground_truth)
    except (ValueError, OverflowError):
        return 0.0
    if pred_norm != gt_norm:
        return 0.0
    if pred_norm not in proposed_answers:
        return 0.0
    return 1.0


async def reward_gs_sample(args, sample: Sample, **kwargs) -> float:
    """Single-sample RM (unused when rewards are set in rollout)."""
    return scalar_correctness_reward(sample.response or "", sample.label or "")
