"""Reward helpers for scaffolding (optional custom RM hook)."""

from __future__ import annotations

from slime.rollout.rm_hub.math_dapo_utils import compute_score
from slime.utils.types import Sample


def scalar_correctness_reward(response: str, label: str) -> float:
    """1.0 if strict boxed answer matches ground truth integer, else 0.0."""
    gt = str(label).strip()
    out = compute_score(response, gt, strict_box_verify=True)
    return 1.0 if out.get("acc") else 0.0


async def reward_gs_sample(args, sample: Sample, **kwargs) -> float:
    """Single-sample RM (unused when rewards are set in rollout)."""
    return scalar_correctness_reward(sample.response, sample.label or "")
