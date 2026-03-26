"""Split-group GRPO reward normalization: solvers normalized among solvers, judges among judges."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

import torch

from slime.utils.types import Sample


def _flatten_samples(samples: list[Any]) -> list[Sample]:
    flat: list[Sample] = samples  # type: ignore[assignment]
    while flat and isinstance(flat[0], list):
        flat = list(itertools.chain.from_iterable(flat))
    return flat


def dual_group_grpo_reward_postprocess(args: Any, samples: list[Sample] | list[list[Sample]]):
    """
    For each prompt group (same ``group_index``), mean-center rewards separately for:
    - samples with ``metadata["round_type"] == "solver"``
    - samples with ``metadata["round_type"] == "judge"``

    Returns ``(raw_rewards, processed_rewards)`` in the same order as the flattened sample list.
    """
    flat = _flatten_samples(samples)
    raw_rewards = [float(s.get_reward_value(args)) for s in flat]

    by_group: dict[int, list[Sample]] = defaultdict(list)
    for s in flat:
        by_group[int(s.group_index)].append(s)

    norm_by_id: dict[int, float] = {}

    for _gid, grp in by_group.items():
        solvers = [s for s in grp if (s.metadata or {}).get("round_type") == "solver"]
        judges = [s for s in grp if (s.metadata or {}).get("round_type") == "judge"]
        if len(solvers) + len(judges) != len(grp):
            raise ValueError(
                f"scaffolding group {gid}: expected every sample to be round_type solver|judge, "
                f"got n={len(grp)} solvers={len(solvers)} judges={len(judges)}"
            )
        solvers.sort(key=lambda s: int((s.metadata or {}).get("attempt_idx", 0)))
        judges.sort(key=lambda s: int((s.metadata or {}).get("judge_idx", 0)))

        def _normalize_group(sub: list[Sample]) -> None:
            if not sub:
                return
            vals = torch.tensor([float(s.get_reward_value(args)) for s in sub], dtype=torch.float32)
            centered = vals - vals.mean()
            if getattr(args, "grpo_std_normalization", False):
                centered = centered / (centered.std(unbiased=False) + 1e-6)
            for s, v in zip(sub, centered.tolist(), strict=True):
                norm_by_id[id(s)] = float(v)

        _normalize_group(solvers)
        _normalize_group(judges)

    processed = [norm_by_id.get(id(s), 0.0) for s in flat]
    return raw_rewards, processed
