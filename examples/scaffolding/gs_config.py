"""
Notebook-aligned defaults from examples/scaffolding/gen-select-nb.ipynb (CFG class).
Override via environment variables with prefix SLIME_SCAFFOLDING_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v != "" else default


@dataclass(frozen=True)
class ScaffoldingCFG:
    """Timing and search defaults aligned with the Kaggle notebook."""

    high_problem_timeout: float = 900.0
    base_problem_timeout: float = 300.0
    notebook_limit: float = 17400.0
    server_timeout: int = 180
    session_timeout: float = 960.0
    jupyter_timeout: float = 6.0
    sandbox_timeout: float = 3.0
    top_logprobs: int = 5
    batch_size: int = 256
    early_stop: int = 4
    attempts: int = 8
    workers: int = 20
    turns: int = 128
    gen_select_threshold: int = 4  # notebook parity; unused in slime rollout
    # Stochastic judges for diverse GRPO groups (override via SLIME_SCAFFOLDING_JUDGE_TEMPERATURE).
    judge_temperature: float = 1.0
    judge_max_tokens: int = 32768
    buffer_tokens: int = 512
    search_tokens: int = 32
    min_p: float = 0.02
    temperature: float = 1.0
    # Dynamic per-problem budget: problems_remaining in notebook starts at 50
    problems_remaining_default: int = 50

    @staticmethod
    def from_env() -> "ScaffoldingCFG":
        p = "SLIME_SCAFFOLDING_"
        return ScaffoldingCFG(
            high_problem_timeout=_env_float(f"{p}HIGH_PROBLEM_TIMEOUT", 900.0),
            base_problem_timeout=_env_float(f"{p}BASE_PROBLEM_TIMEOUT", 300.0),
            notebook_limit=_env_float(f"{p}NOTEBOOK_LIMIT", 17400.0),
            server_timeout=_env_int(f"{p}SERVER_TIMEOUT", 180),
            session_timeout=_env_float(f"{p}SESSION_TIMEOUT", 960.0),
            jupyter_timeout=_env_float(f"{p}JUPYTER_TIMEOUT", 6.0),
            sandbox_timeout=_env_float(f"{p}SANDBOX_TIMEOUT", 3.0),
            top_logprobs=_env_int(f"{p}TOP_LOGPROBS", 5),
            batch_size=_env_int(f"{p}BATCH_SIZE", 256),
            early_stop=_env_int(f"{p}EARLY_STOP", 4),
            attempts=_env_int(f"{p}ATTEMPTS", 8),
            workers=_env_int(f"{p}WORKERS", 20),
            turns=_env_int(f"{p}TURNS", 128),
            gen_select_threshold=_env_int(f"{p}GEN_SELECT_THRESHOLD", 4),
            judge_temperature=_env_float(f"{p}JUDGE_TEMPERATURE", 1.0),
            judge_max_tokens=_env_int(f"{p}JUDGE_MAX_TOKENS", 32768),
            buffer_tokens=_env_int(f"{p}BUFFER_TOKENS", 512),
            search_tokens=_env_int(f"{p}SEARCH_TOKENS", 32),
            min_p=_env_float(f"{p}MIN_P", 0.02),
            temperature=_env_float(f"{p}TEMPERATURE", 1.0),
            problems_remaining_default=_env_int(f"{p}PROBLEMS_REMAINING", 50),
        )


SYSTEM_PROMPT = (
    "You are a world-class International Mathematical Olympiad (IMO) competitor. "
    "Solve the problem step by step. You may execute short Python code via the "
    "tool when helpful. When you have the final answer, put it in \\boxed{} as an integer."
)

# Kept in lockstep with examples/scaffolding/gen-select-nb.ipynb (cell defining GEN_SELECT_PROMPT).
GEN_SELECT_PROMPT = """\
You are judging {num_solutions} candidate solutions (numbered 0 through {max_idx}) to a math problem.

Your ONLY job is to pick the best answer FROM the solutions provided. You are a comparator, not a solver.

Problem:
{problem}

Solutions:
{solutions}

Instructions:

STRICT RULE — DO NOT RE-SOLVE: Do not attempt your own derivation, computation, or independent reasoning about the problem. Judges who re-derive answers get them wrong more often than the majority of solvers. If you catch yourself computing an answer or reasoning about the math independently, STOP and return to comparing the provided solutions.

STRICT RULE — ANSWER MUST COME FROM SOLUTIONS: Your final answer MUST be one of the answers that appears in the solution headers above. You may NOT output any answer that no solution proposed. There are exactly {num_solutions} solutions (numbered 0 through {max_idx}) — do not reference solutions outside this range.

Follow these steps:

1. Read each solution's final answer from the "(final answer: X)" label in its header. Group solutions by answer and count how many support each one. State the vote counts explicitly.

2. The answer with the most votes is the presumptive winner. In case of an exact tie, the answer from the lowest-indexed solution wins.

3. To override the majority, you must identify a specific, concrete error that you can QUOTE directly from every single majority solution's text — a wrong algebraic step, a wrong formula, a missed constraint, or a verifiable arithmetic mistake. The error must be visible in the solution text itself; your own alternative reasoning or re-derivation does NOT count as evidence. An error in an illustrative example does not invalidate the core derivation. Vague doubts do not count. If you cannot quote a concrete error from every majority solution, you MUST select the majority answer.

4. When a solution includes executed code that produced a result, trust the code output over manual reasoning, as long as the code correctly implements the problem constraints.

Your final answer must be a single integer that at least one solution proposed.

End with exactly:

\\boxed{{ANSWER}}

where ANSWER is the integer you select (e.g. \\boxed{{42}})."""
