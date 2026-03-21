"""
Custom rollout: parallel solve attempts + unconditional gen-select judge round.
Trains only on model-generated tokens (loss_mask); tool / observation tokens masked 0.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from examples.scaffolding.gs_config import GEN_SELECT_PROMPT, SYSTEM_PROMPT, ScaffoldingCFG
from examples.scaffolding.python_tool import PersistentPythonSession
from examples.scaffolding.reward_gpt_oss_scaffolding import scalar_correctness_reward
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.misc import load_function
from slime.rollout.sglang_rollout import GenerateState, abort
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
BOXED_INT_RE = re.compile(r"\\boxed\{(-?\d+)\}")
TOOL_OBSERVATION_TEMPLATE = "<|im_end|>\n<|im_start|>tool\n{content}\n<|im_end|>\n<|im_start|>assistant\n"
NOTEBOOK_START_TIME = time.time()


@dataclass
class AttemptResult:
    prompt_ids: list[int]
    response_token_ids: list[int]
    response_text: str
    loss_mask: list[int]
    rollout_log_probs: list[float]
    extracted_answer: str | None
    status: Sample.Status
    metadata: dict[str, Any] = field(default_factory=dict)


def _problem_budget_s(cfg: ScaffoldingCFG, problems_remaining: int, notebook_elapsed: float) -> float:
    """Notebook dynamic budget: max(base, min(time_left - (k-1)*base, high))."""
    time_left = cfg.notebook_limit - notebook_elapsed
    k = max(0, problems_remaining - 1)
    inner = time_left - k * cfg.base_problem_timeout
    return max(cfg.base_problem_timeout, min(inner, cfg.high_problem_timeout))


def _format_solutions_for_judge(attempts: list[AttemptResult]) -> str:
    parts: list[str] = []
    for i, att in enumerate(attempts):
        text = att.response_text[-4000:] if len(att.response_text) > 4000 else att.response_text
        if att.extracted_answer is not None:
            ans = str(att.extracted_answer)
        else:
            m = BOXED_INT_RE.findall(att.response_text)
            ans = m[-1] if m else "(no answer)"
        parts.append(f"--- Solution {i} (final answer: {ans}) ---\n{text}")
    return "\n\n".join(parts)


def _encode_prompt_ids(tokenizer: Any, problem_text: str) -> list[int]:
    return tokenizer.encode(problem_text, add_special_tokens=False)


async def _sglang_generate(
    args: Any,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    session_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    headers = session_headers
    return await post(url, payload, headers=headers)


async def run_one_attempt(
    args: Any,
    state: GenerateState,
    cfg: ScaffoldingCFG,
    problem_text: str,
    attempt_idx: int,
    deadline: float,
    sampling_params: dict[str, Any],
    session_id: str | None,
) -> AttemptResult:
    tokenizer = state.tokenizer
    prompt_ids = _encode_prompt_ids(tokenizer, problem_text)

    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    response_text = ""
    status = Sample.Status.PENDING
    session = PersistentPythonSession()

    headers = None
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and session_id:
        headers = {"X-SMG-Routing-Key": f"{session_id}-att{attempt_idx}"}

    turn = 0
    max_turns = 0 if getattr(args, "ci_test", False) else cfg.turns

    try:
        while turn < max_turns and time.time() < deadline:
            current_ids = prompt_ids + response_token_ids
            max_ctx = getattr(args, "rollout_max_context_len", None) or 65536
            if len(current_ids) >= max_ctx - cfg.buffer_tokens:
                status = Sample.Status.TRUNCATED
                break

            sp = sampling_params.copy()
            sp["max_new_tokens"] = min(
                sp.get("max_new_tokens", 4096),
                max_ctx - len(current_ids) - 1,
            )
            if sp["max_new_tokens"] <= 0:
                status = Sample.Status.TRUNCATED
                break

            out = await _sglang_generate(args, current_ids, sp, headers)

            if out["meta_info"]["finish_reason"]["type"] == "abort":
                status = Sample.Status.ABORTED
                break

            if "output_token_logprobs" in out["meta_info"]:
                new_ids = [item[1] for item in out["meta_info"]["output_token_logprobs"]]
                new_lps = [item[0] for item in out["meta_info"]["output_token_logprobs"]]
            else:
                new_ids = tokenizer.encode(out.get("text", ""), add_special_tokens=False)
                new_lps = [0.0] * len(new_ids)

            chunk_text = tokenizer.decode(new_ids, skip_special_tokens=False)
            response_text += chunk_text
            response_token_ids.extend(new_ids)
            loss_mask.extend([1] * len(new_ids))
            rollout_log_probs.extend(new_lps)

            m = BOXED_INT_RE.findall(response_text)
            if m:
                try:
                    v = int(m[-1])
                    if 0 <= v <= 99999:
                        status = Sample.Status.COMPLETED
                        break
                except ValueError:
                    pass

            match = CODE_BLOCK_RE.search(chunk_text)
            if match and time.time() < deadline:
                code = match.group(1).strip()
                if code:
                    obs = await session.execute(code, cfg.jupyter_timeout)
                    obs_str = TOOL_OBSERVATION_TEMPLATE.format(content=obs)
                    obs_ids = tokenizer.encode(obs_str, add_special_tokens=False)
                    response_token_ids.extend(obs_ids)
                    loss_mask.extend([0] * len(obs_ids))
                    rollout_log_probs.extend([0.0] * len(obs_ids))
                    response_text += obs_str

            finish = out["meta_info"]["finish_reason"]["type"]
            if finish == "length":
                status = Sample.Status.TRUNCATED
                break

            turn += 1
    finally:
        session.close()

    if status == Sample.Status.PENDING:
        status = Sample.Status.COMPLETED if response_token_ids else Sample.Status.TRUNCATED

    m = BOXED_INT_RE.findall(response_text)
    extracted = m[-1] if m else None

    return AttemptResult(
        prompt_ids=prompt_ids,
        response_token_ids=response_token_ids,
        response_text=response_text,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        extracted_answer=extracted,
        status=status,
        metadata={"attempt_idx": attempt_idx, "turns": turn},
    )


def _encode_judge_prompt_ids(tokenizer: Any, problem_text: str, attempt_results: list[AttemptResult]) -> list[int]:
    sols = _format_solutions_for_judge(attempt_results)
    n = len(attempt_results)
    user_prompt = GEN_SELECT_PROMPT.format(
        num_solutions=n,
        max_idx=n - 1,
        problem=problem_text,
        solutions=sols,
    )
    messages = [
        {"role": "system", "content": "You are a careful judge for math competitions."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.encode(
            f"<|im_start|>system\nYou are a careful judge for math competitions.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )


async def run_judge_round(
    args: Any,
    state: GenerateState,
    cfg: ScaffoldingCFG,
    problem_text: str,
    attempt_results: list[AttemptResult],
    deadline: float,
    sampling_params: dict[str, Any],
    session_id: str | None,
) -> AttemptResult:
    tokenizer = state.tokenizer
    prompt_ids = _encode_judge_prompt_ids(tokenizer, problem_text, attempt_results)

    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    response_text = ""
    status = Sample.Status.PENDING

    headers = None
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and session_id:
        headers = {"X-SMG-Routing-Key": f"{session_id}-judge"}

    sp = sampling_params.copy()
    sp["temperature"] = cfg.judge_temperature
    sp["max_new_tokens"] = min(cfg.judge_max_tokens, 8192)

    max_ctx = getattr(args, "rollout_max_context_len", None) or 65536
    while time.time() < deadline:
        current_ids = prompt_ids + response_token_ids
        sp2 = sp.copy()
        sp2["max_new_tokens"] = min(sp2["max_new_tokens"], max_ctx - len(current_ids) - 1)
        if sp2["max_new_tokens"] <= 0:
            status = Sample.Status.TRUNCATED
            break

        out = await _sglang_generate(args, current_ids, sp2, headers)

        if out["meta_info"]["finish_reason"]["type"] == "abort":
            status = Sample.Status.ABORTED
            break

        if "output_token_logprobs" in out["meta_info"]:
            new_ids = [item[1] for item in out["meta_info"]["output_token_logprobs"]]
            new_lps = [item[0] for item in out["meta_info"]["output_token_logprobs"]]
        else:
            new_ids = tokenizer.encode(out.get("text", ""), add_special_tokens=False)
            new_lps = [0.0] * len(new_ids)

        chunk_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        response_text += chunk_text
        response_token_ids.extend(new_ids)
        loss_mask.extend([1] * len(new_ids))
        rollout_log_probs.extend(new_lps)

        if out["meta_info"]["finish_reason"]["type"] == "stop":
            status = Sample.Status.COMPLETED
            break
        if out["meta_info"]["finish_reason"]["type"] == "length":
            status = Sample.Status.TRUNCATED
            break

    if status == Sample.Status.PENDING:
        status = Sample.Status.COMPLETED if response_token_ids else Sample.Status.TRUNCATED

    m = BOXED_INT_RE.findall(response_text)
    extracted = m[-1] if m else None

    return AttemptResult(
        prompt_ids=prompt_ids,
        response_token_ids=response_token_ids,
        response_text=response_text,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        extracted_answer=extracted,
        status=status,
        metadata={"round": "judge"},
    )


def _attempt_to_sample(template: Sample, att: AttemptResult, *, round_number: int) -> Sample:
    s = copy.deepcopy(template)
    s.tokens = att.prompt_ids + att.response_token_ids
    s.response_length = len(att.response_token_ids)
    s.response = att.response_text
    s.loss_mask = att.loss_mask
    s.rollout_log_probs = att.rollout_log_probs
    s.status = att.status
    s.metadata = dict(template.metadata or {})
    s.metadata["round_number"] = round_number
    s.metadata.update(att.metadata)
    s.reward = scalar_correctness_reward(att.response_text, template.label or "")
    return s


async def process_group(args: Any, group: list[Sample]) -> list[Sample]:
    cfg = ScaffoldingCFG.from_env()
    expected = cfg.attempts + 1
    assert len(group) == expected, (
        f"Group size {len(group)} must equal attempts+1={expected}. "
        "Set --n-samples-per-prompt to match SLIME_SCAFFOLDING_ATTEMPTS+1."
    )

    state = GenerateState(args)
    template = group[0]
    problem_text = template.prompt if isinstance(template.prompt, str) else str(template.prompt)

    problems_remaining = max(1, cfg.problems_remaining_default - int(template.group_index or 0))
    meta = template.metadata if isinstance(template.metadata, dict) else {}
    if "problems_remaining" in meta:
        problems_remaining = int(meta["problems_remaining"])

    notebook_elapsed = float(meta.get("notebook_elapsed", time.time() - NOTEBOOK_START_TIME))
    budget = _problem_budget_s(cfg, problems_remaining, notebook_elapsed)
    deadline = time.time() + budget

    sampling_params = state.sampling_params.copy()
    sampling_params["temperature"] = cfg.temperature
    if "min_p" not in sampling_params:
        sampling_params["min_p"] = cfg.min_p

    session_id = getattr(template, "session_id", None) or None

    # Parallel attempts with early-stop
    attempt_tasks: list[asyncio.Task[Any]] = []
    for i in range(cfg.attempts):
        if time.time() >= deadline:
            break
        attempt_tasks.append(
            asyncio.create_task(
                run_one_attempt(
                    args,
                    state,
                    cfg,
                    problem_text,
                    i,
                    deadline,
                    sampling_params,
                    session_id,
                )
            )
        )

    attempts: list[AttemptResult | None] = [None] * cfg.attempts
    vote_counts: dict[str, int] = defaultdict(int)
    early_stop = False

    for fut in asyncio.as_completed(attempt_tasks):
        try:
            att = await fut
        except asyncio.CancelledError:
            continue
        idx = att.metadata.get("attempt_idx", 0)
        if isinstance(idx, int) and 0 <= idx < cfg.attempts:
            attempts[idx] = att
        if att.extracted_answer is not None:
            vote_counts[str(att.extracted_answer)] += 1
            if vote_counts and max(vote_counts.values()) >= cfg.early_stop:
                early_stop = True
        if early_stop:
            break

    for t in attempt_tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*attempt_tasks, return_exceptions=True)

    ref_prompt_ids = _encode_prompt_ids(state.tokenizer, problem_text)
    for a in attempts:
        if a is not None and getattr(a, "prompt_ids", None):
            ref_prompt_ids = a.prompt_ids
            break

    # Fill missing attempts with truncated placeholders
    for i in range(cfg.attempts):
        if attempts[i] is None:
            attempts[i] = AttemptResult(
                prompt_ids=ref_prompt_ids,
                response_token_ids=[],
                response_text="",
                loss_mask=[],
                rollout_log_probs=[],
                extracted_answer=None,
                status=Sample.Status.TRUNCATED,
                metadata={"attempt_idx": i, "cancelled": True},
            )

    # Always run judge (no majority gate)
    if time.time() < deadline:
        judge = await run_judge_round(
            args,
            state,
            cfg,
            problem_text,
            list(attempts),
            deadline,
            sampling_params,
            session_id,
        )
    else:
        judge = AttemptResult(
            prompt_ids=_encode_judge_prompt_ids(state.tokenizer, problem_text, list(attempts)),
            response_token_ids=[],
            response_text="",
            loss_mask=[],
            rollout_log_probs=[],
            extracted_answer=None,
            status=Sample.Status.TRUNCATED,
            metadata={"round": "judge", "timeout": True},
        )

    out_samples: list[Sample] = []
    for i in range(cfg.attempts):
        s = _attempt_to_sample(group[i], attempts[i], round_number=1)
        s.index = group[i].index
        s.group_index = group[i].group_index
        out_samples.append(s)

    sj = _attempt_to_sample(group[cfg.attempts], judge, round_number=2)
    sj.index = group[cfg.attempts].index
    sj.group_index = group[cfg.attempts].group_index
    out_samples.append(sj)

    # wandb extras
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "scaffolding/early_stop": float(early_stop),
                    "scaffolding/problem_budget_s": budget,
                    "scaffolding/r1_mean_reward": sum(s.reward for s in out_samples[: cfg.attempts]) / cfg.attempts,
                    "scaffolding/r2_reward": float(out_samples[-1].reward),
                }
            )
    except Exception:
        pass

    return out_samples


async def generate_rollout_gs_async(
    args: Any,
    rollout_id: int,
    data_source_fn: Callable[[int], list[list[Sample]]],
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    assert args.rollout_global_dataset
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    metric_gatherer = MetricGatherer()
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Scaffolding rollout")

    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source_fn(args.over_sampling_batch_size)
            for group in samples:
                state.pendings.add(asyncio.create_task(process_group(args, group)))
            state.remaining_batch_size += len(samples)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()
            if do_print:
                logger.info(
                    "First scaffolding group: reward=%s",
                    [s.reward for s in group[:3]],
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    aborted_samples = await abort(args, rollout_id)
    state.reset()
    metrics = metric_gatherer.collect()
    metrics["rollout/scaffolding/ok"] = True
    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


def generate_rollout_gs(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    assert args.rollout_global_dataset
    if evaluation:
        return RolloutFnEvalOutput(data={})

    output, aborted = run(generate_rollout_gs_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted)
    return output
