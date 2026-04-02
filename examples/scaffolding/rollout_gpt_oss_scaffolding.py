"""
Custom rollout: parallel solve attempts + parallel gen-select judges.
Trains only on model-generated tokens (loss_mask); tool / observation tokens masked 0.

Each problem yields ``2 * attempts`` samples (default 8 solvers + 8 judges). GRPO normalization is
split across solver and judge groups via ``--custom-reward-post-process-path``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from tqdm import tqdm

from examples.scaffolding.gs_config import GEN_SELECT_PROMPT, PREFERENCE_PROMPT, SYSTEM_PROMPT, TOOL_PROMPT, ScaffoldingCFG
from examples.scaffolding.python_tool import PersistentPythonSession
from examples.scaffolding.reward_gpt_oss_scaffolding import (
    judge_selection_reward,
    scalar_correctness_reward,
)
from examples.scaffolding.scaffolding_boxed import (
    boxed_answer_valid_for_stop,
    extract_last_boxed_integer,
    extract_last_notebook_solver_integer,
    normalize_int_answer,
    stream_answer_valid_for_stop,
    stream_notebook_solver_answer_valid_for_stop,
    notebook_solver_answer_valid_for_stop,
)
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, abort
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

try:
    from openai_harmony import (
        HarmonyEncodingName,
        Author,
        Conversation,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
        TextContent,
        ToolNamespaceConfig,
        load_harmony_encoding,
    )
except ImportError:
    Author = Conversation = Message = ReasoningEffort = Role = SystemContent = TextContent = ToolNamespaceConfig = None
    HarmonyEncodingName = None
    load_harmony_encoding = None

CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TOOL_OBSERVATION_TEMPLATE = "<|im_end|>\n<|im_start|>tool\n{content}\n<|im_end|>\n<|im_start|>assistant\n"
ANALYSIS_ONLY_TOOL_NUDGE = (
    "You have continued with prose-only reasoning. Your very next assistant message must call the python notebook. "
    "Send only executable Python code to recipient python that performs a concrete computation, search, or check "
    "useful for solving this problem. Do not send another analysis-only message. Do not restate the problem in "
    "comments. If you import anything, immediately follow it with a real computation and print(...)."
)
FIRST_TOOL_RETRY_NUDGE = (
    "Restart from scratch. Your first assistant message must call recipient python with only executable Python code. "
    "Do not write any analysis before the tool call. Use the notebook immediately to run a concrete computation, "
    "search, or case-check that advances the solution. Do not restate the problem in comments. Do not send "
    "imports-only code. Your tool call must quickly compute something and print the result."
)
EMPTY_TOOL_OUTPUT_NUDGE = (
    "The previous python call produced no visible output. Your very next assistant message must again call "
    "recipient python with only executable Python code. Run a concrete computation and use print(...) to show a "
    "useful intermediate result. Do not send prose, comments-only code, or imports-only code."
)
FORCED_PYTHON_PREFIX = "<|channel|>analysis to=python<|message|>"
NOTEBOOK_START_TIME = time.time()
_HARMONY_ENCODING = None
_STREAM_HTTP_CLIENT: httpx.AsyncClient | None = None
NO_VISIBLE_OUTPUT_TOOL_PREFIX = "[WARN] No visible output."


def _response_preview(text: str, limit: int = 400) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _tool_obs_requires_python_followup(obs: str) -> bool:
    return (obs or "").strip().startswith(NO_VISIBLE_OUTPUT_TOOL_PREFIX)


def _get_stream_http_client() -> httpx.AsyncClient:
    global _STREAM_HTTP_CLIENT
    if _STREAM_HTTP_CLIENT is None:
        _STREAM_HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(None))
    return _STREAM_HTTP_CLIENT


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


def _augment_problem_text(problem_text: str) -> str:
    return f"{problem_text.rstrip()} {PREFERENCE_PROMPT}".strip()


def _get_harmony_encoding() -> Any | None:
    global _HARMONY_ENCODING
    if load_harmony_encoding is None or HarmonyEncodingName is None:
        return None
    if _HARMONY_ENCODING is None:
        _HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _HARMONY_ENCODING


def _get_harmony_stop_token_ids(existing: list[int] | None = None) -> list[int] | None:
    encoding = _get_harmony_encoding()
    if encoding is None:
        return existing

    merged: list[int] = list(existing or [])
    for token_id in encoding.stop_tokens_for_assistant_actions():
        if token_id not in merged:
            merged.append(token_id)
    return merged


def _build_harmony_solver_conversation(problem_text: str) -> Any | None:
    if Conversation is None or Message is None or Role is None or ToolNamespaceConfig is None:
        return None

    tool_cfg = ToolNamespaceConfig(name="python", description=TOOL_PROMPT, tools=[])
    system = (
        SystemContent.new()
        .with_model_identity(SYSTEM_PROMPT)
        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        .with_tools(tool_cfg)
    )
    return Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.USER, problem_text),
        ]
    )


def _maybe_add_harmony_python_fewshot(conv: Any, cfg: ScaffoldingCFG) -> Any:
    if not getattr(cfg, "harmony_python_fewshot", False):
        return conv
    if conv is None or Message is None or Role is None:
        return conv

    example_problem = (
        "Example mini-problem: How many even-sized subsets does a 3-element set have? "
        "Use the python notebook immediately, then give the boxed final answer."
    )
    example_code = "from math import comb\nprint(sum(comb(3, k) for k in range(4) if k % 2 == 0))"
    example_obs = "4"
    example_final = "\\boxed{4}"

    messages = list(getattr(conv, "messages", []) or [])
    if len(messages) < 2:
        return conv

    messages.insert(1, Message.from_role_and_content(Role.USER, example_problem))
    example_tool_call = (
        Message.from_role_and_content(Role.ASSISTANT, example_code)
        .with_channel("analysis")
        .with_recipient("python")
    )
    messages.insert(2, example_tool_call)
    messages.insert(3, _make_harmony_tool_message(example_obs, channel="analysis"))
    messages.insert(4, Message.from_role_and_content(Role.ASSISTANT, example_final).with_channel("final"))
    conv.messages = messages
    return conv


def _harmony_message_text(message: Any) -> str:
    parts: list[str] = []
    for item in getattr(message, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _make_harmony_tool_message(output: str, *, channel: Any | None = None) -> Any:
    message = Message(author=Author(role=Role.TOOL, name="python"), content=[TextContent(text=output)]).with_recipient(
        "assistant"
    )
    return message.with_channel(channel) if channel else message


def _make_harmony_user_message(text: str) -> Any:
    return Message.from_role_and_content(Role.USER, text)


def _extract_forced_python_code(text: str) -> str:
    stripped = text or ""
    if stripped.startswith(FORCED_PYTHON_PREFIX):
        stripped = stripped[len(FORCED_PYTHON_PREFIX) :]
    marker_idx = stripped.find("<|")
    if marker_idx >= 0:
        stripped = stripped[:marker_idx]
    return stripped.strip()


def _build_solver_prompt_context(
    tokenizer: Any,
    cfg: ScaffoldingCFG,
    problem_text: str,
    extra_user_messages: list[str],
) -> tuple[list[int], Any | None, Any | None]:
    prompt_text = problem_text
    if extra_user_messages:
        prompt_text = "\n\n".join([prompt_text, *extra_user_messages])

    harmony_encoding = _get_harmony_encoding()
    harmony_conv = _build_harmony_solver_conversation(prompt_text)
    harmony_conv = _maybe_add_harmony_python_fewshot(harmony_conv, cfg)
    if harmony_encoding is not None and harmony_conv is not None:
        prompt_ids = harmony_encoding.render_conversation_for_completion(harmony_conv, Role.ASSISTANT)
        return prompt_ids, harmony_encoding, harmony_conv

    return _encode_solver_prompt_ids(tokenizer, prompt_text), None, None


async def _synthesize_forced_python_tool_turn(
    *,
    attempt_idx: int,
    cfg: ScaffoldingCFG,
    tokenizer: Any,
    harmony_encoding: Any,
    harmony_conv: Any,
    session: PersistentPythonSession,
    prompt_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
    turn_token_ids: list[int],
    tool_call_previews: list[str],
    tool_result_previews: list[str],
) -> tuple[bool, int]:
    synthetic_text = tokenizer.decode(turn_token_ids, skip_special_tokens=False)
    code = _extract_forced_python_code(synthetic_text)
    if not code:
        return False, 0

    assistant_msg = Message.from_role_and_content(Role.ASSISTANT, code).with_channel("analysis").with_recipient(
        "python"
    )
    harmony_conv.messages.append(assistant_msg)
    with_assistant_ids = harmony_encoding.render_conversation_for_completion(harmony_conv, Role.ASSISTANT)
    assistant_response_ids = with_assistant_ids[len(prompt_ids) :]
    current_len = len(response_token_ids)
    if assistant_response_ids[:current_len] != response_token_ids:
        logger.warning(
            "Synthetic forced-python assistant render diverged from generated prefix on attempt %s; skipping synthetic tool execution.",
            attempt_idx,
        )
        harmony_conv.messages.pop()
        return False, 0

    assistant_delta_ids = assistant_response_ids[current_len:]
    if assistant_delta_ids:
        response_token_ids.extend(assistant_delta_ids)
        loss_mask.extend([0] * len(assistant_delta_ids))
        rollout_log_probs.extend([0.0] * len(assistant_delta_ids))

    tool_call_previews.append(_response_preview(code, limit=200))
    obs = await session.execute(code, cfg.jupyter_timeout)
    tool_result_previews.append(_response_preview(obs, limit=200))
    if attempt_idx == 0:
        logger.info(
            "Solver attempt 0 synthesized forced python tool call at chunk boundary: code=%s obs=%s",
            tool_call_previews[-1],
            tool_result_previews[-1],
        )

    harmony_conv.messages.append(_make_harmony_tool_message(obs, channel="analysis"))
    with_tool_ids = harmony_encoding.render_conversation_for_completion(harmony_conv, Role.ASSISTANT)
    tool_response_ids = with_tool_ids[len(prompt_ids) :]
    tool_delta_ids = tool_response_ids[len(response_token_ids) :]
    if tool_delta_ids:
        response_token_ids.extend(tool_delta_ids)
        loss_mask.extend([0] * len(tool_delta_ids))
        rollout_log_probs.extend([0.0] * len(tool_delta_ids))

    return True, int("[ERROR]" in obs)


def _collect_scaffolding_metrics(groups: list[list[Sample]]) -> dict[str, float]:
    flat = [sample for group in groups for sample in group]
    if not flat:
        return {}

    solvers = [s for s in flat if (s.metadata or {}).get("round_type") == "solver"]
    judges = [s for s in flat if (s.metadata or {}).get("round_type") == "judge"]

    def _mean_reward(items: list[Sample]) -> float:
        return sum(float(s.reward or 0.0) for s in items) / len(items) if items else 0.0

    def _extract_rate(items: list[Sample]) -> float:
        if not items:
            return 0.0
        extracted = sum(1 for s in items if extract_last_boxed_integer(s.response or "") is not None)
        return extracted / len(items)

    return {
        "scaffolding/solver_mean_reward": _mean_reward(solvers),
        "scaffolding/judge_mean_reward": _mean_reward(judges),
        "scaffolding/solver_extract_rate": _extract_rate(solvers),
        "scaffolding/judge_extract_rate": _extract_rate(judges),
    }


def _problem_budget_s(cfg: ScaffoldingCFG, problems_remaining: int, notebook_elapsed: float) -> float:
    """Notebook dynamic budget: max(base, min(time_left - (k-1)*base, high))."""
    time_left = cfg.notebook_limit - notebook_elapsed
    k = max(0, problems_remaining - 1)
    inner = time_left - k * cfg.base_problem_timeout
    return max(cfg.base_problem_timeout, min(inner, cfg.high_problem_timeout))


def _format_solutions_for_judge(attempts: list[AttemptResult]) -> str:
    parts: list[str] = []
    for i, att in enumerate(attempts):
        last_answer_turn = (att.metadata or {}).get("last_answer_turn")
        if last_answer_turn:
            text = last_answer_turn
        else:
            response = att.response_text or "(no response)"
            boxed_pos = response.rfind("\\boxed{")
            if boxed_pos >= 0:
                start = max(0, boxed_pos - 2500)
                text = ("..." if start > 0 else "") + response[start:]
                if len(text) > 4000:
                    text = "..." + text[-4000:]
            elif len(response) > 3000:
                text = "..." + response[-3000:]
            else:
                text = response
        if att.extracted_answer is not None:
            ans = str(att.extracted_answer)
        else:
            ext = extract_last_boxed_integer(att.response_text)
            ans = ext if ext is not None else "(no answer)"
        parts.append(f"--- Solution {i} (final answer: {ans}) ---\n{text}")
    return "\n\n".join(parts)


def _encode_solver_prompt_ids(tokenizer: Any, problem_text: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.encode(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem_text}<|im_end|>\n"
            f"<|im_start|>assistant\n",
            add_special_tokens=False,
        )


def _encode_judge_prompt_ids(tokenizer: Any, problem_text: str, attempt_results: list[AttemptResult]) -> list[int]:
    """Match gen-select-nb.ipynb: full GEN_SELECT_PROMPT as a single user message (no extra system turn)."""
    sols = _format_solutions_for_judge(attempt_results)
    n = len(attempt_results)
    user_prompt = GEN_SELECT_PROMPT.format(
        num_solutions=n,
        max_idx=n - 1,
        problem=problem_text,
        solutions=sols,
    )
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.encode(
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )


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


async def _sglang_generate_stream(
    args: Any,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    session_headers: dict[str, str] | None = None,
):
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "stream": True,
    }
    headers = session_headers
    client = _get_stream_http_client()
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            yield json.loads(data)


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
    prompt_ids: list[int] = []
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    response_text = ""
    status = Sample.Status.PENDING
    last_answer_turn: str | None = None
    tool_call_count = 0
    tool_error_count = 0
    tool_call_previews: list[str] = []
    tool_result_previews: list[str] = []
    last_finish_reason: str | None = None
    last_harmony_recipient: str | None = None
    last_harmony_channel: str | None = None
    assistant_message_count = 0
    analysis_only_nudge_count = 0
    stall_detected = False
    python_first_retry_count = 0
    pending_python_followup_after_tool = False
    no_output_tool_retry_count = 0
    extra_user_messages: list[str] = []

    headers = None
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and session_id:
        headers = {"X-SMG-Routing-Key": f"{session_id}-att{attempt_idx}"}

    base_sampling_seed = sampling_params.get("sampling_seed", getattr(args, "rollout_seed", 42))
    try:
        base_sampling_seed = int(base_sampling_seed)
    except (TypeError, ValueError):
        base_sampling_seed = 42
    attempt_sampling_seed = int(math.pow(base_sampling_seed + attempt_idx, 2))

    while True:
        prompt_ids, harmony_encoding, harmony_conv = _build_solver_prompt_context(
            tokenizer, cfg, problem_text, extra_user_messages
        )
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = []
        response_text = ""
        status = Sample.Status.PENDING
        last_answer_turn = None
        tool_call_count = 0
        tool_error_count = 0
        tool_call_previews = []
        tool_result_previews = []
        last_finish_reason = None
        last_harmony_recipient = None
        last_harmony_channel = None
        assistant_message_count = 0
        stall_warning_emitted = False
        analysis_only_warning_emitted = False
        analysis_only_nudge_count = 0
        stall_detected = False
        pending_python_followup_after_tool = False
        no_output_tool_retry_count = 0
        no_action_progress_bucket = 0
        retry_with_forced_python_first = False
        turn = 0
        max_turns = 0 if getattr(args, "ci_test", False) else cfg.turns
        session = PersistentPythonSession()

        try:
            while turn < max_turns and time.time() < deadline:
                current_ids = prompt_ids + response_token_ids
                if harmony_encoding is not None and harmony_conv is not None:
                    current_ids = harmony_encoding.render_conversation_for_completion(harmony_conv, Role.ASSISTANT)
                    if current_ids[: len(prompt_ids)] != prompt_ids:
                        logger.warning(
                            "Harmony prompt prefix changed for attempt %s; using rendered prompt as source of truth",
                            attempt_idx,
                        )
                    if len(current_ids) >= len(prompt_ids):
                        response_token_ids = current_ids[len(prompt_ids) :]
                        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
                max_ctx = getattr(args, "rollout_max_context_len", None) or 65536
                if len(current_ids) >= max_ctx - cfg.buffer_tokens:
                    status = Sample.Status.TRUNCATED
                    break

                turn_start = len(response_token_ids)
                turn_done = False
                finish = None
                hit_artificial_chunk_limit = False
                while not turn_done and time.time() < deadline:
                    current_ids = prompt_ids + response_token_ids
                    forced_prefix_ids: list[int] = []
                    forced_prefix_text = ""
                    if (
                        harmony_encoding is not None
                        and harmony_conv is not None
                        and cfg.harmony_force_python_prefix_after_retry
                        and (
                            (
                                tool_call_count == 0
                                and (python_first_retry_count > 0 or analysis_only_nudge_count > 0)
                            )
                            or pending_python_followup_after_tool
                        )
                    ):
                        forced_prefix_text = FORCED_PYTHON_PREFIX
                        forced_prefix_ids = tokenizer.encode(forced_prefix_text, add_special_tokens=False)
                        current_ids = current_ids + forced_prefix_ids
                        if attempt_idx == 0:
                            logger.info(
                                "Solver attempt 0 forcing python recipient prefix: prefix_len=%s retry_count=%s analysis_nudges=%s pending_tool_followup=%s no_output_tool_retries=%s",
                                len(forced_prefix_ids),
                                python_first_retry_count,
                                analysis_only_nudge_count,
                                pending_python_followup_after_tool,
                                no_output_tool_retry_count,
                            )
                    sp = sampling_params.copy()
                    sp["sampling_seed"] = attempt_sampling_seed
                    sp["stop_token_ids"] = _get_harmony_stop_token_ids(sp.get("stop_token_ids"))
                    if attempt_idx == 0:
                        logger.info(
                            "Solver attempt 0 debug: using_harmony=%s stop_token_count=%s prompt_len=%s sampling_seed=%s chunk_tokens=%s use_streaming=%s python_first_retry_count=%s",
                            harmony_encoding is not None and harmony_conv is not None,
                            len(sp.get("stop_token_ids") or []),
                            len(current_ids),
                            attempt_sampling_seed,
                            cfg.generation_chunk_tokens,
                            cfg.use_streaming and cfg.generation_chunk_tokens <= 0,
                            python_first_retry_count,
                        )
                    remaining_tokens = max_ctx - len(current_ids) - 1
                    remaining_response_budget = (getattr(args, "rollout_max_response_len", 0) or 0) - len(
                        response_token_ids
                    )
                    # generation_chunk_tokens <= 0 disables artificial chunking and keeps notebook-like
                    # single-request assistant turns.
                    chunk_limit = (
                        cfg.generation_chunk_tokens if cfg.generation_chunk_tokens > 0 else remaining_response_budget
                    )
                    sp["max_new_tokens"] = min(
                        sp.get("max_new_tokens", remaining_response_budget),
                        chunk_limit,
                        remaining_tokens,
                        remaining_response_budget,
                    )
                    if sp["max_new_tokens"] <= 0:
                        status = Sample.Status.TRUNCATED
                        turn_done = True
                        break

                    use_streaming = cfg.use_streaming and cfg.generation_chunk_tokens <= 0
                    stream_answer_detected = False
                    hit_artificial_chunk_limit = False

                    if use_streaming:
                        finish = "length"
                        turn_text = ""
                        turn_new_ids: list[int] = []
                        turn_new_lps: list[float] = []
                        recent_chunks: list[str] = []

                        async for event in _sglang_generate_stream(args, current_ids, sp, headers):
                            meta = event.get("meta_info", {}) or {}
                            finish = (meta.get("finish_reason") or {}).get("type", finish)
                            full_text = event.get("text", "") or ""
                            delta_text = full_text[len(turn_text) :] if full_text.startswith(turn_text) else full_text

                            streamed_logprobs = meta.get("output_token_logprobs") or []
                            prev_token_count = len(turn_new_ids)
                            delta_items = (
                                streamed_logprobs[prev_token_count:]
                                if len(streamed_logprobs) >= prev_token_count
                                else []
                            )
                            if delta_items:
                                delta_ids = [item[1] for item in delta_items]
                                delta_lps = [item[0] for item in delta_items]
                                if not delta_text:
                                    delta_text = tokenizer.decode(delta_ids, skip_special_tokens=False)
                            else:
                                delta_ids = tokenizer.encode(delta_text, add_special_tokens=False) if delta_text else []
                                delta_lps = [0.0] * len(delta_ids)

                            if delta_text:
                                recent_chunks.append(delta_text)
                                if len(recent_chunks) > cfg.search_tokens:
                                    recent_chunks = recent_chunks[-cfg.search_tokens :]

                            turn_text += delta_text
                            turn_new_ids.extend(delta_ids)
                            turn_new_lps.extend(delta_lps)

                            preview_text = response_text + forced_prefix_text + turn_text
                            preview_token_count = len(response_token_ids) + len(forced_prefix_ids) + len(turn_new_ids)
                            if assistant_message_count == 0 and tool_call_count == 0:
                                if attempt_idx == 0 and preview_token_count >= 512:
                                    bucket = preview_token_count // 512
                                    if bucket > no_action_progress_bucket:
                                        no_action_progress_bucket = bucket
                                        logger.info(
                                            "Solver attempt 0 still has no Harmony action/final answer after %s streamed tokens. preview=%s",
                                            preview_token_count,
                                            _response_preview(preview_text, limit=240),
                                        )
                                if (
                                    cfg.first_tool_retry_tokens > 0
                                    and python_first_retry_count < cfg.first_tool_retry_limit
                                    and preview_token_count >= cfg.first_tool_retry_tokens
                                ):
                                    retry_with_forced_python_first = True
                                    last_finish_reason = "retry_python_first"
                                    logger.warning(
                                        "Solver attempt %s exceeded python-first retry threshold (%s tokens) without any Harmony action/final answer; restarting attempt with a forced python-first user message.",
                                        attempt_idx,
                                        cfg.first_tool_retry_tokens,
                                    )
                                    turn_done = True
                                    break
                                if cfg.stall_warning_tokens > 0 and preview_token_count >= cfg.stall_warning_tokens:
                                    if not stall_warning_emitted:
                                        stall_warning_emitted = True
                                        logger.warning(
                                            "Solver attempt %s has no Harmony action/final answer after %s tokens. preview=%s",
                                            attempt_idx,
                                            preview_token_count,
                                            _response_preview(preview_text, limit=240),
                                        )
                                if cfg.stall_fail_tokens > 0 and preview_token_count >= cfg.stall_fail_tokens:
                                    stall_detected = True
                                    last_finish_reason = "stall_no_action"
                                    status = Sample.Status.TRUNCATED
                                    logger.error(
                                        "Solver attempt %s exceeded stalled-token threshold (%s) without any Harmony action/final answer.",
                                        attempt_idx,
                                        cfg.stall_fail_tokens,
                                    )
                                    turn_done = True
                                    break

                            recent_text = "".join(recent_chunks)
                            if stream_notebook_solver_answer_valid_for_stop(preview_text, delta_text):
                                last_answer_turn = turn_text or last_answer_turn
                                status = Sample.Status.COMPLETED
                                stream_answer_detected = True
                                break

                        if attempt_idx == 0:
                            logger.info(
                                "Solver attempt 0 stream finished: finish=%s turn_tokens=%s total_response_tokens=%s answer_detected=%s preview=%s",
                                finish,
                                len(turn_new_ids),
                                len(response_token_ids) + len(forced_prefix_ids) + len(turn_new_ids),
                                stream_answer_detected,
                                _response_preview(turn_text or response_text, limit=240),
                            )

                        chunk_text = turn_text
                        new_ids = turn_new_ids
                        new_lps = turn_new_lps
                        if forced_prefix_ids:
                            response_text += forced_prefix_text
                            response_token_ids.extend(forced_prefix_ids)
                            loss_mask.extend([0] * len(forced_prefix_ids))
                            rollout_log_probs.extend([0.0] * len(forced_prefix_ids))
                        response_text += chunk_text
                        response_token_ids.extend(new_ids)
                        loss_mask.extend([1] * len(new_ids))
                        rollout_log_probs.extend(new_lps)
                    else:
                        out = await _sglang_generate(args, current_ids, sp, headers)

                        finish = out["meta_info"]["finish_reason"]["type"]
                        if "output_token_logprobs" in out["meta_info"]:
                            new_ids = [item[1] for item in out["meta_info"]["output_token_logprobs"]]
                            new_lps = [item[0] for item in out["meta_info"]["output_token_logprobs"]]
                        else:
                            new_ids = tokenizer.encode(out.get("text", ""), add_special_tokens=False)
                            new_lps = [0.0] * len(new_ids)

                        chunk_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                        if forced_prefix_ids:
                            response_text += forced_prefix_text
                            response_token_ids.extend(forced_prefix_ids)
                            loss_mask.extend([0] * len(forced_prefix_ids))
                            rollout_log_probs.extend([0.0] * len(forced_prefix_ids))
                        response_text += chunk_text
                        response_token_ids.extend(new_ids)
                        loss_mask.extend([1] * len(new_ids))
                        rollout_log_probs.extend(new_lps)

                        if assistant_message_count == 0 and tool_call_count == 0:
                            if attempt_idx == 0 and len(response_token_ids) >= 512:
                                bucket = len(response_token_ids) // 512
                                if bucket > no_action_progress_bucket:
                                    no_action_progress_bucket = bucket
                                    logger.info(
                                        "Solver attempt 0 still has no Harmony action/final answer after %s tokens. preview=%s",
                                        len(response_token_ids),
                                        _response_preview(response_text, limit=240),
                                    )
                            if (
                                cfg.first_tool_retry_tokens > 0
                                and python_first_retry_count < cfg.first_tool_retry_limit
                                and len(response_token_ids) >= cfg.first_tool_retry_tokens
                            ):
                                retry_with_forced_python_first = True
                                last_finish_reason = "retry_python_first"
                                logger.warning(
                                    "Solver attempt %s exceeded python-first retry threshold (%s tokens) without any Harmony action/final answer; restarting attempt with a forced python-first user message.",
                                    attempt_idx,
                                    cfg.first_tool_retry_tokens,
                                )
                                turn_done = True
                                break
                            if cfg.stall_warning_tokens > 0 and len(response_token_ids) >= cfg.stall_warning_tokens:
                                if not stall_warning_emitted:
                                    stall_warning_emitted = True
                                    logger.warning(
                                        "Solver attempt %s has no Harmony action/final answer after %s tokens. preview=%s",
                                        attempt_idx,
                                        len(response_token_ids),
                                        _response_preview(response_text, limit=240),
                                    )
                            if cfg.stall_fail_tokens > 0 and len(response_token_ids) >= cfg.stall_fail_tokens:
                                stall_detected = True
                                last_finish_reason = "stall_no_action"
                                status = Sample.Status.TRUNCATED
                                logger.error(
                                    "Solver attempt %s exceeded stalled-token threshold (%s) without any Harmony action/final answer.",
                                    attempt_idx,
                                    cfg.stall_fail_tokens,
                                )
                                turn_done = True
                                break

                        hit_artificial_chunk_limit = finish == "length" and sp["max_new_tokens"] < remaining_tokens

                    if turn_done:
                        break

                    observed_message_or_tool = False
                    last_finish_reason = finish
                    if finish == "abort":
                        status = Sample.Status.ABORTED
                        turn_done = True
                        break

                    if stream_answer_detected or notebook_solver_answer_valid_for_stop(response_text):
                        last_answer_turn = chunk_text or response_text or last_answer_turn
                        status = Sample.Status.COMPLETED
                        turn_done = True
                        break

                    if harmony_encoding is not None and harmony_conv is not None:
                        turn_token_ids = response_token_ids[turn_start:]
                        parsed_messages = harmony_encoding.parse_messages_from_completion_tokens(
                            turn_token_ids, Role.ASSISTANT
                        )
                        if attempt_idx == 0:
                            if parsed_messages:
                                last = parsed_messages[-1]
                                logger.info(
                                    "Solver attempt 0 parsed %s Harmony messages from completed turn. last_recipient=%s last_channel=%s preview=%s",
                                    len(parsed_messages),
                                    getattr(last, "recipient", None),
                                    getattr(last, "channel", None),
                                    _response_preview(_harmony_message_text(last), limit=240),
                                )
                            else:
                                logger.warning(
                                    "Solver attempt 0 completed a streamed turn but Harmony parsed 0 messages. finish=%s turn_tokens=%s preview=%s",
                                    finish,
                                    len(turn_token_ids),
                                    _response_preview(tokenizer.decode(turn_token_ids, skip_special_tokens=False), limit=240),
                                )
                        harmony_conv.messages.extend(parsed_messages)
                        if parsed_messages:
                            observed_message_or_tool = True
                            assistant_message_count += len(parsed_messages)
                            last = parsed_messages[-1]
                            last_text = _harmony_message_text(last)
                            if attempt_idx == 0:
                                logger.info(
                                    "Solver attempt 0 harmony message: recipient=%s channel=%s preview=%s",
                                    getattr(last, "recipient", None),
                                    getattr(last, "channel", None),
                                    _response_preview(last_text, limit=240),
                                )
                            last_harmony_recipient = getattr(last, "recipient", None)
                            last_harmony_channel = getattr(last, "channel", None)
                            if pending_python_followup_after_tool and last_harmony_recipient != "python":
                                logger.warning(
                                    "Solver attempt %s produced a non-python Harmony message immediately after a no-output python tool call. recipient=%s preview=%s",
                                    attempt_idx,
                                    last_harmony_recipient,
                                    _response_preview(last_text, limit=200),
                                )
                            if notebook_solver_answer_valid_for_stop(last_text) or notebook_solver_answer_valid_for_stop(response_text):
                                last_answer_turn = last_text or last_answer_turn
                                status = Sample.Status.COMPLETED
                                turn_done = True
                                break

                            if tool_call_count == 0 and last_harmony_recipient != "python":
                                if (
                                    cfg.analysis_only_warning_turns > 0
                                    and assistant_message_count >= cfg.analysis_only_warning_turns
                                ):
                                    if not analysis_only_warning_emitted:
                                        analysis_only_warning_emitted = True
                                        logger.warning(
                                            "Solver attempt %s has produced %s Harmony analysis messages without a tool call or final answer. preview=%s",
                                            attempt_idx,
                                            assistant_message_count,
                                            _response_preview(last_text or response_text, limit=240),
                                        )
                                    min_nudge_count = assistant_message_count - cfg.analysis_only_warning_turns + 1
                                    if analysis_only_nudge_count < min_nudge_count:
                                        reminder = _make_harmony_user_message(ANALYSIS_ONLY_TOOL_NUDGE)
                                        harmony_conv.messages.append(reminder)
                                        next_ids = harmony_encoding.render_conversation_for_completion(
                                            harmony_conv, Role.ASSISTANT
                                        )
                                        delta_start = len(prompt_ids) + len(response_token_ids)
                                        reminder_ids = next_ids[delta_start:]
                                        response_token_ids.extend(reminder_ids)
                                        loss_mask.extend([0] * len(reminder_ids))
                                        rollout_log_probs.extend([0.0] * len(reminder_ids))
                                        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
                                        analysis_only_nudge_count += 1
                                        if attempt_idx == 0:
                                            logger.info(
                                                "Solver attempt 0 injected tool-only analysis reminder #%s after %s assistant messages.",
                                                analysis_only_nudge_count,
                                                assistant_message_count,
                                            )
                                if (
                                    cfg.analysis_only_fail_turns > 0
                                    and assistant_message_count >= cfg.analysis_only_fail_turns
                                ):
                                    stall_detected = True
                                    last_finish_reason = "stall_analysis_only"
                                    status = Sample.Status.TRUNCATED
                                    logger.error(
                                        "Solver attempt %s exceeded Harmony analysis-only threshold (%s turns) without a tool call or final answer.",
                                        attempt_idx,
                                        cfg.analysis_only_fail_turns,
                                    )
                                    turn_done = True
                                    break

                            if getattr(last, "recipient", None) == "python" and time.time() < deadline:
                                code = last_text.strip()
                                if code:
                                    tool_call_count += 1
                                    tool_call_previews.append(_response_preview(code, limit=200))
                                    obs = await session.execute(code, cfg.jupyter_timeout)
                                    if "[ERROR]" in obs:
                                        tool_error_count += 1
                                    tool_result_previews.append(_response_preview(obs, limit=200))
                                    pending_python_followup_after_tool = (
                                        cfg.harmony_force_python_prefix_after_retry
                                        and _tool_obs_requires_python_followup(obs)
                                    )
                                    if pending_python_followup_after_tool:
                                        no_output_tool_retry_count += 1
                                    if attempt_idx == 0:
                                        logger.info(
                                            "Solver attempt 0 harmony tool call: code=%s obs=%s",
                                            tool_call_previews[-1],
                                            tool_result_previews[-1],
                                        )
                                    harmony_conv.messages.extend(
                                        [_make_harmony_tool_message(obs, channel=getattr(last, "channel", None))]
                                    )
                                    next_ids = harmony_encoding.render_conversation_for_completion(
                                        harmony_conv, Role.ASSISTANT
                                    )
                                    delta_start = len(prompt_ids) + len(response_token_ids)
                                    obs_ids = next_ids[delta_start:]
                                    response_token_ids.extend(obs_ids)
                                    loss_mask.extend([0] * len(obs_ids))
                                    rollout_log_probs.extend([0.0] * len(obs_ids))
                                    response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
                                    if pending_python_followup_after_tool:
                                        reminder = _make_harmony_user_message(EMPTY_TOOL_OUTPUT_NUDGE)
                                        harmony_conv.messages.append(reminder)
                                        next_ids = harmony_encoding.render_conversation_for_completion(
                                            harmony_conv, Role.ASSISTANT
                                        )
                                        delta_start = len(prompt_ids) + len(response_token_ids)
                                        reminder_ids = next_ids[delta_start:]
                                        response_token_ids.extend(reminder_ids)
                                        loss_mask.extend([0] * len(reminder_ids))
                                        rollout_log_probs.extend([0.0] * len(reminder_ids))
                                        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
                                        if attempt_idx == 0:
                                            logger.warning(
                                                "Solver attempt 0 python tool call produced no visible output; injected follow-up python reminder #%s.",
                                                no_output_tool_retry_count,
                                            )
                        elif (
                            hit_artificial_chunk_limit
                            and forced_prefix_ids
                            and (
                                (
                                    tool_call_count == 0
                                    and (python_first_retry_count > 0 or analysis_only_nudge_count > 0)
                                )
                                or pending_python_followup_after_tool
                            )
                            and time.time() < deadline
                        ):
                            synthesized, synthetic_error = await _synthesize_forced_python_tool_turn(
                                attempt_idx=attempt_idx,
                                cfg=cfg,
                                tokenizer=tokenizer,
                                harmony_encoding=harmony_encoding,
                                harmony_conv=harmony_conv,
                                session=session,
                                prompt_ids=prompt_ids,
                                response_token_ids=response_token_ids,
                                loss_mask=loss_mask,
                                rollout_log_probs=rollout_log_probs,
                                turn_token_ids=turn_token_ids,
                                tool_call_previews=tool_call_previews,
                                tool_result_previews=tool_result_previews,
                            )
                            if synthesized:
                                observed_message_or_tool = True
                                assistant_message_count += 1
                                tool_call_count += 1
                                tool_error_count += synthetic_error
                                pending_python_followup_after_tool = bool(
                                    cfg.harmony_force_python_prefix_after_retry
                                    and tool_result_previews
                                    and _tool_obs_requires_python_followup(tool_result_previews[-1])
                                )
                                if pending_python_followup_after_tool:
                                    no_output_tool_retry_count += 1
                                last_harmony_recipient = "python"
                                last_harmony_channel = "analysis"
                                response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
                    else:
                        if notebook_solver_answer_valid_for_stop(response_text):
                            last_answer_turn = chunk_text or last_answer_turn
                            status = Sample.Status.COMPLETED
                            turn_done = True
                            break

                        match = CODE_BLOCK_RE.search(chunk_text)
                        if match and time.time() < deadline:
                            code = match.group(1).strip()
                            if code:
                                observed_message_or_tool = True
                                tool_call_count += 1
                                tool_call_previews.append(_response_preview(code, limit=200))
                                obs = await session.execute(code, cfg.jupyter_timeout)
                                if "[ERROR]" in obs:
                                    tool_error_count += 1
                                tool_result_previews.append(_response_preview(obs, limit=200))
                                pending_python_followup_after_tool = (
                                    cfg.harmony_force_python_prefix_after_retry
                                    and _tool_obs_requires_python_followup(obs)
                                )
                                if pending_python_followup_after_tool:
                                    no_output_tool_retry_count += 1
                                obs_str = TOOL_OBSERVATION_TEMPLATE.format(content=obs)
                                obs_ids = tokenizer.encode(obs_str, add_special_tokens=False)
                                response_token_ids.extend(obs_ids)
                                loss_mask.extend([0] * len(obs_ids))
                                rollout_log_probs.extend([0.0] * len(obs_ids))
                                response_text += obs_str

                    if (
                        harmony_encoding is not None
                        and harmony_conv is not None
                        and not hit_artificial_chunk_limit
                        and not turn_done
                    ):
                        if attempt_idx == 0:
                            logger.info(
                                "Solver attempt 0 completed a full Harmony assistant turn; advancing to the next turn."
                            )
                        turn_done = True
                        break

                    if hit_artificial_chunk_limit and not turn_done:
                        if observed_message_or_tool:
                            if attempt_idx == 0:
                                logger.info(
                                    "Solver attempt 0 reached artificial chunk boundary after a parseable Harmony message/tool; continuing with the next turn."
                                )
                            turn_done = True
                            break

                        if attempt_idx == 0:
                            logger.info(
                                "Solver attempt 0 reached artificial chunk boundary without a parseable Harmony action; continuing assistant turn."
                            )
                        continue

                if status in (Sample.Status.COMPLETED, Sample.Status.ABORTED, Sample.Status.TRUNCATED):
                    break

                if finish == "length" and not hit_artificial_chunk_limit:
                    if attempt_idx == 0:
                        logger.info(
                            "Solver attempt 0 hit length limit without stop. response_preview=%s",
                            _response_preview(response_text, limit=240),
                            )
                        status = Sample.Status.TRUNCATED
                        turn_done = True
                        break

                    turn_done = True

                if retry_with_forced_python_first:
                    break

                turn += 1
                if retry_with_forced_python_first:
                    break
                if status in (Sample.Status.COMPLETED, Sample.Status.ABORTED, Sample.Status.TRUNCATED):
                    break
        finally:
            session.close()

        if retry_with_forced_python_first:
            python_first_retry_count += 1
            extra_user_messages.append(FIRST_TOOL_RETRY_NUDGE)
            logger.warning(
                "Restarting solver attempt %s from scratch with forced python-first retry #%s.",
                attempt_idx,
                python_first_retry_count,
            )
            continue

        break

    if status == Sample.Status.PENDING:
        status = Sample.Status.COMPLETED if response_token_ids else Sample.Status.TRUNCATED

    extracted = extract_last_notebook_solver_integer(response_text)

    return AttemptResult(
        prompt_ids=prompt_ids,
        response_token_ids=response_token_ids,
        response_text=response_text,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        extracted_answer=extracted,
        status=status,
        metadata={
            "attempt_idx": attempt_idx,
            "turns": turn,
            "last_answer_turn": last_answer_turn,
            "used_harmony": harmony_encoding is not None and harmony_conv is not None,
            "tool_call_count": tool_call_count,
            "tool_error_count": tool_error_count,
            "tool_call_previews": tool_call_previews[:6],
            "tool_result_previews": tool_result_previews[:6],
            "finish_reason": last_finish_reason,
            "last_harmony_recipient": last_harmony_recipient,
            "last_harmony_channel": last_harmony_channel,
            "assistant_message_count": assistant_message_count,
            "analysis_only_nudge_count": analysis_only_nudge_count,
            "stall_detected": stall_detected,
            "python_first_retry_count": python_first_retry_count,
            "pending_python_followup_after_tool": pending_python_followup_after_tool,
            "no_output_tool_retry_count": no_output_tool_retry_count,
        },
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
    judge_idx: int,
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
        headers = {"X-SMG-Routing-Key": f"{session_id}-judge{judge_idx}"}

    sp = sampling_params.copy()
    sp["temperature"] = cfg.judge_temperature
    sp["max_new_tokens"] = cfg.judge_max_tokens

    max_ctx = getattr(args, "rollout_max_context_len", None) or 65536
    while time.time() < deadline:
        current_ids = prompt_ids + response_token_ids
        sp2 = sp.copy()
        remaining_response_budget = cfg.judge_max_tokens - len(response_token_ids)
        sp2["max_new_tokens"] = min(
            sp2["max_new_tokens"],
            max_ctx - len(current_ids) - 1,
            remaining_response_budget,
        )
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

        if boxed_answer_valid_for_stop(response_text):
            status = Sample.Status.COMPLETED
            break

        if out["meta_info"]["finish_reason"]["type"] == "stop":
            status = Sample.Status.COMPLETED
            break
        if out["meta_info"]["finish_reason"]["type"] == "length":
            status = Sample.Status.TRUNCATED
            break

    if status == Sample.Status.PENDING:
        status = Sample.Status.COMPLETED if response_token_ids else Sample.Status.TRUNCATED

    extracted = extract_last_boxed_integer(response_text)

    return AttemptResult(
        prompt_ids=prompt_ids,
        response_token_ids=response_token_ids,
        response_text=response_text,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        extracted_answer=extracted,
        status=status,
        metadata={"round": "judge", "judge_idx": judge_idx},
    )


def _solver_proposal_set(attempts: list[AttemptResult]) -> set[str]:
    proposed: set[str] = set()
    for a in attempts:
        x = extract_last_notebook_solver_integer(a.response_text)
        if x is None:
            continue
        try:
            proposed.add(normalize_int_answer(x))
        except (ValueError, OverflowError):
            continue
    return proposed


def _placeholder_attempt(
    prompt_ids: list[int],
    attempt_idx: int,
    *,
    cancelled: bool = False,
) -> AttemptResult:
    meta = {"attempt_idx": attempt_idx, "turns": 0}
    if cancelled:
        meta["cancelled"] = True
    return AttemptResult(
        prompt_ids=prompt_ids,
        response_token_ids=[],
        response_text="",
        loss_mask=[],
        rollout_log_probs=[],
        extracted_answer=None,
        status=Sample.Status.TRUNCATED,
        metadata=meta,
    )


def _attempt_to_sample(
    template: Sample,
    att: AttemptResult,
    *,
    round_type: str,
    round_number: int,
    proposed_answers: set[str] | None = None,
) -> Sample:
    s = copy.deepcopy(template)
    s.tokens = att.prompt_ids + att.response_token_ids
    s.response_length = len(att.response_token_ids)
    s.response = att.response_text
    s.loss_mask = att.loss_mask
    s.rollout_log_probs = att.rollout_log_probs
    s.status = att.status
    s.metadata = dict(template.metadata or {})
    s.metadata["round_number"] = round_number
    s.metadata["round_type"] = round_type
    s.metadata.update(att.metadata)
    if round_type == "solver":
        s.reward = scalar_correctness_reward(att.response_text, template.label or "")
    else:
        s.reward = judge_selection_reward(
            att.response_text,
            template.label or "",
            proposed_answers or set(),
        )
    return s


async def process_group(args: Any, group: list[Sample]) -> list[Sample]:
    cfg = ScaffoldingCFG.from_env()
    expected = 2 * cfg.attempts
    assert len(group) == expected, (
        f"Group size {len(group)} must equal 2 * attempts = {expected}. "
        "Set --n-samples-per-prompt to 2 * SLIME_SCAFFOLDING_ATTEMPTS."
    )

    state = GenerateState(args)
    template = group[0]
    problem_text = template.prompt if isinstance(template.prompt, str) else str(template.prompt)
    problem_text = _augment_problem_text(problem_text)

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

    attempt_tasks = [
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
        for i in range(cfg.attempts)
    ]
    raw_attempts = await asyncio.gather(*attempt_tasks, return_exceptions=True)

    ref_prompt_ids = _encode_solver_prompt_ids(state.tokenizer, problem_text)
    attempts: list[AttemptResult] = []
    for i, res in enumerate(raw_attempts):
        if isinstance(res, BaseException):
            logger.exception("Solver attempt %s failed: %s", i, res)
            attempts.append(_placeholder_attempt(ref_prompt_ids, i, cancelled=True))
        else:
            attempts.append(res)

    proposed = _solver_proposal_set(attempts)

    judge_tasks = [
        asyncio.create_task(
            run_judge_round(
                args,
                state,
                cfg,
                problem_text,
                list(attempts),
                deadline,
                sampling_params,
                session_id,
                j,
            )
        )
        for j in range(cfg.attempts)
    ]
    raw_judges = await asyncio.gather(*judge_tasks, return_exceptions=True)

    judges: list[AttemptResult] = []
    judge_prompt_ids = _encode_judge_prompt_ids(state.tokenizer, problem_text, list(attempts))
    for j, res in enumerate(raw_judges):
        if isinstance(res, BaseException):
            logger.exception("Judge rollout %s failed: %s", j, res)
            judges.append(
                AttemptResult(
                    prompt_ids=judge_prompt_ids,
                    response_token_ids=[],
                    response_text="",
                    loss_mask=[],
                    rollout_log_probs=[],
                    extracted_answer=None,
                    status=Sample.Status.TRUNCATED,
                    metadata={"round": "judge", "judge_idx": j, "failed": True},
                )
            )
        else:
            judges.append(res)

    assert len(judges) == cfg.attempts, f"expected {cfg.attempts} judge results, got {len(judges)}"

    out_samples: list[Sample] = []
    for i in range(cfg.attempts):
        s = _attempt_to_sample(group[i], attempts[i], round_type="solver", round_number=1)
        s.index = group[i].index
        s.group_index = group[i].group_index
        out_samples.append(s)

    for j in range(cfg.attempts):
        sj = _attempt_to_sample(
            group[cfg.attempts + j],
            judges[j],
            round_type="judge",
            round_number=2,
            proposed_answers=proposed,
        )
        sj.index = group[cfg.attempts + j].index
        sj.group_index = group[cfg.attempts + j].group_index
        out_samples.append(sj)

    try:
        import wandb

        if wandb.run is not None:
            sol_rewards = [s.reward for s in out_samples[: cfg.attempts]]
            jud_rewards = [s.reward for s in out_samples[cfg.attempts :]]
            wandb.log(
                {
                    "scaffolding/problem_budget_s": budget,
                    "scaffolding/solver_mean_reward": sum(sol_rewards) / cfg.attempts,
                    "scaffolding/judge_mean_reward": sum(jud_rewards) / cfg.attempts,
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
                solver0 = group[0]
                judge0 = group[len(group) // 2]
                solver_summaries = [
                    {
                        "idx": idx,
                        "reward": sample.reward,
                        "status": str(sample.status),
                        "extracted": extract_last_boxed_integer(sample.response or ""),
                        "tool_calls": (sample.metadata or {}).get("tool_call_count"),
                        "tool_errors": (sample.metadata or {}).get("tool_error_count"),
                        "finish": (sample.metadata or {}).get("finish_reason"),
                    }
                    for idx, sample in enumerate(group[: len(group) // 2])
                ]
                judge_summaries = [
                    {
                        "idx": idx,
                        "reward": sample.reward,
                        "status": str(sample.status),
                        "extracted": extract_last_boxed_integer(sample.response or ""),
                    }
                    for idx, sample in enumerate(group[len(group) // 2 :])
                ]
                logger.info(
                    "First scaffolding group: reward=%s",
                    [s.reward for s in group[:3]],
                )
                logger.info("First scaffolding solver summaries: %s", solver_summaries)
                logger.info("First scaffolding judge summaries: %s", judge_summaries)
                logger.info(
                    "First solver sample: status=%s extracted=%s used_harmony=%s tool_calls=%s finish_reason=%s recipient=%s channel=%s response_preview=%s",
                    solver0.status,
                    extract_last_boxed_integer(solver0.response or ""),
                    (solver0.metadata or {}).get("used_harmony"),
                    (solver0.metadata or {}).get("tool_call_count"),
                    (solver0.metadata or {}).get("finish_reason"),
                    (solver0.metadata or {}).get("last_harmony_recipient"),
                    (solver0.metadata or {}).get("last_harmony_channel"),
                    _response_preview(solver0.response or ""),
                )
                if (solver0.metadata or {}).get("tool_call_previews"):
                    logger.info("First solver tool call previews: %s", (solver0.metadata or {}).get("tool_call_previews"))
                if (solver0.metadata or {}).get("tool_result_previews"):
                    logger.info(
                        "First solver tool result previews: %s", (solver0.metadata or {}).get("tool_result_previews")
                    )
                logger.info(
                    "First judge sample: status=%s extracted=%s response_preview=%s",
                    judge0.status,
                    extract_last_boxed_integer(judge0.response or ""),
                    _response_preview(judge0.response or ""),
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
    metrics |= _collect_scaffolding_metrics(data)
    metrics["rollout/scaffolding/ok"] = True
    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


def generate_rollout_gs(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    assert args.rollout_global_dataset
    if evaluation:
        return RolloutFnEvalOutput(data={})

    output, aborted = run(generate_rollout_gs_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted)
    return output
