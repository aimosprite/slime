"""Custom generate function + OPD reward/post-processing with multi-turn tool calls."""

import ast
import asyncio
import json
import logging
import re

import aiohttp
import torch
from transformers.tokenization_utils_base import BatchEncoding

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from examples.on_policy_distillation.tool_executor import StatefulPythonExecutor, get_semaphore, get_tool_spec

logger = logging.getLogger(__name__)

# Defaults — overridden by args when available
MAX_TURNS = 8
MAX_TOOL_CALLS = 4
TOOL_TIMEOUT = 30
TOOL_CONCURRENCY = 32

# "Thinking mode for general tasks" sampling defaults (from Qwen3.5 docs)
THINKING_GENERAL_SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

STUDENT_SYSTEM_PROMPT = (
    "You are an expert competition mathematician.\n"
    "You may call the `python` tool for exact computation.\n"
    "Do not hallucinate tool outputs; always wait for tool results.\n"
    "After finishing computation, provide only the final answer in \\boxed{...}."
)

CONCISE_TEACHER_SYSTEM_PROMPT = (
    "You are an expert competition mathematician.\n"
    "Solve the problem with concise reasoning.\n"
    "Avoid repeated prose and unnecessary explanation.\n"
    "If computation helps, you may call the `python` tool.\n"
    "Do not hallucinate tool outputs; always wait for tool results.\n"
    "After finishing computation, provide only the final answer in \\boxed{...}."
)


# ---------------------------------------------------------------------------
# Helpers: tool call parsing & token computation
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_CALL_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)

_teacher_rm_semaphore: asyncio.Semaphore | None = None


def _normalize_token_ids(token_ids) -> list[int]:
    """Coerce tokenizer outputs to a flat list of token ids.

    Newer Transformers builds may return a ``BatchEncoding`` from
    ``apply_chat_template`` where older versions returned a plain list[int].
    """
    if isinstance(token_ids, BatchEncoding):
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def _get_teacher_rm_semaphore(concurrency: int = 8) -> asyncio.Semaphore:
    global _teacher_rm_semaphore
    if _teacher_rm_semaphore is None:
        _teacher_rm_semaphore = asyncio.Semaphore(concurrency)
    return _teacher_rm_semaphore


def _extract_question(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, str):
        return prompt
    for message in reversed(prompt):
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("Unable to recover the original user question from the sample prompt.")


def _build_teacher_messages(question: str, style: str = "concise") -> list[dict[str, str]]:
    if style != "concise":
        raise ValueError(f"Unsupported teacher prompt style: {style}")
    return [
        {"role": "system", "content": CONCISE_TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _get_arguments_config(func_name: str, tools: list[dict] | None) -> dict:
    if tools is None:
        return {}
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        if function.get("name") != func_name:
            continue
        params = function.get("parameters", {})
        if isinstance(params, dict) and "properties" in params:
            return params["properties"]
        if isinstance(params, dict):
            return params
    logger.warning("Tool %r is not defined in the tools list.", func_name)
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict, func_name: str):
    """Adapted from SGLang's qwen3_coder detector."""
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value

    config = param_config[param_name]
    param_type = str(config.get("type", "string")).strip().lower() if isinstance(config, dict) else "string"
    if param_type in {"string", "str", "text", "varchar", "char", "enum"}:
        return param_value
    if any(param_type.startswith(prefix) for prefix in ("int", "uint", "long", "short", "unsigned")):
        try:
            return int(param_value)
        except Exception:
            logger.warning("Failed to parse integer parameter %r for tool %r.", param_name, func_name)
            return param_value
    if param_type.startswith("num") or param_type.startswith("float"):
        try:
            maybe_convert = "." not in param_value and "e" not in param_value.lower()
            value = float(param_value)
            return int(value) if maybe_convert and value.is_integer() else value
        except Exception:
            logger.warning("Failed to parse float parameter %r for tool %r.", param_name, func_name)
            return param_value
    if param_type in {"boolean", "bool", "binary"}:
        return param_value.lower() == "true"
    if param_type in {"object", "array", "arr"} or param_type.startswith("dict") or param_type.startswith("list"):
        try:
            return json.loads(param_value)
        except Exception:
            try:
                return ast.literal_eval(param_value)
            except Exception:
                logger.warning("Failed to parse structured parameter %r for tool %r.", param_name, func_name)
                return param_value
    try:
        return ast.literal_eval(param_value)
    except Exception:
        return param_value


def parse_tool_call(text: str, tools: list[dict] | None = None) -> tuple[str, str] | None:
    """Parse the SGLang `qwen3_coder` tool-call format from model output."""
    if "<tool_call>" not in text:
        return None

    raw_tool_calls = _TOOL_CALL_RE.findall(text)
    if not raw_tool_calls and "<function=" in text:
        raw_tool_calls = [text]

    for tool_content in raw_tool_calls:
        stripped = tool_content.strip()
        if stripped.startswith("{"):
            try:
                tool_call = json.loads(stripped)
            except json.JSONDecodeError:
                tool_call = None
            if isinstance(tool_call, dict):
                func_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                if func_name == "python":
                    code = arguments.get("code", "")
                    if isinstance(code, str) and code.strip():
                        return func_name, code

        funcs = _TOOL_CALL_FUNCTION_RE.findall(tool_content)
        for func_match in funcs:
            func_body = func_match[0] or func_match[1]
            if ">" not in func_body:
                continue
            name_end = func_body.index(">")
            func_name = func_body[:name_end]
            params_str = func_body[name_end + 1 :]

            param_config = _get_arguments_config(func_name, tools)
            parsed_params = {}
            for p_match in _TOOL_CALL_PARAMETER_RE.findall(params_str):
                if ">" not in p_match:
                    continue
                p_idx = p_match.index(">")
                p_name = p_match[:p_idx]
                p_val = p_match[p_idx + 1 :]
                if p_val.startswith("\n"):
                    p_val = p_val[1:]
                if p_val.endswith("\n"):
                    p_val = p_val[:-1]
                parsed_params[p_name] = _convert_param_value(p_val, p_name, param_config, func_name)

            if func_name == "python":
                code = parsed_params.get("code", "")
                if isinstance(code, str) and code.strip():
                    return func_name, code

    return None


def get_tool_response_and_gen_prompt_tokens(tokenizer, tool_name: str, content: str) -> list[int]:
    """Compute exact token IDs for a tool-response message + the next assistant generation prompt.

    Uses the prefix-trick: encode [prefix, tool_msg] with add_generation_prompt and
    subtract prefix tokens.
    """
    prefix_msg = {"role": "user", "content": "FOR CALCULATING TOKENS ONLY"}
    tool_msg = {"role": "tool", "name": tool_name, "content": content}

    prefix_ids = _normalize_token_ids(tokenizer.apply_chat_template([prefix_msg], tokenize=True))
    full_ids = _normalize_token_ids(tokenizer.apply_chat_template(
        [prefix_msg, tool_msg], tokenize=True, add_generation_prompt=True
    ))
    return full_ids[len(prefix_ids):]


# ---------------------------------------------------------------------------
# Custom generate function
# ---------------------------------------------------------------------------

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn tool-calling generate function for OPD.

    Produces ``sample.tokens``, ``sample.response_length``, ``sample.loss_mask``,
    and ``sample.rollout_log_probs`` suitable for OPD training.
    """
    max_turns = getattr(args, "tool_max_turns", MAX_TURNS)
    max_tool_calls = getattr(args, "tool_max_tool_calls", MAX_TOOL_CALLS)
    timeout = getattr(args, "tool_timeout", TOOL_TIMEOUT)
    concurrency = getattr(args, "tool_concurrency", TOOL_CONCURRENCY)

    # Initialise global semaphore once (idempotent after first call)
    get_semaphore(concurrency)

    # Apply "Thinking mode for general tasks" sampling defaults
    sampling_params = sampling_params.copy()
    for key, default_val in THINKING_GENERAL_SAMPLING_DEFAULTS.items():
        sampling_params.setdefault(key, default_val)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_spec = get_tool_spec()

    # ── Build initial prompt ──────────────────────────────────────────
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": sample.prompt},
    ]
    prompt_token_ids: list[int] = _normalize_token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, tools=[tool_spec], add_generation_prompt=True,
        enable_thinking=True,
    ))

    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    rollout_log_probs: list[float] = []
    tool_call_count = 0
    executor = StatefulPythonExecutor(timeout=timeout)

    try:
        for turn in range(max_turns):
            # Check context length budget
            total_len = len(prompt_token_ids) + len(response_token_ids)
            if args.rollout_max_context_len is not None:
                max_ctx = args.rollout_max_context_len
            else:
                max_ctx = args.context_parallel_size * args.max_tokens_per_gpu
            if total_len >= max_ctx:
                sample.status = Sample.Status.TRUNCATED
                break

            # ── Call sglang ───────────────────────────────────────────
            payload = {
                "input_ids": prompt_token_ids + response_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            # Log debug info to wandb
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "debug/total_tokens": total_len,
                            "debug/response_tokens": len(response_token_ids),
                            "debug/tool_calls": tool_call_count,
                            "debug/turn": turn,
                        }
                    )
            except ImportError:
                pass

            output = await post(url, payload)

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            # Extract tokens + log probs
            if "output_token_logprobs" in output["meta_info"]:
                cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                cur_text = output["text"]
                cur_token_ids = tokenizer(cur_text, add_special_tokens=False)["input_ids"]
                cur_log_probs = [0.0] * len(cur_token_ids)

            response_token_ids += cur_token_ids
            loss_masks += [1] * len(cur_token_ids)
            rollout_log_probs += cur_log_probs

            # ── Parse for tool call ───────────────────────────────────
            cur_text = tokenizer.decode(cur_token_ids)
            parsed = parse_tool_call(cur_text, tools=[tool_spec])

            if parsed is None:
                # No tool call → final answer; set status from finish reason
                fr = output["meta_info"]["finish_reason"]["type"]
                if fr == "length":
                    sample.status = Sample.Status.TRUNCATED
                else:
                    sample.status = Sample.Status.COMPLETED
                break

            # ── Execute tool ──────────────────────────────────────────
            tool_name, code = parsed
            tool_call_count += 1
            result = await executor.execute(code)
            # Truncate very long outputs to keep context manageable
            if len(result) > 3000:
                result = result[:3000] + "\n...[output truncated]"

            # ── Compute tool-response tokens (Qwen3 native format) ────
            obs_token_ids = get_tool_response_and_gen_prompt_tokens(tokenizer, tool_name, result)
            response_token_ids += obs_token_ids
            loss_masks += [0] * len(obs_token_ids)
            rollout_log_probs += [0.0] * len(obs_token_ids)

            assert len(response_token_ids) == len(rollout_log_probs), (
                f"Token/logp length mismatch at turn {turn}: "
                f"{len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"
            )
            assert len(response_token_ids) == len(loss_masks), (
                f"Token/mask length mismatch at turn {turn}: "
                f"{len(response_token_ids)} tokens vs {len(loss_masks)} masks"
            )

            if tool_call_count >= max_tool_calls:
                break

            # Check length after tool response
            if output["meta_info"]["finish_reason"]["type"] == "length":
                sample.status = Sample.Status.TRUNCATED
                break
        else:
            # Exhausted max_turns
            sample.status = Sample.Status.TRUNCATED

    finally:
        executor.close()

    # ── Populate sample ───────────────────────────────────────────────
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = tokenizer.decode(response_token_ids)
    sample.loss_mask = loss_masks
    sample.rollout_log_probs = rollout_log_probs

    # Set status if not already set
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    return sample


# ---------------------------------------------------------------------------
# reward_func — identical to on_policy_distillation.py
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    """Call teacher sglang server to get token-level log-probs under a concise teacher prompt."""
    state = GenerateState(args)
    tokenizer = state.tokenizer
    tool_spec = get_tool_spec()
    question = _extract_question(sample.prompt)
    teacher_messages = _build_teacher_messages(question, style=getattr(args, "teacher_prompt_style", "concise"))
    teacher_prompt_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=True,
            tools=[tool_spec],
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    response_ids = sample.tokens[-sample.response_length :] if sample.response_length > 0 else []
    payload = {
        "input_ids": teacher_prompt_ids + response_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    rm_concurrency = getattr(args, "teacher_request_concurrency", 8)
    async with _get_teacher_rm_semaphore(rm_concurrency):
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()


# ---------------------------------------------------------------------------
# post_process_rewards — identical to on_policy_distillation.py
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs and store on samples for OPD KL penalty.

    Returns scalar rewards (0.0 for pure distillation).
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
