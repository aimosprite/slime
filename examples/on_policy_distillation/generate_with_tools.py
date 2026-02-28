"""Custom generate function + OPD reward/post-processing with multi-turn tool calls.

Integrates multi-turn tool calls (Qwen3 native format) with OPD teacher scoring.
Wired in via ``--custom-generate-function-path``.
"""

import json
import logging
import re

import aiohttp
import torch

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

SYSTEM_PROMPT = (
    "You are a helpful assistant that can use a Python code interpreter to solve problems. "
    "When you need to perform calculations or verify results, call the code_interpreter tool. "
    "The interpreter persists state across calls within a conversation."
)


# ---------------------------------------------------------------------------
# Helpers: tool call parsing & token computation
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Parse Qwen3-native ``<tool_call>`` block. Returns (tool_name, code) or None."""
    m = _TOOL_CALL_RE.search(text)
    if m is None:
        return None
    try:
        raw = m.group(1)
        # The model sometimes emits newlines inside JSON values; escape them.
        raw = raw.replace("\n", "\\n")
        obj = json.loads(raw)
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if name == "code_interpreter":
            code = args.get("code", "")
            if code.strip():
                return name, code
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    return None


def get_tool_response_and_gen_prompt_tokens(tokenizer, tool_name: str, content: str) -> list[int]:
    """Compute exact token IDs for a tool-response message + the next assistant generation prompt.

    Uses the prefix-trick: encode [prefix, tool_msg] with add_generation_prompt and
    subtract prefix tokens.
    """
    prefix_msg = {"role": "user", "content": "FOR CALCULATING TOKENS ONLY"}
    tool_msg = {"role": "tool", "name": tool_name, "content": content}

    prefix_ids = tokenizer.apply_chat_template([prefix_msg], tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        [prefix_msg, tool_msg], tokenize=True, add_generation_prompt=True
    )
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

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_spec = get_tool_spec()

    # ── Build initial prompt ──────────────────────────────────────────
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample.prompt},
    ]
    prompt_token_ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, tools=[tool_spec], add_generation_prompt=True
    )

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
            parsed = parse_tool_call(cur_text)

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
    """Call teacher sglang server to get token-level log-probs on the student's rollout."""
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
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
