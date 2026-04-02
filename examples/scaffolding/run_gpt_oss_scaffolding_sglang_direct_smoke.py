#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from examples.scaffolding.gs_config import ScaffoldingCFG
from examples.scaffolding.reward_gpt_oss_scaffolding import judge_selection_reward, scalar_correctness_reward
from examples.scaffolding.rollout_gpt_oss_scaffolding import (
    _augment_problem_text,
    _problem_budget_s,
    _response_preview,
    run_judge_round,
    run_one_attempt,
)
from examples.scaffolding.run_gpt_oss_scaffolding_rl import _resolve_hf_checkpoint_to_local_dir
from slime.backends.sglang_utils.sglang_engine import _compute_server_args, launch_server_process
from slime.utils.http_utils import find_available_port
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct SGLang-only GPT-OSS scaffolding smoke.")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--rollout-num-gpus", type=int, default=8)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=8)
    parser.add_argument("--rollout-max-context-len", type=int, default=16384)
    parser.add_argument("--rollout-max-response-len", type=int, default=8192)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    parser.add_argument("--sglang-pp-size", type=int, default=1)
    parser.add_argument("--sglang-dp-size", type=int, default=1)
    parser.add_argument("--sglang-ep-size", type=int, default=8)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.42)
    parser.add_argument("--sglang-disable-cuda-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge", action="store_true")
    return parser.parse_args()


def _load_jsonl_row(path: str, row_index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index={row_index} is out of range for {path}")


def _make_runtime_args(cli_args: argparse.Namespace, hf_checkpoint: str) -> SimpleNamespace:
    args = SimpleNamespace(
        hf_checkpoint=hf_checkpoint,
        seed=cli_args.seed,
        rollout_seed=cli_args.seed,
        rollout_num_gpus=cli_args.rollout_num_gpus,
        rollout_num_gpus_per_engine=cli_args.rollout_num_gpus_per_engine,
        num_gpus_per_node=cli_args.rollout_num_gpus,
        actor_num_gpus_per_node=0,
        actor_num_nodes=0,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
        use_critic=False,
        colocate=False,
        debug_rollout_only=True,
        offload_rollout=False,
        fp16=False,
        use_rollout_routing_replay=False,
        sglang_pp_size=cli_args.sglang_pp_size,
        sglang_dp_size=cli_args.sglang_dp_size,
        sglang_ep_size=cli_args.sglang_ep_size,
        sglang_server_concurrency=max(1, cli_args.attempts),
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        use_slime_router=False,
        ci_test=False,
        rollout_max_context_len=cli_args.rollout_max_context_len,
        rollout_max_response_len=cli_args.rollout_max_response_len,
        rollout_temperature=cli_args.rollout_temperature,
        rollout_top_p=cli_args.rollout_top_p,
        rollout_top_k=cli_args.rollout_top_k,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_mem_fraction_static=cli_args.sglang_mem_fraction_static,
        sglang_disable_cuda_graph=bool(cli_args.sglang_disable_cuda_graph),
        sglang_context_length=cli_args.rollout_max_context_len,
    )
    return args


def _start_direct_server(args: SimpleNamespace) -> tuple[Any, int]:
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "false")
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    os.environ.setdefault("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "true")
    os.environ.setdefault("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "true")
    os.environ.setdefault("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false")
    os.environ.setdefault("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", "false")
    os.environ.setdefault("SLIME_ENABLE_GLOBAL_SGLANG_PATCH", "1")
    os.environ.setdefault("SLIME_SGLANG_HEALTH_MODE", "basic")
    os.environ.setdefault("SLIME_SGLANG_SKIP_FLUSH_CACHE", "1")

    port = find_available_port(15000)
    nccl_port = find_available_port(21000)
    dist_init_addr = f"127.0.0.1:{find_available_port(22000)}"
    server_args_dict, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr=dist_init_addr,
        nccl_port=nccl_port,
        host="127.0.0.1",
        port=port,
        base_gpu_id=0,
    )
    process = launch_server_process(ServerArgs(**server_args_dict))
    args.sglang_router_ip = "127.0.0.1"
    args.sglang_router_port = port
    logger.info(
        "Direct SGLang server ready at http://127.0.0.1:%s (tp=%s ep=%s mem_fraction_static=%s)",
        port,
        server_args_dict.get("tp_size"),
        server_args_dict.get("ep_size"),
        server_args_dict.get("mem_fraction_static"),
    )
    return process, port


def _make_sampling_params(args: SimpleNamespace) -> dict[str, Any]:
    return {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


async def _run_direct_smoke(cli_args: argparse.Namespace) -> dict[str, Any]:
    hf_checkpoint = _resolve_hf_checkpoint_to_local_dir(cli_args.hf_checkpoint)
    runtime_args = _make_runtime_args(cli_args, hf_checkpoint)
    process = None
    try:
        process, _ = _start_direct_server(runtime_args)
        tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        state = SimpleNamespace(tokenizer=tokenizer)
        cfg = ScaffoldingCFG.from_env()

        row = _load_jsonl_row(cli_args.data_jsonl, cli_args.row_index)
        problem_text = _augment_problem_text(row["question"])
        budget = _problem_budget_s(cfg, cfg.problems_remaining_default, notebook_elapsed=0.0)
        deadline = time.time() + budget
        sampling_params = _make_sampling_params(runtime_args)

        logger.info(
            "Running direct smoke on row id=%s ground_truth=%s attempts=%s budget=%.1fs",
            row.get("id"),
            row.get("ground_truth"),
            cli_args.attempts,
            budget,
        )

        attempt_results = await asyncio.gather(
            *[
                run_one_attempt(
                    runtime_args,
                    state,
                    cfg,
                    problem_text,
                    attempt_idx=i,
                    deadline=deadline,
                    sampling_params=sampling_params.copy(),
                    session_id=row.get("id"),
                )
                for i in range(cli_args.attempts)
            ]
        )

        attempt_dicts: list[dict[str, Any]] = []
        for result in attempt_results:
            reward = scalar_correctness_reward(result.response_text, str(row["ground_truth"]))
            attempt_dicts.append(
                {
                    "status": result.status.value,
                    "extracted_answer": result.extracted_answer,
                    "reward": reward,
                    "response": result.response_text,
                    "response_preview": _response_preview(result.response_text, limit=400),
                    "metadata": result.metadata,
                }
            )

        judge_payload = None
        if cli_args.judge and len(attempt_results) >= 2:
            judge_result = await run_judge_round(
                runtime_args,
                state,
                cfg,
                problem_text,
                attempt_results,
                deadline=deadline,
                sampling_params=sampling_params.copy(),
                session_id=row.get("id"),
                judge_idx=0,
            )
            proposed = {r.extracted_answer for r in attempt_results if r.extracted_answer is not None}
            judge_reward = judge_selection_reward(
                judge_result.response_text, str(row["ground_truth"]), proposed
            )
            judge_payload = {
                "status": judge_result.status.value,
                "extracted_answer": judge_result.extracted_answer,
                "reward": judge_reward,
                "response": judge_result.response_text,
                "response_preview": _response_preview(judge_result.response_text, limit=400),
                "metadata": judge_result.metadata,
            }

        solver_tool_calls_total = sum(int((a.get("metadata") or {}).get("tool_call_count") or 0) for a in attempt_dicts)
        solver_successful_tool_calls_total = sum(
            max(
                0,
                int((a.get("metadata") or {}).get("tool_call_count") or 0)
                - int((a.get("metadata") or {}).get("tool_error_count") or 0),
            )
            for a in attempt_dicts
        )
        solver_positive_rewards = sum(1 for a in attempt_dicts if float(a.get("reward") or 0.0) > 0.0)
        solver_extracted = sum(1 for a in attempt_dicts if a.get("extracted_answer") is not None)

        payload = {
            "row": row,
            "hf_checkpoint": hf_checkpoint,
            "attempts": attempt_dicts,
            "judge": judge_payload,
            "summary": {
                "solver_tool_calls_total": solver_tool_calls_total,
                "solver_successful_tool_calls_total": solver_successful_tool_calls_total,
                "solver_positive_rewards": solver_positive_rewards,
                "solver_extracted_count": solver_extracted,
                "solver_attempt_count": len(attempt_dicts),
            },
        }
        return payload
    finally:
        if process is not None:
            kill_process_tree(process.pid)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
    )
    cli_args = _parse_args()
    payload = asyncio.run(_run_direct_smoke(cli_args))
    output_path = Path(cli_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    first = (payload.get("attempts") or [None])[0]
    if first:
        print(
            "[direct-smoke] first_attempt: "
            f"status={first.get('status')} "
            f"reward={first.get('reward')} "
            f"extracted={first.get('extracted_answer')} "
            f"tool_calls={(first.get('metadata') or {}).get('tool_call_count')} "
            f"tool_errors={(first.get('metadata') or {}).get('tool_error_count')} "
            f"recipient={(first.get('metadata') or {}).get('last_harmony_recipient')} "
            f"channel={(first.get('metadata') or {}).get('last_harmony_channel')}"
        )
        print(f"[direct-smoke] first_attempt_preview={first.get('response_preview')}")
    print(f"[direct-smoke] summary={payload.get('summary')}")
    print(f"[direct-smoke] wrote {output_path}")


if __name__ == "__main__":
    main()
