#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.scaffolding.reward_gpt_oss_scaffolding import scalar_correctness_reward
from examples.scaffolding.run_gpt_oss_scaffolding_rl import _resolve_hf_checkpoint_to_local_dir
from slime.ray.placement_group import allocate_train_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    print(f"[rl-topology-repro] {message}", flush=True)


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real RL-topology 20B repro on a single GPU.")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument(
        "--data-jsonl",
        default="examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--rollout-max-context-len", type=int, default=4096)
    parser.add_argument("--rollout-max-response-len", type=int, default=2048)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.60)
    parser.add_argument("--enable-lora", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_model_args_from_script(model_type: str) -> list[str]:
    script_path = _REPO_ROOT / "scripts" / "models" / f"{model_type}.sh"
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    import subprocess

    cmd = f'source "{script_path}" && printf "%s\\0" "${{MODEL_ARGS[@]}}"'
    output = subprocess.check_output(["bash", "-lc", cmd], cwd=str(_REPO_ROOT), env=os.environ)
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def _load_jsonl_row(path: str, row_index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index={row_index} is out of range for {path}")


def _make_repro_cli_args(
    *,
    hf_checkpoint: str,
    data_jsonl: str,
    rollout_max_context_len: int,
    rollout_max_response_len: int,
    sglang_mem_fraction_static: float,
    enable_lora: bool,
) -> list[str]:
    cli_args = [
        *_load_model_args_from_script("gpt-oss-20B"),
        "--train-backend",
        "megatron",
        "--hf-checkpoint",
        hf_checkpoint,
        "--prompt-data",
        data_jsonl,
        "--input-key",
        "question",
        "--label-key",
        "ground_truth",
        "--num-rollout",
        "2",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "2",
        "--global-batch-size",
        "2",
        "--advantage-estimator",
        "grpo",
        "--rollout-function-path",
        "examples.scaffolding.rollout_gpt_oss_scaffolding.generate_rollout_gs",
        "--custom-reward-post-process-path",
        "examples.scaffolding.grpo_dual_group_reward_postprocess.dual_group_grpo_reward_postprocess",
        "--rollout-max-context-len",
        str(rollout_max_context_len),
        "--rollout-max-response-len",
        str(rollout_max_response_len),
        "--rollout-temperature",
        "1.0",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--num-gpus-per-node",
        "1",
        "--colocate",
        "--sglang-ep-size",
        "1",
        "--sglang-mem-fraction-static",
        str(sglang_mem_fraction_static),
        "--max-tokens-per-gpu",
        "1024",
        "--log-probs-max-tokens-per-gpu",
        "1024",
        "--megatron-to-hf-mode",
        "bridge",
        "--tensor-model-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--sglang-disable-cuda-graph",
        "--save-debug-rollout-data",
        "/root/data/debug_rollout/rl_topology_repro_{rollout_id}.pt",
    ]
    if enable_lora:
        cli_args.extend(
            [
                "--enable-lora",
                "--lora-r",
                "8",
                "--lora-alpha",
                "32",
                "--lora-dropout",
                "0.0",
                "--lora-lr",
                "2e-4",
                "--lora-target-policy",
                "mlp_moe_only",
            ]
        )
    return cli_args


def _parse_rl_args(cli_args: list[str]):
    old_argv = sys.argv
    sys.argv = ["rl-topology-repro", *cli_args]
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    args.train_env_vars = dict(getattr(args, "train_env_vars", {}) or {})
    return args


def _extract_rollout_summary(path: Path, ground_truth: str) -> dict[str, Any]:
    payload = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(sample) for sample in payload.get("samples", [])]
    solvers = [sample for sample in samples if (sample.metadata or {}).get("round_type") == "solver"]
    judges = [sample for sample in samples if (sample.metadata or {}).get("round_type") == "judge"]

    def _summarize_sample(sample: Sample | None) -> dict[str, Any] | None:
        if sample is None:
            return None
        return {
            "status": sample.status.value,
            "reward": sample.reward,
            "response_preview": (sample.response or "")[:400],
            "metadata": sample.metadata,
            "scalar_correctness_reward": scalar_correctness_reward(sample.response or "", ground_truth),
        }

    return {
        "path": str(path),
        "num_samples": len(samples),
        "solver_count": len(solvers),
        "judge_count": len(judges),
        "solver": _summarize_sample(solvers[0] if solvers else None),
        "judge": _summarize_sample(judges[0] if judges else None),
    }


def _wait_for_rollout_engines_healthy(rollout_manager, timeout_s: float = 1800.0) -> None:
    started = time.time()
    engines, _lock, _num_new = ray.get(rollout_manager.get_rollout_engines_and_lock.remote())
    if not engines:
        raise RuntimeError("No rollout engines were created")
    while True:
        statuses = []
        all_ready = True
        for engine in engines:
            try:
                version = ray.get(engine.get_weight_version.remote(), timeout=10.0)
                statuses.append(version)
            except Exception as exc:
                statuses.append(f"not-ready:{type(exc).__name__}")
                all_ready = False
        if all_ready:
            _progress(f"Rollout engines ready weight_versions={statuses}")
            return
        if time.time() - started > timeout_s:
            raise TimeoutError(f"Timed out waiting for rollout engines to become healthy: {statuses}")
        _progress(f"Waiting for rollout engines ready: {statuses}")
        time.sleep(5.0)


def main() -> None:
    faulthandler.dump_traceback_later(300, repeat=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cli = _parse_cli()
    configure_logger()
    os.environ["SLIME_SCAFFOLDING_ATTEMPTS"] = "1"

    hf_checkpoint = _resolve_hf_checkpoint_to_local_dir(cli.hf_checkpoint)
    row = _load_jsonl_row(cli.data_jsonl, cli.row_index)
    ground_truth = str(row["ground_truth"])

    repro_cli_args = _make_repro_cli_args(
        hf_checkpoint=hf_checkpoint,
        data_jsonl=cli.data_jsonl,
        rollout_max_context_len=cli.rollout_max_context_len,
        rollout_max_response_len=cli.rollout_max_response_len,
        sglang_mem_fraction_static=cli.sglang_mem_fraction_static,
        enable_lora=cli.enable_lora,
    )
    args = _parse_rl_args(repro_cli_args)

    if not ray.is_initialized():
        _progress("Initializing Ray")
        ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True, include_dashboard=False)

    pgs = None
    rollout_manager = None
    actor_model = None
    try:
        _progress("Creating placement groups")
        pgs = create_placement_groups(args)
        _progress("Creating rollout manager")
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
        _wait_for_rollout_engines_healthy(rollout_manager)

        baseline_path = Path(args.save_debug_rollout_data.format(rollout_id=0))
        post_sync_path = Path(args.save_debug_rollout_data.format(rollout_id=1))
        for path in (baseline_path, post_sync_path):
            if path.exists():
                path.unlink()

        _progress("Running baseline rollout via real rollout manager")
        ray.get(rollout_manager.generate.remote(0))
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline rollout artifact missing: {baseline_path}")

        _progress("Creating real Megatron actor group")
        actor_model = allocate_train_group(
            args=args,
            num_nodes=args.actor_num_nodes,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            pg=pgs["actor"],
        )
        _progress("Initializing Megatron actor group")
        start_rollout_ids = ray.get(actor_model.async_init(args, role="actor", with_ref=False, with_opd_teacher=False))
        _progress(f"Megatron actor init complete start_rollout_ids={start_rollout_ids}")
        actor_model.set_rollout_manager(rollout_manager)

        _progress("Updating rollout weights through normal RL path")
        actor_model.update_weights()
        _progress("Weight update complete")

        _progress("Running post-sync rollout via real rollout manager")
        ray.get(rollout_manager.generate.remote(1))
        if not post_sync_path.is_file():
            raise FileNotFoundError(f"Post-sync rollout artifact missing: {post_sync_path}")

        payload = {
            "row": row,
            "hf_checkpoint": hf_checkpoint,
            "enable_lora": cli.enable_lora,
            "baseline": _extract_rollout_summary(baseline_path, ground_truth),
            "post_sync": _extract_rollout_summary(post_sync_path, ground_truth),
        }
        output_path = Path(cli.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path)}, indent=2))
    finally:
        if rollout_manager is not None:
            try:
                ray.get(rollout_manager.dispose.remote())
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
