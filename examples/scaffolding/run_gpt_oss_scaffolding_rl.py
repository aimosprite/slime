#!/usr/bin/env python3
"""
Single entry: AIMO filtered JSONL + GPT-OSS (20B or 120B) + dual-group scaffolding rollout + GRPO.

Environment:
  SLIME_SCRIPT_MODEL_SIZE: "20b" | "120b" (default: 20b)
  SLIME_SCRIPT_HF_CHECKPOINT: path to HF weights (required)
  SLIME_SCRIPT_DATA_JSONL: path to train_data_filtered.jsonl (required)
  SLIME_SCRIPT_NUM_GPUS: default 16 (sbatch scripts set 2 or 4)
  SLIME_SCRIPT_TP / SLIME_SCRIPT_EP / SLIME_SCRIPT_ETP: Megatron parallel sizes (defaults 2 / 1 / 1; ETP must stay 1 for GPT-OSS MoE+bias)
  SLIME_SCRIPT_ROLLOUT_TP: SGLang tensor parallel width (default 2; match TP on small GPU counts)
  SLIME_SCRIPT_TRAIN_BACKEND: megatron | fsdp (default: megatron for GPT-OSS MoE)
  SLIME_SCRIPT_GLOBAL_BATCH_SIZE: non-smoke only; default 32 (set to e.g. rollout_batch_size * n_samples_per_prompt on few GPUs)
  SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN / SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN: non-smoke rollout limits (defaults 8192 / 65536)
  SLIME_SCRIPT_SGLANG_MEM_FRACTION: optional override for --sglang-mem-fraction-static (default 0.75)
  SLIME_SCRIPT_MAX_TOKENS_PER_GPU: optional override for --max-tokens-per-gpu (default 4096)
  WANDB_API_KEY: optional; prompted if training with wandb and missing

Example:
  export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
  export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py

Smoke test (20B only, few scaffolding attempts, one rollout step, --debug-rollout-only):
  export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
  export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke

  # Reward / config checks only (no Ray / GPUs):
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke-rewards-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _prompt_wandb_if_needed() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    if os.environ.get("SLIME_SCRIPT_SKIP_WANDB"):
        return
    try:
        key = input("WANDB_API_KEY not set. Enter WANDB API key (or press Enter to disable wandb): ").strip()
    except EOFError:
        key = ""
    if key:
        os.environ["WANDB_API_KEY"] = key
    else:
        os.environ["SLIME_SCRIPT_SKIP_WANDB"] = "1"


def _default_data_path() -> str:
    p = os.environ.get("SLIME_SCRIPT_DATA_JSONL")
    if p:
        return p
    # Repo inspect file from plan (optional)
    cand = _REPO_ROOT / ".hf-dataset-inspect" / "train_data_filtered.jsonl"
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError(
        "Set SLIME_SCRIPT_DATA_JSONL to train_data_filtered.jsonl "
        "(e.g. from aimosprite/aimo branch jason/data/)."
    )


def _apply_smoke_env_defaults() -> None:
    """Defaults for `--smoke` when env vars are unset (20B, small batch, few attempts)."""
    defaults = {
        "SLIME_SCRIPT_SKIP_WANDB": "1",
        "SLIME_SCRIPT_MODEL_SIZE": "20b",
        "SLIME_SCRIPT_NUM_GPUS": "2",
        "SLIME_SCRIPT_NUM_ROLLOUT": "1",
        "SLIME_SCRIPT_ROLLOUT_BATCH_SIZE": "1",
        "SLIME_SCAFFOLDING_ATTEMPTS": "2",
        "SLIME_SCRIPT_ROLLOUT_TP": "2",
        "SLIME_SCRIPT_TP": "2",
        "SLIME_SCRIPT_EP": "1",
        "SLIME_SCRIPT_ETP": "1",
    }
    for k, v in defaults.items():
        if not os.environ.get(k):
            os.environ[k] = v


def _smoke_reward_checks() -> None:
    """Validate solver/judge 0/1 rewards and shared boxed extraction."""
    from examples.scaffolding.reward_gpt_oss_scaffolding import (
        judge_selection_reward,
        scalar_correctness_reward,
    )

    assert scalar_correctness_reward("Thus the answer is \\boxed{42}", "42") == 1.0
    assert scalar_correctness_reward("Thus the answer is \\boxed{43}", "42") == 0.0
    assert scalar_correctness_reward("no boxed answer here", "42") == 0.0
    assert scalar_correctness_reward("wrong \\boxed{1} then \\boxed{42}", "42") == 1.0
    assert judge_selection_reward("\\boxed{42}", "42", {"42"}) == 1.0
    assert judge_selection_reward("\\boxed{42}", "42", {"43"}) == 0.0
    assert judge_selection_reward("\\boxed{43}", "42", {"42", "43"}) == 0.0
    print("[smoke] reward checks OK (solver + judge selection; shared boxed extraction).")


def _smoke_config_consistency() -> None:
    """Ensure launcher n_samples_per_prompt matches 2 * solver attempts (solvers + judges)."""
    from examples.scaffolding.gs_config import ScaffoldingCFG

    cfg = ScaffoldingCFG.from_env()
    attempts = int(os.environ.get("SLIME_SCAFFOLDING_ATTEMPTS", "8"))
    n_sp = 2 * attempts
    if cfg.attempts != attempts:
        raise AssertionError(
            f"SLIME_SCAFFOLDING_ATTEMPTS={attempts} but ScaffoldingCFG.attempts={cfg.attempts} "
            "(gs_config must read the same env)."
        )
    print(
        f"[smoke] config OK: attempts={attempts}, n_samples_per_prompt={n_sp} "
        f"(2 × attempts: solvers + judges)."
    )


def _parse_launcher_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-OSS scaffolding RL launcher (see module docstring).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "20B smoke: set small defaults, run reward/config checks, then train with "
            "--debug-rollout-only (one rollout step, batch size 1 prompt)."
        ),
    )
    p.add_argument(
        "--smoke-rewards-only",
        action="store_true",
        help="Only run reward + config consistency checks (no Ray / GPUs).",
    )
    return p.parse_args()


def main() -> None:
    launcher_args = _parse_launcher_args()
    if launcher_args.smoke_rewards_only:
        _smoke_reward_checks()
        _smoke_config_consistency()
        return

    from slime.utils.external_utils.command_utils import execute_train

    if launcher_args.smoke:
        _apply_smoke_env_defaults()
        os.environ.setdefault("SLIME_SCRIPT_SKIP_WANDB", "1")

    if not launcher_args.smoke:
        _prompt_wandb_if_needed()
    elif not os.environ.get("WANDB_API_KEY"):
        os.environ["SLIME_SCRIPT_SKIP_WANDB"] = "1"

    model_size = os.environ.get("SLIME_SCRIPT_MODEL_SIZE", "20b").lower().strip()
    if model_size not in ("20b", "120b"):
        raise ValueError("SLIME_SCRIPT_MODEL_SIZE must be 20b or 120b")

    if launcher_args.smoke and model_size != "20b":
        raise ValueError("--smoke is only supported with gpt-oss-20b (set SLIME_SCRIPT_MODEL_SIZE=20b).")

    hf_ckpt = os.environ.get("SLIME_SCRIPT_HF_CHECKPOINT", "").strip()
    if not hf_ckpt:
        raise ValueError("Set SLIME_SCRIPT_HF_CHECKPOINT to the local HuggingFace model directory.")

    data_path = _default_data_path()
    if not Path(data_path).is_file():
        raise FileNotFoundError(data_path)

    num_gpus = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "16"))
    backend = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "megatron").lower()
    if backend not in ("megatron", "fsdp"):
        raise ValueError("SLIME_SCRIPT_TRAIN_BACKEND must be megatron or fsdp")

    # attempts solvers + attempts judges (default 8 + 8 = 16)
    attempts = int(os.environ.get("SLIME_SCAFFOLDING_ATTEMPTS", "8"))
    n_sp = 2 * attempts

    megatron_model_type = "gpt-oss-20B" if model_size == "20b" else "gpt-oss-120B"

    wandb_args = ""
    if os.environ.get("WANDB_API_KEY") and not os.environ.get("SLIME_SCRIPT_SKIP_WANDB"):
        wandb_args = (
            "--use-wandb "
            "--wandb-project slime-aimo-scaffolding "
            "--wandb-group gpt-oss-scaffolding "
            f"--wandb-key '{os.environ['WANDB_API_KEY']}' "
        )

    ckpt_args = f"--hf-checkpoint {hf_ckpt} "

    num_rollout = os.environ.get("SLIME_SCRIPT_NUM_ROLLOUT", "64")
    rollout_bs = os.environ.get("SLIME_SCRIPT_ROLLOUT_BATCH_SIZE", "4")

    if launcher_args.smoke:
        rollout_max_resp = os.environ.get("SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_RESPONSE_LEN", "4096")
        rollout_max_ctx = os.environ.get("SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_CONTEXT_LEN", "16384")
        # Align GBS with rollout batching when num_steps_per_rollout is set.
        num_steps_per_rollout = "1"
        gbs = int(rollout_bs) * n_sp // int(num_steps_per_rollout)
        smoke_extra = (
            "--debug-rollout-only "
            f"--num-steps-per-rollout {num_steps_per_rollout} "
        )
    else:
        rollout_max_resp = os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN", "8192")
        rollout_max_ctx = os.environ.get("SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN", "65536")
        gbs = int(os.environ.get("SLIME_SCRIPT_GLOBAL_BATCH_SIZE", "32"))
        smoke_extra = ""

    rollout_args = (
        f"--prompt-data {data_path} "
        "--input-key question "
        "--label-key ground_truth "
        "--rollout-shuffle "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_bs} "
        f"--n-samples-per-prompt {n_sp} "
        f"--rollout-max-response-len {rollout_max_resp} "
        f"--rollout-max-context-len {rollout_max_ctx} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {gbs} "
        f"{smoke_extra}"
        "--advantage-estimator grpo "
        "--rollout-function-path examples.scaffolding.rollout_gpt_oss_scaffolding.generate_rollout_gs "
        "--custom-reward-post-process-path examples.scaffolding.grpo_dual_group_reward_postprocess.dual_group_grpo_reward_postprocess "
    )

    grpo_args = (
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optim_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_mem = os.environ.get("SLIME_SCRIPT_SGLANG_MEM_FRACTION", "0.75")
    max_tok = os.environ.get("SLIME_SCRIPT_MAX_TOKENS_PER_GPU", "4096")
    sglang_args = (
        f"--rollout-num-gpus-per-engine {os.environ.get('SLIME_SCRIPT_ROLLOUT_TP', '2')} "
        f"--sglang-mem-fraction-static {sglang_mem} "
    )

    # Colocated train+rollout on the same GPUs; tune TP/EP/ETP via env (see sbatch scripts).
    tp = os.environ.get("SLIME_SCRIPT_TP", "2")
    ep = os.environ.get("SLIME_SCRIPT_EP", "1")
    # MoE + bias: Megatron requires expert tensor parallel size 1 (see e.g. other run-*.sh MoE configs).
    etp = os.environ.get("SLIME_SCRIPT_ETP", "1")
    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {num_gpus} "
        f"--rollout-num-gpus {num_gpus} "
        "--colocate "
    )

    if backend == "megatron":
        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optim_args} "
            f"{grpo_args} "
            f"{sglang_args} "
            f"{wandb_args} "
            "--train-backend megatron "
            "--megatron-to-hf-mode bridge "
            f"--tensor-model-parallel-size {tp} "
            f"--expert-model-parallel-size {ep} "
            f"--expert-tensor-parallel-size {etp} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--use-dynamic-batch-size "
            f"--max-tokens-per-gpu {max_tok} "
            f"{misc_args} "
        )
        meg_type = megatron_model_type
    else:
        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optim_args} "
            f"{grpo_args} "
            f"{sglang_args} "
            f"{wandb_args} "
            "--train-backend fsdp "
            "--gradient-checkpointing "
            f"{misc_args} "
        )
        meg_type = None

    extra_env = {
        "PYTHONPATH": f"/root/Megatron-LM/:{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
    }
    if os.environ.get("WANDB_API_KEY"):
        extra_env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    if launcher_args.smoke:
        _smoke_reward_checks()
        _smoke_config_consistency()
        print(
            f"[smoke] Launching debug-rollout-only: num_rollout={num_rollout}, "
            f"rollout_batch_size={rollout_bs}, n_samples_per_prompt={n_sp}, global_batch_size={gbs}, "
            f"gpus={num_gpus}."
        )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=meg_type,
        extra_env_vars=extra_env,
    )


if __name__ == "__main__":
    main()
