#!/usr/bin/env python3
"""
Single entry: AIMO filtered JSONL + GPT-OSS (20B or 120B) + dual-group scaffolding rollout + GRPO.

Environment:
  SLIME_SCRIPT_MODEL_SIZE: "20b" | "120b" (default: 20b)
  SLIME_SCRIPT_HF_CHECKPOINT: Hugging Face Hub model id (e.g. openai/gpt-oss-20b) or local weights dir (required)
  SLIME_SCRIPT_DATA_JSONL: path to train_data_filtered.jsonl (required)
  SLIME_SCRIPT_NUM_GPUS: default 16 (sbatch scripts set 2 or 4)
  SLIME_SCRIPT_TP / SLIME_SCRIPT_EP / SLIME_SCRIPT_ETP: Megatron parallel sizes (defaults 2 / 1 / 1; ETP must stay 1 for GPT-OSS MoE+bias)
  SLIME_SCRIPT_ROLLOUT_TP: SGLang tensor parallel width (default 2; match TP on small GPU counts)
  SLIME_SCRIPT_TRAIN_BACKEND: megatron | fsdp (default: megatron for GPT-OSS MoE)
  SLIME_SCRIPT_GLOBAL_BATCH_SIZE: non-smoke only; default 32 (set to e.g. rollout_batch_size * n_samples_per_prompt on few GPUs)
  SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN / SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN: non-smoke rollout limits (defaults 8192 / 65536)
  SLIME_SCRIPT_SGLANG_MEM_FRACTION: optional override for --sglang-mem-fraction-static (default 0.75)
  SLIME_SCRIPT_MAX_TOKENS_PER_GPU: optional override for --max-tokens-per-gpu (default 4096)
  SLIME_SCRIPT_LOG_PROBS_MAX_TOKENS_PER_GPU: optional override for --log-probs-max-tokens-per-gpu
  SLIME_SCRIPT_ATTENTION_BACKEND: Megatron attention backend for Transformer Engine (default: flash)
  WANDB_API_KEY: optional; prompted if training with wandb and missing

Example:
  export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
  export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py

Smoke test (few scaffolding attempts, one rollout step, one real train step):
  export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
  export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke

  # Reward / config checks only (no Ray / GPUs):
  python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke-rewards-only
"""

from __future__ import annotations

import argparse
import os
import shlex
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
    """Defaults for `--smoke` when env vars are unset (tiny end-to-end train run)."""
    defaults = {
        "SLIME_SCRIPT_MODEL_SIZE": "20b",
        "SLIME_SCRIPT_NUM_GPUS": "2",
        "SLIME_SCRIPT_NUM_ROLLOUT": "1",
        "SLIME_SCRIPT_ROLLOUT_BATCH_SIZE": "1",
        "SLIME_SCAFFOLDING_ATTEMPTS": "2",
        "SLIME_SCRIPT_ROLLOUT_TP": "2",
        "SLIME_SCRIPT_TP": "2",
        "SLIME_SCRIPT_EP": "1",
        "SLIME_SCRIPT_ETP": "1",
        "SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_RESPONSE_LEN": "1024",
        "SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_CONTEXT_LEN": "8192",
        "SLIME_SCRIPT_SMOKE_DEBUG_ROLLOUT_PATH": "/tmp/slime-debug-rollout/{rollout_id}.pt",
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
    assert scalar_correctness_reward("Thus the answer is \\boxed{ 42 }", "42") == 1.0
    assert scalar_correctness_reward("final answer is 42", "42") == 1.0
    assert scalar_correctness_reward("Thus the answer is \\boxed{43}", "42") == 0.0
    assert scalar_correctness_reward("no boxed answer here", "42") == 0.0
    assert scalar_correctness_reward("wrong \\boxed{1} then \\boxed{42}", "42") == 1.0
    assert scalar_correctness_reward("<|channel|>final<|message|>\\boxed{42.0}<|end|>", "42") == 0.0
    assert judge_selection_reward("\\boxed{42}", "42", {"42"}) == 1.0
    assert judge_selection_reward("**Judgment:** [42]", "42", {"42"}) == 1.0
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


def _resolve_hf_checkpoint_to_local_dir(hf_ckpt: str) -> str:
    """Megatron bridge loading needs a real directory; Hub ids must be cached locally first."""
    path = Path(hf_ckpt).expanduser()
    if path.is_dir():
        print(f"[launcher] Using explicit local checkpoint dir: {path.resolve()}")
        return str(path.resolve())

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repo_cache_dir = hf_home / "hub" / f"models--{hf_ckpt.strip().replace('/', '--')}"
    if repo_cache_dir.is_dir():
        refs_main = repo_cache_dir / "refs" / "main"
        candidate_snapshot: Path | None = None
        if refs_main.is_file():
            revision = refs_main.read_text().strip()
            if revision:
                snapshot_dir = repo_cache_dir / "snapshots" / revision
                if snapshot_dir.is_dir():
                    candidate_snapshot = snapshot_dir
        if candidate_snapshot is None:
            snapshots_dir = repo_cache_dir / "snapshots"
            snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.is_dir() else []
            if len(snapshot_dirs) == 1:
                candidate_snapshot = snapshot_dirs[0]
        if candidate_snapshot is not None:
            print(f"[launcher] Using cached HF snapshot dir: {candidate_snapshot.resolve()}")
            return str(candidate_snapshot.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for Hugging Face Hub checkpoint ids. "
            "Install it or set SLIME_SCRIPT_HF_CHECKPOINT to a local weights directory."
        ) from e
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        local_dir = snapshot_download(repo_id=hf_ckpt.strip(), token=token, local_files_only=True)
        print(f"[launcher] Using local_files_only HF snapshot dir: {local_dir}")
        return local_dir
    except Exception:
        pass
    print(f"[launcher] Cache miss for {hf_ckpt.strip()} under HF_HOME={hf_home}; downloading from Hub.")
    return snapshot_download(repo_id=hf_ckpt.strip(), token=token)


def _parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="GPT-OSS scaffolding RL launcher (see module docstring).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Set small defaults when unset, run reward/config checks, then execute one tiny "
            "end-to-end RL iteration (one rollout plus one real training step)."
        ),
    )
    p.add_argument(
        "--smoke-rewards-only",
        action="store_true",
        help="Only run reward + config consistency checks (no Ray / GPUs).",
    )
    p.add_argument(
        "--inference-only",
        action="store_true",
        help="Run rollout generation only via the SGLang rollout manager; skip Megatron training actors entirely.",
    )
    launcher_args, passthrough_args = p.parse_known_args()
    return launcher_args, passthrough_args


def main() -> None:
    launcher_args, passthrough_args = _parse_launcher_args()
    if launcher_args.smoke_rewards_only:
        _smoke_reward_checks()
        _smoke_config_consistency()
        return

    from slime.utils.external_utils.command_utils import execute_train

    if launcher_args.smoke:
        _apply_smoke_env_defaults()

    if launcher_args.inference_only:
        has_debug_rollout_only = any(
            arg == "--debug-rollout-only" or arg.startswith("--debug-rollout-only=") for arg in passthrough_args
        )
        if not has_debug_rollout_only:
            passthrough_args = [*passthrough_args, "--debug-rollout-only"]

    if not launcher_args.smoke:
        _prompt_wandb_if_needed()
    elif not os.environ.get("WANDB_API_KEY"):
        os.environ["SLIME_SCRIPT_SKIP_WANDB"] = "1"

    model_size = os.environ.get("SLIME_SCRIPT_MODEL_SIZE", "20b").lower().strip()
    if model_size not in ("20b", "120b"):
        raise ValueError("SLIME_SCRIPT_MODEL_SIZE must be 20b or 120b")

    hf_ckpt = os.environ.get("SLIME_SCRIPT_HF_CHECKPOINT", "").strip()
    if not hf_ckpt:
        raise ValueError(
            "Set SLIME_SCRIPT_HF_CHECKPOINT to a Hugging Face model id (e.g. openai/gpt-oss-20b) "
            "or a local model directory."
        )
    hf_ckpt = _resolve_hf_checkpoint_to_local_dir(hf_ckpt)

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
        wandb_project = os.environ.get("SLIME_SCRIPT_WANDB_PROJECT", "slime-aimo-scaffolding").strip()
        wandb_group = os.environ.get("SLIME_SCRIPT_WANDB_GROUP", "gpt-oss-scaffolding").strip()
        wandb_team = os.environ.get("SLIME_SCRIPT_WANDB_TEAM", "").strip()
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {shlex.quote(wandb_project)} "
            f"--wandb-group {shlex.quote(wandb_group)} "
            f"--wandb-key '{os.environ['WANDB_API_KEY']}' "
        )
        if wandb_team:
            wandb_args += f"--wandb-team {shlex.quote(wandb_team)} "

    ckpt_args = f"--hf-checkpoint {shlex.quote(hf_ckpt)} "

    num_rollout = os.environ.get("SLIME_SCRIPT_NUM_ROLLOUT", "64")
    rollout_bs = os.environ.get("SLIME_SCRIPT_ROLLOUT_BATCH_SIZE", "4")

    if launcher_args.smoke:
        rollout_max_resp = os.environ.get("SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_RESPONSE_LEN", "4096")
        rollout_max_ctx = os.environ.get("SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_CONTEXT_LEN", "16384")
        num_steps_per_rollout = "1"
        gbs = int(
            os.environ.get(
                "SLIME_SCRIPT_GLOBAL_BATCH_SIZE",
                str(int(rollout_bs) * n_sp // int(num_steps_per_rollout)),
            )
        )
        smoke_extra = f"--num-steps-per-rollout {num_steps_per_rollout} "
        has_debug_rollout_override = any(
            arg == "--load-debug-rollout-data"
            or arg.startswith("--load-debug-rollout-data=")
            or arg == "--save-debug-rollout-data"
            or arg.startswith("--save-debug-rollout-data=")
            for arg in passthrough_args
        )
        if not has_debug_rollout_override:
            smoke_debug_rollout_path = os.environ.get("SLIME_SCRIPT_SMOKE_DEBUG_ROLLOUT_PATH", "").strip()
            if smoke_debug_rollout_path:
                smoke_extra += f"--save-debug-rollout-data {shlex.quote(smoke_debug_rollout_path)} "
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
    passthrough_args_str = " ".join(shlex.quote(arg) for arg in passthrough_args)
    if passthrough_args_str:
        passthrough_args_str += " "

    lora_lr = os.environ.get("SLIME_SCRIPT_LORA_LR", "2e-4")
    enable_lora = os.environ.get("SLIME_SCRIPT_ENABLE_LORA", "1").strip().lower() not in {"0", "false", "no"}
    lora_args = ""
    if enable_lora:
        lora_args = (
            "--enable-lora "
            "--lora-r 8 "
            "--lora-alpha 32 "
            "--lora-dropout 0.0 "
            f"--lora-lr {lora_lr} "
            "--lora-target-policy mlp_moe_only "
        )

    optim_args = (
        "--optimizer adam "
        f"--lr {lora_lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_mem = os.environ.get("SLIME_SCRIPT_SGLANG_MEM_FRACTION", "0.75")
    max_tok = os.environ.get("SLIME_SCRIPT_MAX_TOKENS_PER_GPU", "4096")
    log_probs_max_tok = os.environ.get("SLIME_SCRIPT_LOG_PROBS_MAX_TOKENS_PER_GPU", "").strip()
    sglang_args = (
        f"--rollout-num-gpus-per-engine {os.environ.get('SLIME_SCRIPT_ROLLOUT_TP', '2')} "
        f"--sglang-ep-size {os.environ.get('SLIME_SCRIPT_SGLANG_EP_SIZE', '1')} "
        f"--sglang-mem-fraction-static {sglang_mem} "
    )
    dynamic_batch_args = (
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tok} "
    )
    if log_probs_max_tok:
        dynamic_batch_args += f"--log-probs-max-tokens-per-gpu {log_probs_max_tok} "

    # Colocated train+rollout on the same GPUs; tune TP/EP/ETP via env (see sbatch scripts).
    tp = os.environ.get("SLIME_SCRIPT_TP", "2")
    ep = os.environ.get("SLIME_SCRIPT_EP", "1")
    # MoE + bias: Megatron requires expert tensor parallel size 1 (see e.g. other run-*.sh MoE configs).
    etp = os.environ.get("SLIME_SCRIPT_ETP", "1")
    attention_backend = os.environ.get("SLIME_SCRIPT_ATTENTION_BACKEND", "flash").strip()
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
            f"{lora_args} "
            f"{grpo_args} "
            f"{sglang_args} "
            f"{wandb_args} "
            "--train-backend megatron "
            "--megatron-to-hf-mode bridge "
            f"--tensor-model-parallel-size {tp} "
            f"--expert-model-parallel-size {ep} "
            f"--expert-tensor-parallel-size {etp} "
            f"--attention-backend {attention_backend} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            f"{dynamic_batch_args} "
            f"{passthrough_args_str}"
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
            f"{passthrough_args_str}"
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
        mode_label = "inference-only tiny rollout" if launcher_args.inference_only else "end-to-end tiny RL run"
        print(
            f"[smoke] Launching {mode_label}: num_rollout={num_rollout}, "
            f"rollout_batch_size={rollout_bs}, n_samples_per_prompt={n_sp}, global_batch_size={gbs}, "
            f"gpus={num_gpus}, num_steps_per_rollout=1."
        )
        smoke_debug_rollout_path = os.environ.get("SLIME_SCRIPT_SMOKE_DEBUG_ROLLOUT_PATH", "").strip()
        if smoke_debug_rollout_path:
            print(f"[smoke] Debug rollout path template: {smoke_debug_rollout_path}")

    execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=meg_type,
        train_script=(
            "examples/scaffolding/run_gpt_oss_scaffolding_inference_only.py"
            if launcher_args.inference_only
            else "train.py"
        ),
        extra_env_vars=extra_env,
    )


if __name__ == "__main__":
    main()
