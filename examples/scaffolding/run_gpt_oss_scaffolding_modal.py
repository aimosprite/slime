#!/usr/bin/env python3
"""
Single Modal launcher for GPT-OSS scaffolding LoRA RL.

Default targets:
  - gpt-oss-20b  -> 2x H200
  - gpt-oss-120b -> 8x H200

This script delegates training to:
  examples/scaffolding/run_gpt_oss_scaffolding_rl.py
and only sets sensible model/topology defaults for Modal.

Typical usage (weights load from Hugging Face Hub inside Modal; no local upload):
  modal run examples/scaffolding/run_gpt_oss_scaffolding_modal.py \
    --model-size 20b \
    --data-jsonl /root/data/train_data_filtered.jsonl

  # Optional: override hub id or use a local volume path
  modal run ... --hf-checkpoint openai/gpt-oss-20b

Infra (fixed in this file so local + remote imports match — Modal requirement):
  Volume "slime-data" -> /root/data (HF cache under /root/data/hf-cache)
  Secret "slime-training-secrets" -> HF_TOKEN, WANDB_API_KEY, etc.
  To use other names, edit DEFAULT_MODAL_VOLUME / DEFAULT_MODAL_SECRET below.

Container image: slimerl/slime:latest (matches docker/Dockerfile stack). Override _SLIME_IMAGE below
  to pin a tag/digest if you need reproducibility.
"""

import os
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import modal

# Default --hf-checkpoint when omitted: pull weights from Hub inside the worker (no Mac-side download).
DEFAULT_HF_CHECKPOINT_20B = "openai/gpt-oss-20b"
DEFAULT_HF_CHECKPOINT_120B = "openai/gpt-oss-120b"

APP_NAME = "slime-gpt-oss-scaffolding-rl"


def _resolve_repo_root() -> Path:
    """Repo root for image build (local) and for imports inside Modal (worker copies this file to /root/)."""
    here = Path(__file__).resolve()
    # Local: .../slime/examples/scaffolding/<this file>.py
    try:
        cand = here.parents[2]
        if (cand / "setup.py").is_file():
            return cand
    except IndexError:
        pass
    # Remote: module is /root/run_gpt_oss_scaffolding_modal.py; slime is copied to /root/slime
    modal_root = Path("/root/slime")
    if (modal_root / "setup.py").is_file():
        return modal_root
    raise RuntimeError(f"Cannot resolve slime repo root (from {here})")


REPO_ROOT = _resolve_repo_root()

# Do not gate these on os.environ: the worker imports this module without your shell env, which
# would change the dependency graph vs `modal run` locally and triggers ExecutionError.
DEFAULT_MODAL_VOLUME = "slime-data"
DEFAULT_MODAL_SECRET = "slime-training-secrets"

VOLUMES: Dict[str, modal.Volume] = {
    "/root/data": modal.Volume.from_name(DEFAULT_MODAL_VOLUME, create_if_missing=True)
}
SECRETS = [modal.Secret.from_name(DEFAULT_MODAL_SECRET)]

app = modal.App(APP_NAME)

# Use the same prebuilt stack as local Docker (SGLang + patched Megatron, etc.). Plain PyTorch
# images + requirements.txt only pull sglang-router, not `sglang`, so train.py fails at import.
# Overlay this checkout on /root/slime and refresh the editable install without re-resolving deps.
_SLIME_IMAGE = "slimerl/slime:latest"

image = (
    modal.Image.from_registry(_SLIME_IMAGE)
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/slime",
        copy=True,
        ignore=[".git", ".venv", ".venv-modal", "__pycache__", ".mypy_cache", ".pytest_cache"],
    )
    .run_commands("pip install openai_harmony")
    .run_commands("pip install 'numpy<2' 'ortools==9.14.6206'")
    .run_commands("pip install -e /root/slime --no-deps")
)


@dataclass(frozen=True)
class ModelPreset:
    model_size: str
    num_gpus: int
    tp: int
    ep: int
    sglang_ep_size: int
    rollout_tp: int
    sglang_mem_fraction: float
    max_tokens_per_gpu: int
    log_probs_max_tokens_per_gpu: int
    rollout_batch_size: int
    rollout_max_context_len: int
    moe_token_dispatcher_type: str = "alltoall"
    moe_flex_dispatcher_backend: str = ""
    enable_moe_deepep: bool = False
    disable_moe_permute_fusion: bool = False
    disable_moe_grouped_gemm: bool = False
    force_safe_varlen_attn: bool = False
    safe_varlen_attn_block_size: int = 0
    smoke_global_batch_size: int = 4


PRESETS: Dict[str, ModelPreset] = {
    # 20B full run on Modal: use the same 4xH200 / TP=4 shape the repo
    # documents for colocated 16k runs when 2xH200 OOMs.
    "20b": ModelPreset(
        model_size="20b",
        num_gpus=4,
        tp=4,
        ep=4,
        sglang_ep_size=1,
        rollout_tp=2,
        sglang_mem_fraction=0.38,
        max_tokens_per_gpu=1536,
        log_probs_max_tokens_per_gpu=2048,
        rollout_batch_size=1,
        rollout_max_context_len=16384,
        moe_token_dispatcher_type="alltoall",
        disable_moe_grouped_gemm=True,
        force_safe_varlen_attn=True,
        safe_varlen_attn_block_size=128,
    ),
    # Memory-first for 120B MoE: shard experts aggressively with EP=8.
    # Keep TP=1 to avoid extra dense-layer TP collectives.
    "120b": ModelPreset(
        model_size="120b",
        num_gpus=8,
        tp=1,
        ep=8,
        sglang_ep_size=8,
        rollout_tp=8,
        sglang_mem_fraction=0.42,
        max_tokens_per_gpu=2048,
        log_probs_max_tokens_per_gpu=2048,
        rollout_batch_size=1,
        rollout_max_context_len=16384,
        force_safe_varlen_attn=True,
        safe_varlen_attn_block_size=128,
        smoke_global_batch_size=8,
    ),
}

PRESET_20B_SINGLE_GPU_REPRO = ModelPreset(
    model_size="20b",
    num_gpus=1,
    tp=1,
    ep=1,
    sglang_ep_size=1,
    rollout_tp=1,
    sglang_mem_fraction=0.60,
    max_tokens_per_gpu=1024,
    log_probs_max_tokens_per_gpu=1024,
    rollout_batch_size=1,
    rollout_max_context_len=4096,
    force_safe_varlen_attn=True,
    safe_varlen_attn_block_size=128,
    smoke_global_batch_size=2,
)


def _prepare_smoke_data_jsonl(source_path: str, target_path: str, *, max_lines: int = 1) -> str:
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(source_path)

    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept_lines: list[str] = []
    with src.open("r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                kept_lines.append(line)
            if len(kept_lines) >= max_lines:
                break

    if not kept_lines:
        raise ValueError(f"No non-empty JSONL rows found in {source_path}")

    try:
        first_row = json.loads(kept_lines[0])
        question_preview = " ".join(str(first_row.get("question", "")).split())[:200]
        print(
            "[smoke] Using first real training row: "
            f"id={first_row.get('id', '(missing)')} "
            f"ground_truth={first_row.get('ground_truth', '(missing)')} "
            f"question_preview={question_preview}"
        )
    except Exception:
        print(f"[smoke] Using first real training row from {source_path}")

    dst.write_text("".join(kept_lines), encoding="utf-8")
    return str(dst)


def _load_first_jsonl_row(path: str) -> dict:
    src = Path(path)
    with src.open("r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No non-empty JSONL rows found in {path}")


def _prepare_easy_smoke_data_jsonl(target_path: str) -> str:
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "easy_smoke_0",
        "question": (
            "Compute 17 + 25. You may reason briefly, but the final answer must be a single integer in "
            "\\boxed{}."
        ),
        "ground_truth": "42",
    }
    print(f"[smoke] Using easy smoke row: id={row['id']} ground_truth={row['ground_truth']} question={row['question']}")
    dst.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return str(dst)


def _prepare_single_problem_smoke_data_jsonl(
    source_path: str,
    target_path: str,
    *,
    row_id: str,
    expected_oss_correctness: int | None = None,
) -> str:
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(source_path)

    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    selected_row: dict | None = None
    with src.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == row_id:
                selected_row = row
                break

    if selected_row is None:
        raise ValueError(f"Row id={row_id!r} not found in {source_path}")

    actual_oss_correctness = selected_row.get("oss_correctness")
    if expected_oss_correctness is not None and actual_oss_correctness != expected_oss_correctness:
        raise AssertionError(
            f"Expected oss_correctness={expected_oss_correctness} for {row_id}, "
            f"got {actual_oss_correctness!r}"
        )

    question_preview = " ".join(str(selected_row.get("question", "")).split())[:200]
    print(
        "[smoke-single-rl] Using one-problem dataset row: "
        f"id={selected_row.get('id')} "
        f"ground_truth={selected_row.get('ground_truth')} "
        f"oss_correctness={selected_row.get('oss_correctness')} "
        f"question_preview={question_preview}"
    )
    dst.write_text(json.dumps(selected_row) + "\n", encoding="utf-8")
    return str(dst)


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _print_rollout_artifact_debug(artifact_path: str | Path) -> None:
    import sys

    import torch

    if "/root/slime" not in sys.path:
        sys.path.insert(0, "/root/slime")
    from examples.scaffolding.scaffolding_boxed import extract_last_boxed_integer

    payload = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    samples = payload["samples"]
    solvers = [s for s in samples if (s.get("metadata") or {}).get("round_type") == "solver"]
    judges = [s for s in samples if (s.get("metadata") or {}).get("round_type") == "judge"]
    if not solvers or not judges:
        print(f"[smoke] Artifact {artifact_path} is missing solver or judge samples")
        return

    def _print_sample(prefix: str, sample: dict) -> None:
        metadata = sample.get("metadata") or {}
        response = sample.get("response") or ""
        print(
            f"[smoke] {prefix}: "
            f"status={sample.get('status')} reward={sample.get('reward')} "
            f"extracted={extract_last_boxed_integer(response)} "
            f"used_harmony={metadata.get('used_harmony')} "
            f"tool_calls={metadata.get('tool_call_count')} "
            f"tool_errors={metadata.get('tool_error_count')} "
            f"analysis_only_nudges={metadata.get('analysis_only_nudge_count')} "
            f"python_first_retries={metadata.get('python_first_retry_count')} "
            f"finish_reason={metadata.get('finish_reason')} "
            f"recipient={metadata.get('last_harmony_recipient')} "
            f"channel={metadata.get('last_harmony_channel')}"
        )
        if metadata.get("tool_call_previews"):
            print(f"[smoke] {prefix} tool_call_previews={metadata.get('tool_call_previews')}")
        if metadata.get("tool_result_previews"):
            print(f"[smoke] {prefix} tool_result_previews={metadata.get('tool_result_previews')}")
        print(f"[smoke] {prefix} response_preview={_preview(response)}")

    _print_sample("first_solver", solvers[0])
    _print_sample("first_judge", judges[0])


def _validate_scaffolding_rollout_artifact(
    artifact_path: str | Path,
    *,
    expected_answer: str | None = None,
    min_solver_reward_mean: float = 0.25,
    min_judge_reward_mean: float = 0.5,
    min_extract_rate: float = 0.5,
    min_solver_tool_calls_total: int = 0,
    min_solver_successful_tool_calls_total: int = 0,
    require_any_positive_solver_reward: bool = False,
    require_any_positive_judge_reward: bool = False,
    require_solver_harmony: bool = False,
) -> None:
    import sys

    import torch
    if "/root/slime" not in sys.path:
        sys.path.insert(0, "/root/slime")
    from examples.scaffolding.scaffolding_boxed import extract_last_boxed_integer

    payload = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    samples = payload["samples"]
    if not samples:
        raise ValueError(f"No samples found in rollout artifact: {artifact_path}")

    solver_rewards = []
    judge_rewards = []
    extracted = []
    solver_tool_calls_total = 0
    solver_successful_tool_calls_total = 0
    solver_harmony_count = 0
    expected_norm = expected_answer.strip() if expected_answer is not None else None

    for sample in samples:
        metadata = sample.get("metadata") or {}
        round_type = metadata.get("round_type")
        reward = float(sample.get("reward", 0.0) or 0.0)
        response = sample.get("response") or ""

        extracted_answer = extract_last_boxed_integer(response)
        extracted.append(extracted_answer is not None)
        if round_type == "solver":
            solver_rewards.append(reward)
            solver_tool_calls = int(metadata.get("tool_call_count") or 0)
            solver_tool_errors = int(metadata.get("tool_error_count") or 0)
            solver_tool_calls_total += solver_tool_calls
            solver_successful_tool_calls_total += max(0, solver_tool_calls - solver_tool_errors)
            if metadata.get("used_harmony"):
                solver_harmony_count += 1
            if expected_norm is not None and extracted_answer is not None and extracted_answer != expected_norm:
                raise AssertionError(
                    f"Unexpected solver answer in {artifact_path}: got {extracted_answer}, expected {expected_norm}"
                )
        elif round_type == "judge":
            judge_rewards.append(reward)

    if not solver_rewards or not judge_rewards:
        raise AssertionError(f"Expected both solver and judge samples in {artifact_path}")

    solver_mean = sum(solver_rewards) / len(solver_rewards)
    judge_mean = sum(judge_rewards) / len(judge_rewards)
    extract_rate = sum(extracted) / len(extracted)
    if solver_mean < min_solver_reward_mean:
        raise AssertionError(f"solver_mean_reward={solver_mean:.3f} < {min_solver_reward_mean:.3f}")
    if judge_mean < min_judge_reward_mean:
        raise AssertionError(f"judge_mean_reward={judge_mean:.3f} < {min_judge_reward_mean:.3f}")
    if extract_rate < min_extract_rate:
        raise AssertionError(f"extract_rate={extract_rate:.3f} < {min_extract_rate:.3f}")
    if solver_tool_calls_total < min_solver_tool_calls_total:
        raise AssertionError(f"solver_tool_calls_total={solver_tool_calls_total} < {min_solver_tool_calls_total}")
    if solver_successful_tool_calls_total < min_solver_successful_tool_calls_total:
        raise AssertionError(
            "solver_successful_tool_calls_total="
            f"{solver_successful_tool_calls_total} < {min_solver_successful_tool_calls_total}"
        )
    if require_any_positive_solver_reward and not any(reward > 0.0 for reward in solver_rewards):
        raise AssertionError("No positive solver reward found in rollout artifact")
    if require_any_positive_judge_reward and not any(reward > 0.0 for reward in judge_rewards):
        raise AssertionError("No positive judge reward found in rollout artifact")
    if require_solver_harmony and solver_harmony_count == 0:
        raise AssertionError("No solver samples used Harmony")


def _prepare_smoke_train_cfg(cfg: Dict[str, str], preset: ModelPreset) -> Dict[str, str]:
    train_cfg = dict(cfg)
    rollout_bs = int(train_cfg.get("rollout_batch_size", preset.rollout_batch_size))
    required_n_sp = max(
        2 * int(train_cfg.get("attempts", "2")),
        (preset.smoke_global_batch_size + rollout_bs - 1) // rollout_bs,
    )
    if required_n_sp % 2 != 0:
        required_n_sp += 1
    train_cfg["attempts"] = str(required_n_sp // 2)
    train_cfg["global_batch_size"] = str(preset.smoke_global_batch_size)
    return train_cfg


def _build_train_env(cfg: Dict[str, str], preset: ModelPreset, *, smoke: bool = False) -> Dict[str, str]:
    dispatcher_type = str(cfg.get("moe_token_dispatcher_type", preset.moe_token_dispatcher_type))
    flex_dispatcher_backend = str(cfg.get("moe_flex_dispatcher_backend", preset.moe_flex_dispatcher_backend))
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": f"/root/slime:{env.get('PYTHONPATH', '')}",
            "SLIME_SCRIPT_MODEL_SIZE": preset.model_size,
            "SLIME_SCRIPT_HF_CHECKPOINT": cfg["hf_checkpoint"],
            "SLIME_SCRIPT_DATA_JSONL": cfg["data_jsonl"],
            "SLIME_SCRIPT_TRAIN_BACKEND": "megatron",
            "SLIME_SCRIPT_NUM_GPUS": str(preset.num_gpus),
            "SLIME_SCRIPT_TP": str(cfg.get("tp", preset.tp)),
            "SLIME_SCRIPT_EP": str(cfg.get("ep", preset.ep)),
            "SLIME_SCRIPT_ETP": "1",
            "SLIME_SCRIPT_SGLANG_EP_SIZE": str(cfg.get("sglang_ep_size", preset.sglang_ep_size)),
            "SLIME_SCRIPT_ROLLOUT_TP": str(cfg.get("rollout_tp", preset.rollout_tp)),
            "SLIME_SCRIPT_SGLANG_MEM_FRACTION": str(
                cfg.get("sglang_mem_fraction", preset.sglang_mem_fraction)
            ),
            "SLIME_SCRIPT_MAX_TOKENS_PER_GPU": str(
                cfg.get("max_tokens_per_gpu", preset.max_tokens_per_gpu)
            ),
            "SLIME_SCRIPT_LOG_PROBS_MAX_TOKENS_PER_GPU": str(
                cfg.get("log_probs_max_tokens_per_gpu", preset.log_probs_max_tokens_per_gpu)
            ),
            "SLIME_SCRIPT_ROLLOUT_BATCH_SIZE": str(
                cfg.get("rollout_batch_size", preset.rollout_batch_size)
            ),
            "SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN": str(
                cfg.get("rollout_max_context_len", preset.rollout_max_context_len)
            ),
            "SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN": str(cfg.get("rollout_max_response_len", 8192)),
            "SLIME_SCRIPT_NUM_ROLLOUT": str(cfg.get("num_rollout", 16)),
            "SLIME_SCAFFOLDING_ATTEMPTS": str(cfg.get("attempts", 8)),
            "SLIME_SCRIPT_MOE_TOKEN_DISPATCHER_TYPE": dispatcher_type,
            "SLIME_SCRIPT_ENABLE_MOE_DEEPEP": "1"
            if str(cfg.get("enable_moe_deepep", "1" if preset.enable_moe_deepep else "0")) == "1"
            else "0",
            "SLIME_SCRIPT_ENABLE_LORA": "1"
            if str(cfg.get("enable_lora", "1")).strip().lower() not in {"0", "false", "no"}
            else "0",
            "SLIME_BRIDGE_MERGE_ADAPTER_WEIGHTS": "1"
            if str(cfg.get("merge_adapter_weights", "1")).strip().lower() not in {"0", "false", "no"}
            else "0",
            "SLIME_BRIDGE_COMPARE_TO_HF": "1"
            if str(cfg.get("bridge_compare_to_hf", "0")).strip().lower() in {"1", "true", "yes"}
            else "0",
            "SLIME_BRIDGE_COMPARE_LIMIT": str(cfg.get("bridge_compare_limit", "24")),
            "SLIME_BRIDGE_COMPARE_FAIL_FAST": "1"
            if str(cfg.get("bridge_compare_fail_fast", "0")).strip().lower() in {"1", "true", "yes"}
            else "0",
        }
    )
    if dispatcher_type == "flex" and flex_dispatcher_backend:
        env["SLIME_SCRIPT_MOE_FLEX_DISPATCHER_BACKEND"] = flex_dispatcher_backend
    else:
        env.pop("SLIME_SCRIPT_MOE_FLEX_DISPATCHER_BACKEND", None)
    env.setdefault("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "0")
    for cfg_key, env_key in (
        ("wandb_project", "SLIME_SCRIPT_WANDB_PROJECT"),
        ("wandb_group", "SLIME_SCRIPT_WANDB_GROUP"),
        ("wandb_team", "SLIME_SCRIPT_WANDB_TEAM"),
    ):
        if cfg.get(cfg_key):
            env[env_key] = str(cfg[cfg_key])

    if "global_batch_size" in cfg:
        env["SLIME_SCRIPT_GLOBAL_BATCH_SIZE"] = str(cfg["global_batch_size"])
    else:
        attempts = int(env["SLIME_SCAFFOLDING_ATTEMPTS"])
        rollout_bs = int(env["SLIME_SCRIPT_ROLLOUT_BATCH_SIZE"])
        # Keep dual-group scaffolding invariant: GBS ~= rollout_batch_size * 2 * attempts.
        env["SLIME_SCRIPT_GLOBAL_BATCH_SIZE"] = str(rollout_bs * 2 * attempts)

    # If wandb key is absent, keep launcher non-interactive in remote Modal workers.
    if not env.get("WANDB_API_KEY"):
        env["SLIME_SCRIPT_SKIP_WANDB"] = "1"
    if cfg.get("skip_wandb") == "1":
        env["SLIME_SCRIPT_SKIP_WANDB"] = "1"

    # Hugging Face Hub auth (gated models): secret often provides HF_TOKEN only.
    if env.get("HF_TOKEN") and not env.get("HUGGING_FACE_HUB_TOKEN"):
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]

    for cfg_key, env_key in (
        ("cuda_launch_blocking", "CUDA_LAUNCH_BLOCKING"),
        ("nccl_debug", "NCCL_DEBUG"),
        ("nccl_debug_subsys", "NCCL_DEBUG_SUBSYS"),
        ("disable_moe_permute_fusion", "SLIME_SCRIPT_DISABLE_MOE_PERMUTE_FUSION"),
        ("disable_moe_grouped_gemm", "SLIME_SCRIPT_DISABLE_MOE_GROUPED_GEMM"),
    ):
        if cfg.get(cfg_key):
            env[env_key] = cfg[cfg_key]

    if preset.disable_moe_permute_fusion and not cfg.get("disable_moe_permute_fusion"):
        env["SLIME_SCRIPT_DISABLE_MOE_PERMUTE_FUSION"] = "1"
    if preset.disable_moe_grouped_gemm and not cfg.get("disable_moe_grouped_gemm"):
        env["SLIME_SCRIPT_DISABLE_MOE_GROUPED_GEMM"] = "1"
    if str(cfg.get("force_safe_varlen_attn", "1" if preset.force_safe_varlen_attn else "0")) == "1":
        env["SLIME_FORCE_SAFE_VARLEN_ATTN"] = "1"
    if "safe_varlen_attn_block_size" in cfg:
        env["SLIME_SAFE_VARLEN_ATTN_BLOCK_SIZE"] = str(cfg["safe_varlen_attn_block_size"])
    elif preset.safe_varlen_attn_block_size > 0:
        env["SLIME_SAFE_VARLEN_ATTN_BLOCK_SIZE"] = str(preset.safe_varlen_attn_block_size)

    # Persist Hub downloads on the Modal volume (local entrypoint sets this when SLIME_MODAL_VOLUME is set).
    if cfg.get("persist_hf_cache_on_data_volume") == "1":
        env.setdefault("HF_HOME", "/root/data/hf-cache")

    if smoke:
        # Keep smoke runs bounded while still exercising one real rollout and one real train step.
        preserve_model_limits = str(cfg.get("preserve_model_limits_in_smoke", "0")) == "1"
        chunk_tokens_default = "0"
        use_streaming_default = "1"
        stall_warning_default = "0"
        stall_fail_default = "0"
        first_tool_retry_tokens_default = "0"
        first_tool_retry_limit_default = "0"
        harmony_python_fewshot_default = "0"
        harmony_force_python_prefix_after_retry_default = "0"
        analysis_only_warning_default = "0"
        analysis_only_fail_default = "0"
        smoke_rollout_max_context_len = (
            str(cfg.get("rollout_max_context_len", preset.rollout_max_context_len))
            if preserve_model_limits
            else "4096"
        )
        smoke_rollout_max_response_len = (
            str(cfg.get("rollout_max_response_len", 8192)) if preserve_model_limits else "256"
        )
        env["SLIME_SCRIPT_SKIP_WANDB"] = "1"
        env["SLIME_SCRIPT_NUM_ROLLOUT"] = str(cfg.get("num_rollout", "1"))
        env["SLIME_SCRIPT_ROLLOUT_BATCH_SIZE"] = str(cfg.get("rollout_batch_size", "1"))
        env["SLIME_SCAFFOLDING_ATTEMPTS"] = str(cfg.get("attempts", "2"))
        env["SLIME_SCRIPT_GLOBAL_BATCH_SIZE"] = str(
            cfg.get("global_batch_size", preset.smoke_global_batch_size)
        )
        env["SLIME_SCRIPT_ROLLOUT_MAX_CONTEXT_LEN"] = smoke_rollout_max_context_len
        env["SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN"] = smoke_rollout_max_response_len
        env["SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_CONTEXT_LEN"] = smoke_rollout_max_context_len
        env["SLIME_SCRIPT_SMOKE_ROLLOUT_MAX_RESPONSE_LEN"] = smoke_rollout_max_response_len
        env.setdefault("SLIME_SCAFFOLDING_GENERATION_CHUNK_TOKENS", chunk_tokens_default)
        env.setdefault("SLIME_SCAFFOLDING_USE_STREAMING", use_streaming_default)
        env.setdefault("SLIME_SCAFFOLDING_STALL_WARNING_TOKENS", stall_warning_default)
        env.setdefault("SLIME_SCAFFOLDING_STALL_FAIL_TOKENS", stall_fail_default)
        env.setdefault("SLIME_SCAFFOLDING_FIRST_TOOL_RETRY_TOKENS", first_tool_retry_tokens_default)
        env.setdefault("SLIME_SCAFFOLDING_FIRST_TOOL_RETRY_LIMIT", first_tool_retry_limit_default)
        env.setdefault("SLIME_SCAFFOLDING_HARMONY_PYTHON_FEWSHOT", harmony_python_fewshot_default)
        env.setdefault(
            "SLIME_SCAFFOLDING_HARMONY_FORCE_PYTHON_PREFIX_AFTER_RETRY",
            harmony_force_python_prefix_after_retry_default,
        )
        env.setdefault("SLIME_SCAFFOLDING_ANALYSIS_ONLY_WARNING_TURNS", analysis_only_warning_default)
        env.setdefault("SLIME_SCAFFOLDING_ANALYSIS_ONLY_FAIL_TURNS", analysis_only_fail_default)
        env.setdefault("SLIME_SCRIPT_SMOKE_DEBUG_ROLLOUT_PATH", "/root/data/debug_rollout/{rollout_id}.pt")
        env["SLIME_DEBUG_ROLLOUT_REPEAT_TO"] = str(preset.smoke_global_batch_size)
        if preset.model_size == "120b":
            env.setdefault("SLIME_SCRIPT_RAY_WORKER_REGISTER_TIMEOUT_SECONDS", "600")
            env.setdefault("SLIME_SCAFFOLDING_JUPYTER_TIMEOUT", "120")

    return env


def _run_training(
    cfg: Dict[str, str],
    preset: ModelPreset,
    *,
    launcher_args: list[str] | None = None,
    smoke: bool = False,
) -> None:
    train_env = _build_train_env(cfg, preset, smoke=smoke)
    cmd = ["python3", "examples/scaffolding/run_gpt_oss_scaffolding_rl.py"]
    if launcher_args:
        cmd.extend(launcher_args)
    subprocess.run(
        cmd,
        cwd="/root/slime",
        env=train_env,
        check=True,
    )


def _stage_smoke_run(cfg: Dict[str, str], preset: ModelPreset) -> None:
    smoke_cfg = dict(cfg)
    smoke_cfg["skip_wandb"] = "1"
    smoke_cfg.setdefault("persist_hf_cache_on_data_volume", "1")
    source_row: dict | None = None
    if "attempts" not in smoke_cfg:
        rollout_bs = int(smoke_cfg.get("rollout_batch_size", preset.rollout_batch_size))
        min_attempts = max(1, (preset.smoke_global_batch_size + 2 * rollout_bs - 1) // (2 * rollout_bs))
        smoke_cfg["attempts"] = str(min_attempts)
    rollout_bs = int(smoke_cfg.get("rollout_batch_size", preset.rollout_batch_size))
    smoke_cfg["global_batch_size"] = str(rollout_bs * 2 * int(smoke_cfg.get("attempts", "2")))
    if str(smoke_cfg.get("use_easy_smoke_data", "0")) == "1":
        smoke_cfg["data_jsonl"] = _prepare_easy_smoke_data_jsonl(f"/tmp/slime-easy-smoke-{preset.model_size}.jsonl")
    else:
        smoke_cfg["data_jsonl"] = _prepare_smoke_data_jsonl(
            cfg["data_jsonl"],
            f"/tmp/slime-smoke-{preset.model_size}.jsonl",
            max_lines=1,
        )
        source_row = _load_first_jsonl_row(smoke_cfg["data_jsonl"])
    debug_rollout_path = smoke_cfg.get("debug_rollout_path", "/root/data/debug_rollout/{rollout_id}.pt")
    rollout_zero_path = Path(debug_rollout_path.format(rollout_id=0))
    if rollout_zero_path.exists():
        rollout_zero_path.unlink()

    _run_training(
        smoke_cfg,
        preset,
        launcher_args=[
            "--smoke",
            "--debug-rollout-only",
            "--sglang-disable-cuda-graph",
            "--save-debug-rollout-data",
            debug_rollout_path,
        ],
        smoke=True,
    )

    if not rollout_zero_path.is_file():
        raise FileNotFoundError(f"Smoke rollout did not produce expected debug data: {rollout_zero_path}")
    _print_rollout_artifact_debug(rollout_zero_path)
    if str(smoke_cfg.get("validate_easy_smoke", "0")) == "1":
        _validate_scaffolding_rollout_artifact(
            rollout_zero_path,
            expected_answer=smoke_cfg.get("easy_smoke_expected_answer", "42"),
        )
    elif str(smoke_cfg.get("validate_real_smoke", "0")) == "1":
        _validate_scaffolding_rollout_artifact(
            rollout_zero_path,
            expected_answer=(source_row or {}).get("ground_truth"),
            min_solver_reward_mean=0.0,
            min_judge_reward_mean=0.0,
            min_extract_rate=0.0,
            min_solver_tool_calls_total=1,
            min_solver_successful_tool_calls_total=1,
            require_any_positive_solver_reward=True,
            require_any_positive_judge_reward=True,
            require_solver_harmony=True,
        )

    _run_training(
        _prepare_smoke_train_cfg(smoke_cfg, preset),
        preset,
        launcher_args=[
            "--smoke",
            "--load-debug-rollout-data",
            debug_rollout_path,
        ],
        smoke=True,
    )


def _stage_inference_only_smoke_run(cfg: Dict[str, str], preset: ModelPreset) -> None:
    smoke_cfg = dict(cfg)
    smoke_cfg["skip_wandb"] = "1"
    smoke_cfg.setdefault("persist_hf_cache_on_data_volume", "1")
    source_row: dict | None = None
    if "attempts" not in smoke_cfg:
        rollout_bs = int(smoke_cfg.get("rollout_batch_size", preset.rollout_batch_size))
        min_attempts = max(1, (preset.smoke_global_batch_size + 2 * rollout_bs - 1) // (2 * rollout_bs))
        smoke_cfg["attempts"] = str(min_attempts)
    rollout_bs = int(smoke_cfg.get("rollout_batch_size", preset.rollout_batch_size))
    smoke_cfg["global_batch_size"] = str(rollout_bs * 2 * int(smoke_cfg.get("attempts", "2")))
    if str(smoke_cfg.get("use_easy_smoke_data", "0")) == "1":
        smoke_cfg["data_jsonl"] = _prepare_easy_smoke_data_jsonl(f"/tmp/slime-easy-smoke-{preset.model_size}.jsonl")
    else:
        smoke_cfg["data_jsonl"] = _prepare_smoke_data_jsonl(
            cfg["data_jsonl"],
            f"/tmp/slime-smoke-{preset.model_size}.jsonl",
            max_lines=1,
        )
        source_row = _load_first_jsonl_row(smoke_cfg["data_jsonl"])

    debug_rollout_path = smoke_cfg.get("debug_rollout_path", "/root/data/debug_rollout/{rollout_id}.pt")
    rollout_zero_path = Path(debug_rollout_path.format(rollout_id=0))
    if rollout_zero_path.exists():
        rollout_zero_path.unlink()

    _run_training(
        smoke_cfg,
        preset,
        launcher_args=[
            "--smoke",
            "--inference-only",
            "--sglang-disable-cuda-graph",
            "--save-debug-rollout-data",
            debug_rollout_path,
        ],
        smoke=True,
    )

    if not rollout_zero_path.is_file():
        raise FileNotFoundError(f"Inference-only smoke rollout did not produce expected debug data: {rollout_zero_path}")
    _print_rollout_artifact_debug(rollout_zero_path)
    if str(smoke_cfg.get("validate_easy_smoke", "0")) == "1":
        _validate_scaffolding_rollout_artifact(
            rollout_zero_path,
            expected_answer=smoke_cfg.get("easy_smoke_expected_answer", "42"),
        )
    elif str(smoke_cfg.get("validate_real_smoke", "0")) == "1":
        _validate_scaffolding_rollout_artifact(
            rollout_zero_path,
            expected_answer=(source_row or {}).get("ground_truth"),
            min_solver_reward_mean=0.0,
            min_judge_reward_mean=0.0,
            min_extract_rate=0.0,
            min_solver_tool_calls_total=1,
            min_solver_successful_tool_calls_total=1,
            require_any_positive_solver_reward=True,
            require_any_positive_judge_reward=True,
            require_solver_harmony=True,
        )


def _print_direct_inference_artifact_debug(artifact_path: str | Path) -> None:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    attempts = payload.get("attempts") or []
    if not attempts:
        print(f"[direct-smoke] Artifact {artifact_path} has no attempts")
        return

    first = attempts[0]
    metadata = first.get("metadata") or {}
    print(
        "[direct-smoke] first_attempt: "
        f"status={first.get('status')} reward={first.get('reward')} extracted={first.get('extracted_answer')} "
        f"used_harmony={metadata.get('used_harmony')} tool_calls={metadata.get('tool_call_count')} "
        f"tool_errors={metadata.get('tool_error_count')} finish_reason={metadata.get('finish_reason')} "
        f"recipient={metadata.get('last_harmony_recipient')} channel={metadata.get('last_harmony_channel')}"
    )
    if metadata.get("tool_call_previews"):
        print(f"[direct-smoke] first_attempt tool_call_previews={metadata.get('tool_call_previews')}")
    if metadata.get("tool_result_previews"):
        print(f"[direct-smoke] first_attempt tool_result_previews={metadata.get('tool_result_previews')}")
    print(f"[direct-smoke] first_attempt response_preview={first.get('response_preview')}")
    print(f"[direct-smoke] summary={payload.get('summary')}")


def _validate_direct_inference_artifact(
    artifact_path: str | Path,
    *,
    require_harmony: bool = True,
    require_tool_or_answer: bool = True,
) -> None:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    attempts = payload.get("attempts") or []
    if not attempts:
        raise AssertionError(f"No attempts found in direct inference artifact: {artifact_path}")

    any_harmony = False
    any_tool = False
    any_successful_tool = False
    any_answer = False
    for attempt in attempts:
        metadata = attempt.get("metadata") or {}
        any_harmony = any_harmony or bool(metadata.get("used_harmony"))
        tool_calls = int(metadata.get("tool_call_count") or 0)
        tool_errors = int(metadata.get("tool_error_count") or 0)
        any_tool = any_tool or tool_calls > 0
        any_successful_tool = any_successful_tool or max(0, tool_calls - tool_errors) > 0
        any_answer = any_answer or attempt.get("extracted_answer") is not None

    if require_harmony and not any_harmony:
        raise AssertionError("Direct inference smoke produced no Harmony-backed attempts")
    if require_tool_or_answer and not (any_successful_tool or any_answer):
        raise AssertionError("Direct inference smoke produced neither a successful tool call nor an extracted answer")


def _stage_direct_inference_smoke_run(cfg: Dict[str, str], preset: ModelPreset) -> None:
    smoke_cfg = dict(cfg)
    smoke_cfg["skip_wandb"] = "1"
    smoke_cfg.setdefault("persist_hf_cache_on_data_volume", "1")
    smoke_cfg["data_jsonl"] = _prepare_smoke_data_jsonl(
        cfg["data_jsonl"],
        f"/tmp/slime-direct-smoke-{preset.model_size}.jsonl",
        max_lines=1,
    )
    source_row = _load_first_jsonl_row(smoke_cfg["data_jsonl"])

    output_path = smoke_cfg.get("debug_rollout_path", "/root/data/debug_rollout/direct_sglang_smoke.json")
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    direct_env = _build_train_env(smoke_cfg, preset, smoke=True)
    direct_env.setdefault("PYTHONPATH", f"/root/slime:{direct_env.get('PYTHONPATH', '')}")
    direct_env.setdefault("SLIME_SCAFFOLDING_ATTEMPTS", smoke_cfg.get("attempts", "2"))

    cmd = [
        "python3",
        "examples/scaffolding/run_gpt_oss_scaffolding_sglang_direct_smoke.py",
        "--hf-checkpoint",
        smoke_cfg["hf_checkpoint"],
        "--data-jsonl",
        smoke_cfg["data_jsonl"],
        "--output-path",
        output_path,
        "--attempts",
        smoke_cfg.get("attempts", "2"),
        "--rollout-num-gpus",
        str(smoke_cfg.get("rollout_num_gpus", preset.num_gpus)),
        "--rollout-num-gpus-per-engine",
        str(smoke_cfg.get("rollout_num_gpus_per_engine", smoke_cfg.get("rollout_tp", preset.rollout_tp))),
        "--rollout-max-context-len",
        str(smoke_cfg.get("rollout_max_context_len", preset.rollout_max_context_len)),
        "--rollout-max-response-len",
        str(smoke_cfg.get("rollout_max_response_len", 8192)),
        "--sglang-ep-size",
        str(smoke_cfg.get("sglang_ep_size", preset.sglang_ep_size)),
        "--sglang-mem-fraction-static",
        str(smoke_cfg.get("sglang_mem_fraction", preset.sglang_mem_fraction)),
        "--seed",
        "42",
        "--sglang-disable-cuda-graph",
    ]
    subprocess.run(
        cmd,
        cwd="/root/slime",
        env=direct_env,
        check=True,
    )

    if not output_file.is_file():
        raise FileNotFoundError(f"Direct SGLang smoke did not produce expected artifact: {output_file}")

    _print_direct_inference_artifact_debug(output_file)
    _validate_direct_inference_artifact(output_file)
    print(
        "[direct-smoke] validated "
        f"id={source_row.get('id')} ground_truth={source_row.get('ground_truth')} artifact={output_file}"
    )


def _stage_single_problem_rl_smoke_run(cfg: Dict[str, str], preset: ModelPreset) -> None:
    smoke_cfg = dict(cfg)
    smoke_cfg.setdefault("persist_hf_cache_on_data_volume", "1")
    row_id = smoke_cfg.get("single_problem_row_id", "polymath_10017")
    expected_oss_correctness = smoke_cfg.get("single_problem_expected_oss_correctness")
    hardcoded_single_problem_path = smoke_cfg.get(
        "single_problem_data_jsonl",
        "/root/slime/examples/scaffolding/single_problem_smoke_polymath_10017.jsonl",
    )
    if Path(hardcoded_single_problem_path).is_file():
        smoke_cfg["data_jsonl"] = hardcoded_single_problem_path
        row = _load_first_jsonl_row(smoke_cfg["data_jsonl"])
        print(
            "[smoke-single-rl] Using hardcoded one-problem dataset row: "
            f"id={row.get('id')} ground_truth={row.get('ground_truth')} "
            f"oss_correctness={row.get('oss_correctness')}"
        )
        if row.get("id") != row_id:
            raise AssertionError(
                f"Expected hardcoded row id {row_id!r}, found {row.get('id')!r} in {hardcoded_single_problem_path}"
            )
        if expected_oss_correctness is not None and row.get("oss_correctness") != int(expected_oss_correctness):
            raise AssertionError(
                f"Expected oss_correctness={expected_oss_correctness} for {row_id}, "
                f"got {row.get('oss_correctness')!r}"
            )
    else:
        smoke_cfg["data_jsonl"] = _prepare_single_problem_smoke_data_jsonl(
            cfg["data_jsonl"],
            f"/tmp/slime-single-problem-smoke-{preset.model_size}.jsonl",
            row_id=row_id,
            expected_oss_correctness=(
                int(expected_oss_correctness) if expected_oss_correctness is not None else None
            ),
        )
    smoke_cfg.setdefault("preserve_model_limits_in_smoke", "1")
    smoke_cfg.setdefault("attempts", "2")
    attempts = int(smoke_cfg["attempts"])
    n_samples_per_prompt = 2 * attempts
    smoke_cfg.setdefault("global_batch_size", str(preset.smoke_global_batch_size))
    global_batch_size = int(smoke_cfg["global_batch_size"])
    if global_batch_size % n_samples_per_prompt != 0:
        raise AssertionError(
            f"single-problem RL smoke requires global_batch_size divisible by n_samples_per_prompt; "
            f"got global_batch_size={global_batch_size}, n_samples_per_prompt={n_samples_per_prompt}"
        )
    smoke_cfg.setdefault("rollout_batch_size", str(global_batch_size // n_samples_per_prompt))
    rollout_bs = int(smoke_cfg["rollout_batch_size"])
    if rollout_bs * n_samples_per_prompt != global_batch_size:
        raise AssertionError(
            "single-problem RL smoke requires global_batch_size == rollout_batch_size * n_samples_per_prompt; "
            f"got global_batch_size={global_batch_size}, rollout_batch_size={rollout_bs}, "
            f"n_samples_per_prompt={n_samples_per_prompt}"
        )
    smoke_cfg.setdefault("rollout_max_context_len", str(preset.rollout_max_context_len))
    smoke_cfg.setdefault("rollout_max_response_len", "8192")
    smoke_cfg.setdefault("wandb_project", "slime-aimo-scaffolding")
    smoke_cfg.setdefault("wandb_group", f"gpt-oss-single-problem-{row_id}")

    _run_training(
        smoke_cfg,
        preset,
        launcher_args=[
            "--smoke",
            "--sglang-disable-cuda-graph",
        ],
        smoke=True,
    )


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def train_20b_modal(cfg: Dict[str, str]) -> None:
    _run_training(cfg, PRESETS["20b"])


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def train_120b_modal(cfg: Dict[str, str]) -> None:
    _run_training(cfg, PRESETS["120b"])


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def preflight_modal(cfg: Dict[str, str]) -> None:
    model_size = cfg.get("model_size", "20b")
    preset = PRESETS[model_size]
    _run_training(cfg, preset, launcher_args=["--smoke-rewards-only"])


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_20b_modal(cfg: Dict[str, str]) -> None:
    _stage_smoke_run(cfg, PRESETS["20b"])


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_train_20b_modal(cfg: Dict[str, str]) -> None:
    _run_training(
        cfg,
        PRESETS["20b"],
        launcher_args=[
            "--smoke",
            "--load-debug-rollout-data",
            cfg.get("debug_rollout_path", "/root/data/debug_rollout/{rollout_id}.pt"),
        ],
        smoke=True,
    )


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 90,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_120b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg["preserve_model_limits_in_smoke"] = "1"
    smoke_cfg.setdefault("validate_real_smoke", "1")
    _stage_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 90,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_120b_defaults_modal() -> None:
    smoke_cfg = {
        "hf_checkpoint": DEFAULT_HF_CHECKPOINT_120B,
        "data_jsonl": "/root/data/train_data_filtered.jsonl",
        "preserve_model_limits_in_smoke": "1",
        "validate_real_smoke": "1",
    }
    _stage_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:8",
    cpu=32,
    memory=262144,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_infer_120b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg["preserve_model_limits_in_smoke"] = "1"
    smoke_cfg.setdefault("validate_real_smoke", "1")
    _stage_inference_only_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:8",
    cpu=32,
    memory=262144,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_infer_120b_defaults_modal() -> None:
    smoke_cfg = {
        "hf_checkpoint": DEFAULT_HF_CHECKPOINT_120B,
        "data_jsonl": "/root/data/train_data_filtered.jsonl",
        "preserve_model_limits_in_smoke": "1",
        "validate_real_smoke": "1",
    }
    _stage_inference_only_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:1",
    cpu=12,
    memory=98304,
    timeout=60 * 45,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_infer_direct_120b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg["preserve_model_limits_in_smoke"] = "1"
    smoke_cfg.setdefault("attempts", "1")
    smoke_cfg.setdefault("debug_rollout_path", "/root/data/debug_rollout/direct_sglang_smoke.json")
    smoke_cfg.setdefault("rollout_num_gpus", "1")
    smoke_cfg.setdefault("rollout_num_gpus_per_engine", "1")
    smoke_cfg.setdefault("sglang_ep_size", "1")
    smoke_cfg.setdefault("sglang_mem_fraction", "0.70")
    _stage_direct_inference_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:1",
    cpu=12,
    memory=98304,
    timeout=60 * 45,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_infer_direct_20b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg["preserve_model_limits_in_smoke"] = "1"
    smoke_cfg.setdefault("data_jsonl", "/root/slime/examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl")
    smoke_cfg.setdefault("attempts", "1")
    smoke_cfg.setdefault("debug_rollout_path", "/root/data/debug_rollout/direct_sglang_smoke_20b.json")
    smoke_cfg.setdefault("rollout_num_gpus", "1")
    smoke_cfg.setdefault("rollout_num_gpus_per_engine", "1")
    smoke_cfg.setdefault("sglang_ep_size", "1")
    smoke_cfg.setdefault("sglang_mem_fraction", str(PRESET_20B_SINGLE_GPU_REPRO.sglang_mem_fraction))
    smoke_cfg.setdefault("rollout_max_context_len", str(PRESET_20B_SINGLE_GPU_REPRO.rollout_max_context_len))
    smoke_cfg.setdefault("rollout_max_response_len", "2048")
    _stage_direct_inference_smoke_run(smoke_cfg, PRESET_20B_SINGLE_GPU_REPRO)


@app.function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131072,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_megatron_bridge_repro_20b_modal(cfg: dict = None) -> None:
    print("[bridge-repro-modal] function start", flush=True)
    repro_cfg = dict(cfg or {})
    data_jsonl = repro_cfg.get("single_problem_data_jsonl") or "/root/slime/examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl"
    output_path = repro_cfg.get("debug_rollout_path", "/root/data/debug_rollout/megatron_bridge_repro_20b.json")
    rollout_max_context_len = repro_cfg.get("bridge_repro_rollout_max_context_len") or repro_cfg.get("rollout_max_context_len", "4096")
    rollout_max_response_len = repro_cfg.get("bridge_repro_rollout_max_response_len") or repro_cfg.get("rollout_max_response_len", "2048")
    cmd = [
        "python3",
        "-u",
        "examples/scaffolding/run_gpt_oss_scaffolding_megatron_bridge_repro.py",
        "--hf-checkpoint",
        repro_cfg.get("hf_checkpoint", DEFAULT_HF_CHECKPOINT_20B),
        "--data-jsonl",
        data_jsonl,
        "--output-path",
        output_path,
        "--row-index",
        str(repro_cfg.get("row_index", "0")),
        "--rollout-max-context-len",
        str(rollout_max_context_len),
        "--rollout-max-response-len",
        str(rollout_max_response_len),
        "--sglang-mem-fraction-static",
        str(repro_cfg.get("sglang_mem_fraction", PRESET_20B_SINGLE_GPU_REPRO.sglang_mem_fraction)),
        "--skip-baseline",
    ]
    if str(repro_cfg.get("enable_lora", "1")).strip().lower() in {"0", "false", "no"}:
        cmd.append("--no-enable-lora")
    if str(repro_cfg.get("merge_adapter_weights", "1")).strip().lower() in {"0", "false", "no"}:
        cmd.append("--no-merge-adapter-weights")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/root/slime:/root/Megatron-LM:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["SLIME_SGLANG_SERIALIZE_BY_VALUE"] = "1"
    env.setdefault("HF_HOME", "/root/data/hf-cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/data/hf-cache/hub")
    print(f"[bridge-repro-modal] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd="/root/slime", env=env, check=True)


@app.function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131072,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_rl_topology_repro_20b_modal(cfg: dict = None) -> None:
    print("[rl-topology-repro-modal] function start", flush=True)
    repro_cfg = dict(cfg or {})
    data_jsonl = repro_cfg.get("single_problem_data_jsonl") or "/root/slime/examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl"
    output_path = repro_cfg.get("debug_rollout_path", "/root/data/debug_rollout/rl_topology_repro_20b.json")
    rollout_max_context_len = repro_cfg.get("bridge_repro_rollout_max_context_len") or repro_cfg.get("rollout_max_context_len", "4096")
    rollout_max_response_len = repro_cfg.get("bridge_repro_rollout_max_response_len") or repro_cfg.get("rollout_max_response_len", "2048")
    cmd = [
        "python3",
        "-u",
        "examples/scaffolding/run_gpt_oss_scaffolding_rl_topology_repro.py",
        "--hf-checkpoint",
        repro_cfg.get("hf_checkpoint", DEFAULT_HF_CHECKPOINT_20B),
        "--data-jsonl",
        data_jsonl,
        "--output-path",
        output_path,
        "--row-index",
        str(repro_cfg.get("row_index", "0")),
        "--rollout-max-context-len",
        str(rollout_max_context_len),
        "--rollout-max-response-len",
        str(rollout_max_response_len),
        "--sglang-mem-fraction-static",
        str(repro_cfg.get("sglang_mem_fraction", PRESET_20B_SINGLE_GPU_REPRO.sglang_mem_fraction)),
    ]
    if str(repro_cfg.get("enable_lora", "1")).strip().lower() in {"0", "false", "no"}:
        cmd.append("--no-enable-lora")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/root/slime:/root/Megatron-LM:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME", "/root/data/hf-cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/data/hf-cache/hub")
    print(f"[rl-topology-repro-modal] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd="/root/slime", env=env, check=True)


@app.function(
    image=image,
    gpu="H200:1",
    cpu=12,
    memory=98304,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_sglang_refit_repro_20b_modal(cfg: dict = None) -> None:
    repro_cfg = dict(cfg or {})
    data_jsonl = repro_cfg.get("single_problem_data_jsonl") or "/root/slime/examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl"
    output_path = repro_cfg.get("debug_rollout_path", "/root/data/debug_rollout/sglang_refit_repro_20b.json")
    cmd = [
        "python3",
        "-u",
        "examples/scaffolding/run_gpt_oss_scaffolding_sglang_refit_repro.py",
        "--hf-checkpoint",
        repro_cfg.get("hf_checkpoint", DEFAULT_HF_CHECKPOINT_20B),
        "--data-jsonl",
        data_jsonl,
        "--output-path",
        output_path,
        "--row-index",
        str(repro_cfg.get("row_index", "0")),
        "--rollout-max-context-len",
        str(repro_cfg.get("rollout_max_context_len", PRESET_20B_SINGLE_GPU_REPRO.rollout_max_context_len)),
        "--rollout-max-response-len",
        str(repro_cfg.get("rollout_max_response_len", "2048")),
        "--sglang-mem-fraction-static",
        str(repro_cfg.get("sglang_mem_fraction", PRESET_20B_SINGLE_GPU_REPRO.sglang_mem_fraction)),
        "--refit-mode",
        str(repro_cfg.get("refit_mode", "disk")),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/root/slime:/root/Megatron-LM:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME", "/root/data/hf-cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/data/hf-cache/hub")
    subprocess.run(cmd, cwd="/root/slime", env=env, check=True)


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 45,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_easy_120b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg["use_easy_smoke_data"] = "1"
    smoke_cfg["validate_easy_smoke"] = "1"
    smoke_cfg.setdefault("easy_smoke_expected_answer", "42")
    smoke_cfg.setdefault("debug_rollout_path", "/root/data/debug_rollout/easy_smoke_{rollout_id}.pt")
    smoke_cfg.setdefault("attempts", "2")
    smoke_cfg.setdefault("rollout_max_response_len", "1024")
    _stage_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_train_120b_modal(cfg: dict = None) -> None:
    cfg = dict(cfg or {})
    train_cfg = _prepare_smoke_train_cfg(cfg, PRESETS["120b"])
    train_cfg["preserve_model_limits_in_smoke"] = "1"
    _run_training(
        train_cfg,
        PRESETS["120b"],
        launcher_args=[
            "--smoke",
            "--load-debug-rollout-data",
            cfg.get("debug_rollout_path", "/root/data/debug_rollout/{rollout_id}.pt"),
        ],
        smoke=True,
    )


@app.function(
    image=image,
    gpu="H200:8",
    cpu=64,
    memory=786432,
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_single_problem_rl_120b_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg.setdefault("single_problem_row_id", "polymath_10017")
    if "single_problem_data_jsonl" not in smoke_cfg:
        smoke_cfg.setdefault("single_problem_expected_oss_correctness", "10")
    _stage_single_problem_rl_smoke_run(smoke_cfg, PRESETS["120b"])


@app.function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131072,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def smoke_single_problem_rl_20b_small_modal(cfg: dict = None) -> None:
    smoke_cfg = dict(cfg or {})
    smoke_cfg.setdefault(
        "single_problem_data_jsonl",
        "/root/slime/examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl",
    )
    smoke_cfg.setdefault("single_problem_row_id", "fib100_mod_100000")
    smoke_cfg.setdefault("attempts", "1")
    smoke_cfg.setdefault("num_rollout", "1")
    smoke_cfg.setdefault("rollout_batch_size", "1")
    smoke_cfg.setdefault("global_batch_size", "2")
    smoke_cfg.setdefault("rollout_max_context_len", "4096")
    smoke_cfg.setdefault("rollout_max_response_len", "2048")
    smoke_cfg.setdefault("preserve_model_limits_in_smoke", "1")
    smoke_cfg.setdefault("skip_wandb", "1")
    smoke_cfg.setdefault("bridge_compare_to_hf", "1")
    smoke_cfg.setdefault("bridge_compare_limit", "24")
    _stage_single_problem_rl_smoke_run(smoke_cfg, PRESET_20B_SINGLE_GPU_REPRO)


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def repro_20b_modal(cfg: Dict[str, str]) -> None:
    repro_cfg = dict(cfg)
    repro_cfg["num_rollout"] = "1"
    repro_cfg["skip_wandb"] = "1"
    debug_rollout_path = repro_cfg.get("debug_rollout_path", "/root/data/debug_rollout/full_repro_{rollout_id}.pt")
    rollout_zero_path = Path(debug_rollout_path.format(rollout_id=0))
    if rollout_zero_path.exists():
        rollout_zero_path.unlink()

    _run_training(
        repro_cfg,
        PRESETS["20b"],
        launcher_args=[
            "--debug-rollout-only",
            "--sglang-disable-cuda-graph",
            "--save-debug-rollout-data",
            debug_rollout_path,
        ],
    )

    if not rollout_zero_path.is_file():
        raise FileNotFoundError(f"Repro rollout did not produce expected debug data: {rollout_zero_path}")

    _run_training(
        repro_cfg,
        PRESETS["20b"],
        launcher_args=[
            "--load-debug-rollout-data",
            debug_rollout_path,
        ],
    )


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=262144,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def repro_train_20b_modal(cfg: Dict[str, str]) -> None:
    repro_cfg = dict(cfg)
    repro_cfg["num_rollout"] = "1"
    repro_cfg["skip_wandb"] = "1"
    _run_training(
        repro_cfg,
        PRESETS["20b"],
        launcher_args=[
            "--load-debug-rollout-data",
            repro_cfg.get("debug_rollout_path", "/root/data/debug_rollout/full_repro_{rollout_id}.pt"),
        ],
    )


@app.local_entrypoint()
def main(
    mode: str = "full",
    model_size: str = "20b",
    hf_checkpoint: str = "",
    data_jsonl: str = "/root/data/train_data_filtered.jsonl",
    attempts: int = 0,
    num_rollout: int = 16,
    rollout_max_response_len: int = 8192,
    rollout_batch_size: int = 0,
    rollout_max_context_len: int = 0,
    global_batch_size: int = 0,
    sglang_mem_fraction: float = 0.0,
    max_tokens_per_gpu: int = 0,
    log_probs_max_tokens_per_gpu: int = 0,
    tp: int = 0,
    ep: int = 0,
    rollout_tp: int = 0,
    moe_token_dispatcher_type: str = "",
    moe_flex_dispatcher_backend: str = "",
    enable_moe_deepep: int = -1,
    cuda_launch_blocking: int = 0,
    nccl_debug: str = "",
    nccl_debug_subsys: str = "",
    disable_moe_permute_fusion: int = 0,
    disable_moe_grouped_gemm: int = 0,
    force_safe_varlen_attn: int = 0,
    safe_varlen_attn_block_size: int = 0,
    debug_rollout_path: str = "",
    easy_smoke_expected_answer: str = "42",
    single_problem_data_jsonl: str = "",
    single_problem_row_id: str = "",
    single_problem_expected_oss_correctness: str = "",
    enable_lora: int = 1,
    merge_adapter_weights: int = 1,
    bridge_compare_to_hf: int = 0,
    bridge_compare_limit: int = 0,
    bridge_compare_fail_fast: int = 0,
    refit_mode: str = "disk",
) -> None:
    model_size = model_size.lower().strip()
    if model_size not in PRESETS:
        raise ValueError("model_size must be one of: 20b, 120b")

    default_ckpt = DEFAULT_HF_CHECKPOINT_20B if model_size == "20b" else DEFAULT_HF_CHECKPOINT_120B
    effective_ckpt = hf_checkpoint.strip() or default_ckpt

    cfg: Dict[str, str] = {
        "model_size": model_size,
        "hf_checkpoint": effective_ckpt,
        "data_jsonl": data_jsonl,
        "num_rollout": str(num_rollout),
        "rollout_max_response_len": str(rollout_max_response_len),
    }

    if attempts > 0:
        cfg["attempts"] = str(attempts)
    if rollout_batch_size > 0:
        cfg["rollout_batch_size"] = str(rollout_batch_size)
    if rollout_max_context_len > 0:
        cfg["rollout_max_context_len"] = str(rollout_max_context_len)
    if global_batch_size > 0:
        cfg["global_batch_size"] = str(global_batch_size)
    if sglang_mem_fraction > 0:
        cfg["sglang_mem_fraction"] = str(sglang_mem_fraction)
    if max_tokens_per_gpu > 0:
        cfg["max_tokens_per_gpu"] = str(max_tokens_per_gpu)
    if log_probs_max_tokens_per_gpu > 0:
        cfg["log_probs_max_tokens_per_gpu"] = str(log_probs_max_tokens_per_gpu)
    if tp > 0:
        cfg["tp"] = str(tp)
    if ep > 0:
        cfg["ep"] = str(ep)
    if rollout_tp > 0:
        cfg["rollout_tp"] = str(rollout_tp)
    if moe_token_dispatcher_type:
        cfg["moe_token_dispatcher_type"] = moe_token_dispatcher_type
    if moe_flex_dispatcher_backend:
        cfg["moe_flex_dispatcher_backend"] = moe_flex_dispatcher_backend
    if enable_moe_deepep >= 0:
        cfg["enable_moe_deepep"] = str(int(enable_moe_deepep > 0))
    if cuda_launch_blocking > 0:
        cfg["cuda_launch_blocking"] = str(cuda_launch_blocking)
    if nccl_debug:
        cfg["nccl_debug"] = nccl_debug
    if nccl_debug_subsys:
        cfg["nccl_debug_subsys"] = nccl_debug_subsys
    if disable_moe_permute_fusion > 0:
        cfg["disable_moe_permute_fusion"] = str(disable_moe_permute_fusion)
    if disable_moe_grouped_gemm > 0:
        cfg["disable_moe_grouped_gemm"] = str(disable_moe_grouped_gemm)
    if force_safe_varlen_attn > 0:
        cfg["force_safe_varlen_attn"] = str(force_safe_varlen_attn)
    if safe_varlen_attn_block_size > 0:
        cfg["safe_varlen_attn_block_size"] = str(safe_varlen_attn_block_size)
    if debug_rollout_path:
        cfg["debug_rollout_path"] = debug_rollout_path
    if easy_smoke_expected_answer:
        cfg["easy_smoke_expected_answer"] = easy_smoke_expected_answer
    if single_problem_data_jsonl:
        cfg["single_problem_data_jsonl"] = single_problem_data_jsonl
    if single_problem_row_id:
        cfg["single_problem_row_id"] = single_problem_row_id
    if single_problem_expected_oss_correctness:
        cfg["single_problem_expected_oss_correctness"] = single_problem_expected_oss_correctness
    cfg["enable_lora"] = str(int(enable_lora > 0))
    cfg["merge_adapter_weights"] = str(int(merge_adapter_weights > 0))
    if bridge_compare_to_hf > 0:
        cfg["bridge_compare_to_hf"] = str(int(bridge_compare_to_hf > 0))
    if bridge_compare_limit > 0:
        cfg["bridge_compare_limit"] = str(bridge_compare_limit)
    if bridge_compare_fail_fast > 0:
        cfg["bridge_compare_fail_fast"] = str(int(bridge_compare_fail_fast > 0))
    if refit_mode:
        cfg["refit_mode"] = refit_mode

    cfg["persist_hf_cache_on_data_volume"] = "1"

    if mode not in {
        "full",
        "smoke",
        "smoke-easy",
        "smoke-single-rl",
        "smoke-single-rl-small",
        "smoke-bridge-repro",
        "smoke-rl-topology-repro",
        "smoke-sglang-refit-repro",
        "smoke-train",
        "smoke-infer",
        "smoke-infer-direct",
        "repro",
        "repro-train",
        "preflight",
    }:
        raise ValueError(
            "mode must be one of: full, smoke, smoke-easy, smoke-train, smoke-infer, "
            "smoke-infer-direct, smoke-single-rl, smoke-single-rl-small, smoke-bridge-repro, "
            "smoke-rl-topology-repro, smoke-sglang-refit-repro, "
            "repro, repro-train, preflight"
        )

    if mode == "preflight":
        preflight_modal.remote(cfg)
        return

    if mode == "smoke":
        if model_size == "20b":
            smoke_20b_modal.remote(cfg)
        else:
            smoke_120b_modal.remote(cfg)
        return

    if mode == "smoke-easy":
        if model_size != "120b":
            raise ValueError("mode=smoke-easy is currently only supported for model_size=120b")
        smoke_easy_120b_modal.remote(cfg)
        return

    if mode == "smoke-single-rl":
        if model_size != "120b":
            raise ValueError("mode=smoke-single-rl is currently only supported for model_size=120b")
        smoke_single_problem_rl_120b_modal.remote(cfg)
        return

    if mode == "smoke-single-rl-small":
        if model_size != "20b":
            raise ValueError("mode=smoke-single-rl-small is currently only supported for model_size=20b")
        smoke_single_problem_rl_20b_small_modal.remote(cfg)
        return

    if mode == "smoke-bridge-repro":
        if model_size != "20b":
            raise ValueError("mode=smoke-bridge-repro is currently only supported for model_size=20b")
        smoke_megatron_bridge_repro_20b_modal.remote(cfg)
        return

    if mode == "smoke-rl-topology-repro":
        if model_size != "20b":
            raise ValueError("mode=smoke-rl-topology-repro is currently only supported for model_size=20b")
        smoke_rl_topology_repro_20b_modal.remote(cfg)
        return

    if mode == "smoke-sglang-refit-repro":
        if model_size != "20b":
            raise ValueError("mode=smoke-sglang-refit-repro is currently only supported for model_size=20b")
        smoke_sglang_refit_repro_20b_modal.remote(cfg)
        return

    if mode == "smoke-train":
        if model_size == "20b":
            smoke_train_20b_modal.remote(cfg)
        else:
            smoke_train_120b_modal.remote(cfg)
        return

    if mode == "smoke-infer":
        if model_size != "120b":
            raise ValueError("mode=smoke-infer is currently only supported for model_size=120b")
        smoke_infer_120b_modal.remote(cfg)
        return

    if mode == "smoke-infer-direct":
        if model_size == "20b":
            smoke_infer_direct_20b_modal.remote(cfg)
        else:
            smoke_infer_direct_120b_modal.remote(cfg)
        return

    if mode == "repro":
        if model_size != "20b":
            raise ValueError("mode=repro is currently only supported for model_size=20b")
        repro_20b_modal.remote(cfg)
        return

    if mode == "repro-train":
        if model_size != "20b":
            raise ValueError("mode=repro-train is currently only supported for model_size=20b")
        repro_train_20b_modal.remote(cfg)
        return

    if model_size == "20b":
        train_20b_modal.remote(cfg)
    else:
        train_120b_modal.remote(cfg)
