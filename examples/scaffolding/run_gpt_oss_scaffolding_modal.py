#!/usr/bin/env python3
"""
Single Modal launcher for GPT-OSS scaffolding LoRA RL.

Default targets:
  - gpt-oss-20b  -> 2x H200
  - gpt-oss-120b -> 8x H200

This script delegates training to:
  examples/scaffolding/run_gpt_oss_scaffolding_rl.py
and only sets sensible model/topology defaults for Modal.

Typical usage:
  modal run examples/scaffolding/run_gpt_oss_scaffolding_modal.py \
    --model-size 20b \
    --hf-checkpoint /root/data/models/gpt-oss-20b \
    --data-jsonl /root/data/train_data_filtered.jsonl

  modal run examples/scaffolding/run_gpt_oss_scaffolding_modal.py \
    --model-size 120b \
    --hf-checkpoint /root/data/models/gpt-oss-120b \
    --data-jsonl /root/data/train_data_filtered.jsonl

Optional infra (via local env before `modal run`):
  SLIME_MODAL_VOLUME=<volume-name>   # mounted at /root/data
  SLIME_MODAL_SECRET=<secret-name>   # WANDB_API_KEY / HF_TOKEN
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import modal


APP_NAME = "slime-gpt-oss-scaffolding-rl"
REPO_ROOT = Path(__file__).resolve().parents[2]

MODAL_VOLUME_NAME = os.environ.get("SLIME_MODAL_VOLUME", "").strip()
MODAL_SECRET_NAME = os.environ.get("SLIME_MODAL_SECRET", "").strip()

VOLUMES = (
    {"/root/data": modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)}
    if MODAL_VOLUME_NAME
    else {}
)
SECRETS = [modal.Secret.from_name(MODAL_SECRET_NAME)] if MODAL_SECRET_NAME else []

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.02-py3", add_python="3.10")
    .run_commands("pip install -e /root/slime")
)

code_mount = modal.Mount.from_local_dir(str(REPO_ROOT), remote_path="/root/slime")


@dataclass(frozen=True)
class ModelPreset:
    model_size: str
    num_gpus: int
    tp: int
    ep: int
    rollout_tp: int
    sglang_mem_fraction: float
    max_tokens_per_gpu: int
    rollout_batch_size: int
    rollout_max_context_len: int


PRESETS: Dict[str, ModelPreset] = {
    # Fastest practical config on 2xH200 that still keeps comms moderate.
    "20b": ModelPreset(
        model_size="20b",
        num_gpus=2,
        tp=2,
        ep=1,
        rollout_tp=2,
        sglang_mem_fraction=0.65,
        max_tokens_per_gpu=3072,
        rollout_batch_size=1,
        rollout_max_context_len=32768,
    ),
    # Memory-first for 120B MoE: shard experts aggressively with EP=8.
    # Keep TP=1 to avoid extra dense-layer TP collectives.
    "120b": ModelPreset(
        model_size="120b",
        num_gpus=8,
        tp=1,
        ep=8,
        rollout_tp=8,
        sglang_mem_fraction=0.42,
        max_tokens_per_gpu=2048,
        rollout_batch_size=1,
        rollout_max_context_len=16384,
    ),
}


def _build_train_env(cfg: Dict[str, str], preset: ModelPreset) -> Dict[str, str]:
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
            "SLIME_SCRIPT_ROLLOUT_TP": str(cfg.get("rollout_tp", preset.rollout_tp)),
            "SLIME_SCRIPT_SGLANG_MEM_FRACTION": str(
                cfg.get("sglang_mem_fraction", preset.sglang_mem_fraction)
            ),
            "SLIME_SCRIPT_MAX_TOKENS_PER_GPU": str(
                cfg.get("max_tokens_per_gpu", preset.max_tokens_per_gpu)
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
        }
    )

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

    return env


def _run_training(cfg: Dict[str, str], preset: ModelPreset) -> None:
    train_env = _build_train_env(cfg, preset)
    subprocess.run(
        ["python3", "examples/scaffolding/run_gpt_oss_scaffolding_rl.py"],
        cwd="/root/slime",
        env=train_env,
        check=True,
    )


@app.function(
    image=image,
    mounts=[code_mount],
    gpu="H200:2",
    cpu=16,
    memory=131072,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def train_20b_modal(cfg: Dict[str, str]) -> None:
    _run_training(cfg, PRESETS["20b"])


@app.function(
    image=image,
    mounts=[code_mount],
    gpu="H200:8",
    cpu=64,
    memory=524288,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def train_120b_modal(cfg: Dict[str, str]) -> None:
    _run_training(cfg, PRESETS["120b"])


@app.local_entrypoint()
def main(
    model_size: str = "20b",
    hf_checkpoint: str = "",
    data_jsonl: str = "/root/data/train_data_filtered.jsonl",
    attempts: int = 8,
    num_rollout: int = 16,
    rollout_max_response_len: int = 8192,
    rollout_batch_size: int = 0,
    rollout_max_context_len: int = 0,
    global_batch_size: int = 0,
    sglang_mem_fraction: float = 0.0,
    max_tokens_per_gpu: int = 0,
    tp: int = 0,
    ep: int = 0,
    rollout_tp: int = 0,
) -> None:
    model_size = model_size.lower().strip()
    if model_size not in PRESETS:
        raise ValueError("model_size must be one of: 20b, 120b")

    default_ckpt = (
        "/root/data/models/gpt-oss-20b" if model_size == "20b" else "/root/data/models/gpt-oss-120b"
    )
    effective_ckpt = hf_checkpoint.strip() or default_ckpt

    cfg: Dict[str, str] = {
        "hf_checkpoint": effective_ckpt,
        "data_jsonl": data_jsonl,
        "attempts": str(attempts),
        "num_rollout": str(num_rollout),
        "rollout_max_response_len": str(rollout_max_response_len),
    }

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
    if tp > 0:
        cfg["tp"] = str(tp)
    if ep > 0:
        cfg["ep"] = str(ep)
    if rollout_tp > 0:
        cfg["rollout_tp"] = str(rollout_tp)

    if model_size == "20b":
        train_20b_modal.remote(cfg)
    else:
        train_120b_modal.remote(cfg)
