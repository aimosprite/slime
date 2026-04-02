#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.scaffolding.gs_config import ScaffoldingCFG
from examples.scaffolding.reward_gpt_oss_scaffolding import scalar_correctness_reward
from examples.scaffolding.rollout_gpt_oss_scaffolding import (
    _augment_problem_text,
    _problem_budget_s,
    _response_preview,
    run_one_attempt,
)
from examples.scaffolding.run_gpt_oss_scaffolding_sglang_direct_smoke import (
    _load_jsonl_row,
    _make_runtime_args,
    _start_direct_server,
)
from examples.scaffolding.run_gpt_oss_scaffolding_rl import _resolve_hf_checkpoint_to_local_dir
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.value)
        except Exception:
            pass
    if hasattr(value, "name") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.name
        except Exception:
            pass
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct SGLang same-weights refit repro for GPT-OSS.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refit-mode",
        choices=("disk", "tensor"),
        default="disk",
        help="Refit strategy to test.",
    )
    return parser.parse_args()


def _progress(message: str) -> None:
    print(f"[sglang-refit-repro] {message}", flush=True)


def _run_attempt(
    *,
    runtime_args,
    state,
    cfg: ScaffoldingCFG,
    row: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    question = _augment_problem_text(str(row["question"]))
    ground_truth = str(row["ground_truth"])
    notebook_elapsed = float(row.get("notebook_elapsed", 0.0) or 0.0)
    budget_s = _problem_budget_s(cfg, 1, notebook_elapsed)
    sampling_params = {
        "temperature": runtime_args.rollout_temperature,
        "top_p": runtime_args.rollout_top_p,
        "top_k": runtime_args.rollout_top_k,
        "max_new_tokens": runtime_args.rollout_max_response_len,
        "stop": runtime_args.rollout_stop,
        "stop_token_ids": runtime_args.rollout_stop_token_ids,
        "skip_special_tokens": runtime_args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    _progress(f"{label}: run_one_attempt start budget_s={budget_s:.1f}")
    result = asyncio.run(
        run_one_attempt(
            runtime_args,
            state,
            cfg,
            question,
            0,
            time.time() + budget_s,
            sampling_params=sampling_params,
            session_id=f"sglang-refit-{label}",
        )
    )
    reward = scalar_correctness_reward(result.extracted_answer, ground_truth)
    metadata = result.metadata or {}
    payload = {
        "label": label,
        "status": result.status,
        "reward": reward,
        "extracted": result.extracted_answer,
        "tool_calls": metadata.get("tool_call_count"),
        "tool_errors": metadata.get("tool_error_count"),
        "used_harmony": metadata.get("used_harmony"),
        "response_preview": _response_preview(result.response_text, limit=300),
        "response_text": result.response_text,
        "metadata": metadata,
    }
    _progress(
        f"{label}: status={payload['status']} reward={payload['reward']} "
        f"extracted={payload['extracted']} preview={payload['response_preview']!r}"
    )
    return payload


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout_s: float = 1800.0) -> dict[str, Any]:
    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _get_json(base_url: str, endpoint: str, timeout_s: float = 60.0) -> dict[str, Any]:
    response = requests.get(f"{base_url}/{endpoint}", timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _write_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _progress(f"wrote {output_path}")


def _iter_safetensors_tensors(model_dir: str):
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    files_to_keys: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        files_to_keys.setdefault(filename, []).append(key)
    for filename, keys in files_to_keys.items():
        shard_path = Path(model_dir) / filename
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in keys:
                yield key, shard.get_tensor(key)


def _update_same_weights_from_tensor(base_url: str, hf_checkpoint: str) -> dict[str, Any]:
    from slime.backends.fsdp_utils.update_weight_utils import FlattenedTensorBucket, _serialize_flattened_tensor_data

    named_tensors = list(_iter_safetensors_tensors(hf_checkpoint))
    _progress(f"same-weights refit tensor mode: loaded {len(named_tensors)} tensors")
    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    flattened_tensor_data = {
        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
        "metadata": flattened_tensor_bucket.get_metadata(),
    }
    payload = {
        "serialized_named_tensors": [_serialize_flattened_tensor_data(flattened_tensor_data)],
        "load_format": "flattened_bucket_by_value" if os.environ.get("SLIME_SGLANG_SERIALIZE_BY_VALUE", "").strip().lower() in {"1", "true", "yes"} else "flattened_bucket",
        "flush_cache": True,
        "weight_version": "same-weights-refit",
    }
    return _post_json(base_url, "update_weights_from_tensor", payload, timeout_s=3600.0)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_checkpoint = _resolve_hf_checkpoint_to_local_dir(args.hf_checkpoint)
    row = _load_jsonl_row(args.data_jsonl, args.row_index)
    cfg = ScaffoldingCFG()
    cli_args = argparse.Namespace(
        hf_checkpoint=hf_checkpoint,
        seed=args.seed,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_max_context_len=args.rollout_max_context_len,
        rollout_max_response_len=args.rollout_max_response_len,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_mem_fraction_static=args.sglang_mem_fraction_static,
        sglang_disable_cuda_graph=False,
        attempts=1,
        judge=False,
    )
    runtime_args = _make_runtime_args(cli_args, hf_checkpoint)

    process = None
    base_url = None
    started_at = time.time()
    try:
        process, port = _start_direct_server(runtime_args)
        base_url = f"http://127.0.0.1:{port}"
        _progress(f"server ready base_url={base_url}")

        before_version = _get_json(base_url, "get_weight_version")["weight_version"]
        state = argparse.Namespace(tokenizer=load_tokenizer(hf_checkpoint, trust_remote_code=True))
        baseline = _run_attempt(
            runtime_args=runtime_args,
            state=state,
            cfg=cfg,
            row=row,
            label="baseline",
        )
        partial_payload = {
            "hf_checkpoint": hf_checkpoint,
            "row": row,
            "refit_mode": args.refit_mode,
            "elapsed_s": time.time() - started_at,
            "weight_version_before": before_version,
            "weight_version_after": None,
            "baseline": baseline,
            "post_refit": None,
        }
        _write_payload(output_path, partial_payload)

        if args.refit_mode == "disk":
            _progress("same-weights refit start mode=disk")
            refit_response = _post_json(
                base_url,
                "update_weights_from_disk",
                {
                    "model_path": hf_checkpoint,
                    "load_format": None,
                    "flush_cache": True,
                    "weight_version": "same-weights-refit",
                },
                timeout_s=3600.0,
            )
            _progress(f"same-weights refit done response={refit_response}")
        elif args.refit_mode == "tensor":
            _progress("same-weights refit start mode=tensor")
            refit_response = _update_same_weights_from_tensor(base_url, hf_checkpoint)
            _progress(f"same-weights refit done response={refit_response}")
        else:
            raise NotImplementedError(args.refit_mode)

        after_version = _get_json(base_url, "get_weight_version")["weight_version"]
        post_refit = _run_attempt(
            runtime_args=runtime_args,
            state=state,
            cfg=cfg,
            row=row,
            label="post_refit",
        )

        payload = {
            "hf_checkpoint": hf_checkpoint,
            "row": row,
            "refit_mode": args.refit_mode,
            "elapsed_s": time.time() - started_at,
            "weight_version_before": before_version,
            "weight_version_after": after_version,
            "baseline": baseline,
            "post_refit": post_refit,
        }
        _write_payload(output_path, payload)
    finally:
        if process is not None and process.is_alive():
            _progress("terminating direct server")
            process.terminate()
            process.join(timeout=20)


if __name__ == "__main__":
    main()
