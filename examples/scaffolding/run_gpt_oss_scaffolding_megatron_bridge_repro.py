#!/usr/bin/env python3
from __future__ import annotations

print("[bridge-repro] python entrypoint import start", flush=True)

import argparse
import asyncio
import faulthandler
import json
import logging
import multiprocessing
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray
import requests

print("[bridge-repro] third-party imports loaded", flush=True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MEGATRON_ROOT = Path("/root/Megatron-LM")
if _MEGATRON_ROOT.is_dir() and str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(1, str(_MEGATRON_ROOT))

print("[bridge-repro] importing gs_config", flush=True)
from examples.scaffolding.gs_config import ScaffoldingCFG

print("[bridge-repro] importing reward_gpt_oss_scaffolding", flush=True)
from examples.scaffolding.reward_gpt_oss_scaffolding import scalar_correctness_reward

print("[bridge-repro] importing rollout_gpt_oss_scaffolding", flush=True)
from examples.scaffolding.rollout_gpt_oss_scaffolding import (
    _augment_problem_text,
    _problem_budget_s,
    _response_preview,
    run_one_attempt,
)

print("[bridge-repro] importing run_gpt_oss_scaffolding_rl", flush=True)
from examples.scaffolding.run_gpt_oss_scaffolding_rl import _resolve_hf_checkpoint_to_local_dir

print("[bridge-repro] importing ray actor env helpers", flush=True)
from slime.ray.actor_group import FORWARDED_TRAIN_ENV_VARS
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

print("[bridge-repro] importing tokenizer loader", flush=True)
from slime.utils.processing_utils import load_tokenizer

print("[bridge-repro] local module imports loaded", flush=True)

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    print(f"[bridge-repro] {message}", flush=True)


def _ensure_shared_multiprocessing_authkey() -> str:
    authkey_hex = os.environ.get("SLIME_MP_AUTHKEY_HEX", "").strip()
    if not authkey_hex:
        authkey_hex = secrets.token_hex(32)
        os.environ["SLIME_MP_AUTHKEY_HEX"] = authkey_hex
    multiprocessing.current_process().authkey = bytes.fromhex(authkey_hex)
    return authkey_hex


_ensure_shared_multiprocessing_authkey()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GPT-OSS Megatron->SGLang bridge repro on a single GPU.")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument(
        "--data-jsonl",
        default="examples/scaffolding/direct_smoke_fib100_mod_1e9p9.jsonl",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--rollout-max-context-len", type=int, default=4096)
    parser.add_argument("--rollout-max-response-len", type=int, default=2048)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-adapter-weights", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_model_args_from_script(model_type: str) -> list[str]:
    script_path = _REPO_ROOT / "scripts" / "models" / f"{model_type}.sh"
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    cmd = f'source "{script_path}" && printf "%s\\0" "${{MODEL_ARGS[@]}}"'
    output = subprocess.check_output(["bash", "-lc", cmd], cwd=str(_REPO_ROOT), env=os.environ)
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def _load_jsonl_row(path: str, row_index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index={row_index} is out of range for {path}")


def _start_direct_server(args: SimpleNamespace) -> tuple[Any, int]:
    from sglang.srt.server_args import ServerArgs

    from slime.backends.sglang_utils.sglang_engine import _compute_server_args, launch_server_process
    from slime.utils.http_utils import find_available_port

    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "false")
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    os.environ.setdefault("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "true")
    os.environ.setdefault("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "true")
    os.environ.setdefault("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false")
    os.environ.setdefault("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", "false")
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
    old_global_patch = os.environ.get("SLIME_ENABLE_GLOBAL_SGLANG_PATCH")
    os.environ["SLIME_ENABLE_GLOBAL_SGLANG_PATCH"] = "1"
    try:
        process = launch_server_process(ServerArgs(**server_args_dict))
    finally:
        if old_global_patch is None:
            os.environ.pop("SLIME_ENABLE_GLOBAL_SGLANG_PATCH", None)
        else:
            os.environ["SLIME_ENABLE_GLOBAL_SGLANG_PATCH"] = old_global_patch
    args.sglang_router_ip = "127.0.0.1"
    args.sglang_router_port = port
    _progress(
        "Direct SGLang server ready "
        f"port={port} tp={server_args_dict.get('tp_size')} "
        f"ep={server_args_dict.get('ep_size')} "
        f"mem_fraction_static={server_args_dict.get('mem_fraction_static')}"
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


@ray.remote(num_cpus=0.25)
class DirectEngineProxy:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict[str, Any] | None = None, timeout_s: float = 60.0) -> Any:
        _progress(f"DirectEngineProxy POST {endpoint} timeout_s={timeout_s}")
        response = requests.post(f"{self.base_url}/{endpoint}", json=payload or {}, timeout=timeout_s)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = response.json()
            _progress(f"DirectEngineProxy POST {endpoint} -> json")
            return body
        _progress(f"DirectEngineProxy POST {endpoint} -> text")
        return response.text

    def pause_generation(self):
        return self._post("pause_generation", timeout_s=30.0)

    def continue_generation(self):
        return self._post("continue_generation", timeout_s=30.0)

    def abort_all_requests(self):
        return self._post("abort_request", {"abort_all": True}, timeout_s=30.0)

    def flush_cache(self):
        if os.environ.get("SLIME_SGLANG_SKIP_FLUSH_CACHE", "").strip().lower() in {"1", "true", "yes"}:
            _progress("DirectEngineProxy flush_cache skipped by SLIME_SGLANG_SKIP_FLUSH_CACHE")
            return "skipped"
        _progress("DirectEngineProxy flush_cache start")
        for _ in range(60):
            try:
                response = requests.get(f"{self.base_url}/flush_cache")
                if response.status_code == 200:
                    _progress("DirectEngineProxy flush_cache -> success")
                    return response.text
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("Timeout while flushing direct server cache.")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        payload: dict[str, Any] = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._post("update_weights_from_tensor", payload, timeout_s=900.0)

    def get_weight_version(self):
        _progress("DirectEngineProxy get_weight_version")
        response = requests.get(f"{self.base_url}/get_weight_version", timeout=30.0)
        response.raise_for_status()
        ans = response.json()["weight_version"]
        _progress(f"DirectEngineProxy get_weight_version -> {ans}")
        return ans


def _build_debug_actor_class():
    _progress("Importing MegatronTrainRayActor lazily")
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor

    class MegatronBridgeDebugActor(MegatronTrainRayActor):
        def init_from_cli(self, cli_args: list[str]) -> dict[str, Any]:
            from slime.utils.arguments import parse_args

            _progress("MegatronBridgeDebugActor.init_from_cli: parse_args start")
            old_argv = sys.argv
            sys.argv = ["megatron-bridge-repro", *cli_args]
            try:
                args = parse_args()
            finally:
                sys.argv = old_argv
            _progress(
                "MegatronBridgeDebugActor.init_from_cli: parse_args complete "
                f"tp={getattr(args, 'tensor_model_parallel_size', None)} "
                f"ep={getattr(args, 'expert_model_parallel_size', None)} "
                f"pp={getattr(args, 'pipeline_model_parallel_size', None)} "
                f"enable_lora={getattr(args, 'enable_lora', None)}"
            )

            _progress("MegatronBridgeDebugActor.init_from_cli: calling super().init")
            start_rollout_id = super().init(args, role="actor", with_ref=False, with_opd_teacher=False)
            _progress(
                "MegatronBridgeDebugActor.init_from_cli: super().init complete "
                f"start_rollout_id={start_rollout_id}"
            )
            return {
                "start_rollout_id": start_rollout_id,
                "load": args.load,
                "enable_lora": bool(getattr(args, "enable_lora", False)),
                "megatron_to_hf_mode": getattr(args, "megatron_to_hf_mode", None),
                "tensor_model_parallel_size": getattr(args, "tensor_model_parallel_size", None),
                "expert_model_parallel_size": getattr(args, "expert_model_parallel_size", None),
                "expert_tensor_parallel_size": getattr(args, "expert_tensor_parallel_size", None),
            }

        def connect_and_update(self, rollout_engine) -> dict[str, Any]:
            _progress("MegatronBridgeDebugActor.connect_and_update: connecting rollout engine")
            self.weight_updater.connect_rollout_engines([rollout_engine], None)
            _progress("MegatronBridgeDebugActor.connect_and_update: starting weight update")
            self.weight_updater.update_weights()
            _progress(
                f"MegatronBridgeDebugActor.connect_and_update: weight update complete version={self.weight_updater.weight_version}"
            )
            return {
                "weight_version": self.weight_updater.weight_version,
            }

    return MegatronBridgeDebugActor


def _ray_get_with_progress(ref, *, label: str, poll_s: float = 20.0, timeout_s: float | None = None):
    start = time.time()
    while True:
        ready, _ = ray.wait([ref], timeout=poll_s)
        if ready:
            elapsed = time.time() - start
            value = ray.get(ready[0])
            _progress(f"{label} completed after {elapsed:.1f}s")
            return value
        elapsed = time.time() - start
        _progress(f"{label} still pending after {elapsed:.1f}s")
        if timeout_s is not None and elapsed >= timeout_s:
            raise TimeoutError(f"{label} timed out after {elapsed:.1f}s")


def _make_megatron_cli_args(
    *,
    model_type: str,
    hf_checkpoint: str,
    data_jsonl: str,
    rollout_max_context_len: int,
    rollout_max_response_len: int,
    sglang_mem_fraction_static: float,
    enable_lora: bool,
) -> list[str]:
    cli_args = [
        *_load_model_args_from_script(model_type),
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
        "1",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "2",
        "--rollout-max-response-len",
        str(rollout_max_response_len),
        "--rollout-max-context-len",
        str(rollout_max_context_len),
        "--rollout-temperature",
        "1.0",
        "--global-batch-size",
        "2",
        "--advantage-estimator",
        "grpo",
        "--rollout-function-path",
        "examples.scaffolding.rollout_gpt_oss_scaffolding.generate_rollout_gs",
        "--custom-reward-post-process-path",
        "examples.scaffolding.grpo_dual_group_reward_postprocess.dual_group_grpo_reward_postprocess",
        "--optimizer",
        "adam",
        "--lr",
        "2e-4",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--kl-loss-coef",
        "0.0",
        "--kl-loss-type",
        "low_var_kl",
        "--kl-coef",
        "0.0",
        "--entropy-coef",
        "0.0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-ep-size",
        "1",
        "--sglang-mem-fraction-static",
        str(sglang_mem_fraction_static),
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "1024",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--rollout-num-gpus",
        "1",
        "--colocate",
        "--megatron-to-hf-mode",
        "bridge",
        "--tensor-model-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--attention-backend",
        "flash",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
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


def _make_direct_runtime_args(
    *,
    hf_checkpoint: str,
    seed: int,
    rollout_max_context_len: int,
    rollout_max_response_len: int,
    sglang_mem_fraction_static: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint=hf_checkpoint,
        seed=seed,
        rollout_seed=seed,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=1,
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
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_server_concurrency=1,
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        use_slime_router=False,
        ci_test=False,
        rollout_max_context_len=rollout_max_context_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_mem_fraction_static=sglang_mem_fraction_static,
        sglang_disable_cuda_graph=True,
        sglang_context_length=rollout_max_context_len,
    )


def _train_actor_env(*, merge_adapter_weights: bool) -> dict[str, str]:
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = ["/root/slime", "/root/Megatron-LM"]
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    env_vars = {
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
        "PYTHONPATH": ":".join(pythonpath_parts),
        **{name: os.environ[name] for name in FORWARDED_TRAIN_ENV_VARS if os.environ.get(name)},
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
    }
    import torch_memory_saver

    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )
    if not os.path.exists(dynlib_path):
        raise FileNotFoundError(f"LD_PRELOAD so file {dynlib_path} does not exist.")
    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    env_vars["SLIME_BRIDGE_REPRO_TRACE_INIT"] = "1"
    env_vars["SLIME_ENABLE_QWEN_ROTARY_PATCH"] = "0"
    env_vars["CUDA_LAUNCH_BLOCKING"] = "1"
    env_vars["SLIME_SGLANG_FORCE_SINGLE_DTYPE_BUCKETS"] = "1"
    env_vars["SLIME_SGLANG_CLONE_BEFORE_BUCKET"] = "1"
    env_vars["SLIME_SGLANG_SERIALIZE_BY_VALUE"] = "1"
    env_vars["SLIME_BRIDGE_EXPORT_CPU"] = "1"
    env_vars["SLIME_BRIDGE_TRACE_EXPORT_FAILURE"] = "1"
    env_vars["SLIME_BRIDGE_MERGE_ADAPTER_WEIGHTS"] = "1" if merge_adapter_weights else "0"
    env_vars["SLIME_MP_AUTHKEY_HEX"] = _ensure_shared_multiprocessing_authkey()
    return env_vars


def _attempt_to_dict(result: Any, ground_truth: str) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "extracted_answer": result.extracted_answer,
        "reward": scalar_correctness_reward(result.response_text, ground_truth),
        "response": result.response_text,
        "response_preview": _response_preview(result.response_text, limit=400),
        "metadata": result.metadata,
    }


def _probe_direct_engine_controls(engine_proxy, *, label: str) -> dict[str, Any]:
    _progress(f"{label}: probing pause_generation")
    pause_result = _ray_get_with_progress(
        engine_proxy.pause_generation.remote(),
        label=f"{label}.pause_generation",
        poll_s=5.0,
        timeout_s=45.0,
    )
    _progress(f"{label}: probing continue_generation")
    continue_result = _ray_get_with_progress(
        engine_proxy.continue_generation.remote(),
        label=f"{label}.continue_generation",
        poll_s=5.0,
        timeout_s=45.0,
    )
    _progress(f"{label}: probing flush_cache")
    flush_result = _ray_get_with_progress(
        engine_proxy.flush_cache.remote(),
        label=f"{label}.flush_cache",
        poll_s=5.0,
        timeout_s=90.0,
    )
    _progress(f"{label}: probing get_weight_version")
    weight_version = _ray_get_with_progress(
        engine_proxy.get_weight_version.remote(),
        label=f"{label}.get_weight_version",
        poll_s=5.0,
        timeout_s=45.0,
    )
    return {
        "pause_generation": pause_result,
        "continue_generation": continue_result,
        "flush_cache": flush_result,
        "weight_version": weight_version,
    }


async def _drain_direct_engine_after_baseline(engine_proxy, *, label: str) -> dict[str, Any]:
    _progress(f"{label}: aborting any leftover in-flight requests from the streamed baseline solve")
    abort_result = _ray_get_with_progress(
        engine_proxy.abort_all_requests.remote(),
        label=f"{label}.abort_all_requests",
        poll_s=5.0,
        timeout_s=45.0,
    )
    await asyncio.sleep(2.0)
    _progress(f"{label}: probing weight version after abort")
    weight_version = _ray_get_with_progress(
        engine_proxy.get_weight_version.remote(),
        label=f"{label}.get_weight_version",
        poll_s=5.0,
        timeout_s=45.0,
    )
    return {
        "abort_all_requests": abort_result,
        "weight_version": weight_version,
    }


async def _run_repro(cli_args: argparse.Namespace) -> dict[str, Any]:
    hf_checkpoint = _resolve_hf_checkpoint_to_local_dir(cli_args.hf_checkpoint)
    runtime_args = _make_direct_runtime_args(
        hf_checkpoint=hf_checkpoint,
        seed=cli_args.seed,
        rollout_max_context_len=cli_args.rollout_max_context_len,
        rollout_max_response_len=cli_args.rollout_max_response_len,
        sglang_mem_fraction_static=cli_args.sglang_mem_fraction_static,
    )

    process = None
    actor = None
    engine_proxy = None
    baseline_dict = None
    post_sync_dict = None
    direct_server_base_url = None
    update_info = None
    init_info = None
    pre_megatron_controls = None
    post_init_controls = None
    post_baseline_drain = None

    try:
        process, port = _start_direct_server(runtime_args)
        direct_server_base_url = f"http://127.0.0.1:{port}"
        runtime_args.sglang_router_ip = "127.0.0.1"
        runtime_args.sglang_router_port = port

        tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        state = SimpleNamespace(tokenizer=tokenizer)
        cfg = ScaffoldingCFG.from_env()

        row = _load_jsonl_row(cli_args.data_jsonl, cli_args.row_index)
        problem_text = _augment_problem_text(row["question"])
        budget = _problem_budget_s(cfg, cfg.problems_remaining_default, notebook_elapsed=0.0)
        deadline = time.time() + budget
        sampling_params = _make_sampling_params(runtime_args)

        _progress(
            "Running 20B bridge repro on row "
            f"id={row.get('id')} ground_truth={row.get('ground_truth')} enable_lora={cli_args.enable_lora}"
        )

        if cli_args.skip_baseline:
            baseline_dict = {
                "status": "skipped",
                "extracted_answer": None,
                "reward": None,
                "response": None,
                "response_preview": None,
                "metadata": {"skipped": True},
            }
            _progress("Baseline inference skipped")
        else:
            baseline_result = await run_one_attempt(
                runtime_args,
                state,
                cfg,
                problem_text,
                attempt_idx=0,
                deadline=deadline,
                sampling_params=sampling_params.copy(),
                session_id=row.get("id"),
            )
            baseline_dict = _attempt_to_dict(baseline_result, str(row["ground_truth"]))
            _progress(
                "Baseline inference finished "
                f"status={baseline_dict['status']} reward={baseline_dict['reward']} "
                f"extracted={baseline_dict['extracted_answer']}"
            )

        if not ray.is_initialized():
            _progress("Initializing local Ray")
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            _progress("Local Ray initialized")

        _progress("Creating DirectEngineProxy actor")
        engine_proxy = DirectEngineProxy.remote(direct_server_base_url)
        _progress("DirectEngineProxy actor created")
        if cli_args.skip_baseline:
            post_baseline_drain = {"skipped": True}
        else:
            post_baseline_drain = await _drain_direct_engine_after_baseline(
                engine_proxy, label="post_baseline_drain"
            )
            _progress(f"Post-baseline direct engine drain ok: {post_baseline_drain}")
        pre_megatron_controls = _probe_direct_engine_controls(engine_proxy, label="pre_megatron_controls")
        _progress(f"Pre-Megatron direct engine controls ok: {pre_megatron_controls}")
        _progress("Building train actor runtime env")
        train_env = _train_actor_env(merge_adapter_weights=cli_args.merge_adapter_weights)
        _progress("Train actor runtime env built")
        os.environ.setdefault("SLIME_ENABLE_QWEN_ROTARY_PATCH", "0")
        MegatronBridgeDebugActor = _build_debug_actor_class()
        _progress("MegatronTrainRayActor lazy import complete")
        RemoteActor = ray.remote(num_gpus=1, runtime_env={"env_vars": train_env})(MegatronBridgeDebugActor)
        _progress("Creating MegatronBridgeDebugActor handle")
        actor = RemoteActor.remote(1, 0, None, None)
        _progress("MegatronBridgeDebugActor handle created")

        cli = _make_megatron_cli_args(
            model_type="gpt-oss-20B",
            hf_checkpoint=hf_checkpoint,
            data_jsonl=cli_args.data_jsonl,
            rollout_max_context_len=cli_args.rollout_max_context_len,
            rollout_max_response_len=cli_args.rollout_max_response_len,
            sglang_mem_fraction_static=cli_args.sglang_mem_fraction_static,
            enable_lora=cli_args.enable_lora,
        )
        _progress("Starting Megatron actor init")
        init_info = _ray_get_with_progress(
            actor.init_from_cli.remote(cli),
            label="actor.init_from_cli",
            poll_s=15.0,
            timeout_s=1200.0,
        )
        _progress(f"Megatron actor init returned: {init_info}")
        post_init_controls = _probe_direct_engine_controls(engine_proxy, label="post_init_controls")
        _progress(f"Post-init direct engine controls ok: {post_init_controls}")

        _progress("Starting connect_and_update")
        update_info = _ray_get_with_progress(
            actor.connect_and_update.remote(engine_proxy),
            label="actor.connect_and_update",
            poll_s=15.0,
            timeout_s=900.0,
        )
        _progress(f"connect_and_update returned: {update_info}")

        update_info["engine_weight_version"] = _ray_get_with_progress(
            engine_proxy.get_weight_version.remote(),
            label="engine_proxy.get_weight_version",
            poll_s=5.0,
            timeout_s=120.0,
        )

        deadline = time.time() + budget
        post_sync_result = await run_one_attempt(
            runtime_args,
            state,
            cfg,
            problem_text,
            attempt_idx=0,
            deadline=deadline,
            sampling_params=sampling_params.copy(),
            session_id=row.get("id"),
        )
        post_sync_dict = _attempt_to_dict(post_sync_result, str(row["ground_truth"]))

        return {
            "row": row,
            "hf_checkpoint": hf_checkpoint,
            "enable_lora": cli_args.enable_lora,
            "direct_server_base_url": direct_server_base_url,
            "baseline": baseline_dict,
            "post_sync": post_sync_dict,
            "megatron_init": init_info,
            "post_baseline_drain": post_baseline_drain,
            "pre_megatron_controls": pre_megatron_controls,
            "post_init_controls": post_init_controls,
            "weight_update": update_info,
        }
    finally:
        if actor is not None:
            try:
                ray.kill(actor)
            except Exception:
                pass
        if engine_proxy is not None:
            try:
                ray.kill(engine_proxy)
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()
        if process is not None:
            try:
                from sglang.srt.utils import kill_process_tree

                kill_process_tree(process.pid)
            except Exception:
                pass


def main() -> None:
    faulthandler.dump_traceback_later(300, repeat=True)
    cli_args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _progress("main parsed args")
    payload = asyncio.run(_run_repro(cli_args))
    output_path = Path(cli_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "summary": payload.get("weight_update")}, indent=2))


if __name__ == "__main__":
    main()
