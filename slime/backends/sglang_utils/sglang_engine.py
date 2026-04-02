import dataclasses
import base64
import ipaddress
import inspect
import json
import logging
import multiprocessing
import os
import pickle
import secrets
import time
from urllib.parse import quote

import requests
import sglang_router
import torch
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def _ensure_shared_multiprocessing_authkey() -> None:
    authkey_hex = os.environ.get("SLIME_MP_AUTHKEY_HEX", "").strip()
    if not authkey_hex:
        authkey_hex = secrets.token_hex(32)
        os.environ["SLIME_MP_AUTHKEY_HEX"] = authkey_hex
    multiprocessing.current_process().authkey = bytes.fromhex(authkey_hex)


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def _patch_sglang_weight_loader_compat() -> None:
    """Make older SGLang default_weight_loader tolerant of newer keyword args."""
    try:
        from sglang.srt.model_loader import weight_utils
    except Exception:
        return

    default_weight_loader = getattr(weight_utils, "default_weight_loader", None)
    if default_weight_loader is None:
        return

    try:
        sig = inspect.signature(default_weight_loader)
    except (TypeError, ValueError):
        return

    if getattr(default_weight_loader, "_slime_filters_unsupported_kwargs", False):
        return

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return

    supported_kwargs = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    def _wrap_weight_loader(fn):
        if getattr(fn, "_slime_filters_unsupported_kwargs", False):
            return fn
        try:
            fn_sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in fn_sig.parameters.values()):
            return fn
        fn_supported_kwargs = {
            name
            for name, param in fn_sig.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        def _compat_weight_loader(*args, **kwargs):
            filtered_kwargs = {name: value for name, value in kwargs.items() if name in fn_supported_kwargs}
            return fn(*args, **filtered_kwargs)

        _compat_weight_loader._slime_filters_unsupported_kwargs = True
        _compat_weight_loader._slime_original_weight_loader = fn
        return _compat_weight_loader

    def _compat_default_weight_loader(*args, **kwargs):
        filtered_kwargs = {name: value for name, value in kwargs.items() if name in supported_kwargs}
        return default_weight_loader(*args, **filtered_kwargs)

    _compat_default_weight_loader._slime_filters_unsupported_kwargs = True
    weight_utils.default_weight_loader = _compat_default_weight_loader

    try:
        from sglang.srt.models import gpt_oss

        if getattr(gpt_oss, "default_weight_loader", None) is default_weight_loader:
            gpt_oss.default_weight_loader = _compat_default_weight_loader
    except Exception:
        pass

    def _patch_live_model_weight_loaders(model) -> int:
        patched = 0
        visited = set()

        def _maybe_patch_holder(holder):
            nonlocal patched
            loader = getattr(holder, "weight_loader", None)
            if loader is None:
                return
            wrapped = _wrap_weight_loader(loader)
            if wrapped is not loader:
                try:
                    setattr(holder, "weight_loader", wrapped)
                    patched += 1
                except Exception:
                    pass

        for _name, param in model.named_parameters(recurse=True):
            if id(param) in visited:
                continue
            visited.add(id(param))
            _maybe_patch_holder(param)
        for _name, module in model.named_modules():
            if id(module) in visited:
                continue
            visited.add(id(module))
            _maybe_patch_holder(module)
        return patched

    try:
        from sglang.srt.model_executor import model_runner as sglang_model_runner
        from sglang.srt.model_loader.loader import device_loading_context
        from sglang.srt.model_loader.utils import post_load_weights

        for method_name in ("update_weights_from_disk", "update_weights_from_tensor"):
            original = getattr(sglang_model_runner.ModelRunner, method_name, None)
            if original is None or getattr(original, "_slime_live_weight_loader_compat", False):
                continue

            def _make_wrapper(fn, fn_name):
                def _wrapped(self, *args, **kwargs):
                    patched = _patch_live_model_weight_loaders(self.model)
                    if patched:
                        logger.info("Patched %s live model weight_loader callables before %s.", patched, fn_name)
                    result = fn(self, *args, **kwargs)
                    if fn_name == "update_weights_from_tensor":
                        try:
                            success = bool(result[0]) if isinstance(result, tuple) else bool(result)
                        except Exception:
                            success = False
                        if success:
                            target_device = torch.device(self.device)
                            processed = 0
                            for _, module in self.model.named_modules():
                                quant_method = getattr(module, "quant_method", None)
                                if quant_method is None:
                                    continue
                                with device_loading_context(module, target_device):
                                    quant_method.process_weights_after_loading(module)
                                processed += 1
                            try:
                                post_load_weights(self.model, self.model_config)
                            except Exception:
                                pass
                            logger.info(
                                "Reprocessed %s quantized modules after %s.",
                                processed,
                                fn_name,
                            )
                    return result

                _wrapped._slime_live_weight_loader_compat = True
                return _wrapped

            setattr(sglang_model_runner.ModelRunner, method_name, _make_wrapper(original, method_name))
    except Exception:
        pass

    logger.info("Patched SGLang default_weight_loader to ignore unsupported keyword args.")


def _patch_sglang_multiprocessing_serializer_compat() -> None:
    try:
        import torch
        from sglang.srt import utils as sglang_utils
        from sglang.srt.managers import scheduler_update_weights_mixin as sglang_scheduler_update_weights_mixin
        from sglang.srt.utils import common as sglang_common
        from sglang.srt.managers import tp_worker as sglang_tp_worker
    except Exception:
        return

    serializer_cls = getattr(sglang_utils, "MultiprocessingSerializer", None)
    if serializer_cls is None:
        return

    deserialize = getattr(serializer_cls, "deserialize", None)
    if deserialize is None or getattr(deserialize, "_slime_by_value_compat", False):
        return

    def _deserialize_with_by_value_compat(data):
        if isinstance(data, str):
            try:
                payload = json.loads(data)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("format") == "slime-by-value-flattened-bucket-v1":
                raw = base64.b64decode(payload["tensor_bytes_b64"])
                metadata = pickle.loads(base64.b64decode(payload["metadata_pickle_b64"]))
                flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
                return {
                    "flattened_tensor": flat,
                    "metadata": metadata,
                }
        return deserialize(data)

    _deserialize_with_by_value_compat._slime_by_value_compat = True
    serializer_cls.deserialize = staticmethod(_deserialize_with_by_value_compat)
    common_serializer_cls = getattr(sglang_common, "MultiprocessingSerializer", None)
    if common_serializer_cls is not None:
        common_serializer_cls.deserialize = staticmethod(_deserialize_with_by_value_compat)
    tp_worker_serializer_cls = getattr(sglang_tp_worker, "MultiprocessingSerializer", None)
    if tp_worker_serializer_cls is not None:
        tp_worker_serializer_cls.deserialize = staticmethod(_deserialize_with_by_value_compat)
    logger.info("Patched SGLang MultiprocessingSerializer.deserialize with by-value flattened-bucket support.")

    update_weights_from_tensor = getattr(sglang_tp_worker.TpModelWorker, "update_weights_from_tensor", None)
    if update_weights_from_tensor is not None and not getattr(
        update_weights_from_tensor, "_slime_by_value_compat", False
    ):

        def _tp_worker_update_weights_from_tensor(self, recv_req):
            raw_payload = recv_req.serialized_named_tensors[self.tp_rank]
            if recv_req.load_format == "flattened_bucket_by_value" and isinstance(raw_payload, str):
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("format") == "slime-by-value-flattened-bucket-v1":
                    print("[slime-by-value] tp_worker detected by-value flattened bucket", flush=True)
                    raw = base64.b64decode(payload["tensor_bytes_b64"])
                    metadata = pickle.loads(base64.b64decode(payload["metadata_pickle_b64"]))
                    named_tensors = {
                        "flattened_tensor": torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone(),
                        "metadata": metadata,
                    }
                    print(
                        "[slime-by-value] tp_worker reconstructed flattened bucket "
                        f"bytes={len(raw)} metadata_len={len(metadata)}",
                        flush=True,
                    )
                    return self.model_runner.update_weights_from_tensor(
                        named_tensors=named_tensors,
                        load_format="flattened_bucket",
                    )
            print(
                f"[slime-by-value] tp_worker falling back to original serializer path load_format={recv_req.load_format}",
                flush=True,
            )
            return update_weights_from_tensor(self, recv_req)

        _tp_worker_update_weights_from_tensor._slime_by_value_compat = True
        sglang_tp_worker.TpModelWorker.update_weights_from_tensor = _tp_worker_update_weights_from_tensor
        logger.info("Patched SGLang TpModelWorker.update_weights_from_tensor with by-value flattened-bucket support.")

    scheduler_update_weights_from_tensor = getattr(
        sglang_scheduler_update_weights_mixin.SchedulerUpdateWeightsMixin,
        "update_weights_from_tensor",
        None,
    )
    if scheduler_update_weights_from_tensor is not None and not getattr(
        scheduler_update_weights_from_tensor, "_slime_by_value_compat", False
    ):

        # Preserve the original return type behavior without importing the pydantic req output type here.
        def _scheduler_wrapper(self, recv_req):
            raw_payload = recv_req.serialized_named_tensors[0]
            if recv_req.load_format == "flattened_bucket_by_value" and isinstance(raw_payload, str):
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("format") == "slime-by-value-flattened-bucket-v1":
                    print("[slime-by-value] scheduler detected by-value flattened bucket", flush=True)
                    worker = self.draft_worker or self.tp_worker
                    success, message = worker.update_weights_from_tensor(recv_req)
                    if success:
                        if recv_req.flush_cache:
                            self.flush_cache()
                        if recv_req.weight_version is not None:
                            self.weight_version = recv_req.weight_version
                    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqOutput

                    return UpdateWeightsFromTensorReqOutput(success, message)
            return scheduler_update_weights_from_tensor(self, recv_req)

        _scheduler_wrapper._slime_by_value_compat = True
        sglang_scheduler_update_weights_mixin.SchedulerUpdateWeightsMixin.update_weights_from_tensor = (
            _scheduler_wrapper
        )
        logger.info(
            "Patched SGLang SchedulerUpdateWeightsMixin.update_weights_from_tensor with by-value support."
        )


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server

    _ensure_shared_multiprocessing_authkey()
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=_launch_server_with_patches, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _launch_server_with_patches(server_args: ServerArgs) -> None:
    _ensure_shared_multiprocessing_authkey()
    _patch_sglang_weight_loader_compat()
    _patch_sglang_multiprocessing_serializer_compat()

    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    health_mode = os.environ.get("SLIME_SGLANG_HEALTH_MODE", "generate").strip().lower()
    if health_mode == "basic":
        health_path = "/v1/models"
    else:
        health_path = "/health_generate"

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}{health_path}", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        if os.environ.get("SLIME_SGLANG_SKIP_FLUSH_CACHE", "").strip().lower() in {"1", "true", "yes"}:
            return

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break

            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(self, args, rank: int, worker_type: str = "regular", base_gpu_id: int | None = None):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id

    def init(self, dist_init_addr, port, nccl_port, host=None, disaggregation_bootstrap_port=None):
        self.router_ip = self.args.sglang_router_ip
        self.router_port = self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        if self.node_rank == 0 and self.router_ip and self.router_port:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or slime router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                payload = {
                    "url": f"http://{self.server_host}:{self.server_port}",
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                try:
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
):
    nnodes = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": args.rollout_num_gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
    }

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
