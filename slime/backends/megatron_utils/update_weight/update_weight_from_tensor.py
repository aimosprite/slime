import base64
import json
import os
import pickle
from argparse import Namespace
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
import traceback

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        # create the group within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            try:
                print("[bridge-repro-update] pause_generation start", flush=True)
                ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
                print("[bridge-repro-update] pause_generation done", flush=True)
                print("[bridge-repro-update] flush_cache start", flush=True)
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
                print("[bridge-repro-update] flush_cache done", flush=True)
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    print("[bridge-repro-update] pre-load post_process_weights start", flush=True)
                    post_process_weights(
                        restore_weights_before_load=True,
                        post_process_quantization=False,
                        rollout_engines=self.rollout_engines,
                    )
                    print("[bridge-repro-update] pre-load post_process_weights done", flush=True)
            except Exception:
                print("[bridge-repro-update] rank0 pre-update control phase failed", flush=True)
                print(traceback.format_exc(), flush=True)
                raise
        dist.barrier(group=get_gloo_group())

        try:
            print(f"[bridge-repro-update] weights_getter start rank={rank}", flush=True)
            megatron_local_weights = self.weights_getter()
            print(
                f"[bridge-repro-update] weights_getter done rank={rank} count={len(megatron_local_weights)}",
                flush=True,
            )
            if rank == 0:
                sample_items = list(megatron_local_weights.items())[:5]
                for sample_idx, (name, tensor) in enumerate(sample_items):
                    print(
                        "[bridge-repro-update] weight sample "
                        f"idx={sample_idx} name={name} "
                        f"type={type(tensor).__name__} "
                        f"device={tensor.device} dtype={tensor.dtype} "
                        f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                        f"contiguous={tensor.is_contiguous()} "
                        f"pinned={tensor.is_pinned() if tensor.device.type == 'cpu' else 'n/a'}",
                        flush=True,
                    )
                probe_name = "vp_stages.0.decoder.final_layernorm.weight"
                probe_tensor = megatron_local_weights.get(probe_name)
                if probe_tensor is not None:
                    print(
                        "[bridge-repro-update] raw probe start "
                        f"name={probe_name} type={type(probe_tensor).__name__} "
                        f"device={probe_tensor.device} dtype={probe_tensor.dtype} "
                        f"shape={tuple(probe_tensor.shape)} stride={tuple(probe_tensor.stride())} "
                        f"contiguous={probe_tensor.is_contiguous()}",
                        flush=True,
                    )
                    probe_ops = [
                        ("detach", lambda x: x.detach()),
                        ("detach_clone", lambda x: x.detach().clone()),
                        ("detach_cpu", lambda x: x.detach().cpu()),
                        ("detach_float_cpu", lambda x: x.detach().to(dtype=torch.float32).cpu()),
                    ]
                    for probe_label, probe_fn in probe_ops:
                        try:
                            probe_result = probe_fn(probe_tensor)
                            if isinstance(probe_result, torch.Tensor):
                                print(
                                    "[bridge-repro-update] raw probe ok "
                                    f"name={probe_name} op={probe_label} "
                                    f"type={type(probe_result).__name__} "
                                    f"device={probe_result.device} dtype={probe_result.dtype} "
                                    f"shape={tuple(probe_result.shape)}",
                                    flush=True,
                                )
                            else:
                                print(
                                    "[bridge-repro-update] raw probe ok "
                                    f"name={probe_name} op={probe_label} type={type(probe_result).__name__}",
                                    flush=True,
                                )
                        except Exception as probe_exc:
                            print(
                                "[bridge-repro-update] raw probe failed "
                                f"name={probe_name} op={probe_label} "
                                f"exc={type(probe_exc).__name__}: {probe_exc}",
                                flush=True,
                            )
        except Exception:
            print(f"[bridge-repro-update] weights_getter failed rank={rank}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

        try:
            for chunk_idx, hf_named_tensors in enumerate(self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights)):
                if rank == 0:
                    chunk_names = [name for name, _tensor in hf_named_tensors[:3]]
                    print(
                        f"[bridge-repro-update] sending chunk={chunk_idx} rank={rank} "
                        f"chunk_len={len(hf_named_tensors)} sample_names={chunk_names}",
                        flush=True,
                    )
                refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                ray.get(refs)
                if rank == 0:
                    print(f"[bridge-repro-update] sent chunk={chunk_idx} rank={rank}", flush=True)
                del long_lived_tensors
        except Exception:
            print(f"[bridge-repro-update] hf export/send failed rank={rank}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            try:
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    print("[bridge-repro-update] post-load post_process_weights start", flush=True)
                    post_process_weights(
                        restore_weights_before_load=False,
                        post_process_quantization=True,
                        rollout_engines=self.rollout_engines,
                    )
                    print("[bridge-repro-update] post-load post_process_weights done", flush=True)
                print("[bridge-repro-update] continue_generation start", flush=True)
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
                print("[bridge-repro-update] continue_generation done", flush=True)
            except Exception:
                print("[bridge-repro-update] rank0 post-update control phase failed", flush=True)
                print(traceback.format_exc(), flush=True)
                raise
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _serialize_flattened_tensor_data(flattened_tensor_data) -> str:
    if os.environ.get("SLIME_SGLANG_SERIALIZE_BY_VALUE", "").strip().lower() in {"1", "true", "yes"}:
        flat = flattened_tensor_data["flattened_tensor"].detach().cpu().contiguous().view(torch.uint8)
        payload = {
            "format": "slime-by-value-flattened-bucket-v1",
            "tensor_bytes_b64": base64.b64encode(flat.numpy().tobytes()).decode("ascii"),
            "metadata_pickle_b64": base64.b64encode(
                pickle.dumps(flattened_tensor_data["metadata"], protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii"),
        }
        return json.dumps(payload, separators=(",", ":"))

    return MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)


def _by_value_enabled() -> bool:
    return os.environ.get("SLIME_SGLANG_SERIALIZE_BY_VALUE", "").strip().lower() in {"1", "true", "yes"}


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    long_live_tensors = []
    clone_before_bucket = os.environ.get("SLIME_SGLANG_CLONE_BEFORE_BUCKET", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if clone_before_bucket:
        normalized_named_tensors = []
        for name, tensor in hf_named_tensors:
            normalized_tensor = tensor.detach().contiguous().clone()
            normalized_named_tensors.append((name, normalized_tensor))
        hf_named_tensors = normalized_named_tensors

    force_single_dtype_buckets = os.environ.get("SLIME_SGLANG_FORCE_SINGLE_DTYPE_BUCKETS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if dist.get_rank() == ipc_gather_src:
        dtype_counts = Counter(str(tensor.dtype) for _name, tensor in hf_named_tensors)
        device_counts = Counter(str(tensor.device) for _name, tensor in hf_named_tensors)
        contiguous_counts = Counter("contiguous" if tensor.is_contiguous() else "noncontiguous" for _name, tensor in hf_named_tensors)
        print(
            "[bridge-repro-update] colocated send chunk stats "
            f"count={len(hf_named_tensors)} "
            f"dtypes={dict(dtype_counts)} "
            f"devices={dict(device_counts)} "
            f"contiguity={dict(contiguous_counts)} "
            f"force_single_dtype_buckets={force_single_dtype_buckets} "
            f"clone_before_bucket={clone_before_bucket}",
            flush=True,
        )
        weird_samples = [
            (name, tensor)
            for name, tensor in hf_named_tensors
            if tensor.device.type != "cuda" or not tensor.is_contiguous()
        ][:5]
        for sample_idx, (name, tensor) in enumerate(hf_named_tensors[:5]):
            print(
                "[bridge-repro-update] colocated tensor sample "
                f"idx={sample_idx} name={name} device={tensor.device} dtype={tensor.dtype} "
                f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"storage_offset={tensor.storage_offset()} data_ptr={tensor.data_ptr()}",
                flush=True,
            )
        for sample_idx, (name, tensor) in enumerate(weird_samples):
            print(
                "[bridge-repro-update] colocated weird tensor "
                f"idx={sample_idx} name={name} device={tensor.device} dtype={tensor.dtype} "
                f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}",
                flush=True,
            )

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False) and not force_single_dtype_buckets:
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        if dist.get_rank() == ipc_gather_src:
            print(
                "[bridge-repro-update] building flattened bucket "
                f"dtype_group={_dtype} tensor_count={len(named_tensors)} "
                f"sample_names={[name for name, _tensor in named_tensors[:3]]}",
                flush=True,
            )
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(_serialize_flattened_tensor_data(flattened_tensor_data))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket_by_value" if _by_value_enabled() else "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
