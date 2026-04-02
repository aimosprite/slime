import json
import logging
import os
import re
import threading
import time
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    if _is_existing_nonempty_dir(load_path) and _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )

    if _is_existing_nonempty_dir(load_path):
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    if args.megatron_to_hf_mode == "bridge" and args.hf_checkpoint:
        logger.info(
            "args.load=%r is not a local non-empty checkpoint directory; "
            "falling back to hf_checkpoint=%r for bridge initialization",
            args.load,
            args.hf_checkpoint,
        )
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=args.hf_checkpoint,
        )

    raise AssertionError(f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?")


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        try:
            import slime_plugins.mbridge  # noqa: F401
            from mbridge import AutoBridge as MBridgeAutoBridge

            logger.info("Using mbridge.load_weights(memory_efficient=True) for HF -> Megatron initialization")
            bridge = MBridgeAutoBridge.from_pretrained(load_path, trust_remote_code=True)
            bridge.load_weights(ddp_model, load_path, memory_efficient=True)
            logger.info("Finished HF -> Megatron initialization via mbridge.load_weights(memory_efficient=True)")
        except Exception:
            logger.exception(
                "mbridge memory-efficient load failed; falling back to megatron.bridge.load_hf_weights"
            )
            _load_checkpoint_hf_via_megatron_bridge(ddp_model, load_path)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _load_checkpoint_hf_via_megatron_bridge(ddp_model, load_path: str) -> None:
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info("Constructing megatron.bridge AutoBridge from HF checkpoint...")
    start = time.monotonic()
    bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
    logger.info(
        "Constructed megatron.bridge AutoBridge in %.1fs; starting load_hf_weights().",
        time.monotonic() - start,
    )

    heartbeat_interval_s = int(os.environ.get("SLIME_BRIDGE_LOAD_HEARTBEAT_SECS", "60"))
    done = threading.Event()

    def _heartbeat() -> None:
        while not done.wait(heartbeat_interval_s):
            logger.info(
                "Still running megatron.bridge.load_hf_weights(...) after %.1fs.",
                time.monotonic() - start,
            )

    heartbeat = threading.Thread(target=_heartbeat, name="bridge-load-heartbeat", daemon=True)
    heartbeat.start()

    bridge.load_hf_weights(ddp_model)
    done.set()
    heartbeat.join(timeout=0.1)
    logger.info("Finished HF -> Megatron initialization via megatron.bridge.load_hf_weights")


def _read_hf_weight_map(load_path: str | Path) -> dict[str, str]:
    index_path = Path(load_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}

    try:
        index_data = json.loads(index_path.read_text())
    except Exception:
        logger.exception("Failed to parse HF safetensors index at %s", index_path)
        return {}

    weight_map = index_data.get("weight_map", {})
    return weight_map if isinstance(weight_map, dict) else {}


def _hf_uses_composite_gpt_oss_moe_weights(load_path: str | Path) -> bool:
    """Detect GPT-OSS checkpoints whose expert weights are stored as composite tensors.

    GPT-OSS 120B stores expert tensors as layer-level composite entries such as
    ``model.layers.0.mlp.experts.gate_up_proj_blocks`` rather than per-expert
    ``...experts.<id>.gate_proj.weight`` tensors. The installed Megatron Bridge
    loader knows how to map those names, while the memory-efficient mbridge path
    currently assumes the per-expert layout.
    """

    for name in _read_hf_weight_map(load_path):
        if re.fullmatch(
            r"model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)(?:_(bias|blocks|scales))?",
            name,
        ):
            return True

    return False


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)


def _is_existing_nonempty_dir(path: str | Path | None) -> bool:
    if path is None:
        return False
    path = Path(path)
    return path.is_dir() and _is_dir_nonempty(path)
