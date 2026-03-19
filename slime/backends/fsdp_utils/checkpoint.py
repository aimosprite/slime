from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    get_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Replicate

logger = logging.getLogger(__name__)


MODEL_INDEX = "model.safetensors.index.json"
MODEL_SINGLE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
LR_SCHEDULER_FILE = "lr_scheduler.pt"


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _materialize_state_value(value: Any) -> Any:
    if not torch.is_tensor(value):
        return value

    tensor = value
    if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim, async_op=True).to_local()
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()

    return tensor.detach().cpu().contiguous()


def _collect_full_model_state_dict(model: torch.nn.Module) -> dict[str, Any] | None:
    full_state: dict[str, Any] = {} if dist.get_rank() == 0 else None
    for name, value in model.state_dict().items():
        materialized = _materialize_state_value(value)
        if full_state is not None:
            full_state[name] = materialized
    return full_state


def _load_local_model_state_dict(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / MODEL_INDEX
    single_path = model_dir / MODEL_SINGLE

    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        state_dict: dict[str, Any] = {}
        for shard_name in sorted(set(weight_map.values())):
            state_dict.update(load_file(model_dir / shard_name, device="cpu"))
        return state_dict

    if single_path.exists():
        return load_file(single_path, device="cpu")

    raise FileNotFoundError(f"No local safetensors checkpoint found under {model_dir}")


def _has_checkpoint_payload(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file():
            return True
        if child.is_dir() and any(grandchild.is_file() for grandchild in child.iterdir()):
            return True
    return False


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    local_model_exists = (model_dir / MODEL_INDEX).exists() or (model_dir / MODEL_SINGLE).exists()
    if local_model_exists:
        try:
            full_state = _load_local_model_state_dict(model_dir) if dist.get_rank() == 0 else {}
            actor.model = actor._fsdp2_load_full_state_dict(
                actor.model,
                full_state,
                actor.dp_mesh,
                cpu_offload=True if getattr(actor, "fsdp_cpu_offload", False) else None,
            )
            logger.info(f"[FSDP] Loaded local model from {model_dir}")
        except Exception as e:
            logger.error(f"[FSDP] Failed to load local model from {model_dir}: {e}")
            return None
    else:
        model_state = ModelState(actor.model)
        state_dict = {"model_state": model_state}

        try:
            dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
            logger.info(f"[FSDP] Loaded model from {model_dir}")
        except Exception as e:
            logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
            return None

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    local_optimizer_file = optimizer_dir / OPTIMIZER_FILE
    if load_optimizer and local_optimizer_file.exists():
        try:
            optim_state = torch.load(local_optimizer_file, map_location="cpu", weights_only=False)
            options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
            set_optimizer_state_dict(actor.model, actor.optimizer, optim_state, options=options)
            logger.info(f"[FSDP] Loaded optimizer from {local_optimizer_file}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {local_optimizer_file}: {e}")
    elif load_optimizer and _has_checkpoint_payload(optimizer_dir):
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except BaseException as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    local_lr_scheduler_file = lr_scheduler_dir / LR_SCHEDULER_FILE
    load_lr_scheduler = hasattr(actor, "lr_scheduler") and _has_checkpoint_payload(lr_scheduler_dir)
    if load_lr_scheduler and local_lr_scheduler_file.exists():
        try:
            actor.lr_scheduler.load_state_dict(torch.load(local_lr_scheduler_file, map_location="cpu", weights_only=False))
            logger.info(f"[FSDP] Loaded LR scheduler from {local_lr_scheduler_file}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {local_lr_scheduler_file}: {e}")
    elif load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
        except BaseException as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[FSDP] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, iteration: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        # Re-running into an existing iteration directory is fragile with torch.save():
        # stale rng.pt / metadata from a prior failed attempt can trip inline_container
        # errors during overwrite. Start each save from a clean directory instead.
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = _collect_full_model_state_dict(actor.model)
    if dist.get_rank() == 0:
        save_torch_state_dict(
            model_state,
            model_dir,
            filename_pattern="model{suffix}.safetensors",
            safe_serialization=True,
            max_shard_size=os.environ.get("SLIME_FSDP_MAX_SHARD_SIZE", "5GB"),
        )
    del model_state
    dist.barrier()

    # Save optimizer state (skip if --no-save-optim is set)
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        try:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            optimizer_state = get_optimizer_state_dict(actor.model, actor.optimizer, options=options)
            if dist.get_rank() == 0:
                torch.save(optimizer_state, optimizer_dir / OPTIMIZER_FILE)
        except Exception as e:
            logger.warning(f"[FSDP] Failed to save optimizer state to {optimizer_dir}: {e}")

    # Save LR scheduler state (skip if --no-save-optim is set)
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        if dist.get_rank() == 0:
            torch.save(actor.lr_scheduler.state_dict(), lr_scheduler_dir / LR_SCHEDULER_FILE)

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        rng_tmp = checkpoint_dir / "rng.pt.tmp"
        torch.save(rng_state, rng_tmp)
        rng_tmp.replace(checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
