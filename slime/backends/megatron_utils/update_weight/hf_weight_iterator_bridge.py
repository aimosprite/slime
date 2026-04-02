import json
import dataclasses
import inspect
import os
import traceback
from pathlib import Path

import torch

from slime_plugins.mbridge.mxfp4_reference import convert_moe_packed_tensors_reference
from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self._hf_snapshot_index = None
        self._hf_reference_cache = {}

    def _find_hf_pretrained(self):
        for attr in ("hf_pretrained", "_hf_pretrained"):
            value = getattr(self._bridge, attr, None)
            if value is not None:
                return value
        return None

    def _load_reference_tensor(self, hf_param_name: str):
        if hf_param_name in self._hf_reference_cache:
            return self._hf_reference_cache[hf_param_name]

        checkpoint_dir = Path(self.args.hf_checkpoint)
        if not checkpoint_dir.is_dir():
            return None

        composite_ref = self._load_composite_reference_tensor(hf_param_name, checkpoint_dir)
        if composite_ref is not None:
            self._hf_reference_cache[hf_param_name] = composite_ref
            return composite_ref

        index = self._hf_snapshot_index
        if index is None:
            index_path = checkpoint_dir / "model.safetensors.index.json"
            if index_path.is_file():
                try:
                    index = json.loads(index_path.read_text())
                except Exception:
                    index = {}
            else:
                index = {}
            self._hf_snapshot_index = index

        weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
        filename = weight_map.get(hf_param_name)
        if not filename:
            return None

        tensor_path = checkpoint_dir / filename
        try:
            from safetensors import safe_open

            with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(hf_param_name)
        except Exception:
            print(
                f"[bridge-repro-hf-iter] reference-load failed name={hf_param_name} file={tensor_path}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            return None

        self._hf_reference_cache[hf_param_name] = tensor
        return tensor

    def _load_tensor_from_snapshot(self, checkpoint_dir: Path, hf_param_name: str):
        index = self._hf_snapshot_index
        if index is None:
            index_path = checkpoint_dir / "model.safetensors.index.json"
            if index_path.is_file():
                try:
                    index = json.loads(index_path.read_text())
                except Exception:
                    index = {}
            else:
                index = {}
            self._hf_snapshot_index = index

        weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
        filename = weight_map.get(hf_param_name)
        if not filename:
            return None

        tensor_path = checkpoint_dir / filename
        from safetensors import safe_open

        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(hf_param_name)

    def _load_composite_reference_tensor(self, hf_param_name: str, checkpoint_dir: Path):
        if hf_param_name.endswith(".mlp.experts.gate_up_proj"):
            blocks_name = f"{hf_param_name}_blocks"
            scales_name = f"{hf_param_name}_scales"
            try:
                blocks = self._load_tensor_from_snapshot(checkpoint_dir, blocks_name)
                scales = self._load_tensor_from_snapshot(checkpoint_dir, scales_name)
                if blocks is None or scales is None:
                    return None
                return convert_moe_packed_tensors_reference(blocks, scales)
            except Exception:
                print(
                    "[bridge-repro-hf-iter] composite-reference-load failed "
                    f"name={hf_param_name} blocks={blocks_name} scales={scales_name}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                return None

        if hf_param_name.endswith(".mlp.experts.down_proj"):
            blocks_name = f"{hf_param_name}_blocks"
            scales_name = f"{hf_param_name}_scales"
            try:
                blocks = self._load_tensor_from_snapshot(checkpoint_dir, blocks_name)
                scales = self._load_tensor_from_snapshot(checkpoint_dir, scales_name)
                if blocks is None or scales is None:
                    return None
                return convert_moe_packed_tensors_reference(blocks, scales)
            except Exception:
                print(
                    "[bridge-repro-hf-iter] composite-reference-load failed "
                    f"name={hf_param_name} blocks={blocks_name} scales={scales_name}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                return None

        return None

    def _export_hf_weights_compat(self, *, cpu: bool, conversion_tasks, merge_adapter_weights: bool):
        export_fn = self._bridge.export_hf_weights
        export_sig = inspect.signature(export_fn)
        export_kwargs = {}
        if "cpu" in export_sig.parameters:
            export_kwargs["cpu"] = cpu
        if "show_progress" in export_sig.parameters:
            export_kwargs["show_progress"] = True
        if "conversion_tasks" in export_sig.parameters:
            export_kwargs["conversion_tasks"] = conversion_tasks

        if "merge_adapter_weights" in export_sig.parameters:
            export_kwargs["merge_adapter_weights"] = merge_adapter_weights
            print("[bridge-repro-hf-iter] using AutoBridge.export_hf_weights(..., merge_adapter_weights=...)", flush=True)
            return export_fn(self.model, **export_kwargs)

        if merge_adapter_weights:
            print(
                "[bridge-repro-hf-iter] AutoBridge.export_hf_weights has no merge_adapter_weights kwarg; "
                "falling back to older default merge behavior",
                flush=True,
            )
            return export_fn(self.model, **export_kwargs)

        model_bridge = getattr(self._bridge, "_model_bridge", None)
        hf_pretrained = self._find_hf_pretrained()
        if model_bridge is None or hf_pretrained is None:
            print(
                "[bridge-repro-hf-iter] merge_adapter_weights=False requested, but direct model_bridge fallback "
                "is unavailable; using default AutoBridge export",
                flush=True,
            )
            return export_fn(self.model, **export_kwargs)

        stream_fn = model_bridge.stream_weights_megatron_to_hf
        stream_sig = inspect.signature(stream_fn)
        stream_kwargs = {}
        if "cpu" in stream_sig.parameters:
            stream_kwargs["cpu"] = cpu
        if "show_progress" in stream_sig.parameters:
            stream_kwargs["show_progress"] = True
        if "conversion_tasks" in stream_sig.parameters:
            stream_kwargs["conversion_tasks"] = conversion_tasks
        if "merge_adapter_weights" in stream_sig.parameters:
            stream_kwargs["merge_adapter_weights"] = False
        print(
            "[bridge-repro-hf-iter] using direct model_bridge.stream_weights_megatron_to_hf "
            "fallback for merge_adapter_weights=False",
            flush=True,
        )
        return stream_fn(self.model, hf_pretrained, **stream_kwargs)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        try:
            print(
                f"[bridge-repro-hf-iter] get_hf_weight_chunks start local_weight_count={len(renamed_megatron_local_weights)}",
                flush=True,
            )
            with megatron_bridge_utils.patch_megatron_model(self.model):
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                print(f"[bridge-repro-hf-iter] conversion_tasks count={len(conversion_tasks)}", flush=True)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                export_cpu = os.environ.get("SLIME_BRIDGE_EXPORT_CPU", "").strip().lower() in {"1", "true", "yes"}
                merge_adapter_weights = (
                    os.environ.get("SLIME_BRIDGE_MERGE_ADAPTER_WEIGHTS", "1").strip().lower()
                    in {"1", "true", "yes"}
                )
                print(f"[bridge-repro-hf-iter] export_hf_weights cpu={export_cpu}", flush=True)
                print(
                    f"[bridge-repro-hf-iter] export_hf_weights merge_adapter_weights={merge_adapter_weights}",
                    flush=True,
                )

                named_weights = self._export_hf_weights_compat(
                    cpu=export_cpu,
                    conversion_tasks=conversion_tasks,
                    merge_adapter_weights=merge_adapter_weights,
                )

                compare_enabled = (
                    os.environ.get("SLIME_BRIDGE_COMPARE_TO_HF", "").strip().lower() in {"1", "true", "yes"}
                )
                compare_limit = int(os.environ.get("SLIME_BRIDGE_COMPARE_LIMIT", "24") or "24")
                compare_fail_fast = (
                    os.environ.get("SLIME_BRIDGE_COMPARE_FAIL_FAST", "").strip().lower() in {"1", "true", "yes"}
                )
                compare_state = {"seen": 0, "fail_fast": compare_fail_fast}

                named_weights = (
                    self._maybe_compare_against_hf_reference(
                        hf_param_name=hf_param_name,
                        megatron_param_name=megatron_param_name,
                        param=postprocess_hf_param(
                            args=self.args,
                            megatron_param_name=megatron_param_name,
                            hf_param_name=hf_param_name,
                            param=weight,
                        ),
                        compare_enabled=compare_enabled,
                        compare_limit=compare_limit,
                        compare_state=compare_state,
                    )
                    for hf_param_name, weight, megatron_param_name in named_weights
                )

                for chunk_idx, chunk in enumerate(
                    chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)
                ):
                    print(
                        f"[bridge-repro-hf-iter] yielding chunk={chunk_idx} chunk_len={len(chunk)} "
                        f"sample_names={[name for name, _ in chunk[:3]]}",
                        flush=True,
                    )
                    yield chunk
        except Exception:
            print("[bridge-repro-hf-iter] get_hf_weight_chunks failed", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def _maybe_compare_against_hf_reference(
        self,
        *,
        hf_param_name: str,
        megatron_param_name: str,
        param,
        compare_enabled: bool,
        compare_limit: int,
        compare_state: dict,
    ):
        if compare_enabled and compare_state["seen"] < compare_limit:
            ref = self._load_reference_tensor(hf_param_name)
            if ref is None:
                if compare_state["seen"] == 0:
                    print(
                        "[bridge-repro-hf-iter] hf-compare missing reference "
                        f"name={hf_param_name} megatron={megatron_param_name}",
                        flush=True,
                    )
                return (hf_param_name, param)
            compare_state["seen"] += 1
            try:
                ref_cpu = ref.detach().to(device="cpu", copy=True)
                param_cpu = param.detach().to(device="cpu", copy=True)
                if ref_cpu.shape != param_cpu.shape:
                    print(
                        "[bridge-repro-hf-iter] hf-compare shape-mismatch "
                        f"name={hf_param_name} megatron={megatron_param_name} "
                        f"exported_shape={tuple(param_cpu.shape)} ref_shape={tuple(ref_cpu.shape)}",
                        flush=True,
                    )
                else:
                    max_abs = (param_cpu.float() - ref_cpu.float()).abs().max().item()
                    exact = bool(torch.equal(param_cpu, ref_cpu))
                    print(
                        "[bridge-repro-hf-iter] hf-compare "
                        f"name={hf_param_name} megatron={megatron_param_name} "
                        f"exact={exact} max_abs={max_abs:.6g}",
                        flush=True,
                    )
                    if hf_param_name.endswith(".mlp.experts.down_proj"):
                        alt_ref_cpu = ref_cpu.transpose(-1, -2).contiguous()
                        if alt_ref_cpu.shape == param_cpu.shape:
                            alt_max_abs = (param_cpu.float() - alt_ref_cpu.float()).abs().max().item()
                            alt_exact = bool(torch.equal(param_cpu, alt_ref_cpu))
                            print(
                                "[bridge-repro-hf-iter] hf-compare-alt "
                                f"name={hf_param_name} megatron={megatron_param_name} "
                                f"alt=transposed_ref exact={alt_exact} max_abs={alt_max_abs:.6g}",
                                flush=True,
                            )
                    if compare_state.get("fail_fast") and compare_state["seen"] >= compare_limit:
                        raise RuntimeError(
                            "SLIME_BRIDGE_COMPARE_FAIL_FAST: collected requested hf-compare samples "
                            f"(limit={compare_limit}); stopping before rollout"
                        )
            except RuntimeError:
                raise
            except Exception:
                print(
                    "[bridge-repro-hf-iter] hf-compare failed "
                    f"name={hf_param_name} megatron={megatron_param_name}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
        return (hf_param_name, param)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    preview_limit = 8
    preview_state = {"seen": 0}
    export_cpu = os.environ.get("SLIME_BRIDGE_EXPORT_CPU", "").strip().lower() in {"1", "true", "yes"}

    def _handle_one(task):
        if task is None:
            return None
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        preview_state["seen"] += 1
        if preview_state["seen"] <= preview_limit:
            print(
                "[bridge-repro-hf-iter] task preview "
                f"idx={preview_state['seen'] - 1} key={weight_dict_key} "
                f"type={type(new_param_weight).__name__} "
                f"src_device={new_param_weight.device} dtype={new_param_weight.dtype} "
                f"shape={tuple(new_param_weight.shape)} stride={tuple(new_param_weight.stride())} "
                f"contiguous={new_param_weight.is_contiguous()} "
                f"pinned={new_param_weight.is_pinned() if new_param_weight.device.type == 'cpu' else 'n/a'}",
                flush=True,
            )
        try:
            if new_param_weight.device.type == "cpu":
                # TensorBackuper stores pinned CPU copies. Normalize them into an ordinary CPU
                # tensor first. If bridge export is already running in cpu=True mode, keep them
                # on CPU end to end instead of forcing a pointless staging hop back onto CUDA.
                pinned_cpu_weight = new_param_weight.detach().contiguous()
                host_weight = torch.empty(
                    pinned_cpu_weight.shape,
                    dtype=pinned_cpu_weight.dtype,
                    device=torch.device("cpu"),
                )
                host_weight.copy_(pinned_cpu_weight, non_blocking=False)
                if export_cpu:
                    new_param_weight = host_weight
                else:
                    new_param_weight = host_weight.to(
                        device=torch.device("cuda", torch.cuda.current_device()),
                        non_blocking=False,
                    )
            else:
                # Normalize same-device live actor weights into plain tensors before Bridge export.
                # The cheap GPT-OSS repro repeatedly reaches export with a healthy CUDA parameter
                # and then dies on the first host copy. Giving Bridge a detached contiguous tensor
                # here is the closest equivalent to the already-working CPU-snapshot path.
                new_param_weight = new_param_weight.detach().contiguous().clone()
                new_param_weight = new_param_weight.to(device=torch.cuda.current_device(), non_blocking=False)
            if preview_state["seen"] <= preview_limit:
                torch.cuda.synchronize()
                print(
                    "[bridge-repro-hf-iter] task transfer ok "
                    f"idx={preview_state['seen'] - 1} key={weight_dict_key} "
                    f"dst_device={new_param_weight.device}",
                    flush=True,
                )
        except Exception:
            print(
                "[bridge-repro-hf-iter] tensor transfer failed "
                f"key={weight_dict_key} device={new_param_weight.device} "
                f"dtype={new_param_weight.dtype} shape={tuple(new_param_weight.shape)} "
                f"stride={tuple(new_param_weight.stride())} "
                f"contiguous={new_param_weight.is_contiguous()} "
                f"pinned={new_param_weight.is_pinned() if new_param_weight.device.type == 'cpu' else 'n/a'}",
                flush=True,
            )
            raise
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
