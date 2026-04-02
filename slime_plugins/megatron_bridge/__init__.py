import inspect
import logging
import os
import re
from dataclasses import is_dataclass, replace
from types import SimpleNamespace

import torch


logger = logging.getLogger(__name__)


def _patch_single_pp_export_path() -> None:
    """Avoid NCCL object collectives in Megatron Bridge when PP size is 1.

    Megatron Bridge's Megatron->HF export path gathers global parameter names
    across the pipeline-parallel group. In our trainer configurations PP is
    commonly 1, so that collective is unnecessary. The stock implementation
    still runs ``all_gather_object`` on the PP group, which is enough to trip a
    CUDA-side object-collective failure in the cheap GPT-OSS 20B bridge repro.
    """

    try:
        from megatron.bridge.models.conversion.model_bridge import (
            MegatronModelBridge,
            _megatron_local_name_to_global,
        )
        from megatron.bridge.models.conversion.utils import extract_sort_key, persistent_buffers
        from megatron.core import parallel_state
        from megatron.core.utils import unwrap_model
    except Exception:
        logger.exception("Failed to import Megatron Bridge ModelBridge components for PP=1 export patching.")
        return

    if getattr(MegatronModelBridge, "_slime_single_pp_export_patch", False):
        return

    original = MegatronModelBridge._megatron_global_param_names_all_pp_ranks

    def _patched(self, megatron_model):
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_size = pp_group.size() if pp_group is not None else 1
        if pp_size > 1:
            return original(self, megatron_model)

        models = unwrap_model(megatron_model)
        if not isinstance(models, list):
            models = [models]

        global_param_names = []
        seen = set()
        for vp_stage, model in enumerate(models):
            config = getattr(model, "config", None)
            for param_name, _ in model.named_parameters():
                global_name = _megatron_local_name_to_global(
                    models=models,
                    config=config,
                    param_name=param_name,
                    vp_stage=vp_stage,
                )
                if global_name not in seen:
                    seen.add(global_name)
                    global_param_names.append(global_name)

            persistent = persistent_buffers(model)
            if hasattr(persistent, "items"):
                buffer_names = [name for name, _ in persistent.items()]
            else:
                buffer_names = list(persistent)
            for buffer_name in buffer_names:
                global_name = _megatron_local_name_to_global(
                    models=models,
                    config=config,
                    param_name=buffer_name,
                    vp_stage=vp_stage,
                )
                if global_name not in seen:
                    seen.add(global_name)
                    global_param_names.append(global_name)

        return sorted(global_param_names, key=extract_sort_key)

    MegatronModelBridge._megatron_global_param_names_all_pp_ranks = _patched
    MegatronModelBridge._slime_single_pp_export_patch = True
    logger.info("Patched Megatron Bridge export path to skip PP object gather when pipeline size is 1.")


def _patch_gpt_oss_bridge() -> None:
    """Patch the installed GPT-OSS bridge to support SequentialMLP expert names.

    The training path uses ``megatron.bridge.AutoBridge`` directly. Our older
    ``slime_plugins/mbridge`` shims are not imported there, so grouped-gemm-off
    runs still rely on the installed GPT-OSS bridge. Older bridge builds know
    about grouped expert params (``mlp.experts.linear_fc*.weight*``) but not the
    per-expert SequentialMLP layout exposed as
    ``mlp.experts.local_experts.*.linear_fc*`` when grouped GEMM is disabled.
    """

    try:
        from megatron.bridge.models.conversion import param_mapping as param_mapping_mod
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import (
            GPTOSSBridge,
            GPTOSSMLPDownProjMapping,
            GPTOSSMLPGateUpProjMapping,
        )
        from megatron.bridge.utils import common_utils as common_utils_mod
    except Exception:
        logger.exception("Failed to import Megatron Bridge GPT-OSS components for patching.")
        return

    if getattr(GPTOSSBridge, "_slime_local_experts_patch", False):
        return

    def _extract_expert_number_from_param(param_name: str) -> int:
        patterns = [
            r"local_experts\.(\d+)",
            r"experts\.(\d+)",
            r"(?:weight|bias)(\d+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        raise ValueError(f"No expert number found in parameter name: {param_name}")

    def _patch_param_mapping_export_helpers() -> None:
        common_utils_mod.extract_expert_number_from_param = _extract_expert_number_from_param
        param_mapping_mod.extract_expert_number_from_param = _extract_expert_number_from_param

        def _gather_from_ep_ranks(self, megatron_weights, megatron_module, hf_param_name):
            if megatron_module is None:
                num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
            else:
                model_config = self._get_config(megatron_module)
                num_experts = model_config.num_moe_experts
                num_experts_per_rank = num_experts // self.ep_size
                num_experts_per_rank = self.broadcast_obj_from_pp_rank(
                    num_experts_per_rank, "num_experts_per_rank"
                )

            global_expert_number = _extract_expert_number_from_param(self.megatron_param)
            local_expert_number = global_expert_number % num_experts_per_rank
            gathered_expert_param_names = [
                re.sub(
                    r"experts\.(\d+)",
                    f"experts.{int(local_expert_number) + num_experts_per_rank * i}",
                    str(hf_param_name),
                )
                for i in range(self.ep_size)
            ]
            assert str(hf_param_name) in gathered_expert_param_names, (
                f"hf_param_name {hf_param_name} not in gathered_expert_param_names {gathered_expert_param_names}"
            )

            gathered_weights = [param_mapping_mod.torch.empty_like(megatron_weights) for _ in range(self.ep_size)]
            param_mapping_mod.torch.distributed.all_gather(gathered_weights, megatron_weights, group=self.ep_group)

            weights_dict = {}
            for i, param_name in enumerate(gathered_expert_param_names):
                if param_name in weights_dict:
                    weights_dict[param_name] = param_mapping_mod.torch.cat(
                        [weights_dict[param_name], gathered_weights[i].unsqueeze(0)], dim=0
                    )
                else:
                    weights_dict[param_name] = gathered_weights[i].unsqueeze(0)
            for param_name in weights_dict:
                weights_dict[param_name] = weights_dict[param_name].squeeze()
            return weights_dict

        patched = False
        for _, cls in inspect.getmembers(param_mapping_mod, inspect.isclass):
            if hasattr(cls, "gather_from_ep_ranks"):
                try:
                    setattr(cls, "gather_from_ep_ranks", _gather_from_ep_ranks)
                    patched = True
                except Exception:
                    logger.debug("Skipping gather_from_ep_ranks patch for %s", cls, exc_info=True)
        if not patched:
            logger.warning("Did not find a Megatron Bridge mapping class with gather_from_ep_ranks to patch.")

    _patch_param_mapping_export_helpers()

    def _mapping_registry_with_local_experts(self) -> MegatronMappingRegistry:
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.self_attn.o_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.sinks": "decoder.layers.*.self_attention.core_attention.softmax_offset",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.mlp.router.bias": "decoder.layers.*.mlp.router.bias",
            "model.layers.*.mlp.router.weight": "decoder.layers.*.mlp.router.weight",
        }

        mapping_list = [
            AutoMapping(hf_param=hf_param, megatron_param=megatron_param)
            for hf_param, megatron_param in param_mappings.items()
        ]
        mapping_list.extend(
            [
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                ),
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.bias*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.bias*",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.bias",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.bias",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)

    GPTOSSBridge.mapping_registry = _mapping_registry_with_local_experts
    GPTOSSBridge._slime_local_experts_patch = True
    logger.info("Patched Megatron Bridge GPT-OSS mapping_registry with SequentialMLP local_experts support.")


def _patch_export_failure_tracing() -> None:
    """Add env-gated tracing around Megatron Bridge Megatron->HF export.

    The cheap GPT-OSS repro has already shown that even ``cpu=True`` export can
    fail inside Megatron Bridge on ``tensor.cpu()`` before SGLang sees a single
    HF tensor. This patch keeps the same export logic but adds task-level logs so
    we can identify the first failing Megatron/HF parameter pair.
    """

    try:
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge
        from megatron.core.utils import unwrap_model
    except Exception:
        logger.exception("Failed to import Megatron Bridge components for export tracing patch.")
        return

    if getattr(MegatronModelBridge, "_slime_export_failure_trace_patch", False):
        return

    original = MegatronModelBridge.stream_weights_megatron_to_hf
    original_params = inspect.signature(original).parameters
    hf_weight_tuple_params = inspect.signature(HFWeightTuple).parameters

    def _make_hf_weight_tuple(hf_name: str, tensor, *, megatron_param_name: str):
        kwargs = {}
        if "name" in hf_weight_tuple_params:
            kwargs["name"] = hf_name
        elif "hf_name" in hf_weight_tuple_params:
            kwargs["hf_name"] = hf_name
        else:
            first_param = next(iter(hf_weight_tuple_params))
            kwargs[first_param] = hf_name

        if "tensor" in hf_weight_tuple_params:
            kwargs["tensor"] = tensor
        elif "weight" in hf_weight_tuple_params:
            kwargs["weight"] = tensor
        else:
            remaining = [name for name in hf_weight_tuple_params if name not in kwargs]
            if not remaining:
                raise TypeError("HFWeightTuple signature has no tensor field")
            kwargs[remaining[0]] = tensor

        if "megatron_param_name" in hf_weight_tuple_params:
            kwargs["megatron_param_name"] = megatron_param_name

        return HFWeightTuple(**kwargs)

    def _yield_from_original(
        self,
        megatron_model,
        hf_pretrained,
        *,
        cpu: bool,
        show_progress: bool,
        conversion_tasks,
        merge_adapter_weights: bool,
    ):
        model_for_tasks = megatron_model
        if not isinstance(model_for_tasks, list):
            model_for_tasks = [model_for_tasks]

        if not hasattr(self, "hf_config"):
            fallback_hf_config = getattr(hf_pretrained, "config", None)
            if fallback_hf_config is not None:
                self.hf_config = fallback_hf_config
            else:
                logger.warning(
                    "Megatron Bridge export patch could not seed missing hf_config on %s in original export path; "
                    "compatibility behavior may still be incomplete.",
                    type(self).__name__,
                )

        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, model_for_tasks)

        sanitized_tasks = []
        skipped_null = 0
        skipped_null_mapping = 0
        for task_idx, task in enumerate(conversion_tasks):
            if task is None:
                skipped_null += 1
                print(
                    f"[bridge-repro-export] sanitize: dropping null conversion task idx={task_idx}",
                    flush=True,
                )
                continue
            if getattr(task, "mapping", None) is None:
                skipped_null_mapping += 1
                print(
                    "[bridge-repro-export] sanitize: dropping conversion task with null mapping "
                    f"idx={task_idx} global={_task_name(task)} local={_task_local_name(task)}",
                    flush=True,
                )
                continue
            sanitized_tasks.append(task)

        if skipped_null or skipped_null_mapping:
            print(
                "[bridge-repro-export] sanitize summary "
                f"kept={len(sanitized_tasks)} dropped_null={skipped_null} "
                f"dropped_null_mapping={skipped_null_mapping}",
                flush=True,
            )

        kwargs = {}
        if "cpu" in original_params:
            kwargs["cpu"] = cpu
        if "show_progress" in original_params:
            kwargs["show_progress"] = show_progress
        if "conversion_tasks" in original_params:
            kwargs["conversion_tasks"] = sanitized_tasks
        if "merge_adapter_weights" in original_params:
            kwargs["merge_adapter_weights"] = merge_adapter_weights
        yield from original(self, megatron_model, hf_pretrained, **kwargs)

    def _trace_enabled() -> bool:
        return os.environ.get("SLIME_BRIDGE_TRACE_EXPORT_FAILURE", "").strip().lower() in {"1", "true", "yes"}

    def _should_log_task(task_idx: int) -> bool:
        return task_idx < 24 or task_idx % 100 == 0

    def _task_name(task) -> str:
        for attr in ("global_param_name", "param_name", "name"):
            value = getattr(task, attr, None)
            if value:
                return str(value)
        return "<unknown-task>"

    def _task_local_name(task) -> str:
        value = getattr(task, "param_name", None)
        if value:
            return str(value)
        return "<unknown-local>"

    def _tensor_desc(tensor) -> str:
        try:
            return (
                f"device={tensor.device} dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                f"stride={tuple(tensor.stride())} contiguous={tensor.is_contiguous()}"
            )
        except Exception as exc:
            return f"<tensor-desc-failed {type(exc).__name__}: {exc}>"

    def _embeddings_are_tied(model_bridge, model_config) -> bool:
        helper = getattr(model_bridge, "_share_embeddings_and_output_weights", None)
        if callable(helper):
            return bool(helper(model_config))
        if hasattr(model_config, "tie_word_embeddings"):
            return bool(model_config.tie_word_embeddings)
        if hasattr(model_config, "untie_embeddings_and_output_weights"):
            return not bool(model_config.untie_embeddings_and_output_weights)
        logger.warning(
            "Megatron Bridge export patch could not determine whether embeddings are tied; "
            "defaulting to untied embeddings for compatibility."
        )
        return False

    def _should_skip_duplicate_embedding_export(model_bridge, task, megatron_model) -> bool:
        helper = getattr(model_bridge, "_should_skip_mtp_duplicate_embedding_export", None)
        if callable(helper):
            return bool(helper(task, megatron_model))
        return False

    def _maybe_modify_hf_weight(model_bridge, task, converted_weights_dict, hf_state_dict):
        helper = getattr(model_bridge, "maybe_modify_converted_hf_weight", None)
        if callable(helper):
            param_names = list(inspect.signature(helper).parameters)
            num_params = len(param_names)
            first_name = param_names[0] if param_names else None
            second_name = param_names[1] if num_params >= 2 else None
            state_names = {"hf_state_dict", "state_dict", "hf_state"}
            converted_names = {
                "converted_weights_dict",
                "converted_weights",
                "hf_weights",
                "weights_dict",
            }
            task_names = {"task", "conversion_task"}

            candidates = []

            def _add_candidate(*args):
                if len(args) != num_params:
                    return
                if args not in candidates:
                    candidates.append(args)

            if first_name in task_names:
                _add_candidate(task, converted_weights_dict, hf_state_dict)
                _add_candidate(task, converted_weights_dict)
                _add_candidate(task)
            elif first_name in converted_names:
                if second_name in state_names:
                    _add_candidate(converted_weights_dict, hf_state_dict)
                _add_candidate(converted_weights_dict)
                _add_candidate(task, converted_weights_dict)
            else:
                if num_params == 2 and second_name not in state_names:
                    _add_candidate(task, converted_weights_dict)
                    _add_candidate(converted_weights_dict, hf_state_dict)
                else:
                    _add_candidate(task, converted_weights_dict, hf_state_dict)
                    _add_candidate(task, converted_weights_dict)
                    _add_candidate(converted_weights_dict, hf_state_dict)
                _add_candidate(converted_weights_dict)
                _add_candidate(task)
                _add_candidate()

            last_exc = None
            for args in candidates:
                try:
                    return helper(*args)
                except TypeError as exc:
                    last_exc = exc
                    continue
                except AttributeError as exc:
                    msg = str(exc)
                    if "'dict' object has no attribute 'param_name'" in msg:
                        last_exc = exc
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
            return helper()
        return converted_weights_dict

    def _patched(
        self,
        megatron_model,
        hf_pretrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks=None,
        merge_adapter_weights: bool = True,
    ):
        if not _trace_enabled():
            yield from _yield_from_original(
                self,
                megatron_model,
                hf_pretrained,
                cpu=cpu,
                show_progress=show_progress,
                conversion_tasks=conversion_tasks,
                merge_adapter_weights=merge_adapter_weights,
            )
            return

        def _cpu_copy_tensor(task_idx: int, task, hf_name: str, tensor, grouped: bool = False):
            if not cpu:
                return tensor
            try:
                return tensor.cpu()
            except Exception as exc:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    print(
                        "[bridge-repro-export] cpu-copy retry via clone "
                        f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                        f"{_tensor_desc(tensor)}",
                        flush=True,
                    )
                    clone_exc = None
                    try:
                        cloned = tensor.detach().clone()
                        return cloned.cpu()
                    except Exception as retry_exc:
                        clone_exc = retry_exc
                        print(
                            "[bridge-repro-export] clone retry failed "
                            f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                            f"exc={type(retry_exc).__name__}: {retry_exc}",
                            flush=True,
                        )
                    print(
                        "[bridge-repro-export] cpu-copy retry via float32 host transfer "
                        f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                        f"{_tensor_desc(tensor)}",
                        flush=True,
                    )
                    try:
                        float_cpu = tensor.detach().to(dtype=torch.float32).cpu()
                        return float_cpu.to(dtype=tensor.dtype)
                    except Exception as float_exc:
                        extra = (
                            f" clone_exc={type(clone_exc).__name__}: {clone_exc}"
                            if clone_exc is not None
                            else ""
                        )
                        print(
                            "[bridge-repro-export] float32 host-transfer failed "
                            f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                            f"exc={type(float_exc).__name__}: {float_exc}{extra}",
                            flush=True,
                        )
                label = "grouped cpu-copy failed" if grouped else "cpu-copy failed"
                raise RuntimeError(
                    f"{label} "
                    f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                    f"{_tensor_desc(tensor)}"
                ) from exc

        def _wrap_task_mapping(task_idx: int, task):
            if task is None:
                return None

            class _LoggingMappingProxy:
                def __init__(self, mapping):
                    self._mapping = mapping

                def __getattr__(self, name):
                    return getattr(self._mapping, name)

                def megatron_to_hf(self, megatron_weights, megatron_module):
                    if _should_log_task(task_idx):
                        print(
                            "[bridge-repro-export] task start "
                            f"idx={task_idx} global={_task_name(task)} local={_task_local_name(task)} "
                            f"mapping={self._mapping.__class__.__name__} "
                            f"param_weight={_tensor_desc(megatron_weights) if megatron_weights is not None else 'None'}",
                            flush=True,
                        )
                    converted = self._mapping.megatron_to_hf(megatron_weights, megatron_module)
                    if _should_log_task(task_idx):
                        print(
                            "[bridge-repro-export] task converted "
                            f"idx={task_idx} global={_task_name(task)} "
                            f"hf_names={list(converted.keys())[:8]}",
                            flush=True,
                        )
                    return converted

            mapping_proxy = _LoggingMappingProxy(task.mapping)
            if is_dataclass(task):
                return replace(task, mapping=mapping_proxy)
            try:
                task_copy = SimpleNamespace(**vars(task))
                task_copy.mapping = mapping_proxy
                return task_copy
            except Exception:
                try:
                    task.mapping = mapping_proxy
                    return task
                except Exception:
                    return task

        if merge_adapter_weights and not hasattr(self, "build_adapter_conversion_tasks"):
            if conversion_tasks is None:
                conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
            traced_tasks = []
            for task_idx, task in enumerate(conversion_tasks):
                wrapped_task = _wrap_task_mapping(task_idx, task)
                if wrapped_task is not None:
                    traced_tasks.append(wrapped_task)
            print(
                "[bridge-repro-export] adapter tracing unavailable on this Megatron Bridge build; "
                "keeping original adapter merge path but wrapping base conversion tasks for logging",
                flush=True,
            )
            yield from _yield_from_original(
                self,
                megatron_model,
                hf_pretrained,
                cpu=cpu,
                show_progress=show_progress,
                conversion_tasks=traced_tasks,
                merge_adapter_weights=merge_adapter_weights,
            )
            return

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        adapter_tasks_by_base = {}
        if merge_adapter_weights:
            adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)

        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = _embeddings_are_tied(self, model_config)
        hf_state_dict = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        if not hasattr(self, "hf_config"):
            fallback_hf_config = getattr(hf_pretrained, "config", None)
            if fallback_hf_config is not None:
                self.hf_config = fallback_hf_config
            else:
                logger.warning(
                    "Megatron Bridge export patch could not seed missing hf_config on %s; "
                    "compatibility behavior may still be incomplete.",
                    type(self).__name__,
                )

        grouped_buffers = {}

        for task_idx, task in enumerate(
            self._with_progress_tracking(conversion_tasks, "Converting to HuggingFace", show_progress)
        ):
            if task is None:
                print(
                    f"[bridge-repro-export] skipping null conversion task idx={task_idx}",
                    flush=True,
                )
                continue
            if getattr(task, "mapping", None) is None:
                print(
                    "[bridge-repro-export] skipping conversion task with null mapping "
                    f"idx={task_idx} global={_task_name(task)} local={_task_local_name(task)}",
                    flush=True,
                )
                continue
            task_global_name = _task_name(task)
            if _should_log_task(task_idx):
                print(
                    "[bridge-repro-export] task start "
                    f"idx={task_idx} global={task_global_name} local={_task_local_name(task)} "
                    f"mapping={task.mapping.__class__.__name__} "
                    f"param_weight={_tensor_desc(task.param_weight) if task.param_weight is not None else 'None'}",
                    flush=True,
                )

            megatron_weights = task.param_weight
            megatron_module = task.megatron_module
            if _should_skip_duplicate_embedding_export(self, task, megatron_model):
                megatron_weights = None
                megatron_module = None

            converted_weights_dict = task.mapping.megatron_to_hf(megatron_weights, megatron_module)

            if getattr(task.mapping, "is_grouped_export", False):
                merged_result = self._accumulate_grouped_export(
                    task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict
                )
                if merged_result is not None:
                    for hf_name, tensor in merged_result.items():
                        if _should_log_task(task_idx):
                            print(
                                "[bridge-repro-export] grouped cpu-copy "
                                f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                                f"{_tensor_desc(tensor)}",
                                flush=True,
                            )
                        try:
                            final_tensor = _cpu_copy_tensor(task_idx, task, hf_name, tensor, grouped=True)
                        except Exception as exc:
                            print(
                                "[bridge-repro-export] grouped cpu-copy FAILED "
                                f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                                f"{_tensor_desc(tensor)}",
                                flush=True,
                            )
                            raise exc
                        yield _make_hf_weight_tuple(
                            hf_name,
                            final_tensor,
                            megatron_param_name=task_global_name,
                        )
                continue

            converted_weights_dict = _maybe_modify_hf_weight(self, task, converted_weights_dict, hf_state_dict)

            adapter_tasks = None
            if merge_adapter_weights and "to_wrap.weight" in task_global_name:
                task_global_base_prefix, _, _ = task_global_name.partition(".to_wrap.weight")
                adapter_tasks = adapter_tasks_by_base.get(task_global_base_prefix)
            if merge_adapter_weights and adapter_tasks:
                adapter_weights = self.materialize_adapter_weights(adapter_tasks)
                converted_weights_dict = self._merge_lora_adapter_weights(
                    megatron_model,
                    converted_weights_dict,
                    adapter_weights,
                )

            if _should_log_task(task_idx):
                print(
                    "[bridge-repro-export] task converted "
                    f"idx={task_idx} global={task_global_name} "
                    f"hf_names={list(converted_weights_dict.keys())[:8]}",
                    flush=True,
                )

            for hf_name, tensor in converted_weights_dict.items():
                if _should_log_task(task_idx):
                    print(
                        "[bridge-repro-export] cpu-copy "
                        f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                        f"{_tensor_desc(tensor)}",
                        flush=True,
                    )
                try:
                    final_tensor = _cpu_copy_tensor(task_idx, task, hf_name, tensor)
                except Exception as exc:
                    print(
                        "[bridge-repro-export] cpu-copy FAILED "
                        f"idx={task_idx} global={_task_name(task)} hf={hf_name} "
                        f"{_tensor_desc(tensor)}",
                        flush=True,
                    )
                    raise exc

                if not merge_adapter_weights and "to_wrap.weight" in task_global_name:
                    suffix_pos = hf_name.rfind(".")
                    if suffix_pos == -1:
                        hf_name = hf_name + ".base_layer"
                    else:
                        hf_name = hf_name[:suffix_pos] + ".base_layer" + hf_name[suffix_pos:]

                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    yield _make_hf_weight_tuple(
                        hf_name,
                        final_tensor,
                        megatron_param_name=task_global_name,
                    )
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            yield _make_hf_weight_tuple(
                                "lm_head.weight",
                                final_tensor.clone().detach(),
                                megatron_param_name=task_global_name,
                            )
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    raise ValueError(
                        "Encountered lm_head.weight when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    yield _make_hf_weight_tuple(
                        hf_name,
                        final_tensor,
                        megatron_param_name=task_global_name,
                    )

    MegatronModelBridge.stream_weights_megatron_to_hf = _patched
    MegatronModelBridge._slime_export_failure_trace_patch = True
    logger.info("Patched Megatron Bridge export path with env-gated task-level failure tracing.")


_patch_gpt_oss_bridge()
_patch_single_pp_export_path()
_patch_export_failure_tracing()
