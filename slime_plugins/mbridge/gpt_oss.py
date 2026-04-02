import inspect
import logging
import math
import os
import re
from collections import OrderedDict

import torch
from mbridge.core import register_model
from mbridge.models import Qwen2Bridge, Qwen2MoEBridge

from .mxfp4_reference import convert_moe_packed_tensors_reference

logger = logging.getLogger(__name__)


@register_model("gpt_oss")
class GptOssBridge(Qwen2MoEBridge):
    """
    Bridge implementation for GPT-OSS models.

    Handles weight conversion between preprocessed GPT-OSS HF format
    (BF16 per-expert) and Megatron-Core.

    Key differences from Qwen2MoE:
    - All layers are MoE (no dense layers, no shared expert)
    - Has learnable softmax offset (sinks)
    - Has attention bias (q/k/v/o_proj.bias)
    - Has router bias
    - Has expert bias (gate/up/down_proj.bias)
    """

    _ATTENTION_MAPPING = {
        **(Qwen2Bridge._ATTENTION_MAPPING),
        "self_attention.linear_proj.bias": ["model.layers.{layer_number}.self_attn.o_proj.bias"],
        "self_attention.core_attention.softmax_offset": ["model.layers.{layer_number}.self_attn.sinks"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        # GPT-OSS expert tensors are stored as layer-level composites.
        "mlp.experts.linear_fc1.bias": ["model.layers.{layer_number}.mlp.experts.gate_up_proj_bias"],
        "mlp.experts.linear_fc2.bias": ["model.layers.{layer_number}.mlp.experts.down_proj_bias"],
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_blocks",
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_scales",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.down_proj_blocks",
            "model.layers.{layer_number}.mlp.experts.down_proj_scales",
        ],
    }

    _COMPOSITE_HF_CACHE_SIZE = int(os.environ.get("SLIME_MBRIDGE_COMPOSITE_HF_CACHE_SIZE", "4"))

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        elif "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Override to handle expert bias names correctly.

        Base class extracts expert_id by splitting on 'weight', which fails
        for bias parameters. We extract expert_id from after 'bias' as well.
        """
        name = name.replace(".to_wrap", "")
        name = name.replace(".base_linear", "")
        layer_number = name.split(".")[2]

        if ".mlp.experts.local_experts." in name:
            parts = name.split(".")
            expert_idx = parts.index("local_experts")
            expert_id = parts[expert_idx + 1]
            suffix = ".".join(parts[expert_idx + 2 :])

            local_expert_mapping = {
                "linear_fc1.bias": self._MLP_MAPPING["mlp.experts.linear_fc1.bias"],
                "linear_fc2.bias": self._MLP_MAPPING["mlp.experts.linear_fc2.bias"],
                "linear_fc1.weight": self._MLP_MAPPING["mlp.experts.linear_fc1.weight"],
                "linear_fc2.weight": self._MLP_MAPPING["mlp.experts.linear_fc2.weight"],
            }

            if suffix not in local_expert_mapping:
                raise NotImplementedError(f"Unsupported local expert parameter name: {name}")

            return [x.format(layer_number=layer_number, expert_id=expert_id) for x in local_expert_mapping[suffix]]

        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    # Extract expert_id from end of name (after weight/bias)
                    if "weight" in name.split(".")[-1]:
                        expert_id = name.split("weight")[-1]
                    elif "bias" in name.split(".")[-1]:
                        expert_id = name.split("bias")[-1]
                    else:
                        raise ValueError(f"Cannot extract expert_id from: {name}")
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported MLP parameter name: {name}")
        return convert_names

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """Override mbridge loading so GPT-OSS composite expert tensors are cached and sliced correctly."""

        self.safetensor_io = self._get_safetensor_io(weights_path)

        composite_cache: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()

        for model in models:
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }

            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self._should_use_ep_composite_scatter(local_name, hf_names):
                        if self.mpu.ep_rank == 0 and self.mpu.etp_rank == 0:
                            to_load_from_disk.extend(hf_names)
                    elif self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    elif "lm_head.weight" in hf_names:
                        to_load_from_disk.extend(hf_names)

            hf_weights_map = None
            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(to_load_from_disk)

            for local_name, hf_names in local_to_hf_map.items():
                global_name = local_to_global_map[local_name]
                param = model.state_dict()[local_name]

                if self._should_use_ep_composite_scatter(local_name, hf_names):
                    param_to_load = self._load_composite_expert_param_via_ep_scatter(
                        local_name=local_name,
                        global_name=global_name,
                        param=param,
                        hf_names=hf_names,
                        hf_weights_map=hf_weights_map,
                        composite_cache=composite_cache,
                    )
                    param.copy_(param_to_load)
                    continue

                if set(to_load_from_disk) & set(hf_names):
                    if self._uses_composite_expert_tensor(global_name, hf_names):
                        cache_key = tuple(hf_names)
                        if cache_key in composite_cache:
                            hf_weights = composite_cache.pop(cache_key)
                            composite_cache[cache_key] = hf_weights
                        else:
                            hf_weights = self._load_composite_hf_tensor(hf_names, hf_weights_map)
                            if self.mpu.pp_rank == 0 and self.mpu.tp_rank == 0 and self.mpu.ep_rank == 0:
                                logger.info("Loaded GPT-OSS composite expert tensor: %s", hf_names[0])
                            composite_cache[cache_key] = hf_weights
                            while len(composite_cache) > self._COMPOSITE_HF_CACHE_SIZE:
                                composite_cache.popitem(last=False)
                    else:
                        if memory_efficient:
                            hf_weights = [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
                        else:
                            assert hf_weights_map is not None
                            hf_weights = [hf_weights_map[x] for x in hf_names]

                    mcore_weight = self._weight_to_mcore_format(global_name, hf_weights)
                else:
                    mcore_weight = None

                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in global_name:
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            global_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            global_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )

                param.copy_(param_to_load)

    def _should_use_ep_composite_scatter(self, local_name: str, hf_names: list[str]) -> bool:
        return (
            ".mlp.experts.linear_fc" in local_name
            and self.mpu.ep_size > 1
            and self.mpu.etp_size == 1
            and any(".mlp.experts." in hf_name for hf_name in hf_names)
        )

    def _load_composite_expert_param_via_ep_scatter(
        self,
        *,
        local_name: str,
        global_name: str,
        param: torch.Tensor,
        hf_names: list[str],
        hf_weights_map: dict[str, torch.Tensor] | None,
        composite_cache: OrderedDict[tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        cache_key = tuple(hf_names)
        local_expert_id = self._extract_expert_number(local_name)
        param_to_load = torch.empty_like(param)

        if self.mpu.ep_rank == 0:
            if cache_key in composite_cache:
                composite = composite_cache.pop(cache_key)
                composite_cache[cache_key] = composite
            else:
                composite = self._load_composite_hf_tensor(hf_names, hf_weights_map)
                if self.mpu.pp_rank == 0 and self.mpu.tp_rank == 0:
                    logger.info("Loaded GPT-OSS composite expert tensor for EP scatter: %s", hf_names[0])
                composite_cache[cache_key] = composite
                while len(composite_cache) > self._COMPOSITE_HF_CACHE_SIZE:
                    composite_cache.popitem(last=False)

            num_experts = getattr(self.config, "num_moe_experts", self.hf_config.num_local_experts)
            num_experts_per_rank = num_experts // self.mpu.ep_size
            mcore_weights_ep_split = []
            for ep_rank in range(self.mpu.ep_size):
                global_expert_id = ep_rank * num_experts_per_rank + local_expert_id
                expert_weight = self._weight_to_mcore_format_for_expert(global_name, composite, global_expert_id)
                mcore_weights_ep_split.append(expert_weight.to(param.device, dtype=param.dtype).contiguous())
        else:
            mcore_weights_ep_split = None

        torch.distributed.scatter(
            param_to_load,
            mcore_weights_ep_split,
            src=torch.distributed.get_global_rank(self.mpu.ep_group, 0),
            group=self.mpu.ep_group,
        )
        return param_to_load

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights) -> torch.Tensor:
        if ".mlp.experts." not in mcore_weights_name:
            return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

        if "linear_fc1.weight" in mcore_weights_name:
            composite = self._materialize_composite_expert_tensor(hf_weights, "gate_up_proj")
            return self._weight_to_mcore_format_for_expert(
                mcore_weights_name, composite, self._extract_expert_number(mcore_weights_name)
            )

        if "linear_fc1.bias" in mcore_weights_name:
            composite = self._materialize_composite_expert_tensor(hf_weights, "gate_up_proj_bias")
            return self._weight_to_mcore_format_for_expert(
                mcore_weights_name, composite, self._extract_expert_number(mcore_weights_name)
            )

        if "linear_fc2.weight" in mcore_weights_name:
            composite = self._materialize_composite_expert_tensor(hf_weights, "down_proj")
            return self._weight_to_mcore_format_for_expert(
                mcore_weights_name, composite, self._extract_expert_number(mcore_weights_name)
            )

        if "linear_fc2.bias" in mcore_weights_name:
            composite = self._materialize_composite_expert_tensor(hf_weights, "down_proj_bias")
            return self._weight_to_mcore_format_for_expert(
                mcore_weights_name, composite, self._extract_expert_number(mcore_weights_name)
            )

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_mcore_format_for_expert(
        self, mcore_weights_name: str, composite: torch.Tensor, global_expert_id: int
    ) -> torch.Tensor:
        if "linear_fc1.weight" in mcore_weights_name:
            expert_weight = composite[global_expert_id]
            expert_weight = self._align_gate_up_proj_weight(expert_weight)
            return self._interleave_gate_up(expert_weight)

        if "linear_fc1.bias" in mcore_weights_name:
            expert_bias = composite[global_expert_id]
            return self._interleave_gate_up(expert_bias)

        if "linear_fc2.weight" in mcore_weights_name:
            expert_weight = composite[global_expert_id]
            return self._align_down_proj_weight(expert_weight)

        if "linear_fc2.bias" in mcore_weights_name:
            return composite[global_expert_id]

        raise NotImplementedError(f"Unsupported GPT-OSS expert parameter: {mcore_weights_name}")

    @staticmethod
    def _uses_composite_expert_tensor(mcore_weights_name: str, hf_names: list[str]) -> bool:
        if ".mlp.experts.linear_fc" not in mcore_weights_name:
            return False
        return any(".mlp.experts." in hf_name for hf_name in hf_names)

    def _load_composite_hf_tensor(
        self,
        hf_names: list[str],
        hf_weights_map: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if hf_weights_map is None:
            hf_weights = [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
        else:
            hf_weights = [hf_weights_map[x] for x in hf_names]
        kind = self._composite_kind_from_hf_name(hf_names[0])
        return self._materialize_composite_expert_tensor(hf_weights, kind)

    @staticmethod
    def _composite_kind_from_hf_name(hf_name: str) -> str:
        if ".mlp.experts.gate_up_proj" in hf_name:
            if hf_name.endswith("_bias"):
                return "gate_up_proj_bias"
            return "gate_up_proj"
        if ".mlp.experts.down_proj" in hf_name:
            if hf_name.endswith("_bias"):
                return "down_proj_bias"
            return "down_proj"
        raise ValueError(f"Unsupported GPT-OSS composite HF parameter: {hf_name}")

    def _materialize_composite_expert_tensor(self, hf_weights, kind: str) -> torch.Tensor:
        if isinstance(hf_weights, torch.Tensor):
            tensor = hf_weights
        elif len(hf_weights) == 1:
            tensor = hf_weights[0]
        elif len(hf_weights) == 2 and kind in {"gate_up_proj", "down_proj"}:
            tensor = convert_moe_packed_tensors_reference(hf_weights[0], hf_weights[1])
        else:
            raise ValueError(f"Unexpected GPT-OSS HF weight representation for {kind}: {type(hf_weights)}")

        if kind == "down_proj" and tensor.ndim == 3:
            tensor = tensor.transpose(-1, -2).contiguous()
        return tensor

    def _extract_expert_number(self, param_name: str) -> int:
        for pattern in (r"local_experts\.(\d+)", r"(?:weight|bias)(\d+)$"):
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        raise ValueError(f"No expert number found in GPT-OSS parameter name: {param_name}")

    @staticmethod
    def _interleave_gate_up(gate_up_proj: torch.Tensor) -> torch.Tensor:
        return torch.cat((gate_up_proj[::2, ...], gate_up_proj[1::2, ...]), dim=0)

    def _align_gate_up_proj_weight(self, expert_weight: torch.Tensor) -> torch.Tensor:
        hidden_size = self.hf_config.hidden_size
        fused_intermediate = self.hf_config.intermediate_size * 2
        if expert_weight.ndim != 2:
            return expert_weight
        if expert_weight.shape == (hidden_size, fused_intermediate):
            return expert_weight.transpose(0, 1).contiguous()
        if expert_weight.shape == (fused_intermediate, hidden_size):
            return expert_weight
        raise ValueError(
            f"Unexpected GPT-OSS gate_up_proj expert weight shape {tuple(expert_weight.shape)}; "
            f"expected {(hidden_size, fused_intermediate)} or {(fused_intermediate, hidden_size)}."
        )

    def _align_down_proj_weight(self, expert_weight: torch.Tensor) -> torch.Tensor:
        hidden_size = self.hf_config.hidden_size
        intermediate_size = self.hf_config.intermediate_size
        if expert_weight.ndim != 2:
            return expert_weight
        if expert_weight.shape == (intermediate_size, hidden_size):
            return expert_weight.transpose(0, 1).contiguous()
        if expert_weight.shape == (hidden_size, intermediate_size):
            return expert_weight
        raise ValueError(
            f"Unexpected GPT-OSS down_proj expert weight shape {tuple(expert_weight.shape)}; "
            f"expected {(intermediate_size, hidden_size)} or {(hidden_size, intermediate_size)}."
        )

    def _weight_name_mapping_mcore_local_to_global(self, model) -> dict[str, str]:
        """Map local Megatron parameter names to globally-indexed names.

        mbridge's generic MoE implementation assumes grouped-GEMM expert params
        are encoded as ``...weight<expert_id>`` and crashes on GPT-OSS's
        SequentialMLP layout (``local_experts.<id>.linear_fc*.weight``). GPT-OSS
        needs EP-aware renumbering for both layouts.
        """

        from megatron.core import mpu
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()

        model_modules = model if isinstance(model, (list, tuple)) else [model]

        model_module = model_modules[0]
        if hasattr(model_module, "module"):
            model_module = model_module.module
        if hasattr(model_module, "module"):
            model_module = model_module.module
        num_experts = getattr(model_module.config, "num_moe_experts", self.hf_config.num_local_experts)
        expert_offset = ep_rank * num_experts // ep_size if ep_size > 1 else 0

        sig = inspect.signature(get_transformer_layer_offset)
        need_vp_stage = "vp_stage" in sig.parameters

        mapping: dict[str, str] = {}

        def _normalize_one(name: str, layer_offset: int) -> str:
            name = name.replace(".to_wrap", "")
            name = name.replace(".base_linear", "")
            name = re.sub(r"^(module\.)+", "", name)

            if ".adapter." in name:
                return ""

            decoder_match = re.match(r"decoder\.layers\.(\d+)\.(.+)", name)
            if decoder_match:
                layer_idx, rest = decoder_match.groups()
                layer_idx = int(layer_idx) + layer_offset

                local_expert_match = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(.+)", rest)
                if local_expert_match:
                    local_expert_id, suffix = local_expert_match.groups()
                    global_expert_id = int(local_expert_id) + expert_offset
                    return f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{global_expert_id}.{suffix}"

                grouped_match = re.match(r"mlp\.experts\.(.+)\.(weight|bias)(\d+)$", rest)
                if grouped_match:
                    expert_rest, kind, local_expert_id = grouped_match.groups()
                    global_expert_id = int(local_expert_id) + expert_offset
                    return f"decoder.layers.{layer_idx}.mlp.experts.{expert_rest}.{kind}{global_expert_id}"

                return f"decoder.layers.{layer_idx}.{rest}"

            mtp_match = re.match(r"mtp\.layers\.(\d+)\.(.+)", name)
            if mtp_match:
                layer_idx, rest = mtp_match.groups()
                local_expert_match = re.match(r"transformer_layer\.mlp\.experts\.local_experts\.(\d+)\.(.+)", rest)
                if local_expert_match:
                    local_expert_id, suffix = local_expert_match.groups()
                    global_expert_id = int(local_expert_id) + expert_offset
                    return (
                        f"mtp.layers.{layer_idx}.transformer_layer.mlp.experts.local_experts."
                        f"{global_expert_id}.{suffix}"
                    )

                grouped_match = re.match(r"transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)$", rest)
                if grouped_match:
                    expert_rest, kind, local_expert_id = grouped_match.groups()
                    global_expert_id = int(local_expert_id) + expert_offset
                    return f"mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{expert_rest}.{kind}{global_expert_id}"

            return name

        for vp_stage, model_module in enumerate(model_modules):
            unwrapped = model_module
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            layer_offset = (
                get_transformer_layer_offset(unwrapped.config, vp_stage)
                if need_vp_stage
                else get_transformer_layer_offset(unwrapped.config)
            )

            for name in unwrapped.state_dict().keys():
                if "_extra_state" in name:
                    continue
                normalized = _normalize_one(name, layer_offset)
                if normalized:
                    mapping[name] = normalized

        return mapping

    def _build_config(self):
        grouped_gemm_enabled = os.environ.get("SLIME_SCRIPT_DISABLE_MOE_GROUPED_GEMM", "0") != "1"
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE
            moe_ffn_hidden_size=self.hf_config.intermediate_size,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_local_experts,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=grouped_gemm_enabled,
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            # GPT-OSS specific
            add_qkv_bias=True,
            add_bias_linear=True,
            qk_layernorm=False,
            persist_layer_norm=True,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            # SWA
            window_size=(self.hf_config.sliding_window, 0),
            window_attn_skip_freq=2,
            # Learnable softmax
            softmax_type="learnable",
            # Quick GeGLU
            glu_linear_offset=1.0,
            activation_func_clamp_value=getattr(self.hf_config, "swiglu_limit", 7.0),
            # RoPE
            rotary_interleaved=False,
        )

    def _get_transformer_layer_spec(self):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

        return get_gpt_layer_with_transformer_engine_spec()

