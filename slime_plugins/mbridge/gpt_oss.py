"""
Bridge for OpenAI GPT-OSS models (gpt-oss-20b, gpt-oss-120b).

Architecture:
  - MoE layers (all), alternating sliding_window(128) / full attention
  - hidden_size=2880, head_dim=64, num_heads=64, num_kv_heads=8
  - intermediate_size=2880 (per-expert FFN hidden dim, SwiGLU)
  - Attention biases on q/k/v/o projections
  - Router has bias (mlp.router.bias)
  - No shared experts, no QK layer norms

Post tokenizer_swap.py, expert weights are dense bf16:
  gate_up_proj.weight [num_experts, 2*ffn_hidden, hidden]
  down_proj.weight    [num_experts, hidden, ffn_hidden]

Without --moe-grouped-gemm, Megatron uses per-expert SequentialMLP with names like:
  mlp.experts.local_experts.{i}.linear_fc1.weight
We map each to the fused HF tensor and extract expert i.
"""

import re

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("gpt_oss")
class GptOssBridge(Qwen2MoEBridge):
    """Megatron-Core bridge for OpenAI GPT-OSS family."""

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_proj.bias": [
            "model.layers.{layer_number}.self_attn.o_proj.bias"
        ],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
    }

    _OTHER_MAPPING = {
        "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if ".self_attention." in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name or "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            return self._weight_name_mapping_other(mcore_weights_name)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]

        # Per-expert sequential MLP: local_experts.{i}.linear_fc1/fc2.weight/bias
        m = re.search(r"local_experts\.(\d+)\.linear_fc([12])\.(weight|bias)", name)
        if m:
            expert_id = int(m.group(1))
            fc_num = m.group(2)
            wb = m.group(3)
            if fc_num == "1" and wb == "weight":
                return [f"model.layers.{layer_number}.mlp.experts.gate_up_proj.weight"]
            elif fc_num == "1" and wb == "bias":
                return [f"model.layers.{layer_number}.mlp.experts.gate_up_proj_bias"]
            elif fc_num == "2" and wb == "weight":
                return [f"model.layers.{layer_number}.mlp.experts.down_proj.weight"]
            elif fc_num == "2" and wb == "bias":
                return [f"model.layers.{layer_number}.mlp.experts.down_proj_bias"]

        # Router and layernorm
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                return [x.format(layer_number=layer_number) for x in mapping_names]

        raise NotImplementedError(f"GptOssBridge: unsupported MLP param: {name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        # Per-expert: extract single expert from fused [num_experts, out, in] tensor
        m = re.search(r"local_experts\.(\d+)\.linear_fc([12])\.(weight|bias)", mcore_weights_name)
        if m:
            expert_id = int(m.group(1))
            w = hf_weights[0]
            if w.dim() == 2:
                # Bias: [num_experts, out] → [out]
                return w[expert_id].to(torch.bfloat16)
            else:
                # Weight: [num_experts, out, in] → [out, in]
                return w[expert_id].to(torch.bfloat16)

        # QKV concat
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 3
            hf_cfg = self.hf_config
            num_kv_heads = hf_cfg.num_key_value_heads
            num_heads = hf_cfg.num_attention_heads
            head_dim = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // num_heads)
            num_q_per_kv = num_heads // num_kv_heads
            q, k, v = hf_weights
            is_bias = ".bias" in mcore_weights_name

            if is_bias:
                return torch.cat([q, k, v], dim=0)
            else:
                hidden = q.shape[-1]
                real_kv_heads = q.shape[0] // (num_q_per_kv * head_dim)
                q = q.view(real_kv_heads, num_q_per_kv, head_dim, hidden)
                k = k.view(real_kv_heads, 1, head_dim, hidden)
                v = v.view(real_kv_heads, 1, head_dim, hidden)
                qkv = torch.cat([q, k, v], dim=1)
                return qkv.view(-1, hidden).contiguous()

        # Default: single tensor pass-through
        assert len(hf_weights) == 1, (
            f"GptOssBridge: expected 1 tensor for {mcore_weights_name}, got {len(hf_weights)}"
        )
        return hf_weights[0]

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Convert Megatron → HF. Saves per-expert weights back into fused format."""
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        # QKV: split back
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            hf_cfg = self.hf_config
            num_kv_heads = hf_cfg.num_key_value_heads
            num_heads = hf_cfg.num_attention_heads
            head_dim = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // num_heads)
            num_q_per_kv = num_heads // num_kv_heads
            is_bias = ".bias" in mcore_weights_name

            if is_bias:
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                q = mcore_weights[:q_size]
                k = mcore_weights[q_size : q_size + kv_size]
                v = mcore_weights[q_size + kv_size :]
            else:
                hidden = mcore_weights.shape[-1]
                qkv = mcore_weights.view(num_kv_heads, num_q_per_kv + 2, head_dim, hidden)
                q = qkv[:, :num_q_per_kv].reshape(-1, hidden)
                k = qkv[:, num_q_per_kv : num_q_per_kv + 1].reshape(-1, hidden)
                v = qkv[:, num_q_per_kv + 1 :].reshape(-1, hidden)

            return hf_names, [q, k, v]

        return hf_names, [mcore_weights]

    def _build_config(self):
        hf = self.hf_config
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE
            moe_ffn_hidden_size=hf.intermediate_size,
            moe_router_topk=hf.num_experts_per_tok,
            num_moe_experts=hf.num_local_experts,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            # GPT-OSS has biases on q/k/v/o + router
            add_qkv_bias=True,
            add_bias_linear=True,
            qk_layernorm=False,
            # No Apex/TE
            persist_layer_norm=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
        )
