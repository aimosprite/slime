import torch

from mbridge.core import register_model
from mbridge.core.llm_bridge import LLMBridge

from .qwen3_next import Qwen3NextBridge


@register_model("qwen3_5_moe")
class Qwen35MoEBridge(Qwen3NextBridge):
    # Qwen3.5 text_config lacks intermediate_size; provide a default so the
    # CONFIG_MAPPING getattr doesn't raise.  The actual expert/shared-expert
    # FFN sizes are passed explicitly in _build_config.
    _CONFIG_MAPPING = {
        **LLMBridge._CONFIG_MAPPING,
        "ffn_hidden_size": ("intermediate_size", None),
    }

    # GatedDeltaNet (linear attention) layers in Megatron use `self_attention.<param>`
    # directly, while HF nests them under `linear_attn.<param>`.
    # Merge with parent mapping so full-attention layers keep their existing mappings.
    _ATTENTION_MAPPING = {
        **Qwen3NextBridge._ATTENTION_MAPPING,
        # GatedDeltaNet direct params (Megatron) -> HF linear_attn.*
        "self_attention.A_log": ["model.layers.{layer_number}.linear_attn.A_log"],
        "self_attention.conv1d.weight": ["model.layers.{layer_number}.linear_attn.conv1d.weight"],
        "self_attention.dt_bias": ["model.layers.{layer_number}.linear_attn.dt_bias"],
        "self_attention.out_norm.weight": ["model.layers.{layer_number}.linear_attn.norm.weight"],
        "self_attention.out_proj.weight": ["model.layers.{layer_number}.linear_attn.out_proj.weight"],
        # Input layernorm is fused into in_proj for GatedDeltaNet layers
        "self_attention.in_proj.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        # Megatron merges qkv+z+b+a into a single in_proj; HF splits them.
        "self_attention.in_proj.weight": [
            "model.layers.{layer_number}.linear_attn.in_proj_qkv.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_z.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_b.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_a.weight",
        ],
    }

    def _get_text_hf_config(self):
        # Qwen3.5 multimodal checkpoints store LM hyperparameters in text_config.
        return getattr(self.hf_config, "text_config", self.hf_config)

    def _map_language_model_weight_name(self, name: str) -> str:
        # Qwen3.5 multimodal checkpoints nest LM weights under model.language_model.*.
        if not hasattr(self.hf_config, "text_config"):
            return name
        return (
            name.replace("model.layers.", "model.language_model.layers.")
            .replace("model.embed_tokens.", "model.language_model.embed_tokens.")
            .replace("model.norm.", "model.language_model.norm.")
        )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        if "mtp" in mcore_weights_name:
            convert_names = self._convert_mtp_param(mcore_weights_name)
        else:
            convert_names = super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        return [self._map_language_model_weight_name(name) for name in convert_names]

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 3
            cfg = self._get_text_hf_config()
            num_key_value_heads = cfg.num_key_value_heads
            hidden_dim = cfg.hidden_size
            num_attention_heads = cfg.num_attention_heads
            num_querys_per_group = num_attention_heads // cfg.num_key_value_heads
            head_dim = getattr(cfg, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(
                    [
                        real_num_key_value_heads,
                        num_querys_per_group,
                        2,
                        head_dim,
                        -1,
                    ]
                )
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        # GatedDeltaNet: merge HF split in_proj_{qkv,z,b,a} -> single in_proj
        if "self_attention.in_proj.weight" in mcore_weights_name:
            # hf_weights order: [qkv, z, b, a] (matches _ATTENTION_MAPPING)
            assert len(hf_weights) == 4
            qkv, z, b, a = hf_weights
            # Megatron splits as [q, k, v, z, b, a] along dim 0
            return torch.cat([qkv, z, b, a], dim=0).contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _get_gptmodel_args(self) -> dict:
        cfg = self._get_text_hf_config()
        rope_params = getattr(cfg, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rotary_base = rope_params.get("rope_theta", 10000)
        else:
            rotary_base = getattr(cfg, "rope_theta", 10000)
        return dict(
            vocab_size=cfg.vocab_size,
            max_sequence_length=cfg.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rotary_base,
        )

    def _build_config(self):
        mtp_args = {}
        cfg = self._get_text_hf_config()
        if hasattr(cfg, "num_nextn_predict_layers"):
            mtp_args["mtp_num_layers"] = cfg.num_nextn_predict_layers
        elif hasattr(cfg, "mtp_num_hidden_layers"):
            mtp_args["mtp_num_layers"] = cfg.mtp_num_hidden_layers

        text_config_key = "text_config" if hasattr(self.hf_config, "text_config") else None

        # Qwen3.5 MoE has no intermediate_size; derive ffn_hidden_size from
        # shared_expert_intermediate_size (matches the model-args script).
        ffn_hidden_size = getattr(cfg, "intermediate_size", None) or getattr(
            cfg, "shared_expert_intermediate_size", cfg.moe_intermediate_size
        )

        # Gated delta net (hybrid linear/full attention) config
        linear_attn_args = {}
        full_attention_interval = getattr(cfg, "full_attention_interval", None)
        if full_attention_interval is not None:
            linear_attn_args.update(
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=full_attention_interval,
                linear_conv_kernel_dim=getattr(cfg, "linear_conv_kernel_dim", 4),
                linear_key_head_dim=getattr(cfg, "linear_key_head_dim", 128),
                linear_value_head_dim=getattr(cfg, "linear_value_head_dim", 128),
                linear_num_key_heads=getattr(cfg, "linear_num_key_heads", None),
                linear_num_value_heads=getattr(cfg, "linear_num_value_heads", None),
            )

        return self._build_base_config(
            text_config_key=text_config_key,
            use_cpu_initialization=False,
            ffn_hidden_size=ffn_hidden_size,
            moe_ffn_hidden_size=cfg.moe_intermediate_size,
            moe_shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=cfg.num_experts_per_tok,
            num_moe_experts=cfg.num_experts,
            moe_aux_loss_coeff=cfg.router_aux_loss_coef,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            attention_output_gate=True,
            moe_shared_expert_gate=True,
            **linear_attn_args,
            **mtp_args,
        )
