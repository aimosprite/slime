import torch

from mbridge.core import register_model

from .qwen3_next import Qwen3NextBridge


@register_model("qwen3_5_moe")
class Qwen35MoEBridge(Qwen3NextBridge):
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
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _build_config(self):
        mtp_args = {}
        cfg = self._get_text_hf_config()
        if hasattr(cfg, "num_nextn_predict_layers"):
            mtp_args["mtp_num_layers"] = cfg.num_nextn_predict_layers
        elif hasattr(cfg, "mtp_num_hidden_layers"):
            mtp_args["mtp_num_layers"] = cfg.mtp_num_hidden_layers

        return self._build_base_config(
            use_cpu_initialization=False,
            moe_ffn_hidden_size=cfg.moe_intermediate_size,
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
            **mtp_args,
        )
