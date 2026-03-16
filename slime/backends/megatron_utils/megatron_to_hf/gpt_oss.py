"""
GPT-OSS Megatron → HF weight converter.

GPT-OSS specifics vs Qwen3MoE:
- Fused expert weights: gate_up_proj [num_experts, 2*ffn, hidden], down_proj [num_experts, hidden, ffn]
- Router has bias (mlp.router.bias)
- Attention has biases on q/k/v/o projections
- Per-expert SequentialMLP names: local_experts.{i}.linear_fc{1,2}.weight
"""

import re

import torch


def convert_gpt_oss_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Per-expert SequentialMLP: local_experts.{i}.linear_fc{1,2}.{weight,bias}
        expert_pattern = r"mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.(weight|bias)"
        m = re.match(expert_pattern, rest)
        if m:
            expert_id = m.group(1)
            fc_num = m.group(2)
            wb = m.group(3)
            if fc_num == "1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                if wb == "weight":
                    return [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight", gate_weight),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight", up_weight),
                    ]
                else:  # bias
                    return [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.bias", gate_weight),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.bias", up_weight),
                    ]
            elif fc_num == "2":
                if wb == "weight":
                    return [(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight", param)]
                else:
                    return [(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.bias", param)]

        # Also handle grouped gemm format: mlp.experts.linear_fc{1,2}.{weight,bias}{expert_id}
        grouped_expert_pattern = r"mlp\.experts\.linear_fc([12])\.(weight|bias)(\d+)"
        m = re.match(grouped_expert_pattern, rest)
        if m:
            fc_num, wb, expert_id = m.groups()
            if fc_num == "1":
                gate_part, up_part = param.chunk(2, dim=0)
                if wb == "weight":
                    return [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight", gate_part),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight", up_part),
                    ]
                else:  # bias
                    return [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.bias", gate_part),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.bias", up_part),
                    ]
            elif fc_num == "2":
                if wb == "weight":
                    return [(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight", param)]
                else:
                    return [(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.bias", param)]

        # Attention
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_proj.bias":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.bias", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, [value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param, [value_num_per_group * head_dim, head_dim, head_dim], dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]

        # Layernorms
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.router.weight", param)]
        elif rest == "mlp.router.bias":
            return [(f"model.layers.{layer_idx}.mlp.router.bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
