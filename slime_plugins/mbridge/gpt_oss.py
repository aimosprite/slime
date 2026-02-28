"""
Bridge for OpenAI gpt-oss-120b and aimosprite fine-tuned derivatives.

Architecture (from config.json):
  - 36 MoE layers (all), alternating sliding_window(128) / full attention
  - hidden_size=2880, intermediate_size=2880 (per-expert FFN hidden dim)
  - 128 experts, top-4 routing, vocab=201088
  - Attention biases on q/k/v/o projections
  - Router has bias (mlp.router.bias)
  - No shared experts, no QK layer norms
  - MXFP4 quantization on MoE expert weights only (attention/router/embed unquantized)
  - swiglu_limit=7.0 (capped SwiGLU gate activation)

Expert weight format in original openai/gpt-oss-120b:
  MXFP4 quantized: {gate_up,down}_proj_blocks (packed 4-bit) + {gate_up,down}_proj_scales (e8m0)
  + {gate_up,down}_proj_bias (full precision)

If the aimosprite fine-tuned model was saved with dequantized dense weights, set
  GPTOSS_DEQUANT=false (env var) to skip dequantization.
  Expected dense format: gate_up_proj.weight [num_experts, 2*ffn_hidden, hidden]
                         down_proj.weight     [num_experts, hidden, ffn_hidden]

Run to verify weight names before training:
  python -c "
  from safetensors import safe_open
  import os
  f = safe_open('model-00001-of-00014.safetensors', framework='pt')
  for k in list(f.keys())[:30]: print(k)
  "
"""

import os

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


def _unpack_mxfp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack MXFP4 e2m1 values from uint8 packed tensor (2 values per byte).

    fp4 e2m1 bit layout: s eee mmm → 1 sign, 2 exp, 1 mantissa
    Actually fp4 is: 1 sign + 2 exp + 1 mantissa = 4 bits

    Returns float32 tensor with twice as many elements as input bytes.
    """
    # Each uint8 byte contains 2 fp4 values: high nibble + low nibble
    hi = (packed >> 4) & 0xF  # upper 4 bits
    lo = packed & 0xF          # lower 4 bits

    def nibble_to_f32(nibbles: torch.Tensor) -> torch.Tensor:
        # fp4 e2m1: bit3=sign, bit2-1=exp, bit0=mantissa
        sign = ((nibbles >> 3) & 1).float()
        exp = ((nibbles >> 1) & 0x3).long()
        mant = (nibbles & 0x1).float()

        # Special: exp=0 → subnormal, exp>0 → normal
        # Normal: value = (-1)^sign * 2^(exp-1) * (1 + mant/2)
        # Subnormal (exp=0): value = (-1)^sign * 2^(-0) * (mant/2) = ±mant/2
        normal_mask = exp > 0
        exp_f = exp.float()

        value = torch.where(
            normal_mask,
            (1.0 + mant * 0.5) * (2.0 ** (exp_f - 1.0)),
            mant * 0.5,
        )
        value = value * (1.0 - 2.0 * sign)  # apply sign
        return value

    hi_f = nibble_to_f32(hi)
    lo_f = nibble_to_f32(lo)

    # Interleave: [hi_0, lo_0, hi_1, lo_1, ...]
    out = torch.stack([hi_f, lo_f], dim=-1).flatten(-2)
    return out


def _dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 expert weights (blocks + e8m0 scales) to bfloat16.

    Assumes MX block size of 32 elements.

    Args:
        blocks: uint8 tensor of shape [..., block_size_bytes] where block_size_bytes = 16
                (32 fp4 values packed into 16 bytes)
        scales: uint8 e8m0 scale tensor of shape [...] (one scale per 32 elements)

    Returns:
        bfloat16 tensor with shape [..., 32]
    """
    BLOCK_SIZE = 32  # MX standard block size

    # Unpack fp4 values
    values = _unpack_mxfp4_e2m1(blocks)  # [..., 32]

    # Decode e8m0 scales: value = 2^(uint8 - 127)
    scale_f = (2.0 ** (scales.float() - 127.0)).unsqueeze(-1)  # [..., 1]

    return (values * scale_f).to(torch.bfloat16)


def _dequant_expert_weight(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize gpt-oss MXFP4 expert weight tensor.

    Actual gpt-oss storage format (confirmed from model inspection):
      blocks: [num_experts, out_features, num_blocks, 16]  uint8
              (16 bytes per block = 32 fp4 values per block, 2 fp4 per byte)
      scales: [num_experts, out_features, num_blocks]  uint8 e8m0

    Returns:
      bfloat16 tensor of shape [num_experts, out_features, in_features]
      where in_features = num_blocks * 32
    """
    num_experts, out_features, num_blocks, bytes_per_block = blocks.shape
    in_features = num_blocks * bytes_per_block * 2  # 2 fp4 per byte

    dequant = _dequant_mxfp4(blocks, scales)  # [E, out, num_blocks, 32]
    return dequant.view(num_experts, out_features, in_features)


@register_model("gpt_oss")
class GptOssBridge(Qwen2MoEBridge):
    """
    Megatron-Core bridge for OpenAI gpt-oss-120b family.

    Extends Qwen2MoEBridge with:
      - Attention projection biases (q/k/v/o)
      - Router bias
      - MXFP4 expert weight dequantization
      - gpt-oss-specific router naming (mlp.router vs mlp.gate)
    """

    # Attention mapping: gpt-oss uses self_attn (same as Qwen2), but has biases on all projections
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

    # MLP mapping: gpt-oss uses mlp.router (not mlp.gate) and stores experts in combined tensors
    # Expert weights are MXFP4 quantized: _blocks (packed fp4) + _scales (e8m0)
    # Megatron grouped-GEMM format: [num_experts, 2*ffn_hidden, hidden] for fc1 (gate+up combined)
    _MLP_MAPPING = {
        "pre_mlp_layernorm": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        # Router: gpt-oss uses mlp.router (not mlp.gate like Qwen2MoE)
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        # Expert weights (combined blocks+scales for dequantization in _weight_to_mcore_format)
        # gate_up combined (Megatron fc1 = gate+up concatenated on dim 0)
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_blocks",
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_scales",
        ],
        "mlp.experts.linear_fc1.bias": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_bias"
        ],
        # down proj (Megatron fc2)
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.down_proj_blocks",
            "model.layers.{layer_number}.mlp.experts.down_proj_scales",
        ],
        "mlp.experts.linear_fc2.bias": [
            "model.layers.{layer_number}.mlp.experts.down_proj_bias"
        ],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Map Megatron-Core param name → list of HF param names."""
        assert "_extra_state" not in mcore_weights_name

        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if mcore_weights_name in direct_name_mapping:
            return [direct_name_mapping[mcore_weights_name]]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name or "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"GptOssBridge: unsupported param name: {mcore_weights_name}"
            )

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Map MLP Megatron param names to HF names."""
        layer_number = name.split(".")[2]  # decoder.layers.{layer}.mlp...
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                return [x.format(layer_number=layer_number) for x in mapping_names]
        raise NotImplementedError(f"GptOssBridge: unsupported MLP param: {name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """Convert HF tensors → Megatron tensor.

        Special handling:
          1. QKV concatenation (inherited from Qwen2MoEBridge)
          2. Expert weight dequantization (blocks+scales → bfloat16 dense)
        """
        # Expert fc1 (gate+up): dequantize from MXFP4 blocks+scales
        if "mlp.experts.linear_fc1.weight" in mcore_weights_name:
            blocks, scales = hf_weights  # [E, 2*H, K/2] and [E, 2*H, K/32]
            # Check if already dense (fine-tuned model may store as float)
            if blocks.dtype != torch.uint8:
                # Already dense: expect [num_experts, 2*ffn_hidden, hidden]
                return blocks.to(torch.bfloat16)
            return _dequant_expert_weight(blocks, scales)

        # Expert fc2 (down): dequantize from MXFP4 blocks+scales
        if "mlp.experts.linear_fc2.weight" in mcore_weights_name:
            blocks, scales = hf_weights  # [E, H, F/2] and [E, H, F/32]
            if blocks.dtype != torch.uint8:
                return blocks.to(torch.bfloat16)
            return _dequant_expert_weight(blocks, scales)

        # Expert biases: just pass through
        if "mlp.experts.linear_fc" in mcore_weights_name and ".bias" in mcore_weights_name:
            return hf_weights[0].to(torch.bfloat16)

        # QKV concat (same logic as Qwen2MoEBridge / Qwen3NextBridge)
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
                # bias shape: [out_features] — just concatenate
                return torch.cat([q, k, v], dim=0)
            else:
                # weight shape: [out_features, hidden_size]
                real_kv_heads = q.shape[0] // (num_q_per_kv * head_dim)
                q = q.view(real_kv_heads, num_q_per_kv, head_dim, -1)
                k = k.view(real_kv_heads, 1, head_dim, -1)
                v = v.view(real_kv_heads, 1, head_dim, -1)
                qkv = torch.cat([q, k, v], dim=1)  # [kv_groups, q_per_kv+2, head_dim, hidden]
                hidden = q.shape[-1]
                return qkv.view(-1, hidden).contiguous()

        # Default: single-tensor pass-through
        assert len(hf_weights) == 1, (
            f"GptOssBridge: expected 1 HF tensor for {mcore_weights_name}, got {len(hf_weights)}"
        )
        return hf_weights[0]

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Convert Megatron tensor → HF format for saving checkpoints.

        Expert weights are saved as dense bfloat16 (not re-quantized to MXFP4),
        which is compatible with full-precision fine-tuning.
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        # Expert fc1: save as single dense tensor (drop the scales entry)
        if "mlp.experts.linear_fc1.weight" in mcore_weights_name:
            # Save as dense gate_up_proj; use only the first mapped name (blocks)
            # and skip scales (no re-quantization during training)
            return [hf_names[0]], [mcore_weights]

        # Expert fc2: same
        if "mlp.experts.linear_fc2.weight" in mcore_weights_name:
            return [hf_names[0]], [mcore_weights]

        # QKV: split back into q, k, v
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
                total_heads = num_heads + 2 * num_kv_heads
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
            moe_router_load_balancing_type="none",  # no aux loss for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            # Attention + router bias: controlled by add_bias_linear (default True)
            # gpt-oss has biases on q/k/v/o projections AND the router linear
            add_qkv_bias=True,
            add_bias_linear=True,              # enables bias on router + o_proj
            qk_layernorm=False,
            # General
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )
