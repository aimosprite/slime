# gpt-oss-120b (OpenAI) / aimosprite/gpt-oss-120b-reap-openmathreasoning-hard-tir-private
#
# Architecture (from config.json):
#   36 MoE layers (all), alternating sliding_window(128) / full_attention
#   hidden_size=2880, head_dim=64, num_heads=64, num_kv_heads=8
#   intermediate_size=2880 (per-expert FFN hidden dim)
#   128 experts, top-4 routing, vocab=201088
#   RoPE theta=150000, YaRN factor=32
#   All biases enabled (q/k/v/o + router + expert biases)
#   MXFP4 quantization on expert weights (handled by mbridge dequantization)
#
# Bridge mode: --megatron-to-hf-mode bridge
#   GptOssBridge is registered for model_type="gpt_oss"

N_MOE_LAYERS=36

MODEL_ARGS=(
    # Transformer geometry
    --group-query-attention
    --num-layers $N_MOE_LAYERS
    --hidden-size 2880
    --num-attention-heads 64
    --num-query-groups 8
    --kv-channels 64

    # Attention settings
    --add-qkv-bias
    --add-bias-linear                  # output projection has bias

    # Layer norm + activation
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights

    # Positional embedding (YaRN RoPE)
    --position-embedding-type rope
    --rotary-base 150000
    --vocab-size 201088

    # MoE settings (all 36 layers are MoE)
    --moe-ffn-hidden-size 2880
    --moe-layer-freq "[1]*${N_MOE_LAYERS}"
    --moe-router-topk 4
    --num-experts 128
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type none      # no aux loss for RL
    --moe-router-score-function softmax
    --moe-router-dtype fp32
    --moe-permute-fusion

    # Bridge mode (loads HF weights via mbridge, no convert_hf_to_torch_dist needed)
    --megatron-to-hf-mode bridge
)
