# gpt-oss-20b (OpenAI) — post tokenizer swap to Qwen3.5
#
# Architecture (from config.json):
#   24 MoE layers (all), alternating sliding_window(128) / full_attention
#   hidden_size=2880, head_dim=64, num_heads=64, num_kv_heads=8
#   intermediate_size=2880 (per-expert FFN hidden dim, SwiGLU)
#   32 experts, top-4 routing
#   RoPE theta=150000, YaRN factor=32
#   All biases enabled (q/k/v/o + router + expert biases)
#   vocab_size=248320 (post Qwen3.5 tokenizer swap; original=201088)
#
# Note: head_dim=64 explicitly set in config (NOT 2880/64=45).
#   q_proj: [4096, 2880], k_proj: [512, 2880], v_proj: [512, 2880]

N_MOE_LAYERS=24

MODEL_ARGS=(
    # Transformer geometry
    --group-query-attention
    --num-layers $N_MOE_LAYERS
    --hidden-size 2880
    --ffn-hidden-size 2880
    --num-attention-heads 64
    --num-query-groups 8
    --kv-channels 64

    # Attention settings — GPT-OSS has biases on all projections
    # bias is enabled by default in Megatron (use --disable-bias-linear to turn off)
    --add-qkv-bias

    # Layer norm + activation
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --untie-embeddings-and-output-weights

    # Positional embedding (YaRN RoPE)
    --position-embedding-type rope
    --rotary-base 150000
    --vocab-size 248320

    # MoE settings (all 24 layers are MoE)
    --moe-ffn-hidden-size 2880
    --moe-layer-freq '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
    --moe-router-topk 4
    --num-experts 32
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type none
    --moe-router-score-function softmax
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0

    # Impl
    --transformer-impl "${TRANSFORMER_IMPL:-transformer_engine}"
    --sequence-parallel
    --no-rope-fusion
    --no-bias-swiglu-fusion
)
