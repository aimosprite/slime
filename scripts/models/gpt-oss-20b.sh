# OpenAI GPT-OSS-20B (MoE) model args for SLIME bridge-mode runs.
#
# NOTE:
# - This script is intended for `bridge_mode: true` RL runs in this repo.
# - If your SLIME branch ships a dedicated GPT-OSS script, prefer that one.

MODEL_ARGS=(
   --num-layers 24
   --hidden-size 2880
   --ffn-hidden-size 2880

   --num-attention-heads 64
   --group-query-attention
   --num-query-groups 8
   --kv-channels 64

   --normalization RMSNorm
   --norm-epsilon 1e-5
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-150000}"

   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 201088

   # MoE
   --moe-ffn-hidden-size 2880
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 4
   --moe-layer-freq [1]*24
   --num-experts 32
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0
)
