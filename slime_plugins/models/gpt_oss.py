"""
Monkey-patch DotProductAttention to support packed sequences via flash_attn_varlen_func.

This is applied at import time. With --transformer-impl local, Megatron uses
DotProductAttention which rejects packed sequences. This patch adds flash
attention varlen support for the packed case while keeping the original
behavior for non-packed inputs.

Import this module before model construction:
    import slime_plugins.models.gpt_oss  # applies patch
"""

import torch
from flash_attn import flash_attn_varlen_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention

_original_forward = DotProductAttention.forward


def _patched_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    attn_mask_type=None,
    attention_bias=None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
):
    if packed_seq_params is None:
        return _original_forward(
            self, query, key, value, attention_mask, attn_mask_type,
            attention_bias, packed_seq_params, **kwargs,
        )

    # query/key/value shape: [sq, b, np, hn]
    sq, b, np, hn = query.shape
    _, _, nkv, _ = key.shape

    q = query.reshape(sq * b, np, hn)
    k = key.reshape(sq * b, nkv, hn)
    v = value.reshape(sq * b, nkv, hn)

    cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(torch.int32)
    cu_seqlens_k = packed_seq_params.cu_seqlens_kv.to(torch.int32)
    max_seqlen_q = packed_seq_params.max_seqlen_q
    max_seqlen_k = packed_seq_params.max_seqlen_kv

    softmax_scale = hn ** -0.5

    attn_output = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=self.attention_dropout.p if self.training else 0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    return attn_output.reshape(sq, b, np, hn)


DotProductAttention.forward = _patched_forward
