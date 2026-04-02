"""
Megatron-compatible core attention module using flash attention with learnable softmax.

This replaces TEDotProductAttention for models that need learnable softmax with packed
sequences (THD format), which TE 2.10.0 does not support.
"""

import math
import os

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.utils import divide
from torch import Tensor

from slime_plugins.models.learnable_softmax_attention import learnable_softmax_flash_attn_varlen


def _expand_kv_heads_for_gqa(kv: Tensor, num_query_heads: int) -> Tensor:
    """Expand KV heads to match query heads for local fallback attention."""
    if kv.size(1) == num_query_heads:
        return kv
    if num_query_heads % kv.size(1) != 0:
        raise ValueError(
            f"Cannot expand {kv.size(1)} KV heads to {num_query_heads} query heads for local attention fallback."
        )
    return kv.repeat_interleave(num_query_heads // kv.size(1), dim=1)


def _safe_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    softmax_scale: float,
    softmax_offset: Tensor | None,
    causal: bool,
    window_size: tuple[int, int],
    dropout_p: float,
    training: bool,
) -> Tensor:
    """Safe chunked-attention fallback for packed THD tensors.

    This avoids flash-attn's varlen kernel on the long-context GPT-OSS training path we
    exercise in Modal. For full attention we keep memory bounded by chunking queries;
    for local attention the same code path only touches the relevant window.
    """
    left_window, right_window = window_size
    if left_window < 0 and right_window < 0:
        block_size = int(os.environ.get("SLIME_SAFE_VARLEN_ATTN_BLOCK_SIZE", "128"))
    else:
        block_size = int(os.environ.get("SLIME_SAFE_VARLEN_ATTN_BLOCK_SIZE", "1024"))
    out = torch.empty_like(q)
    q_heads = q.size(1)
    offset_fp32 = softmax_offset.float() if softmax_offset is not None else None

    for seq_idx in range(cu_seqlens_q.numel() - 1):
        q_start = int(cu_seqlens_q[seq_idx].item())
        q_end = int(cu_seqlens_q[seq_idx + 1].item())
        k_start = int(cu_seqlens_k[seq_idx].item())
        k_end = int(cu_seqlens_k[seq_idx + 1].item())

        q_seq = q[q_start:q_end]
        k_seq = _expand_kv_heads_for_gqa(k[k_start:k_end], q_heads)
        v_seq = _expand_kv_heads_for_gqa(v[k_start:k_end], q_heads)
        seq_len_q = q_end - q_start
        seq_len_k = k_end - k_start

        for block_q_start in range(0, seq_len_q, block_size):
            block_q_end = min(block_q_start + block_size, seq_len_q)
            q_chunk = q_seq[block_q_start:block_q_end]

            if left_window >= 0:
                block_k_start = max(0, block_q_start - left_window)
            else:
                block_k_start = 0

            if causal:
                max_k_index = block_q_end - 1 + max(right_window, 0)
            elif right_window >= 0:
                max_k_index = block_q_end - 1 + right_window
            else:
                max_k_index = seq_len_k - 1
            block_k_end = min(seq_len_k, max_k_index + 1)

            k_chunk = k_seq[block_k_start:block_k_end]
            v_chunk = v_seq[block_k_start:block_k_end]

            scores = torch.einsum("qhd,khd->qhk", q_chunk.float(), k_chunk.float()) * softmax_scale

            q_positions = torch.arange(block_q_start, block_q_end, device=q.device).unsqueeze(1)
            k_positions = torch.arange(block_k_start, block_k_end, device=q.device).unsqueeze(0)
            valid = torch.ones((block_q_end - block_q_start, block_k_end - block_k_start), device=q.device, dtype=torch.bool)
            if left_window >= 0:
                valid &= k_positions >= (q_positions - left_window)
            if right_window >= 0:
                valid &= k_positions <= (q_positions + right_window)
            if causal:
                valid &= k_positions <= q_positions
            scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)

            lse = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            if dropout_p > 0.0 and training:
                probs = F.dropout(probs, p=dropout_p, training=True)

            out_chunk = torch.einsum("qhk,khd->qhd", probs, v_chunk.float())
            if offset_fp32 is not None:
                out_chunk = out_chunk * torch.sigmoid(lse - offset_fp32.unsqueeze(0)).unsqueeze(-1)
            out[q_start + block_q_start : q_start + block_q_end] = out_chunk.to(q.dtype)

    return out


class FlashDotProductAttention(MegatronModule):
    """Core attention using flash attention directly, bypassing TE.

    Supports learnable softmax + THD (packed sequences) + sliding window attention,
    a combination that TE 2.10.0 cannot handle.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = max(1, layer_number)

        if softmax_scale is not None:
            self.softmax_scale = softmax_scale
        else:
            self.softmax_scale = 1.0 / math.sqrt(config.kv_channels)

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        if is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number):
            self.window_size = config.window_size  # (left, right) tuple
        else:
            self.window_size = (-1, -1)

        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout

        # Learnable softmax offset
        self.current_max_attn_logits = None  # compatibility with qk-clip interface
        if pg_collection is None:
            from megatron.core import parallel_state

            world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            world_size = pg_collection.tp.size()
        num_heads_per_partition = divide(config.num_attention_heads, world_size)

        if config.softmax_type == "vanilla":
            self.softmax_offset = None
        elif config.softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                num_heads_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        elif config.softmax_type == "learnable":
            self.register_parameter(
                "softmax_offset",
                torch.nn.Parameter(
                    torch.empty(
                        num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                ),
            )
            if config.perform_initialization:
                self.softmax_offset = config.init_method(self.softmax_offset)
        else:
            raise ValueError(f"Unsupported softmax type: {config.softmax_type}")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        """Forward pass using flash attention.

        Handles both THD (packed) and SBHD formats.
        """
        assert attention_bias is None, "Attention bias not supported with FlashDotProductAttention"

        causal = (
            attn_mask_type
            in (
                AttnMaskType.causal,
                AttnMaskType.padding_causal,
                AttnMaskType.causal_bottom_right,
            )
            if attn_mask_type is not None
            else True
        )

        is_training = self.training
        dropout_p = self.attention_dropout if is_training else 0.0

        if packed_seq_params is not None and hasattr(packed_seq_params, "qkv_format"):
            qkv_format = packed_seq_params.qkv_format
        else:
            qkv_format = "sbhd"

        if qkv_format == "thd":
            # THD: packed sequences. q/k/v are (total_tokens, nheads, headdim)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_k = packed_seq_params.cu_seqlens_kv
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_k = packed_seq_params.max_seqlen_kv
        else:
            # SBHD: (seq_len, batch, nheads, headdim) → convert to varlen format
            s, b, h, d = query.shape
            query = query.permute(1, 0, 2, 3).reshape(b * s, h, d)
            key = key.permute(1, 0, 2, 3).reshape(b * s, key.size(2), d)
            value = value.permute(1, 0, 2, 3).reshape(b * s, value.size(2), d)
            cu_seqlens_q = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=query.device)
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_q = s
            max_seqlen_k = s

        force_safe_varlen = os.environ.get("SLIME_FORCE_SAFE_VARLEN_ATTN", "0") == "1"
        use_safe_varlen = qkv_format == "thd" and (force_safe_varlen or self.window_size != (-1, -1))

        if use_safe_varlen:
            output = _safe_varlen_attention(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                self.softmax_scale,
                self.softmax_offset,
                causal,
                self.window_size,
                dropout_p,
                is_training,
            )
        elif self.softmax_offset is not None:
            # Use learnable softmax flash attention
            output = learnable_softmax_flash_attn_varlen(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.softmax_offset,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=self.window_size,
                dropout_p=dropout_p,
                deterministic=self.config.deterministic_mode if hasattr(self.config, "deterministic_mode") else False,
            )
        else:
            # Vanilla flash attention
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=self.window_size,
                deterministic=self.config.deterministic_mode if hasattr(self.config, "deterministic_mode") else False,
            )

        if qkv_format != "thd":
            # Convert back to SBHD: (b*s, h, d) → (s, b, h, d)
            output = output.reshape(b, s, h, d).permute(1, 0, 2, 3)

        return output
