"""Periodic test-loss evaluation hook for SFT embedding surgery.

Used via --custom-megatron-before-train-step-hook-path slime.hooks.eval_hook.eval_before_step

All TP ranks must participate in the forward pass (NCCL collectives).
Data is prepared on TP-rank-0 and broadcast to other TP ranks.
Loss is computed via vocab_parallel_cross_entropy (TP-aware).
"""

import json
import logging
import os
import random

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

logger = logging.getLogger(__name__)

_state = {"test_samples": None, "initialized": False}

EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "50"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
TEST_DATA_PATH = os.environ.get(
    "TEST_DATA_PATH", "/root/slime/models/test-eval-sample.jsonl"
)
NUM_TEST_CACHE = int(os.environ.get("NUM_TEST_CACHE", "2000"))


def _init(args):
    """Load and tokenize test data (TP-rank-0 only)."""
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer

    logger.info(f"[eval_hook] Loading test data: {TEST_DATA_PATH}")
    tok = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    mask_gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=args.loss_mask_type)

    with open(TEST_DATA_PATH) as f:
        rows = [json.loads(line) for line in f]

    indices = random.Random(123).sample(range(len(rows)), min(NUM_TEST_CACHE, len(rows)))
    test_samples = []
    for idx in indices:
        msgs = rows[idx]["messages"]
        try:
            token_ids, loss_mask = mask_gen.get_loss_mask(msgs)
        except Exception:
            continue
        if len(token_ids) > args.seq_length:
            continue
        test_samples.append((token_ids, loss_mask))

    _state["test_samples"] = test_samples
    logger.info(f"[eval_hook] Cached {len(test_samples)} test samples")


def _build_batch(samples, batch_indices):
    """Build packed THD batch from selected test samples."""
    all_tokens = []
    all_loss_masks = []
    cu_seqlens = [0]

    for i in batch_indices:
        token_ids, loss_mask = samples[i]
        all_tokens.append(torch.tensor(token_ids, dtype=torch.long))
        all_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float32))
        cu_seqlens.append(cu_seqlens[-1] + len(token_ids))

    tokens_cat = torch.cat(all_tokens).cuda()
    loss_masks_cat = torch.cat(all_loss_masks).cuda()

    # Pad to TP-aligned size
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pad_size = tp_size * 128
    pad = (pad_size - tokens_cat.size(0) % pad_size) % pad_size
    if pad > 0:
        tokens_cat = F.pad(tokens_cat, (0, pad), value=0)
        loss_masks_cat = F.pad(loss_masks_cat, (0, pad), value=0.0)
        cu_seqlens.append(cu_seqlens[-1] + pad)

    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int, device="cuda")
    max_seqlen = (cu_seqlens_t[1:] - cu_seqlens_t[:-1]).max().item()

    return tokens_cat, loss_masks_cat, cu_seqlens_t, max_seqlen


def eval_before_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Hook called before each training step on ALL ranks."""
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

    # Only DP-rank-0's TP group participates (other DP groups skip)
    if mpu.get_data_parallel_rank(with_context_parallel=True) != 0:
        return

    num_steps_per_rollout = args.global_batch_size // (
        args.micro_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
    )
    accumulated_step = rollout_id * num_steps_per_rollout + step_id

    if accumulated_step % EVAL_INTERVAL != 0:
        return

    is_main = mpu.get_tensor_model_parallel_rank() == 0
    tp_group = mpu.get_tensor_model_parallel_group()

    # Initialize on main rank only
    if not _state["initialized"]:
        if is_main:
            _init(args)
        _state["initialized"] = True

    # Main rank picks batch indices and builds tokens; broadcast to other TP ranks
    if is_main and _state["test_samples"]:
        samples = _state["test_samples"]
        batch_indices = random.sample(range(len(samples)), min(EVAL_BATCH_SIZE, len(samples)))
        tokens, loss_mask, cu_seqlens, max_seqlen = _build_batch(samples, batch_indices)
        # Broadcast metadata: total_len and num_seqlens
        meta = torch.tensor([tokens.size(0), cu_seqlens.size(0), max_seqlen], dtype=torch.long, device="cuda")
    else:
        meta = torch.zeros(3, dtype=torch.long, device="cuda")

    torch.distributed.broadcast(meta, src=torch.distributed.get_process_group_ranks(tp_group)[0], group=tp_group)
    total_len, num_seqlens, max_seqlen = meta[0].item(), meta[1].item(), meta[2].item()

    if not is_main:
        tokens = torch.zeros(total_len, dtype=torch.long, device="cuda")
        loss_mask = torch.zeros(total_len, dtype=torch.float32, device="cuda")
        cu_seqlens = torch.zeros(num_seqlens, dtype=torch.int, device="cuda")

    src_rank = torch.distributed.get_process_group_ranks(tp_group)[0]
    torch.distributed.broadcast(tokens, src=src_rank, group=tp_group)
    torch.distributed.broadcast(loss_mask, src=src_rank, group=tp_group)
    torch.distributed.broadcast(cu_seqlens, src=src_rank, group=tp_group)

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    tokens_input = tokens.unsqueeze(0)  # [1, T]

    # Forward pass (all TP ranks participate)
    m = model[0].module if hasattr(model[0], "module") else model[0]
    was_training = m.training
    m.eval()

    with torch.no_grad():
        # Returns [batch, seq, vocab/tp] = [1, T, vocab/tp] on each TP rank
        output = m(
            input_ids=tokens_input,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
        )

        # Remove batch dim: [1, T, vocab/tp] -> [T, vocab/tp]
        logits = output.squeeze(0) if output.dim() == 3 else output

        # Shift for next-token prediction
        shift_logits = logits[:-1, :]    # [T-1, vocab/tp]
        shift_labels = tokens[1:]        # [T-1]
        shift_mask = loss_mask[1:]       # [T-1]

        # TP-aware cross entropy (handles partial logits + all-reduce internally)
        per_token_loss = vocab_parallel_cross_entropy(
            shift_logits.unsqueeze(0).float(),  # [1, T-1, vocab/tp]
            shift_labels.unsqueeze(0),           # [1, T-1]
        ).squeeze(0)  # [T-1]

        masked_loss = (per_token_loss * shift_mask).sum()
        n_tokens = shift_mask.sum()
        mean_loss = (masked_loss / n_tokens).item() if n_tokens > 0 else float("nan")

    if was_training:
        m.train()

    # Log only from main rank
    if is_main:
        from slime.utils import logging_utils
        log_dict = {
            "test/loss": mean_loss,
            "test/n_tokens": int(n_tokens.item()),
            "train/step": accumulated_step,
        }
        logging_utils.log(args, log_dict, step_key="train/step")
        logger.info(f"[eval_hook] step {accumulated_step}: test/loss={mean_loss:.4f} "
                    f"({int(n_tokens.item()):,} tokens)")
