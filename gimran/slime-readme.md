# Slime SFT Pipeline — How It Works

This document explains the exact code path when running SFT (supervised fine-tuning) with slime, specifically for the embedding surgery experiment.

## Quick Overview

```
bash script (launch) → train_async.py (orchestrator)
                          ├── RolloutManager (CPU: data loading + tokenization)
                          └── MegatronTrainRayActor (8 GPUs: forward/backward/update)
```

## Entry Point

**Script:** `scripts/sft-gpt-oss-20b-embedding-swap.sh`

The bash script:
1. Loads config from yaml via `scripts/lib/config.sh` (flat yaml → env vars)
2. Validates all required vars are set (no silent defaults)
3. Optionally runs prep (download model, tokenizer swap, megatron conversion, dataset)
4. Kills leftover processes
5. Starts Ray head node
6. Starts HF checkpoint shipper (background loop)
7. Submits a Ray job: `python3 train_async.py` with all args

**Config loader:** `scripts/lib/config.sh`
- `load_config()`: Reads yaml, converts `snake_case: value` → `export SNAKE_CASE='value'`
- `load_env()`: Sources `.env` for secrets (WANDB_KEY, HF_TOKEN)

## Orchestration

**File:** `train_async.py`

Creates two Ray actors:
- **`RolloutManager`** (`slime/ray/rollout.py`) — Single CPU actor that manages data
- **`MegatronTrainRayActor`** (`slime/ray/megatron.py`) — 8 GPU workers (TP=4, DP=2)

The training loop (simplified):
```python
for rollout_id in range(total_rollouts):
    # 1. RolloutManager fetches + tokenizes a batch
    samples = rollout_manager.generate(rollout_id)

    # 2. MegatronTrainRayActor trains on that batch
    train_actor.train_step(samples)
```

With `rollout_batch_size == global_batch_size == 128`, each rollout = exactly 1 training step.

## Data Loading

### Dataset source
**File:** `slime/rollout/data_source.py`

- Reads the parquet file (`am-qwen3-distilled-train.parquet`) into memory at init
- Shuffles with a deterministic seed
- `get_samples(n)` returns the next `n` samples from a sequential cursor
- When cursor reaches the end, wraps to next epoch and reshuffles

### Tokenization + filtering
**File:** `slime/rollout/sft_rollout.py`

Called once per training step by `RolloutManager.generate()`:

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    # 1. Keep fetching from data_buffer until we have enough valid samples
    while len(filtered) < target:
        samples = data_buffer.get_samples(fetch_size)
        for sample in samples:
            # 2. Tokenize messages using the model's tokenizer
            token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

            # 3. Skip samples longer than seq_length (8192)
            if len(token_ids) > max_len:
                continue

            # 4. Attach tokens + loss mask to sample
            sample.tokens = token_ids
            sample.loss_mask = loss_mask[-response_length:]
            filtered.append(sample)

    return filtered  # exactly rollout_batch_size samples
```

### Loss masking
**File:** `slime/utils/mask_utils.py`

`MultiTurnLossMaskGenerator` with `tokenizer_type="qwen3"`:
- Applies the chat template to messages
- Tokenizes the full conversation
- Creates a binary mask: **1** for assistant response tokens (compute loss), **0** for user/system/template tokens (ignore)
- This means the model only learns to predict assistant outputs, not user inputs

## Training

### Batch construction
**File:** `slime/backends/megatron_utils/data.py`

Takes the list of variable-length Samples and packs them into a single batch:
- Concatenates all token sequences into one long tensor
- Builds `cu_seqlens` (cumulative sequence lengths) for flash attention
- Pads to TP-aligned boundaries
- Creates `PackedSeqParams(cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, qkv_format="thd")`

This is the "THD packing" format — multiple variable-length sequences in one batch, handled efficiently by flash attention without padding waste.

### Forward + backward + optimizer
**File:** `slime/backends/megatron_utils/model.py`

`MegatronTrainRayActor` wraps a Megatron GPTModel. Each training step:

1. **Forward pass:**
   - `input_ids [1, T]` → `embed_tokens` (vocab_parallel) → embeddings
   - → 24 transformer layers (MoE, 32 experts, top-4 routing) — **all frozen**
   - → `lm_head` (output_layer, vocab_parallel) → logits `[1, T, vocab/tp]`

2. **Loss computation:**
   - `vocab_parallel_cross_entropy(logits, labels)` — TP-aware, each GPU has vocab/4 logits
   - Masked by loss_mask (only assistant tokens contribute)
   - `calculate_per_token_loss=True` — loss is averaged per token, not per sequence

3. **Backward pass:**
   - Gradients flow through the full model BUT only `embed_tokens` and `lm_head` have `requires_grad=True`
   - Enforced by `--only-train-params-name-list embedding output_layer`
   - All 24 transformer layers, all 32×24 expert FFNs — zero gradient, zero update

4. **Optimizer step:**
   - Adam with lr=5e-4, cosine decay, 1% warmup
   - Only updates the two embedding matrices (~1.4B params total)
   - Gradient clipping at 1.0

### Parameter freezing
**File:** `slime/backends/megatron_utils/model.py` (during model init)

```python
# --only-train-params-name-list embedding output_layer
for name, param in model.named_parameters():
    if not any(train_name in name for train_name in ["embedding", "output_layer"]):
        param.requires_grad = False
```

The two trainable layers:
- `embedding.word_embeddings.weight` — shape `[248320, 2880]` (split across TP=4 → `[62080, 2880]` per GPU)
- `output_layer.weight` — shape `[248320, 2880]` (same split)

Both are randomly initialized (N(0, 0.02)) since the original GPT-OSS tokenizer was swapped for Qwen3.5.

## Evaluation

**File:** `slime/hooks/eval_hook.py`

Registered via `--custom-megatron-before-train-step-hook-path`. Called before every training step (eval_interval=1).

1. On first call: loads test parquet, tokenizes up to 2000 samples, caches in memory
2. Each eval: picks 32 random cached samples, packs into a THD batch
3. Forward pass with `torch.no_grad()` on all TP ranks
4. Computes masked cross-entropy via `vocab_parallel_cross_entropy`
5. Logs `test/loss` to WandB

Only DP-rank-0's TP group runs eval (the other DP group skips to avoid duplicate work).

## Logging + Checkpointing

### WandB
**File:** `slime/utils/logging_utils.py`

Metrics logged every step:
- `train/loss` — masked per-token cross-entropy on training batch
- `train/grad_norm` — gradient norm (of the two trainable layers)
- `train/lr-pg_0` — current learning rate
- `test/loss` — masked per-token cross-entropy on test batch (from eval hook)

### Checkpointing
**File:** `slime/backends/megatron_utils/checkpoint.py`

Every `save_interval=100` steps, saves a Megatron `torch_dist` checkpoint to `models/gpt-oss-20b-qwen3.5-tokenizer_slime/`.

### HF shipping
**File:** `scripts/lib/hf_shipper.sh`

Background bash loop that polls every 30s for new checkpoints and uploads them to HuggingFace (`aimosprite/gpt-oss-20b-embedding-surgery`).

## Parallelism

```
8x H100 80GB GPUs
├── TP group 0: GPU 0,1,2,3  (tensor parallel — model split 4 ways)
└── TP group 1: GPU 4,5,6,7  (tensor parallel — same split)

DP = 8 / (TP=4 × PP=1 × CP=1) = 2 data parallel groups
```

Each training step:
- Both DP groups process different micro-batches from the same global batch
- Gradients are all-reduced across the 2 DP groups
- Each TP group handles 1/4 of the vocabulary for embed_tokens and lm_head

## File Summary

| File | Role |
|------|------|
| `scripts/sft-gpt-oss-20b-embedding-swap.sh` | Launch script |
| `scripts/lib/config.sh` | Config loader (yaml → env vars) |
| `scripts/lib/hf_shipper.sh` | Background checkpoint uploader |
| `scripts/models/gpt-oss-20b.sh` | Model architecture args (49 flags) |
| `configs/sft-gpt-oss-20b-embedding-surgery-stage1.yaml` | All training hyperparameters |
| `train_async.py` | Ray orchestrator |
| `slime/ray/rollout.py` | RolloutManager actor |
| `slime/ray/megatron.py` | MegatronTrainRayActor actor |
| `slime/rollout/sft_rollout.py` | Data tokenization + length filtering |
| `slime/rollout/data_source.py` | Sequential dataset cursor |
| `slime/utils/mask_utils.py` | Loss mask generation (assistant tokens only) |
| `slime/backends/megatron_utils/data.py` | THD batch packing |
| `slime/backends/megatron_utils/model.py` | Forward/backward/optimizer step |
| `slime/backends/megatron_utils/checkpoint.py` | Megatron checkpoint save/load |
| `slime/hooks/eval_hook.py` | Test loss evaluation hook |
| `slime/utils/logging_utils.py` | WandB logging |
