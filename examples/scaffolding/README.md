# AIMO GPT-OSS scaffolding RL

This example implements **parallel solve rollouts + parallel gen-select judge rollouts**, **split-group GRPO** (solvers normalized among solvers, judges among judges), and **loss masking** so only model-generated tokens are trained (tool/interpreter observations are masked out).

## Layout

| File | Purpose |
|------|---------|
| `gs_config.py` | Notebook timing/search defaults (`SLIME_SCAFFOLDING_*` env overrides). |
| `scaffolding_boxed.py` | Shared `\boxed{}` integer extraction for rollout stop + rewards (consistent semantics). |
| `python_tool.py` | Python tool runner with process timeout (aligned with notebook `jupyter_timeout`). |
| `rollout_gpt_oss_scaffolding.py` | Custom `--rollout-function-path`: SGLang `/generate` multi-turn tool loop, **8 solver + 8 judge** samples per problem (default), Harmony-style chat prompts. |
| `reward_gpt_oss_scaffolding.py` | Solver 0/1 (boxed vs ground truth) and judge 0/1 (pick GT from among solver proposals). |
| `grpo_dual_group_reward_postprocess.py` | `--custom-reward-post-process-path`: mean-center (optional std) **separately** for solver and judge subgroups. |
| `run_gpt_oss_scaffolding_rl.py` | Single launcher (wandb key prompt, `execute_train`, `PYTHONPATH`). |
| `../../scripts/models/gpt-oss-120B.sh` | Megatron `MODEL_ARGS` for **openai/gpt-oss-120b** (36 layers, 128 experts). |

## Requirements

- `--n-samples-per-prompt` must equal **`2 * SLIME_SCAFFOLDING_ATTEMPTS`** (default **8 + 8 = 16**).
- Training data JSONL fields: `question`, `ground_truth` (see `.hf-dataset-inspect/train_data_filtered.jsonl` in-repo inspect file).
- Set `SLIME_SCRIPT_HF_CHECKPOINT` to a local GPT-OSS HF tree for SGLang + (Megatron) training.

## Run

```bash
export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b-or-120b
export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
export SLIME_SCRIPT_MODEL_SIZE=20b   # or 120b
python examples/scaffolding/run_gpt_oss_scaffolding_rl.py
```

Tune GPUs and parallelism via `SLIME_SCRIPT_NUM_GPUS`, `SLIME_SCRIPT_TP`, `SLIME_SCRIPT_EP`, `SLIME_SCRIPT_ROLLOUT_TP`, etc.

### Modal single-script launcher

Use `examples/scaffolding/run_gpt_oss_scaffolding_modal.py` to launch the same RL entrypoint on Modal with model-size defaults:

- `20b`: `2xH200`, `TP=2`, `EP=1`, `rollout_tp=2`
- `120b`: `8xH200`, `TP=1`, `EP=8`, `rollout_tp=8`

From the **repo root**, the wrapper script sets volume/secret defaults and runs `modal run`:

```bash
examples/scaffolding/scripts/prep-modal.sh                    # venv + volume (once)
examples/scaffolding/scripts/prep-modal.sh --upload-sample-data   # optional: sample JSONL on volume
examples/scaffolding/scripts/run-gpt-oss-20b-scaffolding-modal.sh   # 20B; Hub weights + default data path
```

Or invoke Modal directly (volume `slime-data` and secret `slime-training-secrets` are wired in the `.py`):

```bash
modal run examples/scaffolding/run_gpt_oss_scaffolding_modal.py \
  --model-size 20b \
  --data-jsonl /root/data/train_data_filtered.jsonl
```

### Smoke test (20B, few prompts)

Runs reward checks + config consistency (`n_samples_per_prompt` = `2 * SLIME_SCAFFOLDING_ATTEMPTS`), then **`--debug-rollout-only`** with one rollout step and batch size 1 (override via env).

```bash
export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke
```

Reward/config only (no Ray; no GPU):

```bash
python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke-rewards-only
```

Core training flags set by the launcher include:

- `--rollout-function-path examples.scaffolding.rollout_gpt_oss_scaffolding.generate_rollout_gs`
- `--custom-reward-post-process-path examples.scaffolding.grpo_dual_group_reward_postprocess.dual_group_grpo_reward_postprocess`
- `--advantage-estimator grpo`

## Colocation memory tuning (SGLang + Megatron on the same GPUs)

When `--colocate` is set, Slime enables **train offload** (`torch_memory_saver`) so rollout and training time-share VRAM, but **both engines still contend at init**: SGLang loads first; after `release_memory_occupation`, Megatron must allocate large **DDP grad buffers** while a **SGLang footprint** remains. If Megatron’s first big `torch.zeros` (grad pool) fails, you need **more free VRAM**, not guesswork.

### 1) Read the failure (no binary search on the first variable)

From the log, find either:

- `torch.OutOfMemoryError: Tried to allocate X GiB. ... Y GiB is free` → **shortfall** \( \approx X - Y \) (GiB).
- `TorchMemorySaver::malloc return OOM ... (alloc)size=... free_bytes=...` → convert bytes to GiB:  
  `alloc_gib = alloc_bytes / 2^30`, `free_gib = free_bytes / 2^30`, **shortfall** \( \approx \texttt{alloc\_gib} - \texttt{free\_gib} \).

You need roughly **shortfall** more contiguous free memory before that allocation succeeds (plus a few GiB slack for fragmentation).

### 2) First-order budget (usable for any GPU size / future 120B)

Let:

- \(M\) = **usable per-GPU VRAM** (GiB). Example: H200 **~140 GiB** in our logs (`total capacity of 139.80 GiB`).
- \(f\) = `--sglang-mem-fraction-static` (what SGLang is allowed to treat as its **static** memory pool, as a fraction of device memory).
- \(G\) = **largest Megatron allocation** you observed (GiB). Empirically on **GPT-OSS 20B, TP=4, DP=1**, the failing line was **~73 GiB** for the grad buffer (`Tried to allocate 72.90 GiB`; `alloc)size=78274101248` → ~72.9 GiB).
- `margin` = `train_memory_margin_bytes` / \(2^{30}\) (default **1 GiB** in slime when offload is on).
- `overhead` = CUDA context, fragmentation, non-PyTorch use — budget **~3–8 GiB** in practice.

**Ballpark constraint** (per GPU, after SGLang release):

\[
(1 - f)\,M - \texttt{overhead} \;\gtrsim\; G + \texttt{margin}
\quad\Rightarrow\quad
f \;\lesssim\; 1 - \frac{G + \texttt{margin} + \texttt{overhead}}{M}.
\]

**Plug in our 4×H200 20B numbers** (from job `10983236`):

- \(M \approx 140\), \(G \approx 73\), `margin` \(= 1\), pick `overhead` \(= 6\):  
  \(f \lesssim 1 - 80/140 \approx 0.43\).

The earlier run used **\(f = 0.52\)**, giving **~74 GiB** process memory still tied up and only **~60.5 GiB** free — consistent with \(f \cdot M \approx 0.52 \times 140 \approx 73\) GiB reserved to inference-side pools. That matches **~13 GiB shortfall** vs the **~73 GiB** Megatron allocation.

**Action from a measured shortfall** (GiB): lower \(f\) by about **shortfall / M** (linear first step), then re-run:

\[
\Delta f \approx \frac{\texttt{shortfall\_gib}}{M}.
\]

Example: shortfall **13 GiB**, \(M = 140\) → \(\Delta f \approx 0.093\) → from 0.52 go to **~0.43**; the preemptable 4×H200 script defaults a bit lower (**0.38**) for headroom.

### 3) `--max-tokens-per-gpu` (Megatron dynamic batch)

This caps **training** token microbatches (not SGLang’s KV pool). It does **not** fix the usual **startup** OOM (large contiguous **DDP grad buffer** vs. leftover SGLang VRAM); tune **`--sglang-mem-fraction-static`** for that. Use this knob only if **train steps** OOM after init (e.g. 2048 → 1536 → 1024). Activation memory scales with **tokens × hidden × layers / recompute**; at 20B with full recompute it is often modest next to weights + grad/optimizer, but long **packed** sequences can still spike.

### 4) Fragmentation (`expandable_segments` vs colocation)

PyTorch’s `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (or `PYTORCH_ALLOC_CONF=...`) can help **Megatron** tolerate a fragmented CUDA heap, but **Slime’s colocated path must not set it globally**: SGLang’s scheduler uses **TorchMemorySaver**, which **errors out** if expandable segments are enabled (`RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet`), killing `RolloutManager` / `SGLangEngine` at startup.

**Rule for `--colocate`:** leave expandable segments **unset** (default allocator). Rely on **`--sglang-mem-fraction-static`** and the shortfall math above. If you ever split train and rollout onto **separate** processes with disjoint CUDA allocators, you could revisit allocator settings for the train side only.

### 5) Scaling toward GPT-OSS 120B

- **\(G\)** (Megatron peak chunk per rank) grows with model size and **shrinks** with more **tensor parallel** width (each rank holds fewer parameters). You cannot reuse the numeric **\(f\)** from 20B on 120B without re-estimating \(G\) (or re-reading a single OOM line).
- **Rule**: keep the same **inequality**; after changing model / TP / EP, if OOM returns, read **new** `alloc_gib` / `free_gib` and apply **\(\Delta f \approx \texttt{shortfall}/M\)** again.
- **More GPUs**: often **raise TP** (or EP for MoE) so per-rank **\(G\)** drops; **\(M\)** unchanged per device unless you change batch or sequence caps.

## Notes

- **Solvers**: `SLIME_SCAFFOLDING_ATTEMPTS` parallel trajectories; each scored **1** iff the last `\boxed{}` integer matches `ground_truth` (any integer; no fixed competition range in prompts or rewards).
- **Judges**: same count of parallel trajectories; each scored **1** iff its boxed pick equals `ground_truth` **and** that integer was proposed by at least one solver (last boxed extract per solver).
- **Judge sampling** defaults to **temperature 1.0** (`SLIME_SCAFFOLDING_JUDGE_TEMPERATURE`); override for deterministic judges.
- **Problem timeout** follows the notebook formula in `rollout_gpt_oss_scaffolding._problem_budget_s` (override `notebook_elapsed` / `problems_remaining` via sample `metadata` if needed).
- **SGLang / GPT-OSS**: standard `/generate` with `return_logprob=True` for training logprobs.
