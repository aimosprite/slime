# ORCD: interactive GPU allocation for scaffolding experiments

This folder holds MIT ORCD Slurm entrypoints and the GPT-OSS scaffolding RL launcher. When you are **debugging or iterating quickly**, waiting for a new `sbatch` allocation on every attempt wastes time. Prefer **one long-lived Slurm allocation** and run many commands on that node until you are done.

## Why use a persistent allocation?

- **`sbatch`**: Each job enters the queue separately. You pay queue latency every time you change a flag or fix a crash.
- **`salloc` + interactive shell**: Slurm grants GPUs once (for up to the partition walltime). You can start Ray, Apptainer runs, short tests, and reruns **without** a new allocation, until time expires or you exit/`scancel`.

Match the same resources as the batch scripts you are debugging (see `#SBATCH` in `sbatch_train_small_full_mit_orcd*.sh` or `sbatch_train_small_full_mit_preemptable_4gpu_16k.sh`), e.g.:

- **Normal GPU queue:** `mit_normal_gpu`, `--gres=gpu:h200:2`, `--cpus-per-task=16`, `--mem=128G`, `--time=5:59:00`
- **Preemptable + 4×H200 (scaffolding full run):** `mit_preemptable`, `--gres=gpu:h200:4`, `--cpus-per-task=32`, `--mem=256G`, `--time=5:59:00` (jobs may be preempted; walltime up to 2 days on that partition per `sinfo`)

Use a **different tmux session name** per partition (e.g. `slime_interactive_preempt` vs `slime_interactive_alloc`) so you do not attach to the wrong allocation.

## Recommended pattern: `tmux` + `salloc` + `srun`

Run this from the **ORCD login node** (SSH session). `tmux` keeps your session alive if SSH drops so you do not lose the allocation.

```bash
# Optional: reuse or pick a session name
tmux new-session -d -s slime_interactive_alloc \
  salloc \
    --partition=mit_normal_gpu \
    --gres=gpu:h200:2 \
    --cpus-per-task=16 \
    --mem=128G \
    --nodes=1 \
    --ntasks=1 \
    --time=5:59:00 \
    --job-name=slime-interactive \
    srun --pty bash -l
```

**Same idea on `mit_preemptable` with 4×H200** (matches `sbatch_train_small_full_mit_preemptable_4gpu_16k.sh`):

```bash
tmux new-session -d -s slime_interactive_preempt \
  salloc \
    --partition=mit_preemptable \
    --gres=gpu:h200:4 \
    --cpus-per-task=32 \
    --mem=256G \
    --nodes=1 \
    --ntasks=1 \
    --time=5:59:00 \
    --job-name=slime-interactive-preempt \
    srun --pty bash -l
```

Then: `tmux attach -t slime_interactive_preempt`

When Slurm prints `Granted job allocation <JOBID>` and you see a prompt on a compute node (e.g. `[user@nodeXXXX ...]$`), you are on the GPU node.

- **Attach:** `tmux attach -t slime_interactive_alloc`
- **Detach (leave job running):** `Ctrl-b` then `d`
- **End allocation:** `exit` the shell on the compute node, or `scancel <JOBID>` from another shell

## Running experiments on that node

1. **Secrets:** Put `WANDB_API_KEY` and `HF_TOKEN` in `examples/scaffolding/.env` (this repo path). The sbatch scripts source that file by absolute path so it still works when Slurm copies the batch script elsewhere.

2. **Same commands as batch, without `sbatch`:** From the compute-node shell inside the allocation, run the body of the script directly, for example:

   ```bash
   cd /home/rohin/slime
   bash examples/scaffolding/sbatch_train_small_full_mit_orcd_16k_ctx.sh
   ```

   Or tee logs:

   ```bash
   bash examples/scaffolding/sbatch_train_small_full_mit_orcd_16k_ctx.sh 2>&1 | tee examples/scaffolding/my-run.log
   ```

3. **Apptainer:** Those scripts `module load cuda` / `apptainer` and run `python3 examples/scaffolding/run_gpt_oss_scaffolding_rl.py` inside `slime.sif` when the image exists at `/home/rohin/slime/slime.sif`.

4. **Do not rely on a second concurrent `srun` on the same allocation** unless you know the first step left free GPUs—the interactive pattern here uses **one** `srun` step (your shell) and you launch training from inside it.

## When to use `sbatch` instead

Use `sbatch` for unattended long runs, strict log files per job ID, requeue behavior, or when you do not need a shell. Use **`salloc` + tmux** when you expect many failures and quick retries.

## Related files

| File | Role |
|------|------|
| `sbatch_train_small_full_mit_orcd.sh` | Full run, ~32k rollout context cap |
| `sbatch_train_small_full_mit_orcd_16k_ctx.sh` | Same, ~16k context if 32k OOMs |
| `sbatch_train_small_full_mit_preemptable_4gpu_16k.sh` | ~16k context, **4×H200**, `mit_preemptable` (use when 2×H200 colocate OOMs) |
| `run_gpt_oss_scaffolding_rl.py` | Python launcher; reads `SLIME_SCRIPT_*` env vars |
| `.env` | `WANDB_API_KEY`, `HF_TOKEN` (not committed; keep private) |

## Colocated train + rollout memory (GPT-OSS 20B)

The launcher uses the **classic colocated** Megatron path: **full** activation recompute, **dynamic** batching, **`--expert-tensor-parallel-size 1`**, and tokenizer model derived from **`--hf-checkpoint`** when needed. Scaffolding uses **`--n-samples-per-prompt` = 2 × `SLIME_SCAFFOLDING_ATTEMPTS`** (default **16** = 8 solvers + 8 judges); set **`SLIME_SCRIPT_GLOBAL_BATCH_SIZE`** accordingly (e.g. **`rollout_batch_size × 16`** when using batch scripts’ defaults). On tight GPUs you may still hit CUDA OOM; optional overrides: **`SLIME_SCRIPT_SGLANG_MEM_FRACTION`**, **`SLIME_SCRIPT_MAX_TOKENS_PER_GPU`**. For more headroom per device, use **more GPUs** with matching **`SLIME_SCRIPT_TP`** / **`SLIME_SCRIPT_ROLLOUT_TP`** (e.g. `sbatch_train_small_full_mit_preemptable_4gpu_16k.sh` with TP=4 on 4×H200).

If you add new Slurm flags or env vars, keep **batch scripts and this workflow** in sync so interactive and batch runs behave the same.
