# AIMO GPT-OSS scaffolding RL

This example implements notebook-aligned **parallel solve attempts + unconditional gen-select judge**, plain **GRPO**, and **loss masking** so only model-generated tokens are trained (tool/interpreter observations are masked out).

## Layout

| File | Purpose |
|------|---------|
| `gs_config.py` | Notebook timing/search defaults (`SLIME_SCAFFOLDING_*` env overrides). |
| `python_tool.py` | Python tool runner with process timeout (aligned with notebook `jupyter_timeout`). |
| `rollout_gpt_oss_scaffolding.py` | Custom `--rollout-function-path` implementation: SGLang `/generate` multi-turn tool loop, problem-level deadline, early vote stop, **always-on judge**, one `Sample` per attempt + one for judge. |
| `reward_gpt_oss_scaffolding.py` | Strict boxed correctness helper (used when building each sample). |
| `run_gpt_oss_scaffolding_rl.py` | Single launcher (wandb key prompt, `execute_train`, `PYTHONPATH`). |
| `../../scripts/models/gpt-oss-120B.sh` | Megatron `MODEL_ARGS` for **openai/gpt-oss-120b** (36 layers, 128 experts). |

## Requirements

- `--n-samples-per-prompt` must equal **`SLIME_SCAFFOLDING_ATTEMPTS + 1`** (default 8 attempts + 1 judge = **9**).
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

### Smoke test (20B, few prompts)

Runs strict boxed reward checks + config consistency (`n_samples_per_prompt` = `SLIME_SCAFFOLDING_ATTEMPTS + 1`), then **`--debug-rollout-only`** with one rollout step and batch size 1 (override via env).

```bash
export SLIME_SCRIPT_HF_CHECKPOINT=/path/to/gpt-oss-20b
export SLIME_SCRIPT_DATA_JSONL=/path/to/train_data_filtered.jsonl
python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke
```

Reward/config only (no Ray; no extra Python deps beyond stdlib + `math_dapo_utils`):

```bash
python examples/scaffolding/run_gpt_oss_scaffolding_rl.py --smoke-rewards-only
```

Core training flags set by the launcher include:

- `--rollout-function-path examples.scaffolding.rollout_gpt_oss_scaffolding.generate_rollout_gs`
- `--advantage-estimator grpo`

## Notes

- **Judge always runs** (no majority-vote gate): the judge round is trained every time alongside attempts.
- **Problem timeout** follows the notebook formula in `gs_config._problem_budget_s` (override `notebook_elapsed` / `problems_remaining` via sample `metadata` if needed).
- **SGLang / GPT-OSS**: uses the standard `/generate` API with `return_logprob=True` for training logprobs.
