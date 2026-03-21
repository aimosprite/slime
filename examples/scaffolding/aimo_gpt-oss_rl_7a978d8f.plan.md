---
name: AIMO GPT-OSS RL
overview: Build a single-entry slime RL pipeline for AIMO-style math scaffolding that stays close to the notebook’s gen-select/tool behavior while using slime-native SGLang rollouts, plain GRPO training, reward plumbing, Ray orchestration, and wandb logging. The plan keeps GPU allocation and launch behavior configurable for a 16xH100 default, supports both GPT-OSS-20B and GPT-OSS-120B for pipeline validation, always runs both gen-select rounds, and uses the AIMO filtered training data as the reward source.
todos:
  - id: launcher
    content: Design a single Python launcher for setup, secrets, resource config, data resolution, and slime job submission.
    status: pending
  - id: gptoss_models
    content: Add GPT-OSS-20B and GPT-OSS-120B model args/config support using the repo’s existing GPT-OSS Megatron bridge/spec integration.
    status: pending
  - id: rollout
    content: Implement a custom SGLang GPT-OSS multi-turn rollout with Python tool execution, multiple attempts, unconditional two-round gen-select, correct token masking, and simultaneous learning from both rounds.
    status: pending
  - id: reward
    content: Implement correctness-based reward plus grouped gen-select metrics/post-processing aligned with AIMO labels and notebook behavior.
    status: pending
  - id: observability
    content: Wire full wandb logging and add a small-debug validation path before the default 16xH100 configuration.
    status: pending
isProject: false
---

# AIMO GPT-OSS Scaffolding RL Plan

## Goal

Create a single command/script that automatically prepares data + credentials, launches a slime RL run for GPT-OSS-20B or GPT-OSS-120B, and preserves as much of the notebook behavior in [examples/scaffolding/gen-select-nb.ipynb](/Users/rohin/Desktop/code/slime/examples/scaffolding/gen-select-nb.ipynb) as practical while staying on slime’s normal training path and using plain GRPO.

## Implementation Shape

- Keep the main loop on slime’s existing trainer stack: [train.py](/Users/rohin/Desktop/code/slime/train.py), [slime/utils/arguments.py](/Users/rohin/Desktop/code/slime/slime/utils/arguments.py), and [slime/utils/external_utils/command_utils.py](/Users/rohin/Desktop/code/slime/slime/utils/external_utils/command_utils.py).
- Follow the repo’s existing “single entry script” pattern from [examples/true_on_policy/run_simple.py](/Users/rohin/Desktop/code/slime/examples/true_on_policy/run_simple.py) and [examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py](/Users/rohin/Desktop/code/slime/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py), but make it more self-contained for setup, secrets, and resource configuration.
- Reuse the custom multi-turn rollout pattern from [examples/retool/generate_with_retool.py](/Users/rohin/Desktop/code/slime/examples/retool/generate_with_retool.py), but replace the XML/Qwen-specific interaction with GPT-OSS-oriented SGLang requests and tool rounds.
- Use SGLang’s current GPT-OSS support rather than the notebook’s older vLLM + Harmony completion stream. The rollout should preserve the notebook semantics where possible: reasoning effort, Python tool usage, multi-turn solving, multiple attempts per prompt, answer extraction, entropy-based selection, always-on two-round gen-select, and notebook-style problem-level timeout behavior.

## Planned Files

- Add a top-level launcher such as [examples/scaffolding/run_gpt_oss_scaffolding_rl.py](/Users/rohin/Desktop/code/slime/examples/scaffolding/run_gpt_oss_scaffolding_rl.py).
- Add a custom rollout module such as [examples/scaffolding/rollout_gpt_oss_scaffolding.py](/Users/rohin/Desktop/code/slime/examples/scaffolding/rollout_gpt_oss_scaffolding.py).
- Add reward / grouped-selection helpers such as [examples/scaffolding/reward_gpt_oss_scaffolding.py](/Users/rohin/Desktop/code/slime/examples/scaffolding/reward_gpt_oss_scaffolding.py).
- Add a small tool sandbox/helper module if needed, likely next to the rollout and borrowing structure from [examples/retool/generate_with_retool.py](/Users/rohin/Desktop/code/slime/examples/retool/generate_with_retool.py).
- Add a GPT-OSS-20B validation entry and model config alongside the 120B path so the exact same pipeline can be exercised at smaller scale first.
- Add a GPT-OSS-120B model args script modeled on [scripts/models/gpt-oss-20B.sh](/Users/rohin/Desktop/code/slime/scripts/models/gpt-oss-20B.sh), e.g. [scripts/models/gpt-oss-120B.sh](/Users/rohin/Desktop/code/slime/scripts/models/gpt-oss-120B.sh).

## Core Workstreams

### 1. Single-entry launcher

- Implement one Python entry script that:
  - accepts defaults for `16xH100` but exposes knobs for train GPUs, rollout GPUs, GPUs per SGLang engine, TP/PP/EP, and colocated vs split rollout/train placement.
  - checks for `WANDB_API_KEY` and any model/data credentials, prompting interactively if absent.
  - validates or fetches the AIMO training file from the user’s indicated source and stages it into a stable path usable by slime.
  - assembles the full slime CLI and launches via `execute_train(...)` so the run remains repo-native.
- Prefer environment-variable overrides similar to existing example scripts so changing hardware only needs env edits, not code edits.

### 2. GPT-OSS model/backend wiring

- Keep the existing `20B` path usable as the first validation target and add a parallel `120B` model script derived from [scripts/models/gpt-oss-20B.sh](/Users/rohin/Desktop/code/slime/scripts/models/gpt-oss-20B.sh), filling all size-dependent flags for `gpt-oss-120b` from the actual HF config rather than assuming 20B-compatible dimensions.
- Wire the launcher so Megatron training uses the repo’s GPT-OSS support in [slime_plugins/models/gpt_oss.py](/Users/rohin/Desktop/code/slime/slime_plugins/models/gpt_oss.py) and [slime_plugins/mbridge/gpt_oss.py](/Users/rohin/Desktop/code/slime/slime_plugins/mbridge/gpt_oss.py).
- Keep the backend/resource layout configurable enough to move off the initial `16xH100` assumption later without restructuring the example, and let the single launcher switch cleanly between 20B and 120B presets.

### 3. Notebook-style rollout in slime

- Implement a custom rollout function that keeps the notebook’s behavior close to training-time execution:
  - multi-turn solve loop with a Python tool.
  - multiple attempts per prompt.
  - answer extraction focused on boxed integer answers.
  - token/logprob capture for entropy-based attempt scoring.
  - mandatory second-round gen-select / judge pass for every prompt group, with no majority gate.
  - notebook-style problem-level timeout handling, including per-problem budget checks and cancellation of leftover work when the budget is exhausted or early-stop criteria fire.
- Structure the rollout so both rounds become trainable trajectories in the same GRPO update:
  - round 1 samples the candidate attempts.
  - round 2 consumes the round 1 candidates and produces the judge/gen-select answer.
  - both rounds are retained and scored so the policy learns candidate generation and final selection simultaneously.
- Base the integration on slime’s custom rollout hooks in [slime/utils/arguments.py](/Users/rohin/Desktop/code/slime/slime/utils/arguments.py) and [examples/retool/generate_with_retool.py](/Users/rohin/Desktop/code/slime/examples/retool/generate_with_retool.py), but target SGLang’s GPT-OSS tool/reasoning path.
- Use SGLang’s current GPT-OSS parser conventions instead of reproducing the notebook’s raw Harmony token parsing. Important caveat to account for: GPT-OSS tool parsing drops `analysis` content, so tool rounds should be completed by sending tool outputs back as `role="tool"` messages before expecting final user-visible text.
- Preserve the notebook’s key settings as explicit config values: reasoning effort, `attempts`, `max_turns`, tool timeout, judge temperature, context lengths, search window for boxed answers, early-stop criteria, and the notebook’s problem-level timeout controls, but remove the old majority-threshold gate entirely.
- Mirror the notebook’s timeout hierarchy explicitly:
  - per-tool execution timeout.
  - per-attempt / per-turn stopping behavior.
  - per-problem wall-clock budget and dynamic budget bookkeeping.
  - early termination of remaining attempts when a timeout or early-stop condition makes more work invalid or wasteful.
  - launcher-level defaults that match the notebook first, while still allowing easy overrides for different clusters.
- Make loss masking explicit and testable:
  - only tokens generated by the model in each round receive `loss_mask=1`.
  - prompt/context tokens, prior-round context carried into later rounds, tool outputs, and any synthetic glue text receive `loss_mask=0`.
  - round 2 masking is computed independently so the model is trained only on the judge/gen-select tokens it actually emits.

### 4. Reward + selection logic

- Use the AIMO filtered dataset as the training source with correctness reward against `ground_truth` / final integer answer.
- Reuse the existing math normalization and strict box checking logic from [slime/rollout/rm_hub/math_dapo_utils.py](/Users/rohin/Desktop/code/slime/slime/rollout/rm_hub/math_dapo_utils.py) rather than inventing a new evaluator.
- Use plain GRPO as the RL objective, not REINFORCE-style training or custom policy-gradient variants.
- Add custom reward/group post-processing so training can still log notebook-like gen-select metrics even though RL reward is ultimately supervised by the labeled answer.
- Treat notebook-style gen-select as both:
  - a rollout-time aggregation strategy for picking/logging the final answer among attempts.
  - a learned second-round decision process that is always executed and is included in training/logging on every prompt group.
  - a diagnostic/auxiliary signal logged to wandb, including vote count, entropy-weighted score, judge usage, agreement rate, tool usage, round-1 correctness, round-2 correctness, and selection correctness.
- Keep the scalar optimization objective simple at first: correctness-centered reward with notebook-style auxiliary metrics, to minimize train/test mismatch without making the first version overly fragile.

### 5. Data plumbing and schema alignment

- Normalize the AIMO JSONL into the fields slime expects, likely mapping:
  - prompt/question -> rollout input
  - ground truth -> label
  - optional notebook trace / metadata -> extra logging fields only
- If the local AIMO branch path is unavailable, make the launcher able to read from the provided GitHub source or a local override path.
- Preserve long-form questions verbatim to stay close to notebook prompts and avoid accidental train/test drift from aggressive preprocessing.

### 6. Wandb and observability

- Ensure the launcher always enables wandb unless explicitly disabled.
- Log standard slime training metrics plus custom scaffolding metrics, including at minimum:
  - reward and accuracy.
  - parsed-answer rate and boxed-answer rate.
  - tool-call count, tool success/failure, timeout count, average turns.
  - per-attempt entropy summaries and agreement statistics.
  - judge invocation rate and judge override rate.
  - round-1 vs round-2 reward/accuracy split.
  - mask coverage metrics for each round, so incorrect training on prompt/tool tokens is easy to detect.
  - problem-level elapsed time, timeout-trigger rate, and attempt-cancel statistics.
  - selection-vs-single-sample correctness.
  - token/response length, truncation rate, rollout finish reasons.
- Keep naming aligned with existing `train/*`, `rollout/*`, `eval/*` conventions so dashboards are easy to compare with other slime runs.

## Default Configuration Targets

- Default to a `16xH100` configuration but expose these as first-class settings in the launcher:
  - total training GPUs.
  - total rollout GPUs.
  - GPUs per rollout engine.
  - Megatron parallelism values.
  - SGLang memory fraction and engine count.
  - batch sizes, samples per prompt, max response/context length, and eval cadence.
- Start with notebook-like generation defaults where feasible, then translate them into slime/SGLang-native flags rather than hard-coding notebook server assumptions.

## Validation

- First validate the pipeline on a tiny debug slice of the AIMO data with 1 prompt / 1 rollout / few turns, using GPT-OSS-20B as the fast pipeline shakeout target.
- Then validate a small multi-attempt rollout to confirm:
  - tool round-trips work on SGLang with GPT-OSS.
  - both gen-select rounds always execute.
  - entropy scoring and gen-select aggregation produce stable outputs.
  - correctness reward matches dataset labels.
  - round-specific masks only train on model-generated tokens.
  - both round 1 and round 2 samples are present in the same GRPO training batch/update.
  - notebook-style problem-level timeouts stop work when expected and are reflected in rollout metrics.
  - wandb captures both training and custom scaffolding metrics.
- After the 20B pipeline is healthy, run the same path with 120B under the default `16xH100` launch profile.

## Key Risks To Handle In Implementation

- SGLang GPT-OSS tool semantics are not identical to the notebook’s old Harmony stream parser, so the rollout should preserve behavior semantically, not token-for-token.
- The repo has a ready 20B GPT-OSS script but not a 120B one, so the 120B model args must be derived carefully from the real checkpoint config.
- The user-specified AIMO local path may not always exist, so the launcher should have a robust fallback path resolution story.
- Grouped gen-select behavior spans multiple attempts and two trainable rounds, so it may require custom reward post-processing or grouped logging rather than a simple per-sample reward hook.
- Incorrect masking across tool outputs or carried-over round context would silently corrupt training, so the rollout needs explicit mask invariants and debug logging from the start.
- Problem-level timeout semantics are part of the task definition here, so drifting from the notebook’s budgeting/cancellation behavior would create train-test mismatch even if prompts and rewards look correct.

