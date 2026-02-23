# Add Stateful Python Tool to On-Policy Distillation

## Context

The OPD training (Qwen3-8B teacher → Qwen3-1.7B student) currently generates plain text rollouts. We want the student model to learn to use a Python code interpreter tool during generation — Jupyter-style, where variables persist across tool calls within a single rollout.

The existing `examples/retool/` has a working tool-augmented training setup, but its `PythonSandbox` is **stateless** (fresh subprocess per call). We need a **stateful** variant.

## Plan

### File 1: `examples/on_policy_distillation/python_sandbox.py` (new)

**`StatefulPythonSandbox` class** — persistent subprocess REPL per sample.

- **Subprocess REPL worker**: A Python script (embedded as string constant) that runs a `while True` loop reading code blocks from stdin (delimited by a sentinel `<<<__SANDBOX_INPUT_END__>>>`), executing them via `exec(code, globals_dict)` in a persistent globals dict, and writing output to stdout (delimited by `<<<__SANDBOX_OUTPUT_END__>>>`).
- **`start()`**: Writes REPL script to a temp file, spawns `python3 -u` subprocess with `PYTHONUNBUFFERED=1`, `resource.setrlimit()` for 4GB memory cap.
- **`execute_code(code)`**: Safety check → send code over stdin pipe → read output until sentinel (via `asyncio.run_in_executor` to avoid blocking the event loop) → return result. On timeout (`asyncio.wait_for`), kill and restart subprocess. On crash (EOF from readline), restart.
- **`cleanup()`**: Kill subprocess, remove temp dir. Called in `finally` block of generate function.
- **Safety checks**: Reuse retool's dangerous pattern blacklist (import os/sys/subprocess, eval, exec, open, `__dunder__`). Whitelist of allowed modules: math, random, datetime, collections, itertools, functools, statistics, decimal, fractions, re, json, copy, numpy, sympy.
- **Concurrency**: Global `asyncio.Semaphore(32)` guards `execute_code()` calls (same pattern as retool).
- **Config**: `max_turns=10`, `max_tool_calls=8`, `python_timeout=60`, `python_memory_limit_gb=4`.

### File 2: `examples/on_policy_distillation/generate_with_python_tool.py` (new)

**Custom generate function** — multi-turn loop with tool execution. Follows the exact pattern of `examples/retool/generate_with_retool.py`.

- **Reuse from retool**: `format_conversation_with_tools()` (Jinja2 Qwen3 tool template), `postprocess_predictions()` (parse `<tool_call>`, `<code>`, `` ```python ```, `Answer: \boxed{}`), `postprocess_responses()`, `get_tool_specs()`.
- **`generate(args, sample, sampling_params)`**:
  1. Create `StatefulPythonSandbox` per sample
  2. Format prompt with tool spec via Jinja2 template
  3. Multi-turn loop (max 10 turns): sglang generate → parse tool call → `sandbox.execute_code()` → append observation with `loss_mask=0` and dummy `log_prob=0.0` → continue
  4. Set `sample.tokens`, `.response_length`, `.loss_mask`, `.rollout_log_probs`
  5. `finally: await sandbox.cleanup()`
- **Output truncation**: Cap tool output at 2000 chars to prevent context overflow from accumulated state.
- **OPD compatibility**: The existing `reward_func` and `post_process_rewards` in `on_policy_distillation.py` are unchanged. They send the full `sample.tokens` to the teacher, extract teacher log probs, and trim to `response_length`. The `loss_mask` ensures tool output tokens don't contribute to training loss.

### File 3: `examples/on_policy_distillation/run-qwen3-8B-opd-fsdp.sh` (modify)

Add one argument to the ray job submit:
```bash
--custom-generate-function-path examples.on_policy_distillation.generate_with_python_tool.generate
```

## Key files to reference during implementation

| File | What to reuse |
|------|--------------|
| `examples/retool/tool_sandbox.py` | Safety checks, dangerous patterns, allowed modules, TOOL_CONFIGS, semaphore pattern |
| `examples/retool/generate_with_retool.py` | Jinja2 template, `format_conversation_with_tools`, `postprocess_predictions`, `postprocess_responses`, `get_tool_specs`, multi-turn loop structure, loss mask + log prob handling |
| `examples/on_policy_distillation/on_policy_distillation.py` | Unchanged — reward_func and post_process_rewards work as-is with tool-augmented samples |
| `slime/rollout/sglang_rollout.py` | `GenerateState`, `post` utility, custom generate function loading |

## Verification

1. Submit a short test run with `--num-rollout 4 --rollout-batch-size 2 --n-samples-per-prompt 2` to verify:
   - Sandbox processes start and clean up (no zombie processes)
   - Variables persist across tool calls within a sample (e.g., `x=5` then `print(x)`)
   - Loss masks are correct (1 for model tokens, 0 for `<interpreter>...</interpreter>`)
   - Log prob arrays align with token counts
   - OPD teacher log probs are extracted correctly
2. Check that no sandbox subprocesses remain after the job completes
