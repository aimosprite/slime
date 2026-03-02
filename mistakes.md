# MISTAKES.md

## 1. Fixed SGLang startup errors one-by-one instead of reading ERROR_HISTORY.md

**Date:** 2024-02-24

**What happened:** When SGLang failed to start during eval, I fixed each error
individually (no sglang installed, no CUDA_HOME, no Python.h, no real nvcc)
across multiple crash-restart cycles. This took ~30 minutes of the user's time.

**What I should have done:** `eval/ERROR_HISTORY.md` already documented errors #2
(deep_gemm CUDA_HOME assertion) and the nvidia-cuda-nvcc workaround. Reading it
first would have saved several iterations.

**Lesson:** Always check existing error logs and documentation before debugging
from scratch. The codebase likely has prior art for the exact problem.

## 2. nvidia-cuda-nvcc-cu12 pip package is not a real CUDA toolkit

**Date:** 2024-02-24

**What happened:** Installed `nvidia-cuda-nvcc-cu12` pip package thinking it
provided `nvcc`. It only has `ptxas` — no actual nvcc binary. SGLang's JIT
kernel compilation (Triton, flashinfer, sgl_kernel) needs real nvcc from the
CUDA toolkit.

**Fix:** Install `cuda-nvcc-12-8` from NVIDIA's apt repo. This provides
`/usr/local/cuda-12.8/bin/nvcc`.

## 3. setup.sh used relative paths without cd to script directory

**Date:** 2024-02-24

**What happened:** Running `bash ../setup.sh` from `eval/` created `models/`
and `slime/` inside `eval/` instead of the repo root. The 120B model (126GB)
was downloaded to the wrong location.

**Fix:** Added `cd "$(dirname "$0")"` at the top of setup.sh.

## 4. SGLang rejects `<|channel|>` tags in multi-turn tool use messages

**Date:** 2024-02-24

**What happened:** gpt-oss models emit structured `<|channel|>analysis`,
`<|channel|>commentary`, `<|channel|>final` tags in their responses. In the
multi-turn tool use loop (`generate_with_tools`), the raw assistant response
was appended to the message history. When sent back to SGLang on the next
turn, SGLang returned 400: "You have passed a message containing `<|channel|>`
tags in the content field."

**Fix:** Added `_strip_channel_tags()` helper in `eval_and_log.py` that
extracts content between `<|message|>` and `<|end|>` tags, stripping the
channel wrapper. Applied to assistant messages before appending to the
conversation history (line ~696).

## 5. `_CHANNEL_CODE_RE` regex referenced but never defined in harness.py

**Date:** 2024-02-27

**What happened:** `_extract_code_blocks()` had a fallback branch using
`_CHANNEL_CODE_RE` to extract code from Harmony channel tags, but the regex was
never defined. Every GPT-OSS tool_use eval crashed with `NameError: name
'_CHANNEL_CODE_RE' is not defined`.

**Fix:** Added the regex: `_CHANNEL_CODE_RE = re.compile(r"<\|channel\|>code
(?:<\|recipient\|>\w+)?<\|message\|>(.*?)<\|end\|>", re.DOTALL)`

## 6. `<|end|>` stop token kills GPT-OSS after first channel segment

**Date:** 2024-02-27

**What happened:** GPT-OSS (Harmony format) emits multi-segment responses:
`<|channel|>analysis<|message|>...<|end|><|channel|>code<|message|>...<|end|>`.
The stop token `<|end|>` fired after the analysis segment, cutting off the model
before it could write code or `\boxed{}`. Then `finish_reason == "stop"` triggered
the break logic, ending the conversation with `extracted=None` for every problem.

**Fix:** In the "no code blocks found" branch, detect channel-format output
(`<|channel|>` or `<|message|>` in text) and `continue` the loop instead of
breaking. The model just needs another turn to produce its code/answer segment.
Also added `model_format: harmony` scaffolding_arg and GPT-OSS-specific system
prompt that instructs the model to use ```python blocks instead of channel tags.

## 7. ToolUseScaffolding stop tokens are Harmony-only — Qwen gets no stops

**Date:** 2026-02-27

**What happened:** `_TOOL_USE_STOP_TOKENS` was `["<|end|>", "<|start|>"]` —
Harmony-specific tokens. Qwen models never emit these, so the model generated
its entire response (reasoning + code + simulated output + answer) in one turn.
`turns=1` and `exec=0` for almost every problem despite `tool_use` scaffolding.
The model wrote code blocks but also hallucinated their output and continued to
`\boxed{}` without pausing for real execution.

**Fix:** Split stop tokens per `model_format`. Default (Qwen) uses
`["```output", "```Output", "```\nOutput", "```\noutput"]` to catch the model
when it tries to simulate code output. Harmony keeps the original tokens.
Stop tokens are now `self._stop_tokens` set in `__init__` based on format.

## 8. Missing model_format: harmony in GPT-OSS configs

**Date:** 2025-02-27

**What happened:** Ran GPT-OSS 120B with `scaffolding: tool_use` but forgot
`model_format: harmony` in `scaffolding_args`. The model emits code inside
Harmony `<|channel|>code<|message|>...<|end|>` tags, not standard ```python
blocks. Without `model_format: harmony`:
- Code blocks were never extracted or executed (exec=0 on most problems)
- Stop tokens were wrong (using ```output instead of <|end|>/<|start|>)
- Model rushed to \boxed{} with placeholder/wrong answers
- Score dropped from ~90% to ~64%

**Fix:** Always set `model_format: harmony` in `scaffolding_args` for any
GPT-OSS model, even with `scaffolding: default`. Updated all configs and README.

**Lesson:** GPT-OSS uses Harmony channel tags, not standard markdown. This
applies to both tool_use AND default scaffolding. If eval results look
suspiciously bad and `exec=0`, check `model_format` first.

## 9. Megatron prep torchrun fails with "No module named 'megatron'"

**Date:** 2026-03-02

**What happened:** `DO_PREP=1` ran `torchrun convert_hf_to_torch_dist.py` without
setting `PYTHONPATH`. Megatron-LM was cloned to `/root/slime/Megatron-LM/` but
not on the system path, so the import failed immediately.

**Fix:** Prepend `PYTHONPATH="${MEGATRON_PATH}"` to the torchrun call in the prep
section. Also installed Megatron-LM with `sudo pip3 install -e .` from the
cloned directory as a belt-and-suspenders measure.

## 10. transformer_engine and apex not installed — training will fail

**Date:** 2026-03-02

**What happened:** `scripts/models/qwen3-8B.sh` hardcodes `--transformer-impl
transformer_engine`. Neither `transformer_engine` nor `apex` are installed on
this machine. Training would have crashed at model initialization.

**Fix:** Made `--transformer-impl` configurable via `${TRANSFORMER_IMPL:-transformer_engine}`
in `qwen3-8B.sh`. Set `transformer_impl: local` in the config YAML to use the
pure-PyTorch fallback until the proper libraries are installed.

**To install properly:** Run `setup-env.sh` (requires uv, CUDA toolkit, ~30 min compile)
or: `pip install transformer_engine[pytorch]==2.10.0 --no-build-isolation`

## 11. Models downloaded to /root/ instead of models/ folder

**Date:** 2026-03-02

**What happened:** `POOL_DIR` in the script defaulted to `/root`, so model
artifacts (Qwen3-8B, Qwen3-8B-random-emb) were placed in `/root/` instead of
the repo's `models/` subdirectory. Created a duplicate at `/root/Qwen3-8B`.

**Fix:** Changed `POOL_DIR` default to `/root/slime/models`. Moved artifacts.
Deleted the duplicate `/root/Qwen3-8B` (16GB freed). Added `models/` and
`Megatron-LM/` to `.gitignore`.
