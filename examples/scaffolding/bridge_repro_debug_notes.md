# GPT-OSS Bridge Repro Notes

## Goal

Cheaply reproduce the RL-only inference corruption with the smallest setup that still exercises the Megatron -> HF -> SGLang weight handoff.

## Current minimal target

- Model: `openai/gpt-oss-20b`
- GPU shape: `1xH200`
- Baseline path: direct SGLang inference
- Repro path: direct SGLang baseline, then initialize a single-process Megatron actor, export/update weights through `UpdateWeightFromTensor`, then rerun the same inference

## Confirmed observations

- Direct 20B single-GPU SGLang inference works.
- Direct 120B single-GPU Fibonacci smoke worked and produced a real boxed answer.
- RL-backed 120B smoke produced gibberish before any useful Harmony/tool boundary.
- RL-backed 120B Fibonacci smoke reproduced the same gibberish behavior, so the hard problem was not the cause.
- The rollout max response length bug was real and has already been fixed elsewhere:
  - solver generation was previously ignoring `rollout_max_response_len`
  - after the fix, bad RL generations now stop at the expected capped length

## Cheap 20B repro progress

### Failure 1

- App: `ap-uJ4ueuUOhttV2PKQ6VpM9f`
- Failure: `ModuleNotFoundError: No module named 'examples'`
- Cause: direct repro script did not add `/root/slime` to `sys.path`
- Fix: insert repo root into `sys.path` at script startup

### Failure 2

- App: `ap-vBNz8RDRAjCobx0ZWJKu03`
- Failure: `ModuleNotFoundError: No module named 'megatron.training'`
- Cause: the cheap repro environment did not expose `/root/Megatron-LM` on the driver `PYTHONPATH`, even though the normal RL path does
- Fix applied:
  - add `/root/Megatron-LM` to `sys.path` in `run_gpt_oss_scaffolding_megatron_bridge_repro.py`
  - set `PYTHONPATH` for the Modal subprocess in `run_gpt_oss_scaffolding_modal.py`
  - add the same `PYTHONPATH` shape to the Ray actor runtime env inside the repro script

### Failure 3

- App: `ap-hOgmPgxM0HE6i4SJXxL893`
- Failure: `ModuleNotFoundError: No module named 'examples.scaffolding'`
- Cause: I put `/root/Megatron-LM` ahead of `/root/slime` on `PYTHONPATH`, which likely shadowed the repo `examples` package
- Fix applied:
  - keep `/root/slime` ahead of `/root/Megatron-LM` in both `sys.path` and `PYTHONPATH`

### Failure 4

- App: `ap-MdcCjsKZX3TaaSWcXYKvnJ`
- Failure: Megatron HF validation rejected the minimal repro CLI before model init
- Symptom:
  - `hidden_size None`
  - `num_layers None`
  - `num_attention_heads None`
  - `rotary_base 10000` mismatch vs HF `150000`
- Cause: the cheap repro was hand-building only generic train args and was missing the real GPT-OSS `MODEL_ARGS` that the launcher sources from `scripts/models/gpt-oss-20B.sh`
- Fix applied:
  - load exact `MODEL_ARGS` from `scripts/models/gpt-oss-20B.sh`
  - keep LoRA flags enabled on top of those exact model args

### Failure 5

- App: `ap-kb9yaeqzMcn5mlj4jtUYgp`
- Failure: `torch_memory_saver` asserted during Megatron actor init because `LD_PRELOAD` was empty
- Cause: the cheap repro was creating its Ray actor runtime env manually and did not copy the normal `RayTrainGroup` setup for colocated Megatron offload
- Fix applied:
  - add `LD_PRELOAD=<torch_memory_saver_hook_mode_preload...so>`
  - add `TMS_INIT_ENABLE=1`
  - add `TMS_INIT_ENABLE_CPU_BACKUP=1`

## Working hypothesis

- The likely bad path is still the RL-only Megatron -> HF -> SGLang weight handoff.
- LoRA is still a strong suspect because it is enabled in RL paths and the direct smoke does not rely on the same export/update path.
- This cheap repro should let us separate:
  - baseline direct inference sanity
  - post-Megatron-weight-sync corruption
  - LoRA-specific corruption vs generic bridge/update corruption

## Next checks

1. Get the cheap 20B bridge repro to run end to end with LoRA enabled.
2. Compare baseline output vs post-sync output.
3. If corrupted, rerun the same repro with `--no-enable-lora`.
4. Use the result to patch the larger RL path only after the cheap repro is understood.

### Failure 6

- App: `ap-FoMusLZ6hDghKfnAveJV75`
- Observation: the repro got all the way through:
  - baseline direct inference
  - Megatron actor init
  - LoRA attachment
  - HF -> Megatron load
  - offload
- Then it went silent right after `Timer train_wait start`.
- Interpretation:
  - `train_wait start` means actor `init()` returned and the actor is idle, not stuck inside init.
  - So the likely stall moved downstream into the main script's `ray.get(actor.connect_and_update.remote(...))` or the first part of `connect_and_update()`.
- Fix applied:
  - add explicit logs inside `DirectEngineProxy` RPC methods
  - add explicit logs inside `MegatronBridgeDebugActor.connect_and_update()`
  - wrap the driver-side `ray.get(...)` calls in polling progress logs with timeouts so the repro no longer goes silent
  - switch key driver progress markers from `logger.info(...)` to plain `print(..., flush=True)` because the driver logger output was not surfacing reliably in Modal app logs

### Failure 7

- Cheap repro proxy was not respecting `SLIME_SGLANG_SKIP_FLUSH_CACHE`.
   - The direct server startup path sets `SLIME_SGLANG_SKIP_FLUSH_CACHE=1`, and the real `SGLangEngine.flush_cache()` returns early in that case.
   - The custom `DirectEngineProxy.flush_cache()` in the repro was still hitting `/flush_cache` directly, so the repro could hang in `connect_and_update()` even though the real direct server path intentionally skips that call.
   - Fixed by making `DirectEngineProxy.flush_cache()` honor the same env flag.

### Failure 8

- App: `ap-D72dMnzIoD7WFaRpF7ou7S`
- Symptom: attached run stayed alive for a long time but app logs only showed the NVIDIA CUDA banner, with no repro-script output at all.
- Interpretation:
  - the blind spot moved earlier than the first existing `_progress(...)` call
  - possible boundaries were:
    - Modal function before `subprocess.run(...)`
    - Python child process startup
    - heavy top-level imports in `run_gpt_oss_scaffolding_megatron_bridge_repro.py`
- Fix applied:
  - add `[bridge-repro-modal]` prints before launching the child command
  - run child as `python3 -u`
  - set `PYTHONUNBUFFERED=1`
  - add `[bridge-repro] python entrypoint import start`
  - add `[bridge-repro] third-party imports loaded`
  - add `[bridge-repro] local module imports loaded`
  - enable `faulthandler.dump_traceback_later(300, repeat=True)` in `main()` to dump stacks if the repro hangs after startup

### Failure 9

- App: `ap-Mqdiqs4MdbItlX8VrH2pB1`
- Observation:
  - the child process now definitely starts
  - it gets through:
    - `[bridge-repro-modal] function start`
    - child launch command
    - `[bridge-repro] python entrypoint import start`
    - `[bridge-repro] third-party imports loaded`
  - but it still does not reach `local module imports loaded`
- Interpretation:
  - the current stall is inside the local import block, not Modal startup and not generic Python/third-party imports
- Fix applied:
  - split local imports into per-module checkpoints so the next rerun identifies the exact module:
    - `gs_config`
    - `reward_gpt_oss_scaffolding`
    - `rollout_gpt_oss_scaffolding`
    - `run_gpt_oss_scaffolding_rl`
    - `run_gpt_oss_scaffolding_sglang_direct_smoke`
    - `MegatronTrainRayActor`
  - later interrupt showed the run had already advanced into `_run_repro()` and was blocked in `_resolve_hf_checkpoint_to_local_dir()` -> `snapshot_download(...)`
  - cause: the bridge repro wrapper was not explicitly setting `HF_HOME=/root/data/hf-cache`, so it could miss the shared Modal volume cache and redownload `gpt-oss-20b`
  - fix applied: set both `HF_HOME=/root/data/hf-cache` and `HUGGINGFACE_HUB_CACHE=/root/data/hf-cache/hub` in `smoke_megatron_bridge_repro_20b_modal()`

### Failure 10

- App: `ap-7TAja5xJPCy9heRNAV6QXO`
- Observation:
  - the run was still dying during import, before the actual baseline-vs-post-sync comparison
  - the captured stack showed child-process re-import of `run_gpt_oss_scaffolding_megatron_bridge_repro.py` descending into:
    - `run_gpt_oss_scaffolding_sglang_direct_smoke`
    - `sglang.__init__`
    - `sglang.lang.*`
    - `sglang.utils`
    - `IPython.display`
- Interpretation:
  - the repro script was importing the direct-smoke helper module at top level
  - because the direct-smoke module imports SGLang at top level, every spawned child re-import of the main module paid that SGLang/IPython import cost before reaching Megatron init
  - this is a harness bug, not yet evidence of model corruption
- Fix applied:
  - remove the top-level import of `run_gpt_oss_scaffolding_sglang_direct_smoke`
  - inline `_load_jsonl_row()` and `_make_sampling_params()` in the repro script
  - move SGLang-specific imports behind the local `_start_direct_server()` helper so only the direct-server startup path imports SGLang

### Failure 11

- App: `ap-F7G104f8KlVisljxz4J82b`
- Observation:
  - the repro now gets past the old direct-smoke import trap and reaches:
    - `importing run_gpt_oss_scaffolding_rl`
    - `importing MegatronTrainRayActor`
  - after stopping the run, the last stack was no longer in SGLang; it was in:
    - `megatron.core.inference.contexts.dynamic_context`
    - `import wandb`
  - every frame still flowed through `sitecustomize._patched_import`
- Interpretation:
  - after the direct SGLang server starts, the repro leaves `SLIME_ENABLE_GLOBAL_SGLANG_PATCH=1` set in the parent process
  - later child imports for Ray/Megatron inherit that env var and pay the global import-hook cost even though they are no longer on the direct-server path
- Fix applied:
  - scope `SLIME_ENABLE_GLOBAL_SGLANG_PATCH=1` only to the direct-server `launch_server_process(...)` call, then restore the parent env immediately after launch
  - keep `SLIME_SGLANG_SKIP_FLUSH_CACHE=1` and health-mode envs in the parent because the proxy still reads them

### Failure 12

- Observation:
  - even after the direct-server import cleanup, the repro still front-loads Megatron actor import at module import time
  - that means the script can spend a long time inside Megatron package import work before it even reaches the baseline direct-inference comparison
- Interpretation:
  - for the minimal repro, the actor path only needs to start after the direct baseline inference is complete
  - keeping the actor import at module import time makes child-process startup noisier and harder to separate from the direct baseline path
- Fix applied:
  - remove the top-level `MegatronTrainRayActor` import
  - build the debug actor subclass lazily inside `_run_repro()` right before creating the Ray actor

### Failure 13

- App: `ap-YbAhWNhxKW1CA3uToG9ZiW`
- Observation:
  - the repro got through:
    - direct 20B baseline inference
    - local Ray startup
    - Megatron actor init
    - LoRA attachment
    - HF -> Megatron initialization via `mbridge.load_weights(memory_efficient=True)`
  - then `actor.connect_and_update` stayed pending for minutes
  - actor-side logs reached:
    - `MegatronBridgeDebugActor.connect_and_update: connecting rollout engine`
    - `MegatronBridgeDebugActor.connect_and_update: starting weight update`
  - proxy-side logs reached:
    - `DirectEngineProxy POST pause_generation`
  - but never reached:
    - `DirectEngineProxy POST pause_generation -> ...`
- Interpretation:
  - the hang is earlier than tensor export or `update_weights_from_tensor`
  - the first blocking point is likely the direct SGLang `/pause_generation` control endpoint
  - that still leaves two possibilities:
    - `/pause_generation` is already unhealthy on the untouched direct server after one baseline inference
    - or Megatron init changes something that makes `/pause_generation` hang before the actual bridge update starts
- Fix applied:
  - add explicit timeouts to the repro proxy control requests
  - add a pre-Megatron direct-engine control probe:
    - `pause_generation`
    - `continue_generation`
    - `flush_cache`
    - `get_weight_version`
  - add the same probe again after Megatron actor init but before `connect_and_update()`
  - record both probe results in the repro artifact

### Failure 14

- App: `ap-YblWt8blGyJVe02EeA6ghB`
- Observation:
  - the new pre-Megatron control probe fires before Megatron init
  - it still hangs immediately on:
    - `pre_megatron_controls.pause_generation`
  - the proxy reports:
    - `DirectEngineProxy POST pause_generation timeout_s=30.0`
    - `pre_megatron_controls.pause_generation still pending after 5.0s`
    - `pre_megatron_controls.pause_generation still pending after 10.0s`
    - `pre_megatron_controls.pause_generation still pending after 15.0s`
- Interpretation:
  - the first broken boundary is now clearly before Megatron init
  - the strongest explanation is that the streamed baseline solve returns as soon as the repro detects a valid answer, while the direct SGLang server keeps decoding that request in the background
  - that leaves `/pause_generation` blocked even on the untouched direct server
- Fix applied:
  - add `DirectEngineProxy.abort_all_requests()` hitting `/abort_request` with `{"abort_all": true}`
  - after the baseline streamed solve, the repro now explicitly aborts any leftover in-flight request and rechecks `get_weight_version` before probing `pause_generation`

### Failure 15

- App: `ap-LY8kPLfkpHL1A4DulDtPgp`
- Observation:
  - the cheap repro now gets through:
    - direct baseline inference
    - direct-engine drain
    - pre/post-init control probes
    - Megatron actor init
    - LoRA attachment
    - HF -> Megatron load
    - the start of `connect_and_update()`
  - the new instrumentation narrowed the crash to bridge export:
    - `weights_getter done rank=0 count=3387`
    - `get_hf_weight_chunks start local_weight_count=3387`
  - then Megatron Bridge fails while collecting global param names:
    - `torch.distributed.all_gather_object(...)`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - this is no longer a generic SGLang control problem
  - the failure is inside Megatron Bridge's Megatron->HF export path
  - on this cheap repro the pipeline model parallel size is 1, so gathering object names across the PP group is unnecessary
  - the most plausible bridge bug is that the stock export path still runs an NCCL-backed object collective even when `PP=1`
- Fix applied:
  - patch the installed Megatron Bridge `MegatronModelBridge._megatron_global_param_names_all_pp_ranks()` path via `slime_plugins/megatron_bridge/__init__.py`
  - for `PP=1`, skip the object gather and enumerate global param names locally instead
  - also tighten repro progress logging so `_ray_get_with_progress()` only prints `completed` after `ray.get(...)` succeeds

### Failure 16

- App: `ap-EWRlXNpnZlEnlCWUtdLk1J`
- Observation:
  - with the PP=1 export patch in place, the cheap repro again gets through:
    - direct baseline inference
    - local Ray startup
    - direct-engine drain
    - pre-Megatron control probes
    - actor import
    - actor handle creation
  - it then stalls earlier than the old bridge-export crash:
    - `Starting Megatron actor init`
    - `actor.init_from_cli still pending after 15.0s`
    - `actor.init_from_cli still pending after 30.0s`
    - `actor.init_from_cli still pending after 45.0s`
    - `actor.init_from_cli still pending after 60.0s`
  - the app was killed by a local `modal run` keyboard interrupt, not by an internal traceback
- Interpretation:
  - after fixing the PP=1 bridge-export crash, the next unknown boundary is inside `init_from_cli()`
  - current logs are too coarse to tell whether the stall is:
    - `parse_args()`
    - Megatron distributed init
    - HF config/tokenizer load
    - model setup / LoRA wrapping
    - checkpoint load
    - or weight_updater construction
- Fix applied:
  - add detailed init phase tracing in `run_gpt_oss_scaffolding_megatron_bridge_repro.py`
  - add detailed init phase tracing in `slime/backends/megatron_utils/actor.py`
  - add detailed init phase tracing in `slime/backends/megatron_utils/model.py`
  - set `SLIME_BRIDGE_REPRO_TRACE_INIT=1` in the cheap repro actor runtime env

### Failure 17

- App: `ap-hvWgQQESwFQtahKTDz3Wjs`
- Observation:
  - detached rerun got through:
    - direct server startup
    - direct baseline inference
    - post-baseline drain
    - pre-Megatron control probes
  - it then died before any of the new `init_from_cli()` phase markers fired
  - last live boundary in logs:
    - `Importing MegatronTrainRayActor lazily`
    - `package init start`
    - `importing deep_ep`
    - `imported deep_ep`
    - `importing megatron.bridge qwen rotary patch targets`
  - app ended with:
    - subprocess `SIGSEGV`
- Interpretation:
  - the new crash is earlier than actor init and earlier than the bridge export path
  - the strongest current suspect is the unconditional Qwen-VL rotary patch import in `slime/backends/megatron_utils/__init__.py`
  - that import is unrelated to GPT-OSS and runs in the same process that already owns the live SGLang server
- Fix applied:
  - gate the Qwen rotary patch import behind `SLIME_ENABLE_QWEN_ROTARY_PATCH`
  - disable that patch in the cheap GPT-OSS repro main process and actor runtime env

### Failure 18

- App: `ap-mrUZToZ7CW95zO9ZNIuL3F`
- Observation:
  - the cheap repro now gets through the full actor-init path:
    - `parse_args`
    - Megatron distributed init
    - LoRA attachment
    - HF -> Megatron load
    - weights_backuper creation
    - weight_updater construction
    - actor offload
    - post-init direct-engine control probes
  - it then fails in `connect_and_update()` with a concrete bridge-export error:
    - `hf_weight_iterator_bridge.py:76`
    - `new_param_weight.cuda()`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the small repro is now using the same Megatron + LoRA + bridge export path as the trainer and reproducing a real narrow failure
  - the bridge iterator is using a different CPU->GPU transfer call than the known-good direct iterator
  - `hf_weight_iterator_direct.py` uses:
    - `.to(device=torch.cuda.current_device(), non_blocking=True)`
  - `hf_weight_iterator_bridge.py` was using:
    - `.cuda()`
  - the current best hypothesis is that the bridge iterator's transfer path is wrong for these pinned CPU backup tensors
- Fix applied:
  - change bridge iterator tensor transfer to match the direct iterator:
    - `.to(device=torch.cuda.current_device(), non_blocking=True)`
  - add failure diagnostics printing tensor key/device/dtype/shape if that transfer still fails

### Failure 19

- App: `ap-qhAU0fziPeuaHx2bHveO9r`
- Observation:
  - the cheap repro now completes the full 20B actor-init path with LoRA enabled:
    - direct baseline inference
    - Megatron actor init
    - LoRA attachment
    - HF -> Megatron load
    - weights_backuper creation
    - weight_updater construction
    - post-init rollout-engine control probes
  - it still fails in `connect_and_update()` at the first Bridge export transfer:
    - `hf_weight_iterator_bridge.py:79`
    - `new_param_weight.to(device=torch.cuda.current_device(), non_blocking=True)`
    - `torch.AcceleratorError: CUDA error: invalid argument`
  - this reproduces after the earlier `.cuda()` -> `.to(...)` change, so the issue is not just the choice of one CUDA copy API
- Interpretation:
  - the remaining fault domain is now very narrow:
    - Bridge export is consuming tensors returned by `weights_backuper.get("actor")`
    - those tensors are pinned CPU backups created by `TensorBackuper`
  - the current best hypothesis is that some of those CPU backup tensors have a layout / stride / dtype combination that Megatron Bridge's naive CPU -> CUDA handoff cannot handle directly
- Fix applied:
  - force synchronous CUDA errors in the cheap repro actor env with `CUDA_LAUNCH_BLOCKING=1`
  - log representative `weights_getter()` tensor metadata before Bridge export
  - log the first few Bridge conversion task keys and tensor metadata
  - sanitize CPU backup tensors before handoff by materializing a contiguous CPU tensor and then copying into a fresh CUDA tensor explicitly

### Failure 20

- App: `ap-U7pD1Ofm14rsTYOmzsOYb7`
- Observation:
  - the new diagnostics show `weights_getter()` is returning normal-looking pinned CPU backup tensors:
    - contiguous
    - standard row-major strides
    - BF16 for the first sampled tensors
  - example first tensor:
    - `vp_stages.0.embedding.word_embeddings.weight`
    - `device=cpu`
    - `dtype=torch.bfloat16`
    - `shape=(201088, 2880)`
    - `stride=(2880, 1)`
    - `pinned=True`
  - Bridge export still fails almost immediately after `get_hf_weight_chunks start`, now at:
    - `gpu_weight.copy_(new_param_weight, non_blocking=False)`
    - same `torch.AcceleratorError: CUDA error: invalid argument`
  - importantly, no `task preview` or `conversion_tasks count` lines were emitted before the crash
- Interpretation:
  - the bridge crash is still happening while trying to re-materialize actor weights from CPU backup into CUDA for export
  - this makes the pinned-CPU backuper round-trip itself the strongest current suspect, not the particular CUDA copy API
  - Bridge mode should not need `TensorBackuper.get("actor")` for rollout export at all, because update_weights is exporting the live actor parameters immediately after training / resume
- Fix applied:
  - change the Megatron actor's bridge-mode `weight_updater` getter to use live actor tensors directly via `named_params_and_buffers(..., convert_to_global_name=False, translate_gpu_to_cpu=False)`
  - keep `TensorBackuper` in place for ref / old-actor bookkeeping, but stop routing bridge export through the pinned CPU actor snapshot

### Failure 21

- App: `ap-FhbuUoxrl5qtuwKSrnzQze`
- Observation:
  - the live-actor bridge getter change worked in the sense that the old CPU-backup round-trip crash is gone
  - `weights_getter()` now returns live CUDA actor tensors:
    - sampled tensors were on `cuda:0`
    - sampled tensors were contiguous BF16 tensors with standard strides
  - Bridge export gets past `weights_getter()` and into `_send_hf_params(...)`
  - the new failure point is later, inside SGLang flattened-bucket construction:
    - `FlattenedTensorBucket(named_tensors=...)`
    - `torch.cat(flattened_tensors, dim=0)`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the primary fault boundary moved forward from "CPU backup -> CUDA rematerialization" to "flattening the exported HF tensors for colocated SGLang update"
  - this makes the current best hypothesis a bad tensor mix in the first exported HF chunk, especially the SGLang multi-dtype flattened-bucket path
- Fix applied:
  - log the first colocated HF chunk's device / dtype / contiguity histogram before bucketing
  - add an env-controlled repro-only fallback to split flattened buckets by dtype:
    - `SLIME_SGLANG_FORCE_SINGLE_DTYPE_BUCKETS=1`

### Failure 22

- App: `ap-AedQjXY8hyjV9H1d5BArC4`
- Observation:
  - the new chunk logging showed the crash is even narrower than "mixed dtype bucket"
  - the very first exported HF chunk is:
    - `chunk_len=1`
    - `sample_names=['model.norm.weight']`
    - dtype histogram `{'torch.bfloat16': 1}`
    - device histogram `{'cuda:0': 1}`
    - contiguity histogram `{'contiguous': 1}`
  - even with per-dtype bucketing forced, `FlattenedTensorBucket(named_tensors=[('model.norm.weight', ...)])` still fails on:
    - `torch.cat(flattened_tensors, dim=0)`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the current culprit is not the mixed-dtype path
  - the problem is more likely that Bridge export is yielding a CUDA tensor whose storage / view semantics look contiguous but are still incompatible with SGLang's bucket flattening path
  - the next best hypothesis is that we need to materialize each exported HF tensor into fresh storage before handing it to `FlattenedTensorBucket`
- Fix applied:
  - add repro-only tensor normalization before bucketing:
    - `SLIME_SGLANG_CLONE_BEFORE_BUCKET=1`
    - each exported HF tensor is now `detach().contiguous().clone()`d on-device before bucket construction
  - log the first few exported HF tensors' `storage_offset` and `data_ptr` so we can distinguish raw bridge views from fresh cloned tensors

### Failure 23

- App: `ap-NGhl3nZRsmGKjDcyUi1s7L`
- Observation:
  - the repro reached the exact same first exported HF chunk:
    - `chunk_len=1`
    - `sample_names=['model.norm.weight']`
  - but now the failure happens even earlier than `FlattenedTensorBucket`
  - the first exported HF tensor is invalid enough that:
    - `tensor.detach().contiguous().clone()`
    - immediately raises `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the broken object is the GPU tensor yielded by `Megatron Bridge export_hf_weights(cpu=False)` itself
  - this strongly points to the bridge GPU-export path, not the downstream SGLang send logic
- Fix applied:
  - add a repro-only bridge export toggle:
    - `SLIME_BRIDGE_EXPORT_CPU=1`
  - `HfWeightIteratorBridge` now logs whether `export_hf_weights` is running with `cpu=True` or `cpu=False`
  - next test is to keep the same Megatron + LoRA + bridge path, but force the bridge exporter to emit CPU tensors

### Failure 24

- App: `ap-SVMogjRZvUe2KO3tL6wV8E`
- Observation:
  - the repro now gets all the way through:
    - direct baseline inference
    - local Ray startup
    - direct-engine drain
    - Megatron actor init
    - LoRA attachment
    - HF -> Megatron initialization
    - actor backup / updater construction
    - `connect_and_update()` startup
  - with `SLIME_BRIDGE_EXPORT_CPU=1`, the failure moves earlier than SGLang bucketing:
    - `weights_getter done rank=0 count=3387`
    - `get_hf_weight_chunks start local_weight_count=3387`
    - then Megatron Bridge itself fails in:
      - `stream_weights_megatron_to_hf`
      - `final_tensor = tensor.cpu() if cpu else tensor`
      - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - this proves the current bad tensor exists before `UpdateWeightFromTensor` hands anything to SGLang
  - `cpu=True` does not fix the export, so the defect is upstream of SGLang flattening/bucketing
  - the next missing fact is the exact exported task / HF param that first trips `tensor.cpu()`
- Fix applied:
  - add an env-gated tracing patch around `MegatronModelBridge.stream_weights_megatron_to_hf`
  - keep the same bridge export logic, but log:
    - task index
    - `task.global_param_name`
    - converted HF names
    - tensor metadata
    - the exact HF name at the failing `tensor.cpu()` boundary
  - enable that tracing in the cheap repro actor env via `SLIME_BRIDGE_TRACE_EXPORT_FAILURE=1`

### Failure 25

- App: `ap-0M8t8uYuZrzKXHKBxACIvL`
- Observation:
  - the cheap repro reached the direct baseline solve and then the full trainer-side path:
    - local Ray
    - lazy Megatron actor import
    - LoRA attachment
    - HF -> Megatron initialization
    - actor backup / updater construction
  - disabling SGLang CUDA graph capture for the cheap repro materially reduced startup time without changing the Megatron/export path
  - the baseline direct 20B inference completed with a sane numeric answer (`extracted=50975`) rather than gibberish, which is enough to use as a control for bridge corruption
- Fix applied:
  - set `sglang_disable_cuda_graph=True` in the cheap repro's direct-server runtime args

### Failure 26

- App: `ap-0M8t8uYuZrzKXHKBxACIvL`
- Observation:
  - the task-level export tracer patch itself broke the bridge path before the original export bug could surface
  - at `connect_and_update()` the run failed with:
    - `AttributeError: 'GPTOSSBridge' object has no attribute 'build_adapter_conversion_tasks'`
  - this means the installed Megatron Bridge build in Modal is older than the API shape I assumed when copying the tracing shim
- Interpretation:
  - this is a repro harness bug, not evidence about the underlying inference corruption
  - the tracer must not change control flow on older bridge builds
- Fix applied:
  - if `build_adapter_conversion_tasks` is missing, the tracing shim now logs that adapter tracing is unavailable and falls back to the original `stream_weights_megatron_to_hf`

### Failure 27

- App: `ap-cMm9sM260CyvV9X6O8481x`
- Observation:
  - the cheap 20B repro now has a valid control path:
    - direct `1xH200` SGLang startup succeeded
    - baseline direct inference on `fib100_mod_100000` finished with `reward=1.0` and `extracted=15075`
  - the same run then entered the trainer-style Megatron path with LoRA enabled:
    - local Ray actor startup
    - GPT-OSS Megatron init from trainer-like args
    - LoRA attachment
    - HF -> Megatron load
    - actor backup / updater construction
    - `connect_and_update()`
  - the failure is now decisively inside the original Megatron Bridge export path, before SGLang receives any tensors:
    - `weights_getter done rank=0 count=3387`
    - `get_hf_weight_chunks start local_weight_count=3387`
    - `/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/conversion/model_bridge.py:641`
    - `final_tensor = tensor.cpu() if cpu else tensor`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - direct inference can be correct on the exact same prompt/problem
  - corruption is upstream of SGLang update/bucketing
  - the most important remaining discriminator is whether this only happens with LoRA enabled
- Fix applied:
  - expose `enable_lora` on the cheap repro Modal entrypoint so the same bridge repro can be rerun with LoRA disabled

### Failure 28

- App: `ap-xyrHUhIEk7865ARQTYLY3b`
- Observation:
  - the no-LoRA control is now real:
    - same cheap 20B bridge repro
    - same `fib100_mod_100000` row
    - `enable_lora=False` confirmed in logs
  - direct baseline inference remained sane rather than gibberish:
    - `Baseline inference finished status=completed reward=0.0 extracted=50975`
  - the run drained and probed the direct engine successfully, then moved into trainer-style Megatron init
  - but it never reached bridge export
  - failure moved earlier to Megatron actor initialization:
    - `torch.OutOfMemoryError: Tried to allocate 38.96 GiB`
    - direct SGLang process was already holding about `124.43 GiB`
    - only about `15.36 GiB` free remained on the H200
- Interpretation:
  - no-LoRA is not yet a clean discriminator for the bridge bug because it dies before `connect_and_update()`
  - the failure is still informative: full-trainable Megatron actor init needs substantially more free memory than the LoRA-enabled path
  - to test no-LoRA on 1xH200, the direct debug server must use a much smaller KV/cache footprint
- Next step:
  - rerun the same no-LoRA control with much lower `sglang_mem_fraction_static` and smaller context/response limits so actor init can complete

### Failure 29

- App: `ap-YyZ9Optk42SPrZ6E2H0s7t`
- Observation:
  - reducing the direct 20B server cache footprint from `0.60` to `0.25` improved free memory on the H200, but the no-LoRA control still never reached bridge export
  - the run again confirmed the intended control setup:
    - same cheap 20B bridge repro
    - same `fib100_mod_100000` row
    - `enable_lora=False`
  - the direct server came up with:
    - `mem_fraction_static=0.25`
  - but trainer-style Megatron actor init still failed before `connect_and_update()`:
    - `torch.OutOfMemoryError: Tried to allocate 77.91 GiB`
    - only about `25.15 GiB` free remained
    - the direct SGLang process was still using about `114.64 GiB`
  - the run also exposed a harness mismatch:
    - the bridge repro Modal function was still ignoring generic `rollout_max_context_len` / `rollout_max_response_len` overrides
    - so this run still launched with the old `4096/2048` limits
- Interpretation:
  - no-LoRA on `1xH200` is still blocked by actor-init memory, not by the bridge export itself
  - the most useful next step is to honor the smaller rollout caps and squeeze the direct server further before deciding whether a 2-GPU split is necessary
- Fix applied:
  - the bridge repro Modal entrypoint now falls back to generic rollout cap overrides when bridge-specific keys are absent

### Failure 30

- App: `ap-BqP51Q5aoOtS9gCYEQyPmK`
- Observation:
  - the cheap LoRA-enabled 20B repro made it much farther with the tighter `1024/512` caps:
    - direct baseline inference finished
    - local Ray initialized
    - direct-engine drain and control probes passed
    - Megatron actor init completed far enough to attach LoRA and start HF -> Megatron loading
    - the run reached `connect_and_update()`
  - the new failure was not the original bridge export bug
  - instead, the task-logging shim I added for older bridge builds tripped first:
    - `AttributeError: 'NoneType' object has no attribute 'param_weight'`
    - source: iterating the processed `conversion_tasks` wrapper in the tracer path before handing it back to the original exporter
- Interpretation:
  - this is a tracer harness bug, not evidence that the underlying LoRA export problem changed
  - the useful part is that the cheap repro now reliably reaches `connect_and_update()` on the real trainer-like LoRA path
- Fix applied:
  - skip `None` entries when wrapping conversion tasks for older bridge builds so the tracer can stay on the original adapter-merge path

### Failure 31

- App: `ap-40JsasmkFfYU30e850jo04`
- Observation:
  - the second LoRA-enabled rerun again reached the real trainer path:
    - baseline direct inference
    - local Ray
    - full Megatron actor init with bridge LoRA attached
    - `connect_and_update()`
  - the tracer still failed before the original bridge export because the processed conversion-task iterator itself did not tolerate `None` task entries
  - exact failure:
    - `AttributeError: 'NoneType' object has no attribute 'param_weight'`
    - source: `HfWeightIteratorBridge._process_conversion_tasks()._handle_one`
- Interpretation:
  - the cheap repro now reliably reaches the exact export/update boundary
  - the remaining blocker is still tracer-side iterator hygiene, not the underlying export corruption
- Fix applied:
  - `_process_conversion_tasks()._handle_one` now passes through `None` task entries unchanged

### Failure 32

- App: `ap-cB7wDeUMdHUkuMD4MyOOM4`
- Observation:
  - the cheap LoRA-enabled 20B repro again reached the true trainer-side export boundary:
    - direct `1xH200` server
    - local Ray
    - LoRA-wrapped Megatron actor init
    - HF -> Megatron initialization
    - `connect_and_update()`
  - the latest failure was still tracer-side, but now much narrower:
    - `AttributeError: 'WeightConversionTask' object has no attribute 'global_param_name'`
    - source: the logging wrapper in `slime_plugins/megatron_bridge.__init__` when trying to print task metadata before `task.mapping.megatron_to_hf(...)`
- Interpretation:
  - the installed Bridge build uses a leaner `WeightConversionTask` schema than the newer one the tracer was written against
  - the useful part is that the repro is now definitely inside the original export loop before the tracer trips
- Fix applied:
  - make the tracer use safe task-name fallbacks (`global_param_name`, then `param_name`, then `name`) instead of assuming `global_param_name` exists

### Failure 33

- App: `ap-P1VFOh57QtI3Q2VdXVJmvg`
- Observation:
  - the cheap LoRA-enabled 20B repro now cleanly reproduces the real trainer-side bug end to end on `1xH200`
  - it reaches:
    - direct SGLang startup
    - local Ray
    - full LoRA-wrapped Megatron actor init
    - HF -> Megatron initialization
    - weights_backuper + `UpdateWeightFromTensor`
    - `connect_and_update()`
    - `weights_getter done rank=0 count=3387`
    - `get_hf_weight_chunks start local_weight_count=3387`
  - then fails in original Megatron Bridge export before any HF chunk is yielded:
    - `model_bridge.py:641`
    - `final_tensor = tensor.cpu() if cpu else tensor`
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the cheap repro is now sufficient to debug the real issue
  - the failure is upstream of SGLang send/bucketing and downstream of local Megatron tensor collection
  - the fault domain is now narrowed to Bridge export of converted HF tensors on the LoRA-enabled path
- Fix applied:
  - add a repro-only toggle so Bridge export can be rerun with `merge_adapter_weights=False` while keeping the same LoRA-wrapped Megatron actor path

### Failure 34

- App: `ap-z93lag909xa7AbuYpD6Jn5`
- Observation:
  - the no-LoRA control with the same cheap `1xH200`, `skip-baseline`, and `0.25` SGLang mem fraction still does not reach export
  - it fails earlier in Megatron DDP grad-buffer allocation:
    - `torch.OutOfMemoryError: Tried to allocate 77.91 GiB`
    - only about `25.21 GiB` free remained on the H200
- Interpretation:
  - the single-GPU no-LoRA control is still not a usable export discriminator at this footprint
  - the practical cheap fault domain remains the LoRA-enabled path, because that is the only one that fits far enough to reproduce the export bug on `1xH200`

### Failure 35

- App: `ap-Oo5zgkNowLjWESOzmQ9tmf`
- Observation:
  - the merge-disabled LoRA repro now gets past the old `AutoBridge.export_hf_weights(..., merge_adapter_weights=...)` signature mismatch
  - it also gets past the earlier CUDA-side `tensor.cpu()` invalid-argument crash that appeared on the default LoRA merge path
  - with `merge_adapter_weights=False`, the first new failure is instead:
    - `AttributeError: 'GPTOSSBridge' object has no attribute '_share_embeddings_and_output_weights'`
    - source: our traced `stream_weights_megatron_to_hf` compatibility patch in `slime_plugins/megatron_bridge.__init__`
- Interpretation:
  - this is a compatibility bug in our fallback export shim, not the underlying trainer/export corruption
  - more importantly, the fact that the merge-disabled run bypassed the old CUDA export crash is strong evidence that the original `invalid argument` failure is specifically in Bridge LoRA adapter merging, not generic base-weight export
- Fix applied:
  - make the traced export patch fall back to `model_config.tie_word_embeddings` / `untie_embeddings_and_output_weights` when the older Bridge build does not provide `_share_embeddings_and_output_weights`

### Failure 36

- App: `ap-cYkdpPNK9xlXf4F7yqIFo8`
- Observation:
  - the merge-disabled LoRA repro got one step further into the patched direct `stream_weights_megatron_to_hf` fallback
  - it began the first real base-weight export task:
    - `task start idx=0 global=decoder.final_layernorm.weight`
  - then failed on another helper that exists on newer Bridge builds but not this one:
    - `AttributeError: 'GPTOSSBridge' object has no attribute '_should_skip_mtp_duplicate_embedding_export'`
    - source: our traced export shim in `slime_plugins/megatron_bridge.__init__`
- Interpretation:
  - this is still compatibility fallout from running the newer traced export shim against an older Bridge build
  - the useful part is unchanged: merge-disabled export is still progressing farther than the original merge-enabled crash path, which continues to support the hypothesis that the original CUDA invalid-argument bug is in adapter merging
- Fix applied:
  - add a compatibility fallback that treats missing `_should_skip_mtp_duplicate_embedding_export` as `False`

### Failure 37

- App: `ap-b0q7da1GiI7j1TMyRORySu`
- Observation:
  - the merge-disabled LoRA repro got farther again:
    - through `connect_and_update()`
    - through `weights_getter`
    - into the first base export task
    - `task start idx=0 global=decoder.final_layernorm.weight`
  - the next failure is still a compatibility mismatch in the older Bridge build:
    - `TypeError: GPTOSSBridge.maybe_modify_converted_hf_weight() takes 3 positional arguments but 4 were given`
    - source: our traced export shim calling the newer 3-argument form
- Interpretation:
  - this is still not the underlying export corruption
  - it confirms the merge-disabled path is now executing real base-weight conversion logic, which keeps strengthening the case that the original CUDA invalid-argument crash is adapter-merge-specific
- Fix applied:

### Failure 38

- App: `ap-sGKLCLmFIPa9idDTFrL4Wg`
- Observation:
  - the cheap repro was replaced with the real RL topology on `gpt-oss-20b` / `1xH200`
  - this run used the actual trainer-side actors:
    - `MegatronTrainRayActor`
    - `RolloutManager`
    - `SGLangEngine`
    - normal `actor_model.update_weights()`
  - it got through real SGLang bring-up and into the real Megatron -> HF -> SGLang handoff
  - the first two HF chunks uploaded successfully:
    - `model.norm.weight`
    - `model.embed_tokens.weight`
  - then it failed during Bridge export with:
    - `AttributeError: 'NoneType' object has no attribute 'mapping'`
    - source: `model_bridge.py` inside `stream_weights_megatron_to_hf`
- Interpretation:
  - the simplified cheap repro was no longer necessary to see a real shared-path bug
  - the actual RL path had a concrete conversion-task bug before the original gibberish-output symptom
- Fix applied:
  - sanitize Bridge `conversion_tasks` on the active/original export branch and drop:
    - `None` tasks
    - tasks whose `mapping` is `None`

### Failure 39

- App: `ap-HiwaqfbHfH1it1n1hN5bfp`
- Observation:
  - the real `20b` / `1xH200` RL-topology smoke got past the old Bridge export crash
  - `update_weights()` completed successfully on the real trainer path
  - after that, the run started the actual rollout generation and reproduced the original bad behavior:
    - `Solver attempt 0 debug: using_harmony=True ...`
    - no Harmony action or final answer after `512`, `1024`, `1536` streamed tokens
    - preview text was punctuation-heavy gibberish such as:
      - `, to, (,. , ( (!/,, a ----- - - ...,./::., ...`
- Interpretation:
  - the real shared-path Bridge crash was fixable
  - once that crash was removed, the original post-sync inference corruption reappeared on the cheap real RL topology
  - the remaining bug is now downstream of successful weight sync, not an early export/transport failure
- Next focus:
  - determine whether the post-sync weights differ from the original checkpoint
  - top suspect remains LoRA-enabled export/update behavior, since direct notebook-like inference without this path remains sane

### Failure 40

- App: `ap-Vm8h1cTQ4Ksv93A4oy0ZYi`
- Observation:
  - the real small-RL `20b` / `1xH200` control with LoRA fully disabled did use the correct no-LoRA command
  - it did not reach rollout quality comparison
  - it failed during Megatron DDP grad-buffer allocation with:
    - `torch.OutOfMemoryError: Tried to allocate 77.91 GiB`
    - only about `42.26 GiB` free remained on the H200
- Interpretation:
  - the no-LoRA control is now real, not a launcher placebo
  - but on this cheap topology it is still memory-blocked before `update_weights()` / rollout
  - so it does not yet discriminate inference corruption vs no-LoRA behavior

### Failure 41

- App: `ap-IosZef2N8WbYxtLjloC0bG`
- Observation:
  - the real small-RL `20b` / `1xH200` control with:
    - LoRA enabled
    - `merge_adapter_weights=0`
    - exported-vs-original tensor-compare instrumentation
  - got to `update_weights()` on the shared path
  - then failed in the active non-tracing export branch with:
    - `AttributeError: 'GPTOSSBridge' object has no attribute 'hf_config'`
    - source: installed `GPTOSSBridge.maybe_modify_converted_hf_weight`
- Interpretation:
  - this is another old-Bridge compatibility bug, not the underlying inference corruption
  - the active/original export branch was not seeding `self.hf_config`, even though the traced branch already had that compatibility fallback
- Fix applied:
  - seed `self.hf_config` from `hf_pretrained.config` in the active/original `_yield_from_original(...)` path too
  - make the traced export shim call `maybe_modify_converted_hf_weight` based on the bound-method signature at runtime

### Failure 38

- App: `ap-XelR3f4pAZLQFQNmyRZqEt`
- Observation:
  - the merge-disabled LoRA repro got through:
    - full trainer-style Megatron init
    - LoRA wrapping
    - HF -> Megatron load
    - `connect_and_update()`
    - `weights_getter`
    - the first base export task
  - then failed inside the older GPT-OSS bridge helper:
    - `AttributeError: 'GPTOSSBridge' object has no attribute 'hf_config'`
    - source: `gpt_oss_bridge.py` inside `maybe_modify_converted_hf_weight(...)`
- Interpretation:
  - this is still compatibility fallout in the older Bridge build, not the underlying export corruption
  - the useful result is unchanged: merge-disabled export continues to get deeper into real base-weight conversion than the original merge-enabled invalid-argument crash path
- Fix applied:
  - seed missing `self.hf_config` from `hf_pretrained.config` inside the traced export shim before task conversion starts

### Failure 39

- App: `ap-E1AeWmPHndG8uRO9WRmOYy`
- Observation:
  - the merge-disabled LoRA repro got through the same export boundary again
  - the missing `hf_config` issue is gone
  - the next compatibility failure is still in the older GPT-OSS bridge helper:
    - `AttributeError: 'dict' object has no attribute 'param_name'`
    - source: `maybe_modify_converted_hf_weight(...)`
- Interpretation:
  - the older bound method expects `task` as its first argument
  - my previous runtime shim only switched on arity, so on this build it incorrectly passed `converted_weights_dict` into the `task` slot
- Fix applied:
  - make the traced export shim dispatch `maybe_modify_converted_hf_weight` by parameter names as well as arity, so older `(task, converted_weights_dict)` and newer forms both route correctly

### Failure 40

- App: `ap-Y8QVcWFcqmIqplPg3ege5z`
- Observation:
  - the merge-disabled LoRA repro finally got through all currently-known old-bridge compatibility mismatches
  - it reproduced the original hard failure again during export:
    - `torch.AcceleratorError: CUDA error: invalid argument`
    - source: `tensor.cpu()` inside the traced export loop
- Interpretation:
  - this is the most important result so far
  - disabling adapter merging does **not** eliminate the crash
  - so the original failure is not just the LoRA adapter-merge path; base-weight export itself can still produce a bad tensor on this cheap repro
- Fix applied:
  - wrap the traced `tensor.cpu()` failure with task/hf-name metadata in the raised exception so the next rerun tells us exactly which exported tensor is bad

### Failure 41

- App: `ap-ac8ymF2ssFEZZXKoYWHWeF`
- Observation:
  - the cheap 20B repro completed the full regular-trainer-style path up through:
    - LoRA wrapping
    - HF -> Megatron init
    - actor backup
    - `UpdateWeightFromTensor` construction
    - `connect_and_update()`
  - then failed on the same first export tensor as before:
    - `idx=0`
    - Megatron global name: `decoder.final_layernorm.weight`
    - HF name: `model.norm.weight`
    - descriptor: `device=cuda:0 dtype=torch.bfloat16 shape=(2880,) stride=(1,) contiguous=True`
  - the first fallback attempt (`tensor.detach().clone().cpu()`) did not resolve it
- Interpretation:
  - this confirms the failure is not just pre-init or bridge-construction noise
  - it survives the full regular Megatron path and still dies on the first small base tensor export
  - simple clone-before-host-copy is not sufficient
- Fix applied:
  - add more detailed retry logging to distinguish:
    - clone/materialization failure
    - vs. host transfer failure after materialization
  - add a second retry path via `float32` host transfer to test whether the failure is specific to direct bf16 host copy

### Failure 42

- App: `ap-akGwmB7kzqYnr6rzKohIah`
- Observation:
  - the cheap 20B repro got all the way through:
    - LoRA wrapping
    - HF -> Megatron init
    - `UpdateWeightFromTensor` construction
    - `connect_and_update()`
    - `weights_getter()`
  - the new raw probe on `vp_stages.0.decoder.final_layernorm.weight` showed:
    - `detach()` succeeds
    - `detach().clone()` fails with `CUDA error: invalid argument`
    - `detach().cpu()` fails with `CUDA error: invalid argument`
    - `detach().to(dtype=torch.float32).cpu()` fails with `CUDA error: an illegal memory access was encountered`
  - `get_hf_weight_chunks()` then dies immediately, before the first task preview
- Interpretation:
  - the corruption is present on the raw live actor `Parameter` before Bridge conversion starts
  - Bridge export is not the first corrupter
  - feeding live actor tensors into bridge mode is the likely design bug; we should export from the TensorBackuper snapshot instead
- Fix applied:
  - switch bridge mode back to the TensorBackuper actor snapshot by default, while keeping an env-gated escape hatch for the old live-tensor behavior

### Failure 43

- App: `ap-2TtC6WyUWY5eY3AJMnFtZz`
- Observation:
  - the cheap 20B repro completed the full exact-trainer init path with LoRA enabled:
    - HF -> Megatron init
    - actor backup
    - `UpdateWeightFromTensor` construction
    - `actor.init_from_cli()` completion
  - the getter log confirmed bridge mode was now reading from the actor snapshot:
    - `weight_updater getter uses TensorBackuper actor snapshot for bridge mode`
  - the raw probe on `vp_stages.0.decoder.final_layernorm.weight` was now healthy CPU data:
    - `type=Tensor device=cpu dtype=torch.bfloat16 shape=(2880,) stride=(1,) contiguous=True`
    - `detach_clone`, `detach_cpu`, and `detach_float_cpu` all succeeded
  - `connect_and_update()` then reached the first bridge task preview and failed on:
    - `idx=0`
    - `key=vp_stages.0.decoder.final_layernorm.weight`
    - `src_device=cpu dtype=torch.bfloat16 shape=(2880,) stride=(1,) contiguous=True pinned=True`
    - failure site: `_handle_one()` on `gpu_weight.copy_(new_param_weight, non_blocking=False)`
    - exception: `CUDA error: invalid argument`
- Interpretation:
  - the old toxic live-parameter bug is fixed for this path
  - the remaining failure is now specifically pinned CPU snapshot -> manual CUDA staging in `_handle_one()`
  - Bridge traversal is no longer the blocker; the copy kernel is
- Fix applied:
  - replace the manual `torch.empty(..., device='cuda'); copy_(pinned_cpu_tensor)` path with:
    - an ordinary CPU tensor copy first
    - then `.to(cuda)` for the staging transfer

### Failure 44

- App: `ap-ZRFSSeRzqghNMuf0TkkSeR`
- Observation:
  - the rerun again completed the exact-trainer init path and reached `connect_and_update()`
  - the same first task was still the boundary:
    - `key=vp_stages.0.decoder.final_layernorm.weight`
  - changing `copy_(pinned_cpu_tensor)` into `host_weight.to(cuda)` did not help
  - it still failed on the first CPU snapshot -> CUDA materialization with:
    - `torch.AcceleratorError: CUDA error: invalid argument`
- Interpretation:
  - the remaining failure is not about the exact CUDA copy primitive
  - forcing the snapshot tensor back onto CUDA is likely the wrong move when the bridge export path is already running with `cpu=True`
- Fix applied:
  - if `SLIME_BRIDGE_EXPORT_CPU=1` and the source weight is already a CPU snapshot tensor, keep it on CPU end to end
  - still normalize pinned CPU tensors into ordinary CPU tensors first to avoid propagating pinned-memory semantics into bridge conversion

### Failure 45

- Observation:
  - the local GPT-OSS MXFP4 helper in both:
    - `slime_plugins/mbridge/gpt_oss.py`
    - `tools/preprocess_gpt_oss.py`
    was handwritten and not obviously sourced from the current upstream GPT-OSS implementation.
  - Hugging Face transformers has an upstream GPT-OSS MXFP4 path in `transformers.integrations.mxfp4.convert_moe_packed_tensors`.
  - The most important semantic difference from our handwritten helper is that the upstream path returns:
    - `out.transpose(1, 2).contiguous()`
    after unpacking, while our old helper returned the flattened tensor without that transpose.
- Interpretation:
  - the handwritten local helper was not safe to trust as ground truth.
  - if the output layout is wrong, MoE expert weights can be structurally corrupted even if the scalar nibble decode is otherwise correct.
- Fix applied:
  - vendor the HF reference implementation into `slime_plugins/mbridge/mxfp4_reference.py`
  - route both:
    - `slime_plugins/mbridge/gpt_oss.py`
    - `tools/preprocess_gpt_oss.py`
    through that shared reference implementation instead of keeping two handwritten copies

### Failure 46

- Observation:
  - the real GPT-OSS load and export paths are split across two different bridge implementations:
    - load path: `slime_plugins/mbridge/gpt_oss.py` via local `mbridge` registration
    - export path: installed `megatron.bridge` GPT-OSS bridge, patched by `slime_plugins/megatron_bridge/__init__.py`, used from `HfWeightIteratorBridge`
  - these two codepaths are not obviously inverse of each other.
  - there are at least three expert-weight representations active in the codebase:
    - original HF checkpoint: `*_blocks` / `*_scales`
    - bridge export/update path: dense composite `gate_up_proj` / `down_proj`
    - direct Megatron->HF converter: per-expert `gate_proj` / `up_proj` / `down_proj`
- Interpretation:
  - this split-brain bridge architecture is now a stronger root-cause candidate than the local handwritten MXFP4 helper by itself.
  - even after swapping to the HF reference MXFP4 dequantizer, the real RL-path repro still produced gibberish, which makes a broader load/export mismatch more likely.
  - the current tensor-compare hook is also blind to the most suspicious tensors, because exported composite names like `model.layers.*.mlp.experts.gate_up_proj` do not exist in the original HF checkpoint `weight_map`, which only contains `*_blocks` / `*_scales`.

### Failure 47

- Observation:
  - the original tensor-compare hook in `hf_weight_iterator_bridge.py` only did exact-name lookup into the HF checkpoint `weight_map`.
  - that could never validate exported GPT-OSS composite expert tensors like:
    - `model.layers.*.mlp.experts.gate_up_proj`
    - `model.layers.*.mlp.experts.down_proj`
    because the original checkpoint only stores:
    - `*_blocks`
    - `*_scales`
- Interpretation:
  - the compare path was blind exactly where the most suspicious tensors live.
  - earlier compare runs failing to stop before rollout do not imply the exported expert tensors matched; they may simply have been skipped.
- Fix applied:
  - teach `_load_reference_tensor(...)` to materialize dense composite expert references from the original HF checkpoint:
    - `gate_up_proj` -> dequantize `gate_up_proj_blocks/scales`
    - `down_proj` -> dequantize `down_proj_blocks/scales` and transpose to the dense exported layout

### Failure 48

- Observation:
  - the new dense-composite compare path did start firing on the real RL export path.
  - however, `bridge_compare_fail_fast=1` did not stop the run before rollout.
  - cause: `_maybe_compare_against_hf_reference(...)` raised `RuntimeError` for fail-fast inside a broad `except Exception:` block, then swallowed it and continued.
- Interpretation:
  - earlier fail-fast compare runs were not actually authoritative about where comparison stopped.
- Fix applied:
  - let `RuntimeError` propagate out of `_maybe_compare_against_hf_reference(...)` so fail-fast now really aborts before rollout.


## Failure 49: real expert compare isolates down_proj mismatch
- Run `ap-RlY3PGHYYgvVIXKn6MY4DO` reached real `update_weights()` and fail-fast compare before rollout.
- Early/base tensors still compare exactly.
- `model.layers.1.mlp.experts.gate_up_proj` compares exact with `max_abs=0`.
- `model.layers.1.mlp.experts.down_proj` compares non-exact with `max_abs=32.0312`.
- This sharply narrows the bad post-sync inference to the expert `down_proj` path, not generic MoE export or MXFP4 dequantization.
- Most likely remaining causes are a down-proj transpose/layout mismatch between local HF->Megatron load (`slime_plugins/mbridge/gpt_oss.py`) and installed Megatron Bridge Megatron->HF export, or a bug in the new dense reference compare orientation for down-proj.


## Failure 50: down_proj mismatch was a compare-orientation bug, not an export bug
- Run `ap-WXJf5086O4vqdD8gpOdlvs` used corrected compare logic that treats exported `down_proj` as HF-format and also logs an alternate `transposed_ref` comparison.
- Result: `model.layers.0/1.mlp.experts.down_proj` compare exact against the raw checkpoint reference with `max_abs=0`.
- The alternate `transposed_ref` comparison is non-exact (`max_abs=24` on layer 0, `32.0312` on layer 1).
- Therefore the earlier apparent `down_proj` mismatch was caused by comparing against the wrong orientation.
- Combined with earlier exact matches for gate_up/router/qkv/norms, the exported tensors on the active RL path are exact for all sampled early tensors, including expert weights.
- This shifts the likely root cause away from HF->Megatron export and toward either:
  - the SGLang `update_weights_from_tensor` application path, or
  - an untested later tensor subset that is still mismatched.


## Failure 51: direct same-weights SGLang refit control added
- Added `examples/scaffolding/run_gpt_oss_scaffolding_sglang_refit_repro.py`.
- This direct control starts SGLang 20B on 1xH200, runs baseline Fibonacci inference, calls `update_weights_from_disk` with the exact same checkpoint, then runs the same inference again.
- Goal: separate generic SGLang hot-refit bugs from Megatron/export bugs. If baseline is good and post-refit is bad here, the remaining culprit is SGLang hot refit itself, not Megatron.
- Modal mode added: `smoke-sglang-refit-repro`, live app `ap-Z3Q00RyTQ2FmvoD2nfKFsY`.


## Failure 52: SGLang tensor hot-update skips quant postprocessing
- Deep audit of upstream SGLang found a concrete asymmetry:
  - `ModelRunner.update_weights_from_disk(...)` calls `loader.load_weights_and_postprocess(...)`
  - `ModelRunner.update_weights_from_tensor(...)` only calls `self.model.load_weights(...)`
- For GPT-OSS this is a strong bug candidate because the quantized MoE path depends on `quant_method.process_weights_after_loading(module)` after load.
- This matches the observed symptom cluster:
  - exported tensors compare exact before upload
  - RL path uses `update_weights_from_tensor`
  - post-sync inference immediately degenerates into punctuation-heavy garbage
- Local compatibility fix added in `slime/backends/sglang_utils/sglang_engine.py`:
  - after successful `update_weights_from_tensor`, rerun quant postprocessing for every module with `quant_method`
  - then best-effort `post_load_weights(self.model, self.model_config)`
- Validation path:
  - direct `20b` / `1xH200`
  - no-tool row `examples/scaffolding/direct_smoke_boxed_17.jsonl`
  - same-checkpoint `refit_mode=tensor`
  - compare baseline vs post-refit behavior
