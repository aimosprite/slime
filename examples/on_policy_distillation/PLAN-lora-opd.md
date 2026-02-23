# Repo Onboarding + LoRA OPD Implementation Plan

## Part 1: Repo Onboarding

### What is slime?

slime is an **RL post-training framework for LLMs** with two training backends — **Megatron-LM** (tensor/pipeline parallelism) and **FSDP** (HuggingFace-native, simpler setup) — plus **SGLang** for fast rollout/inference. It uses **Ray** for orchestration across GPUs/nodes. Think of it as: Ray orchestrates placement groups of GPUs, SGLang handles rollout generation, and the chosen backend (Megatron or FSDP) handles training forward/backward passes.

### Core Architecture

```
train.py (entry point)
  ├── parse_args()                          # slime/utils/arguments.py
  ├── create_placement_groups()             # Ray GPU allocation
  ├── create_rollout_manager()              # SGLang inference engines
  ├── create_training_models()              # Megatron or FSDP actor (+ optional critic)
  └── training loop:
       for rollout_id in range(num_rollout):
           1. rollout_manager.generate()     # SGLang generates responses
           2. actor_model.async_train()      # Backend trains on rollout data
           3. actor_model.update_weights()   # Push updated weights to SGLang
```

### Key directories

| Path | Purpose |
|------|---------|
| `train.py` | Entry point - the training loop |
| `slime/ray/` | Ray orchestration: placement groups, actor groups, rollout manager |
| `slime/backends/megatron_utils/` | Megatron backend: model init, loss functions, checkpointing, weight update |
| `slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor` - the main training worker |
| `slime/backends/megatron_utils/model.py` | Model init (`setup_model_and_optimizer`), forward, train loops |
| `slime/backends/megatron_utils/model_provider.py` | `GPTModel` builder + `freeze_model_params()` |
| `slime/backends/megatron_utils/loss.py` | All loss functions: policy_loss, value_loss, advantage computation, OPD KL |
| `slime/backends/megatron_utils/update_weight/` | Weight sync to SGLang: `common.py` (all-gather + naming), `update_weight_from_distributed.py`, `update_weight_from_tensor.py` |
| `slime/utils/arguments.py` | All CLI arguments (including `--train-backend fsdp\|megatron`) |
| `slime/utils/tensor_backper.py` | `TensorBackuper` — stores/restores multiple weight copies (actor, ref, teacher) |
| `slime/utils/types.py` | `Sample`, `RolloutBatch` types |
| `scripts/models/` | Model architecture configs (Megatron args as shell arrays) |
| `examples/` | Example run scripts |

### How model weights are managed (Megatron backend)

Uses a `TensorBackuper` (`slime/utils/tensor_backper.py`) to manage multiple sets of weights for the **same Megatron model architecture**:

- **"actor"**: The policy being trained (gradients flow here)
- **"ref"**: Reference model for KL penalty (frozen copy)
- **"teacher"**: Teacher model for OPD (frozen copy, megatron OPD mode only)

`_switch_model(tag)` calls `weights_backuper.restore(tag)` to swap the model's parameters in-place from CPU backup. This means the ref and teacher models **must have the same architecture** as the actor.

Key code path (`actor.py:102-112`):
```python
self.weights_backuper = TensorBackuper.create(
    source_getter=lambda: named_params_and_buffers(self.args, self.model, ...),
    single_tag=None if args.enable_weights_backuper else "actor",
)
self.weights_backuper.backup("actor")
```

### How weight sync to SGLang works

After training, `update_weights()` pushes the actor's weights to the SGLang rollout engines. The flow (in `update_weight/common.py`):

1. `weights_backuper.get("actor")` → dict of `{name: tensor}` for all params
2. `all_gather_param()` → gather TP-sharded params to full tensors (handles `linear_fc1` GLU rechunking and `linear_fc2` dim fix)
3. `convert_to_hf()` → rename Megatron param names to HF format
4. Broadcast to SGLang engines via NCCL (distributed mode) or IPC (colocated mode)

### Megatron MLP layer naming (Qwen3)

With `--swiglu`, Megatron fuses gate_proj + up_proj into a single layer:

| Megatron name | Type | HF equivalent | Shape (per TP rank) |
|---------------|------|---------------|---------------------|
| `mlp.linear_fc1` | `ColumnParallelLinear` | `gate_proj` + `up_proj` (fused) | `[2*ffn_hidden/TP, hidden]` |
| `mlp.linear_fc2` | `RowParallelLinear` | `down_proj` | `[hidden, ffn_hidden/TP]` |

For Qwen3-8B: hidden=4096, ffn_hidden=12288. With TP=1: `linear_fc1` is `[24576, 4096]`, `linear_fc2` is `[4096, 12288]`.

### How OPD works

On-Policy Distillation adds a KL penalty to the advantages:

```
advantage[i] = base_advantage[i] - opd_kl_coef * (student_logp[i] - teacher_logp[i])
```

This is **orthogonal to the advantage estimator** (GRPO, PPO, etc.) - it's just an additive term.

**Two modes:**

1. **`--opd-type sglang`**: Teacher runs on a separate SGLang server. Teacher log-probs are obtained during rollout via HTTP calls. The teacher can have a **different architecture** and **different size**. This is what you want for Qwen3-32B → Qwen3-8B (different architectures).

2. **`--opd-type megatron`**: Teacher weights are loaded into the same Megatron model via `load_other_checkpoint("teacher", ...)`. **Requires same architecture.**

### Your experiment: Qwen3-32B-Base (teacher) → Qwen3-8B-Base (student)

**Critical constraint**: Since Qwen3-32B and Qwen3-8B have different architectures, you **must** use `--opd-type sglang`.

**Backend choice: Megatron** — preserves TP/SP/CP optimizations, and the experiment will serve as a foundation for LoRA support in the Megatron backend (which benefits larger models that need TP).

**Baseline run script**: `run-qwen3-8B-opd.sh`

---

## Part 2: LoRA OPD Implementation Plan (Megatron Backend)

### Goal

Run OPD with Qwen3-32B-Base teacher → Qwen3-8B-Base student, using the **Megatron backend** with `--opd-type sglang`. **Freeze the student base model** and only train LoRA adapters. LoRA rank = 128.

### Key insight: megatron-bridge has native LoRA

**LoRA is built into megatron-bridge** ([docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html)). It handles:

- TP-aware injection into `linear_fc1`, `linear_fc2`, `linear_qkv`, `linear_proj`
- Automatic freezing of base weights (only adapter params get gradients)
- Adapter-only checkpointing
- Integration after Megatron Core init but before distributed wrapping

This means we do **not** need to build custom LoRA wrapper modules. The implementation reduces to: **pass the LoRA config to the bridge provider** and handle the slime-specific RL training concerns (ref model, weight sync to SGLang).

### How megatron-bridge LoRA works

```python
from megatron.bridge.peft.lora import LoRA

# Configuration via bridge provider
provider.peft = LoRA(
    target_modules=["linear_fc1", "linear_fc2"],  # which layers get adapters
    dim=128,                                        # LoRA rank
    alpha=256,                                      # scaling factor
    dropout=0.0,
)
```

Supported target layers:
- `linear_qkv` — combined QKV attention projection
- `linear_proj` — attention output projection
- `linear_fc1` — first MLP layer (fused gate+up with SwiGLU)
- `linear_fc2` — second MLP layer (down_proj)

Wildcard patterns also work: `"*.layers.0.*.linear_qkv"` targets only layer 0.

### Current state in the repo

Slime already has **bridge mode** (`--megatron-to-hf-mode bridge`) which uses `AutoBridge` in three places:

1. **Model provider** (`model_provider.py:83-99`): `AutoBridge.from_hf_pretrained()` → `to_megatron_provider()` — builds the GPTModel
2. **Checkpoint loading** (`checkpoint.py:129-150`): `bridge.load_hf_weights()` — loads HF checkpoints into Megatron
3. **Weight sync** (`hf_weight_iterator_bridge.py`): `bridge.export_hf_weights()` — converts Megatron weights back to HF format for SGLang

The LoRA config slots into **(1)** — the model provider path — and the bridge handles the rest.

### Implementation plan

#### Step 0: Get vanilla Megatron OPD running

- Run `run-qwen3-8B-opd.sh` end-to-end without LoRA
- Verify the Megatron backend works with `--opd-type sglang` + SGLang teacher
- Must use `--megatron-to-hf-mode bridge` (required for LoRA)
- **Status**: In progress

#### Step 1: Pass LoRA config to bridge provider

**Modify: `slime/backends/megatron_utils/model_provider.py`**

In the bridge mode path (`get_model_provider_func`), add LoRA config before `provider.finalize()`:

```python
if args.megatron_to_hf_mode == "bridge":
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    # ... existing parallelism config ...

    # NEW: LoRA configuration
    if getattr(args, 'use_lora', False):
        from megatron.bridge.peft.lora import LoRA
        provider.peft = LoRA(
            target_modules=args.lora_target_modules,
            dim=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    provider.finalize()
    return provider.provide
```

The bridge will:
- Inject LoRA adapters into the specified layers (TP-aware)
- Set `requires_grad=False` on base weights
- The optimizer created by Megatron will only track adapter params

**No new `lora.py` file needed. No manual TP-aware wrappers. No RowParallelLinear injection concerns.**

**Validation**: After the model is built, verify the bridge did its job:
```python
# Quick check: count trainable vs frozen params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
# Expect: trainable ≈ 208M (LoRA), frozen ≈ 7.6B (base)
```

#### Step 2: Reference model — disable LoRA instead of weight swapping

With LoRA, the ref model is just the base model with LoRA disabled. Since base weights are frozen, the ref model never changes.

**Modify: `slime/backends/megatron_utils/actor.py`**

The bridge's LoRA modules should support enable/disable (scaling factor control). In `train_actor()`, where ref log-probs are computed:

```python
# Current code (actor.py:404-414):
if "ref" in self.weights_backuper.backup_tags:
    self._switch_model("ref")
    rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))

# New code for LoRA:
if self.args.use_lora:
    self._disable_lora()  # set adapter scaling=0 on all LoRA modules
    rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))
    self._enable_lora()   # restore scaling
elif "ref" in self.weights_backuper.backup_tags:
    self._switch_model("ref")
    rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))
```

Helper methods on `MegatronTrainRayActor`:
```python
def _disable_lora(self):
    """Disable LoRA adapters for ref model forward pass."""
    for module in self.model[0].module.modules():
        if hasattr(module, 'lora_scaling'):  # bridge LoRA attribute name TBD
            module._saved_scaling = module.lora_scaling
            module.lora_scaling = 0.0

def _enable_lora(self):
    """Re-enable LoRA adapters after ref forward pass."""
    for module in self.model[0].module.modules():
        if hasattr(module, '_saved_scaling'):
            module.lora_scaling = module._saved_scaling
            del module._saved_scaling
```

**Note**: The exact attribute name for the scaling factor depends on the bridge's LoRA implementation. Need to inspect the bridge's LoRA module to find the right attribute (likely `scaling`, `lora_scaling`, or similar). An alternative is to zero out `lora_B` weights temporarily, but scaling is cleaner.

**Benefits**:
- No need to load `--ref-load` checkpoint (saves init time)
- No need to store ref weights on CPU via TensorBackuper (saves ~16GB host RAM)
- No weight swap overhead per training step
- Mathematically equivalent: ref = base model = actor with LoRA disabled

#### Step 3: Weight sync to SGLang — merge before push

SGLang expects standard HF model weights. Before pushing, merge LoRA into base weights so the existing weight sync pipeline works unchanged.

**Approach**: Merge LoRA before `weights_backuper.backup("actor")`, unmerge after. Then `update_weights()` uses the already-merged backup — zero changes to the weight sync pipeline.

```python
# In train_actor(), after the training step (around actor.py:489):

if self.args.use_lora:
    self._merge_lora()                       # W_base += B @ A * scaling
self.weights_backuper.backup("actor")        # backs up merged weights
if self.args.use_lora:
    self._unmerge_lora()                     # restore original base for next step
```

The bridge may provide merge/unmerge helpers. If not, implement manually:
```python
def _merge_lora(self):
    """Merge LoRA weights into base model for weight sync."""
    for module in self.model[0].module.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # W += B @ A * scaling (operates on TP-local weights)
            module.weight.data += (module.lora_B.weight @ module.lora_A.weight) * module.scaling

def _unmerge_lora(self):
    """Unmerge LoRA weights to restore base model."""
    for module in self.model[0].module.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            module.weight.data -= (module.lora_B.weight @ module.lora_A.weight) * module.scaling
```

**Note on numerical stability**: For bf16, repeated merge/unmerge may introduce drift. Mitigations:
- Store a copy of the original base weight (extra memory but exact unmerge)
- Perform the merge/unmerge in fp32
- This only happens once per rollout, so drift is minimal in practice

The existing weight sync pipeline (`hf_weight_iterator_bridge.py` → `bridge.export_hf_weights()`) needs no changes since the merged weights look like standard base model weights.

#### Step 4: TensorBackuper integration

With LoRA + the merge-before-backup approach:
- **"actor" backup** stores merged weights (base + LoRA folded in) — used by weight sync
- **"ref" backup** is not needed (LoRA disable/enable replaces it)
- The backuper's `named_params_and_buffers()` source iterates all model params including LoRA adapter params, but since we merge before backup, the LoRA contribution is already in the base weights

**Skip ref backup in init** when using LoRA:
```python
# In actor.py init:
if with_ref and not self.args.use_lora:
    self.load_other_checkpoint("ref", args.ref_load)
```

#### Step 5: Checkpoint save/load

The bridge handles adapter-only checkpointing natively: "Only adapter parameters are saved and loaded during checkpointing; base model weights remain frozen and unchanged." ([docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html))

Need to verify how this interacts with slime's existing checkpoint flow (`checkpoint.py:save_checkpoint` / `load_checkpoint`). The bridge checkpoint machinery should:
- **Save**: Only write adapter weights (~400MB vs ~16GB for full model)
- **Load**: Load adapter weights on top of the base model

If the bridge doesn't handle this automatically through the standard Megatron save/load path, add a manual adapter save:
```python
# In save_model():
if self.args.use_lora:
    lora_state = {name: p.detach().cpu() for name, p in model.named_parameters() if 'lora' in name}
    torch.save(lora_state, f"{args.save}/lora_adapter_rollout{rollout_id}.pt")
```

#### Step 6: New CLI arguments

**Modify: `slime/utils/arguments.py`**

```python
# LoRA arguments
parser.add_argument('--use-lora', action='store_true', help='Enable LoRA training (requires --megatron-to-hf-mode bridge)')
parser.add_argument('--lora-rank', type=int, default=128, help='LoRA rank (dim)')
parser.add_argument('--lora-alpha', type=int, default=256, help='LoRA alpha scaling factor')
parser.add_argument('--lora-target-modules', type=str, nargs='+',
                    default=['linear_fc1', 'linear_fc2'],
                    help='Megatron module names to apply LoRA to')
parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
```

Add validation:
```python
if args.use_lora:
    assert args.megatron_to_hf_mode == "bridge", "--use-lora requires --megatron-to-hf-mode bridge"
```

#### Step 7: Run script

**New file: `examples/on_policy_distillation/run-qwen3-8B-lora-opd.sh`**

Based on `run-qwen3-8B-opd.sh` with changes:
```bash
CKPT_ARGS=(
   --hf-checkpoint ${POOL_DIR}/Qwen3-8B
   --load ${POOL_DIR}/Qwen3-8B                  # HF checkpoint (bridge mode loads directly)
   --save ${POOL_DIR}/Qwen3-8B_lora_opd/
   --save-interval 20
   --megatron-to-hf-mode bridge                  # REQUIRED for LoRA
   # NOTE: no --ref-load needed (ref = base model with LoRA disabled)
)

LORA_ARGS=(
    --use-lora
    --lora-rank 128
    --lora-alpha 256
    --lora-target-modules linear_fc1 linear_fc2
    --lora-dropout 0.0
)
```

### Files to modify

| File | Changes |
|------|---------|
| `slime/backends/megatron_utils/model_provider.py` | Add `provider.peft = LoRA(...)` in bridge mode when `--use-lora` |
| `slime/backends/megatron_utils/actor.py` | LoRA-aware ref model (disable/enable), merge/unmerge around backup, skip ref checkpoint loading |
| `slime/utils/arguments.py` | New `--use-lora`, `--lora-rank`, `--lora-alpha`, `--lora-target-modules`, `--lora-dropout` args + validation |
| `examples/.../run-qwen3-8B-lora-opd.sh` | **NEW** — run script for LoRA OPD |

### What we no longer need (vs. previous plan)

| Previous plan | Why it's gone |
|---------------|---------------|
| `slime/backends/megatron_utils/lora.py` (new file) | Bridge handles TP-aware LoRA injection natively |
| Custom `LoRAColumnParallelLinear` wrapper | Bridge handles it |
| Custom `LoRARowParallelLinear` wrapper | Bridge handles it |
| RowParallelLinear forward hook / monkey-patch | Bridge handles it |
| Manual TP-aware A/B matrix sharding | Bridge handles it |
| DDP + frozen params verification | Bridge handles freezing correctly |
| Optimizer compatibility concerns | Bridge ensures only adapter params are trainable |

### Remaining risks and unknowns

1. **Bridge provider LoRA API**: The exact API for setting LoRA on the provider needs verification. The docs show `ConfigContainer(peft=LoRA(...))`, but slime uses `bridge.to_megatron_provider()`. We assume `provider.peft = LoRA(...)` works — need to verify by inspecting the megatron-bridge source or testing.

2. **LoRA module interface for disable/enable**: Need to discover the exact attribute names on the bridge's LoRA modules (scaling factor, weight names) to implement `_disable_lora()` / `_enable_lora()` and `_merge_lora()` / `_unmerge_lora()`. Inspect the bridge's LoRA module source once available.

3. **Weight sync with LoRA params in backuper**: The `named_params_and_buffers()` source will include bridge-injected LoRA params. The merge-before-backup approach avoids needing to handle these in the HF conversion pipeline, but need to verify the bridge's LoRA param names don't interfere with `export_hf_weights()`.

4. **Checkpoint compatibility**: Need to verify the bridge's adapter-only checkpointing works with slime's checkpoint flow, or implement a manual adapter save/load as fallback.

### Implementation order

1. **Get vanilla OPD running** with `run-qwen3-8B-opd.sh` using `--megatron-to-hf-mode bridge`
2. **Add CLI args** (`arguments.py`) — straightforward, no risk
3. **Add LoRA to bridge provider** (`model_provider.py`) — verify model builds with adapters, check param counts
4. **Implement ref model disable/enable** (`actor.py`) — inspect bridge LoRA module attributes
5. **Implement merge-before-backup** (`actor.py`) — verify SGLang gets correct weights
6. **Verify checkpoint save/load** — adapter-only saves
7. **End-to-end test** with `run-qwen3-8B-lora-opd.sh`
