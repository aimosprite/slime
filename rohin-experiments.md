# Rohin's Experiment Ops (On-Policy Distillation on SF Compute)

## Architecture

- **Ray-distributed training** with 3 components: training loop (Ray head), Megatron actors (training), SGLang engines (inference/rollout)
- **On-Policy Distillation (OPD):** large teacher (Qwen3.5-122B-A10B) distills into smaller student (Qwen3.5-35B-A3B) via KL divergence on teacher logprobs
- **GRPO-style RL** with verifiable math rewards + OPD KL loss

## Cluster Setup (2-Node H100)

- **Node A (teacher/head):** 8× H100 running teacher SGLang server (TP=8, EP=8) + Ray head
- **Node B (student/worker):** GPU 0-3 for Megatron training (TP=2, EMP=2, ETP=2, DP=2), GPU 4-7 for student rollout SGLang engines
- Single-node alternative exists with `use_colocate=1` and CPU optimizer offload

## One-Shot Launch

- `setup.sh teacher` on Node A — installs Docker/NVIDIA toolkit, pulls `slimerl/slime:latest`, creates `.env` (Wandb + HF tokens), starts teacher server + Ray head
- `setup.sh student <NODE_A_IP>` on Node B — same install, joins Ray cluster as worker
- Everything runs inside Docker with `--gpus all --network host --shm-size=64g`

## Preparation Pipeline (`prep-opd.sh`)

- Downloads teacher + student HF models to `${POOL_DIR}/`
- Downloads dataset (e.g. `BytedTsinghua-SIA/DAPO-Math-17k` or `OpenEvals/IMO-AnswerBench`)
- **Converts HF → Megatron format** via `torchrun tools/convert_hf_to_torch_dist.py` with model-specific args from `scripts/models/qwen3.5-35B-A3B.sh`

## Training Run (`run-opd.sh`)

1. Optionally runs prep (AUTO_PREP=1)
2. Loads config hierarchy: CLI args > `.env` > `train-config.yaml` > `config-16xh100.env` > defaults
3. Starts teacher SGLang server on designated GPUs, waits for health check (up to 300s)
4. Starts Ray cluster (`ray start --head`)
5. Launches background checkpoint shipper (polls for new checkpoints, ships to HF Hub)
6. Submits Ray job: `python3 train.py` with ~100 args across model/rollout/optimizer/GRPO/wandb/sglang groups

## Training Loop (`train.py`)

- Creates Ray placement groups for GPU allocation
- Initializes RolloutManager (SGLang engines) + MegatronTrainRayActor (training)
- Each step: generate rollouts → get teacher logprobs → compute rewards → Megatron forward-backward (policy loss + OPD KL loss) → update weights → sync weights to rollout engines
- Periodic eval + checkpoint save

## Weight Sync (Megatron → SGLang)

- Gathers distributed weights across TP/EP/DP ranks → unified tensor dict → CPU (Ray object store) → broadcast to SGLang engines
- Uses `HfWeightIteratorBridge` for Megatron→HF format conversion
- Separate paths for colocated (same node) vs distributed (multi-node)

## Rollout / Inference

- RolloutManager runs SGLang engines for student generation
- Supports multi-turn tool calling (Python code interpreter via `generate_with_tools.py`)
- Gets teacher logprobs from teacher server for OPD
- Rule-based reward model for math verification (exact match)

## Checkpointing

- Saves Megatron distributed checkpoints every `save_interval` steps to `${POOL_DIR}/${STUDENT_SHORT}_slime/`
- Background shipper auto-uploads to HF Hub every `checkpoint_ship_every` steps — ephemeral node safe
- Resume: auto-detects `latest_checkpointed_iteration.txt`

## Monitoring

- **Wandb:** project `slime-dev`, group `{teacher}-to-{student}-opd` — logs loss, advantage, entropy, pass rate
- **Ray dashboard:** `http://NODE_A_IP:8265`
- **Logs:** teacher SGLang → `/tmp/sglang_*.log`, shipper → `/tmp/slime_checkpoint_shipper.log`, main → `${POOL_DIR}/run-opd.log`

## Config Knobs (`train-config.yaml`)

- `opd_kl_coef: 1.0` — weight of teacher KL loss
- `kl_loss_coef: 0.0` — ref model KL (0 = skip ref model, saves ~32GB)
- `rollout_batch_size: 16`, `n_samples_per_prompt: 4`, `global_batch_size: 64`
- `learning_rate: 1e-6`, `weight_decay: 0.1`, `adam_beta2: 0.98`
- `rollout_max_response_len: 16384`, `rollout_temperature: 1`
- `num_steps: 100`, `save_interval: 10`, `eval_interval: 10`

## Recent Fixes (branch `rohin/opd-sfcompute`)

- CUDA visibility: set `NCCL_CUMEM_ENABLE=0` to match SGLang
- Qwen3.5 MoE converter routing: match `qwen3_5_moe` (with underscore) for stacked expert weights
- Memory: CPU optimizer offload, reduced margin (1GB→256MB), skip ref model when `kl_loss_coef=0`
- Teacher server: expert parallelism (EP=4 as sub-division of TP), memory fraction tuning for MoE

## Key Files

- `examples/on_policy_distillation/sfcompute/setup.sh` — one-shot node setup
- `examples/on_policy_distillation/sfcompute/run-opd.sh` — training runner
- `examples/on_policy_distillation/sfcompute/prep-opd.sh` — model download + conversion
- `examples/on_policy_distillation/sfcompute/train-config.yaml` — hyperparameters
- `examples/on_policy_distillation/sfcompute/config-16xh100.env` — paths & cluster layout
- `train.py` — main training loop
- `slime/ray/rollout.py` — RolloutManager
- `slime/backends/megatron_utils/actor.py` — Megatron training actor
- `scripts/models/qwen3.5-35B-A3B.sh` — student model args
