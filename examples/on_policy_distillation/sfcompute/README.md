# sfcompute OPD

Run on-policy distillation (OPD) on SF Compute H100 nodes via Docker.

Current defaults in `train-config.yaml` are for a 2-node run:

- 2 nodes total (`cluster_num_nodes: 2`)
- 8 GPUs per node (`gpus_per_node: 8`)
- Node A = teacher + Ray head
- Node B = Ray worker (student training + rollout)

## Two-node quickstart (frictionless)

### 1) SSH into both nodes and clone repo

On **both nodes**:

```bash
cd ~
git clone -b rohin/opd-sfcompute https://github.com/aimosprite/slime.git
cd ~/slime
```

### 2) Configure `train-config.yaml` on Node A

Set at minimum:

```yaml
cluster_num_nodes: 2
gpus_per_node: 8
ray_head_ip: <NODE_A_IP>
teacher_ip: <NODE_A_IP>
```

Use Node A reachable IP (private/LAN preferred).  
Quick way to get it on Node A: `hostname -I | awk '{print $1}'`.

### 3) Start worker role on Node B

On **Node B**:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh student
```

This now does everything for worker mode: installs Docker/tooling if needed, pulls image, and starts a blocking Ray worker in Docker.

### 4) Start teacher/head role on Node A

On **Node A**:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh teacher
```

This does everything for head mode: installs tooling, creates `.env` if needed, and launches OPD training.

Use tmux on both nodes if desired:

```bash
tmux new -s opd
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t opd`.

## What `setup.sh` does

1. Installs Docker + NVIDIA container toolkit (skips if already present).
2. Verifies GPU passthrough.
3. Pulls `slimerl/slime:latest`.
4. `teacher`/`single` mode: prompts for `.env` tokens (`WANDB_API_KEY`, `HF_TOKEN`, optional `CHECKPOINT_HF_REPO_ID`) if missing.
5. Launches role:
   - `teacher`/`single` -> `docker-run.sh train`
   - `student` -> `docker-run.sh worker` (joins Ray cluster and blocks)

## Commands summary

- Node A (head/teacher):

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh teacher
```

- Node B (student worker):

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh student
```

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-shot host setup + training launcher |
| `docker-run.sh` | Docker wrapper for prep/train |
| `prep-opd.sh` | Idempotent prep (dataset/model download + checkpoint conversion) |
| `run-opd.sh` | Launches teacher + Ray OPD training (auto-runs prep if needed) |
| `config-8xh100.env` | Configurable paths and GPU layout |
| `train-config.yaml` | Training hyperparameters (save/eval cadence, optimizer, checkpoint shipping) |

## Hugging Face auth

- `.env` `HF_TOKEN` is the primary auth source — scripts authenticate automatically.
- If `CHECKPOINT_HF_REPO_ID` is empty, checkpoint uploads default to `<hf_username>/<student>-from-<teacher>-opd` (auto-derived from model names in `train-config.yaml`).

## Checkpoint safety on ephemeral nodes

`run-opd.sh` ships checkpoints to Hugging Face Hub every 10 steps by default (configurable in `train-config.yaml`). A preflight check runs automatically at startup to verify auth, repo creation, and uploads before training begins.

To run the preflight manually (without starting training):

```bash
bash examples/on_policy_distillation/sfcompute/docker-run.sh preflight
```

This verifies your HF token, creates the checkpoint repo if needed, and uploads a test file. Takes a few seconds and will surface any auth or permissions issues before you commit to a long run.

Note: the preflight requires Docker and a `.env` with your `HF_TOKEN`, so you cannot run it before `setup.sh`. However, the preflight runs automatically at the start of every training run (before any model downloads or GPU work), so if HF auth is broken the script will exit immediately without wasting time. Use the manual preflight command above for subsequent runs or after changing tokens.

## Get a node on SF Compute

1. Run `bash sfacquire.sh` from the aimo repo, answer all the questions (make sure that pass_along_node_to_setup.sh exists)
2. Do `sf nodes ssh root@...` based on whatever node you have there
    - If that doesn't work, try [this](https://discord.com/channels/1447431405788463157/1461129003896410282/1476321257203957991)

## Single-node runbook (legacy/simple)

If you only have one 8x H100 node, set this in `train-config.yaml`:

```yaml
cluster_num_nodes: 1
gpus_per_node: 8
ray_visible_gpus: auto
```

Then run:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh single
```

`single` defaults auto-split GPUs (student lower half, teacher upper half), enable colocate, and enable CPU optimizer offload.
