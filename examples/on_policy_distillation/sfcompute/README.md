# sfcompute OPD

Run on-policy distillation (OPD) on SF Compute H100 nodes via Docker.

Current defaults in `train-config.yaml` are for a single 8x H100 node:

- 1 node total (`cluster_num_nodes: 1`)
- 8 GPUs on the node (`gpus_per_node: 8`)
- GPUs `0..5` = student train + rollout
- GPUs `6..7` = teacher scoring server

## Two-node quickstart (frictionless)

### 1) SSH into both nodes, clone repo, set up Tailscale

On **both nodes**:

```bash
cd ~
git clone -b rohin/opd-sfcompute https://github.com/aimosprite/slime.git
cd ~/slime

curl -fsSL https://tailscale.com/install.sh | sh  
sudo tailscale up
```

Optional sanity check on each node (recommended):

```bash
tailscale status
tailscale ip -4
```

### 2) Start teacher/head role on Node A

On **Node A**:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh teacher
```

This does everything for head mode: installs tooling, auto-detects the node IP (prefers Tailscale IPv4 when available), writes it into `train-config.yaml` (`ray_head_ip` + `teacher_ip`), creates `.env` if needed, and launches OPD training.

It also prints the exact student command to run.
For 2-node configs, it pauses and asks you to press Enter after the student worker is up.

### 3) Start worker role on Node B (using teacher IP)

On **Node B**, run the command printed by teacher setup:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh student <NODE_A_IP>
```

This does everything for worker mode: installs Docker/tooling if needed, pulls image, and starts a blocking Ray worker in Docker connected to the teacher/head IP.

If you set a non-default Ray port on Node A, pass it in on Node B too:

```bash
RAY_PORT=6379 bash examples/on_policy_distillation/sfcompute/setup.sh student <NODE_A_IP>
```

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
   - `student <teacher_ip>` -> `docker-run.sh worker` (joins Ray cluster and blocks)

## Tailscale + ports defaults

Defaults now work out-of-the-box for Tailscale-connected nodes:

- `setup.sh teacher` picks Tailscale IPv4 first (`tailscale ip -4`) for `ray_head_ip` and `teacher_ip`.
- `setup.sh student <teacher_ip>` joins Ray at `RAY_PORT` (default `6379`).
- `run-opd.sh` starts Ray head on:
  - `RAY_PORT` (default `6379`)
  - dashboard `RAY_DASHBOARD_PORT` (default `8265`)
- Teacher sglang server listens on `TEACHER_PORT` (default `13141`).

If you changed firewall/Tailscale ACL policy, ensure worker node can reach Node A on:

- `6379/tcp` (or your `RAY_PORT`)
- `8265/tcp` (or your `RAY_DASHBOARD_PORT`)
- `13141/tcp` (or your `TEACHER_PORT`)

## Commands summary

- Node A (head/teacher):

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh teacher
```

- Node B (student worker):

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh student <NODE_A_IP>
```

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-shot host setup + training launcher |
| `docker-run.sh` | Docker wrapper for prep/train |
| `prep-opd.sh` | Idempotent prep (dataset/model download + checkpoint conversion) |
| `run-opd.sh` | Launches teacher + Ray OPD training (auto-runs prep if needed) |
| `config-16xh100.env` | Configurable paths and GPU layout (2 nodes x 8 H100) |
| `train-config.yaml` | Training hyperparameters (save/eval cadence, optimizer, checkpoint shipping) |

## To clean up dead runs
```bash
docker kill $(docker ps -q) 2>/dev/null; pkill -9 -f torchrun; pkill -9 -f sglang; pkill -9 ray
```

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


## Single-node runbook

The checked-in config already targets the single-node 8x H100 path.

Then run:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh single
```

`single` uses the active layout in `train-config.yaml` and `config-16xh100.env`.
