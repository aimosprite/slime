# sfcompute OPD

Run Qwen3-8B on-policy distillation on a single 8x H100 node via Docker.

## Quickstart

1) Clone the repo:

```bash
cd ~
git clone -b rohin/opd-sfcompute https://github.com/aimosprite/slime.git
cd ~/slime
```

2) Start a tmux session (training is long-running — tmux keeps it alive if your SSH disconnects):

```bash
tmux new -s opd
```

3) Run setup and start training:

```bash
bash examples/on_policy_distillation/sfcompute/setup.sh
```

This single script handles everything: installs Docker + NVIDIA container toolkit if missing, pulls the training image, prompts you for tokens (WandB, HF, optional checkpoint repo), and launches training.

4) Detach from tmux when you want to disconnect:

```
Ctrl-b d
```

Re-attach later with `tmux attach -t opd`.

## What `setup.sh` does

1. Installs Docker + NVIDIA container toolkit (skips if already present).
2. Verifies GPU passthrough.
3. Pulls `slimerl/slime:latest`.
4. Prompts for `.env` tokens (`WANDB_API_KEY`, `HF_TOKEN`, optional `CHECKPOINT_HF_REPO_ID`). Skipped if `.env` already exists.
5. Launches `docker-run.sh train`, which auto-runs prep (model/dataset download + checkpoint conversion) then starts the OPD training loop.

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-shot host setup + training launcher |
| `docker-run.sh` | Docker wrapper for prep/train |
| `prep-qwen3-8B-opd.sh` | Idempotent prep (dataset/model download + checkpoint conversion) |
| `run-qwen3-8B-opd.sh` | Launches teacher + Ray OPD training (auto-runs prep if needed) |
| `config-8xh100.env` | Configurable paths and GPU layout |
| `train-config.yaml` | Training hyperparameters (save/eval cadence, optimizer, checkpoint shipping) |

## Hugging Face auth

- `.env` `HF_TOKEN` is the primary auth source — scripts authenticate automatically.
- If `CHECKPOINT_HF_REPO_ID` is empty, checkpoint uploads default to `<hf_username>/qwen3-8b-opd-checkpoints`.

## Checkpoint safety on ephemeral nodes

`run-qwen3-8B-opd.sh` ships checkpoints to Hugging Face Hub every 10 steps by default (configurable in `train-config.yaml`).
