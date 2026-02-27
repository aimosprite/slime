# sfcompute OPD

Run on-policy distillation on a single 8x H100 node via Docker. Teacher and student models are configured in `train-config.yaml`.

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
