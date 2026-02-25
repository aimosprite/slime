# sfcompute OPD scripts

This folder contains scripts for running Qwen3-8B OPD over SSH on a single 8x H100 node (no Slurm wrapper).

## Files

- `config-8xh100.env`: configurable paths and GPU layout.
- `prep-qwen3-8B-opd.sh`: runs shared prep flow using config values.
- `run-qwen3-8B-opd.sh`: launches teacher + Ray job with 8x H100 defaults.

## End-to-end commands

Run these commands in order.

1) Clone repos and set up Megatron-LM (slime pins a specific commit + patch):

```bash
cd ~
git clone https://github.com/your-org/slime.git
git clone https://github.com/NVIDIA/Megatron-LM.git

# Use slime-pinned Megatron commit (required for convert_hf_to_torch_dist, etc.)
MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
cd ~/Megatron-LM
git checkout ${MEGATRON_COMMIT}
git checkout -- .  # reset before applying patch
git apply ~/slime/docker/patch/latest/megatron.patch
cd ~
```

2) Bootstrap Python tooling (`uv` + venv) and CUDA 12 (required for Transformer Engine):

```bash
apt update
apt install -y curl ca-certificates python3 python3-venv wget

# Transformer Engine requires CUDA 12+. Add NVIDIA's CUDA repo (ubuntu2204/ubuntu2004/ubuntu2404):
UBUNTU_VER=$(lsb_release -rs | tr -d '.')  # e.g. 22.04 -> 2204
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-4
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv ~/venvs/slime
source ~/venvs/slime/bin/activate
```

3) Install Python dependencies:

```bash
source ~/venvs/slime/bin/activate
cd ~/Megatron-LM
uv pip install -e .
cd ~/slime
uv pip install -e .
uv pip install "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
uv pip install "numpy<2"  # Megatron does not support numpy 2.x
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
uv pip install -U ray sglang huggingface_hub pandas pyarrow wandb
```

4) Log in to Hugging Face:

```bash
source ~/venvs/slime/bin/activate
huggingface-cli login
```

5) Create/edit WandB env file:

```bash
cd ~/slime
cat > .env <<'EOF'
WANDB_API_KEY=your_key_here
EOF
```

6) Configure sfcompute defaults:

```bash
cd ~/slime
$EDITOR examples/on_policy_distillation/sfcompute/config-8xh100.env
```

7) Download dataset parquet to `${POOL_DIR}`:

```bash
source ~/venvs/slime/bin/activate
cd ~/slime
source examples/on_policy_distillation/sfcompute/config-8xh100.env
mkdir -p "${POOL_DIR}/dapo-math-17k"
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
  --repo-type dataset \
  --local-dir "${POOL_DIR}/dapo-math-17k"
```

8) Run prep:

```bash
source ~/venvs/slime/bin/activate
cd ~/slime
bash examples/on_policy_distillation/sfcompute/prep-qwen3-8B-opd.sh
```

9) Run training:

```bash
source ~/venvs/slime/bin/activate
cd ~/slime
bash examples/on_policy_distillation/sfcompute/run-qwen3-8B-opd.sh
```

10) (Optional) run with a custom config file:

```bash
source ~/venvs/slime/bin/activate
cd ~/slime
CONFIG_FILE=/path/to/your-config.env \
  bash examples/on_policy_distillation/sfcompute/run-qwen3-8B-opd.sh
```
