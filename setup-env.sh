#!/bin/bash
# Setup slime environment using uv
# Adapted from build_conda.sh

set -ex

export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
export ROOT_DIR="/home/rohin"
export SLIME_DIR="${ROOT_DIR}/slime"

# ---- Megatron-LM (correct commit + patch) ----

if [ ! -d "${ROOT_DIR}/Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "${ROOT_DIR}/Megatron-LM"
fi

cd "${ROOT_DIR}/Megatron-LM"
git checkout ${MEGATRON_COMMIT}
# Apply patch (reset first to avoid re-apply issues)
git checkout -- .
git apply "${SLIME_DIR}/docker/patch/latest/megatron.patch"
uv pip install -e .

# ---- sglang (correct commit + patch) ----

if [ ! -d "${ROOT_DIR}/sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git "${ROOT_DIR}/sglang"
fi

cd "${ROOT_DIR}/sglang"
git checkout ${SGLANG_COMMIT}
git checkout -- .
git apply "${SLIME_DIR}/docker/patch/latest/sglang.patch" || true
uv pip install -e "python[all]"

# ---- Python dependencies ----

cd "${SLIME_DIR}"

# PyTorch (skip if already installed)
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || \
    uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# flash-attn (requires compilation)
MAX_JOBS=2 uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# mbridge (weight conversion HF <-> Megatron)
uv pip install "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps

# Transformer Engine (required for Megatron backend)
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

# Apex (fused kernels for Megatron)
NVCC_APPEND_FLAGS="--threads 4" \
  uv pip install --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  "apex @ git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

# Other deps
uv pip install "git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b" --force-reinstall
uv pip install "git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl" --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
uv pip install flash-linear-attention==0.4.0
uv pip install einops
uv pip install nvidia-cudnn-cu12==9.16.0.29
uv pip install "numpy<2"
uv pip install cmake ninja

# ---- Install slime itself ----

cd "${SLIME_DIR}"
uv pip install -e .

echo ""
echo "========================================="
echo "Setup complete."
echo "========================================="
