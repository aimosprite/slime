"""
Convert the latest Megatron checkpoint to HF format and upload to HuggingFace.

Usage:
    modal run gimran/emb-surgery-sft/oss20b/modal/convert_and_upload.py
"""

import modal
import os
import subprocess
import shutil
from pathlib import Path

app = modal.App("gimran-convert-upload")
vol = modal.Volume.from_name("slime-models", create_if_missing=True)
secrets = modal.Secret.from_name("slime-secrets")

POOL_DIR = "/root/slime/models"
REPO_DIR = "/root/slime"
MEGATRON_PATH = "/root/Megatron-LM"

# Source paths (on volume)
SWAPPED_DIR = f"{POOL_DIR}/gpt-oss-20b-qwen3.5-tokenizer"
SLIME_CKPT_DIR = f"{POOL_DIR}/gpt-oss-20b-qwen3.5-tokenizer_slime"

# Output
HF_OUTPUT_DIR = f"{POOL_DIR}/gpt-oss-20b-surgery-hf"
HF_UPLOAD_REPO = "aimosprite/gpt-oss-20b-surgery-hf"

_local_file = Path(__file__).resolve()
_is_local = len(_local_file.parents) > 4

if _is_local:
    REPO_ROOT = _local_file.parents[4]
    image = (
        modal.Image.from_registry("slimerl/slime:latest")
        .run_commands("pip install 'typing_extensions>=4.12' 'pydantic>=2.0'")
        .add_local_dir(str(REPO_ROOT / "slime"), remote_path="/root/slime/slime")
        .add_local_dir(str(REPO_ROOT / "slime_plugins"), remote_path="/root/slime/slime_plugins")
        .add_local_dir(str(REPO_ROOT / "tools"), remote_path="/root/slime/tools")
        .add_local_file(str(REPO_ROOT / "setup.py"), "/root/slime/setup.py")
        .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/slime/pyproject.toml")
    )
else:
    image = modal.Image.from_registry("slimerl/slime:latest")


@app.function(
    image=image,
    volumes={POOL_DIR: vol},
    secrets=[secrets],
    gpu="H100",
    timeout=7200,
    memory=262144,  # 256GB RAM for conversion
)
def convert_and_upload():
    """Convert latest Megatron checkpoint to HF using Bridge and upload."""
    # Find latest checkpoint
    tracker = f"{SLIME_CKPT_DIR}/latest_checkpointed_iteration.txt"
    if not Path(tracker).exists():
        raise RuntimeError(f"No checkpoint tracker at {tracker}")

    latest_iter = int(Path(tracker).read_text().strip())
    iter_dir = f"{SLIME_CKPT_DIR}/iter_{latest_iter:07d}"
    print(f"=== Converting checkpoint iter {latest_iter} ===")
    print(f"  Input: {iter_dir}")
    print(f"  Origin HF: {SWAPPED_DIR}")
    print(f"  Output: {HF_OUTPUT_DIR}")

    if not Path(iter_dir).exists():
        raise RuntimeError(f"Checkpoint dir not found: {iter_dir}")
    if not Path(SWAPPED_DIR).exists():
        raise RuntimeError(f"Swapped HF dir not found: {SWAPPED_DIR}")

    # List checkpoint contents
    ckpt_files = list(Path(iter_dir).iterdir())
    print(f"  Checkpoint files: {len(ckpt_files)}")
    for f in sorted(ckpt_files)[:10]:
        print(f"    {f.name} ({f.stat().st_size / 1e6:.0f} MB)")

    # Clean output
    if Path(HF_OUTPUT_DIR).exists():
        shutil.rmtree(HF_OUTPUT_DIR)

    # Convert using raw converter (supports GPT-OSS via gpt_oss.py converter)
    env = {
        **os.environ,
        "PYTHONPATH": f"{MEGATRON_PATH}:{REPO_DIR}",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29600",
    }

    cmd = [
        "python3", f"{REPO_DIR}/tools/convert_torch_dist_to_hf.py",
        "--input-dir", iter_dir,
        "--output-dir", HF_OUTPUT_DIR,
        "--origin-hf-dir", SWAPPED_DIR,
        "--vocab-size", "248320",
        "-f",
    ]

    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Converter failed with exit code {result.returncode}")

    # Verify output
    safetensors = list(Path(HF_OUTPUT_DIR).glob("*.safetensors"))
    config = Path(HF_OUTPUT_DIR) / "config.json"
    tokenizer = Path(HF_OUTPUT_DIR) / "tokenizer.json"
    print(f"\n=== Conversion complete ===")
    print(f"  Safetensors files: {len(safetensors)}")
    print(f"  config.json: {config.exists()}")
    print(f"  tokenizer.json: {tokenizer.exists()}")
    total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
    print(f"  Total size: {total_gb:.1f} GB")

    if not safetensors:
        raise RuntimeError("No safetensors files produced")

    # Login to HF
    token = os.environ.get("HF_TOKEN", "")
    subprocess.run(
        ["huggingface-cli", "login", "--token", token, "--add-to-git-credential"],
        capture_output=True,
    )

    # Create repo if needed
    subprocess.run(
        ["huggingface-cli", "repo", "create", HF_UPLOAD_REPO.split("/")[1],
         "--organization", HF_UPLOAD_REPO.split("/")[0], "--type", "model"],
        capture_output=True,
    )

    # Upload
    print(f"\n=== Uploading to {HF_UPLOAD_REPO} ===")
    result = subprocess.run(
        ["huggingface-cli", "upload", HF_UPLOAD_REPO, HF_OUTPUT_DIR, ".",
         "--repo-type", "model"],
        env={**os.environ, "HF_TOKEN": token},
    )
    if result.returncode != 0:
        raise RuntimeError("Upload failed")

    print(f"\n=== DONE ===")
    print(f"  HF link: https://huggingface.co/{HF_UPLOAD_REPO}")

    vol.commit()


@app.local_entrypoint()
def main():
    convert_and_upload.remote()
