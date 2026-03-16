"""
GPT-OSS-20b embedding surgery stage 1.5 — Modal training script.

Pure Python. No bash, no YAML, no config files.
Resumes from checkpoint iter 299 at aimosprite/gpt-oss-20b-embedding-surgery.

Usage:
    modal run gimran/emb-surgery-sft/oss20b/modal/train.py            # prep + train
    modal run gimran/emb-surgery-sft/oss20b/modal/train.py::prep      # prep only
    modal run gimran/emb-surgery-sft/oss20b/modal/train.py::train     # train only
    modal run --detach gimran/emb-surgery-sft/oss20b/modal/train.py   # detached
"""

import modal
import os
import subprocess
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Config — hardcoded from WandB run 5pmajcpv
# ═══════════════════════════════════════════════════════════════════

# Paths (inside container)
POOL_DIR = "/root/slime/models"
MEGATRON_PATH = "/root/Megatron-LM"
REPO_DIR = "/root/slime"

# Model
HF_MODEL = "openai/gpt-oss-20b"
DONOR_TOKENIZER = "Qwen/Qwen3.5-35B-A3B"
NEW_VOCAB_SIZE = 248320
INIT_STD = 0.02
INIT_SEED = 42

# Derived paths
MODEL_NAME = HF_MODEL.split("/")[-1]
MODEL_DIR = f"{POOL_DIR}/{MODEL_NAME}"
DONOR_TOKENIZER_DIR = f"{POOL_DIR}/Qwen3.5-35B-A3B-tokenizer"
SWAPPED_DIR = f"{POOL_DIR}/{MODEL_NAME}-qwen3.5-tokenizer"
MEGATRON_REF_DIR = f"{POOL_DIR}/{MODEL_NAME}-qwen3.5-tokenizer_torch_dist"
SLIME_CKPT_DIR = f"{POOL_DIR}/{MODEL_NAME}-qwen3.5-tokenizer_slime"
# Dataset: FineWeb-Edu for CLM embedding alignment (not SFT data)
DATASET_PATH = f"{POOL_DIR}/fineweb-clm-train.parquet"
TEST_DATA_PATH = f"{POOL_DIR}/fineweb-clm-test.parquet"
FINEWEB_NUM_SAMPLES = 500_000

# Resume checkpoint
HF_CHECKPOINT_REPO = "aimosprite/gpt-oss-20b-embedding-surgery"
RESUME_ITER = 1649  # phase 2 checkpoint (loss ~4-5 on SFT, embeddings partially trained)

# ── Phase config ──
# Set PHASE=1 for 12.4% trainable warmup, PHASE=2 for 100% trainable from checkpoint
PHASE = 2

if PHASE == 1:
    # Phase 1: emb + output + boundary layers. --no-save-optim avoids MoE checkpoint bug.
    TRAIN_PARAMS = ["embedding", "output_layer", "decoder.layers.0", "decoder.layers.23"]
    SEQ_LENGTH = 8192
    GLOBAL_BATCH_SIZE = 64
    ROLLOUT_BATCH_SIZE = 64
    MICRO_BATCH_SIZE = 1
    NUM_EPOCH = 1
    SAVE_INTERVAL = 10
    ROLLOUT_SEED = 43
    LR = 5e-4
    MIN_LR = 1e-5
    LR_DECAY_STYLE = "cosine"
    LR_WARMUP_FRACTION = 0.01
    WEIGHT_DECAY = 0.01
    CLIP_GRAD = 1.0
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.95
    TP = 2
    PP = 1
    CP = 1
    EP = 1
    ETP = 1
    NUM_GPUS = 8
    CPU_OFFLOAD = False
    USE_DISTRIBUTED_OPTIMIZER = False
    WANDB_GROUP = "gpt-oss-20b-embedding-surgery-stage1p75-phase1"

elif PHASE == 2:
    # CLM embedding alignment on FineWeb: embeddings + output_layer only, frozen transformer
    # Start from random embeddings (iter 299), train on diverse web text
    TRAIN_PARAMS = ["embedding", "output_layer"]  # embeddings only — freeze transformer body
    SEQ_LENGTH = 4096
    GLOBAL_BATCH_SIZE = 128
    ROLLOUT_BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 1
    NUM_EPOCH = 10
    SAVE_INTERVAL = 50
    ROLLOUT_SEED = 43
    LR = 1e-4   # moderate for embeddings — only emb+output_layer are trainable
    MIN_LR = 1e-5
    LR_DECAY_STYLE = "cosine"
    LR_WARMUP_FRACTION = 0.01
    WEIGHT_DECAY = 0.01
    CLIP_GRAD = 1.0
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.95
    # Embeddings-only is small — fits on H100 easily, no need for B200
    TP = 2
    PP = 1
    CP = 1
    EP = 1
    ETP = 1
    NUM_GPUS = 8
    CPU_OFFLOAD = False
    USE_DISTRIBUTED_OPTIMIZER = False
    WANDB_GROUP = "gpt-oss-20b-emb-surgery-clm-fineweb-embonly"

# WandB
WANDB_PROJECT = "slime-dev"
# WANDB_GROUP set per phase above

# Checkpoint shipping
CHECKPOINT_HF_REPO_ID = "aimosprite/gpt-oss-20b-embedding-surgery"

# GPT-OSS-20b Megatron model args (from scripts/models/gpt-oss-20b.sh)
MODEL_ARGS = [
    "--group-query-attention",
    "--num-layers", "24",
    "--hidden-size", "2880",
    "--ffn-hidden-size", "2880",
    "--num-attention-heads", "64",
    "--num-query-groups", "8",
    "--kv-channels", "64",
    "--add-qkv-bias",
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-5",
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--position-embedding-type", "rope",
    "--rotary-base", "150000",
    "--vocab-size", str(NEW_VOCAB_SIZE),
    "--moe-ffn-hidden-size", "2880",
    "--moe-layer-freq", "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
    "--moe-router-topk", "4",
    "--num-experts", "32",
    "--moe-token-dispatcher-type", "alltoall",
    "--moe-router-load-balancing-type", "none",
    "--moe-router-score-function", "softmax",
    "--moe-router-dtype", "fp32",
    "--moe-aux-loss-coeff", "0",
    "--transformer-impl", "transformer_engine",
    "--sequence-parallel",
    "--no-rope-fusion",
    "--no-bias-swiglu-fusion",
]

# ═══════════════════════════════════════════════════════════════════
# Modal setup
# ═══════════════════════════════════════════════════════════════════

app = modal.App("gimran-oss20b-emb-surgery")
vol = modal.Volume.from_name("slime-models", create_if_missing=True)
secrets = modal.Secret.from_name("slime-secrets")

# Paths for bundling local files — only resolved locally, not on remote container
_local_file = Path(__file__).resolve()
_is_local = len(_local_file.parents) > 4  # remote container has /root/train.py (shallow)

if _is_local:
    TOOLS_DIR = _local_file.parent.parent  # oss20b/
    REPO_ROOT = _local_file.parents[4]     # slime repo root
    image = (
        modal.Image.from_registry("slimerl/slime:latest")
        .run_commands(
            "pip install 'typing_extensions>=4.12' 'pydantic>=2.0'",
        )
        .add_local_file(str(TOOLS_DIR / "tokenizer_swap.py"), "/root/tools/tokenizer_swap.py")
        .add_local_file(str(TOOLS_DIR / "prep_datasets_with_splits.py"), "/root/tools/prep_datasets_with_splits.py")
        .add_local_file(str(TOOLS_DIR / "prep_fineweb.py"), "/root/tools/prep_fineweb.py")
        .add_local_dir(str(REPO_ROOT / "slime_plugins"), remote_path="/root/slime/slime_plugins")
        .add_local_dir(str(REPO_ROOT / "slime"), remote_path="/root/slime/slime")
        .add_local_file(str(REPO_ROOT / "setup.py"), "/root/slime/setup.py")
        .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/slime/pyproject.toml")
    )
else:
    image = modal.Image.from_registry("slimerl/slime:latest")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def run(cmd, env=None, check=True):
    """Run a command, print output live."""
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, env=env, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def hf_login():
    """HF_TOKEN env var is sufficient for downloads. Just set git credential for uploads."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        os.environ["HF_TOKEN"] = token
        subprocess.run(
            ["git", "config", "--global", "credential.helper", "store"],
            capture_output=True,
        )
        subprocess.run(
            ["huggingface-cli", "login", "--token", token, "--add-to-git-credential"],
            capture_output=True,  # suppress noise
        )


def dir_exists_and_valid(path, min_size_mb=0):
    """Check if directory exists and optionally has minimum size."""
    p = Path(path)
    if not p.is_dir():
        return False
    if min_size_mb > 0:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return total > min_size_mb * 1024 * 1024
    return True


# ═══════════════════════════════════════════════════════════════════
# Prep
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={POOL_DIR: vol},
    secrets=[secrets],
    gpu="H100",
    timeout=7200,
)
def prep():
    """Download model, swap tokenizer, convert to Megatron, get dataset, download checkpoint."""
    hf_login()

    # 1. Download base model
    if not Path(f"{MODEL_DIR}/config.json").exists():
        print(f"=== Downloading {HF_MODEL} ===")
        run(["huggingface-cli", "download", HF_MODEL, "--local-dir", MODEL_DIR])
    else:
        print(f"=== {MODEL_NAME} already exists, skipping ===")

    # 2. Download donor tokenizer
    if not Path(f"{DONOR_TOKENIZER_DIR}/tokenizer.json").exists():
        print(f"=== Downloading donor tokenizer {DONOR_TOKENIZER} ===")
        run([
            "huggingface-cli", "download", DONOR_TOKENIZER,
            "--include", "tokenizer*", "special_tokens*", "chat_template*",
            "--local-dir", DONOR_TOKENIZER_DIR,
        ])
    else:
        print("=== Donor tokenizer already exists, skipping ===")

    # 3. Tokenizer swap + MXFP4 dequant + embedding resize
    # Validate: original model has 2 shards, swapped model should too (~40GB)
    if not dir_exists_and_valid(SWAPPED_DIR, min_size_mb=30000):
        if Path(SWAPPED_DIR).exists():
            print("=== Removing incomplete swapped model ===")
            shutil.rmtree(SWAPPED_DIR)
        print(f"=== Tokenizer swap + dequant (std={INIT_STD}, seed={INIT_SEED}) ===")
        run([
            "python3", "/root/tools/tokenizer_swap.py",
            "--input-dir", MODEL_DIR,
            "--donor-tokenizer-dir", DONOR_TOKENIZER_DIR,
            "--output-dir", SWAPPED_DIR,
            "--new-vocab-size", str(NEW_VOCAB_SIZE),
            "--init-std", str(INIT_STD),
            "--seed", str(INIT_SEED),
        ])
    else:
        print("=== Swapped model already exists, skipping ===")

    # 4. Convert HF -> Megatron torch_dist
    if not dir_exists_and_valid(MEGATRON_REF_DIR, min_size_mb=1000):
        if Path(MEGATRON_REF_DIR).exists():
            shutil.rmtree(MEGATRON_REF_DIR)
        print("=== Converting HF -> Megatron (single process) ===")
        convert_env = {
            **os.environ,
            "PYTHONPATH": f"{MEGATRON_PATH}:{REPO_DIR}",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29600",
        }
        run([
            "python3", f"{REPO_DIR}/tools/convert_hf_to_torch_dist.py",
            *MODEL_ARGS,
            "--hf-checkpoint", SWAPPED_DIR,
            "--save", MEGATRON_REF_DIR,
            "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1",
            "--expert-model-parallel-size", "1",
            "--expert-tensor-parallel-size", "1",
            "--untie-embeddings-and-output-weights",
            "--no-gradient-accumulation-fusion",
        ], env=convert_env)
    else:
        print("=== Megatron checkpoint already exists, skipping ===")

    # 5. Download & convert dataset (FineWeb for CLM embedding alignment)
    if not Path(DATASET_PATH).exists():
        print(f"=== Downloading FineWeb ({FINEWEB_NUM_SAMPLES} samples) ===")
        run([
            "python3", "/root/tools/prep_fineweb.py",
            "--output-dir", POOL_DIR,
            "--num-samples", str(FINEWEB_NUM_SAMPLES),
            "--test-fraction", "0.02",
            "--seed", "42",
        ])
    else:
        print("=== FineWeb dataset already exists, skipping ===")

    # 6. Download resume checkpoint from HF
    iter_dir = f"{SLIME_CKPT_DIR}/iter_{RESUME_ITER:07d}"
    tracker = f"{SLIME_CKPT_DIR}/latest_checkpointed_iteration.txt"

    if Path(iter_dir).exists() and Path(tracker).exists():
        # Fix tracker if it points to a different iter (e.g. going back to an older checkpoint)
        stored = int(Path(tracker).read_text().strip())
        if stored != RESUME_ITER:
            print(f"=== Resetting tracker from {stored} to {RESUME_ITER} ===")
            with open(tracker, "w") as f:
                f.write(str(RESUME_ITER))
            vol.commit()
        print(f"=== Checkpoint iter {RESUME_ITER} already exists, skipping ===")
    else:
        print(f"=== Downloading checkpoint iter {RESUME_ITER} from {HF_CHECKPOINT_REPO} ===")
        os.makedirs(SLIME_CKPT_DIR, exist_ok=True)

        run([
            "huggingface-cli", "download", HF_CHECKPOINT_REPO,
            "--include", "checkpoint/*",
            "--local-dir", "/tmp/hf_ckpt",
        ])

        if Path(iter_dir).exists():
            shutil.rmtree(iter_dir)
        shutil.move("/tmp/hf_ckpt/checkpoint", iter_dir)
        shutil.rmtree("/tmp/hf_ckpt", ignore_errors=True)

        with open(tracker, "w") as f:
            f.write(str(RESUME_ITER))

        print(f"=== Checkpoint saved to {iter_dir} ===")

    # Verify
    missing = []
    for d in [SWAPPED_DIR, MEGATRON_REF_DIR]:
        if not Path(d).is_dir():
            missing.append(d)
    if not Path(DATASET_PATH).is_file():
        missing.append(DATASET_PATH)
    if not Path(iter_dir).is_dir():
        missing.append(iter_dir)
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}")

    vol.commit()
    print("=== Prep complete ===")


# ═══════════════════════════════════════════════════════════════════
# Preflight — verify everything works before burning GPU hours
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={POOL_DIR: vol},
    secrets=[secrets],
    gpu="H100",  # 1 GPU is enough for checks
    timeout=600,
)
def preflight():
    """Verify all artifacts, WandB, and HF checkpoint shipping work."""
    import time

    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_token = os.environ.get("WANDB_TOKEN", "")
    errors = []

    print("=" * 60)
    print("  PREFLIGHT CHECK")
    print("=" * 60)

    # 1. Check artifacts exist
    print("\n[1/6] Checking artifacts...")
    artifacts = {
        "Swapped model": f"{SWAPPED_DIR}/config.json",
        "Megatron ref checkpoint": MEGATRON_REF_DIR,
        "Training dataset": DATASET_PATH,
        "Test dataset": TEST_DATA_PATH,
        # Resume checkpoint removed — starting fresh (old ckpt incompatible with this Megatron)
    }
    for name, path in artifacts.items():
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}  ({path})")
        if not exists:
            errors.append(f"Missing: {name} at {path}")

    # 2. Check resume iteration matches
    print("\n[2/6] Checking resume checkpoint...")
    tracker_path = f"{SLIME_CKPT_DIR}/latest_checkpointed_iteration.txt"
    if Path(tracker_path).exists():
        stored_iter = int(Path(tracker_path).read_text().strip())
        if stored_iter != RESUME_ITER:
            print(f"  WARNING: tracker says iter {stored_iter}, script expects {RESUME_ITER}")
            errors.append(f"Checkpoint mismatch: tracker={stored_iter}, expected={RESUME_ITER}")
        else:
            print(f"  Resume iter: {stored_iter} OK")

        # Check the actual iter directory has files
        iter_dir = f"{SLIME_CKPT_DIR}/iter_{stored_iter:07d}"
        if Path(iter_dir).exists():
            n_files = len(list(Path(iter_dir).iterdir()))
            total_mb = sum(f.stat().st_size for f in Path(iter_dir).rglob("*") if f.is_file()) / 1e6
            print(f"  Checkpoint dir: {n_files} files, {total_mb:.0f} MB")
            if total_mb < 1000:
                errors.append(f"Checkpoint suspiciously small: {total_mb:.0f} MB")
        else:
            errors.append(f"Iter directory missing: {iter_dir}")

    # 3. Check HF auth
    print("\n[3/6] Checking HF auth...")
    if hf_token:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True, text=True,
            env={**os.environ, "HF_TOKEN": hf_token},
        )
        if result.returncode == 0:
            print(f"  HF auth: OK ({result.stdout.strip().split(chr(10))[0]})")
        else:
            print(f"  HF auth: FAILED")
            errors.append("HF authentication failed")
    else:
        errors.append("HF_TOKEN not set")

    # 4. Test HF upload (use a unique preflight file, not reusing old ones)
    print("\n[4/6] Testing HF checkpoint upload...")
    if hf_token:
        preflight_content = f"preflight-{int(time.time())}"
        preflight_file = f"/tmp/preflight_{int(time.time())}.txt"
        Path(preflight_file).write_text(preflight_content)

        result = subprocess.run(
            [
                "huggingface-cli", "upload", CHECKPOINT_HF_REPO_ID,
                preflight_file, f".preflight_{int(time.time())}.txt",
                "--repo-type", "model",
            ],
            capture_output=True, text=True,
            env={**os.environ, "HF_TOKEN": hf_token},
        )
        Path(preflight_file).unlink(missing_ok=True)
        if result.returncode == 0:
            print(f"  HF upload: OK (wrote to {CHECKPOINT_HF_REPO_ID})")
        else:
            print(f"  HF upload: FAILED — {result.stderr.strip()[:200]}")
            errors.append(f"HF upload failed: {result.stderr.strip()[:100]}")

    # 5. Check WandB
    print("\n[5/6] Checking WandB...")
    if wandb_token and len(wandb_token) >= 40:
        print(f"  WandB key: OK ({len(wandb_token)} chars)")
        print(f"  Project: {WANDB_PROJECT}")
        print(f"  Group: {WANDB_GROUP}")
    elif wandb_token:
        print(f"  WandB key: SUSPICIOUS (only {len(wandb_token)} chars, need 40+)")
        errors.append(f"WandB key too short: {len(wandb_token)} chars")
    else:
        errors.append("WANDB_TOKEN not set")

    # 6. Check GPU
    print("\n[6/6] Checking GPU...")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        gpus = result.stdout.strip().split("\n")
        print(f"  GPUs available: {len(gpus)}")
        for g in gpus[:2]:
            print(f"    {g.strip()}")
        if len(gpus) < 8:
            print(f"  NOTE: Only {len(gpus)} GPU(s) on preflight (train uses 8)")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  PREFLIGHT FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        print("=" * 60)
        raise RuntimeError(f"Preflight failed: {errors}")
    else:
        print("  PREFLIGHT PASSED — all checks OK")
        print("  Ready to train. Run:")
        print("    modal run --detach .../modal/train.py::train")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={POOL_DIR: vol},
    secrets=[secrets],
    gpu="H100:8",
    timeout=86400,
    memory=1048576,  # 1 TB RAM (Modal max)
)
def train():
    """Run embedding surgery training."""
    hf_login()

    os.makedirs(SLIME_CKPT_DIR, exist_ok=True)

    wandb_key = os.environ.get("WANDB_TOKEN", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    # Start background checkpoint shipper — uploads every new checkpoint to HF
    # This ensures checkpoints survive even if the Volume gets wiped
    shipper_script = f"""
import time, subprocess, os
tracker = "{SLIME_CKPT_DIR}/latest_checkpointed_iteration.txt"
last_synced = ""
while True:
    if os.path.exists(tracker):
        step = open(tracker).read().strip()
        if step and step != last_synced:
            iter_dir = f"{SLIME_CKPT_DIR}/iter_{{int(step):07d}}"
            if os.path.isdir(iter_dir):
                print(f"[shipper] Uploading step {{step}}...")
                r = subprocess.run(["huggingface-cli", "upload", "{CHECKPOINT_HF_REPO_ID}",
                    iter_dir, f"checkpoint_stage1p75/iter_{{int(step):07d}}",
                    "--repo-type", "model"], capture_output=True, text=True)
                if r.returncode == 0:
                    subprocess.run(["huggingface-cli", "upload", "{CHECKPOINT_HF_REPO_ID}",
                        tracker, "checkpoint_stage1p75/latest_checkpointed_iteration.txt",
                        "--repo-type", "model"], capture_output=True)
                    last_synced = step
                    print(f"[shipper] Step {{step}} uploaded")
                else:
                    print(f"[shipper] Upload failed: {{r.stderr[:200]}}")
    time.sleep(30)
"""
    shipper_proc = subprocess.Popen(
        ["python3", "-c", shipper_script],
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    print(f"Checkpoint shipper started (PID {shipper_proc.pid})")

    # NVLink detection
    nvlink_result = subprocess.run(
        "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l",
        shell=True, capture_output=True, text=True,
    )
    has_nvlink = "1" if int(nvlink_result.stdout.strip() or "0") > 0 else "0"
    print(f"NVLink: {has_nvlink}")

    # Kill leftovers (NOT python — that would kill the Modal container itself)
    for proc in ["sglang", "ray"]:
        subprocess.run(["pkill", "-9", proc], capture_output=True)

    # Start Ray
    run([
        "ray", "start", "--head",
        "--node-ip-address", "127.0.0.1",
        "--num-gpus", str(NUM_GPUS),
        "--disable-usage-stats",
        "--dashboard-host=0.0.0.0",
        "--dashboard-port=8265",
    ])

    # Build train_async.py args
    train_args = [
        "python3", f"{REPO_DIR}/train_async.py",
        # Ray
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(NUM_GPUS),
        # Model
        *MODEL_ARGS,
        # Checkpoints
        "--hf-checkpoint", SWAPPED_DIR,
        "--ref-load", MEGATRON_REF_DIR,
        "--load", f"{SLIME_CKPT_DIR}/",
        "--save", f"{SLIME_CKPT_DIR}/",
        "--save-interval", str(SAVE_INTERVAL),
        "--no-save-optim",
        "--no-save-rng",
        # SFT
        "--rollout-function-path", "slime.rollout.sft_rollout.generate_rollout",
        "--prompt-data", DATASET_PATH,
        "--input-key", "messages",
        "--loss-mask-type", "qwen3",
        "--tool-key", "tools",
        "--rollout-shuffle",
        "--rollout-seed", str(ROLLOUT_SEED),
        "--rollout-max-context-len", str(SEQ_LENGTH),
        "--num-epoch", str(NUM_EPOCH),
        "--rollout-batch-size", str(ROLLOUT_BATCH_SIZE),
        "--global-batch-size", str(GLOBAL_BATCH_SIZE),
        "--loss-type", "sft_loss",
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        # Freeze (empty = train all params)
        *(["--only-train-params-name-list", *TRAIN_PARAMS] if TRAIN_PARAMS else []),
        # Perf
        "--bf16",
        "--tensor-model-parallel-size", str(TP),
        "--pipeline-model-parallel-size", str(PP),
        "--context-parallel-size", str(CP),
        "--expert-model-parallel-size", str(EP),
        "--expert-tensor-parallel-size", str(ETP),
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--micro-batch-size", str(MICRO_BATCH_SIZE),
        # Optimizer
        "--optimizer", "adam",
        "--lr", str(LR),
        "--lr-decay-style", LR_DECAY_STYLE,
        "--min-lr", str(MIN_LR),
        "--lr-warmup-fraction", str(LR_WARMUP_FRACTION),
        "--weight-decay", str(WEIGHT_DECAY),
        "--clip-grad", str(CLIP_GRAD),
        "--adam-beta1", str(ADAM_BETA1),
        "--adam-beta2", str(ADAM_BETA2),
        *(["--optimizer-cpu-offload", "--use-precision-aware-optimizer"] if CPU_OFFLOAD else []),
        *(["--use-distributed-optimizer"] if USE_DISTRIBUTED_OPTIMIZER else []),
        # NOT using --overlap-cpu-optimizer-d2h-h2d (staging buffers use ~21GB GPU memory)
        *(["--no-load-optim", "--no-load-rng"] if PHASE == 2 else []),
        # WandB
        "--use-wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-group", WANDB_GROUP,
        "--wandb-key", wandb_key,
        # Eval hook
        "--custom-megatron-before-train-step-hook-path", "slime.hooks.eval_hook.eval_before_step",
        # Misc
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        # NOT using --accumulate-allreduce-grads-in-fp32 (wastes 42GB on fp32 grad buffers)
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--no-gradient-accumulation-fusion",
        "--seq-length", str(SEQ_LENGTH),
        "--distributed-timeout-minutes", "30",
        "--train-memory-margin-bytes", str(256 * 1024 * 1024),  # 256MB margin (default 1GB wastes memory)
    ]

    runtime_env = {
        "PYTHONPATH": MEGATRON_PATH,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",  # force NVLink on (Modal H100s have NVLink)
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_TIMEOUT": "1800",  # 30 min timeout (default 10 min was too short for first step)
        "NCCL_DEBUG": "WARN",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TEST_DATA_PATH": TEST_DATA_PATH,
        "EVAL_INTERVAL": "1",
        "EVAL_BATCH_SIZE": "4",
    }

    import json
    runtime_env_json = json.dumps({"env_vars": runtime_env})

    ray_cmd = [
        "ray", "job", "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={runtime_env_json}",
        "--",
        *train_args,
    ]

    print(f"=== Resuming from iter {RESUME_ITER}, 8x H100 ===")
    print(f"=== WandB: {WANDB_PROJECT}/{WANDB_GROUP} ===")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MASTER_ADDR": "127.0.0.1",
        "no_proxy": "127.0.0.1",
        "RAY_memory_usage_threshold": "0.99",
    }

    result = subprocess.run(ray_cmd, env=env, cwd=REPO_DIR)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("=== Training complete ===")


# ═══════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    print("=== Step 1: Prep ===")
    prep.remote()
    print("=== Step 2: Preflight ===")
    preflight.remote()
    print("=== Step 3: Train (8x H100) ===")
    train.remote()
    print("=== Done ===")
