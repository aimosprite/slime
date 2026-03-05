"""
Prepare both datasets for GPT-OSS embedding surgery with train/test splits.

1. AM-Qwen3-Distilled (1.89M traces) → train + test parquet
2. Qwen3.5-35B rollouts (968 correct) → train + test parquet

Usage:
    python tools/prep_datasets_with_splits.py \
        --output-dir models/ \
        --test-fraction 0.05 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


# ── Schema ──────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([pa.field("messages", pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
])))])


def write_parquet(rows: list, path: str):
    arr = pa.array(rows, type=SCHEMA.field("messages").type)
    table = pa.table({"messages": arr})
    pq.write_table(table, path)
    print(f"  Wrote {len(rows)} rows → {path}")


# ── AM-Qwen3-Distilled ─────────────────────────────────────────────────────

_AM_JSONL_FILES = ["code.jsonl", "if.jsonl", "math.jsonl", "multiturn.jsonl", "other.jsonl", "science.jsonl"]
_ROLE_MAP = {"human": "user", "gpt": "assistant", "assistant": "assistant"}


def _conv_to_messages(row: dict) -> list[dict] | None:
    system = (row.get("system") or "").strip()
    convs = row.get("conversations", [])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in convs:
        role = _ROLE_MAP.get(turn.get("from", ""), None)
        if role is None:
            return None
        messages.append({"role": role, "content": turn.get("value", "")})
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) < 2:
        return None
    return messages


def prep_am_dataset(output_dir: Path, test_frac: float, seed: int):
    print("=== AM-Qwen3-Distilled ===")
    cache_dir = output_dir / ".am_jsonl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download
    for fname in _AM_JSONL_FILES:
        local = cache_dir / fname
        if not local.exists():
            print(f"  Downloading {fname}...")
            hf_hub_download("a-m-team/AM-Qwen3-Distilled", fname, repo_type="dataset", local_dir=str(cache_dir))
        else:
            print(f"  Cached: {fname}")

    # Stream convert with hash-based train/test split (memory efficient)
    import hashlib
    train_path = str(output_dir / "am-qwen3-distilled-train.parquet")
    test_path = str(output_dir / "am-qwen3-distilled-test.parquet")
    train_writer = None
    test_writer = None
    train_batch, test_batch = [], []
    batch_size = 5000
    total, skipped, n_train, n_test = 0, 0, 0, 0

    def flush_batches():
        nonlocal train_writer, test_writer, train_batch, test_batch
        if train_batch:
            arr = pa.array(train_batch, type=SCHEMA.field("messages").type)
            t = pa.table({"messages": arr})
            if train_writer is None:
                train_writer = pq.ParquetWriter(train_path, SCHEMA)
            train_writer.write_table(t)
            train_batch = []
        if test_batch:
            arr = pa.array(test_batch, type=SCHEMA.field("messages").type)
            t = pa.table({"messages": arr})
            if test_writer is None:
                test_writer = pq.ParquetWriter(test_path, SCHEMA)
            test_writer.write_table(t)
            test_batch = []

    for fname in _AM_JSONL_FILES:
        print(f"  Processing {fname}...", flush=True)
        with open(cache_dir / fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                msgs = _conv_to_messages(row)
                if msgs is None:
                    skipped += 1
                    continue
                total += 1
                # Deterministic hash-based split
                h = int(hashlib.md5(f"{seed}:{total}".encode()).hexdigest(), 16)
                if (h % 1000) < int(test_frac * 1000):
                    test_batch.append(msgs)
                    n_test += 1
                else:
                    train_batch.append(msgs)
                    n_train += 1
                if len(train_batch) + len(test_batch) >= batch_size:
                    flush_batches()

    flush_batches()
    if train_writer: train_writer.close()
    if test_writer: test_writer.close()
    print(f"  Total: {total} samples, {skipped} skipped, train={n_train}, test={n_test}")
    return train_path, test_path


# ── Qwen3.5 Rollouts ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given a math problem. "
    "Think step by step and provide your answer."
)


def _rollout_to_messages(problem: str, rollout: dict) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": problem})
    gen_text = rollout.get("generation_text", "")
    messages.append({"role": "assistant", "content": gen_text})
    return messages


def prep_qwen35_rollouts(output_dir: Path, test_frac: float, seed: int):
    print("\n=== Qwen3.5 Rollouts ===")
    dataset_name = "aimosprite/qwen3.5-35b-eval-run-20260302"
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    jsonl_files = sorted([f for f in all_files if f.endswith(".jsonl")])

    all_rows = []
    skipped = 0
    for jf in jsonl_files:
        print(f"  Processing {jf}...")
        path = hf_hub_download(dataset_name, jf, repo_type="dataset")
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                problem = data.get("problem", "")
                for r in data.get("rollouts", []):
                    if not r.get("correct", False):
                        skipped += 1
                        continue
                    msgs = _rollout_to_messages(problem, r)
                    all_rows.append(msgs)
    print(f"  Total: {len(all_rows)} correct rollouts, {skipped} incorrect skipped")

    # Split
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_test = max(1, int(len(all_rows) * test_frac))
    test_rows = all_rows[:n_test]
    train_rows = all_rows[n_test:]

    train_path = str(output_dir / "qwen35-rollouts-train.parquet")
    test_path = str(output_dir / "qwen35-rollouts-test.parquet")
    write_parquet(train_rows, train_path)
    write_parquet(test_rows, test_path)
    return train_path, test_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="models/")
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-am", action="store_true", help="Skip AM-Qwen3-Distilled")
    parser.add_argument("--skip-qwen35", action="store_true", help="Skip Qwen3.5 rollouts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_am:
        am_train, am_test = prep_am_dataset(output_dir, args.test_fraction, args.seed)
    if not args.skip_qwen35:
        q35_train, q35_test = prep_qwen35_rollouts(output_dir, args.test_fraction, args.seed)

    print("\n=== Dataset Summary ===")
    if not args.skip_am:
        t = pq.read_table(am_train)
        t2 = pq.read_table(am_test)
        print(f"AM-Qwen3-Distilled: train={len(t)}, test={len(t2)}")
    if not args.skip_qwen35:
        t = pq.read_table(q35_train)
        t2 = pq.read_table(q35_test)
        print(f"Qwen3.5 Rollouts:   train={len(t)}, test={len(t2)}")


if __name__ == "__main__":
    main()
