"""
Download a random sample of FineWeb-Edu and format as parquet for CLM training.

Uses HF Hub API directly (no datasets library) to avoid dependency conflicts.

Formats each document as a single-turn conversation:
  [{"role": "user", "content": ""}, {"role": "assistant", "content": "<text>"}]

This way the SFT loss mask computes loss on ALL tokens (the entire text is
the "assistant response"), effectively giving us CLM loss.

Usage:
    python prep_fineweb.py --output-dir models/ --num-samples 500000
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_tree


SCHEMA = pa.schema([pa.field("messages", pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
])))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="models/")
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=16000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(output_dir / "fineweb-clm-train.parquet")
    test_path = str(output_dir / "fineweb-clm-test.parquet")

    if Path(train_path).exists():
        print(f"=== FineWeb dataset already exists at {train_path}, skipping ===")
        return

    print(f"=== Downloading FineWeb-Edu sample ({args.num_samples} docs) ===")

    # Download individual parquet files from fineweb-edu sample-10BT
    # Each file has ~100K-500K rows. We only need a few files.
    repo_id = "HuggingFaceFW/fineweb-edu"
    config = "sample-10BT"

    # List parquet files in the dataset
    files = list(list_repo_tree(repo_id, repo_type="dataset", path_in_repo=f"{config}/train"))
    parquet_files = sorted([f.path for f in files if f.path.endswith(".parquet")])
    print(f"  Found {len(parquet_files)} parquet files")

    rng = random.Random(args.seed)
    # Shuffle and pick a few files
    rng.shuffle(parquet_files)

    train_writer = None
    test_writer = None
    train_batch, test_batch = [], []
    batch_size = 5000
    n_train, n_test, n_skipped = 0, 0, 0
    target = args.num_samples

    def flush():
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

    for pf in parquet_files:
        if n_train + n_test >= target:
            break

        print(f"  Downloading {pf}...")
        local_path = hf_hub_download(repo_id, pf, repo_type="dataset")

        # Read parquet file
        table = pq.read_table(local_path, columns=["text"])
        texts = table.column("text").to_pylist()
        print(f"    {len(texts)} docs in file")

        for text in texts:
            if n_train + n_test >= target:
                break

            text = (text or "").strip()
            if len(text) < args.min_chars:
                n_skipped += 1
                continue
            if len(text) > args.max_chars:
                text = text[:args.max_chars]

            # Format as conversation: empty user turn + assistant turn with full text
            msgs = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text},
            ]

            if rng.random() < args.test_fraction:
                test_batch.append(msgs)
                n_test += 1
            else:
                train_batch.append(msgs)
                n_train += 1

            if len(train_batch) + len(test_batch) >= batch_size:
                flush()

        if (n_train + n_test) % 50000 == 0 and (n_train + n_test) > 0:
            print(f"  Progress: {n_train + n_test}/{target}")

        # Free memory
        del table, texts

    flush()
    if train_writer:
        train_writer.close()
    if test_writer:
        test_writer.close()

    print(f"  Done: train={n_train}, test={n_test}, skipped={n_skipped}")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")


if __name__ == "__main__":
    main()
