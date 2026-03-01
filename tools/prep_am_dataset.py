"""
Download AM-Qwen3-Distilled and convert to SLIME SFT format (parquet with 'messages' column).

Usage:
    python tools/prep_am_dataset.py \
        --dataset a-m-team/AM-Qwen3-Distilled \
        --output /root/am-qwen3-distilled.parquet \
        --max-samples 0
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def inspect_and_convert(dataset_name: str, output_path: str, max_samples: int = 0):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    print(f"Dataset size: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    # Print first sample to understand format
    sample = ds[0]
    print("\n--- First sample keys and types ---")
    for k, v in sample.items():
        if isinstance(v, str):
            print(f"  {k}: str (len={len(v)}, preview={v[:200]!r})")
        elif isinstance(v, list):
            print(f"  {k}: list (len={len(v)}, first={v[0] if v else 'empty'})")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}")

    # Try to auto-detect the format and convert to messages
    if "messages" in ds.column_names:
        print("\nFound 'messages' column — using directly.")
        converted = ds
    elif "conversations" in ds.column_names:
        print("\nFound 'conversations' column — renaming to 'messages'.")
        converted = ds.rename_column("conversations", "messages")
    elif "prompt" in ds.column_names and "response" in ds.column_names:
        print("\nFound 'prompt' + 'response' columns — converting to messages format.")

        def to_messages(row):
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            return {"messages": messages}

        converted = ds.map(to_messages, remove_columns=ds.column_names)
    elif "query" in ds.column_names and "response" in ds.column_names:
        print("\nFound 'query' + 'response' columns — converting to messages format.")

        def to_messages(row):
            messages = [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]},
            ]
            return {"messages": messages}

        converted = ds.map(to_messages, remove_columns=ds.column_names)
    else:
        print("\n!!! Could not auto-detect format. Dumping first sample as JSON for inspection:")
        print(json.dumps({k: str(v)[:500] for k, v in sample.items()}, indent=2))
        print("\nPlease update this script to handle this format.")
        return

    if max_samples > 0:
        converted = converted.select(range(min(max_samples, len(converted))))
        print(f"\nSubsampled to {len(converted)} samples.")

    # Verify
    sample_msg = converted[0]["messages"]
    print(f"\n--- Converted sample ---")
    if isinstance(sample_msg, str):
        sample_msg = json.loads(sample_msg)
    for msg in sample_msg[:3]:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:200]
        print(f"  [{role}]: {content!r}...")

    print(f"\nSaving to {output_path}...")
    converted.to_parquet(output_path)
    print(f"Done! {len(converted)} samples saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="a-m-team/AM-Qwen3-Distilled")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all samples")
    args = parser.parse_args()

    inspect_and_convert(args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    main()
