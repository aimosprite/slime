"""
Download AM-Qwen3-Distilled and convert to SLIME SFT format (parquet with 'messages' column).

AM-Qwen3-Distilled is stored as JSONL files on HF (not parquet), with format:
  {"conversations": [{"from": "human", "value": "...", "info": {...}},
                     {"from": "assistant", "value": "..."}],
   "system": "..."}

We convert to:
  {"messages": [{"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}

Usage:
    python tools/prep_am_dataset.py \
        --dataset a-m-team/AM-Qwen3-Distilled \
        --output /root/slime/models/am-qwen3-distilled.parquet \
        --max-samples 0
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_tree

# Dataset files (in order; order matters for reproducibility)
_AM_JSONL_FILES = ["code.jsonl", "if.jsonl", "math.jsonl", "multiturn.jsonl", "other.jsonl", "science.jsonl"]

_ROLE_MAP = {"human": "user", "gpt": "assistant", "assistant": "assistant"}


def _conv_to_messages(row: dict) -> list[dict] | None:
    """Convert one AM row to a list of role/content message dicts."""
    system = (row.get("system") or "").strip()
    convs = row.get("conversations", [])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for turn in convs:
        role = _ROLE_MAP.get(turn.get("from", ""), None)
        if role is None:
            return None  # skip malformed rows
        messages.append({"role": role, "content": turn.get("value", "")})

    # Need at least user + assistant
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) < 2:
        return None

    return messages


def convert_am_dataset(dataset_name: str, output_path: str, max_samples: int = 0):
    print(f"Dataset: {dataset_name}")
    print(f"Output:  {output_path}")

    # Download all JSONL files
    cache_dir = Path(output_path).parent / ".am_jsonl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = []
    for fname in _AM_JSONL_FILES:
        local = cache_dir / fname
        if local.exists():
            print(f"  Cached: {fname}")
        else:
            print(f"  Downloading: {fname} ...", flush=True)
            hf_hub_download(
                repo_id=dataset_name,
                filename=fname,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
        jsonl_paths.append(local)

    # Stream, convert, collect into pyarrow table in batches
    batch_size = 10_000
    total = 0
    skipped = 0
    writer = None
    schema = pa.schema([pa.field("messages", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ])))])

    output_path = str(output_path)
    rows_batch = []

    def flush_batch(rows, writer_ref):
        arr = pa.array(rows, type=schema.field("messages").type)
        table = pa.table({"messages": arr})
        if writer_ref[0] is None:
            writer_ref[0] = pq.ParquetWriter(output_path, schema)
        writer_ref[0].write_table(table)
        return []

    writer_ref = [None]

    for path in jsonl_paths:
        print(f"  Processing: {path.name}", flush=True)
        with open(path) as f:
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

                rows_batch.append(msgs)
                total += 1

                if len(rows_batch) >= batch_size:
                    rows_batch = flush_batch(rows_batch, writer_ref)

                if 0 < max_samples <= total:
                    break
            if 0 < max_samples <= total:
                break

    if rows_batch:
        flush_batch(rows_batch, writer_ref)

    if writer_ref[0] is not None:
        writer_ref[0].close()

    print(f"\nDone! {total} samples written, {skipped} skipped → {output_path}")

    # Quick verify
    table = pq.read_table(output_path, columns=["messages"])
    print(f"Parquet rows: {len(table)}")
    sample_msgs = table["messages"][0].as_py()
    print("First sample messages:")
    for msg in sample_msgs[:3]:
        content_preview = (msg.get("content") or "")[:120]
        print(f"  [{msg['role']}]: {repr(content_preview)}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="a-m-team/AM-Qwen3-Distilled")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all samples")
    args = parser.parse_args()

    convert_am_dataset(args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    main()
