"""
Convert aimosprite/qwen3.5-35b-eval-run-20260302 rollouts to SLIME SFT format.

Input format (per JSONL line):
  {"problem": "...", "rollouts": [{"correct": true, "generation_text": "...", "turns": [...], ...}]}

Output format (parquet):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Takes only correct rollouts. For multi-turn (tool-call) conversations, reconstructs
the full message history from turns.

Usage:
    python tools/prep_qwen35_rollouts.py \
        --dataset aimosprite/qwen3.5-35b-eval-run-20260302 \
        --output models/qwen35-rollouts.parquet \
        --correct-only
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given a math problem. "
    "Think step by step and provide your answer."
)


def _rollout_to_messages(problem: str, rollout: dict) -> list[dict]:
    """Convert a single rollout into a list of messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": problem})

    turns = rollout.get("turns", [])
    if len(turns) <= 1:
        # Single-turn: use generation_text directly
        gen_text = rollout.get("generation_text", "")
        messages.append({"role": "assistant", "content": gen_text})
    else:
        # Multi-turn: reconstruct from turns
        for turn in turns:
            reasoning = turn.get("reasoning_content", "")
            content = turn.get("content", "")
            # Combine reasoning and content (Qwen3.5 format: <think>...</think>answer)
            full_content = ""
            if reasoning:
                full_content = f"<think>{reasoning}</think>"
            if content:
                full_content += content
            messages.append({"role": "assistant", "content": full_content})

            # Add tool call results if any
            tool_calls = turn.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict) and "function" in tc:
                    # Tool call response would come from the next turn's input
                    pass

    return messages


def convert_rollouts(dataset_name: str, output_path: str, correct_only: bool = True, max_per_problem: int = 0):
    print(f"Dataset: {dataset_name}")
    print(f"Output:  {output_path}")
    print(f"Filter:  correct_only={correct_only}, max_per_problem={max_per_problem}")

    # List JSONL files
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    jsonl_files = sorted([f for f in all_files if f.endswith(".jsonl")])
    print(f"Found {len(jsonl_files)} JSONL files")

    schema = pa.schema([pa.field("messages", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ])))])

    writer = None
    total = 0
    skipped = 0
    batch = []
    batch_size = 500

    def flush():
        nonlocal writer, batch
        if not batch:
            return
        arr = pa.array(batch, type=schema.field("messages").type)
        table = pa.table({"messages": arr})
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        batch = []

    for jf in jsonl_files:
        print(f"  Processing {jf}...")
        path = hf_hub_download(dataset_name, jf, repo_type="dataset")
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                problem = data.get("problem", "")
                rollouts = data.get("rollouts", [])
                count = 0
                for r in rollouts:
                    if correct_only and not r.get("correct", False):
                        skipped += 1
                        continue
                    if 0 < max_per_problem <= count:
                        break
                    msgs = _rollout_to_messages(problem, r)
                    batch.append(msgs)
                    total += 1
                    count += 1
                    if len(batch) >= batch_size:
                        flush()

    flush()
    if writer is not None:
        writer.close()

    print(f"\nDone! {total} samples written, {skipped} skipped → {output_path}")

    # Verify
    table = pq.read_table(output_path, columns=["messages"])
    print(f"Parquet rows: {len(table)}")
    if len(table) > 0:
        sample = table["messages"][0].as_py()
        for msg in sample[:3]:
            content = (msg.get("content") or "")[:120]
            print(f"  [{msg['role']}]: {repr(content)}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aimosprite/qwen3.5-35b-eval-run-20260302")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--correct-only", action="store_true", default=True)
    parser.add_argument("--all-rollouts", action="store_true", help="Include incorrect rollouts too")
    parser.add_argument("--max-per-problem", type=int, default=0, help="Max rollouts per problem (0=all)")
    args = parser.parse_args()

    if args.all_rollouts:
        args.correct_only = False

    convert_rollouts(args.dataset, args.output, args.correct_only, args.max_per_problem)


if __name__ == "__main__":
    main()
