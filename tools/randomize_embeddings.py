"""
Randomize embedding and output projection weights of a HuggingFace model checkpoint.

Memory-efficient: processes safetensor shards individually, never loads full model.

Usage:
    python tools/randomize_embeddings.py \
        --input-dir /root/Qwen3-8B \
        --output-dir /root/Qwen3-8B-random-emb \
        --init-std 0.02 \
        --seed 42
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

EMBED_KEY = "model.embed_tokens.weight"
OUTPUT_KEY = "lm_head.weight"


def randomize_checkpoint(input_dir: Path, output_dir: Path, init_std: float, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all non-safetensor files (config, tokenizer, etc.)
    for f in input_dir.iterdir():
        if f.name.startswith("."):
            continue
        if f.suffix == ".safetensors":
            continue
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)

    torch.manual_seed(seed)

    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        files_with_targets = set()
        if EMBED_KEY in weight_map:
            files_with_targets.add(weight_map[EMBED_KEY])
        if OUTPUT_KEY in weight_map:
            files_with_targets.add(weight_map[OUTPUT_KEY])

        all_shard_files = sorted(set(weight_map.values()))
        for shard_file in all_shard_files:
            shard_path = input_dir / shard_file
            tensors = load_file(shard_path)

            if shard_file in files_with_targets:
                for key in [EMBED_KEY, OUTPUT_KEY]:
                    if key in tensors:
                        shape, dtype = tensors[key].shape, tensors[key].dtype
                        print(f"  Randomizing {key}: shape={list(shape)}, dtype={dtype}")
                        tensors[key] = torch.normal(0, init_std, size=shape).to(dtype)

            save_file(tensors, output_dir / shard_file)
            print(f"  Saved {shard_file}")

        shutil.copy2(index_path, output_dir / "model.safetensors.index.json")
    else:
        # Single safetensors file
        shard_path = input_dir / "model.safetensors"
        tensors = load_file(shard_path)

        for key in [EMBED_KEY, OUTPUT_KEY]:
            if key in tensors:
                shape, dtype = tensors[key].shape, tensors[key].dtype
                print(f"  Randomizing {key}: shape={list(shape)}, dtype={dtype}")
                tensors[key] = torch.normal(0, init_std, size=shape).to(dtype)

        save_file(tensors, output_dir / "model.safetensors")

    print(f"\nDone. Saved to {output_dir}")
    print(f"Init: N(0, {init_std}), seed={seed}")


def main():
    parser = argparse.ArgumentParser(description="Randomize embedding + output weights in HF checkpoint")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    assert input_dir.exists(), f"Input dir not found: {input_dir}"
    assert input_dir != output_dir, "Input and output dirs must be different"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Init:   N(0, {args.init_std}), seed={args.seed}")
    print()

    randomize_checkpoint(input_dir, output_dir, args.init_std, args.seed)


if __name__ == "__main__":
    main()
