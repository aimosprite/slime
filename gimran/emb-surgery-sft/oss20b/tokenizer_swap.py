"""
Tokenizer swap + MXFP4 dequantization + embedding resize for GPT-OSS → Qwen3.5 tokenizer.

Does the following in one pass:
  1. Dequantizes MXFP4 expert weights (blocks+scales → bf16 dense)
  2. Copies all non-safetensor files from input model
  3. Overwrites tokenizer files with donor tokenizer
  4. Updates config.json: new vocab_size, removes quantization_config
  5. Replaces embed_tokens and lm_head with random init at new vocab size
  6. Drops attention sink tensors (self_attn.sinks) — inference-only
  7. Saves all weights in bf16

Memory-efficient: processes safetensor shards individually, never loads full model.

Usage:
    python tools/tokenizer_swap.py \
        --input-dir  models/gpt-oss-20b \
        --donor-tokenizer-dir models/Qwen3.5-35B-A3B-tokenizer \
        --output-dir models/gpt-oss-20b-qwen3.5-tokenizer \
        --new-vocab-size 248320 \
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

# Tokenizer-related files to copy from donor
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def unpack_mxfp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack MXFP4 e2m1 values from uint8 packed tensor (2 values per byte).

    fp4 e2m1: 1 sign + 2 exp + 1 mantissa = 4 bits per value.
    Each uint8 byte holds 2 fp4 values (high nibble + low nibble).
    """
    hi = (packed >> 4) & 0xF
    lo = packed & 0xF

    def nibble_to_f32(nibbles: torch.Tensor) -> torch.Tensor:
        sign = ((nibbles >> 3) & 1).float()
        exp = ((nibbles >> 1) & 0x3).long()
        mant = (nibbles & 0x1).float()

        normal_mask = exp > 0
        exp_f = exp.float()

        value = torch.where(
            normal_mask,
            (1.0 + mant * 0.5) * (2.0 ** (exp_f - 1.0)),
            mant * 0.5,
        )
        value = value * (1.0 - 2.0 * sign)
        return value

    hi_f = nibble_to_f32(hi)
    lo_f = nibble_to_f32(lo)
    return torch.stack([hi_f, lo_f], dim=-1).flatten(-2)


def dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 expert weights (blocks + e8m0 scales) to bfloat16.

    Args:
        blocks: uint8 [num_experts, out_features, num_blocks, 16] — 32 fp4 per block
        scales: uint8 e8m0 [num_experts, out_features, num_blocks]
    Returns:
        bfloat16 [num_experts, out_features, in_features]
    """
    num_experts, out_features, num_blocks, bytes_per_block = blocks.shape
    in_features = num_blocks * bytes_per_block * 2  # 2 fp4 per byte

    values = unpack_mxfp4_e2m1(blocks)  # [E, out, num_blocks, 32]
    scale_f = (2.0 ** (scales.float() - 127.0)).unsqueeze(-1)  # [E, out, num_blocks, 1]
    return (values * scale_f).reshape(num_experts, out_features, in_features).to(torch.bfloat16)


def dequant_shard(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize all MXFP4 expert weights in a shard. Returns new dict with dense bf16 weights.

    Transforms:
      gate_up_proj_blocks + gate_up_proj_scales → gate_up_proj.weight
      down_proj_blocks + down_proj_scales → down_proj.weight
    Drops: _blocks, _scales suffixes (absorbed), self_attn.sinks (inference-only)
    """
    result = {}
    # Collect blocks/scales pairs
    blocks_keys = [k for k in tensors if k.endswith("_blocks")]
    scales_map = {}
    for bk in blocks_keys:
        sk = bk.replace("_blocks", "_scales")
        if sk in tensors:
            scales_map[bk] = sk

    processed = set()
    for bk, sk in scales_map.items():
        blocks = tensors[bk]
        scales = tensors[sk]
        # Derive output name: gate_up_proj_blocks → gate_up_proj.weight
        out_name = bk.replace("_blocks", ".weight")
        if blocks.dtype == torch.uint8:
            print(f"  Dequantizing {bk} ({list(blocks.shape)}) → {out_name}")
            result[out_name] = dequant_mxfp4(blocks, scales)
        else:
            # Already dense (shouldn't happen in original model, but handle gracefully)
            result[out_name] = blocks.to(torch.bfloat16)
        processed.add(bk)
        processed.add(sk)

    for k, v in tensors.items():
        if k in processed:
            continue
        if k.endswith("_scales"):
            # Orphan scale without blocks in this shard — skip
            continue
        if "self_attn.sinks" in k:
            print(f"  Dropping {k} (attention sinks, inference-only)")
            continue
        # Cast everything else to bf16
        if v.dtype != torch.bfloat16:
            result[k] = v.to(torch.bfloat16)
        else:
            result[k] = v

    return result


def tokenizer_swap(
    input_dir: Path,
    donor_tokenizer_dir: Path,
    output_dir: Path,
    new_vocab_size: int,
    init_std: float,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = None

    # 1. Copy non-safetensor files from input model
    for f in input_dir.iterdir():
        if f.name.startswith("."):
            continue
        if f.suffix == ".safetensors":
            continue
        if f.is_dir():
            continue
        # Skip tokenizer files (will be replaced by donor)
        if f.name in TOKENIZER_FILES:
            continue
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)

    # 2. Copy tokenizer files from donor
    for fname in TOKENIZER_FILES:
        src = donor_tokenizer_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Tokenizer: copied {fname}")
        else:
            print(f"  Tokenizer: {fname} not found in donor, skipping")

    # 3. Update config.json
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    old_vocab = config.get("vocab_size", "unknown")
    config["vocab_size"] = new_vocab_size
    config.pop("quantization_config", None)  # No longer quantized
    hidden_size = config.get("hidden_size", 2880)

    # Update eos/pad token IDs from donor tokenizer_config.json
    donor_tok_config = donor_tokenizer_dir / "tokenizer_config.json"
    if donor_tok_config.exists():
        with open(donor_tok_config) as f:
            tok_cfg = json.load(f)
        if "eos_token" in tok_cfg:
            from transformers import AutoTokenizer
            donor_tok = AutoTokenizer.from_pretrained(str(donor_tokenizer_dir))
            old_eos = config.get("eos_token_id")
            old_pad = config.get("pad_token_id")
            config["eos_token_id"] = donor_tok.eos_token_id
            if donor_tok.pad_token_id is not None:
                config["pad_token_id"] = donor_tok.pad_token_id
            print(f"  Config: eos_token_id {old_eos} → {config['eos_token_id']}")
            print(f"  Config: pad_token_id {old_pad} → {config.get('pad_token_id')}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: vocab_size {old_vocab} → {new_vocab_size}, removed quantization_config")

    # 4. Process safetensor shards
    torch.manual_seed(seed)

    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Expected sharded checkpoint, got no index at {index_path}")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Build new weight map (accounting for renamed/dropped tensors)
    new_weight_map = {}
    all_shard_files = sorted(set(weight_map.values()))

    for shard_file in all_shard_files:
        print(f"\n  Processing {shard_file}...")
        shard_path = input_dir / shard_file
        tensors = load_file(shard_path)

        # Dequantize MXFP4 and clean up
        tensors = dequant_shard(tensors)

        # Resize embeddings if present
        for key in [EMBED_KEY, OUTPUT_KEY]:
            if key in tensors:
                old_shape = tensors[key].shape
                print(f"  Resizing {key}: {list(old_shape)} → [{new_vocab_size}, {hidden_size}]")
                tensors[key] = torch.normal(
                    0, init_std, size=(new_vocab_size, hidden_size), dtype=torch.bfloat16
                )

        # Save processed shard
        save_file(tensors, output_dir / shard_file)
        print(f"  Saved {shard_file} ({len(tensors)} tensors)")

        # Update weight map
        for k in tensors:
            new_weight_map[k] = shard_file

    # 5. Save updated index
    new_index = {"metadata": {}, "weight_map": new_weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"\nDone. Saved to {output_dir}")
    print(f"  Vocab: {old_vocab} → {new_vocab_size}")
    print(f"  Embeddings: N(0, {init_std}), seed={seed}")
    print(f"  Expert weights: dequantized MXFP4 → bf16")
    print(f"  Tokenizer: from {donor_tokenizer_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer swap + MXFP4 dequant + embedding resize for GPT-OSS"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Path to original GPT-OSS checkpoint")
    parser.add_argument("--donor-tokenizer-dir", type=str, required=True, help="Path to donor tokenizer files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for swapped model")
    parser.add_argument("--new-vocab-size", type=int, required=True, help="New vocabulary size (e.g. 248320)")
    parser.add_argument("--init-std", type=float, default=0.02, help="Std dev for random embedding init")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for embedding randomization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    donor_tokenizer_dir = Path(args.donor_tokenizer_dir)
    output_dir = Path(args.output_dir)

    assert input_dir.exists(), f"Input dir not found: {input_dir}"
    assert donor_tokenizer_dir.exists(), f"Donor tokenizer dir not found: {donor_tokenizer_dir}"
    assert input_dir != output_dir, "Input and output dirs must be different"

    print(f"Input model:      {input_dir}")
    print(f"Donor tokenizer:  {donor_tokenizer_dir}")
    print(f"Output:           {output_dir}")
    print(f"New vocab size:   {args.new_vocab_size}")
    print(f"Embedding init:   N(0, {args.init_std}), seed={args.seed}")
    print()

    tokenizer_swap(input_dir, donor_tokenizer_dir, output_dir, args.new_vocab_size, args.init_std, args.seed)


if __name__ == "__main__":
    main()
