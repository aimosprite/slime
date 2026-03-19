#!/usr/bin/env python3
"""Export GPT-OSS to a fused BF16 HuggingFace checkpoint.

This is the FSDP/rollout counterpart to preprocess_gpt_oss.py. Instead of
splitting experts into Megatron bridge format, it materializes the
Transformers-native dequantized BF16 state dict and saves it back out as a
standard HF checkpoint. That gives FSDP train, ref, and SGLang rollout one
shared representation to load and sync against.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _copy_support_files(input_dir: Path, output_dir: Path) -> None:
    skip_names = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    for src in input_dir.iterdir():
        if src.name in skip_names:
            continue
        dst = output_dir / src.name
        if dst.exists():
            continue
        if src.is_file():
            shutil.copy2(src, dst)


def _clear_existing_weight_files(output_dir: Path) -> None:
    for pattern in (
        "model*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_weight_files(output_dir)

    print(f"Loading GPT-OSS from {input_dir} as fused BF16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(input_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    print(f"Saving fused BF16 checkpoint to {output_dir}...", flush=True)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "dtype"):
        model.config.dtype = "bfloat16"
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(input_dir), trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:
        print(f"Tokenizer export skipped: {exc}", flush=True)

    _copy_support_files(input_dir, output_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
