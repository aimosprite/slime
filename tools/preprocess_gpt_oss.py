"""Preprocess GPT-OSS model: dequantize MXFP4 experts and unfuse into per-expert format.

This converts the GPT-OSS HF checkpoint from:
  - MXFP4 quantized fused expert weights (gate_up_proj_blocks/scales, down_proj_blocks/scales)
To:
  - BF16 per-expert weights (experts.{e}.gate_proj.weight, experts.{e}.up_proj.weight, etc.)

Usage:
    python tools/preprocess_gpt_oss.py \
        --input /path/to/gpt-oss-20b \
        --output /path/to/gpt-oss-20b-bf16
"""

import argparse
import json
import math
import os
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from slime_plugins.mbridge.mxfp4_reference import convert_moe_packed_tensors_reference


def preprocess_gpt_oss(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(os.path.join(input_dir, "config.json")) as f:
        config = json.load(f)

    num_experts = config["num_local_experts"]
    intermediate_size = config["intermediate_size"]

    # Remove quantization config and ensure torch_dtype is bfloat16
    new_config = {k: v for k, v in config.items() if k != "quantization_config"}
    new_config["torch_dtype"] = "bfloat16"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy non-weight files
    for fname in os.listdir(input_dir):
        if fname in ("config.json", "model.safetensors.index.json"):
            continue
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Process safetensors: collect all weight names
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single file
        weight_map = None

    # Group weights by safetensors file
    if weight_map:
        files = set(weight_map.values())
    else:
        files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]

    all_output_tensors = OrderedDict()
    new_weight_map = {}

    for sf_file in sorted(files):
        sf_path = os.path.join(input_dir, sf_file)
        print(f"Processing {sf_file}...")
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)

                # Check if this is a quantized expert weight
                if key.endswith("_blocks"):
                    base_name = key[: -len("_blocks")]
                    scales_key = base_name + "_scales"
                    # Load scales from same or different file
                    try:
                        scales = f.get_tensor(scales_key)
                    except Exception:
                        # Try loading from other files
                        scales = _load_tensor_from_files(input_dir, files, scales_key)

                    print(f"  Dequantizing {base_name}...")
                    dequantized = convert_moe_packed_tensors_reference(tensor, scales)
                    _unfuse_experts(
                        base_name, dequantized, num_experts, intermediate_size, all_output_tensors, new_weight_map
                    )

                elif key.endswith("_scales"):
                    continue  # handled with _blocks

                elif ".mlp.experts.gate_up_proj_bias" in key:
                    # Unfuse bias: [E, 2*intermediate] -> per-expert gate_bias + up_bias
                    # GPT-OSS uses interleaved format: even=gate, odd=up
                    layer_prefix = key.rsplit(".mlp.experts.gate_up_proj_bias", 1)[0]
                    for e in range(num_experts):
                        gate_bias = tensor[e, 0::2]
                        up_bias = tensor[e, 1::2]
                        gname = f"{layer_prefix}.mlp.experts.{e}.gate_proj.bias"
                        uname = f"{layer_prefix}.mlp.experts.{e}.up_proj.bias"
                        all_output_tensors[gname] = gate_bias.contiguous()
                        all_output_tensors[uname] = up_bias.contiguous()
                        new_weight_map[gname] = "model.safetensors"
                        new_weight_map[uname] = "model.safetensors"

                elif ".mlp.experts.down_proj_bias" in key:
                    # Unfuse bias: [E, hidden] -> per-expert
                    layer_prefix = key.rsplit(".mlp.experts.down_proj_bias", 1)[0]
                    for e in range(num_experts):
                        dname = f"{layer_prefix}.mlp.experts.{e}.down_proj.bias"
                        all_output_tensors[dname] = tensor[e].contiguous()
                        new_weight_map[dname] = "model.safetensors"

                else:
                    all_output_tensors[key] = tensor
                    new_weight_map[key] = "model.safetensors"

    # Save output
    print(f"Saving {len(all_output_tensors)} tensors...")

    # Split into chunks of ~5GB each
    chunk_size = 5 * 1024 * 1024 * 1024  # 5GB
    chunks = []
    current_chunk = OrderedDict()
    current_size = 0

    for name, tensor in all_output_tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = OrderedDict()
            current_size = 0
        current_chunk[name] = tensor
        current_size += tensor_size

    if current_chunk:
        chunks.append(current_chunk)

    final_weight_map = {}
    if len(chunks) == 1:
        out_file = "model.safetensors"
        save_file(chunks[0], os.path.join(output_dir, out_file))
        for k in chunks[0]:
            final_weight_map[k] = out_file
    else:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            out_file = f"model-{i+1:05d}-of-{total:05d}.safetensors"
            save_file(chunk, os.path.join(output_dir, out_file))
            for k in chunk:
                final_weight_map[k] = out_file

        # Save index
        total_size = sum(t.numel() * t.element_size() for t in all_output_tensors.values())
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": final_weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)

    print(f"Done! Output saved to {output_dir}")


def _unfuse_experts(base_name, dequantized, num_experts, intermediate_size, output_tensors, weight_map):
    """Unfuse 3D expert tensor into per-expert format."""
    layer_prefix = base_name.rsplit(".mlp.experts.", 1)[0]
    weight_type = base_name.rsplit(".mlp.experts.", 1)[1]

    if "gate_up_proj" in weight_type:
        # [E, 2*intermediate, hidden] -> per-expert gate + up
        for e in range(num_experts):
            expert_weight = dequantized[e]  # [2*intermediate, hidden]
            # GPT-OSS uses interleaved format: [g0,u0,g1,u1,...] (even=gate, odd=up)
            gate = expert_weight[0::2]  # [intermediate, hidden]
            up = expert_weight[1::2]  # [intermediate, hidden]
            gname = f"{layer_prefix}.mlp.experts.{e}.gate_proj.weight"
            uname = f"{layer_prefix}.mlp.experts.{e}.up_proj.weight"
            output_tensors[gname] = gate.contiguous()
            output_tensors[uname] = up.contiguous()
            weight_map[gname] = "model.safetensors"
            weight_map[uname] = "model.safetensors"
    elif "down_proj" in weight_type:
        # [E, hidden, intermediate] -> per-expert
        for e in range(num_experts):
            dname = f"{layer_prefix}.mlp.experts.{e}.down_proj.weight"
            output_tensors[dname] = dequantized[e].contiguous()
            weight_map[dname] = "model.safetensors"


def _load_tensor_from_files(input_dir, files, key):
    """Load a tensor by searching across safetensors files."""
    for sf_file in files:
        sf_path = os.path.join(input_dir, sf_file)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"Tensor {key} not found in any safetensors file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GPT-OSS model")
    parser.add_argument("--input", required=True, help="Input HF model directory")
    parser.add_argument("--output", required=True, help="Output directory for BF16 model")
    args = parser.parse_args()
    preprocess_gpt_oss(args.input, args.output)
