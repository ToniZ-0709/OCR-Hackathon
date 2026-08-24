#!/usr/bin/env python3
"""Merge the selected LoRA adapter into an unquantized BF16 Qwen3-VL-4B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights not found under {args.adapter}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)

    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)

    # vLLM 0.11 ships a tokenizer stack that expects this legacy field name.
    tokenizer_config_path = args.output / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    extra_tokens = tokenizer_config.pop("extra_special_tokens", None)
    if isinstance(extra_tokens, list):
        tokenizer_config["additional_special_tokens"] = extra_tokens
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization_config"):
        raise RuntimeError("Merged BF16 config unexpectedly contains quantization_config")

    tensor_names = [name for name, _ in model.named_parameters()]
    if any("lora_" in name for name in tensor_names):
        raise RuntimeError("Merged model still contains LoRA tensors")

    print(f"Saved merged BF16 model to {args.output}")


if __name__ == "__main__":
    main()
