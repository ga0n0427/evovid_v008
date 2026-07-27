#!/usr/bin/env python3
"""Merge one PEFT LoRA adapter into its immutable base model for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from verl.utils.lora_train_loader import resolve_lora_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Base + LoRA, merge the adapter, and save an inference-only Hugging Face model."
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_shard_size", default="5GB")
    return parser.parse_args()


def has_model_weights(path: Path) -> bool:
    return any(path.glob("model*.safetensors")) or (path / "pytorch_model.bin").is_file()


def main() -> None:
    args = parse_args()
    adapter_path = resolve_lora_adapter(args.adapter_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        auto_class = AutoModelForImageTextToText
    else:
        auto_class = AutoModelForCausalLM

    print(f"Loading immutable base model for merge: {args.base_model}")
    base_model = auto_class.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter for merge: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    except (OSError, ValueError, TypeError) as exc:
        print(f"No processor saved for this model: {exc}")
    else:
        processor.save_pretrained(output_dir)

    if not (output_dir / "config.json").is_file() or not has_model_weights(output_dir):
        raise RuntimeError(f"Merged model is incomplete: {output_dir}")
    print(f"Saved merged inference model to: {output_dir}")


if __name__ == "__main__":
    main()
