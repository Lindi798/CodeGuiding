#!/usr/bin/env python3
"""
Run inference on first sample from apps_test.jsonl using Qwen model.
Direct question input without prompt wrapper.

Usage:
    python infer_apps_sample.py \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --jsonl-file ./data/apps/apps_test.jsonl \
        --output output_sample_1.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_model(model_name: str, use_4bit: bool, adapter_path: Optional[str], device_map: str = "auto"):
    """Load model with optional LoRA adapter."""
    print(f"? Loading model: {model_name}")
    
    if use_4bit:
        try:
            from pgt.training.utils import build_bnb_config
            bnb_config = build_bnb_config(use_4bit=True)
            print("? Using 4-bit quantization")
        except ImportError:
            print("? pgt not available, loading without quantization")
            bnb_config = None
    else:
        bnb_config = None
    
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    if adapter_path:
        print(f"? Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    print("? Model loaded")
    return model


def _generate(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    verbose: bool = False,
) -> str:
    """Generate response from model."""
    if verbose:
        print(f"\n? Prompt length: {len(prompt_text)} chars")
    
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    if verbose:
        print(f"? Input token count: {input_ids.shape[1]}")
    
    do_sample = temperature > 0
    
    print(f"? Generating with max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}...")
    
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    new_tokens = gen_ids[0][input_ids.shape[1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    if verbose:
        print(f"? Output token count: {len(new_tokens)}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on APPS dataset sample"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/home/liyongqi/models/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        default="./data/apps/apps_test.jsonl",
        help="JSONL dataset file",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_output.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA adapter path (optional)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Load sample
    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"? File not found: {args.jsonl_file}")
        sys.exit(1)
    
    print(f"? Loading sample {args.sample_index} from {args.jsonl_file}...")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == args.sample_index:
                sample = json.loads(line)
                break
        else:
            print(f"? Sample index {args.sample_index} out of range")
            sys.exit(1)
    
    # Get question
    question = sample.get("question", "")
    if not question:
        print("? 'question' field not found in sample")
        sys.exit(1)
    
    print(f"? Loaded sample with question length: {len(question)} chars")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = _load_model(
        args.model_name,
        use_4bit=args.use_4bit,
        adapter_path=args.adapter,
        device_map=args.device_map,
    )
    
    # Generate
    print("\n" + "=" * 80)
    print("INFERENCE")
    print("=" * 80)
    
    response = _generate(
        model,
        tokenizer,
        question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )
    
    print(f"? Generation complete\n")
    
    # Prepare output
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "adapter_path": args.adapter,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "sample_index": args.sample_index,
        "sample_fields": {
            k: v for k, v in sample.items()
            if k not in ["solutions", "input_output"]  # Skip large fields
        },
        "question": question,
        "response": response,
    }
    
    # Display
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"\n? Model: {args.model_name}")
    print(f"? Sample index: {args.sample_index}")
    print(f"\n? Question ({len(question)} chars):")
    print("-" * 80)
    print(question[:500] + ("..." if len(question) > 500 else ""))
    print("-" * 80)
    print(f"\n? Response ({len(response)} chars):")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n? Output saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
