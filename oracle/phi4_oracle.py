#!/usr/bin/env python3
"""PyTorch oracle for Phi-4-mini persistent decode validation.

Phi-4-mini (microsoft/Phi-4-mini-instruct) uses the Phi3ForCausalLM architecture:
32 full-attention layers, GQA (24 heads, 8 kv-heads), head_dim=128,
partial_rotary_factor=0.75 → rot_dim=96, LongRoPE with short_factor/long_factor
each length 48, tied embeddings, no QK-norm, fused qkv_proj / gate_up_proj.

JSON output matches `crates/runner/src/oracle.rs::OracleOutput`. This minimal
version emits prefill_logits, decode_logits, and generated_token_ids — enough
to drive Rust `--validate` end-to-end once the phi4 megakernel lands. Per-layer
hidden-state dumps can be added later when debugging the kernel.

Usage:
    python3 phi4_oracle.py \
        --model-dir /path/to/Phi-4-mini-instruct \
        --prompt "Hello, world" \
        --max-new-tokens 8 \
        --dtype bf16
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Path to local Phi-4-mini snapshot (config.json + safetensors + tokenizer).")
    p.add_argument("--prompt", help="Plain text prompt.")
    p.add_argument("--prompt-ids",
                   help="Comma-separated token IDs (alternative to --prompt; bypasses tokenizer).")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cpu", help="cpu, cuda:0, etc.")
    args = p.parse_args()
    if args.prompt is None and args.prompt_ids is None:
        p.error("one of --prompt or --prompt-ids is required")
    return args


def greedy_decode(
    model,
    device: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[float], list[list[float]], list[int]]:
    """Prefill + `max_new_tokens` greedy decode steps. No EOS stopping so the
    Rust-side fixed-length comparison always has the same sample count."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    prefill_logits = out.logits[0, -1, :].float().cpu().tolist()
    past = out.past_key_values
    next_token = int(out.logits[0, -1, :].argmax())

    decode_logits: list[list[float]] = []
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        generated_ids.append(next_token)
        tok = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.no_grad():
            step = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = step.past_key_values
        decode_logits.append(step.logits[0, -1, :].float().cpu().tolist())
        next_token = int(step.logits[0, -1, :].argmax())
    return prefill_logits, decode_logits, generated_ids


def main() -> None:
    args = parse_args()
    torch_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = args.device

    load_started = time.perf_counter()
    # Use the built-in Phi3ForCausalLM from transformers rather than the
    # `modeling_phi3.py` shipped in the model snapshot — that custom file
    # imports `LossKwargs` which was removed in transformers 5.x. The
    # built-in implementation is functionally equivalent for inference.
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch_dtype,
        **({"device_map": device} if device != "cpu" else {}),
    )
    model.eval()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    if args.prompt_ids is not None:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not prompt_ids:
        print("prompt tokenized to empty list", file=sys.stderr)
        sys.exit(2)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    prefill_started = time.perf_counter()
    with torch.no_grad():
        # Warmup pass so prefill timing excludes first-run lazy init.
        _ = model(input_ids=input_ids[:, :1], use_cache=False)
    _ = time.perf_counter() - prefill_started  # discarded

    prefill_started = time.perf_counter()
    prefill_logits, decode_logits, generated_ids = greedy_decode(
        model, device, input_ids, args.max_new_tokens,
    )
    total_ms = (time.perf_counter() - prefill_started) * 1000.0
    # Coarse split: call the first pass prefill, remainder decode.
    prefill_ms = total_ms * (1.0 / (1.0 + args.max_new_tokens))
    decode_ms = total_ms - prefill_ms

    output = {
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "generated_tokens": len(generated_ids),
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "generated_token_ids": generated_ids,
        "prompt_token_ids": prompt_ids,
    }
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
