#!/usr/bin/env python3
"""Minimal PyTorch oracle for Gemma 4 E2B persistent decode validation.

Text-only greedy decode against `Gemma4ForConditionalGeneration`. Captures
last-token logits at every step so SuperSonic can compare its kernel logits
once the Gemma 4 megakernel exists. Skips state export (KV/conv) — that lives
in a future "full oracle" pass.

JSON schema matches `crates/runner/src/oracle.rs::OracleOutput`:
  load_ms, prefill_ms, decode_ms       (f64)
  prompt_tokens, generated_tokens      (usize)
  prefill_logits                       (Vec<f32>, vocab=262144)
  decode_logits                        (Vec<Vec<f32>>, one per step)
  generated_token_ids                  (Vec<u32>)

Usage:
    python3 gemma4_oracle.py \
        --model-dir /path/to/gemma-4-E2B \
        --prompt "Hello, world" \
        --max-new-tokens 4 \
        --dtype bf16
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Path to local Gemma 4 snapshot dir (config.json + safetensors + tokenizer.json)")
    p.add_argument("--prompt", required=True, help="Plain text prompt")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cpu", help="cpu, cuda:0, etc.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch_dtype,
    )
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    load_ms = (time.perf_counter() - t0) * 1000.0

    device = next(model.parameters()).device
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)

    # Prefill — text-only forward pass, no images/audio.
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # Gemma4ForConditionalGeneration returns logits [batch, seq, vocab].
    prefill_last_logits = out.logits[0, -1, :].float().cpu().tolist()
    past = out.past_key_values
    next_token = int(out.logits[0, -1, :].argmax())

    decode_logits: list[list[float]] = []
    generated_ids: list[int] = []
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        generated_ids.append(next_token)
        token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=token_input, past_key_values=past, use_cache=True)
        past = out.past_key_values
        decode_logits.append(out.logits[0, -1, :].float().cpu().tolist())
        next_token = int(out.logits[0, -1, :].argmax())
    decode_ms = (time.perf_counter() - t0) * 1000.0

    payload = {
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "prefill_logits": prefill_last_logits,
        "decode_logits": decode_logits,
        "generated_token_ids": generated_ids,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
