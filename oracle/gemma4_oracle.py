#!/usr/bin/env python3
"""PyTorch oracle for Gemma 4 E2B persistent decode validation.

Text-only greedy decode against `Gemma4ForConditionalGeneration`. Captures
last-token logits per step. With `--emit-state`, also dumps post-prefill
hidden state and per-layer KV caches so the future Rust megakernel can boot
from a known-good prefill and validate decode step-by-step.

Gemma 4 E2B has 35 transformer layers but `num_kv_shared_layers: 20`, so HF's
DynamicCache only allocates 15 entries (one per *unique* K/V slot). Layers
15-34 reuse those slots per Gemma 4's sharing pattern; the kernel resolves
that mapping from `text_config`. Layer types alternate `[SWA*4, FULL]`, so
SWA entries have head_dim=256 and FULL entries head_dim=512.

JSON schema matches `crates/runner/src/oracle.rs::OracleOutput`:
  load_ms, prefill_ms, decode_ms       (f64)
  prompt_tokens, generated_tokens      (usize)
  prefill_logits                       (Vec<f32>, vocab=262144)
  decode_logits                        (Vec<Vec<f32>>, one per step)
  generated_token_ids                  (Vec<u32>)
With --emit-state, additionally:
  prefill_hidden                       (base64 BF16, shape [1, 1, hidden])
  prefill_hidden_shape                 (Vec<usize>)
  kv_caches                            (Vec<KvCacheDump>, 15 entries for E2B)

Usage:
    python3 gemma4_oracle.py \
        --model-dir /path/to/gemma-4-E2B \
        --prompt "Hello, world" \
        --max-new-tokens 4 \
        --dtype bf16 \
        [--emit-state]
"""

import argparse
import base64
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
    p.add_argument("--emit-state", action="store_true",
                   help="Also emit prefill_hidden + kv_caches for Rust decode bootstrap")
    return p.parse_args()


def tensor_to_b64(t: torch.Tensor) -> str:
    # numpy can't represent BF16; round-trip via raw torch storage bytes.
    # `.clone()` forces fresh contiguous storage sized to the logical tensor —
    # without it, a sliced view can leave the parent storage attached and
    # `untyped_storage()` returns the parent's bytes, not the slice's.
    flat = t.detach().cpu().contiguous().clone()
    return base64.b64encode(bytes(flat.untyped_storage())).decode("ascii")


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

    state_payload: dict = {}
    if args.emit_state:
        # Re-run the language stack with output_hidden_states=True to grab the
        # post-final-norm hidden state at the last prompt token. Keep use_cache=False
        # to avoid mutating `past` (we keep the prefill cache from the call above).
        with torch.no_grad():
            inner = model.model(
                input_ids=input_ids, use_cache=False, output_hidden_states=True,
            )
        last_hidden = inner.last_hidden_state[:, -1:, :]  # [1, 1, hidden]
        state_payload["prefill_hidden"] = tensor_to_b64(last_hidden.to(torch_dtype))
        state_payload["prefill_hidden_shape"] = list(last_hidden.shape)

        # 15 cache entries for E2B (35 layers - 20 shared). SWA layers store
        # k/v at head_dim=256, full layers at head_dim=512. We preserve HF's
        # native layout [1, num_kv_heads, seq, head_dim].
        kv_caches = []
        for i, layer in enumerate(past.layers):
            k, v = layer.keys, layer.values
            if k is None or v is None:
                raise RuntimeError(f"cache layer {i} ({type(layer).__name__}) "
                                   f"missing keys/values; cannot dump state")
            kv_caches.append({
                "layer": i,
                "k": tensor_to_b64(k.to(torch_dtype)),
                "k_shape": list(k.shape),
                "v": tensor_to_b64(v.to(torch_dtype)),
                "v_shape": list(v.shape),
            })
        state_payload["kv_caches"] = kv_caches

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
    payload.update(state_payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
