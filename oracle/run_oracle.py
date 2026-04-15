#!/usr/bin/env python3
"""Minimal PyTorch oracle for Qwen3.5 persistent decode validation.

Usage:
    python3 run_oracle.py --model-id Qwen/Qwen3.5-0.8B \
        --prompt-ids 9707 \
        --max-new-tokens 8 \
        --dtype bf16

Outputs JSON with:
  - prefill_logits: last-token logits after prefill
  - decode_logits: list of logit vectors, one per decode step
  - generated_token_ids: list of greedy-sampled token IDs
  - prefill_hidden: base64-encoded BF16 hidden state after prefill [1, 1, hidden_dim]
  - prefill_kv_caches: list of {k: base64, v: base64} per full-attention layer
  - prefill_conv_states: list of base64 per linear-attention layer
  - prefill_recurrent_states: list of base64 per linear-attention layer
"""

import argparse
import base64
import json
import time

import torch
from transformers import AutoModelForCausalLM


def tensor_to_b64(t: torch.Tensor) -> str:
    # Use torch's raw bytes — numpy doesn't support BF16
    raw = bytes(t.contiguous().cpu().untyped_storage())
    return base64.b64encode(raw).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-ids", required=True, help="Comma-separated token IDs")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emit-state", action="store_true",
                        help="Emit prefill hidden + layer states for Rust decode engine")
    args = parser.parse_args()

    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x]
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    load_ms = (time.perf_counter() - t0) * 1000

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    prefill_logits = out.logits[0, -1, :].float().cpu().tolist()
    past = out.past_key_values
    next_token = int(out.logits[0, -1, :].argmax())

    # Extract prefill state if requested
    state_payload = {}
    if args.emit_state:
        # Get the hidden state after all layers but before final norm + lm_head.
        # We re-run with output_hidden_states=True.
        with torch.no_grad():
            out2 = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        # Last hidden state: [batch, seq, hidden] — take last token
        last_hidden = out2.last_hidden_state[:, -1:, :]  # [1, 1, hidden]
        state_payload["prefill_hidden"] = tensor_to_b64(last_hidden.to(torch_dtype))
        state_payload["prefill_hidden_shape"] = list(last_hidden.shape)

        # Extract KV caches and conv/recurrent states from past_key_values.
        # Qwen3.5 DynamicCache has .layers — each is either:
        #   DynamicLayer (full attention): .keys [1, nkv, seq, hd], .values
        #   LinearAttentionLayer: .conv_states [1, qkv_dim, kernel-1], .recurrent_states [1, nh, hk, hv]
        kv_caches = []
        conv_states = []
        recurrent_states = []

        for i, layer in enumerate(past.layers):
            layer_type = type(layer).__name__
            if layer_type == "DynamicLayer":
                k, v = layer.keys, layer.values
                kv_caches.append({
                    "layer": i,
                    "k": tensor_to_b64(k.to(torch_dtype)),
                    "k_shape": list(k.shape),
                    "v": tensor_to_b64(v.to(torch_dtype)),
                    "v_shape": list(v.shape),
                })
            elif layer_type == "LinearAttentionLayer":
                cs = layer.conv_states
                rs = layer.recurrent_states
                # Conv state: PyTorch [1, 6144, 4] → kernel [6144, 3]
                # Kernel state = sliding window of previous kern-1 inputs.
                # PyTorch stores the full kernel_size window; last kern-1 columns
                # are the carry-forward state (most recent values).
                cs_squeezed = cs.squeeze(0)[:, -3:]  # [6144, 3]
                # Recurrent state: PyTorch [1, 16, 128, 128] BF16 → kernel [16, 128, 128] F32
                rs_squeezed = rs.squeeze(0).float()  # [16, 128, 128] F32
                conv_states.append({
                    "layer": i,
                    "data": tensor_to_b64(cs_squeezed.to(torch_dtype)),
                    "shape": list(cs_squeezed.shape),
                })
                recurrent_states.append({
                    "layer": i,
                    "data": tensor_to_b64(rs_squeezed),
                    "shape": list(rs_squeezed.shape),
                    "dtype": "f32",
                })
            else:
                raise RuntimeError(f"Unknown cache layer type: {layer_type} at layer {i}")

        state_payload["kv_caches"] = kv_caches
        state_payload["conv_states"] = conv_states
        state_payload["recurrent_states"] = recurrent_states

    # Decode
    decode_logits = []
    generated_ids = []
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        generated_ids.append(next_token)
        token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=token_input, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_vec = out.logits[0, -1, :].float().cpu().tolist()
        decode_logits.append(logits_vec)
        next_token = int(out.logits[0, -1, :].argmax())
    decode_ms = (time.perf_counter() - t0) * 1000

    payload = {
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "generated_token_ids": generated_ids,
    }
    payload.update(state_payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
