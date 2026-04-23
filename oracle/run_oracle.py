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
import gc
import glob
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

import safetensors.torch


def tensor_to_b64(t: torch.Tensor) -> str:
    # Serialize the tensor's logical byte view, not the full backing storage.
    # Some hooked tensors are views; exporting the entire storage can append
    # unrelated bytes and poison downstream parity checks.
    raw = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def _load_fp8_model(bf16_model_id, fp8_model_dir, torch_dtype, block_size=128):
    """Load model structure with random init, then fill from FP8-dequanted values.

    FP8 E4M3 weights are dequanted via: bf16(fp8.float() * scale_inv.float())
    This matches the precision level our GPU kernel uses.

    Uses low_cpu_mem_usage + ignore_mismatched_sizes to avoid downloading BF16
    checkpoint weights. Peak RAM ≈ 1x model size.
    """
    import sys
    from transformers import AutoConfig

    # Load config, create model with random weights (no checkpoint download)
    config = AutoConfig.from_pretrained(bf16_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config.text_config if hasattr(config, "text_config") else config,
        torch_dtype=torch_dtype,
    )

    sd = model.state_dict()
    replaced = 0
    for sf in sorted(glob.glob(os.path.join(fp8_model_dir, "*.safetensors"))):
        tensors = safetensors.torch.load_file(sf, device="cpu")
        scales = {n.replace("_scale_inv", ""): t
                  for n, t in tensors.items() if "scale_inv" in n}
        for name, t in tensors.items():
            if "scale_inv" in name:
                continue
            hf_name = name.replace("model.language_model.", "model.")
            if hf_name not in sd:
                continue
            if t.dtype == torch.float8_e4m3fn and name in scales:
                s = scales[name].float()
                w = t.float()
                rows, cols = w.shape
                se = s.repeat_interleave(block_size, 0)[:rows] \
                      .repeat_interleave(block_size, 1)[:, :cols]
                sd[hf_name].copy_((w * se).to(torch_dtype))
                replaced += 1
            elif sd[hf_name].shape == t.shape:
                sd[hf_name].copy_(t.to(torch_dtype))
                replaced += 1
        del tensors, scales
        gc.collect()

    print(f"[oracle-fp8] replaced {replaced}/{len(sd)} weights", file=sys.stderr)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-ids", required=True, help="Comma-separated token IDs")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emit-state", action="store_true",
                        help="Emit prefill hidden + layer states for Rust decode engine")
    parser.add_argument("--fp8-model-dir",
                        help="Path to FP8 safetensors dir. Weights are dequanted to BF16 "
                             "and injected into --model-id's architecture.")
    args = parser.parse_args()

    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x]
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    t0 = time.perf_counter()
    if args.fp8_model_dir:
        model = _load_fp8_model(args.model_id, args.fp8_model_dir, torch_dtype)
    else:
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
        attn_residual_states = [None] * len(model.model.layers)
        post_attn_norm_states = [None] * len(model.model.layers)
        mlp_outputs = [None] * len(model.model.layers)
        layer_outputs = [None] * len(model.model.layers)
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            def make_pre_hook(idx):
                def hook(module, inputs):
                    attn_residual_states[idx] = inputs[0][:, -1:, :].detach().to(torch_dtype).cpu()
                return hook
            def make_forward_hook(idx, store):
                def hook(module, inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    store[idx] = out[:, -1:, :].detach().to(torch_dtype).cpu()
                return hook
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(make_pre_hook(layer_idx)))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_forward_hook(layer_idx, post_attn_norm_states)))
            hooks.append(layer.mlp.register_forward_hook(make_forward_hook(layer_idx, mlp_outputs)))
            hooks.append(layer.register_forward_hook(make_forward_hook(layer_idx, layer_outputs)))
        with torch.no_grad():
            out2 = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        for hook in hooks:
            hook.remove()
        # Last hidden state: [batch, seq, hidden] — take last token
        last_hidden = out2.last_hidden_state[:, -1:, :]  # [1, 1, hidden]
        state_payload["prefill_hidden"] = tensor_to_b64(last_hidden.to(torch_dtype))
        state_payload["prefill_hidden_shape"] = list(last_hidden.shape)
        state_payload["layer_attn_residual_states"] = [
            tensor_to_b64(h) for h in attn_residual_states
        ]
        state_payload["layer_post_attn_norm_states"] = [
            tensor_to_b64(h) for h in post_attn_norm_states
        ]
        state_payload["layer_mlp_outputs"] = [
            tensor_to_b64(h) for h in mlp_outputs
        ]
        state_payload["layer_hidden_states"] = [
            tensor_to_b64(h)
            for h in layer_outputs
        ]

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
