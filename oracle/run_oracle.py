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
import ctypes as ct
import gc
import glob
import json
import os
import time
from math import prod

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import safetensors.torch


def _load_bnb_cuda_ops():
    from bitsandbytes.backends.cuda.ops import (
        _cuda_device_of,
        _get_tensor_stream,
        get_ptr,
        lib,
    )

    return _cuda_device_of, _get_tensor_stream, get_ptr, lib


def tensor_to_b64(t: torch.Tensor) -> str:
    # Serialize the tensor's logical byte view, not the full backing storage.
    # Some hooked tensors are views; exporting the entire storage can append
    # unrelated bytes and poison downstream parity checks.
    raw = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def int8_vectorwise_quant_export(a: torch.Tensor, threshold: float):
    """BitsAndBytes CUDA vectorwise quant with the same kernel, minus the
    upstream `.view(-1)` bug that trips on this single-row Llama trace.
    """
    _cuda_device_of, _get_tensor_stream, get_ptr, lib = _load_bnb_cuda_ops()
    if a.dtype != torch.float16:
        raise TypeError(f"expected float16 activation for export, got {a.dtype}")
    rows = prod(a.shape[:-1])
    cols = a.shape[-1]
    row_stats = torch.empty(rows, device=a.device, dtype=torch.float32)
    out_row = torch.empty(a.shape, device=a.device, dtype=torch.int8)
    outlier_cols = None
    if threshold > 0.0:
        outliers = a.abs() >= threshold
        if outliers.any():
            reduce_dims = tuple(range(outliers.ndim - 1))
            outlier_mask = outliers.any(dim=reduce_dims) if reduce_dims else outliers
            outlier_cols = torch.argwhere(outlier_mask).reshape(-1)
        else:
            outlier_cols = torch.empty(0, device=a.device, dtype=torch.int64)
    with _cuda_device_of(a):
        lib.cint8_vector_quant(
            get_ptr(a),
            get_ptr(out_row),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            _get_tensor_stream(a),
        )
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0
    return out_row, row_stats, outlier_cols


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
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load the model with BitsAndBytes INT8 quantization")
    parser.add_argument("--fp8-model-dir",
                        help="Path to FP8 safetensors dir. Weights are dequanted to BF16 "
                             "and injected into --model-id's architecture.")
    parser.add_argument("--trace-full-attn-layer", type=int,
                        help="Emit internal tensors for one full-attention layer at the last prompt token")
    args = parser.parse_args()

    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x]
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    t0 = time.perf_counter()
    if args.fp8_model_dir:
        model = _load_fp8_model(args.model_id, args.fp8_model_dir, torch_dtype)
    elif args.load_in_8bit:
        if args.device == "cpu":
            raise SystemExit("--load-in-8bit requires a CUDA/XPU/HPU device, not cpu")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            device_map={"": args.device},
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype, trust_remote_code=True,
        )
    model.eval()
    if args.device != "cpu" and not args.load_in_8bit:
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
        trace_layer_input = None
        trace_normed_input = None
        trace_gated_actual = None
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            if args.trace_full_attn_layer == layer_idx:
                def capture_layer_input(module, inputs):
                    nonlocal trace_layer_input
                    trace_layer_input = inputs[0].detach()
                def capture_normed_input(module, inputs, kwargs):
                    nonlocal trace_normed_input
                    hidden_states = kwargs.get("hidden_states")
                    if hidden_states is None and len(inputs) > 0:
                        hidden_states = inputs[0]
                    if hidden_states is None:
                        raise RuntimeError(
                            f"Failed to capture self_attn hidden_states for layer {args.trace_full_attn_layer}"
                        )
                    trace_normed_input = hidden_states.detach()
                def capture_gated_actual(module, inputs):
                    nonlocal trace_gated_actual
                    if len(inputs) == 0:
                        raise RuntimeError(
                            f"Failed to capture o_proj input for layer {args.trace_full_attn_layer}"
                        )
                    trace_gated_actual = inputs[0].detach()
                hooks.append(layer.register_forward_pre_hook(capture_layer_input))
                hooks.append(
                    layer.self_attn.register_forward_pre_hook(
                        capture_normed_input, with_kwargs=True
                    )
                )
                hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_gated_actual))
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

        if args.trace_full_attn_layer is not None:
            if trace_layer_input is None or trace_normed_input is None:
                raise RuntimeError(f"Failed to capture full-attention trace inputs for layer {args.trace_full_attn_layer}")
            layer_idx = args.trace_full_attn_layer
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            model_type = getattr(model.model.config, "model_type", "")
            input_shape = trace_normed_input.shape[:-1]
            batch_size, seq_len = trace_normed_input.shape[:2]
            num_heads = getattr(attn, "num_heads", attn.config.num_attention_heads)
            num_kv_heads = getattr(attn, "num_key_value_heads", attn.config.num_key_value_heads)
            q_dim = num_heads * attn.head_dim
            seq_len = trace_layer_input.shape[1]
            position_ids = torch.arange(seq_len, device=device).view(1, seq_len)
            q_proj_out = attn.q_proj(trace_normed_input)
            if q_proj_out.shape[-1] == q_dim * 2:
                q_raw, gate = torch.chunk(q_proj_out, 2, dim=-1)
            elif q_proj_out.shape[-1] == q_dim:
                q_raw = q_proj_out
                gate = torch.zeros_like(q_raw)
            else:
                raise RuntimeError(
                    f"Unexpected q_proj width {q_proj_out.shape[-1]} for layer {layer_idx}; "
                    f"expected {q_dim} or {q_dim * 2}"
                )

            q_raw = q_raw.view(batch_size, seq_len, num_heads, attn.head_dim)
            k_raw = attn.k_proj(trace_normed_input).view(batch_size, seq_len, num_kv_heads, attn.head_dim)
            v_raw = attn.v_proj(trace_normed_input).view(batch_size, seq_len, num_kv_heads, attn.head_dim)

            q_norm = getattr(attn, "q_norm", None)
            k_norm = getattr(attn, "k_norm", None)
            if q_norm is not None:
                q_raw = q_norm(q_raw)
            if k_norm is not None:
                k_raw = k_norm(k_raw)

            query_states = q_raw.transpose(1, 2)
            key_states = k_raw.transpose(1, 2)
            value_states = v_raw.transpose(1, 2)
            q_pre_rope = query_states
            k_pre_rope = key_states

            if model_type == "qwen3_5":
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    ALL_ATTENTION_FUNCTIONS,
                    apply_rotary_pos_emb,
                    create_causal_mask,
                    eager_attention_forward,
                )

                position_ids_qwen = torch.arange(seq_len, device=device).view(1, 1, -1).expand(
                    4, batch_size, -1
                )
                text_position_ids = position_ids_qwen[0]
                rope_position_ids = position_ids_qwen[1:]
                causal_mask = create_causal_mask(
                    config=model.model.config,
                    inputs_embeds=trace_layer_input,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )
                cos, sin = model.model.rotary_emb(trace_normed_input, rope_position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    attn.config._attn_implementation, eager_attention_forward
                )
                attn_output, _ = attention_interface(
                    attn,
                    query_states,
                    key_states,
                    value_states,
                    causal_mask,
                    dropout=0.0,
                    scaling=attn.scaling,
                )
            elif model_type == "llama":
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

                cos, sin = model.model.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                if num_heads != num_kv_heads:
                    repeat_factor = num_heads // num_kv_heads
                    key_states_for_attn = key_states.repeat_interleave(repeat_factor, dim=1)
                    value_states_for_attn = value_states.repeat_interleave(repeat_factor, dim=1)
                else:
                    key_states_for_attn = key_states
                    value_states_for_attn = value_states
                scores = torch.matmul(query_states, key_states_for_attn.transpose(-1, -2)) * attn.scaling
                causal_mask = torch.full(
                    (1, 1, seq_len, seq_len),
                    torch.finfo(scores.dtype).min,
                    dtype=scores.dtype,
                    device=device,
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                probs = torch.softmax((scores + causal_mask).float(), dim=-1).to(query_states.dtype)
                attn_output = torch.matmul(probs, value_states_for_attn)
            else:
                raise RuntimeError(
                    f"--trace-full-attn-layer currently supports qwen3_5 and llama, got model_type={model_type!r}"
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            gate_last = gate[:, -1:, :]
            pre_gate_last = attn_output[:, -1:, :]
            if torch.count_nonzero(gate_last).item() == 0:
                gated_last = pre_gate_last
            else:
                gated_last = pre_gate_last * torch.sigmoid(gate_last)
            state_payload["traced_full_attn_layer"] = layer_idx
            state_payload["traced_full_attn_input"] = tensor_to_b64(trace_layer_input[:, -1:, :].to(torch_dtype))
            state_payload["traced_full_attn_normed"] = tensor_to_b64(trace_normed_input[:, -1:, :].to(torch_dtype))
            state_payload["traced_full_attn_q_proj"] = tensor_to_b64(
                q_pre_rope.transpose(1, 2).reshape(*input_shape, -1)[:, -1:, :].to(torch_dtype)
            )
            state_payload["traced_full_attn_gate_proj"] = tensor_to_b64(gate_last.to(torch_dtype))
            state_payload["traced_full_attn_k_proj"] = tensor_to_b64(
                k_pre_rope.transpose(1, 2).reshape(*input_shape, -1)[:, -1:, :].to(torch_dtype)
            )
            state_payload["traced_full_attn_v_proj"] = tensor_to_b64(
                value_states.transpose(1, 2).reshape(*input_shape, -1)[:, -1:, :].to(torch_dtype)
            )
            state_payload["traced_full_attn_q_rope"] = tensor_to_b64(
                query_states.transpose(1, 2).reshape(*input_shape, -1)[:, -1:, :].to(torch_dtype)
            )
            state_payload["traced_full_attn_k_rope"] = tensor_to_b64(
                key_states.transpose(1, 2).reshape(*input_shape, -1)[:, -1:, :].to(torch_dtype)
            )
            state_payload["traced_full_attn_pre_gate"] = tensor_to_b64(pre_gate_last.to(torch_dtype))
            state_payload["traced_full_attn_gated"] = tensor_to_b64(gated_last.to(torch_dtype))
            if trace_gated_actual is None:
                raise RuntimeError(
                    f"Failed to capture actual gated tensor for layer {args.trace_full_attn_layer}"
                )
            state_payload["traced_full_attn_gated_actual"] = tensor_to_b64(
                trace_gated_actual[:, -1:, :].to(torch_dtype)
            )
            post_norm_last = post_attn_norm_states[layer_idx].to(device)
            mlp_gate = layer.mlp.gate_proj(post_norm_last)
            mlp_up = layer.mlp.up_proj(post_norm_last)
            mlp_swiglu = layer.mlp.act_fn(mlp_gate) * mlp_up
            mlp_down = layer.mlp.down_proj(mlp_swiglu)
            state_payload["traced_mlp_gate"] = tensor_to_b64(mlp_gate.to(torch_dtype))
            state_payload["traced_mlp_up"] = tensor_to_b64(mlp_up.to(torch_dtype))
            state_payload["traced_mlp_swiglu"] = tensor_to_b64(mlp_swiglu.to(torch_dtype))
            state_payload["traced_mlp_down"] = tensor_to_b64(mlp_down.to(torch_dtype))
            if args.load_in_8bit:
                down_state = getattr(layer.mlp.down_proj, "state", None)
                threshold = float(getattr(down_state, "threshold", 0.0) or 0.0)
                mlp_swiglu_2d = mlp_swiglu.reshape(-1, mlp_swiglu.shape[-1])
                ca, sca, outlier_cols = int8_vectorwise_quant_export(
                    mlp_swiglu_2d.to(torch.float16), threshold=threshold
                )
                ca_dense, sca_dense, _ = int8_vectorwise_quant_export(
                    mlp_swiglu_2d.to(torch.float16), threshold=0.0
                )
                mixed_direct, suba = torch.ops.bitsandbytes.int8_mixed_scaled_mm(
                    mlp_swiglu_2d,
                    ca,
                    down_state.CB,
                    sca,
                    down_state.SCB,
                    outlier_cols,
                    None,
                )
                if outlier_cols is not None and outlier_cols.numel():
                    subb_t = torch.ops.bitsandbytes.int8_vectorwise_dequant.default(
                        down_state.CB[:, outlier_cols].contiguous(),
                        down_state.SCB,
                    ).to(mlp_swiglu_2d.dtype)
                    subb_t = subb_t.contiguous()
                else:
                    subb_t = torch.empty(
                        (down_state.CB.shape[0], 0),
                        device=mlp_swiglu_2d.device,
                        dtype=mlp_swiglu_2d.dtype,
                    )
                state_payload["traced_mlp_down_ca"] = tensor_to_b64(ca)
                state_payload["traced_mlp_down_ca_shape"] = list(ca.shape)
                state_payload["traced_mlp_down_sca"] = sca.float().cpu().tolist()
                state_payload["traced_mlp_down_ca_dense"] = tensor_to_b64(ca_dense)
                state_payload["traced_mlp_down_ca_dense_shape"] = list(ca_dense.shape)
                state_payload["traced_mlp_down_sca_dense"] = sca_dense.float().cpu().tolist()
                state_payload["traced_mlp_down_suba"] = tensor_to_b64(suba.to(torch_dtype))
                state_payload["traced_mlp_down_suba_shape"] = list(suba.shape)
                state_payload["traced_mlp_down_subb_t"] = tensor_to_b64(subb_t.to(torch_dtype))
                state_payload["traced_mlp_down_subb_t_shape"] = list(subb_t.shape)
                state_payload["traced_mlp_down_outlier_cols"] = (
                    []
                    if outlier_cols is None
                    else [int(v) for v in outlier_cols.int().cpu().tolist()]
                )
                state_payload["traced_mlp_down_outlier_threshold"] = threshold

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
