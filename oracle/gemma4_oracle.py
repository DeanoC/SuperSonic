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
  prefill_per_layer_hidden             (Vec<base64 BF16>, 35 entries for E2B — post-
                                        decoder-block hidden state at last prompt
                                        token, one per layer. Drives Rust layer-by-
                                        layer kernel validation.)
  prefill_per_layer_hidden_shape       (Vec<usize>, shape of each entry — always
                                        [1, 1, hidden])
  prefill_per_layer_pre_ple            (Vec<base64 BF16>, 35 entries — hidden state
                                        at the same checkpoint BEFORE the Per-Layer-
                                        Embeddings (PLE) branch and layer_scalar
                                        multiply. Useful for Rust kernels that do
                                        not yet plumb PLE.)
  per_layer_inputs                     (base64 BF16, shape
                                        [num_hidden_layers, hidden_size_per_layer_input]).
                                        Per-layer conditioning vector at the last
                                        prompt token. Used inside each layer's PLE
                                        branch: gelu_tanh(gate(hidden)) * per_layer_inputs[i].
  per_layer_inputs_shape               (Vec<usize>)
  per_layer_inputs_by_step             (Vec<base64 BF16>, one entry per decode step.
                                        Entry k's input token is `last_prompt_token`
                                        for k==0 and `generated_token_ids[k-1]` for
                                        k > 0. Each entry is shape
                                        [num_hidden_layers, hidden_size_per_layer_input]
                                        and is consumed by the Rust decode validator
                                        so it does not need to implement
                                        `project_per_layer_inputs` itself.)

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
        # Install forward pre-hooks on each decoder layer's `per_layer_input_gate`
        # to snapshot the hidden state right BEFORE the PLE branch runs (which is
        # the checkpoint our Rust kernel can reach without plumbing PLE). The
        # hook fires with a 1-tuple `(hidden_states,)` matching the gate's
        # forward signature.
        pre_ple_snapshots: dict[int, torch.Tensor] = {}
        hook_handles: list = []
        # AutoModelForImageTextToText returns Gemma4ForConditionalGeneration whose
        # `.model` attribute is a Gemma4Model multimodal wrapper; the text stack
        # with `.layers` lives under `model.model.language_model` (see HF's
        # Gemma4Model.__init__: `self.language_model = AutoModel.from_config(...)`).
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            language_model = model.model.language_model
        elif hasattr(model, "model"):
            language_model = model.model
        else:
            language_model = model
        for layer_idx, layer in enumerate(language_model.layers):
            if not getattr(layer, "hidden_size_per_layer_input", 0):
                continue
            gate = layer.per_layer_input_gate

            def make_hook(idx: int):
                def hook(_module, inputs):
                    pre_ple_snapshots[idx] = inputs[0].detach().clone()
                return hook

            hook_handles.append(gate.register_forward_pre_hook(make_hook(layer_idx)))

        # Re-run the language stack with output_hidden_states=True to grab the
        # post-final-norm hidden state at the last prompt token. Keep use_cache=False
        # to avoid mutating `past` (we keep the prefill cache from the call above).
        # Call the text model directly (`Gemma4TextModel`) so we get a clean
        # `BaseModelOutputWithPast` with `.last_hidden_state` + `.hidden_states`,
        # rather than the multimodal wrapper's aggregated output.
        try:
            with torch.no_grad():
                inner = language_model(
                    input_ids=input_ids, use_cache=False, output_hidden_states=True,
                )
        finally:
            for h in hook_handles:
                h.remove()
        last_hidden = inner.last_hidden_state[:, -1:, :]  # [1, 1, hidden]
        state_payload["prefill_hidden"] = tensor_to_b64(last_hidden.to(torch_dtype))
        state_payload["prefill_hidden_shape"] = list(last_hidden.shape)

        # HuggingFace returns `hidden_states` as a tuple of length num_layers+1:
        # index 0 is the input embedding (pre-layer-0), indices 1..=N are the
        # post-decoder-block outputs for layers 0..N-1. We want per-layer POST-block
        # outputs, so skip index 0 and take the last token of each remaining entry.
        # Each slice has shape [1, seq_len, hidden] — we save [1, 1, hidden] at
        # position -1 so Rust can compare against its own single-token kernel
        # output. Note: these are pre-final-norm; `prefill_hidden` above is
        # post-final-norm and comes from the same forward pass.
        hidden_tuple = inner.hidden_states
        if hidden_tuple is None:
            raise RuntimeError("output_hidden_states=True did not populate hidden_states")
        per_layer: list[str] = []
        per_layer_shape: list[int] = []
        for layer_h in hidden_tuple[1:]:
            last = layer_h[:, -1:, :].to(torch_dtype)
            per_layer.append(tensor_to_b64(last))
            if not per_layer_shape:
                per_layer_shape = list(last.shape)
        state_payload["prefill_per_layer_hidden"] = per_layer
        state_payload["prefill_per_layer_hidden_shape"] = per_layer_shape

        # Drain pre-PLE snapshots collected via the forward pre-hooks on the
        # second forward pass above. Each snapshot has shape [1, seq_len, hidden];
        # take the last prompt token, same slicing convention as the post-layer dump.
        pre_ple_entries: list[str] = []
        for i in range(len(hidden_tuple) - 1):
            snap = pre_ple_snapshots.get(i)
            if snap is None:
                raise RuntimeError(
                    f"pre-PLE hook fired no snapshot for layer {i}; does this layer "
                    "have hidden_size_per_layer_input=0?"
                )
            last = snap[:, -1:, :].to(torch_dtype)
            pre_ple_entries.append(tensor_to_b64(last))
        state_payload["prefill_per_layer_pre_ple"] = pre_ple_entries

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

        # Per-layer-input tensor consumed by each decoder layer's PLE branch.
        # HF computes this once per forward by combining the raw per-layer
        # embedding table lookup with a projection of the main embedding:
        #   per_layer_inputs_raw = embed_tokens_per_layer(input_ids)
        #       .reshape(..., num_layers, hidden_size_per_layer_input)
        #   per_layer_projection = per_layer_model_projection(embed_tokens(ids)
        #       * hidden_size**0.5) * hidden_size**-0.5
        #   per_layer_projection = per_layer_projection_norm(per_layer_projection)
        #   per_layer_inputs = (per_layer_projection + per_layer_inputs_raw)
        #                       * per_layer_input_scale   (scale = 2**-0.5)
        # We dump only the last prompt token's slice: shape
        # [num_hidden_layers, hidden_size_per_layer_input].
        with torch.no_grad():
            pli_raw = language_model.get_per_layer_inputs(input_ids, None)
            embeds_main = language_model.embed_tokens(input_ids)
            pli_full = language_model.project_per_layer_inputs(embeds_main, pli_raw)
        last_pli = pli_full[0, -1, :, :].to(torch_dtype)
        state_payload["per_layer_inputs"] = tensor_to_b64(last_pli)
        state_payload["per_layer_inputs_shape"] = list(last_pli.shape)

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

    if args.emit_state:
        # Per-step per_layer_inputs for decode validation. Each step feeds a
        # single token through get_per_layer_inputs + project_per_layer_inputs.
        # Both calls are position-independent (pure embedding table lookups +
        # elementwise math), so running them on a 1-token batch reproduces the
        # value HF would see mid-decode.
        last_prompt_token = int(input_ids[0, -1])
        step_input_ids: list[int] = [last_prompt_token] + generated_ids[:-1]
        pli_by_step: list[str] = []
        for tok_id in step_input_ids:
            tok_tensor = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            with torch.no_grad():
                raw = language_model.get_per_layer_inputs(tok_tensor, None)
                main = language_model.embed_tokens(tok_tensor)
                full = language_model.project_per_layer_inputs(main, raw)
            pli_by_step.append(tensor_to_b64(full[0, 0, :, :].to(torch_dtype)))
        state_payload["per_layer_inputs_by_step"] = pli_by_step

    payload = {
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "prefill_logits": prefill_last_logits,
        "decode_logits": decode_logits,
        "generated_token_ids": generated_ids,
        # Prompt token IDs (including any special tokens the tokenizer added).
        # Needed by the Rust single-layer validator so it can reconstruct the
        # layer-0 input hidden = embed_tokens[last_prompt_token] * sqrt(hidden).
        "prompt_token_ids": [int(x) for x in input_ids[0].cpu().tolist()],
    }
    payload.update(state_payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
