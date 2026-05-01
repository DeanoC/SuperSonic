#!/usr/bin/env python3
"""
Multi-layer reference forward for Qwen3.6-MoE — PR 4c step 1.

Chains the parity-tested per-block references — `reference_full_attention_layer`
(`qwen36_moe_oracle.py`), `reference_linear_attn_layer`
(`qwen36_moe_linear_oracle.py`), `reference_moe_ffn_block`
(`qwen36_moe_ffn_oracle.py`) — into one decode step across N transformer
layers. Walks the hybrid pattern (every 4th layer is full attention, indices
3/7/11/...; the other three are linear attention), applies the final RMSnorm,
and projects through `lm_head` to logits. This oracle is the parity ground
truth for PR 4c step 2's host-orchestrated multi-launch driver.

This script is pure orchestration: every per-layer math primitive — RMSnorm,
RoPE, GQA, conv1d, delta-rule recurrent update, top-k MoE dispatch, shared
expert, INT4 min/max group-quant — is imported from the per-block oracles
unchanged. The INT4 sidecar self-checks fire transitively per layer when
`--int4` is set; this script adds no new math.

Modes:
  --synthetic --num-layers 4
      4 layers (3 linear + 1 full at idx 3) at small synthetic geometry
      (default `hidden=256`). Primary parity gate: fits in <1 GiB host RAM
      and runs in seconds. INT4 mode pinned to `group_size=128`.
  --checkpoint --num-layers 8 --model-dir /path/to/Qwen3.6-MoE
      Loads per-layer tensors on demand from a real checkpoint to keep host
      RAM honest (35B BF16 is ~65 GiB — won't fit 64 GiB host). Production-
      geometry sanity check; not a CI gate. The final RMSnorm + `lm_head`
      stay synthesized in checkpoint mode too — loading the lm_head tensor
      from a multimodal Qwen3.5MoE checkpoint requires care around tied
      embeddings and tokenizer vocab, which is out of scope for step 1.

Schema:
  qwen36-moe-oracle-multilayer-v1            (BF16 weights)
  qwen36-moe-oracle-multilayer-int4-v1       (--int4)

Output JSON shape:
  {
    "schema": "...",
    "mode":   "synthetic" | "checkpoint",
    "position": int,
    "num_layers": int,
    "config": { hidden, vocab, attn{...}, lin{...}, ffn{...}, rms_norm_eps,
                int4_group_size? },
    "layers": [{ "layer_idx": int, "kind": "full" | "linear" }, ...],
    "input_hidden":  bf16_b64,
    "final_norm_w":  bf16_b64,
    "lm_head_w":     bf16_b64,
    "weights_per_layer":  [{ "attn": {...}, "ffn": {...} }, ...],
    "intermediates_per_layer": [{ layer_idx, kind,
                                  output_after_attn: bf16_b64,
                                  output_after_ffn:  bf16_b64 }, ...],
    "final_hidden": bf16_b64,
    "logits":       bf16_b64,
    "int4_weights_per_layer": [{ "attn": {name: {packed,scale,zero}},
                                 "ffn": {...} }, ...]    # only with --int4
  }

Step 2 will read this JSON, upload the per-layer weights + initial input
to the GPU, run the kernel chain, and compare against `intermediates_per_layer`
+ `final_hidden` + `logits`.
"""

from __future__ import annotations

import argparse
import base64  # noqa: F401  -- re-exported via per-block helpers; left explicit for grep.
import json
import sys
from pathlib import Path

# Per-block oracles live alongside this file; make them importable when this
# script is run directly (`python oracle/qwen36_moe_multilayer_oracle.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from qwen36_moe_oracle import (
    INT4_ATTN_TARGETS,                                                  # noqa: F401
    b64_bf16,
    b64_f32,
    b64_u8,
    load_layer_from_checkpoint as load_full_attn_from_checkpoint,
    quantize_int4_attn_weights,
    reference_full_attention_layer,
    rms_norm,
    synthesize_layer as synthesize_full_attn_layer,
)
from qwen36_moe_linear_oracle import (
    INT4_LINEAR_TARGETS,                                                # noqa: F401
    load_layer_from_checkpoint as load_linear_attn_from_checkpoint,
    quantize_int4_linear_weights,
    reference_linear_attn_layer,
    synthesize_layer as synthesize_linear_attn_layer,
)
from qwen36_moe_ffn_oracle import (
    INT4_FFN_TARGETS,                                                   # noqa: F401
    load_block_from_checkpoint as load_ffn_block_from_checkpoint,
    quantize_int4_ffn_weights,
    reference_moe_ffn_block,
    synthesize_block as synthesize_ffn_block,
)


# Hybrid pattern: every 4th layer is full attention. Indices 3, 7, 11, ...
# are full; everything else is linear. Matches Qwen3.6-MoE 35B-A3B layout.
HYBRID_FULL_ATTN_STRIDE = 4


def is_full_attn_layer(layer_idx: int) -> bool:
    return (layer_idx + 1) % HYBRID_FULL_ATTN_STRIDE == 0


# ---------------------------------------------------------------------------
# Per-layer weight synthesis. Each layer needs distinct synthetic weights or
# the chained forward becomes degenerate (every layer applies the same
# transform). `layer_seed` is derived from the base seed below.
# ---------------------------------------------------------------------------
def derive_layer_seed(base_seed: int, layer_idx: int, kind: str) -> int:
    """Distinct seed per (layer, kind) pair so attn and ffn weights aren't
    aliased and layer N differs from layer N+1. Two large coprimes mixed
    with the kind so attn/ffn within one layer also differ — without this
    they share the leading `randn(hidden)` draw and `input_hidden` would
    alias `post_attn_norm_w`."""
    kind_offset = 0 if kind == "attn" else 0x9E37_79B1
    return (base_seed + layer_idx * 1009 + kind_offset) & 0x7FFF_FFFF


def synthesize_layer_weights(
    *,
    layer_idx: int,
    hidden: int,
    attn_cfg: dict,
    lin_cfg: dict,
    ffn_cfg: dict,
    base_seed: int,
    dtype: torch.dtype,
    warm_state: bool,
) -> dict:
    attn_seed = derive_layer_seed(base_seed, layer_idx, "attn")
    ffn_seed = derive_layer_seed(base_seed, layer_idx, "ffn")

    if is_full_attn_layer(layer_idx):
        attn_w = synthesize_full_attn_layer(
            hidden=hidden,
            H=attn_cfg["num_attention_heads"],
            Hkv=attn_cfg["num_kv_heads"],
            d=attn_cfg["head_dim"],
            seed=attn_seed,
            dtype=dtype,
        )
    else:
        attn_w = synthesize_linear_attn_layer(
            hidden=hidden,
            K=lin_cfg["num_k_heads"],
            V=lin_cfg["num_v_heads"],
            k_dim=lin_cfg["head_k_dim"],
            v_dim=lin_cfg["head_v_dim"],
            kernel=lin_cfg["conv_kernel_dim"],
            seed=attn_seed,
            dtype=dtype,
            warm_state=warm_state,
        )
    # Chained forward: each layer's input comes from the previous layer's
    # output, not from a freshly synthesised vector. Drop the per-layer
    # `input_hidden` the per-block synth routines include.
    attn_w.pop("input_hidden", None)

    ffn_w = synthesize_ffn_block(
        hidden=hidden,
        num_experts=ffn_cfg["num_experts"],
        moe_intermediate=ffn_cfg["moe_intermediate"],
        shared_intermediate=ffn_cfg["shared_intermediate"],
        seed=ffn_seed,
        dtype=dtype,
    )
    ffn_w.pop("input_hidden", None)

    return {"attn": attn_w, "ffn": ffn_w}


def load_layer_weights_from_checkpoint(
    *,
    layer_idx: int,
    model_dir: Path,
    weight_prefix: str,
    device: str,
    dtype: torch.dtype,
    hidden: int,
    lin_cfg: dict,
    state_kind: str,
    base_seed: int,
) -> dict:
    """Per-block checkpoint load wrappers fetch the trained projection
    weights for one layer. Linear-attn layers also need conv + recurrent
    state, which the per-block loader doesn't supply (the kernel-side caller
    owns state lifetime), so we synthesise them here using the same warm/fresh
    convention as the per-block linear oracle."""
    if is_full_attn_layer(layer_idx):
        attn_w = load_full_attn_from_checkpoint(
            model_dir=model_dir,
            layer_idx=layer_idx,
            weight_prefix=weight_prefix,
            device=device,
        )
        # Norm weights stored BF16 in the checkpoint; cast to working dtype
        # so the matmuls don't silently mix dtypes.
        for k in ("input_norm_w", "q_norm_w", "k_norm_w"):
            attn_w[k] = attn_w[k].to(dtype)
    else:
        attn_w = load_linear_attn_from_checkpoint(
            model_dir=model_dir,
            layer_idx=layer_idx,
            weight_prefix=weight_prefix,
            device=device,
        )
        for k in ("input_norm_w", "norm_w"):
            attn_w[k] = attn_w[k].to(dtype)

        K = lin_cfg["num_k_heads"]
        V = lin_cfg["num_v_heads"]
        k_dim = lin_cfg["head_k_dim"]
        v_dim = lin_cfg["head_v_dim"]
        kernel = lin_cfg["conv_kernel_dim"]
        qkv_dim = 2 * K * k_dim + V * v_dim

        # Reproducible per-layer state — same RNG style as the per-block
        # linear oracle's checkpoint mode, with a layer-specific seed so
        # different layers see different state.
        g = torch.Generator().manual_seed(
            derive_layer_seed(base_seed, layer_idx, "state")
        )
        if state_kind == "warm":
            attn_w["conv_state_before"] = (
                torch.randn(qkv_dim, kernel - 1, generator=g, dtype=torch.float32) * 0.5
            ).to(dtype)
            attn_w["recurrent_state_before"] = (
                torch.randn(V, k_dim, v_dim, generator=g, dtype=torch.float32) * 0.1
            )
        else:
            attn_w["conv_state_before"] = torch.zeros(qkv_dim, kernel - 1, dtype=dtype)
            attn_w["recurrent_state_before"] = torch.zeros(
                V, k_dim, v_dim, dtype=torch.float32
            )

    ffn_w = load_ffn_block_from_checkpoint(
        model_dir=model_dir,
        layer_idx=layer_idx,
        weight_prefix=weight_prefix,
        device=device,
    )
    ffn_w["post_attn_norm_w"] = ffn_w["post_attn_norm_w"].to(dtype)

    return {"attn": attn_w, "ffn": ffn_w}


# ---------------------------------------------------------------------------
# Decode loop. Walks the hybrid pattern, residuals chained, returns per-layer
# intermediates + final_hidden + logits.
# ---------------------------------------------------------------------------
def run_multilayer_decode(
    *,
    layers: list[dict],
    initial_hidden: torch.Tensor,
    final_norm_w: torch.Tensor,
    lm_head_w: torch.Tensor,
    attn_cfg: dict,
    lin_cfg: dict,
    ffn_cfg: dict,
    rms_norm_eps: float,
    position: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    h = initial_hidden
    intermediates: list[dict] = []

    for layer_idx, layer in enumerate(layers):
        if is_full_attn_layer(layer_idx):
            attn_out = reference_full_attention_layer(
                input_hidden=h,
                position=position,
                num_heads=attn_cfg["num_attention_heads"],
                num_kv_heads=attn_cfg["num_kv_heads"],
                head_dim=attn_cfg["head_dim"],
                rotary_dim=attn_cfg["rotary_dim"],
                rope_theta=attn_cfg["rope_theta"],
                rms_norm_eps=rms_norm_eps,
                **layer["attn"],
            )
            kind = "full"
        else:
            attn_out = reference_linear_attn_layer(
                input_hidden=h,
                num_k_heads=lin_cfg["num_k_heads"],
                num_v_heads=lin_cfg["num_v_heads"],
                head_k_dim=lin_cfg["head_k_dim"],
                head_v_dim=lin_cfg["head_v_dim"],
                conv_kernel_dim=lin_cfg["conv_kernel_dim"],
                rms_norm_eps=rms_norm_eps,
                **layer["attn"],
            )
            kind = "linear"

        h_after_attn = attn_out["output_hidden"]
        ffn_out = reference_moe_ffn_block(
            input_hidden=h_after_attn,
            num_experts=ffn_cfg["num_experts"],
            moe_intermediate=ffn_cfg["moe_intermediate"],
            shared_intermediate=ffn_cfg["shared_intermediate"],
            top_k=ffn_cfg["top_k"],
            rms_norm_eps=rms_norm_eps,
            **layer["ffn"],
        )
        h_after_ffn = ffn_out["output_hidden"]

        intermediates.append({
            "layer_idx": layer_idx,
            "kind": kind,
            "output_after_attn": h_after_attn,
            "output_after_ffn": h_after_ffn,
        })
        h = h_after_ffn

    final_hidden = rms_norm(h, final_norm_w, rms_norm_eps)
    # F32 GEMV → cast to working dtype for the parity gate. Production
    # decode samples in F32 too; matching that here keeps the kernel-side
    # comparison apples-to-apples once step 2 lands.
    logits = (
        final_hidden.to(torch.float32) @ lm_head_w.to(torch.float32).T
    ).to(initial_hidden.dtype)

    return intermediates, final_hidden, logits


# ---------------------------------------------------------------------------
# Encoding helpers for the JSON payload.
# ---------------------------------------------------------------------------
def encode_attn_weights(
    layer_idx: int, attn_w: dict, encode
) -> dict[str, str]:
    """Encode the projection + norm + state tensors for one layer's attention
    block. Linear-attn `recurrent_state_before` is F32 in production so it's
    encoded as F32 even when the working dtype is BF16; everything else
    follows the working dtype."""
    out: dict[str, str] = {}
    for name, tensor in attn_w.items():
        if tensor is None:
            continue
        if not is_full_attn_layer(layer_idx) and name == "recurrent_state_before":
            out[name] = b64_f32(tensor)
        else:
            out[name] = encode(tensor)
    return out


def encode_ffn_weights(ffn_w: dict, encode) -> dict[str, str]:
    return {name: encode(tensor) for name, tensor in ffn_w.items() if tensor is not None}


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-layer Qwen3.6-MoE reference forward (PR 4c step 1)"
    )
    p.add_argument("--mode", choices=["synthetic", "checkpoint"], default="synthetic")
    p.add_argument("--model-dir", type=Path,
                   help="HuggingFace safetensors dir (checkpoint mode)")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--num-layers", type=int, default=4,
                   help="Number of transformer layers to chain. Default 4 = "
                        "3 linear + 1 full (smallest hybrid window).")
    p.add_argument("--position", type=int, default=0,
                   help="Decode position (only affects full-attn layers' RoPE).")
    p.add_argument("--state", choices=["fresh", "warm"], default="warm",
                   help="Initial state for linear-attn layers. Warm exercises "
                        "decay + delta update; fresh starts from zeros.")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="Base seed for synthesis (per-layer + per-kind seeds "
                        "derived from this).")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON output path")

    # Geometry — synthetic defaults are tuned to satisfy INT4 group_size=128
    # divisibility on every quantized projection at the smallest size.
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--vocab", type=int, default=1024,
                   help="lm_head output dim. Stays BF16 (no INT4 path).")

    # Full-attn geometry (small synthetic; production is H=16, Hkv=2, d=256).
    p.add_argument("--num-attention-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--rotary-dim", type=int, default=32)
    p.add_argument("--rope-theta", type=float, default=1e7)

    # Linear-attn geometry (small synthetic; production is K=16, V=32, k=v=128).
    p.add_argument("--num-k-heads", type=int, default=4)
    p.add_argument("--num-v-heads", type=int, default=8)
    p.add_argument("--head-k-dim", type=int, default=32)
    p.add_argument("--head-v-dim", type=int, default=32)
    p.add_argument("--conv-kernel-dim", type=int, default=4)

    # FFN geometry (small synthetic; production is E=256, I=Is=512, top_k=8).
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--moe-intermediate", type=int, default=128,
                   help="Per-expert intermediate dim. Must be ≥ INT4 "
                        "group_size (128) so down_proj's in_features fits.")
    p.add_argument("--shared-intermediate", type=int, default=128)
    p.add_argument("--top-k", type=int, default=2)

    p.add_argument("--rms-norm-eps", type=float, default=1e-6)

    p.add_argument("--int4", action="store_true",
                   help="Quantize each layer's projection weights via the "
                        "per-block oracles' INT4 helpers (min/max group-quant). "
                        "Schema becomes `qwen36-moe-oracle-multilayer-int4-v1`.")
    p.add_argument("--int4-group-size", type=int, default=128,
                   help="Group size for INT4 min/max quant. Pinned to 128 "
                        "across the runtime + bake.")

    p.add_argument("--no-emit-weights", action="store_true",
                   help="Skip per-layer weight emission. Use for cheap shape/"
                        "schema checks; step 2 (kernel parity) needs them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    attn_cfg = {
        "num_attention_heads": args.num_attention_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "rotary_dim": args.rotary_dim,
        "rope_theta": args.rope_theta,
    }
    lin_cfg = {
        "num_k_heads": args.num_k_heads,
        "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim,
        "head_v_dim": args.head_v_dim,
        "conv_kernel_dim": args.conv_kernel_dim,
    }
    ffn_cfg = {
        "num_experts": args.num_experts,
        "moe_intermediate": args.moe_intermediate,
        "shared_intermediate": args.shared_intermediate,
        "top_k": args.top_k,
    }

    # Synthesise initial input + final RMSnorm + lm_head with the base seed.
    # Both modes synthesise these — loading lm_head from a multimodal Qwen3.5MoE
    # checkpoint requires care around tied embeddings + tokenizer vocab and is
    # out of scope for step 1 (the per-layer parity is what we're after).
    g = torch.Generator().manual_seed(args.seed)
    initial_hidden = (
        torch.randn(args.hidden, generator=g, dtype=torch.float32) / args.hidden ** 0.5
    ).to(dtype)
    final_norm_w = (
        torch.ones(args.hidden, dtype=torch.float32)
        + torch.randn(args.hidden, generator=g, dtype=torch.float32) * 0.02
    ).to(dtype)
    # lm_head: [vocab, hidden]. ~N(0, 1/sqrt(hidden)) keeps logits O(1).
    lm_head_w = (
        torch.randn(args.vocab, args.hidden, generator=g, dtype=torch.float32)
        / args.hidden ** 0.5
    ).to(dtype)

    initial_hidden = initial_hidden.to(args.device)
    final_norm_w = final_norm_w.to(args.device)
    lm_head_w = lm_head_w.to(args.device)

    # Per-layer weights.
    layers: list[dict] = []
    for layer_idx in range(args.num_layers):
        if args.mode == "synthetic":
            layer = synthesize_layer_weights(
                layer_idx=layer_idx,
                hidden=args.hidden,
                attn_cfg=attn_cfg,
                lin_cfg=lin_cfg,
                ffn_cfg=ffn_cfg,
                base_seed=args.seed,
                dtype=dtype,
                warm_state=(args.state == "warm"),
            )
        else:
            if args.model_dir is None:
                raise SystemExit("--model-dir is required in checkpoint mode")
            layer = load_layer_weights_from_checkpoint(
                layer_idx=layer_idx,
                model_dir=args.model_dir,
                weight_prefix=args.weight_prefix,
                device=args.device,
                dtype=dtype,
                hidden=args.hidden,
                lin_cfg=lin_cfg,
                state_kind=args.state,
                base_seed=args.seed,
            )
        # Move to device. recurrent_state_before stays F32 (production
        # convention); everything else follows working dtype.
        layer["attn"] = {
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in layer["attn"].items()
        }
        layer["ffn"] = {
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in layer["ffn"].items()
        }
        layers.append(layer)

    # INT4 quantize-in-place. The per-block helpers self-check that
    # (packed, scale, zero) round-trips byte-for-byte to the BF16
    # reconstruction they write back into `layer["attn"|"ffn"]`. So this
    # loop transitively gives us the per-layer INT4 self-check the
    # acceptance criteria require.
    int4_per_layer: list[dict] | None = None
    if args.int4:
        int4_per_layer = []
        for layer_idx, layer in enumerate(layers):
            entry: dict = {}
            if is_full_attn_layer(layer_idx):
                entry["attn"] = quantize_int4_attn_weights(
                    layer["attn"], args.int4_group_size
                )
            else:
                entry["attn"] = quantize_int4_linear_weights(
                    layer["attn"], args.int4_group_size
                )
            entry["ffn"] = quantize_int4_ffn_weights(
                layer["ffn"], args.int4_group_size
            )
            int4_per_layer.append(entry)

    # Run the chained decode.
    intermediates, final_hidden, logits = run_multilayer_decode(
        layers=layers,
        initial_hidden=initial_hidden,
        final_norm_w=final_norm_w,
        lm_head_w=lm_head_w,
        attn_cfg=attn_cfg,
        lin_cfg=lin_cfg,
        ffn_cfg=ffn_cfg,
        rms_norm_eps=args.rms_norm_eps,
        position=args.position,
    )

    # Build JSON.
    encode = b64_bf16 if args.dtype == "bf16" else b64_f32

    config = {
        "hidden": args.hidden,
        "vocab": args.vocab,
        "rms_norm_eps": args.rms_norm_eps,
        "attn": {
            **attn_cfg,
            "rms_norm_eps": args.rms_norm_eps,
            "attn_output_gate": True,
        },
        "lin": {**lin_cfg, "rms_norm_eps": args.rms_norm_eps},
        "ffn": {**ffn_cfg, "rms_norm_eps": args.rms_norm_eps},
    }
    if args.int4:
        config["int4_group_size"] = args.int4_group_size

    layer_meta = [
        {"layer_idx": i, "kind": "full" if is_full_attn_layer(i) else "linear"}
        for i in range(args.num_layers)
    ]

    intermediate_payload = [
        {
            "layer_idx": item["layer_idx"],
            "kind": item["kind"],
            "output_after_attn": encode(item["output_after_attn"]),
            "output_after_ffn": encode(item["output_after_ffn"]),
        }
        for item in intermediates
    ]

    out = {
        "schema": (
            "qwen36-moe-oracle-multilayer-int4-v1"
            if args.int4
            else "qwen36-moe-oracle-multilayer-v1"
        ),
        "mode": args.mode,
        "state": args.state,
        "position": args.position,
        "num_layers": args.num_layers,
        "dtype": args.dtype,
        "config": config,
        "layers": layer_meta,
        "input_hidden": encode(initial_hidden),
        "final_norm_w": encode(final_norm_w),
        "lm_head_w": encode(lm_head_w),
        "intermediates_per_layer": intermediate_payload,
        "final_hidden": encode(final_hidden),
        "logits": encode(logits),
    }

    if not args.no_emit_weights:
        out["weights_per_layer"] = [
            {
                "attn": encode_attn_weights(i, layer["attn"], encode),
                "ffn": encode_ffn_weights(layer["ffn"], encode),
            }
            for i, layer in enumerate(layers)
        ]

    if int4_per_layer is not None and not args.no_emit_weights:
        out["int4_weights_per_layer"] = [
            {
                "attn": {
                    name: {
                        "packed": b64_u8(t["packed"]),
                        "scale": b64_bf16(t["scale"]),
                        "zero": b64_bf16(t["zero"]),
                    }
                    for name, t in entry["attn"].items()
                },
                "ffn": {
                    name: {
                        "packed": b64_u8(t["packed"]),
                        "scale": b64_bf16(t["scale"]),
                        "zero": b64_bf16(t["zero"]),
                    }
                    for name, t in entry["ffn"].items()
                },
            }
            for entry in int4_per_layer
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    # Eyeball line — drift across layers is what step 2's parity gate compares.
    in_norm = initial_hidden.to(torch.float32).norm().item()
    final_norm = final_hidden.to(torch.float32).norm().item()
    logits_norm = logits.to(torch.float32).norm().item()
    full_count = sum(1 for i in range(args.num_layers) if is_full_attn_layer(i))
    lin_count = args.num_layers - full_count
    int4_tag = "int4" if args.int4 else "bf16-w"
    sys.stderr.write(
        f"[multilayer-oracle] mode={args.mode} layers={args.num_layers} "
        f"({lin_count} linear + {full_count} full) "
        f"weights={int4_tag} state={args.state} pos={args.position} "
        f"|input|={in_norm:.4f} |final|={final_norm:.4f} "
        f"|logits|={logits_norm:.4f}\n"
    )


if __name__ == "__main__":
    main()
