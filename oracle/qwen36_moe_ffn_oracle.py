#!/usr/bin/env python3
"""
PyTorch reference for one Qwen3.6-MoE FFN block's decode step.

Companion to PR 4b4 of docs/qwen36-moe-plan.md. The attention oracle
(`qwen36_moe_oracle.py`) covers the pre-FFN half of a layer; this oracle
covers the post-attention half — RMS norm, top-k MoE expert dispatch,
shared-expert path, and the final residual add. The HIP kernel's
parity test reads the JSON this script emits and compares its own
intermediates point for point.

The MoE FFN block on Qwen3-Next (35B-A3B):

  h_norm        = rms_norm(input_hidden, post_attn_norm_w, eps)

  # router
  router_logits = h_norm @ gate.T                         # [E=256]
  router_probs  = softmax(router_logits, dim=-1)
  topk_w, topk_idx = topk(router_probs, k=8)              # [k], [k]
  topk_w        = topk_w / topk_w.sum()                   # renormalised

  # per-expert FFN (gate_up_proj is fused: rows 0..512 = gate, 512..1024 = up)
  for j in range(k):
      e   = topk_idx[j]
      gu  = gate_up_proj[e] @ h_norm                      # [2 * I]
      gp, up = split(gu, [I, I])
      mid = silu(gp) * up                                 # [I]
      moe_out_j = down_proj[e] @ mid                      # [hidden]

  moe_out = sum(topk_w[j] * moe_out_j)                    # [hidden]

  # shared expert (always-on, smaller, gated by its own sigmoid)
  shared_gate = sigmoid(shared_expert_gate @ h_norm)      # [1] scalar
  shared_mid  = silu(shared_expert.gate_proj @ h_norm) \
              * (shared_expert.up_proj  @ h_norm)         # [Is = 512]
  shared_out  = shared_gate * (shared_expert.down_proj @ shared_mid)

  output_hidden = input_hidden + moe_out + shared_out

`input_hidden` here is the *post-attention residual* — i.e. the output
of the attention oracle, NOT the original input to the layer. The
kernel's MoE FFN entry takes the same buffer the attention kernel
wrote to.

Why hand-rolled rather than transformers: same reason as the attention
oracle — the multimodal HF class for this checkpoint has vision + MTP
heads we don't care about, and instantiating it costs ~70 GiB of host
RAM. We load just the per-layer FFN tensors.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Tensor I/O helpers (same encoding the attention oracle uses)
# ---------------------------------------------------------------------------
def b64_bf16(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().tobytes()
    ).decode()


def b64_f32(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.float32).contiguous().cpu().numpy().tobytes()
    ).decode()


def b64_i32(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.int32).contiguous().cpu().numpy().tobytes()
    ).decode()


def find_shard_for(model_dir: Path, name: str) -> Path:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wm = json.loads(idx_path.read_text())["weight_map"]
        if name not in wm:
            raise SystemExit(f"tensor not in index: {name}")
        return model_dir / wm[name]
    single = model_dir / "model.safetensors"
    if single.exists():
        return single
    raise SystemExit(f"no safetensors at {model_dir}")


def load_tensor(model_dir: Path, name: str, device: str = "cpu") -> torch.Tensor:
    path = find_shard_for(model_dir, name)
    with safe_open(str(path), framework="pt", device=device) as f:
        return f.get_tensor(name)


# ---------------------------------------------------------------------------
# Math primitives — kept structurally identical to what the kernel will do
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm without add-unit-offset. F32 internals, output in input dtype.
    Matches the attention oracle's `rms_norm` byte-for-byte so the kernel can
    reuse the same implementation across the attn + FFN halves of a layer."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return out.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU computed in F32 then cast back. The HIP kernel does the same;
    keeping this explicit matches the BF16 round-tripping behaviour
    (cast-up, op, cast-down with RNE) the kernel implements."""
    in_dtype = x.dtype
    return (x.to(torch.float32) * torch.sigmoid(x.to(torch.float32))).to(in_dtype)


# ---------------------------------------------------------------------------
# Per-block reference forward
# ---------------------------------------------------------------------------
def reference_moe_ffn_block(
    *,
    input_hidden: torch.Tensor,             # [hidden] — post-attention residual
    post_attn_norm_w: torch.Tensor,         # [hidden]
    gate_w: torch.Tensor,                   # [E=256, hidden]
    gate_up_proj_w: torch.Tensor,           # [E, 2*I=1024, hidden]   (gate || up)
    down_proj_w: torch.Tensor,              # [E, hidden, I=512]
    shared_gate_proj_w: torch.Tensor,       # [Is=512, hidden]
    shared_up_proj_w: torch.Tensor,         # [Is, hidden]
    shared_down_proj_w: torch.Tensor,       # [hidden, Is]
    shared_expert_gate_w: torch.Tensor,     # [1, hidden]
    num_experts: int,
    moe_intermediate: int,                  # I = 512 (per-expert, half of fused dim)
    shared_intermediate: int,               # Is = 512 (= moe_intermediate on 35B)
    top_k: int,
    rms_norm_eps: float,
) -> dict:
    """Apply one MoE FFN block for a single token.
    Returns a dict of every intermediate, all in the input dtype."""
    dtype = input_hidden.dtype
    hidden = input_hidden.shape[-1]
    E = num_experts
    I = moe_intermediate
    Is = shared_intermediate
    K = top_k

    # 1. Post-attention RMS norm.
    h_norm = rms_norm(input_hidden, post_attn_norm_w, rms_norm_eps)

    # 2. Router. Matmul + softmax + top-k. Score-renormalisation is the
    #    Qwen3-Next convention (renormalise across just the top-k probs so
    #    they sum to 1 within the dispatch); the kernel's stage 1 must
    #    reproduce both indices AND weights exactly.
    router_logits = (h_norm.to(torch.float32) @ gate_w.to(torch.float32).T)  # [E]
    router_probs = F.softmax(router_logits, dim=-1)                          # [E]
    topk_vals, topk_idx = torch.topk(router_probs, K, dim=-1)                # [K]
    topk_w_renorm = topk_vals / topk_vals.sum(dim=-1, keepdim=True)          # [K]

    # 3. Per-expert MLP. The fused weight stores `gate || up` along its row
    #    axis: rows [0..I) are the gate projection, rows [I..2I) are the up
    #    projection. The kernel reads the slice for each selected expert.
    per_expert_outs = []
    for j in range(K):
        e = int(topk_idx[j].item())
        gu = h_norm.to(torch.float32) @ gate_up_proj_w[e].to(torch.float32).T   # [2*I]
        gp = gu[:I]
        up = gu[I:]
        mid = (gp * torch.sigmoid(gp)) * up                                     # [I]
        eo = mid @ down_proj_w[e].to(torch.float32).T                           # [hidden]
        per_expert_outs.append(eo)
    expert_stack = torch.stack(per_expert_outs, dim=0)                          # [K, hidden]
    moe_out = (topk_w_renorm.unsqueeze(-1) * expert_stack).sum(dim=0).to(dtype) # [hidden]

    # 4. Shared expert. Always-on path that runs in parallel; its output is
    #    sigmoid-gated by `shared_expert_gate` (a single 1×hidden lane).
    sg_logit = (h_norm.to(torch.float32) @ shared_expert_gate_w.to(torch.float32).T)  # [1]
    sg_scalar = torch.sigmoid(sg_logit).squeeze(-1)                                   # scalar
    sgp = h_norm.to(torch.float32) @ shared_gate_proj_w.to(torch.float32).T           # [Is]
    sup = h_norm.to(torch.float32) @ shared_up_proj_w.to(torch.float32).T             # [Is]
    smid = (sgp * torch.sigmoid(sgp)) * sup                                            # [Is]
    sdown = smid @ shared_down_proj_w.to(torch.float32).T                              # [hidden]
    shared_out = (sg_scalar * sdown).to(dtype)                                         # [hidden]

    # 5. Residual sum. The MoE block's post-residual output replaces
    #    `input_hidden` for the next layer's input.
    output_hidden = (input_hidden.to(torch.float32) + moe_out.to(torch.float32)
                     + shared_out.to(torch.float32)).to(dtype)

    return {
        "h_norm":          h_norm,
        "router_logits":   router_logits.to(dtype),
        "router_probs":    router_probs.to(dtype),
        "topk_idx":        topk_idx.to(torch.int32),
        "topk_weights":    topk_w_renorm.to(dtype),
        "expert_stack":    expert_stack.to(dtype).reshape(K * hidden),
        "moe_out":         moe_out,
        "shared_gate":     sg_scalar.to(dtype),
        "shared_mid":      smid.to(dtype),
        "shared_out":      shared_out,
        "output_hidden":   output_hidden,
    }


# ---------------------------------------------------------------------------
# Driver: synthetic + from-checkpoint modes
# ---------------------------------------------------------------------------
def synthesize_block(
    *,
    hidden: int,
    num_experts: int,
    moe_intermediate: int,
    shared_intermediate: int,
    seed: int,
    dtype: torch.dtype,
) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        # ~N(0, 1/sqrt(fan_in)) so projection outputs stay O(1) — same
        # convention as the attention oracle.
        fan_in = shape[-1] if len(shape) >= 2 else 1
        scale = (1.0 / fan_in) ** 0.5
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    def norm_w(n: int) -> torch.Tensor:
        return (torch.ones(n, dtype=torch.float32)
                + torch.randn(n, generator=g, dtype=torch.float32) * 0.02).to(dtype)

    return dict(
        input_hidden=randn(hidden),
        post_attn_norm_w=norm_w(hidden),
        gate_w=randn(num_experts, hidden),
        gate_up_proj_w=randn(num_experts, 2 * moe_intermediate, hidden),
        down_proj_w=randn(num_experts, hidden, moe_intermediate),
        shared_gate_proj_w=randn(shared_intermediate, hidden),
        shared_up_proj_w=randn(shared_intermediate, hidden),
        shared_down_proj_w=randn(hidden, shared_intermediate),
        # The shared-expert sigmoid gate is one row × hidden; small values
        # keep sigmoid()<1 so the test catches a missing scalar multiply.
        shared_expert_gate_w=randn(1, hidden),
    )


def load_block_from_checkpoint(
    *, model_dir: Path, layer_idx: int, weight_prefix: str, device: str
) -> dict:
    lp = f"{weight_prefix}.layers.{layer_idx}"
    mp = f"{lp}.mlp"
    return dict(
        post_attn_norm_w=load_tensor(model_dir, f"{lp}.post_attention_layernorm.weight", device),
        gate_w=load_tensor(model_dir, f"{mp}.gate.weight", device),
        gate_up_proj_w=load_tensor(model_dir, f"{mp}.experts.gate_up_proj", device),
        down_proj_w=load_tensor(model_dir, f"{mp}.experts.down_proj", device),
        shared_gate_proj_w=load_tensor(model_dir, f"{mp}.shared_expert.gate_proj.weight", device),
        shared_up_proj_w=load_tensor(model_dir, f"{mp}.shared_expert.up_proj.weight", device),
        shared_down_proj_w=load_tensor(model_dir, f"{mp}.shared_expert.down_proj.weight", device),
        shared_expert_gate_w=load_tensor(model_dir, f"{mp}.shared_expert_gate.weight", device),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3.6-MoE single-block FFN reference (PR 4b4 oracle)"
    )
    p.add_argument("--mode", choices=["synthetic", "checkpoint"], default="synthetic")
    p.add_argument("--model-dir", type=Path,
                   help="Path to the HuggingFace safetensors dir (checkpoint mode)")
    p.add_argument("--layer-idx", type=int, default=0,
                   help="Layer index. The MoE FFN is identical across full and "
                        "linear attention layers, so layer 0 (linear) is fine.")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="Synthesis seed (also seeds the synthetic input_hidden in "
                        "checkpoint mode)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cpu",
                   help="Run on this device (cpu, cuda:0). Defaults to cpu so the "
                        "oracle is reproducible without a GPU. The 35B-A3B per-layer "
                        "MoE weights total ~5.4 GiB BF16; a CPU run on 64 GiB "
                        "host fits without trouble.")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON output path")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--moe-intermediate", type=int, default=512,
                   help="Per-expert intermediate dim (I in the docstring; half of "
                        "the fused gate_up_proj's row count)")
    p.add_argument("--shared-intermediate", type=int, default=512,
                   help="Shared-expert intermediate dim (= moe_intermediate on "
                        "35B-A3B but kept separate in case future configs differ)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.mode == "synthetic":
        weights = synthesize_block(
            hidden=args.hidden,
            num_experts=args.num_experts,
            moe_intermediate=args.moe_intermediate,
            shared_intermediate=args.shared_intermediate,
            seed=args.seed,
            dtype=dtype,
        )
    else:
        if args.model_dir is None:
            raise SystemExit("--model-dir is required in checkpoint mode")
        weights = load_block_from_checkpoint(
            model_dir=args.model_dir,
            layer_idx=args.layer_idx,
            weight_prefix=args.weight_prefix,
            device=args.device,
        )
        torch.manual_seed(args.seed)
        weights["input_hidden"] = (
            torch.randn(args.hidden, dtype=torch.float32) / args.hidden**0.5
        ).to(dtype)
        # Norm weights stored BF16 — cast to working dtype.
        weights["post_attn_norm_w"] = weights["post_attn_norm_w"].to(dtype)

    weights = {k: v.to(args.device) for k, v in weights.items()}

    intermediates = reference_moe_ffn_block(
        num_experts=args.num_experts,
        moe_intermediate=args.moe_intermediate,
        shared_intermediate=args.shared_intermediate,
        top_k=args.top_k,
        rms_norm_eps=args.rms_norm_eps,
        **weights,
    )

    encode = b64_bf16 if args.dtype == "bf16" else b64_f32

    # `topk_idx` is int32 regardless of the working dtype; everything else
    # follows the working dtype.
    enc_weights = {k: encode(v) for k, v in weights.items()}
    enc_intermediates: dict[str, str] = {}
    for k, v in intermediates.items():
        if k == "topk_idx":
            enc_intermediates[k] = b64_i32(v)
        else:
            enc_intermediates[k] = encode(v)

    out = {
        "schema": "qwen36-moe-oracle-ffn-v1",
        "mode": args.mode,
        "layer_idx": args.layer_idx,
        "dtype": args.dtype,
        "config": {
            "hidden": args.hidden,
            "num_experts": args.num_experts,
            "moe_intermediate": args.moe_intermediate,
            "shared_intermediate": args.shared_intermediate,
            "top_k": args.top_k,
            "rms_norm_eps": args.rms_norm_eps,
        },
        "weights": enc_weights,
        "intermediates": enc_intermediates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    out_norm = intermediates["output_hidden"].to(torch.float32).norm().item()
    in_norm = weights["input_hidden"].to(torch.float32).norm().item()
    delta = (intermediates["output_hidden"] - weights["input_hidden"]
             ).to(torch.float32).norm().item()
    selected = intermediates["topk_idx"].tolist()
    sys.stderr.write(
        f"[oracle-ffn] layer={args.layer_idx} mode={args.mode} "
        f"|input|={in_norm:.4f} |output|={out_norm:.4f} |delta|={delta:.4f} "
        f"top_k={selected}\n"
    )


if __name__ == "__main__":
    main()
