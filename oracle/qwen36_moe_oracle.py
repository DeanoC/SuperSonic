#!/usr/bin/env python3
"""
PyTorch reference for one Qwen3.6-MoE full-attention layer's decode step.

Companion to PR 4b of docs/qwen36-moe-plan.md. Loads exactly the weights
needed for one layer (no full-model instantiation — 35B doesn't fit
24 GiB), runs the forward in PyTorch using primitives we control end
to end, and emits every intermediate as JSON + base64. The HIP kernel's
parity test reads this JSON and compares its own intermediates point
for point.

Why hand-rolled rather than transformers: the published checkpoint is
a multimodal class (Qwen3_5MoeForConditionalGeneration) with vision +
MTP heads we don't care about. Pulling it in via AutoModelForCausalLM
would either flatten/strip silently or fail to dispatch, plus would
need ~70 GiB of RAM to instantiate. Loading just the tensors we need
keeps this script honest about what's being tested.

Math implemented (per Qwen3-Next full-attention with attn_output_gate):

  x_norm = rmsnorm(input, w_input_norm, eps)
  q_raw  = x_norm @ q_proj.T                       # [2 * H * d]  H=16, d=256
  q, gate = split(q_raw, axis=-1, [H*d, H*d])
  k      = x_norm @ k_proj.T                       # [Hkv * d]    Hkv=2
  v      = x_norm @ v_proj.T                       # [Hkv * d]
  q      = rmsnorm_per_head(q, w_q_norm)           # over last dim per head
  k      = rmsnorm_per_head(k, w_k_norm)
  q, k   = rope(q, k, position, rotary_dim=64)     # partial RoPE
  scores = q @ k.T / sqrt(d)                       # GQA: each Q head pairs
                                                   #      with floor(h / (H/Hkv))
  attn   = softmax(scores) @ v
  attn   = sigmoid(gate) * attn                    # output gate
  out    = attn @ o_proj.T
  return input + out

This is the prefill-position-zero variant: no prior KV cache, so the
attention reduces to a single-token "self-attention" against itself.
The kernel's first parity gate is on this exact path; KV-cache extension
to non-zero positions follows the same shape, just with a longer K/V
sequence.
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
# Tensor I/O helpers
# ---------------------------------------------------------------------------
def b64_bf16(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().tobytes()
    ).decode()


def b64_f32(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.float32).contiguous().cpu().numpy().tobytes()
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
    """RMS norm without add-unit-offset. Computed in F32, output in input dtype."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return out.to(in_dtype)


def build_rope_tables(
    rotary_dim: int, seq_len: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for partial RoPE. Returns `[seq_len, rotary_dim/2]` each
    in the requested dtype. The kernel reads these at the position index."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, half]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope_partial(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Rotate the first `rotary_dim` channels of `x[..., d]` using the
    half-pair convention: pair (i, i+half) for i in [0, half). Leaves
    channels [rotary_dim:d] untouched."""
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_pair_a = x_rot[..., :half]
    x_pair_b = x_rot[..., half:]
    rot_a = x_pair_a * cos - x_pair_b * sin
    rot_b = x_pair_b * cos + x_pair_a * sin
    return torch.cat([rot_a, rot_b, x_pass], dim=-1)


def gqa_attention(
    q: torch.Tensor,  # [H, d]
    k: torch.Tensor,  # [Hkv, d]
    v: torch.Tensor,  # [Hkv, d]
    head_dim: int,
) -> torch.Tensor:
    """Single-token self-attention with GQA broadcast. Returns [H, d]."""
    H = q.shape[0]
    Hkv = k.shape[0]
    assert H % Hkv == 0, f"GQA requires H ({H}) % Hkv ({Hkv}) == 0"
    rep = H // Hkv
    k_full = k.repeat_interleave(rep, dim=0)  # [H, d]
    v_full = v.repeat_interleave(rep, dim=0)  # [H, d]

    # Query length 1 against key length 1: scores per head are scalar.
    scale = 1.0 / (head_dim**0.5)
    scores = (q * k_full).sum(dim=-1, keepdim=True) * scale  # [H, 1]
    weights = F.softmax(scores.to(torch.float32), dim=-1).to(v_full.dtype)
    return weights * v_full  # [H, d]


# ---------------------------------------------------------------------------
# Per-layer reference forward
# ---------------------------------------------------------------------------
def reference_full_attention_layer(
    *,
    input_hidden: torch.Tensor,        # [hidden]
    input_norm_w: torch.Tensor,        # [hidden]
    q_proj_w: torch.Tensor,            # [2*H*d, hidden]   (attn_output_gate=true)
    k_proj_w: torch.Tensor,            # [Hkv*d, hidden]
    v_proj_w: torch.Tensor,            # [Hkv*d, hidden]
    q_norm_w: torch.Tensor,            # [d]
    k_norm_w: torch.Tensor,            # [d]
    o_proj_w: torch.Tensor,            # [hidden, H*d]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
    position: int,
) -> dict:
    """Apply one full-attention layer for a single token at `position`.
    Returns a dict of every intermediate, all in the input dtype."""
    dtype = input_hidden.dtype
    hidden = input_hidden.shape[-1]
    H = num_heads
    Hkv = num_kv_heads
    d = head_dim

    # 1. Pre-attention RMS norm.
    x_norm = rms_norm(input_hidden, input_norm_w, rms_norm_eps)

    # 2. Q/K/V projections. Q's output has 2*H*d channels (Q + output gate).
    q_raw = x_norm @ q_proj_w.T  # [2*H*d]
    k_raw = x_norm @ k_proj_w.T  # [Hkv*d]
    v_raw = x_norm @ v_proj_w.T  # [Hkv*d]

    # 3. Split Q's gate off and reshape per-head.
    q_lanes = q_raw[: H * d]
    out_gate_lanes = q_raw[H * d :]
    q_heads = q_lanes.reshape(H, d)            # [H, d]
    k_heads = k_raw.reshape(Hkv, d)            # [Hkv, d]
    v_heads = v_raw.reshape(Hkv, d)            # [Hkv, d]
    out_gate = out_gate_lanes.reshape(H, d)    # [H, d]

    # 4. Per-head Q/K RMS norm.
    q_normed = rms_norm(q_heads, q_norm_w, rms_norm_eps)
    k_normed = rms_norm(k_heads, k_norm_w, rms_norm_eps)

    # 5. Partial RoPE on Q and K.
    seq_len = max(position + 1, 1)
    cos_table, sin_table = build_rope_tables(rotary_dim, seq_len, rope_theta, dtype)
    cos_pos = cos_table[position].to(dtype)  # [rotary_dim/2]
    sin_pos = sin_table[position].to(dtype)
    q_rot = apply_rope_partial(q_normed, cos_pos, sin_pos, rotary_dim)
    k_rot = apply_rope_partial(k_normed, cos_pos, sin_pos, rotary_dim)

    # 6. Self-attention against the just-emitted (k, v) (KV cache len = 1).
    attn = gqa_attention(q_rot, k_rot, v_heads, d)  # [H, d]

    # 7. Output gate (Qwen3-Next innovation).
    gate = torch.sigmoid(out_gate.to(torch.float32)).to(dtype) * attn

    # 8. O projection + residual.
    gate_flat = gate.reshape(H * d)
    o_out = gate_flat @ o_proj_w.T   # [hidden]
    output_hidden = input_hidden + o_out

    return {
        "x_norm":         x_norm,
        "q_raw":          q_raw,
        "k_raw":          k_raw,
        "v_raw":          v_raw,
        "q_lanes":        q_lanes,
        "out_gate_lanes": out_gate_lanes,
        "q_normed":       q_normed.reshape(H * d),
        "k_normed":       k_normed.reshape(Hkv * d),
        "q_rot":          q_rot.reshape(H * d),
        "k_rot":          k_rot.reshape(Hkv * d),
        "attn":           attn.reshape(H * d),
        "gated_attn":     gate_flat,
        "o_out":          o_out,
        "output_hidden":  output_hidden,
    }


# ---------------------------------------------------------------------------
# Driver: synthetic + from-checkpoint modes
# ---------------------------------------------------------------------------
def synthesize_layer(
    *, hidden: int, H: int, Hkv: int, d: int, seed: int, dtype: torch.dtype
) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        # ~N(0, 1/sqrt(fan_in)) so projection outputs stay O(1) — keeps
        # the synthetic forward numerically tame for parity comparison.
        fan_in = shape[-1] if len(shape) >= 2 else 1
        scale = (1.0 / fan_in) ** 0.5
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    # Norm weights at ~1.0 plus a small jitter so an off-by-one in the unit
    # offset would be visible.
    def norm_w(n: int) -> torch.Tensor:
        return (torch.ones(n, dtype=torch.float32)
                + torch.randn(n, generator=g, dtype=torch.float32) * 0.02).to(dtype)

    return dict(
        input_hidden=randn(hidden),
        input_norm_w=norm_w(hidden),
        q_proj_w=randn(2 * H * d, hidden),
        k_proj_w=randn(Hkv * d, hidden),
        v_proj_w=randn(Hkv * d, hidden),
        q_norm_w=norm_w(d),
        k_norm_w=norm_w(d),
        o_proj_w=randn(hidden, H * d),
    )


def load_layer_from_checkpoint(
    *, model_dir: Path, layer_idx: int, weight_prefix: str, device: str
) -> dict:
    lp = f"{weight_prefix}.layers.{layer_idx}"
    fa = f"{lp}.self_attn"
    return dict(
        input_norm_w=load_tensor(model_dir, f"{lp}.input_layernorm.weight", device),
        q_proj_w=load_tensor(model_dir, f"{fa}.q_proj.weight", device),
        k_proj_w=load_tensor(model_dir, f"{fa}.k_proj.weight", device),
        v_proj_w=load_tensor(model_dir, f"{fa}.v_proj.weight", device),
        q_norm_w=load_tensor(model_dir, f"{fa}.q_norm.weight", device),
        k_norm_w=load_tensor(model_dir, f"{fa}.k_norm.weight", device),
        o_proj_w=load_tensor(model_dir, f"{fa}.o_proj.weight", device),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3.6-MoE single-layer reference forward (PR 4b oracle)"
    )
    p.add_argument("--mode", choices=["synthetic", "checkpoint"], default="synthetic")
    p.add_argument("--model-dir", type=Path,
                   help="Path to the HuggingFace safetensors dir (checkpoint mode)")
    p.add_argument("--layer-idx", type=int, default=3,
                   help="Full-attention layer index (default 3 = first full layer)")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--position", type=int, default=0)
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="Synthesis seed for `--mode synthetic`")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cpu",
                   help="Run on this device (cpu, cuda:0). Defaults to cpu so the "
                        "oracle is reproducible without a GPU; `cuda:0` here = "
                        "ROCm GPU on this machine via PyTorch's HIP wheel.")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON output path")
    p.add_argument("--num-attention-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--rotary-dim", type=int, default=64)
    p.add_argument("--rope-theta", type=float, default=1e7)
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.mode == "synthetic":
        weights = synthesize_layer(
            hidden=args.hidden,
            H=args.num_attention_heads,
            Hkv=args.num_kv_heads,
            d=args.head_dim,
            seed=args.seed,
            dtype=dtype,
        )
    else:
        if args.model_dir is None:
            raise SystemExit("--model-dir is required in checkpoint mode")
        weights = load_layer_from_checkpoint(
            model_dir=args.model_dir,
            layer_idx=args.layer_idx,
            weight_prefix=args.weight_prefix,
            device=args.device,
        )
        # Embed_tokens isn't needed; we synthesise the input_hidden the same
        # way as synthetic mode so the parity test has a deterministic input
        # without paying ~1 GiB to load the embedding table.
        torch.manual_seed(args.seed)
        weights["input_hidden"] = (
            torch.randn(args.hidden, dtype=torch.float32) / args.hidden**0.5
        ).to(dtype)
        # Norm weights from 35B-A3B are stored BF16; cast to working dtype.
        for k in ("input_norm_w", "q_norm_w", "k_norm_w"):
            weights[k] = weights[k].to(dtype)

    weights = {k: v.to(args.device) for k, v in weights.items()}

    intermediates = reference_full_attention_layer(
        num_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        position=args.position,
        **weights,
    )

    encode = b64_bf16 if args.dtype == "bf16" else b64_f32
    out = {
        "schema": "qwen36-moe-oracle-layer-v1",
        "mode": args.mode,
        "layer_idx": args.layer_idx,
        "position": args.position,
        "dtype": args.dtype,
        "config": {
            "hidden": args.hidden,
            "num_attention_heads": args.num_attention_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "rotary_dim": args.rotary_dim,
            "rope_theta": args.rope_theta,
            "rms_norm_eps": args.rms_norm_eps,
            "attn_output_gate": True,
        },
        "weights": {k: encode(v) for k, v in weights.items()},
        "intermediates": {k: encode(v) for k, v in intermediates.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    # Quick eyeball at stderr — gives an at-a-glance "is this remotely sane".
    out_norm = intermediates["output_hidden"].to(torch.float32).norm().item()
    in_norm = weights["input_hidden"].to(torch.float32).norm().item()
    sys.stderr.write(
        f"[oracle] layer={args.layer_idx} pos={args.position} mode={args.mode} "
        f"|input|={in_norm:.4f} |output|={out_norm:.4f} "
        f"|delta|={(intermediates['output_hidden'] - weights['input_hidden']).to(torch.float32).norm().item():.4f}\n"
    )


if __name__ == "__main__":
    main()
