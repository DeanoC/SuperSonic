#!/usr/bin/env python3
"""
PyTorch reference for one Qwen3.6-MoE full-attention layer's decode step.

Companion to PR 4b (BF16) and PR 4b6 (INT4) of docs/qwen36-moe-plan.md.
Loads exactly the weights needed for one layer (no full-model
instantiation — 35B doesn't fit 24 GiB), runs the forward in PyTorch
using primitives we control end to end, and emits every intermediate
as JSON + base64. The HIP kernel's parity test reads this JSON and
compares its own intermediates point for point.

`--int4` switches into INT4 mode: the four projection weights that the
INT4 bake quantizes (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are
min/max group-quantized at gs=128 with BF16 scale + zero. The `weights`
block then carries the BF16-rounded *reconstruction* of those tensors
(so the existing BF16 kernel path is still exercised) and a parallel
`int4_weights` block carries the packed nibbles + scale + zero sidecars
the INT4 kernel path will read. Schema becomes
`qwen36-moe-oracle-layer-int4-v1`. Norms (input/q/k) stay BF16 — the
bake excludes them.

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


def b64_u8(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.to(torch.uint8).contiguous().cpu().numpy().tobytes()
    ).decode()


# Tensors the INT4 bake quantizes for one full-attention layer. The norm
# weights (input_norm, q_norm, k_norm) stay BF16 — the bake excludes them
# from the INT4 budget for the same reason `crates/qwen36_moe/src/weights.rs`
# does.
INT4_ATTN_TARGETS: tuple[str, ...] = (
    "q_proj_w",
    "k_proj_w",
    "v_proj_w",
    "o_proj_w",
)


@torch.no_grad()
def minmax_int4_packed_and_recon(
    W: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min/max INT4 group-quant for a 2D `[out, in]` weight. Mirrors the
    `minmax_int4_packed_and_recon` in `qwen36_moe_ffn_oracle.py` exactly,
    which in turn mirrors `oracle/bake_int4.py`.

    Returns (packed/scale/zero on CPU; recon stays on `W.device`):

      packed   uint8  shape `[out, in/2]`     two nibbles per byte,
                                               even col → low nibble.
      scale    f32    shape `[out/gs, in/gs]` BF16 values stored as f32.
      zero     f32    shape `[out/gs, in/gs]` BF16 values stored as f32.
      recon    bf16   shape `[out, in]`        single-rounding
                                               `bf16(q*s - z*s)` —
                                               matches the kernel's
                                               `bf16_round_rne_f32_finite(
                                                  n*s - zs)` exactly.
    """
    if W.dim() != 2:
        raise ValueError(f"expected 2D, got shape {tuple(W.shape)}")
    out_f, in_f = W.shape
    gs = group_size
    if in_f % gs != 0 or in_f % 2 != 0:
        raise ValueError(
            f"in_features {in_f} must be divisible by group_size={gs} and even"
        )
    if out_f % gs != 0:
        raise ValueError(
            f"out_features {out_f} must be divisible by group_size={gs}"
        )
    sr = out_f // gs
    sc = in_f // gs

    slab = W.to(torch.float32)
    tiles = slab.reshape(sr, gs, sc, gs)
    tmax = tiles.amax(dim=(1, 3))
    tmin = tiles.amin(dim=(1, 3))
    rng = tmax - tmin
    s = torch.where(rng > 0, rng / 15.0, torch.ones_like(rng))
    z = torch.where(rng > 0, -tmin / s, torch.zeros_like(rng))
    # Round through BF16 — sidecars are stored BF16 in the bake and the
    # kernel reads BF16 scale/zero.
    s = s.to(torch.bfloat16).to(torch.float32)
    z = z.to(torch.bfloat16).to(torch.float32)
    s_full = s.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    z_full = z.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    q = torch.clamp(
        torch.round(slab / s_full + z_full), 0.0, 15.0
    ).to(torch.uint8)
    # Single-rounding reconstruction: bf16(q*s - z*s). The kernel computes
    # `bf16(n*s - zs)` with `zs = z*s` precomputed in F32, so the rounding
    # boundary is identical.
    recon = (q.to(torch.float32) * s_full - z_full * s_full).to(torch.bfloat16)
    # Pack 2 nibbles/byte: even col → low, odd col → high.
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed.cpu(), s.cpu(), z.cpu(), recon


@torch.no_grad()
def dequant_int4_packed(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference dequant — inverse of `minmax_int4_packed_and_recon`. Used
    for the in-process self-check below."""
    out_f = packed.shape[-2]
    in_half = packed.shape[-1]
    in_f = in_half * 2
    gs = group_size
    q = torch.empty((out_f, in_f), dtype=torch.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = (packed >> 4) & 0x0F
    s_full = scale.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    z_full = zero.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    recon = (q.to(torch.float32) * s_full - z_full * s_full).to(torch.bfloat16)
    return recon.to(torch.float32)


def quantize_int4_attn_weights(
    weights: dict[str, torch.Tensor], group_size: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Quantize the INT4-targeted attn weights in-place: replace each entry
    in `weights` with its BF16 reconstruction (cast to that weight's
    original dtype + device). Return a parallel dict of
    `{name: {"packed", "scale", "zero"}}` sidecars on CPU."""
    int4_sidecars: dict[str, dict[str, torch.Tensor]] = {}
    for name in INT4_ATTN_TARGETS:
        if name not in weights:
            raise SystemExit(f"INT4 mode missing required weight: {name}")
        W = weights[name]
        packed, scale, zero, recon = minmax_int4_packed_and_recon(W, group_size)

        # In-process self-check: verify (packed, scale, zero) round-trips
        # to exactly `recon`. Catches packing bugs before the kernel ever
        # sees the JSON.
        recon_check = dequant_int4_packed(packed, scale, zero, group_size)
        diff = (recon.to(torch.float32).cpu() - recon_check).abs().max().item()
        if diff != 0.0:
            raise RuntimeError(
                f"INT4 self-check failed for {name}: "
                f"max |recon - dequant(packed)| = {diff:.3e}"
            )

        int4_sidecars[name] = {"packed": packed, "scale": scale, "zero": zero}
        weights[name] = recon.to(dtype=W.dtype, device=W.device)
    return int4_sidecars


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
    p.add_argument("--mode", choices=["synthetic", "checkpoint", "bake"], default="synthetic")
    p.add_argument("--model-dir", type=Path,
                   help="Path to the HuggingFace safetensors dir (checkpoint mode)")
    p.add_argument("--bake-dir", type=Path,
                   help="Path to a SuperSonic INT4 GPTQ bake directory "
                        "(`bake` mode). Loads layer's INT4 weights directly + "
                        "reconstructs to BF16 via the kernel's exact "
                        "`bf16(q*s - z*s)` formula. Schema becomes "
                        "`qwen36-moe-oracle-layer-int4-v1` with `int4_weights` "
                        "carrying the bake's *actual* packed/scale/zero — the "
                        "harness for verifying the kernel produces the right "
                        "output on the bake's specific INT4 patterns.")
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
    p.add_argument("--int4", action="store_true",
                   help="Quantize the four projection weights (q/k/v/o_proj) "
                        "to INT4 (min/max group-quant). The `weights` block "
                        "carries the BF16 reconstruction; an additional "
                        "`int4_weights` block carries (packed, scale, zero) "
                        "sidecars for the kernel's INT4 path. Schema becomes "
                        "`qwen36-moe-oracle-layer-int4-v1`.")
    p.add_argument("--int4-group-size", type=int, default=128,
                   help="Group size for INT4 min/max quant. Must divide "
                        "out_features and in_features of every quantized "
                        "tensor. The runtime + bake both pin to 128.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    int4_sidecars: dict[str, dict[str, torch.Tensor]] | None = None

    if args.mode == "synthetic":
        weights = synthesize_layer(
            hidden=args.hidden,
            H=args.num_attention_heads,
            Hkv=args.num_kv_heads,
            d=args.head_dim,
            seed=args.seed,
            dtype=dtype,
        )
    elif args.mode == "bake":
        if args.bake_dir is None:
            raise SystemExit("--bake-dir is required in bake mode")
        # Bake mode: load INT4 weights from the SuperSonic bake directly, use
        # the bake's BF16 reconstructions as the "weights" for the reference
        # forward, and emit the bake's actual (packed, scale, zero) sidecars
        # so the kernel-ffi parity test verifies the kernel reproduces what
        # Python computes on these specific INT4 patterns.
        from _bake_loader import load_bf16, load_int4

        bake_dir = str(args.bake_dir)
        lp = f"{args.weight_prefix}.layers.{args.layer_idx}"
        fa = f"{lp}.self_attn"

        weights = {}
        int4_sidecars = {}
        for key, name in [
            ("q_proj_w", f"{fa}.q_proj.weight"),
            ("k_proj_w", f"{fa}.k_proj.weight"),
            ("v_proj_w", f"{fa}.v_proj.weight"),
            ("o_proj_w", f"{fa}.o_proj.weight"),
        ]:
            recon, packed, scale, zero = load_int4(bake_dir, name)
            weights[key] = recon
            int4_sidecars[key] = {
                "packed": packed,
                "scale": scale.to(torch.float32),
                "zero": zero.to(torch.float32),
            }
        weights["input_norm_w"] = load_bf16(bake_dir, f"{lp}.input_layernorm.weight")
        weights["q_norm_w"] = load_bf16(bake_dir, f"{fa}.q_norm.weight")
        weights["k_norm_w"] = load_bf16(bake_dir, f"{fa}.k_norm.weight")

        # Synthesise input_hidden the same way checkpoint mode does so the
        # parity test has a deterministic input without loading embed_tokens.
        torch.manual_seed(args.seed)
        weights["input_hidden"] = (
            torch.randn(args.hidden, dtype=torch.float32) / args.hidden**0.5
        ).to(dtype)
        for k in ("input_norm_w", "q_norm_w", "k_norm_w"):
            weights[k] = weights[k].to(dtype)

        # Force the INT4 path so the schema/JSON shape matches what the
        # kernel-ffi INT4 parity tests expect.
        args.int4 = True
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

    if args.int4 and int4_sidecars is None:
        # Quantize before computing intermediates so the reference uses the
        # same BF16-reconstructed weights the kernel will see — intermediates
        # are then valid for both the BF16 and INT4 kernel paths.
        # (Skipped when bake mode already populated int4_sidecars from disk.)
        int4_sidecars = quantize_int4_attn_weights(weights, args.int4_group_size)

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
    config = {
        "hidden": args.hidden,
        "num_attention_heads": args.num_attention_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "rotary_dim": args.rotary_dim,
        "rope_theta": args.rope_theta,
        "rms_norm_eps": args.rms_norm_eps,
        "attn_output_gate": True,
    }
    if args.int4:
        config["int4_group_size"] = args.int4_group_size

    out = {
        "schema": ("qwen36-moe-oracle-layer-int4-v1"
                   if args.int4 else "qwen36-moe-oracle-layer-v1"),
        "mode": args.mode,
        "layer_idx": args.layer_idx,
        "position": args.position,
        "dtype": args.dtype,
        "config": config,
        "weights": {k: encode(v) for k, v in weights.items()},
        "intermediates": {k: encode(v) for k, v in intermediates.items()},
    }
    if int4_sidecars is not None:
        out["int4_weights"] = {
            name: {
                "packed": b64_u8(t["packed"]),
                "scale": b64_bf16(t["scale"]),
                "zero": b64_bf16(t["zero"]),
            }
            for name, t in int4_sidecars.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    # Quick eyeball at stderr — gives an at-a-glance "is this remotely sane".
    out_norm = intermediates["output_hidden"].to(torch.float32).norm().item()
    in_norm = weights["input_hidden"].to(torch.float32).norm().item()
    int4_tag = "int4" if args.int4 else "bf16-w"
    sys.stderr.write(
        f"[oracle] layer={args.layer_idx} pos={args.position} mode={args.mode} "
        f"weights={int4_tag} "
        f"|input|={in_norm:.4f} |output|={out_norm:.4f} "
        f"|delta|={(intermediates['output_hidden'] - weights['input_hidden']).to(torch.float32).norm().item():.4f}\n"
    )


if __name__ == "__main__":
    main()
