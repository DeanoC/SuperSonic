#!/usr/bin/env python3
"""
PyTorch reference for one Qwen3.6-MoE linear-attention layer's decode step.

Companion to PR 4b3 step 1 (BF16) and PR 4b6 step 3 (INT4). The 3-of-4
layers in the hybrid pattern that aren't full-attention are
linear-attention (delta-rule recurrent state + depthwise conv pre-mix),
and they need their own staged kernel. This oracle is the parity
ground-truth for that kernel.

`--int4` switches into INT4 mode: the three projection weights that the
INT4 bake quantizes (`in_proj_qkv`, `in_proj_z`, `out_proj`) are
min/max group-quantized at gs=128 with BF16 scale + zero. `in_proj_a`
and `in_proj_b` (small per-V-head scalars) plus the conv1d, dt_bias,
A_log, norms and state buffers all stay BF16 — the bake excludes them
from the INT4 budget; see `crates/qwen36_moe/src/weights.rs::lin_int4`.

Schema becomes `qwen36-moe-oracle-linear-int4-v1` when --int4 is set.

As with `qwen36_moe_oracle.py`, the math is hand-rolled to avoid pulling
in the multimodal `Qwen3_5MoeForConditionalGeneration` class — it would
need ~70 GiB of RAM to instantiate and we only care about one layer's
worth of tensors.

Math implemented (single decode step, all per qwen35 production code):

  x_norm     = rmsnorm(input_hidden, w_input_norm, eps)

  qkv_raw    = x_norm @ in_proj_qkv.T              # [qkv_dim]
  z_raw      = x_norm @ in_proj_z.T                # [V*v]
  a_raw      = x_norm @ in_proj_a.T                # [V]
  b_raw      = x_norm @ in_proj_b.T                # [V]

  conv_input = concat(conv_state_before, qkv_raw)  # [qkv_dim, kernel]
  conv_out   = depthwise_conv1d(conv_input, conv_w[, conv_b])  # [qkv_dim]
  silu_out   = silu(conv_out)
  conv_state_after = conv_input[:, 1:]             # shifted

  q_raw, k_raw, v_raw = split(silu_out)            # along channel
  q          = l2_normalize(q_raw.reshape(K, k))
  k          = l2_normalize(k_raw.reshape(K, k))
  v          =                v_raw.reshape(V, v)

  rep        = V // K
  q_rep      = q.repeat_interleave(rep, dim=0)     # [V, k]
  k_rep      = k.repeat_interleave(rep, dim=0)
  q_scaled   = q_rep * (1 / sqrt(k))

  beta       = sigmoid(b_raw)                      # [V]
  g          = -softplus(a_raw + dt_bias) * exp(A_log)   # [V]   (F32)

  state      = recurrent_state_before * exp(g)[:, None, None]
  kv_mem     = sum_k(state * k_rep[..., None])     # [V, v]
  delta      = (v - kv_mem) * beta[..., None]
  state_aft  = state + k_rep[..., None] * delta[:, None, :]
  rec_out    = sum_k(state_aft * q_scaled[..., None])    # [V, v]

  out_normed = rmsnorm_per_head(rec_out, norm_w, eps)
  z_silu     = silu(z_raw.reshape(V, v))
  out_gated  = out_normed * z_silu

  o_out      = out_gated.reshape(V*v) @ out_proj.T
  output_hidden = input_hidden + o_out

Where K = num_k_heads, V = num_v_heads, k = head_k_dim, v = head_v_dim,
qkv_dim = 2*K*k + V*v.

The first decode step from a fresh prefill has conv_state_before and
recurrent_state_before both zero. To exercise the recurrent update
properly the oracle also supports a "warm" mode where prior state is
seeded from the same RNG that produces the synthetic weights.
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
# Tensor I/O helpers (identical to qwen36_moe_oracle.py)
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


# Tensors the INT4 bake quantizes for one linear-attention layer. The
# small per-V-head scalars `in_proj_a` / `in_proj_b` plus all the
# small/conv/state tensors stay BF16 — the bake excludes them. See
# `crates/qwen36_moe/src/weights.rs::lin_int4` for the matching budget.
INT4_LINEAR_TARGETS: tuple[str, ...] = (
    "in_proj_qkv_w",
    "in_proj_z_w",
    "out_proj_w",
)


@torch.no_grad()
def minmax_int4_packed_and_recon(
    W: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min/max INT4 group-quant for a 2D `[out, in]` weight. Mirrors the
    helpers in `qwen36_moe_oracle.py` and `qwen36_moe_ffn_oracle.py`,
    which in turn mirror `oracle/bake_int4.py`. Single-rounding
    `bf16(q*s - z*s)` reconstruction matches the kernel's
    `int4_dequant_scalar`.

    Returns (packed/scale/zero on CPU; recon stays on `W.device`):
      packed   uint8  shape `[out, in/2]`     two nibbles per byte,
                                               even col → low nibble.
      scale    f32    shape `[out/gs, in/gs]` BF16 values stored as f32.
      zero     f32    shape `[out/gs, in/gs]` BF16 values stored as f32.
      recon    bf16   shape `[out, in]`        kernel's reconstruction.
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
    s = s.to(torch.bfloat16).to(torch.float32)
    z = z.to(torch.bfloat16).to(torch.float32)
    s_full = s.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    z_full = z.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
    q = torch.clamp(
        torch.round(slab / s_full + z_full), 0.0, 15.0
    ).to(torch.uint8)
    recon = (q.to(torch.float32) * s_full - z_full * s_full).to(torch.bfloat16)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed.cpu(), s.cpu(), z.cpu(), recon


@torch.no_grad()
def dequant_int4_packed(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference dequant — inverse of `minmax_int4_packed_and_recon`."""
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


def quantize_int4_linear_weights(
    weights: dict[str, torch.Tensor], group_size: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Quantize the INT4-targeted linear-attn weights in-place: replace
    each entry in `weights` with its BF16 reconstruction (cast to the
    weight's original dtype + device). Return a parallel dict of
    `{name: {"packed", "scale", "zero"}}` sidecars on CPU."""
    int4_sidecars: dict[str, dict[str, torch.Tensor]] = {}
    for name in INT4_LINEAR_TARGETS:
        if name not in weights:
            raise SystemExit(f"INT4 mode missing required weight: {name}")
        W = weights[name]
        packed, scale, zero, recon = minmax_int4_packed_and_recon(W, group_size)

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
# Math primitives
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm without add-unit-offset. Computed in F32, output in input dtype."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return out.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU = x * sigmoid(x), computed in F32 then cast back."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    return (xf * torch.sigmoid(xf)).to(in_dtype)


def l2_normalize_per_head(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """F.normalize(p=2, dim=-1, eps=eps) — divides by max(norm, eps)."""
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# Per-layer reference forward (single decode step)
# ---------------------------------------------------------------------------
def reference_linear_attn_layer(
    *,
    input_hidden: torch.Tensor,            # [hidden]
    input_norm_w: torch.Tensor,            # [hidden]
    in_proj_qkv_w: torch.Tensor,           # [qkv_dim, hidden]
    in_proj_z_w: torch.Tensor,             # [V*v, hidden]
    in_proj_a_w: torch.Tensor,             # [V, hidden]
    in_proj_b_w: torch.Tensor,             # [V, hidden]
    conv1d_w: torch.Tensor,                # [qkv_dim, 1, kernel]   (depthwise)
    conv1d_bias: torch.Tensor | None,      # [qkv_dim] or None
    dt_bias: torch.Tensor,                 # [V]
    a_log: torch.Tensor,                   # [V]
    norm_w: torch.Tensor,                  # [v]   per-head normalization weight
    out_proj_w: torch.Tensor,              # [hidden, V*v]
    conv_state_before: torch.Tensor,       # [qkv_dim, kernel - 1]
    recurrent_state_before: torch.Tensor,  # [V, k, v]   F32
    num_k_heads: int,                      # K
    num_v_heads: int,                      # V
    head_k_dim: int,                       # k
    head_v_dim: int,                       # v
    conv_kernel_dim: int,
    rms_norm_eps: float,
) -> dict:
    dtype = input_hidden.dtype
    K = num_k_heads
    V = num_v_heads
    k_dim = head_k_dim
    v_dim = head_v_dim
    key_dim = K * k_dim
    value_dim = V * v_dim
    qkv_dim = 2 * key_dim + value_dim
    assert in_proj_qkv_w.shape[0] == qkv_dim, (
        f"in_proj_qkv_w out_dim {in_proj_qkv_w.shape[0]} != expected {qkv_dim}"
    )

    # 1. Pre-attn RMS norm.
    x_norm = rms_norm(input_hidden, input_norm_w, rms_norm_eps)

    # 2. In-projections (qkv, z, a, b).
    qkv_raw = x_norm @ in_proj_qkv_w.T                    # [qkv_dim]
    z_raw = x_norm @ in_proj_z_w.T                        # [V*v]
    a_raw = x_norm @ in_proj_a_w.T                        # [V]
    b_raw = x_norm @ in_proj_b_w.T                        # [V]

    # 3. Conv1d step.
    # conv_state_before holds the prior `kernel-1` channels per
    # qkv_dim. Append the new qkv_raw column to get a `[qkv_dim, kernel]`
    # window, then sum element-wise against conv1d_w (depthwise: one
    # filter per channel, no cross-channel mixing).
    conv_input = torch.cat(
        [conv_state_before.to(dtype), qkv_raw.unsqueeze(-1)], dim=-1
    )                                                      # [qkv_dim, kernel]
    conv_w_2d = conv1d_w.to(dtype).squeeze(1)              # [qkv_dim, kernel]
    conv_out_f32 = (conv_input.to(torch.float32) * conv_w_2d.to(torch.float32)).sum(dim=-1)
    if conv1d_bias is not None:
        conv_out_f32 = conv_out_f32 + conv1d_bias.to(torch.float32)
    conv_out = conv_out_f32.to(dtype)                      # [qkv_dim]
    silu_out = silu(conv_out)                              # [qkv_dim]
    conv_state_after = conv_input[:, 1:]                   # [qkv_dim, kernel-1]

    # 4. Split q, k, v out of silu_out.
    q_raw = silu_out[:key_dim]
    k_raw = silu_out[key_dim:2 * key_dim]
    v_raw = silu_out[2 * key_dim:2 * key_dim + value_dim]
    q_heads = q_raw.reshape(K, k_dim)
    k_heads = k_raw.reshape(K, k_dim)
    v_heads = v_raw.reshape(V, v_dim)

    # 5. L2-normalize Q and K per head.
    q_normed = l2_normalize_per_head(q_heads, eps=1e-6)
    k_normed = l2_normalize_per_head(k_heads, eps=1e-6)

    # 6. GQA fan-out (linear attn replicates K-heads to match V-heads).
    head_repeat = V // K
    if head_repeat > 1:
        q_rep = q_normed.repeat_interleave(head_repeat, dim=0)   # [V, k_dim]
        k_rep = k_normed.repeat_interleave(head_repeat, dim=0)
    else:
        q_rep = q_normed
        k_rep = k_normed

    # 7. Scale Q.
    q_scaled = q_rep * (1.0 / (k_dim ** 0.5))

    # 8. Beta gate and decay g (F32 throughout — matches qwen35 oracle).
    beta = torch.sigmoid(b_raw.to(torch.float32))                  # [V]
    dt_bias_f = dt_bias.to(torch.float32)
    a_log_exp = a_log.to(torch.float32).exp()
    g = -torch.log1p(torch.exp(a_raw.to(torch.float32) + dt_bias_f)) * a_log_exp   # [V]

    # 9. Delta-rule recurrent update (single step).
    # state shape: [V, k_dim, v_dim]
    state = recurrent_state_before.to(torch.float32).clone()
    g_step = g.exp()                                                # [V]
    state = state * g_step.unsqueeze(-1).unsqueeze(-1)              # decay
    k_rep_f = k_rep.to(torch.float32)
    q_scaled_f = q_scaled.to(torch.float32)
    v_heads_f = v_heads.to(torch.float32)
    kv_mem = (state * k_rep_f.unsqueeze(-1)).sum(dim=1)             # [V, v_dim]
    delta = (v_heads_f - kv_mem) * beta.unsqueeze(-1)               # [V, v_dim]
    state_after = state + k_rep_f.unsqueeze(-1) * delta.unsqueeze(1)
    recurrent_out = (state_after * q_scaled_f.unsqueeze(-1)).sum(dim=1)   # [V, v_dim]

    # 10. Per-head RMS norm of recurrent output.
    rec_out_dtype = recurrent_out.to(dtype)
    out_normed = rms_norm(rec_out_dtype, norm_w, rms_norm_eps)      # [V, v_dim]

    # 11. Z gating (silu(z) elementwise multiply).
    z_heads = z_raw.reshape(V, v_dim)
    z_silu = silu(z_heads)
    out_gated = out_normed * z_silu                                 # [V, v_dim]

    # 12. Out projection + residual.
    out_flat = out_gated.reshape(V * v_dim)
    o_out = out_flat @ out_proj_w.T                                 # [hidden]
    output_hidden = input_hidden + o_out

    return {
        "x_norm":                x_norm,
        "qkv_raw":               qkv_raw,
        "z_raw":                 z_raw,
        "a_raw":                 a_raw,
        "b_raw":                 b_raw,
        "conv_input":            conv_input.reshape(-1),
        "conv_out":              conv_out,
        "silu_out":              silu_out,
        "q_raw":                 q_raw,
        "k_raw":                 k_raw,
        "v_raw":                 v_raw,
        "q_normed":              q_normed.reshape(-1),
        "k_normed":              k_normed.reshape(-1),
        "q_rep":                 q_rep.reshape(-1),
        "k_rep":                 k_rep.reshape(-1),
        "v_heads":               v_heads.reshape(-1),
        "beta":                  beta,
        "g":                     g,
        "q_scaled":              q_scaled.reshape(-1),
        "state_after":           state_after.reshape(-1),
        "recurrent_out":         recurrent_out.reshape(-1),
        "out_normed":            out_normed.reshape(-1),
        "z_silu":                z_silu.reshape(-1),
        "out_gated":             out_gated.reshape(-1),
        "o_out":                 o_out,
        "output_hidden":         output_hidden,
        "conv_state_after":      conv_state_after.reshape(-1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def synthesize_layer(
    *,
    hidden: int,
    K: int,
    V: int,
    k_dim: int,
    v_dim: int,
    kernel: int,
    seed: int,
    dtype: torch.dtype,
    warm_state: bool,
) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        fan_in = shape[-1] if len(shape) >= 2 else 1
        scale = (1.0 / fan_in) ** 0.5
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    def norm_w(n: int) -> torch.Tensor:
        return (
            torch.ones(n, dtype=torch.float32)
            + torch.randn(n, generator=g, dtype=torch.float32) * 0.02
        ).to(dtype)

    key_dim = K * k_dim
    value_dim = V * v_dim
    qkv_dim = 2 * key_dim + value_dim

    if warm_state:
        # Seeded "post-prefill" state. Conv state stays BF16 (matches the
        # production layout); recurrent state is F32 (production keeps it
        # in F32 for stability).
        conv_state = (torch.randn(qkv_dim, kernel - 1, generator=g, dtype=torch.float32) * 0.5).to(dtype)
        recurrent_state = torch.randn(V, k_dim, v_dim, generator=g, dtype=torch.float32) * 0.1
    else:
        conv_state = torch.zeros(qkv_dim, kernel - 1, dtype=dtype)
        recurrent_state = torch.zeros(V, k_dim, v_dim, dtype=torch.float32)

    return dict(
        input_hidden=randn(hidden),
        input_norm_w=norm_w(hidden),
        in_proj_qkv_w=randn(qkv_dim, hidden),
        in_proj_z_w=randn(value_dim, hidden),
        in_proj_a_w=randn(V, hidden),
        in_proj_b_w=randn(V, hidden),
        # depthwise conv: [out_channels, 1, kernel] with each channel having
        # its own filter. Initialise small so the silu output stays tame.
        conv1d_w=(torch.randn(qkv_dim, 1, kernel, generator=g, dtype=torch.float32) * 0.3).to(dtype),
        conv1d_bias=None,
        # dt_bias and A_log are per-V-head scalars; use small magnitudes so
        # the softplus chain stays finite.
        dt_bias=(torch.randn(V, generator=g, dtype=torch.float32) * 0.1).to(dtype),
        a_log=(torch.randn(V, generator=g, dtype=torch.float32) * 0.5).to(dtype),
        norm_w=norm_w(v_dim),
        out_proj_w=randn(hidden, value_dim),
        conv_state_before=conv_state,
        recurrent_state_before=recurrent_state,
    )


def load_layer_from_checkpoint(
    *, model_dir: Path, layer_idx: int, weight_prefix: str, device: str
) -> dict:
    lp = f"{weight_prefix}.layers.{layer_idx}"
    la = f"{lp}.linear_attn"

    def maybe_load(name: str) -> torch.Tensor | None:
        try:
            return load_tensor(model_dir, name, device)
        except SystemExit:
            return None

    return dict(
        input_norm_w=load_tensor(model_dir, f"{lp}.input_layernorm.weight", device),
        in_proj_qkv_w=load_tensor(model_dir, f"{la}.in_proj_qkv.weight", device),
        in_proj_z_w=load_tensor(model_dir, f"{la}.in_proj_z.weight", device),
        in_proj_a_w=load_tensor(model_dir, f"{la}.in_proj_a.weight", device),
        in_proj_b_w=load_tensor(model_dir, f"{la}.in_proj_b.weight", device),
        conv1d_w=load_tensor(model_dir, f"{la}.conv1d.weight", device),
        conv1d_bias=maybe_load(f"{la}.conv1d.bias"),
        dt_bias=load_tensor(model_dir, f"{la}.dt_bias", device),
        a_log=load_tensor(model_dir, f"{la}.A_log", device),
        norm_w=load_tensor(model_dir, f"{la}.norm.weight", device),
        out_proj_w=load_tensor(model_dir, f"{la}.out_proj.weight", device),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3.6-MoE single-layer linear-attention reference forward"
    )
    p.add_argument("--mode", choices=["synthetic", "checkpoint"], default="synthetic")
    p.add_argument("--model-dir", type=Path,
                   help="Path to the HuggingFace safetensors dir (checkpoint mode)")
    p.add_argument("--layer-idx", type=int, default=0,
                   help="Linear-attention layer index (default 0)")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--state", choices=["fresh", "warm"], default="fresh",
                   help="Prior state: 'fresh' (zeros, first decode step) or "
                        "'warm' (seeded random; exercises decay + delta update)")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    # Linear-attn geometry (35B-A3B defaults).
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--conv-kernel-dim", type=int, default=4)
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    p.add_argument("--int4", action="store_true",
                   help="Quantize the three projection weights "
                        "(in_proj_qkv, in_proj_z, out_proj) to INT4 "
                        "(min/max group-quant). Schema becomes "
                        "`qwen36-moe-oracle-linear-int4-v1`.")
    p.add_argument("--int4-group-size", type=int, default=128,
                   help="Group size for INT4 min/max quant. Must divide "
                        "out_features and in_features of every quantized "
                        "tensor. The runtime + bake both pin to 128.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.mode == "synthetic":
        weights = synthesize_layer(
            hidden=args.hidden,
            K=args.num_k_heads,
            V=args.num_v_heads,
            k_dim=args.head_k_dim,
            v_dim=args.head_v_dim,
            kernel=args.conv_kernel_dim,
            seed=args.seed,
            dtype=dtype,
            warm_state=(args.state == "warm"),
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
        # Reproducible synthetic input + state (same RNG style as
        # qwen36_moe_oracle's checkpoint mode).
        torch.manual_seed(args.seed)
        weights["input_hidden"] = (
            torch.randn(args.hidden, dtype=torch.float32) / args.hidden ** 0.5
        ).to(dtype)
        # Norm weights from 35B-A3B are stored BF16; cast to working dtype.
        for k in ("input_norm_w", "norm_w"):
            weights[k] = weights[k].to(dtype)

        key_dim = args.num_k_heads * args.head_k_dim
        value_dim = args.num_v_heads * args.head_v_dim
        qkv_dim = 2 * key_dim + value_dim
        if args.state == "warm":
            weights["conv_state_before"] = (
                torch.randn(qkv_dim, args.conv_kernel_dim - 1, dtype=torch.float32) * 0.5
            ).to(dtype)
            weights["recurrent_state_before"] = (
                torch.randn(
                    args.num_v_heads, args.head_k_dim, args.head_v_dim, dtype=torch.float32
                ) * 0.1
            )
        else:
            weights["conv_state_before"] = torch.zeros(
                qkv_dim, args.conv_kernel_dim - 1, dtype=dtype
            )
            weights["recurrent_state_before"] = torch.zeros(
                args.num_v_heads, args.head_k_dim, args.head_v_dim, dtype=torch.float32
            )

    weights = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
               for k, v in weights.items()}

    int4_sidecars: dict[str, dict[str, torch.Tensor]] | None = None
    if args.int4:
        # Quantize before computing intermediates so the reference uses the
        # same BF16-reconstructed weights the kernel will see.
        int4_sidecars = quantize_int4_linear_weights(weights, args.int4_group_size)

    intermediates = reference_linear_attn_layer(
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
        conv_kernel_dim=args.conv_kernel_dim,
        rms_norm_eps=args.rms_norm_eps,
        **weights,
    )

    encode = b64_bf16 if args.dtype == "bf16" else b64_f32
    weight_payload = {}
    for k, v in weights.items():
        if v is None:
            continue
        # recurrent_state_before is F32 always; encode appropriately.
        if k == "recurrent_state_before":
            weight_payload[k] = b64_f32(v)
        else:
            weight_payload[k] = encode(v)

    intermediate_payload = {}
    for k, v in intermediates.items():
        # state_after is F32 (production keeps recurrent state in F32);
        # everything else follows the working dtype.
        if k == "state_after":
            intermediate_payload[k] = b64_f32(v)
        else:
            intermediate_payload[k] = encode(v)

    config = {
        "hidden": args.hidden,
        "num_k_heads": args.num_k_heads,
        "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim,
        "head_v_dim": args.head_v_dim,
        "conv_kernel_dim": args.conv_kernel_dim,
        "rms_norm_eps": args.rms_norm_eps,
    }
    if args.int4:
        config["int4_group_size"] = args.int4_group_size

    out = {
        "schema": ("qwen36-moe-oracle-linear-int4-v1"
                   if args.int4 else "qwen36-moe-linear-oracle-layer-v1"),
        "mode": args.mode,
        "state": args.state,
        "layer_idx": args.layer_idx,
        "dtype": args.dtype,
        "config": config,
        "weights": weight_payload,
        "intermediates": intermediate_payload,
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

    # Eyeball line.
    out_norm = intermediates["output_hidden"].to(torch.float32).norm().item()
    in_norm = weights["input_hidden"].to(torch.float32).norm().item()
    delta = (intermediates["output_hidden"] - weights["input_hidden"]).to(torch.float32).norm().item()
    sys.stderr.write(
        f"[linear-oracle] layer={args.layer_idx} state={args.state} mode={args.mode} "
        f"|input|={in_norm:.4f} |output|={out_norm:.4f} |delta|={delta:.4f}\n"
    )


if __name__ == "__main__":
    main()
