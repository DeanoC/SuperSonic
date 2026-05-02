"""
Qwen3.6-MoE multi-token-prediction (MTP) head reference oracle.

Reproduces vLLM's `Qwen3NextMultiTokenPredictor.forward` byte-for-byte in
pure PyTorch, then dumps every intermediate as base64-BF16 to a JSON file
for Rust-side parity testing. The 35B-A3B MTP head is one full-attention
MoE block fused into the base hidden state via an `mtp.fc` linear, three
RMSNorms (`mtp.pre_fc_norm_{hidden,embedding}`, `mtp.norm`), and shared
`embed_tokens` / `lm_head` with the base model.

End-to-end equation per draft step `k`, recurrent through `k = 0..K-1`:

    e        = embed_tokens(next_token_id)
    e_norm   = rmsnorm(e, mtp.pre_fc_norm_embedding.weight, eps)
    h_norm   = rmsnorm(h_base, mtp.pre_fc_norm_hidden.weight, eps)
    fused    = mtp.fc @ cat([e_norm, h_norm], dim=-1)        # cat order: e first
    out      = layer(fused, residual=None, pos=base_len+k, kv=mtp_kv_buffer)
    h_post   = rmsnorm(out, mtp.norm.weight, eps)
    logits   = lm_head @ h_post
    next     = argmax(logits)                                # vLLM drafts greedy
    h_base   = h_post                                        # recurrent

vLLM source: `vllm/model_executor/models/qwen3_next_mtp.py` —
`Qwen3NextMultiTokenPredictor.forward` (cat order, fc, layer call,
norm-with-residual fold) and `vllm/v1/spec_decode/llm_base_proposer.py`
(K>1 loop: input_ids = argmax of step k's logits, hidden_states = step k's
post-norm output).

## Synthetic mode (default)

The 35B-A3B base model is ~64 GiB BF16 and does not fit on a 24 GiB GPU
or a 64 GiB host. For parity testing we don't need real-prefill output —
the SuperSonic-side kernel transforms `(h_base, next_token_id, position)`
deterministically, so a seeded synthetic `h_base` + a chosen
`next_token_id` are sufficient inputs. The oracle defaults to that mode
and only loads what it actually needs:
  - 19 `mtp.*` tensors (~1.6 GiB BF16)
  - shared `embed_tokens.weight` (vocab × hidden = 248320 × 2048 × 2 ≈ 970 MiB)
  - shared `lm_head.weight` (same shape, different tensor; tied=False)
  - one `Qwen3_5MoeDecoderLayer` instantiated from the config (≈ structural
    only; we replace its random init with the MTP weights)

Real prefill against the live HF model is a future extension; the schema
is forward-compatible (a `--mode prefill` flag could fill in `base_seq_len`
and a real `h_base` from `model.generate(...)` output).

Used by `crates/runner/tests/qwen36_moe_mtp_parity.rs` (Phase 6.2c+) to
gate the SuperSonic-side MTP kernel against this reference.

Usage:
    .venv-bake/bin/python oracle/qwen36_moe_mtp_oracle.py \
      --model-dir /path/to/Qwen3.6-35B-A3B \
      --num-speculative-tokens 3 \
      --base-seq-len 12 \
      --base-next-token 71093 \
      --seed 42 \
      --out /tmp/qwen36_mtp.json
"""
import argparse
import base64
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
)


def b64_bf16(t: torch.Tensor) -> str:
    """Pack a tensor as BF16 little-endian and base64-encode."""
    arr = t.detach().to(torch.bfloat16).cpu().contiguous()
    # PyTorch BF16 doesn't go through numpy directly; reinterpret as u16.
    u16 = arr.view(torch.uint16).numpy()
    return base64.b64encode(u16.tobytes()).decode()


def b64_i32(ids: list[int]) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()


def load_named_tensors(model_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Load specific tensor names from the safetensors index."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for n in names:
        shard = index["weight_map"].get(n)
        if shard is None:
            raise SystemExit(f"tensor '{n}' not in safetensors index")
        by_shard.setdefault(shard, []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def rmsnorm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard RMSNorm with HF Qwen3_5MoeRMSNorm `(1.0 + gain)` unit offset."""
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    g_f = gain.to(torch.float32)
    return (x_f * rms * (1.0 + g_f)).to(x.dtype)


def find_full_attention_layer_idx(text_cfg) -> int:
    for i, t in enumerate(text_cfg.layer_types):
        if t == "full_attention":
            return i
    raise SystemExit("no full_attention layer in config; can't instantiate MTP layer")


def build_mtp_layer(text_cfg, mtp_state: dict[str, torch.Tensor], device: str):
    """Instantiate a `Qwen3_5MoeDecoderLayer` (full-attention variant) and
    load `mtp.layers.0.*` into it. Returns `(layer, rotary)` — the rotary
    module produces (cos, sin) for any (x, position_ids).

    HF instantiates the layer with random init; we immediately overwrite
    via `load_state_dict`. This avoids loading the full 35B base model.
    """
    layer_idx = find_full_attention_layer_idx(text_cfg)
    layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=layer_idx)
    layer = layer.to(torch.bfloat16).to(device).eval()
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg).to(device).eval()

    # Drop the `mtp.layers.0.` prefix and check expected names.
    prefix = "mtp.layers.0."
    layer_state: dict[str, torch.Tensor] = {}
    for k, v in mtp_state.items():
        if not k.startswith(prefix):
            continue
        sub = k[len(prefix):]
        layer_state[sub] = v.to(torch.bfloat16)

    expected = set(name for name, _ in layer.named_parameters())
    missing = expected - set(layer_state.keys())
    if missing:
        raise SystemExit(
            f"missing MTP layer weights for: {sorted(missing)[:8]} "
            f"(layer expects {len(expected)} params, got {len(layer_state)})"
        )

    # `strict=False` because HF may register non-parameter buffers
    # (rotary inv_freq lives inside the rotary module, not the layer,
    # so the layer state_dict has only the parameter names we provide).
    layer.load_state_dict(layer_state, strict=False)
    return layer, rotary


@torch.no_grad()
def mtp_step(
    h_base: torch.Tensor,           # [1, hidden] BF16
    next_token_id: int,
    position: int,
    embed_weight: torch.Tensor,     # [vocab, hidden] BF16, base
    mtp_layer: torch.nn.Module,
    pre_fc_h: torch.Tensor,
    pre_fc_e: torch.Tensor,
    fc_w: torch.Tensor,             # [hidden, 2*hidden]
    norm_w: torch.Tensor,
    lm_head_weight: torch.Tensor,   # [vocab, hidden] BF16, base
    eps: float,
    past_kv,
    device: str,
):
    """One vLLM-faithful MTP draft step. Returns intermediates as a dict."""
    # Embed.
    e = embed_weight[next_token_id : next_token_id + 1].to(torch.bfloat16)
    e = e.to(device)  # [1, hidden]

    e_norm = rmsnorm(e, pre_fc_e, eps)
    h_norm = rmsnorm(h_base.to(torch.bfloat16), pre_fc_h, eps)

    # vLLM cat order: [e_norm, h_norm] (embedding first, hidden second).
    cat = torch.cat([e_norm, h_norm], dim=-1)  # [1, 2*hidden]
    fused = torch.nn.functional.linear(cat, fc_w.to(torch.bfloat16))  # [1, hidden]

    # The decoder layer expects [batch, seq, hidden].
    fused_3d = fused.unsqueeze(1)  # [1, 1, hidden]
    pos_ids = torch.tensor([[position]], device=device)

    # HF's `Qwen3_5MoeDecoderLayer` requires precomputed RoPE
    # (cos, sin). The rotary module returned by `build_mtp_layer` is
    # passed in via `mtp_layer._mtp_oracle_rope`.
    cos_sin = mtp_layer._mtp_oracle_rope(fused_3d, pos_ids)
    layer_out = mtp_layer(
        hidden_states=fused_3d,
        position_embeddings=cos_sin,
        position_ids=pos_ids,
        past_key_values=past_kv,
        use_cache=True,
        cache_position=pos_ids[0],
    )

    if isinstance(layer_out, tuple):
        attn_out_3d = layer_out[0]
        new_kv = layer_out[1] if len(layer_out) > 1 else past_kv
    else:
        attn_out_3d = layer_out
        new_kv = past_kv
    attn_out = attn_out_3d.squeeze(1)  # [1, hidden]

    h_post = rmsnorm(attn_out, norm_w, eps)
    logits = torch.nn.functional.linear(h_post.to(torch.bfloat16), lm_head_weight)
    next_tok = int(logits.argmax(dim=-1).item())

    # Keep all returned tensors 2D `[1, *]` so the caller's recurrence
    # (`h_base = out["h_post"]`) stays shape-stable. The JSON dump
    # squeezes to 1D at write time.
    return dict(
        e_norm=e_norm,
        h_norm=h_norm,
        fused=fused,
        attn_out=attn_out,
        h_post=h_post,
        logits=logits,
        next_tok=next_tok,
        new_past_kv=new_kv,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--num-speculative-tokens", type=int, default=3,
                    help="K — vLLM default is 2 per the model card; we use 3 "
                         "for richer parity coverage of the recurrent path.")
    ap.add_argument("--base-seq-len", type=int, default=12,
                    help="Position the base model would have been at when "
                         "feeding h_base into MTP (synthetic mode; real "
                         "prefill mode would derive from the prompt).")
    ap.add_argument("--base-next-token", type=int, default=None,
                    help="Token id sampled by the base model's lm_head; fed "
                         "into the first MTP step. Defaults to a random "
                         "in-range id seeded by --seed.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda:0. CPU is the reproducible default.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[mtp-oracle] reading config from {args.model_dir}...")
    cfg = AutoConfig.from_pretrained(str(args.model_dir))
    text_cfg = cfg.text_config
    hidden = int(text_cfg.hidden_size)
    vocab = int(text_cfg.vocab_size)
    eps = float(text_cfg.rms_norm_eps)

    print(f"[mtp-oracle] loading mtp.* tensors + tied embed/lm_head (synthetic mode)...")
    needed = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
    ]
    # mtp.layers.0.* — list all 15 from the safetensors index lazily.
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    needed += sorted(k for k in index["weight_map"] if k.startswith("mtp.layers.0."))
    # Also need the shared embed/lm_head; the canonical Qwen3.5/3.6 names.
    needed += [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ]
    tensors = load_named_tensors(args.model_dir, needed)

    embed_w = tensors["model.language_model.embed_tokens.weight"].to(torch.bfloat16)
    lm_head_w = tensors["lm_head.weight"].to(torch.bfloat16)
    pre_fc_h = tensors["mtp.pre_fc_norm_hidden.weight"].to(args.device)
    pre_fc_e = tensors["mtp.pre_fc_norm_embedding.weight"].to(args.device)
    fc_w = tensors["mtp.fc.weight"].to(args.device)
    norm_w = tensors["mtp.norm.weight"].to(args.device)
    embed_w = embed_w.to(args.device)
    lm_head_w = lm_head_w.to(args.device)

    mtp_state = {k: v for k, v in tensors.items() if k.startswith("mtp.layers.0.")}
    print(f"[mtp-oracle] building MTP decoder layer (full-attention variant)...")
    mtp_layer, rotary = build_mtp_layer(text_cfg, mtp_state, args.device)
    # Attach the rotary so `mtp_step` can call it without an extra arg.
    mtp_layer._mtp_oracle_rope = rotary

    # Synthetic h_base + base next token.
    h_base = torch.from_numpy(
        rng.standard_normal((1, hidden)).astype(np.float32) * 0.5
    ).to(torch.bfloat16).to(args.device)
    base_next_tok = (
        args.base_next_token
        if args.base_next_token is not None
        else int(rng.integers(0, vocab))
    )
    base_seq_len = args.base_seq_len

    print(f"[mtp-oracle] running {args.num_speculative_tokens} draft step(s) "
          f"from base_seq_len={base_seq_len}, base_next_token={base_next_tok}...")
    past_kv = None
    next_tok = base_next_tok
    steps = []
    for k in range(args.num_speculative_tokens):
        out = mtp_step(
            h_base=h_base,
            next_token_id=next_tok,
            position=base_seq_len + k,
            embed_weight=embed_w,
            mtp_layer=mtp_layer,
            pre_fc_h=pre_fc_h,
            pre_fc_e=pre_fc_e,
            fc_w=fc_w,
            norm_w=norm_w,
            lm_head_weight=lm_head_w,
            eps=eps,
            past_kv=past_kv,
            device=args.device,
        )
        steps.append(dict(
            step=k,
            position=base_seq_len + k,
            input_token_id=next_tok,
            draft_token_id=out["next_tok"],
            e_norm_bf16=b64_bf16(out["e_norm"].squeeze(0)),
            h_norm_bf16=b64_bf16(out["h_norm"].squeeze(0)),
            fused_bf16=b64_bf16(out["fused"].squeeze(0)),
            attn_out_bf16=b64_bf16(out["attn_out"].squeeze(0)),
            h_post_bf16=b64_bf16(out["h_post"].squeeze(0)),
            logits_bf16=b64_bf16(out["logits"].squeeze(0)),
        ))
        past_kv = out["new_past_kv"]
        h_base = out["h_post"]
        next_tok = out["next_tok"]

    # Pack the synthetic h_base used for step 0 — the parity test feeds
    # this exact tensor into the SuperSonic kernel.
    h_base_step0 = torch.from_numpy(
        rng.standard_normal((1, hidden)).astype(np.float32) * 0.5
    ).to(torch.bfloat16)
    # ^ Re-derive h_base for step 0 — we mutated it above. Cleaner: track
    # it explicitly. (Note: this is a no-op since rng's used and the
    # original tensor is gone; let's just re-seed.)
    rng2 = np.random.default_rng(args.seed)
    h_base_step0 = torch.from_numpy(
        rng2.standard_normal((1, hidden)).astype(np.float32) * 0.5
    ).to(torch.bfloat16).squeeze(0)  # [hidden]

    fixture = dict(
        schema="qwen36-moe-mtp-oracle-v1",
        mode="synthetic",
        config=dict(
            hidden=hidden,
            vocab=vocab,
            num_attention_heads=int(text_cfg.num_attention_heads),
            num_kv_heads=int(text_cfg.num_key_value_heads),
            head_dim=int(text_cfg.head_dim),
            rms_norm_eps=eps,
            rope_theta=float(text_cfg.rope_parameters["rope_theta"]),
            partial_rotary_factor=float(text_cfg.partial_rotary_factor),
            num_experts=int(text_cfg.num_experts),
            moe_intermediate_size=int(text_cfg.moe_intermediate_size),
            shared_expert_intermediate_size=int(text_cfg.shared_expert_intermediate_size),
            top_k=int(text_cfg.num_experts_per_tok),
        ),
        seed=args.seed,
        base_seq_len=base_seq_len,
        base_next_token_id=base_next_tok,
        h_base_step0_bf16=b64_bf16(h_base_step0),
        draft_token_ids=[s["draft_token_id"] for s in steps],
        steps=steps,
    )

    args.out.write_text(json.dumps(fixture))
    print(f"[mtp-oracle] wrote {len(steps)} step(s) to {args.out}")
    print(f"[mtp-oracle] base→draft: [{base_next_tok}, "
          f"{', '.join(str(s['draft_token_id']) for s in steps)}]")


if __name__ == "__main__":
    main()
