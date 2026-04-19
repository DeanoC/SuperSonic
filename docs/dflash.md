# DFlash Speculative Decoding — Canonical Reference

This document is the confirmed ground-truth spec for the DFlash draft model as
shipped in `z-lab/Qwen3.5-9B-DFlash`. All M2+ implementation in SuperSonic
defers to this document; the z-lab paper and GitHub repo are secondary
references.

**Primary sources:**
- `z-lab/Qwen3.5-9B-DFlash/dflash.py` (MIT, shipped with checkpoint, referenced
  via `auto_map: AutoModel: dflash.DFlashDraftModel` in `config.json`)
- `z-lab/Qwen3.5-9B-DFlash/config.json`
- `z-lab/Qwen3.5-9B-DFlash/model.safetensors` (58 tensors, ~2.0 GiB BF16)
- `github.com/z-lab/dflash` (inference loop — `dflash_generate` entry point)

Paper: arXiv 2602.06036. The paper does not fully specify the injection
arithmetic — use the code, not the paper, as the source of truth.

## 1. Target configuration

The draft was trained to tap `Qwen3.5-9B` (target has `num_target_layers: 32`
per DFlash config). Tap indices are **read from the checkpoint's config.json,
not hardcoded**:

```
target_layer_ids: [1, 8, 15, 22, 29]     # 5 taps
```

Each tap is the hidden state at the **output** of the specified target layer
(post-residual, post-MLP). M1 in SuperSonic writes these at the same point
`component_decode_step_4b_trace_layer` reads from, i.e. after
`hidden_io` is finalized for that layer.

## 2. Draft architecture

```
num_hidden_layers: 5
layer_types: [full_attention] * 5        # all full, no sliding
hidden_size: 4096
intermediate_size: 12288
num_attention_heads: 32                  # GQA 4:1
num_key_value_heads: 8
head_dim: 128
vocab_size: 248320
block_size: 16                           # candidates per round
mask_token_id: 248070                    # fill for unrevealed positions
rope_theta: 1e7
rms_norm_eps: 1e-6
tie_word_embeddings: false               # (moot — draft owns neither tensor)
```

Safetensors keys per layer (58 tensors total):
```
layers.{i}.input_layernorm.weight             [4096]           — pre-attn
layers.{i}.self_attn.q_proj.weight            [4096, 4096]
layers.{i}.self_attn.k_proj.weight            [1024, 4096]
layers.{i}.self_attn.v_proj.weight            [1024, 4096]
layers.{i}.self_attn.o_proj.weight            [4096, 4096]
layers.{i}.self_attn.q_norm.weight            [128]            — per-head RMSN
layers.{i}.self_attn.k_norm.weight            [128]            — per-head RMSN
layers.{i}.post_attention_layernorm.weight    [4096]           — pre-MLP
layers.{i}.mlp.gate_proj.weight               [12288, 4096]
layers.{i}.mlp.up_proj.weight                 [12288, 4096]
layers.{i}.mlp.down_proj.weight               [4096, 12288]
```

Top-level:
```
fc.weight            [4096, 20480]             — tap fuser (no bias)
hidden_norm.weight   [4096]                    — post-fuser RMSNorm
norm.weight          [4096]                    — final RMSNorm (pre-lm_head)
```

Note: `fc.weight` has the PyTorch nn.Linear convention — rows = out_features
(4096), cols = in_features (5 * 4096 = 20480). The matmul is
`y = x @ fc.weight.T`.

## 3. Tap fuser (runs ONCE per decode round, reused by all 5 draft layers)

```python
# target_hidden: list of 5 tensors, each [B=16, 4096] from taps [1,8,15,22,29]
# Tile each tap to all B=16 positions (same vector replicated).
target_hidden = concat(target_hidden, dim=-1)        # [B, 5*4096] = [B, 20480]
target_hidden = linear_no_bias(target_hidden, fc.weight)   # [B, 4096]
target_hidden = rms_norm(target_hidden, hidden_norm.weight)
```

The tap vector is the SAME across all 16 candidate positions — it's conditioned
on the single committed prefix token, not on the block. `ctx_len = B = 16` in
attention because the fused vector is tiled along the context dimension.

(Verify: `dflash.py:174-175`.)

## 4. Per-layer forward — the injection formula

From `dflash.py:56-100` (`Qwen3DFlashAttention.forward`):

```python
# hidden_states: [B=16, q_len=16, 4096]   draft-side input
# target_hidden: [B=16, ctx_len=16, 4096] output of the fuser

# ---- Q: draft only ----
q = q_proj(hidden_states)                                # [B, 16, 4096]
q = q.view(B, 16, 32, 128)
q = q_norm(q)                                            # per-head RMSN
q = q.transpose(1, 2)                                    # [B, 32, 16, 128]

# ---- K, V: SAME weights applied to BOTH streams, CONCAT along seq ----
k_ctx   = k_proj(target_hidden)                          # [B, 16, 1024]
k_noise = k_proj(hidden_states)                          # [B, 16, 1024]
v_ctx   = v_proj(target_hidden)
v_noise = v_proj(hidden_states)
k = concat([k_ctx, k_noise], dim=1)                      # [B, 32, 1024]
v = concat([v_ctx, v_noise], dim=1)
k = k.view(B, 32, 8, 128)
k = k_norm(k)                                            # per-head RMSN
k = k.transpose(1, 2)                                    # [B, 8, 32, 128]
v = v.view(B, 32, 8, 128).transpose(1, 2)

# ---- RoPE on full concat (NOT on q_ctx since Q has no ctx part) ----
# Position IDs span 0..ctx_len+q_len = 32. apply_rotary_pos_emb slices cos/sin
# to the last q_len for Q, uses full for K. See dflash.py:20-26.
q, k = apply_rotary_pos_emb(q, k, cos, sin)

# ---- Attention (bidirectional, no mask) ----
# is_causal = False. sliding_window = None (all layers full_attention).
attn = softmax(q @ k.transpose(-1,-2) * (1/sqrt(128))) @ v     # [B, 32, 16, 128]
attn = attn.transpose(1, 2).reshape(B, 16, 4096)
out  = o_proj(attn)                                       # [B, 16, 4096]
```

**Key facts:**
- K and V each have seq length 32 (16 context + 16 noise). Q has seq length 16.
- Attention output shape is `q_len = 16` — the context tokens contribute to
  attention output only via being attended-TO.
- `q_norm` applied in HEAD view (over head_dim=128), not hidden_size. Same for
  `k_norm`. Standard Qwen3.
- `q_norm` applied BEFORE transpose; `k_norm` applied AFTER reshape to head
  view but BEFORE transpose (line 77). Matches Qwen3.
- No K/V injection projections exist — `k_inject`/`v_inject` from the plan
  assumption are NOT in the checkpoint.

Decoder layer (`dflash.py:111-143`) is standard pre-norm:
```
x = hidden_states
x = x + attn(input_layernorm(x), target_hidden)
x = x + mlp(post_attention_layernorm(x))           # SwiGLU, standard Qwen3
```

Final: `norm(last_layer_output)` → returned to caller (caller applies target's
lm_head; draft has no lm_head tensor).

## 5. Input preparation (speculative loop side)

From `github.com/z-lab/dflash/dflash/model.py` (`dflash_generate`, confirmed
by a second inspection targeting exact line numbers):

### 5.1 Draft forward signature

```python
# line 111-119 of dflash_generate:
noise_embedding = target.model.embed_tokens(block_output_ids)   # [B=1, q_len=16, hidden]
draft_hidden = model(
    target_hidden  = target_hidden,                              # [B, ctx_len, hidden]  — variable ctx_len!
    noise_embedding= noise_embedding,
    position_ids   = position_ids[:, past_draft_cache_len : start + block_size],
    past_key_values= past_key_values_draft,                      # persists + grows across rounds
    use_cache      = True,
    is_causal      = False,
)
block_logits = target.lm_head(draft_hidden[:, 1 - block_size :, :])  # last q_len rows
# line 120:
past_key_values_draft.crop(start)                                # rollback this round's ctx+noise append
```

### 5.2 `target_hidden` is NOT pre-tiled to block_size

`extract_context_feature` (line 45) returns `cat([tapped_states], dim=-1)` →
`[B, T, num_taps * hidden]` where `T` is the target's output length. On the
very first round `T = 1` (target was invoked with `logits_to_keep=1`). On
subsequent rounds `T = acceptance_length + 1` (up to `block_size`).

So **`ctx_len` varies per round, `1 ≤ ctx_len ≤ block_size`** — it is NOT
always 16. The fuser projects each ctx position independently (shape
`[B, ctx_len, num_taps*hidden] → [B, ctx_len, hidden]`). The `[ctx_len, hidden]`
result is passed in directly; no tiling / no broadcast.

### 5.3 Position IDs cover past_cache gap + ctx + noise, monotonically

```python
# line 82 (outer loop):
position_ids = torch.arange(max_length + block_size).unsqueeze(0)   # [1, big]
```
Each round's slice `[:, past_draft_cache_len : start + block_size]` covers:
- any gap between the draft cache end and `start` (usually zero after prior crop)
- the `ctx_len` ctx positions (taps for the last `acceptance_length+1` tokens)
- the `q_len = block_size` noise/draft positions

These are contiguous real-sequence positions. RoPE over K uses the full slice;
RoPE over Q uses `cos[..., -q_len:, :]` (tail — the noise positions).

### 5.4 Draft KV cache lifecycle

The draft owns its own KV cache (`past_key_values_draft`). Per round:
1. Cache is at length `start` (last committed position), from prior-round crop.
2. Inside forward, each layer appends `ctx_len + q_len` new (K,V) entries.
3. Attention runs bidirectionally (`is_causal=False`) over
   `cache_len + ctx_len + q_len` total positions.
4. After forward returns, `crop(start)` rolls back the appended
   `ctx_len + q_len` — only the committed prefix persists.
5. Next round: `start += acceptance_length + 1`, cache length stays at the new
   `start` after the crop at the top of the next round.

**Implication for SuperSonic:** the draft needs its own KV cache buffer (not
shared with target). Design mirrors standard Qwen3 GQA KV cache — shape
`[1, num_kv_heads=8, max_ctx, head_dim=128]` per layer, BF16. Size budget is
small: `5 layers × 2 (K+V) × 8 × max_ctx × 128 × 2 bytes ≈ 20 KiB/token`, so
4K ctx = 80 MiB, 32K = 640 MiB. Comfortable within the ~2 GiB draft budget.

### 5.5 Attention mask: None

`attention_mask` is never passed to the draft. `is_causal=False` in
`Qwen3DFlashAttention` (line 39). Attention is fully bidirectional — every
position (cache + ctx + noise) attends to every other position.

## 6. Accept/reject (M3, not M2)

1. Verify: run target prefill for the 16-candidate block at positions
   `[L, L+16)`, with `commit_kv_filled = false`.
2. `block_argmax = argmax_per_pos(verify_logits)`.
3. `accepted = longest prefix i s.t. block_tokens[0..i] == block_argmax[0..i]`
   at temperature 0. Commit `accepted + 1` tokens (accepted + 1 bonus).
4. `kv_filled = L + accepted + 1`. KV past that offset is harmlessly
   overwritten next round (no tree attention, no rollback machinery).
5. Mask-token semantics are only relevant to the draft's INPUT (for positions
   it hasn't seen yet) — they do not enter verification.

## 7. Sharing with target (Arc pattern)

The draft has no `embed_tokens` and no `lm_head` tensors. SuperSonic shares
these via `Arc::clone(&target.embed_tokens)` / `Arc::clone(&target.lm_head)`.
Pattern already established in `crates/qwen35/src/weights.rs:37,142,148-151`.
The draft crate must NOT load these from its own safetensors — they don't
exist there.

## 8. Divergences from the pre-M2 plan

These are the items the plan assumed that the checkpoint inspection corrected:

| Plan assumption | Actual |
|---|---|
| `K = K_proj(x) + K_inject(fused)` (additive) | `K = concat([k_proj(fused), k_proj(x)], seq)` (same weights, concat) |
| `k_inject` / `v_inject` projections exist | Do not exist; no extra attention weights |
| 3 tap layers `[2, 18, 33]` | 5 tap layers `[1, 8, 15, 22, 29]` |
| Fuser `[3*hidden, hidden]` | Fuser `fc: [4096, 20480]` = `[5*hidden, hidden]` |
| `ctx_len = block_size = 16` (tap tiled) | `1 ≤ ctx_len ≤ block_size`; taps are per-ctx-position, not tiled |
| Attention seq = 32 (fixed) | Seq = `draft_cache_len + ctx_len + q_len`, variable per round |
| Draft has no KV cache (one-shot) | Draft owns a KV cache; appended + crop-rolled-back per round |

Plan items that held up:
- Draft is 5 layers (`num_hidden_layers: 5`).
- Bidirectional attention (`is_causal=False`).
- Draft shares embed_tokens + lm_head with target via Arc.
- Block size 16.
- All full_attention layers (no sliding).

### 8.1 Impact on M2 vs M3

The original M2 gate asked for a one-shot `forward(prefix_token, target_taps)`
returning 16 tokens. That maps cleanly onto the "first-round" case:
- `draft_cache_len = 0`
- `ctx_len = 1` (single tap after the prefill bonus decode)
- `q_len = block_size = 16`
- Attention seq length = 17 per layer.

M2 smoke can exercise exactly this path and meet the gate (finite, in-vocab,
<2 ms). The general case (`draft_cache_len > 0`, `ctx_len ≥ 1`) requires a
draft KV cache + the per-round crop protocol. That richer lifecycle is the
natural home of M3 (speculative loop), not M2. The forward function signature
should accept `(ctx, noise, position_ids, past_kv)` so M3 can drive it without
refactoring.

## 9. References

- `/home/deano/models/qwen35-9b-dflash/dflash.py` (local canonical)
- `/home/deano/models/qwen35-9b-dflash/config.json`
- https://github.com/z-lab/dflash — `dflash/model.py`, `dflash/benchmark.py`
- https://huggingface.co/z-lab/Qwen3.5-9B-DFlash
- https://arxiv.org/abs/2602.06036
