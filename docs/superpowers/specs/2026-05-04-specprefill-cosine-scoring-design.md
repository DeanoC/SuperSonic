# SpecPrefill cosine-scoring fast path — design

**Branch:** `research/specprefill-speculator-fastpath`
**Status:** design approved 2026-05-04; ready for writing-plans.
**Worktree:** `/home/deano/projects/SuperSonicBase-spec-fastpath`

## Goal

Eliminate the SpecPrefill speculator slow path (currently ~5.7s for the
lookahead decode on Qwen3.5-9B + 0.8B draft, making SpecPrefill 1.5×
SLOWER than dense prefill on gfx1100). Replace per-layer
softmax(Q·Kᵀ) importance scoring with single-layer cosine-similarity
scoring patterned on hipfire's PFlash. Drop the lookahead decode steps
entirely; the speculator just runs dense prefill and we score from one
of its K caches.

Target: SpecPrefill keep=0.50 TTFT on 1353-token Qwen3.5-9B prompt
< 5.0s (dense baseline). Currently 7.7s.

## Non-goals

- Deleting the existing `prefill_with_lookahead_attention` /
  `decode_step_with_query_capture` / `lookahead_attention_scores`
  paths in this PR. They stay behind a `--specprefill-algorithm`
  flag for A/B comparison and as a fallback. Phase E removal.
- Cross-family draft (Qwen3.5 → Qwen3.6).
- Sampling-based decode.
- Lifting the SpecPrefill+KV-FP8 validation gate added in PR #188
  (separate Phase D follow-up — needs BF16→FP8 step-copy quantise).

## What ships

Three landable PRs:

### PR D1 — cosine kernel + FFI + parity test

- New HIP kernel `pfx_pflash_cosine_score_kernel` in
  `kernels/prefill_helpers.hip`. One workgroup per block, lane-strided
  over the per-position K vector (`kv_dim = num_kv_heads * head_dim`),
  shared-memory reduction for `(dot, ||block_mean||², ||last_k||²)`,
  emits `score = dot / sqrt(||block_mean||² × ||last_k||²)` per block.
  Reads BF16 K directly (no Q8 dequant — our drafter cache is BF16,
  unlike hipfire's Q8 cache).
- HIP bridge entry + CUDA/Metal stubs + Rust FFI in
  `crates/kernel-ffi/src/prefill_ffi.rs`.
- Standalone parity test `crates/runner/tests/specprefill_pflash_cosine_parity.rs`
  against a CPU reference (deterministic synthetic K), asserting per-block
  cosine values match within 1e-5 and the per-row sum-of-squares-rebalance
  invariant holds.

### PR D2 — orchestrator integration + flag + validation

- `crates/runner/src/specprefill_engine.rs` gains a `score_blocks_cosine`
  function:
  1. Drafter dense prefill (existing `DecodeEngine::prefill_native_with_final_norm`
     path — fast megakernel, no decode steps).
  2. Pick scoring layer (default = shallowest full-attention layer
     index from `config.layer_types`; override via env var
     `SUPERSONIC_SPECPREFILL_SCORE_LAYER=N`).
  3. For mode=`shallowest` (default A): one cosine kernel launch on that
     layer's K cache. For mode=`all_max` (B): one launch per
     full-attention layer (6 on Qwen3.5-0.8B); element-wise max across
     layers per block. Mode chosen via env var
     `SUPERSONIC_SPECPREFILL_LAYERS=shallowest|all_max`.
  4. Project per-block scores → per-token vector: every token in block
     `b` gets `score[b]`. This lets the existing
     `select_kept_positions` (forced prefix/suffix bands, top-K within
     chunk, deterministic tie-break) consume the new scoring without
     change. `--specprefill-pool-window` becomes a no-op (block-level
     scoring already smooths) — flag stays in place; help text gets
     a "no-op when --specprefill-algorithm=cosine" note.
- `crates/runner/src/main.rs` adds CLI flag
  `--specprefill-algorithm <cosine|lookahead>` with default `lookahead`
  (preserves existing behaviour). PR D3 flips the default.
- `crates/runner/src/policy.rs::validate_specprefill_flags` validates
  the flag value against the two known algorithms.
- New test `crates/runner/tests/specprefill_qwen35_9b_cosine_parity.rs`
  mirrors the existing `specprefill_qwen35_9b_parity.rs` but with
  `--specprefill-algorithm=cosine`. Same bars (cossim ≥ 0.65 keep=0.50,
  cossim ≥ 0.999 keep=1.00, top-5 ≥ 4, argmax match, byte-equal
  multitoken text at keep=1.00).
- Manual TTFT measurement run during PR D2 prep, captured in the PR
  description: target `cosine + keep=0.50 + 1353-tok prompt` < dense
  5.1s baseline.

### PR D3 — flip default + doc updates

- CLI default `--specprefill-algorithm` flips from `lookahead` to `cosine`.
- `docs/specprefill.md` gets an "Algorithm" subsection listing the two
  modes and when to use each (cosine = production default;
  lookahead = legacy/research only).
- `docs/feature-compatibility.md` SpecPrefill subsection updates the
  "Performance note" paragraph to reflect the new TTFT win
  (cosine path measurement).
- `docs/performance.md` § Runtime feature impact: SpecPrefill rows
  update from "1.53× SLOWER" → measured cosine number; new comparison
  row for `cosine vs lookahead` if useful for future readers.
- A vs B (`shallowest` vs `all_max`) measured comparison documented
  in `docs/specprefill.md` so the env-var override is grounded.

## Architecture

```
specprefill_engine::run_specprefill(...)
    │
    ├── load draft + target weights, build draft engine
    ├── if cli.specprefill_algorithm == cosine:        # NEW path
    │       ├── draft_engine.prefill_native_with_final_norm(prompt_ids)
    │       ├── pick scoring layer (env or default = shallowest full-attn)
    │       ├── (single layer)  one launch of pfx_pflash_cosine_score_kernel
    │       │   OR (all_max)    one launch per full-attn layer; max-reduce
    │       ├── project per-block scores → per-token Vec<f32>
    │       └── (importance vector ready)
    ├── else (lookahead):                              # EXISTING Phase C path
    │       └── draft_engine.prefill_with_lookahead_attention(...)  # slow
    │       └── per-layer aggregation (max heads → max layers → mean lookahead)
    │       └── (importance vector ready)
    │
    ├── select_kept_positions(&importance, &cfg)       # UNCHANGED
    ├── target_engine.prefill_kept_native(...)         # UNCHANGED
    ├── decode loop with rope_pos / kv_slot decoupling # UNCHANGED
    └── detokenise + print
```

Only the highlighted "NEW path" branch is added. Everything below
`select_kept_positions` is untouched. The existing Phase C path stays
selectable via `--specprefill-algorithm=lookahead`.

## Kernel layout (the new bit)

`pfx_pflash_cosine_score_kernel<T>` (T = `hip_bfloat16` only on Phase D;
F16 stub returns "not implemented"):

```
inputs:
  k_cache: BF16 [1, kv_heads, cap, head_dim]   (drafter K cache after prefill)
  n_pos:   int                                 (prompt token count)
  kv_heads: int
  head_dim: int
  block_size: int
  n_blocks: int                                (= ceil(n_pos / block_size))
  last_pos: int                                (= n_pos - 1)

outputs:
  scores:  F32 [n_blocks]                      (one cosine per block)

grid:    n_blocks workgroups
block:   wave size threads (queried via hipDeviceProp_t.warpSize)

per workgroup:
  block_idx = blockIdx.x
  block_start = block_idx * block_size
  block_end   = min(block_start + block_size, n_pos)
  block_len   = block_end - block_start
  if block_len == 0: emit 0.0 and return

  # Lane-strided over kv_dim = kv_heads * head_dim
  for d = lane; d < kv_dim; d += warpSize:
      block_mean_d = mean(K[pos][d] for pos in block_start..block_end)
      last_k_d     = K[last_pos][d]
      partial_dot  += block_mean_d * last_k_d
      partial_nb2  += block_mean_d * block_mean_d
      partial_nl2  += last_k_d * last_k_d

  # Three-way warp reduction
  dot = warp_sum(partial_dot)
  nb2 = warp_sum(partial_nb2)
  nl2 = warp_sum(partial_nl2)

  if lane == 0:
      denom = sqrt(nb2 * nl2)
      scores[block_idx] = denom > 0 ? dot / denom : 0.0
```

K cache layout note: drafter cache is `[1, kv_heads, cap, head_dim]`
(per `qwen35::state::ensure_kv_capacity` — same layout as the
SpecPrefill PR #177 lookahead path). Per-position K starts at offset
`(h * cap + pos) * head_dim` for kv-head `h`. The kernel computes
indices accordingly.

## Bench script

Add `tests/gfx1100/bench_specprefill_cosine.sh` mirroring the existing
matrix bench's discipline (warmup + 3 runs + median + 3s cooldown):

```bash
# A vs B vs lookahead vs dense, on the 1353-token prompt
./bench_specprefill_cosine.sh
=> | mode                   | TTFT ms | speedup vs dense |
   | dense (baseline)       |    5057 |             1.00× |
   | cosine, shallowest (A) |     ??? |              ???× |
   | cosine, all_max (B)    |     ??? |              ???× |
   | lookahead (Phase C)    |    7739 |             0.65× |
```

Required for PR D2 + D3 to land. Numbers go into `docs/specprefill.md`
and the PR D3 description.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Single-layer cosine signal is weaker than multi-layer softmax-attention; argmax-match might fail on Qwen3.5-9B at low keep ratios. | Validation harness reuses Phase A2's bars (cossim ≥ 0.65 keep=0.50, argmax match). If A fails, try B (max over all full-attn layers) before pivoting to other approaches. |
| 2 | Shallowest-full-attn-layer choice is hipfire's heuristic; their reasoning ("RoPE-OOD NaN cascade on small drafters") may not apply to Qwen3.5-0.8B at our prompt lengths. | Env-var override `SUPERSONIC_SPECPREFILL_SCORE_LAYER=N` lets us probe other layers. PR D3 documents the chosen default. |
| 3 | Pivot loses the "max over heads" half of Phase A2's aggregation, since cosine is over the full kv_dim vector at once. | Per-head scoring is option (C) from the brainstorm — explicitly deferred. Add as Phase E follow-up if both A and B fail. |
| 4 | The `--specprefill-pool-window` flag becomes a no-op for cosine but remains for lookahead. Confusing for users who set it expecting smoothing. | Help text + doc update document the per-algorithm semantics. The flag still works for the lookahead path. |
| 5 | Drafter K cache layout assumption (`[1, kv_heads, cap, head_dim]`) is shared with the existing PR #177 lookahead path. If the layout ever changes, both kernels break the same way — no new exposure. | None; this is consistent with the existing pattern. |

## Acceptance criteria

PR D1 lands when:
- New cosine kernel + FFI compiles cleanly across HIP / CUDA stub /
  Metal stub.
- `specprefill_pflash_cosine_parity.rs` test passes against the CPU
  reference (max abs error < 1e-3 on synthetic BF16 K, sum-of-squares
  invariant holds).

PR D2 lands when:
- `--specprefill-algorithm` CLI flag accepts `cosine` and `lookahead`.
- `validate_specprefill_flags` rejects unknown algorithm values.
- `specprefill_qwen35_9b_cosine_parity.rs` test passes the existing
  bars on Qwen3.5-9B + 0.8B (cossim ≥ 0.65 keep=0.50, cossim ≥ 0.999
  keep=1.00, top-5 ≥ 4, argmax match, byte-equal multitoken at keep=1.00).
- The lookahead path remains the default and existing tests still pass.
- Manual TTFT measurement: cosine + keep=0.50 + 1353-tok prompt
  TTFT < 5.0s (the dense baseline). Recorded in PR description.

PR D3 lands when:
- Default flips to `cosine` and existing parity test (renamed to refer
  to cosine) still passes.
- Documentation reflects the new default + the cosine TTFT win.
- A vs B measurement included in `docs/specprefill.md`.
