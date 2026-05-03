# SpecPrefill — Phase C end-to-end design

**Branch:** `research/specprefill`
**Companion to:**
- `docs/research/2026-05-03-specprefill-feasibility.md` (overall plan)
- `docs/research/2026-05-03-specprefill-phase-a-results.md` (Phase A numbers — 4B target)
- `docs/research/2026-05-03-specprefill-phase-a2-cross-target.md` (Phase A2 numbers — 9B target)
- `crates/runner/src/specprefill.rs` (host selection, landed Phase A)
- `crates/kernel-ffi/src/prefill_ffi.rs::apply_rope_prefill_indirect` (RoPE-indirect, landed Phase B)

**Status:** design only.

## Scope

End-to-end SpecPrefill on Qwen3.5-9B target + Qwen3.5-0.8B draft, single GPU, single batch, **greedy decode only**. The HIP backend is the implementation target. Out of scope this phase: cross-family draft (Qwen3.5 → Qwen3.6-MoE), sampling-based decoding, CUDA/Metal backends, INT4/FP8 quantised target.

## Phase A2 findings that bind the design

- Argmax preservation is universal across all tested (keep_ratio × prompt_len) cells on the 9B target. Greedy decoding picks the same first generated token at every keep ratio in [0.1, 0.9].
- Cossim degrades on long prompts at low/mid keep ratios (0.30 → 0.68). At keep ≥ 0.50 the 9B target holds cossim ≥ 0.927 with top-5 ≥ 5/5 long-prompt.
- **Default keep_ratio = 0.50** (not 0.30 as originally suggested in the feasibility memo).
- **Integration-test cossim bar = 0.90** (not 0.95). `argmax_match` is the primary correctness gate; cossim is a regression backstop.

## Architecture

```
                ┌───────────────────────────────────────────────────────┐
                │ run_specprefill (new) crates/runner/src/specprefill_engine.rs │
                └───────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────────────────┐
        ▼                 ▼                             ▼
   load draft      load target weights            (existing)
   (Qwen3.5-       (Qwen3.5-9B BF16,              kernel_ffi
    0.8B BF16)      via prefill_engine             rotary, dispatch
                    setup path)
        │                 │
        ▼                 │
  draft prefill +         │
  look-ahead              │
  attention export        │
  → host scratch          │
        │                 │
        ▼                 │
  importance              │
  aggregation             │
  (max heads/layers,      │
  mean lookahead)         │
  + selection             │
  (specprefill.rs,        │
  landed)                 │
        │                 │
        ▼                 ▼
  kept_positions ─→ target sparse prefill
                    (compacted embeds +
                     apply_rope_prefill_indirect +
                     unchanged causal mask)
                          │
                          ▼
                    (existing) decode loop
                    seqlen_offset = original_T
```

The dispatch entry point is `crates/runner/src/main.rs`: when `--specprefill-draft-dir` is set, route to `run_specprefill` instead of the standard prefill+decode path. Pattern mirrors `--dflash` → `run_qwen35_dflash`.

## Component 1: look-ahead attention kernel (new)

**Why a new kernel, not extending `supersonic_qwen35_full_attention_prefill_kernel`:** the dense prefill kernel computes `softmax(QKᵀ) · V` in one fused streaming-softmax pass that consumes the post-softmax weights immediately. Materialising those weights would require either (a) buffering unnormalised exponentials per `(q_row, k_pos)` and dividing post-loop, or (b) branching inside the inner loop. Both touch the production prefill hot path that's already validated end-to-end on the target. A standalone look-ahead kernel keeps the dense prefill bit-identical.

**Kernel:** `supersonic_qwen35_lookahead_attention_scores` (HIP) in `kernels/full_attention.hip`, FFI in `crates/kernel-ffi/src/prefill_ffi.rs`.

Inputs:
- `q`: BF16, `[heads, lookahead_count, head_dim]` — already RoPE'd, only the last `lookahead_count` query rows.
- `k`: BF16, `[kv_heads, kv_len, head_dim]` — the layer's K cache after dense prefill on the speculator.
- `kv_len`: `int` — number of valid prompt keys (== prompt token count for the draft after its own dense prefill).
- `lookahead_count`: `int` — paper §3.3 N+1 (the last prompt token + N look-ahead query rows).
- `num_kv_groups`: `int` — `q_heads / kv_heads` (GQA).
- `scale`: `float` — `1 / sqrt(head_dim)`.

Output:
- `scores`: F32, `[heads, lookahead_count, kv_len]` — post-softmax per-row attention against the prompt context. Heads are query heads (kv broadcasting via `num_kv_groups`).

Algorithm (per `(q_head, q_row)`):
1. Pass 1: walk `k_pos` ∈ [0, kv_len), accumulate `score = scale · q · k`, track `max`.
2. Pass 2: walk again, write `expf(score - max)` to scratch, accumulate `denom`.
3. Pass 3: divide each scratch entry by `denom`, write to `scores`.

Implementation note: passes 2 and 3 can be fused — keep the unnormalised exponentials in shared memory if `kv_len` fits (typically it does — Qwen3.5-0.8B has kv_heads=4, head_dim=256; one row is 4 × kv_len floats = 21 KiB at kv_len=1354, comfortably in LDS). For longer prompts, fall back to the three-pass form. Matches the streaming-softmax structure already in the dense kernel.

The look-ahead kernel does **not** apply causality — every look-ahead query row attends to all `kv_len` prompt keys. This is what the paper requires (§3.3): the last prompt token attends to its full preceding context, and each look-ahead token attends to the full prompt (not to other look-ahead tokens — those keys aren't in this kernel's K input anyway).

CUDA + Metal stubs: error-only stubs that return "not implemented" at runtime. The Phase C scope is HIP-only; CUDA/Metal can land later.

**Cost:** `lookahead_count / kv_len ≈ 5/1354 ≈ 0.4 %` of the dense prefill QK cost on the speculator. One launch per layer (the speculator has 6 full-attention layers in Qwen3.5-0.8B; 8 in 9B).

**Test:** `crates/kernel-ffi/tests/specprefill_lookahead_attention_parity.rs`. Construct synthetic Q, K with known values; verify F32 scores match a CPU reference (numpy-style softmax) within 1e-5. Also verify the per-row sum equals 1.0 (softmax invariant).

## Component 2: speculator prefill driver (new)

A new public function on `crates/runner/src/prefill_engine.rs`:

```rust
pub fn prefill_with_lookahead_attention(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    lookahead_count: usize,
) -> Result<PrefillWithLookaheadResult>
```

Where `PrefillWithLookaheadResult` adds to `PrefillResult`:

```rust
pub struct PrefillWithLookaheadResult {
    pub base: PrefillResult,
    /// Per full-attention layer: [num_q_heads, lookahead_count, kv_len] F32.
    /// One Vec entry per full-attention layer, in layer order.
    pub layer_scores: Vec<Vec<f32>>,
}
```

Convention: `lookahead_count = cli.specprefill_lookahead + 1`. The `+1` accounts for the last prompt token's query row, which is what the paper's importance signal includes alongside `lookahead` look-ahead-token query rows. With the default `--specprefill-lookahead = 4` this is `lookahead_count = 5`.

Internals:
1. `prefill_inner` on the full prompt as today. After this, each full-attention layer's K cache holds `prompt_len` entries, and the post-RoPE Q for the last prompt position sits in workspace.
2. For `i in 0..(lookahead_count - 1)`: greedy-sample the next token from the previous step's logits, run a single decode step on the speculator, capture its post-RoPE Q row at each full-attention layer. Expose the capture via a new internal `decode_step_with_query_capture(token_id, layers: &[usize]) -> (logits, Vec<query_row_per_layer>)` on the speculator's `DecodeEngine` — the per-layer post-RoPE Q is already a tensor the existing decode path computes; capturing it is a host copy.
3. Stack the captured Q rows alongside the prompt's last-row Q → `[lookahead_count, head_dim]` per query head per layer.
4. Per full-attention layer, launch `supersonic_qwen35_lookahead_attention_scores` with those Q rows and the layer's K cache, passing `kv_len = prompt_len` so the kernel only attends to the prompt context. (The decode steps in step 2 do append K rows past `prompt_len`; we ignore them by setting `kv_len = prompt_len`.) Capture scores into per-layer F32 buffers and copy host-side at the end.

**Why this is the right shape:** the look-ahead query rows carry "what would the model attend to next" because they came from real decode steps with the actual sampled tokens. K is fixed at the prompt boundary so the per-row softmax denominators are computed against the same prompt context for every row — exactly what the paper's aggregation expects. We considered (and rejected) the simpler "feed lookahead-many padding tokens through dense prefill" alternative: padding tokens have arbitrary IDs and their attention rows would not match what the model would actually compute during real decode, so the importance signal would be a poor proxy.

**Cost:** `lookahead_count - 1` decode steps on the 0.8B draft (~3 ms each on RX 7900 XTX from existing decode benchmarks) + per-layer look-ahead kernel launches (negligible). Total speculator overhead: ~30 ms for the default `lookahead = 4` on a 1354-token prompt.

## Component 3: target sparse prefill (extension)

A new public function on `crates/runner/src/prefill_engine.rs`:

```rust
pub fn prefill_kept(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    kept_positions: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
) -> Result<PrefillResult>
```

Implementation — `prefill_inner` learns one new optional parameter, `kept_positions: Option<&[u32]>`:

1. **Embedding compaction:** at the entry of the prefill pipeline, after `embed_tokens` lookup but before any layer compute, if `kept_positions` is `Some`, gather rows from the embedding table at `prompt_ids[kept_positions[i]]` instead of `prompt_ids[0..T]`. The result is `[len(kept), hidden_dim]`. Done host-side or via a small `gather_rows` HIP kernel — host-side is fine for prefill (one-shot copy of `len(kept)` rows × 4 KiB BF16 ≈ 700 KiB at kept=678).

2. **RoPE indirection:** every call site that today invokes `apply_rope_prefill` switches to `apply_rope_prefill_indirect` when `kept_positions` is `Some`, passing `pos_ids = kept_positions` uploaded as `ScalarType::U32`. The new kernel was landed in Phase B with parity tests — it produces bit-identical output to the dense kernel when `kept_positions == [0, 1, ..., T-1]`.

3. **Causal mask:** unchanged. The full-attention prefill kernel computes a lower-triangular mask via `causal_limit = min(kv_len, seqlen_offset + q_pos + 1)` (full_attention.hip:282). Over the compacted sequence this is exactly correct: kept token at compacted slot `i` sees compacted slots `0..=i`, which corresponds to prompt positions `kept_positions[0..=i]`. Phase B's parity test verified this.

4. **KV cache layout after sparse prefill:** the layer's K and V caches hold `len(kept)` entries, each tagged with its source prompt position via the RoPE rotation it received. `kv_filled = len(kept)` post-prefill.

5. **Linear-attention layers:** Qwen3.5 architecture has 24 (0.8B) / 32 (9B) layers split as `[linear, linear, linear, full]` × N. The linear-attention layers do *not* use RoPE and are stateful in time. **Open question 1**: do they sparsify cleanly? The paper deals with full-attention models; Qwen3.5's linear-attention layers compute a recurrent state, and dropping middle tokens could change the converged state in a way that doesn't have an interpretation analogous to "sparse causal attention." See Risks section.

## Component 4: orchestrator engine (new)

`crates/runner/src/specprefill_engine.rs`, ~600 LoC. Mirrors `qwen35_dflash_engine.rs`:

```
fn run_specprefill(cli: &Cli, model_variant, entry, ordinal, total_vram) -> Result<()> {
    // 1. Validate CLI: --specprefill-draft-dir set, batch_size==1, no --kv-fp8 (Phase C).
    // 2. Load tokenizer + target config + draft config.
    // 3. VRAM check: draft fixed + target fixed + KV (using kept count upper bound from
    //    crates/runner/src/specprefill.rs::keep_count) + scratch + overhead_factor.
    //    Bail with a useful message if the budget is exceeded.
    // 4. Load target weights (BF16).
    // 5. Build target DecodeEngine (existing path, prefill_chunk_size + kv_chunk_size from CLI).
    // 6. Load draft weights into a separate Qwen35Weights + draft DecodeEngine.
    //    Both engines share gpu-hal device context (ordinal).
    // 7. Tokenise prompt (target tokenizer; fail if draft tokenizer differs in vocab_size or
    //    BOS handling — strict same-family for Phase C).
    // 8. Speculator phase:
    //    a. Build SelectionConfig from CLI flags (defaults: keep=0.50, chunk=32, pool=5,
    //       prefix=4, suffix=4, lookahead=4).
    //    b. Call prefill_with_lookahead_attention on the draft.
    //    c. Aggregate per-layer scores via the formula from oracle/specprefill_oracle.py:
    //       max over heads → max over layers → mean over lookahead steps → [T] F32.
    //       Implemented host-side as a small loop in specprefill_engine.rs.
    //    d. Call select_kept_positions (already in specprefill.rs) → Vec<u32>.
    //    e. (Optional) Free draft weights to claw back VRAM:
    //       --specprefill-unload-draft (default off; opt-in for tight memory).
    // 9. Target phase: prefill_kept(prompt_ids, &kept_positions) → PrefillResult.
    // 10. Hand off to standard decode loop with seqlen_offset = prompt_ids.len()
    //     (NOT kept_positions.len()).
}
```

Decode glue: the existing decode path takes a `seqlen_offset` argument per call (verified in the prefill engine — see `apply_rope_prefill_indirect` and the dflash engine's `decode_step_with_taps_kernel(tok, seqlen_offset, ...)` call sites). The orchestrator passes `seqlen_offset = prompt_ids.len()` for the first generated token (not `kept_positions.len()`), and increments by 1 per decoded token as today. **Confirm during PR C3 prep** by reading the full decode loop to make sure no internal path uses `kv_filled` as a position substitute (Open Question 2 in Risks).

## Component 5: CLI surface

In `crates/runner/src/main.rs`:

```rust
#[clap(long)]
pub specprefill_draft_dir: Option<PathBuf>,

#[clap(long, default_value = "0.50", value_parser = clap::value_parser!(f32))]
pub specprefill_keep_ratio: f32,

#[clap(long, default_value = "32")]
pub specprefill_chunk_size: usize,

#[clap(long, default_value = "5")]
pub specprefill_pool_window: usize,

#[clap(long, default_value = "4")]
pub specprefill_lookahead: usize,

#[clap(long, default_value = "4")]
pub specprefill_always_keep_prefix: usize,

#[clap(long, default_value = "4")]
pub specprefill_always_keep_suffix: usize,

#[clap(long, default_value_t = false)]
pub specprefill_unload_draft: bool,
```

Validation in `run_specprefill`:
- `keep_ratio` ∈ [0.05, 1.0]. Below 0.05 the forced prefix/suffix bands dominate and the "selection" is mostly forced; above 1.0 is meaningless.
- `pool_window` odd, ≥ 1.
- `lookahead` ∈ [1, 16].

## Component 6: integration test

`crates/runner/tests/specprefill_qwen35_9b_parity.rs`. Pattern: shell-out to a release-built `runner` binary, twice — once with `--specprefill-draft-dir` (sparse) and once without (dense). Skipped when the model dirs aren't set in env (`SUPERSONIC_QWEN35_9B_DIR`, `SUPERSONIC_QWEN35_0_8B_DIR`) or when HIP isn't compiled.

Assertions on the last prefill step's logits:
1. `argmax(last_step_logits)` matches between sparse and dense runs (primary correctness gate per Phase A2).
2. `cossim(last_step_logits) ≥ 0.90` (regression backstop; Phase A2 measured 0.927 at keep=0.50 long-prompt, so 0.90 has ~30 % headroom against measured noise).
3. `top5_overlap ≥ 4/5` (Phase A2 measured 5/5 at keep=0.50 long-prompt; 4/5 absorbs the short-prompt 3/5 edge case if it appears on the test prompt).

Decode-stream parity is **not** part of this test — once the prefill diverges, decode-step divergence accumulates and is a separate quality concern (Phase A's Observation 1). The test's job is "did the prefill produce a usable KV cache for greedy decode?" and assertion 1 answers that.

Reference shell-out pattern: `crates/runner/tests/qwen36_moe_kv_fp8_parity.rs`.

## Data flow summary

| Step | On | Reads | Writes | Cost (1354-tok prompt, 9B+0.8B) |
|---|---|---|---|---|
| 1. Draft dense prefill | GPU | prompt_ids, draft weights | draft KV cache, last logits | ~700 ms |
| 2. Draft 4 lookahead decode steps + Q capture | GPU | draft KV cache, lm_head | 4 × Q row per layer, 4 next-token logits | ~12 ms |
| 3. lookahead_attention_scores per layer | GPU | Q rows, draft K cache | F32 [heads, 5, T] per layer → host | ~5 ms |
| 4. Importance aggregation | CPU | per-layer scores | [T] F32 importance | < 1 ms |
| 5. select_kept_positions | CPU | importance | Vec<u32> kept_positions | < 1 ms |
| 6. Target sparse prefill | GPU | prompt_ids, kept_positions, target weights | target KV cache (kept-len), last logits | ~50 % of dense @ keep=0.50 |
| 7. Decode | GPU | target KV cache, lm_head | tokens | unchanged |

Total speculator overhead: ~720 ms on the 0.8B draft. Dense target prefill on 1354 tokens for 9B is ~6 s on RX 7900 XTX (estimate from existing benchmarks). At keep=0.50 the sparse target prefill is ~3 s. Net wallclock: 720 ms + 3 s = 3.7 s vs 6 s dense → ~38 % TTFT win at keep=0.50, ~58 % at keep=0.30. Numbers are estimates; benchmark in Phase C wrap-up.

## Risks and open questions

| # | Question | Mitigation |
|---|---|---|
| 1 | Linear-attention layers' state diverges under non-contiguous sparse prefill — the recurrent state at compacted slot `i` doesn't equal the dense state at prompt position `kept_positions[i]`. | Phase A2 measured end-to-end logits cossim ≥ 0.927 at keep=0.50 on the 9B model — the linear-attention layers were exercised through HF transformers and the algorithm still preserved argmax. We rely on the same empirical observation. If the SuperSonic linear-attention prefill diverges further (kernel-level differences from HF), we'll see it in the integration test's cossim or argmax assertions. Worst case we add a "skip linear layers in sparse prefill" branch — but the Phase A numbers say we don't need to. |
| 2 | Decode position cursor: does the existing decode path use `kv_filled` or a separate position counter? If `kv_filled`, sparse prefill (where `kv_filled = len(kept) < original_T`) will write decode tokens at the wrong RoPE position. | 30-min code read of `decode_engine.rs` before kernel work. If the decode path needs a separate `position_offset` field, add it. (Reading the existing code, `seqlen_offset` is already a per-call argument on the decode kernels — this should "just work.") |
| 3 | VRAM: target 9B BF16 ≈ 18 GiB + draft 0.8B BF16 ≈ 1.6 GiB + KV (~70 MiB at len=1354 keep=0.50) + scratch ≈ 21 GiB peak on a 24 GiB GPU. | Tight. Budget calc in `run_specprefill` is the gate. `--specprefill-unload-draft` (default off) frees the draft after step 5 to recover ~1.6 GiB before target prefill. If still tight, fall back to Qwen3.5-4B target (Phase A baseline). |
| 4 | Tokeniser mismatch between draft and target on cross-family pairs. | Strict same-family check in `run_specprefill` for Phase C. Cross-family (Qwen3.5 → Qwen3.6) is deferred to a later phase. |
| 5 | The look-ahead kernel computes softmax over a single layer's Q-row attention to the prompt K. The aggregation `max heads → max layers → mean lookahead` is done on the host. Cost of host-side aggregation at e.g. 8 layers × 16 heads × 5 lookahead × 1354 tokens × 4 bytes = ~3.4 MiB transfer. | Negligible at PCIe 4.0 speeds (< 1 ms). |
| 6 | Codex review bot will flag inline P1/P2 issues on PR open. | Expected; address before merge per `~/.claude/projects/-home-deano-projects-SuperSonic/memory/reference_codex_review_bot.md`. |

## Implementation sequence

Three landable PRs:

**PR C1 — look-ahead attention kernel + FFI + parity test.** ~250 LoC HIP + ~150 LoC Rust. Standalone, testable in isolation against a CPU reference. CUDA/Metal stubs return "not implemented."

**PR C2 — `prefill_with_lookahead_attention` + `prefill_kept` plumbing in prefill_engine.** Wires the look-ahead kernel into the speculator prefill path; adds the `kept_positions` parameter to `prefill_inner` and the new public wrapper. ~400 LoC Rust. Unit tests: dense path with `kept_positions = None` is unchanged (cossim 1.0 vs current); dense path with `kept_positions = [0..T]` matches the `None` path bit-identically.

**PR C3 — `specprefill_engine.rs` + CLI flags + integration test.** ~600 LoC Rust. Adds the orchestrator and the end-to-end test that runs full vs sparse prefill on the 9B model.

PRs land in order; each is reviewable independently. Total scope: ~1.4k LoC.

## What this design does not specify

- The decode-path changes if open question 2 turns out to need a new `position_offset` field. Decided in PR C3 prep.
- The exact VRAM budget formula (depends on whether `unload_draft` is the default — currently off). Will pin down in `run_specprefill` review.
- Benchmarking scaffolding (TTFT measurement vs dense prefill). Out of scope for the design; will land separately as a perf script.
- Cross-family extension. Deferred.

## Acceptance criteria

A merged PR C3 satisfies:
1. `cargo test -p runner --release` passes (existing tests unchanged).
2. The new `specprefill_qwen35_9b_parity` integration test passes when `SUPERSONIC_QWEN35_9B_DIR` and `SUPERSONIC_QWEN35_0_8B_DIR` are set:
   - `argmax_match == 1` at the last prefill position.
   - `cossim ≥ 0.90`.
   - `top5_overlap ≥ 4`.
3. Codex review bot's P1 comments are resolved before merge.
4. The dense prefill code path (no `--specprefill-draft-dir`) is byte-identical to the current `main` branch behaviour. Verified by re-running an existing prefill regression check.
