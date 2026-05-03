# SpecPrefill (arXiv 2502.02789) — implementation feasibility for SuperSonic

**Branch:** `research/specprefill`
**Status:** investigation only; no code changes yet.
**Paper:** [Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789), Liu et al., ICML 2025.
**Reference impl:** <https://github.com/Jingyu6/speculative_prefill> — vLLM monkey-patch.

## TL;DR

The paper's algorithm is a clean fit for SuperSonic's prefill path on Qwen3.5 / Qwen3.6-MoE: same-family small + large checkpoints already coexist in the runtime, the prefill engine is in-tree (`crates/runner/src/prefill_engine.rs`, ~3.4k LoC), and DFlash already has a "draft model alongside target" pattern we can borrow from. The kernel-side work is **non-trivial but bounded**: the prefill RoPE and attention masking both currently assume contiguous positions, and we don't expose attention scores to the host. Ballpark: **2–4 weeks for a complete Qwen3.5 implementation**, with optional later extension to Qwen3.6-MoE. **It does not affect the persistent decode megakernel** — it's a pure prefill optimization.

## What the paper does

1. **Speculator forward + N look-ahead steps** on the prompt (paper uses Llama-3.1-8B alongside 70B/405B targets).
2. **Cache the speculator's Q and K** during the look-ahead.
3. **Token importance** = the look-ahead tokens' attention scores against the context, aggregated as `mean_lookahead( max_layers( max_heads( score ) ) )`.
4. **Selection**: 1-D average-pool the per-token scores, then top-K within fixed chunks. Keep ratios benchmarked at 10/30/50/70/90 %.
5. **Target prefill on the selected subset only**, but with **original (non-contiguous) position IDs** — RoPE rotates each kept token by its source-prompt position, not its compacted slot index.
6. **Decode continues from the original tail position** — e.g. if the prompt had 10 tokens and we kept 5, decode positions are still 10, 11, 12 …, not 5, 6, 7 …

The paper headline: 7.66× TTFT on Llama-405B / 8× H100 / TP=8 at 10 % keep ratio, with LongBench quality at 99.7 % of baseline.

## Why this fits SuperSonic well

| Question | Answer |
|---|---|
| Is prefill the bottleneck? | At long context yes — for Qwen3.6-35B-A3B INT4 at e.g. 8k prompt, prefill dominates over the persistent decode kernel's win. |
| Do we have a same-family speculator? | Yes — Qwen3.5-0.8B/2B already live in the registry; both are natural drafts for Qwen3.6-35B-A3B (Qwen3.6-MoE shares the Qwen3-Next attention shape) and for Qwen3.5-9B. |
| Two-model orchestration in the runtime? | Partially — DFlash (`crates/runner/src/qwen35_dflash_engine.rs`) already pairs a 5-layer draft checkpoint with a 9B target on one GPU. The "load two model dirs and share a stream" boilerplate is in place. |
| Does it touch the persistent megakernel? | No — SpecPrefill is purely prefill-side. The decode loop is unchanged. Orthogonal to all the FP8/INT4/MoE-VMM work that just landed. |
| Memory? | gfx1100 24 GiB: Qwen3.5-0.8B BF16 ≈ 1.6 GiB; tight but fits alongside Qwen3.6-35B-A3B INT4 (~17.5 GiB) plus KV. Could fall back to FP8/INT4 speculator if needed (or share the speculator's weights with a pre-existing `--dflash` draft load if family allows). |

## What's missing in the codebase

### 1. Prefill RoPE with arbitrary position IDs

`kernels/full_attention.hip::supersonic_qwen35_hip_apply_rope_prefill` (FFI in `crates/kernel-ffi/src/prefill_ffi.rs:670`) takes `seq_len` and indexes `cos_table` / `sin_table` by *position-in-stream*. SpecPrefill needs RoPE keyed by an *original-position array* `pos_ids[i]` (with `i` ∈ kept-token slots). Two ways to add this:

- **Option A (cheap)**: precompute the cos/sin gather on the host, pass per-token cos/sin slices to the existing kernel. Wastes 2× cos/sin bandwidth but no kernel changes.
- **Option B (clean)**: add `apply_rope_prefill_indirect(..., pos_ids: *const c_int)` that gathers `cos_table[pos_ids[t]]` per token in the kernel. One new HIP entry, ~30 LoC.

**Recommendation: Option B.** Mirrors the same change needed for FlashAttention-style sparse prefill in the future; cheap to add now.

### 2. Attention mask for non-contiguous positions

The full-attention prefill kernel (`kernels/full_attention.hip` and the WMMA prefill GEMMs at `kernels/full_attention_4b.hip`) build the causal mask implicitly from `seq_len` (lower-triangular). Once we drop tokens from the middle, we need to either:

- **Option A**: Pass a `bool[seq_kept × seq_kept]` mask (or a packed bitmap). Simple but `seq_kept`-quadratic memory.
- **Option B**: Re-derive the mask in-kernel from `pos_ids`: `mask[i, j] = pos_ids[j] <= pos_ids[i]`. Compact (no extra buffer), kernel-side branch.

**Recommendation: Option B**, lifted into the same kernel that does RoPE-indirect.

### 3. Exposing speculator attention scores to the host

The prefill kernels write attention to `workspace[OFF_ATTN]` then immediately consume it for the V-weighted reduction; the raw `[H, T, T]` attention tensor is never materialized. SpecPrefill needs the **softmax(QKᵀ)** for the *last (look-ahead) token's row* against the full-context K — i.e., a `[num_heads, num_layers, T]` slice per look-ahead step.

Cheapest path: add a "trace mode" to the speculator's prefill that, at the last `lookahead_count` rows of the QK matmul, writes the post-softmax row into a host-visible buffer. ~50 LoC of HIP, gated on a flag so the non-trace path stays unchanged.

Alternatively, since the speculator is *small* and we only need the attention rows for the final N tokens, we can run a **special "look-ahead" forward** that just does QKᵀ + softmax for those rows (skipping the V-weighted reduction and the rest of the layer). Different but simpler.

### 4. Two-model orchestration

DFlash (`crates/runner/src/qwen35_dflash_engine.rs`) already loads a target + a draft model into the same `gpu-hal` device context and runs them on the same HIP stream. SpecPrefill needs the same shape but:
- Draft is a **full model** (Qwen3.5-0.8B or 2B), not a 5-layer head — bigger weights but still fits.
- The two models run **sequentially** (draft → selection → target) per request, not interleaved.
- Selection runs on the **host** (it's a tiny scan over scalar importance scores).

Most of this is `Vec<LayerBuffers>` ownership plumbing — no new HAL primitives needed.

### 5. Selection logic (host-side)

Pure CPU work. ~80 LoC of Rust:
- 1-D average-pool the importance scores with a small window (paper uses 5–10).
- Within each chunk (paper uses 32 or 64 tokens), select the top-K-fraction by score.
- Always keep the BOS / system-prompt prefix and the last few tokens (recommended in §3.4 for stability).
- Emit the `pos_ids: Vec<i32>` of kept tokens.

### 6. CLI + registry surface

- `--specprefill <draft-model>` flag pointing at a same-family small model dir.
- `--specprefill-keep-ratio <0.10..0.90>` (paper benchmarks at these five values).
- `--specprefill-lookahead <N>` (paper uses 4 by default).
- Registry: probably a new `RegistryEntry` per (target, speculator) pair if we want VRAM pre-flight to be honest. Or a runtime-only check that adds the speculator's weight bytes to the budget if SpecPrefill is on.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| Cross-architecture speculator (e.g. Qwen3.5-0.8B → Qwen3.6-MoE-35B-A3B) — paper uses **same-family** drafts. There's a follow-up "Cross-Family Speculative Prefill" paper hinted at in search results, but worth verifying this combo retains quality before committing. | Start with Qwen3.5-9B + Qwen3.5-0.8B (both pure-attention same-architecture). Extend to Qwen3.6-MoE only after verifying. |
| RDNA3 `gfx1100` is far less prefill-bound than 8× H100 TP=8 (the paper's setup). The 7.66× TTFT win came from 405B+TP communication overhead, not just FLOPs. Our wins on 9B / 35B-A3B will be smaller. | Set quality bar (cossim ≥ 0.99 on LongBench-style prompt) and benchmark at long context (≥ 4k); accept whatever the speedup turns out to be. The cost is small if we land it cleanly. |
| Attention export changes the prefill kernel's hot path. Risk of regressing the existing (already validated) prefill. | Gate the trace path behind a kernel arg; non-SpecPrefill prefill calls hit the same code as today byte-for-byte. |
| Speculator + target both live in 24 GiB VRAM on gfx1100. Tight with Qwen3.6-35B-A3B INT4. | Either run speculator at INT4 (Qwen3.5-2B INT4 ≈ 1 GiB) or unload speculator weights between draft and target phases (load takes <1 s for 2B from mmap'd bake). |
| Original-position RoPE may interact badly with very long contexts (RoPE base 10M for Qwen3.6) where dropped middle tokens stretch the effective frequency span. | Measure perplexity drop empirically; the paper's LongBench numbers suggest this is fine but our context lengths are shorter so impact may differ. |

## Suggested implementation phasing

If we pursue this, three landable PRs:

**PR A — host-side scaffolding + CLI (no kernel changes, no quality)**
- Two-model loader patterned on DFlash.
- Importance-scoring host code (mock attention scores; just verify the selection + position-ID restoration math against a Python reference).
- `--specprefill-dryrun` that runs the speculator, prints the kept positions, and exits before the target runs.
- ~500 LoC Rust; one PR.

**PR B — kernel changes for RoPE + mask indirection**
- New `apply_rope_prefill_indirect` HIP entry.
- Sparse-causal mask derivation in the prefill attention kernel.
- Per-block parity test against the existing dense kernel with `pos_ids = [0, 1, ..., T-1]` (must be bit-identical).
- ~200 LoC HIP + 100 LoC Rust + 1 parity test.

**PR C — attention export + end-to-end wiring**
- Speculator look-ahead pass that materializes the attention scores for selection.
- End-to-end `--specprefill` flag: speculator run → selection → target prefill on subset → decode continues with the original-position cursor.
- LongBench-style logits cossim test at keep ratios 0.3 / 0.5 / 0.9.
- ~400 LoC Rust + small kernel tweak.

Total: roughly 1–2k LoC, three reviewable PRs. **Phase A alone is useful** as a research probe even if Phases B/C are deferred — it answers "would the kept-token set actually preserve quality on Qwen3.6-MoE on our prompts?" without any kernel work.

## What to do next

Three options in order of cost:

1. **Stop here**: investigation memo only; revisit when long-prompt TTFT becomes a perf goal.
2. **Phase A only**: ship the host-side scaffolding + a Python reference for the kept-token math, validate the quality story on a few real prompts before committing to kernels. ~1 week.
3. **Full A+B+C**: ~2–4 weeks; first practical 7900 XTX SpecPrefill on a Qwen target.

My recommendation: **Phase A first.** It's a contained, low-risk probe that answers the only real "does this work for our models on our hardware" question before anyone touches a HIP kernel. If the kept-token reconstruction matches the full prefill within cossim ≥ 0.999 on representative prompts, Phase B+C is straightforward; if it doesn't, the paper doesn't translate to our setup and we've spent 1 week instead of 4.
