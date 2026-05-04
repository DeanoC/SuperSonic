# Qwen3.6-MoE position-threading audit: dense → MTP → SpecPrefill → SpecPrefill+MTP

**Date:** 2026-05-04
**Branch:** `research/qwen36-mtp-specprefill`
**Spec:** `docs/superpowers/specs/2026-05-04-qwen36-mtp-specprefill-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-qwen36-mtp-specprefill.md`

## TL;DR

Across the four decode modes the Qwen3.6-MoE engine supports, two distinct timelines flow through the kernel: the **absolute RoPE position** (used to rotate query/key vectors) and the **KV cache slot** (used as the index into the per-layer K/V cache buffer). They agree in three of the four modes and diverge in the fourth. The R1 `(position: i32, cache_pos: Option<i32>)` shape papered over the divergence with an inheritance sentinel; the new `PositionPair { rope, cache }` carries both timelines explicitly so no consumer can silently assume they're the same.

## The four modes

| Mode | rope timeline | cache timeline | rope == cache? |
|---|---|---|---|
| Dense decode | absolute (= step index) | absolute (= step index) | yes |
| MTP-only (`--speculative-decode`) | abs base + k (verify); abs base + k (draft chain) | abs base + k (verify); k (draft chain, per-MTP-session cache) | yes (in the verify chain — base + k for both) |
| SpecPrefill-only (`--specprefill-draft-dir`) | absolute prompt position | compact slot count | **no** (this is the case PR #211 fixed at the kernel level) |
| SpecPrefill + MTP (this branch) | absolute prompt position + k | compact slot count + k | **no** |

Notes on MTP-only: the draft chain has its own per-session K/V cache (slot starts at 0 for each new draft chain) — that's a third "cache" timeline disjoint from the base model's. It happens to use `k` as the slot, which agrees with `base + k` only when `base = 0`. But the MTP draft head's cache is internal; the *base model* in MTP-only sees `base + k` for both rope and cache, which is the dense-equivalent shape.

## Where each timeline originates

- **rope** — the *drafter input space*. The model reads tokens from a string of length `prompt_ids.len()` and rotates each position's query/key at its place in that string. SpecPrefill prunes positions but does not relabel them — kept tokens still rotate at their original prompt index.
- **cache** — the *KV-physical space*. The K/V cache is laid out contiguously along its sequence axis. In dense mode the cache index equals the prompt position. In SpecPrefill mode the cache index is the count of *kept* tokens written so far, so kept position 87 might land at cache slot 43.

## What `PositionPair` enforces

- `rope` is always populated. No `Option`. No "inherit from position." The consumer knows exactly which timeline this is.
- `cache` is always populated. Same reason.
- `is_dense()` returns `true` exactly when the two agree — callers that branch on the chained sibling fns (which take only `position`) gate on this rather than `cache_pos.is_some()`.
- The two ctors `dense(p)` and `split(rope, cache)` make the call site explicit about which mode it's in.

## Where the pair is computed

One helper, `current_position(...)` in `engine.rs`, computes the pair from `(step, loop_state, kept_positions, effective_prompt_len, prompt_ids.len())`. The chain step (via `Qwen36ChainStep::position`) and the speculative extension (via `Qwen36SpeculativeExtension::base_position`) both consume the same pair — there's no second site that can drift.

## Why R1's `Option<cache_pos>` worked but ages badly

R1 added `cache_pos: Option<i32>` because only the chained kernel supported `cache_pos`; the persistent kernel inherited from `position`. The `Option` thus encoded "I want decoupling" vs "I don't care" — but the same `None` value also meant "I don't even know if decoupling is possible." Two distinct concepts mapped to one value.

PR #211 made the persistent kernel `cache_pos`-aware. The "is decoupling possible" question is settled (yes, always). What remains is "what are the two timeline values" — and that's a pair, not an option.

## The MTP draft chain stays unchanged

`crates/runner/src/qwen36_moe/mtp.rs::run_mtp_draft_chain` takes a single `base_position: i32` and uses `base_position + k` for RoPE rotation while `cache_pos = k` indexes the per-chain (fresh) MTP K/V cache. We don't touch this. The change is upstream: the speculative driver now passes `base_rope` (the absolute timeline) where it previously passed `base_position` overloaded to be either the absolute or compact value depending on the mode. With `base_rope` always being the absolute timeline, MTP draft RoPE is automatically correct in all modes.

## Where the speculative driver split happens

`crates/runner/src/qwen36_moe/speculative.rs::run_speculative_decode_step{,_batched}` now take both `base_rope: i32` and `base_cache: i32`:

- **MTP draft chain** receives `base_rope` (absolute timeline). It rotates at `base_rope + k` and writes into its own per-chain K/V cache at slot `k`.
- **Verify replay closures** receive `PositionPair { rope: base_rope + k, cache: base_cache + k }` for each verify step. These closures dispatch into `run_chain_step` (or `run_spec_chain_step` in the batched path), which pass `(rope, cache)` straight into the persistent kernel.

In dense MTP, `base_rope == base_cache == loop_state.position` and every `PositionPair` collapses to `dense(p)` — the path is bit-equivalent to the pre-change code (verified by the existing MTP regression tests). In SpecPrefill+MTP, the cache stays on the compact timeline while RoPE stays absolute.

## Empirical validation

- `specprefill_qwen36_moe_cosine_parity::cosine_qwen36_moe_keep_100_near_identity` continues to hit `cossim = 1.000000` (regression check — the SpecPrefill-only path with `MTP off`).
- `specprefill_qwen36_moe_cosine_parity::cosine_qwen36_moe_keep_050_topic_alignment` continues to pass with the same topic-alignment bar.
- `specprefill_qwen36_moe_mtp_cosine_parity` (new): two cells:
  - `keep=1.00 near-identity` against dense+MTP baseline. Bar: cossim ≥ 0.99, argmax match, top-5 overlap = 5. Test stderr line `[mtp+specprefill parity mtp+cosine keep=1.00 near-identity] cossim=...` carries the actual measurement.
  - `keep=0.50 topic-alignment` against dense+MTP baseline. Bar: cossim ≥ 0.65, top-5 overlap ≥ 3.

The keep=1.00 cell is the strongest correctness probe: when every prompt position is kept, `kept_positions[step] == step` for all steps, so `base.rope == base.cache` and the SpecPrefill+MTP run is *definitionally* equivalent to dense+MTP on the input side. Any divergence is a plumbing bug. Ditto for the chain-step path through the SpecPrefill-only parity test that PR #211 already validated at `cossim = 1.0`.

## Loose ends / known gaps

- **Batched-spec-verify in SpecPrefill+MTP** is plumbed (the batched closure receives `PositionPair` pairs and dispatches via `run_batched_spec_verify_inputs`) but the parity test above only exercises the sequential path. The linear-attn snapshot/restore semantics in SpecPrefill mode warrant separate validation when batched-verify becomes a hot path. Sequential is the production default and the `Qwen36SpeculativeExtension` switches to batched only when `linear_attn_snapshot` is `Some` — so the test as written takes the sequential path through `run_speculative_decode_step`. To exercise batched, `--batched-spec-verify` would have to be on; today there's no equivalent flag in the test harness.
- **Performance characterisation** of SpecPrefill+MTP is out of scope here. SpecPrefill-only gives ~2.78× TTFT speedup (PR #210); MTP-only gives ~1.3-1.5× decode speedup on accepted tokens. Combined headline is theoretical ~3.5-4× but actual numbers are a separate measurement project.
- **Issue #209** (keep<1.00 layer-load hang after SIGKILL'd processes) remains orthogonal: the hang is at `load_decode_layers_with_vmm_strategy`, pre-decode-kernel. The PositionPair plumbing doesn't touch that path. The OOMs we hit during testing in this branch were the same fingerprint and recovered cleanly when the GPU freed up.
- **Forensic eprintln** at `engine.rs:165` (the lifted bail) prints `[specprefill+mtp] composed run: ...` once per process when both flags are set. Conservative breadcrumb; can be removed once the combined mode has miles on it. Doesn't appear in dense/SpecPrefill-only/MTP-only runs.

## Files in this branch

- `crates/runner/src/qwen36_moe/types.rs` — `PositionPair` definition.
- `crates/runner/src/qwen36_moe/chain.rs` — `Qwen36ChainStep::position: PositionPair`, body simplification.
- `crates/runner/src/qwen36_moe/engine.rs` — `current_position()` helper, lift the bail, both call sites updated.
- `crates/runner/src/qwen36_moe/spec_verify.rs` — `Qwen36SpeculativeExtension::base_position`, `Qwen36SpecChainStep::position`, replay-pair threading, chained-fallback gate on `is_dense()`.
- `crates/runner/src/qwen36_moe/speculative.rs` — `run_speculative_decode_step{,_batched}` take `base_rope + base_cache`; closures receive `PositionPair`.
- `crates/runner/src/qwen36_moe/decode_loop.rs` — `speculative_replay_inputs` / `partial_accept_replay_inputs` take a base `PositionPair` and return `Vec<(PositionPair, u32)>`.
- `crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs` — new combined-mode parity test.
- `docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md` — this memo.
- `docs/superpowers/specs/2026-05-04-qwen36-mtp-specprefill-design.md` — design spec.
- `docs/superpowers/plans/2026-05-04-qwen36-mtp-specprefill.md` — implementation plan.
