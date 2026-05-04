# Qwen3.6-MoE: MTP + SpecPrefill composition

**Date:** 2026-05-04
**Branch:** `research/qwen36-mtp-specprefill`
**Builds on:** PR #211 (`research/qwen36-persistent-cache-pos`, just merged) which plumbed `cache_pos` through the persistent decode megakernel.

## Goal

Lift the `bail!` in `crates/runner/src/qwen36_moe/engine.rs:165` that prevents `--specprefill-draft-dir` and `--speculative-decode` from being used together on Qwen3.6-MoE, with correctness guaranteed by a (rope_pos, cache_slot) audit through the speculative path and a dedicated parity test.

The R1 mutex gate's stated reason — "MTP uses the persistent decode kernel which takes only `position` (no `cache_pos` decoupling)" — is now stale: PR #211 added `cache_pos` to the persistent kernel. But naively removing the bail would silently produce garbage drafts because `loop_state.position` is overloaded as the *compact KV slot* in SpecPrefill mode and the speculative driver currently uses it as both the draft cache slot AND the RoPE timeline base.

## Background: where positions diverge

After SpecPrefill prefill at keep_ratio=0.50 over a 1393-token prompt:

| Quantity | Dense | SpecPrefill |
|---|---|---|
| `loop_state.position` (compact slot index) | 1393 | ~700 |
| Absolute RoPE position of next-to-write token | 1393 | 1393 |

Today `engine.rs::decode_text` already correctly computes a per-step `(rope_pos, chain_cache_pos)` tuple at line 389-400 and passes it into `run_chain_step`. But:

- The R1 chain step API uses `Qwen36ChainStep::{ position: i32, cache_pos: Option<i32> }` with `None ⇒ inherit`. The `Option<i32>` is a stopgap from when only the chained kernel supported `cache_pos`; now the persistent kernel does too, so the inheritance sentinel is a bit of premature abstraction we can clean up.
- The speculative path in `spec_verify.rs::run_speculative_extension` passes `loop_state.position` to the speculative driver as `base_position`. The driver uses `base_position + k` for both the MTP draft chain's RoPE rotation and the verify replay positions. In SpecPrefill mode this is wrong: RoPE should rotate at `prompt_len_full + n_generated_so_far + k`, not `compact_slot_count + k`.

## Design

### 1. `PositionPair` type

In `crates/runner/src/qwen36_moe/types.rs` (alongside `MultiLayerGeom`):

```rust
/// Per-step position pair. Decouples RoPE rotation timeline from
/// the KV cache slot index, which differ in SpecPrefill mode where
/// kept tokens land at compact slots while still rotating at their
/// original prompt positions.
///
/// In dense decode and pre-SpecPrefill code paths the two are equal;
/// the `dense(p)` constructor is the one-arg shortcut for that case.
#[derive(Debug, Clone, Copy)]
pub struct PositionPair {
    /// Absolute RoPE position. Always advances on the absolute
    /// timeline: `prompt_len_full + n_generated_so_far` for
    /// generation steps, `kept_positions[step]` for SpecPrefill
    /// prefill steps.
    pub rope: i32,
    /// KV cache slot index. Equals `rope` in dense decode; equals
    /// the compact slot count (`loop_state.position`) in SpecPrefill
    /// mode.
    pub cache: i32,
}

impl PositionPair {
    /// Dense-decode shortcut: rope and cache slot agree.
    pub const fn dense(p: i32) -> Self { Self { rope: p, cache: p } }
    /// Decoupled SpecPrefill / MTP-style pair.
    pub const fn split(rope: i32, cache: i32) -> Self { Self { rope, cache } }
}
```

No `Option`-shaped sentinel. Every consumer takes a `PositionPair` and uses both fields — no "what does None mean again" branch.

### 2. `chain.rs` refactor

Replace the existing fields:

```rust
// before
pub(crate) position: i32,
pub(crate) cache_pos: Option<i32>,

// after
pub(crate) position: PositionPair,
```

`run_chain_step` body shrinks: drop the `let cache_pos_arg = args.cache_pos.unwrap_or(CACHE_POS_INHERIT);` line, replace internal references with `args.position.rope` / `args.position.cache`. The chained-fallback branch's `if let Some(cache_pos) = args.cache_pos` becomes `if args.position.rope != args.position.cache` — non-trivial only when they differ.

The `CACHE_POS_INHERIT` constant in `qwen36_moe_persistent_decode.rs` stays — the FFI safe wrapper still takes a raw `i32` and `INHERIT` is the right sentinel at the C boundary.

Two call sites to update in `engine.rs::decode_text`:
- Dense: `position: PositionPair::dense(loop_state.position)`
- SpecPrefill: `position: PositionPair::split(rope_pos, loop_state.position)`

### 3. Speculative extension wiring

`Qwen36SpeculativeExtension` gains a `base_position: PositionPair` field replacing the implicit `args.loop_state.position` use. The caller (engine.rs:526) constructs the pair using the same `(rope_pos, chain_cache_pos)` arithmetic that the dense decode loop already computes at `engine.rs:389-400` — the absolute rope for the just-sampled token (= `kept_positions[step]` during prefill, = `prompt_ids.len() + gen_off` during generation) and the compact `loop_state.position` slot. Factoring this into a `current_position(step, loop_state, kept_positions, effective_prompt_len, prompt_ids.len()) -> PositionPair` helper avoids drift between the chain step and the speculative extension call sites — they consume the same pair.

Inside `run_speculative_extension`, the call to `run_speculative_decode_step{,_batched}` passes `base.rope` as the `base_position` argument the driver uses for RoPE math (`base.rope + k`), and `base.cache` is threaded into the verify replay's `run_spec_chain_step` calls so accepted draft tokens write at the right compact slot.

The MTP draft chain itself (`mtp.rs`) is unchanged: it already takes a single `base_position` and does `position = base_position + k` (RoPE) with `cache_pos = k` (its own per-chain cache, separate from base). When we pass `base.rope` instead of the overloaded compact slot, RoPE math is automatically correct.

### 4. `run_spec_chain_step` thread-through

`Qwen36SpecChainStep::position: i32` becomes `position: PositionPair`. Inside, the persistent path call gets `position.rope` and the new `cache_pos` arg gets `position.cache`. The chained fallback in spec_verify currently calls `run_chained_decode_fast` (no cache_pos) — for SpecPrefill+MTP we need to switch that to `run_chained_decode_fast_with_cache_pos` when `position.rope != position.cache`. Mirror the chain.rs branching exactly.

### 5. Lift the gate

`engine.rs:165`:

```rust
// before
if keep_mask.is_some() && speculative_decode {
    anyhow::bail!("SpecPrefill ... cannot be combined with --speculative-decode ...");
}
// after — diagnostic only
if keep_mask.is_some() && speculative_decode {
    eprintln!(
        "[specprefill+mtp] composed run: keep_ratio depends on \
         drafter, k={QWEN36_NUM_SPECULATIVE_TOKENS}. Position threading: \
         rope on absolute timeline, cache on compact slot."
    );
}
```

The eprintln is conservative — leaves a forensic breadcrumb in user logs without spamming dense-mode runs. Can be removed in a follow-up once the combined mode has miles on it.

### 6. Parity test

`crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs`. Mirrors `specprefill_qwen36_moe_cosine_parity.rs` but with `--speculative-decode` set:

- `cosine_qwen36_moe_mtp_keep_100_near_identity`: keep=1.00, cossim ≥ 0.99 against dense+MTP baseline (NOT against dense-only — MTP introduces its own draft-vs-base interaction independent of SpecPrefill that we don't want to chase).
- `cosine_qwen36_moe_mtp_keep_050_topic_alignment`: keep=0.50, cossim ≥ 0.65, top-5 overlap ≥ 3 against dense+MTP baseline.

Test env vars match the existing tests: `SUPERSONIC_QWEN36_35B_A3B_DIR` + `SUPERSONIC_QWEN35_0_8B_DIR`. Skips silently when unset.

The dense+MTP baseline is run inside the test (same machinery as the existing parity test, just with `--speculative-decode` added) so we're isolating the SpecPrefill-on-top-of-MTP interaction, not the MTP-alone sampling noise.

### 7. Audit memo

`docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md`. Authored alongside the implementation, documents:

- The (rope, cache) split semantics across the four modes: dense, MTP-only, SpecPrefill-only, SpecPrefill+MTP.
- Where each timeline originates: rope = absolute (drafter-input space), cache = compact (KV-physical space).
- The invariants `PositionPair` enforces vs. the R1 `(position, Option<cache_pos>)` shape: `rope` is always populated, `cache` is always populated, no inheritance ambiguity.
- Loose ends / known gaps: anything that surfaced during implementation that's worth a follow-up branch.

## Testing

| Test | Purpose | Existing? |
|---|---|---|
| `specprefill_qwen35_9b_cosine_parity` (3 cases) | Qwen3.5-9B regression — different compile unit, sanity check after API touch | ✓ regression |
| `specprefill_qwen36_moe_cosine_parity` (2 cases) | Qwen3.6-MoE SpecPrefill-only regression — `cossim=1.0` at keep=1.00 must still hold | ✓ regression |
| `qwen36_moe_speculative_decode_*` (existing MTP tests) | MTP-only regression | ✓ regression |
| `specprefill_qwen36_moe_mtp_cosine_parity` (2 cases) | New: combined-mode parity | new |

Build: `HIP_ARCH=gfx1100 cargo build --release` clean.

## Out of scope

- Performance optimization of the combined mode. R1 baseline shows SpecPrefill alone gives 2.78× TTFT speedup; MTP alone gives ~1.3-1.5× decode speedup on accepted tokens. Combined speedup is theoretical headline ~3.5-4× but actual perf is a separate measurement project.
- DFlash / batched-spec-verify with SpecPrefill — the current spec_verify.rs has both `sequential` and `batched` paths; we update both API-wise but only test `sequential` in the parity test (batched needs its own follow-up since linear-attn snapshot/restore semantics in SpecPrefill mode warrant separate validation).
- Lifting any other R1 caveats. We touch only the MTP gate.

## Files changed

| File | Change |
|---|---|
| `crates/runner/src/qwen36_moe/types.rs` | `+PositionPair` type |
| `crates/runner/src/qwen36_moe/chain.rs` | `Qwen36ChainStep::position: PositionPair`, body simplification |
| `crates/runner/src/qwen36_moe/engine.rs` | call-site update + lift the bail |
| `crates/runner/src/qwen36_moe/spec_verify.rs` | `Qwen36SpeculativeExtension::base_position: PositionPair`, `Qwen36SpecChainStep::position: PositionPair`, replay thread-through, chained-fallback branch on rope ≠ cache |
| `crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs` | new combined-mode parity test |
| `docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md` | new audit memo |
