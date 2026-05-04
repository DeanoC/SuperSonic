# Qwen3.6-MoE MTP + SpecPrefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the `engine.rs:165` mutex gate that prevents `--specprefill-draft-dir + --speculative-decode` on Qwen3.6-MoE, with correctness ensured by a `PositionPair { rope, cache }` API refactor and a new combined-mode parity test.

**Architecture:** Replace `Qwen36ChainStep::{ position: i32, cache_pos: Option<i32> }` and `Qwen36SpecChainStep::position: i32` with `PositionPair`, lifted into a new `qwen36_moe::types::PositionPair` type. Compute the pair at one helper site (`current_position(...)`) so the chain step and speculative extension consume the same pair. MTP draft chain (`mtp.rs`) is unchanged — we just feed it the absolute RoPE base instead of the compact slot.

**Tech Stack:** Rust (workspace), HIP via hipcc (no kernel changes), existing parity-test machinery (cargo test --release -p runner).

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/runner/src/qwen36_moe/types.rs` | `PositionPair` type + ctors |
| `crates/runner/src/qwen36_moe/chain.rs` | `Qwen36ChainStep::position: PositionPair`, body collapse |
| `crates/runner/src/qwen36_moe/engine.rs` | `current_position` helper, lift the bail, update both `run_chain_step` + `run_speculative_extension` call sites |
| `crates/runner/src/qwen36_moe/spec_verify.rs` | `Qwen36SpeculativeExtension::base_position`, `Qwen36SpecChainStep::position`, replay rewires, chained-fallback branching on `rope ≠ cache` |
| `crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs` | New combined-mode parity test |
| `docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md` | Position-threading audit memo |

---

## Task 1: Add `PositionPair` type

**Files:**
- Modify: `crates/runner/src/qwen36_moe/types.rs` (insert after the `MultiLayerGeom` definition around line 45)

- [ ] **Step 1: Add the type and ctors**

```rust
/// Per-step position pair. Decouples the absolute RoPE rotation
/// timeline from the KV cache slot index. They differ in
/// SpecPrefill mode where kept tokens land in compact slots while
/// still rotating at their original prompt positions; MTP-style
/// decoupling uses the same shape (RoPE = base + k, cache slot = k).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionPair {
    /// Absolute RoPE position (always-advancing timeline). Equals
    /// `kept_positions[step]` during SpecPrefill prefill, and
    /// `prompt_ids.len() + gen_offset` during generation.
    pub rope: i32,
    /// KV cache slot index. Equals `rope` in dense decode; equals
    /// the compact slot count (`loop_state.position`) in
    /// SpecPrefill mode.
    pub cache: i32,
}

impl PositionPair {
    /// Dense-decode shortcut: rope and cache slot agree.
    #[inline]
    pub const fn dense(p: i32) -> Self {
        Self { rope: p, cache: p }
    }

    /// Decoupled SpecPrefill / MTP-style pair.
    #[inline]
    pub const fn split(rope: i32, cache: i32) -> Self {
        Self { rope, cache }
    }

    /// `true` when rope and cache agree — i.e. the dense case.
    /// Useful for the chained-fallback branch in chain.rs that
    /// only needs the cache_pos sibling fns when the two diverge.
    #[inline]
    pub const fn is_dense(self) -> bool {
        self.rope == self.cache
    }
}
```

- [ ] **Step 2: Verify it compiles in isolation**

Run: `cd /home/deano/projects/SuperSonicBase-qwen36-mtp-specprefill && HIP_ARCH=gfx1100 cargo check -p runner`
Expected: compiles clean (only the type added, no consumers yet).

- [ ] **Step 3: Commit**

```bash
git add crates/runner/src/qwen36_moe/types.rs
git commit -m "qwen36-moe: add PositionPair type for (rope, cache) decoupling

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Refactor `Qwen36ChainStep` to use `PositionPair`

**Files:**
- Modify: `crates/runner/src/qwen36_moe/chain.rs` (struct definition, `run_chain_step` body)

- [ ] **Step 1: Update imports + struct**

Replace the existing `position: i32, cache_pos: Option<i32>` fields and drop the `CACHE_POS_INHERIT` import (it's no longer used at this layer):

```rust
// In imports:
use crate::qwen36_moe_persistent_decode::{LmHeadFold, PersistentScratch};
use crate::qwen36_moe_types::{
    DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom, PositionPair,
};

// In Qwen36ChainStep struct:
pub(crate) struct Qwen36ChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) moe_expert_residency: Option<&'a mut MoeExpertResidencyManager>,
    pub(crate) moe_runtime: &'a mut MoeRuntimeConfig,
    pub(crate) moe_routes: &'a mut MoeRouteRuntime,
    pub(crate) initial_hidden: &'a [u8],
    /// `(rope, cache)` for this step. Dense decode uses
    /// `PositionPair::dense(loop_state.position)`; SpecPrefill uses
    /// `PositionPair::split(rope_pos, loop_state.position)`.
    pub(crate) position: PositionPair,
    pub(crate) step: usize,
    pub(crate) is_gen_step: bool,
    pub(crate) emit_stage_timings: bool,
    pub(crate) fold: Option<LmHeadFold<'a>>,
}
```

- [ ] **Step 2: Update `run_chain_step` body**

Replace the persistent and chained branches to use `args.position.rope` / `args.position.cache` directly. The persistent branch always passes both values to the kernel; the chained branch picks the with-cache-pos sibling when `!args.position.is_dense()`:

```rust
    let mut lm_head_folded = false;
    let outputs = if let Some(scratch) = args.persistent_scratch.as_deref_mut() {
        // Persistent kernel takes (rope, cache) directly. The
        // megakernel's full-attn phase consumes cache_pos via
        // `eff_cache_pos = (cache_pos >= 0) ? cache_pos : position`,
        // so passing `position.cache` works for both dense and
        // SpecPrefill cases.
        let rope = args.position.rope;
        let cache = args.position.cache;
        if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
            drop(args.fold);
            let mut prefetch = |phase: ExpertPrefetchPhase,
                                layer_idx: usize,
                                routes: &[ExpertRoute]|
             -> Result<()> {
                handle_moe_expert_prefetch(
                    manager,
                    args.store,
                    args.moe_runtime.prefetch_mode,
                    args.moe_runtime.prefetch_ranks,
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
                    phase,
                    layer_idx,
                    routes,
                )
            };
            scratch
                .run_sparse_with_expert_prefetch(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    &mut prefetch,
                )
                .with_context(|| {
                    format!(
                        "segmented persistent sparse decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        } else {
            lm_head_folded = args.fold.is_some();
            scratch
                .run(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.fold,
                )
                .with_context(|| {
                    format!(
                        "persistent decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        }
    } else {
        drop(args.fold);
        let rope = args.position.rope;
        if !args.position.is_dense() {
            let cache = args.position.cache;
            if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        args.store,
                        args.moe_runtime.prefetch_mode,
                        args.moe_runtime.prefetch_ranks,
                        &args.moe_routes.previous_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        args.moe_routes.route_telemetry.as_mut(),
                        args.moe_routes.transition_predictors.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                run_chained_decode_fast_with_expert_prefetch_and_cache_pos(
                    args.ordinal,
                    args.geom,
                    args.layers,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.emit_stage_timings,
                    &mut prefetch,
                )
            } else {
                run_chained_decode_fast_with_cache_pos(
                    args.ordinal,
                    args.geom,
                    args.layers,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.emit_stage_timings,
                )
            }
            .with_context(|| {
                format!(
                    "chained sparse-prefill decode (step {}, rope {}, cache {})",
                    args.step, rope, cache
                )
            })?
        } else if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
            let mut prefetch = |phase: ExpertPrefetchPhase,
                                layer_idx: usize,
                                routes: &[ExpertRoute]|
             -> Result<()> {
                handle_moe_expert_prefetch(
                    manager,
                    args.store,
                    args.moe_runtime.prefetch_mode,
                    args.moe_runtime.prefetch_ranks,
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
                    phase,
                    layer_idx,
                    routes,
                )
            };
            run_chained_decode_fast_with_expert_prefetch(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.emit_stage_timings,
                &mut prefetch,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?
        } else {
            run_chained_decode_fast(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.emit_stage_timings,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?
        }
    };
```

(Note: `PersistentScratch::run` and `run_sparse_with_expert_prefetch` already take `cache_pos: i32` as a separate parameter from PR #211 — we're just stopping packaging it as `Option<i32>` at this Rust layer.)

- [ ] **Step 3: Verify chain.rs compiles in isolation against still-broken call sites in engine.rs**

Run: `cd /home/deano/projects/SuperSonicBase-qwen36-mtp-specprefill && HIP_ARCH=gfx1100 cargo check -p runner 2>&1 | head -30`
Expected: errors at the `engine.rs::run_chain_step` call site (Task 3 will fix these). Errors should mention `position: PositionPair` mismatch with `position: i32`.

- [ ] **Step 4: Commit**

```bash
git add crates/runner/src/qwen36_moe/chain.rs
git commit -m "qwen36-moe: replace chain Option<cache_pos> with PositionPair

The Option<i32> sentinel was a stopgap from R1 when only the chained
kernel supported cache_pos. PR #211 added cache_pos to the persistent
kernel; the inheritance ambiguity at the Rust layer is no longer
load-bearing. PositionPair carries (rope, cache) explicitly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Update `engine.rs` call sites + add `current_position` helper

**Files:**
- Modify: `crates/runner/src/qwen36_moe/engine.rs` (lines 389-400 + line 459 call site, plus new helper)

- [ ] **Step 1: Add `current_position` helper near the top of engine.rs**

After existing imports / above `decode_text`, define:

```rust
/// Compute the `(rope, cache)` PositionPair for one step of the
/// decode loop. In dense mode the rope and cache agree; in
/// SpecPrefill mode the rope tracks the absolute prompt-token
/// position (during prefill of kept tokens) or the absolute
/// generation position (after prefill ends) while the cache slot
/// is the compact `loop_state.position`.
fn current_position(
    step: usize,
    loop_state_position: i32,
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    full_prompt_len: usize,
) -> PositionPair {
    match keep_mask {
        None => PositionPair::dense(loop_state_position),
        Some(_) => {
            let rope = if step < effective_prompt_len {
                kept_positions[step] as i32
            } else {
                let gen_off = step - effective_prompt_len;
                (full_prompt_len + gen_off) as i32
            };
            PositionPair::split(rope, loop_state_position)
        }
    }
}
```

- [ ] **Step 2: Replace the inline tuple computation at engine.rs:389-400**

Replace the existing match-block that produces `(rope_pos, chain_cache_pos): (i32, Option<i32>)` with:

```rust
        // Per-step position pair. Dense mode: rope == cache. SpecPrefill
        // mode: rope on absolute timeline, cache = compact slot count.
        // See `current_position` for the arithmetic.
        let position = current_position(
            step,
            loop_state.position,
            keep_mask.as_ref(),
            &kept_positions,
            effective_prompt_len,
            prompt_ids.len(),
        );
```

- [ ] **Step 3: Update the `run_chain_step` call at engine.rs:448-464**

Replace the two fields:

```rust
            // before
            position: rope_pos,
            cache_pos: chain_cache_pos,

            // after
            position,
```

- [ ] **Step 4: Add the PositionPair import**

In engine.rs's import block:

```rust
use crate::qwen36_moe_types::{
    ..., PositionPair, ..., // alphabetize per existing convention
};
```

- [ ] **Step 5: Verify chain.rs + engine.rs compile**

Run: `HIP_ARCH=gfx1100 cargo check -p runner 2>&1 | head -30`
Expected: compiles, OR errors only at the speculative_extension call site (which Task 4 will address). Specifically, the chain.rs and engine.rs::decode_text path should both compile.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/qwen36_moe/engine.rs
git commit -m "qwen36-moe: thread PositionPair through decode_text loop

current_position() helper centralizes the (rope, cache) calculation
so the chain step and the speculative extension (Task 4) consume the
same pair. Dense mode unchanged at runtime; SpecPrefill paths now
pass an unambiguous PositionPair rather than (i32, Option<i32>).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rewire speculative_extension + spec_verify replay

**Files:**
- Modify: `crates/runner/src/qwen36_moe/spec_verify.rs` (`Qwen36SpeculativeExtension`, `Qwen36SpecChainStep`, `Qwen36SequentialSpecVerifyInput`, `Qwen36BatchedSpecVerifyInputs`, `run_spec_chain_step`, `run_sequential_spec_verify_input`, `run_speculative_extension`)
- Modify: `crates/runner/src/qwen36_moe/engine.rs` (the `run_speculative_extension` call site at ~line 526)

- [ ] **Step 1: Update `Qwen36SpecChainStep` and `run_spec_chain_step`**

```rust
pub(crate) struct Qwen36SpecChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    /// `(rope, cache)` for this verify-replay step. In dense MTP
    /// the two agree; in SpecPrefill+MTP the rope is on the
    /// absolute prompt timeline while cache is the compact slot.
    pub(crate) position: PositionPair,
    pub(crate) input: u32,
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn run_spec_chain_step(args: Qwen36SpecChainStep<'_>) -> Result<DecodeOutputs> {
    let t_embed_start = std::time::Instant::now();
    let initial_hidden = lookup_embed_row(
        args.store,
        args.weight_prefix,
        args.input as usize,
        args.geom.hidden as usize,
    )
    .with_context(|| {
        format!(
            "spec verify embed lookup token {} at rope {} cache {}",
            args.input, args.position.rope, args.position.cache
        )
    })?;
    args.stage_timings.record_embed(t_embed_start.elapsed());

    let t_chain_start = std::time::Instant::now();
    let outputs = if let Some(scratch) = args.persistent_scratch {
        scratch.run(
            args.ordinal,
            &initial_hidden,
            args.position.rope,
            args.position.cache,
            None,
        )?
    } else if !args.position.is_dense() {
        // SpecPrefill+MTP verify replay on the chained fallback
        // path: kept-token replay needs the cache_pos sibling so K/V
        // lands at the compact slot while RoPE rotates absolute.
        run_chained_decode_fast_with_cache_pos(
            args.ordinal,
            args.geom,
            args.layers,
            &initial_hidden,
            args.position.rope,
            args.position.cache,
            args.emit_stage_timings,
        )?
    } else {
        run_chained_decode_fast(
            args.ordinal,
            args.geom,
            args.layers,
            &initial_hidden,
            args.position.rope,
            args.emit_stage_timings,
        )?
    };
    args.stage_timings
        .record_chain(t_chain_start.elapsed(), &outputs);
    args.stage_timings.count_generation_step();
    Ok(outputs)
}
```

Add the import: `use crate::qwen36_moe_decode::run_chained_decode_fast_with_cache_pos;` near the top of spec_verify.rs.

- [ ] **Step 2: Update `Qwen36SequentialSpecVerifyInput` + `run_sequential_spec_verify_input`**

This wraps run_spec_chain_step and just needs to forward the `position` field type. Replace `position: i32` with `position: PositionPair` in the struct, and pass it through.

```rust
pub(crate) struct Qwen36SequentialSpecVerifyInput<'a> {
    // ... unchanged fields above ...
    pub(crate) position: PositionPair,
    pub(crate) input: u32,
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn run_sequential_spec_verify_input(
    args: Qwen36SequentialSpecVerifyInput<'_>,
) -> Result<(u32, Vec<u8>)> {
    let outputs = run_spec_chain_step(Qwen36SpecChainStep {
        ordinal: args.ordinal,
        geom: args.geom,
        store: args.store,
        weight_prefix: args.weight_prefix,
        layers: args.layers,
        persistent_scratch: args.persistent_scratch,
        stage_timings: args.stage_timings,
        position: args.position,
        input: args.input,
        emit_stage_timings: args.emit_stage_timings,
    })?;
    // ... rest unchanged ...
}
```

- [ ] **Step 3: Update `Qwen36BatchedSpecVerifyInputs` + `run_batched_spec_verify_inputs`**

Read the existing function body (around spec_verify.rs:130-200) and replace its loop's `position: i32` with `PositionPair`. The batched path receives a `Vec<(i32, u32)>` of `(position, input_token)` pairs from the speculative driver; promote that to `Vec<(PositionPair, u32)>`.

The speculative driver at `crates/runner/src/qwen36_moe/speculative.rs` constructs these pairs as `(base_position + k, input_k)`. Since the driver only knows `base_position`, the pairs it constructs are *rope-only*. The verify path needs to compute the cache slot for each (which is `base_cache + k`). Add a `base_cache: i32` argument to `run_speculative_decode_step{,_batched}` that the verify-replay closure converts into a PositionPair.

Concrete signature change in `speculative.rs::run_speculative_decode_step`:

```rust
pub fn run_speculative_decode_step<F>(
    // ... existing args ...
    base_position: i32,        // RoPE base (absolute)
    base_cache: i32,           // KV cache base (compact in SpecPrefill, equals base_position in dense)
    num_drafts: usize,
    base_step: F,
) -> Result<SpeculativeStepResult>
where
    F: FnMut(PositionPair, u32) -> Result<(u32, Vec<u8>)>,
```

The `base_step` closure now receives a `PositionPair { rope: base_position + k, cache: base_cache + k }` directly, removing per-call-site arithmetic.

In the body, every `let pos = base_position + (k as i32);` becomes `let pos = PositionPair::split(base_position + k as i32, base_cache + k as i32);` (in SpecPrefill+MTP) or just `PositionPair::dense(base_position + k as i32)` (when caller passes `base_cache == base_position`).

Apply the same shape change to `run_speculative_decode_step_batched`.

- [ ] **Step 4: Update `Qwen36SpeculativeExtension` to take a `PositionPair` base**

```rust
pub(crate) struct Qwen36SpeculativeExtension<'a> {
    // ... unchanged fields above loop_state ...
    pub(crate) loop_state: &'a Qwen36DecodeLoopState,
    /// `(rope, cache)` of the just-sampled token that the
    /// speculative pass starts from. The speculative driver uses
    /// `rope + k` for RoPE rotation of draft step k, and
    /// `cache + k` for the KV slot. In dense mode they agree; in
    /// SpecPrefill mode rope is the absolute prompt position while
    /// cache is the compact slot count.
    pub(crate) base_position: PositionPair,
    pub(crate) h_base_in: &'a [u8],
    pub(crate) first_token: u32,
    // ... unchanged below ...
}
```

In `run_speculative_extension`'s body, replace both `args.loop_state.position` arguments to `run_speculative_decode_step{,_batched}` with two args: `args.base_position.rope, args.base_position.cache`. Update the verify-input closures so they construct `Qwen36SequentialSpecVerifyInput { ..., position: pair, ... }` from the `pair: PositionPair` the speculative driver hands them.

- [ ] **Step 5: Update the call site in engine.rs**

Around line 526:

```rust
            let result = run_speculative_extension(Qwen36SpeculativeExtension {
                ordinal,
                geom: &geom,
                store: &store,
                weight_prefix,
                layers: &mut layers,
                persistent_scratch: persistent_scratch.as_mut(),
                mtp: &mut mtp,
                forward_scratch: &mut mtp_forward_scratch,
                chain_scratch: &mut mtp_chain_scratch,
                embed_w: &embed_w_buf,
                final_norm_w: &final_norm_w_buf,
                lm_head_w: &lm_head_w_buf,
                final_hidden: &mut final_hidden_buf,
                logits: &mut logits_buf,
                counter: &mut counter_buf,
                linear_attn_snapshot: linear_attn_snapshot.as_mut(),
                loop_state: &loop_state,
                base_position: position, // the PositionPair already computed earlier in the step
                h_base_in: &outputs.final_hidden_bytes,
                first_token: sampled,
                stage_timings: &mut stage_timings,
                emit_stage_timings,
            })?;
```

(`position` here is the same `PositionPair` from Task 3, computed at the top of the loop iteration. After the chain step ran at this position and we sampled, the speculative pass's `base_position` is exactly that pair.)

- [ ] **Step 6: Verify the worktree compiles**

Run: `HIP_ARCH=gfx1100 cargo build --release 2>&1 | tail -20`
Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add crates/runner/src/qwen36_moe/spec_verify.rs crates/runner/src/qwen36_moe/speculative.rs crates/runner/src/qwen36_moe/engine.rs
git commit -m "qwen36-moe: thread PositionPair through speculative path

run_speculative_decode_step{,_batched} now take base_position +
base_cache; verify-input closures receive a PositionPair directly.
Qwen36SpeculativeExtension and Qwen36SpecChainStep take PositionPair
instead of i32. Decouples MTP draft RoPE timeline from the compact
KV slot — required for SpecPrefill+MTP composition since
loop_state.position is the compact slot, not the absolute RoPE
position.

No behaviour change in dense mode (PositionPair::dense(p) makes
rope == cache and both branches collapse to the previous code).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Lift the mutex gate

**Files:**
- Modify: `crates/runner/src/qwen36_moe/engine.rs:163-174`

- [ ] **Step 1: Replace the bail with a one-line eprintln**

```rust
    // before
    if keep_mask.is_some() && speculative_decode {
        anyhow::bail!(
            "SpecPrefill (sparse prefill via --specprefill-draft-dir) cannot be \
             combined with --speculative-decode (MTP self-speculative). MTP uses \
             the persistent decode kernel, which takes only `position` (no \
             `cache_pos` decoupling), so it would write at wrong KV slots when \
             the prompt has been pruned. R1 follow-up: plumb cache_pos through \
             MTP too."
        );
    }

    // after
    if keep_mask.is_some() && speculative_decode {
        eprintln!(
            "[specprefill+mtp] composed run: rope on absolute prompt timeline, \
             cache on compact KV slot. See \
             docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md."
        );
    }
```

- [ ] **Step 2: Verify build clean**

Run: `HIP_ARCH=gfx1100 cargo build --release 2>&1 | tail -3`
Expected: `Finished release profile`.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/src/qwen36_moe/engine.rs
git commit -m "qwen36-moe: lift specprefill+mtp mutex gate

Replaces the bail at engine.rs:165 with a forensic eprintln. The
gate's premise — that the persistent kernel doesn't accept
cache_pos — is now stale (PR #211); the compact-slot vs absolute-
RoPE split (the actual interaction risk) is now handled correctly
by PositionPair threading from Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Regression check existing parity tests

- [ ] **Step 1: Confirm GPU is available**

Run: `rocm-smi --showpids 2>&1 | grep -E "PID|supersonic"`
Expected: no other supersonic processes (the `perf/qwen36-sparse-vmm-longctx-regression` worktree may be active; if so, defer to its current cell finishing — model load needs ~17 GiB VRAM).

- [ ] **Step 2: Run Qwen3.5-9B same-family regression**

Run:
```bash
SUPERSONIC_QWEN35_9B_DIR=/mnt/data/models/Qwen3.5-9B \
SUPERSONIC_QWEN35_0_8B_DIR=/mnt/data/models/Qwen3.5-0.8B \
HIP_ARCH=gfx1100 \
cargo test --release -p runner --test specprefill_qwen35_9b_cosine_parity -- --test-threads=1 --nocapture
```
Expected: 3 passed, 0 failed.

- [ ] **Step 3: Run Qwen3.6-MoE SpecPrefill-only regression**

Run:
```bash
SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B \
SUPERSONIC_QWEN35_0_8B_DIR=/mnt/data/models/Qwen3.5-0.8B \
HIP_ARCH=gfx1100 \
cargo test --release -p runner --test specprefill_qwen36_moe_cosine_parity -- --test-threads=1 --nocapture
```
Expected: 2 passed, 0 failed. `keep_100_near_identity` must still hit `cossim=1.000000`.

- [ ] **Step 4: If existing MTP-only tests exist, run them**

Run: `ls crates/runner/tests | grep -iE "mtp|speculative"` to find them, then run with the appropriate env vars. Expected: pass.

- [ ] **Step 5: Commit a marker if any test outputs are interesting**

Skip if all green. If there's a noteworthy regression-related observation (e.g. timing shift), capture it as a memo line in the audit memo (Task 9).

---

## Task 7: Write `specprefill_qwen36_moe_mtp_cosine_parity` test

**Files:**
- Create: `crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs`

- [ ] **Step 1: Build the new test by adapting the existing one**

The existing `specprefill_qwen36_moe_cosine_parity.rs` already exposes the right machinery:
- `run_supersonic_capture_logits(&[args]) -> anyhow::Result<Vec<f32>>` (lines 34-56) — runs supersonic with `--dump-last-logits` and parses the `LAST_LOGITS:` line from stdout.
- `cossim`, `argmax`, `top5` (lines 58-91) — pure helpers.
- `check_specprefill_env() -> Option<(String, String)>` (line 92+) — env var skip logic.

Write the new test as:

```rust
//! Combined-mode parity for Qwen3.6-MoE: --specprefill-draft-dir
//! together with --speculative-decode. Validates the PositionPair
//! threading (PR following #211) by running a dense+MTP baseline
//! and a sparse+MTP run through the same harness.
//!
//! Two cells:
//!  - keep=1.00 near-identity: every prompt position kept, so
//!    rope == cache for every step; should be ≥ 0.99 cossim against
//!    dense+MTP (BF16 noise floor, not bit-equal because MTP draft
//!    sampling re-orders some fused ops).
//!  - keep=0.50 topic-alignment: kept tokens land in compact slots
//!    while rotating at their original RoPE; combined-mode fluency
//!    bar (cossim ≥ 0.65, top-5 overlap ≥ 3, argmax not required).
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR or SUPERSONIC_QWEN35_0_8B_DIR
//!    unset/missing.
//!  - SUPERSONIC_SPECPREFILL_PARITY=0.

use gpu_hal::Backend;
use std::collections::HashSet;
use std::process::Command;

// Helpers identical to specprefill_qwen36_moe_cosine_parity.rs:34-91.
// Copy verbatim:
//   - run_supersonic_capture_logits (lines 34-56)
//   - cossim                         (lines 58-67)
//   - argmax                         (lines 69-80)
//   - top5                           (lines 82-86)
//   - check_specprefill_env          (lines 92+ until the next #[test])

fn run_combined_parity_check(
    target: &str,
    draft: &str,
    keep_ratio: &str,
    label: &str,
    cossim_floor: f64,
    require_argmax_match: bool,
    top5_overlap_floor: usize,
) {
    // Same prompt fixture archived in PR #211 (1393 tokens — long
    // enough that SpecPrefill prunes meaningfully, short enough
    // that the test runs in <1 min wall time).
    let prompt = std::fs::read_to_string(
        "tests/fixtures/specprefill/specprefill_c1393_target1393_actual1393.txt",
    )
    .expect("read 1393-token prompt fixture (committed in PR #211)");

    // Dense + MTP baseline (no SpecPrefill).
    let baseline_logits = run_supersonic_capture_logits(&[
        "--backend", "hip",
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", target,
        "--int4",
        "--prompt", &prompt,
        "--max-new-tokens", "1",
        "--persistent-decode",
        "--speculative-decode",
    ])
    .expect("dense+MTP baseline run");

    // SpecPrefill + MTP combined.
    let combined_logits = run_supersonic_capture_logits(&[
        "--backend", "hip",
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", target,
        "--int4",
        "--prompt", &prompt,
        "--max-new-tokens", "1",
        "--persistent-decode",
        "--speculative-decode",
        "--specprefill-draft-dir", draft,
        "--specprefill-algorithm", "cosine",
        "--specprefill-keep-ratio", keep_ratio,
    ])
    .expect("specprefill+MTP combined run");

    assert_eq!(
        baseline_logits.len(),
        combined_logits.len(),
        "vocab size mismatch ({} vs {})",
        baseline_logits.len(),
        combined_logits.len(),
    );

    let cs = cossim(&baseline_logits, &combined_logits);
    let dense_top5 = top5(&baseline_logits);
    let combined_top5 = top5(&combined_logits);
    let overlap = dense_top5.intersection(&combined_top5).count();
    let dense_argmax = argmax(&baseline_logits);
    let combined_argmax = argmax(&combined_logits);

    eprintln!(
        "[mtp+specprefill parity {label}] cossim={cs:.6} dense_argmax={dense_argmax} \
         combined_argmax={combined_argmax} top5_overlap={overlap}/5"
    );

    assert!(cs >= cossim_floor, "cossim {cs:.6} < floor {cossim_floor}");
    assert!(
        overlap >= top5_overlap_floor,
        "top-5 overlap {overlap} < floor {top5_overlap_floor}"
    );
    if require_argmax_match {
        assert_eq!(dense_argmax, combined_argmax, "argmax mismatch at {label}");
    }
}

#[test]
fn cosine_qwen36_moe_mtp_keep_100_near_identity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // keep=1.00: every position kept ⇒ rope == cache for every
    // step. Combined run should be near bit-equal to dense+MTP (BF16
    // floor — same numerical relaxation as the SpecPrefill-only
    // parity test for the same reason: MTP introduces non-trivial
    // draft-vs-base op-shape variance).
    run_combined_parity_check(
        &target,
        &draft,
        "1.00",
        "mtp+cosine keep=1.00 near-identity",
        /* cossim_floor */ 0.99,
        /* require_argmax_match */ true,
        /* top5_overlap_floor */ 5,
    );
}

#[test]
fn cosine_qwen36_moe_mtp_keep_050_topic_alignment() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    run_combined_parity_check(
        &target,
        &draft,
        "0.50",
        "mtp+cosine keep=0.50 topic-alignment",
        /* cossim_floor */ 0.65,
        /* require_argmax_match */ false,
        /* top5_overlap_floor */ 3,
    );
}
```

The "copy verbatim" instructions above the new test point to specific line ranges in the existing test (lines 34-56, 58-91, and the `check_specprefill_env` body). Use `Read` to get those lines and paste them in — do not transcribe by hand.

- [ ] **Step 2: Verify test compiles**

Run: `HIP_ARCH=gfx1100 cargo build --release --tests -p runner 2>&1 | tail -5`
Expected: clean build of test binary.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/tests/specprefill_qwen36_moe_mtp_cosine_parity.rs
git commit -m "qwen36-moe: combined-mode parity test for SpecPrefill+MTP

Two cells: keep=1.00 near-identity (cossim >= 0.99) and keep=0.50
topic-alignment (cossim >= 0.65, top-5 overlap >= 3). Validates the
PositionPair plumbing: combined run vs dense+MTP baseline. Skips
silently when env vars / GPU unavailable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Run new combined-mode parity test

- [ ] **Step 1: GPU check**

Run: `rocm-smi --showpids 2>&1 | grep -E "supersonic"`
Expected: no other GPU users.

- [ ] **Step 2: Run new test**

```bash
SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B \
SUPERSONIC_QWEN35_0_8B_DIR=/mnt/data/models/Qwen3.5-0.8B \
HIP_ARCH=gfx1100 \
cargo test --release -p runner --test specprefill_qwen36_moe_mtp_cosine_parity -- --test-threads=1 --nocapture
```
Expected: 2 passed.

- [ ] **Step 3: If keep=0.50 fails the topic-alignment bar**

Inspect the log printout (`[mtp+specprefill parity mtp+cosine keep=0.50 topic-alignment] cossim=...`). Likely causes:
- cossim < 0.65 → MTP draft acceptance is collapsing in SpecPrefill mode → check that `base.rope` is being passed correctly through `run_speculative_decode_step{,_batched}` (Task 4 step 3-4); a missed call site would mean RoPE is still on the compact timeline.
- top5_overlap < 3 → similar root cause.

If failure persists after API audit, capture the failure mode in the audit memo and adjust the floor with justification — but only after verifying no plumbing bug.

- [ ] **Step 4: If keep=1.00 fails near-identity**

`cossim < 0.99` at keep=1.00 means the dense+MTP and combined runs diverge even when every position is kept. Since `kept_positions[step] == step` at keep=1.00, `position.rope == position.cache` for every step — should be bit-equal to dense+MTP if PositionPair is wired right. A failure here is a real bug in the plumbing.

---

## Task 9: Write the position-threading audit memo

**Files:**
- Create: `docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md`

- [ ] **Step 1: Draft the memo**

```markdown
# Qwen3.6-MoE position-threading audit: dense → MTP → SpecPrefill → SpecPrefill+MTP

**Date:** 2026-05-04
**Branch:** `research/qwen36-mtp-specprefill`
**Spec:** `docs/superpowers/specs/2026-05-04-qwen36-mtp-specprefill-design.md`

## TL;DR

Across the four decode modes the Qwen3.6-MoE engine supports, two
distinct timelines flow through the kernel: the **absolute RoPE
position** (used to rotate K queries/keys) and the **KV cache slot**
(used as the index into the per-layer K/V cache buffer). They agree
in three of the four modes and diverge in the fourth. The R1
`(position: i32, cache_pos: Option<i32>)` shape papered over the
divergence with an inheritance sentinel; the new `PositionPair {
rope, cache }` carries both timelines explicitly so no consumer can
silently assume they're the same.

## The four modes

| Mode | rope timeline | cache timeline | rope == cache? |
|---|---|---|---|
| Dense decode | absolute (= step index) | absolute (= step index) | yes |
| MTP-only (--speculative-decode) | absolute base + k | absolute base + k (verify); k (draft, per-chain) | yes (verify replay); decoupled inside MTP draft, but driver math is uniform |
| SpecPrefill-only (--specprefill-draft-dir) | absolute prompt position | compact slot count | **no** (this is the case PR #211 fixed at the kernel level) |
| SpecPrefill + MTP (this branch) | absolute prompt position + k | compact slot count + k | **no** |

## Where each timeline originates

- **rope**: drafter-input space. The model reads tokens from a
  string of length `prompt_ids.len()` and rotates each position's
  query/key at its place in that string. SpecPrefill prunes
  positions but does not relabel them — kept tokens still rotate at
  their original prompt index.
- **cache**: KV-physical space. The K/V cache is laid out
  contiguously along its sequence axis. In dense mode the cache
  index equals the prompt position. In SpecPrefill mode the cache
  index is the count of *kept* tokens written so far, so kept
  position 87 might land at cache slot 43.

## What `PositionPair` enforces

- `rope` is always populated. No `Option`. No "inherit from
  position." The consumer knows exactly which timeline this is.
- `cache` is always populated. Same reason.
- `is_dense()` returns `true` exactly when the two agree —
  callers that branch on the chained sibling fns (which take only
  `position`) gate on this rather than `cache_pos.is_some()`.
- The two ctors `dense(p)` and `split(rope, cache)` make the call
  site explicit about which mode it's in.

## Where the pair is computed

One helper, `current_position(...)` in `engine.rs`, computes the
pair from `(step, loop_state, kept_positions, effective_prompt_len,
prompt_ids.len())`. The chain step and the speculative extension
both consume the same pair — there's no second site that can drift.

## Why R1's `Option<cache_pos>` worked but ages badly

R1 added `cache_pos: Option<i32>` because only the chained kernel
supported `cache_pos`; the persistent kernel inherited from
`position`. The `Option` thus encoded "I want decoupling" vs "I
don't care" — but the same `None` value also meant "I don't even
know if decoupling is possible." Two distinct concepts mapped to
one value.

PR #211 made the persistent kernel cache_pos-aware. The "is
decoupling possible" question is settled (yes, always). What
remains is "what are the two timeline values" — and that's a pair,
not an option.

## Loose ends / known gaps

- The `qwen36_moe_speculative.rs::run_speculative_decode_step{,_batched}`
  body math `let pos = base_position + (k as i32);` was changed to
  produce a `PositionPair` based on a separate `base_cache + k`. In
  dense mode the two arithmetics are identical; in SpecPrefill+MTP
  the cache stays on the compact timeline. Verified by
  `cosine_qwen36_moe_mtp_keep_050_topic_alignment` — see test stderr
  line `[mtp+specprefill parity ...] cossim=...` for the exact
  numbers.
- Batched-spec-verify (`run_speculative_decode_step_batched`) is
  similarly updated. The linear-attn snapshot/restore semantics in
  SpecPrefill+MTP+batched-verify weren't directly tested in this
  branch — sequential-verify is the production default and the
  parity test exercises that. Batched correctness in SpecPrefill
  mode is a follow-up.
- The MTP draft chain inside `mtp.rs` is unchanged. It uses
  `cache_pos = k` (draft-chain-local) and `position = base + k`
  (RoPE). When the caller passes `base = base_position.rope`, the
  draft RoPE is automatically on the absolute timeline.
```

After Task 8 runs, the memo's "Loose ends" bullet referencing the keep=0.50 test result already points at the test's stderr printout — no manual fill-in needed. If the cossim came back at the low end of the bar (≤ 0.70), add a one-line "observed cossim was X.XX, sitting just above the 0.65 floor — flag for follow-up if MTP+SpecPrefill becomes a hot path" right below the bullet.

- [ ] **Step 2: Commit memo**

```bash
git add docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md
git commit -m "docs(research): qwen36 position-threading audit memo

Documents the (rope, cache) split semantics across the four decode
modes (dense, MTP-only, SpecPrefill-only, SpecPrefill+MTP) and the
invariants PositionPair enforces vs. R1's Option<cache_pos>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist

After all tasks complete:

- [ ] Spec coverage: every section of the spec has a corresponding task. (PositionPair → T1, chain.rs → T2, engine.rs → T3, speculative + spec_verify → T4, mutex gate → T5, regression → T6, new parity test → T7-T8, audit memo → T9.)
- [ ] No "TBD" / "TODO" / placeholder strings in any task.
- [ ] Type names consistent: `PositionPair` everywhere, never `PositionTuple` or similar drift.
- [ ] All `--no-verify` / `--no-gpg-sign` etc. absent (they should not appear).
- [ ] Each task has a working build → test → commit shape.
- [ ] All commit messages end with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` line.

---

## After all tasks pass

Use `superpowers:finishing-a-development-branch` to wrap up: present merge / PR / keep / discard options.
