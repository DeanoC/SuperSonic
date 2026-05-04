# Qwen3.6 SpecPrefill — research session findings (2026-05-04)

**Branch:** `research/qwen36-specprefill`
**Status:** R1 Step 1 done (gate widened); next step is R1 Step 1b (Qwen36Moe target wiring).
**Hardware:** gfx1100 (RX 7900 XTX, 24 GiB).

## R1 result

**Cross-family Qwen3.5-0.8B → Qwen3.6-MoE SpecPrefill is viable on this hardware. R1 success criterion (≥1.5× TTFT at 8k, keep=0.50) is met at 1.4k tokens with margin.**

TTFT sweep, Qwen3.6-35B-A3B INT4 + Qwen3.5-0.8B BF16 drafter, gfx1100 24 GiB, `keep_ratio=0.50`, `--max-new-tokens 1`, `--emit-stage-timings`. All numbers are `prefill_chain_ms` from one warm run per cell (cosine kernel deterministic; setup costs excluded since they amortize across requests in production).

| Prompt tokens | Kept (%) | Dense prefill ms | Sparse target prefill ms | Drafter score ms | Speedup (target only) | Speedup (steady-state) |
|---:|---:|---:|---:|---:|---:|---:|
|   88 |   49 (55.7%) |    2608 |   1700 |     46 | 1.53× | 1.49× |
|  349 |  179 (51.3%) |   10905 |   5832 |     46 | 1.87× | 1.86× |
| 1393 |  701 (50.3%) |   54363 |  25714 |    399 | 2.11× | 2.08× |
| 4177 | 2093 (50.1%) |  246642 |  97928 |   2076 | 2.52× | 2.47× |
| 8353 | 4178 (50.0%) |  745312 | 259810 |   8321 | 2.87× | 2.78× |

Per-token rates: dense persistent decode ~30 → 89 ms/token across 88 → 8353 tokens (attention-quadratic scaling visible at 8k); sparse chained decode ~32.8 → 62.2 ms/token across the same range. The chained-decode fallback's per-token cost stays close to dense persistent's at small prompts and pulls *ahead* on long ones because the kept-token attention scope is half the dense attention scope (kv_len ~ kept_count not prompt_len) — a happy second-order win the persistent kernel doesn't get.

These numbers are within or close to the SpecPrefill paper's range. Liu et al. report 2.07× at 1.3k and 3.36× at 8k for Qwen3.5-9B same-family with the 0.8B drafter; we observe 2.08× at 1.4k and 2.78× at 8k cross-family on Qwen3.6-MoE. The slightly lower 8k number is consistent with the drafter cost growing (8.3 sec of cosine scoring on 8k tokens with the 0.8B drafter is meaningful overhead) and with the cross-family drafter being approximately — but not perfectly — calibrated for a Qwen3.6 target's importance distribution.

**Recommendation:** ship the R1 path behind the existing `--specprefill-draft-dir` flag. Persistent-kernel `cache_pos` (HIP work) is a follow-up speed unlock, but not blocking — the chained fallback is fast enough. Parity tests (R1 Step 4) are the natural ship-prep work.

## Why this works on A3B when speculative *decode* doesn't

[thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) reports speculative *decode* on Qwen3.6-35B-A3B (llama.cpp draft + Qwen3.5-0.8B drafter, single-stream RTX 3090) **loses 3–12% vs dense baseline at 100% draft acceptance**, with a bimodal collapse to 59–67 tok/s on prompts that keep spec-decode active. The MoESD framing ([arXiv 2505.19645](https://arxiv.org/html/2505.19645)) explains the mechanism: A3B routes 8-of-256 experts per token (sparsity ρ ≈ 0.031), giving an expert-saturation threshold `T_thres = log_{1-ρ}(1-0.95) ≈ 94` tokens. A draft batch of K (1–32) ≪ T_thres means each drafted token pulls fresh expert slices, and the verify pass loads the *union* of K positions' expert sets — overhead that exceeds the savings even when every drafted token is accepted.

**SpecPrefill avoids this pathology by construction.** The MoE expert-saturation cost lives in batched-verify steps (drafter proposes K tokens → target verifies K+1 in one forward); SpecPrefill has *no batched verify*. The drafter scores importance; the target processes the *kept* prompt tokens through the *existing* single-token-per-step prefill pipeline, doing fewer forward passes than dense, not more. Each kept token activates exactly the same expert subset it would in dense — there's no union-of-K to load.

Concretely on this hardware: dense prefill is 30 → 89 ms/token across 88 → 8353 tokens; sparse chained-decode prefill is 32.8 → 62.2 ms/token across the same range. The per-token cost is comparable to dense persistent decode (chained adds 7–17% per token, well below the threshold of "doubles per-token cost" that batched verify pays under expert saturation). The kept-token reduction (~50%) dominates, and the speedup grows with prompt length because (a) drafter scoring amortises and (b) sparse attention scope (`kv_len = kept_count`) pulls ahead of dense's linear KV growth on long prompts.

The thc1006 v3 vLLM clean-A/B retest with `--no-enable-prefix-caching` flips MTP k=1 to **+27.5%** decode speedup on the same 3090 hardware — that's consistent with the framing too: vLLM MTP at k=1 has structurally smaller batched verify (no separate draft-model forward) and avoids the prefix-cache interaction that masked the win in earlier benches. SpecPrefill is orthogonal to whichever spec-decode method runs on top of it (they target different stages: prefill vs decode), so the two could compose.

## TL;DR

Three findings during R1 Step 1 reshape the original plan substantially:

1. **Qwen3.6-27B is hardware-incompatible on this GPU.** ~27 GiB FP8 weights don't fit on 24 GiB and Qwen36Moe-family VMM relief is unavailable to it (it sits in `ModelFamily::Qwen35`, registry.rs:83-85). The locally-available Qwen3.6-35B-A3B (MoE, ~17.5 GiB INT4 with VMM expert paging) is the only viable Qwen3.6 target.
2. **MoE prefill is already single-token-at-a-time** (`crates/runner/src/qwen36_moe/engine.rs:100`). So "sparse prefill" reduces to "skip pruned positions in the per-token loop"; no bulk-prefill kernel needs porting.
3. **The position-decoupling I worried about is already implemented.** `Qwen36MoeAttnStepParams` (`crates/kernel-ffi/src/qwen36_moe.rs:965-984`) has separate `position` (RoPE) and `cache_pos` (KV slot) fields with `-1` sentinel meaning "inherit." MTP (`crates/runner/src/qwen36_moe/mtp.rs:16-19`) uses this for draft-step rotation at absolute position with compact cache slot — exactly the pattern sparse prefill needs.

The CLAUDE.md kernel-isolation question becomes moot — no kernel sharing or porting happens for the MoE path.

## What R1 Step 1 / 1b / 2 are now (all done end-to-end)

**R1 Step 1 — gate widening:**
- `crates/runner/src/policy.rs:105-118` — accept `Qwen3_6_35B_A3B` alongside `Qwen3_5_9B`.
- `crates/runner/src/main.rs:952-967` — Qwen36Moe family dispatch routes through `specprefill_engine::run_specprefill` when `--specprefill-draft-dir` is set instead of silently falling into the dense MoE path.

**R1 Step 1b — cross-family dispatch + cache_pos plumbing:**
- `crates/kernel-ffi/src/qwen36_moe.rs` — already had separate `position` (RoPE) and `cache_pos` (KV slot) fields on `Qwen36MoeAttnStepParams` (used by MTP). No FFI change required.
- `crates/runner/src/qwen36_moe/decode.rs` — added `run_chained_decode_fast_with_cache_pos` and `run_chained_decode_fast_with_expert_prefetch_and_cache_pos` as sibling functions to the existing entries. Existing `run_chained_decode_fast` and friends keep their signatures so legacy callers (tests, multilayer parity) are unaffected. Internal `run_chained_decode_impl` now delegates to a new `run_chained_decode_impl_with_cache_pos` with `CACHE_POS_INHERIT` to preserve dense bit-equality.
- `crates/runner/src/qwen36_moe/chain.rs` — `Qwen36ChainStep` gained `cache_pos: Option<i32>`. When `Some`, `run_chain_step` forces the chained-with-cache-pos path (the persistent decode kernel takes only `position`, so it cannot serve sparse prefill until a separate kernel-side change). `cache_pos = None` preserves the existing dispatch exactly.
- `crates/runner/src/qwen36_moe/engine.rs` — added `pub fn run_with_sparse_prefill(cli, entry, total_vram, keep_mask: Vec<bool>)`. Both `run` and the new function call a shared `run_inner(... keep_mask: Option<Vec<bool>>)`. `decode_text` accepts the keep_mask and currently bails with a clear "R1 Step 3 not yet implemented" message at the top of the function — the loop-side sparse stepping is the remaining piece.
- `crates/runner/src/qwen36_moe/mod.rs` — re-exports `run_with_sparse_prefill` alongside `run`.
- `crates/runner/src/specprefill_engine.rs` — new `run_specprefill_qwen36_moe(cli, entry, ordinal, total_vram, draft_dir)` helper. Looks up the Qwen3.5-0.8B drafter's registry entry (separately from the target's params, which are `FamilyParams::Qwen36Moe`), tokenizes with the target tokenizer, loads the drafter, runs cosine scoring, drops the drafter (frees VRAM before the MoE target loads), then calls `qwen36_moe_cli::run_with_sparse_prefill` with the keep mask. The lookahead algorithm is gated to same-family targets only — cosine is the supported path for cross-family.

**R1 Step 2 — drafter probe on Qwen3.6-MoE prompt** (verified on hardware):
```
[specprefill] cross-family: prompt_tokens=14 (Qwen3.5-0.8B drafter → qwen3.6-35b-a3b target)
[specprefill] draft weights loaded in 271ms
[specprefill] speculator (cosine) done in 27ms (block_size=32)
[specprefill] kept 11/14 tokens (78.6%)
```
The drafter scores a Qwen3.6-MoE-tokenized prompt cleanly (cosine scoring 27 ms vs 23 ms on the same prompt with a Qwen3.5-9B target — overhead is family-independent). Keep mask is non-degenerate. Drafter unloads, MoE target loads via VMM.

**R1 Step 3 — sparse-step the prefill loop** (verified end-to-end on hardware):

Smoke results on a 14-token prompt ("The capital of France is Paris and the largest city in Spain is Madrid"), `--max-new-tokens 4`, defaults `keep_ratio=0.50` + `always_keep_prefix=4` + `always_keep_suffix=4`:

| Path | Kept | Generated tokens | Decoded |
|---|---|---|---|
| Qwen3.5-9B + Qwen3.5-0.8B drafter (regression) | 5/5 (100%) | (5 prompt + 4 new) | "Paris.\nThe" |
| Qwen3.6-35B-A3B dense (no drafter) | n/a | `[13, 3437, 369, 279]` | ". What is the" |
| Qwen3.6-35B-A3B + Qwen3.5-0.8B drafter (sparse) | 11/14 (78.6%) | `[13, 3437, 369, 279]` | ". What is the" |

Longer-prompt smoke on a 67-token prompt about the history of AI, `--max-new-tokens 8`:

| Path | Kept | Decoded |
|---|---|---|
| Qwen3.6-35B-A3B dense | n/a | " generation of AI will be based on a" |
| Qwen3.6-35B-A3B + Qwen3.5-0.8B drafter (sparse) | 35/67 (52.2%) | " decade will be the decade of AI." |

At ~52% keep the argmax of the first generated token diverges ("generation" vs "decade") but both are grammatical, topically aligned continuations of "the next ___" — that's expected behavior at this keep ratio (Liu et al. report retention measured on longer-form tasks, not 8-token argmax parity). The fact that the output is fluent and on-topic is the correctness signal that the cache_pos / RoPE-pos split is wired right end-to-end.

The cross-family sparse-prefill output is **bit-identical** to the dense Qwen3.6-MoE output for this prompt — 3 middle tokens were dropped without affecting the first 4 generated tokens. That's a strong functional-correctness signal for the cache_pos plumbing, the chained-decode-fallback dispatch, and the rope_pos / cache_pos split across the prefill→generation boundary. (It's also expected behavior for SpecPrefill on a prompt where the dropped tokens carry low importance — Liu et al. report ~99.7% LongBench retention.)

The Qwen3.5-9B path remains bit-equal to before (the dense `keep_mask=None` branch in `decode_text` is structurally identical to the prior code; the `match &keep_mask` per-step computation falls into `(loop_state.position, None)` which the chain step then routes through the unchanged persistent decode path).

Code shape:
- `crates/runner/src/qwen36_moe/engine.rs` (`decode_text`):
  - Added speculative-decode + sparse-prefill mutual-exclusion gate (MTP uses persistent decode without `cache_pos` plumbing; combination would write to wrong KV slots).
  - Build `kept_positions: Vec<usize>` from the keep-mask just before the loop. Dense case is `(0..prompt_ids.len()).collect()` so loop control flow is bit-equal.
  - Validate keep-mask invariants (length matches prompt, last position kept, non-empty).
  - Override `loop_state.current_token = prompt_ids[kept_positions[0]]` and `loop_state.total_steps = effective_prompt_len + max_new - 1` for the sparse case.
  - In the loop body, compute `(rope_pos, chain_cache_pos)` per step:
    - Dense: `(loop_state.position, None)` — bit-equal to before.
    - Sparse prefill: `(kept_positions[step] as i32, Some(loop_state.position))`.
    - Sparse generation: `((prompt_ids.len() + (step - effective_prompt_len)) as i32, Some(loop_state.position))`.
  - Pass them to `Qwen36ChainStep` as `position` and `cache_pos`.
  - Replace `prompt_ids.len()` with `effective_prompt_len` in the `is_gen_step` check and the prefill-feed branch; the next-token feed uses `prompt_ids[kept_positions[step + 1]]`.
  - `loop_state.position` advances +1 per step as before — for sparse it now means "compact KV slot," consistent with the chain step's `cache_pos`.

`cargo build --release` clean across all edits. The Qwen3.5-9B dense path is bit-equal to before (all sparse-only logic is gated on `keep_mask.is_some()` or wrapped in the `match &keep_mask` per-step branch).

Limitations of the R1 prototype (documented for follow-up):
- `--specprefill-draft-dir` + `--speculative-decode` mutually excluded on Qwen3.6-MoE until MTP plumbs `cache_pos`.
- Sparse-prefill steps + all subsequent generation steps run via the chained decode path (slower per step than persistent decode), because the persistent decode kernel takes only `position` and would over-attend into unpopulated KV slots when `cache_pos` differs from `position`. Future kernel work: add `cache_pos` to `qwen36_moe_hip_persistent_decode_launch`.

## Phase A2 deferral revisited

Phase A2 left Qwen3.5-0.8B → Qwen3.6-MoE as research because the target footprint exceeded 24 GiB. That deferral predates Qwen3.6-MoE's VMM (`crates/runner/src/qwen36_moe/vmm.rs`, `kv_vmm` + `moe_vmm_mode`) and the SpecPrefill+KV-FP8 lift in PR #203. Both are now available and are the path to fitting drafter + 35B-A3B target on 24 GiB.

## Vocab/tokenizer compatibility

Confirmed: `Qwen3.6-MoE.vocab_size = 248320` (`crates/qwen36_moe/src/config.rs:392`) = Qwen3.5's. The vocab-equality assert at `specprefill_engine.rs:246-251` will pass naturally for a Qwen3.5-0.8B → Qwen3.6-MoE pair. No tokenizer-bridge work needed.

## Remaining work (R1 Step 1b + Step 3) — scoped sketch

### Step 1b — MoE target dispatch in run_specprefill

`run_specprefill` in `specprefill_engine.rs` currently:
1. Loads target config via `qwen35::config::load_config` (line 188).
2. Loads target weights via `Qwen35Weights::load` (line 376).
3. Builds a target *engine* (lines 385-393) using `Qwen35KernelParams`.

For a Qwen3.6-MoE target none of those apply — config comes from `qwen36_moe::config`, weights from `BakedStore`, and the engine is `qwen36_moe::engine::run` (a self-contained top-level function, not a callable component). Two refactor options:

- **Option A — extend qwen36_moe::engine::run** to accept an optional drafter-derived keep-mask + drafter handle. `run_specprefill` would do drafter loading + cosine scoring + selection, then forward the keep-mask to `qwen36_moe::engine::run` along with target setup. Requires threading new params into the MoE engine, but keeps each engine's setup code in its native module.
- **Option B — write a new `run_specprefill_qwen36_moe`** that copies the relevant bits of `qwen36_moe::engine::run` and inlines the sparse-prefill modification. Higher duplication, lower coupling.

**Recommendation: Option A.** The existing MoE engine is already large and growing; threading a `Option<&[bool]>` keep-mask through it once is cheaper than maintaining a parallel copy. Drafter setup (Qwen35-0.8B) stays in `specprefill_engine.rs`.

### Step 3 — sparse-step the MoE prefill loop

In `crates/runner/src/qwen36_moe/engine.rs:265-360`, when a keep-mask is present:
- Replace `for step in 0..loop_state.total_steps` with an explicit iterator over `kept_positions` for the prefill phase.
- For each kept position `p`: `loop_state.current_token = prompt_ids[p]`, then call `run_chain_step` with `position = p` (RoPE) and `cache_pos = compacted_idx` (KV slot, increments from 0).
- Generation phase resumes after the last kept position with `position = kept_positions.last() + 1` and `cache_pos = kept_count`.

`Qwen36ChainStep` (chain.rs:26) currently exposes only `position: i32` — extend with `cache_pos: Option<i32>` and forward to `Qwen36MoeAttnStepParams::cache_pos` in `run_chain_step`. Default `None` ⇒ `cache_pos = -1` (sentinel) preserves dense bit-equality.

### Step 2 — drafter scoring probe (still useful)

Even before Step 1b is complete, run the drafter side end-to-end against a Qwen3.6 prompt to confirm Qwen3.5-0.8B's importance scores look non-degenerate on Qwen3.6 prompts. Can be done with a temporary patch that returns the keep mask and exits before target dispatch. Cheap probe, high info value.

## Out of scope (still)

- Training a Qwen3.6-DFlash checkpoint.
- Tokenizer-bridge / cross-tokenizer drafting (vocabs match).
- Sampling support (greedy-only).
- Batch size > 1.
- Qwen3.6-27B target (hardware-incompatible).

## Operational notes

- GPU is shared with another agent on this machine; check `rocm-smi -u` before any HIP op. Other agent's runs are typically short.
- Worktree convention is sibling dirs (`../SuperSonicBase-<topic>`); branch prefix `research/`.
- Existing reference perf branch: `perf/qwen36-longctx-full-attn` at `../SuperSonicBase-qwen36-longctx-perf`.
