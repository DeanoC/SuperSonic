# Qwen3.6-MoE persistent decode + `cache_pos`: SpecPrefill on the megakernel

**Date:** 2026-05-04
**Branch:** `research/qwen36-persistent-cache-pos`
**Builds on:** R1 SpecPrefill cross-family prototype (PR #210, `docs/research/2026-05-04-qwen36-specprefill-research.md`)

## TL;DR

Plumbed `cache_pos` through the Qwen3.6-MoE persistent decode megakernel so SpecPrefill sparse-prefill no longer has to fall back to the chained 80-launches-per-token path. Sparse-prefill steps now run on the same single-launch megakernel as dense decode, recovering the 7–17%-per-token overhead the chained fallback was paying.

The cleanest correctness signal is a tighter parity bar: the cross-family parity test `cosine_qwen36_moe_keep_100_near_identity` was previously gated at `cossim ≥ 0.99` because R1's sparse-prefill step ran through chained kernels while the dense reference ran through the megakernel — the two paths use bit-different fused-op shapes. With both sides now on the megakernel, **`cossim` measures 1.000000 (bit-equal)** at keep=1.00 instead of just clearing the 0.99 floor.

## Why this matters

The R1 prototype routed sparse-prefill steps through the chained decode driver because the persistent kernel's `position` parameter was used for both RoPE rotation and the KV cache slot. SpecPrefill needs them decoupled: kept tokens rotate at their original prompt positions but write into compact KV slots. The chained side was already plumbed (via `Qwen36MoeAttnStepParams::cache_pos`); the persistent kernel was not.

The chained path adds ~7–17% per-token overhead vs the persistent megakernel (40 attn + 40 ffn + 1 lm_head launches per token, each ~30 µs HIP launch overhead = ~2.4 ms/token, ~9% of the 27 ms/token chain time). At long prompts every kept token pays that overhead, so closing the gap is a roughly-uniform speed unlock across the SpecPrefill TTFT curve.

## Implementation

Five-file change, all on top of R1 main (`7de2e8d`):

| File | Change |
|---|---|
| `kernels/qwen36_moe_persistent/persistent_decode.hip` | Template gains `int cache_pos` after `position`; passes it to `qwen36_moe_attn_step_device` instead of the hard-coded `-1`. |
| `kernels/qwen36_moe_bridge.cpp` | `qwen36_moe_hip_persistent_decode_launch` accepts `int cache_pos`, forwards into both WMMA + non-WMMA `hipLaunchKernelGGL` paths. |
| `crates/kernel-ffi/src/qwen36_moe.rs` | Extern decl + safe wrappers `persistent_decode_launch` / `persistent_decode_launch_range` accept `cache_pos: i32`. |
| `crates/runner/src/qwen36_moe/persistent_decode.rs` | `PersistentScratch::run` and `run_sparse_with_expert_prefetch` take `cache_pos: i32`; re-exports `CACHE_POS_INHERIT` for callers that don't decouple. |
| `crates/runner/src/qwen36_moe/chain.rs` | Drops the "force-chained when `cache_pos.is_some()`" gate. Persistent path now handles both dense (`None` → `INHERIT`) and SpecPrefill (`Some(slot)`); chained-with-cache-pos siblings remain as the `--no-persistent-decode` fallback. |

The reference reader for `cache_pos` is `kernels/qwen36_moe_persistent/full_attn_phase.cuh:730`:

```c
const int eff_cache_pos = (cache_pos >= 0) ? cache_pos : position;
const int kv_len = use_kv_cache ? (eff_cache_pos + 1) : 1;
```

That logic was written for MTP (where the draft head writes step `k` at slot `k` while RoPE rotates at `base_seq_len + k`) and is byte-identical to what SpecPrefill needs.

## Validation

### Parity tests

| Test | Pre-change | Post-change | Notes |
|---|---|---|---|
| `cosine_qwen36_moe_keep_100_near_identity` | cossim ≥ 0.99 (gated) | **cossim = 1.000000** | bit-equal dense; was floor-gated because sparse used chained kernels with different fused-op shape |
| `cosine_qwen36_moe_keep_050_topic_alignment` | cossim ≈ 0.93, top-5 ≥ 3 | cossim = 0.931045, top-5 = 3/5, argmax matches | unchanged keep=0.50 bar |
| `specprefill_qwen35_9b_cosine_parity` (3 cases) | all pass | **all pass** | Qwen3.5-9B same-family regression — different compilation unit, expected unaffected |

The keep=1.00 bit-equality is the strongest correctness proof. When `cache_pos == position` for every step, the `cache_pos`-aware path is *definitionally* identical to the dense path; observing `cossim = 1.0` confirms the wiring through bridge → FFI → wrapper → runner is byte-clean.

### Build

`HIP_ARCH=gfx1100 cargo build --release` clean (only pre-existing warnings).

### TTFT A/B (Qwen3.6-35B-A3B INT4, keep=0.50, gfx1100, 24 GiB)

The R1 sweep used `/tmp/qwen36_specprefill_prompt_*.txt` fixtures that were not archived in-tree and got wiped between R1 and this branch (see "Follow-up — archive prompt fixtures" below). Rather than guess at byte-equivalent prompts, this run uses freshly-generated token-budget-controlled prompts archived under `tests/fixtures/specprefill/` and switches methodology to a **chained-vs-persistent A/B in the same process session**: same prompt, same model load, same drafter, only the kernel-path flag (`--persistent-decode` vs `--no-persistent-decode`) differs. Cleaner isolation than the R1 cross-day baseline.

`prefill_chain_ms` per cell from `--emit-stage-timings`:

| Prompt tokens | Persistent (this branch) | Chained (`--no-persistent-decode`) | Δ ms | **Speedup (persistent)** |
|---:|---:|---:|---:|---:|
|   88 |  1420 |  1534 |  +114 | **7.4%** |
|  349 |  5388 |  5828 |  +440 | **7.6%** |
| 1393 | 22794 | 24611 | +1817 | **7.4%** |
| 4177 | 72641 | 78127 | +5486 | **7.0%** |
| 8353 | _skipped — chained at 8k = ~260 s, full pair ~9 min, expected ~7% by curve continuity_ | | | |

**Headline:** persistent decode + `cache_pos` recovers a uniform **7.0–7.6%** of per-token wall-clock across the full prompt-length range, sitting cleanly inside the predicted 7–17% chained-overhead band. The flatness of the speedup is itself useful evidence: the win is purely launch-overhead amortization (40 attn + 40 ffn + 1 lm_head launches/token folded into 1 cooperative launch), so it scales with `prefill_steps` rather than with attention scope.

Extrapolating to the R1 8k cell: chained sparse target prefill of 259 810 ms × 0.93 ≈ **242 000 ms** on the persistent path; combined with the unchanged 8.3 s drafter scoring and the R1 dense baseline of 745 312 ms gives a total-speedup estimate of **~2.97× at 8k**, vs R1's measured 2.78× — moving toward the 3.0–3.3× target band.

#### Failures encountered + retried

The first A/B pass had two sporadic page-fault crashes during layer load (matching the issue #209 fingerprint exactly: `Memory access fault by GPU node-1 ... Page not present or supervisor privilege.` at `loading 40 layers`). These were re-run after a clean GPU and both passed. The flakiness is **independent of the persistent-vs-chained choice** — one failure was on the chained side (c0088 chained) and one on the persistent side (c1393 persistent), and both succeeded on retry with the same flags. This is consistent with the R1 hypothesis that residual VRAM allocations from prior killed processes (the earlier OOM'd run of this same branch's sweep) trigger the VMM-load instability.

Per-cell logs at `/tmp/qwen36_ttft_ab_<cell>_<mode>.log` (and retries at `/tmp/qwen36_ttft_ab_retry_*.log`).

## Issue #209 (keep<1.00 hang after SIGKILL'd processes)

[Issue #209](https://github.com/DeanoC/SuperSonic/issues/209) reports that sparse-prefill `--specprefill-keep-ratio < 1.0` hangs in `load_decode_layers_with_vmm_strategy` (at `crates/runner/src/qwen36_moe/vmm.rs`) after a prior `supersonic` process was `kill -9`'d and ROCm leaked its VRAM allocations.

**Persistent vs chained doesn't change this.** The hang is at *layer-load* time — the VMM expert-page mapping path — which runs **before** any decode kernel is invoked. The chained-vs-persistent choice is decode-time, downstream of the hang. So this work neither helps nor hurts #209's reproducibility.

The R1 hypothesis (driver-allocator fragmentation interacting with the MoE expert VMM page-in path) and the suggested investigations (SIGTERM handler that drops `MoeExpertResidencyManager` + VMM mappings, `hipMemUnmap` audit on process exit, fragmented-state detect-and-bail) all stand.

Recommend leaving #209 open and untagged-by-this-branch — the persistent path is orthogonal to the bug.

## Follow-up

- **8k cell TTFT**: this run skipped the 8353-token cell to keep the A/B inside one tight session window; the curve at 88/349/1393/4177 is flat at ~7%, so the 8k value is well-predicted but worth confirming once the gfx1100 has a quiet ~10-min slot.
- **Archive prompt fixtures done — keep them**: regenerating prompts with the Qwen3.6 tokenizer landed at `tests/fixtures/specprefill/specprefill_c{0088,0349,1393,4177,8353}_target<N>_actual<N>.txt` (token-exact). The generator is `tests/fixtures/specprefill/gen_prompts.py` (committed under this branch). Replaces the lost R1 `/tmp` fixtures.
- **Add a TTFT A/B bench script in-tree**: `/tmp/qwen36_ttft_ab.sh` is the harness used here; it should land under `tests/gfx1100/` next to `bench_specprefill_cosine.sh` so this comparison is repeatable.
- **MTP + SpecPrefill composition**: with the persistent kernel now decoupled from the dense `cache_pos == position` assumption, the MTP draft head (which has always wanted decoupled slots, see `crates/runner/src/qwen36_moe/mtp.rs:16-19`) and SpecPrefill share one path. The R1 memo's "Out of scope" caveat about `--specprefill-draft-dir + --speculative-decode mutually excluded on Qwen3.6-MoE until MTP plumbs cache_pos` can probably now be lifted; worth a follow-up branch.

## References

- R1 PR #210 (merged): https://github.com/DeanoC/SuperSonic/pull/210
- Issue #209 (keep<1.00 hang): https://github.com/DeanoC/SuperSonic/issues/209
- R1 research memo: `docs/research/2026-05-04-qwen36-specprefill-research.md`
- Reference `cache_pos` consumer: `kernels/qwen36_moe_persistent/full_attn_phase.cuh:730`
- MTP cache_pos usage: `crates/runner/src/qwen36_moe/mtp.rs:16-19`
