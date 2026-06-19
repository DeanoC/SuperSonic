# Qwen3.6 Lucebox Parity Log

Last updated: 2026-06-18

Purpose: durable working log for the Qwen3.6-27B DFlash Lucebox parity effort on RX 7900 XTX / gfx1100. Keep this file current before and after profiling or benchmark runs so context compaction does not cause stale results, repeated sweeps, or reverted experiments to be treated as new evidence.

## Compaction Recovery Checkpoint

This file is the authoritative memory for this performance thread. On resume:

1. Read this checkpoint before running benchmarks or changing code.
2. Do not repeat any sweep listed in "Valid SuperSonic Sweep Results" unless the code changed in a way that affects that path.
3. Do not use stale artifacts listed in "Ideas Already Tried Or Ruled Out" as evidence.
4. Update this file before starting a new idea, and again immediately after each benchmark with the artifact path and keep/revert decision.

Current exact next action:

- Do not start another performance idea from stale memory. PR #264 is merged into `main`; the `100 tok/s` mean target on the Lucebox 10-prompt Qwen3.6 suite was reached by Phase 2V on branch `codex/qwen36-100tok-profile2`.
- Active next-performance branch: `codex/qwen36-100tok-profile2`.
- Source on merged `main` is `c89b257 Merge pull request #264 from DeanoC/codex/qwen36-next-roofline`.
- Source on merged `main` has `DDTREE_DEFAULT_BUDGET = 15` and `DDTREE_DEFAULT_TOP_K = 4` in `crates/runner/src/qwen35_dflash_engine.rs`.
- New roofline report: `docs/qwen36-lucebox-next-roofline.md`.
- Current confirmed best:
  - `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256.json`: mean 100.86 tok/s, weighted 99.40 tok/s, min 87.87, max 118.20, generated 1654, stopped early 10/10.
  - Repeat `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256_rerun.json`: mean 100.79 tok/s, weighted 99.38 tok/s, min 87.87, max 117.92, generated 1654, stopped early 10/10.
  - Combined output hash stayed `032209e65467e8aa6c74025dc8b70b325f0ec767054ddff614e04550bb11f3bf` and per-prompt generated-token counts stayed `[179,232,99,159,174,154,216,114,165,162]`.
  - Validation passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`; ignored GPU test `dflash_append_delta_q8_direct_attention_matches_extract_path`; one-prompt enabled/disabled A/B; FFI shape profile confirmed the new direct recurrent row and removed `dflash_extract_recurrent_attn`.
- Fresh post-PR #264 full-suite baseline for the 100 tok/s continuation:
  - `target/qwen36_100tok/post264_main_10x256.json`
  - mean 86.54 tok/s, min 76.10, max 100.81, generated 1654, stopped early 10/10.
  - This is slightly above the PR #264 kept artifact `target/qwen36_lucebox_next/tree_conv_source_map_10x256.json` at 86.42 mean / 85.30 weighted, and leaves about a 15.5% mean-throughput gap to 100 tok/s.
- Fresh post-PR #264 one-prompt profile for the 100 tok/s continuation:
  - `target/qwen36_100tok/post264_profile_verify_ffi_shapes_he01_fulltail.json`
  - he_01 generated 179 tokens, 2913 ms decode, 61.46 tok/s under FFI/HAL instrumentation.
  - DFlash breakdown under instrumentation: draft=427 ms, verify=2368 ms, rollback=117 ms.
  - Top FFI rows: `matmul_rhs_transposed_int4[b=1 m=16 n=17408 k=5120 qt=12]` 2688 calls / 412.86 ms; `matmul_mmq_q8_1_q6_k[b=1 m=16 n=5120 k=17408]` 672 calls / 350.26 ms; `delta_recurrent_tree_prefill_capture_q8_trace_attn` 1008 calls / 206.86 ms; `matmul_rhs_transposed_int4[b=1 m=16 n=5120 k=17408 qt=12]` 672 calls / 171.40 ms; `matmul_rhs_transposed_int4[b=1 m=16 n=5120 k=6144 qt=13]` 1008 calls / 130.30 ms; `full_attention_tree_prefill` 336 calls / 108.47 ms.
  - Interpretation: the next material target is the small-M projection train plus tree attention; host overhead, rollback, and 30 ms helper cleanups are not enough to reach 100 tok/s.
  - Do not repeat the rejected direct BA scalar fusion, dense MLP gate/up pair helper, Q6_K MMQ lm-head path, or strided prefix default promotion unless a new kernel design changes those paths materially.
- Active Phase 2A experiment:
  - Implemented a HIP-only raw-GGML `m=16` matmul residual-add epilogue for tree full-attention and tree linear-attention output projections.
  - Rollback gate: `SUPERSONIC_DFLASH_DISABLE_TREE_RESIDUAL_FUSED_MATMUL=1`.
  - The fused epilogue intentionally rounds the projection accumulator to BF16 before adding the BF16 residual, matching the old `matmul -> element_add` numerical order.
  - Scope is intentionally narrow: it does not touch MLP/Q6_K residual fusion yet, and it does not retry dense MLP pair-helper, Q6_K lm-head, or bounded `m <= 16` block-dequant experiments.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
  - One-prompt A/B: `target/qwen36_100tok/tree_residual_fused_he01.json` 78.49 tok/s vs `target/qwen36_100tok/tree_residual_fused_disabled_he01.json` 78.37 tok/s. Fused, disabled, and `post264_main_10x256.json` he_01 stdout tails match exactly.
  - FFI profile: `target/qwen36_100tok/tree_residual_fused_profile_ffi_shapes_he01.json`. The fused path engages for tree output projections: `matmul_rhs_transposed_int4_residual_add[b=1 m=16 n=5120 k=6144 qt=13]` 1008 calls / 131.69 ms and `qt=12` 336 calls / 35.79 ms. `qwen.element_add` drops from the post-#264 profile's 2688 calls / 67.45 ms to 1344 calls / 34.07 ms.
  - Full-suite same-build A/B: `tree_residual_fused_10x256.json` mean 86.39 / weighted 85.26 vs rollback gate `tree_residual_fused_disabled_10x256.json` mean 86.31 / weighted 85.16. Both generated 1654 and stopped early 10/10. This is only a tiny same-build win and remains below `post264_main_10x256.json` mean 86.54 / weighted 85.44 due normal run noise.
  - Normal verify profile without FFI syncs: `target/qwen36_100tok/tree_residual_fused_profile_verify_he01.json`, he_01 179 tokens, 2286 ms decode, 78.31 tok/s. DFlash breakdown: draft=387 ms, verify=1812 ms, rollback=87 ms. Summed tree verify buckets over 21 rounds: linear attention 1178.48 ms, full attention 535.44 ms, logits/greedy 72.22 ms, MLP 9.02 ms, all other buckets below 3 ms.
  - Hot-shape microbench: `target/release/int4_test` with `SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1`, `SUPERSONIC_INT4_TEST_BENCH_M16_HOT=1`, `SUPERSONIC_INT4_TEST_BENCH_ITERS=10`. Current m16 rows: Q4_K gate/up `m=16 n=17408 k=5120` mean 0.2695 ms; Q4_K down `m=16 n=5120 k=17408` mean 0.4285 ms; Q5_K linear `m=16 n=5120 k=6144` mean 0.2323 ms; Q6_K down generic mean 0.6301 ms; Q6_K down MMQ mean 0.7268 ms. This does not support another simple Q6_K MMQ flip as the next move.
  - Correctness harness: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip target/release/int4_test` passed, including new byte-parity checks for `matmul_rhs_transposed_int4_residual_add` versus regular matmul plus `element_add` on Q4_K, Q5_K, and Q6_K m16 fixtures.
  - Decision: keep only as a small launch cleanup with a rollback gate; it does not materially move the 100 tok/s target. Next target must be a larger projection-kernel or acceptance/round-count win.
- Active Phase 2B probe:
  - Testing whether the existing generic GGML gate/up pair helper is worth revisiting for `m=16`, or whether it remains rejected versus two fixed-qtype m16 launches.
  - New harness switch: `SUPERSONIC_INT4_TEST_BENCH_GGML_PAIR_M16=1`.
  - This is not a repeat of the rejected dense MLP pair-helper default promotion: the first step is a focused microbench that compares existing `matmul_rhs_transposed_ggml_pair` against the current exact fixed-qtype `m=16` path and checks byte parity against two separate launches.
  - Harness command: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1 SUPERSONIC_INT4_TEST_BENCH_GGML_PAIR_M16=1 SUPERSONIC_INT4_TEST_BENCH_ITERS=20 target/release/int4_test`.
  - Result: existing generic pair is byte-exact against two fixed m16 launches (`max_abs=0`, `byte_mismatch=0/557056`) but slower: two fixed m16 launches mean `0.3658 ms`, generic pair mean `0.5479 ms`, speedup `0.668x`.
  - Decision: do not wire/promote existing `matmul_rhs_transposed_ggml_pair` for the gate/up path. A new fixed-qtype m16 pair kernel is still a distinct possible experiment, but it must beat the `0.3658 ms` two-launch reference before runtime integration.
  - Follow-up prototype in progress: `matmul_rhs_transposed_ggml_pair` now dispatches to a fixed-qtype `m=16,k%256,n_each%16` HIP pair kernel unless `SUPERSONIC_DFLASH_DISABLE_GGML_PAIR_M16_QTYPE=1` is set. This remains harness/prototype-only until a microbench shows it beats two fixed m16 launches.
  - Fixed-qtype pair result: default pair FFI remains byte-exact (`max_abs=0`, `byte_mismatch=0/557056`) and is faster than two fixed m16 launches: two fixed mean `0.3831 ms`, fixed pair mean `0.2639 ms`, speedup `1.451x`.
  - Rollback-gated result with `SUPERSONIC_DFLASH_DISABLE_GGML_PAIR_M16_QTYPE=1`: two fixed mean `0.3975 ms`, old generic pair mean `0.6432 ms`, speedup `0.618x`.
  - Decision update: keep the fixed-qtype pair kernel and wire it into the runtime gate/up path behind a runtime rollback gate only if full deterministic output parity holds.
  - Runtime wiring in progress: `prefill_mlp_layer` now tries a HIP-only raw-GGML `seq_len=16` gate/up pair into packed `[gate, up]`, then consumes it with existing `swiglu_mul_split`. High-level rollback gate: `SUPERSONIC_DFLASH_DISABLE_GGML_MLP_GATE_UP_PAIR=1`; low-level fixed-pair rollback gate: `SUPERSONIC_DFLASH_DISABLE_GGML_PAIR_M16_QTYPE=1`.
  - One-prompt A/B: `target/qwen36_100tok/gateup_pair_he01.json` 78.55 tok/s vs `target/qwen36_100tok/gateup_pair_disabled_he01.json` 78.31 tok/s. Both generated 179 tokens and normalized stdout tails match.
  - FFI shape profile: `target/qwen36_100tok/gateup_pair_profile_ffi_shapes_he01.json`. The runtime path engages: `qwen.matmul_rhs_transposed_ggml_pair[b=1 m=16 n_each=17408 k=5120 qt=12]` is 1344 calls / 340.30 ms, `qwen.swiglu_mul_split` is 1344 calls / 36.73 ms, and the previous 2688-call Q4_K `m=16,n=17408,k=5120` single-matmul row is absent. The remaining `qt=8` row is a separate draft/target path, 210 calls / 45.55 ms.
  - Full-suite A/B: `target/qwen36_100tok/gateup_pair_10x256.json` mean 86.46 / weighted 85.31 vs rollback gate `target/qwen36_100tok/gateup_pair_disabled_10x256.json` mean 86.15 / weighted 85.02. Both generated 1654 tokens, stopped early 10/10, and normalized stdout tails match.
  - Decision: keep the fixed-qtype m16 GGML gate/up pair path default-on with both rollback gates. The full-suite win is small (`+0.31` mean, `+0.29` weighted) and does not materially change the 100 tok/s gap; the next target must be the remaining Q6_K down/projection cost or tree attention, not another gate/up pair retry.
- Phase 2C implemented and kept:
  - Extending the existing m16 raw-GGML residual-add epilogue to non-Q6 MLP down projection only. This targets the remaining Q4_K down row (`matmul_rhs_transposed_int4[b=1 m=16 n=5120 k=17408 qt=12]`, 672 calls / 174.58 ms) plus about half of the remaining `qwen.element_add` calls.
  - The Q6_K MLP down MMQ path remains untouched because previous microbench evidence did not support falling back to generic Q6_K matmul.
  - New high-level rollback gate: `SUPERSONIC_DFLASH_DISABLE_MLP_DOWN_RESIDUAL_FUSED_MATMUL=1`. Existing `SUPERSONIC_DFLASH_DISABLE_TREE_RESIDUAL_FUSED_MATMUL=1` also disables the shared residual-add helper.
  - One-prompt A/B: `target/qwen36_100tok/mlp_down_residual_fused_he01.json` 78.43 tok/s vs rollback gate `target/qwen36_100tok/mlp_down_residual_fused_disabled_he01.json` 78.37 tok/s. Both generated 179 tokens and normalized stdout tails match.
  - FFI shape profile: `target/qwen36_100tok/mlp_down_residual_fused_profile_ffi_shapes_he01.json`. The intended Q4_K MLP-down row now uses `qwen.matmul_rhs_transposed_int4_residual_add[b=1 m=16 n=5120 k=17408 g=128 qt=12]` 672 calls / 174.59 ms, and `qwen.element_add` drops again to 672 calls / 17.03 ms. Remaining top rows are Q6_K MLP-down MMQ 672 calls / 350.69 ms, fixed gate/up pair 1344 calls / 340.40 ms, recurrent tree attention 1008 calls / 206.91 ms, Q4_K MLP-down residual-add 672 calls / 174.59 ms, and Q5_K tree linear residual-add 1008 calls / 132.29 ms.
  - Full-suite same-session A/B: `target/qwen36_100tok/mlp_down_residual_fused_10x256.json` mean 86.40 / weighted 85.27 vs rollback gate `target/qwen36_100tok/mlp_down_residual_fused_disabled_10x256.json` mean 86.31 / weighted 85.17. Both generated 1654 tokens and stopped early 10/10.
  - Normalized full-suite output hash matches fused vs rollback: `9edf60ede5bec95535088ad27a110aac06e0fef81e966e61c391841ff4f793c0`.
  - Decision: keep as a tiny launch-cleanup/default-on path with rollback gate, but do not treat it as meaningful progress toward 100 tok/s. It remains within normal run noise and below the earlier fresh post-#264 baseline artifact (`post264_main_10x256.json` mean 86.54 / weighted 85.44).
  - Next target must be larger than residual-add cleanup: Q6_K MLP-down MMQ, fixed gate/up pair kernel efficiency, recurrent tree attention, or a round/acceptance-count improvement.
- Phase 2D implemented and kept:
  - Candidate: fuse the fixed raw-GGML `m=16` gate/up pair matmul with the immediately following SwiGLU epilogue for the tree verifier MLP path.
  - This is distinct from the rejected existing generic pair-helper promotion. The current kept fixed-qtype pair is already faster than two separate m16 matmuls, but it still writes `[gate, up]` BF16 and launches `swiglu_mul_split`.
  - Exactness requirement: the fused path must round gate and up accumulators to BF16 before applying `silu(gate) * up`, matching current `pair -> swiglu_mul_split` order.
  - Rollback gate: `SUPERSONIC_DFLASH_DISABLE_GGML_MLP_GATE_UP_SWIGLU_FUSED=1`, default off.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic` and `HIP_ARCH=gfx1100 cargo build --release --bin int4_test`.
  - Harness command: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1 SUPERSONIC_INT4_TEST_BENCH_GGML_PAIR_M16=1 SUPERSONIC_INT4_TEST_BENCH_ITERS=20 target/release/int4_test`.
  - Harness result: fused gate/up+SwiGLU is byte-exact versus `matmul_rhs_transposed_ggml_pair -> swiglu_mul_split` for the Q4_K m16 hot shape (`byte_mismatch=0/278528`, `max_abs=0`). Microbench means: pair+split `0.2523 ms`, fused `0.2323 ms`, speedup `1.086x`.
  - One-prompt A/B: `target/qwen36_100tok/gateup_swiglu_fused_he01.json` 80.39 tok/s vs rollback gate `target/qwen36_100tok/gateup_swiglu_fused_disabled_he01.json` 78.37 tok/s. Both generated 179 tokens and normalized stdout hashes match (`cfd9f35e3bef39c9fb36286b32cc3c5c7fb0a62ef48eda9b40e23009809b1304`).
  - FFI shape profile: `target/qwen36_100tok/gateup_swiglu_fused_profile_ffi_shapes_he01.json`. The fused runtime row engages: `qwen.matmul_rhs_transposed_ggml_pair_swiglu[b=1 m=16 n_each=17408 k=5120 qt=12]` 1344 calls / 328.52 ms. The previous `qwen.swiglu_mul_split` row is gone. The FFI-profile MLP bucket drops from about 44.7 ms/round in Phase 2C to about 42.5 ms/round.
  - Full-suite A/B: `target/qwen36_100tok/gateup_swiglu_fused_10x256.json` mean 88.52 / weighted 87.35 vs rollback gate `target/qwen36_100tok/gateup_swiglu_fused_disabled_10x256.json` mean 86.17 / weighted 85.03. Both generated 1654 tokens, stopped early 10/10, and normalized output hashes match (`9edf60ede5bec95535088ad27a110aac06e0fef81e966e61c391841ff4f793c0`).
  - Decision: keep default-on with the rollback gate. This is the first material Phase 2 win, improving `+2.35` mean / `+2.32` weighted over same-build rollback and `+1.98` mean / `+1.91` weighted over the fresh post-#264 baseline.
  - Current best artifact: `target/qwen36_100tok/gateup_swiglu_fused_10x256.json`, mean 88.52 tok/s, weighted 87.35 tok/s, min 77.64, max 102.99, generated 1654. Remaining gap to 100 tok/s mean is about 13%.
  - New top FFI rows after Phase 2D: Q6_K MLP-down MMQ 672 calls / 351.58 ms, fused Q4_K gate/up+SwiGLU 1344 calls / 328.52 ms, recurrent tree attention 1008 calls / 208.22 ms, Q4_K MLP-down residual-add 672 calls / 174.82 ms, Q5_K tree linear residual-add 1008 calls / 133.05 ms, full tree attention 336 calls / 108.62 ms, RMSNorm 2961 calls / 104.10 ms, lm-head Q6_K row 42 calls / 96.36 ms.
  - Next target: Q6_K MLP-down MMQ or recurrent tree attention. Do not retry unfused pair+split or old generic pair-helper.
- Phase 2E implemented and kept:
  - Candidate: add a Q6_K MMQ residual-add epilogue for the MLP-down path so the Q6_K down projection writes directly into `hidden`, matching the current `MMQ -> BF16 proj_buf -> element_add` order by BF16-rounding the MMQ projection before adding the BF16 residual.
  - Scope is deliberately small: do not change Q8_1 activation quantization, do not change the Q6_K MMA tile math, and do not retry the rejected generic Q6_K fallback.
  - Rollback gate: `SUPERSONIC_DFLASH_DISABLE_Q6_K_MMQ_MLP_DOWN_RESIDUAL_FUSED=1`, default off.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic` and `HIP_ARCH=gfx1100 cargo build --release --bin int4_test`.
  - Harness command: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1 SUPERSONIC_INT4_TEST_BENCH_M16_HOT=1 SUPERSONIC_INT4_TEST_BENCH_ITERS=10 target/release/int4_test`.
  - Harness result: Q6_K MMQ residual-add is byte-exact versus `matmul_mmq_q8_1_q6_k -> element_add` on the hot m16 down shape (`byte_mismatch=0/81920`, `max_abs=0`). Microbench means: MMQ+add `0.5139 ms`, fused residual `0.4611 ms`, speedup `1.115x`. Q6_K mid-linear m16 also exact and improved `1.170x`; vocab row-scan exact but only `1.005x`.
  - One-prompt A/B: `target/qwen36_100tok/q6_mmq_residual_fused_he01.json` 80.52 tok/s vs rollback gate `target/qwen36_100tok/q6_mmq_residual_fused_disabled_he01.json` 80.32 tok/s. Both generated 179 tokens and normalized stdout hashes match (`cfd9f35e3bef39c9fb36286b32cc3c5c7fb0a62ef48eda9b40e23009809b1304`).
  - FFI shape profile: `target/qwen36_100tok/q6_mmq_residual_fused_profile_ffi_shapes_he01.json`. The runtime row engages: `qwen.matmul_mmq_q8_1_q6_k_residual_add[b=1 m=16 n=5120 k=17408]` 672 calls / 351.90 ms. The old `qwen.element_add` row is absent; the FFI-profile MLP bucket drops again to roughly 41.6-41.9 ms/round.
  - Full-suite A/B: `target/qwen36_100tok/q6_mmq_residual_fused_10x256.json` mean 88.75 / weighted 87.59 vs rollback gate `target/qwen36_100tok/q6_mmq_residual_fused_disabled_10x256.json` mean 88.49 / weighted 87.32. Both generated 1654 tokens, stopped early 10/10, and normalized output hashes match (`9edf60ede5bec95535088ad27a110aac06e0fef81e966e61c391841ff4f793c0`).
  - Decision: keep default-on with rollback gate. This is a small cleanup win (`+0.26` mean / `+0.27` weighted over same-build rollback) and brings the current best to mean 88.75 / weighted 87.59.
  - Current best artifact: `target/qwen36_100tok/q6_mmq_residual_fused_10x256.json`, mean 88.75 tok/s, weighted 87.59 tok/s, min 77.82, max 103.41, generated 1654. Remaining gap to 100 tok/s mean is about 12.7%.
  - New top FFI rows after Phase 2E: Q6_K MLP-down MMQ residual-add 672 calls / 351.90 ms, fused Q4_K gate/up+SwiGLU 1344 calls / 329.02 ms, recurrent tree attention 1008 calls / 207.24 ms, Q4_K MLP-down residual-add 672 calls / 174.87 ms, Q5_K tree linear residual-add 1008 calls / 132.37 ms, full tree attention 336 calls / 108.51 ms, RMSNorm 2961 calls / 102.07 ms, lm-head Q6_K row 42 calls / 96.73 ms.
  - Next target must be a core-kernel or algorithmic win: Q6_K MMQ math/layout, fused gate/up core efficiency, recurrent tree attention, or acceptance/round count. Residual/launch cleanups are now exhausted for the MLP path.
- Phase 2F tried and rejected:
  - Candidate: reduce redundant scalar work in `delta_recurrent_tree_prefill_capture_q8_trace_attn`. The current tree recurrent kernel recomputes `expf(g[t])` and reloads `beta[t]` and `value[t, v]` in every K lane for each `(head, token, value)` block, even though those values are scalar for the block.
  - Implementation idea: load `expf(g)`, `beta`, and `value_v` once per block into shared scalars, then let all K lanes reuse them. Preserve the existing path behind a rollback gate.
  - Build passed, but one-prompt A/B was worse: `target/qwen36_100tok/tree_rec_scalar_bcast_he01.json` 80.06 tok/s vs rollback/original `target/qwen36_100tok/tree_rec_scalar_bcast_disabled_he01.json` 80.26 tok/s.
  - Decision: reverted the code and did not keep an env-gated variant. The extra synchronization appears to cost more than the saved scalar `expf`/loads on this kernel.
  - Do not retry this scalar-broadcast form unless the recurrent kernel is substantially reorganized to avoid the added per-token synchronization.
- Resume after local cleanup:
  - After the previous PR/branch cleanup, the Phase 2D/2E continuation patch was preserved in stash `pre-clean qwen36-100tok uncommitted perf work`, then re-applied cleanly on fresh branch `codex/qwen36-100tok-profile2`.
  - Build/hygiene passed on this branch: `cargo fmt --check`, `git diff --check`, `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`, and `HIP_ARCH=gfx1100 cargo build --release --bin int4_test`.
  - Fresh current FFI/HAL shape profile: `target/qwen36_100tok_profile2/profile_current_he01_ffi_shapes.json`.
  - The profile matches Phase 2E evidence: he_01 generated 179 tokens; instrumentation decode was 2746 ms / 65.19 tok/s; DFlash breakdown was draft=431 ms, verify=2198 ms, rollback=117 ms.
  - Current top FFI rows: Q6_K MLP-down MMQ residual-add 672 calls / 351.08 ms; fused Q4_K gate/up+SwiGLU 1344 calls / 327.87 ms; recurrent tree attention 1008 calls / 207.16 ms; Q4_K MLP-down residual-add 672 calls / 175.11 ms; Q5_K tree linear residual-add 1008 calls / 132.60 ms; full tree attention 336 calls / 108.60 ms; RMSNorm 2961 calls / 102.78 ms; lm-head Q6_K row 42 calls / 96.26 ms.
  - Interpretation: the 100 tok/s gap is still too large for residual/launch cleanup. Next experiments should target Q6_K MMQ math/layout, fused gate/up core efficiency, recurrent tree attention without extra synchronization, or acceptance/round-count.
- Budget-15 knob check after Phase 2E:
  - Current he_01 baseline artifact: `target/qwen36_100tok/q6_mmq_residual_fused_he01.json`, 179 tokens, 80.52 tok/s, stdout hash `1eeaabe120546b85be3dfecb0bd471063b093efa887da90ed2190c7a83c707ef`.
  - `target/qwen36_100tok_profile2/smoke_budget15_top8_he01.json`: 179 tokens, 80.65 tok/s, same hash.
  - `target/qwen36_100tok_profile2/smoke_budget15_top4_nochain_he01.json`: 179 tokens, 80.58 tok/s, same hash.
  - `target/qwen36_100tok_profile2/smoke_budget15_top8_nochain_he01.json`: 179 tokens, 80.39 tok/s, same hash.
  - Decision: do not promote top-k 8 or no-chain settings from this evidence. The first prompt is flat/noisy; a full-suite knob run is low value unless a kernel change materially changes verifier cost or acceptance.
- Active Phase 2G experiment:
  - Target the hottest current FFI row, Q6_K MLP-down MMQ residual-add (`m=16 n=5120 k=17408`, 672 calls / 351.08 ms in the fresh profile).
  - Candidate: add a hot exact-shape dispatch for Q6_K MMQ when `m == 16`, `n % 128 == 0`, and `k % 256 == 0`. The exact path can skip partial-tile row/column checks in Q6/Q8 shared loads and output stores. Also fix the Q6 scale (`x_df`) load loop so it does not load rows 0-127 twice per tile.
  - Rollback gate: `SUPERSONIC_DFLASH_DISABLE_Q6_K_MMQ_HOT_EXACT=1`, default off.
  - Acceptance before keeping: `int4_test` Q6_K MMQ byte parity must pass, one-prompt output hash must match, and full-suite weighted throughput must improve before promotion.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic` and `HIP_ARCH=gfx1100 cargo build --release --bin int4_test`.
  - Harness command: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1 SUPERSONIC_INT4_TEST_BENCH_M16_HOT=1 SUPERSONIC_INT4_TEST_BENCH_ITERS=20 target/release/int4_test`.
  - Harness result: Q6_K MMQ residual-add remains byte-exact versus MMQ plus add on the hot m16 down shape (`byte_mismatch=0/81920`). Same-binary hot-shape A/B: exact path `mmq_residual_fused mean_ms=0.3203`, rollback gate `0.4162`, about `1.30x` faster for that row.
  - One-prompt A/B: `target/qwen36_100tok_profile2/q6_hot_exact_he01.json` 83.89 tok/s vs rollback gate `target/qwen36_100tok_profile2/q6_hot_exact_disabled_he01.json` 80.45 tok/s. Both generated 179 tokens and matched the previous stdout hash `1eeaabe120546b85be3dfecb0bd471063b093efa887da90ed2190c7a83c707ef`.
  - Full-suite same-binary A/B: `target/qwen36_100tok_profile2/q6_hot_exact_10x256.json` mean 92.60 / weighted 91.38 vs rollback gate `target/qwen36_100tok_profile2/q6_hot_exact_disabled_10x256.json` mean 88.35 / weighted 87.17. Both generated 1654 tokens and matched the previous suite hash `0908ee95c4c36c99b008a742f33ee6bd6dab48e4e2852263216d1af1987b1b69`.
  - Delta versus previous best `target/qwen36_100tok/q6_mmq_residual_fused_10x256.json`: +3.85 mean tok/s / +3.79 weighted tok/s. Every prompt improved versus the rollback gate; no prompt-level collapse.
  - Fresh FFI profile after keeping Phase 2G: `target/qwen36_100tok_profile2/q6_hot_exact_profile_ffi_shapes_he01.json`.
  - New top FFI rows: fused Q4_K gate/up+SwiGLU 1344 calls / 330.25 ms; Q6_K MLP-down MMQ residual-add 672 calls / 255.36 ms; recurrent tree attention 1008 calls / 207.22 ms; Q4_K MLP-down residual-add 672 calls / 174.69 ms; Q5_K tree linear residual-add 1008 calls / 132.45 ms; full tree attention 336 calls / 108.52 ms; RMSNorm 2961 calls / 102.18 ms.
  - High-level per-round buckets under instrumentation are now roughly MLP 37.1 ms, linear attention 39.0 ms, full attention 14-17 ms. This shifts the next target toward fused gate/up core efficiency or tree linear/recurrent attention; Q6_K down is no longer the top row.
  - Decision: keep default-on with the rollback gate. Current best is 92.60 mean / 91.38 weighted; remaining mean gap to 100 tok/s is about 8.0%.
- Phase 2H tried and rejected:
  - Target the current top FFI row, fused Q4_K gate/up+SwiGLU (`matmul_rhs_transposed_ggml_pair_swiglu`, 1344 calls / 330.25 ms).
  - Candidate: relax only this kernel's `__launch_bounds__` from `(32, 8)` to `(32, 4)` to see whether lower occupancy pressure lets the compiler keep the two WMMA accumulators/dequant temporaries in faster registers. This is a compile-time tuning probe, not a logic change.
  - Build passed for both `supersonic` and `int4_test` while the probe was present.
  - Harness command: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 SUPERSONIC_BACKENDS=hip SUPERSONIC_INT4_TEST_BENCH_GGML_HOT=1 SUPERSONIC_INT4_TEST_BENCH_GGML_PAIR_M16=1 SUPERSONIC_INT4_TEST_BENCH_ITERS=30 target/release/int4_test`.
  - Harness result: byte parity stayed exact (`pair_vs_two_fixed_m16 byte_mismatch=0/557056`, `fused_swiglu_vs_pair_split byte_mismatch=0/278528`) and `fused_pair_swiglu mean_ms=0.2179`, but this did not translate into a useful model-level win.
  - One-prompt smoke: `target/qwen36_100tok_profile2/gateup_launchbounds4_he01.json` ran 83.82 tok/s with 179 generated tokens, essentially flat and slightly below the Phase 2G one-prompt result of 83.89 tok/s.
  - Decision: rejected and reverted. Keep the fused pair+SwiGLU kernel at `__launch_bounds__(32, 8)`. Do not retry this exact launch-bounds-only tweak unless a later kernel rewrite changes register/occupancy behavior materially.
- Rebuild checkpoint after rejecting Phase 2H:
  - Source has the fused pair+SwiGLU kernel back at `__launch_bounds__(32, 8)`.
  - `cargo fmt --check`: passed.
  - `git diff --check`: passed.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin int4_test`: passed with existing warnings.
  - Q6 hot harness after rebuild: byte parity still exact for Q6_K MMQ residual-add on the hot m16 down shape (`byte_mismatch=0/81920`); `mmq_residual_fused mean_ms=0.3213`.
  - Real one-prompt smoke after rebuild: `target/qwen36_100tok_profile2/q6_hot_exact_rebuild_he01.json`, he_01 generated 179 tokens and ran 84.03 tok/s.
  - Interpretation: the 88+ tok/s work is not reverted. Current source still includes the kept Phase 2D/2E/2G optimizations; only the rejected Phase 2H launch-bounds tweak was reverted. The current best full-suite artifact remains `target/qwen36_100tok_profile2/q6_hot_exact_10x256.json` at 92.60 mean / 91.38 weighted.
- Phase 2I DDTree knob smoke rejected/no-promote:
  - Reason: Phase 2G materially reduced verifier cost and moved the full-suite mean to 92.60 tok/s, so a limited DDTree setting smoke is now justified even though earlier budget/top-k checks were flat.
  - Scope: one-prompt `he_01` only, corrected Lucebox mode, Q8 draft, `n_gen=256`, direct rollback, budgets `{14,15,16}` and top-k `{4,8}`.
  - Baseline anchor after rebuild: `target/qwen36_100tok_profile2/q6_hot_exact_rebuild_he01.json`, budget 15/top-k 4, 179 generated tokens, 84.03 tok/s.
  - Results:
    - `target/qwen36_100tok_profile2/smoke_p2i_budget14_top4_he01.json`: 179 generated tokens, 60.72 tok/s.
    - `target/qwen36_100tok_profile2/smoke_p2i_budget14_top8_he01.json`: 179 generated tokens, 60.57 tok/s.
    - `target/qwen36_100tok_profile2/smoke_p2i_budget15_top8_he01.json`: 179 generated tokens, 83.89 tok/s.
    - `target/qwen36_100tok_profile2/smoke_p2i_budget16_top4_he01.json`: 179 generated tokens, 40.70 tok/s.
    - `target/qwen36_100tok_profile2/smoke_p2i_budget16_top8_he01.json`: 179 generated tokens, 40.68 tok/s.
  - Decision: do not run a full-suite DDTree sweep and do not promote any settings. Budget 15/top-k 4 remains the default; top-k 8 is flat/noisy and budgets 14/16 are clear regressions after Phase 2G.
  - Next target returns to kernel work: fused Q4_K gate/up+SwiGLU core efficiency, tree recurrent attention, Q4/Q5 small-M projection cost, or full tree attention. The DDTree knob path will not close the remaining 8% gap to 100 tok/s.
- Phase 2J implemented and kept:
  - Target: `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn`, currently 1008 calls / about 207 ms in `target/qwen36_100tok_profile2/q6_hot_exact_profile_ffi_shapes_he01.json`.
  - Candidate: add a warp32 Q8 tree recurrent direct-attention kernel. It was first tested behind `SUPERSONIC_DFLASH_ENABLE_TREE_RECURRENT_WARP32=1`, then promoted default-on after validation. Rollback gate: `SUPERSONIC_DFLASH_DISABLE_TREE_RECURRENT_WARP32=1`.
  - Rationale: the current kernel launches 128 threads per `(head, value_channel)` and performs two block-wide K reductions per tree token, each crossing four waves through LDS and `__syncthreads`. The warp32 variant keeps a single wave per block, each lane owns four K positions, computes four 32-lane partials, and combines them as `(p0 + p2) + (p1 + p3)` to preserve the current block-reduction order as closely as possible.
  - This is deliberately different from rejected Phase 2F: it does not add shared scalar broadcasts or per-token cross-wave synchronization. It removes the block-level synchronization from the recurrent attention reductions and keeps only one single-wave `__syncthreads()` per tree token to preserve parent/child trace visibility.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
  - Ignored GPU parity passed with the opt-in env: `cargo test -p runner --test dflash_tree_delta_parity dflash_tree_delta_q8_direct_attention_matches_extract_path --release -- --ignored --nocapture`.
  - One-prompt smoke with opt-in env: `target/qwen36_100tok_profile2/tree_recurrent_warp32_he01.json`, 179 generated tokens, 86.96 tok/s. Baseline rebuild smoke `target/qwen36_100tok_profile2/q6_hot_exact_rebuild_he01.json` was 179 tokens / 84.03 tok/s. Stdout-tail hashes matched exactly.
  - FFI shape profile with opt-in env: `target/qwen36_100tok_profile2/tree_recurrent_warp32_profile_ffi_shapes_he01.json`, 179 generated tokens, instrumentation 69.30 tok/s. The recurrent row moved from `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn` 1008 calls / 207.22 ms to 1008 calls / 137.73 ms. The linear-attention bucket moved from about 39 ms/round to about 35.6-36.2 ms/round.
  - Full-suite opt-in run: `target/qwen36_100tok_profile2/tree_recurrent_warp32_10x256.json`, mean 95.75 tok/s, weighted 94.45 tok/s, min 83.89, max 111.48, generated 1654, stopped early 10/10.
  - Prompt-level comparison versus previous best `target/qwen36_100tok_profile2/q6_hot_exact_10x256.json`: all 10 prompts improved, generated-token counts are identical, and stdout-tail hashes match for every prompt.
  - After promotion to default-on, rebuild passed again: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
  - No-env one-prompt smoke: `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_he01.json`, 179 generated tokens, 86.66 tok/s, stdout-tail hash matches the Phase 2G baseline.
  - Rollback-gated one-prompt smoke: `target/qwen36_100tok_profile2/tree_recurrent_warp32_disabled_he01.json`, 179 generated tokens, 83.96 tok/s, stdout-tail hash matches.
  - Promoted no-env full-suite run: `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_10x256.json`, mean 95.63 tok/s, weighted 94.36 tok/s, min 83.82, max 111.36, generated 1654, stopped early 10/10.
  - Prompt-level comparison versus previous best `target/qwen36_100tok_profile2/q6_hot_exact_10x256.json`: all 10 prompts improved, generated-token counts are identical, and stdout-tail hashes match for every prompt.
  - Decision: keep default-on with rollback gate `SUPERSONIC_DFLASH_DISABLE_TREE_RECURRENT_WARP32=1`. Current best default artifact is `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_10x256.json`; remaining gap to 100 tok/s mean is about 4.6%.
- Rejected Phase 2K micro-follow-up:
  - Target: the new warp32 recurrent kernel from Phase 2J.
  - Candidate: compute/load scalar per-token/per-value values (`exp(g[t])`, `beta[t]`, and `value[t, v]`) only in lane 0 and broadcast them with `__shfl`.
  - Rationale: this is the safe version of the rejected Phase 2F scalar idea. It stays inside one wave and adds no shared-memory broadcasts or new `__syncthreads`; it should preserve exact values by broadcasting lane 0's result.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
  - Ignored GPU parity passed: `cargo test -p runner --test dflash_tree_delta_parity dflash_tree_delta_q8_direct_attention_matches_extract_path --release -- --ignored --nocapture`.
  - One-prompt smoke: `target/qwen36_100tok_profile2/tree_recurrent_warp32_scalar_bcast_he01.json`, 179 generated tokens, 86.66 tok/s.
  - Decision: rejected and reverted. The result was flat versus the promoted Phase 2J no-env smoke `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_he01.json` at 86.66 tok/s, so this adds complexity without a measured win. Keep Phase 2J default-on; do not retry this lane-0 scalar broadcast unless a later recurrent-kernel rewrite changes the load pattern materially.
- Refreshed profile after Phase 2J:
  - Artifact: `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_profile_ffi_shapes_he01.json`.
  - he_01 instrumentation throughput: 179 generated tokens, 69.20 tok/s.
  - DFlash breakdown: draft=435 ms, verify=2032 ms, rollback=120 ms.
  - Top FFI rows: fused Q4_K gate/up+SwiGLU 1344 calls / 329.04 ms; Q6_K MLP-down MMQ residual-add 672 calls / 255.38 ms; Q4_K MLP-down residual-add 672 calls / 174.90 ms; warp32 tree recurrent attention 1008 calls / 137.23 ms; Q5_K tree projection residual-add 1008 calls / 132.85 ms; full tree attention 336 calls / 108.51 ms; RMSNorm 2961 calls / 102.47 ms; greedy lm-head 42 calls / 96.33 ms.
  - Interpretation: after the warp32 recurrent win, no single remaining row can close the 4.6% gap alone. The next useful attempts must either improve acceptance/round count, reduce the fused Q4_K gate/up core, or combine several smaller verifier-kernel wins.
- Rejected Phase 2L Q6_K MMQ Y64 tile probe:
  - Target: Q6_K MLP-down MMQ residual-add at `m=16,n=5120,k=17408`, which still accounts for 255 ms in the post-Phase-2J FFI profile.
  - Candidate: specialize the Q6_K MMQ hot path with a smaller output tile (`Y=64` instead of `Y=128`) to double block count and reduce per-block LDS pressure.
  - Harness result with the probe enabled: byte exact against the current fused residual path (`byte_mismatch=0/81920`), but `mmq_residual_fused mean_ms=0.6218`.
  - Same-binary rollback gate result with the probe disabled: `mmq_residual_fused mean_ms=0.4366`. Post-revert harness was also byte exact and `0.4335 ms`.
  - Decision: rejected and reverted. Do not retry this Y64 tiling form; it is much slower for the hot shape despite exact output.
- Rejected Phase 2M full tree attention hard-sync removal:
  - Target: the BF16 tiled tree attention launcher in `kernels/full_attention_bridge.cpp`, which still used `hipDeviceSynchronize()` in `launch_tree_tiled`.
  - Candidate: change only that post-launch sync to `maybe_sync()`, matching other bridge helpers unless `SUPERSONIC_SYNC_EACH_KERNEL=1`.
  - One-prompt smoke: `target/qwen36_100tok_profile2/tree_attention_maybesync_he01.json`, 179 generated tokens, 86.51 tok/s versus current default `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_he01.json` at 86.66 tok/s.
  - Full-suite run: `target/qwen36_100tok_profile2/tree_attention_maybesync_10x256.json`, mean 93.99 tok/s, min 83.26, max 110.62, generated 1654, stopped early 10/10. This regressed from the current default `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_10x256.json` at 95.63 mean.
  - Decision: rejected and reverted. Keep the hard sync in `launch_tree_tiled`; removing it is not a valid default-on optimization for this path.
- Phase 2N DDTree knob check rejected/no-promote:
  - Reason: Phase 2J materially changed verifier cost, so the earlier Phase 2I budget/top-k smoke needed a cheap re-check under the current default kernel stack.
  - Same-session default anchor: `target/qwen36_100tok_profile2/smoke_p2j_default_rerun_he01.json`, 179 generated tokens, 86.13 tok/s, stdout-tail hash `1eeaabe120546b85be3dfecb0bd471063b093efa887da90ed2190c7a83c707ef`.
  - `target/qwen36_100tok_profile2/smoke_p2j_budget15_top8_he01.json`: 179 generated tokens, 85.98 tok/s, same hash.
  - `target/qwen36_100tok_profile2/smoke_p2j_budget15_top4_nochain_he01.json`: 179 generated tokens, 86.06 tok/s, same hash.
  - `target/qwen36_100tok_profile2/smoke_p2j_budget15_top8_nochain_he01.json`: 179 generated tokens, 86.06 tok/s, same hash.
  - Decision: do not promote top-k 8 or no-chain and do not run a full-suite sweep. These settings are flat to slightly lower on the representative prompt, with identical output, and cannot explain the remaining 4.6% gap to 100 tok/s.
- Rejected Phase 2O fast-SwiGLU sigmoid probe:
  - Target: the current hottest row, fused Q4_K `matmul_rhs_transposed_ggml_pair_swiglu[b=1 m=16 n_each=17408 k=5120 qt=12]`.
  - Candidate: add an opt-in Q4_K path using `supersonic_qwen35_sigmoid_fast` / `__expf` inside the fused pair+SwiGLU kernel instead of exact `expf`, behind `SUPERSONIC_DFLASH_ENABLE_GGML_MLP_GATE_UP_SWIGLU_FAST_SIGMOID=1`.
  - Harness result with the probe enabled: fused output stayed byte-exact versus existing `pair -> swiglu_mul_split` on the hot shape (`byte_mismatch=0/278528`), and the same-build microbench was slightly faster for fused pair+SwiGLU (`0.2231 ms` enabled vs `0.2328 ms` disabled). The microbench remains noisy.
  - One-prompt model smoke: `target/qwen36_100tok_profile2/fast_swiglu_sigmoid_he01.json`, 179 generated tokens, 91.16 tok/s, stdout-tail hash `1eeaabe120546b85be3dfecb0bd471063b093efa887da90ed2190c7a83c707ef`; same-build disabled anchor `target/qwen36_100tok_profile2/fast_swiglu_sigmoid_disabled_he01.json` was 86.43 tok/s with the same hash.
  - Full-suite opt-in run: `target/qwen36_100tok_profile2/fast_swiglu_sigmoid_10x256.json`, mean 95.21 tok/s, min 83.68, max 105.82, generated 1654, stopped early 10/10. All prompt hashes and generated-token counts matched current default, but it regressed from `target/qwen36_100tok_profile2/tree_recurrent_warp32_default_10x256.json` at 95.63 mean. Gains on he_01/he_06 were outweighed by losses on he_07/he_09.
  - Decision: rejected and reverted. Do not promote fast-SwiGLU sigmoid as a default; only revisit if a later full-suite A/B shows prompt-level stability and mean above the current 95.63 tok/s default.
- Rejected Phase 2P fused-SwiGLU output-branch cleanup:
  - Target: the same fused Q4_K gate/up+SwiGLU row.
  - Candidate: remove the `if (out_col_idx < n_each)` guard from `supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel`, since the bridge already requires `n_each % 16 == 0` for this m16 hot path.
  - Harness result: byte-exact versus existing pair+split (`byte_mismatch=0/278528`) and hot fused pair+SwiGLU measured `0.2200 ms` in the 40-iteration harness. This looked slightly better than nearby noisy runs but did not carry into the model.
  - One-prompt smoke: `target/qwen36_100tok_profile2/swiglu_no_outcol_branch_he01.json`, 179 generated tokens, 86.28 tok/s. This is below the current default he_01 anchors (`86.43` same-session disabled from Phase 2O and `86.66` promoted Phase 2J no-env smoke), with the same output hash expected from byte-exact math.
  - Decision: rejected and reverted. Do not promote output-branch removal from this evidence; microbench noise is not enough without model-level gain.
- Phase 2Q implemented and kept:
  - Target: the generic m16 GGML qtype WMMA kernel `supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel`, not the fused pair+SwiGLU kernel from rejected Phase 2H.
  - Candidate: reduce launch bounds from `__launch_bounds__(32, 8)` to `__launch_bounds__(32, 4)` for the generic m16 qtype path used by Q4/Q5 residual projections and Q6_K lm-head shapes. This keeps the Phase 2D/2E/2G/2J optimizations default-on.
  - One-prompt smoke: `target/qwen36_100tok_profile2/m16_qtype_launchbounds4_he01.json`, 179 generated tokens, 91.91 tok/s, output hash unchanged.
  - Full-suite run: `target/qwen36_100tok_profile2/m16_qtype_launchbounds4_10x256.json`, mean 96.40 tok/s, weighted 94.87 tok/s, min 84.39, max 113.38, generated 1654, stopped early 10/10. All prompt output hashes and generated-token counts match the previous default.
  - Prompt-level note: `he_07` regressed by about 3.0 tok/s versus `tree_recurrent_warp32_default_10x256.json`, but the suite minimum improved and there was no token-count or output-hash change.
  - Rejected subprobe: `__launch_bounds__(32, 2)` gave `target/qwen36_100tok_profile2/m16_qtype_launchbounds2_he01.json`, 179 generated tokens, 85.25 tok/s, so it was reverted.
  - Rejected follow-up subprobes after refreshing the profile: `__launch_bounds__(32, 3)` gave `target/qwen36_100tok_profile2/m16_qtype_launchbounds3_he01.json`, 179 generated tokens, 85.18 tok/s; `__launch_bounds__(32, 5)` gave `target/qwen36_100tok_profile2/m16_qtype_launchbounds5_he01.json`, 179 generated tokens, 87.34 tok/s. The restored `(32, 4)` binary gave `target/qwen36_100tok_profile2/m16_qtype_launchbounds4_restored_he01.json`, 179 generated tokens, 87.64 tok/s in the same noisy smoke band. All three hashes matched, but neither `(32, 3)` nor `(32, 5)` justified promotion or a full suite.
  - Decision: keep `(32, 4)` default-on. `target/release/supersonic` was rebuilt after restoring `(32, 4)` so the executable no longer contains the rejected `(32, 2)` probe.
- Refreshed profile after Phase 2Q:
  - Artifact: `target/qwen36_100tok_profile2/m16_qtype_launchbounds4_profile_ffi_shapes_he01.json`. This is diagnostic only because FFI/HAL profiling inserts syncs and changed the instrumented generated length to 193 tokens.
  - Top rows: fused Q4_K gate/up+SwiGLU 1408 calls / 344.29 ms; Q6_K MMQ MLP-down residual-add 704 calls / 267.36 ms; warp32 recurrent attention 1056 calls / 208.03 ms; Q4_K residual-add projection 704 calls / 182.60 ms; Q5_K projection 1056 calls / 137.89 ms; full tree attention 352 calls / 111.94 ms; RMSNorm rows 3102 calls / 107.31 ms; Q6_K lm-head 44 calls / 93.09 ms.
  - Diagnostic residual-fusion gate check: disabling `SUPERSONIC_DFLASH_DISABLE_TREE_RESIDUAL_FUSED_MATMUL=1` moved the Q4 residual row back to plain matmul and doubled element-add calls, so the residual-fused path is still useful and should stay enabled.
  - Next likely target: MLP path fusion around fused gate/up+SwiGLU plus Q6_K down quantization/MMQ. Avoid repeating rejected fast-SwiGLU sigmoid, output-branch cleanup, DDTree knob changes, or launch-bounds-only probes without new evidence.
- Rejected Phase 2R Q6_K MMQ launch-bounds probe:
  - Target: the second hottest refreshed row, `qwen.matmul_mmq_q8_1_q6_k_residual_add[b=1 m=16 n=5120 k=17408]`, 704 calls / 267.36 ms in the Phase 2Q FFI profile.
  - Candidate: change `supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel` launch bounds from `__launch_bounds__(256, 2)` to `__launch_bounds__(256, 1)` to relax register pressure constraints for the hot exact path.
  - One-prompt smoke: `target/qwen36_100tok_profile2/q6_mmq_launchbounds1_he01.json`, 179 generated tokens, 87.49 tok/s, output hash matching the restored `(32, 4)` anchor. Same-session restored current source anchor `target/qwen36_100tok_profile2/m16_qtype_launchbounds4_restored_he01.json` was 87.64 tok/s.
  - Decision: rejected and reverted. This is flat/noisy and does not justify a full-suite run.
- Phase 2S implemented and kept:
  - Target: Q6_K lm-head greedy argmax in the prefill-append verifier. The Phase 2Q profile showed Q6_K lm-head plus argmax as a remaining hot slice, but the first implementation only wired the cached tree greedy path and did not engage in Lucebox serving mode.
  - Candidate: add a fused raw-GGML Q6_K m16 lm-head argmax path, guarded by `SUPERSONIC_DFLASH_DISABLE_Q6_K_LM_HEAD_ARGMAX_FUSED=1`. The kernel computes BF16-rounded tile logits, reduces per 16-column tile to a row winner, then reduces tile winners to token IDs, avoiding full `[16, vocab]` logits materialization for greedy scans.
  - Harness: `HIP_ARCH=gfx1100 cargo run --release --bin int4_test` passed, including `q6_k m16 fused argmax parity` with `fused_argmax_mismatches=0/16`.
  - Initial smoke before active-path wiring was flat because the fused row did not appear in FFI shapes; it was only connected to `PrefillTreeVerifyCache::compute_greedy_ids`, while the benchmark reported `[dflash] using prefill-append target verifier`.
  - Corrected one-prompt A/B after wiring `compute_greedy_for_range`: `target/qwen36_100tok_profile2/q6_lm_head_argmax_range_fused_he01.json` 87.49 tok/s versus rollback gate `target/qwen36_100tok_profile2/q6_lm_head_argmax_range_fused_disabled_he01.json` 87.34 tok/s. Both generated 179 tokens and stdout-tail hash `632ea843784fa7b9ed25a50807653dbc3bceca0a5bd2a9c9983145385424d0cf`.
  - Corrected FFI shape profile: `target/qwen36_100tok_profile2/q6_lm_head_argmax_range_fused_profile_ffi_shapes_he01.json`. It shows `qwen.matmul_q6_k_m16_argmax[b=1 m=16 n=248320 k=5120]` engaged for 21 calls / 44.79 ms, the old Q6_K full lm-head row reduced from 42 calls to 21 calls, and `qwen.argmax_bf16_rows` disappeared. The remaining 21 full lm-head calls are from logits/NLL work and should not be confused with a failed dispatch.
  - Full-suite run: `target/qwen36_100tok_profile2/q6_lm_head_argmax_range_fused_10x256.json`, mean 96.89 tok/s, weighted 95.51 tok/s, min 84.60, max 113.51, generated 1654, stopped early 10/10. Prompt output hash matches Phase 2Q (`bddf89d09566256324051dad1f102f1b0ca31717d22def40abe2b92498ca601e`) and per-prompt generated-token counts are unchanged.
  - Decision: keep default-on with rollback gate. This is a modest but valid full-suite win over Phase 2Q (`96.40` mean / `94.87` weighted). It does not close the 100 tok/s goal; the remaining gap is about 3.1 mean tok/s.
- Phase 2T implemented and kept:
  - Target: the draft candidate generator's identical Q6_K lm-head + argmax pattern. After Phase 2S, the remaining `m=16,n=248320` row came from draft candidate generation, not target verifier greedy output.
  - Candidate: reuse the same fused Q6_K m16 argmax path in `draft_forward_and_sample` only when DDTree probing is absent, because DDTree top-k probing still needs the full draft logits buffer. Added persistent `lm_head_block_best_*` scratch buffers to `DFlashScratch` to avoid per-round allocation. The same rollback gate, `SUPERSONIC_DFLASH_DISABLE_Q6_K_LM_HEAD_ARGMAX_FUSED=1`, disables both target and draft fused argmax paths.
  - One-prompt A/B: `target/qwen36_100tok_profile2/q6_lm_head_argmax_target_draft_fused_he01.json` 87.72 tok/s versus rollback gate `target/qwen36_100tok_profile2/q6_lm_head_argmax_target_draft_fused_disabled_he01.json` 87.11 tok/s. Both generated 179 tokens and stdout-tail hash `632ea843784fa7b9ed25a50807653dbc3bceca0a5bd2a9c9983145385424d0cf`.
  - FFI shape profile: `target/qwen36_100tok_profile2/q6_lm_head_argmax_target_draft_fused_profile_ffi_shapes_he01.json` shows `qwen.matmul_q6_k_m16_argmax[b=1 m=16 n=248320 k=5120]` for 42 calls / 90.28 ms, and the old full `m=16,n=248320` lm-head row is gone.
  - Full-suite run: `target/qwen36_100tok_profile2/q6_lm_head_argmax_target_draft_fused_10x256.json`, mean 96.94 tok/s, weighted 95.57 tok/s, min 84.39, max 113.25, generated 1654, stopped early 10/10. Prompt output hash matches Phase 2S (`bddf89d09566256324051dad1f102f1b0ca31717d22def40abe2b92498ca601e`) and per-prompt generated-token counts are unchanged.
  - Decision: keep. This is a small full-suite win over Phase 2S and removes the last full-logits lm-head materialization in no-DDTree Lucebox serving mode. The remaining gap to 100 tok/s mean is about 3.06 tok/s.
- Phase 2U implemented and kept:
  - Target: append verifier attention output projections still used `matmul_proj -> element_add`, while the tree verifier attention paths already used the raw-GGML residual-add epilogue.
  - Candidate: extend the existing residual-add projection helper to append full-attention and append linear-attention O projections. The linear-attention path keeps the old materialized `proj_buf2` path when `SUPERSONIC_TRACE_LINEAR_ATTN=1` so debug tracing remains intact. The shared rollback gate is still `SUPERSONIC_DFLASH_DISABLE_TREE_RESIDUAL_FUSED_MATMUL=1`, default off.
  - One-prompt A/B: `target/qwen36_100tok_profile2/append_attn_residual_fused_he01.json` and rollback-gated `target/qwen36_100tok_profile2/append_attn_residual_fused_disabled_he01.json` were exactly flat at 179 generated tokens, 2042 ms decode, 87.64 tok/s, with stdout-tail hash `632ea843784fa7b9ed25a50807653dbc3bceca0a5bd2a9c9983145385424d0cf`.
  - FFI shape profile: `target/qwen36_100tok_profile2/append_attn_residual_fused_profile_ffi_shapes_he01.json` confirms the append path engages. `qwen.element_add` disappeared from the top rows, `qwen.matmul_rhs_transposed_int4_residual_add[b=1 m=16 n=5120 k=6144 g=128 qt=13]` appeared for 1008 calls / 132.76 ms, and `qt=12` appeared for 336 calls / 35.06 ms. Versus the Phase 2T FFI profile, the two residual-add rows cost about 2.6 ms more than the old plain matmul rows but remove the 33.8 ms `element_add` row.
  - Full-suite run: `target/qwen36_100tok_profile2/append_attn_residual_fused_10x256.json`, mean 97.12 tok/s, weighted 95.75 tok/s, min 84.67, max 113.51, generated 1654, stopped early 10/10.
  - Combined output hash matches Phase 2T (`032209e65467e8aa6c74025dc8b70b325f0ec767054ddff614e04550bb11f3bf`) and per-prompt generated-token counts are unchanged.
  - Decision: keep. This is a small full-suite win over Phase 2T (`+0.18` mean, `+0.18` weighted) and leaves about 2.88 tok/s mean gap to the 100 tok/s goal. Do not repeat append O-projection residual fusion; it is already default-on through the shared helper.
- Phase 2V implemented and kept:
  - Target: append linear-attention recurrent capture still used the old 128-thread Q8 trace kernel plus a separate `dflash_extract_recurrent_attn` pass. The Phase 2U FFI profile showed `qwen.delta_recurrent_prefill_capture_q8_trace` at 1008 calls / 199.52 ms and `qwen.dflash_extract_recurrent_attn` at 1008 calls / 31.39 ms.
  - Candidate: add a direct append recurrent Q8 path that mirrors the accepted tree warp32 reduction order, writes BF16 attention output directly, updates the persistent F32 recurrent state directly, and still writes the exact Q8 rollback trace. Rollback gate: `SUPERSONIC_DFLASH_DISABLE_APPEND_RECURRENT_WARP32=1`, default off.
  - Correctness: `cargo test -p runner --test dflash_tree_delta_parity dflash_append_delta_q8_direct_attention_matches_extract_path --release -- --ignored --nocapture` passed. The test checks BF16 attention bytes, Q8 trace bytes, and final recurrent state versus the old Q8 capture plus extract path.
  - Build: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic` passed with existing warnings.
  - One-prompt A/B: `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_he01.json` 91.16 tok/s versus rollback gate `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_disabled_he01.json` 87.87 tok/s. Both generated 179 tokens and stdout-tail hash `632ea843784fa7b9ed25a50807653dbc3bceca0a5bd2a9c9983145385424d0cf`.
  - FFI shape profile: `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_profile_ffi_shapes_he01.json` shows `qwen.delta_recurrent_prefill_capture_q8_trace_attn` 1008 calls / 132.42 ms. The old `qwen.delta_recurrent_prefill_capture_q8_trace` and `qwen.dflash_extract_recurrent_attn` rows are gone. Total FFI-profile time dropped from 2471.78 ms in Phase 2U to 2368.06 ms.
  - Full-suite run: `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256.json`, mean 100.86 tok/s, weighted 99.40 tok/s, min 87.87, max 118.20, generated 1654, stopped early 10/10.
  - Repeat full-suite run: `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256_rerun.json`, mean 100.79 tok/s, weighted 99.38 tok/s, min 87.87, max 117.92, generated 1654, stopped early 10/10.
  - Combined output hash matches Phase 2U (`032209e65467e8aa6c74025dc8b70b325f0ec767054ddff614e04550bb11f3bf`) and per-prompt generated-token counts are unchanged.
  - Decision: keep default-on with rollback gate. This meets the explicit `100 tok/s` mean objective twice, without prompt-level token-count collapse or output drift. Weighted throughput remains just below 100 tok/s, so future work can still target weighted >=100 if desired, but this phase satisfies the stated mean-throughput goal.
- Previous post-PR #263 full-suite baseline, kept for historical comparison:
  - `target/qwen36_lucebox_next/baseline_10x256.json`
  - mean 85.52 tok/s, weighted 84.39 tok/s, min 74.96, max 99.50, generated 1654.
  - This reproduces the PR #263 checkpoint (`85.35 mean / 84.23 weighted`) within noise.
- Fresh internal verify profile:
  - `target/qwen36_lucebox_next/profile_verify_he01_fulltail.json`
  - he_01 generated 179 tokens, 2314 ms decode, 77.34 tok/s.
  - DFlash breakdown: draft=385 ms, verify=1843 ms, rollback=86 ms.
  - Verify buckets over 21 rounds: linear attention=1166.90 ms, full attention=584.54 ms, logits/greedy=69.47 ms, MLP=8.25 ms.
- Fresh FFI shape profile:
  - `target/qwen36_lucebox_next/profile_ffi_shapes_he01.json`
  - Use for shape/call ranking only; FFI profiling inserts syncs and distorts macro bucket times.
- `rocprofv3` status:
  - Installed AMD ROCm 7.1.1 rpath `rocprofiler-sdk` because Fedora has no native `rocprofv3` provider on this machine.
  - Fixed the profiler/runtime mismatch by also installing matching AMD rpath runtime packages: `hip-runtime-amd-rpath7.1.1`, `hsa-rocr-rpath7.1.1`, `hsa-amd-aqlprofile-rpath7.1.1`, `comgr-rpath7.1.1`, and `rocminfo-rpath7.1.1`.
  - Use `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib SUPERSONIC_BACKENDS=hip /opt/rocm-7.1.1/bin/rocprofv3 ...`.
  - `--version` and `--list-avail` work; counter inventory is at `target/qwen36_lucebox_next/rocprofv3-list-avail.txt`.
  - Trace collection now works for both `/usr/bin/hip_add_kernel` and SuperSonic. Key artifacts: `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_runtime_kernel_trace.csv`, `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_tiny_trace_kernel_trace.csv`, and `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_trace_kernel_trace.csv`.
  - Full-tail trace artifact `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_trace_kernel_stats.csv`: he_01 generated 179 tokens, decode 2451 ms, DFlash draft=389 ms, verify=1908 ms, rollback=154 ms. Top kernel rows are dense INT4 matmul 1020 ms, GGML qtype 12 669 ms, Q6_K MMQ 386 ms, recurrent prefill 300 ms, tree recurrent 182 ms, qtype 14 179 ms, full tree attention 92 ms.
  - PMC works if counter groups are small. Working artifacts include `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_pmc_sqwaves_counter_collection.csv` and `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_pmc_sqwaves_allkernels_counter_collection.csv`.
  - For SuperSonic PMC, prefer all-kernel collection on a representative `n_gen=256` run and filter CSV offline; `--kernel-include-regex` and short 2/32-token runs can miss the tree verifier path or silently produce no useful counter CSV.
  - Do not combine too many PMC counters in one pass. The broad set `SQ_WAVES Wavefronts OccupancyPercent VALUInsts FETCH_SIZE WRITE_SIZE L2CacheHit` exceeded hardware collection capabilities and left a child benchmark process that had to be killed.
  - `GPUBusy` read as zero in the first tree-targeted PMC smoke; rely on `SQ_WAVES` and trace timing first, then add separate memory/VALU/L2 passes only after checking each group.
  - Keep the isolated `/opt/rocm-7.1.1` profiling lane for RX 7900 XTX. Revisit a system-wide ROCm upgrade when the planned RDNA4 card arrives.
- Current implementation direction:
  - Phase 1A implemented and kept: Q8 tree recurrent direct-attention output. It preserves Q8 rollback trace bytes, writes BF16 attention rows directly, and skips the old `dflash_extract_recurrent_attn` launch on the default Q8 tree trace path.
  - Rollback gate: `SUPERSONIC_DFLASH_DISABLE_TREE_DIRECT_ATTENTION=1`.
  - Phase 1B implemented and kept: tree verify now reuses the existing HIP `prepare_conv_input_tail` helper, replacing the old `transpose_pad_conv` + `fill_conv_tail` pair on the tree path. It writes the helper's next-tail output to scratch `linear_new_tail` only, so tree acceptance/rollback semantics remain deferred.
  - Rollback gate: existing `SUPERSONIC_DFLASH_DISABLE_FUSED_CONV_PREP=1`.
  - Phase 1C tried and rejected as a default: direct BA projection + beta/g fusion. The opt-in path exists behind `SUPERSONIC_DFLASH_ENABLE_FUSED_BA_DIRECT=1`, with `SUPERSONIC_DFLASH_DISABLE_FUSED_BA_DIRECT=1` also respected for bisects, but profiling showed the scalar fused projection is much slower than the existing generic BF16 matmul plus beta/g epilogue.
  - Phase 1D implemented and kept: tree full-attention K/V transposes now use one paired HIP launch instead of two separate `transpose_shd_hsd` launches. Rollback gate: `SUPERSONIC_DFLASH_DISABLE_TREE_FULL_KV_TRANSPOSE=1`.
  - Phase 1E implemented but default-off: strided tree full-attention prefix K/V avoids per-round contiguous prefix allocation/copy and has exact parity, but the current kernel is slower than the contiguous-prefix fallback. Keep it as an opt-in profiling/future-kernel path with `SUPERSONIC_DFLASH_ENABLE_TREE_FULL_PREFIX_STRIDED=1`; `SUPERSONIC_DFLASH_DISABLE_TREE_FULL_PREFIX_STRIDED=1` is a hard off-switch.
  - Phase 1F implemented and kept: precompute the tree convolution source-column map once in `PrefillTreeVerifyCache` and use an indexed `linear_tree_conv_pack` HIP helper so each channel element does not re-walk `parent_ids`. Rollback gate: `SUPERSONIC_DFLASH_DISABLE_TREE_CONV_SOURCE_MAP=1`.
  - Build passed: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
  - New ignored GPU parity test passed with `/opt/rocm-7.1.1`: `cargo test -p runner --test dflash_tree_delta_parity dflash_tree_delta_q8_direct_attention_matches_extract_path --release -- --ignored --nocapture`.
  - New ignored GPU conv-prep parity test passed with `/opt/rocm-7.1.1`: `cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_conv_input_prepare_matches_transpose_plus_tail --release -- --ignored --nocapture`.
  - Full A/B artifacts with explicit `SUPERSONIC_DFLASH_DDTREE_VERIFY=1 SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1`:
    - Direct: `target/qwen36_lucebox_next/tree_direct_attn_10x256.json`, mean 85.69 tok/s, weighted 84.56, min 75.13, generated 1654.
    - Gate disabled path: `target/qwen36_lucebox_next/tree_direct_attn_disabled_10x256.json`, mean 85.08 tok/s, weighted 83.96, min 74.63, generated 1654.
    - Combined direct-attn + tree conv-prep: `target/qwen36_lucebox_next/tree_convprep_direct_attn_10x256.json`, mean 86.11 tok/s, weighted 84.98, min 75.70, max 100.30, generated 1654.
    - Combined gated fallback (`SUPERSONIC_DFLASH_DISABLE_TREE_DIRECT_ATTENTION=1 SUPERSONIC_DFLASH_DISABLE_FUSED_CONV_PREP=1`): `target/qwen36_lucebox_next/tree_convprep_direct_attn_disabled_10x256.json`, mean 85.12 tok/s, weighted 84.01, min 74.68, max 99.01, generated 1654.
    - Fresh current default after leaving direct BA opt-in only: `target/qwen36_lucebox_next/tree_phase1_default_10x256.json`, mean 86.05 tok/s, weighted 84.93, min 75.53, max 100.10, generated 1654.
    - K/V pair transpose current default: `target/qwen36_lucebox_next/tree_full_kv_pair_10x256.json`, mean 86.12 tok/s, weighted 84.97, min 75.53, max 100.50, generated 1654.
    - K/V pair transpose rollback gate: `target/qwen36_lucebox_next/tree_full_kv_pair_disabled_10x256.json`, mean 85.97 tok/s, weighted 84.85, min 75.41, max 100.20, generated 1654.
    - Strided prefix opt-in path: `target/qwen36_lucebox_next/tree_full_prefix_strided_10x256.json`, mean 86.21 tok/s, weighted 85.07, min 75.59, max 100.30, generated 1654.
    - Strided prefix disabled/current default path: `target/qwen36_lucebox_next/tree_full_prefix_strided_disabled_10x256.json`, mean 86.27 tok/s, weighted 85.14, min 75.70, max 100.50, generated 1654.
    - Rebuilt final default after making strided prefix opt-in: `target/qwen36_lucebox_next/tree_phase1e_final_default_10x256.json`, mean 86.30 tok/s, weighted 85.18, min 75.70, max 100.91, generated 1654.
    - Indexed tree conv source-map current default: `target/qwen36_lucebox_next/tree_conv_source_map_10x256.json`, mean 86.42 tok/s, weighted 85.30, min 75.93, max 100.81, generated 1654.
    - Indexed tree conv source-map rollback gate: `target/qwen36_lucebox_next/tree_conv_source_map_disabled_10x256.json`, mean 86.34 tok/s, weighted 85.18, min 75.76, max 100.81, generated 1654.
    - Fresh baseline: `target/qwen36_lucebox_next/baseline_10x256.json`, mean 85.52 tok/s, weighted 84.39, min 74.96, generated 1654.
  - FFI shape profile: `target/qwen36_lucebox_next/tree_direct_attn_profile_ffi_shapes_he01.json`, new `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn` is 1008 calls / 207.58 ms, and `qwen.dflash_extract_recurrent_attn` is no longer present on the Q8 tree path.
  - Combined FFI shape profile: `target/qwen36_lucebox_next/tree_convprep_direct_attn_profile_ffi_shapes_he01.json`, `qwen.prepare_conv_input_tail` is 1008 calls / 27.43 ms; `qwen.transpose_pad_conv`, `qwen.fill_conv_tail`, and `qwen.dflash_extract_recurrent_attn` are absent from the optimized tree path. Remaining nearby costs include `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn` 1008 calls / 206.65 ms, `qwen.full_attention_tree_prefill` 336 calls / 108.51 ms, and `qwen.linear_tree_conv_pack` 1008 calls / 30.59 ms.
  - Direct BA opt-in FFI profile: `target/qwen36_lucebox_next/tree_direct_ba_profile_ffi_shapes_he01.json`, `qwen.project_ba_compute_beta_g_bf16` is 960 calls / 271.90 ms. Default-off profile: `target/qwen36_lucebox_next/tree_direct_ba_defaultoff_profile_ffi_shapes_he01.json`, old `qwen.matmul_rhs_transposed[b=1 m=16 n=96 k=5120 dtype=BF16]` is 1008 calls / 67.14 ms and `qwen.compute_beta_g_ba_bf16` is 1008 calls / 26.67 ms. The fused BA candidate is therefore rejected until it gets an MFMA-class implementation.
  - K/V pair FFI shape profile: `target/qwen36_lucebox_next/tree_full_kv_pair_profile_ffi_shapes_he01.json`, `qwen.transpose_shd_hsd` drops to 336 calls / 14.09 ms, with new `qwen.transpose_shd_hsd_pair` at 336 calls / 8.94 ms. Previous Phase 1 default had `qwen.transpose_shd_hsd` at 1008 calls / 31.41 ms.
  - Strided prefix FFI shape profile: `target/qwen36_lucebox_next/tree_full_prefix_strided_profile_ffi_shapes_he01.json`, `qwen.full_attention_tree_prefill_strided` is 336 calls / 121.96 ms versus the contiguous path at 336 calls / 108.36 ms in `tree_full_kv_pair_profile_ffi_shapes_he01.json`. HAL allocation calls drop from 897 to 225 and D2D bytes drop from 1.21 GB to 0.89 GB, but macro throughput is fractionally better with striding disabled, so the path is opt-in only.
  - Indexed tree conv source-map FFI shape profile: `target/qwen36_lucebox_next/tree_conv_source_map_profile_ffi_shapes_he01.json`, `qwen.linear_tree_conv_pack_indexed` is 1008 calls / 30.06 ms versus the old parent-walk `qwen.linear_tree_conv_pack` at 1008 calls / 30.58 ms. Extra source-map upload raises H2D calls from 105 to 126 and bytes from 10 KB to 15 KB, so the win is intentionally tiny.
  - New ignored GPU K/V pair parity test passed with `/opt/rocm-7.1.1`: `cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_full_kv_pair_transpose_matches_separate_calls --release -- --ignored --nocapture`.
  - New ignored GPU strided-prefix parity test passed with `/opt/rocm-7.1.1`: `cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_full_attention_strided_prefix_matches_contiguous_prefix --release -- --ignored --nocapture`.
  - New ignored GPU indexed-conv parity test passed with `/opt/rocm-7.1.1`: `cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_conv_pack_indexed_matches_parent_walk_fixture --release -- --ignored --nocapture`.
  - Normalized stdout tails and generated-token counts in `tree_convprep_direct_attn_10x256.json`, `tree_phase1_default_10x256.json`, `tree_full_kv_pair_10x256.json`, `tree_full_kv_pair_disabled_10x256.json`, `tree_full_prefix_strided_10x256.json`, `tree_full_prefix_strided_disabled_10x256.json`, `tree_phase1e_final_default_10x256.json`, `tree_conv_source_map_10x256.json`, and `tree_conv_source_map_disabled_10x256.json` match both `baseline_10x256.json` and `tree_convprep_direct_attn_disabled_10x256.json`.
  - Combined result is a small `+1.2%` weighted win versus the two-gate fallback, not the full next performance step.
  - Next target remains the tree verify small-M projection train plus tree attention: `matmul_rhs_transposed_int4`/Q6_K projection shapes, `delta_recurrent_tree_prefill_capture_q8_trace_attn`, `linear_tree_conv_pack`, remaining transposes, and full tree attention.
  - Second target is tree full attention.
  - Do not repeat direct BA scalar fusion, MLP-only, lm-head, rollback, or host setup unless a new profile changes the bucket shape.
- Invalid smoke runs from this phase, do not use as evidence:
  - `target/qwen36_lucebox_next/tree_direct_attn_profile_he01.json` and `target/qwen36_lucebox_next/tree_direct_attn_disabled_profile_he01.json` accidentally used `q4km-gptq` and generated 175 tokens.
  - `target/qwen36_lucebox_next/tree_direct_attn_ddtree_he01.json`, `target/qwen36_lucebox_next/tree_direct_attn_disabled_ddtree_he01.json`, `target/qwen36_lucebox_next/tree_direct_attn_ddtree_profile_he01.json`, and `target/qwen36_lucebox_next/tree_direct_attn_disabled_ddtree_profile_he01.json` missed `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1`, used `commit=append-reverify`, and are invalid for PR #263 comparisons.
- Best validated no-env default before the budget change:
  - `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_10x256.json`
  - mean 66.23 tok/s, weighted 65.39, min 58.24, max 77.34, generated 1654.
- Best current validated no-env artifact:
  - `target/qwen36_lucebox20/tree_budget15_default_noenv_10x256.json`
  - mean 85.35 tok/s, weighted 84.23, min 74.85, max 99.40, generated 1654.
  - all 10 output tails and generated-token counts match the explicit budget-15 artifact byte-for-byte.
  - this is 127.4% of the Lucebox Q4 reference and 122.7% of the Lucebox Q8 reference by mean tok/s.
- Completed no-env validation:
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
  - Smoke artifact `target/qwen36_lucebox20/tree_budget15_default_noenv_profile_he01.json`: he_01 generated 179 tokens, 2311 ms decode, 77.46 tok/s; stderr profile shows `tree_verify len=16`, 21 rounds, mean accepted 8.48, DFlash breakdown draft=385 ms, verify=1840 ms, rollback=86 ms.
  - Full artifact `target/qwen36_lucebox20/tree_budget15_default_noenv_10x256.json`: mean 85.35, weighted 84.23, generated 1654.
  - Compared against `tree_budget15_top4_default_10x256.json`: all stdout tails and generated-token counts match.
- Next work: inspect final diff/status, keep this log in the PR, and prepare the PR/checks. Do not rerun budget sweeps.
- Current aggregate FFI/HAL profile is `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_ffi_he01.json`.
- Current shape-level FFI/HAL profile after Q6_K MLP-down is `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_ffi_shapes_he01_fulltail.json`.
- Do not repeat allocation-cache, greedy-cache, GPU-tap, dense MLP pair-helper, GGML small-M N64, bounded `m <= 16`, fused GGML SwiGLU, Q6_K MMQ lm-head, BF16 trace, tree Q8 pre-exp G, or shape-profile experiments unless code changes in those paths.
- Keep `SUPERSONIC_DISABLE_Q6_K_MMQ_MLP_DOWN=1` as the Q6_K MLP-down bisect env, and keep `SUPERSONIC_DFLASH_DDTREE_BUDGET=14` available as the budget default rollback env.

Current best branch artifact:
`target/qwen36_lucebox20/tree_budget15_default_noenv_10x256.json`

Run GPU benchmarks with escalated permissions; sandboxed runs cannot access the HIP device and have previously failed with `HIP error: supersonic_query_gpu_info failed with status 100`.

## Current Repo State

- Repo: `/home/deano/projects/SuperSonicBase`
- Baseline branch: `main`
- Current merge point: `27259d8 Merge pull request #262 from DeanoC/codex/tree-verify-speedup`
- PR #262 status: merged and remote branch deleted.
- Active branch: `codex/lucebox-parity-tree-qkvz`
- Current local edits:
  - `crates/runner/src/prefill_engine.rs`: validated tree fusions, greedy cache, GPU tap capture, rollback-buffer reuse, and default-on Q6_K MMQ MLP-down.
  - `crates/runner/src/decode_engine.rs`: tree GPU tap capture path and owned tree commit recycling for rollback-cache reuse.
  - `crates/runner/src/qwen35_dflash_engine.rs`: tree GPU tap capture path, direct tree rollback commit wiring through the owned commit path, and `DDTREE_DEFAULT_BUDGET = 15`.
  - `kernels/full_attention_bridge.cpp`: validated RMSNorm and generic SwiGLU hard-sync removal.
  - `kernels/full_attention_bridge_4b.cpp`: validated i8 WMMA support-probe fix for Q6_K MMQ; lm-head perf follow-up rejected/no-promote.
  - `docs/qwen36-lucebox-parity-log.md`: this durable log.

## Benchmark Rules That Matter

- Use the corrected Lucebox-style benchmark mode from PR #262.
- Use ChatML with Qwen3 no-thinking prefill:
  `<think>\n\n</think>\n\n`
- Correct prompt token counts for the 10 HumanEval prompts are:
  `[142, 134, 103, 137, 136, 112, 132, 112, 134, 116]`
- Treat old `target/qwen36_he_supersonic_*.json` fallback results as stale.
- Treat pre-Lucebox-mode artifacts as historical only, even if their filenames mention tree cache or tiled scratch.

Canonical SuperSonic command shape:

```bash
SUPERSONIC_DFLASH_DDTREE_VERIFY=1 \
SUPERSONIC_DFLASH_DDTREE_BUDGET=<budget> \
SUPERSONIC_DFLASH_DDTREE_TOP_K=<top_k> \
SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1 \
python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
  --binary target/release/supersonic \
  --model qwen3.6-27b \
  --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
  --quant q4km --backend hip \
  --lucebox-serving-mode --dflash-draft-variant lucebox-q8-0 \
  --n-gen 256 --limit 10 --timeout 1200 \
  --out-json target/qwen36_lucebox20/<name>.json
```

The runner flag for direct binary invocations is `--sampling-seed`, not `--seed`.

## Valid Reference Numbers

Lucebox artifacts:

| Artifact | Draft | Mean tok/s | Output tokens | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `target/qwen36_lucebox20/lucebox_he_10x256_budget8.json` | Q4_K_M | 66.99 | 1636 | 90% |
| `target/qwen36_lucebox20/lucebox_he_10x256_budget8_q8draft.json` | Q8_0 | 69.58 | 1671 | 80% |

Best corrected merged SuperSonic baseline:

| Artifact | Budget | Top-k | Temp | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `target/qwen36_lucebox20/sweep_lucebox_mode_q8_nothink_budget14_top4_10x256.json` | 14 | 4 | default | 59.43 | 58.43 | 52.33 | 69.40 | 1654 |

Best current branch artifact:

| Artifact | Budget | Top-k | Temp | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `target/qwen36_lucebox20/tree_budget15_default_noenv_10x256.json` | 15 | 4 | default | 85.35 | 84.23 | 74.85 | 99.40 | 1654 |

Current gap from best current branch run:

- Versus Lucebox Q4_K_M reference: `85.35 / 66.99 = 127.4%`, about 27.4% ahead.
- Versus Lucebox Q8_0 reference: `85.35 / 69.58 = 122.7%`, about 22.7% ahead.
- This is the no-budget-env result after changing `DDTREE_DEFAULT_BUDGET` to 15 in source.

## Valid SuperSonic Sweep Results

All rows below used corrected Lucebox serving mode, Q8 draft GGUF, no-thinking prefill, 10 prompts, `n_gen=256`.

| Artifact | Budget | Top-k | Temp | Mean tok/s | Weighted tok/s | Min tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `supersonic_lucebox_mode_q8draft_nothink_budget8_10x256.json` | 8 | 4 | default | 50.59 | 50.10 | 46.17 | Apples-to-apples with Lucebox documented budget 8, but not best for SuperSonic. |
| `sweep_lucebox_mode_q8_nothink_budget4_top4_10x256.json` | 4 | 4 | default | 33.18 | 33.02 | 32.02 | Too little acceptance, do not repeat. |
| `sweep_lucebox_mode_q8_nothink_budget8_top8_10x256.json` | 8 | 8 | default | 50.63 | 50.13 | 46.23 | Top-k 8 does not rescue budget 8. |
| `sweep_lucebox_mode_q8_nothink_budget12_top4_10x256.json` | 12 | 4 | default | 57.31 | 56.48 | 47.94 | Close but below budget 14. |
| `sweep_lucebox_mode_q8_nothink_budget12_top8_10x256.json` | 12 | 8 | default | 58.80 | 57.71 | 51.28 | Better than top-k 4, still below budget 14. |
| `sweep_lucebox_mode_q8_nothink_budget14_top4_10x256.json` | 14 | 4 | default | 59.43 | 58.43 | 52.33 | Historical merged-baseline best before this branch's verify/cache/kernel work. |
| `sweep_lucebox_mode_q8_nothink_budget14_top8_10x256.json` | 14 | 8 | default | 59.36 | 58.36 | 52.25 | No useful gain over top-k 4. |
| `sweep_lucebox_mode_q8_nothink_budget14_top4_temp07_10x256.json` | 14 | 4 | 0.7 | 59.44 | 58.46 | 52.49 | Noise-level difference; not a real win. |
| `sweep_lucebox_mode_q8_nothink_budget14_top4_temp13_10x256.json` | 14 | 4 | 1.3 | 59.40 | 58.40 | 52.36 | Noise-level difference; not a real win. |
| `sweep_lucebox_mode_q8_nothink_budget16_top4_10x256.json` | 16 | 4 | default | 41.81 | 41.31 | 37.44 | Too much verify cost, do not repeat unless verify kernel changes. |
| `tree_budget15_top4_default_10x256.json` | 15 | 4 | default via env | 85.38 | 84.24 | 74.74 | Explicit budget-15 run before source default promotion. |
| `tree_budget15_default_noenv_10x256.json` | 15 | 4 | default | 85.35 | 84.23 | 74.85 | Validated source default after promotion; current best branch artifact. |

Takeaway: historical knob tuning alone was not enough, but budget 15 became a major win after the current branch's Q6_K MLP-down and tree verify/cache changes because it aligns the tree workload to `m=16`. Do not rerun old budget sweeps unless code changes materially.

## Profiling Evidence

Corrected Lucebox-mode Q8 draft, budget 8, he_01 profile artifact:
`target/qwen36_lucebox20/profile_lucebox_mode_q8_he01_10prompt_parity.json`

Representative result:

- `he_01`: 179 generated tokens, 3772 ms decode, 47.44 tok/s.
- DFlash breakdown: draft 483 ms, verify 3142 ms, rollback 106 ms, other 42 ms.
- Per tree verify round at len 9:
  - setup/upload: about 0.10 ms
  - embed: about 0.03 ms
  - input norm: about 28.5 ms
  - full attention: about 13-15 ms
  - linear attention: about 31-32 ms
  - post norm: about 10.6 ms
  - MLP: about 29.5 ms
  - taps: about 2.7-2.9 ms
  - logits/greedy: about 8.0 ms

Budget 14/top-k 4 was also manually profiled in the prior run:

- `he_01`: about 3317 ms decode, 18.53 ms/token, 179 generated.
- Tree verify len 15 remained about 130 ms per round.
- Similar bucket shape: input norm, linear attention, MLP, and logits/greedy dominate; full attention is not the largest bucket.

Takeaway: rollback is not the bottleneck. Verify dominates, with draft a distant second.

## Ideas Already Tried Or Ruled Out

- Do not use old fallback benchmark results as evidence. They predate corrected Lucebox-mode prompt handling.
- Do not cite `current_main_supersonic_budget8_append_10x256.json` for parity. It is old append-mode evidence, mean about 5.26 tok/s, and unrelated to the current corrected tree path.
- Do not treat these pre-Lucebox-mode artifacts as valid current wins:
  - `tree_cache_greedy_reuse_10x256.json`: mean 32.61 tok/s
  - `tree_cache_scratch_reuse_10x256.json`: mean 32.74 tok/s
  - `tree_tiled_scratch_defaults_10x256.json`: mean 38.79 tok/s
- Budget/top-k/temp sweeps listed above are already done. Repeating them without code changes is low value; budget 15 has already been validated as the current no-env default on this branch.
- A local experiment to reuse greedy final-norm/logits/argmax buffers in `PrefillTreeVerifyCache` was present after PR #262 but was reverted during cleanup because it was unvalidated and not part of merged `main`. If revisited, put it on a branch and benchmark against budget 14/top-k 4 before keeping it.
- SuperSonic source now defaults to `DDTREE_DEFAULT_BUDGET = 15` and `DDTREE_DEFAULT_TOP_K = 4` in `crates/runner/src/qwen35_dflash_engine.rs`; the no-env validation artifacts named in the checkpoint passed.
- Dense MLP gate/up pairing via the existing `prefill_ffi::matmul_rhs_transposed_ggml_pair` helper was tried after rollback-buffer reuse and rejected. It slowed the one-prompt profile, changed acceptance-round shape, and was reverted completely. Do not retry that helper as a quick performance slice; a real dense HIP gate/up improvement needs new measured kernel work or a packed-weight path.
- `SUPERSONIC_DFLASH_ENABLE_GGML_SMALL_M_N64=1` was tried and rejected for the current DDTree `m=15` workload: 42.02 tok/s versus 58.17 tok/s baseline on the one-prompt smoke.
- A bounded `m <= 16` GGML block-dequant kernel was implemented, benchmarked, rejected, and reverted: 45.05 tok/s versus 58.17 tok/s baseline on the one-prompt smoke.
- A fused GGML gate/up+SwiGLU HIP kernel was implemented, benchmarked, rejected, and reverted: 43.40 tok/s versus 58.17 tok/s baseline on the one-prompt smoke. It made the MLP bucket tiny but moved total verify time badly, with input-norm bucket around 107 ms and verify at 3629 ms.
- Q6_K MMQ MLP-down was implemented with a reusable Q8_1 activation workspace, but the corrected Lucebox-mode smoke failed before generating tokens with HIP status 309 (`device_supports_wmma_i8` false in the MMQ bridge on this gfx1100 path). It was reverted. The i8 WMMA support probe has since been fixed and validated with `int4_test`, so this can be revisited only as a new logged default-off experiment against the current `tree_swiglu_maybesync...` baseline.
- Q6_K MMQ lm-head was re-tested after the i8 WMMA support-probe fix and rejected/no-promote: `target/qwen36_lucebox20/tree_q6k_mmq_lm_head_after_probe_budget14_top4_profile_he01.json` ran 58.62 tok/s versus 59.92 tok/s for the current best smoke, changed generated tokens/round count, and should not be full-suited.
- BF16 recurrent rollback trace was tried with `SUPERSONIC_DFLASH_DISABLE_Q8_ROLLBACK_TRACE=1` and rejected/no-promote: 55.04 tok/s versus 58.17 tok/s for the default Q8-trace one-prompt smoke. Do not promote BF16 trace as the default for this branch.
- Tree Q8 recurrent pre-exp G was implemented, benchmarked, rejected/no-promote, and reverted: 58.11 tok/s versus 58.17 tok/s for the default Q8-trace one-prompt smoke. Do not retry simple pre-exponentiation of `g` unless the recurrent kernel design changes more substantially.
- Shape-level FFI profiling was already run; do not rerun it until the fused SwiGLU path or another code slice changes the relevant matmul shapes.

## Current Validation Queue

Completed:

1. Rebuilt the release binary after changing `DDTREE_DEFAULT_BUDGET` from 14 to 15.
2. Ran the no-budget-env one-prompt smoke artifact named in the checkpoint.
3. Ran the no-budget-env full 10-prompt artifact named in the checkpoint.
4. Compared no-env budget-15 outputs against `tree_budget15_top4_default_10x256.json`; all output tails and generated-token counts match.
5. Updated this log with the no-env result.

Next:

1. Inspect final diff/status.
2. Prepare PR/checks.
3. Only start more kernel work if a fresh review or CI result requires it.

## Active Experiment: Q6_K MMQ MLP-Down After gfx11 Support-Probe Fix

Reason to try:

- Shape-level FFI profile shows dense MLP down projection as a major remaining bucket:
  - `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=14]`: 672 calls, 353.380 ms total.
  - `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=12]`: 672 calls, 292.592 ms total.
- The previous Q6_K MMQ MLP-down attempt failed before a useful performance artifact because the old gfx1100 i8 WMMA support probe returned false.
- The support probe is now fixed and validated by `int4_test`, including `GGML Q6_K MMQ m8` and `GGML Q6_K MMQ m16`.
- This is a fresh default-off experiment, not a promotion.

Implementation plan:

- Add a grow-only `PrefillScratch` U8 workspace for MMQ Q8_1 activations.
- Add default-off env gate `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`.
- In `prefill_mlp_layer`, route only HIP, non-int8, raw GGML Q6_K down projections with no AWQ/native-int4 side tensors through:
  1. reusable Q8_1 quantization of `scratch.mlp_buf`,
  2. `matmul_mmq_q8_1_q6_k` into `scratch.proj_buf`.
- Fallback remains the current `matmul_proj` path.

Validation plan:

1. `cargo fmt --check`
2. `cargo check -p runner --bin supersonic`
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
4. One corrected Lucebox-mode `he_01` profile with `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`.
   - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_after_probe_budget14_top4_profile_he01.json`
   - Compare against current best smoke `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_profile_he01.json`: 179 generated tokens, 2988 ms decode, 16.69 ms/token, 59.92 tok/s.
5. If faster and generated/acceptance shape is sane, run the full 10-prompt suite:
   - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_after_probe_budget14_top4_10x256.json`
6. If it errors, slows total decode, or changes acceptance shape badly, revert only this MLP-down source slice and keep the support-probe fix.

Current status:

- Implemented in `crates/runner/src/prefill_engine.rs`.
- `cargo fmt --check`: passed.
- `cargo check -p runner --bin supersonic`: passed with existing warnings.
- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- One-prompt smoke/profile:
  - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_after_probe_budget14_top4_profile_he01.json`
  - Env: `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`, corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback, verify profile enabled.
  - Result: 179 generated tokens, 2960 ms decode, 16.54 ms/token, 60.46 tok/s.
  - Current best smoke baseline `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_profile_he01.json`: 179 generated tokens, 2988 ms decode, 16.69 ms/token, 59.92 tok/s.
  - Same 21 rounds and same mean accepted per round (`8.48`).
  - DFlash breakdown moved from draft=388/verify=2503/rollback=97 ms to draft=386/verify=2482/rollback=92 ms.
  - Decision before full suite: healthy and faster; run the full 10-prompt suite before keeping/promoting.
- Full corrected 10-prompt suite with `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`:
  - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_after_probe_budget14_top4_10x256.json`
  - Result: mean 66.32 tok/s, weighted 65.49 tok/s, min 58.38, max 77.40, generated tokens 1654.
  - Previous best `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_10x256.json`: mean 65.78 tok/s, weighted 64.68 tok/s, min 57.97, max 76.80, generated tokens 1654.
  - Delta: +0.54 mean tok/s, +0.81 weighted tok/s, about +0.83% mean.
  - Prompt-level timing: 9 prompts improved and `he_05` regressed from 71.68 to 68.17 tok/s; this is not a collapse because output tails and generated-token counts were byte-for-byte identical for all 10 prompts, and `he_05` remains above the suite mean/min.
  - Gap after this run: `66.32 / 66.99 = 99.0%` of Lucebox Q4_K_M reference, about 1.0% behind; `66.32 / 69.58 = 95.3%` of Lucebox Q8_0 reference, about 4.7% behind.
  - Decision: promote this path to default for matching raw Q6_K MLP-down projections, while keeping `SUPERSONIC_DISABLE_Q6_K_MMQ_MLP_DOWN=1` as a bisect/rollback env.
  - Completed validation after promotion: reran the one-prompt smoke and full 10-prompt suite without `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN`.
- Promotion edit:
  - `q6_k_mmq_mlp_down_enabled()` now defaults to on and is disabled only by `SUPERSONIC_DISABLE_Q6_K_MMQ_MLP_DOWN=1`.
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Default no-env one-prompt smoke/profile:
  - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 2977 ms decode, 16.63 ms/token, 60.13 tok/s.
  - Previous best smoke baseline `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_profile_he01.json`: 179 generated tokens, 2988 ms decode, 16.69 ms/token, 59.92 tok/s.
- Default no-env full corrected 10-prompt suite:
  - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_10x256.json`
  - Result: mean 66.23 tok/s, weighted 65.39 tok/s, min 58.24, max 77.34, generated tokens 1654.
  - Previous best `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_10x256.json`: mean 65.78 tok/s, weighted 64.68 tok/s, min 57.97, max 76.80, generated tokens 1654.
  - Prompt-level timing: 9 prompts improved and `he_05` regressed from 71.68 to 68.03 tok/s; all 10 output tails and generated-token counts are byte-for-byte identical to the previous best artifact.
  - Gap after default promotion: `66.23 / 66.99 = 98.9%` of Lucebox Q4_K_M reference, about 1.1% behind; `66.23 / 69.58 = 95.2%` of Lucebox Q8_0 reference, about 4.8% behind.
  - Keep decision: keep the promoted default Q6_K MMQ MLP-down path.

## Completed Diagnostic: Shape-Level INT4 FFI Profile

Reason to run:

- The post-rollback FFI profile proves `qwen.matmul_rhs_transposed_int4` dominates total timed FFI, but aggregate op timing hides which projection shapes matter most.
- `SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES=1` is already wired in `crates/kernel-ffi/src/prefill_ffi.rs` for INT4 and BF16 matmul wrappers. It records shape-qualified profile keys such as `b/m/n/k/group/quant_type`.

Planned command:

- Corrected Lucebox-mode Q8 draft, budget 14, top-k 4, direct rollback.
- Env:
  - `SUPERSONIC_DFLASH_DDTREE_VERIFY=1`
  - `SUPERSONIC_DFLASH_DDTREE_BUDGET=14`
  - `SUPERSONIC_DFLASH_DDTREE_TOP_K=4`
  - `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1`
  - `SUPERSONIC_DFLASH_PROFILE_VERIFY=1`
  - `SUPERSONIC_DFLASH_PROFILE_FFI=1`
  - `SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES=1`
- Artifact: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_ffi_shapes_he01.json`

Expected use:

- Use this as diagnostic evidence only; FFI profiling inserts synchronizations and slows absolute throughput.
- Choose the next implementation slice from the largest shape-qualified INT4/BF16 matmul totals. Do not start another cache/knob experiment from this run.

Result:

- Artifact: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_ffi_shapes_he01.json`
- Corrected Lucebox-mode Q8 draft, budget 14/top-k 4, `he_01`, `n_gen=256`.
- Instrumented throughput: 179 generated tokens, 49.63 tok/s. Use as diagnostic only.
- DFlash breakdown under FFI instrumentation: draft 429 ms, verify 3057 ms, rollback 120 ms.
- FFI total: 28665 calls, 3396.593 ms.
- HAL total: 88899 calls, 3281.979 ms, allocation calls 896, allocated bytes 990714265.

Top shape-qualified FFI entries:

| Shape/op | Calls | Mean ms | Total ms | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `matmul_rhs_transposed_int4[b=1 m=15 n=17408 k=5120 g=128 qt=12]` | 2688 | 0.2395 | 643.642 | Dense MLP gate/up projection shape; largest single bucket. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=14]` | 672 | 0.5259 | 353.380 | Dense MLP down projection shape. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=12]` | 672 | 0.4354 | 292.592 | Dense MLP down projection shape. |
| `delta_recurrent_tree_prefill_capture_q8_trace` | 1008 | 0.2002 | 201.850 | Linear recurrent tree kernel. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=6144 g=128 qt=13]` | 1008 | 0.1925 | 194.064 | Linear attention output/projection shape. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=248320 k=5120 g=128 qt=14]` | 21 | 7.1371 | 149.880 | Tree greedy lm-head. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=16384 k=5120 g=0 qt=12]` | 504 | 0.2233 | 112.543 | Full-attention projection shape. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=10240 k=5120 g=128 qt=14]` | 504 | 0.2162 | 108.956 | Full-attention projection shape. |
| `full_attention_tree_prefill` | 336 | 0.3210 | 107.846 | Tree full-attention kernel. |
| `rms_norm_rows_plain` | 2961 | 0.0346 | 102.445 | Norm overhead. |

Decision:

- The next slice should target the `m=15` dense INT4 projection path, especially MLP gate/up/down shapes.
- Do not retry the rejected dense gate/up pair helper. The viable directions are either a better existing low-bit kernel path for these shapes or a targeted HIP matmul improvement.

## Active Diagnostic: Shape-Level FFI Profile After Default Q6_K MMQ MLP-Down

Reason to run:

- The promoted default Q6_K MMQ MLP-down path moved the full suite to 66.23 tok/s mean, about 1.1% behind Lucebox Q4.
- The previous shape-level profile predates this path, so its down-projection timing is stale.
- Need current shape-qualified FFI evidence before picking the next small slice.

Planned command:

- Corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback, one prompt (`he_01`).
- Env:
  - `SUPERSONIC_DFLASH_DDTREE_VERIFY=1`
  - `SUPERSONIC_DFLASH_DDTREE_BUDGET=14`
  - `SUPERSONIC_DFLASH_DDTREE_TOP_K=4`
  - `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1`
  - `SUPERSONIC_DFLASH_PROFILE_VERIFY=1`
  - `SUPERSONIC_DFLASH_PROFILE_FFI=1`
  - `SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES=1`
- Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_ffi_shapes_he01.json`

Expected use:

- Use this as diagnostic evidence only; FFI profiling inserts synchronizations and slows absolute throughput.
- Compare hot buckets against `tree_rollback_cache_budget14_top4_profile_ffi_shapes_he01.json`.
- Choose the next implementation slice only if a remaining bucket is both large and not already rejected in this log.

Current status:

- Primary artifact with full stderr tail: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_ffi_shapes_he01_fulltail.json`
- Result: 179 generated tokens, 3614 ms decode in the first run and 49.33-49.53 tok/s across profile runs. Use as diagnostic only.
- FFI total: 29337 calls, 3424.780 ms in the full-tail run.
- HAL total: 90244 calls, 3451.359 ms, allocation calls 897, allocated bytes 991008025.

Top shape-qualified FFI entries after default Q6_K MMQ MLP-down:

| Shape/op | Calls | Mean ms | Total ms | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `matmul_rhs_transposed_int4[b=1 m=15 n=17408 k=5120 g=128 qt=12]` | 2688 | 0.2401 | 645.334 | Dense MLP gate/up remains the dominant bucket. |
| `matmul_mmq_q8_1_q6_k[b=1 m=15 n=5120 k=17408]` | 672 | 0.5175 | 347.747 | New promoted Q6_K MLP-down path. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=12]` | 672 | 0.4360 | 292.980 | Q4_K half of MLP down remains. |
| `delta_recurrent_tree_prefill_capture_q8_trace` | 1008 | 0.2015 | 203.098 | Linear recurrent tree kernel. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=6144 g=128 qt=13]` | 1008 | 0.1934 | 194.950 | Linear attention projection shape. |
| `matmul_rhs_transposed_int4[b=1 m=15 n=248320 k=5120 g=128 qt=14]` | 21 | 7.2191 | 151.601 | Tree greedy lm-head; existing MMQ lm-head retest remains rejected. |
| `full_attention_tree_prefill` | 336 | 0.3213 | 107.971 | Tree full-attention kernel. |
| `rms_norm_rows_plain` | 2961 | 0.0345 | 102.278 | Norm overhead. |

Decision:

- Do not retry the rejected dense MLP pair helper, GGML small-M N64, bounded `m <= 16`, or fused GGML gate/up+SwiGLU paths.
- There is no existing Q4_K/Q5_K MMQ matmul consumer to reuse; only Q8_1 quantization and Q6_K MMQ matmul exist.
- Next low-risk test: current default at `SUPERSONIC_DFLASH_DDTREE_BUDGET=15` and top-k 4. This gives `tree_len=16`, potentially hitting the existing exact m16 GGML kernels that are visible in the profile, and it has not been swept on the current code.

## Active Experiment: DDTree Budget 15 For M16 Alignment

Reason to try:

- Current best default budget 14 uses `tree_len=15`, so the hottest MLP gate/up and down shapes run as `m=15`.
- Existing exact m16 GGML kernels are present and appear faster for comparable Q4/Q6 shapes in the current profile.
- Prior corrected sweeps tested budgets 14 and 16 but not 15, and they predate the current promoted MLP-down path.

Validation plan:

1. One corrected Lucebox-mode `he_01` profile with budget 15/top-k 4.
   - Artifact: `target/qwen36_lucebox20/tree_budget15_top4_default_profile_he01.json`
   - Compare against current best default smoke `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_he01.json`: 179 generated tokens, 2977 ms decode, 60.13 tok/s.
2. If faster and generated/acceptance shape is sane, run the full 10-prompt suite.
   - Artifact: `target/qwen36_lucebox20/tree_budget15_top4_default_10x256.json`
3. If slower or acceptance changes badly, reject/no-promote budget 15 and keep budget 14 default.

Current status:

- One-prompt smoke/profile:
  - Artifact: `target/qwen36_lucebox20/tree_budget15_top4_default_profile_he01.json`
  - Result: 179 generated tokens, 2314 ms decode, 12.93 ms/token, 77.34 tok/s.
  - Budget 14 current default smoke `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_profile_he01.json`: 179 generated tokens, 2977 ms decode, 16.63 ms/token, 60.13 tok/s.
  - Same output tail and generated-token count as budget 14.
  - Same 21 rounds and same mean accepted per round (`8.48`).
  - DFlash breakdown moved from draft=386/verify=2482/rollback=92 ms at budget 14 to draft=385/verify=1840/rollback=89 ms at budget 15.
  - Decision before full suite: very strong positive signal; run the full corrected 10-prompt suite.
- Full corrected 10-prompt suite with `SUPERSONIC_DFLASH_DDTREE_BUDGET=15`:
  - Artifact: `target/qwen36_lucebox20/tree_budget15_top4_default_10x256.json`
  - Result: mean 85.38 tok/s, weighted 84.24 tok/s, min 74.74, max 99.60, generated tokens 1654.
  - Budget 14 current default artifact `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_default_budget14_top4_10x256.json`: mean 66.23 tok/s, weighted 65.39 tok/s, min 58.24, max 77.34, generated tokens 1654.
  - All 10 prompts improved versus budget 14, and all 10 output tails/generated-token counts are byte-for-byte identical.
  - Gap after budget-15 run: `85.38 / 66.99 = 127.5%` of Lucebox Q4_K_M reference and `85.38 / 69.58 = 122.7%` of Lucebox Q8_0 reference.
  - Decision: promote `DDTREE_DEFAULT_BUDGET` from 14 to 15. Keep `SUPERSONIC_DFLASH_DDTREE_BUDGET=14` available as the env override if needed.
  - Completed validation after promotion: rebuilt and reran the one-prompt smoke plus full 10-prompt suite without setting `SUPERSONIC_DFLASH_DDTREE_BUDGET`.
- Source-default promotion validation:
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
  - No-budget-env smoke artifact: `target/qwen36_lucebox20/tree_budget15_default_noenv_profile_he01.json`.
  - Smoke result: `he_01`, 179 generated tokens, 2311 ms decode, 77.46 tok/s. Stderr profile shows `tree_verify len=16`, 21 rounds, mean accepted 8.48, DFlash breakdown draft=385 ms, verify=1840 ms, rollback=86 ms.
  - No-budget-env full artifact: `target/qwen36_lucebox20/tree_budget15_default_noenv_10x256.json`.
  - Full result: mean 85.35 tok/s, weighted 84.23 tok/s, min 74.85, max 99.40, generated tokens 1654.
  - All 10 generated-token counts and stdout tails match `tree_budget15_top4_default_10x256.json`.
  - Final decision: keep `DDTREE_DEFAULT_BUDGET = 15`.

## Rejected Slice: Evaluate GGML Small-M N64 Kernel

Reason to try:

- The shape profile shows nearly all remaining hot INT4 calls are GGML K-block matmuls with `m=15`.
- The HIP bridge already has a guarded raw-GGML small-M N64 dispatch behind `SUPERSONIC_DFLASH_ENABLE_GGML_SMALL_M_N64=1`.
- The N64 kernel bounds output rows by `m`, so it can safely run with `m=15` without padding output buffers.

Planned validation:

1. One corrected Lucebox-mode Q8 draft `he_01` profile with budget 14/top-k 4 and `SUPERSONIC_DFLASH_ENABLE_GGML_SMALL_M_N64=1`.
   - Artifact: `target/qwen36_lucebox20/tree_ggml_small_m_n64_budget14_top4_profile_he01.json`
   - Compare against `tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
2. If healthy, run the full corrected 10-prompt suite.
3. If it regresses, record and do not promote the env/default.

Result:

- Artifact: `target/qwen36_lucebox20/tree_ggml_small_m_n64_budget14_top4_profile_he01.json`
- Result: 179 generated tokens, 42.02 tok/s.
- Baseline rollback-cache smoke: 179 generated tokens, 58.17 tok/s.

Reject decision:

- Do not enable/promote `SUPERSONIC_DFLASH_ENABLE_GGML_SMALL_M_N64` for DDTree verify.
- It is materially slower for the current `m=15` tree matmul workload.
- Next candidate: build or test a bounded `m <= 16` block-dequant small-M kernel, because the existing optimized `m16` block path is skipped by the DDTree `m=15` shape.

## Rejected Slice: Bounded GGML M<=16 Block Kernel

Reason to try:

- Current DDTree verify uses `m=15`, so it skips the existing exact-`m16` GGML block-dequant small-M kernel.
- The new path is based on the existing `m16` block-loop kernel but keeps `m` row bounds and true output stride.
- It is guarded by `SUPERSONIC_DFLASH_ENABLE_GGML_MLE16_BLOCK=1` for initial validation; default behavior is unchanged until benchmark evidence says to promote it.

Implementation under test:

- `kernels/full_attention_4b.hip`: added `supersonic_qwen35_matmul_ggml_dequant_wmma_mle16_qtype_kernel<QTYPE, TRUNC_DEQUANT>`.
- `kernels/full_attention_bridge_4b.cpp`: dispatches to it for raw GGML K-block matmuls when:
  - `8 < m < 16`,
  - `n` is divisible by 16,
  - `k` is divisible by 256,
  - `awq_inv_scale == nullptr`,
  - `SUPERSONIC_DFLASH_ENABLE_GGML_MLE16_BLOCK=1`.

Validation plan:

1. Rebuild release binary.
2. One corrected Lucebox-mode `he_01` profile with the env gate enabled.
   - Artifact: `target/qwen36_lucebox20/tree_ggml_mle16_block_budget14_top4_profile_he01.json`
3. If healthy, run a shape-level FFI profile to confirm hot `m=15` INT4 totals moved.
4. If one-prompt throughput improves, run the full corrected 10-prompt suite.

Result:

- Release build passed with existing warnings.
- Artifact: `target/qwen36_lucebox20/tree_ggml_mle16_block_budget14_top4_profile_he01.json`
- Result: 179 generated tokens, 45.05 tok/s.
- Baseline rollback-cache smoke: 179 generated tokens, 58.17 tok/s.

Reject decision:

- Revert this slice. The block-loop adaptation is slower than the current generic `m=15` qtype kernel.
- Do not retry bounded `m <= 16` block-dequant dispatch as a quick win.

## Rejected Slice: Fused GGML Gate/Up SwiGLU

Reason to try:

- Shape profile shows the largest bucket is dense MLP gate/up:
  `matmul_rhs_transposed_int4[b=1 m=15 n=17408 k=5120 g=128 qt=12]`, 2688 calls, 643.642 ms.
- Current path launches gate matmul, up matmul, then `swiglu_mul`, materializing two `[tree_len, intermediate]` BF16 buffers before the SwiGLU output.
- A fused gate/up+SwiGLU kernel can compute both Q4_K dot products for the same output column while reusing the LHS row tile, round each accumulator to BF16 to match current materialization, apply the existing `gate / (1 + exp(-gate)) * up`, and write only `mlp_buf`.

Guard used during experiment:

- New runtime path must be gated by `SUPERSONIC_DFLASH_ENABLE_FUSED_GGML_SWIGLU_PAIR=1` for initial validation.
- It should activate only on HIP, batch 1, small `seq_len`, matching raw GGML gate/up quant types, no AWQ/sidecar/native-int4 scale tensors, and BF16 output.
- The rejected `matmul_rhs_transposed_ggml_pair` helper remains rejected; this is a different kernel because it fuses the SwiGLU epilogue and preserves BF16 materialization semantics before activation.

Implementation status:

- Implemented:
  - `kernels/full_attention_4b.hip`: HIP kernel body.
  - `kernels/full_attention_bridge_4b.cpp`: bridge/export.
  - `crates/kernel-ffi/src/prefill_ffi.rs`: extern/wrapper/profile key.
  - `crates/runner/src/prefill_engine.rs`: gated MLP call path.
- Pre-benchmark validation:
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Benchmark result:
  - Artifact: `target/qwen36_lucebox20/tree_fused_ggml_swiglu_pair_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 43.40 tok/s.
  - Baseline rollback-cache smoke: 179 generated tokens, 58.17 tok/s.
  - DFlash breakdown: draft 400 ms, verify 3629 ms, rollback 95 ms.
  - Profile shape: MLP bucket fell to about 0.2 ms, but input-norm bucket jumped to about 107 ms; total verify/decode regressed badly.
- Reject decision:
  - Revert this slice completely.
  - Do not retry this fused GGML gate/up+SwiGLU approach as implemented. Any future revisit needs a different kernel design and must start from a fresh hypothesis about synchronization/timing side effects.

Post-result action:

- Source changes for this slice were removed from `full_attention_4b.hip`, `full_attention_bridge_4b.cpp`, `prefill_ffi.rs`, and `prefill_engine.rs`.
- Keep the artifact only as negative evidence.
- Post-revert validation passed:
  - `cargo fmt --check`
  - `cargo check -p runner --bin supersonic`
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`

## Rejected Slice: Existing Q6_K MMQ Lm-Head For Tree M=15

Reason to try:

- Shape-level FFI profile shows tree greedy lm-head as `matmul_rhs_transposed_int4[b=1 m=15 n=248320 k=5120 g=128 qt=14]`, 21 calls, 149.880 ms total.
- `prefill_engine.rs` already has `maybe_matmul_q6_k_mmq_lm_head`, but its default gate only enables for `m == 8`; the env var `SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD=1` enables it for other `m` values.
- This is an env-only experiment using existing code, not a new kernel.

Planned validation:

1. One corrected Lucebox-mode `he_01` profile with budget 14/top-k 4 and `SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD=1`.
   - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_lm_head_budget14_top4_profile_he01.json`
   - Compare against `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
2. If healthy, run shape-level FFI profile to confirm lm-head shape moved.
3. If one-prompt throughput improves, run full corrected 10-prompt suite.

Result:

- Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_lm_head_budget14_top4_profile_he01.json`
- Result: 179 generated tokens, 58.17 tok/s, decode 3078 ms.
- Baseline rollback-cache smoke: 179 generated tokens, 58.17 tok/s.
- DFlash breakdown: draft 399 ms, verify 2584 ms, rollback 95 ms.
- Logits/greedy bucket stayed around 7.9-8.2 ms, effectively unchanged from baseline.

Reject/no-promote decision:

- Do not promote `SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD=1` for tree `m=15`.
- Do not spend a shape-level FFI run on this env-only path unless the implementation changes to reuse q8 workspace or otherwise change the actual lm-head path.

## Rejected Follow-Up: Q6_K MMQ Lm-Head After gfx11 Support-Probe Fix

Reason to re-test:

- The original Q6_K MMQ lm-head test likely did not exercise the MMQ bridge on gfx1100 because `device_supports_wmma_i8(device_ordinal)` returned false.
- `kernels/full_attention_bridge_4b.cpp` was changed so the i8 WMMA support probe follows the same runtime gfx11 device probe as BF16 WMMA instead of relying on host-side `__gfx1100__` macros.
- `HIP_ARCH=gfx1100 cargo build --release --bin int4_test` passed, and `target/release/int4_test` passed the `GGML Q6_K MMQ m8` and `GGML Q6_K MMQ m16` correctness cases.

Validation:

- Env-only run with corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback, and `SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD=1`.
- Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_lm_head_after_probe_budget14_top4_profile_he01.json`
- Result: 193 generated tokens, 3294 ms decode, 17.06 ms/token, 58.62 tok/s.
- Current best smoke baseline: `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_profile_he01.json`, 179 generated tokens, 2988 ms decode, 16.69 ms/token, 59.92 tok/s.
- DFlash breakdown in after-probe run: draft 443 ms, verify 2758 ms, rollback 93 ms, 24 rounds, mean accepted 8.00.
- The logits/greedy bucket dropped from about 8.4 ms to about 4.0 ms per tree verify, but total decode regressed and the generated-token/round shape changed.

Reject/no-promote decision:

- Do not promote `SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD=1`.
- Do not run a full 10-prompt suite for this env path.
- The support-probe fix itself remains validated by `int4_test`; the negative perf result is specific to the lm-head env path.

## Rejected Slice: Q6_K MMQ MLP Down Projection

Reason to try:

- Shape-level FFI profile shows dense MLP down projection as a major remaining bucket:
  - `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=14]`: 672 calls, 353.380 ms total.
  - `matmul_rhs_transposed_int4[b=1 m=15 n=5120 k=17408 g=128 qt=12]`: 672 calls, 292.592 ms total.
- Existing MMQ support only consumes Q6_K weights, so this slice targets the `qt=14` half of the down-projection bucket first.
- The rejected lm-head MMQ test allocated a Q8_1 workspace inside the helper and was flat; this slice adds reusable `PrefillScratch` Q8_1 workspace so MLP down calls do not allocate per layer/round.

Implementation attempted:

- Add `PrefillScratch` grow-only U8 workspace for MMQ Q8_1 activations.
- Add default-off env gate: `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`.
- In `prefill_mlp_layer`, route only non-int8 Q6_K down projections with no AWQ/native-int4 side tensors through:
  1. `quantize_mmq_q8_1` on `scratch.mlp_buf`,
  2. `matmul_mmq_q8_1_q6_k` into `scratch.proj_buf`.
- Fallback remains the current `matmul_proj` path.

Validation plan used:

1. `cargo fmt --check`
2. `cargo check -p runner --bin supersonic`
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
4. One corrected Lucebox-mode `he_01` profile:
   - Env: `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`
   - Artifact: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_budget14_top4_profile_he01.json`
   - Compare against `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
5. If healthy and faster, run shape-level FFI profile and full 10-prompt suite.
6. If one-prompt throughput is flat/worse or acceptance shape changes badly, revert this slice and record it here.

Result:

- Implemented in `crates/runner/src/prefill_engine.rs`.
- `cargo fmt --check`: passed.
- `cargo check -p runner --bin supersonic`: passed with existing warnings.
- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Benchmark command:
  - Env: `SUPERSONIC_ENABLE_Q6_K_MMQ_MLP_DOWN=1`, corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback.
  - Artifact path requested: `target/qwen36_lucebox20/tree_q6k_mmq_mlp_down_budget14_top4_profile_he01.json`.
- Benchmark failed immediately before producing a useful performance artifact:
  - Error: `q6_k_mmq MLP down matmul: HIP error: matmul_mmq_q8_1_q6_k failed: 309`.
  - Status 309 maps to `device_supports_wmma_i8(device_ordinal)` failing in `kernels/full_attention_bridge_4b.cpp`.

Reject/revert decision:

- Source changes for this slice were reverted from `crates/runner/src/prefill_engine.rs`.
- Post-revert validation passed:
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Do not retry Q6_K MMQ MLP-down on gfx1100 unless the implementation avoids the i8 WMMA requirement or the bridge support changes.

## Rejected Diagnostic: BF16 Recurrent Trace Versus Q8 Trace

Reason to run:

- The next isolated hot kernel after dense INT4 matmuls is `qwen.delta_recurrent_tree_prefill_capture_q8_trace`: 1008 calls, 201.850 ms total in the shape-level `he_01` profile.
- The Q8 trace path quantizes each recurrent state block during verify; rollback apply is now relatively small after rollback-buffer reuse.
- The repo already has a BF16 transposed trace path guarded by existing env controls, so this can be tested without new code.

Planned validation:

1. One corrected Lucebox-mode Q8 draft `he_01` profile with budget 14/top-k 4 and `SUPERSONIC_DFLASH_DISABLE_Q8_ROLLBACK_TRACE=1`.
   - Artifact: `target/qwen36_lucebox20/tree_bf16_trace_budget14_top4_profile_he01.json`
   - Compare against `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
2. If one-prompt throughput is clearly faster and acceptance shape does not collapse, run the full corrected 10-prompt suite.
3. If it is flat or slower, keep Q8 trace default and move to kernel/code optimization.

Result:

- Artifact: `target/qwen36_lucebox20/tree_bf16_trace_budget14_top4_profile_he01.json`
- Env-only run: `SUPERSONIC_DFLASH_DISABLE_Q8_ROLLBACK_TRACE=1`, corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback, verify profile enabled.
- Result: 193 generated tokens, 3508 ms decode, 55.04 tok/s.
- Baseline default Q8-trace smoke: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json`, 179 generated tokens, 3076 ms decode, 58.17 tok/s.
- BF16 trace lowered the profiled per-round linear-attn bucket to about 17.6 ms, but it changed the generated output/rounding path, required 24 rounds instead of 21, and slowed the total decode.

Reject/no-promote decision:

- Keep the default Q8 recurrent trace path.
- Do not run the full 10-prompt suite for BF16 trace.
- Future linear-attention work should optimize the Q8 trace kernel directly or change the recurrent trace representation with a deterministic acceptance/parity validation plan.

## Rejected Slice: Tree Q8 Recurrent Pre-Exp G

Reason to try:

- The tree Q8 recurrent kernel computes `exp(g[h,t])` inside every K/V lane, even though `g[h,t]` is one scalar per value-head/tree-node.
- For Qwen3.6 tree verify this repeats the same exponential across `K * V = 16384` lanes for each head/node scalar.
- The existing fused BA producer already writes `beta` and `g` once per `[head, node]`, so an opt-in tree path can write pre-exponentiated decay once and use a matching Q8 recurrent kernel variant.
- This preserves the Q8 recurrent trace representation and rollback apply path, unlike the rejected BF16 trace diagnostic.

Implementation plan:

- Add default-off env gate: `SUPERSONIC_DFLASH_TREE_RECURRENT_PREEXP_G=1`.
- Add HIP helper `compute_beta_expg_ba_bf16` that writes `beta[h,t]` and `exp(g_log[h,t])` into the existing `linear_g` scratch buffer.
- Add Q8 tree recurrent capture FFI variant that consumes pre-exponentiated `g` directly instead of calling `expf(g)` inside the hot K/V lane loop.
- Wire only the tree path when:
  - rollback trace dtype is Q8/U8,
  - fused BA weights are present,
  - HIP backend is active,
  - the env gate is set.
- Fallback remains the current Q8 trace path.

Validation plan:

1. `cargo fmt --check`
2. `cargo check -p runner --bin supersonic`
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
4. One corrected Lucebox-mode `he_01` profile with `SUPERSONIC_DFLASH_TREE_RECURRENT_PREEXP_G=1`.
   - Artifact: `target/qwen36_lucebox20/tree_q8_preexp_g_budget14_top4_profile_he01.json`
   - Compare against `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
5. If faster and generated/acceptance shape is sane, run shape-level FFI or full 10-prompt suite depending on how large the one-prompt win is.

Result:

- Implemented in:
  - `kernels/prefill_helpers.hip`
  - `kernels/prefill_helpers_bridge.cpp`
  - `kernels/full_attention.hip`
  - `kernels/full_attention_bridge.cpp`
  - `crates/kernel-ffi/src/prefill_ffi.rs`
  - `crates/runner/src/prefill_engine.rs`
- Pre-benchmark validation passed:
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Artifact: `target/qwen36_lucebox20/tree_q8_preexp_g_budget14_top4_profile_he01.json`
- Env: `SUPERSONIC_DFLASH_TREE_RECURRENT_PREEXP_G=1`, corrected Lucebox-mode Q8 draft, budget 14/top-k 4, direct rollback, verify profile enabled.
- Result: 179 generated tokens, 3080 ms decode, 58.11 tok/s.
- Baseline default Q8-trace smoke: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json`, 179 generated tokens, 3076 ms decode, 58.17 tok/s.

Reject/revert decision:

- Do not promote `SUPERSONIC_DFLASH_TREE_RECURRENT_PREEXP_G`.
- The simple pre-exp split is effectively flat and slightly slower at end-to-end decode, even though it removes repeated `expf` from the recurrent kernel.
- Source changes for this slice were reverted.
- Post-revert validation passed:
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Any future recurrent-kernel work needs a larger design change than precomputing `g`, such as changing work decomposition/reduction/trace writes with parity validation.

## Validated Slice: Remove Generic RMSNorm Hard Sync

Reason to try:

- `prefill_ffi::rms_norm_rows_plain` calls the generic HIP symbol `supersonic_qwen35_hip_rms_norm` in `kernels/full_attention_bridge.cpp`.
- That bridge's `rms_norm_device` still calls `hipDeviceSynchronize()` unconditionally after every RMSNorm launch.
- The tree verifier runs input and post-attention RMSNorm for every layer; shape-level FFI profile shows `qwen.rms_norm_rows_plain` at 2961 calls and 102.445 ms in the instrumented `he_01` run.
- The 4b bridge's RMSNorm device already uses the normal `maybe_sync()` convention, so this change aligns the generic bridge with existing local behavior: sync only when `SUPERSONIC_SYNC_EACH_KERNEL` is set.

Implementation plan:

- In `kernels/full_attention_bridge.cpp`, change only `rms_norm_device` from unconditional `hipDeviceSynchronize()` to `maybe_sync()`.
- Do not touch other generic bridge hard syncs in this slice.
- Preserve launch-error checking and return codes.

Validation plan:

1. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
2. One corrected Lucebox-mode `he_01` profile with budget 14/top-k 4.
   - Artifact: `target/qwen36_lucebox20/tree_rms_norm_maybesync_budget14_top4_profile_he01.json`
   - Compare against `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json` at 58.17 tok/s.
3. If faster and generated/acceptance shape is unchanged, run the full corrected 10-prompt suite.

Current status:

- Implementation is present in `kernels/full_attention_bridge.cpp`.
- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- One-prompt smoke/profile:
  - Artifact: `target/qwen36_lucebox20/tree_rms_norm_maybesync_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 3008 ms decode, 16.80 ms/token, 59.52 tok/s.
  - Baseline rollback-cache smoke: 179 generated tokens, 3076 ms decode, 17.19 ms/token, 58.17 tok/s.
  - Same generated-token count and same 21-round acceptance shape as baseline.
  - DFlash breakdown moved from draft=400/verify=2585/rollback=92 ms to draft=389/verify=2523/rollback=95 ms.
- Full corrected 10-prompt suite:
  - Artifact: `target/qwen36_lucebox20/tree_rms_norm_maybesync_budget14_top4_10x256.json`
  - Result: mean 65.34 tok/s, weighted 64.24 tok/s, min 57.54, max 76.28, generated tokens 1654.
  - Baseline rollback-cache suite: mean 63.78 tok/s, weighted 62.71 tok/s, min 56.21, max 74.46, generated tokens 1654.
  - All prompts improved; there is no prompt-level collapse in this suite.

Keep/revert decision:

- Keep. This slice is validated and becomes the current best branch artifact.

## Validated Slice: Remove Generic SwiGLU Hard Sync

Reason to try:

- `prefill_ffi::swiglu_mul` calls the generic HIP symbol `supersonic_qwen35_hip_swiglu_mul` in `kernels/full_attention_bridge.cpp`.
- That bridge's `swiglu_mul_device` called `hipDeviceSynchronize()` unconditionally, while the 4b bridge's matching helper already uses `maybe_sync()`.
- Shape-level FFI profile showed `qwen.swiglu_mul` at 1449 calls and 37.771 ms in the instrumented `he_01` run.

Implementation:

- In `kernels/full_attention_bridge.cpp`, changed only `swiglu_mul_device` from unconditional `hipDeviceSynchronize()` to `maybe_sync()`.
- Did not touch `swiglu_mul_split_device` or unrelated helpers.

Validation:

- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- One-prompt smoke:
  - Artifact: `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 2988 ms decode, 16.69 ms/token, 59.92 tok/s.
  - RMSNorm-only smoke baseline: 179 generated tokens, 3008 ms decode, 16.80 ms/token, 59.52 tok/s.
  - Same generated-token count and same 21-round acceptance shape.
- Full corrected 10-prompt suite:
  - Artifact: `target/qwen36_lucebox20/tree_swiglu_maybesync_budget14_top4_10x256.json`
  - Result: mean 65.78 tok/s, weighted 64.68 tok/s, min 57.97, max 76.80, generated tokens 1654.
  - RMSNorm-only suite baseline: mean 65.34 tok/s, weighted 64.24 tok/s, min 57.54, max 76.28, generated tokens 1654.
  - All prompts improved; there is no prompt-level collapse in this suite.

Keep/revert decision:

- Keep. This slice is validated and becomes the current best branch artifact.

## Branch And Logging Discipline

For this active parity PR/branch:

1. Continue from `codex/lucebox-parity-tree-qkvz` while its validated QKVZ/QKV-prep/gated-epilogue, greedy-cache, GPU-tap, and rollback-cache work is still unmerged.
2. Do not restart from `main` or recreate earlier changes unless the user explicitly asks or this branch is merged/abandoned.
3. Add new benchmark output under `target/qwen36_lucebox20/` with a descriptive name.
4. Update this log before coding a new idea with the hypothesis, target files, command/env, and expected artifact.
5. Update this log after every benchmark or failed run with:
   - command/env used
   - artifact path
   - mean, weighted, min, max tok/s
   - whether it is valid corrected Lucebox-mode evidence
   - whether to keep, revert, or investigate further

Do not open a PR from a performance slice until the log says which baseline it compares against and whether the full 10-prompt result moves the parity gap.

## Validated Branch Work: Tree QKVZ Fusion

Branch: `codex/lucebox-parity-tree-qkvz`

Change under test:

- `crates/runner/src/prefill_engine.rs`
- `prefill_tree_linear_attention_layer` now mirrors the append verifier's fused QKVZ path.
- The gate is the existing `SUPERSONIC_DISABLE_FUSED_QKVZ` / `SUPERSONIC_ENABLE_FUSED_QKVZ` behavior plus the Qwen3.6 default condition `(hidden_size == 5120 && num_hidden_layers == 64)`.
- When enabled and `lw.qkvz_proj_w` exists on HIP, tree linear attention runs one `qkvz` projection into `scratch.mlp_buf`, then `split_qkvz_bf16` into `scratch.proj_buf` and `scratch.proj_buf2`.
- Fallback keeps the prior separate QKV and Z projections exactly.
- Second pass added after the first full run:
  - Tree linear attention now uses existing HIP `split_norm_transpose_qkv_bf16` under `SUPERSONIC_DFLASH_DISABLE_FUSED_QKV_PREP` control, mirroring append.
  - Tree linear attention now uses existing HIP `rms_norm_gated_sfirst_bf16` under `SUPERSONIC_DFLASH_DISABLE_FUSED_GATED_EPILOGUE` control, mirroring append.
  - Both fall back to the prior split/cast/norm/transpose and gated RMSNorm sequences.
- Third pass under test:
  - `PrefillTreeVerifyCache` now owns reusable `greedy_logits_gpu` and `greedy_indices_gpu` buffers sized to `[tree_len, vocab]` and `[tree_len]`.
  - Cached tree verify uses these buffers for final RMSNorm, lm-head, GPU argmax, and ID D2H when `greedy_only` is true.
  - This intentionally does not resize `PrefillScratch.logits_buf`, which must stay small for normal large prefill.

Expected validation:

1. `cargo fmt --check`
2. `cargo check -p runner --bin supersonic`
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
4. One corrected Lucebox-mode smoke/profile run with budget 14/top-k 4.
5. Full corrected 10-prompt budget 14/top-k 4 run against `sweep_lucebox_mode_q8_nothink_budget14_top4_10x256.json`.

Keep criteria:

- No correctness/build regression.
- Full 10-prompt mean improves over 59.43 tok/s, or profiling shows a clear linear-attention reduction worth combining with another small change.
- If mean is flat or worse and profile shows no linear-attention reduction, revert this branch change and record the result here.

Validation so far:

- `cargo fmt --check`: passed.
- `cargo check -p runner --bin supersonic`: passed with existing warnings.
- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- Smoke/profile:
  - Command: corrected Lucebox-mode Q8 draft, budget 14, top-k 4, prompt `he_01`, `n_gen=256`, `SUPERSONIC_DFLASH_PROFILE_VERIFY=1`.
  - Artifact: `target/qwen36_lucebox20/tree_qkvz_fusion_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 3257 ms decode, 18.19 ms/token, 54.98 tok/s.
  - Prior valid budget 14/top-k 4 artifact for `he_01`: 3305 ms decode, 54.17 tok/s.
  - Profile tail still shows tree verify dominance: draft 408 ms, verify 2713 ms, rollback 101 ms, other 35 ms.
  - Representative tree verify len 15 buckets: input norm about 29 ms, full attention about 14-16 ms, linear attention about 32.8-33.2 ms, MLP about 30.1-30.6 ms, logits/greedy about 8.0 ms.
  - This is positive enough for a full 10-prompt run, but not yet proof of a meaningful parity gain.

Full 10-prompt validation:

- Command: corrected Lucebox-mode Q8 draft, budget 14, top-k 4, `n_gen=256`.
- Artifact: `target/qwen36_lucebox20/tree_qkvz_fusion_budget14_top4_10x256.json`
- Result: mean 60.36 tok/s, weighted 59.34 tok/s, min 53.19 tok/s, max 70.67 tok/s, generated 1654 tokens.
- Prior valid baseline artifact: `target/qwen36_lucebox20/sweep_lucebox_mode_q8_nothink_budget14_top4_10x256.json`
- Prior baseline: mean 59.43 tok/s, weighted 58.43 tok/s, min 52.33 tok/s, max 69.40 tok/s, generated 1654 tokens.
- Delta: +0.94 mean tok/s, +0.91 weighted tok/s, about +1.6%.
- Prompt-level result: all 10 prompts improved; no hidden prompt-level collapse.
- Gap after this change:
  - Versus Lucebox Q4_K_M reference: `60.36 / 66.99 = 90.1%`, about 9.9% behind.
  - Versus Lucebox Q8_0 reference: `60.36 / 69.58 = 86.8%`, about 13.2% behind.
- Keep decision: keep the QKVZ tree fusion change. It is not enough for parity, but it is valid corrected Lucebox-mode progress.

Next validation needed after second pass:

- Re-run `cargo fmt --check`, `cargo check -p runner --bin supersonic`, and release build. Done; all passed with existing warnings.
- Re-run one corrected Lucebox-mode `he_01` profile with budget 14/top-k 4. Done:
  - Artifact: `target/qwen36_lucebox20/tree_linear_fusions_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 3148 ms decode, 17.59 ms/token, 56.85 tok/s.
  - QKVZ-only smoke: 3257 ms decode, 54.98 tok/s.
  - Prior valid baseline smoke row: 3305 ms decode, 54.17 tok/s.
  - DFlash breakdown: draft 409 ms, verify 2599 ms, rollback 105 ms, other 36 ms.
  - Profile bucket caveat: linear-attention bucket dropped to about 18.1-18.4 ms, while post-norm rose to about 19.5-19.9 ms. Treat per-bucket movement as async boundary-sensitive; the useful signal is lower total verify/decode time.
- If profile is healthy, re-run the full 10-prompt benchmark with a new artifact name.

Full 10-prompt validation after second pass:

- Command: corrected Lucebox-mode Q8 draft, budget 14, top-k 4, `n_gen=256`.
- Artifact: `target/qwen36_lucebox20/tree_linear_fusions_budget14_top4_10x256.json`
- Result: mean 62.54 tok/s, weighted 61.50 tok/s, min 55.10 tok/s, max 73.05 tok/s, generated 1654 tokens.
- Delta versus QKVZ-only artifact:
  - QKVZ-only: mean 60.36 tok/s, weighted 59.34 tok/s, min 53.19 tok/s.
  - Combined linear fusions: +2.18 mean tok/s, +2.15 weighted tok/s, about +3.6%.
- Delta versus original valid budget 14/top-k 4 baseline:
  - Baseline: mean 59.43 tok/s, weighted 58.43 tok/s, min 52.33 tok/s.
  - Combined linear fusions: +3.12 mean tok/s, +3.06 weighted tok/s, about +5.2%.
- Prompt-level result: all 10 prompts improved; generated-token count stayed identical, so no hidden prompt-level collapse in this suite.
- Gap after this change:
  - Versus Lucebox Q4_K_M reference: `62.54 / 66.99 = 93.4%`, about 6.6% behind.
  - Versus Lucebox Q8_0 reference: `62.54 / 69.58 = 89.9%`, about 10.1% behind.
- Keep decision: keep the tree QKVZ, QKV prepare, and gated epilogue fusions. They are valid corrected Lucebox-mode progress and materially reduce the remaining gap.

Next validation needed after greedy-cache pass:

- Build validation is already done for the greedy-cache pass:
  - `cargo fmt --check`: passed.
  - `cargo check -p runner --bin supersonic`: passed with existing warnings.
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
- One corrected Lucebox-mode `he_01` profile with budget 14/top-k 4 is done:
  - Artifact: `target/qwen36_lucebox20/tree_linear_fusions_greedy_cache_budget14_top4_profile_he01.json`
  - Result: 179 generated tokens, 3139 ms decode, 17.54 ms/token, 57.01 tok/s.
  - Prior combined-fusions smoke artifact: `target/qwen36_lucebox20/tree_linear_fusions_budget14_top4_profile_he01.json`
  - Prior result: 179 generated tokens, 3148 ms decode, 17.59 ms/token, 56.85 tok/s.
  - Delta: +0.16 tok/s, -9 ms decode, same token count and same mean accepted per round (`8.48`).
  - DFlash breakdown moved from draft 409 ms, verify 2599 ms, rollback 105 ms, other 36 ms to draft 409 ms, verify 2593 ms, rollback 101 ms, other 36 ms.
  - Keep decision before full suite: healthy but tiny. Run the full 10-prompt benchmark before keeping or reverting.
- Run full 10-prompt benchmark:
  - Artifact: `target/qwen36_lucebox20/tree_linear_fusions_greedy_cache_budget14_top4_10x256.json`
  - Result: mean 62.63 tok/s, weighted 61.57 tok/s, min 55.19 tok/s, max 73.10 tok/s, generated 1654 tokens.
  - Prior combined-fusions full artifact: `target/qwen36_lucebox20/tree_linear_fusions_budget14_top4_10x256.json`
  - Prior combined-fusions full result: mean 62.54 tok/s, weighted 61.50 tok/s, min 55.10 tok/s, max 73.05 tok/s, generated 1654 tokens.
  - Delta versus combined fusions: +0.09 mean tok/s, +0.08 weighted tok/s, same generated-token count.
  - Delta versus original valid budget 14/top-k 4 baseline: +3.20 mean tok/s, +3.14 weighted tok/s, about +5.4%.
  - Prompt-level result versus combined fusions: 8 prompts improved, 1 unchanged, 1 regressed slightly (`he_03` by about 0.10 tok/s). No hidden prompt-level collapse.
  - Gap after this change:
    - Versus Lucebox Q4_K_M reference: `62.63 / 66.99 = 93.5%`, about 6.5% behind.
    - Versus Lucebox Q8_0 reference: `62.63 / 69.58 = 90.0%`, about 10.0% behind.
  - Keep decision: keep the greedy-cache pass. The win is tiny, but full-suite result is positive, generated-token count is unchanged, and it removes repeated greedy-path allocation pressure from cached tree verify.

Next implementation direction:

- Remaining profile buckets on `he_01` after current branch are still dominated by tree verify:
  - input norm: about 29 ms per len-15 verify
  - MLP: about 30 ms
  - post norm: about 19.6-19.8 ms
  - linear attention: about 18.2 ms
  - full attention: about 14-16 ms
  - logits/greedy: about 8.0 ms
- Do not spend more time on greedy-cache unless a later profile shows it regressed. The next likely useful area is reducing per-layer norm/MLP overhead or checking whether append-path MLP/norm fusions are missing from tree verify.
- Quick search for a dense Qwen35 HIP fused MLP gate/up path found no existing primitive to reuse. Existing `gate_up` support appears to be Metal-only dense Qwen MLP or Qwen3/Qwen3.6 MoE-specific. Do not repeat that search as a quick branch slice; a dense HIP gate/up fusion would mean adding a new kernel or a new packed weight path.

## Completed Slice: Tree GPU Tap Capture

Reason to try:

- Current direct tree rollback still downloads every tree row for each DFlash tap layer during verify, then uploads only accepted rows back into draft tap history.
- Current profile bucket for taps is about 2.8 ms per len-15 tree verify on `he_01`.
- Append verify already has a GPU tap sink path. Tree verify should be able to mirror that by capturing tap rows into reusable cache-owned GPU storage, then gathering accepted tree rows into `tap_history_gpu` during commit.

Implementation intent:

- Keep existing host tap path as fallback.
- Use existing `SUPERSONIC_DFLASH_DISABLE_GPU_TAP_HISTORY` behavior: when GPU tap history is enabled, tree verify should avoid host tap materialization.
- Add cache-owned reusable tree tap capture storage; do not allocate per round.
- Add a commit-side GPU gather for accepted tree indices.
- If this regresses or complicates rollback correctness, revert only this slice and keep the validated tree linear fusions plus greedy-cache changes.

Validation required after implementation:

1. `cargo fmt --check`: passed.
2. `cargo check -p runner --bin supersonic`: passed with existing warnings.
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
4. One corrected Lucebox-mode `he_01` profile: done.
   - Artifact: `target/qwen36_lucebox20/tree_gpu_taps_budget14_top4_profile_he01.json`
   - Compare taps bucket against `target/qwen36_lucebox20/tree_linear_fusions_greedy_cache_budget14_top4_profile_he01.json`.
   - Result: 179 generated tokens, 3129 ms decode, 17.48 ms/token, 57.21 tok/s.
   - Previous greedy-cache smoke: 179 generated tokens, 3139 ms decode, 17.54 ms/token, 57.01 tok/s.
   - Taps bucket moved from about 2.8 ms to 0.06-0.07 ms per len-15 verify.
   - Total verify stayed about 2593 ms because per-bucket timing shifted into input norm; use total decode/rollback as the main signal.
   - DFlash breakdown moved from draft 409 ms, verify 2593 ms, rollback 101 ms, other 36 ms to draft 408 ms, verify 2593 ms, rollback 92 ms, other 36 ms.
   - Keep decision before full suite: healthy enough for the full 10-prompt run.
5. Full 10-prompt benchmark: done.
   - Artifact: `target/qwen36_lucebox20/tree_gpu_taps_budget14_top4_10x256.json`
   - Result: mean 62.76 tok/s, weighted 61.73 tok/s, min 55.34 tok/s, max 73.31 tok/s, generated 1654 tokens.
   - Previous greedy-cache full artifact: `target/qwen36_lucebox20/tree_linear_fusions_greedy_cache_budget14_top4_10x256.json`
   - Previous greedy-cache result: mean 62.63 tok/s, weighted 61.57 tok/s, min 55.19 tok/s, max 73.10 tok/s, generated 1654 tokens.
   - Delta versus greedy-cache full run: +0.14 mean tok/s, +0.15 weighted tok/s, same generated-token count.
   - Delta versus original valid budget 14/top-k 4 baseline: +3.34 mean tok/s, +3.29 weighted tok/s, about +5.6%.
   - Prompt-level result versus greedy-cache full run: all 10 prompts improved; no hidden prompt-level collapse.
   - Gap after this change:
     - Versus Lucebox Q4_K_M reference: `62.76 / 66.99 = 93.7%`, about 6.3% behind.
     - Versus Lucebox Q8_0 reference: `62.76 / 69.58 = 90.2%`, about 9.8% behind.
   - Keep decision: keep tree GPU tap capture. It removes tree tap host materialization in GPU tap-history mode and gives a small suite-level win.

## Completed Slice: Pre-Rollback FFI Profile

Reason to run:

- Coarse tree verify buckets are async-boundary sensitive after the tap and linear-attention changes.
- Needed current FFI/HAL aggregate timing before choosing the rollback-cache optimization.

Command intent:

- Corrected Lucebox-mode Q8 draft, budget 14, top-k 4, direct rollback.
- `SUPERSONIC_DFLASH_PROFILE_VERIFY=1` plus `SUPERSONIC_DFLASH_PROFILE_FFI=1`.
- One prompt (`he_01`), no warmup, large harness tail.

Artifact:

- `target/qwen36_lucebox20/tree_gpu_taps_budget14_top4_profile_ffi_he01.json`

Result:

- Profiling artifact: `target/qwen36_lucebox20/tree_gpu_taps_budget14_top4_profile_ffi_he01.json`
- Absolute speed is slowed by FFI instrumentation and should not be used as the perf benchmark:
  - 179 generated tokens, about 48.66 tok/s.
- FFI summary:
  - total calls: 28665
  - total timed FFI: 3417.431 ms
  - `qwen.matmul_rhs_transposed_int4`: 8589 calls, 2514.541 ms total, 0.2928 ms mean, 9.256 ms max.
  - `qwen.delta_recurrent_tree_prefill_capture_q8_trace`: 1008 calls, 204.365 ms total.
  - `qwen.full_attention_tree_prefill`: 336 calls, 108.096 ms total.
  - `qwen.rms_norm_rows_plain`: 2961 calls, 101.768 ms total.
  - `qwen.element_add`: 2688 calls, 67.171 ms total.
  - `qwen.matmul_rhs_transposed`: 1008 calls, 66.695 ms total.
  - `qwen.swiglu_mul`: 37.856 ms total.
  - `qwen.dflash_apply_tree_rollback_q8_trace`: 35.067 ms total.
- HAL summary:
  - total HAL calls: 94147
  - total timed HAL: 3352.293 ms
  - allocation calls: 3456
  - allocated bytes: 13396679065
  - synchronization calls: 57351, inflated by FFI profiling syncs; do not overinterpret this count.
  - free total: 42.738 ms
  - alloc total: 21.881 ms
  - D2D copy total: 23.960 ms.
- Inference:
  - INT4 matmul dominates compute and remains the big structural gap.
  - The actionable low-risk next slice was rollback-buffer reuse because allocation traces matched per-round tree rollback captures:
    - full-attention rollback K/V buffers per verified round,
    - linear rollback conv-input and recurrent-trace buffers per linear layer per verified round.
  - Do not repeat this pre-rollback FFI profile; use `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_ffi_he01.json` as the current FFI/HAL evidence.

## Completed Slice: Tree Rollback-Buffer Reuse

Reason to try:

- Current tree verify still allocates rollback capture buffers inside each verified round.
- The FFI/HAL profile shows 3456 allocation calls and about 13.4 GB allocated during an instrumented `he_01` run.
- Expected win is mostly reduced allocator/runtime overhead and cleaner profiling; do not expect this alone to close the full Lucebox gap.

Pre-work already present before final wiring:

- `PrefillTreeVerifyCache` has a `rollback: Option<PrefillTreeRollback>` field initialized to `None`.
- `PrefillTreeVerifyCache::take_rollback` and `PrefillTreeVerifyCache::recycle_rollback` exist.
- `alloc_tree_rollback` allocates grow/reuse candidates for:
  - full-attention `tree_k` and `tree_v` buffers sized to `[num_key_value_heads, tree_len, head_dim]` BF16,
  - linear `conv_input` sized to `[qkv_dim, kern - 1 + tree_len]` BF16,
  - linear `recurrent_trace` using `dflash_q8_trace_bytes(...)` for Q8 trace or the existing non-Q8 trace shape.
- `tree_rollback_matches` checks layer count, device ordinal, tree length, dtype, and buffer byte lengths.
- `prefill_tree_verify_impl` now calls `cache.take_rollback(config, prefix_len)?` when `capture_rollback` is enabled.

Implemented:

- In `prefill_tree_full_attention_layer`, rollback capture now reuses the allocated `Full { tree_k, tree_v }` slot instead of allocating fresh K/V buffers.
- In `prefill_tree_linear_attention_layer`, rollback capture now reuses the allocated `Linear { conv_input, recurrent_trace }` slot instead of allocating fresh conv-input and recurrent-trace buffers.
- Added `delta_recurrent_tree_prefill_capture_with_trace` to share the Q8/BF16/default recurrent trace capture dispatch between cached and uncached trace storage.
- Added `DecodeEngine::commit_prefill_tree_verify_owned` to apply rollback and recycle `result.rollback.take()` back into `PrefillTreeVerifyCache`.
- Routed direct tree rollback in `qwen35_dflash_engine` through the owned commit path while preserving:
  - GPU tap gather into `tap_history_gpu`,
  - host tap fallback flattening when GPU tap history is disabled.

Validation:

1. `cargo fmt --check`: passed.
2. `cargo check -p runner --bin supersonic`: passed with existing warnings.
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
4. Corrected Lucebox-mode one-prompt budget 14/top-k 4 profile:
   - Artifact: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_he01.json`
   - Result: 179 generated tokens, 58.17 tok/s.
   - Previous GPU-tap smoke: 179 generated tokens, 57.21 tok/s.
5. Corrected Lucebox-mode full 10-prompt budget 14/top-k 4:
   - Artifact: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_10x256.json`
   - Result: mean 63.78 tok/s, weighted 62.71 tok/s, min 56.21 tok/s, max 74.46 tok/s, generated 1654 tokens.
   - Previous GPU-tap full artifact: `target/qwen36_lucebox20/tree_gpu_taps_budget14_top4_10x256.json`
   - Previous GPU-tap full result: mean 62.76 tok/s, weighted 61.73 tok/s, min 55.34 tok/s, max 73.31 tok/s, generated 1654 tokens.
   - Delta versus GPU-tap full run: +1.02 mean tok/s, +0.98 weighted tok/s, about +1.6%.
   - Delta versus original valid budget 14/top-k 4 baseline: +4.35 mean tok/s, +4.27 weighted tok/s, about +7.3%.
   - Prompt-level result versus GPU-tap full run: all 10 prompts improved; generated-token count stayed identical, so no hidden prompt-level collapse.
   - Gap after this change:
     - Versus Lucebox Q4_K_M reference: `63.78 / 66.99 = 95.2%`, about 4.8% behind.
     - Versus Lucebox Q8_0 reference: `63.78 / 69.58 = 91.7%`, about 8.3% behind.
6. Corrected Lucebox-mode one-prompt FFI/HAL rerun:
   - Artifact: `target/qwen36_lucebox20/tree_rollback_cache_budget14_top4_profile_ffi_he01.json`
   - Result under instrumentation: 179 generated tokens, 49.55 tok/s; do not use this as the perf benchmark.
   - Allocation calls dropped from 3456 to 896.
   - Allocated bytes dropped from 13396679065 to 990714265.
   - Timed allocation cost dropped from 21.881 ms to 6.744 ms; timed free cost dropped from 42.738 ms to 7.956 ms.

Keep decision: keep rollback-buffer reuse. It is a valid corrected Lucebox-mode win, all prompts improved, token count is unchanged, and allocation pressure dropped materially.

## Rejected Slice: Dense MLP Gate/Up Pair Helper

Reason to try:

- Post-rollback FFI/HAL evidence still shows `qwen.matmul_rhs_transposed_int4` dominating the run.
- Dense MLP gate and up projections are two large same-input INT4 matmuls per layer, so using an existing paired helper looked like a plausible low-risk launch/packing reduction before writing a new HIP kernel.

Implementation tried:

- Temporarily routed `prefill_mlp_layer` gate/up projections through `prefill_ffi::matmul_rhs_transposed_ggml_pair`.
- Added a local bisect env var `SUPERSONIC_DFLASH_DISABLE_MLP_GATE_UP_PAIR` during the experiment.
- This was reverted completely after validation; the env var and paired path should not be present in the current diff.

Validation while the experiment was present:

1. `cargo fmt --check`: passed.
2. `cargo check -p runner --bin supersonic`: passed with existing warnings.
3. `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`: passed with existing warnings.
4. Enabled one-prompt corrected Lucebox-mode profile:
   - Artifact: `target/qwen36_lucebox20/tree_mlp_gate_up_pair_budget14_top4_profile_he01.json`
   - Result: 179 generated tokens, 57.87 tok/s.
   - Acceptance shape changed to 20 rounds, mean accepted 8.90.
   - MLP bucket was about 33.6 ms and input norm about 34 ms.
5. Disabled A/B in the same build:
   - Artifact: `target/qwen36_lucebox20/tree_mlp_gate_up_pair_disabled_budget14_top4_profile_he01.json`
   - Result: 179 generated tokens, 58.00 tok/s.
   - Acceptance shape was 21 rounds, mean accepted 8.48.
   - MLP bucket was about 30.3 ms.

Reject decision:

- Reverted this slice completely.
- Do not retry `matmul_rhs_transposed_ggml_pair` for dense MLP gate/up as a quick optimization. It was slower and perturbed the acceptance/round profile.
- The release binary was rebuilt after reverting and passed:
  - `cargo fmt --check`
  - `cargo check -p runner --bin supersonic`
  - `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
