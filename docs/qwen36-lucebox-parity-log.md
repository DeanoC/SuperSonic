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

- Do not start another performance idea from stale memory. PR #263 is merged into `main`; the winning budget-15 change is now the merged baseline.
- Active next-performance branch: `codex/qwen36-next-roofline`.
- Source on merged `main` has `DDTREE_DEFAULT_BUDGET = 15` and `DDTREE_DEFAULT_TOP_K = 4` in `crates/runner/src/qwen35_dflash_engine.rs`.
- New roofline report: `docs/qwen36-lucebox-next-roofline.md`.
- Fresh post-PR #263 full-suite baseline:
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
