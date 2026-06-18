# Qwen3.6 Lucebox Next Roofline

Last updated: 2026-06-18

This is the next performance checkpoint for SuperSonic Qwen3.6-27B DFlash on RX 7900 XTX / `gfx1100`, after PR #263 was merged into `main`.

## Baseline

Branch and build:

- Branch: `codex/qwen36-next-roofline`
- Baseline commit: `636e61b Merge pull request #263 from DeanoC/codex/lucebox-parity-tree-qkvz`
- Build: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
- Suite: Lucebox HumanEval 10-prompt Qwen3.6-27B serving-mode suite, Q4_K_M target, Q8_0 draft GGUF, no-thinking ChatML prefill, stop on EOS, `n_gen=256`
- DDTree comparison settings: `SUPERSONIC_DFLASH_DDTREE_VERIFY=1`, budget 15, top-k 4, and `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1` for the PR #263 comparison path. Source still gates DDTree and direct rollback by environment variable; missing the direct-rollback env measures the old append-reverify commit path and is not comparable.

Fresh full-suite artifact:

- `target/qwen36_lucebox_next/baseline_10x256.json`
- Mean: `85.52 tok/s`
- Weighted: `84.39 tok/s`
- Min: `74.96 tok/s`
- Max: `99.50 tok/s`
- Generated tokens: `1654`
- Stopped early: `10/10`

This reproduces the PR #263 checkpoint (`85.35 mean / 84.23 weighted`) within normal run noise.

| Prompt | Generated | Decode ms | tok/s |
| --- | ---: | ---: | ---: |
| he_01 | 179 | 2305 | 77.64 |
| he_02 | 232 | 3094 | 74.96 |
| he_03 | 99 | 1274 | 77.70 |
| he_04 | 159 | 1969 | 80.78 |
| he_05 | 174 | 1981 | 87.87 |
| he_06 | 154 | 1639 | 93.98 |
| he_07 | 216 | 2641 | 81.77 |
| he_08 | 114 | 1293 | 88.18 |
| he_09 | 165 | 1658 | 99.50 |
| he_10 | 162 | 1746 | 92.76 |

## Machine Ceilings

Artifact: `target/qwen36_lucebox_next/machine-profile-summary.json`

- GPU: RX 7900 XTX class `gfx1100`, 48 CUs, wave32, 24 GiB VRAM
- Measured VRAM read: `782.09 GB/s`
- Measured VRAM write: `814.35 GB/s`
- Measured VRAM copy: `776.44 GB/s`
- Measured BF16 MMA: `70.34 TFLOP/s`
- Measured F16 MMA: `71.43 TFLOP/s`
- Measured i8 MMA: `71.10 TFLOP/s`
- PCIe H2D/D2H plateau: about `27-29 GB/s`

The practical roofline for the current decode path should be based on the measured ceilings above, not advertised peak compute. Small-M tree kernels are launch/memory/occupancy sensitive before they are pure matrix-throughput workloads.

## Profiling Artifacts

Internal verify profile:

- Artifact: `target/qwen36_lucebox_next/profile_verify_he01_fulltail.json`
- Prompt: `he_01`
- Result: 179 generated tokens, 2314 ms decode, `77.34 tok/s`
- DFlash breakdown: draft `385 ms`, verify `1843 ms`, rollback `86 ms`
- Rounds: 21
- Mean accepted per round: 8.48

Summed normal-profile tree verify buckets over the 21 rounds:

| Bucket | Total ms | Share of verify bucket | Notes |
| --- | ---: | ---: | --- |
| Tree linear attention | 1166.90 | 63.5% | Dominant next target |
| Tree full attention | 584.54 | 31.8% | Second target |
| Logits/greedy | 69.47 | 3.8% | Not first |
| MLP | 8.25 | 0.4% | No longer hot after PR #263 |
| Setup/upload + embed + norms + taps | 7.22 | 0.4% | Not first |

Shape-level FFI profile:

- Artifact: `target/qwen36_lucebox_next/profile_ffi_shapes_he01.json`
- Result under instrumentation: 179 generated tokens, 2996 ms decode, `59.74 tok/s`
- Use this profile for shape/call ranking only; instrumentation inserts syncs and distorts macro bucket times.

Top relevant FFI entries:

| Op | Calls | Total ms | Mean ms | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `matmul_rhs_transposed_int4[b=1 m=16 n=17408 k=5120 g=128 qt=12]` | 2688 | 416.64 | 0.155 | Small-M projection launch train |
| `matmul_mmq_q8_1_q6_k[b=1 m=16 n=5120 k=17408]` | 672 | 350.67 | 0.522 | Small-M projection train; not an isolated MLP-only problem |
| `delta_recurrent_tree_prefill_capture_q8_trace` | 1008 | 212.62 | 0.211 | Recurrent tree linear-attention core |
| `matmul_rhs_transposed_int4[b=1 m=16 n=5120 k=17408 g=128 qt=12]` | 672 | 171.38 | 0.255 | Projection path around linear/MLP work |
| `matmul_rhs_transposed_int4[b=1 m=16 n=5120 k=6144 g=128 qt=13]` | 1008 | 130.93 | 0.130 | Linear-attention projection path |
| `full_attention_tree_prefill` | 336 | 108.48 | 0.323 | Full-attention tree kernel |
| `matmul_rhs_transposed_int4[b=1 m=16 n=248320 k=5120 g=128 qt=14]` | 42 | 96.42 | 2.296 | LM-head, limited total impact |
| `transpose_pad_conv` | 1008 | 33.51 | 0.033 | Adjacent linear-attention data movement |
| `transpose_shd_hsd` | 1008 | 31.70 | 0.031 | Adjacent linear-attention data movement |
| `dflash_extract_recurrent_attn` | 1008 | 31.33 | 0.031 | Adjacent linear-attention extraction |
| `linear_tree_conv_pack` | 1008 | 30.93 | 0.031 | Adjacent linear-attention packing |
| `split_norm_transpose_qkv_bf16` | 1008 | 28.59 | 0.028 | Adjacent linear-attention preparation |
| `compute_beta_g_ba_bf16` | 1008 | 26.90 | 0.027 | Adjacent linear-attention preparation |
| `fill_conv_tail` | 1008 | 26.10 | 0.026 | Adjacent linear-attention preparation |
| `argmax_bf16_rows` | 42 | 11.97 | 0.285 | Greedy return path is already small |

## rocprofv3 Status

`rocprofv3` now works on this box when the profiler and HIP/HSA runtime are kept on the same AMD ROCm 7.1.1 rpath stack. The original failure was a mixed-stack registration mismatch, not sudo, container isolation, or lack of direct GPU access.

Fedora 44 currently exposes `/usr` HIP/HSA runtime packages and `rocprofiler-register`, but no native `rocprofv3` provider. The first install therefore put only the AMD rpath profiler SDK under `/opt/rocm-7.1.1`:

- `rocm-core-rpath7.1.1`
- `rocprofiler-register-rpath7.1.1`
- `rocprofiler-sdk-rocpd-rpath7.1.1`
- `rocprofiler-sdk-roctx-rpath7.1.1`
- `rocprofiler-sdk-rpath7.1.1`

That was not enough by itself: `/opt/rocm-7.1.1/bin/rocprofv3` could launch workloads against Fedora `/usr` HIP/HSA, but trace and counter runs emitted only `*_config.json` files with `profile.threads=0` and `profile.contexts=0`. The same zero-context behavior reproduced with `/usr/bin/hip_add_kernel` and under `sudo`, which ruled out SuperSonic and permissions.

The fix was to install the matching AMD rpath runtime packages alongside the profiler:

- `hip-runtime-amd-rpath7.1.1`
- `hsa-rocr-rpath7.1.1`
- `hsa-amd-aqlprofile-rpath7.1.1`
- `comgr-rpath7.1.1`
- `rocminfo-rpath7.1.1`

Then run profiling with:

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib \
  SUPERSONIC_BACKENDS=hip \
  /opt/rocm-7.1.1/bin/rocprofv3 ...
```

Verified after the runtime fix:

- `/opt/rocm-7.1.1/bin/rocprofv3 --version`: reports rocprofiler-sdk `1.0.0`, ROCm `7.1.1`
- `/opt/rocm-7.1.1/bin/rocprofv3 --list-avail`: works outside the sandbox and wrote `target/qwen36_lucebox_next/rocprofv3-list-avail.txt`
- `ldd /usr/bin/hip_add_kernel` with `LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib` resolves HIP, HSA, COMGR, and `rocprofiler-register` from `/opt/rocm-7.1.1/lib`
- `ldd target/release/supersonic` with the same environment resolves HIP, HSA, COMGR, and `rocprofiler-register` from `/opt/rocm-7.1.1/lib`

Working trace artifacts:

- `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_runtime_kernel_stats.csv`
- `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_runtime_kernel_trace.csv`
- `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_runtime_results.db`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_tiny_trace_kernel_stats.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_tiny_trace_kernel_trace.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_trace_kernel_stats.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_trace_kernel_trace.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_trace_results.db`

Representative full-tail `he_01` trace under profiler:

- 179 generated tokens
- Decode: `2451 ms`, `13.69 ms/token`
- DFlash breakdown: draft `389 ms`, verify `1908 ms`, rollback `154 ms`
- Profiling overhead versus the non-rocprof baseline is roughly 6% on this prompt.

Top kernels from `supersonic_he01_256_trace_kernel_stats.csv`:

| Kernel | Calls | Total ms | Share |
| --- | ---: | ---: | ---: |
| `supersonic_qwen35_matmul_int4_dequant_wmma_kernel` | 470 | 1020.0 | 30.5% |
| `matmul_ggml_dequant_wmma_m16_qtype_kernel<12,true>` | 5544 | 669.0 | 20.0% |
| `matmul_mmq_q8_1_q6_k_kernel` | 704 | 385.9 | 11.6% |
| `delta_recurrent_prefill_kernel<float>` | 48 | 300.1 | 9.0% |
| `delta_recurrent_tree_prefill_capture_q8_trace` | 1008 | 181.9 | 5.5% |
| `matmul_ggml_dequant_wmma_m16_qtype_kernel<14,true>` | 714 | 179.5 | 5.4% |
| `full_attention_tree_prefill_tiled` | 336 | 92.3 | 2.8% |
| `__amd_rocclr_copyBuffer` | 30612 | 55.1 | 1.7% |

Domain stats in the same run:

- Kernel dispatch: 64,249 calls, `3340 ms`, 78.4%
- Memory copy: 1,099 calls, `820 ms`, 19.2%; this is mostly model-load H2D traffic, not decode-loop D2H logits movement
- Memory allocation: 4,738 calls, `101 ms`, 2.4%

Working PMC/counter artifacts:

- `target/qwen36_lucebox_next/rocprof_opt/hip_add_opt_pmc_sqwaves_counter_collection.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_tiny_pmc_tree_counter_collection.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_2_pmc_sqwaves_allkernels_counter_collection.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_32_pmc_sqwaves_allkernels_counter_collection.csv`
- `target/qwen36_lucebox_next/rocprof_opt/supersonic_he01_256_pmc_sqwaves_allkernels_counter_collection.csv`

PMC notes:

- Keep counter passes small. The combined pass `SQ_WAVES Wavefronts OccupancyPercent VALUInsts FETCH_SIZE WRITE_SIZE L2CacheHit` failed with `Request exceeds the capabilities of the hardware to collect` and left a child benchmark process running until it was killed.
- Prefer all-kernel PMC on a representative full-tail run, then filter CSV offline. `--kernel-include-regex` can silently produce no SuperSonic counter CSV if the regex misses profiler-visible aliases.
- A 2-token or 32-token run does not exercise the same tree-kernel set as a full-tail 256-token run, so use `n_gen=256` for tree-kernel PMC evidence.
- `GPUBusy` reported zero in the first targeted tree pass and should not be used as a reliable signal here.
- `SQ_WAVES` is useful. In `supersonic_he01_256_pmc_sqwaves_allkernels_counter_collection.csv`, offline filtering shows:
  - `delta_recurrent_prefill_capture_q8_trace`: 960 calls, 23,592,960 total waves, mean `171.5 us`
  - `full_attention_prefill_tiled`: 336 calls, 178,176 total waves, mean `297.1 us`; 320 of those calls use the tree-sized grid
  - `linear_prefill_conv_pack`: 1008 calls, 7,096,320 total waves, mean `9.3 us`
  - `dflash_extract_recurrent_attn`: 1008 calls, 24,901,632 total waves, mean `6.0 us`

Use the trace files for exact kernel names and timing. Use PMC files for wave/resource counters after mapping aliases back to the traced hot kernels.

Blocked smoke attempts retained for history:

- `target/qwen36_lucebox_next/rocprof/smoke_config_config.json`
- `target/qwen36_lucebox_next/rocprof/smoke_runtime_config.json`
- `target/qwen36_lucebox_next/rocprof/smoke_pmc_gpubusy_config.json`
- `target/qwen36_lucebox_next/rocprof/smoke_hsa_tools_config.json`
- `target/qwen36_lucebox_next/rocprof_debug/hip_add_runtime_config.json`
- `target/qwen36_lucebox_next/rocprof_debug/hip_add_register_hsa_tools_config.json`
- `target/qwen36_lucebox_next/rocprof_debug/hip_add_preload_register_config.json`
- `/tmp/rocprof_root_debug/hip_add_root_runtime_config.json`

Decision: external tracing is now usable enough for the next optimization pass. Keep the isolated `/opt/rocm-7.1.1` lane for this RX 7900 XTX work, and revisit a system-wide ROCm upgrade when the planned RDNA4 card arrives.

## Roofline Classification

Current `he_01` decode time is about `2314 ms` for 179 generated tokens. To reach `95 tok/s` on the same token count, decode time must fall to about `1884 ms`, saving about `430 ms`. To reach `105 tok/s`, decode time must fall to about `1705 ms`, saving about `609 ms`.

The only single bucket large enough to provide the first meaningful gain is tree linear attention:

- A 15% reduction in tree linear attention saves about `175 ms` on `he_01`.
- A 25% reduction saves about `292 ms`.
- A 35% reduction saves about `408 ms`.
- A combined 20% linear-attention reduction plus 15% full-attention reduction saves about `321 ms`.
- A combined 30% linear-attention reduction plus 20% full-attention reduction saves about `467 ms`.

Classification:

| Area | Classification | Evidence | Next action |
| --- | --- | --- | --- |
| Tree linear attention | Launch/memory/occupancy limited until proven otherwise | 63.5% of verify, many small-M `m=16` recurrent/projection/pack/extract kernels | First implementation track |
| Tree full attention | Memory/occupancy limited, with possible prefix-KV reload waste | 31.8% of verify, `full_attention_tree_prefill` visible but bucket also includes nearby projections | Second implementation track |
| MLP | Not currently hot | Normal profile MLP bucket is only 8.25 ms over 21 rounds | Do not spend the next pass here |
| Logits/greedy | Secondary | 69.47 ms normal bucket; GPU argmax already returns row IDs | Revisit only after linear/full attention |
| Host/runtime overhead | Not first | Setup/upload total is about 2 ms; no full-host logits movement in current greedy path | Keep avoiding regressions |
| Rollback | Not first | 86 ms on `he_01`, far below verify | Leave alone unless a commit-path change falls out of tree work |

## Realistic Targets

Use weighted throughput for acceptance because the suite stops early on all prompts.

- Current weighted baseline: `84.39 tok/s`
- Moderate kernel wins: `95-105 tok/s`
  - Requires roughly 11-20% decode-time reduction.
  - Plausible from linear-attention launch/data movement reduction plus a smaller full-attention cleanup.
- Strong tree-kernel wins: `110-125 tok/s`
  - Requires roughly 23-33% decode-time reduction.
  - Requires substantial improvement in linear attention and full attention, not just one helper kernel.
- Beyond `125 tok/s`
  - Treat as algorithmic work: better acceptance, fewer verifier rows, or a verifier redesign.
  - Kernel polishing alone is unlikely to get there from this profile.

## First Implementation Phase

Start with tree verify small-M projection/linear-attention work, not lm-head, rollback, or host setup.

Scope:

1. Inspect the `m=16` recurrent tree linear-attention path around `delta_recurrent_tree_prefill_capture_q8_trace`.
2. Include adjacent kernels in the same pass: `dflash_extract_recurrent_attn`, `linear_tree_conv_pack`, `transpose_pad_conv`, `transpose_shd_hsd`, `split_norm_transpose_qkv_bf16`, `compute_beta_g_ba_bf16`, and `fill_conv_tail`.
3. Look for fusions or data-layout changes that reduce global traffic and launch count without changing Q8 recurrent trace semantics.
4. Keep a rollback env for risky kernel rewrites.
5. Validate deterministic output tails and generated-token counts against `target/qwen36_lucebox_next/baseline_10x256.json`.

The first PR should target the tree verify projection/attention launch train and should not repeat rejected work from the parity log: Q6_K lm-head, dense MLP pair-helper, GGML small-M N64, bounded `m <= 16`, fused GGML SwiGLU, BF16 rollback trace, simple Q8 pre-exp G, or scalar direct BA fusion.

### Phase 1A Result: Direct Q8 Tree Attention Output

Implemented a narrow Q8 tree recurrent fast path that preserves the rollback Q8 trace but writes BF16 attention rows directly from the recurrent kernel. This removes the separate `dflash_extract_recurrent_attn` launch and avoids writing the unused float final-state slab to `linear_delta_out` for the default Q8 tree trace path.

Rollback gate:

- `SUPERSONIC_DFLASH_DISABLE_TREE_DIRECT_ATTENTION=1`

Validation artifacts:

- Build: `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`
- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_delta_parity dflash_tree_delta_q8_direct_attention_matches_extract_path --release -- --ignored --nocapture`
- FFI shape profile: `target/qwen36_lucebox_next/tree_direct_attn_profile_ffi_shapes_he01.json`
- Full direct suite: `target/qwen36_lucebox_next/tree_direct_attn_10x256.json`
- Full env-disabled suite: `target/qwen36_lucebox_next/tree_direct_attn_disabled_10x256.json`

FFI profile evidence:

- New op: `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn`, 1008 calls, `207.58 ms` total on `he_01`.
- Removed from the Q8 tree path: `qwen.dflash_extract_recurrent_attn`.
- `full_attention_tree_prefill` remains at `108.60 ms` under the same instrumented run.
- `linear_tree_conv_pack` remains at `30.60 ms`.

Benchmark result:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_10x256.json` | 85.52 | 84.39 | 74.96 | 1654 | Fresh post-PR #263 reference |
| `tree_direct_attn_disabled_10x256.json` | 85.08 | 83.96 | 74.63 | 1654 | Same code with rollback gate enabled |
| `tree_direct_attn_10x256.json` | 85.69 | 84.56 | 75.13 | 1654 | Kept; small `+0.7%` weighted vs gate |

Decision: keep this change as a small, correctness-tested launch/memory cleanup. It is not enough to move the realistic target band; the next substantial work still needs to attack the remaining tree linear/full attention kernels and projection train.

### Phase 1B Result: Reuse Fused Conv Prep In Tree Verify

Reused the existing HIP `prepare_conv_input_tail` helper in tree verify. The tree path now prepares `[qkv_dim, pad + tree_len]` conv input from the previous conv tail and current tree QKV rows in one launch, with `linear_new_tail` used as a disposable output so acceptance/rollback semantics stay unchanged.

Rollback gate:

- `SUPERSONIC_DFLASH_DISABLE_FUSED_CONV_PREP=1`

Validation artifacts:

- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_conv_input_prepare_matches_transpose_plus_tail --release -- --ignored --nocapture`
- Combined FFI shape profile: `target/qwen36_lucebox_next/tree_convprep_direct_attn_profile_ffi_shapes_he01.json`
- Combined full suite: `target/qwen36_lucebox_next/tree_convprep_direct_attn_10x256.json`
- Combined gated fallback: `target/qwen36_lucebox_next/tree_convprep_direct_attn_disabled_10x256.json`

FFI profile evidence:

- New tree op: `qwen.prepare_conv_input_tail`, 1008 calls, `27.43 ms` total on `he_01`.
- Removed from the optimized tree path: `qwen.transpose_pad_conv`, `qwen.fill_conv_tail`, and `qwen.dflash_extract_recurrent_attn`.
- Remaining nearby costs: `qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn` 1008 calls / `206.65 ms`, `qwen.full_attention_tree_prefill` 336 calls / `108.51 ms`, `qwen.linear_tree_conv_pack` 1008 calls / `30.59 ms`.

Combined benchmark result:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_10x256.json` | 85.52 | 84.39 | 74.96 | 99.50 | 1654 | Fresh post-PR #263 reference |
| `tree_convprep_direct_attn_disabled_10x256.json` | 85.12 | 84.01 | 74.68 | 99.01 | 1654 | Both cleanup gates enabled |
| `tree_convprep_direct_attn_10x256.json` | 86.11 | 84.98 | 75.70 | 100.30 | 1654 | Kept; `+1.2%` weighted vs gated fallback |

All prompt-level generated-token counts and normalized stdout tails match the baseline and gated fallback. The gain is modest but consistent across all 10 prompts; it is a launch-count cleanup, not the larger recurrent/full-attention rewrite needed for the `95-105 tok/s` target band.

### Phase 1C Result: Direct BA Fusion Rejected For Default Path

Tried a fused HIP helper that projects the fused BA weight and computes beta/g directly, preserving the BF16 rounding point from the previous two-step path. The helper is correct on deterministic GPU parity tests, but the first implementation is scalar and much slower than the existing generic BF16 matmul plus beta/g epilogue.

The path is therefore opt-in only:

- `SUPERSONIC_DFLASH_ENABLE_FUSED_BA_DIRECT=1`
- `SUPERSONIC_DFLASH_DISABLE_FUSED_BA_DIRECT=1` is also respected for bisects

Validation artifacts:

- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_direct_ba_beta_g_matches_matmul_path --release -- --ignored --nocapture`
- Opt-in FFI shape profile: `target/qwen36_lucebox_next/tree_direct_ba_profile_ffi_shapes_he01.json`
- Default-off FFI shape profile: `target/qwen36_lucebox_next/tree_direct_ba_defaultoff_profile_ffi_shapes_he01.json`
- Fresh current default suite: `target/qwen36_lucebox_next/tree_phase1_default_10x256.json`

FFI profile evidence:

| Path | Op | Calls | Total ms | Mean ms |
| --- | --- | ---: | ---: | ---: |
| Opt-in fused BA | `qwen.project_ba_compute_beta_g_bf16` | 960 | 271.90 | 0.283 |
| Default old path | `qwen.matmul_rhs_transposed[b=1 m=16 n=96 k=5120 dtype=BF16]` | 1008 | 67.14 | 0.067 |
| Default old path | `qwen.compute_beta_g_ba_bf16` | 1008 | 26.67 | 0.027 |

Fresh current default benchmark:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tree_convprep_direct_attn_10x256.json` | 86.11 | 84.98 | 75.70 | 100.30 | 1654 | Previous kept Phase 1A/1B checkpoint |
| `tree_phase1_default_10x256.json` | 86.05 | 84.93 | 75.53 | 100.10 | 1654 | Current default after leaving direct BA opt-in |

All prompt-level generated-token counts and normalized stdout tails match the baseline and Phase 1A/1B checkpoint. Do not revisit this direct BA idea unless the projection side is rewritten around the existing MFMA-class matmul path or an equivalent tiled implementation.

### Phase 1D Result: Paired K/V Transpose For Tree Full Attention

The FFI profile showed `transpose_shd_hsd` at 1008 calls in the Phase 1 default path. Mapping the call sites showed this was exactly `3 x full_attention_tree_prefill` over the 336 tree full-attention calls: K transpose, V transpose, and Q transpose. Phase 1D fuses only the equal-shape K/V transposes into one HIP launch and leaves Q unchanged.

Rollback gate:

- `SUPERSONIC_DFLASH_DISABLE_TREE_FULL_KV_TRANSPOSE=1`

Validation artifacts:

- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_full_kv_pair_transpose_matches_separate_calls --release -- --ignored --nocapture`
- FFI shape profile: `target/qwen36_lucebox_next/tree_full_kv_pair_profile_ffi_shapes_he01.json`
- Full suite: `target/qwen36_lucebox_next/tree_full_kv_pair_10x256.json`
- Rollback-gated full suite: `target/qwen36_lucebox_next/tree_full_kv_pair_disabled_10x256.json`

FFI profile evidence:

| Artifact | Op | Calls | Total ms | Mean ms |
| --- | --- | ---: | ---: | ---: |
| Phase 1 default | `qwen.transpose_shd_hsd` | 1008 | 31.41 | 0.031 |
| K/V pair path | `qwen.transpose_shd_hsd` | 336 | 14.09 | 0.042 |
| K/V pair path | `qwen.transpose_shd_hsd_pair` | 336 | 8.94 | 0.027 |

Full-suite result:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tree_phase1_default_10x256.json` | 86.05 | 84.93 | 75.53 | 100.10 | 1654 | Phase 1A/1B with direct BA default-off |
| `tree_full_kv_pair_disabled_10x256.json` | 85.97 | 84.85 | 75.41 | 100.20 | 1654 | Same code with K/V pair rollback gate |
| `tree_full_kv_pair_10x256.json` | 86.12 | 84.97 | 75.53 | 100.50 | 1654 | Kept; tiny macro win, useful launch cleanup |

All prompt-level generated-token counts and normalized stdout tails match the baseline, Phase 1 default, and rollback-gated K/V run. This is intentionally a small full-attention launch cleanup; it does not change the roofline target or the next major optimization track.

### Phase 1E Result: Strided Prefix K/V For Tree Full Attention Default-Off

Tried a strided-prefix variant of the tree full-attention kernel so the verifier can read prefix K/V rows directly from the layer KV cache, using `kv_capacity` as the prefix stride. This avoids the fallback path's per-round contiguous prefix K/V allocation and per-head D2D copy, and keeps exact parity with the current contiguous-prefix attention output.

The path is kept for profiling and future kernel work, but it is not default-on:

- Opt in with `SUPERSONIC_DFLASH_ENABLE_TREE_FULL_PREFIX_STRIDED=1`
- Force off with `SUPERSONIC_DFLASH_DISABLE_TREE_FULL_PREFIX_STRIDED=1`

Validation artifacts:

- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_full_attention_strided_prefix_matches_contiguous_prefix --release -- --ignored --nocapture`
- Opt-in FFI shape profile: `target/qwen36_lucebox_next/tree_full_prefix_strided_profile_ffi_shapes_he01.json`
- Opt-in full suite: `target/qwen36_lucebox_next/tree_full_prefix_strided_10x256.json`
- Disabled/current-default full suite: `target/qwen36_lucebox_next/tree_full_prefix_strided_disabled_10x256.json`
- Rebuilt final default full suite: `target/qwen36_lucebox_next/tree_phase1e_final_default_10x256.json`

FFI/HAL profile evidence:

| Artifact | Op / HAL metric | Value |
| --- | --- | ---: |
| K/V pair contiguous path | `qwen.full_attention_tree_prefill` | 336 calls / 108.36 ms |
| Strided prefix path | `qwen.full_attention_tree_prefill_strided` | 336 calls / 121.96 ms |
| K/V pair contiguous path | HAL allocation calls | 897 |
| Strided prefix path | HAL allocation calls | 225 |
| K/V pair contiguous path | HAL D2D bytes | 1.21 GB |
| Strided prefix path | HAL D2D bytes | 0.89 GB |

Full-suite result:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tree_full_kv_pair_10x256.json` | 86.12 | 84.97 | 75.53 | 100.50 | 1654 | Previous kept Phase 1D checkpoint |
| `tree_full_prefix_strided_10x256.json` | 86.21 | 85.07 | 75.59 | 100.30 | 1654 | Correct, fewer allocations/copies, but slower attention kernel |
| `tree_full_prefix_strided_disabled_10x256.json` | 86.27 | 85.14 | 75.70 | 100.50 | 1654 | Default decision; fractionally faster macro result |
| `tree_phase1e_final_default_10x256.json` | 86.30 | 85.18 | 75.70 | 100.91 | 1654 | Rebuilt final source with strided prefix opt-in only |

All prompt-level generated-token counts and normalized stdout tails match the baseline and Phase 1D checkpoint. The allocation/copy reduction is real, but the strided kernel loses the contiguous-prefix locality benefit and is not a measured throughput win. Do not re-promote this path unless the kernel is retiled around strided prefix reads or a future memory/capacity profile makes the allocation pressure dominant.

### Phase 1F Result: Indexed Tree Conv Source Map

The tree linear-attention conv helper previously walked `parent_ids` inside every `(tree row, channel)` element to select each convolution source column. Phase 1F precomputes the tiny `[tree_len, kernel_size]` source-column map once in `PrefillTreeVerifyCache`, uploads it with the other tree metadata, and uses an indexed HIP helper for `linear_tree_conv_pack`.

Rollback gate:

- `SUPERSONIC_DFLASH_DISABLE_TREE_CONV_SOURCE_MAP=1`

Validation artifacts:

- Parity test: `env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib HIP_ARCH=gfx1100 cargo test -p runner --test dflash_tree_conv_pack_parity dflash_tree_conv_pack_indexed_matches_parent_walk_fixture --release -- --ignored --nocapture`
- FFI shape profile: `target/qwen36_lucebox_next/tree_conv_source_map_profile_ffi_shapes_he01.json`
- Full suite: `target/qwen36_lucebox_next/tree_conv_source_map_10x256.json`
- Rollback-gated full suite: `target/qwen36_lucebox_next/tree_conv_source_map_disabled_10x256.json`

FFI/HAL profile evidence:

| Path | Op / HAL metric | Value |
| --- | --- | ---: |
| Parent-walk conv | `qwen.linear_tree_conv_pack` | 1008 calls / 30.58 ms |
| Indexed source-map conv | `qwen.linear_tree_conv_pack_indexed` | 1008 calls / 30.06 ms |
| Parent-walk conv | HAL H2D calls / bytes | 105 / 10 KB |
| Indexed source-map conv | HAL H2D calls / bytes | 126 / 15 KB |

Full-suite result:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Max tok/s | Generated | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tree_phase1e_final_default_10x256.json` | 86.30 | 85.18 | 75.70 | 100.91 | 1654 | Previous default checkpoint |
| `tree_conv_source_map_disabled_10x256.json` | 86.34 | 85.18 | 75.76 | 100.81 | 1654 | Same code with source-map rollback gate |
| `tree_conv_source_map_10x256.json` | 86.42 | 85.30 | 75.93 | 100.81 | 1654 | Kept; tiny branch-work cleanup |

All prompt-level generated-token counts and normalized stdout tails match the baseline. This is a valid cleanup, but the measured win is about `+0.1%` weighted and does not change the realistic target band. The next meaningful speedup still needs to attack the projection train or the recurrent/full-attention kernels themselves.

Invalid smoke runs, do not use as evidence:

- `tree_direct_attn_profile_he01.json` and `tree_direct_attn_disabled_profile_he01.json` used the script default `q4km-gptq`, generated 175 tokens, and are not comparable with the Q4_K_M baseline.
- `tree_direct_attn_ddtree_he01.json`, `tree_direct_attn_disabled_ddtree_he01.json`, `tree_direct_attn_ddtree_profile_he01.json`, and `tree_direct_attn_disabled_ddtree_profile_he01.json` missed `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1`, used `commit=append-reverify`, and measured the old slow commit path instead of the PR #263 comparison path.

## Follow-Up Phase

If the linear-attention pass materially reduces verify cost, run a limited DDTree sweep only around:

- Budgets: `{14, 15, 16}`
- Top-k: `{4, 8}`
- Chain-seed: current default first, off only if verify cost changes enough to justify checking

Promote a new default only if weighted throughput improves and weak prompts do not collapse behind the mean.

References:

- AMD HIP performance optimization: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html
- AMD HIP performance guidelines: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- rocprofv3 usage: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- DeFT: Decoding with Flash Tree-attention: https://arxiv.org/abs/2404.00242
