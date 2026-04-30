# Certified KV Paper-to-Code Audit Map

This document maps the certified quantised attention paper in
`docs/papers/Certified_Quantised_Attention.tex` to the PyTorch oracle and the
Rust/CUDA execution path. It is intended as a maintenance checklist: if one row
changes, update the corresponding oracle, Rust, CUDA, and tests together.

## Scope

- Paper source of truth: `docs/papers/Certified_Quantised_Attention.tex`.
- Executable oracle: `oracle/certified_kv_llama31.py`.
- Runtime path: `crates/runner/src/decode_engine.rs`, `crates/qwen35/src/state.rs`,
  `crates/kernel-ffi/src/certified_kv.rs`, and
  `kernels/certified_kv_bridge_cuda.cu`.
- Current implementation target: LLaMA 3.1 certified KV decode path.

## Non-Negotiable Invariants

- Tier 1 VRAM stores only compressed complete blocks plus metadata:
  `key_i8`, `key_scale`, `key_zero`, `value_i4`, `value_scale`,
  `value_zero`, `value_error`, and `value_norm`.
- Tier 2 stores the full original K/V in host memory:
  `certified_kv_host_k` and `certified_kv_host_v`.
- A trailing incomplete block is not quantised. It remains BF16 in the tail
  scratch buffers `certified_kv_tail_k` and `certified_kv_tail_v`.
- GPU memory must not contain a full BF16 key/value corpus except transient
  scratch populated by the fallback system. The persistent GPU corpus is the
  compressed Tier 1 representation.
- Rung 3 and Rung 4 fallback use Tier 2 originals copied into scratch/prefix
  buffers, not dequantised Tier 1 data.
- The oracle remains the readable executable specification. Runtime changes
  that affect semantics must first be representable in the oracle.

## Configuration Defaults

| Concept | Paper default | Oracle | Rust/CLI |
| --- | --- | --- | --- |
| Block size | `B = 16` | `CertifiedKvConfig.block_size`, `oracle/certified_kv_llama31.py:21` | `CertifiedKvConfig`, `crates/runner/src/certified_kv.rs:7`; `CertifiedKvDecodeParams`, `crates/runner/src/decode_engine.rs:1122` |
| Value group size | `g = 16` | `CertifiedKvConfig.value_group_size`, `oracle/certified_kv_llama31.py:23` | `--certified-kv-value-group-size`, `crates/runner/src/main.rs`; runtime params at `crates/runner/src/decode_engine.rs:1122` |
| Coverage threshold | `tau_cov = 0.995` | `CertifiedKvConfig.tau_cov`, `oracle/certified_kv_llama31.py:24` | `certified_kv_tau_cov` in `crates/runner/src/main.rs`; default copied into `CertifiedKvDecodeParams::trace_default`, `crates/runner/src/decode_engine.rs:1139` |
| `K_min`, `K_max` | `2`, `128` | `CertifiedKvConfig.k_min/k_max`, `oracle/certified_kv_llama31.py:25` | CLI fields in `crates/runner/src/main.rs`; runtime params in `crates/runner/src/decode_engine.rs:1122` |
| Value tolerance | `v_tol = 0.05` | `CertifiedKvConfig.v_tol`, `oracle/certified_kv_llama31.py:27` | `--certified-kv-v-tol`, `crates/runner/src/main.rs:450`; runtime params in `crates/runner/src/decode_engine.rs:1122` |
| Ranking depth | `r = 1` | `CertifiedKvConfig.ranking_r`, `oracle/certified_kv_llama31.py:28` | `--certified-kv-ranking-r`, `crates/runner/src/main.rs:454` |
| Rung 1 expansion | threshold `0.005`, multiplier `2.0` | `CertifiedKvConfig.rung1_threshold/rung1_multiplier`, `oracle/certified_kv_llama31.py:29` | `--certified-kv-rung1-*`, `crates/runner/src/main.rs:458` |
| Tail guard factor | `exp(3 Delta)` | `CertifiedKvConfig.delta_guard_factor`, `oracle/certified_kv_llama31.py:32` | `--certified-kv-delta-guard-factor`, `crates/runner/src/main.rs:470` |
| Exploration and guard epsilon | `0.01`, `0.0001` | `score_exploration_rate`, `eps_guard`, `oracle/certified_kv_llama31.py:31` | CLI fields in `crates/runner/src/main.rs:474` and `crates/runner/src/main.rs:482` |
| Promoted value cache | implementation scratch cache | N/A; oracle models semantics, not cache policy | `--certified-kv-value-cache-blocks`, default `128`; runtime params in `crates/runner/src/decode_engine.rs` |

## Paper-to-Code Map

| Paper part | Oracle path | Rust runtime path | FFI/CUDA path | Audit expectation |
| --- | --- | --- | --- | --- |
| Tiered cache object | `TieredKvCache`, `oracle/certified_kv_llama31.py:38`; `build_tiered_kv_cache`, `oracle/certified_kv_llama31.py:165` | State fields in `crates/qwen35/src/state.rs:33`; allocation/update in `crates/runner/src/decode_engine.rs:4392`, `:4422`, `:4744`, `:4872` | Quantisation kernels in `kernels/certified_kv_bridge_cuda.cu:72` and `:205` | Complete blocks go to compressed Tier 1; full originals go to Tier 2 host; incomplete tail stays BF16. |
| Per-channel asymmetric INT8 keys | `quantize_keys_int8_asymmetric`, `oracle/certified_kv_llama31.py:94`; `dequantize_keys`, `oracle/certified_kv_llama31.py:121` | State fields `certified_kv_key_i8/key_scale/key_zero`, `crates/qwen35/src/state.rs:33`; block quantisation call path around `crates/runner/src/decode_engine.rs:4446` | FFI wrapper `quantize_bf16_keys_range`, `crates/kernel-ffi/src/certified_kv.rs:630`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:72`, zero write at `:122` | Scales and zeros are per block and per channel. Zero points are required; symmetric INT8 is not paper-exact. |
| Per-group INT4 values | `quantize_values_int4`, `oracle/certified_kv_llama31.py:130`; `dequantize_values_int4`, `oracle/certified_kv_llama31.py:152` | State fields `certified_kv_value_i4/value_scale/value_zero`, `crates/qwen35/src/state.rs:37`; block quantisation/update around `crates/runner/src/decode_engine.rs:4744` and `:5254` | FFI wrapper `quantize_bf16_values_range`, `crates/kernel-ffi/src/certified_kv.rs:750`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:205` | Packed INT4 is the normal value path. BF16 values are bring-up/debug only unless a fallback promotes from Tier 2. |
| Value error annotation and norms | `value_errors` and `value_norms` in `TieredKvCache`, `oracle/certified_kv_llama31.py:43` | State fields `certified_kv_value_error/value_norm`, `crates/qwen35/src/state.rs:40`; runtime D2H inputs around `crates/runner/src/decode_engine.rs:5415` and `:5424` | CUDA writes value error/norm in `kernels/certified_kv_bridge_cuda.cu:286` | `eta_b` and value norms are computed at quantisation time from original BF16 values. |
| Phase 1 INT8 block scoring | `certified_attention_step`, `oracle/certified_kv_llama31.py:514`; log masses built before selection | Runtime scoring call `crates/runner/src/decode_engine.rs:5381` | FFI `score_blocks_int8`, `crates/kernel-ffi/src/certified_kv.rs:902`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:386` and extern `:1335` | Scoring uses INT8 keys, key scales, and key zeros. It produces per-head/per-block log mass and max score. |
| Score error bound `Delta` | `score_delta_bound`, `oracle/certified_kv_llama31.py:283` | Device selector uses per-block key scale norms cached in `certified_kv_device_meta_key_scale_norm_blocks`, `crates/qwen35/src/state.rs`; fallback/host validation helpers remain in `crates/runner/src/decode_engine.rs` | FFI `key_scale_norms` and `select_blocks_device`, `crates/kernel-ffi/src/certified_kv.rs`; CUDA scale-norm and selector kernels in `kernels/certified_kv_bridge_cuda.cu` | The fast path computes the paper's Cauchy-Schwarz score bound on GPU from query norm and per-block key-scale norms. Host-side bound code is audit/debug fallback only. |
| Adaptive top-K and Rung 1 expansion | `adaptive_topk_mask`, `oracle/certified_kv_llama31.py:300` | GPU selector dispatch and scratch buffers in `crates/runner/src/decode_engine.rs`; host selector remains only for debug/fallback modes | CUDA `certified_kv_select_blocks_kernel` writes `promote_index`, `value_promote_index`, selected counts, fallback flags, `Delta`, `E_key`, and tail-bound telemetry | If `K_max` prevents the target tail certificate and uncertified tail is disallowed, the affected query head escalates to Rung 3 per-head dense fallback instead of silently continuing or forcing a whole-layer fallback. Normal selection must not require per-step score/metadata D2H. |
| Key error certificate `E_key` | `certified_attention_step` telemetry, `oracle/certified_kv_llama31.py:514` | Device selector path keeps `E_key` and tail-bound scalars on GPU; host telemetry reads them only when `SUPERSONIC_CERTIFIED_HOST_TELEMETRY` is set | CUDA selector writes `e_key_by_head`, `delta_tail_by_head`, `vmax_by_head`, and `true_tail_by_head`; ranking fallback flags are consumed on GPU | Uses achieved tail mass after clamp/expansion and `exp(guard * Delta)`. Default performance path is GPU-resident; audit telemetry may synchronize by opt-in. |
| Value error certificate `E_val` | `value_error_bound`, `oracle/certified_kv_llama31.py:366`; final masses returned by `_mixed_attention`, `oracle/certified_kv_llama31.py:477` | Runtime reduces final Phase-2 token probabilities to block masses after attention in `crates/runner/src/decode_engine.rs`; telemetry in `crates/runner/src/llama31_engine.rs:1425` | CUDA probability-to-block-mass reduction `supersonic_llama31_certified_kv_block_masses_from_probs`; CUDA value certificate/promotion evaluator `supersonic_llama31_certified_kv_value_promotions_from_block_masses` | `E_val` must use final masses from the actual key precision path, not Phase-1 INT8 mass estimates. Promoted value blocks are excluded from the INT4 value error sum. |
| Rung 2 value promotion | `certified_attention_step` builds `value_promote_mask`, oracle reruns with promoted values when required | Runtime seeds value promotion from Phase-1 masses, then checks final Phase-2 masses and reruns attention if any block exceeds `rho_b * eta_b > v_tol` | Mixed/all-promoted kernels consume INT4 Tier 1 for the fast path; CUDA promotion evaluator writes `value_promote_index`; promoted values are gathered from Tier 2 into BF16 scratch | Any promoted value path must source BF16 values from `certified_kv_host_v`, never from dequantised INT4. Final `E_val` is computed after any required rerun. |
| Promoted value cache | Not modeled; cache is an implementation optimization | Per-layer `certified_kv_promoted_value_cache` plus tags/LRU in `crates/qwen35/src/state.rs`; initial and final value-promotion remap/gather logic in `crates/runner/src/decode_engine.rs`; telemetry in `DecodeStageTimings` and `llama31_engine.rs` | Existing Tier-2 gather kernels fill cache slots; attention kernels consume cache slots through `value_promote_index` | Cache entries are BF16 value blocks fetched from Tier 2 into scratch VRAM. They are not Tier 1 and are bounded by `--certified-kv-value-cache-blocks`; disabling the cache must preserve exact output and only increase value H2D. A cache is used only when the current step's promoted set fits per KV head, so an in-step LRU eviction cannot invalidate a live promotion index. |
| Ranking consistency order check | `ranking_consistency_fallback_heads`, `oracle/certified_kv_llama31.py:413` | Device ranking path in `crates/runner/src/decode_engine.rs`; host comparison helpers remain for non-device/debug modes | CUDA `selected_fp16_log_masses` plus `ranking_flags_device`; per-head flagged fallback kernel `dense_flagged_heads_out_bf16` consumes device flags directly | If promoted-set rankings disagree, affected heads use dense flagged-head fallback from Tier 2. Default path must not copy fallback flags to the host. |
| Ranking boundary verification | `ranking_consistency_fallback_heads`, `oracle/certified_kv_llama31.py:413` | Device ranking path in `crates/runner/src/decode_engine.rs`; host boundary helper remains for fallback/debug modes | CUDA `certified_kv_ranking_flags_kernel` checks promoted ranking and tail-boundary intervals on device | Tail block upper-bound log mass must not enter the top-`r` promoted FP16 ranking. Boundary failures set per-head fallback flags on GPU. |
| Score consistency and exploration | Oracle scoring/fallback telemetry in `certified_attention_step`, `oracle/certified_kv_llama31.py:514` | GPU score-consistency dispatch in `crates/runner/src/decode_engine.rs`; deterministic exploration adds extra promoted blocks before gather | FFI `score_consistency`, `crates/kernel-ffi/src/certified_kv.rs`; CUDA kernel `certified_kv_score_consistency_kernel`, `kernels/certified_kv_bridge_cuda.cu` | Non-all-promoted selected blocks compare INT8 and Tier-2 BF16 scores against the declared bound; violation flags are cleared on every call and any violating query head escalates to Rung 3 per-head dense fallback. |
| Phase 2 mixed attention | `certified_attention_step`, `oracle/certified_kv_llama31.py:514` | Runtime mixed-attention dispatch at `crates/runner/src/decode_engine.rs:6035` | FFI `attend_mixed_key_int4_with_bf16_tail_strided`, `crates/kernel-ffi/src/certified_kv.rs:1651`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:706` and extern `:1842` | Selected/promoted keys use BF16 scratch from Tier 2; tail uses BF16 tail scratch; unpromoted complete blocks use INT8 keys and INT4 values. |
| Rung 3 per-head fallback | Oracle returns fallback heads from ranking check | Device fallback flags in `crates/runner/src/decode_engine.rs`; layer-local prefix scratch cache incrementally pages Tier 2 K/V into VRAM | FFI `dense_flagged_heads_out_bf16` for GPU flags and `dense_selected_heads_out_bf16` for host/debug fallback; CUDA externs in `kernels/certified_kv_bridge_cuda.cu` | Only affected heads are recomputed densely, using Tier 2 originals plus BF16 tail. Prefix scratch is fallback scratch, not Tier 1 BF16 KV. |
| Rung 4 full dense fallback | Oracle dense-equivalent fallback contract in `certified_attention_step` | `force_dense_layer_fallback`, `crates/runner/src/decode_engine.rs:5454`; dense path begins at `crates/runner/src/decode_engine.rs:5737` | Dense fallback uses BF16 Tier 2/prefix buffers and BF16 tail | This is the unconditional terminal state. It must not depend on compressed data for correctness. |
| PG-19 target-NLL QA endpoint | Oracle benchmark driver computes perplexity from target logits | Teacher-forced LLaMA 3.1 path accumulates decode target NLL in `DecodeEngine::component_decode_step_4b_accumulate_target_nll`; final scalar is read once | CUDA `supersonic_qwen35_cuda_accumulate_target_nll_bf16`, `kernels/prefill_helpers_bridge_cuda.cu` | Full-vocab logits must not be copied to host per decode token. This affects benchmark scoring only, not certified attention semantics. |
| Memory/caching telemetry | Oracle telemetry dictionary from `certified_attention_step` | `CertifiedKvShadowStats`, `crates/runner/src/decode_engine.rs`; `certified_kv_memory_stats`; decode telemetry in `crates/runner/src/llama31_engine.rs` | Device telemetry stays on GPU unless `SUPERSONIC_CERTIFIED_HOST_TELEMETRY=1` | Telemetry must distinguish compressed VRAM, Tier 2 host bytes, tail scratch, ranking prefix scratch, H2D bytes, and cache hits. Audit telemetry may synchronize; the default fast path should not. |

## Test and Benchmark Coverage

| Coverage target | File/command | What it protects |
| --- | --- | --- |
| Oracle semantics | `python3 -m unittest tests/test_certified_kv_llama31_oracle.py` | INT8 key quantisation, INT4 value packing, bounds, fallback decisions, and arXiv-v1 small-context QA fixtures. |
| CUDA wrapper/kernels | `cargo test -p kernel-ffi certified_kv --lib -- --nocapture` | Key zero points, value norms, INT4 packing, score blocks, mixed attention, and dense selected-head fallback. |
| Rust integration compile | `cargo check -p kernel-ffi -p qwen35 -p runner` | Cross-crate API shape, CLI/config propagation, state fields, and FFI signatures. |
| Formatting | `cargo fmt` | Rust source formatting. |

## Audit Checklist for Future Changes

1. If key quantisation changes, update `quantize_keys_int8_asymmetric`,
   `quantize_bf16_keys_range`, the CUDA key quantisation kernel, and the
   key-zero/kernel tests together.
2. If value quantisation changes, update `quantize_values_int4`,
   `quantize_bf16_values_range`, the CUDA value kernel, value error/norm
   tests, and the storage accounting.
3. If any bound changes, update the oracle bound function first, then the Rust
   helper, then telemetry names/meaning if needed.
4. If selection or fallback changes, verify both ranking order and boundary
   checks still trigger dense selected-head or dense layer fallback from Tier 2.
5. If memory layout changes, re-run `certified_kv_memory_stats` telemetry and
   verify the GPU full-corpus BF16 invariant still holds.
6. If performance code introduces caches, confirm cached data is scratch/prefix
   fallback data and not a persistent full BF16 mirror unless explicitly
   reported as such.

## Known Implementation Choices

- The runtime's default certified path keeps key selection, ranking flags,
  value-promotion checks, and PG-19 target-NLL accumulation on GPU. Host reads
  for selector/fallback/certificate scalars are opt-in through
  `SUPERSONIC_CERTIFIED_HOST_TELEMETRY=1` or explicit trace/debug paths.
- When the key budget covers every complete block, the runtime skips Phase-1
  INT8 scoring and uses `init_all_promoted_indices` to initialize key/value
  promotion indices on GPU. This makes `E_key = 0` for complete blocks while
  avoiding per-token CPU index uploads. This shortcut is only valid when
  `K_max >= num_blocks`; Rung 1 expansion is conditional and must not be used
  to skip scoring ahead of time.
- The runtime computes value certificates from final Phase-2 attention masses.
  The CUDA attention kernels leave normalized token probabilities in
  `score_scratch`; `block_masses_from_token_probs` reduces those probabilities
  to per-head/per-block masses. A CUDA evaluator consumes those masses plus
  `eta_b`, writes the final value-promotion index, and computes per-head
  `E_val`. If final masses require value promotions, the runtime gathers the
  promoted BF16 values from Tier 2 and reruns attention before reporting
  `E_val`.
- `bf16_values` is retained for bring-up/debug modes. Paper-exact certified
  execution uses INT4 values in Tier 1 and BF16 values only when fetched from
  Tier 2 by fallback/promotion.
- The ranking prefix cache is a scratch optimization for fallback. It is tracked
  separately from Tier 1 compressed VRAM and Tier 2 host memory.
- The promoted value cache is also scratch VRAM populated only from Tier 2. It
  caches selected BF16 value blocks across decode steps for both the initial
  Phase-1-seeded value promotions and the final post-mass Rung-2 value
  promotions, while the normal value corpus remains INT4 in Tier 1.
- CUDA bridge helpers should not call `cudaDeviceSynchronize` in the steady
  path. Default-stream ordering is sufficient between kernels; explicit
  synchronization is reserved for timing, host readback, or temporary-device
  allocations that must not be freed before queued work completes.
