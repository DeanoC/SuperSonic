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

## Paper-to-Code Map

| Paper part | Oracle path | Rust runtime path | FFI/CUDA path | Audit expectation |
| --- | --- | --- | --- | --- |
| Tiered cache object | `TieredKvCache`, `oracle/certified_kv_llama31.py:38`; `build_tiered_kv_cache`, `oracle/certified_kv_llama31.py:165` | State fields in `crates/qwen35/src/state.rs:33`; allocation/update in `crates/runner/src/decode_engine.rs:4392`, `:4422`, `:4744`, `:4872` | Quantisation kernels in `kernels/certified_kv_bridge_cuda.cu:72` and `:205` | Complete blocks go to compressed Tier 1; full originals go to Tier 2 host; incomplete tail stays BF16. |
| Per-channel asymmetric INT8 keys | `quantize_keys_int8_asymmetric`, `oracle/certified_kv_llama31.py:94`; `dequantize_keys`, `oracle/certified_kv_llama31.py:121` | State fields `certified_kv_key_i8/key_scale/key_zero`, `crates/qwen35/src/state.rs:33`; block quantisation call path around `crates/runner/src/decode_engine.rs:4446` | FFI wrapper `quantize_bf16_keys_range`, `crates/kernel-ffi/src/certified_kv.rs:630`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:72`, zero write at `:122` | Scales and zeros are per block and per channel. Zero points are required; symmetric INT8 is not paper-exact. |
| Per-group INT4 values | `quantize_values_int4`, `oracle/certified_kv_llama31.py:130`; `dequantize_values_int4`, `oracle/certified_kv_llama31.py:152` | State fields `certified_kv_value_i4/value_scale/value_zero`, `crates/qwen35/src/state.rs:37`; block quantisation/update around `crates/runner/src/decode_engine.rs:4744` and `:5254` | FFI wrapper `quantize_bf16_values_range`, `crates/kernel-ffi/src/certified_kv.rs:750`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:205` | Packed INT4 is the normal value path. BF16 values are bring-up/debug only unless a fallback promotes from Tier 2. |
| Value error annotation and norms | `value_errors` and `value_norms` in `TieredKvCache`, `oracle/certified_kv_llama31.py:43` | State fields `certified_kv_value_error/value_norm`, `crates/qwen35/src/state.rs:40`; runtime D2H inputs around `crates/runner/src/decode_engine.rs:5415` and `:5424` | CUDA writes value error/norm in `kernels/certified_kv_bridge_cuda.cu:286` | `eta_b` and value norms are computed at quantisation time from original BF16 values. |
| Phase 1 INT8 block scoring | `certified_attention_step`, `oracle/certified_kv_llama31.py:514`; log masses built before selection | Runtime scoring call `crates/runner/src/decode_engine.rs:5381` | FFI `score_blocks_int8`, `crates/kernel-ffi/src/certified_kv.rs:902`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:386` and extern `:1335` | Scoring uses INT8 keys, key scales, and key zeros. It produces per-head/per-block log mass and max score. |
| Score error bound `Delta` | `score_delta_bound`, `oracle/certified_kv_llama31.py:283` | `certified_kv_score_delta_blocks`, `crates/runner/src/decode_engine.rs:59`; runtime use at `crates/runner/src/decode_engine.rs:5463` | Uses CPU-side copied key scales during certification logic | Bound must use per-channel scales and query values for the selected GQA group. |
| Adaptive top-K and Rung 1 expansion | `adaptive_topk_mask`, `oracle/certified_kv_llama31.py:300` | `certified_kv_select_block_indices_from_scores`, `crates/runner/src/decode_engine.rs:175`; runtime use at `crates/runner/src/decode_engine.rs:5473`; dense fallback on uncertified tail at `:5491` | Selection is host-side; selected heads/blocks drive the mixed attention FFI call | If `K_max` prevents the target tail certificate and uncertified tail is disallowed, the runtime must escalate instead of silently continuing. |
| Key error certificate `E_key` | `certified_attention_step` telemetry, `oracle/certified_kv_llama31.py:514` | `certified_kv_key_error_bound`, `crates/runner/src/decode_engine.rs:99`; runtime computation at `crates/runner/src/decode_engine.rs:5520`; telemetry in `crates/runner/src/llama31_engine.rs:1424` | Host-side certificate calculation | Uses true tail upper bound and `exp(guard * Delta)`; telemetry must report achieved values after expansion. |
| Value error certificate `E_val` | `value_error_bound`, `oracle/certified_kv_llama31.py:366` | Runtime value-bound calculation around `crates/runner/src/decode_engine.rs:5623`; telemetry in `crates/runner/src/llama31_engine.rs:1425` | Host-side certificate calculation using masses plus `value_error` | Promoted value blocks are excluded from the INT4 value error sum. |
| Rung 2 value promotion | `certified_attention_step` builds `value_promote_mask`, `oracle/certified_kv_llama31.py:514` | Runtime uses Tier 2 values at `crates/runner/src/decode_engine.rs:5643`; mixed attention call at `:6035` | Mixed kernel currently consumes INT4 Tier 1 for the fast path; dense fallback consumes BF16 Tier 2 | Any promoted value path must source BF16 values from `certified_kv_host_v`, never from dequantised INT4. |
| Ranking consistency order check | `ranking_consistency_fallback_heads`, `oracle/certified_kv_llama31.py:413` | `certified_kv_ranking_mismatch`, `crates/runner/src/decode_engine.rs:391`; runtime use at `crates/runner/src/decode_engine.rs:5703` | Per-head fallback kernel `dense_selected_heads_out_bf16`, `crates/kernel-ffi/src/certified_kv.rs:1953`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:875` | If promoted-set rankings disagree, affected heads use dense selected-head fallback from Tier 2. |
| Ranking boundary verification | `ranking_consistency_fallback_heads`, `oracle/certified_kv_llama31.py:413` | `certified_kv_ranking_boundary_violators`, `crates/runner/src/decode_engine.rs:349`; runtime use at `crates/runner/src/decode_engine.rs:5547` | Host-side check before fallback dispatch | Tail block upper-bound log mass must not enter the top-`r` promoted FP16 ranking. |
| Score consistency and exploration | Oracle scoring/fallback telemetry in `certified_attention_step`, `oracle/certified_kv_llama31.py:514` | `certified_kv_score_consistency_violates`, `crates/runner/src/decode_engine.rs:103`; runtime use at `crates/runner/src/decode_engine.rs:5664`; dense fallback at `:5685` | FP16 comparison data comes from Tier 2/prefix scratch | Any score bound violation escalates to dense layer fallback. |
| Phase 2 mixed attention | `certified_attention_step`, `oracle/certified_kv_llama31.py:514` | Runtime mixed-attention dispatch at `crates/runner/src/decode_engine.rs:6035` | FFI `attend_mixed_key_int4_with_bf16_tail_strided`, `crates/kernel-ffi/src/certified_kv.rs:1651`; CUDA kernel `kernels/certified_kv_bridge_cuda.cu:706` and extern `:1842` | Selected/promoted keys use BF16 scratch from Tier 2; tail uses BF16 tail scratch; unpromoted complete blocks use INT8 keys and INT4 values. |
| Rung 3 per-head fallback | Oracle returns fallback heads from ranking check | Runtime fallback head collection at `crates/runner/src/decode_engine.rs:5731`; prefix scratch cache at `:6249`; dispatch at `:6338` | FFI `dense_selected_heads_out_bf16`, `crates/kernel-ffi/src/certified_kv.rs:1953`; CUDA extern `kernels/certified_kv_bridge_cuda.cu:1949` | Only affected heads are recomputed densely, using Tier 2 originals plus BF16 tail. |
| Rung 4 full dense fallback | Oracle dense-equivalent fallback contract in `certified_attention_step` | `force_dense_layer_fallback`, `crates/runner/src/decode_engine.rs:5454`; dense path begins at `crates/runner/src/decode_engine.rs:5737` | Dense fallback uses BF16 Tier 2/prefix buffers and BF16 tail | This is the unconditional terminal state. It must not depend on compressed data for correctness. |
| Memory/caching telemetry | Oracle telemetry dictionary from `certified_attention_step` | `CertifiedKvShadowStats`, `crates/runner/src/decode_engine.rs:1164`; `certified_kv_memory_stats`, `crates/runner/src/decode_engine.rs:7705`; decode telemetry in `crates/runner/src/llama31_engine.rs:1386` | N/A | Telemetry must distinguish compressed VRAM, Tier 2 host bytes, tail scratch, ranking prefix scratch, H2D bytes, and cache hits. |

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

- The Rust runtime currently performs some CPU-side certification logic using
  copied score/metadata buffers. This is an implementation choice for auditability
  and can be optimized later if it preserves the same telemetry and fallback
  semantics.
- `bf16_values` is retained for bring-up/debug modes. Paper-exact certified
  execution uses INT4 values in Tier 1 and BF16 values only when fetched from
  Tier 2 by fallback/promotion.
- The ranking prefix cache is a scratch optimization for fallback. It is tracked
  separately from Tier 1 compressed VRAM and Tier 2 host memory.
