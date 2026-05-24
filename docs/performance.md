# Performance

Measured decode throughput for the shipping kernels. Numbers are steady-state
tokens/second, single-sequence (`--batch-size 1`) unless noted, recorded with
a 16-token generation on the 6-token `"The quick brown fox jumps over"` prompt.

If you reproduce these and get materially different results, please open an
issue with your GPU arch, ROCm/CUDA versions, and the exact command line.

## HIP — `gfx1100` (AMD Radeon RX 7900 XTX, 24 GiB)

Discrete dGPU; 96 CUs, RDNA3 WMMA. The full quant matrix (BF16, INT4 GPTQ,
FP8 runtime, FP8 KV cache) is supported across every shipping model on
this arch. Measurements last validated 2026-05-04 (warmup + median-of-3
with cooldown between cells; matches the 2026-04-30 baseline at the
[gemma4-fp8-runtime](https://github.com/DeanoC/SuperSonic/pull/51) merge —
see [Methodology](#methodology) below for the cooldown control added
after PR #184 published numbers biased upward by serial-bench thermal
accumulation). 6-token prompt, 16-token generation, single sequence,
`--batch-size 1`. Each cell is the median of 3 `ms/step` measurements
from the runner's `[result] ms_per_step=N` / `ms_per_tok=N` line after
one warm-up run; reproduce with `tests/gfx1100/bench_matrix.sh`.

<!-- AUTOGEN BELOW: bench-perf-matrix -->
| Model           |  BF16 |
| --------------- | ----: |
| qwen3.5-0.8b    |   8.0 |

<!-- AUTOGEN END: bench-perf-matrix -->

| Model           | BF16  | INT4  | FP8r  | KV-FP8 |
|-----------------|------:|------:|------:|-------:|
| qwen3.5-0.8b    |   8   |  10   |  10   |   85¹  |
| qwen3.5-2b      |  11   |  11   |  15   |  126¹  |
| qwen3.5-4b      |  21   |  15   |  30   |  223¹  |
| qwen3.5-9b      |  32   |  26   |  48   |  347¹  |
| gemma4-e2b      |  28   |  34   |  36   |   29   |
| gemma4-e4b      |  46   |  49   |  61   |   47   |
| phi4-mini       |  38.3 |  39.7 |  53.1 |   78.1 |
| qwen3.6-35b-a3b |   —²  |  28.3 |   —²  |   28.5 |

¹ Qwen 3.5 `--kv-fp8` falls back to a *replayed-prefill* decode path
("`single-sequence CUDA KV-FP8 uses replayed GPU prefill for
correctness`"). The decode kernel itself is fine; the slow column is
the per-step prefill replay needed to keep the FP8 KV cache
self-consistent. KV-FP8 on Qwen is currently a memory feature
(headroom for longer contexts), not a throughput feature. Gemma 4 's
`--kv-fp8` is wired into the persistent kernel directly and is
~free vs BF16.

² qwen3.6-35b-a3b doesn't ship a BF16 lane (the source FP8 weights would
  expand to ~70 GiB which exceeds 24 GiB), and FP8 runtime isn't wired
  for the MoE family. INT4-GPTQ is the only viable lane on gfx1100 for
  this model. The dedicated [Qwen3.6-MoE on gfx1100](#qwen36-moe-on-gfx1100)
  section below has the per-stage breakdown.

<!-- AUTOGEN BELOW: hipfire-comparison -->
<!-- AUTOGEN END: hipfire-comparison -->

### Translated to tokens/sec

| Model           | BF16   | INT4   | FP8r   | KV-FP8 |
|-----------------|-------:|-------:|-------:|-------:|
| qwen3.5-0.8b    | 125.0  | 100.0  | 100.0  |  11.8  |
| qwen3.5-2b      |  90.9  |  90.9  |  66.7  |   7.9  |
| qwen3.5-4b      |  47.6  |  66.7  |  33.3  |   4.5  |
| qwen3.5-9b      |  31.3  |  38.5  |  20.8  |   2.9  |
| gemma4-e2b      |  35.7  |  29.4  |  27.8  |  34.5  |
| gemma4-e4b      |  21.7  |  20.4  |  16.4  |  21.3  |
| phi4-mini       |  26.1  |  25.2  |  18.8  |  12.8  |

### Cross-row notes

- **INT4 vs BF16** — INT4 wins on the larger Qwen variants
  (`qwen3.5-4b`: 1.4×, `qwen3.5-9b`: 1.23×) because they're
  memory-bandwidth-bound on 7900 XTX and INT4 halves the weight bytes
  read per step. INT4 is roughly neutral or slightly slower on small
  models (Qwen 0.8B/2B, Gemma E2B, Phi-4-mini) where the per-step
  dequant overhead matches the bandwidth savings.
- **FP8 runtime overhead** — FP8r runs 1.0–1.4× the BF16 ms/step on
  every model. The slowdown is the LDS-LUT-driven per-element FP8
  dequant in the matmul inner loops (`g4_fp8_dequant_weight_lut` /
  `fp8_dequant_weight_lut`); on bandwidth-saturated configs this is
  partly hidden by the 2× weight-bytes-saved, but on compute-tight
  Qwen 0.8B / Gemma E2B the dequant cost wins. FP8 runtime is a
  memory feature first (~half the weight footprint, see VRAM table
  below) and a throughput feature only when paired with KV-FP8 on
  Gemma 4 to free up KV headroom.
- **`--fp8-runtime` cannot combine with `--int4`** on any model
  (separate kernel families). Gemma 4 `--fp8-runtime` and `--kv-fp8`
  additionally require `--batch-size=1` because the FP8 paths are
  wired into the single-batch persistent decode kernel only; the
  batched and INT4 Gemma kernels stay BF16-weights / BF16-KV.

### Methodology

The matrix above is reproduced from `tests/gfx1100/bench_matrix.sh`. As of
2026-05-04 the script applies a **3-second cooldown** between cells and
takes the **median of 3 measurement runs** per cell after a warmup pass.
This control was added after PR #184 published numbers that were biased
upward by serial-bench thermal accumulation — running 7 models × 4 quants
back-to-back with no cooldown left the GPU hot enough that the larger
Qwen3.5 cells measured 1.5–2× slower than their steady-state values.
The 2026-05-04 re-validation against the original 2026-04-30 baseline
confirmed every cell matches when the cooldown + median-of-3 controls
are applied.

If you reproduce these numbers and see materially different results,
please confirm:
- Each cell ran with at least one warmup pass.
- Cooldown between cells was at least 3 seconds.
- The reported number is the median (not the first run; cold first runs
  include kernel JIT compile time).

### VRAM footprint (gfx1100, weights+scratch only)

Approximate steady-state device memory for the weights+scratch portion of
the engine, before any KV cache. KV cache adds linearly with context
length (`num_kv_heads × head_dim × max_t × 2 bytes/elem` BF16, halved
under `--kv-fp8` plus a small per-(head, position) F32 scale overhead).

| Model           | BF16    | INT4      | FP8r      |
|-----------------|--------:|----------:|----------:|
| qwen3.5-0.8b    |   2 GiB |  ~0.7 GiB |  ~1.2 GiB |
| qwen3.5-2b      |   5 GiB |  ~1.9 GiB |  ~3.0 GiB |
| qwen3.5-4b      |  10 GiB |  ~3.7 GiB |  ~6.0 GiB |
| qwen3.5-9b      |  18 GiB |  ~6.7 GiB |  ~10.8 GiB|
| gemma4-e2b      |  11 GiB |  ~4.1 GiB |  ~6.6 GiB |
| gemma4-e4b      |  10 GiB |  ~3.7 GiB |  ~6.0 GiB |
| phi4-mini       |   8 GiB |  ~3.0 GiB |  ~4.8 GiB |

The `INT4` and `FP8r` columns are derived from the registry's
BF16 `fixed_bytes` × the engine's quant scale factor (0.37× for INT4,
0.6× for FP8 — see `crates/runner/src/main.rs` and
`crates/runner/src/phi4_engine.rs`). The same scaling is applied to
the VRAM admission preflight, so memory-constrained cards that pass
preflight will fit at runtime.

### Qwen3.6-MoE on `gfx1100`

`qwen3.6-35b-a3b` is the first MoE model shipped on SuperSonic: 40
layers (30 linear-attention + 10 full-attention in a hybrid pattern),
256 experts with top-8 routing, ~3B active parameters per token,
INT4-GPTQ from the published FP8 source weights. BF16 doesn't fit in
24 GiB; the INT4 bake is the only HIP lane. Steady-state, single-
sequence on RX 7900 XTX with the published v2-int4-gptq bake,
6-token prompt + 16-token generation:

**Production decode (async dispatch, no per-step sync)** — what
`./target/release/supersonic … --max-new-tokens 16` actually runs:

| Stage         | ms/step  | tok/s if alone |
|---------------|---------:|---------------:|
| chain (40 L)  |    24.72 |           40.5 |
| lm_head       |     1.53 |          653.6 |
| sample/detok  |     0.27 |         3703.7 |
| **total**     |**26.53** |       **37.7** |

The chain wall-clock here is host-real-time end-to-end (the chain
ends in a D2H copy that drains the queue), but the per-kernel-class
breakdown isn't observable in this mode — the host can't tell how
much of `chain_ms` is full-attn vs linear-attn vs FFN without
forcing a sync between each step launch.

**With `--emit-stage-timings`** (per-step `gpu_hal::sync`, slower but
attributable):

| Stage          | ms/step  | tok/s if alone | share of chain |
|----------------|---------:|---------------:|---------------:|
| ↳ FFN (40 L)         | 11.42 |          87.6 |             43% |
| ↳ linear-attn (30 L) | 12.29 |          81.4 |             46% |
| ↳ full-attn (10 L)   |  2.46 |         406.5 |              9% |
| sub-stage sum        | 26.17 |               |            ~99% |
| chain (40 L)         | 26.42 |          37.9 |               — |
| lm_head              |  1.53 |         653.6 |               — |
| sample/detok         |  0.27 |        3703.7 |               — |
| **total**            |**28.21**|     **35.4**|               — |

Per-stage `tok/s` is `1000 / ms_per_step` — the throughput that
stage would sustain if it were the only cost. The 1.7 ms gap between
the two `total`s is the per-step sync overhead (~80 syncs/token); the
production headline is the async **37.7 tok/s** number.

This is **3.0× the original `12.6 tok/s` baseline** (PR #74's concurrent
expert dispatch). The cumulative gain across the WMMA + dispatch
optimisation arc:

| Land    | Phase                              | total ms | tok/s | Δ          |
|---------|------------------------------------|---------:|------:|-----------:|
| PR #74  | concurrent K_top routed experts    |    79.3  |  12.6 | (baseline) |
| PR #76  | lm_head WMMA tile                  |    55.9  |  17.9 |       +42% |
| PR #77  | per-expert FFN INT4 WMMA           |    36.2  |  27.7 |      +120% |
| PR #78  | linear-attn INT4 WMMA              |    29.9  |  33.5 |      +167% |
| PR #79  | full-attn INT4 WMMA                |    28.3  |  35.3 |      +180% |
| PR #80  | defer per-step bridge syncs        |    26.5  |  37.7 |      +199% |
| PR #128 #138 #140 | Phase 3a-f persistent megakernel + lm_head fold (default-on) | 28.6  | 35.0 |  +178%¹ |

¹ Re-measured 2026-05-03 on the canonical fox prompt + 16 gen tokens.
The chained baseline now measures 31.5 ms (vs PR #80's 26.5 ms) on the
same prompt, presumably because of intervening thermal/driver state
drift across the months the table covers; persistent saves ~3 ms vs
the chained baseline as measured TODAY (`chain 29.63 + lm_head 1.59
ms` → `chain 28.25 + lm_head 0.08 ms`, **+10.2% tok/s**), which is
the more reproducible number.

Architectural notes:

- **Decode INT4 GEMVs run through RDNA3 WMMA**
  (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`) for every weight matmul:
  q/k/v/o_proj (full-attn), in_proj_qkv/z + out_proj (linear-attn),
  per-expert gate_up + down_proj (FFN), and lm_head. INT4 weights are
  dequanted to BF16 in LDS per WMMA tile, so bandwidth utilisation is
  near-peak on the matmul-bound phases. Helper:
  `wmma_int4_matvec_partial_16rows` in `kernels/qwen36_moe.hip`.
- **`USE_WMMA` template parameter** on each step kernel
  (`qwen36_moe_attn_step_kernel`, `qwen36_moe_linear_step_kernel`,
  `qwen36_moe_ffn_step_kernel`). Bridge picks the instantiation at
  launch time based on `device_supports_wmma_bf16(ord)` + dim
  divisibility checks. `SUPERSONIC_QWEN4B_DISABLE_WMMA=1` forces
  every WMMA path back to the scalar fallback for A/B work.
- **Concurrent K_top expert dispatch** (PR #74) preserves block
  partitioning across PRs #76-79: `group_id = blockIdx.x % top_k`,
  `sub_id = blockIdx.x / top_k`. Each routed expert group runs G/H/I
  in parallel; the shared expert still runs sequentially ahead (the
  9th-group experiment regressed by ~4 ms from L2 pressure, see
  PR #77 for details). 35B-A3B uses top_k=8, exactly at the FFN
  sync_buf counter cap of 16 slots.
- **Async chain dispatch** (PR #80): the 80 step launches per token
  no longer `hipDeviceSynchronize` between steps — the default stream
  serializes and the chain-end D2H of `final_hidden_bytes` is the
  natural barrier. With `--emit-stage-timings` the per-step sync comes
  back so the breakdown above stays accurate; the production hot path
  runs async and saves ~1.8 ms/token.

Remaining wedges (looking forward):

- **Phase G of linear-attn (delta-rule recurrent state update)** is
  *state-bound*, not weight-bound — the per-V-head recurrent state
  matrix is ~2 MiB per layer and the kernel reads/writes it five times
  per step. WMMA can't help; further wins would need a state-layout
  redesign or fused-state kernel.
- **Persistent megakernel landed** (Phase 3a-f, PRs #118 → #140).
  Default decode path on qwen3.6-MoE/HIP since PR #138 — one
  cooperative HIP launch processes all 40 layers per token plus the
  final RMSnorm + lm_head GEMV (Phase 3f, PR #140). The chained
  per-step launchers stay reachable via `--no-persistent-decode` for
  A/B work or bisecting a suspected megakernel-side regression. Bit-
  exact greedy output to the chained baseline, gated by:
  - `multilayer_persistent_decode_matches_chained` parity test on
    synthetic fixtures.
  - The verify suite's `chained_vs_persistent` SHA256 gate
    (logits + final hidden + generated_ids byte-identical) on real
    PG-19 + RULER prompts. Validated across 10 case configurations:
    PG-19 + RULER × {128, 512, 2K, 4K, 8K} × INT4 — 10/10 byte-
    identical, all positive perf delta.
  | Family | Context | Δ chain | Δ total |
  |---|---|---:|---:|
  | PG-19  |   128 | +8.4% | +7.9% |
  | PG-19  |   512 | +7.7% | +7.5% |
  | PG-19  |    2K | +4.5% | +4.5% |
  | PG-19  |    4K | +3.8% | +3.9% |
  | PG-19  |    8K | +2.6% | +2.6% |
  | RULER  |   128 | +7.3% | +6.7% |
  | RULER  |   512 | +7.6% | +7.3% |
  | RULER  |    2K | +5.3% | +5.2% |
  | RULER  |    4K | +1.0% | +1.1% |
  | RULER  |    8K | +2.5% | +2.6% |
  Absolute reclaim is ~2.5-3 ms/token regardless of context (the
  chained path's HIP launch overhead: 80 step launches × ~30 µs +
  one separate `lm_head_launch` × ~30 µs). Relative speedup shrinks
  at longer context because chain compute grows with context while
  launch overhead stays per-token-constant.
- **Speculative decode + persistent** (Phase 3e.3, PR #136). The
  K+1 verify chains and replay chains in `--speculative-decode` go
  through the same persistent path, saving the same ~2.7 ms/chain.
  Spec + persistent A/B on the 8-token fox prompt × 32 gen: chain
  29.59 → 26.65 ms (+9.9%), total 31.22 → 28.22 ms (**32.0 → 35.4
  tok/s, +10.7%**), generation bit-identical to spec + chained.
- **lm_head fold details** (Phase 3f, PR #140). The folded final
  RMSnorm + lm_head GEMV runs only at gen steps (the fold is a
  no-op on prefill: `step + 1 < prompt_ids.len()` ⇒ pass `None`).
  Spec-verify chains skip the fold too — they batch K+1 lm_heads
  through the dedicated `lm_head_batched_launch` (Phase 6.4a)
  which amortizes the 970 MiB BF16 weight read better than K+1
  per-chain folds would. The work-stealing WMMA path in
  `lm_head_phase.cuh` reuses the persistent kernel's existing
  9 KiB LDS + 96 blocks × 8 waves = 768 concurrent waves chewing
  through vocab=248k tiles, vs the standalone WMMA kernel's
  15.5k blocks × 1 wave (which queue rather than run concurrently).
- **Speculative decode** is the realistic path to 100+ tok/s — needs a
  draft head (MTP/Eagle) and a verification kernel that batches K
  candidates × layers in one launch.

INT4 weight + scratch on the 35B-A3B bake: ~17 GiB on disk, ~21 GiB
runtime including KV cache at the default context. Within the 24 GiB
budget. Calibration needs more host RAM than typical 7900 XTX rigs
carry, so the bake is produced on a bigger box and distributed via
GitHub releases (see [bake-distribution.md](bake-distribution.md));
consumers pull it automatically on first run.

**Sparse MoE VMM residency sweep** — measured 2026-05-03 on the same
RX 7900 XTX with `tests/gfx1100/bench_qwen36_sparse_caps.py`, the
canonical fox prompt, 16 generated tokens, INT4 weights, persistent decode,
and `--emit-stage-timings`. The dense row is fully resident virtual expert
slabs; sparse rows set `SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=N` and capture the
runner's VMM telemetry JSON.

| Mode | total ms/tok | tok/s | total resident GiB | MoE resident GiB | KV resident GiB | peak pages | page misses | evicted pages | ids match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| dense | 28.71 | 34.83 | 15.04 | 15.00 | 0.04 | - | - | - | yes |
| cap8 | 137.95 | 7.25 | 0.07 | 0.03 | 0.04 | 16 | 13099 | 13083 | yes |
| cap32 | 140.26 | 7.13 | 0.16 | 0.12 | 0.04 | 64 | 13099 | 13035 | yes |
| cap64 | 152.71 | 6.55 | 0.29 | 0.25 | 0.04 | 128 | 13099 | 12971 | yes |
| cap128 | 140.89 | 7.10 | 0.54 | 0.50 | 0.04 | 256 | 13099 | 12843 | yes |
| cap256 | 142.70 | 7.01 | 1.04 | 1.00 | 0.04 | 512 | 13099 | 12587 | yes |
| cap320 | 104.52 | 9.57 | 1.29 | 1.25 | 0.04 | 640 | 8357 | 7717 | yes |

The curve is a memory win but not a throughput win: cap320 cuts total VMM
resident memory from ~15.0 GiB to ~1.3 GiB, but it is still ~3.6× slower than
dense on this short prompt because page misses dominate. Sparse islands remain
an opt-in small-VRAM mode for now; the runtime should not auto-default to
`SUPERSONIC_MOE_ISLAND_CAP_EXPERTS` on gfx1100 until prefetching/reuse reduces
the miss rate substantially.

The sweep helper can compare prefetch policies in one run. Use
`--prefetch-rank-sweep none,1,2,4,all` to expand every sparse cap across no
lookahead, rank-limited lookahead, and full top-k lookahead rows. Use
`--prefetch-mode-sweep disabled,previous-token,previous-token-resident,transition`
together with `--prefetch-rank-sweep none,1,all` to compare normal
previous-token prefetch, resident-only LRU refresh, and online transition-aware
admission without hand-running separate commands. The markdown table includes
same-rank repeat, previous-rank reuse, and best-transition columns derived from
the route transition matrix.
Use `--protected-sweep none,32,64,96,128` to expand each sparse row across
protected eviction-band sizes; this sets
`SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS` for sparse rows only and adds
protected-page/protect-hit columns to the markdown output.

Short protected-band smoke on 2026-05-03 (`cap320`, 8 generated tokens,
no warmup) kept resident memory fixed at 1.29 GiB and preserved greedy output,
but it did not improve throughput: `p0` measured 110.43 ms/tok, `p32` 112.54,
`p64` 113.81, `p96` 113.87, and `p128` 123.97. Larger protected bands also
increased demand misses on this prompt, so the protected band is telemetry and
tuning infrastructure for now, not a default policy.

**Sparse MoE previous-token prefetch sweep** — measured 2026-05-03 after
non-evicting prefetch admission landed. Same host/GPU/model/prompt as above,
cap fixed at `SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=320`, 16 generated tokens,
no warmup:

| Mode | total ms/tok | tok/s | total resident GiB | prefetch | ranks | page misses | prefetch page misses | prefetch skipped | rank0 resident | rank0 repeat | evicted pages | ids match |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| dense | 28.73 | 34.81 | 15.04 | - | - | - | - | - | - | - | - | yes |
| cap320-none | 108.66 | 9.20 | 1.29 | disabled | 0 | 8357 | 0 | 0 | 52.4% | 52.3% | 7717 | yes |
| cap320-r1 | 119.62 | 8.36 | 1.29 | previous-token | 1 | 9815 | 0 | 616 | 42.6% | 52.3% | 9175 | yes |
| cap320-r1-resident | 116.18 | 8.61 | 1.29 | previous-token-resident | 1 | 9752 | 0 | 0 | 43.9% | 52.3% | 9112 | yes |
| cap320-r2 | 131.78 | 7.59 | 1.29 | previous-token | 2 | 11190 | 0 | 2196 | 25.8% | 52.3% | 10550 | yes |
| cap320-r4 | 140.39 | 7.12 | 1.29 | previous-token | 4 | 12310 | 0 | 5511 | 9.5% | 52.3% | 11670 | yes |
| cap320-all | 147.75 | 6.77 | 1.29 | previous-token | 8 | 12927 | 0 | 11961 | 0.6% | 52.3% | 12287 | yes |

Non-evicting admission successfully prevents prefetch page misses, but this
previous-token policy is still a regression on the fox prompt. As ranks
increase, skipped prefetches and demand page misses rise monotonically. The
resident-only row keeps prefetch page misses and skipped prefetches at zero,
but still regresses versus no lookahead, so even refreshing previous-token
residents perturbs LRU in the wrong direction for this prompt. Treat
previous-token prefetch as diagnostic only; the next useful policy needs a
stronger admission signal than "same layer's previous token top-k".

**Sparse MoE transition-prefetch smoke** — measured 2026-05-03 with
`SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=320`, 8 generated tokens, no warmup, and
`--prefetch-transition-min-obs 1` to force the online predictor path to admit
within the short run. IDs matched across all rows. The transition path remained
non-evicting and recorded zero prefetch page misses, but it regressed this short
fox prompt (`126.31 ms/tok`) versus no lookahead (`108.63 ms/tok`) and
previous-token rank-1 (`122.00 ms/tok`). Treat transition prefetch as an
experimental policy that needs longer prompts and prompt diversity before it can
be considered for defaults.

Reproduce:

```bash
cargo build --release --bin supersonic
./target/release/supersonic --model qwen3.6-35b-a3b \
  --model-dir /path/to/Qwen3.6-35B-A3B-FP8 \
  --prompt "The quick brown fox jumps over" \
  --max-new-tokens 16 --emit-stage-timings
```

#### Long-context prefill (batched-Q, default since 2026-05-05)

Long-context prefill on `qwen3.6-35b-a3b` was the dominant wall-clock
cost for the research workload that drove this work — 9.6 minutes to
prefill 8K tokens on the per-token persistent megakernel. The
batched-Q prefill path (now the default) chunks the prompt and runs
all 32 full-attention layers' attention through a K-tiled
FlashAttention-style kernel that shares K/V tile loads across the
chunk's queries, plus a permute-by-expert grouped INT4 GEMM for the
MoE FFN that runs all 256 experts in one launch per layer per chunk.

Bench (gfx1100, qwen3.6-35b-a3b INT4, NIAH-style synthetic prompts,
`prefill_total_ms` from `--emit-stage-timings`):

| Context | Per-token (legacy) | Batched-Q (default) | Speedup |
|--------:|-------------------:|--------------------:|--------:|
|     512 |             13.4 s |               8.9 s |   1.50× |
|    2048 |             61.1 s |              38.6 s |   1.58× |
|    4096 |            134.4 s |              75.0 s |  **1.79×** |

Chunk size is chosen at runtime from the prompt's prefix context,
capped at the largest size that gives 100% WMMA utilization in the
grouped MoE GEMM (`chunk = 16 × num_experts / top_k = 512` for
`top_k=8` and `num_experts=256`); larger chunks land marginal extra
K-tile-share in attention. Trailing partial chunks use the exact
remaining size in one call.

Bisect/escape hatches (each defaults OFF — i.e. batched is on):
  - `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0` — revert entire prefill
    loop to the legacy per-token persistent megakernel.
  - `SUPERSONIC_QWEN36_MOE_BATCHED_ATTN=0` — keep the chunked
    orchestrator but call per-token chain step inside each chunk
    (no perf benefit, just structural).
  - `SUPERSONIC_QWEN36_MOE_GROUPED_FFN=0` — keep batched attention
    but fall back to per-token FFN inside the chunk.

FP8 KV cache and SpecPrefill / sparse-prefill modes automatically
fall through to the per-token path (`supports_batched_path()` in
`crates/runner/src/qwen36_moe/batched_prefill.rs`).

Reproduce the A/B:

```bash
# Baseline (legacy per-token):
python3 tests/gfx1100/bench_qwen36_longctx.py --no-batched-prefill \
  --contexts 512,2048,4096 --modes int4-vmm --max-new-tokens 4

# Batched (default):
python3 tests/gfx1100/bench_qwen36_longctx.py \
  --contexts 512,2048,4096 --modes int4-vmm --max-new-tokens 4
```

## HIP — `gfx1150` (AMD Radeon 890M iGPU)

16 CUs, 2.9 GHz core, shared with system memory. Measurements on
2026-04-20 at current main (0.8B-native kernel deleted, 2× grid
oversubscription merged, registry-driven cooperative-launch preset
for 0.8B).

### Qwen3.5

| Model              | Quant | ms/tok | tok/s |
|--------------------|-------|-------:|------:|
| qwen3.5-0.8b       | BF16  |   34   | 29.4  |
| qwen3.5-0.8b       | INT4  |   44   | 22.7  |
| qwen3.5-2b         | BF16  |   78   | 12.8  |
| qwen3.5-2b         | INT4  |   58   | 17.2  |
| qwen3.5-4b         | BF16  |  160   |  6.3  |
| qwen3.5-4b         | INT4  |  110   |  9.1  |
| qwen3.5-9b         | FP8   |  697   |  1.4  |

Notes:

- `qwen3.5-0.8b` decode (both BF16 and INT4) runs through the 4B persistent
  megakernel. The dedicated 0.8B-native kernel was deleted on 2026-04-20 —
  it had no INT4/FP8 path and was ~2.8× slower than the 4B-routed path
  even for BF16.
- The 0.8B HIP registry entry carries a `hip_launch_preset` of `(32 blocks,
  cooperative=true)`, installed automatically at startup. Cooperative
  launch on gfx1150 caps conservatively at 24 blocks for 0.8B's 14 KB LDS
  footprint — that's where the ~1.2× speedup over the plain 2× default
  (16 blocks) comes from. 2B/4B/9B have no preset because their larger LDS
  caps the cooperative grid at or below the 2× default; they stay on the
  non-cooperative path. `SUPERSONIC_QWEN4B_BLOCKS` / `SUPERSONIC_QWEN4B_COOP`
  env vars still override the preset.
- `qwen3.5-9b` INT4 bake runs out of VRAM during GPTQ calibration on 16 GiB
  cards. Consumers pull the released bake from GitHub releases (see
  [bake-distribution.md](bake-distribution.md)); the INT4 runtime itself is
  supported.
- FP8-runtime and FP8-KV paths (`--fp8-runtime`, `--kv-fp8`) are only wired
  for the Qwen family on HIP. Gemma 4 and Phi-4-mini reject both flags.

### Gemma 4

| Model        | Quant | ms/tok | tok/s |
|--------------|-------|-------:|------:|
| gemma4-e2b   | BF16  |  246   | 4.07  |
| gemma4-e2b   | INT4¹ |  230   | 4.35  |
| gemma4-e4b   | BF16  |  425   | 2.35  |

¹ Gemma 4 E2B INT4 runs but quality is degraded — the GPTQ bake is
distributed from releases and produces coherent first tokens but
devolves into repetition within a few generations. INT4 quality
calibration for the E2B bake is parked pending a revisit. E4B INT4
calibration OOMs on this machine and is also parked — BF16 is the
shipping path for E4B on gfx1150.

**Gemma 4 BF16 prefill** routes through a WMMA
(`v_wmma_f32_16x16x16_bf16`) tiled matmul when `seq_len >= 16` on
gfx11xx; shorter prefills (and decode, which is always `seq_len == 1`)
stay on the work-stealing scalar kernel. Gated by
`SUPERSONIC_GEMMA4_DISABLE_WMMA=1`. Prefill speedups measured
2026-04-20:

| Model        | Quant | Prompt tokens | Scalar    | WMMA    | Speedup |
|--------------|-------|--------------:|----------:|--------:|:--------|
| gemma4-e2b   | BF16  |          1021 | 417190 ms | 8863 ms |  47.1×  |
| gemma4-e4b   | BF16  |           241 | 206045 ms | 5935 ms |  34.7×  |
| gemma4-e2b   | INT4  |          1021 | 182116 ms | 4866 ms |  37.4×  |

The ratio is larger than the Qwen WMMA port (~2–4×) because the Gemma 4
scalar path was a one-block-per-output-element work-stealing matvec
(fine for decode, terrible for prefill), not a tiled matmul.

### Phi-4-mini

| Model        | Quant | ms/tok | tok/s |
|--------------|-------|-------:|------:|
| phi4-mini    | BF16  |  298   | 3.36  |
| phi4-mini    | INT4  |  359   | 2.78  |

### Scaling with context length

Qwen3.5 BF16, 8-token generation, prefill ms + decode ms/tok at varying
prompt size. Decode grows slowly with KV size; prefill grows linearly
with prompt tokens. Prefill numbers reflect the RDNA3 WMMA
(`v_wmma_f32_16x16x16_bf16`) matmul shipped 2026-04-20; set
`SUPERSONIC_QWEN4B_DISABLE_WMMA=1` to fall back to the scalar kernel.

| Variant            | 1020 tok prefill (WMMA) | 1020 tok prefill (scalar) | WMMA speedup |
|--------------------|------------------------:|--------------------------:|:-------------|
| qwen3.5-0.8b BF16  |        5879 ms          |        11244 ms           |   1.91×      |
| qwen3.5-2b  BF16   |        9206 ms          |        24963 ms           |   2.71×      |
| qwen3.5-4b  BF16   |       29891 ms          |        70075 ms           |   2.34×      |
| qwen3.5-2b  INT4   |        6681 ms          |        24532 ms           |   3.67×      |
| qwen3.5-4b  INT4   |       18669 ms          |        73065 ms           |   3.91×      |
| qwen3.5-9b  INT4   |       33486 ms          |       126646 ms           |   3.78×      |

INT4 gains more than BF16 because the scalar INT4 kernel also pays the
dequant cost every iteration — moving the loop body to WMMA BF16 with the
dequant bundled into the B-matrix load amortizes both savings together.

At 1026-token prompts **prefill is 11-13× the total decode time for an
8-token reply**. Any further decode optimization (cooperative launch
tweaks, VGPR reduction, etc.) is invisible to long-prompt users
compared to a prefill win.

### Where time goes at 1026-token context

Per-decode-step breakdown from `--emit-stage-timings` (sum across
layers, divided by step count to give ms/step-of-that-section):

**qwen3.5-0.8b BF16 decode @1026 ctx** (95 ms/step persistent):
| Section           | ms/step | share |
|-------------------|-------:|------:|
| linear_out        |  21.9  |  23%  |
| mlp_gate_up       |  18.3  |  19%  |
| linear_proj       |  15.8  |  17%  |
| linear_core       |  14.2  |  15%  |
| mlp_down          |   7.7  |   8%  |
| full_attn_proj    |   7.3  |   8%  |
| full_attn_out     |   5.1  |   5%  |
| full_attn_core    |   4.4  |   5%  |

**qwen3.5-2b BF16 decode @1026 ctx** (185 ms/step persistent):
| Section           | ms/step | share |
|-------------------|-------:|------:|
| mlp_gate_up       |  38.2  |  21%  |
| mlp_down          |  34.9  |  19%  |
| linear_core       |  30.3  |  16%  |
| linear_out        |  30.0  |  16%  |
| linear_proj       |  19.4  |  10%  |
| full_attn_out     |  13.0  |   7%  |
| full_attn_core    |   9.7  |   5%  |
| full_attn_proj    |   9.6  |   5%  |

Implications for future work:

- **Qwen3.5 decode is linear-attention-dominated**, not full-attention
  (linear_proj + linear_core + linear_out = 55% on 0.8B, 42% on 2B).
  A `--kv-fp8` win would only touch `full_attn_core` which is ≤5%;
  that's a VRAM feature, not a throughput feature on Qwen.
- `full_attn_core` grows with KV size as expected for per-decode-step
  attention (one query × kv_len past positions is O(kv_len)): 1.1 →
  4.4 ms/step on 0.8B from 64 → 1024 ctx. The measured 4× at 16× ctx
  is sub-linear because fixed kernel-launch and barrier overhead
  dominates at small KV sizes. In absolute terms it stays small
  (≤5% of decode step time across tested contexts).
- Prefill's dominant matmul used to be a naive scalar-FMA tiled kernel
  eating ~80% of prefill time. As of 2026-04-20 it runs on a WMMA
  (`v_wmma_f32_16x16x16_bf16`) port that lands 1.9–2.7× on prefill
  end-to-end for the three BF16 Qwen variants (see the table above).
  Further wins on top are possible — shared-memory tiling across
  multiple waves, larger output tiles per block, dual-issue packing —
  but require a second pass.

## CUDA — `sm86` (NVIDIA RTX 3090-class)

24 GB VRAM, 936 GB/s memory bandwidth. Measurements below were refreshed on
2026-04-23 at commit `7837902` for the Llama INT8 lane and commit `5a34190`
for the Qwen rows.

Quick checked paths (`PROMPT_REPEAT=8`, `MAX_NEW_TOKENS=8`, `RUNS=1`):

| Model                       | Path                         | Prefill   | Decode     |
|----------------------------|------------------------------|-----------|------------|
| qwen3.5-0.8b              | default (hero)               | 544 tok/s | 106.7 tok/s |
| qwen3.5-4b `--batch-size 1` | default (single kernel)      | 124.7 tok/s | 26.0 tok/s |
| qwen3.5-4b `--batch-size 2` | default (batched kernel)     | 122.9 tok/s | 15.4 tok/s¹ |

Warmed native single-stream `4B` hero lane
(`./tests/sm86/bench_qwen4b_single.sh`, `pp533 / tg16`, commit `5a34190`):

| Model                       | Path                    | Prefill   | Decode     | Persistent |
|----------------------------|-------------------------|-----------|------------|------------|
| qwen3.5-4b `--batch-size 1` | `--force-kernel-decode` | 101.5 tok/s | 22.0 tok/s | 654.6 ms   |

Current `llama3.1-8b` CUDA INT8 single-sequence lane
(`--int8`, prompt `Hello`, `tg32`, commit `7837902`):

| Model         | Path                    | Prefill   | Decode     |
|---------------|-------------------------|-----------|------------|
| llama3.1-8b   | baked INT8 component path | n/a     | 38.9 tok/s |

Notes:

- `qwen3.5-4b` batch-1 CUDA decode now defaults to the kernel path. The older
  replayed-prefill decode path is legacy debugging behavior and must be
  requested explicitly with `--force-replay-decode`.
- The warmed `4B` hero-lane benchmark above also recorded these stage means:
  `full_attn_core=121.1 ms`, `linear_proj=35.1 ms`, `linear_out=78.7 ms`,
  `mlp_gate_up=160.6 ms`, `mlp_down=212.9 ms`.
- `llama3.1-8b --int8` currently runs through the shared component decode path,
  not a dedicated persistent kernel. It uses CUDA fast-greedy lm-head scoring,
  reusable component MLP scratch, and strided-KV decode attention. A `pp2/tg16`
  staged run recorded `392.4 ms` total over 15 timed tokens
  (`26.2 ms/token`): `full_attn=154.1 ms`, `mlp=205.6 ms`,
  `rms_norm=13.1 ms`, `lm_head=19.5 ms`.
- ¹ The batched decode figure is aggregate tokens/second across
  `--batch-size 2`.

Llama 3.1 8B arxiv_v1 retrieval smoke QA
(`./tests/sm86/bench_llama31_arxiv_v1_smoke.sh`, commit `9d00178`):

The current CUDA certified-KV runtime stores completed blocks in Tier-1
compressed form (INT8 keys + INT4 values) and retains BF16 originals in
host-pinned Tier-2 storage. The live decode path runs the adaptive selector,
pages selected key blocks from Tier-2 into a compact device scratch buffer, and
uses INT4 values for aligned blocks. Value escalation and ranking fallback are
still not wired in live decode, so the quality contract is not yet the full
paper ladder.

| Subtask           | Path              | Context | Score | DotCache ref | Decode ms/tok |
|-------------------|-------------------|--------:|------:|-------------:|--------------:|
| niah_single       | dense INT8        |    4096 | 1.000 |        1.000 |         397.6 |
| niah_single       | certified KV INT8 |    4096 | 1.000 |        1.000 |          74.5 |
| niah_multikey     | dense INT8        |    4096 | 1.000 |        1.000 |         402.1 |
| niah_multikey     | certified KV INT8 |    4096 | 1.000 |        1.000 |          82.5 |
| niah_multiquery   | dense INT8        |    4096 | 1.000 |        1.000 |         404.6 |
| niah_multiquery   | certified KV INT8 |    4096 | 1.000 |        1.000 |          83.2 |

The arxiv_v1 smoke harness replays the DotCache synthetic retrieval subtasks
with deterministic seeds, scores only the generated suffix, compares against
the normalized DotCache reference results from
`/workspace/DotCache/benchmarks/results/arxiv_v1_20260420`, and fails on
critical certified-vs-dense regressions. The 4K smoke above passed all gates.

Llama 3.1 8B PG-19 teacher-forced smoke QA is covered separately by
`./tests/sm86/bench_llama31_pg19_smoke.sh`. It uses the Rust
`--teacher-forced` scorer, which prefills the first token, feeds the true next
token through dense or certified-KV CUDA decode, and accumulates NLL from the
returned logits. A tiny local-text probe (`CONTEXTS=32`, one chunk, commit
`8bffbca`) passed the dense-vs-certified gate:

| Source        | Path              | Context | Chunks | PPL     | Decode ms/tok |
|---------------|-------------------|--------:|-------:|--------:|--------------:|
| local fixture | dense INT8        |      32 |      1 | 239.558 |          37.0 |
| local fixture | certified KV INT8 |      32 |      1 | 235.822 |          36.6 |
| PG-19 stream  | dense INT8        |     512 |      1 |   6.727 |          53.6 |
| PG-19 stream  | certified KV INT8 |     512 |      1 |   6.783 |          38.6 |
| PG-19 stream  | dense INT8        |    4096 |      1 |   6.279 |         222.7 |
| PG-19 stream  | certified KV INT8 |    4096 |      1 |   6.294 |          99.1 |

The 512-token PG-19 smoke (`target/pg19_smoke_real_512.json`, one streamed
test chunk, commit `8bffbca` + docs update) passed the default
`MAX_CERTIFIED_DELTA=0.10` gate with certified delta `+0.055` ppl. This is
still a quick smoke baseline rather than a final quality number.

The 4K reference-grade smoke (`target/pg19_smoke_reference_4k.json`) uses the
DotCache PG-19 protocol: dense scores the full 4095-token target stream, while
certified uses a 50% dense prefix (`dense_prefix_len=2048`), skips the boundary
target, and scores the certified suffix (`4094` scored tokens,
`2047` certified decode steps). It passed `REFERENCE_SMOKE=1` and
`FAIL_ABOVE_REFERENCE=1` against
`/workspace/DotCache/benchmarks/results/arxiv_v1_20260420`: dense PPL
`6.279` vs DotCache `6.259`, certified PPL `6.294` vs DotCache `6.284`, and
certified-vs-dense delta `+0.015` ppl. Use:
`CONTEXTS=4096 REFERENCE_SMOKE=1 FAIL_ABOVE_REFERENCE=1` for this lane.

CUDA `sm86` tracks detailed kernel-level optimization history for both the
`0.8B` and `4B` hero lanes in
[qwen35-sm86-optimization.md](qwen35-sm86-optimization.md).

## CUDA — `sm90` (NVIDIA H100 80GB HBM3)

Measurements recorded 2026-05-07 on an NVIDIA H100 80GB HBM3, driver
580.126.09, CUDA toolkit 13.0 (`nvcc` 13.0.88). The H100 path compiles native
SM90 CUDA objects but currently reuses the CUDA `sm86` registry geometry and
kernel families. Treat these as a compatibility baseline, not a
Hopper-optimized result.

Before measuring, a resident vLLM `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
server using ~74 GiB of VRAM was stopped. The benchmark model directories were
empty before the run; SuperSonic downloaded the `bakes-v2` BF16 release
artifacts and bundled HF metadata on first use:

- `/home/deano/models/Qwen3.5-0.8B-sm90`
- `/home/deano/models/Qwen3.5-4B-sm90`

Smoke checks:

| Model         | Prompt tokens | Generated | Prefill | Decode | Result |
|---------------|--------------:|----------:|--------:|-------:|--------|
| qwen3.5-0.8b  |             6 |         4 |   39 ms |  64 ms | pass |
| qwen3.5-4b    |             6 |         4 |  103 ms | 127 ms | pass |

Warmed benchmark settings:

```bash
PATH=/home/deano/.cargo/bin:$PATH \
SUPERSONIC_BACKENDS=cuda TARGET_PROMPT_TOKENS=224 MAX_NEW_TOKENS=32 \
WARMUP_RUNS=3 TIMED_RUNS=5 COMPARE_LEGACY=0 \
  ./tests/sm86/bench_qwen08.sh /home/deano/models/Qwen3.5-0.8B-sm90

PATH=/home/deano/.cargo/bin:$PATH \
SUPERSONIC_BACKENDS=cuda TARGET_PROMPT_TOKENS=224 MAX_NEW_TOKENS=32 \
WARMUP_RUNS=3 TIMED_RUNS=5 \
  ./tests/sm86/bench_qwen4b_single.sh /home/deano/models/Qwen3.5-4B-sm90

PATH=/home/deano/.cargo/bin:$PATH \
SUPERSONIC_BACKENDS=cuda TARGET_PROMPT_TOKENS=224 MAX_NEW_TOKENS=32 \
WARMUP_RUNS=3 TIMED_RUNS=5 BATCH_SIZE=2 \
  ./tests/sm86/bench_qwen4b_batch.sh /home/deano/models/Qwen3.5-4B-sm90
```

The prompt calibrator selected `prompt_repeat=32`, producing 416 prompt
tokens. The table reports the mean over 5 timed runs after 3 warmup runs.

| Model                       | Path                    | Prefill     | Decode      | Decode mean |
|-----------------------------|-------------------------|------------:|------------:|------------:|
| qwen3.5-0.8b                | fast-greedy BF16        | 1362.1 tok/s | 32.2 tok/s  | 994.8 ms / 32 tok |
| qwen3.5-4b `--batch-size 1` | `--force-kernel-decode` | 784.9 tok/s  | 26.4 tok/s  | 1210.2 ms / 32 tok |
| qwen3.5-4b `--batch-size 2` | batched BF16            | 785.8 tok/s  | 11.3 tok/s¹ | 5675.2 ms / 64 aggregate tok |

¹ Aggregate tokens/second across `--batch-size 2`.

The quick combined harness also passed:

```bash
PATH=/home/deano/.cargo/bin:$PATH \
SUPERSONIC_BACKENDS=cuda RUNS=3 MAX_NEW_TOKENS=16 PROMPT_REPEAT=16 \
BATCH_SIZE_4B=2 \
  ./tests/sm86/bench.sh \
  /home/deano/models/Qwen3.5-0.8B-sm90 \
  /home/deano/models/Qwen3.5-4B-sm90
```

That quick pass produced 224 prompt tokens and 16 generated tokens. The first
0.8B prefill included a cold-start outlier (`4551 ms`, then `189/186 ms`), so
the warmed table above is the preferred H100 baseline.

## Metal — `apple-m4` (Apple M4)

The current Metal numbers are still prototype-grade and are scoped narrowly to
`qwen3.5-0.8b`, but they are now fast enough to serve as a stable Apple-silicon
performance checkpoint instead of a pure bring-up lane.

Measurements below were recorded on the Apple M4 development machine using the
checked-in Metal bughunt harness and the local cached Qwen3.5 0.8B snapshot.
The benchmark command was:

```bash
target/debug/qwen35_bughunt \
  --mode bench \
  --backend metal \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt-manifest crates/runner/bughunt/qwen35_metal_manifest.json \
  --prompt hello_world \
  --iters 3 \
  --warmup 1 \
  --decode-tokens 4 \
  --profile-ops
```

Current checkpoint:

| Model         | Path                     | Metric                  | Value      |
|---------------|--------------------------|-------------------------|-----------:|
| qwen3.5-0.8b  | native prefill           | prefill wall time       |   107 ms   |
| qwen3.5-0.8b  | greedy prefill           | first-token wall time   |    99.7 ms |
| qwen3.5-0.8b  | replay decode            | decode wall time        |    84.0 ms/tok |
| qwen3.5-0.8b  | component decode proto   | decode wall time        |    35.2 ms/tok |

What moved this checkpoint materially:

- runtime profiling exposed command-buffer creation / wait overhead instead of
  treating Metal as a black box
- lazy batch encoder creation removed the worst encoder churn in prefill
- standalone matvec now uses native Metal by default instead of the host path
- component decode now reuses the persistent argmax buffer instead of allocating
  and flushing an argmax buffer every token

What still dominates:

- prefill is no longer host-fallback dominated on the benchmarked path
- replay decode is still mostly a correctness/reference lane
- component decode is now mostly bounded by the single per-token command-buffer
  wait, which implies the next real win should come from deeper decode fusion
  or fewer queued decode sub-operations rather than more host-side cleanup

## Metal — `apple-m5-max` (Apple M5 Max)

Apple M5 Max Metal is the main Apple-silicon target for Qwen3.6 bring-up. The
current benchmark lane is deliberately narrow: `qwen3.6-35b-a3b` with INT4
weights on the chained Metal decode path. This section is a performance harness
checkpoint, not a claim that the HIP feature set has been ported to Metal.
The latest checkpoint promotes the fused Qwen3.6 stage-5 linear-attention INT4
Metal path into the default lane, with `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5=1`
kept as the host fallback escape hatch. The default FFN lane remains the
host-orchestrated INT4 fallback, but routed expert gate/up and down work is now
batched across top-k experts to reduce per-layer thread orchestration. Native
FFN projection work remains explicit opt-in with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5=1`; profile runs no longer
switch the FFN implementation underneath the headline lane. A newer
gate/up-only tiled routed-expert kernel is also available as a diagnostic
microbench and explicit decode experiment via
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GATE_UP_TILED=1`, but it is not
promoted because the real decode path must still synchronize back to host for
expert down/finalize. The follow-up combined routed-expert gate/up +
down/finalize path is available behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5=1`; it keeps the
`expert_mid` workspace device-side, but remains diagnostic-only because the
real model profile points at command-buffer wait and bake-buffer residency
rather than raw shader arithmetic. The packed active-expert variant behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5=1` copies only routed
top-k expert slabs into compact scratch buffers before launching the same
combined shader. It is a residency attribution experiment, not a default
runtime path.

Reproduce the run with:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" cargo run --release -p supersonic-bench --bin bench-perf -- --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4
```

The local-main-target workflow for this machine is:

1. quick smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" cargo test --release -p runner --test qwen36_moe_metal_smoke -- --ignored --nocapture`
2. headline decode gate: the `bench-perf --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4` command above
3. long-context smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke`
4. profile smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke --metal-profile`
5. batched-prefill MoE feasibility: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke --batched-prefill-feasibility`
6. batched-prefill Metal prototype: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke --batched-prefill-prototype`
7. batched-prefill variant sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_batched_prefill_variants.py --metal-profile`
8. MTP tensor audit: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/audit_qwen36_mtp.py --require-complete-bake`
9. MTP acceptance/policy probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mtp_acceptance.py`
10. MTP Metal K=1 experiment: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mtp_acceptance.py --metal-experiment`
11. MTP Metal prompt-suite sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_mtp_acceptance.py --prompt-set smoke --metal-experiment`
12. static top-N resident-table probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_static_topn.py`
13. static top-N warm runtime sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_static_topn_runtime.py --modes default,static,static-hotset,mps-static-partial --metal-profile`
14. MPS resident-table viability probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mps_resident_table.py --run-pilot --require-pilot`
15. route residency sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_route_residency.py --prompt-set smoke`
16. fused routed INT4 sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_fused_routed_int4.py --prompt-set smoke --metal-profile`
17. SOTA gate refresh plan: `python3 tests/metal/refresh_qwen36_sota_gates.py --max-age-hours 24`
18. SOTA gate summary: `python3 tests/metal/summarize_qwen36_sota_gates.py --require --max-age-hours 24`
19. routed-expert FFN microbench: `target/release/qwen36_ffn_expert_microbench --iters 20 --warmup 3`
20. long-context comparison: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset comparison`

The Metal long-context harness writes `target/qwen36_metal_longctx.json` and
`target/qwen36_metal_longctx.md`. It uses deterministic NIAH-style prompts and
the supported `int4` Metal lane only, then reports generated-token sanity,
NIAH hit/miss, `stage_timings`, `chain_breakdown`, and `lifecycle_timings`.
When `--metal-profile` is set, the harness also records parsed `metal_profile`
and `hal_profile` summaries from the machine-readable profile lines. When
`--batched-prefill-feasibility` is set, the harness still forces the known-good
Metal per-token prefill path, but it enables route capture and records
`batched_prefill_feasibility` rows that summarize the grouped-MoE permutation
metadata a future Metal prefill kernel would consume: profiled tokens, chunks,
expert segments, average rows per touched expert, and WMMA16 assignment
coverage. It also emits `[qwen36-batched-prefill-plan]` rows for candidate
64/128/256/512/1024-token chunks, recording scalar tail assignments, WMMA16
padded assignments, and padding overhead. With `--batched-prefill-prototype`,
the harness sets
`SUPERSONIC_QWEN36_MOE_METAL_BATCHED_PREFILL_PROTOTYPE=1` and runs the
experimental Metal batched-prefill path: Metal batched full-attention plus a
direct routed-expert INT4 gate/up and down/combine kernel pair, with
router/top-k and shared-expert work still on the existing host/primitive path.
`--batched-prefill-variant` names the measured env-gated prototype probes
without requiring hand-built environment overrides: `linear-direct-off`,
`full-attn-tmajor`, `split-qgate`, `router-topk`, and `fused-residual`.
The harness records the selected variant and its env overrides per row so A/B
comparison outputs are self-describing. The long-context JSON schema is now
`qwen36-moe-metal-longctx-bench-v5` and
records `batched_prefill_prototype` at top level plus
`metal_batched_prefill_prototype` and `batched_prefill_variant` per row;
feasibility rows remain under `batched_prefill_plans`; set
`SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_PLAN_CHUNKS=...` to override the planner
chunk list without adding a SuperSonic CLI flag.
`tests/metal/sweep_qwen36_batched_prefill_variants.py` wraps those variant
knobs into a parity-preserving A/B harness. It runs the supported `baseline`
mode plus `prototype-default` and the named prototype variants against the same
deterministic NIAH prompt per context, then writes
`target/qwen36_metal_batched_prefill_variant_sweep.{json,md}` with generated ID
parity, prefill ratios versus baseline, lifecycle/stage rows, optional
Metal/HAL profiles, and the exact variant/env gate used by each row. Its v2
schema adds a nonfatal `promotion_gate` summary: a candidate must preserve
generated IDs, improve prefill, headline decode, and `ffn_ms_avg`, avoid more
than the configured full-attention/linear-attention/lm-head regression ratio,
and include non-regressed `command_buffer_wait` profile evidence unless
`--no-promotion-require-profile` is used.
The 512-token smoke is a real prefill run and is slow on the current chained
Metal path; use `--preset comparison` as a long-running sweep before selecting
the next runtime optimization target. The first v3 feasibility smoke on this
machine profiled 417 prefill tokens and showed the 512/1024-token plans tied at
one chunk with 82.7% WMMA16 assignment coverage, 23,048 scalar-tail assignments,
and 54.6% WMMA16 padding overhead. Chunk 64 fell to 37.3% WMMA16 coverage and
228.0% padding overhead, so the first Metal grouped-compute prototype should
start at the moderate/full-prompt chunk end, not tiny chunks.
The first opt-in prototype smoke on this machine used the 512-token preset and
generated the same `[271]` one-token sanity row. In normal mode it measured
22.15s prefill, 268.55 ms/token decode, and no NIAH hit because only one token
was requested. The profiled variant measured 34.20s prefill and 253.74
ms/token decode; profile overhead is expected because
`SUPERSONIC_METAL_PROFILE=1` splits the routed-expert phases. The profile rows
showed `qwen36_batched_prefill_grouped_expert_direct` at 5.024s native wall
across 40 layers, GPU timestamps of 1.903s for
`command_buffer_gpu:qwen36_batched_prefill_grouped_expert_gate_up` and 2.045s
for `command_buffer_gpu:qwen36_batched_prefill_grouped_expert_down_combine`,
while `command_buffer_wait` was still the top Metal row at 33.184s and HAL
`copy_h2d` accounted for 4.335s. On Apple UMA this is buffer
materialization/copy bookkeeping rather than PCIe upload.

The next measured slice removes the shared-expert scalar-gate host broadcast in
the Metal batched-prefill prototype. A native BF16 row-scalar sigmoid multiply
keeps the `[N, 1]` gate on Metal and writes the `[N, hidden]` shared output
directly. The follow-up 512-token normal prototype smoke measured 12.47s
prefill and 172.54 ms/token with the same `[271]` generated-token sanity row.
The profiled run still carries heavy attribution overhead, but the new
`sigmoid_mul_row_scalar` row itself is small: 40 calls, 11.859 ms native wall,
and 0.725 ms GPU timestamp total. It also removes 40 D2H scalar-gate reads, 40
expanded-gate H2D writes, 40 transient temp allocations, and about 68 MB of D2D
traffic from the prior prototype profile. The remaining top rows are
`command_buffer_wait`, `qwen36_linear_int4_stage5`,
`qwen36_batched_prefill_grouped_expert_direct`, and
`full_attention_prefill_strided`; HAL `copy_h2d` is still dominated by
materializing the baked model buffers. That makes the next measured target
prefill orchestration, linear-attention command-buffer volume, full-attention
prefill, and routed-expert direct work, not per-token MPS slab materialization.

The next orchestration slice targets `qwen36_linear_int4_stage5`, which had
12,540 waited calls in the profiled 512-token run. The normal Metal prototype
now keeps the native stage-5 temporary output separate from the final residual
destination, opens one Metal batch per linear-attention layer, and writes final
rows directly into `chunk_hidden`; this removes the per-token CPU D2D row copy
and collapses the waited linear submits for normal runs. Attribution runs keep
the old waited path when `SUPERSONIC_METAL_PROFILE=1` so the per-phase profile
rows stay comparable. On the 512-token smoke, the direct-off control
(`SUPERSONIC_QWEN36_MOE_METAL_LINEAR_PREFILL_DIRECT=0`) measured 13.30s
prefill and 191.99 ms/token with `[271]`; the direct-row batch measured 10.89s
prefill and 179.14 ms/token with the same generated ID. The next measured
targets remain full-attention prefill and routed-expert direct compute, while a
proper batched linear-attention kernel would be the larger follow-up.

The full-attention follow-up keeps only allocation reuse on by default. The
batched prefill path now reuses Q-after-norm, K-after-norm, and KV-prefix
scratch buffers instead of allocating them inside every full-attention layer.
Two apparent layout fixes were measured but left opt-in because they did not
beat the default path on the 512-token Metal smoke: direct time-major KV
attention via `SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_TMAJOR=1` measured 11.09s
prefill, and native Q/gate splitting via
`SUPERSONIC_QWEN36_MOE_METAL_SPLIT_QGATE=1` measured 11.00s prefill. With both
probes disabled, the default path measured 10.73s prefill and generated the
same `[271]` sanity row. The measured next bottleneck remains routed expert
compute/residency, not these full-attention layout probes.

Two routed-FFN micro-orchestration probes are also measured negative and remain
opt-in only. `SUPERSONIC_QWEN36_MOE_METAL_FUSED_FFN_RESIDUAL=1` fuses
`chunk_hidden += combined` and `chunk_hidden += shared_out` into one Metal
kernel while preserving the two BF16 rounding points, but measured 11.03s
prefill versus a 10.65s disabled-path control. `SUPERSONIC_QWEN36_MOE_METAL_ROUTER_TOPK=1`
is the matching router probe: it runs router softmax/top-k on Metal and batches
it with routed expert direct in normal runs. It preserves the `[271]` sanity
row, but measured 11.63s prefill versus an 11.05s host-top-k control. The
default lane keeps both off. The next FFN work should target routed expert
compute/residency, not standalone router top-k or residual-add reshuffling.
`tests/metal/audit_qwen36_mtp.py` is the speculative-decode readiness audit. It
checks the source snapshot for the split MTP expert tensors and the INT4 bake
for the 19 folded `mtp.*` tensors loaded by the runtime. On this local M5 Max
cache, the audit reports `source=complete` with 1,560 `mtp.*` tensors and
`bake=complete` with all 19 runtime MTP tensors, so the model files are ready
for the MTP acceptance gate. The runner now emits a machine-readable
`[qwen36-mtp-acceptance]` row when
`SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE=1` or `--emit-stage-timings` is used
with speculative decode. The row records drafted tokens, accepted tokens,
acceptance rate, emitted tokens, base verify steps, batched replay steps, and
target steps per emitted token. `tests/metal/probe_qwen36_mtp_acceptance.py`
captures that telemetry on enabled backends and records the expected
`policy_blocked` result on Metal today. Metal speculative decode remains
unsupported by default. For measurement only, `--metal-experiment` sets
`SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT=1` and runs the sequential K=1 Metal
path. The probe forces `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0` on Metal so
the experiment does not accidentally enter the HIP/CUDA-only grouped prefill
launcher. Batched verify and K>1 promotion remain blocked until that row
reports real acceptance without mixing the result into FFN latency. The first
local K=1 smoke completed in 24.3s with `drafted_tokens=2`,
`accepted_tokens=1`, `acceptance_rate=0.5`, and
`target_steps_per_emitted=1.0`; this proves the Metal path can run and measure
acceptance, but it is not a throughput win by itself because every emitted token
still required one target-model step. `tests/metal/sweep_qwen36_mtp_acceptance.py`
extends that one-prompt result across a smoke or comparison prompt suite and
writes aggregate `drafted_tokens`, `accepted_tokens`, `acceptance_rate`, and
`target_steps_per_emitted` rows to
`target/qwen36_mtp_acceptance_sweep.{json,md}`. The sweep now reports a
machine-readable `promotion_gate` using aggregate acceptance and target-model
steps per emitted token, and `--metal-profile` preserves parsed Metal/HAL
attribution rows for each prompt. The first smoke sweep completed both rows in
34.7s: the profiling prompt accepted 0/2 drafts, the coding prompt accepted
1/2 drafts, aggregate acceptance was 25.0%, and aggregate
`target_steps_per_emitted` remained 1.0. This keeps K=1 as an instrumentation
path, not a supported-speed path.
`tests/metal/probe_qwen36_static_topn.py` is the first static resident-table
probe. The runner now has gated machine-readable route dumps:
`SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_TOPN_LAYERS=1` emits
`[qwen36-route-topn-layer]` rows with per-layer expert IDs and counts, while
`SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_CALLS=1` emits `[qwen36-route-call]`
rows for real active expert sets. The probe uses those rows to build static
top-N sets from a calibration prompt, evaluate them against a separate
coding-shaped prompt, export a `static_tables` JSON object for runtime probes,
and size both native INT4 resident tables and resident FP16 MPS RHS tables. For
Qwen3.6 geometry, each resident expert costs roughly 1.50 MiB as native INT4
packed weights plus GPTQ sidecars, or 6 MiB of FP16 MPS RHS data (gate/up plus
down). Capacities 2/4/8/16 across 40 layers imply about
0.12/0.23/0.47/0.94 GiB for native INT4 residency and
0.47/0.94/1.88/3.75 GiB for FP16 RHS residency before h_norm/output scratch or
miss fallback.
The packed native INT4 static-table runtime probe is opt-in on top of the
packed stage-5 path:
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5=1`,
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN=1`,
`SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE=target/qwen36_static_topn_mps_probe.json`,
and optionally
`SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY=<N>`. It uses a
per-layer resident native INT4 table on full hits and falls back to the
existing packed/hotset path when any active expert is missing.
The first local two-prompt smoke used a profiling/agentic calibration prompt
and a coding-shaped evaluation prompt at context 256. It collected 160
calibration top-N rows and 880 evaluation route calls. Assignment coverage on
the evaluation prompt was 9.1%/14.9%/23.1%/35.8% for capacities 2/4/8/16, and
capacity 16 fully covered only 0.5% of layer calls, leaving 876/880 calls on the
miss fallback. The result is a useful negative gate: a small static resident MPS
table is unlikely to beat the default path unless the fallback is very cheap or
the table is prompt/domain-specialized.

Current 512-token `--metal-profile` smoke checkpoint on this M5 Max after the
FFN fallback tightening and profile parser work:

| Context | total ms/tok | tok/s | prefill s | lm-head ms | full-attn ms | linear-attn ms | FFN ms | likely row bottleneck | top Metal op |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 512 | 269.24 | 3.71 | 71.71 | 97.92 | 33.60 | 73.31 | 63.96 | lm-head | `qwen36_ffn_int4_stage5` |

The original 512-token smoke measured roughly 2598.56 ms/token with
prefill_total_ms=1001228.792 and FFN as the top per-token stage. The first
row-parallel FFN pass cut prefill to roughly 553 seconds and moved the measured
bottleneck to linear-attention; follow-up linear-attention and full-attention
projection passes cut prefill to roughly 160 seconds. The current profiled
smoke cuts prefill to roughly 72 seconds, records a clear NIAH miss row, and
shows two distinct bottlenecks: the one-token row is lm-head/tail dominated,
while Metal profile totals are still dominated by the FFN stage and
`command_buffer_wait`.

Capture the native Metal hardware profile before starting runtime optimization:

```bash
SUPERSONIC_BACKENDS=metal cargo run -p machine-profile --bin machine-profile -- show --raw
```

The Metal profile records the Apple GPU device name, Metal capability metadata,
unified-memory bandwidth, threadgroup-memory bandwidth, F16
`simdgroup_multiply_accumulate` throughput rows split by accumulator dtype, a
small sweep over simdgroups-per-threadgroup/threadgroups-per-core/accumulator
count, an MPS FP16 GEMM roofline row, and Qwen3.6-shaped INT4 GEMV
microkernels. On the current M5 Max machine the profiler identifies the GPU as
`apple-m5-max` with 40 GPU cores from `system_profiler`; published profiles must
remain sanitized and must not include serial numbers, hardware UUIDs, display
serials, or provisioning identifiers. Treat the MPS row as the vendor-library
hardware ceiling and the explicit simdgroup rows as SuperSonic kernel-design
probes. The profile also includes a public-MSL tiled simdgroup GEMM row at
`2048^3`, a same-size MPS row, and guarded MPP + Metal Tensor matmul rows. The
MPP rows are the only rows intended to represent the supported M5 Neural
Accelerator path; they remain `null` unless the runtime can compile the MPP
shader, bind `MTLTensor` arguments through Metal 4 argument tables, complete the
dispatch, and read back a nonzero tensor result. The support probe is a single
full `64x64 * 64x32 -> 64x32` MPP tile; the large MPP rows are equivalent-GEMM
throughput measurements built from repeated exact `64x32x64` MPP tiles and are
independently guarded by their own output readback. They are not yet a claim
that one whole square `MTLTensor` matmul invocation is working. The raw profile
also records `mpp_tensor_matmul_probe_status`, `mpp_tensor_write_probe_value`,
and `mpp_tensor_matmul_probe_value` so a failed MPP row distinguishes tensor
binding from the MPP operation itself. If the isolated MMA sweep stays flat and
the public tiled GEMM remains far below same-size MPS, the next
optimization step is MPP/Metal-Tensor or MLX interop for large dense phases, not
another single occupancy tweak.

The harness records a warmup plus median-of-3 headline run at
`--max-new-tokens 16`, then performs one additional `--emit-stage-timings`
attribution run. For this Metal lane it also forces the dense prefill token loop
(`SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0` and
`SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP=1`) because the default Qwen3.6
batched prefill router-permute path is HIP/CUDA-only. The attribution run also
enables `SUPERSONIC_METAL_QWEN36_MPP_PILOT=1`, which emits a separate
`[qwen36-moe mpp-pilot]` row for repeated exact `64x32x64` MPP tiles. It also
enables `SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT=1`, which emits a resident
FP16 Metal Performance Shaders row for Qwen3.6 active-expert GEMV shapes. It
also enables `SUPERSONIC_METAL_PROFILE=1` for the attribution run so the profile
JSON includes `metal_profile` and `hal_profile` objects with parseable per-op
rows. These are runtime-adjacent MPP/MPS pilot measurements, not model matmul
replacements. The attribution maps are stored in the same schema-v7 perf JSON as
`stage_timings`, `chain_breakdown`, `lifecycle_timings`, `mpp_pilot`,
`mps_expert_pilot`, `metal_profile`, and `hal_profile` without feeding back into
the headline median. Schema v7 also preserves typed Qwen3.6 expert-residency
policy rows with `resident_format`, `scope`, `miss_policy`, `capacity`, and the
numeric counters so perf artifacts retain the scheduler identity.

<!-- AUTOGEN BELOW: apple-m5-max-metal -->
| Model           |  INT4 |
| --------------- | ----: |
| qwen3.6-35b-a3b | 150.6 |

<!-- AUTOGEN END: apple-m5-max-metal -->

Latest local attribution run
(`target/bench-runs/2026-05-23-796ea84/perf/qwen3.6-35b-a3b_int4.json`):

| Metric | Value |
|---|---:|
| Headline median | 150.6 ms/token |
| Stage total | 173.014 ms/token |
| Chain | 167.612 ms/token |
| LM head | 4.342 ms/token |
| FFN | 96.761 ms/token |
| Linear attention | 54.181 ms/token |
| Full attention | 16.470 ms/token |
| Prefill total | 1070.741 ms |
| MPP pilot | 15.230 TFLOP/s |
| MPS expert pilot gate/up | 0.619 ms, 5.423 TFLOP/s |
| MPS expert pilot down | 0.433 ms, 3.878 TFLOP/s |

The headline median is the unprofiled median-of-3 run (`149.2`, `150.6`,
`153.5` ms/token). The stage table comes from the extra profile attribution run
and carries profiling overhead, but follows the same default FFN path as the
headline samples. A default one-token smoke generated the expected token `[11]`;
the latest local cold one-token profile measured `ffn_ms_avg=128.573`,
with `qwen36_ffn_host_expert_gate_up` at 56.839 ms total and
`qwen36_ffn_host_expert_down` at 36.181 ms total. The explicit full native FFN
escape hatch also generated `[11]`, but remains too slow to promote
(`ffn_ms_avg=640.553` in the same one-token smoke shape).

The focused routed-expert microbench exercises the exact Qwen3.6 stage-5 INT4
shape (`hidden=2048`, `num_experts=256`, `moe_intermediate=512`, `top_k=8`,
`group_size=128`) without the rest of decode:

```bash
cargo build --release -p runner --bin qwen36_ffn_expert_microbench
target/release/qwen36_ffn_expert_microbench --iters 20 --warmup 3
```

The binary reports gate/up-only, the original combined tiled stage-5 path, the
direct-gather fused routed INT4 path, and the GPU-pack fused routed INT4 path.
All four rows use the same synthetic Qwen3.6 stage-5 geometry and validate
against the CPU oracle, so the GPU-pack row is the apples-to-apples microbench
for the route-sweep fallback when static residency misses too often.

On this M5 Max, the four-row Metal validation run reports `mean_ms=0.5351` for
gate/up, `0.4984` for the original combined stage-5 path, `0.3369` for
direct-gather fused stage-5, and `0.5331` for GPU-pack fused stage-5; every row
has `mismatches=0`. Wired into decode with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GATE_UP_TILED=1`, the same kernel
still generates `[11]`, and the Metal profile shows only 8.418 ms total GPU
time for `command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled` across
40 layers. The wall-time op row is much worse, however:
`qwen36_ffn_int4_expert_gate_up_tiled` records 277.396 ms total because each
layer must wait before the host expert-down path reads `expert_mid`. That makes
the gate/up-only decode experiment a diagnostic step, not the default path.

The combined opt-in path with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5=1` also generates
`[11]`, but it is not promotable. An unprofiled one-token smoke measured
`ffn_ms_avg=1276.670`; the profiled one-token smoke measured
`ffn_ms_avg=1458.677`, with
`qwen36_ffn_int4_expert_gate_up_down_finalize_tiled` at 1414.646 ms wall time
across 40 layers, but only 19.760 ms total GPU time for
`command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_down_finalize_tiled`.
`command_buffer_wait` accounts for 1561.827 ms total in the same run. The
combined microbench proves the arithmetic path is cheap on synthetic resident
buffers; the real decode path is now pointing at command-buffer waits plus
large GPTQ bake-buffer residency/page movement.

The packed active-expert path is the first residency experiment on the real
model. With `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5=1`, the
unprofiled one-token smoke still generates `[11]` and improves the pathological
combined path to `337.4 ms/token` with `ffn_ms_avg=177.706`. The profiled run
measures `376.1 ms/token`, `ffn_ms_avg=184.798`, `100.287 ms` total in
`qwen36_ffn_int4_expert_pack_stage5`, `43.556 ms` total in
`qwen36_ffn_int4_expert_packed_stage5`, and `21.075 ms` total GPU time for the
combined shader. `command_buffer_wait` drops from the pathological
`1561.827 ms` to `182.291 ms`, so packing confirms that the giant expert
buffers caused most of the previous wall time. It is still slower than the
default FFN lane because the per-token CPU pack copies active expert slabs from
every layer. The next runtime optimization should therefore focus on persistent
hot-expert packing, prefetch/residency reuse across tokens, or an MPS/MPP-backed
expert matvec bridge before any routed-expert FFN path is made default.

A follow-up reuse-cache probe is intentionally opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_CACHE=1` while the packed expert
path is enabled. On a four-token profiled smoke, the cache preserved the same
tokens `[11, 353, 599, 264]` and reduced HAL alloc/free churn from 2431/2431
calls to 1711/1471 calls, but it did not improve latency: the cached run
measured `234.7 ms/token`, `ffn_ms_avg=128.730`, and `280.597 ms` total in
`qwen36_ffn_int4_expert_pack_stage5`; the no-cache control measured
`217.6 ms/token`, `ffn_ms_avg=126.803`, and `248.724 ms` in the same pack bucket.
That rules out a simple per-layer scratch-slab cache as the next promotion
target. The next FFN work should either keep packed experts resident without
recopying on route churn, or move the expert matvecs to a Metal Performance
Shaders / MPP bridge that avoids the CPU slab-pack step entirely.
Profile runs now emit `[qwen36-expert-residency]` plus one
`[qwen36-expert-residency-policy]` row per resident expert policy. The legacy
`[qwen36-pack-cache]` line is still emitted for older parsers. A four-token
Apple M5 Max profile with the packed expert path and exact-route pack cache
enabled generated `[11, 353, 599, 264]` and measured
`279.5 ms/token`, `ffn_ms_avg=162.420`, and
`qwen36_ffn_int4_expert_pack_stage5=384.124 ms` across 160 layer calls. The
cache profile reported `calls=160`, `entries=40`, `exact_hits=0`,
`route_refills=120`, `allocations=40`, and `copied_bytes=2014248960`, with
`avg_copy_bytes=12589056` per refill/allocation. That confirms the scratch cache
is saving allocation churn but not slab-copy churn: every post-allocation layer
call saw a different active-expert set. The next packed-path experiment needs a
larger resident hot set or a different addressing scheme; an exact-route
per-layer cache is not worth promoting.

The resident hot-set follow-up is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET=1`; capacity defaults to
16 and can be set with
`SUPERSONIC_METAL_QWEN36_FFN_EXPERT_HOTSET_CAPACITY`. It reuses the same packed
kernel but maps each top-k expert to a resident slot, so only slot misses are
recopied. The 16-slot Apple M5 Max profile preserved `[11, 353, 599, 264]` and
cut copied bytes to `1356470784`, with `slot_hits=418`, `slot_misses=862`,
`slot_hit_rate=0.326562`, and `evictions=222`, but measured `298.0 ms/token`
and `ffn_ms_avg=161.142`. A 32-slot run avoided evictions and improved host pack
time to `337.429 ms`, but copied bytes barely changed (`1345455360`) and wall
time worsened to `304.2 ms/token`, with
`qwen36_ffn_int4_expert_packed_hotset_stage5=185.272 ms`. That rules out a
straight LRU hotset as the next promotion path.

The static top-N resident follow-up is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN=1` while the packed
expert path is enabled. It loads `static_tables` from
`SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE`, chooses the largest
exported capacity unless
`SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY` is set, and fills a
per-layer native INT4 resident table once. Full-hit calls remap top-k experts
to resident slots and run `qwen36_ffn_int4_expert_packed_static_topn_stage5`;
misses record `miss_policy=static_topn` in the expert-residency profile and
fall through to the existing packed/hotset/default path.
The first Apple M5 Max one-token smoke used the regenerated v2 probe table.
The probe's capacity-64 row covered 69.858% of evaluation assignments, and the
runtime smoke preserved generation parity with `[11]`. It is not a latency win
on the cold first token: `decode_ms=507`, `ffn_ms_avg=369.071`, and the
residency profile reported `exact_hits=9/40`, `slot_hit_rate=0.731250`, and
`copied_bytes=879660288` for static-table allocations. That validates the
runtime wiring and miss fallback, but leaves promotion blocked on warm
multi-token reuse and/or a cheaper hybrid fallback for the 31/40 non-full-hit
layer calls.
`tests/metal/sweep_qwen36_static_topn_runtime.py` is the follow-up warm-token
comparison harness for that question. It runs separate process rows for modes
such as `default`, `static`, and `static-hotset`, keeps generated IDs as the
per-prompt parity key, and records stage timings, chain breakdown, lifecycle
timings, expert-residency totals, and per-policy rows in
`target/qwen36_static_topn_runtime_sweep.{json,md}`. With `--metal-profile`
it also preserves parsed `metal_profile` and `hal_profile` objects per row and
renders the top Metal/HAL attribution in Markdown, which makes the comparison
prompt set usable as a profiling gate rather than a Hello-only smoke. The v3
schema adds a nonfatal `promotion_gate`: a resident mode must preserve
generated IDs versus `default`, improve headline ms/token and `ffn_ms_avg`,
keep full-attention, linear-attention, and lm-head inside the configured
regression ratio, and include non-regressed `command_buffer_wait` profile
evidence unless `--no-promotion-require-profile` is used.
The first four-token smoke preserved `[11, 353, 599, 264]` across all three
modes but ruled out promotion: default measured `decode_ms=702` and
`ffn_ms_avg=98.761`, static measured `decode_ms=951`, `ffn_ms_avg=177.563`,
`exact_hits=10/160`, `slot_hit_rate=0.508594`, and `copied_bytes=980372736`,
and static+hotset measured `decode_ms=1450`, `ffn_ms_avg=262.215`, and
`copied_bytes=2234557440`. The native static table is useful as a measured
residency scaffold, but Qwen3.6 Metal should next target either a dense
resident MPS/MPP table that can serve partial hits cheaply, or return to the
prefill/orchestration buckets already shown to dominate long-context runs.

`tests/metal/probe_qwen36_mps_resident_table.py` turns that fork into an
explicit gate. It consumes `target/qwen36_static_topn_mps_probe.json`, optionally
runs the existing `[qwen36-moe mps-expert-pilot]` row, and writes
`target/qwen36_mps_resident_table_probe.{json,md}` with all-resident MPS,
full-hit-only, and optimistic partial-hit estimates. The v2 report adds a
nonfatal `viability_gate` with resident-RHS size, projected speedup, assignment
coverage, and full-hit-rate thresholds; a passing partial-hit row is a reason
to prototype only if the runtime can avoid per-token FP16 RHS rebuilds, not a
default-promotion signal. The first direct
`--run-pilot` smoke measured `gate_up_ms=1.312` and `down_ms=0.757`, giving an
all-resident FP16 MPS floor of 82.76 ms/token versus the 98.761 ms/token
default FFN baseline. Capacity 64 costs 15.00 GiB of FP16 MPS RHS storage,
covers 69.9% of routed assignments, but fully serves only 9.8% of layer calls.
The full-hit-only estimate is therefore only 97.20 ms/token, while the
deliberately optimistic partial-hit estimate is 87.58 ms/token. That keeps a
dense resident MPS path interesting only if it can serve resident hits and miss
fallbacks inside the same layer without rebuilding per-token FP16 slabs; a
full-hit-only bridge is not enough.
`tests/metal/sweep_qwen36_route_residency.py` is the prompt-suite version of
the route-locality rows. It runs the default Metal lane with
`SUPERSONIC_QWEN36_ROUTE_PROFILE=1`, aggregates `[qwen36-route-profile]`,
`[qwen36-route-cache-sim]`, and `[qwen36-route-topn]` rows across prompts, and
writes `target/qwen36_route_residency_sweep.{json,md}`. Its v1
`decision_gate` compares LRU hot-set hit rate with oracle static top-N coverage
so the next residency fork can be selected from measured route evidence before
more slab-cache or fused-INT4 work begins.
`tests/metal/summarize_qwen36_sota_gates.py` is the aggregation step after the
individual sweeps. It reads the batched-prefill variant sweep, static top-N
runtime sweep, fused routed INT4 runtime sweep, MPS resident-table probe, route
residency sweep, and MTP acceptance sweep JSON reports, then writes
`target/qwen36_sota_gate_summary.{json,md}` with input status, report age,
passed/failed gate IDs, candidate failures, refresh commands, and the next
action. Missing reports are preserved as rows by default; use
`--require --max-age-hours 24` when a local validation run should fail closed on
absent, malformed, schema-mismatched, stale, or missing-gate artifacts.
The v5 summary also records `superseded_gates`; this prevents an older
estimate-only pass from taking `next_action` after the corresponding runtime
candidate has already been measured and rejected.
`tests/metal/refresh_qwen36_sota_gates.py` is the operational companion for
that summary: by default it writes
`target/qwen36_sota_gate_refresh_plan.{json,md}` with only the missing, stale,
malformed, schema-mismatched, or missing-gate rows selected; add `--run` to
execute those trusted local refresh commands in order, or `--only <gate_id>` to
force-refresh one gate even when its current report is already OK.

The first partial-hit resident MPS runtime prototype is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_STATIC_TOPN_PARTIAL=1` plus the
same static-table env used by native INT4 static top-N. It materializes a
per-layer FP16 MPS RHS table once from the static top-N experts, remaps only the
resident-hit routed groups to table slots, runs an indexed MPS bridge for those
hits, then computes miss groups on the existing host INT4 path and combines
both contributions before the residual write. Profile rows use
`resident_format=fp16_mps`, `miss_policy=static_topn`, and stable op names
`qwen36_ffn_int4_expert_mps_static_topn_pack_f16_lut`,
`qwen36_ffn_int4_expert_mps_static_topn_partial_f16`, and
`qwen36_ffn_host_expert_mps_static_topn_miss_*`. The warm sweep mode
`mps-static-partial` compares this path against `default`, `static`, and
`static-hotset` with the same generated-ID parity key. This is still a
diagnostic path, not a promoted default.

The first measured result is negative. A profiled one-token smoke preserved the
generated id `[11]`, but reported `decode_ms=6324`, `ffn_ms_avg=6073.323`,
`slot_hit_rate=0.731250`, and `copied_bytes=15753805824`. The native indexed
MPS bridge itself was `365.052 ms` across 40 layer calls, while
`qwen36_ffn_int4_expert_mps_static_topn_pack_f16_lut` took `5630.663 ms` on
the host and HAL `copy_h2d` accounted for `4481.851 ms` / `17886298368` bytes.
The warm four-token sweep preserved `[11, 353, 599, 264]`, but measured
`default` at `decode_ms=702`, `ffn_ms_avg=94.930` versus
`mps-static-partial` at `decode_ms=7839`, `ffn_ms_avg=1845.066`,
`slot_hit_rate=0.507812`, and `copied_gib=14.672`. This confirms the prototype
as a correctness/profiling harness only: the RHS materialization and
MPS/host-split overhead swamp the resident-hit matmuls.

The GPU-side active-slab pack probe is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5=1` on top of the
packed expert path. It allocates compact per-layer scratch once, copies the
current top-k expert slabs from the original baked Metal buffers in the FFN
command buffer, remaps `topk_idx` to compact group IDs, then runs the existing
packed gate/up and down/finalize shader. The four-token Apple M5 Max smoke
preserved `[11, 353, 599, 264]`, so the remap and packed shader parity are good,
but it measured `777.3 ms/token`, `ffn_ms_avg=641.772`, and
`qwen36_ffn_int4_expert_gpu_pack_stage5=2417.156 ms` across 160 layer calls.
The command-buffer GPU attribution for the fused pack+expert shader was only
`64.400 ms`, while `command_buffer_wait` was `2678.881 ms`; moving slab
materialization from CPU to GPU therefore did not solve the residency/wait
problem.

The direct-gather follow-up is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5=1`. It keeps
the original top-k expert IDs, reads the baked expert buffers directly, and uses
a 256-thread tiled down/finalize kernel so the down projection has the same
wide reduction shape as gate/up. It also preserved `[11, 353, 599, 264]`, but
the unprofiled four-token smoke measured `308.8 ms/token` with
`ffn_ms_avg=249.990`; the profiled run measured `756.8 ms/token`,
`ffn_ms_avg=616.225`, and
`qwen36_ffn_int4_expert_direct_gather_stage5=2318.450 ms` across 160 layer
calls, while the command-buffer GPU attribution for the direct gather command
was only `55.965 ms`. That confirms the direct original-buffer gather is still
wait/residency dominated on this model. The useful next FFN direction is an
explicit resident representation that avoids per-token active-slab rebuilds and
avoids random giant-buffer gathers. The native INT4 static top-N probe now
covers the narrow static-table branch; MPS/MPP remains the next heavier
resident-matvec option if static full-hit rates are not high enough.

The fused routed INT4 variants are now covered by a promotion-gated runtime
sweep rather than only one-off smoke notes:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_fused_routed_int4.py \
    --prompt-set smoke --metal-profile
```

The sweep compares `default`, `direct-gather`, and `gpu-pack` under the same
prompt and generated-token parity key, records Metal/HAL profile rows when
requested, and writes `target/qwen36_fused_routed_int4_sweep.{json,md}`. Its
nonfatal `promotion_gate` requires generated IDs to match default, headline
decode and `ffn_ms_avg` to improve, full-attention/linear-attention/lm-head not
to regress beyond the configured threshold, and `command_buffer_wait` evidence
to be present and non-regressed. The SOTA summary consumes this report as the
machine-readable gate for the fused routed INT4 fork. The first one-token
profiled smoke preserved `[11]` for all modes and rejected both candidates:
`default` measured `decode_ms=406` / `ffn_ms_avg=190.904`, `direct-gather`
measured `806` / `661.302`, and `gpu-pack` measured `863` / `693.948`, with
both fused candidates failing headline, FFN, and command-buffer-wait gates.

The first MPS bridge step is now an attribution probe, not a decode path. With
`SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT=1`, the runner appends a
`[qwen36-moe mps-expert-pilot]` row and bench perf JSON schema v7 records it as
`mps_expert_pilot`. This probe uses resident FP16 MPSMatrix inputs shaped like
the active-expert gate/up and down GEMVs; it does not consume the GPTQ INT4
expert tensors. On a one-token M5 Max smoke, the model still generated `[11]`
and the probe measured `gate_up_ms=3.260`, `down_ms=2.975`,
`gate_up_tflops=1.029`, and `down_tflops=0.564` for 100 repeated GEMMs. In the
full `bench-perf` attribution run, the same resident-shape pilot measured
`gate_up_ms=0.619` and `down_ms=0.433`; the default INT4 host expert path
reported `qwen36_ffn_host_expert_gate_up=963.798 ms` and
`qwen36_ffn_host_expert_down=508.565 ms` across the profiled prefill+decode
calls.

The first real MPS bridge is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_BRIDGE=1`. It uses the real
active GPTQ experts, transposes/dequantizes those slabs to FP16 MPS layout on
the CPU by default only when
`SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE=1`; otherwise it uses a
GPU-side INT4-to-FP16 transcode with a 16-entry threadgroup LUT per GPTQ
scale/zero group before MPSMatrix gate/up and down consume the FP16 slabs. The
original CPU-pack profile generated `[11]`, but measured `1707.3 ms/token` with
`ffn_ms_avg=1502.344`. Profile attribution named the first blocker:
`qwen36_ffn_int4_expert_mps_bridge_pack_f16=1365.389 ms` across 40 layers,
while `command_buffer_gpu:qwen36_ffn_int4_expert_mps_bridge_f16=34.032 ms`.
The GPU LUT transcode path is correct and the normal async smoke improved
slightly to `1683.6 ms/token`, with `ffn_ms_avg=1473.939` and generated token
`[11]`. It is still not promotable: the profiled GPU-transcode run measured
`1766.4 ms/token`, `ffn_ms_avg=1538.500`,
`qwen36_ffn_int4_expert_mps_transcode_int4_f16=1404.914 ms` wall time, and
`command_buffer_gpu:qwen36_ffn_int4_expert_mps_transcode_int4_f16=66.239 ms`
across 40 layers. That rules out per-token FP16 MPS slab materialization as the
mainline route. A tiled packed-byte CPU LUT pack is available for investigation
with `SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_LUT=1`. Its release
microbench improves the 50.4 MB active-slab pack from `44.725 ms` to
`16.602 ms` by mapping each packed INT4 byte to a pair of FP16 values and
transposing through cache-sized tiles. The LUT bridge now materializes directly
into Metal shared buffers, avoiding the prior intermediate CPU slab plus
`MTLBuffer` memcpy. The optional
`SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_STREAM=1` mode uses paired
non-temporal ARM stores for the one-way transposed FP16 flush. The real bridge
generated `[11]` and improved the old LUT result substantially, but it still is
not promotable: unprofiled decode measured `868.1 ms/token`,
`ffn_ms_avg=688.849`, and the profiled stream-store run measured
`907.5 ms/token`, `ffn_ms_avg=695.173`, and
`qwen36_ffn_int4_expert_mps_bridge_pack_f16_lut=556.425 ms` across 40 layers.
On Apple UMA this is not PCIe upload cost; the remaining blocker is per-token
FP16 MPS slab rebuild/consumption. The next FFN experiment should either keep
active FP16 experts resident across route reuse, or return to a fully fused
routed-expert INT4 path that avoids MPSMatrix RHS rebuilds entirely. Profile
runs now emit Qwen3.6 route-locality lines to decide between those paths:
`[qwen36-route-profile]` reports adjacent-token same-layer reuse. By default,
`[qwen36-route-cache-sim]` simulates per-layer LRU resident-slab budgets of
2/4/8/16/32/64 experts, and `[qwen36-route-topn]` reports oracle top-N coverage
for the same budgets; override with
`SUPERSONIC_QWEN36_ROUTE_PROFILE_CAPACITIES`. A 4-token Apple M5 Max profile generated `[11, 353, 599,
264]` and measured `adjacent_hit_rate=0.400000`; the per-layer LRU hit rates
were 0.5%/2.5%/23.0%/32.7% for capacities 2/4/8/16, while oracle top-N
coverage was 19.2%/34.4%/57.6%/83.2%. That is enough to reject a tiny LRU
resident cache as the next immediate optimization, but it leaves a larger
hot-set cache or fused routed INT4 path as the next measured fork.
`SUPERSONIC_QWEN36_ROUTE_PROFILE_LAYERS` defaults to `40` for
Qwen3.6-35B-A3B and can be overridden for smaller parity runs.
`tests/metal/sweep_qwen36_route_residency.py` preserves those rows as a
prompt-suite report with a nonfatal `decision_gate`: if LRU hit-rate clears the
configured threshold, the next branch is a larger resident cache; if only
oracle top-N coverage clears, the next branch is a static resident table; if
neither clears, the report recommends a fused routed INT4 path over additional
slab-residency experiments.

Unsupported Metal constraints remain explicit for this target: persistent
decode, KV-FP8, speculative decode, batching, and Metal VMM are not benchmarked
or claimed here yet. Because the Qwen3.6 hero lane uses GPTQ-packed INT4
weights and the public MPP tile currently consumes FP16 `MTLTensor` inputs, MPP
stays attribution-only until an INT4-compatible packing/dequant bridge is
measured.

## How to reproduce

```bash
# HIP / gfx1100 — full quant matrix sweep
cargo build --release --bin supersonic
MODEL_DIR_08B=/path/to/Qwen3.5-0.8B \
MODEL_DIR_2B=/path/to/Qwen3.5-2B \
MODEL_DIR_4B=/path/to/Qwen3.5-4B \
MODEL_DIR_9B=/path/to/Qwen3.5-9B \
MODEL_DIR_GEMMA_E2B=/path/to/gemma-4-E2B \
MODEL_DIR_GEMMA_E4B=/path/to/gemma-4-E4B \
MODEL_DIR_PHI4=/path/to/Phi-4-mini-instruct \
  tests/gfx1100/bench_matrix.sh

# HIP / gfx1150
cargo build --release --bin supersonic
./target/release/supersonic --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "The quick brown fox jumps over" \
  --max-new-tokens 16

# Add --int4 / --fp8-runtime / --kv-fp8 as supported per the matrix in README.md.

# CUDA / sm86
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench.sh \
  /path/to/Qwen3.5-0.8B /path/to/Qwen3.5-4B

# CUDA / sm86 warmed 4B single-sequence hero lane
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_single.sh \
  /path/to/Qwen3.5-4B

# CUDA / sm86 Llama 3.1 8B INT8 single-sequence lane
SUPERSONIC_BACKENDS=cuda ./target/release/supersonic \
  --backend cuda \
  --model llama3.1-8b \
  --model-dir /path/to/Meta-Llama-3.1-8B \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --int8

# CUDA / sm86 Llama 3.1 8B arxiv_v1 retrieval smoke QA
CONTEXTS='4096' SUBTASKS='niah_single niah_multikey niah_multiquery' \
  SAMPLES=1 CONFIG=both TIMEOUT=900 \
  ./tests/sm86/bench_llama31_arxiv_v1_smoke.sh \
  /path/to/Meta-Llama-3.1-8B

# CUDA / sm86 Llama 3.1 8B PG-19 teacher-forced smoke QA
CONTEXTS='512' NUM_CHUNKS=1 CONFIG=both \
  ./tests/sm86/bench_llama31_pg19_smoke.sh \
  /path/to/Meta-Llama-3.1-8B
```

## Runtime feature impact

Measured delta of each runtime feature on its canonical workload, on
gfx1100 unless noted. **TBM = to be measured** — the feature is shipped
and validated for correctness but the perf measurement script hasn't
landed. Open an issue with reproduction notes if you want to fill one
in.

| Feature | Canonical workload | Baseline | With feature | Delta | Source |
|---|---|---|---|---|---|
| KV-FP8 | qwen3.5-9b INT4 + 6-token prompt + 16 generated tokens, gfx1100 | 26 ms/step | 347 ms/step¹ | 13.3× SLOWER (decode replay-prefill path) | tests/gfx1100/bench_matrix.sh, validated 2026-05-04 |
| KV-FP8 | qwen3.6-35b-a3b INT4 + 6-token prompt + 16 generated tokens, gfx1100 | 28.3 ms/step | 28.5 ms/step | +0.7% (effectively free; the win is VRAM headroom for long contexts) | manual run, 2026-05-03 |
| KV-FP8 sidecar window | qwen3.6-35b-a3b INT4 + 22-token context (test prompt + 16 gen) | 28.5 ms/step | 28.5 ms/step | identical at this context length (window=256 covers all 22 tokens; the BF16 sidecar is the win at LONG contexts not measured here) | manual run, 2026-05-03 |
| VMM | qwen3.6-35b-a3b INT4 + 8192-token context, gfx1100 | OOM (24 GiB exceeded) | runs | enables the workload | tests/gfx1100/bench_qwen36_sparse_caps.py |
| SpecPrefill (cosine, keep=0.50) | qwen3.5-9b BF16 + 1353-token prompt, gfx1100 | 4941 ms TTFT | 2385 ms TTFT (default `--specprefill-algorithm cosine`, shallowest layer + drafter early-exit) | **2.07× FASTER** | tests/gfx1100/bench_specprefill_cosine.sh, 2026-05-04 |
| SpecPrefill (cosine all_max, keep=0.50) | qwen3.5-9b BF16 + 1353-token prompt, gfx1100 | 4941 ms TTFT | 4144 ms TTFT (`SUPERSONIC_SPECPREFILL_LAYERS=all_max`) | **1.19× FASTER** | tests/gfx1100/bench_specprefill_cosine.sh, 2026-05-04 |
| SpecPrefill (lookahead, keep=0.50) | qwen3.5-9b BF16 + 1353-token prompt, gfx1100 | 4941 ms TTFT | 7846 ms TTFT (legacy `--specprefill-algorithm lookahead`) | **1.59× SLOWER**² | tests/gfx1100/bench_specprefill_cosine.sh, 2026-05-04 |
| SpecPrefill + KV-FP8 (cosine) | qwen3.5-9b BF16 + KV-FP8 + 1353-token prompt | dense + KV-FP8 (replay-prefill decode) | sparse target prefill (TTFT win preserved) + replay-prefill decode³ | TTFT win same as cosine without KV-FP8; decode bounded by KV-FP8 replay cost | crates/runner/tests/specprefill_qwen35_9b_cosine_kvfp8_parity.rs, 2026-05-04 |
| SpecPrefill + KV-FP8 (lookahead) | qwen3.5-9b BF16 + KV-FP8 + 1353-token prompt | — | rejected by validation⁴ | **REJECTED** | n/a (CLI guard, lookahead-only since 2026-05-04) |
| DFlash (B=3) | qwen3.5-9b INT4 greedy decode, gfx1100 | ~32 ms/step | ~12 ms/step (effective; 2.5-3× speedup) | 2.5-3× FASTER | docs/dflash.md M4.3 numbers |
| MoE prefetch | qwen3.6-35b-a3b INT4 decode, gfx1100 | included | (default-on) | — | the 28.3 ms/step row above already includes prefetch — the persistent megakernel default path uses it. A/B vs no-prefetch needs `--no-persistent-decode` which falls back to a different decode path entirely. |
| Certified KV (shadow-validate) | llama3.1-8b INT8 + 1024-token prompt, sm86 | TBM ms/step | TBM ms/step | TBM | (script TBM, sm86-only — not measurable on a HIP-only dev box) |

¹ Qwen3.5 KV-FP8 falls back to the *replayed-prefill* decode path (see
  the gfx1100 main matrix's footnote on KV-FP8). The decode kernel itself
  is fine; the slow column is the per-step prefill replay needed to keep
  the FP8 KV cache self-consistent. KV-FP8 on Qwen3.5 is currently a
  memory feature (headroom for longer contexts), not a throughput feature.

² The legacy `--specprefill-algorithm lookahead` path is slower than
  dense prefill on gfx1100 *at short prompts* because the speculator's
  lookahead decode steps route through the component decode path
  (per-head D2D K/V copy fallback added in PR #177) instead of the
  persistent megakernel. At ≥4k-token prompts the target-prefill
  savings overtake the speculator overhead and lookahead does pull
  ahead of dense (1.18× at 4k, 1.55× at 8k) — but it is still beaten
  by the default `cosine` path at every measured length. Phase D
  (2026-05-04) replaced the default scoring algorithm with `cosine`
  — a hipfire-PFlash-style single-layer cosine-similarity score that
  does one drafter prefill and a single small HIP kernel launch,
  dropping the lookahead decode steps entirely. A follow-on change
  wired in a drafter early-exit (`prefill_kv_through`) so the drafter
  stops after the chosen scoring layer and skips the rest of its
  model. The combined default is 2.07× faster than dense at 1.3k
  tokens, **3.36× faster at 8k tokens** (the speedup compounds with
  prompt length), and scores marginally higher on correctness (cossim
  0.820 vs 0.708 against the dense reference at keep=0.50). See
  [specprefill.md § Algorithm](specprefill.md#algorithm) and
  [specprefill.md § Performance](specprefill.md#performance) for the
  full prompt-length sweep.

³ For the cosine path, SpecPrefill + KV-FP8 reuses the same
  replay-prefill decode strategy plain `--kv-fp8` already uses on
  Qwen3.5-9B (`replay_kv_fp8_enabled` in main.rs). The first prefill
  is sparse via `prefill_kept` (TTFT win preserved); each subsequent
  decode step rebuilds the KV cache from the full unsparsified history
  via `rebuild_prefill_state`. Per-token decode cost matches plain
  KV-FP8 decode — bounded by replay-prefill, same as footnote ¹.

⁴ The lookahead path's SpecPrefill + KV-FP8 combo remains rejected
  upfront by `validate_specprefill_flags`. The underlying issue: the
  speculator's lookahead decode runs through the component decode path
  with `copy_step_bf16` (BF16-only) as its K/V step write, and the
  drafter has no replay-prefill workaround (it doesn't rebuild a
  scratch state per step like the target does). Lifting it would need
  a BF16→FP8 quantise-on-the-fly K/V step write plus an FP8 attention
  read in component decode — a much bigger kernel job. Since the
  cosine path is the default and works with KV-FP8, the lookahead
  combo is documented-unsupported, not an active follow-up.

The DFlash numbers are pulled from [dflash.md](dflash.md)'s M4.3
single-pass fused-verify section. The KV-FP8 number is the gfx1100
matrix delta from the per-arch table at the top of this doc — it is
*recorded* there for one workload and *re-stated here* with the
feature label so the picker doc has a one-stop reference.

The "Baseline" column is the comparison point — the dense / no-feature
run on the same hardware, model, and prompt. The "Source" column names
the bench script or test that produced (or will produce) the
measurement.

<!-- AUTOGEN BELOW: bench-perf-matrix-sm86 -->
| Model           |  INT4 | Spec050 | Spec075 |
| --------------- | ----: | ----: | ----: |
| qwen3.6-35b-a3b | 11138.5 | 6599.2 | 9497.7 |

<!-- AUTOGEN END: bench-perf-matrix-sm86 -->
