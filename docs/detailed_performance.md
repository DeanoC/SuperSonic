# Detailed Performance

This is the long-form performance log: methodology, attribution, historical
optimization checkpoints, and generated benchmark zones. For the quick headline
matrix, see [performance.md](performance.md).

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
[optimization/qwen35-sm86-optimization.md](optimization/qwen35-sm86-optimization.md).

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
kept as the host fallback escape hatch. Decode now lets that native stage-5
linear path publish directly into the next residual buffer by default, avoiding
the old `attn_output -> residual` D2D handoff; set
`SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT=1` to retain the older
handoff for bisection. The default FFN lane remains the host-orchestrated INT4
fallback, but routed expert gate/up and down work is now batched across top-k
experts to reduce per-layer thread orchestration. Native FFN projection work
remains explicit opt-in with
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

The direct-output handoff is an orchestration cleanup, not a new headline
bottleneck fix. On the 2026-05-24 Apple M5 Max comparison, the default lane
measured `144.7 ms/token` versus `145.2 ms/token` with
`SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT=1`; HAL `copy_d2d` fell
from 840 calls / 3.44 MiB to 210 calls / 0.86 MiB. That confirms the old copy
handoff is avoidable but too small to explain current decode time.
The lm-head tail now has the same measured-gate treatment:
`SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX=1` keeps greedy top-1
selection on Metal and reads back only the chosen token when full host logits
are not needed. It remains opt-in until the lm-head tail sweep proves a
headline and `lm_head_ms_avg` win with generated-ID parity. The first
four-token smoke preserved IDs `[11, 271, 40, 599]` and lowered full-logit D2H
from `0.067 ms` total to `0.004 ms`, but failed promotion because
`lm_head_ms_avg` moved from `9.044` to `9.389 ms`; the measured next lm-head
idea should fuse top-1 selection into the lm-head tail or change the dense
matmul shape, not add a separate argmax dispatch.

Reproduce the run with:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" cargo run --release -p supersonic-bench --bin bench-perf -- --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4
```

### Qwen3.5-35B-A3B Q4_K_M public comparison checkpoint

The Qwen3.5-35B-A3B Q4_K_M Metal lane is tracked separately from the Qwen3.6
INT4 gate because it is intended to compare against llama.cpp and MLX M5 Max
numbers for the same model/quant target. The SuperSonic control checkpoint uses
the Q4_K_M-sourced GPTQ/native-INT4 bake (`--q4km-gptq`), while the raw
external-equivalence checkpoint uses staged mixed-layout raw GGML K-blocks
(`--q4km`). The external adapter harness records llama.cpp from a raw GGUF
Q4_K_M file and MLX from a matching MLX model directory. Raw `--q4km` accepts
mixed Q4_K_M bakes where dense/shared projections remain native INT4 sidecars
and routed experts use GGML K-block tensors. The SuperSonic control run is
direct single-sequence greedy generation, empty prompt (`""`, tokenized as one
prompt token), 512 generated tokens, and no attribution or Metal profile pass:

```bash
target/release/supersonic --backend metal \
  --model qwen3.5-35b-a3b \
  --model-dir "$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b" \
  --q4km-gptq \
  --prompt "" \
  --context-size 1024 \
  --max-new-tokens 512 \
  --temperature 0 \
  --top-k 1 \
  --sampling-seed 20260504 \
  --no-download
```

Current local result on this M5 Max:

| Engine | Model/quant | Workload | ms/tok | tok/s | Source |
|---|---|---|---:|---:|---|
| SuperSonic | Qwen3.5-35B-A3B Q4_K_M | 1-token prompt + 512 generated, promoted exact-order FFN path | 67.3 | 14.9 | `/tmp/qwen35_default_after_scalar_fuse_patch_512.log` |
| SuperSonic | Qwen3.5-35B-A3B raw Q4_K_M | empty prompt + 512 generated, staged mixed-layout raw path | 73.6 | 13.6 | `target/bench-runs/2026-06-02-0f7b114/perf/qwen3.5-35b-a3b_q4km.json` |
| llama.cpp | Qwen3.5-35B-A3B raw GGUF Q4_K_M | empty/BOS-compatible prompt + 512 generated, five independent `llama-bench` samples | 15.1 | 66.0 | `target/bench-runs/2026-06-02-0f7b114/external/llama.cpp/qwen3.5-35b-a3b_q4km.json` |
| llama.cpp | Qwen3.5-35B-A3B Q4_K_M | older public M5 Max generation number | ~11.0 | 91.0 | historical public reference |
| MLX | Qwen3.5-35B-A3B Q4_K_M | older public M5 Max generation number | ~7.2 | 139.0 | historical public reference; local refresh pending MLX artifact |

To refresh local external references, pin the first line of
`tools/external/llama-cpp-version.txt` to `llama-cli --version` and the first
line of `tools/external/mlx-lm-version.txt` to `python3 -m mlx_lm --version`,
then run the raw GGUF llama.cpp reference. On the local Homebrew build,
`llama-bench` does not expose `--version` or an explicit `--ctx-size` flag; the
adapter records `context_size=1024` as workload metadata, uses `-p 1` for the
empty/BOS-compatible prompt, and records the command's default `batch-size=2048`
and `ubatch-size=512` context:

```bash
python3 -m oracle.bench.external.external_main \
  --engine llama.cpp \
  --models qwen3.5-35b-a3b \
  --quants q4km \
  --model-dir qwen3.5-35b-a3b=/path/to/qwen3.5-35b-a3b-q4_k_m.gguf \
  --prompt "" \
  --prompt-tokens 0 \
  --context-size 1024 \
  --max-new-tokens 512 \
  --measurement-runs 5
```

Run the matching MLX model-directory reference separately:

```bash
python3 -m oracle.bench.external.external_main \
  --engine mlx-lm \
  --models qwen3.5-35b-a3b \
  --quants q4km \
  --model-dir qwen3.5-35b-a3b=/path/to/mlx/qwen3.5-35b-a3b-q4 \
  --prompt "" \
  --prompt-tokens 0 \
  --context-size 1024 \
  --max-new-tokens 512 \
  --measurement-runs 5
```

The external JSON cells live under the latest `target/bench-runs/*/external/`
run directory and record engine version, exact command, workload metadata,
samples, median `ms_per_step`, derived `tok_per_s`, and engine-specific
batch/context notes. The 2026-06-02 local llama.cpp cell used
`version: 9430 (d48a56eff)` with command:

```bash
llama-bench \
  -m "$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf" \
  -p 1 \
  -n 512 \
  -r 1
```

It ran one warmup and five measured samples:
`[14.015, 14.436, 15.145, 15.394, 16.609] ms/token`, median
`15.145 ms/token` (`66.03 tok/s`). The raw SuperSonic staged lane is
reproduced with:

```bash
cargo run --release -p supersonic-bench --bin bench-perf -- \
  --preset qwen35-raw-q4km-m5-max-gen512 \
  --model-dir qwen3.5-35b-a3b="$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b"
```

The 2026-06-02 run (`target/bench-runs/2026-06-02-0f7b114`) used one 512-token
warmup and five measured samples: `[74.3, 73.6, 77.6, 71.7, 71.0]`, median
`73.6 ms/token` (`13.6 tok/s`). Against the refreshed local llama.cpp median,
raw SuperSonic is currently about 20.6% of llama.cpp throughput. The
fusion/megakernel optimization gate is therefore closed: start that phase only
after raw SuperSonic reaches at least 90% of the local llama.cpp throughput,
currently about `59.4 tok/s`.

Before changing the raw `--q4km` implementation or matrix row, use the manifest
audit to inventory the raw bake and keep layout coverage explicit:

```bash
cargo run -p runner --bin qwen36_q4km_manifest_audit -- \
  --model-dir /path/to/qwen3.5-35b-a3b
```

The current audit model treats raw GGML K-block dense, linear-attention,
shared-expert, routed-expert, and lm-head layouts as supported by the staged
Metal correctness path. It still reports missing tensors and unsupported layouts
as blockers.

The 2026-06-02 local raw-bake audit passed for
`$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b` with 40 layers
(`full=10`, `linear=30`), 331 projections, 251 native INT4 sidecar layouts,
80 raw GGML K-block layouts, and zero missing or unsupported blockers. A
one-token Metal smoke then completed with:

```bash
cargo run --release -p runner --bin supersonic -- \
  --backend metal \
  --model qwen3.5-35b-a3b \
  --model-dir "$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b" \
  --q4km \
  --prompt "Hello" \
  --context-size 64 \
  --max-new-tokens 1 \
  --temperature 0 \
  --top-k 1 \
  --sampling-seed 20260504 \
  --no-download \
  --emit-stage-timings \
  --emit-generated-json
```

That smoke generated token id `[49602]` and measured `2478.3 ms/token`
(`full_attn=37.7 ms`, `linear_attn=109.3 ms`, `ffn=2191.7 ms`,
`lm_head=138.1 ms`). Treat this only as raw-path load/decode evidence: the
staged raw GGML expert path is correctness-oriented and far slower than the
`--q4km-gptq` control lane.

The follow-up 8-token deterministic gate used the same prompt, context, greedy
sampling, and seed without `--emit-stage-timings`. Two consecutive runs produced
identical generated IDs
`[49602, 165189, 184475, 145239, 31375, 47477, 11625, 58985]` and measured
`373.7 ms/token` then `379.0 ms/token`. This confirms repeatable short raw
decode on the mixed-layout bake; it is still a staged correctness gate, not a
headline performance result.

The 128-token raw profiling gate then ran with `--context-size 256`,
`--max-new-tokens 128`, and `--emit-stage-timings`. It preserved the 8-token
prefix above and measured `105.7 ms/token` over 128 generated tokens. Stage
attribution was `chain=100.3 ms/token`, `lm_head=3.8 ms/token`,
`full_attn=7.8 ms/token`, `linear_attn=12.4 ms/token`, and
`ffn=79.9 ms/token`; generation wall time was 13.53s. This makes FFN, and
specifically staged raw routed-expert work, the first optimization target once
the raw lane reaches the 512-token benchmark gate.

Short smoke runs are too noisy for headline comparison: 16-token samples ranged
from ~125 to ~201 ms/token depending on command-buffer scheduling and warm
state, while longer runs settle much lower. The 2026-06-01 FFN Q4_K lane-pair
helper was retained only for the gate/up projection. In the 64-token split
check, scalar Q4_K measured 91.5 ms/token, gate/up-pair-only measured 79.9
ms/token with identical generated IDs, down-pair-only measured 87.6 ms/token
but diverged at token 12, and both pair helpers measured 82.6 ms/token but
diverged at token 9. The default therefore keeps gate/up pair-dot enabled and
leaves the down pair-dot path behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_Q4K_PAIR_DOWN`. The full preset run after
that change reported samples `[70.7, 68.7, 69.6, 69.7, 69.6]`, so the
headline was 69.6 ms/token. A follow-up reset-sync-buffer cleanup kept the
counter reset on fallback kernels but skipped the unused reset before Metal
native attention/FFN phases. The repeatable preset then reported samples
`[69.0, 69.0, 68.7, 69.0, 68.8]`, setting the pre-exact-order headline to a
69.0 ms/token median. Direct 512-token no-JSON controls on that path measured
67.6 and 67.0 ms/token and generated identical 512-token streams; they matched
the earlier default through token 173. The 128-token control matched the earlier
default exactly. A separate `--emit-generated-json` control diverged early, so
the benchmark preset and reproduction command omit that flag.

The 2026-06-01 exact-order FFN update promoted two Metal paths that preserve the
default generated stream while recovering some of the speed from previously
unsafe tiled/SIMD experiments:

- Router logits now use the exact-order SIMD path by default unless
  `SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_STAGE5_EXACT_SIMD=1` is set. It
  keeps lane 0's accumulation order aligned with the serial router while the
  other lanes materialize products. A 128-token default-vs-promoted check
  matched every generated ID and moved from 68.3 to 62.6 ms/token
  (`/tmp/qwen35_q4km_default_128_after_exact.log` vs.
  `/tmp/qwen35_q4km_default_exact_promoted_128.log`).
- Shared gate/up now uses the exact-order SIMD path by default unless
  `SUPERSONIC_METAL_DISABLE_QWEN36_FFN_SHARED_GATE_UP_EXACT_SIMD=1` is set. It
  keeps the host-order INT4 pair-add pattern for gate/up while parallelizing the
  dequant/product step. A split FFN profile measured
  `qwen36_ffn_int4_shared_gate_up_exact_simd` at 0.0759 ms/layer versus the
  previous shared gate/up phase around 0.164 ms/layer, with matching generated
  IDs through the 16-token profile and the 64-token smoke.
- The promoted default 512-token comparison initially measured 65.5 ms/token
  (`/tmp/qwen35_q4km_default_exact_promoted_512.log`). A later current-tree
  control after the native full-attention diagnostics measured 67.3 ms/token
  (`/tmp/qwen35_default_after_scalar_fuse_patch_512.log`), matching the earlier
  generated stream through token 286 before late argmax drift. The quick
  headline now uses the later current-tree 67.3 ms/token checkpoint. This is a
  one-shot direct decode checkpoint, not yet a refreshed five-repetition
  `bench-perf --preset qwen35-q4km-m5-max-gen512` median.
- A follow-up exact-order shared-scalar SIMD probe reduced the split-profile
  shared scalar phase from about 0.117 to 0.102 ms/layer and matched the first
  128 generated tokens, but diverged later in the 512-token stream. It remains
  opt-in behind `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_EXACT_SIMD=1`
  and is not part of the headline path.
  The decode-batch shared/routed parity taps now support position/layer filters
  so late shared-scalar drift can be inspected without snapshotting every layer
  of every generated token:
  `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_POSITION`,
  `..._LAYER`, and the routed equivalents
  `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_POSITION` /
  `..._LAYER`. A filtered smoke at position 1, layer 0 emitted a single shared
  parity line and correctly labeled the path as
  `gate_up_exact_simd+scalar_exact_simd`
  (`/tmp/qwen35_filtered_shared_parity_smoke.log`).
- Follow-up compatibility probes did not improve the headline. The
  expression-matched shared-down exact SIMD path matched the 128-token stream
  but slowed that check to 64.9 ms/token
  (`/tmp/qwen35_q4km_shared_down_pair_expr_exact_simd_128.log`), so it remains
  opt-in behind `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_EXACT_SIMD=1`.
  A top-k-parallel expert-down/finalize path also matched the 128-token stream
  but slowed to 65.4 ms/token in that early check
  (`/tmp/qwen35_q4km_expert_down_topk_parallel_128.log`). A later 512-token
  gate revalidated the same one-row top-k shape as deterministic, so current
  raw GGML expert-down/finalize uses it by default with
  `SUPERSONIC_METAL_DISABLE_QWEN36_FFN_EXPERT_DOWN_TOPK_PARALLEL=1` retained for
  A/B against the older multirow finalizer.
  The router exact-multirow probe was slower and diverged late in the 128-token
  stream; it is opt-in only behind
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_EXACT_MULTIROW=1`.

The 2026-06-02 raw `--q4km` parity/perf pass rechecked the existing down
pair-dot switch on the mixed-layout raw bake. With
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_Q4K_PAIR_DOWN=1`, the 8-token deterministic
smoke preserved the known generated IDs, but the 128-token gate diverged near
the end of the stream. Same-session 128-token profiles also did not show a
stable win: the flagged run measured `100.5 ms/token` with `ffn=74.8 ms/token`,
while the following default control measured `96.8 ms/token` with
`ffn=72.6 ms/token`. Keep down pair-dot opt-in. The built-in split profiler
(`SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES=1`) still slows the run enough to
confirm FFN dominance, so a compact stdout summary was added behind the
additional
`SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES_STDOUT=1` opt-in. A 4-token raw
Q4_K_M smoke on Apple M5 Max exited cleanly and emitted one aggregate row per
`qwen36_ffn_int4_*` command-buffer label, with
`qwen36_ffn_int4_router_topk_stage5_exact_simd` the largest split phase in that
cold capture (`82.790 ms` total across 160 invocations). Use both env flags for
future 128-token gates when Metal trace tooling is too heavyweight.
`SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES_BASELINE=1` now adds a whole-stage
FFN baseline row to the same split-profile run before the per-phase submits.
That baseline intentionally runs the FFN stage twice, so it is not a throughput
mode; it exists to compare the normal single-command-buffer stage GPU time with
the sum of split command buffers and quantify launch/wait attribution.
A 1-token raw Q4_K_M smoke emitted the expected
`qwen36_ffn_int4_stage5_with_router_profile_baseline` row and preserved
`Generated ids: [49602]`. The baseline was `69.374 ms` total across 40 FFN
calls, while the split subphase summary summed to roughly the same total
(`/tmp/supersonic-qwen35-raw-q4km-ffn-baseline-profile-1tok.log`), which
points the next FFN speed work back at in-kernel work and inter-stage scheduling
rather than hidden split-profile GPU timing inflation.
The next router/top-k probe adds
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_TOPK_PARALLEL_SELECT=1`. It keeps the
default top-k kernel's BF16-rounded probability scratch, then replaces the
thread-0 serial selected-expert scan with a threadgroup reduction that uses the
same value/index tie-break. A 1-token raw Q4_K_M router-split profile preserved
`Generated ids: [49602]` and reduced the profiled router top-k phase from
`8.909 ms` total across 40 FFN calls
(`/tmp/supersonic-qwen35-raw-q4km-router-split-baseline-profile-1tok.log`) to
`1.421 ms`
(`/tmp/supersonic-qwen35-raw-q4km-router-topk-parallel-profile-1tok.log`).
A 32-token A/B matched generated IDs exactly and measured `81.4 ms/token`
default vs `69.7 ms/token` with the opt-in
(`/tmp/supersonic-qwen35-raw-q4km-router-topk-{default,parallel}-32.log`).
A 128-token serial gate also matched exactly, measuring `65.5 ms/token`
default vs `61.6 ms/token` opt-in
(`/tmp/supersonic-qwen35-raw-q4km-router-topk-{default,parallel}-128-serial.log`).
The 512-token promotion gate failed, though: the first opt-in run diverged from
default at generated-token index 44, and a repeat opt-in run diverged at index
264 while also differing from the first opt-in run at index 44. Timings were
still slightly faster (`63.2 ms/token` default vs `62.5` and `60.6 ms/token`
opt-in), but the stream instability keeps the parallel selector diagnostic-only
(`/tmp/supersonic-qwen35-raw-q4km-router-topk-{default,parallel}-512-serial.log`,
`/tmp/supersonic-qwen35-raw-q4km-router-topk-parallel-512-repeat.log`).

The 2026-06-03 bug hunt isolated that instability to the no-tap Metal command
cadence around the top-k selector outputs. Targeted decode-batch router parity
taps at the observed divergent positions (`44`, then `227`, then `39`) matched
all 40 layers and also made the generated prefix match, so the tap's extra
snapshot/copy ordering was masking the issue rather than exposing a local
router math error. A single-float-scratch reduction and then an 8-way block
selector both still produced at least one divergent 512-token repeat without
an explicit output barrier. The accepted fix keeps the 8-way deterministic
block selector and adds a Metal `memoryBarrierWithResources` for the top-k
workspace and output-index buffers immediately after the top-k dispatch.
After rebuilding, the opt-in path matched the stable default stream for the
128-token gate (`/tmp/supersonic-qwen35-raw-q4km-router-topk-blockselect-128.log`)
and for two 512-token gates:
`/tmp/supersonic-qwen35-raw-q4km-router-topk-resourcebarrier-512.log` and
`/tmp/supersonic-qwen35-raw-q4km-router-topk-resourcebarrier-512-repeat.log`.
The matched 512-token samples measured `88.4` and `86.0 ms/token`, versus the
same-session stable default at `92.2 ms/token`
(`/tmp/supersonic-qwen35-raw-q4km-router-topk-default-512-fixed.log`).
The flag remained opt-in at that point until broader prompt/model coverage
could confirm the fix beyond the original target prompt.

A broader 2026-06-03 validation sweep then covered three 512-token prompts with
one default control and two opt-in repeats each:
`hello` (`1` prompt token), `code` (`25` prompt tokens), and `long_prefill`
(`49` prompt tokens). All generated ID streams matched exactly: opt-in A,
opt-in B, and opt-in A vs. B had no first mismatch for all three prompts.
The sweep summary is
`/tmp/supersonic-router-topk-sweep-20260603/summary.json`; per-run logs live
under `/tmp/supersonic-router-topk-sweep-20260603/`. The matched timings were
`94.2` default vs `90.1`/`92.9 ms/token` for `hello`, `88.6` default vs
`88.6`/`88.6` for `code`, and `91.1` default vs `89.2`/`89.7` for
`long_prefill`. With that coverage in place, the block-scan top-k selector is
promoted to default and the serial scan remains available with
`SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_TOPK_PARALLEL_SELECT=1`.
The post-promotion 512-token `hello` gate matched the disable-flag serial scan
exactly. The first promoted-default timing sample was noisy at `96.3 ms/token`
(`/tmp/supersonic-router-topk-promoted-default-512.log`), but the immediate
repeat landed at `90.1 ms/token`
(`/tmp/supersonic-router-topk-promoted-default-512-repeat.log`) against the
serial-disable comparison at `90.5 ms/token`
(`/tmp/supersonic-router-topk-promoted-disable-512.log`).

A current-tree 2026-06-02 rerun after the online-attention rebuild kept the raw
default 32-token stream unchanged:
`[49602, 165189, 184475, 145239, 31375, 47477, 11625, 58985, 757, 155221,
22937, 2926, 79609, 6906, 124641, 171294, 16544, 93186, 2661, 53625, 154878,
1686, 91745, 25088, 9696, 23920, 206853, 198643, 60811, 10332, 10265,
233548]`, measuring `120.3 ms/token` with `ffn_ms_avg=93.189` on the short
smoke. The compact split profile again put router/top-k first:
`qwen36_ffn_int4_router_topk_stage5_exact_simd` was `85.733 ms` total across
160 invocations, ahead of expert down (`26.336 ms`) and shared-gate scalar
(`22.911 ms`). Disabling the exact SIMD router path diverged at token 3 and did
not materially improve the same 32-token timing (`120.5 ms/token`), so the next
router optimization should keep the exact-order logits path and focus on a
parity-safe top-k/softmax reduction or launch/barrier consolidation.

The follow-up added a router subphase profile gate:
`SUPERSONIC_METAL_PROFILE_QWEN36_ROUTER_PHASES=1` splits the existing FFN phase
profile into router norm, router logits, and router top-k labels. On the same
4-token raw Q4_K_M smoke, default router subphases were norm `38.943 ms`,
logits `16.848 ms`, and top-k `17.561 ms` across 160 invocations. An opt-in
parity-safe router norm variant,
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_NORM_PARALLEL_STORE=1`, keeps the
mean-square sum in the original serial order but parallelizes the normalized
store. It preserved the 32-token generated stream and cut the profiled router
norm phase to `22.607 ms`, but the end-to-end 32-token smoke stayed slower
(`145.1 ms/token`, `ffn_ms_avg=116.280`), so it remains opt-in.

The next parity-safe router-norm probe narrowed that store parallelism to one
SIMD-width group. It also keeps the mean-square sum serial and preserved the
same 32-token generated stream. The short router subphase profile reported norm
`54.421 ms`, logits `48.140 ms`, and top-k `53.072 ms` across 160 invocations.
A controlled 2026-06-03 A/B pass promoted this path to default: three
32-token pairs measured median total `139.9 ms/token` with the SIMD-width
store vs `149.9 ms/token` for the old serial-store default, with FFN median
`112.6` vs `121.2 ms/token`. A two-pair 128-token confirmation preserved the
known 128-token stream and measured `90.5`/`88.9 ms/token` vs old-default
`97.7`/`94.1 ms/token`. Set
`SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_NORM_WARP_STORE=1` to recover the
old serial-store router norm path, or combine that disable with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_NORM_PARALLEL_STORE=1` to re-test
the older 256-thread store experiment.

The 2026-06-02 128-token raw Q4_K_M follow-up used the same two split-profile
env flags on Apple M5 Max, prompt `Hello`, context 256, and greedy seed
`20260504`. It preserved the known 128-token stream and measured
`145.7 ms/token` under the intentionally slowed split profile. Stage timing
still pointed at FFN (`ffn=117.5 ms/token`, `full_attn=8.5 ms/token`,
`linear_attn=13.2 ms/token`). The compact phase summary reported 5120 calls per
FFN subphase: router/top-k exact SIMD was largest at `2113.243 ms` total,
followed by expert down finalize (`648.422 ms`), shared scalar (`578.275 ms`),
shared gate/up exact SIMD (`384.761 ms`), shared down (`343.259 ms`), and
expert gate/up tiled (`286.565 ms`). The split rows account for only part of
the FFN stage wall time, so the remaining optimization target is still
submission/wait structure around the FFN stage, not just one arithmetic kernel.

A normal-lane 128-token Metal/HAL profile with `SUPERSONIC_METAL_PROFILE=1` and
without split profiling measured `69.2 ms/token` and preserved the same
generated IDs. Top Metal rows were `command_buffer_wait=4317.971 ms`,
`command_buffer_gpu:qwen36_decode_batch_ffn=3931.515 ms`, and
`qwen36_ffn_int4_stage5_with_router=3232.937 ms` across 5120 FFN calls.
`command_buffer_gpu:qwen36_decode_batch_linear_attn` was much smaller at
`711.453 ms`, while the final lm-head GEMV profile row
(`matmul_rhs_transposed_gemv_m1_tiled`) was `369.473 ms`. Retesting the existing
deferred FFN wait switch
(`SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT=1`) preserved the same
128-token stream but slowed to `73.1 ms/token`, so it remains diagnostic-only.

The next 2026-06-03 FFN follow-up checked the opt-in expert-down top-k-parallel
finalizer against the raw Q4_K_M lane and external source shape. llama.cpp's
Metal Q4_K path keeps QK_K=256 block dot work lane-local with scale/min terms
inside the dot kernel
([source](https://fossies.org/linux/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal)),
while MLX's MoE path expresses routed FFN as `SwitchGLU` over `gather_qmm` and
sorts expert indices only once the selected-index set is large enough
([source](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/switch_layers.py)).
For SuperSonic's single-token raw decode, the analogous safe step was to reuse
the existing raw GGML Q4_K pair-dot helper in the top-k-parallel down finalizer
instead of adding another dequant order. After that fix, three 128-token
top-k-parallel raw runs preserved the known generated stream and measured
`62.6`, `60.9`, and `62.6 ms/token`. A same-session A/B kept all six streams
identical and measured default `60.6`, `61.3`, `67.3 ms/token` versus
top-k-parallel `63.3`, `64.6`, `63.7 ms/token`
(`/tmp/supersonic-ffn-topk-ab-20260603-103332`). The one-row top-k path was
later promoted to the automatic raw-GGML expert-down candidate after repeated
512-token checks stayed deterministic.

A follow-up implemented the MLX-shaped selected-expert down path behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DOWN_GATHERED=1`. Instead of looping
top-k inside each output row, it writes per-selected-expert down outputs as
`[top_k, hidden]` into the existing FFN workspace slot and then combines the
top-k weights in a separate finalizer kernel. The 8-token deterministic raw
Q4_K_M smoke matched the default stream, and a same-session 128-token A/B
matched all generated IDs while measuring default `62.3 ms/token`
(`/tmp/supersonic-ffn-gathered-default-128.log`) versus gathered
`62.0 ms/token` (`/tmp/supersonic-ffn-gathered-enabled-128.log`). Keep it
diagnostic-only for now: this proves the MLX-style shape is parity-safe in the
SuperSonic workspace, but the extra dispatch/barrier does not yet move the
larger command-buffer-wait bottleneck.

The immediate fusion/scheduling follow-up also stayed negative. Rechecking the
existing FFN deferred-commit interval on the same 128-token raw Q4_K_M stream
kept IDs aligned but slowed interval 2 to `67.6 ms/token`
(`/tmp/supersonic-ffn-commit-interval2-128.log`) and interval 4 to
`71.5 ms/token` (`/tmp/supersonic-ffn-commit-interval4-128.log`) versus the
same-session `62.3 ms/token` control. A local top_k=8 unrolled expert-down
finalizer prototype matched the 8-token smoke but diverged in the 128-token run
and slowed to `69.4 ms/token` (`/tmp/supersonic-ffn-topk8-unrolled-128.log`),
so it was removed instead of retained as an opt-in flag. The lesson is the same
as the older commit-interval result: simple loop unrolling or wider FFN command
grouping is not enough; any deeper FFN fusion has to preserve the compiler's
current exact reduction shape or move a larger, naturally synchronized phase.

A fresh 2026-06-03 default raw `--q4km` 512-token profile after committing the
diagnostic gathered path measured `90.8 ms/token` under `--emit-stage-timings`
(`/tmp/supersonic-qwen35-raw-q4km-512-stage-20260603.log`). The breakdown was
`chain=85.536 ms/token`, `lm_head=3.125`, `full_attn=16.767`,
`linear_attn=11.428`, and `ffn=57.165`, so the profiled raw lane is still
FFN-dominated at the long benchmark length. A narrow router top-k parallel
selection prototype was then tried: it kept the same BF16 softmax probability
rounding and attempted to parallelize only the selected-expert scan. The
128-token same-settings A/B was faster in isolation (`70.7 ms/token` versus
default `73.3`) but diverged from the default generated stream at token 9
(`/tmp/supersonic-router-topk-parallel-select-128-20260603.log` versus
`/tmp/supersonic-router-topk-default-128-20260603.log`), so that prototype was
removed. The next FFN attempt should therefore target a larger exact router/FFN
phase consolidation with explicit top-k parity taps, not a silent replacement
of the top-k scan order.

The follow-up added the decode-batch-native router parity tap
`SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP=1`, with
`_MAX_CALLS`, `_POSITION`, and `_LAYER` filters. This keeps the default
decode-batch FFN path active while recomputing the host router reference from
captured input/workspace/output-index snapshots. The legacy non-batch router
tap now also accepts
`SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_LAYER`. A one-token raw
Q4_K_M smoke at layer 0 emitted a matching decode-batch row, and an 8-token
cross-layer probe (`/tmp/supersonic-decode-batch-router-tap-8tok.log`) captured
320 router rows with `topk_idx_match=1`, `workspace_idx_match=1`, and
`output_idx_match=1` throughout. Router selection is therefore not the current
source of divergence in the default raw lane; optimization can focus on
preserving the exact selection while reducing the router/FFN phase cost.

A current-tree router-subphase split profile on the same raw Q4_K_M lane
(`/tmp/supersonic-qwen35-raw-q4km-ffn-router-subphase-16.log`) preserved the
known 16-token greedy stream:
`[49602, 165189, 184475, 145239, 31375, 47477, 11625, 58985, 757, 155221,
22937, 2926, 79609, 6906, 124641, 171294]`. The intentionally slowed split
profile measured `289.3 ms/token` and still showed FFN dominance. Across 640
FFN calls, the largest split GPU labels were expert down finalize
(`89.415 ms`, `0.1397 ms/call`), shared scalar (`80.992 ms`, `0.1266 ms/call`),
router norm warp-store (`74.047 ms`, `0.1157 ms/call`), router top-k from
logits (`71.739 ms`, `0.1121 ms/call`), and router logits exact SIMD
(`66.545 ms`, `0.1040 ms/call`). The next kernel experiment should avoid
changing top-k ordering and instead look for a parity-safe consolidation or
scratch/barrier reduction across the exact router subphases and adjacent FFN
work.

An opt-in fused exact router selector was added on 2026-06-03:
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT=1`. It routes the native
stage-5 FFN through the existing monolithic router kernel instead of the default
split exact-SIMD norm/logits/top-k sequence, and the diagnostic labels now report
`router_path=fused-exact`. The local raw GGML Q4_K_M bake was not present
(`/Users/deano/.cache/supersonic-metal-models/qwen3.6-35b-a3b/.supersonic/v2-q4km`
was missing), so the validation used the available
`v2-int4-gptq` control lane. With decode-batch active and the router parity tap
enabled, the 8-token probe
(`/tmp/supersonic-qwen35-int4-router-fused-exact-decode-batch-tap-8.log`)
captured all 320 layer rows with `topk_idx_match=1`,
`workspace_idx_match=1`, `output_idx_match=1`, and zero h-norm/logit/top-k
weight deltas. Same-session 128-token no-tap A/B preserved identical generated
IDs, but the fused candidate was slower: default exact-SIMD
(`/tmp/supersonic-qwen35-int4-router-exact-simd-decode-batch-128.log`) measured
`55.4 ms/token` (`18.05 tok/s`), while fused exact
(`/tmp/supersonic-qwen35-int4-router-fused-exact-decode-batch-128.log`) measured
`60.9 ms/token` (`16.42 tok/s`). Keep it diagnostic-only; it proves the fused
router can be made parity-clean, but the current monolithic shape is not the
speed win.

The actual raw Q4_K_M target lane was rechecked immediately after that control
run using the existing local bake at
`$HOME/.cache/supersonic-metal-models/qwen3.5-35b-a3b/.supersonic/v2-q4km`.
The 8-token fused-router tap
(`/tmp/supersonic-qwen35-raw-q4km-router-fused-exact-decode-batch-tap-8.log`)
again captured all 320 layer rows with exact router agreement against the host
reference. However, no-tap 128-token A/B showed the monolithic fused router is
not stream-parity-safe on raw Q4_K_M: default exact-SIMD
(`/tmp/supersonic-qwen35-raw-q4km-router-exact-simd-decode-batch-128.log`)
measured `67.3 ms/token` (`14.86 tok/s`), while fused exact
(`/tmp/supersonic-qwen35-raw-q4km-router-fused-exact-decode-batch-128.log`)
measured `68.5 ms/token` (`14.60 tok/s`) and diverged at generated-token index
12 (`79609` vs. `194939`). Keep
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT=1` diagnostic-only; local
router parity is not enough for stream parity on the target raw lane.

Follow-up boundary checks showed that the fused router math was not the source
of the drift. A position-12 layer-output tap matched all 80 captured
attention/FFN rows between exact-SIMD and fused exact, and no-tap fused runs
matched the exact generated stream at 13, 16, and 32 generated tokens. The
fused 64-token run
(`/tmp/supersonic-qwen35-raw-q4km-fused-exact-notap-64.log`) diverged later at
generated-token index 33 (`27797` vs. `207577`), while the exact-SIMD 64-token
control (`/tmp/supersonic-qwen35-raw-q4km-exact-simd-notap-64.log`) matched the
128-token exact prefix. Forcing the whole decode batch into the sync-phase
cadence with `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SYNC_PHASES=1`
(`/tmp/supersonic-qwen35-raw-q4km-fused-exact-sync-phases-64.log`) restored the
64-token fused stream but slowed it from `68.1 ms/token` to `81.3 ms/token`.
Using the narrower existing
`SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL=9999`
(`/tmp/supersonic-qwen35-raw-q4km-fused-exact-ffn-commit-9999-64.log`) also
restored the stream at `76.4 ms/token` in the same investigation. The runtime
now applies that no-per-layer-FFN-deferred-commit cadence automatically when
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT=1` is set, unless an
explicit `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL` override is
provided. Rebuilt verification remained parity-clean
(`/tmp/supersonic-qwen35-raw-q4km-fused-exact-default-safe-64-repeat.log`), with
current warmed samples landing at `89.2 ms/token` for the default fused-safe
cadence and `83.6 ms/token` for an explicit interval-9999 rerun
(`/tmp/supersonic-qwen35-raw-q4km-fused-exact-ffn-commit-9999-64-newbin.log`).

The current gap is therefore not a one-line launch-count fix; the Metal profile
still points at the chained per-layer decode structure and command-buffer waits.
For the promoted exact-SIMD router, the default path keeps the existing
per-phase commit ordering for overlap. For the diagnostic fused-exact router,
the stream-safe FFN commit cadence is useful evidence that the drift is a
scheduling/resource-ordering problem, but it is not a performance promotion.

The 2026-06-04 follow-up tested a narrower router-logits consolidation shape:
a temporary 2-expert exact-pair kernel kept one simdgroup per expert and the
same lane-0 32-column accumulation order as
`qwen36_ffn_int4_router_logits_stage5_exact_simd`, but packed two experts into
each threadgroup to reduce threadgroup count without using the previously
rejected 8-expert multirow shape. It was parity-clean but slower, so the code
was removed rather than kept as another startup-time pipeline. The 8-token
smoke preserved the known prefix
(`/tmp/supersonic-router-logits-exact-pair-smoke-8.log`). A 16-token split
profile preserved the generated IDs but moved router logits from the current
default `62.574 ms` total (`0.098 ms/call`) to `72.777 ms` total
(`0.114 ms/call`) across 640 calls
(`/tmp/supersonic-router-logits-exact-pair-profile-16.log`). A normal-lane
128-token A/B also matched the default stream exactly, but slowed from
`97.8 ms/token` (`ffn=70.213 ms/token`) to `100.2 ms/token`
(`ffn=72.061 ms/token`)
(`/tmp/supersonic-router-logits-exact-pair-ab-{default,pair}-128.log`). This
rules out small exact-multirow router-logits packing as the next promotion
target; the next FFN target should either avoid changing router-logit work
ordering entirely or move to a different bucket such as shared-scalar
scheduling or expert-down accumulation.

A same-day `SUPERSONIC_METAL_PROFILE=1` default diagnostic
(`/tmp/supersonic-current-default-metal-profile-128.log`) should be treated as
timing-only because profiling changed the generated stream at token 9. It
measured `90.1 ms/token`, with `ffn=65.348 ms/token`, full attention
`8.834 ms/token`, linear attention `12.243 ms/token`, and lm-head
`3.314 ms/token`. The top profile rows were
`qwen36_ffn_int4_stage5_with_router` at `8196.516 ms`,
`command_buffer_wait` at `7437.951 ms`, aggregate `command_buffer_gpu` at
`3984.655 ms`, and labeled FFN command-buffer GPU time at `2949.334 ms` over
5120 FFN calls. This keeps the next target in FFN scheduling/command-buffer
structure rather than another local router-logits packing kernel.

Follow-up optimization probes on 2026-06-01 kept the headline unchanged but
identified the next safe work area:

- Disabling decode-batch preserved the 128-token stream but slowed the run to
  87.9 ms/token (`/tmp/qwen35_q4km_disable_decode_batch_128.log`), so the
  decode-batch path remains required for the headline lane.
- Existing shared-FFN tiled variants are faster but not parity-safe:
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED=1` measured 58.7 ms/token
  but diverged at token 0; shared gate/up tiled measured 61.1 ms/token and
  diverged at token 0; shared down tiled measured 67.5 ms/token and diverged at
  token 2; shared scalar SIMD measured 68.1 ms/token and diverged at token 0;
  shared gate/up `exp2` measured 71.3 ms/token and diverged at token 21.
- A refreshed split FFN phase profile on the promoted exact path
  (`/tmp/qwen35_ffn_phase_profile_current_16.log`) kept the generated IDs
  aligned for 16 tokens and again made the router block the largest FFN
  sub-phase: `qwen36_ffn_int4_router_topk_stage5_exact_simd` averaged
  0.4293 ms/layer, followed by expert down finalize at 0.1305 ms/layer, shared
  scalar at 0.1188 ms/layer, shared gate/up exact SIMD at 0.0777 ms/layer,
  shared down at 0.0666 ms/layer, and expert gate/up at 0.0533 ms/layer. The
  profile is wall-clock distorted by phase flushes but remains useful for
  ranking the next kernel target.
- Follow-up router/scheduling probes did not improve the headline. An opt-in
  exact SIMD router-norm kernel
  (`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_NORM_EXACT_SIMD=1`) matched the
  128-token stream but slowed to 68.6 ms/token
  (`/tmp/qwen35_router_norm_exact_simd_128.log`). Retesting exact-multirow
  router logits matched 128 tokens but slowed to 73.6 ms/token
  (`/tmp/qwen35_router_exact_multirow_128_retest.log`). Replacing router exact
  scratch barriers with SIMD-group barriers diverged after token 104 and was
  reverted; replacing the scratch/barrier reduction with SIMD shuffles preserved
  the 128-token stream but slowed to 72.7 ms/token and was also reverted. A
  narrower FFN deferred-commit interval probe
  (`SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL=2`) matched the
  128-token stream but slowed to 71.7 ms/token
  (`/tmp/qwen35_ffn_commit_interval2_128.log`), confirming that the current
  per-FFN commits are buying enough CPU/GPU overlap to offset their launch cost.
  A current-tree revisit of the same scheduling lever kept 64-token streams
  aligned for intervals 1, 2, 4, 8, and 9999; interval 9999 was fastest in that
  short smoke (`77.8 ms/token` vs `81.2 ms/token` for interval 1,
  `/tmp/supersonic-qwen35-raw-q4km-exact-ffn-interval-9999-64.log`). Two
  128-token pairs also matched and showed a small interval-9999 edge
  (`66.05` vs `67.35 ms/token` median), but the 512-token gate did not promote:
  interval 9999 diverged at generated-token index 408 for only `0.1 ms/token`
  improvement (`65.4` vs `65.5 ms/token`;
  `/tmp/supersonic-qwen35-raw-q4km-exact-interval9999-512-gate.log`).
  Keep the promoted exact-SIMD path on the default interval-1 cadence.
  A 2026-06-04 scheduling revisit kept the generated stream aligned but did
  not prove a chain-level win. FFN commit intervals 4, 8, and 9999 matched the
  128-token stream; after warm repeat, interval 9999 tied the default
  (`95.0` vs `95.1 ms/token`). Coarse
  `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SYNC_PHASES=1` was stream-safe but
  slower (`96.1 ms/token`). A temporary linear-attention commit interval hook
  also matched the stream, but linear-only intervals 8 and 9999 did not beat the
  warmed default (`95.3` and `96.6 ms/token`). Combining linear interval 8 with
  FFN interval 9999 matched all 512 generated IDs across two runs and measured
  `88.8` and `89.2 ms/token`, but both runs had slower chain/FFN buckets than
  the warmed default control (`chain=85.769/86.116`, `ffn=57.707/58.048` vs
  default `chain=84.483`, `ffn=55.955`). The hook was removed; keep this bucket
  pointed at a larger FFN residency/submit-wait redesign rather than another
  phase-commit cadence tweak
  (`/tmp/supersonic-{ffn-commit-interval-4-128,ffn-commit-interval-8-128,ffn-commit-interval-9999-128,ffn-commit-9999-repeat-128,decode-batch-sync-phases-128,linear-commit-interval-8-128,linear-commit-interval-9999-128,linear8-ffn9999-512,linear8-ffn9999-repeat-512,decode-batch-default-repeat-512}.log`).
  A follow-up generic Metal-batch barrier probe also matched generated streams
  but was slower. Temporarily skipping the helper-level post-encode
  `memoryBarrierWithScope:MTLBarrierScopeBuffers` preserved the 8-token prefix
  and the full 128-token stream, but the 128-token control was faster
  (`88.1 ms/token`, `ffn=64.182`) than the skip variant (`92.2 ms/token`,
  `ffn=66.129`). Keep the helper-level barrier in place; if submit/wait is
  revisited, target a more explicit resident FFN representation or a staged
  command-buffer design rather than removing generic hazard ordering
  (`/tmp/supersonic-skip-post-encode-barrier-{smoke-8,smoke-repeat-8,128}.log`,
  `/tmp/supersonic-post-encode-barrier-default-control-{8,128}.log`).
  Rechecking the MLX-shaped gathered expert-down path in the same tree also
  tied the default rather than improving it: three 128-token pairs measured
  equal medians of `64.3 ms/token`, with pairwise generated streams matching in
  two of three pairs and one late pairwise divergence at token 97
  (`/tmp/supersonic-qwen35-raw-q4km-gathered-current-ab-*.log`).
  The split profiler now labels raw Q4_K_M expert subphases precisely as
  `qwen36_ffn_int4_expert_gate_up_multirow_stage5` and
  `qwen36_ffn_int4_expert_down_finalize_multirow`; the 4-token label smoke
  emitted both names and preserved the known prefix
  (`/tmp/supersonic-qwen35-raw-q4km-multirow-label-smoke-4.log`).
  A follow-up rowpair/top-k-parallel expert-down probe is available only behind
  `SUPERSONIC_METAL_DIAG_QWEN36_FFN_EXPERT_DOWN_ROWPAIR_TOPK_PARALLEL=1`.
  It computes two output rows per threadgroup while assigning one simdgroup per
  top-k expert. The 8-token smoke preserved the known raw Q4_K_M prefix
  (`/tmp/supersonic-qwen35-raw-q4km-rowpair-topk-smoke-8.log`), and a
  128-token profiled A/B matched the generated stream while reducing expert
  down finalize from `586.979 ms` (`0.1146 ms/call`) to `521.111 ms`
  (`0.1018 ms/call`) and moving the profiled run from `120.1` to
  `116.3 ms/token`
  (`/tmp/supersonic-qwen35-raw-q4km-rowpair-{default,enabled}-ab-128.log`).
  It is not promotable: the 512-token gate was faster (`63.0` vs
  `64.8 ms/token`) but diverged at generated-token index 251
  (`/tmp/supersonic-qwen35-raw-q4km-rowpair-{default,enabled}-gate-512.log`).
  Repeated investigation runs confirmed that this is a nondeterministic
  two-row rowpair issue rather than top-k parallelism in general: rowpair
  repeats diverged from each other, a same-input scratch compare reported
  `diff_count=0`, and the one-row top-k-parallel control matched the default
  stream twice over 512 tokens (`65.8` and `62.8 ms/token`). The current
  automatic raw-GGML top-k run without the enable env also matched the
  default 512-token stream at `64.7 ms/token`
  (`/tmp/supersonic-qwen35-raw-q4km-topk-auto-gate-512-a.log`), and a
  1-token split-profile smoke confirmed the dispatch label
  `qwen36_ffn_int4_expert_down_finalize_topk_parallel`
  (`/tmp/supersonic-qwen35-raw-q4km-topk-auto-profile-1tok.log`). Keep
  rowpair quarantined behind
  `SUPERSONIC_METAL_DIAG_QWEN36_FFN_EXPERT_DOWN_ROWPAIR_TOPK_PARALLEL=1`.
  A safer two-row follow-up is available behind
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DOWN_MULTIROW_TOPK_PARALLEL=1`.
  It keeps one simdgroup per selected expert, computes two rows serially inside
  each expert simdgroup, stores isolated `[2][top_k]` scratch, and combines both
  rows from one owner simdgroup. It is deterministic but not faster: the
  1-token profile smoke emitted
  `qwen36_ffn_int4_expert_down_finalize_multirow_topk_parallel`
  (`/tmp/supersonic-qwen35-raw-q4km-multirow-topk-profile-1tok.log`), the
  8-token smoke matched the known prefix
  (`/tmp/supersonic-qwen35-raw-q4km-multirow-topk-smoke-8.log`), and two
  512-token gates matched the default stream exactly at `66.6` and
  `66.7 ms/token`
  (`/tmp/supersonic-qwen35-raw-q4km-multirow-topk-gate-512-{a,b}.log`).
  Keep the one-row top-k path as the default.
- The opt-in native full-attention path is also a real speed lever but not
  promotable yet. `SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE=1` measured
  58.4 ms/token on a 128-token run but diverged at token 0. Adding
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_HOST_ORDER_STAGE5=1` preserved the first
  four generated tokens but slowed to 77.4 ms/token and diverged at token 4.
  Layer-output taps for one generated token put the first plain-native checksum
  mismatch at layer 19 attention; the host-order variant's first checksum
  mismatch moved to layer 27 attention.
  Follow-up probes split the native host-order switch into opt-in exact
  sub-boundaries:
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXACT_INPUT_NORM=1`,
  `..._EXACT_PROJECTIONS=1`, `..._EXACT_QK_NORM_ROPE_CACHE=1`,
  `..._EXACT_SCORES=1`, `..._EXACT_VALUES=1`, and
  `..._EXACT_OUT_PROJ=1`. On layer 3 at position 1, exact input RMSNorm alone
  reduced the layer-attention BF16 mismatch from 150 elements to 1 and restored
  the second generated token; exact input+out-proj made that layer output
  byte-identical. On all native full-attention layers, exact input alone and
  exact input+out-proj still diverged after two matching tokens, while the full
  exact mask still matched only four tokens. A checksum sweep showed exact
  input+out-proj first differed at layer 27 attention on position 0, pointing at
  later-layer projection/cache sensitivity; these switches remain diagnostic
  only.
- A later layer-3-only native-full-attention diagnosis narrowed the earliest
  remaining full-exact drift to the softmax/value substage rather than K/V cache
  or projection layout. With
  `SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE=1`,
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_NATIVE_MAX_LAYER=3`, and
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_HOST_ORDER_STAGE5=1`, the generated stream
  matched baseline through four tokens and diverged on token five
  (`/tmp/qwen35_native_l3_full_exact_8.log`: `[219673, 177277, 79609, 239919,
  100335, ...]` vs baseline token five `96946`). The new opt-in K/V cache tap
  (`SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP=1`) showed byte-identical K/V
  prefixes for layer 3 through position 4
  (`/tmp/qwen35_direct_kv_l3_5.log`,
  `/tmp/qwen35_native_l3_full_exact_kv_5.log`). The new workspace-region tap
  (`SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP=1`) showed byte-identical `q_raw`,
  `k_raw`, `v_raw`, `q_normed`, `k_normed`, `q_rot`, and `k_rot` at layer 3
  position 4; the first differing region was `attn`
  (`/tmp/qwen35_direct_ws_l3_pos4.log`,
  `/tmp/qwen35_native_l3_full_exact_ws_pos4.log`). Rounding the native attention
  vector before the gate did not improve the generated prefix and was reverted.
  A follow-up probability tap
  (`SUPERSONIC_METAL_QWEN36_FULL_ATTN_PROB_TAP=1`) confirmed that the first
  full-exact native difference is already in the softmax probabilities:
  `/tmp/qwen35_direct_ws_prob_l3_pos4.log` and
  `/tmp/qwen35_native_l3_full_exact_ws_prob_pos4.log` had matching upstream
  regions but different `prob_tap` checksums (`e4d583f9b9cf9f71` vs
  `da883fa3c9439dff`), with the visible head differing by one low byte on the
  fourth probability (`...03,ec,19,3c` vs `...04,ec,19,3c`). Enabling
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_PRECISE_EXP=1` did not improve the
  generated prefix after the diagnostic flag was wired through host-order mode
  (`/tmp/qwen35_native_l3_full_exact_precise_exp_fixed_8.log`).
  BF16-rounding the native softmax probabilities before the value accumulation
  was also negative, diverging earlier at the third generated token
  (`/tmp/qwen35_native_l3_full_exact_round_probs_8.log`), and that probe was
  reverted.
  A later score tap (`SUPERSONIC_METAL_QWEN36_FULL_ATTN_SCORE_TAP=1`) proved
  the raw QK scores are not the source of that drift: at layer 3 / position 4,
  direct and native host-order both reported `score_tap=cee588fd52bf23ca`,
  while the probability tap still differed (`e4d583f9b9cf9f71` direct vs
  `da883fa3c9439dff` native). The comparison logs are
  `/tmp/qwen35_direct_ws_score_prob_l3_pos4.log` and
  `/tmp/qwen35_native_ws_score_prob_l3_pos4.log`, so the remaining parity target
  is specifically the softmax `exp/sum/div` boundary.
  Follow-up exp/denominator taps narrowed that further: with
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXP_TAP=1` and
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_DENOM_TAP=1`, direct and native still had
  matching scores (`cee588fd52bf23ca`) and denominators (`20595a8ca0a4a804`),
  but differed in `exp_tap` (`e2ceea3c19782125` direct vs
  `009160bee7917489` native), carrying into the same probability mismatch.
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_PRECISE_EXP=1` produced the same native
  exp/prob checksums, so the mismatch is Metal's exp result versus the CPU/Rust
  exp used by the direct path rather than a fast-math option. A transient
  `exp2(x * log2(e))` compatibility probe was worse, increasing the visible
  exp mismatches from 26/80 words to 62/80 words and leaving the same divergent
  fifth token (`/tmp/qwen35_native_ws_softmax_words_exp2_l3_pos4.log`), so that
  hook was removed. A Cephes-style polynomial exp diagnostic
  (`SUPERSONIC_METAL_QWEN36_FULL_ATTN_POLY_EXP=1`) moved the softmax boundary in
  the right direction. With explicit f32 materialization points in the
  polynomial, the same layer/position reduced exp mismatches to 4/80 words and
  probability mismatches to 7/80 words, but still produced the native fifth
  token (`100335` instead of direct `96946`). A `volatile` materialization
  helper produced the same taps as the bitcast helper
  (`/tmp/qwen35_native_ws_softmax_words_poly_volatile_exp_l3_pos4.log`).
  The remaining fifth-token flip was then isolated to value accumulation rather
  than the softmax probabilities: a transient four-input exp overfit made
  `score_tap`, `exp_tap`, `denom_tap`, and `prob_tap` byte-identical while the
  `attn` checksum still differed and the generated token stayed native
  (`/tmp/qwen35_native_ws_softmax_words_poly_overfit_exp_l3_pos4.log`).
  Adding explicit f32 round points to the host-order value accumulation
  (`acc += p * v`) made `attn` and `gated` byte-identical under the same
  transient exp overfit and restored the direct fifth token
  (`/tmp/qwen35_native_ws_softmax_words_poly_overfit_rounded_values_l3_pos4.log`).
  With the overfit removed, the retained polynomial-exp plus rounded-value path
  now matches the direct generated stream through token 10 and first diverges at
  token 11 on a 16-token layer-3-limited probe
  (`/tmp/qwen35_native_l3_poly_value_round_16.log` vs
  `/tmp/qwen35_direct_16_after_value_round.log`).
  Forcing `precise` locals in that polynomial was not viable because the
  runtime Metal compile failed with status 1200
  (`/tmp/qwen35_native_ws_softmax_words_poly_precise_exp_l3_pos4.log`). Logs:
  `/tmp/qwen35_direct_ws_softmax_taps_l3_pos4.log`,
  `/tmp/qwen35_native_ws_softmax_taps_l3_pos4.log`, and
  `/tmp/qwen35_native_ws_softmax_taps_precise_exp_l3_pos4.log`; word dumps:
  `/tmp/qwen35_native_ws_softmax_words_poly_round_exp_l3_pos4.log`.
  A 2026-06-02 all-layer retest with
  `SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE=1`,
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_HOST_ORDER_STAGE5=1`, and
  `SUPERSONIC_METAL_QWEN36_FULL_ATTN_POLY_EXP=1` did not generalize the
  layer-3-limited improvement: it diverged on the second generated token
  (`[49602, 58985, ...]` instead of `[49602, 165189, ...]`) and measured
  `151.5 ms/token` on a 16-token smoke. Keep this path diagnostic-only until a
  broader softmax/value compatibility fix is found.
  A later native-values dispatch cleanup removed redundant lanes from the
  current values kernel: the shader computes one `(head, dim)` row per
  threadgroup and only lane 0 writes, so dispatching 32 lanes duplicated the
  same softmax/value accumulation 32 times. With the dispatch reduced to one
  thread per group, the opt-in plain native full-attention 128-token probe
  improved to 53.6 ms/token
  (`/tmp/qwen35_native_full_attn_values_1lane_128.log`) while remaining
  non-promotable because it still diverges at token 0. The host-order native
  diagnostic preserved the known first-four-token prefix and still diverged
  afterward (`/tmp/qwen35_native_host_order_values_1lane_8.log`), so the next
  correctness target remains the full-attention softmax probability drift.
  A follow-up rewrite changed the native values kernel from one threadgroup per
  `(head, dim)` to one threadgroup per head: lane 0 computes the softmax once
  into threadgroup memory and the 256 lanes accumulate value dimensions in
  parallel. This removes the structural 256x softmax recomputation per head,
  but the opt-in plain native 128-token probe stayed in the same performance
  band at 53.0 ms/token (`/tmp/qwen35_native_plain_tgsoftmax_128.log`), so the
  immediate wall-clock bottleneck is not the duplicated values softmax alone.
  The host-order polynomial-exp diagnostic retained the same 16-token boundary
  as before after this rewrite
  (`/tmp/qwen35_native_l3_poly_value_round_tgsoftmax_16.log`: matches direct
  through token 10, diverges at token 11).
  Offline comparison of the captured score taps showed the direct CPU path is
  exactly Darwin/libSystem `expf` on both position-4 and position-11 dumps,
  while Python double `exp` and a musl/Arm-style double-intermediate table exp
  each miss one captured word. A temporary Metal table-exp probe using double
  intermediates was removed after runtime shader compilation failed with
  `'double' is not supported in Metal`
  (`/tmp/qwen35_native_table_exp_compile_error.log`). Small coefficient and
  range-reduction nudges to the retained float polynomial improved the captured
  mismatch count only from 12 to 11 words and were not kept.
  A 2026-06-02 follow-up stopped tuning the known-drifting split
  score/probability path and added an opt-in online-softmax full-attention
  kernel behind `SUPERSONIC_METAL_QWEN36_FULL_ATTN_ONLINE=1`. The new kernel
  follows the shipped MLX `sdpa_vector` and llama.cpp Metal flash-attention
  recurrence: each sequence partition carries `(max_score, sum_exp_score,
  output_accumulator)`, then partitions are merged with the same max-rescale
  factor instead of materializing normalized probabilities first. On the
  available local `qwen3.6-35b-a3b` `--int4` bake, the online path compiled
  and matched both default and split-native generated IDs for 32-token
  deterministic smokes. The current rebuilt check generated
  `[11, 271, 40, 599, 264, 3377, 440, 264, 1957, 13, 271, 40, 599, 264, 1957,
  421, 15339, 264, 999, 13, 561, 999, 5435, 264, 1103, 314, 4105, 13, 353,
  1144, 310, 1301]` on both paths. The warm online run measured
  `53.1 ms/token` with `full_attn_ms_avg=6.406`; the split-native comparison
  measured `55.7 ms/token` with `full_attn_ms_avg=6.558`. The online gate now
  has its own 1024-token cap instead of the split-native path's 128-token score
  cap. A prior 160-token deterministic INT4 smoke with `--context-size 256`
  completed
  through the old limit and matched the split/native-fallback stream exactly;
  online measured `48.2 ms/token` with `full_attn_ms_avg=5.240`, while the
  split/native-fallback comparison measured `48.3 ms/token` with
  `full_attn_ms_avg=5.417`. A 512-token online INT4 smoke with
  `--context-size 1024` also completed with `KV cache cap = 513`, measuring
  `62.0 ms/token`, `full_attn_ms_avg=15.433`, `linear_attn_ms_avg=17.168`, and
  `ffn_ms_avg=25.365`. Keep the online path opt-in for now: it is the right
  upstream-shaped comparison lane, but it is not yet a headline speed win, and
  the local raw `--q4km` / `--q4km-gptq` bakes were unavailable under
  `--no-download` in this checkout. Two closer MLX-vector micro-shapes were
  tried and rejected on the 32-token smoke: moving the per-lane V accumulator
  into private locals stayed token-identical but raised `full_attn_ms_avg` to
  `7.596` with a local array and `6.837` with scalar accumulators; caching the
  eight per-lane Q values before the key loop also stayed token-identical but
  measured `full_attn_ms_avg=6.531`. Those edits were not kept because the
  current Metal codegen appears to prefer the lower-register-pressure shared
  scratch form in this kernel.
- A split full-attention phase profiler
  (`SUPERSONIC_METAL_PROFILE_QWEN36_FULL_ATTN_PHASES=1`) showed that the native
  values rewrite is no longer the main wall-clock target. Because decode-batch
  coalesces labels, the useful diagnostic run disabled decode batching:
  `/tmp/qwen35_native_full_attn_phase_profile_nobatch_4.log`
  (`SUPERSONIC_METAL_QWEN36_DISABLE_DECODE_BATCH=1`,
  `SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE=1`). Over 4 generated tokens
  and 10 full-attention layers per token, full-attention GPU time was roughly
  6.1 ms total, about 1.5 ms/token in the unbatched/profiled shape. The largest
  full-attention sub-phases were projections at 3.808 ms total and output
  projection/finalize at 1.939 ms total; values were only 0.259 ms total. The
  same profile still made FFN/decode scheduling the dominant optimization area,
  not native full-attention values.
- An opt-in fused shared gate/up + shared-scalar FFN probe was added behind
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_SCALAR_FUSED=1`. The
  first variant preserved the promoted exact shared gate/up path and computed
  the shared scalar in an extra threadgroup using the same lane-0 serial dot
  order as the default scalar kernel. A 128-token check matched the default
  generated IDs exactly and moved from 75.3 to 70.9 ms/token
  (`/tmp/qwen35_default_after_scalar_fuse_patch_128.log` vs.
  `/tmp/qwen35_shared_scalar_fused_serial_128.log`). The 512-token gate failed:
  the fused path diverged at token 4 and measured 65.2 ms/token
  (`/tmp/qwen35_shared_scalar_fused_serial_512.log`), while the current default
  512-token control measured 67.3 ms/token
  (`/tmp/qwen35_default_after_scalar_fuse_patch_512.log`). A follow-up row-0
  fused variant kept the original shared gate/up grid shape and computed the
  scalar from row 0 of that kernel. It avoided the early token-4 divergence but
  still drifted late in the 512-token stream and slowed to 69.9 ms/token
  (`/tmp/qwen35_shared_scalar_fused_row0_512.log`). The fused scalar path
  therefore remains diagnostic/opt-in and is not part of the headline path.
- Split FFN profiling is distorted by the extra phase flushes, but it is still
  useful directionally: on a 32-token profile run, router/top-k was the largest
  FFN GPU sub-phase (`qwen36_ffn_int4_router_topk_stage5`, 543.7 ms total),
  followed by shared gate/up (195.7 ms), expert down finalize (152.5 ms),
  shared scalar (139.0 ms), shared down (77.6 ms), and expert gate/up (62.5
  ms). A future router optimization still needs a full generated-stream parity
  gate. The current SIMD router probe matched top-k indices and weights across
  664 tapped router calls through a 17-token run, with maximum observed
  differences of 0.015625 in normalized hidden values, 0.0625 in router logits,
  and 0.0009765625 in top-k weights; however, the full 128-token SIMD-router
  run still diverged at token 16 and did not show a consistent speed win.

The repeatable harness preset for future comparisons is:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo run --release -p supersonic-bench --bin bench-perf -- \
  --preset qwen35-q4km-m5-max-gen512
```

This preset tracks the public `llama-bench` generation shape (`tg512`:
`n_prompt=0`, `n_gen=512`, five repetitions). llama.cpp defaults to
`n_batch=2048` and `n_ubatch=512`; those batch knobs affect prompt processing
and prefilled-context tests, while this comparison is single-stream decode at
zero prefilled depth. SuperSonic currently uses the BOS token for an empty
prompt, so its closest comparable workload is one prompt token plus 512
generated tokens with `--context-size 1024`.

The local-main-target workflow for this machine is:

1. quick smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" cargo test --release -p runner --test qwen36_moe_metal_smoke -- --ignored --nocapture`
2. headline decode gate: the `bench-perf --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4` command above
3. long-context smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke`
4. profile smoke: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke --metal-profile`
5. batched-prefill MoE feasibility: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke --batched-prefill-feasibility`
6. batched-prefill Metal default: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset smoke`
7. batched-prefill variant sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_batched_prefill_variants.py --metal-profile`
8. MTP tensor audit: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/audit_qwen36_mtp.py --require-complete-bake`
9. MTP acceptance/policy probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mtp_acceptance.py`
10. MTP Metal K=1 experiment: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mtp_acceptance.py --metal-experiment`
11. MTP Metal prompt-suite sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_mtp_acceptance.py --prompt-set smoke --metal-experiment`
12. static top-N resident-table probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_static_topn.py`
13. static top-N warm runtime sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_static_topn_runtime.py --modes default,static,static-hotset,mps-static-partial --metal-profile`
14. linear decode variant sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_linear_decode.py --prompt-set smoke --metal-profile`
15. full-attention decode variant sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_full_decode.py --prompt-set smoke --metal-profile`
16. lm-head tail variant sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_lm_head_tail.py --prompt-set smoke --metal-profile`
17. MPS resident-table viability probe: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/probe_qwen36_mps_resident_table.py --run-pilot --require-pilot`
18. route residency sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_route_residency.py --prompt-set smoke`
19. LRU resident-cache sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_lru_resident_cache.py --capacities 32,64 --metal-profile`
20. fused routed INT4 sweep: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/sweep_qwen36_fused_routed_int4.py --prompt-set smoke --metal-profile`
21. SOTA gate refresh plan: `python3 tests/metal/refresh_qwen36_sota_gates.py --max-age-hours 24`
22. SOTA gate summary: `python3 tests/metal/summarize_qwen36_sota_gates.py --require --max-age-hours 24`
23. next bottleneck selector: `python3 tests/metal/select_qwen36_next_bottleneck.py --require-selected`
24. routed-expert FFN microbench: `target/release/qwen36_ffn_expert_microbench --iters 20 --warmup 3`
24. long-context comparison: `SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" python3 tests/metal/bench_qwen36_longctx.py --preset comparison`

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
padded assignments, and padding overhead. The harness now runs the Metal
batched-prefill path by default: Metal batched full-attention plus a direct
routed-expert INT4 gate/up and down/combine kernel pair, with router/top-k
still on the existing host path. Set `--legacy-prefill-baseline` when a report
needs the older per-token Metal prefill baseline; `--batched-prefill-prototype`
is retained as a compatibility/provenance tag for older A/B report modes. The
shared-expert tail now opens one Metal batch per layer/chunk by default and
uses the fused residual add to avoid the old batch-flushing two-add sequence;
set
`SUPERSONIC_QWEN36_MOE_METAL_SHARED_EXPERT_BATCH=0` to bisect back to the
primitive sequence. Full-attention prefill now uses a prefill-specific
time-major vector kernel by default. One threadgroup owns each `(query, head)`
row, reduces the Q·K dot across `head_dim` once per KV token, and has lanes
accumulate V dimensions; set `SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_VEC=0` to
fall back to the older scalar-per-output-dimension attention path.
`--batched-prefill-variant` names the measured env-gated prototype probes
without requiring hand-built environment overrides: `linear-direct-off`,
`full-attn-vec-off`, `full-attn-tmajor`, `split-qgate`, `router-topk`,
`fused-residual-off`, and `shared-expert-batch-off`.
The harness records the selected variant and its env overrides per row so A/B
comparison outputs are self-describing. The long-context JSON schema is now
`qwen36-moe-metal-longctx-bench-v6` and
records `batched_prefill_prototype` at top level plus
`metal_batched_prefill_prototype` and `batched_prefill_variant` per row;
feasibility rows remain under `batched_prefill_plans`, and each row can carry
`prefill_progress` entries from `[qwen36-moe prefill-progress]` so an 8192-token
crash still preserves chunks/tokens completed, active variant, and elapsed
prefill time; set
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
ms/token decode; profile overhead was expected because that run split the
routed-expert phases. Current profile runs keep Qwen3.6 FFN phases aggregate by
default and only restore those per-phase waited command buffers when
`SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES=1` is set. The split profile rows
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

The current prefill-reduction pass adds crash-resilient progress telemetry,
promotes the Metal shared-expert tail from separate primitive submissions into
a batched command-buffer section, and adds the prefill-specific vector
full-attention kernel. Profile runs split the shared section into stable labels
(`qwen36_batched_prefill_shared_gate_up`,
`qwen36_batched_prefill_shared_down_scalar`,
`qwen36_batched_prefill_ffn_finalize`) while also recording aggregate
`qwen36_batched_prefill_shared_expert_int4` wall time. The vector attention
kernel is labelled `full_attention_prefill_tmajor_vec`.

The first validation run for that pass used one-token NIAH smokes on 2026-05-25
with no warmup. Baseline rows are the pre-change legacy Metal prefill path;
prototype/vector rows use the promoted Metal batched-prefill path with the
shared-expert batch enabled:

| Context | Legacy prefill | Shared-batch prototype | Promoted vector prefill | Improvement vs legacy | Vector ms/tok | Progress row |
|---:|---:|---:|---:|---:|---:|:---|
| 512 | 25.33 s | 9.71 s | 7.01 s | 72.3% | 100.38 | 417 tokens, 1 chunk |
| 2048 | 165.24 s | 75.31 s | 28.24 s | 82.9% | 199.31 | 1762 tokens, 4 chunks |
| 8192 | - | 1018.00 s | 147.94 s | - | 613.29 | 7142 tokens, 14 chunks |

The 8192 prototype row completed cleanly with generated ID `[271]`; the prior
`-11` long-prefill failure did not reproduce on either prototype path. NIAH
still reports NO for these rows because the run intentionally generates only
one token. A no-`--batched-prefill-prototype` default rerun confirms the
promoted runtime path: progress rows report
`metal-default+full-attn-vec+shared-expert-batch` for 512, 2048, and 8192. The
post-vector 512 profile names `command_buffer_wait` (8.57 s),
`qwen36_linear_int4_stage5` (4.48 s), and
`qwen36_batched_prefill_grouped_expert_direct` (3.47 s) as the largest native
rows. The post-vector 2048 profile reports 37.63 s prefill under
profiling, with `command_buffer_wait` (34.98 s), `qwen36_linear_int4_stage5`
(18.78 s native), `qwen36_batched_prefill_grouped_expert_direct` (12.20 s
native), HAL `copy_h2d` (10.78 s), and `full_attention_prefill_tmajor_vec`
(2.01 s native) as the largest rows. The next measured bottleneck is therefore
linear-attention command-buffer volume plus routed expert work at short
contexts, and full-attention/KV bandwidth becomes visible again as context
length grows.

The earlier linear-attention orchestration slice targeted
`qwen36_linear_int4_stage5`, which had
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
compute/residency at the 512-token smoke size, while the 2048/8192 rows above
shift the next long-context target to full-attention/KV bandwidth.

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
`--max-new-tokens 16`, then performs one additional unprofiled
`--emit-stage-timings` attribution run and a separate Metal/HAL profile
attribution run. For this Metal lane it also forces the dense prefill token loop
(`SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0` and
`SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP=1`) because the default Qwen3.6
batched prefill router-permute path is HIP/CUDA-only. The attribution run also
enables `SUPERSONIC_METAL_QWEN36_MPP_PILOT=1`, which emits a separate
`[qwen36-moe mpp-pilot]` row for repeated exact `64x32x64` MPP tiles. It also
enables `SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT=1`, which emits a resident
FP16 Metal Performance Shaders row for Qwen3.6 active-expert GEMV shapes. It
also enables `SUPERSONIC_METAL_PROFILE=1` only for the profile attribution run
so the profile JSON includes `metal_profile` and `hal_profile` objects with
parseable per-op rows without replacing the unprofiled stage table. These are
runtime-adjacent MPP/MPS pilot measurements, not model matmul replacements. The
attribution maps are stored in the same schema-v9 perf JSON as
`stage_timings`, `chain_breakdown`, `lifecycle_timings`, `mpp_pilot`,
`mps_expert_pilot`, `metal_profile`, and `hal_profile` without feeding back into
the headline median. When the profile pass carries extra split-dispatch
overhead, its timing maps are kept separately as `profile_stage_timings`,
`profile_chain_breakdown`, and `profile_lifecycle_timings`. Schema v9 also
preserves typed Qwen3.6 expert-residency policy rows with `resident_format`,
`scope`, `miss_policy`, `capacity`, and the numeric counters so perf artifacts
retain the scheduler identity, and `meta.json` records git dirty paths plus a
worktree diff hash so selector auto-discovery consumes only artifacts matching
the current checkout unless a historical JSON is supplied explicitly.

Qwen3.6 linear stage-5 profile attribution is aggregate by default: the
ordinary `SUPERSONIC_METAL_PROFILE=1` pass keeps `qwen36_linear_int4_stage5` as
one native profile row and one GPU timestamp row. Use
`SUPERSONIC_METAL_PROFILE_QWEN36_LINEAR_PHASES=1` only for explicit phase
attribution; it restores the per-phase waited command buffers for
`qwen36_linear_int4_input_norm`, projections, recurrent update, output gate,
and out-proj finalize. This keeps default perf JSON from promoting
phase-profile wait overhead into the next-bottleneck decision.

<!-- AUTOGEN BELOW: apple-m5-max-metal -->
| Model           |  INT4 |
| --------------- | ----: |
| qwen3.6-35b-a3b |  58.3 |

<!-- AUTOGEN END: apple-m5-max-metal -->

Current retained Apple M5 Max headline:

- median `58.3 ms/token` (`17.2 tok/s`) from
  `target/bench-runs/2026-05-25-d20a655-9/perf/qwen3.6-35b-a3b_int4.json`
- samples: `58.3`, `60.3`, `58.2`
- unprofiled attribution: `ffn_ms_avg=32.611`,
  `linear_attn_ms_avg=15.954`, `full_attn_ms_avg=5.737`, and
  `lm_head_ms_avg=4.659`
- profile attribution still names `command_buffer_wait` and
  `qwen36_linear_int4_stage5` first, followed by FFN host expert gate/up and
  down

Historical 2026-05-24 attribution run
(`target/bench-runs/2026-05-24-aa72613/perf/qwen3.6-35b-a3b_int4.json`):

| Metric | Value |
|---|---:|
| Headline median | 145.5 ms/token |
| Stage total | 139.930 ms/token |
| Profile stage total | 174.814 ms/token |
| Chain | 134.301 ms/token |
| LM head | 5.393 ms/token |
| FFN | 89.297 ms/token |
| Linear attention | 28.690 ms/token |
| Profile linear attention | 60.827 ms/token |
| Full attention | 16.113 ms/token |
| Prefill total | 801.291 ms |
| MPP pilot | 9.958 TFLOP/s |
| MPS expert pilot gate/up | 0.915 ms, 3.666 TFLOP/s |
| MPS expert pilot down | 0.338 ms, 4.961 TFLOP/s |

The headline median is the unprofiled median-of-3 run (`150.4`, `145.5`,
`141.8` ms/token). Current schema-v9 runs store unprofiled stage timings in
`stage_timings` and preserves the split-profile timing maps under the
`profile_*` fields; the profile pass still shows `qwen36_linear_int4_stage5`
and `command_buffer_wait` as the top Metal rows, but the normal stage table no
longer charges the seven-way linear profiling split to `linear_attn_ms_avg`.
The latest linear INT4 projection kernel consumes both nibbles from each
packed byte per lane load, preserving BF16 dequant rounding while halving the
packed-byte traffic for the projection and out-proj inner loops. Against the
previous schema-v8 run (`162.6 ms/token`, `linear_attn_ms_avg=31.335`), this
measured `145.5 ms/token` and `linear_attn_ms_avg=28.690`; split-profile GPU
rows show `qwen36_linear_int4_projections` dropping from `175.821 ms` to
`70.896 ms` total and `qwen36_linear_int4_out_proj_finalize` from `58.545 ms`
to `28.171 ms` total. Follow-up beta/g recurrent-update hoist probes were
measured and rejected: moving beta/g into the q/k repeat dispatch was headline
flat (`145.1 ms/token`) and regressed split recurrent/qk rows, while a
lane-0 threadgroup variant was also headline flat (`145.6 ms/token`) with a
slower split recurrent row. Keep the paired-nibble kernel as the default
linear-attention state. A default one-token smoke generated the expected token
`[11]`;
the latest local cold one-token profile measured `ffn_ms_avg=128.573`,
with `qwen36_ffn_host_expert_gate_up` at 56.839 ms total and
`qwen36_ffn_host_expert_down` at 36.181 ms total. The explicit full native FFN
escape hatch also generated `[11]`, but remains too slow to promote
(`ffn_ms_avg=640.553` in the same one-token smoke shape).

The 2026-05-25 host INT4 dequant LUT updates supersede the `145.5 ms/token`
baseline for the default lane. The FFN host dot helpers first moved to a
16-entry BF16-rounded dequant table per scale/zero group, avoiding per-element
BF16 rounding in the inner loop. A follow-up precomputes the active dense and
top-k expert tables once per layer/token before the row-parallel dot loops,
removing repeated table construction from the hot rows.

The follow-up Metal linear INT4 update groups packed-byte dot loops by their
GPTQ scale column inside the native `qwen36_linear_int4_stage5` projection and
out-projection kernels. That reuses each scale/zero pair across its 128-value
group instead of recomputing the sidecar index and reloading the pair for every
packed byte.

The latest FFN host router update parallelizes the 256-row BF16 router-logit
matvec after the h_norm write, then reuses that h_norm snapshot through the
later host FFN phases. The current Apple M5 Max `bench-perf` run
(`target/bench-runs/2026-05-25-7ec91da/perf/qwen3.6-35b-a3b_int4.json`)
measured median `96.5 ms/token` with samples `108.8`, `96.5`, `96.1`.
Unprofiled attribution is now `ffn_ms_avg=60.912`,
`linear_attn_ms_avg=19.545`, `full_attn_ms_avg=11.610`, and
`lm_head_ms_avg=4.514`; the one-token Metal smoke still generated `[11]`.
The follow-up host orchestration update routes the hot Qwen3.6 host row helpers
through a persistent worker pool with atomic countdown completion instead of
spawning scoped threads for every router/shared/expert phase. The current Apple
M5 Max `bench-perf` run
(`target/bench-runs/2026-05-25-fb8eb60-12/perf/qwen3.6-35b-a3b_int4.json`)
measured median `88.3 ms/token`, improving the `96.5 ms/token` router baseline.
Unprofiled attribution is now `ffn_ms_avg=53.543`,
`linear_attn_ms_avg=19.088`, `full_attn_ms_avg=10.232`, and
`lm_head_ms_avg=4.928`; the one-token Metal smoke still generated `[11]`.
Profile attribution still names `qwen36_linear_int4_stage5` and
`command_buffer_wait` as the top rows, followed by FFN host expert gate/up and
down.

The next retained FFN arithmetic cleanup keeps the exact INT4 LUT dequant
semantics, but shares an inlined paired-nibble accumulator across dense and
expert LUT dot products so each inner group consumes eight packed bytes per
loop body. The current Apple M5 Max `bench-perf` run
(`target/bench-runs/2026-05-25-e7d8c04/perf/qwen3.6-35b-a3b_int4.json`)
measured samples `85.2`, `86.2`, `89.4` with unprofiled
`total_ms_avg=85.877`. Attribution is now `ffn_ms_avg=51.564`,
`linear_attn_ms_avg=19.102`, `full_attn_ms_avg=9.987`, and
`lm_head_ms_avg=4.602`; the one-token Metal smoke still generated `[11]`.
Profile attribution has FFN expert gate/up at `554.605 ms` total and expert
down at `305.734 ms`, so the next real target is still a larger routed expert
compute/residency step rather than more router work.

The following compute-path update splits the same paired-nibble LUT dot loop
into independent accumulators, reducing the long scalar dependency chain while
staying within the existing FFN parity tolerance. The current Apple M5 Max
`bench-perf` run
(`target/bench-runs/2026-05-25-0c3f940/perf/qwen3.6-35b-a3b_int4.json`)
measured samples `83.6`, `78.0`, `73.7` with unprofiled
`total_ms_avg=72.762`. Attribution is now `ffn_ms_avg=42.501`,
`linear_attn_ms_avg=17.073`, `full_attn_ms_avg=8.078`, and
`lm_head_ms_avg=4.638`; the one-token Metal smoke still generated `[11]`.
Profile attribution shows FFN expert gate/up reduced to `457.864 ms` total and
expert down to `266.605 ms`. FFN remains the selected bottleneck, but the next
candidate needs to be a larger routed expert compute/residency path rather than
another reduction-loop cleanup.

The retained AArch64 host INT4 LUT dot path uses NEON table lookup to
materialize BF16-rounded weights from packed nibbles and accumulates four F32
vectors per eight packed bytes. The confirmation `bench-perf` run
(`target/bench-runs/2026-05-25-822b7d7-4/perf/qwen3.6-35b-a3b_int4.json`)
measured median `61.5 ms/token` with samples `61.5`, `66.2`, `57.9`, improving
the previous `78.0 ms/token` checkpoint. Unprofiled attribution was
`ffn_ms_avg=32.238`, `linear_attn_ms_avg=16.016`,
`full_attn_ms_avg=5.800`, and `lm_head_ms_avg=4.336`.

The latest default-lane cleanup removes the per-layer `h_norm` heap copy from
the Qwen3.6 Metal host FFN fallback. The normalized BF16-rounded row is already
resident in the FFN workspace, so router/shared/expert host phases now read it
from there while writing disjoint workspace regions. This keeps the same INT4
dot arithmetic and generated-token behavior; it only trims host allocation and
copy overhead. The first `bench-perf` run
(`target/bench-runs/2026-05-25-d20a655-8/perf/qwen3.6-35b-a3b_int4.json`)
measured median `58.1 ms/token` with samples `58.1`, `58.9`, `57.9`; the repeat
(`target/bench-runs/2026-05-25-d20a655-9/perf/qwen3.6-35b-a3b_int4.json`)
confirmed median `58.3 ms/token` with samples `58.3`, `60.3`, `58.2`. The
repeat stage table is `ffn_ms_avg=32.611`, `linear_attn_ms_avg=15.954`,
`full_attn_ms_avg=5.737`, and `lm_head_ms_avg=4.659`. Profile attribution still
names `command_buffer_wait` (`993.270 ms`) and `qwen36_linear_int4_stage5`
(`992.118 ms`) first, followed by FFN host expert gate/up (`436.736 ms`) and
down (`245.985 ms`), so the next measured target is the Metal linear stage/wait
pair plus the remaining FFN expert rows.

The closeout long-context pass confirms that decode has improved enough for
prefill stability and orchestration to be the next Apple target. The interrupted
comparison run completed usable console rows for 512 and 2048 requested context
tokens: 512 measured `72.683 ms/token` (`13.758 tok/s`) with
`prefill_total_ms=22799.604`, while 2048 measured `159.044 ms/token`
(`6.288 tok/s`) with `prefill_total_ms=159455.302`. Both rows missed the NIAH
answer with the short generation cap used for the smoke. A separate recorded
8192-token no-warmup row wrote
`target/qwen36_metal_longctx_8192_final.{json,md}` after `1431.64s`, but
returned `-11` before generated IDs, stage timings, or lifecycle timings were
emitted. The prompt did contain the expected needle (`SSB-NEEDLE-68696`), so
this is a runtime stability/long-prefill failure, not a prompt construction
failure. Treat 512/2048 as the current long-context performance evidence and
8192 as a failing gate that must be fixed before claiming long-context support
on Metal.

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
prompt set usable as a profiling gate rather than a Hello-only smoke. The v5
schema keeps the nonfatal `promotion_gate`: a resident mode must preserve
generated IDs versus `default`, improve headline ms/token and `ffn_ms_avg`,
keep full-attention, linear-attention, and lm-head inside the configured
regression ratio, and include non-regressed `command_buffer_wait` profile
evidence unless `--no-promotion-require-profile` is used. It also adds a
separate `experimental_family_parity` block for packed/static-family modes.
That section is diagnostic only: it can confirm that residency variants still
match the packed-family stream, but it cannot promote a mode unless the normal
default-based gate also passes.
The first four-token static smoke established a packed/static-family stream of
`[11, 353, 599, 264]` and ruled out latency promotion: the default-row baseline
in that run measured `decode_ms=702` and `ffn_ms_avg=98.761`, static measured
`decode_ms=951`, `ffn_ms_avg=177.563`, `exact_hits=10/160`,
`slot_hit_rate=0.508594`, and `copied_bytes=980372736`, and static+hotset
measured `decode_ms=1450`, `ffn_ms_avg=262.215`, and
`copied_bytes=2234557440`. Later divergence work showed that the packed,
direct-gather, static, and static-partial FFN family currently follows
`[11, 353, 599, 264]`, while the default/full-native path follows
`[11, 271, 40, 599]`. Treat the packed-family stream as experimental residency
coverage only until the standalone routed-expert arithmetic drift is fixed.
The v5 validation smoke in `target/qwen36_static_topn_family_parity4.{json,md}`
records that split explicitly: `generated_ids_match=false`,
`experimental_family_generated_ids_match=true`, and
`promotion_gate_passed=false` for `default,packed,static,static-partial`.

The native partial-hit static Top-N follow-up is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN_PARTIAL=1`. It uses the
same resident packed INT4 static table as the full-hit static path, remaps only
resident routed groups into the first hit-count top-k slots, runs the existing
native packed FFN shader for those hits, restores the original route workspace,
then computes miss groups with the host fallback and adds them into the partial
`moe_out`. This explicitly tests the non-MPS version of the partial resident
fork. The first one-token profiled smoke matched generated IDs across
`default`, `static`, and `static-partial`, but rejected the candidate:
`static-partial` measured `574.7 ms/token`, `ffn_ms_avg=483.074`, and copied
`3.670 GiB` while allocating all 40 resident tables
(`target/qwen36_static_topn_partial_smoke.{json,md}`). A four-token warm sweep
also rejected it: both static modes diverged from the default stream after the
first token, and `static-partial` was slower than full-hit `static`
(`210.6` vs `104.2 ms/token`, `ffn=159.293` vs `76.689`;
`target/qwen36_static_topn_partial_warm4.{json,md}`). Keep this mode
diagnostic; partial resident INT4 needs either a cheaper all-layer prewarm plus
a parity-safe default comparison, or a different in-kernel partial combine
before it is worth promoting.

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
residency sweep, MTP acceptance sweep, LRU resident-cache sweep, linear decode
sweep, and full-attention decode sweep JSON reports, then writes
`target/qwen36_sota_gate_summary.{json,md}` with input status, report age,
passed/failed gate IDs, candidate failures, refresh commands, and the next
action. Missing reports are preserved as rows by default; use
`--require --max-age-hours 24` when a local validation run should fail closed on
absent, malformed, schema-mismatched, stale, or missing-gate artifacts.
The v9 summary also records `superseded_gates`; this prevents an older
estimate or decision pass from taking `next_action` after the corresponding
runtime candidate has already been measured and rejected.
`tests/metal/refresh_qwen36_sota_gates.py` is the operational companion for
that summary: by default it writes
`target/qwen36_sota_gate_refresh_plan.{json,md}` with only the missing, stale,
malformed, schema-mismatched, or missing-gate rows selected; add `--run` to
execute those trusted local refresh commands in order, or `--only <gate_id>` to
force-refresh one gate even when its current report is already OK.
`tests/metal/select_qwen36_next_bottleneck.py` is the follow-up when that
summary lands on `keep_default_lane_and_select_next_measured_bottleneck`. It
reads the refreshed gate summary plus the profiled default rows from the
runtime sweeps, ranks decode buckets, marks FFN exhausted when the resident,
static, fused, MPS, and LRU forks all have negative runtime evidence, marks
linear and full attention exhausted after their variant gates fail, and writes
`target/qwen36_next_bottleneck.{json,md}` with the next bucket to prototype.

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
The warm four-token sweep preserved the packed-family stream
`[11, 353, 599, 264]`, but measured the default-row baseline at
`decode_ms=702`, `ffn_ms_avg=94.930` versus
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
preserved the packed-family stream `[11, 353, 599, 264]`, so the remap and
packed shader agree with the experimental packed path, but it measured
`777.3 ms/token`, `ffn_ms_avg=641.772`, and
`qwen36_ffn_int4_expert_gpu_pack_stage5=2417.156 ms` across 160 layer calls.
The command-buffer GPU attribution for the fused pack+expert shader was only
`64.400 ms`, while `command_buffer_wait` was `2678.881 ms`; moving slab
materialization from CPU to GPU therefore did not solve the residency/wait
problem.

The direct-gather follow-up is opt-in behind
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5=1`. It keeps
the original top-k expert IDs, reads the baked expert buffers directly, and uses
a 256-thread tiled down/finalize kernel so the down projection has the same
wide reduction shape as gate/up. It also preserved the packed-family stream
`[11, 353, 599, 264]`, but the unprofiled four-token smoke measured
`308.8 ms/token` with
`ffn_ms_avg=249.990`; the profiled run measured `756.8 ms/token`,
`ffn_ms_avg=616.225`, and
`qwen36_ffn_int4_expert_direct_gather_stage5=2318.450 ms` across 160 layer
calls, while the command-buffer GPU attribution for the direct gather command
was only `55.965 ms`. That confirms the direct original-buffer gather is still
wait/residency dominated on this model. Current divergence taps put the first
packed/default checksum split in layer 33 FFN on token 1, with shared output and
top-k routing matching but routed-expert arithmetic drifting before the final
add. The useful next FFN direction is therefore two-track: keep packed/static
residency experiments behind the diagnostic family-parity report, and fix
standalone routed-expert parity against the default/full-native path before any
packed-family optimization can graduate.

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
`[qwen36-moe mps-expert-pilot]` row and bench perf JSON schema v9 records it as
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
