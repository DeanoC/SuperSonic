# Performance

Measured decode throughput for the shipping kernels. Numbers are steady-state
tokens/second, single-sequence (`--batch-size 1`) unless noted, recorded with
a 16-token generation on the 6-token `"The quick brown fox jumps over"` prompt.

If you reproduce these and get materially different results, please open an
issue with your GPU arch, ROCm/CUDA versions, and the exact command line.

## HIP — `gfx1100` (AMD Radeon RX 7900 XTX, 24 GiB)

Discrete dGPU; 96 CUs, RDNA3 WMMA. The full quant matrix (BF16, INT4 GPTQ,
FP8 runtime, FP8 KV cache) is supported across every shipping model on
this arch. Measurements recorded 2026-04-30 at the
[`gemma4-fp8-runtime`](https://github.com/DeanoC/SuperSonic/pull/51) merge,
6-token prompt, 16-token generation, single sequence, `--batch-size 1`.
Each cell is `ms/step` from the runner's `[result] ms_per_step=N` /
`ms_per_tok=N` line after one warm-up run; reproduce with
`tests/gfx1100/bench_matrix.sh`.

| Model           | BF16  | INT4  | FP8r  | KV-FP8 |
|-----------------|------:|------:|------:|-------:|
| qwen3.5-0.8b    |   8   |  10   |  10   |   85¹  |
| qwen3.5-2b      |  11   |  11   |  15   |  126¹  |
| qwen3.5-4b      |  21   |  15   |  30   |  223¹  |
| qwen3.5-9b      |  32   |  26   |  48   |  347¹  |
| gemma4-e2b      |  33   |  36   |  40   |   33   |
| gemma4-e4b      |  54   |  51   |  66   |   52   |
| phi4-mini       |  38.5 |  40.2 |  45.9 |   78.0 |

¹ Qwen 3.5 `--kv-fp8` falls back to a *replayed-prefill* decode path
("`single-sequence CUDA KV-FP8 uses replayed GPU prefill for
correctness`"). The decode kernel itself is fine; the slow column is
the per-step prefill replay needed to keep the FP8 KV cache
self-consistent. KV-FP8 on Qwen is currently a memory feature
(headroom for longer contexts), not a throughput feature. Gemma 4 's
`--kv-fp8` is wired into the persistent kernel directly and is
~free vs BF16.

### Translated to tokens/sec

| Model           | BF16   | INT4   | FP8r   | KV-FP8 |
|-----------------|-------:|-------:|-------:|-------:|
| qwen3.5-0.8b    | 125.0  | 100.0  | 100.0  |  11.8  |
| qwen3.5-2b      |  90.9  |  90.9  |  66.7  |   7.9  |
| qwen3.5-4b      |  47.6  |  66.7  |  33.3  |   4.5  |
| qwen3.5-9b      |  31.3  |  38.5  |  20.8  |   2.9  |
| gemma4-e2b      |  30.3  |  27.8  |  25.0  |  30.3  |
| gemma4-e4b      |  18.5  |  19.6  |  15.2  |  19.2  |
| phi4-mini       |  26.0  |  24.9  |  21.8  |  12.8  |

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

| Stage          | ms/step  |  tok/s  | share |
|----------------|---------:|--------:|------:|
| chain (40 L)   |    24.72 |    40.5 |   93% |
| ↳ FFN (40 L)   |    11.42 |    87.6 |   43% |
| ↳ linear-attn (30 L) | 12.29 |  81.4 |   46% |
| ↳ full-attn (10 L)   |  2.46 |   406 |    9% |
| lm_head        |     1.53 |   653.6 |    6% |
| sample/detok   |     0.27 |  3703.7 |    1% |
| **total**      | **26.53**|**37.7** | 100%  |

Per-stage `tok/s` is `1000 / ms_per_step` — the throughput that stage
would sustain if it were the only cost. The headline rate is the
total row: roughly **37.7 tok/s** on greedy decode (production async
dispatch; numbers above include the per-step sync needed for the
chain breakdown to be accurate, which costs ~1.8 ms).

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
- **Persistent megakernel** (Phase 3 in the roadmap) is now a
  smaller wedge than originally projected, since PR #80 already
  reclaimed ~80 syncs/token of launch overhead. Folding the 80 launches
  into one cooperative kernel still has architectural value as a
  prerequisite for speculative-decode verification but the absolute
  ms reclaim is now ~1-2 ms.
- **Speculative decode** is the realistic path to 100+ tok/s — needs a
  draft head (MTP/Eagle) and a verification kernel that batches K
  candidates × layers in one launch.

INT4 weight + scratch on the 35B-A3B bake: ~17 GiB on disk, ~21 GiB
runtime including KV cache at the default context. Within the 24 GiB
budget. Calibration needs more host RAM than typical 7900 XTX rigs
carry, so the bake is produced on a bigger box and distributed via
GitHub releases (see [bake-distribution.md](bake-distribution.md));
consumers pull it automatically on first run.

Reproduce:

```bash
cargo build --release --bin supersonic
./target/release/supersonic --model qwen3.6-35b-a3b \
  --model-dir /path/to/Qwen3.6-35B-A3B-FP8 \
  --prompt "The quick brown fox jumps over" \
  --max-new-tokens 16 --emit-stage-timings
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
