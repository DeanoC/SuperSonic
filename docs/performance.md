# Performance

Measured decode throughput for the shipping kernels. Numbers are steady-state
tokens/second, single-sequence (`--batch-size 1`) unless noted, recorded with
a 16-token generation on the 6-token `"The quick brown fox jumps over"` prompt.

If you reproduce these and get materially different results, please open an
issue with your GPU arch, ROCm/CUDA versions, and the exact command line.

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

24 GB VRAM, 936 GB/s memory bandwidth. Current behavior depends on whether
the path is using replayed prefill for correctness.

Quick checked paths (`PROMPT_REPEAT=8`, `MAX_NEW_TOKENS=8`, `RUNS=1`):

| Model                       | Path                         | Prefill   | Decode     |
|----------------------------|------------------------------|-----------|------------|
| qwen3.5-0.8b              | default (hero)               | 563 tok/s | 29.9 tok/s |
| qwen3.5-4b `--batch-size 2` | default (batched)            | 124 tok/s | 21.6 tok/s |
| qwen3.5-4b `--batch-size 1` | replay-prefill correctness   | 124 tok/s | 1.1 tok/s  |

Warmed native single-stream `4B` hero lane
(`./tests/sm86/bench_qwen4b_single.sh`, `pp533 / tg128`, commit `e5f244d`):

| Model                       | Path                    | Prefill   | Decode     | Persistent |
|----------------------------|-------------------------|-----------|------------|------------|
| qwen3.5-4b `--batch-size 1` | `--force-kernel-decode` | 118.5 tok/s | 15.2 tok/s | 7714 ms    |

CUDA `sm86` tracks detailed kernel-level optimization history for both the
`0.8B` and `4B` hero lanes in
[qwen35-sm86-optimization.md](qwen35-sm86-optimization.md).

## How to reproduce

```bash
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
```
