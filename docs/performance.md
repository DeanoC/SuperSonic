# Performance

Measured decode throughput for the shipping kernels. Numbers are steady-state
tokens/second, single-sequence (`--batch-size 1`) unless noted, recorded with
a 16-token generation on the 6-token `"The quick brown fox jumps over"` prompt.

If you reproduce these and get materially different results, please open an
issue with your GPU arch, ROCm/CUDA versions, and the exact command line.

## HIP — `gfx1150` (AMD Radeon 890M iGPU)

16 CUs, 2.9 GHz core, shared with system memory. Measurements from
2026-04-20 on commit `d91a993` (2× grid oversubscription merged).

| Model              | Quant | ms/tok | tok/s |
|--------------------|-------|-------:|------:|
| qwen3.5-0.8b       | BF16  | 91     | 11.0  |
| qwen3.5-0.8b       | INT4  | 78     | 12.8  |
| qwen3.5-2b         | BF16  | 159    | 6.3   |
| qwen3.5-2b         | INT4  | 118    | 8.5   |
| qwen3.5-4b         | BF16  | 514    | 1.9   |
| qwen3.5-4b         | INT4  | 286    | 3.5   |
| qwen3.5-9b         | FP8   | 697    | 1.4   |

Notes:

- `qwen3.5-0.8b` decode (both BF16 and INT4) routes through the 4B persistent
  megakernel. The native 0.8B decode kernel has a documented page-fault on
  this machine and is not on the shipping path.
- `qwen3.5-9b` INT4 bake runs out of VRAM during GPTQ calibration on 16 GiB
  cards. Consumers pull the released bake from GitHub releases (see
  [bake-distribution.md](bake-distribution.md)); the INT4 runtime itself is
  supported.
- FP8-runtime and FP8-KV paths (`--fp8-runtime`, `--kv-fp8`) are only wired
  for the Qwen family on HIP. Gemma 4 and Phi-4-mini reject both flags.

Gemma 4 and Phi-4-mini timings on gfx1150 are not in the current measurement
set — add them here when next measured.

## CUDA — `sm86` (NVIDIA RTX 3090-class)

24 GB VRAM, 936 GB/s memory bandwidth. Measurements below were refreshed on
2026-04-22 at commit `5a34190`.

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

Notes:

- `qwen3.5-4b` batch-1 CUDA decode now defaults to the kernel path. The older
  replayed-prefill decode path is legacy debugging behavior and must be
  requested explicitly with `--force-replay-decode`.
- The warmed `4B` hero-lane benchmark above also recorded these stage means:
  `full_attn_core=121.1 ms`, `linear_proj=35.1 ms`, `linear_out=78.7 ms`,
  `mlp_gate_up=160.6 ms`, `mlp_down=212.9 ms`.
- ¹ The batched decode figure is aggregate tokens/second across
  `--batch-size 2`.

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

# CUDA / sm86 warmed 4B single-sequence hero lane
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_single.sh \
  /path/to/Qwen3.5-4B
```
