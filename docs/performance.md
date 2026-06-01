# Performance Headlines

Quick headline decode throughput for the validated platform/model/options
surface. Numbers are steady-state single-sequence decode unless noted. Lower
`ms/tok` is better; `tok/s` is `1000 / ms/tok`.

For methodology, stage attribution, historical optimization notes, long-context
runs, and runtime-feature deltas, see
[detailed_performance.md](detailed_performance.md).

## HIP / AMD

### `gfx1100` — Radeon RX 7900 XTX, 24 GiB

| Model | BF16 | INT4 | FP8 runtime | KV-FP8 |
|---|---:|---:|---:|---:|
| qwen3.5-0.8b | 8.0 ms / 125.0 tok/s | 10.0 ms / 100.0 tok/s | 10.0 ms / 100.0 tok/s | 85.0 ms / 11.8 tok/s |
| qwen3.5-2b | 11.0 ms / 90.9 tok/s | 11.0 ms / 90.9 tok/s | 15.0 ms / 66.7 tok/s | 126.0 ms / 7.9 tok/s |
| qwen3.5-4b | 21.0 ms / 47.6 tok/s | 15.0 ms / 66.7 tok/s | 30.0 ms / 33.3 tok/s | 223.0 ms / 4.5 tok/s |
| qwen3.5-9b | 32.0 ms / 31.3 tok/s | 26.0 ms / 38.5 tok/s | 48.0 ms / 20.8 tok/s | 347.0 ms / 2.9 tok/s |
| gemma4-e2b | 28.0 ms / 35.7 tok/s | 34.0 ms / 29.4 tok/s | 36.0 ms / 27.8 tok/s | 29.0 ms / 34.5 tok/s |
| gemma4-e4b | 46.0 ms / 21.7 tok/s | 49.0 ms / 20.4 tok/s | 61.0 ms / 16.4 tok/s | 47.0 ms / 21.3 tok/s |
| phi4-mini | 38.3 ms / 26.1 tok/s | 39.7 ms / 25.2 tok/s | 53.1 ms / 18.8 tok/s | 78.1 ms / 12.8 tok/s |
| qwen3.6-35b-a3b | - | 28.3 ms / 35.3 tok/s | - | 28.5 ms / 35.1 tok/s |

### `gfx1150` — Radeon 890M iGPU

| Model | Option | ms/tok | tok/s |
|---|---|---:|---:|
| qwen3.5-0.8b | BF16 | 34 | 29.4 |
| qwen3.5-0.8b | INT4 | 44 | 22.7 |
| qwen3.5-2b | BF16 | 78 | 12.8 |
| qwen3.5-2b | INT4 | 58 | 17.2 |
| qwen3.5-4b | BF16 | 160 | 6.3 |
| qwen3.5-4b | INT4 | 110 | 9.1 |
| qwen3.5-9b | FP8 runtime | 697 | 1.4 |
| gemma4-e2b | BF16 | 246 | 4.1 |
| gemma4-e2b | INT4 | 230 | 4.4 |
| gemma4-e4b | BF16 | 425 | 2.4 |
| phi4-mini | BF16 | 298 | 3.4 |
| phi4-mini | INT4 | 359 | 2.8 |

## CUDA / NVIDIA

### `sm86` — RTX 3090-class

| Model | Option | Prefill | Decode |
|---|---|---:|---:|
| qwen3.5-0.8b | BF16 default | 544 tok/s | 106.7 tok/s |
| qwen3.5-4b | BF16 batch 1 default | 124.7 tok/s | 26.0 tok/s |
| qwen3.5-4b | BF16 batch 2 default | 122.9 tok/s | 15.4 tok/s aggregate |
| qwen3.5-4b | BF16 warmed `--force-kernel-decode` | 101.5 tok/s | 22.0 tok/s |
| llama3.1-8b | INT8 component path | n/a | 38.9 tok/s |
| llama3.1-8b | certified KV INT8, PG-19 4K | n/a | 99.1 ms/tok |

### `sm90` — H100 80GB HBM3

| Model | Option | Prefill | Decode |
|---|---|---:|---:|
| qwen3.5-0.8b | BF16 fast-greedy | 1362.1 tok/s | 32.2 tok/s |
| qwen3.5-4b | BF16 `--force-kernel-decode` | 784.9 tok/s | 26.4 tok/s |
| qwen3.5-4b | BF16 batch 2 | 785.8 tok/s | 11.3 tok/s aggregate |

## Metal / Apple Silicon

### `apple-m4`

| Model | Option | Metric |
|---|---|---:|
| qwen3.5-0.8b | native prefill | 107 ms |
| qwen3.5-0.8b | greedy prefill | 99.7 ms first token |
| qwen3.5-0.8b | replay decode | 84.0 ms/tok |
| qwen3.5-0.8b | component decode prototype | 35.2 ms/tok |

### `apple-m5-max`

| Model | Option | ms/tok | tok/s |
|---|---|---:|---:|
| qwen3.6-35b-a3b | INT4, 6-token prompt + 16 generated tokens | 58.3 | 17.2 |
| qwen3.6-35b-a3b | INT4, 512-token long-context smoke | 72.683 | 13.8 |
| qwen3.6-35b-a3b | INT4, 2048-token long-context smoke | 159.044 | 6.3 |

Apple M5 Max Metal also has correctness coverage for `qwen3-30b-a3b` INT4,
Gemma 4 BF16/INT4, and Phi-4 mini BF16/INT4/FP8-runtime. Those lanes are
tracked in [supported-matrix.md](supported-matrix.md); only Qwen3.6 has a
published headline performance row today.

## Runtime Options

| Feature | Canonical workload | Headline effect |
|---|---|---|
| Qwen3.6 KV-FP8 | `qwen3.6-35b-a3b` INT4 on `gfx1100` | 28.3 -> 28.5 ms/tok; effectively free, mainly memory headroom |
| Qwen3.5 KV-FP8 | `qwen3.5-9b` INT4 on `gfx1100` | 26 -> 347 ms/tok; replay-prefill path, memory feature not throughput feature |
| VMM | `qwen3.6-35b-a3b` INT4, 8192-token context on `gfx1100` | enables a workload that otherwise exceeds 24 GiB |
| SpecPrefill cosine | `qwen3.5-9b` BF16, 1353-token prompt on `gfx1100` | 4941 -> 2385 ms TTFT; 2.07x faster |
| DFlash B=3 | `qwen3.5-9b` INT4 on `gfx1100` | roughly 32 -> 12 ms/step effective; 2.5-3x faster |

See [detailed_performance.md#runtime-feature-impact](detailed_performance.md#runtime-feature-impact)
for the full runtime-feature table and caveats.
