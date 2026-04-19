# SuperSonic

Optimized LLM inference with persistent decode megakernels. Each supported (model, backend, GPU) combination gets a hand-tuned kernel — no fallback to generic slow paths.

Currently supports:

- Qwen3.5-0.8B and Qwen3.5-4B on AMD `gfx1150` (RDNA 3.5) via HIP
- Gemma 4 E2B and E4B on AMD `gfx1150` (RDNA 3.5) via HIP
- Qwen3.5-0.8B and Qwen3.5-4B on NVIDIA `sm86` (RTX 3090-class) via CUDA

CUDA v1 is BF16-first. `--int4` and `--fp8-runtime` remain unsupported on CUDA. A hidden unstable CUDA `--kv-fp8` debug path exists for targeted validation work on `qwen3.5-4b`, but it is not part of the public supported surface.

## Supported Matrix

| Backend | GPU arch | Models | Status |
| --- | --- | --- | --- |
| HIP | `gfx1150` | `qwen3.5-0.8b`, `qwen3.5-4b` | validated |
| HIP | `gfx1150` | `gemma4-e2b`, `gemma4-e4b` | upstream validated |
| CUDA | `sm86` | `qwen3.5-0.8b`, `qwen3.5-4b` | validated |

CUDA support is currently a narrow v1 surface:

- hand-maintained CUDA sources only; no generic fallback backend
- BF16 decode path only
- validated on NVIDIA `sm86` hardware (RTX 3090-class)
- validated for both baked weights and direct `--no-bake` safetensors loads
- hidden unstable CUDA `--kv-fp8` debug coverage currently exists only for `qwen3.5-4b` on `sm86`

CUDA v1 does not currently support:

- `--int4`
- `--fp8-runtime`

CUDA KV-FP8 is currently a debug-only surface:

- hidden behind `--allow-unstable-cuda-kv-fp8`
- validated only on `qwen3.5-4b` / `sm86`
- exercised on the real persistent kernel path with `--force-kernel-decode`
- checked against the CPU oracle, not presented as a general CUDA v1 feature
- the BF16 KV sidecar is a bring-up aid for parity-sensitive reads/debugging, not part of normal BF16 CUDA runs
- normal CUDA BF16 runs do not allocate or use the sidecar
- the sidecar can be disabled for A/B work with `SUPERSONIC_DEBUG_DISABLE_KV_FP8_BF16_SIDECAR=1`

CUDA batched decode is currently validated only for:

- `qwen3.5-4b`
- NVIDIA `sm86`
- `--batch-size 2`

`--gpu-validate` is now part of the checked `sm86` debug surface for `qwen3.5-0.8b` and `qwen3.5-4b`.
It is intentionally slower than normal decode: each step replays the full token history through the validated GPU prefill path and compares the resulting last-token logits against native decode.
For `qwen3.5-4b` single-sequence CUDA decode on `sm86`, the current correctness-first path also uses replayed GPU prefill per decode step.
That keeps longer prompt behavior closer to the oracle, but it is materially slower than the batched CUDA decode path.

## Quick Start

```bash
# Build with the backend(s) you want compiled in.
# Omit SUPERSONIC_BACKENDS to build the default configured backend set.
SUPERSONIC_BACKENDS=cuda cargo build --release

# Run (auto-bakes weights on first run)
SUPERSONIC_BACKENDS=cuda cargo run --release --bin supersonic -- \
  --backend cuda \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Hello, world" \
  --max-new-tokens 8
```

On first run, SuperSonic bakes the HuggingFace safetensors into an optimized format at `{model_dir}/.supersonic/v1/`. Subsequent runs load from this baked format for faster startup.

If a local bake is missing and one is available in the repo's GitHub releases,
SuperSonic can download that package instead of rebuilding it locally. Pass
`--no-download` to disable network fetches. See [docs/bake-distribution.md](docs/bake-distribution.md)
for the producer workflow and release layout.

## Producing And Publishing Bakes

On a machine with enough VRAM/RAM for GPTQ calibration, `oracle/bake_all.sh`
can bake and optionally upload every configured model in one pass. Unset model
directory env vars are skipped, so this is the reference "big producer box"
command:

```bash
pip install -r oracle/requirements-upload.txt
gh auth login

QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
QWEN_2B_DIR=/models/Qwen3.5-2B \
QWEN_4B_DIR=/models/Qwen3.5-4B \
QWEN_9B_DIR=/models/Qwen3.5-9B \
GEMMA_E2B_DIR=/models/gemma-4-E2B \
GEMMA_E4B_DIR=/models/gemma-4-E4B \
./oracle/bake_all.sh --upload
```

By default this produces INT4-GPTQ bakes for every configured model. Add
`--bf16` to also publish Qwen BF16 bakes, `--force` to rebuild, or drop
`--upload` to keep the output local.

## CUDA

### Build requirements

- NVIDIA driver + CUDA runtime/toolkit usable from the build machine
- Rust toolchain able to build this repo
- Python 3 with `torch` and `transformers` for oracle validation
- local model weights for `Qwen3.5-0.8B` and/or `Qwen3.5-4B`

### Validated commands

These are the checked CUDA `sm86` commands:

```bash
# Qwen3.5-0.8B
SUPERSONIC_BACKENDS=cuda TIMEOUT=600 ./tests/sm86/run.sh /path/to/Qwen3.5-0.8B

# Qwen3.5-0.8B long-context CPU-oracle check
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 ./tests/sm86/run_long.sh /path/to/Qwen3.5-0.8B

# Qwen3.5-4B
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 CORPUS_TIMEOUT=600 ./tests/sm86/run_4b.sh /path/to/Qwen3.5-4B

# Qwen3.5-4B long-context CPU-oracle check
SUPERSONIC_BACKENDS=cuda TIMEOUT=1200 CORPUS_TIMEOUT=1200 ./tests/sm86/run_4b_long.sh /path/to/Qwen3.5-4B

# Qwen3.5-4B batched decode
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 CORPUS_TIMEOUT=600 ./tests/sm86/run_batch.sh /path/to/Qwen3.5-4B

# Combined wrapper
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run_all.sh \
  /path/to/Qwen3.5-0.8B \
  /path/to/Qwen3.5-4B
```

Each `sm86` script currently validates:

- baked `.supersonic/v1` loading
- direct `--no-bake` loading
- oracle logit deltas
- replay-based `--gpu-validate` deltas and token agreement
- golden corpus coverage

`tests/sm86/run_batch.sh` adds `qwen3.5-4b --batch-size 2` coverage on the same `sm86` target.
`tests/sm86/run_fast_greedy.sh` checks that the CUDA fast-greedy 0.8B path
matches the legacy host-logits sampling path on short, medium, and long prompts.
`tests/sm86/run_negative.sh` covers unsupported CUDA v1 flags and explicit failure modes.
The default short/medium `sm86` scripts still validate against the CUDA oracle.
The long-context scripts use the CPU oracle on this box, because that is the stable reference
for longer `4B` prompts today.
`tests/sm86/run_long.sh` and `tests/sm86/run_4b_long.sh` add explicit long-context coverage
against the CPU oracle using focused long-only golden corpora.

The hidden CUDA KV-FP8 debug surface is validated separately with commands like:

```bash
target/release/supersonic --backend cuda --oracle-device cpu \
  --model qwen3.5-4b --model-dir /path/to/Qwen3.5-4B \
  --prompt '中国的首都是' --max-new-tokens 8 \
  --batch-size 2 --kv-fp8 --allow-unstable-cuda-kv-fp8 --force-kernel-decode --validate

CORPUS_TIMEOUT=1200 tests/corpus/run_golden.sh \
  qwen3.5-4b /path/to/Qwen3.5-4B tests/corpus/golden_4b_batch2.json \
  target/release/supersonic --backend cuda --oracle-device cpu \
  --batch-size 2 --kv-fp8 --allow-unstable-cuda-kv-fp8 --force-kernel-decode
```

### Benchmark baseline

For CUDA baseline measurements on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench.sh \
  /path/to/Qwen3.5-0.8B \
  /path/to/Qwen3.5-4B
```

For a warmed Lucebox-style `qwen3.5-0.8b` parity run on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen08.sh \
  /path/to/Qwen3.5-0.8B
```

That harness defaults to batch-1 BF16, a roughly `pp520` prompt target,
`tg128`, `10` warmup runs, `20` timed runs, and prints aggregated native decode
stage timings from `--emit-stage-timings`.

For a warmed single-sequence native-kernel `qwen3.5-4b` run on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_single.sh \
  /path/to/Qwen3.5-4B
```

That harness forces `--force-kernel-decode` so the run measures the native
single-sequence `4B` kernel instead of the default replayed-prefill
correctness path.

The current `qwen3.5-0.8b` CUDA `sm86` optimization record, benchmark progression,
remaining gap to Lucebox, and carry-forward process for the other supported Qwen3.5
CUDA models are tracked in [docs/qwen35-sm86-optimization.md](/workspace/SuperSonic/docs/qwen35-sm86-optimization.md).

For a one-token Nsight Compute pass over the non-4B persistent decode kernel on
`sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/profile_qwen08_decode.sh \
  /path/to/Qwen3.5-0.8B
```

Set `PROFILE_MODE=fast` to disable the hero path while keeping CUDA fast-greedy,
or `PROFILE_MODE=legacy` to force the old host-logits decode path.

Current behavior on this `sm86` box now depends on whether the path is using replayed prefill for correctness.
With a quick harness pass (`PROMPT_REPEAT=8`, `MAX_NEW_TOKENS=8`, `RUNS=1`):

- `qwen3.5-0.8b`: prefill `199 ms` for 112 prompt tokens (`563 tok/s`), decode `268 ms` for 8 generated tokens (`29.9 tok/s`)
- `qwen3.5-4b` single-sequence: prefill `901 ms` for 112 prompt tokens (`124 tok/s`), decode `7486 ms` for 8 generated tokens (`1.1 tok/s`)
- `qwen3.5-4b --batch-size 2`: prefill `906 ms` for 112 prompt tokens (`124 tok/s`), decode `741 ms` for 16 aggregate generated tokens (`21.6 tok/s`)

So the current CUDA `4B` story on this box is split:

- single-sequence decode is correctness-first and much slower because it replays prefill
- batched decode remains the fast path and is still the better place to do performance work

There is now also an explicit native single-sequence hero lane for `4B`
behind `--force-kernel-decode`. The exact lane is:

- CUDA + `sm86`
- `qwen3.5-4b`
- BF16
- baked load
- `--force-kernel-decode`
- `--batch-size 1`
- warmed `pp533 / tg128`
- hero attention guard `B == 1 && bs == 256 && hd == 256`

The current warmed result on this box comes in at roughly:

- prefill `4484 ms` (`119 tok/s`)
- decode `12097 ms` (`10.6 tok/s`)

The first kept single-stream `4B` CUDA pass on this lane removed
unconditional full-attention trace-buffer writes from the hot path and left
those writes enabled only for the explicit trace workflow. The next two kept
passes then tightened the single-stream BF16 attention-core inner loop: first
by moving to packed two-dimension BF16 score/value work inside the existing
`B == 1` schedule, and then by raising that packed path to four dimensions per
active thread. The latest kept pass then stopped the idle threads in that same
hero branch from doing useless BF16 query staging plus zero-value score/softmax
work. The next kept pass then moved off attention entirely once the temporary
split showed `linear_core` was now mostly recurrent-state traffic: it fused the
serial recurrent update from four state walks down to two and reduced the
warmed single-lane linear core from about `3471 ms` to `3275 ms` on this
machine while leaving the attention core flat at about `4600 ms`. The latest
kept pass then staged the normalized per-head `q/k` vectors for that serial
linear-attention loop into shared memory once per head pair, trimming the
warmed single-lane linear core again from about `3275 ms` to `3236 ms`. The
next kept pass temporarily split the single-stream attention core and showed
the real remaining cost was score-side work, not value accumulation, then
collapsed the hero branch from a two-wave `64 x 4` score reduction to a
one-wave `32 x 8` mapping. That cut warmed single-lane full-attention core
from about `4600 ms` to `4166 ms` and improved warmed decode from about
`12463 ms` to `12097 ms` on this machine.

That lane is intended for Lucebox-style single-stream optimization work; the
validated production throughput lane remains `qwen3.5-4b --batch-size 2`.

## E2E Tests

Tests are machine-specific — each GPU architecture has its own test script under `tests/`. A test runs the full decode pipeline with PyTorch oracle validation and checks that the output delta is below a threshold.

### Running tests

```bash
# gfx1150 (RDNA 3.5) — Qwen3.5-0.8B
./tests/gfx1150/run.sh /path/to/Qwen3.5-0.8B

# gfx1150 (RDNA 3.5) — Qwen3.5-4B
./tests/gfx1150/run_4b.sh /path/to/Qwen3.5-4B

# Or set env vars once
export SUPERSONIC_MODEL_DIR=/path/to/Qwen3.5-0.8B
export SUPERSONIC_MODEL_DIR_4B=/path/to/Qwen3.5-4B
./tests/gfx1150/run.sh
./tests/gfx1150/run_4b.sh

# sm86 (RTX 3090-class) — Qwen3.5-0.8B / 4B
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run.sh /path/to/Qwen3.5-0.8B
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run_4b.sh /path/to/Qwen3.5-4B
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run_batch.sh /path/to/Qwen3.5-4B
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run_all.sh /path/to/Qwen3.5-0.8B /path/to/Qwen3.5-4B
```

### Adding tests for a new machine

1. Create `tests/<gpu_arch>/run.sh` (copy an existing one as a starting point)
2. Adjust the model, prompt, thresholds, or add additional test cases
3. The test exercises both the baked and `--no-bake` (safetensors) loading paths

### Test prerequisites

- ROCm/HIP runtime for HIP builds, or CUDA toolkit/runtime for CUDA builds
- Python 3 with `torch` and `transformers` (for oracle)
- Model weights downloaded locally

### Configuration

```bash
# Max acceptable logit divergence from oracle (default: 1.0)
MAX_DELTA_THRESHOLD=0.5 ./tests/gfx1150/run.sh /path/to/model

# Per-test timeout in seconds (default: 120)
TIMEOUT=180 ./tests/gfx1150/run.sh /path/to/model
```

### Known issues

The persistent decode megakernel can occasionally hang the GPU at 100% utilization. The test script has a timeout (default 120s) and will report failure rather than blocking forever. If this happens you may need to reset the GPU (`rocm-smi --resetgpu`) or reboot before re-running.

For CUDA specifically, treat `sm86` as the validated target for now. Other NVIDIA architectures may work, but they are not yet part of the checked support matrix.
