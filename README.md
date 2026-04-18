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
`--bf16` to also publish Qwen BF16 bakes, `--fp8-native` for FP8-native
bakes (Qwen only; source checkpoint must ship FP8 tensors), `--force` to
rebuild, or drop `--upload` to keep the output local.

**9B INT4 note:** `qwen3.5-9b` GPTQ calibration loads the full BF16 model
(~18 GiB) into GPU memory, so it OOMs on ≤16 GiB cards (including gfx1150).
Run it on a box with ≥24 GiB GPU RAM — the small-VRAM consumer then pulls
the resulting bake from the release. Leave `QWEN_9B_DIR` unset to skip.

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

Current behavior on this `sm86` box now depends on whether the path is using replayed prefill for correctness.
With a quick harness pass (`PROMPT_REPEAT=8`, `MAX_NEW_TOKENS=8`, `RUNS=1`):

- `qwen3.5-0.8b`: prefill `199 ms` for 112 prompt tokens (`563 tok/s`), decode `268 ms` for 8 generated tokens (`29.9 tok/s`)
- `qwen3.5-4b` single-sequence: prefill `901 ms` for 112 prompt tokens (`124 tok/s`), decode `7486 ms` for 8 generated tokens (`1.1 tok/s`)
- `qwen3.5-4b --batch-size 2`: prefill `906 ms` for 112 prompt tokens (`124 tok/s`), decode `741 ms` for 16 aggregate generated tokens (`21.6 tok/s`)

So the current CUDA `4B` story on this box is split:

- single-sequence decode is correctness-first and much slower because it replays prefill
- batched decode remains the fast path and is still the better place to do performance work

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
