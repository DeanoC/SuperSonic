# SuperSonic

Optimized LLM inference with persistent decode megakernels. Each supported (model, backend, GPU) combination gets a hand-tuned kernel — no fallback to generic slow paths.

Currently supports Qwen3.5-0.8B on AMD gfx1150 (RDNA 3.5) via HIP.

## Quick Start

```bash
# Build
cargo build --release

# Run (auto-bakes weights on first run)
cargo run --release --bin supersonic -- \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Hello, world" \
  --max-new-tokens 8
```

On first run, SuperSonic bakes the HuggingFace safetensors into an optimized format at `{model_dir}/.supersonic/v1/`. Subsequent runs load from this baked format for faster startup.

## E2E Tests

Tests are machine-specific — each GPU architecture has its own test script under `tests/`. A test runs the full decode pipeline with PyTorch oracle validation and checks that the output delta is below a threshold.

### Running tests

```bash
# gfx1150 (RDNA 3.5)
./tests/gfx1150/run.sh /path/to/Qwen3.5-0.8B

# Or set the env var once
export SUPERSONIC_MODEL_DIR=/path/to/Qwen3.5-0.8B
./tests/gfx1150/run.sh
```

### Adding tests for a new machine

1. Create `tests/<gpu_arch>/run.sh` (copy an existing one as a starting point)
2. Adjust the model, prompt, thresholds, or add additional test cases
3. The test exercises both the baked and `--no-bake` (safetensors) loading paths

### Test prerequisites

- ROCm/HIP runtime
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
