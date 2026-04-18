# SuperSonic

Optimized LLM inference with persistent decode megakernels. Each supported (model, backend, GPU) combination gets a hand-tuned kernel — no fallback to generic slow paths.

Currently supports Qwen3.5-0.8B and Qwen3.5-4B on AMD gfx1150 (RDNA 3.5) via HIP.

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

If a local bake isn't possible (e.g. INT4 calibration OOMs on small-VRAM hardware), SuperSonic auto-downloads a pre-baked package from the repo's GitHub releases. Pass `--no-download` to disable. See `docs/bake-distribution.md` for how producers publish bakes with `oracle/upload_bake.py`.

## Producing and publishing bakes (big-box producer)

On a machine with enough VRAM/RAM for GPTQ calibration, run `oracle/bake_all.sh`
to bake everything you have weights for and upload each to the GitHub release
as it finishes. Unset env vars are silently skipped, so this command is the
"everything" reference — set whichever dirs you have and copy-paste:

```bash
pip install -r oracle/requirements-upload.txt   # one-time: zstandard
gh auth login                                   # one-time: gh CLI

QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
QWEN_2B_DIR=/models/Qwen3.5-2B \
QWEN_4B_DIR=/models/Qwen3.5-4B \
QWEN_9B_DIR=/models/Qwen3.5-9B \
GEMMA_E2B_DIR=/models/gemma-4-E2B \
GEMMA_E4B_DIR=/models/gemma-4-E4B \
./oracle/bake_all.sh --upload
```

By default this produces INT4-GPTQ bakes for every configured model. Add
`--bf16` to also publish Qwen BF16 bakes, `--force` to re-bake, or drop
`--upload` to bake locally without publishing. Both Python bakers checkpoint
per-layer, so an interrupted run resumes cleanly. The producer can be any
arch — a CUDA machine's output loads on a HIP consumer byte-for-byte (see
`docs/bake-distribution.md`).

As new models are added to the registry, extend the `MODELS` table at the top
of `oracle/bake_all.sh` so this command keeps working as the one-stop batch.

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
