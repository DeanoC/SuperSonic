# Testing

End-to-end test runner, prerequisites, and notes on adding tests for a
new machine. For unit tests run via `cargo test`, see the per-crate
README files (`crates/runner/`, `crates/kernel-ffi/`, etc.).

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
SUPERSONIC_BACKENDS=cuda ./target/release/supersonic --backend cuda \
  --model llama3.1-8b --model-dir /path/to/Meta-Llama-3.1-8B \
  --prompt "Hello" --max-new-tokens 32 --int8
CONTEXTS='4096' SUBTASKS='niah_single niah_multikey niah_multiquery' \
  SAMPLES=1 CONFIG=both TIMEOUT=900 \
  ./tests/sm86/bench_llama31_arxiv_v1_smoke.sh \
  /path/to/Meta-Llama-3.1-8B
CONTEXTS='512' NUM_CHUNKS=1 CONFIG=both \
  ./tests/sm86/bench_llama31_pg19_smoke.sh \
  /path/to/Meta-Llama-3.1-8B
CONTEXTS='4096' NUM_CHUNKS=1 CONFIG=both REFERENCE_SMOKE=1 \
  FAIL_ABOVE_REFERENCE=1 TIMEOUT=1200 \
  ./tests/sm86/bench_llama31_pg19_smoke.sh \
  /path/to/Meta-Llama-3.1-8B

# Apple silicon — Qwen3.5-0.8B Metal bughunt gate
SUPERSONIC_BACKENDS=metal QWEN35_MODEL_DIR=/path/to/Qwen3.5-0.8B ./tests/metal/qwen35_bughunt_gate.sh

# Apple M5 Max large-model Metal coverage
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo test --release -p runner --test metal_large_model_smoke -- --ignored --nocapture

# Apple M5 Max Qwen3.6-MoE INT4 smoke
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo test --release -p runner --test qwen36_moe_metal_smoke -- --ignored --nocapture
```

### Adding tests for a new machine

1. Create `tests/<gpu_arch>/run.sh` (copy an existing one as a starting point)
2. Adjust the model, prompt, thresholds, or add additional test cases
3. The test exercises both the baked and `--no-bake` (safetensors) loading paths

### Test prerequisites

- ROCm/HIP runtime for HIP builds, CUDA toolkit/runtime for CUDA builds, or Apple silicon with Metal for Metal builds
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

For Metal specifically, Apple M4 remains the small Qwen3.5 validation lane.
Apple M5 Max is the current large-model lane: it covers Qwen3.5 BF16 smokes,
Qwen3/Qwen3.6 MoE INT4 smokes, Gemma 4 BF16/INT4 smokes, and Phi-4 mini
BF16/INT4/FP8-runtime smokes through the component/chained Metal paths. The
Qwen3.6-MoE Apple M5 Max performance gate is
`bench-perf --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4`.

## Per-feature parity tests

Several runtime features ship a Rust integration test that shells out
to the `supersonic` binary and asserts last-step logits parity (or text
parity) between dense and feature-on runs. Run them all with:

```bash
cargo test -p runner --release \
    --test specprefill_qwen35_9b_parity \
    --test specprefill_rope_indirect_parity \
    --test specprefill_lookahead_attention_parity \
    --test qwen36_moe_kv_fp8_parity \
    -- --nocapture --test-threads=1
```

Each test self-skips when its required model dirs aren't set in the
environment (e.g. `SUPERSONIC_QWEN35_9B_DIR`,
`SUPERSONIC_QWEN35_0_8B_DIR`, `SUPERSONIC_QWEN36_35B_A3B_DIR`). See
[feature-compatibility.md](feature-compatibility.md) for the full list
of feature flags each test exercises.
