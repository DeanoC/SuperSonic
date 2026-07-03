# Testing

End-to-end test runner, prerequisites, and notes on adding tests for a
new machine. For unit tests run via `cargo test`, see the per-crate
README files (`crates/runner/`, `crates/kernel-ffi/`, etc.).
For repeatable performance, Lucebox, external-reference, and profiler commands,
see [benchmarks.md](benchmarks.md).

Support lanes with named gates are being captured in
[`support/matrix.toml`](../support/matrix.toml). Run
`python3 tools/check-support-matrix.py` after changing supported architectures,
gate scripts, or benchmark references.

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

# gfx1201 (RDNA 4 / R9700) — starter RDNA4 lane
HIP_VISIBLE_DEVICES=1 ./tests/gfx1201/run_matrix.sh
QWEN36_27B_MODEL_DIR=/path/to/supersonic-qwen36-27b-lucebox \
  HIP_VISIBLE_DEVICES=1 ./tests/gfx1201/run_matrix.sh

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

### FLM Main-Path Smoke

The Qwen3.6 27B dense FLM path treats the `.flm` file as the complete model
source. A runnable no-HF artifact must validate under geo-quant's
`runnable-no-hf` profile before it is used for SuperSonic smoke tests:

```bash
/home/deano/.config/superpowers/worktrees/geo-quant/flm-export/.venv-rocm/bin/python \
  -m geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  --profile runnable-no-hf \
  --verify-payload-hashes
```

Then run the storage upload and runner smoke gates:

```bash
SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p model-store flm_qwen36_27b_direct_views_upload_to_hip -- --nocapture

SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

The runner smoke asserts that config, tokenizer, and weights come from FLM,
that BLAKE3 verification is enabled on the single FLM source open, that weights
load from the already-open source, and that no `[fetch]` or `[bake]` path is
entered.

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

For HIP `gfx1201`, `tests/gfx1201/run_matrix.sh` is the bring-up gate. It
builds a dual `gfx1100,gfx1201` HIP binary, runs the RDNA4 WMMA/int4 kernel
harness, then runs a short Qwen3.6 27B Q4KM-GPTQ smoke and the Lucebox/DFlash
smoke when local artifacts are available. Qwen3.5 rows in
`supported-matrix.md` should stay TBM until that script grows per-model token
checks for the local R9700.

For the production inference refactor, the baseline HIP gate is the existing
`tests/gfx1201/run_matrix.sh` starter matrix plus the `gfx1100` Qwen3.6 Lucebox
benchmark recipe in `docs/benchmarks.md`. Runtime/server refactor PRs that touch
DFlash, Qwen3.6-27B loading, kernel groups, or generation scheduling must state
whether they ran this gate, skipped it for lack of local artifacts, or used a
smaller compile/unit-test gate because the change was docs-only.

For Metal specifically, Apple M4 remains the small Qwen3.5 validation lane.
Apple M5 Max is the current large-model correctness lane: it covers Qwen3.5
BF16 smokes, Qwen3/Qwen3.6 MoE INT4 smokes, Gemma 4 BF16/INT4 smokes, and
Phi-4 mini BF16/INT4/FP8-runtime smokes through the component/chained Metal
paths. The corresponding benchmark and external-reference commands live in
[benchmarks.md](benchmarks.md).

To inspect a raw Q4_K_M bake before benchmarking it in SuperSonic, run the
manifest audit. It reads `manifest.json` and `config.json` only, reports every
required Qwen3.5/3.6 MoE projection, and exits `2` only for missing tensors or
unsupported layouts:

```bash
cargo run -p runner --bin qwen36_q4km_manifest_audit -- \
  --model-dir /path/to/qwen3.5-35b-a3b
```

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
