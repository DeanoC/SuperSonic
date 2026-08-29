# Testing

The testing contract protects the direct Qwen3.8-27B GQH path, deterministic
greedy generation, optional NextN/MTP token equivalence, and DFlash2 semantic
quality. CPU-safe checks do
not require an accelerator. Artifact-dependent checks accept explicit paths
locally; the configured self-hosted workflow fails during preflight instead of
turning a missing artifact into a pass.

## CPU pull-request gate

Run the formatting, compile, manifest, documentation, and Python contract
checks from the repository root:

```bash
git diff --check
cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
python3 tools/check-support-matrix.py
python3 tools/check-kernel-groups.py
python3 tools/check-tool-inventory.py
python3 tools/check-retained-source-terms.py
python3 tools/check-active-docs.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Focused retained-path tests include the GQH codec, header and pointer
registry, Qwen3.8 MTP acceptance policy, CLI rejection of unsupported values,
and startup validation of the model-directory/GGUF pair. They are safe to run
without a visible GPU.

## `gfx1201` artifact gate

The self-hosted runner selects one physical R9700 device, masks it to logical
device zero, waits for three idle samples, and runs the artifact checks with
one Rust test thread. Preflight must pass before the release build or any
large artifact is loaded:

```bash
set -euo pipefail
export HIP_ARCH=gfx1201
export RUST_TEST_THREADS=1
export SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1
export SUPERSONIC_GQH_GGUF="${SUPERSONIC_GQH_GGUF:-/home/deano/models/qwen38-gqh-shaped.gguf}"
export SUPERSONIC_QWEN38_MODEL_DIR="${SUPERSONIC_QWEN38_MODEL_DIR:-/data/models/Qwen3.8-27B}"
export SUPERSONIC_GQH_8192_GGUF="${SUPERSONIC_GQH_8192_GGUF:-/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq-8192.gguf}"
python3 tools/check-qwen38-artifacts.py --require-8192
RUST_TEST_THREADS=1 cargo test --release -p kernel-ffi --lib 'gqh::tests::' \
  -- --include-ignored --test-threads=1 --nocapture
RUST_TEST_THREADS=1 cargo test --release -p qwen38 --test qwen38_gqh_gguf_crawl \
  -- --include-ignored --test-threads=1 --nocapture
RUST_TEST_THREADS=1 cargo test --release -p supersonic-runtime \
  --test qwen38_gqh_decode_rung11 -- --include-ignored --test-threads=1 --nocapture
```

The gate covers official GQH vectors, tensor geometry, deterministic component
decode, chat-template generation, and the ordinary-versus-MTP token sequence.
DFlash2 artifact tests additionally compare the draft-forward, target-capture,
rollback, and generation-limit state paths.
Keep `--include-ignored` and `--test-threads=1`: the canonical artifact is
large and these cases are deliberately serialized.

## Benchmark acceptance

The artifact gate is a prerequisite for the reproducible benchmark candidate.
After it passes, use the exact commands in [benchmarks](benchmarks.md). The
implemented CLI requires the captured static GPU provenance JSON; it is not an
optional or planned flag:

```bash
set -euo pipefail
HIP_ARCH=gfx1201 cargo build --release --workspace
run_id="quick-manual-$(date +%s)"
RUST_TEST_THREADS=1 timeout --foreground 660s \
  python3 tools/supersonic-bench.py run \
  --suite quick \
  --model-dir "$SUPERSONIC_QWEN38_MODEL_DIR" \
  --artifact "$SUPERSONIC_GQH_GGUF" \
  --artifact-semantic-id qwen3.8-27b-gqh-q3kxl-hf-91bc7e33 \
  --artifact-quantization GQH-Q3KXL \
  --tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 \
  --chat-template-sha256 c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 \
  --physical-gpu "$SUPERSONIC_R9700_GPU_ID" \
  --gpu-static-json target/benchmarks/manual/amd-smi-provenance.json \
  --rocm-version-file target/benchmarks/manual/rocm-driver-version.txt \
  --hip-version-file target/benchmarks/manual/hipcc-version.txt \
  --logical-gpu "$SUPERSONIC_GPU_LOGICAL" \
  --gpu-arch "$SUPERSONIC_R9700_GPU_ARCH" \
  --device "$SUPERSONIC_DEVICE" \
  --chat \
  --clock-policy locked \
  --gpu-clock-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_MHZ" \
  --gpu-clock-tolerance-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_TOLERANCE_MHZ" \
  --memory-clock-mhz "$SUPERSONIC_BENCHMARK_MEMORY_CLOCK_MHZ" \
  --power-cap-watts "$SUPERSONIC_BENCHMARK_POWER_CAP_WATTS" \
  --performance-level "$SUPERSONIC_BENCHMARK_PERFORMANCE_LEVEL" \
  --seed 1 \
  --run-id "$run_id" \
  --output target/benchmarks/candidate
python3 tools/supersonic-bench.py validate --publishable \
  "target/benchmarks/candidate/$run_id"
```

The quick harness has a 600-second (10-minute) hard budget, while its
workflow job cap is 30 minutes. The explicitly manual full workflow uses the
same evidence contract with `--suite full` and `--peer-artifact`; it has a
20,700-second minimum inside a 21,600-second hard budget and its workflow cap
is 450 minutes. Full measurements run in complete balanced rounds and reserve
the budget tail for the last round and manifest finalization.
These caps include checkout, static provenance, idle checks, artifact
preflight, and the release build. A full run is never silently substituted for
the quick gate.

The executable suites record only fresh-process `cold-load` evidence.
`warm-resident` fails preflight until same-process adapter reuse is verified;
prefix-cache cases remain unsupported until their transitions are verified.
`uncontrolled-clocks` results may be retained for diagnosis but are excluded
from headline and peer speedup claims.
The candidate directory is diagnostic until a reviewer checks raw samples,
the validator's raw-sample value/count/completeness checks, the renderer's
deterministically derived median/MAD/count, correctness, cache/clock evidence,
artifact digests, and comparability, then promotes only portable records in a
code-reviewed change.

## Failure policy

Unsupported model values, missing sidecars, incompatible GQH headers, missing
GPU artifacts, ambiguous device discovery, and token mismatches are failures.
Only an unconfigured local artifact-dependent test may report a documented
local skip; the configured workflow must fail closed.

The benchmark harness distinguishes `complete`, `failed`, and `incomplete`.
Budget exhaustion, interruption, timeout, missing samples, quality mismatch,
or an evidence violation preserves completed diagnostics but makes the bundle
ineligible for `validate --publishable`. A partial quick or full run is never
published as a complete aggregate. Performance telemetry is report-only; a
deterministic quality failure blocks immediately.

### GPU integrity fail-stop policy

The HIP bridges own process-global GQH metadata, streams, events, scratch
allocations, dequant resources, and decode graphs. If a device switch,
synchronization, event/stream operation, free/destroy, or device restoration
fails after one of those objects can reference a model allocation, the bridge
logs the operation, HIP status, and device ordinal and deliberately aborts the
process. Continuing would let Rust field drops free memory that an in-flight
kernel or process-global bridge state could still dereference. Validation,
allocation, and other failures before that ownership/async boundary remain
ordinary returned errors.

The deterministic developer-only death test compiles the injection seam out of
normal builds (only the exact value `SUPERSONIC_GPU_FAILURE_TESTS=1` enables it;
unset or `=0` leaves it out) and verifies the tracked-wire unregister boundary:

```bash
SUPERSONIC_GPU_FAILURE_TESTS=1 HIP_ARCH=gfx1201 \
  cargo test -p kernel-ffi --lib \
  'gqh::tests::fatal_' -- --nocapture
```
