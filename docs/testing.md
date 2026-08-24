# Testing

The testing contract protects the direct Qwen3.8-27B GQH path, deterministic
greedy generation, and optional NextN/MTP token equivalence. CPU-safe checks do
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
export SUPERSONIC_GQH_GGUF="${SUPERSONIC_GQH_GGUF:-/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf}"
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
Keep `--include-ignored` and `--test-threads=1`: the canonical artifact is
large and these cases are deliberately serialized.

## Failure policy

Unsupported model values, missing sidecars, incompatible GQH headers, missing
GPU artifacts, ambiguous device discovery, and token mismatches are failures.
Only an unconfigured local artifact-dependent test may report a documented
local skip; the configured workflow must fail closed.

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
