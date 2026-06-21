# SuperSonic

Optimized LLM inference with persistent decode megakernels. Each supported
(model, backend, GPU) combination gets a hand-tuned kernel — no fallback to
generic slow paths.

Measured decode throughput: see [docs/performance.md](docs/performance.md).

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

On first run, SuperSonic bakes the HuggingFace safetensors into an optimized
format at `{model_dir}/.supersonic/v1/`. Subsequent runs load from this baked
format. If a local bake is missing, SuperSonic can download a published bake
from the repo's GitHub releases — see
[docs/bake-distribution.md](docs/bake-distribution.md). Pass `--no-download`
to disable network fetches.

## Documentation

- **[Supported matrix](docs/supported-matrix.md)** — model × quant × arch
  validated combinations with per-arch caveats.
- **[Feature compatibility](docs/feature-compatibility.md)** — runtime
  features (KV-FP8, VMM, SpecPrefill, DFlash, MoE prefetch, certified-KV)
  with the feature×feature grid and a picker for common use cases.
- **[Performance](docs/performance.md)** — measured decode throughput per
  (model, arch, quant), summarized as headline matrices.
- **[Detailed performance](docs/detailed_performance.md)** — methodology,
  attribution tables, historical logs, and runtime-feature impact.
- **[Benchmarks](docs/benchmarks.md)** — repeatable test, benchmark,
  Lucebox, and profiling commands, including the Qwen3.6 27B 100 tok/s run.
- **[Build and run](docs/build-and-run.md)** — per-backend build commands
  and the validated `supersonic` invocation set.
- **[OpenAI-compatible server](docs/server.md)** — `supersonic-serve`
  endpoints and harness compatibility notes.
- **[Producing release bakes](docs/bake-distribution.md)** — how to
  produce, sign, and publish bakes for a new model variant.
- **[Testing gates](docs/testing.md)** — E2E test runner, prerequisites,
  architecture scripts, and per-feature parity tests.
- **[Development architecture](docs/development/repo-architecture.md)** —
  repository ownership boundaries for runner, runtime, model crates, FFI,
  kernels, tests, and docs.
- **[Consolidation roadmap](docs/development/consolidation-roadmap.md)** —
  staged cleanup plan and non-breaking inventory for current tools.
- **[DFlash speculative decode](docs/dflash.md)** — Qwen3.5-9B INT4
  speculative decode design and milestones.
- **[Qwen3-30B-A3B HIP bring-up](docs/bringup/qwen3-30b-a3b-hip.md)** —
  separate Qwen3 MoE scaffolding and INT4 bake contract for HIP.
- **[SpecPrefill](docs/specprefill.md)** — long-prompt TTFT optimization
  via speculator-driven sparse prefill.
- **[Certified KV (Llama 3.1)](docs/certified-kv-audit-map.md)** — KV
  provenance for retrieval / safety-critical contexts.
- **[Low-level memory](docs/lowlevel-memory.md)** — VMM design, virtual
  KV cache mapping, eviction.
