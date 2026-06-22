# SuperSonic Repository Architecture

This map records the intended ownership boundaries for SuperSonic while the
project is still an experimental inference engine. It is descriptive first:
current code may still cross these boundaries, but new work should make those
crossings rarer and easier to remove.

## Ownership Boundaries

| Area | Owns | Should avoid |
| --- | --- | --- |
| `crates/runner` | CLI orchestration, experiment entrypoints, validation wiring, and compatibility aliases for existing commands. | Becoming the stable library API, owning reusable generation/session contracts, or hiding model/backend policy that server users need. |
| `crates/runtime` | Stable library and server-facing runtime API. This is the intended home for reusable generation, session, request, and response types. | Depending on `runner`, importing CLI-only code, or depending on lab experiment helpers. |
| `crates/core` | Shared cross-crate concepts such as model ids, backend ids, architecture ids, registry metadata, and capability/support vocabulary. | Model-specific runtime implementation, CLI parsing, or backend launch details. |
| Model crates (`crates/qwen35`, `crates/qwen35_dflash`, `crates/qwen3_moe`, `crates/qwen36_moe`, `crates/gemma4`, `crates/phi4`) | Model metadata, weight/state definitions, bake contracts, and model-local helpers. | CLI orchestration, global device probing policy, or unrelated backend selection logic. |
| `crates/kernel-ffi` | Typed host/FFI wrappers, kernel launch wrappers, profiling hooks, and error translation for backend kernels. | High-level model orchestration, benchmark policy, or CLI experiment routing. |
| `kernels` | Backend kernels grouped by model/backend until later split work proves coverage. | Rust ownership of model/session state or command-line behavior. |
| `crates/gpu-hal` | Backend memory/device abstractions, buffers, device selection primitives, and low-level synchronization helpers. | Model-specific scheduling policy or benchmark reporting. |
| `crates/model-store` | Bake discovery, versioning, download layout, and local artifact storage contracts. | Runtime generation loops or backend kernel selection. |
| `crates/bench` and `crates/kernel-lab` | Repeatable benchmarks, kernel task definitions, candidate-vs-baseline comparison, and lab-grade performance artifacts. | Replacing user-facing CLI entrypoints or becoming an implicit release dependency. |
| `tests/` | Architecture and workload validation scripts, smoke tests, parity checks, and reproducible benchmark harnesses. | Long optimization narratives, ad hoc one-off logs, or hidden production entrypoints. |
| `docs/` | Operator docs, support matrix, testing gates, benchmark recipes, development maps, plans, and historical bring-up notes. | Source of runtime truth that cannot be validated by code or scripts. |

## Current Boundary Debt

These are known seams to clean up after this documentation foundation lands:

- `runner` still carries compatibility shims and larger model-specific runtime
  paths while public APIs settle. Keep moving stable generation/session code
  toward `runtime` or a small shared crate so the runtime boundary stays
  library-first.
- `crates/runner/src/bin/*` contains stable-ish validation tools, true
  microbenches, diagnostics, and lab experiments in one flat namespace. Command
  names must keep working, but the repo needs a classification layer before
  files move.
- The main runner still carries model-specific wiring and temporary module
  aliases for Qwen3.6 MoE bring-up. Those are acceptable while APIs settle, but
  should not become the long-term extension model.
- `crates/kernel-ffi/build.rs` compiles a broad bridge surface. Later PRs should
  introduce explicit model/backend feature groups while preserving the default
  build behavior until coverage is proven.
- Support status, benchmark artifacts, and per-architecture gates are mostly
  documented separately. The long-term goal is a single capability/support data
  source that docs and tests can share where practical.

## Change Rules

- Keep existing command names, test scripts, and CLI flags working until
  replacement commands are documented, validated, and wrapper aliases exist.
- Promote a model/backend/architecture path only when the support matrix, test
  gate, and benchmark recipe are all named.
- Treat lab and microbench tools as first-class development tools, but keep them
  out of stable runtime contracts.
- Prefer narrow extraction PRs over broad moves. A move should have an explicit
  owner, validation gate, and rollback path.
- Do not split kernel build behavior until the old default path and the new
  grouped path can be compared on at least one representative backend.

## Related Docs

- [Supported matrix](../supported-matrix.md)
- [Testing gates](../testing.md)
- [Benchmark recipes](../benchmarks.md)
- [Kernel lab](kernel-lab.md)
- [Consolidation roadmap](consolidation-roadmap.md)
