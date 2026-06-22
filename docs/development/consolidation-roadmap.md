# SuperSonic Consolidation Roadmap

The consolidation goal is shipability with no breakage. The first pass is
documentation-only: it makes ownership, tool classes, and validation vocabulary
explicit before any runtime, kernel, or script moves.

## First PR Scope

- Add the development repo map and this roadmap.
- Inventory current binaries and script groups without renaming anything.
- Link contributors to the support matrix, testing gates, benchmark recipes,
  repo map, and roadmap from the top-level README.
- Make no Rust module moves, CLI flag changes, test script renames, kernel file
  moves, or build behavior changes.

## Tracks

### 1. Support And Validation Taxonomy

Normalize the terms used across support docs, test gates, and benchmark
artifacts.

| Term | Meaning |
| --- | --- |
| `supported` | Documented in the support matrix with model, quantization, backend, architecture, and caveats. |
| `validated` | Has a named smoke or parity gate in `docs/testing.md` or an architecture script under `tests/`. |
| `benchmarked` | Has a repeatable command or artifact recipe in `docs/benchmarks.md`. |
| `experimental` | Usable for bring-up or optimization work, but not yet promoted to supported status. |
| `lab` | Useful for local exploration, profiling, diagnostics, or one-off hypotheses. Not a release contract. |

Acceptance for promotion:

- The support matrix names the exact model/backend/architecture combination.
- The relevant test gate is documented and runnable by command name.
- Performance-sensitive changes point to a benchmark recipe or kernel-lab task.
- Caveats are explicit, especially for architecture-specific paths.

### 2. Runner/Lab Separation

Classify the existing runner binaries before moving anything. The first code PR
after this roadmap should add metadata, wrappers, or a directory convention that
keeps every current command name available.

Classification:

- `stable`: user-facing command expected to remain part of the normal workflow.
- `validation`: correctness, parity, smoke, or artifact integrity check.
- `microbench`: focused performance measurement for a small path.
- `lab`: diagnostics, exploratory profiling, or bottleneck search.
- `legacy`: preserved for compatibility while a replacement is documented.

Current runner binary inventory:

The checked source of truth is
[`tools/tool-inventory.toml`](../../tools/tool-inventory.toml). Validate it with
`python3 tools/check-tool-inventory.py`. The table below is the human-readable
summary and should stay aligned with the manifest until it is generated.

| Command | Current file | Class | Future home |
| --- | --- | --- | --- |
| `supersonic` | `crates/runner/src/main.rs` | stable | Keep as the primary CLI until runtime/server APIs are fully split. |
| `int4_test` | `crates/runner/src/bin/int4_test.rs` | validation | `tools/validation` or validation metadata with the existing alias. |
| `int4_real_test` | `crates/runner/src/bin/int4_real_test.rs` | validation | `tools/validation` or validation metadata with the existing alias. |
| `bench_prefill_attn` | `crates/runner/src/bin/bench_prefill_attn.rs` | microbench | `tools/microbench` or microbench metadata with the existing alias. |
| `llama31_certified_kv_microbench` | `crates/runner/src/bin/llama31_certified_kv_microbench.rs` | microbench | `tools/microbench`. |
| `llama31_int8_downproj_diag` | `crates/runner/src/bin/llama31_int8_downproj_diag.rs` | lab | `tools/lab`. |
| `llama31_int8_matmul_test` | `crates/runner/src/bin/llama31_int8_matmul_test.rs` | validation | `tools/validation`. |
| `qwen35_bughunt` | `crates/runner/src/bin/qwen35_bughunt.rs` | lab | `tools/lab`; keep feature-gated behavior intact. |
| `qwen36_ffn_expert_microbench` | `crates/runner/src/bin/qwen36_ffn_expert_microbench.rs` | microbench | `tools/microbench`. |
| `qwen36_q4km_manifest_audit` | `crates/runner/src/bin/qwen36_q4km_manifest_audit.rs` | validation | `tools/validation`. |
| `gemma4_batched_divergent_test` | `crates/runner/src/bin/gemma4_batched_divergent_test.rs` | validation | `tools/validation`. |
| `gemma4_bench` | `crates/runner/src/bin/gemma4_bench.rs` | microbench | `tools/microbench`. |
| `gemma4_corpus_test` | `crates/runner/src/bin/gemma4_corpus_test.rs` | validation | `tools/validation`. |
| `gemma4_decode_validate` | `crates/runner/src/bin/gemma4_decode_validate.rs` | validation | `tools/validation`. |
| `gemma4_e2e_validate` | `crates/runner/src/bin/gemma4_e2e_validate.rs` | validation | `tools/validation`. |
| `gemma4_fused_decode_validate` | `crates/runner/src/bin/gemma4_fused_decode_validate.rs` | validation | `tools/validation`. |
| `gemma4_fused_int4_validate` | `crates/runner/src/bin/gemma4_fused_int4_validate.rs` | validation | `tools/validation`. |
| `gemma4_int4_layer_diag` | `crates/runner/src/bin/gemma4_int4_layer_diag.rs` | lab | `tools/lab`. |
| `gemma4_int4_matvec_test` | `crates/runner/src/bin/gemma4_int4_matvec_test.rs` | validation | `tools/validation`. |
| `gemma4_layer0_validate` | `crates/runner/src/bin/gemma4_layer0_validate.rs` | validation | `tools/validation`. |
| `gemma4_mega_decode_validate` | `crates/runner/src/bin/gemma4_mega_decode_validate.rs` | validation | `tools/validation`. |
| `gemma4_primitive_test` | `crates/runner/src/bin/gemma4_primitive_test.rs` | validation | `tools/validation`. |

Current script and test inventory:

| Path pattern | Class | Future home |
| --- | --- | --- |
| `tests/gfx*/run*.sh`, `tests/sm86/run*.sh` | validation | Keep under `tests/<arch>/` with support-matrix ownership metadata. |
| `tests/*/bench*.sh`, `tests/*/bench*.py` | microbench | Keep under `tests/<arch>/` or move behind benchmark metadata. |
| `tests/*/profile*.sh`, `tests/*/profile*.py` | lab | Keep under `tests/<arch>/lab` or classify in metadata. |
| `tests/*/sweep*.py` | lab | Keep under `tests/<arch>/lab` or classify in metadata. |
| `tests/*/probe*.py`, `tests/*/audit*.py` | validation or lab | Classify per script once the taxonomy source exists. |
| `tests/*/refresh*.py`, `tests/*/summarize*.py`, `tests/*/select*.py` | lab | Keep as analysis helpers unless promoted to a gate. |
| `tests/corpus/run_golden.sh` and corpus generators | validation | Keep as corpus validation fixtures. |
| `tests/dflash/corpus_smoke.sh` | validation | Keep as DFlash smoke coverage. |
| `tests/dflash/block_sweep.sh` | microbench | Keep as DFlash optimization coverage. |
| `tests/test_*.py` | validation | Keep as pytest-facing validation harnesses. |

### 3. Runtime Boundary Cleanup

Remove the `supersonic-runtime -> runner` dependency by extracting shared
generation and session interfaces.

Initial extraction candidates:

- request/session configuration shared by CLI and server paths
- generation result and token event types
- model/backend selection structs that are not CLI-specific
- error types that server users need without importing runner code

Rules for this track:

- Keep `supersonic` CLI flags byte-for-byte compatible.
- Prefer type re-exports during transition so downstream imports can move
  gradually.
- Add focused build checks for `supersonic-runtime`, `supersonic-runner`, and
  `supersonic-server` after the boundary changes.

### 4. Kernel/Build Split

Replace broad "compile every bridge" behavior with explicit backend/model
feature groups only after validation coverage is visible.

Target behavior:

- Default builds preserve today's behavior until grouped builds are proven.
- Feature groups are named by backend and model family, not by incidental file
  layout.
- Kernel-lab and architecture smoke scripts identify the minimum coverage for
  each group.
- Profiling hooks remain available through `kernel-ffi` typed wrappers.

## Follow-Up PR Sequence

| PR | Scope | Guardrail |
| --- | --- | --- |
| PR 2 | Add a `tools/` classification path, `crates/runner/src/bin/lab/` convention, or metadata manifest. Keep wrapper aliases for existing binary names. | Existing `cargo run --bin ...` commands continue to work. |
| PR 3 | Extract shared runtime/generation/session interfaces so `supersonic-runtime` no longer depends on `runner`. | Build runner, runtime, and server; keep CLI behavior unchanged. |
| PR 4 | Introduce a single capability/support data source used by docs/tests where practical. | Generated or checked docs must still be easy to review. |
| PR 5 | Split `kernel-ffi/build.rs` into model/backend compile groups with default behavior preserved. | Compare old default behavior with grouped builds on one representative backend. |
| PR 6+ | Move larger model-specific runtime implementations out of `runner` once public interfaces are stable. | One benchmark or parity artifact for each moved runtime path. |

## First PR Test Plan

- Run `git diff --check`.
- Verify every referenced docs path exists.
- Optionally run markdown link checks if available.
- No full build is required because this PR changes documentation only.

## Later Refactor Test Plan

- `cargo build --release --bin supersonic --bin int4_test`
- Architecture-specific smoke for the touched backend.
- One representative benchmark or parity artifact for any moved runtime path.
- Existing command aliases remain available until replacements are documented
  and validated.
