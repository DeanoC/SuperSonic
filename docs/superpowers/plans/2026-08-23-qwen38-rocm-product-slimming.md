# Qwen3.8 ROCm Product Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SuperSonic into a clean ROCm/HIP-only Qwen3.8-27B inference engine with a direct custom GQH GGUF product path, internal-only FLM foundations, focused tests, reproducible R9700 CI, and accurate performance-first documentation.

**Architecture:** Preserve the proven Qwen3.8 GQH loader and HIP decode/prefill/MTP chain while deleting unsupported families and backends in call-site order. First establish executable contract tests, then narrow the CLI and runtime, isolate shared MTP/GQH code from misleading legacy names, reduce the workspace and build system, and finish with CI and documentation. Every deletion phase must leave the retained product gates green.

**Tech Stack:** Rust 2021, Cargo, clap 4, ROCm/HIP, C++/HIP kernel bridges, custom GQH GGUF, internal FLM codecs, Python 3 manifest validators, GitHub Actions with self-hosted `gfx1201` runners.

**Spec:** `docs/superpowers/specs/2026-08-23-qwen38-rocm-product-slimming-design.md`

## Global Constraints

- The public product supports only ROCm/HIP, Qwen3.8-27B, custom GQH GGUF, `gfx1100`/`gfx1201`, single-sequence inference, greedy generation, and optional Qwen3.8 NextN/MTP.
- Keep `--model`; its only accepted value is `qwen3.8-27b`.
- Remove `--backend` and `SUPERSONIC_BACKENDS`; do not add compatibility shims.
- Keep `HIP_ARCH`, `HIP_VISIBLE_DEVICES`, `ROCM_PATH`, and `HIP_PATH` as ROCm controls.
- Keep FLM foundations internal and compile-tested; do not expose `--flm-file` or make public FLM claims.
- `--model-dir` supplies `config.json`, tokenizer data, and the chat template; `--gguf-file` supplies project-specific GQH weights.
- Preserve every encoding used by the canonical Qwen3.8 artifact, including Q2_K, Q3_K, Q8_0, GGML-K, ROCmFP mixes, and GQH 108-111.
- Preserve ordinary greedy and MTP token equivalence.
- GPU tests run serially and select the `gfx1201` device explicitly.
- Do not introduce blocking throughput thresholds until CI variance is measured.
- Use `gpt-5.6-luna` with `max` reasoning for delegated implementation unless a task explicitly requires primary-agent integration judgment.

---

## File Structure and Ownership After the Refactor

- `crates/core/`: Qwen3.8 model identity and AMD architecture registry only.
- `crates/gpu-hal/`: HIP allocation, copy, event, and device operations only.
- `crates/kernel-ffi/`: Qwen3.8 HIP decode/prefill, GQH, and MTP FFI only.
- `crates/model-store/`: custom GGUF/GQH readers and internal FLM codec foundations.
- `crates/qwen38/`: renamed current Qwen3.8 loader, state, descriptors, and weights.
- `crates/runtime/`: Qwen3.8 decode, prefill, generation, tokenizer/chat, and MTP only.
- `crates/runner/`: one `supersonic` binary, public CLI, startup validation, and structured output.
- `tools/`: narrow manifest, docs, and CI preflight validators.
- `.github/workflows/`: one CPU PR workflow and one `gfx1201` artifact workflow.
- `docs/`: seven public product documents plus narrow contributor architecture guidance.

---

### Task 1: Freeze the Public CLI Contract

**Files:**
- Create: `crates/runner/tests/qwen38_cli_contract.rs`
- Modify: `crates/runner/src/cli.rs`
- Modify: `crates/runner/src/lib.rs`
- Modify: `crates/runner/src/policy.rs`
- Modify: `crates/runner/Cargo.toml`

**Interfaces:**
- Consumes: existing clap `Cli` parser and `ModelVariant` parsing.
- Produces: `pub fn parse_cli_from<I, T>(args: I) -> Result<Cli, clap::Error>` for contract tests; a CLI containing only the fields allowed by the spec.

- [ ] **Step 1: Expose parsing without changing behavior**

Add this wrapper next to `Cli`:

```rust
pub fn parse_cli_from<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)
}
```

Re-export it from `runner::cli` for integration tests.

- [ ] **Step 2: Write failing retained-contract tests**

Test that this minimal command parses:

```rust
let cli = parse_cli_from([
    "supersonic",
    "--model", "qwen3.8-27b",
    "--model-dir", "/models/qwen38",
    "--gguf-file", "/models/qwen38.gqh.gguf",
    "--prompt", "Hello",
]).unwrap();
assert_eq!(cli.model, "qwen3.8-27b");
assert_eq!(cli.max_new_tokens, 8);
```

Also test missing `--model-dir`, missing `--gguf-file`, and a model value of
`qwen3.5-0.8b` fail.

- [ ] **Step 3: Write failing removed-option tests**

Loop over `--backend`, `--flm-file`, `--q4km`, `--dflash`,
`--specprefill`, and `--certified-kv`; assert each produces
`ErrorKind::UnknownArgument`.

- [ ] **Step 4: Run RED tests**

Run:

```bash
cargo test -p runner --test qwen38_cli_contract -- --nocapture
```

Expected: failures because the obsolete options still exist and required
arguments/model validation are not yet enforced.

- [ ] **Step 5: Reduce `Cli` to the approved fields**

Keep model, model directory, GGUF file, prompt/chat, special-token control,
maximum tokens, EOS, greedy controls, context, device, prefill chunking,
Qwen3.8 MTP, and structured output/timing. Remove all fields listed in the
spec. Make `--model-dir` and `--gguf-file` required. Parse `--model` with a
one-value enum or value parser so unsupported names fail during parsing.

- [ ] **Step 6: Remove obsolete CLI policy checks**

Delete policy functions whose inputs no longer exist. Replace broad policy
validation with checks for single-sequence greedy/MTP constraints only.

- [ ] **Step 7: Run GREEN tests and CLI help smoke**

```bash
cargo test -p runner --test qwen38_cli_contract -- --nocapture
cargo run -p runner --bin supersonic -- --help
```

Expected: all contract tests pass; help contains Qwen3.8/GQH/ROCm language and
none of `backend|FLM|CUDA|Metal|Gemma|Phi|Llama|DFlash|SpecPrefill|Certified`.

- [ ] **Step 8: Commit**

```bash
git add crates/runner
git commit -m "refactor(cli): expose only Qwen3.8 GQH inference"
```

---

### Task 2: Make Startup a Direct Qwen3.8 GQH Path

**Files:**
- Create: `crates/runner/tests/qwen38_startup_contract.rs`
- Modify: `crates/runner/src/main.rs`
- Modify: `crates/runner/src/bakes.rs`
- Modify: `crates/runner/src/model_files.rs`
- Modify: `crates/runner/src/qwen35_startup.rs`
- Modify: `crates/runner/src/qwen35_engine_setup.rs`
- Modify: `crates/runner/src/qwen35_runtime.rs`
- Modify: `crates/runner/src/backend_runtime.rs`

**Interfaces:**
- Consumes: reduced `Cli`, `Qwen35Weights::load_gguf`, runtime decode/prefill.
- Produces: `pub fn validate_input_contract(cli: &Cli) -> anyhow::Result<()>` and a single HIP/Qwen3.8 execution path.

- [ ] **Step 1: Write failing filesystem-contract tests**

Use `tempfile::tempdir()` to assert actionable errors for missing
`config.json`, missing tokenizer data, missing GGUF, and a non-GQH GGUF.
Assert error text names the missing path and required artifact role.

- [ ] **Step 2: Run RED tests**

```bash
cargo test -p runner --test qwen38_startup_contract -- --nocapture
```

Expected: failure because no narrow validator exists.

- [ ] **Step 3: Implement preflight validation**

Validate inputs before GPU allocation. Require `config.json`, supported
tokenizer metadata, and a readable GQH GGUF. Preserve the existing GQH magic,
architecture, geometry, and qtype checks from the direct branch in `bakes.rs`.

- [ ] **Step 4: Collapse `main` dispatch**

Remove backend choice, model inference, registry family dispatch, bake fetch,
FLM public dispatch, MoE, DFlash, SpecPrefill, oracle, and validation routes.
Set HIP directly, resolve the AMD architecture, look up the Qwen3.8 entry, load
the direct GQH artifact, and call the retained generation path.

- [ ] **Step 5: Keep FLM unreachable from public startup**

Retain internal FLM modules only where required by model-store compilation.
Delete runner-level FLM source/tokenizer selection and any environment fallback
that makes FLM reachable without a public argument.

- [ ] **Step 6: Run GREEN tests and production check**

```bash
cargo test -p runner --test qwen38_startup_contract -- --nocapture
HIP_ARCH=gfx1201 cargo check -p runner --bin supersonic
```

- [ ] **Step 7: Commit**

```bash
git add crates/runner
git commit -m "refactor(runner): make Qwen3.8 GQH the direct startup path"
```

---

### Task 3: Parameterize and Harden Canonical Artifact Tests

**Files:**
- Modify: `crates/model-store/src/gguf.rs`
- Modify: `crates/kernel-ffi/src/gqh.rs`
- Modify: `crates/qwen35/tests/qwen38_gqh_gguf_crawl.rs`
- Modify: `crates/runtime/tests/qwen38_gqh_decode_rung11.rs`
- Create: `tools/check-qwen38-artifacts.py`
- Create: `tests/test_qwen38_artifact_preflight.py`

**Interfaces:**
- Consumes: `SUPERSONIC_GQH_GGUF`, `SUPERSONIC_QWEN38_MODEL_DIR`, and optional `SUPERSONIC_GQH_8192_GGUF`.
- Produces: one shared artifact-path contract and a CI preflight that fails before tests when configured inputs are absent.

- [ ] **Step 1: Write failing Python preflight tests**

Use temporary directories to test missing environment variables, missing GGUF,
missing `config.json`, and a valid fixture layout. The checker must return a
nonzero exit and name every missing item.

- [ ] **Step 2: Run RED test**

```bash
python3 -m unittest tests.test_qwen38_artifact_preflight -v
```

Expected: failure because the checker does not exist.

- [ ] **Step 3: Implement artifact preflight**

Read the three environment variables, require the canonical GGUF and model
directory, and treat the 8192 artifact as optional unless an explicit
`--require-8192` argument is passed.

- [ ] **Step 4: Remove hard-coded developer paths from Rust tests**

Add one helper per crate that returns `Option<PathBuf>` locally but panics when
`SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1` and a required variable/path is missing.
Use it in every artifact-dependent test.

- [ ] **Step 5: Split short smoke from heavy crawl**

Keep file metadata/geometry checks in a fast artifact smoke. Mark upload,
full-64-layer, and all-tensor coverage with `#[ignore = "requires R9700 artifact CI"]`
so the GPU workflow selects them explicitly with `--ignored`.

- [ ] **Step 6: Run GREEN CPU tests**

```bash
python3 -m unittest tests.test_qwen38_artifact_preflight -v
cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
cargo test -p supersonic-runtime --lib mtp_accept_tests -- --nocapture
```

- [ ] **Step 7: Commit**

```bash
git add crates/model-store crates/kernel-ffi crates/qwen35/tests crates/runtime/tests tools tests
git commit -m "test(qwen38): define canonical GQH artifact gates"
```

---

### Task 4: Remove Unsupported Workspace Crates and Test Suites

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/runtime/Cargo.toml`
- Modify: `crates/runner/Cargo.toml`
- Delete: `crates/qwen35_dflash/`
- Delete: `crates/qwen36_moe/`
- Delete: `crates/server/`
- Delete: `crates/bench/`
- Delete: `crates/kernel-lab/`
- Delete: `crates/machine-profile/`
- Delete: obsolete runner binaries and unsupported Rust/Python/shell tests identified by the design audit.

**Interfaces:**
- Consumes: direct runner path and hardened product tests from Tasks 1-3.
- Produces: a seven-crate product workspace with no unsupported test targets.

- [ ] **Step 1: Capture the expected failing all-target baseline**

```bash
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

Expected: current Qwen3.6-MoE test compilation failures.

- [ ] **Step 2: Remove workspace members and dependency edges**

Delete unsupported members from root Cargo configuration. Remove their
dependencies and public module declarations from runtime and runner.

- [ ] **Step 3: Delete unsupported tests and binaries**

Delete Qwen3.5/Qwen3.6/MoE/DFlash/SpecPrefill/CUDA/Metal/Gemma/Phi/Llama/
Certified-KV/bake/oracle suites. Preserve only tests proven to exercise the
direct Qwen3.8 GQH or internal FLM foundation.

- [ ] **Step 4: Remove dead runner/runtime modules until compilation closes**

Use compiler errors as the call-site inventory. Do not stub deleted behavior.
When an MTP call reaches DFlash-named shared code, leave that code for Task 5.

- [ ] **Step 5: Regenerate lockfile mechanically**

```bash
cargo metadata --no-deps --format-version 1 >/dev/null
```

- [ ] **Step 6: Run GREEN all-target check**

```bash
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

Expected: exit 0 with no references to deleted crates.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove unsupported model and framework crates"
```

---

### Task 5: Isolate Qwen3.8 MTP and GQH from Legacy Names

**Files:**
- Modify: `crates/runtime/src/decode_engine.rs`
- Modify: `crates/runtime/src/prefill_engine.rs`
- Create: `crates/runtime/src/mtp.rs`
- Modify: `crates/runtime/src/lib.rs`
- Modify: `crates/kernel-ffi/src/qwen35.rs`
- Modify: `crates/kernel-ffi/src/layer_desc.rs`
- Modify: `crates/qwen35/src/desc_builder.rs`

**Interfaces:**
- Produces: `MtpVerifyCache`, `MtpVerifyScratch`, and Qwen3.8-named MTP helpers with unchanged token/state behavior.
- Preserves: GQH sidecar ABI currently carried through `INT4ScaleDesc`.

- [ ] **Step 1: Add failing naming/boundary tests**

Add a source-boundary test that scans retained runtime modules and fails if
Qwen3.8 MTP types contain `DFlash` or `MetalV2` names. Keep the existing MTP
acceptance tests as behavioral protection.

- [ ] **Step 2: Run RED tests**

```bash
cargo test -p supersonic-runtime mtp_accept_tests -- --nocapture
python3 tools/check-retained-source-terms.py
```

Expected: source-term checker fails on shared legacy names.

- [ ] **Step 3: Extract shared MTP verification**

Move the Qwen3.8-used fused verification cache, prefill-append verification,
scratch allocation, and state-restore helpers into `runtime::mtp`. Do not move
outer tree/tap/rollback behavior.

- [ ] **Step 4: Rename HIP MTP controls and helpers**

Rename retained Qwen3.5/DFlash/Metal identifiers and environment controls to
Qwen3.8/MTP terminology. Do not change external kernel ABI and behavior in the
same step unless a focused parity test covers it.

- [ ] **Step 5: Document the retained GQH descriptor ABI**

Rename Rust-side types only if all FFI layout assertions pass. Otherwise keep
the ABI type and add a precise comment that its GQH pointers are not legacy
GPTQ support.

- [ ] **Step 6: Run GREEN tests**

```bash
cargo test -p supersonic-runtime mtp_accept_tests -- --nocapture
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
python3 tools/check-retained-source-terms.py
```

- [ ] **Step 7: Commit**

```bash
git add crates/runtime crates/kernel-ffi crates/qwen35 tools
git commit -m "refactor(qwen38): isolate MTP verification from legacy paths"
```

---

### Task 6: Reduce GPU HAL and Kernel Build to HIP

**Files:**
- Modify: `crates/gpu-hal/build.rs`
- Modify: `crates/gpu-hal/src/backend.rs`
- Modify: `crates/gpu-hal/src/lib.rs`
- Modify: `crates/gpu-hal/src/ops.rs`
- Delete: CUDA and Metal HAL modules and bridges.
- Modify: `crates/kernel-ffi/build.rs`
- Modify: `crates/kernel-ffi/kernel-groups.toml`
- Modify: `crates/kernel-ffi/src/lib.rs`
- Delete: Certified-KV, DFlash, Qwen3.6-MoE, CUDA, and Metal FFI modules and sources.

**Interfaces:**
- Produces: HIP-only HAL initialization and two retained kernel groups: Qwen3.8 dense/GQH support and GQH.

- [ ] **Step 1: Write failing backend/build-manifest tests**

Extend `tools/check-kernel-groups.py` tests to assert every group has
`backend = "hip"`, no source name contains `_cuda`, `metal`, `gemma`, `phi`,
`dflash`, or `moe`, and the GQH plus dense 4B/prefill sources are present.

- [ ] **Step 2: Run RED validator tests**

```bash
python3 -m unittest tests.test_kernel_groups -v
```

Expected: failure on current broad groups/sources.

- [ ] **Step 3: Remove backend selection from HAL**

Replace `Auto|Cuda|Metal|Hip` choice with direct HIP operations. Keep device
ordinal selection. Delete CUDA/Metal detection, cfg emission, link logic, and
stubs.

- [ ] **Step 4: Reduce kernel build groups**

Retain the HIP full-attention 4B, prefill helper, GQH, and MTP restore sources.
Remove every other bridge and rerun path. Keep `HIP_ARCH`; remove
`SUPERSONIC_BACKENDS` parsing.

- [ ] **Step 5: Run GREEN build and validators**

```bash
python3 -m unittest tests.test_kernel_groups -v
python3 tools/check-kernel-groups.py
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

- [ ] **Step 6: Commit**

```bash
git add -A crates/gpu-hal crates/kernel-ffi kernels tools tests
git commit -m "refactor(hip): remove CUDA and Metal build surfaces"
```

---

### Task 7: Reduce Registry and Rename the Qwen3.8 Model Crate

**Files:**
- Modify: `crates/core/src/registry.rs`
- Modify: `crates/core/src/backend.rs`
- Rename: `crates/qwen35/` to `crates/qwen38/`
- Modify: root and dependent `Cargo.toml` files.
- Modify: retained Rust imports and Qwen3.5-named product types.

**Interfaces:**
- Produces: `ModelVariant::Qwen3_8_27B`, `ModelFamily::Qwen38`, Qwen3.8 crate types, and only `gfx1100`/`gfx1201` registry rows.

- [ ] **Step 1: Write failing registry tests**

Assert canonical parsing accepts only `qwen3.8-27b`, supported architectures
are exactly `gfx1100` and `gfx1201`, and every registry row uses HIP/Qwen3.8.

- [ ] **Step 2: Run RED tests**

```bash
cargo test -p supersonic-core registry -- --nocapture
```

- [ ] **Step 3: Reduce registry enums and rows**

Remove all unsupported model/family/architecture variants and fallback lists.
Introduce Qwen3.8 names atomically across registry, runtime policy, and tests.

- [ ] **Step 4: Rename crate and product-facing identifiers**

Move the directory with `git mv`, update package/import names, and rename
loader/state/weight types. Keep kernel symbol names only where ABI stability or
compiler risk makes a separate change safer; document those internal symbols.

- [ ] **Step 5: Run GREEN checks**

```bash
cargo test -p supersonic-core registry -- --nocapture
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(qwen38): make Qwen3.8 the sole model identity"
```

---

### Task 8: Reduce Model Store While Preserving Internal FLM

**Files:**
- Modify: `crates/model-store/src/lib.rs`
- Modify: `crates/model-store/src/flm.rs`
- Modify: `crates/model-store/src/gqh.rs`
- Modify: `crates/model-store/Cargo.toml`
- Delete: bake/fetch/manifest/store/transform modules not required by GQH or internal FLM codec tests.
- Create: `crates/model-store/tests/flm_internal_contract.rs`

**Interfaces:**
- Produces: public GQH GGUF loading and crate-private/internal FLM codec primitives; no public runner loader.

- [ ] **Step 1: Write failing FLM-boundary tests**

Test that GQH qtype IDs map consistently between GGUF and FLM codecs and that
an internal FLM tensor descriptor round-trips without GPU access. Add a source
check confirming runner has no `--flm-file` or FLM startup path.

- [ ] **Step 2: Run RED/characterization tests**

```bash
cargo test -p model-store --test flm_internal_contract -- --nocapture
```

- [ ] **Step 3: Localize shared codec constants**

Move constants needed by `gqh.rs` into a small codec module owned by
model-store rather than retaining Qwen3.6 runtime descriptors.

- [ ] **Step 4: Delete legacy bake and distribution surface**

Remove HF baking, published-bake fetching, old manifests, Q4KM/GPTQ conversion,
and GPU-direct FLM product loading not required by the internal format tests.
Trim dependencies after compilation proves they are unused.

- [ ] **Step 5: Run GREEN tests**

```bash
cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
cargo test -p model-store --test flm_internal_contract -- --nocapture
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

- [ ] **Step 6: Commit**

```bash
git add crates/model-store Cargo.lock
git commit -m "refactor(model-store): keep GQH with internal FLM foundations"
```

---

### Task 9: Establish CPU and R9700 CI

**Files:**
- Delete: `.github/workflows/kernel-lab.yml`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/qwen38-gfx1201.yml`
- Rewrite: `support/matrix.toml`
- Rewrite: `crates/kernel-ffi/kernel-groups.toml`
- Modify: `tools/check-support-matrix.py`
- Modify: `tools/check-kernel-groups.py`
- Modify: `tools/tool-inventory.toml`
- Create/modify: validator unit tests under `tests/`.

**Interfaces:**
- Produces: deterministic CPU PR gate and serial self-hosted R9700 gate with structured artifacts.

- [ ] **Step 1: Write failing manifest tests**

Require one public support row per retained architecture, model
`qwen3.8-27b`, source `gqh-gguf`, and a named correctness gate. Reject other
models/backends/sources in the active matrix.

- [ ] **Step 2: Run RED tests**

```bash
python3 -m unittest tests.test_support_matrix -v
```

- [ ] **Step 3: Implement CPU workflow**

Run diff/format checks, `cargo check --workspace --all-targets`, focused GQH
and MTP tests, CLI tests, manifest validators, and active-doc checks. Set
`HIP_ARCH=gfx1201` only where build scripts require an architecture; do not
require a GPU.

- [ ] **Step 4: Implement R9700 workflow**

Use labels `[self-hosted, linux, rocm, gfx1201]`, explicit device selection,
artifact preflight, `RUST_TEST_THREADS=1`, a 45-minute timeout, GPU idle
polling, release build, GQH kernel tests, artifact crawl, decode/chat tests,
ordinary-versus-MTP token comparison, and nonblocking throughput telemetry.

- [ ] **Step 5: Run local workflow-equivalent commands**

```bash
git diff --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
python3 tools/check-support-matrix.py
python3 tools/check-kernel-groups.py
python3 tools/check-tool-inventory.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

- [ ] **Step 6: Commit**

```bash
git add .github support tools tests crates/kernel-ffi/kernel-groups.toml
git commit -m "ci: gate Qwen3.8 GQH on CPU and R9700"
```

---

### Task 10: Rewrite the Product Documentation and Remove History Noise

**Files:**
- Rewrite: `README.md`
- Rewrite: `AGENTS.md`
- Remove or reduce: `CLAUDE.md`
- Rewrite: `docs/build-and-run.md`
- Rewrite: `docs/supported-matrix.md`
- Create: `docs/artifact-format.md`
- Rewrite: `docs/testing.md`
- Rewrite: `docs/benchmarks.md`
- Rewrite: `docs/performance.md`
- Rewrite: retained contributor architecture documentation.
- Delete: obsolete active-branch feature, performance-history, bring-up, optimization, paper, and superseded plan/spec documents, except this spec and implementation plan.
- Create: `tools/check-active-docs.py`
- Create: `tests/test_active_docs.py`

**Interfaces:**
- Produces: seven-document public information architecture and a forbidden-term/link validator.

- [ ] **Step 1: Write failing documentation tests**

The test must crawl README and the seven public documents, verify relative
links exist, and reject unsupported product terms/instructions including
`--backend`, `SUPERSONIC_BACKENDS`, CUDA build commands, Metal, Gemma, Phi,
Llama, Qwen3.5, Qwen3.6, DFlash, SpecPrefill, and Certified-KV. Permit FLM only
in an explicitly marked internal contributor section.

- [ ] **Step 2: Run RED test**

```bash
python3 -m unittest tests.test_active_docs -v
```

Expected: many failures against current documentation.

- [ ] **Step 3: Rewrite public documents**

Use the approved positioning. Include one direct GQH quickstart without
`--backend`, the model-dir/GGUF pairing, qualified 37.2 tok/s evidence only if
the exact commit/artifact/workload is named, and explicit supported targets.

- [ ] **Step 4: Rewrite contributor guidance**

Document crate ownership, internal FLM status, kernel ABI caveats, test tiers,
and the rule that unsupported combinations fail explicitly. Remove duplicate
and contradictory agent guidance.

- [ ] **Step 5: Delete obsolete documents**

Use Git history as the archive. Keep this design and implementation plan until
the work is integrated; delete superseded unrelated plans/specs.

- [ ] **Step 6: Run GREEN documentation tests**

```bash
python3 -m unittest tests.test_active_docs -v
python3 tools/check-active-docs.py
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add -A README.md AGENTS.md CLAUDE.md docs tools tests
git commit -m "docs: reposition SuperSonic around Qwen3.8 ROCm performance"
```

---

### Task 11: Format the Retained Baseline and Make Warnings Actionable

**Files:**
- Modify: retained Rust source formatted by `cargo fmt`.
- Modify: retained build scripts and source files producing warnings.
- Modify: `.github/workflows/ci.yml`.

**Interfaces:**
- Produces: a format-clean retained workspace and a documented warning policy.

- [ ] **Step 1: Record current formatting failure**

```bash
cargo fmt --all --check
```

Expected: failure on the pre-existing retained baseline.

- [ ] **Step 2: Format mechanically**

```bash
cargo fmt --all
```

Review the diff to ensure it contains only rustfmt output.

- [ ] **Step 3: Eliminate retained-backend warning residue**

Remove unreachable CUDA/Metal arms, unused cfg variables, dead imports, and
unexpected feature cfgs that remain in product crates. Do not suppress warnings
globally.

- [ ] **Step 4: Run GREEN format and all-target checks**

```bash
cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "style: format the narrowed Qwen3.8 workspace"
```

---

### Task 12: Full Local and R9700 Verification

**Files:**
- Modify only if verification exposes a regression; every fix follows a new failing test.
- Record: structured verification and benchmark artifacts under the workflow-defined output directory, not Git.

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: evidence that the branch meets every spec success criterion.

- [ ] **Step 1: Run the complete CPU-safe gate**

```bash
git diff --check
cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
cargo test -p kernel-ffi --lib 'gqh::tests::register_and_lookup_header_by_pointer' -- --nocapture
cargo test -p supersonic-runtime --lib mtp_accept_tests -- --nocapture
cargo test -p runner --test qwen38_cli_contract -- --nocapture
cargo test -p runner --test qwen38_startup_contract -- --nocapture
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 tools/check-support-matrix.py
python3 tools/check-kernel-groups.py
python3 tools/check-tool-inventory.py
python3 tools/check-active-docs.py
```

- [ ] **Step 2: Run the R9700 artifact preflight**

```bash
export HIP_VISIBLE_DEVICES=1
export HIP_ARCH=gfx1201
export SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1
python3 tools/check-qwen38-artifacts.py
```

- [ ] **Step 3: Run serial GPU correctness**

```bash
RUST_TEST_THREADS=1 cargo test --release -p kernel-ffi --lib 'gqh::tests::' -- --nocapture
RUST_TEST_THREADS=1 cargo test --release -p qwen38 --test qwen38_gqh_gguf_crawl -- --ignored --nocapture
RUST_TEST_THREADS=1 cargo test --release -p supersonic-runtime --test qwen38_gqh_decode_rung11 -- --ignored --nocapture
```

- [ ] **Step 4: Run ordinary and MTP generation equivalence**

Run the documented eight-token chat smoke once normally and once with the
retained MTP option. Compare structured generated-token arrays exactly; fail on
any mismatch.

- [ ] **Step 5: Collect nonblocking performance telemetry**

Run documented warmups and repeated greedy decode measurements. Record commit,
GPU, HIP version, artifact identity, prompt, token count, correctness hash, and
median throughput. Do not change CI thresholds from this single run.

- [ ] **Step 6: Audit the spec success criteria**

For every bullet under the spec's `Success Criteria`, cite the command output,
test, source path, or deleted path that proves it. Fix uncovered gaps through a
new red/green cycle.

- [ ] **Step 7: Commit verification-only fixes if any**

```bash
git add -A
git commit -m "fix: close Qwen3.8 slimming verification gaps"
```

Skip this commit when verification required no source changes.

- [ ] **Step 8: Request final code review**

Invoke `superpowers:requesting-code-review`, review the complete branch diff
against the spec, address findings with `superpowers:receiving-code-review`,
and rerun the full affected gates.
