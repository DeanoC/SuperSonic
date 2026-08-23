# Task 8 Fix Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline with test-first checkpoints; the parent task forbids subagents.

**Goal:** Make FLM genuinely crate-private and Qwen3.8-oriented while retaining public GQH GGUF loading and testing the actual FLM runtime wire.

**Architecture:** Keep neutral GQH identifier constants and FLM parser modules private to `model-store`. The integration contract test observes only public GQH behavior and the absence of runner/legacy surfaces; FLM row-table and runtime-directory characterization stays in `flm.rs` unit tests.

**Tech Stack:** Rust 2021, Cargo, model-store GQH/FLM parsers, source-boundary tests.

**Spec:** `docs/superpowers/specs/2026-08-23-qwen38-rocm-product-slimming-design.md`, Task 8 review findings.

## Global Constraints

- Preserve public custom GQH GGUF loading.
- Keep FLM CPU/parser foundations internal and unreachable from runner startup.
- Do not invent a test-only FLM wire format; use the production row-table/string-pool and runtime-directory parser.
- No GPU/download/artifact fabrication; report exact external artifact readiness blockers.

### Step 1: Write boundary RED tests

- Remove FLM imports from `flm_internal_contract.rs` and assert only public GQH qtype behavior, codec-16 mapping through public GQH qtype, runner absence, and deleted legacy surfaces.
- Add crate-unit assertions that `codec`/`flm` are not `pub mod`, GQH FLM-named methods are not public, and no FTD1 helper remains.

### Step 2: Run boundary RED

```bash
cargo test -p model-store --test flm_internal_contract -- --nocapture
cargo test -p model-store --lib -- --nocapture
```

Expected: the external test fails to compile against removed FLM imports until its assertions are rewritten; unit checks fail against the current public modules/FTD1 API.

### Step 3: Write internal FLM RED coverage

- Add `flm.rs` unit coverage using a minimal production-format runtime fixture.
- Assert logical-tensor row-table/string-pool parsing, Stage 3 descriptor semantics, `FlmRuntimeDirectory::parse_identity`, and Qwen3.8 identity; remove FTD1 round-trip assertions.

### Step 4: Implement the private Qwen3.8 foundation

- Change `codec` and `flm` module declarations to private.
- Remove FTD1 alias/encoder/decoder.
- Make `GqhRung::from_flm_codec`/`flm_codec` private or eliminate them while retaining internal mapping coverage.
- Replace Qwen3.6/MoE architecture/config/model IDs and public accessors with the minimal internal Qwen3.8 identity; remove unused MoE schema paths.

### Step 5: Run focused GREEN

```bash
cargo test -p model-store --test flm_internal_contract -- --nocapture
cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
cargo test -p model-store --lib flm -- --nocapture
```

### Step 6: Run full verification, append report, and commit

Run full model-store/runner CLI/startup tests, workspace checks, format/source validators, and artifact readiness checks without downloading. Append RED/GREEN and exact blocker evidence to the Task 8 report, then commit the focused fix.
