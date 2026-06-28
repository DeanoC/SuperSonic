# Production Inference Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor SuperSonic from rapid bring-up code into a production-ready HIP-first inference engine for local coding and agent workloads, with Qwen3.6-27B DFlash as the first product lane and Qwen3.6-35B-A3B as the MoE expansion lane.

**Architecture:** Keep `supersonic-runtime` as the stable library and server-facing boundary, keep `runner` as CLI/lab orchestration with compatibility wrappers, and move production model/session policy behind typed runtime contracts. Preserve the measured Qwen3.6/DFlash performance path while replacing env-var-only controls and runner-local ownership with explicit config, capabilities, telemetry, and gates.

**Tech Stack:** Rust workspace, HIP/ROCm via `gpu-hal` and `kernel-ffi`, OpenAI-compatible Axum server, `model-store` baked weights, Python oracle/benchmark harnesses, TOML support/kernel/tool manifests.

---

## Scope And Ground Rules

This refactor is intentionally phased. Do not combine runtime API changes, Qwen3.6 engine moves, kernel build changes, and server compatibility changes in one PR.

The first production lane is:

- backend: HIP
- architectures: `gfx1100` and `gfx1201`
- model: `qwen3.6-27b`
- serving mode: native DFlash through `supersonic-serve`
- quantization: `--q4km` or `--q4km-gptq`
- client compatibility: OpenAI-compatible Chat Completions and Responses for OpenCode/Hermes-style clients

The second production lane is:

- backend: HIP
- model: `qwen3.6-35b-a3b`
- serving mode: MoE decode and long-context work after the runtime boundary is stable
- quantization: INT4/Q4-family first; FP8/KV-FP8 only where the support matrix and validation suite say it is ready

Keep these compatibility rules throughout:

- Existing `cargo run --bin ...` command names keep working until replacements are documented and wrapper aliases exist.
- Existing CLI flags keep their behavior unless the phase explicitly adds a replacement and a compatibility test.
- HIP kernel compilation units stay isolated by model family.
- Performance work must cite a benchmark artifact or repeatable benchmark command.

## Files And Responsibilities

- `docs/development/repo-architecture.md` documents ownership boundaries and should be updated only when an ownership decision changes.
- `docs/development/consolidation-roadmap.md` tracks sequencing and should link to this plan after Task 1.
- `docs/server.md` is the operator contract for `supersonic-serve`.
- `docs/benchmarks.md` owns repeatable performance recipes and Qwen3.6 Lucebox artifact references.
- `docs/testing.md` owns validation gate commands.
- `support/matrix.toml` is the machine-readable support/capability seed.
- `tools/check-support-matrix.py`, `tools/check-tool-inventory.py`, and `tools/check-kernel-groups.py` are manifest guards.
- `crates/runtime/src/state.rs` owns server/runtime loading policy.
- `crates/runtime/src/session.rs` owns the production session dispatch contract.
- `crates/runtime/src/generate.rs` owns request-time generation behavior and scheduler interaction.
- `crates/runtime/src/dflash.rs` owns native DFlash runtime serving behavior.
- `crates/runtime/src/builders.rs` owns production engine construction.
- `crates/runner/src/main.rs` remains the compatibility CLI dispatcher.
- `crates/runner/src/qwen36_moe/` remains the source for Qwen3.6 MoE pieces until each piece is extracted behind a runtime-facing boundary.
- `crates/kernel-ffi/src/qwen36_moe.rs` is a split candidate, but the first kernel-FFI phase must be mechanical and behavior-preserving.
- `crates/kernel-ffi/kernel-groups.toml` is the kernel build grouping source.

---

### Task 1: Baseline Production Contract And Refactor Gates

**Files:**
- Modify: `docs/development/consolidation-roadmap.md`
- Modify: `docs/testing.md`
- Modify: `docs/benchmarks.md`
- Modify: `support/matrix.toml`
- Test: `tools/check-support-matrix.py`

- [ ] **Step 1: Add this plan to the consolidation roadmap**

Add a short section after `## Follow-Up PR Sequence` in `docs/development/consolidation-roadmap.md`:

```markdown
## Production Inference Refactor Track

The production inference refactor is tracked in
[`docs/superpowers/plans/2026-06-28-production-inference-refactor.md`](../superpowers/plans/2026-06-28-production-inference-refactor.md).
It promotes the native HIP `qwen3.6-27b` DFlash server lane first, then moves
Qwen3.6 MoE runtime pieces behind `supersonic-runtime` once the server/session
contract is stable.

Promotion rule: a lane is product-ready only when it has a support-matrix row,
a named validation gate, an operator-facing server command when applicable, and
a repeatable benchmark or smoke recipe.
```

- [ ] **Step 2: Name the production baseline gate in testing docs**

In `docs/testing.md`, add this paragraph near the `gfx1201` known-issues section:

```markdown
For the production inference refactor, the baseline HIP gate is the existing
`tests/gfx1201/run_matrix.sh` starter matrix plus the `gfx1100` Qwen3.6 Lucebox
benchmark recipe in `docs/benchmarks.md`. Runtime/server refactor PRs that touch
DFlash, Qwen3.6-27B loading, kernel groups, or generation scheduling must state
whether they ran this gate, skipped it for lack of local artifacts, or used a
smaller compile/unit-test gate because the change was docs-only.
```

- [ ] **Step 3: Add a server smoke recipe for the production lane**

In `docs/benchmarks.md`, after the `RDNA4 / R9700 Smoke Benchmark` section, add:

```markdown
## Production Server Smoke: Qwen3.6 27B DFlash

This smoke checks the production-facing path rather than the CLI benchmark
path. It assumes `supersonic-serve` was built with HIP and the local Qwen3.6
27B target/draft artifacts exist.

```bash
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100,gfx1201 \
  cargo build --release -p server

SUPERSONIC_BACKENDS=hip HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
target/release/supersonic-serve \
  --backend hip \
  --model qwen3.6-27b \
  --model-dir "$MODEL_DIR" \
  --max-context 4096 \
  --q4km-gptq \
  --dflash \
  --dflash-draft-dir "$DRAFT_DIR" \
  --host 127.0.0.1 \
  --port 8013 \
  --api-key local \
  --no-download \
  --prefix-cache-disable
```

Then run the OpenAI-compatible smoke:

```bash
SUPERSONIC_BASE_URL=http://127.0.0.1:8013 \
SUPERSONIC_API_KEY=local \
node scripts/openai_compat_smoke.mjs
```
```

- [ ] **Step 4: Ensure `support/matrix.toml` names server readiness separately from correctness**

Add a `notes` sentence to `hip-gfx1201-qwen36-27b-int4`:

```toml
notes = "R9700 starter lane: RDNA4 WMMA/int4 harness, direct smoke, optional Lucebox/DFlash smoke, and native server smoke when local target/draft artifacts are present."
```

- [ ] **Step 5: Run docs and manifest guards**

Run:

```bash
python3 tools/check-support-matrix.py
python3 tools/check-tool-inventory.py
python3 tools/check-kernel-groups.py
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 6: Commit Task 1**

```bash
git add docs/development/consolidation-roadmap.md docs/testing.md docs/benchmarks.md support/matrix.toml
git commit -m "docs: define production inference refactor gates"
```

---

### Task 2: Runtime Session Contract For Production Engines

**Files:**
- Modify: `crates/runtime/src/session.rs`
- Modify: `crates/runtime/src/generate.rs`
- Modify: `crates/runtime/src/state.rs`
- Test: `crates/runtime/src/session.rs` unit tests or `crates/server/tests/protocol_mock.rs`

- [ ] **Step 1: Add an explicit session capability summary**

In `crates/runtime/src/session.rs`, add this public struct above `InferenceSession`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionFeatures {
    pub plain_prefill_decode: bool,
    pub native_dflash_generate: bool,
    pub prefix_snapshot: bool,
    pub disk_prefix_snapshot: bool,
}
```

- [ ] **Step 2: Implement `InferenceSession::features`**

Add this method to `impl InferenceSession`:

```rust
pub fn features(&self) -> SessionFeatures {
    match self {
        Self::Qwen(_) => SessionFeatures {
            plain_prefill_decode: true,
            native_dflash_generate: false,
            prefix_snapshot: true,
            disk_prefix_snapshot: true,
        },
        Self::QwenDFlash(_) => SessionFeatures {
            plain_prefill_decode: false,
            native_dflash_generate: true,
            prefix_snapshot: false,
            disk_prefix_snapshot: false,
        },
        Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => SessionFeatures {
            plain_prefill_decode: true,
            native_dflash_generate: false,
            prefix_snapshot: true,
            disk_prefix_snapshot: false,
        },
    }
}
```

- [ ] **Step 3: Use session features in generation instead of shape checks**

In `crates/runtime/src/generate.rs`, replace direct `session.is_dflash()` routing checks with:

```rust
let features = session.features();
if features.native_dflash_generate {
    // existing DFlash path
} else if features.plain_prefill_decode {
    // existing prefill/decode path
} else {
    anyhow::bail!("loaded session does not expose a supported generation path");
}
```

Keep behavior identical: DFlash uses `generate_dflash_greedy`, normal sessions use `prefill` and `decode_step`.

- [ ] **Step 4: Add a mock/session policy test**

If a focused unit test can be added without constructing GPU engines, add a small enum-free helper in `session.rs`:

```rust
pub fn should_use_dflash_generation(features: SessionFeatures) -> bool {
    features.native_dflash_generate
}
```

Add test:

```rust
#[test]
fn generation_mode_uses_dflash_only_when_session_exposes_it() {
    assert!(should_use_dflash_generation(SessionFeatures {
        plain_prefill_decode: false,
        native_dflash_generate: true,
        prefix_snapshot: false,
        disk_prefix_snapshot: false,
    }));
    assert!(!should_use_dflash_generation(SessionFeatures {
        plain_prefill_decode: true,
        native_dflash_generate: false,
        prefix_snapshot: true,
        disk_prefix_snapshot: true,
    }));
}
```

- [ ] **Step 5: Run focused runtime/server checks**

```bash
cargo test -p supersonic-runtime session --lib
cargo test -p server --test protocol_mock
```

Expected: tests pass. If HIP/CUDA linking is unavailable on the machine, record the linker error and run `cargo check -p supersonic-runtime -p server` instead.

- [ ] **Step 6: Commit Task 2**

```bash
git add crates/runtime/src/session.rs crates/runtime/src/generate.rs crates/runtime/src/state.rs
git commit -m "runtime: expose production session features"
```

---

### Task 3: Typed Production Loader Policy

**Files:**
- Modify: `crates/runtime/src/state.rs`
- Modify: `crates/runtime/src/builders.rs`
- Modify: `crates/runtime/src/dflash.rs`
- Test: `crates/runtime/src/state.rs` policy tests

- [ ] **Step 1: Split runtime policy from CLI-shaped loader flags**

Add this struct near `LoaderConfig` in `crates/runtime/src/state.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLane {
    PlainDecode,
    NativeDFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePolicy {
    pub lane: RuntimeLane,
    pub low_bit_target_required: bool,
    pub prefix_cache_allowed: bool,
}
```

- [ ] **Step 2: Add policy resolution**

Add:

```rust
fn resolve_runtime_policy(cfg: &LoaderConfig, variant: &ModelVariant) -> Result<RuntimePolicy> {
    let q4km_like = cfg.q4km || cfg.q4km_gptq;
    if cfg.dflash {
        if !matches!(variant, ModelVariant::Qwen3_5_9B | ModelVariant::Qwen3_6_27B) {
            bail!("--dflash is supported for --model qwen3.5-9b and qwen3.6-27b (got {variant})");
        }
        if !(cfg.int4 || q4km_like) {
            bail!("--dflash requires a low-bit target bake (--int4, --q4km, or --q4km-gptq)");
        }
        if cfg.dflash_draft_dir.is_none() {
            bail!("--dflash requires --dflash-draft-dir");
        }
        if cfg.kv_fp8 {
            bail!("--dflash does not support --kv-fp8");
        }
        return Ok(RuntimePolicy {
            lane: RuntimeLane::NativeDFlash,
            low_bit_target_required: true,
            prefix_cache_allowed: false,
        });
    }
    Ok(RuntimePolicy {
        lane: RuntimeLane::PlainDecode,
        low_bit_target_required: false,
        prefix_cache_allowed: true,
    })
}
```

- [ ] **Step 3: Thread policy into `build`**

In `build`, compute:

```rust
let runtime_policy = resolve_runtime_policy(&cfg, &variant)?;
```

Use it when constructing `PrefixCacheConfig`:

```rust
enabled: cfg.prefix_cache_enabled && runtime_policy.prefix_cache_allowed,
```

This makes the documented DFlash prefix-cache disable behavior enforceable even if the caller forgets `--prefix-cache-disable`.

- [ ] **Step 4: Add policy tests**

Extend the existing `state.rs` tests with:

```rust
#[test]
fn policy_disables_prefix_cache_for_native_dflash() {
    let mut c = cfg();
    c.dflash = true;
    c.q4km_gptq = true;
    c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));
    let policy = resolve_runtime_policy(&c, &ModelVariant::Qwen3_6_27B).unwrap();
    assert_eq!(policy.lane, RuntimeLane::NativeDFlash);
    assert!(!policy.prefix_cache_allowed);
    assert!(policy.low_bit_target_required);
}
```

- [ ] **Step 5: Run policy checks**

```bash
cargo test -p supersonic-runtime state::tests --lib
```

Expected: policy tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add crates/runtime/src/state.rs crates/runtime/src/builders.rs crates/runtime/src/dflash.rs
git commit -m "runtime: make production loader policy explicit"
```

---

### Task 4: Qwen3.6 Model Taxonomy Cleanup

**Files:**
- Modify: `crates/core/src/registry.rs`
- Modify: `crates/runtime/src/builders.rs`
- Modify: `docs/supported-matrix.md`
- Test: `crates/core/src/registry.rs` tests

- [ ] **Step 1: Add architecture-family terminology without renaming CLI variants**

In `crates/core/src/registry.rs`, add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureFamily {
    QwenHybridDense,
    QwenHybridMoE,
    Gemma4,
    Phi4,
    Llama31,
}

impl ModelVariant {
    pub fn architecture_family(&self) -> ArchitectureFamily {
        match self {
            Self::Qwen3_5_0_8B
            | Self::Qwen3_5_2B
            | Self::Qwen3_5_4B
            | Self::Qwen3_5_9B
            | Self::Qwen3_6_27B => ArchitectureFamily::QwenHybridDense,
            Self::Qwen3_5_35B_A3B | Self::Qwen3_30B_A3B | Self::Qwen3_6_35B_A3B => {
                ArchitectureFamily::QwenHybridMoE
            }
            Self::Gemma4_E2B | Self::Gemma4_E4B => ArchitectureFamily::Gemma4,
            Self::Phi4_Mini => ArchitectureFamily::Phi4,
            Self::Llama3_1_8B => ArchitectureFamily::Llama31,
        }
    }
}
```

- [ ] **Step 2: Use architecture family in runtime builder comments and match guards**

In `crates/runtime/src/builders.rs`, replace comments that say `caller filtered to Qwen3.5` for `Qwen3_6_27B` paths with `caller filtered to Qwen hybrid dense`.

Keep `ModelFamily` unchanged in this task. This task clarifies intent without moving registry entries or changing CLI aliases.

- [ ] **Step 3: Add registry tests**

Add:

```rust
#[test]
fn qwen36_27b_is_dense_hybrid_architecture() {
    assert_eq!(
        ModelVariant::Qwen3_6_27B.architecture_family(),
        ArchitectureFamily::QwenHybridDense
    );
    assert_eq!(
        ModelVariant::Qwen3_6_35B_A3B.architecture_family(),
        ArchitectureFamily::QwenHybridMoE
    );
}
```

- [ ] **Step 4: Document the naming split**

In `docs/supported-matrix.md`, near the Qwen3.6 footnotes, add:

```markdown
Naming note: `qwen3.6-27b` uses the dense Qwen hybrid-attention runtime shape,
while `qwen3.6-35b-a3b` uses the MoE runtime shape. The CLI model names stay
marketing/model identifiers; runtime code distinguishes architecture family
separately so production paths can share only the pieces that actually match.
```

- [ ] **Step 5: Run registry tests**

```bash
cargo test -p supersonic-core registry --lib
```

Expected: tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add crates/core/src/registry.rs crates/runtime/src/builders.rs docs/supported-matrix.md
git commit -m "core: distinguish Qwen model and architecture families"
```

---

### Task 5: Server Compatibility Gate For OpenCode And Hermes

**Files:**
- Modify: `scripts/openai_compat_smoke.mjs`
- Modify: `docs/server.md`
- Modify: `crates/server/tests/protocol_mock.rs`
- Test: `cargo test -p server --test protocol_mock`

- [ ] **Step 1: Ensure mock protocol tests cover Chat Completions and Responses**

In `crates/server/tests/protocol_mock.rs`, add test cases for:

```rust
// Chat Completions accepts developer messages and returns usage.
// Responses accepts previous_response_id tool-loop shapes without executing tools.
// Unsupported tool_choice="required" returns 400.
// Queue-full maps to 429.
```

Use existing mock server helpers in the file; do not require a GPU model.

- [ ] **Step 2: Add client mode labels to the smoke script**

Extend `scripts/openai_compat_smoke.mjs` to print a mode label before each endpoint:

```javascript
console.log("[smoke] chat.completions");
console.log("[smoke] responses");
console.log("[smoke] tokenize");
```

Keep request payloads unchanged unless a mock test shows a protocol gap.

- [ ] **Step 3: Update `docs/server.md` with production smoke expectations**

In the OpenAI SDK harness smoke section, add:

```markdown
For production refactor PRs, this smoke is the minimum client-compatibility
gate. It must cover Chat Completions, Responses create/get/delete, tokenization,
streaming, and explicit unsupported-feature errors. For DFlash server mode,
the smoke should be run with `--prefix-cache-disable` or rely on runtime policy
to disable prefix-cache admission automatically.
```

- [ ] **Step 4: Run protocol tests**

```bash
cargo test -p server --test protocol_mock
```

Expected: tests pass without GPU artifacts.

- [ ] **Step 5: Commit Task 5**

```bash
git add scripts/openai_compat_smoke.mjs docs/server.md crates/server/tests/protocol_mock.rs
git commit -m "server: define OpenAI-compatible production smoke"
```

---

### Task 6: Extract Qwen3.6 MoE Runtime Config From Env-Only Controls

**Files:**
- Modify: `crates/runner/src/qwen36_moe/vmm_config.rs`
- Modify: `crates/runner/src/qwen36_moe/telemetry.rs`
- Create: `crates/runtime/src/qwen36_moe_config.rs`
- Modify: `crates/runtime/src/lib.rs`
- Test: existing `vmm_config` tests plus new runtime config tests

- [ ] **Step 1: Add runtime-owned config types**

Create `crates/runtime/src/qwen36_moe_config.rs`:

```rust
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpertVmmMode {
    Auto,
    Disabled,
    Force,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeIslandPrefetchMode {
    Disabled,
    PreviousToken,
    PreviousTokenResidentOnly,
    Transition,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36MoeRuntimeConfig {
    pub vmm_mode: MoeExpertVmmMode,
    pub island_cap_experts: Option<usize>,
    pub protected_experts: Option<usize>,
    pub fixed_hot_experts: Option<usize>,
    pub prefetch_mode: MoeIslandPrefetchMode,
    pub prefetch_ranks: usize,
    pub async_prefetch: bool,
    pub async_staging_pages: usize,
    pub prefetch_evict: bool,
    pub prefetch_evict_min_probability: f64,
}

impl Default for Qwen36MoeRuntimeConfig {
    fn default() -> Self {
        Self {
            vmm_mode: MoeExpertVmmMode::Auto,
            island_cap_experts: None,
            protected_experts: None,
            fixed_hot_experts: None,
            prefetch_mode: MoeIslandPrefetchMode::Disabled,
            prefetch_ranks: 4,
            async_prefetch: false,
            async_staging_pages: 2,
            prefetch_evict: false,
            prefetch_evict_min_probability: 0.90,
        }
    }
}
```

- [ ] **Step 2: Export the config module**

In `crates/runtime/src/lib.rs`, add:

```rust
pub mod qwen36_moe_config;
```

- [ ] **Step 3: Move pure parsers before moving engine code**

Move only pure string parsing helpers from `runner/src/qwen36_moe/vmm_config.rs` into `runtime/src/qwen36_moe_config.rs`. Keep env reads in runner for now, but make them call runtime parsers.

Example parser:

```rust
pub fn parse_optional_positive_usize(name: &str, raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .map_err(|e| anyhow!("parse {name}={raw:?} as positive integer: {e}"))?;
    if value == 0 {
        anyhow::bail!("{name} must be > 0");
    }
    Ok(Some(value))
}
```

- [ ] **Step 4: Add parser tests**

In the new module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_positive_usize_accepts_absent_and_positive() {
        assert_eq!(parse_optional_positive_usize("X", None).unwrap(), None);
        assert_eq!(parse_optional_positive_usize("X", Some("7")).unwrap(), Some(7));
    }

    #[test]
    fn optional_positive_usize_rejects_zero_and_bad_values() {
        assert!(parse_optional_positive_usize("X", Some("0")).is_err());
        assert!(parse_optional_positive_usize("X", Some("abc")).is_err());
    }
}
```

- [ ] **Step 5: Run config tests**

```bash
cargo test -p supersonic-runtime qwen36_moe_config --lib
cargo test -p runner qwen36_moe --lib
```

Expected: runtime config tests pass; runner tests continue to pass or self-skip.

- [ ] **Step 6: Commit Task 6**

```bash
git add crates/runtime/src/lib.rs crates/runtime/src/qwen36_moe_config.rs crates/runner/src/qwen36_moe/vmm_config.rs crates/runner/src/qwen36_moe/telemetry.rs
git commit -m "runtime: introduce typed Qwen3.6 MoE runtime config"
```

---

### Task 7: Move Qwen3.6 MoE Pieces Behind Runtime-Facing Modules

**Files:**
- Modify: `crates/runtime/Cargo.toml`
- Modify: `crates/runtime/src/lib.rs`
- Create: `crates/runtime/src/qwen36_moe/`
- Modify: `crates/runner/src/lib.rs`
- Modify: `crates/runner/src/main.rs`
- Test: Qwen3.6 MoE parity/smoke tests that self-skip without model artifacts

- [ ] **Step 1: Add `qwen36_moe` dependency to runtime**

In `crates/runtime/Cargo.toml`, add:

```toml
qwen36_moe = { path = "../qwen36_moe" }
```

- [ ] **Step 2: Create a runtime Qwen3.6 MoE module shell**

Create `crates/runtime/src/qwen36_moe/mod.rs`:

```rust
//! Runtime-facing Qwen3.6 MoE surface.
//!
//! This module starts as a thin home for config and state contracts. Move
//! runner-owned implementation pieces here only when the move has a focused
//! parity or smoke gate.

pub mod config {
    pub use crate::qwen36_moe_config::*;
}
```

In `crates/runtime/src/lib.rs`, add:

```rust
pub mod qwen36_moe;
```

- [ ] **Step 3: Move data-only types first**

Move data-only types from `crates/runner/src/qwen36_moe/types.rs` that have no CLI dependency into `crates/runtime/src/qwen36_moe/types.rs`. Start with:

```rust
pub const HYBRID_FULL_ATTN_STRIDE: i32 = 4;

pub fn is_full_attn_layer(layer_idx: i32) -> bool {
    (layer_idx + 1) % HYBRID_FULL_ATTN_STRIDE == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionPair {
    pub rope: i32,
    pub cache: i32,
}
```

Keep re-exports in runner so existing tests compile:

```rust
pub use supersonic_runtime::qwen36_moe::types::{is_full_attn_layer, PositionPair, HYBRID_FULL_ATTN_STRIDE};
```

- [ ] **Step 4: Run compile checks before moving engine logic**

```bash
cargo check -p supersonic-runtime -p runner -p server
cargo test -p runner --test qwen36_moe_linear_state -- --nocapture
```

Expected: compile succeeds; model-artifact tests either pass or clearly self-skip.

- [ ] **Step 5: Move one implementation cluster per follow-up PR**

Use this order, each with its own compile/test commit:

```text
1. Position/state snapshot helpers.
2. Telemetry data structs that do not print CLI output.
3. Residency config and pure policy.
4. Engine construction inputs.
5. Production decode session wrapper.
```

Do not move CLI output, dry-run report printing, or lab-only profiling into runtime.

- [ ] **Step 6: Commit Task 7 shell and first type move**

```bash
git add crates/runtime/Cargo.toml crates/runtime/src/lib.rs crates/runtime/src/qwen36_moe crates/runner/src/lib.rs crates/runner/src/main.rs
git commit -m "runtime: start Qwen3.6 MoE runtime module"
```

---

### Task 8: Mechanical Split Of Qwen3.6 Kernel FFI

**Files:**
- Modify: `crates/kernel-ffi/src/qwen36_moe.rs`
- Create: `crates/kernel-ffi/src/qwen36_moe/`
- Modify: `crates/kernel-ffi/src/lib.rs`
- Test: `cargo check -p kernel-ffi` and kernel group guard

- [ ] **Step 1: Create submodules without moving behavior**

Create:

```text
crates/kernel-ffi/src/qwen36_moe/mod.rs
crates/kernel-ffi/src/qwen36_moe/descriptors.rs
crates/kernel-ffi/src/qwen36_moe/launch.rs
crates/kernel-ffi/src/qwen36_moe/prefill.rs
crates/kernel-ffi/src/qwen36_moe/persistent.rs
crates/kernel-ffi/src/qwen36_moe/profile.rs
```

Move public structs and FFI launch wrappers in this order:

```text
1. Descriptor structs and constants.
2. Profile/debug helpers.
3. Prefill launch wrappers.
4. Persistent decode launch wrappers.
5. Step launch wrappers.
```

- [ ] **Step 2: Keep public exports identical**

In `crates/kernel-ffi/src/qwen36_moe/mod.rs`, re-export moved items:

```rust
pub use descriptors::*;
pub use launch::*;
pub use persistent::*;
pub use prefill::*;
pub use profile::*;
```

The goal is that existing imports like `kernel_ffi::qwen36_moe::attn_step_launch` still compile.

- [ ] **Step 3: Run compile and manifest checks after each move chunk**

```bash
cargo check -p kernel-ffi
python3 tools/check-kernel-groups.py
```

Expected: both pass after every chunk.

- [ ] **Step 4: Run one Qwen3.6 compile consumer**

```bash
cargo check -p runner --bin qwen36_q4km_manifest_audit
```

Expected: compile succeeds.

- [ ] **Step 5: Commit Task 8**

```bash
git add crates/kernel-ffi/src/lib.rs crates/kernel-ffi/src/qwen36_moe.rs crates/kernel-ffi/src/qwen36_moe
git commit -m "kernel-ffi: split Qwen3.6 MoE host wrappers"
```

---

### Task 9: Production Telemetry And Metrics

**Files:**
- Modify: `crates/runtime/src/dflash.rs`
- Modify: `crates/runtime/src/generate.rs`
- Modify: `crates/server/src/routes/mod.rs`
- Modify: `crates/server/src/schemas.rs`
- Test: `crates/server/tests/protocol_mock.rs`

- [ ] **Step 1: Add runtime generation statistics**

Create this struct in `crates/runtime/src/generate.rs`:

```rust
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub dflash_rounds: Option<usize>,
    pub dflash_accepted_total: Option<usize>,
    pub decode_ms: Option<f64>,
}
```

- [ ] **Step 2: Extend done events without breaking routes**

Change `GenEvent::Done` to include `stats: GenerationStats`, then update route code to keep existing response fields:

```rust
GenEvent::Done {
    reason,
    stats,
}
```

Routes should continue using `stats.prompt_tokens`, `stats.completion_tokens`, and `stats.cached_prompt_tokens` for OpenAI usage.

- [ ] **Step 3: Record DFlash stats**

When `generate_dflash_greedy` returns `DFlashGenerateOutput`, populate:

```rust
GenerationStats {
    prompt_tokens,
    completion_tokens,
    cached_prompt_tokens: 0,
    dflash_rounds: Some(output.rounds_run),
    dflash_accepted_total: Some(output.accepted_total),
    decode_ms: Some(output.decode_ms),
}
```

- [ ] **Step 4: Expose metrics**

Add counters or gauges to `/metrics` output:

```text
supersonic_generation_active
supersonic_generation_queued
supersonic_dflash_last_rounds
supersonic_dflash_last_accepted_total
supersonic_dflash_last_decode_ms
```

Keep the first implementation in-memory and process-local. Do not add Prometheus dependencies unless the existing route already uses them.

- [ ] **Step 5: Run server protocol tests**

```bash
cargo test -p server --test protocol_mock
```

Expected: existing OpenAI response JSON remains compatible and metrics route includes the new names.

- [ ] **Step 6: Commit Task 9**

```bash
git add crates/runtime/src/dflash.rs crates/runtime/src/generate.rs crates/server/src/routes/mod.rs crates/server/src/schemas.rs crates/server/tests/protocol_mock.rs
git commit -m "server: expose production generation telemetry"
```

---

### Task 10: Production Lane Verification Pass

**Files:**
- Modify: `docs/benchmarks.md`
- Modify: `docs/server.md`
- Modify: `docs/performance.md` if new measured results are collected
- Test: production gates below

- [ ] **Step 1: Run non-GPU repository guards**

```bash
python3 tools/check-support-matrix.py
python3 tools/check-tool-inventory.py
python3 tools/check-kernel-groups.py
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run Rust compile and mock protocol checks**

```bash
cargo check -p supersonic-core -p supersonic-runtime -p server -p runner
cargo test -p server --test protocol_mock
```

Expected: compile and mock protocol tests pass.

- [ ] **Step 3: Run HIP/RDNA4 starter gate when hardware/artifacts are available**

```bash
HIP_VISIBLE_DEVICES=1 ./tests/gfx1201/run_matrix.sh
```

Expected: RDNA4 WMMA/int4 harness passes, direct Qwen3.6-27B smoke passes when `QWEN36_27B_MODEL_DIR` exists, Lucebox/DFlash smoke passes when target and draft artifacts exist.

- [ ] **Step 4: Run Qwen3.6 27B Lucebox benchmark when hardware/artifacts are available**

```bash
HIP_VISIBLE_DEVICES=0 \
python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
  --binary target/release/supersonic \
  --target-profile qwen36-27b-lucebox \
  --model-dir "$MODEL_DIR" \
  --backend hip \
  --context-size 512 \
  --n-gen 256 \
  --dflash \
  --dflash-draft-dir "$DRAFT_DIR" \
  --prompt-source jsonl \
  --prompt-format chatml-no-thinking \
  --lucebox-jsonl "$JSONL" \
  --out-json target/qwen36_production_refactor/current_10x256.json
```

Expected: generated-token counts and combined output hash match the historical production lane unless an intentional algorithm change was made and documented. Throughput should not regress outside normal run noise for boundary-only PRs.

- [ ] **Step 5: Run production server smoke when artifacts are available**

Start server:

```bash
SUPERSONIC_BACKENDS=hip HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
target/release/supersonic-serve \
  --backend hip \
  --model qwen3.6-27b \
  --model-dir "$MODEL_DIR" \
  --max-context 4096 \
  --q4km-gptq \
  --dflash \
  --dflash-draft-dir "$DRAFT_DIR" \
  --host 127.0.0.1 \
  --port 8013 \
  --api-key local \
  --no-download
```

Run smoke:

```bash
SUPERSONIC_BASE_URL=http://127.0.0.1:8013 \
SUPERSONIC_API_KEY=local \
node scripts/openai_compat_smoke.mjs
```

Expected: Chat Completions, Responses, streaming, tokenize/detokenize, and unsupported-feature checks pass.

- [ ] **Step 6: Commit verification docs if new artifacts are recorded**

```bash
git add docs/benchmarks.md docs/server.md docs/performance.md
git commit -m "docs: record production inference refactor validation"
```

Only commit this step if new measured results or commands were added.

---

## Execution Order

Recommended PR order:

1. Task 1 only: docs and gates.
2. Tasks 2 and 3: runtime/session policy foundation.
3. Task 4: Qwen taxonomy cleanup.
4. Task 5: server compatibility smoke.
5. Task 6: typed Qwen3.6 MoE config extraction.
6. Task 7: Qwen3.6 MoE runtime module shell and first type move.
7. Task 8: mechanical kernel-FFI split.
8. Task 9: telemetry.
9. Task 10: full production lane verification.

Stop after each PR if any of these happen:

- Qwen3.6 27B DFlash generated-token hash changes unexpectedly.
- `tests/gfx1201/run_matrix.sh` fails on a change that touched HIP kernel, Qwen3.6, DFlash, or runtime load policy.
- `supersonic-serve` loses an OpenAI-compatible behavior already covered by mock tests.
- kernel group validation fails after a kernel-FFI split.

## Self-Review

- Spec coverage: The plan covers runtime/server ownership, Qwen3.6/DFlash production lane, Qwen3.6 MoE follow-up, kernel-FFI split, support matrix, validation gates, OpenCode/Hermes-compatible server surface, and telemetry.
- Red-flag scan: No open implementation markers are present. Follow-up tasks are scoped by concrete files, commands, and expected results.
- Type consistency: `SessionFeatures`, `RuntimePolicy`, `RuntimeLane`, `ArchitectureFamily`, and `Qwen36MoeRuntimeConfig` are introduced before later tasks reference them.
- Scope check: This remains an umbrella plan, but each task is independently reviewable and produces either a docs-only gate, a runtime contract, a config extraction, a mechanical split, or a verification pass.
