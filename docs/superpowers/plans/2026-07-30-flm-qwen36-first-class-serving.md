# FLM Qwen3.6 First-Class Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a self-contained Qwen3.6-35B-A3B native INT4 FLM once into a long-lived HIP `supersonic-serve` process and serve an OpenAI-compatible agentic coding workload without an HF snapshot or CLI subprocess.

**Architecture:** geo-quant adds a native UTF-8 chat-template asset and makes it mandatory for the strict SuperSonic Qwen3.6 profile. SuperSonic consumes that asset, moves the reusable FLM and production Qwen3.6 lifecycle from `runner` into `supersonic-runtime`, adapts both CLI and HTTP server to one persistent engine, and verifies the result with the existing CLI gate plus a real OpenAI SDK and coding-agent server gate.

**Tech Stack:** Python 3.11, pytest, Jinja2, Rust 2021, Cargo, Tokio, Axum, MiniJinja, tokenizers, ROCm/HIP, OpenAI Node SDK, OpenCode.

## Global Constraints

- SuperSonic worktree: `/home/deano/projects/SuperSonicBase/.worktree/flm-qwen36-serving`, branch `codex/flm-qwen36-serving`, based on merged `main` commit `c3012ff`.
- Create a separate geo-quant worktree from `origin/main` commit `c3f3fe74cd8016ffb0404ffc6cc62a11068d643d`; do not modify the dirty `/home/deano/projects/geo-quant` checkout.
- The native asset contract is exactly `kind_id=5`, `name="chat_template"`, flags `REQUIRED_FOR_RUNTIME | TEXT_UTF8`, payload equal to the selected UTF-8 Jinja source.
- Adding asset kind 5 does not bump FLM runtime-directory version 4 because no record layout changes.
- `--hf-compat-assets omit` must retain the native chat template and omit HF JSON compatibility assets.
- Existing FLMs remain readable by general profiles; only `supersonic-qwen36-moe-native-int4` newly requires the template.
- Normal serving must not read adjacent files, invoke HF metadata helpers, download a bake, or spawn the `supersonic` CLI.
- `supersonic-runtime` must not depend on `runner`; `server` must not depend on `runner`.
- Backend and environment parsing stay at CLI/server boundaries. `Qwen36MoeEngine` receives resolved policy.
- MTP, DFlash, sparse prefill, continuous batching, prefix snapshots, and hipFile are not added to the first server engine.
- The existing CLI verifier strings and structured evidence remain stable.
- Every behavior change follows RED -> GREEN -> REFACTOR; do not write production behavior before observing the targeted test fail.
- Use `CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target` for focused SuperSonic builds.

---

### Task 1: Emit the Native Chat Template from geo-quant

**Repository:** geo-quant worktree created from `origin/main`

**Files:**
- Modify: `pyproject.toml`
- Modify: `geoquant/formats/flm_runtime.py`
- Modify: `geoquant/formats/qwen36_flm_runtime.py`
- Modify: `tests/test_flm_runtime.py`
- Modify: `tests/test_qwen36_flm.py`

**Interfaces:**
- Produces: `ASSET_CHAT_TEMPLATE_UTF8 = 5`
- Produces: `validate_chat_template_payload(payload: bytes, *, context: str) -> str`
- Produces: `resolve_qwen36_chat_template(model_dir: str | Path) -> bytes`
- Produces: an unconditional native runtime asset consumed by Tasks 2 and 3

- [ ] **Step 1: Create the isolated geo-quant worktree**

```bash
cd /home/deano/projects/geo-quant
git check-ignore -q .worktrees
git worktree add .worktrees/flm-qwen36-serving -b codex/flm-qwen36-serving origin/main
cd .worktrees/flm-qwen36-serving
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q \
  tests/test_flm_runtime.py tests/test_qwen36_flm.py
```

Expected: the two baseline files pass. Record any unrelated baseline failure before proceeding.

- [ ] **Step 2: Write producer and round-trip tests**

Add tests that require:

```python
assert runtime.assets[5].kind_id == ASSET_CHAT_TEMPLATE_UTF8
assert runtime.assets[5].name == "chat_template"
assert runtime.assets[5].flags == (
    ASSET_FLAG_REQUIRED_FOR_RUNTIME | ASSET_FLAG_TEXT_UTF8
)
assert runtime.assets[5].payload == template_source.encode("utf-8")
```

Cover root `chat_template.jinja` precedence, string-valued
`tokenizer_config.json`, a named `default` template, omitted HF assets,
missing default, empty source, invalid UTF-8, and invalid Jinja.

- [ ] **Step 3: Run the focused tests and confirm RED**

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q tests/test_qwen36_flm.py \
  -k 'chat_template or omit_hf_compat_assets'
```

Expected: failures report missing constant/asset/resolver behavior.

- [ ] **Step 4: Implement resolution, validation, and emission**

Add `jinja2>=3.1` as a direct dependency. Implement:

```python
ASSET_CHAT_TEMPLATE_UTF8 = 5

def validate_chat_template_payload(payload: bytes, *, context: str) -> str:
    try:
        source = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context}: chat template is not UTF-8: {exc}") from exc
    if not source.strip():
        raise ValueError(f"{context}: chat template is empty")
    try:
        Environment().from_string(source)
    except TemplateError as exc:
        raise ValueError(f"{context}: invalid chat template: {exc}") from exc
    return source
```

`resolve_qwen36_chat_template` must prefer root `chat_template.jinja`, then
accept a string or exactly one named `default` from `tokenizer_config.json`.
Add asset 5 outside the HF compatibility condition.

- [ ] **Step 5: Run focused and complete producer tests**

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q \
  tests/test_flm_runtime.py tests/test_qwen36_flm.py \
  -k 'chat_template or omit_hf_compat_assets or asset_table'
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q \
  tests/test_flm_runtime.py tests/test_qwen36_flm.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml geoquant/formats/flm_runtime.py \
  geoquant/formats/qwen36_flm_runtime.py \
  tests/test_flm_runtime.py tests/test_qwen36_flm.py
git commit -m "feat(flm): embed native Qwen chat templates"
```

---

### Task 2: Strictly Validate and Document the Native Template

**Repository:** geo-quant worktree

**Files:**
- Modify: `geoquant/formats/flm_validate.py`
- Modify: `tests/test_qwen36_flm.py`
- Modify: `docs/fast-load-model-format-design.md`

**Interfaces:**
- Consumes: `validate_chat_template_payload` and `ASSET_CHAT_TEMPLATE_UTF8`
- Produces: `_validate_qwen36_chat_template_asset(runtime, issues) -> None`
- Produces: strict profile issue codes consumed by artifact gates

- [ ] **Step 1: Write malformed-artifact tests**

Add mutations covering missing and duplicate kind 5 plus wrong name, wrong
flags, whitespace-only payload, invalid UTF-8, and invalid Jinja. Assert:

```text
runtime.chat_template_missing
runtime.chat_template_duplicate
runtime.chat_template_name
runtime.chat_template_flags
runtime.chat_template_empty
runtime.chat_template_utf8
runtime.chat_template_syntax
```

- [ ] **Step 2: Run validator tests and confirm RED**

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q tests/test_qwen36_flm.py \
  -k 'supersonic_profile_rejects_ and chat_template'
```

Expected: malformed artifacts are incorrectly accepted or issue codes are absent.

- [ ] **Step 3: Implement strict profile validation**

Implement:

```python
def _validate_qwen36_chat_template_asset(
    runtime: FlmRuntimeDirectory,
    issues: list[FlmValidationIssue],
) -> None:
    matches = [
        asset for asset in runtime.assets.values()
        if asset.kind_id == ASSET_CHAT_TEMPLATE_UTF8
    ]
    # Report missing/duplicate before validating exact name, flags, and payload.
```

Call it only from `_validate_supersonic_qwen36_moe_native_int4_profile`.
Require exact flags and name; reuse the producer payload validator.

- [ ] **Step 4: Update the durable FLM design**

Document kind 5, exact flags/name, producer precedence, no version bump, strict
profile requirement, and that `--hf-compat-assets omit` does not remove the
native asset.

- [ ] **Step 5: Run focused and full geo-quant tests**

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q \
  tests/test_flm_runtime.py tests/test_qwen36_flm.py
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m pytest -q
```

Expected: the repository suite passes, or unrelated baseline failures are
reproduced on `origin/main` and documented.

- [ ] **Step 6: Prove the old canonical artifact is stale**

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --profile supersonic-qwen36-moe-native-int4
```

Expected: failure with `runtime.chat_template_missing`.

- [ ] **Step 7: Commit**

```bash
git add geoquant/formats/flm_validate.py tests/test_qwen36_flm.py \
  docs/fast-load-model-format-design.md
git commit -m "feat(flm): strictly validate native Qwen chat templates"
```

---

### Task 3: Add the Native Template ABI to SuperSonic

**Repository:** SuperSonic worktree

**Files:**
- Modify: `crates/model-store/src/flm.rs`
- Modify: `crates/model-store/src/store.rs`
- Test: `crates/model-store/src/flm.rs`
- Test: `crates/model-store/tests/flm_qwen36_native_layout.rs`

**Interfaces:**
- Produces: `ASSET_CHAT_TEMPLATE_UTF8`, `ASSET_FLAG_TEXT_UTF8`
- Produces: `FlmRuntimeDirectory::required_chat_template_source() -> Result<&str, Error>`
- Consumed by: runtime FLM source in Task 4

- [ ] **Step 1: Write parser and accessor tests**

Build a runtime fixture containing:

```rust
FlmAsset {
    asset_id: 5,
    kind_id: ASSET_CHAT_TEMPLATE_UTF8,
    flags: ASSET_FLAG_REQUIRED_FOR_RUNTIME | ASSET_FLAG_TEXT_UTF8,
    name: "chat_template".into(),
    payload: b"{% for message in messages %}{{ message.content }}{% endfor %}".to_vec(),
}
```

Test exact round-trip and reject missing, duplicate kind, wrong name, wrong
flags, invalid UTF-8, and empty source.

- [ ] **Step 2: Run model-store tests and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p model-store flm_chat_template -- --nocapture
```

Expected: compile failure for missing constants/accessor.

- [ ] **Step 3: Implement constants and strict accessor**

Add:

```rust
pub const ASSET_CHAT_TEMPLATE_UTF8: u16 = 5;
pub const ASSET_FLAG_TEXT_UTF8: u16 = 1 << 2;

impl FlmRuntimeDirectory {
    pub fn required_chat_template_source(&self) -> Result<&str, crate::Error> {
        // Select exactly one asset by kind_id, then require exact name/flags.
    }
}
```

Do not reject unknown future asset kinds during generic runtime parsing.

- [ ] **Step 4: Run model-store gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p model-store flm
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p model-store --test flm_qwen36_native_layout
```

- [ ] **Step 5: Commit**

```bash
git add crates/model-store/src/flm.rs crates/model-store/src/store.rs \
  crates/model-store/tests/flm_qwen36_native_layout.rs
git commit -m "feat(flm): read native chat template assets"
```

---

### Task 4: Move Generic FLM Source and Tokenizer Ownership into Runtime

**Files:**
- Create: `crates/runtime/src/flm_model_source.rs`
- Create: `crates/runtime/src/flm_tokenizer.rs`
- Modify: `crates/runtime/src/lib.rs`
- Modify: `crates/runner/src/flm_model_source.rs`
- Modify: `crates/runner/src/flm_tokenizer.rs`
- Modify: `crates/runner/src/lib.rs`
- Test: `crates/runner/tests/flm_model_source.rs`
- Test: `crates/runner/tests/flm_tokenizer.rs`

**Interfaces:**
- Produces: `supersonic_runtime::flm_model_source::FlmModelSource`
- Produces: `FlmModelSource::chat_template_source() -> Result<&str>`
- Produces: runtime-owned Qwen BPE loading and timings
- Preserves: runner compatibility re-exports

- [ ] **Step 1: Add runtime-facing integration assertions**

Change the two runner integration tests to import runtime types directly while
also asserting the old runner paths compile through re-exports.

- [ ] **Step 2: Run tests and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p runner --test flm_model_source --test flm_tokenizer
```

Expected: unresolved runtime modules.

- [ ] **Step 3: Move implementation without behavior changes**

Move the implementation and unit tests into runtime, then leave wrappers:

```rust
pub use supersonic_runtime::flm_model_source::*;
```

and:

```rust
pub use supersonic_runtime::flm_tokenizer::*;
```

Add `chat_template_source` by calling the model-store strict accessor. No
runner CLI type may appear in either runtime module.

- [ ] **Step 4: Run runtime and runner tests**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime flm_
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p runner --test flm_model_source --test flm_tokenizer
```

- [ ] **Step 5: Commit**

```bash
git add crates/runtime/src crates/runner/src crates/runner/tests/flm_model_source.rs \
  crates/runner/tests/flm_tokenizer.rs
git commit -m "refactor(runtime): own FLM source and tokenizer"
```

---

### Task 5: Move Qwen3.6 Direct-Plan Selection into Runtime

**Files:**
- Create: `crates/runtime/src/qwen36_moe/source.rs`
- Modify: `crates/runtime/src/qwen36_moe/mod.rs`
- Modify: `crates/runner/src/qwen36_moe/flm_source.rs`
- Test: `crates/runtime/src/qwen36_moe/source.rs`
- Test: `crates/runner/tests/flm_moe_main_path.rs`

**Interfaces:**
- Produces: `Qwen36MoeSource`, `Qwen36WeightMode`, `Qwen36MoeDirectProfile`
- Produces: `Qwen36MoeSource::open(path, options) -> Result<Self>`
- Requires: every required logical weight has a compatible direct plan

- [ ] **Step 1: Port the direct-profile tests to runtime**

Require exact outcomes:

```rust
assert_eq!(profile.native_int4, 330);
assert_eq!(profile.bf16_fallback, 0);
```

Retain synthetic small-config tests for missing, mixed, and wrong-kind plans.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::source
```

- [ ] **Step 3: Split source policy from CLI policy**

Implement a runtime source constructor taking only a path and
`FlmModelSourceOptions`. Keep CLI flag exclusion and model-argument matching
in runner. Runtime owns descriptor parsing, config, tokenizer, template,
weight mode, and direct profile.

- [ ] **Step 4: Restore the runner adapter**

Make runner `open_qwen36_moe_flm_source` validate CLI choices, construct the
runtime source, and preserve the existing evidence log strings.

- [ ] **Step 5: Run focused gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::source
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p runner --test flm_moe_main_path
```

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/qwen36_moe crates/runner/src/qwen36_moe/flm_source.rs \
  crates/runner/tests/flm_moe_main_path.rs
git commit -m "refactor(runtime): own Qwen3.6 FLM plan selection"
```

---

### Task 6: Move Concrete Qwen3.6 Buffers and Residency into Runtime

**Files:**
- Expand: `crates/runtime/src/qwen36_moe/types.rs`
- Expand: `crates/runtime/src/qwen36_moe/residency.rs`
- Create: `crates/runtime/src/qwen36_moe/residency_pages.rs`
- Create: `crates/runtime/src/qwen36_moe/prefetch.rs`
- Create: `crates/runtime/src/qwen36_moe/load_policy.rs`
- Modify: matching runner modules into compatibility adapters
- Test: existing runtime and runner residency tests

**Interfaces:**
- Produces: runtime-owned `MultiLayerGeom`, `LayerBuffers`, `ResidentWeight`
- Produces: runtime-owned `MoeExpertResidencyManager`
- Produces: resolved `Qwen36MoeLoadPolicy`; no environment parsing

- [ ] **Step 1: Add ownership and reset-contract tests**

Move pure residency tests first. Add a test proving mapped expert reservations,
statistics, and stable virtual addresses survive a mutable-state reset helper.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::residency
```

- [ ] **Step 3: Move leaf types and residency implementation**

Merge existing runtime contracts with runner implementations. Replace runner
files with re-exports/adapters. Keep file-writing and stdout telemetry in
runner.

- [ ] **Step 4: Move resolved VMM policy**

Expose:

```rust
pub struct Qwen36MoeLoadPolicy {
    pub persistent_decode: bool,
    pub kv_fp8: bool,
    pub kv_vmm: Qwen36KvVmmMode,
    pub moe: Qwen36MoeRuntimeConfig,
    pub virtual_transfer_backend: VirtualArenaTransferBackend,
}
```

CLI/environment translation remains in runner and server.

- [ ] **Step 5: Run residency, config, and compile gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe_config
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo check -p runner --lib --bin supersonic
```

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/qwen36_moe crates/runner/src/qwen36_moe \
  crates/runner/src/lib.rs
git commit -m "refactor(runtime): own Qwen3.6 resident buffers"
```

---

### Task 7: Move Production Layer Load, Decode, and Batched Prefill

**Files:**
- Create: `crates/runtime/src/qwen36_moe/geometry.rs`
- Create: `crates/runtime/src/qwen36_moe/weights.rs`
- Create: `crates/runtime/src/qwen36_moe/layers.rs`
- Create: `crates/runtime/src/qwen36_moe/layer_loader.rs`
- Create: `crates/runtime/src/qwen36_moe/lm_head.rs`
- Create: `crates/runtime/src/qwen36_moe/chain.rs`
- Create: `crates/runtime/src/qwen36_moe/decode.rs`
- Create: `crates/runtime/src/qwen36_moe/persistent_decode.rs`
- Create: `crates/runtime/src/qwen36_moe/prefill.rs`
- Modify: matching runner modules into adapters/re-exports

**Interfaces:**
- Produces: `LoadedQwen36Layers`
- Produces: production single-token chain and batched-prefill functions
- Excludes: CLI sampling, stdout, profiling files, MTP, sparse prefill

- [ ] **Step 1: Redirect one parity test at a time to runtime**

Start with multilayer decode, then persistent decode, then batched attention,
router/permute, and grouped expert parity tests.

- [ ] **Step 2: Confirm the first redirected test is RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p runner --test qwen36_moe_multilayer_parity --no-run
```

Expected: missing runtime exports.

- [ ] **Step 3: Move leaf modules in dependency order**

Use `supersonic_core::registry::Qwen36MoeKernelParams` in geometry. Remove
`Qwen36DecodeLoopState` and CLI `current_position` dependencies from batched
prefill; accept explicit token slices and dense positions.

- [ ] **Step 4: Add compatibility exports**

Use module aliases in `runner/src/lib.rs`, for example:

```rust
pub use supersonic_runtime::qwen36_moe::decode as qwen36_moe_decode;
pub use supersonic_runtime::qwen36_moe::persistent_decode
    as qwen36_moe_persistent_decode;
pub use supersonic_runtime::qwen36_moe::types as qwen36_moe_types;
```

- [ ] **Step 5: Run compile and parity gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo check -p supersonic-runtime -p runner --lib --bin supersonic
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p runner --test qwen36_moe_batched_prefill_attn_kernel_parity -- --nocapture
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p runner --test qwen36_moe_batched_prefill_router_permute_parity -- --nocapture
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p runner --test qwen36_moe_batched_prefill_grouped_expert_parity -- --nocapture
```

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/qwen36_moe crates/runner/src/qwen36_moe \
  crates/runner/src/lib.rs crates/runner/tests
git commit -m "refactor(runtime): own Qwen3.6 production decode"
```

---

### Task 8: Build the Persistent Qwen3.6 Engine and Reset Contract

**Files:**
- Create: `crates/runtime/src/qwen36_moe/engine.rs`
- Modify: `crates/runtime/src/qwen36_moe/mod.rs`
- Test: `crates/runtime/src/qwen36_moe/engine.rs`
- Create: `crates/runtime/tests/qwen36_moe_engine_hip.rs`

**Interfaces:**
- Produces: `Qwen36MoeLoadConfig`, `Qwen36MoeLoadEvidence`
- Produces: `Qwen36MoeEngine::load`, `reset`, `tokenizer`, `chat_template_source`, `eos_ids`

- [ ] **Step 1: Write load-policy and ownership tests**

Test empty/invalid context, non-HIP backend, fallback profile, mismatched model,
and pointer ownership. Require load evidence with positive native INT4 and zero
fallback.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::engine
```

- [ ] **Step 3: Implement the load API**

Use:

```rust
pub struct Qwen36MoeLoadConfig {
    pub flm_path: PathBuf,
    pub backend: Backend,
    pub device_ordinal: usize,
    pub max_context_len: usize,
    pub policy: Qwen36MoeLoadPolicy,
    pub verify_block_hashes: bool,
}
```

The engine explicitly owns source/store, tokenizer/template, layers/arenas,
residency manager, descriptors, state buffers, and scratch. Backend is set and
validated once during load.

- [ ] **Step 4: Write reset tests before reset implementation**

Dirty linear state, mapped KV ranges, counters, persistent scratch state, and
route history. Assert reset zeros mutable state while weight pointers, mapped
virtual addresses, source open count, and allocations remain unchanged.

- [ ] **Step 5: Implement reset**

Zero only mapped VMM ranges, never unmapped virtual address space. Return a
phase-labelled integrity error if any state reset fails.

- [ ] **Step 6: Run focused and HIP reset gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::engine
SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p supersonic-runtime --test qwen36_moe_engine_hip \
  load_and_reset_preserve_resident_model -- --nocapture
```

- [ ] **Step 7: Commit**

```bash
git add crates/runtime/src/qwen36_moe crates/runtime/tests/qwen36_moe_engine_hip.rs
git commit -m "feat(runtime): add persistent Qwen3.6 FLM engine"
```

---

### Task 9: Add Engine Prefill and Incremental Decode

**Files:**
- Modify: `crates/runtime/src/qwen36_moe/engine.rs`
- Modify: `crates/runtime/src/qwen36_moe/prefill.rs`
- Test: `crates/runtime/src/qwen36_moe/engine.rs`
- Test: `crates/runtime/tests/qwen36_moe_engine_hip.rs`

**Interfaces:**
- Produces: `prefill(&[u32]) -> Result<Vec<f32>>`
- Produces: `decode_step(token_id, absolute_pos) -> Result<Vec<f32>>`

- [ ] **Step 1: Write pure lifecycle validation tests**

Require empty prompt rejection, prompt/context overflow rejection, decode
before prefill rejection, duplicate/skipped position rejection, and reset
returning the engine to prefill-ready state.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe::engine::tests
```

- [ ] **Step 3: Implement prefill**

Process `prompt_ids[..len-1]` through batched prefill, process the final prompt
token through the production chain and LM head, return F32 logits, and set
`next_position = len`.

- [ ] **Step 4: Implement decode step**

Require `pos == next_position`, run one persistent production step, return full
F32 logits, and increment `next_position` only after success.

- [ ] **Step 5: Add the real lifecycle HIP test**

Run:

```text
load -> reset -> prefill(prompt) -> decode_step -> reset -> prefill(prompt)
```

Assert repeat-prefill logits are bit-identical, generated token count is
positive, and the second request does not change load sequence or weight
pointers.

- [ ] **Step 6: Run parity and real engine gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime qwen36_moe
SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test --release -p supersonic-runtime --test qwen36_moe_engine_hip -- --nocapture
```

- [ ] **Step 7: Commit**

```bash
git add crates/runtime/src/qwen36_moe crates/runtime/tests/qwen36_moe_engine_hip.rs
git commit -m "feat(runtime): serve Qwen3.6 prefill and decode"
```

---

### Task 10: Make the CLI Use the Runtime Engine

**Files:**
- Modify: `crates/runner/src/qwen36_moe/engine.rs`
- Modify: `crates/runner/src/qwen36_moe/mod.rs`
- Modify: `crates/runner/src/qwen36_moe/output.rs`
- Modify: `crates/runner/src/qwen36_moe/timing.rs`
- Modify: `crates/runner/src/lib.rs`
- Test: `crates/runner/tests/flm_moe_main_path.rs`
- Test: `tests/test_qwen36_flm_first_class_e2e.py`

**Interfaces:**
- Consumes: `Qwen36MoeEngine`
- Preserves: existing CLI flags, output evidence, and first-class verifier

- [ ] **Step 1: Add a test proving CLI engine construction delegates once**

Require one FLM source open and preserve all existing no-HF negative assertions.

- [ ] **Step 2: Run and confirm RED after removing direct CLI construction**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

- [ ] **Step 3: Rewrite the standard FLM path as an adapter**

The CLI retains sampling, stdout, stage timing, profiling, and verifier
evidence. Plain FLM prefill/decode calls runtime engine methods. Keep
experimental MTP/specprefill/legacy branches separate until later extraction.

- [ ] **Step 4: Preserve exact verifier evidence**

Continue emitting:

```text
[qwen36-moe] FLM weight mode: INT4 native FLM
[qwen36-moe] FLM direct plans: required=... raw_dense=... native_int4=... bf16_fallback=0
[FLM runtime weights] ready-for-decode: YES
```

- [ ] **Step 5: Run CLI and Python gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p runner --test flm_moe_main_path
python3 -m unittest tests.test_qwen36_he_supersonic_bench \
  tests.test_qwen36_flm_first_class_e2e -v
```

- [ ] **Step 6: Run the real CLI reuse gate**

```bash
python3 tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  --geoquant-root /home/deano/projects/geo-quant/.worktrees/flm-qwen36-serving \
  --geoquant-python /home/deano/projects/geo-quant/.venv-rocm/bin/python \
  --flm /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --binary /home/deano/projects/SuperSonicBase/target/release/supersonic \
  --out-json target/qwen36_35b_a3b_flm_first_class_e2e.json
```

- [ ] **Step 7: Commit**

```bash
git add crates/runner tests/test_qwen36_flm_first_class_e2e.py
git commit -m "refactor(runner): use persistent Qwen3.6 runtime engine"
```

---

### Task 11: Add First-Class FLM Server Source Resolution

**Files:**
- Modify: `crates/server/src/main.rs`
- Modify: `crates/runtime/src/state.rs`
- Create: `crates/runtime/src/model_source.rs`
- Test: `crates/runtime/src/state.rs`
- Test: `crates/server/tests/protocol_mock.rs`

**Interfaces:**
- Produces: `ModelSource::{Directory(PathBuf), Flm(PathBuf)}`
- Produces: `resolve_model_source(flm_file, model_dir, explicit_model)`
- Consumes: FLM model descriptor; constructs `Qwen36MoeLoadConfig`

- [ ] **Step 1: Write CLI/source policy tests**

Cover `--flm-file`, file-valued `--model-dir`, mutual exclusion, model omitted,
matching model, mismatching model, and rejection of quant/DFlash flags.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime model_source
```

- [ ] **Step 3: Implement source resolution**

Change `Cli.model` to `Option<String>`, add `flm_file: Option<PathBuf>`, and
normalize a file-valued model-dir. Directory startup still requires model.
FLM startup resolves model from the descriptor and validates an optional
explicit model.

- [ ] **Step 4: Bypass HF and bake paths for FLM**

Branch before `ensure_hf_metadata_present`, filesystem tokenizer/template
loading, and bake builders. Construct the runtime engine from the FLM source.

- [ ] **Step 5: Run state/server tests**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p supersonic-runtime state
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p server --test protocol_mock
```

- [ ] **Step 6: Commit**

```bash
git add crates/server/src/main.rs crates/runtime/src/model_source.rs \
  crates/runtime/src/state.rs crates/server/tests/protocol_mock.rs
git commit -m "feat(server): resolve Qwen3.6 models from FLM"
```

---

### Task 12: Integrate the Qwen3.6 Session and Truthful Cache Capabilities

**Files:**
- Modify: `crates/runtime/src/session.rs`
- Modify: `crates/runtime/src/generate.rs`
- Modify: `crates/runtime/src/prefix_cache.rs`
- Modify: `crates/runtime/src/state.rs`
- Test: corresponding runtime modules

**Interfaces:**
- Produces: `InferenceSession::Qwen36Moe(Qwen36MoeEngine)`
- Produces: feature report with plain decode true and all snapshot/speculative fields false
- Produces: `SessionFailureClass::{RequestLocal, IntegrityLost}`

- [ ] **Step 1: Write session feature and dispatch tests**

Require Qwen3.6 reset/prefill/decode dispatch and explicit unsupported snapshot
errors. Add a deterministic session-boundary test double for generate tests.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p supersonic-runtime session::tests generate::tests
```

- [ ] **Step 3: Implement session dispatch**

Add the variant to every exhaustive match. `prefix_snapshot_bytes` returns
`usize::MAX`; snapshot/load/restore return typed unsupported errors.

- [ ] **Step 4: Bypass prefix cache before lookup/capture**

In generation, check `features.prefix_snapshot` before any cache lookup,
anchor-prefill, size estimate, or capture. Qwen3.6 usage reports
`cached_tokens=0`.

- [ ] **Step 5: Add integrity failure classification**

Classify GPU device loss, resident descriptor/pointer errors, and reset failure
as `IntegrityLost`; context, sampling, protocol, and cancellation errors are
`RequestLocal`. Add an atomic readiness state to `ServerState`.

- [ ] **Step 6: Run runtime and server protocol tests**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p supersonic-runtime
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p server --test protocol_mock
```

- [ ] **Step 7: Commit**

```bash
git add crates/runtime/src/session.rs crates/runtime/src/generate.rs \
  crates/runtime/src/prefix_cache.rs crates/runtime/src/state.rs
git commit -m "feat(server): run persistent Qwen3.6 FLM sessions"
```

---

### Task 13: Expose FLM Readiness, Evidence, and Metrics

**Files:**
- Modify: `crates/core/src/capabilities.rs`
- Modify: `crates/runtime/src/state.rs`
- Modify: `crates/server/src/schemas.rs`
- Modify: `crates/server/src/routes/models.rs`
- Test: `crates/server/tests/protocol_mock.rs`

**Interfaces:**
- Consumes: immutable `Qwen36MoeLoadEvidence`
- Produces: basename-only HTTP evidence and process-local load sequence
- Produces: Prometheus metrics for load/direct profile and request lifecycle

- [ ] **Step 1: Write response and metric tests**

Assert readiness false before load and after integrity loss; assert capabilities
show FLM source, model id, native INT4 count > 0, fallback 0, snapshots false,
and load sequence 1. Assert no absolute source path appears.

- [ ] **Step 2: Run and confirm RED**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -p server --test protocol_mock capabilities
```

- [ ] **Step 3: Extend schemas and routes**

Add an optional `flm` evidence object to capabilities/health and render metrics:

```text
supersonic_model_loads_total 1
supersonic_flm_native_int4_direct_weights <positive>
supersonic_flm_bf16_fallback_weights 0
supersonic_flm_source_bytes <positive>
supersonic_flm_device_upload_bytes <positive>
supersonic_flm_startup_seconds <positive>
```

- [ ] **Step 4: Run protocol and support gates**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p server --test protocol_mock
python3 tools/check-support-matrix.py
```

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/capabilities.rs crates/runtime/src/state.rs \
  crates/server/src/schemas.rs crates/server/src/routes/models.rs \
  crates/server/tests/protocol_mock.rs
git commit -m "feat(server): report first-class FLM load evidence"
```

---

### Task 14: Add the Qwen3.6 FLM Server Protocol Harness

**Files:**
- Create: `tests/gfx1100/run_qwen36_flm_server_e2e.py`
- Create: `tests/test_qwen36_flm_server_e2e.py`
- Modify: `scripts/openai_compat_smoke.mjs`
- Create: `scripts/openai_agent_tool_smoke.mjs`
- Modify: `docs/server.md`
- Modify: `docs/testing.md`

**Interfaces:**
- Produces: a phase-aware server lifecycle harness and structured JSON report
- Exercises: OpenAI SDK, tool call/result loop, streaming cancellation, load count

- [ ] **Step 1: Write Python command/report tests**

Test server command uses only `--flm-file`, backend/device/context, host/port,
and deployment flags. Test startup timeout, readiness failure, subprocess
cleanup, structured evidence validation, and load-count invariance.

- [ ] **Step 2: Run and confirm RED**

```bash
python3 -m unittest tests.test_qwen36_flm_server_e2e -v
```

Expected: missing harness/functions.

- [ ] **Step 3: Implement the server harness**

The harness must allocate an unused loopback port, start a new process group,
poll `/ready`, run SDK scripts, terminate/reap on every path, and write:

```json
{
  "model": "qwen3.6-35b-a3b",
  "source": "flm",
  "load_sequence": 1,
  "native_int4": 330,
  "bf16_fallback": 0,
  "requests": {},
  "startup": {},
  "throughput": {},
  "cancellation": {}
}
```

Validate values strictly; reject booleans as integers and non-finite numbers.

- [ ] **Step 4: Add a deterministic tool-loop SDK smoke**

Submit a coding-style function tool, accept either a model-generated valid
tool call or fail with captured raw output, send the function result through
Chat Completions and Responses continuation, and require a subsequent
assistant result. Add a streaming request that aborts after the first delta.

- [ ] **Step 5: Document startup and acceptance commands**

Document:

```bash
target/release/supersonic-serve \
  --flm-file /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --backend hip --device 0 --max-context 4096 \
  --host 127.0.0.1 --port 8080 --api-key local --no-download
```

- [ ] **Step 6: Run unit and protocol tests**

```bash
python3 -m unittest tests.test_qwen36_flm_server_e2e -v
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p server
```

- [ ] **Step 7: Commit**

```bash
git add tests/gfx1100/run_qwen36_flm_server_e2e.py \
  tests/test_qwen36_flm_server_e2e.py scripts/openai_compat_smoke.mjs \
  scripts/openai_agent_tool_smoke.mjs docs/server.md docs/testing.md
git commit -m "test(server): gate Qwen3.6 FLM agent serving"
```

---

### Task 15: Regenerate the Canonical FLM and Run Complete Acceptance

**Repositories:** both clean worktrees

**Files:**
- Artifact: `/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm`
- Report: `target/qwen36_35b_a3b_flm_first_class_e2e.json`
- Report: `target/qwen36_35b_a3b_flm_server_e2e.json`

**Interfaces:**
- Proves: producer, strict format, CLI parity, persistent server, HTTP compatibility, agent tool loop

- [ ] **Step 1: Build release binaries with HIP**

```bash
cd /home/deano/projects/SuperSonicBase/.worktree/flm-qwen36-serving
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100 \
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo build --release -p runner -p server
```

- [ ] **Step 2: Regenerate atomically through the accepted producer gate**

```bash
python3 tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  --regenerate \
  --geoquant-root /home/deano/projects/geo-quant/.worktrees/flm-qwen36-serving \
  --geoquant-python /home/deano/projects/geo-quant/.venv-rocm/bin/python \
  --hf-source /mnt/data/models/Qwen3.6-35B-A3B \
  --flm /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --binary /home/deano/projects/SuperSonicBase/target/release/supersonic \
  --out-json target/qwen36_35b_a3b_flm_first_class_e2e.json \
  --export-timeout 7200 \
  --validation-timeout 1800
```

Expected: native template present, all BLAKE3 payload hashes pass, CLI emits at
least one token, native INT4 coverage is positive, fallback is zero.

- [ ] **Step 3: Run the real persistent server gate**

```bash
python3 tests/gfx1100/run_qwen36_flm_server_e2e.py \
  --flm /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --binary /home/deano/projects/SuperSonicBase/target/release/supersonic-serve \
  --device 0 \
  --max-context 4096 \
  --out-json target/qwen36_35b_a3b_flm_server_e2e.json
```

Expected: all endpoint, streaming, tool-loop, cancellation, evidence, and
single-load assertions pass.

- [ ] **Step 4: Run a real OpenCode smoke**

Start the server on a fixed local port, configure an
`@ai-sdk/openai-compatible` provider with model `qwen3.6-35b-a3b`, and run one
agent request that invokes a harmless filesystem-read tool. Require a complete
tool result continuation and record elapsed time plus server request metrics.

- [ ] **Step 5: Run focused regression suites**

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p model-store -p supersonic-runtime -p server
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p runner --test flm_moe_main_path
python3 -m unittest tests.test_qwen36_flm_first_class_e2e \
  tests.test_qwen36_flm_server_e2e tests.test_support_matrix -v
python3 tools/check-support-matrix.py
git diff --check
```

- [ ] **Step 6: Review and publish geo-quant**

Request code review, fix findings test-first, push the geo-quant branch, and
open a draft PR describing the native template ABI and strict-profile
compatibility boundary.

- [ ] **Step 7: Review and publish SuperSonic**

Request whole-branch code review, fix findings test-first, push
`codex/flm-qwen36-serving`, and open a draft PR containing the engine
extraction, server integration, real ROCm evidence, OpenCode smoke, and any
confirmed pre-existing workspace baseline failures.

## Plan Self-Review Checklist

- Every requirement in the approved design maps to Tasks 1-15.
- The geo-quant worktree is isolated from the dirty checkout.
- No task introduces a runtime/runner dependency cycle.
- Native template production precedes strict consumer/server validation.
- Engine extraction preserves the CLI before server integration begins.
- Prefix caching is bypassed truthfully rather than pretending snapshots work.
- Streaming is token-live through the existing generation channel.
- Real acceptance proves a second request does not reload the model.
- MTP, DFlash, continuous batching, prefix snapshots, and hipFile remain later stages.
- The final gate includes an actual coding-agent tool loop, not only curl.
