# FLM Single Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen3.6 27B dense FLM main path open the FLM once, optionally verify hashes once, and reuse that opened source for config, tokenizer, and weights.

**Architecture:** Keep `FlmModelSource` as the runner-facing source object. Move the opened source inside `Qwen35Startup`, pass it to `load_qwen35_engine`, and add a weight-loading helper that consumes an already-open source instead of reopening the same FLM. Existing HF-directory and bake paths stay on their current code path.

**Tech Stack:** Rust workspace, `runner`, `model-store::BakedStore`, `qwen35::weights::Qwen35Weights`, HIP/ROCm env-gated smoke tests.

---

## File Structure

- Modify `crates/runner/src/bakes.rs`
  - Own shared FLM source option calculation.
  - Add an already-open FLM weight-loading helper.
  - Keep the existing standalone `load_qwen35_weights` fallback for callers that do not have a startup source.
- Modify `crates/runner/src/qwen35_startup.rs`
  - Remove startup-only FLM options.
  - Open FLM with final CLI verification options.
  - Store `Option<FlmModelSource>` in `Qwen35Startup`.
- Modify `crates/runner/src/qwen35_engine_setup.rs`
  - Accept `Option<&FlmModelSource>`.
  - Route FLM-backed runs to the already-open weight helper.
- Modify `crates/runner/src/qwen35_runtime.rs`
  - Carry `flm_source` from startup into engine setup.
- Modify `crates/runner/tests/flm_main_path.rs`
  - Assert single source-open logging.
  - Assert weight loading uses the already-open source.
- Modify `docs/testing.md`
  - Document the single-open smoke expectation.

## Task 1: Shared FLM Source Options

**Files:**
- Modify: `crates/runner/src/bakes.rs`
- Modify: `crates/runner/src/qwen35_startup.rs`

- [ ] **Step 1: Add failing option tests in `bakes.rs`**

Replace the existing multi-line `use super::` import block in the
`#[cfg(test)] mod tests` section with:

```rust
    use super::{
        effective_flm_source, effective_quant_profile, ensure_hf_metadata_present,
        flm_source_is_authoritative_for_model, flm_source_open_options, should_fetch_bake,
        should_fetch_exact_bake, validate_effective_flm_source_model,
        validate_flm_weight_source_options,
    };
```

Then add these tests near the existing FLM source tests:

```rust
#[test]
fn flm_source_open_options_enable_hash_verification_for_single_source() {
    let cli = cli_with_model_dir(
        "/tmp/model.flm",
        &["--int4", "--verify-flm-hashes"],
    );

    let options = flm_source_open_options(&cli).expect("valid FLM source options");

    assert!(options.int4_runtime);
    assert!(options.verify_block_hashes);
}

#[test]
fn flm_source_open_options_keep_hash_verification_opt_in() {
    let cli = cli_with_model_dir("/tmp/model.flm", &["--int4"]);

    let options = flm_source_open_options(&cli).expect("valid FLM source options");

    assert!(options.int4_runtime);
    assert!(!options.verify_block_hashes);
}
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
cargo test -q -p runner flm_source_open_options_enable_hash_verification_for_single_source
```

Expected: FAIL because `flm_source_open_options` does not exist.

- [ ] **Step 3: Implement shared source option helper in `bakes.rs`**

Update the imports at the top of `crates/runner/src/bakes.rs`:

```rust
use crate::flm_model_source::{is_flm_model_path, FlmModelSourceOptions};
```

Add this helper after `validate_flm_weight_source_options`:

```rust
pub(crate) fn flm_source_open_options(cli: &Cli) -> Result<FlmModelSourceOptions> {
    let profile = effective_quant_profile(cli)?;
    Ok(FlmModelSourceOptions {
        int4_runtime: profile.is_native_int4_runtime(),
        verify_block_hashes: cli.verify_flm_hashes,
    })
}
```

- [ ] **Step 4: Remove startup-only deferred hash tests**

In `crates/runner/src/qwen35_startup.rs`, remove `flm_startup_open_options` from
the test module import list and delete the whole function blocks named:

```text
flm_startup_open_options_track_cli_quant_flags
flm_startup_open_options_defer_payload_hash_verification_to_weights
flm_startup_open_options_track_weight_quant_profile_aliases
```

The replacement coverage now lives in `bakes.rs` because there is no separate
startup-only option policy after this stage.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cargo test -q -p runner flm_source_open_options
cargo test -q -p runner qwen35_startup
```

Expected: PASS for the new option tests and the remaining startup tests.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/bakes.rs crates/runner/src/qwen35_startup.rs
git commit -m "runner: share FLM source open options"
```

## Task 2: Startup Owns The Opened FLM Source

**Files:**
- Modify: `crates/runner/src/qwen35_startup.rs`

- [ ] **Step 1: Add source ownership to `Qwen35Startup`**

Change the struct definition:

```rust
pub(crate) struct Qwen35Startup {
    pub(crate) text_config: qwen35::config::TextConfig,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
    pub(crate) flm_source: Option<FlmModelSource>,
}
```

This will create compile failures at destructuring sites until Task 4 threads the
field forward.

- [ ] **Step 2: Use the shared option helper when opening startup FLM**

Update the imports in `crates/runner/src/qwen35_startup.rs`:

```rust
use crate::bakes::{flm_source_open_options, validate_effective_flm_source_model};
use crate::flm_model_source::{is_flm_model_path, FlmModelSource};
```

Replace `open_flm_startup_source` with:

```rust
fn open_flm_startup_source(cli: &Cli) -> Result<Option<FlmModelSource>> {
    let Some(path) = flm_config_path(cli) else {
        return Ok(None);
    };
    let options = flm_source_open_options(cli)?;
    eprintln!(
        "[flm] opening model source at {}{}{}",
        path.display(),
        if options.int4_runtime {
            " (FLM logical INT4 aliases enabled)"
        } else {
            ""
        },
        if options.verify_block_hashes {
            " (BLAKE3 hash verification enabled)"
        } else {
            ""
        }
    );
    FlmModelSource::open_with_options(path, options)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("opening FLM startup source {}: {e}", path.display()))
}
```

Delete the old private `flm_startup_open_options` function.

- [ ] **Step 3: Return the opened source from startup**

In `load_qwen35_startup`, include `flm_source` in the returned bundle:

```rust
    Ok(Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
        flm_source,
    })
```

- [ ] **Step 4: Run compiler check to expose threading sites**

Run:

```bash
cargo check -q -p runner
```

Expected: FAIL with a pattern/destructure error in `qwen35_runtime.rs` because
`Qwen35Startup` now has an additional field that is not handled.

- [ ] **Step 5: Commit after Task 4 resolves compile errors**

Do not commit this task independently if the tree does not compile. Commit it
together with Task 4 after the source is threaded into engine setup.

## Task 3: Already-Open FLM Weight Loading

**Files:**
- Modify: `crates/runner/src/bakes.rs`

- [ ] **Step 1: Import `FlmModelSource`**

Update the existing `flm_model_source` import in `crates/runner/src/bakes.rs`:

```rust
use crate::flm_model_source::{is_flm_model_path, FlmModelSource, FlmModelSourceOptions};
```

- [ ] **Step 2: Add the already-open weight helper**

Add this function after `flm_source_open_options`:

```rust
pub(crate) fn load_qwen35_weights_from_flm_source(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    ordinal: usize,
    weight_prefix: &str,
    q4km_like: bool,
    source: &FlmModelSource,
) -> Result<qwen35::weights::Qwen35Weights> {
    validate_effective_flm_source_model(cli, model_variant)?;
    validate_flm_weight_source_options(cli, q4km_like)?;

    eprintln!(
        "[weights] loading FLM weights from already-open source at {}",
        source.path.display()
    );
    qwen35::weights::Qwen35Weights::load_baked(
        &source.store,
        text_config,
        ordinal,
        weight_prefix,
    )
    .map_err(|e| anyhow::anyhow!("load FLM weights: {e}"))
}
```

- [ ] **Step 3: Make the standalone FLM path use the same helper**

In `load_qwen35_weights`, replace the current `if let Some(flm_file)` block with:

```rust
    if let Some(flm_file) = effective_flm_source(cli) {
        let options = flm_source_open_options(cli)?;
        eprintln!(
            "[flm] opening model source at {}{}{}",
            flm_file.display(),
            if options.int4_runtime {
                " (FLM logical INT4 aliases enabled)"
            } else {
                ""
            },
            if options.verify_block_hashes {
                " (BLAKE3 hash verification enabled)"
            } else {
                ""
            }
        );
        let source = FlmModelSource::open_with_options(flm_file, options)
            .map_err(|e| anyhow::anyhow!("open FLM store: {e}"))?;
        return load_qwen35_weights_from_flm_source(
            cli,
            model_variant,
            text_config,
            ordinal,
            weight_prefix,
            q4km_like,
            &source,
        );
    }
```

This preserves standalone callers that reach `load_qwen35_weights` without a
startup-owned source.

- [ ] **Step 4: Run focused compile check**

Run:

```bash
cargo check -q -p runner
```

Expected: still FAIL until Task 4 passes `flm_source` into `load_qwen35_engine`.

## Task 4: Thread FLM Source Into Engine Setup

**Files:**
- Modify: `crates/runner/src/qwen35_engine_setup.rs`
- Modify: `crates/runner/src/qwen35_runtime.rs`

- [ ] **Step 1: Update engine setup imports**

In `crates/runner/src/qwen35_engine_setup.rs`, replace the `bakes` import:

```rust
use crate::bakes::{load_qwen35_weights, load_qwen35_weights_from_flm_source};
use crate::flm_model_source::FlmModelSource;
```

- [ ] **Step 2: Add `flm_source` to `load_qwen35_engine`**

Add this parameter after `text_config`:

```rust
    flm_source: Option<&FlmModelSource>,
```

The top of the function should then load weights with:

```rust
    let t0 = std::time::Instant::now();
    let weights = if let Some(source) = flm_source {
        load_qwen35_weights_from_flm_source(
            cli,
            model_variant,
            text_config,
            ordinal,
            params.weight_prefix,
            q4km_like,
            source,
        )?
    } else {
        load_qwen35_weights(
            cli,
            model_variant,
            text_config,
            ordinal,
            params.weight_prefix,
            bootstrap_downloaded,
            q4km_like,
        )?
    };
```

- [ ] **Step 3: Destructure startup source in runtime**

In `crates/runner/src/qwen35_runtime.rs`, update the startup destructure:

```rust
    let Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
        flm_source,
    } = load_qwen35_startup(&cli)?;
```

- [ ] **Step 4: Pass the source into engine setup**

Update the `load_qwen35_engine` call:

```rust
    } = load_qwen35_engine(
        &cli,
        &model_variant,
        &text_config,
        flm_source.as_ref(),
        params,
        backend,
        gpu_arch,
        ordinal,
        bootstrap_downloaded,
        q4km_like,
        context_tokens,
    )?;
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cargo test -q -p runner qwen35_startup
cargo test -q -p runner flm_source_open_options
cargo check -q -p runner
```

Expected: PASS.

- [ ] **Step 6: Commit Tasks 2-4 together**

```bash
git add crates/runner/src/bakes.rs crates/runner/src/qwen35_startup.rs crates/runner/src/qwen35_engine_setup.rs crates/runner/src/qwen35_runtime.rs
git commit -m "runner: reuse opened FLM source for dense weights"
```

## Task 5: Strengthen Main-Path Smoke Assertions

**Files:**
- Modify: `crates/runner/tests/flm_main_path.rs`
- Modify: `docs/testing.md`

- [ ] **Step 1: Add a log occurrence helper**

In `crates/runner/tests/flm_main_path.rs`, add this helper after
`combined_output`:

```rust
#[cfg(target_os = "linux")]
fn occurrence_count(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}
```

- [ ] **Step 2: Replace old weight/hash assertions**

Replace the current assertions for `"[weights] loading FLM container"` and
`"BLAKE3 hash verification enabled"` with:

```rust
    assert_eq!(
        occurrence_count(&combined, "[flm] opening model source"),
        1,
        "FLM main path should open the source exactly once:\n{combined}"
    );
    assert!(
        combined.contains("[weights] loading FLM weights from already-open source"),
        "weights were not loaded from the already-open FLM source:\n{combined}"
    );
    assert!(
        combined.contains("BLAKE3 hash verification enabled"),
        "--verify-flm-hashes was not threaded to the single FLM source open:\n{combined}"
    );
    assert!(
        !combined.contains("[weights] loading FLM container"),
        "FLM main path reopened the FLM container during weight loading:\n{combined}"
    );
```

- [ ] **Step 3: Update testing docs**

In `docs/testing.md`, update the FLM Main-Path Smoke paragraph after the runner
command to:

```markdown
The runner smoke asserts that config, tokenizer, and weights come from FLM,
that BLAKE3 verification is enabled on the single FLM source open, that weights
load from the already-open source, and that no `[fetch]` or `[bake]` path is
entered.
```

- [ ] **Step 4: Run the smoke without the env var**

Run:

```bash
cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: PASS with a skip message when `SUPERSONIC_QWEN36_27B_NO_HF_FLM` is not set.

- [ ] **Step 5: Run the real HIP smoke**

Run:

```bash
SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: PASS. Output includes:

```text
[flm] opening model source
[weights] loading FLM weights from already-open source
BLAKE3 hash verification enabled
```

Output does not include:

```text
[weights] loading FLM container
[fetch]
[bake]
config.json
tokenizer.json
.supersonic
```

- [ ] **Step 6: Commit**

```bash
git add crates/runner/tests/flm_main_path.rs docs/testing.md
git commit -m "runner: assert single FLM source main path"
```

## Task 6: Verification Sweep

**Files:**
- No code changes expected.

- [ ] **Step 1: Format touched Rust files**

Run:

```bash
rustfmt --edition 2021 crates/runner/src/bakes.rs crates/runner/src/qwen35_startup.rs crates/runner/src/qwen35_engine_setup.rs crates/runner/src/qwen35_runtime.rs crates/runner/tests/flm_main_path.rs
```

Expected: no output.

- [ ] **Step 2: Run focused runner tests**

Run:

```bash
cargo test -q -p runner qwen35_startup
cargo test -q -p runner flm
```

Expected: PASS.

- [ ] **Step 3: Run model-store FLM tests**

Run:

```bash
cargo test -q -p model-store flm
```

Expected: PASS.

- [ ] **Step 4: Run real FLM smoke with verification**

Run:

```bash
SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Check formatting without touching known unrelated drift**

Run:

```bash
rustfmt --edition 2021 --check crates/runner/src/bakes.rs crates/runner/src/qwen35_startup.rs crates/runner/src/qwen35_engine_setup.rs crates/runner/src/qwen35_runtime.rs crates/runner/tests/flm_main_path.rs
git diff --check
```

Expected: PASS. Do not run full `cargo fmt --check` as the repository currently
has unrelated formatting drift in `crates/runner/src/bin/int4_test.rs`.

- [ ] **Step 6: Final status**

Run:

```bash
git status -sb
git log --oneline --decorate -5
```

Expected: branch `codex/flm-single-source` has the implementation commits and
no uncommitted changes.

## Self-Review

Spec coverage:

- Single source ownership is covered by Tasks 2 and 4.
- Single hash verification at source open is covered by Tasks 1 and 5.
- Already-open weight loading is covered by Tasks 3 and 4.
- HF/bake behavior preservation is covered by keeping the non-FLM branch in
  `load_qwen35_weights` and by the verification sweep.
- No-HF smoke behavior is covered by Task 5.

Placeholder scan:

- The plan contains exact files, code snippets, commands, and expected results.
- There are no deferred implementation placeholders.

Type consistency:

- `FlmModelSourceOptions` fields match `crates/runner/src/flm_model_source.rs`.
- `FlmModelSource` fields `path` and `store` match the current struct.
- `Qwen35Startup` and `load_qwen35_engine` signatures are updated at every call site.
