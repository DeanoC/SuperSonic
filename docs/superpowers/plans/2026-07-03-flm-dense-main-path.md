# FLM Dense Main Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a no-HF Qwen3.6 27B dense INT4 FLM file a first-class SuperSonic model source for config, tokenizer, weights, and a real one-token HIP smoke.

**Architecture:** Keep FLM as a source format that feeds the existing Qwen dense runtime. Add a small typed FLM source wrapper so startup and weights use the same runtime flags, make startup open the FLM once for config/tokenizer, preserve explicit policy rejections for unsupported paths, and add an env-gated e2e smoke that proves the normal `--model-dir model.flm` path needs no HF snapshot or bake directory.

**Tech Stack:** Rust workspace, `runner`, `model-store`, `qwen35`, HIP/ROCm via `gpu-hal`, geo-quant FLM validator for artifact preflight.

---

All paths below are relative to `/home/deano/projects/SuperSonicBase/.worktree/flm-main-model-path`.

## Scope And Ground Rules

This is the "single path complete first" stage. The supported runtime target is:

- `--model qwen3.6-27b`
- `--model-dir /path/to/qwen36-27b-int4-stage3-direct.flm`
- `--int4`
- `--verify-flm-hashes`
- HIP backend
- native FLM runtime config and tokenizer assets
- no HF `config.json`, no HF `tokenizer.json`, no safetensors shards, no `.supersonic` bake directory

Do not widen this stage to Qwen3.6 35B-A3B MoE, DFlash, SpecPrefill, Q4KM, INT8, NVFP4/MXFP8 dense decode, GPUDirect, or O_DIRECT load planning. Existing direct-view upload tests for NVFP4/MXFP8 remain useful storage-path checks, but the main decode smoke in this plan uses the validated INT4 FLM artifact.

The known good local artifact for full verification is:

```bash
/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm
```

It was validated with:

```bash
/home/deano/.config/superpowers/worktrees/geo-quant/flm-export/.venv-rocm/bin/python \
  -m geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  --profile runnable-no-hf \
  --verify-payload-hashes
```

Expected output:

```text
[flm-validate] OK path=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm profile=runnable-no-hf tensors=1998 warnings=0
```

Avoid `/mnt/data/runs/geo-quant/qwen36-27b-int4-runnable.flm`; it currently fails FLM head CRC validation.

## Files And Responsibilities

- `crates/runner/src/flm_model_source.rs` owns runner-facing FLM source opening options and runtime access helpers.
- `crates/runner/src/qwen35_startup.rs` owns Qwen dense startup config/tokenizer resolution and should open FLM startup metadata once.
- `crates/runner/src/bakes.rs` owns HF metadata bootstrap, bake selection, and FLM weight loading policy.
- `crates/runner/tests/flm_main_path.rs` will own the env-gated end-to-end no-HF FLM smoke through the `supersonic` binary.
- `docs/testing.md` owns operator/developer commands for env-gated model and GPU tests.

---

### Task 1: Typed FLM Source Open Options

**Files:**
- Modify: `crates/runner/src/flm_model_source.rs`
- Test: `crates/runner/src/flm_model_source.rs`

- [ ] **Step 1: Add failing tests for FLM open option mapping**

In `crates/runner/src/flm_model_source.rs`, replace the current `#[cfg(test)] mod tests` block with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_flm_model_paths_by_extension() {
        assert!(is_flm_model_path(std::path::Path::new("qwen36.flm")));
        assert!(is_flm_model_path(std::path::Path::new("QWEN36.FLM")));
        assert!(!is_flm_model_path(std::path::Path::new("qwen36")));
        assert!(!is_flm_model_path(std::path::Path::new("qwen36.bin")));
    }

    #[test]
    fn open_options_enable_int4_aliases_and_hash_verification() {
        let load = FlmModelSourceOptions {
            int4_runtime: true,
            verify_block_hashes: true,
        }
        .to_load_options();

        assert!(load.flm_int4_logical_aliases);
        assert!(load.verify_block_hashes);
    }

    #[test]
    fn default_open_options_do_not_enable_runtime_conversions_or_hashes() {
        let load = FlmModelSourceOptions::default().to_load_options();

        assert!(!load.flm_int4_logical_aliases);
        assert!(!load.verify_block_hashes);
    }
}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cargo test -q -p runner open_options_enable_int4_aliases_and_hash_verification
```

Expected: compile failure mentioning `FlmModelSourceOptions` is not found.

- [ ] **Step 3: Implement typed options and runtime helpers**

In `crates/runner/src/flm_model_source.rs`, add this struct above `pub struct FlmModelSource`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlmModelSourceOptions {
    pub int4_runtime: bool,
    pub verify_block_hashes: bool,
}

impl FlmModelSourceOptions {
    pub fn to_load_options(self) -> model_store::FlmLoadOptions {
        model_store::FlmLoadOptions {
            flm_int4_logical_aliases: self.int4_runtime,
            verify_block_hashes: self.verify_block_hashes,
        }
    }
}
```

Then replace the current `impl FlmModelSource` block with:

```rust
impl FlmModelSource {
    pub fn open(
        path: &std::path::Path,
        int4_runtime: bool,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(
            path,
            FlmModelSourceOptions {
                int4_runtime,
                ..Default::default()
            },
        )
    }

    pub fn open_with_options(
        path: &std::path::Path,
        options: FlmModelSourceOptions,
    ) -> anyhow::Result<Self> {
        let store = model_store::BakedStore::open_flm_with_options(
            path,
            options.to_load_options(),
        )?;
        Ok(Self {
            path: path.to_path_buf(),
            store,
        })
    }

    pub fn runtime(&self) -> anyhow::Result<&model_store::FlmRuntimeDirectory> {
        self.store.flm_runtime().ok_or_else(|| {
            anyhow::anyhow!("FLM {} has no runtime directory", self.path.display())
        })
    }

    pub fn qwen_config(&self) -> anyhow::Result<qwen35::config::Config> {
        let runtime = self.runtime()?;
        let cfg = runtime.qwen36_config().ok_or_else(|| {
            anyhow::anyhow!("FLM {} is not Qwen3.6 dense v1", self.path.display())
        })?;
        let config = qwen35::config::Config::try_from_flm_qwen36_dense(cfg).map_err(|e| {
            anyhow::anyhow!(
                "invalid FLM Qwen3.6 dense config in {}: {e}",
                self.path.display()
            )
        })?;
        Ok(config.normalized())
    }

    pub fn qwen_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        crate::flm_tokenizer::load_qwen_bpe_from_flm(self.runtime()?)
            .map_err(|e| anyhow::anyhow!("loading FLM Qwen tokenizer: {e}"))
    }
}
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
cargo test -q -p runner flm_model_source
```

Expected: the `flm_model_source` unit and integration tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add crates/runner/src/flm_model_source.rs
git commit -m "runner: type FLM source open options"
```

---

### Task 2: Verified Single FLM Startup Source

**Files:**
- Modify: `crates/runner/src/qwen35_startup.rs`
- Test: `crates/runner/src/qwen35_startup.rs`

- [ ] **Step 1: Add a startup option regression test**

In `crates/runner/src/qwen35_startup.rs`, update the test import block from:

```rust
    use super::{
        flm_config_path, qwen_tokenizer_source, validate_qwen35_startup, QwenTokenizerSource,
    };
```

to:

```rust
    use super::{
        flm_config_path, flm_startup_open_options, qwen_tokenizer_source,
        validate_qwen35_startup, QwenTokenizerSource,
    };
```

Then add this test after `effective_flm_source_selects_flm_native_tokenizer_without_tokenizer_json`:

```rust
    #[test]
    fn flm_startup_open_options_track_cli_runtime_flags() {
        let cli = cli(
            "/tmp/model.flm",
            &["--int4", "--verify-flm-hashes"],
        );

        let options = flm_startup_open_options(&cli);

        assert!(options.int4_runtime);
        assert!(options.verify_block_hashes);
    }
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cargo test -q -p runner flm_startup_open_options_track_cli_runtime_flags
```

Expected: compile failure mentioning `flm_startup_open_options` is not found.

- [ ] **Step 3: Thread `FlmModelSourceOptions` into startup**

In `crates/runner/src/qwen35_startup.rs`, change this import:

```rust
use crate::flm_model_source::{is_flm_model_path, FlmModelSource};
use crate::flm_tokenizer::load_qwen_bpe_from_flm;
```

to:

```rust
use crate::flm_model_source::{is_flm_model_path, FlmModelSource, FlmModelSourceOptions};
```

Replace `load_qwen35_startup` with:

```rust
pub(crate) fn load_qwen35_startup(cli: &Cli) -> Result<Qwen35Startup> {
    let flm_source = open_flm_startup_source(cli)?;
    let config = load_qwen35_config(cli, flm_source.as_ref())?;
    let text_config = config.text_config;
    eprintln!(
        "[config] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    );

    let tokenizer = load_qwen_tokenizer(cli, flm_source.as_ref())?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);

    Ok(Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    })
}
```

Add these helper functions below `load_qwen35_startup`:

```rust
fn open_flm_startup_source(cli: &Cli) -> Result<Option<FlmModelSource>> {
    let Some(path) = flm_config_path(cli) else {
        return Ok(None);
    };
    FlmModelSource::open_with_options(path, flm_startup_open_options(cli))
        .map(Some)
        .map_err(|e| anyhow::anyhow!("opening FLM startup source {}: {e}", path.display()))
}

fn flm_startup_open_options(cli: &Cli) -> FlmModelSourceOptions {
    FlmModelSourceOptions {
        int4_runtime: cli.int4,
        verify_block_hashes: cli.verify_flm_hashes,
    }
}
```

Replace `fn load_qwen35_config(cli: &Cli) -> Result<qwen35::config::Config>` with:

```rust
fn load_qwen35_config(
    cli: &Cli,
    flm_source: Option<&FlmModelSource>,
) -> Result<qwen35::config::Config> {
    if let Some(source) = flm_source {
        return load_flm_qwen35_config(source);
    }

    qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))
}
```

Replace `load_flm_qwen35_config` with:

```rust
fn load_flm_qwen35_config(source: &FlmModelSource) -> Result<qwen35::config::Config> {
    eprintln!(
        "[config] loading FLM runtime descriptor at {}",
        source.path.display()
    );
    source
        .qwen_config()
        .map_err(|e| anyhow::anyhow!("loading FLM Qwen config: {e}"))
}
```

Replace `fn load_qwen_tokenizer(cli: &Cli) -> Result<tokenizers::Tokenizer>` with:

```rust
fn load_qwen_tokenizer(
    cli: &Cli,
    flm_source: Option<&FlmModelSource>,
) -> Result<tokenizers::Tokenizer> {
    match qwen_tokenizer_source(cli) {
        QwenTokenizerSource::Flm(path) => {
            eprintln!(
                "[tokenizer] loading FLM tokenizer assets at {}",
                path.display()
            );
            let source = flm_source.ok_or_else(|| {
                anyhow::anyhow!(
                    "internal error: FLM tokenizer source {} was not opened",
                    path.display()
                )
            })?;
            source.qwen_tokenizer()
        }
        QwenTokenizerSource::TokenizerJson(model_dir) => {
            let tokenizer_path = model_dir.join("tokenizer.json");
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))
        }
    }
}
```

- [ ] **Step 4: Update old `FlmModelSource::open` call sites**

Run:

```bash
rg -n "FlmModelSource::open" crates/runner/src crates/runner/tests
```

Expected output after Task 2 implementation:

```text
crates/runner/src/qwen35_startup.rs:<line>:    FlmModelSource::open_with_options(path, flm_startup_open_options(cli))
```

There should be no remaining `FlmModelSource::open(path, cli.int4)` calls.

- [ ] **Step 5: Run startup-focused tests**

Run:

```bash
cargo test -q -p runner qwen35_startup
cargo test -q -p runner flm_startup_open_options_track_cli_runtime_flags
```

Expected: both commands pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add crates/runner/src/qwen35_startup.rs
git commit -m "runner: open FLM startup source with verified options"
```

---

### Task 3: Lock No-HF Bootstrap Policy And Document The Gate

**Files:**
- Modify: `crates/runner/src/bakes.rs`
- Modify: `docs/testing.md`
- Test: `crates/runner/src/bakes.rs`

- [ ] **Step 1: Add a regression test for FLM metadata bootstrap bypass**

In `crates/runner/src/bakes.rs`, update the `use super::{...};` list in the test module from:

```rust
    use super::{
        effective_flm_source, effective_quant_profile, flm_source_is_authoritative_for_model,
        should_fetch_bake, should_fetch_exact_bake, validate_effective_flm_source_model,
        validate_flm_weight_source_options,
    };
```

to:

```rust
    use super::{
        effective_flm_source, effective_quant_profile, ensure_hf_metadata_present,
        flm_source_is_authoritative_for_model, should_fetch_bake, should_fetch_exact_bake,
        validate_effective_flm_source_model, validate_flm_weight_source_options,
    };
```

Then add this test after `flm_model_dir_is_authoritative_for_hf_metadata_bootstrap`:

```rust
    #[test]
    fn qwen36_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config() {
        let cli = cli_with_model_dir(
            "/tmp/qwen36-27b-no-hf.flm",
            &["--int4", "--verify-flm-hashes"],
        );

        let downloaded = ensure_hf_metadata_present(&cli, &ModelVariant::Qwen3_6_27B)
            .expect("authoritative FLM source should bypass HF metadata bootstrap");

        assert!(!downloaded);
    }
```

- [ ] **Step 2: Run the focused policy test**

Run:

```bash
cargo test -q -p runner qwen36_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config
```

Expected: pass. If it attempts a bake lock, fetch, or config lookup, this test should fail before any network operation.

- [ ] **Step 3: Add FLM main-path smoke commands to testing docs**

In `docs/testing.md`, add this section after the "Running tests" command block and before "Adding tests for a new machine":

````markdown
### FLM Main-Path Smoke

The Qwen3.6 27B dense FLM path treats the `.flm` file as the complete model
source. A runnable no-HF artifact must validate under geo-quant's
`runnable-no-hf` profile before it is used for SuperSonic smoke tests:

```bash
/home/deano/.config/superpowers/worktrees/geo-quant/flm-export/.venv-rocm/bin/python \
  -m geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  --profile runnable-no-hf \
  --verify-payload-hashes
```

Then run the storage upload and runner smoke gates:

```bash
SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p model-store flm_qwen36_27b_direct_views_upload_to_hip -- --nocapture

SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

The runner smoke asserts that config, tokenizer, and weights come from FLM,
that BLAKE3 verification is enabled, and that no `[fetch]` or `[bake]` path is
entered.
````

- [ ] **Step 4: Run docs and policy checks**

Run:

```bash
cargo test -q -p runner qwen36_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config
git diff --check
```

Expected: test passes and `git diff --check` exits `0`.

- [ ] **Step 5: Commit Task 3**

```bash
git add crates/runner/src/bakes.rs docs/testing.md
git commit -m "runner: lock FLM no-HF bootstrap policy"
```

---

### Task 4: Env-Gated No-HF Main-Path Smoke

**Files:**
- Create: `crates/runner/tests/flm_main_path.rs`
- Test: `crates/runner/tests/flm_main_path.rs`

- [ ] **Step 1: Add the integration smoke test**

Create `crates/runner/tests/flm_main_path.rs` with:

```rust
#[cfg(target_os = "linux")]
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::process::Command;

#[cfg(target_os = "linux")]
fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[cfg(target_os = "linux")]
#[test]
fn qwen36_dense_flm_model_dir_runs_without_hf_snapshot() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_27B_NO_HF_FLM") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_27B_NO_HF_FLM is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        eprintln!(
            "skipping: SUPERSONIC_QWEN36_27B_NO_HF_FLM path does not exist: {}",
            path.display()
        );
        return;
    }

    let backend = std::env::var("SUPERSONIC_FLM_MAIN_PATH_BACKEND")
        .unwrap_or_else(|_| "hip".to_string());
    let device = std::env::var("SUPERSONIC_FLM_MAIN_PATH_DEVICE")
        .unwrap_or_else(|_| "0".to_string());

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args([
        "--backend",
        backend.as_str(),
        "--device",
        device.as_str(),
        "--model",
        "qwen3.6-27b",
        "--model-dir",
    ]);
    cmd.arg(&path);
    cmd.args([
        "--int4",
        "--verify-flm-hashes",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "1",
        "--context-size",
        "16",
        "--emit-generated-json",
    ]);

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("run supersonic FLM main-path smoke: {e}"));
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "FLM main-path smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("[config] loading FLM runtime descriptor"),
        "config was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[tokenizer] loading FLM tokenizer assets"),
        "tokenizer was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[weights] loading FLM container"),
        "weights were not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("BLAKE3 hash verification enabled"),
        "--verify-flm-hashes was not threaded to FLM weight loading:\n{combined}"
    );
    assert!(
        combined.contains("[tokens] "),
        "decode did not emit generated token ids:\n{combined}"
    );
    assert!(
        combined.contains("[generated_json] "),
        "decode did not emit generated text JSON:\n{combined}"
    );

    for forbidden in ["[fetch]", "[bake]", "config.json", "tokenizer.json", ".supersonic"] {
        assert!(
            !combined.contains(forbidden),
            "FLM main path unexpectedly referenced {forbidden:?}:\n{combined}"
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn qwen36_dense_flm_model_dir_runs_without_hf_snapshot() {
    eprintln!("skipping: FLM main-path smoke is Linux/HIP-only");
}
```

- [ ] **Step 2: Run the smoke without the artifact env**

Run:

```bash
cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected output includes:

```text
skipping: SUPERSONIC_QWEN36_27B_NO_HF_FLM is unset
```

Expected result: test binary exits `0`.

- [ ] **Step 3: Run the full no-HF artifact smoke**

Run:

```bash
SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: the test exits `0` and the combined output contains all of:

```text
[config] loading FLM runtime descriptor
[tokenizer] loading FLM tokenizer assets
[weights] loading FLM container
BLAKE3 hash verification enabled
[tokens]
[generated_json]
```

Expected: the combined output does not contain any of:

```text
[fetch]
[bake]
config.json
tokenizer.json
.supersonic
```

- [ ] **Step 4: Commit Task 4**

```bash
git add crates/runner/tests/flm_main_path.rs
git commit -m "runner: add no-HF FLM main-path smoke"
```

---

### Task 5: Full Verification Gate And Push

**Files:**
- Verify: Rust workspace tests and local FLM artifact

- [ ] **Step 1: Re-run geo-quant no-HF artifact validation**

Run:

```bash
/home/deano/.config/superpowers/worktrees/geo-quant/flm-export/.venv-rocm/bin/python \
  -m geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  --profile runnable-no-hf \
  --verify-payload-hashes
```

Expected output:

```text
[flm-validate] OK path=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm profile=runnable-no-hf tensors=1998 warnings=0
```

- [ ] **Step 2: Run runner unit and integration coverage**

Run:

```bash
cargo test -q -p runner flm
cargo test -q -p runner qwen35_startup
cargo test -q -p runner --test flm_tokenizer -- --nocapture
cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: all commands exit `0`. The `flm_tokenizer` and `flm_main_path` integration tests may self-skip when their env vars are unset.

- [ ] **Step 3: Run model-store baseline**

Run:

```bash
cargo test -q -p model-store
```

Expected: all model-store unit tests pass.

- [ ] **Step 4: Run HIP direct-view upload for the INT4 FLM artifact**

Run:

```bash
SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p model-store flm_qwen36_27b_direct_views_upload_to_hip -- --nocapture
```

Expected output includes:

```text
[flm-upload] OK — FLM direct views uploaded to HIP
```

- [ ] **Step 5: Run the full no-HF SuperSonic smoke**

Run:

```bash
SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm \
  cargo test -q -p runner --test flm_main_path -- --nocapture
```

Expected: one-token decode succeeds and the output satisfies the assertions from Task 4.

- [ ] **Step 6: Run formatting and diff checks**

Run:

```bash
cargo fmt --check
git diff --check
git status --short
```

Expected: `cargo fmt --check` and `git diff --check` exit `0`. `git status --short` should show no uncommitted changes after the task commits.

- [ ] **Step 7: Push the branch**

Run:

```bash
git push -u origin codex/flm-main-model-path
```

Expected: branch pushes successfully.

If a draft PR already exists for `codex/flm-main-model-path`, update it with the new commits. If no PR exists, create one as a draft against `main` with this body:

```markdown
## Summary

- opens Qwen3.6 dense FLM startup metadata with typed runtime/hash options
- reuses one FLM source for config and native tokenizer startup
- locks the no-HF metadata bootstrap bypass for `--model-dir model.flm`
- adds an env-gated no-HF Qwen3.6 27B INT4 FLM main-path smoke

## Verification

- `cargo test -q -p runner flm`
- `cargo test -q -p runner qwen35_startup`
- `cargo test -q -p runner --test flm_tokenizer -- --nocapture`
- `cargo test -q -p model-store`
- `/home/deano/.config/superpowers/worktrees/geo-quant/flm-export/.venv-rocm/bin/python -m geoquant.formats.flm_validate /mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm --profile runnable-no-hf --verify-payload-hashes`
- `SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm cargo test -q -p model-store flm_qwen36_27b_direct_views_upload_to_hip -- --nocapture`
- `SUPERSONIC_QWEN36_27B_NO_HF_FLM=/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm cargo test -q -p runner --test flm_main_path -- --nocapture`
```

---

## Self-Review Checklist

- Spec coverage: Tasks 1 and 2 cover verified FLM opening, config, and tokenizer. Task 3 covers no HF bootstrap and docs. Task 4 covers the real SuperSonic main-path smoke. Task 5 covers geo-quant validation, direct-view HIP upload, runner smoke, and baseline tests.
- Unsupported combinations remain explicit: existing policy tests in `policy.rs`, `bakes.rs`, and `qwen35_alt_runtime.rs` continue to cover non-Qwen3.6, `--no-bake`, Q4KM, INT8, DFlash, and SpecPrefill FLM rejections.
- Type consistency: `FlmModelSourceOptions` is the only new options type; startup and source helpers use the same field names, and `to_load_options()` maps directly into `model_store::FlmLoadOptions`.
- Artifact consistency: full decode uses `/mnt/data/runs/geo-quant/qwen36-27b-int4-stage3-direct.flm`, not the stale CRC-broken `qwen36-27b-int4-runnable.flm`.
