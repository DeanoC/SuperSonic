# Qwen3.6 MoE FLM Main Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `qwen3.6-35b-a3b` load config, tokenizer, and decode weights from one FLM file and run a HIP one-token smoke without a Hugging Face snapshot or bake directory.

**Architecture:** Keep `FlmModelSource` as the single opened source. Add a small Qwen3.6-MoE FLM source helper that owns the opened FLM, validated MoE config, FLM tokenizer, and execution weight mode, then thread it through dry-run and decode so existing MoE loaders consume the source's `BakedStore`.

**Tech Stack:** Rust workspace, `qwen36_moe`, `runner`, `model-store::BakedStore`, native FLM tokenizer assets, HIP/ROCm env-gated integration smoke.

---

## File Structure

- Modify `crates/qwen36_moe/Cargo.toml`
  - Add the existing workspace `model-store` dependency for FLM config conversion.
- Modify `crates/qwen36_moe/src/config.rs`
  - Add `Config::try_from_flm_qwen36_moe`.
  - Add tests for real 35B-A3B geometry and invalid descriptors.
- Modify `crates/runner/src/flm_model_source.rs`
  - Add a `qwen_moe_config` accessor that requires `ARCH_QWEN3_6_MOE`.
- Modify `crates/runner/src/bakes.rs`
  - Allow FLM sources for `ModelVariant::Qwen3_6_35B_A3B`.
  - Keep rejections for unrelated variants and unsupported quant flags.
- Create `crates/runner/src/qwen36_moe/flm_source.rs`
  - Open and validate a MoE FLM source once.
  - Resolve the FLM MoE runtime profile to `Qwen36WeightMode::Int4`.
- Modify `crates/runner/src/qwen36_moe/mod.rs`
  - Export the new `flm_source` module inside the CLI module tree.
- Modify `crates/runner/src/qwen36_moe/dry_run.rs`
  - Add a config-provided dry-run entry point and FLM store accounting.
- Modify `crates/runner/src/qwen36_moe/prompt.rs`
  - Add an already-loaded tokenizer prompt path.
- Modify `crates/runner/src/qwen36_moe/engine.rs`
  - Open the FLM source before bake handling.
  - Skip bake download and bake selection when the source is FLM.
  - Reuse the source store for layer loading.
- Create `crates/runner/tests/flm_moe_main_path.rs`
  - Add the env-gated real-artifact smoke test.
- Modify `docs/testing.md`
  - Document the MoE FLM smoke command and assertions.

## Task 1: Accept MoE FLM As An Authoritative Source

**Files:**
- Modify: `crates/runner/src/bakes.rs`

- [ ] **Step 1: Add failing policy tests**

Add these tests in the existing `#[cfg(test)] mod tests` in `crates/runner/src/bakes.rs`:

```rust
#[test]
fn flm_model_dir_is_authoritative_for_qwen36_moe_hf_metadata_bootstrap() {
    assert!(flm_source_is_authoritative_for_model(
        &cli_with_model_dir("/tmp/qwen36-35b-a3b.flm", &["--int4"]),
        &ModelVariant::Qwen3_6_35B_A3B
    ));
}

#[test]
fn qwen36_moe_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config() {
    let cli = cli_with_model_dir(
        "/tmp/qwen36-35b-a3b-no-hf.flm",
        &["--int4", "--verify-flm-hashes"],
    );

    let downloaded = ensure_hf_metadata_present(&cli, &ModelVariant::Qwen3_6_35B_A3B)
        .expect("authoritative MoE FLM source should bypass HF metadata bootstrap");

    assert!(!downloaded);
}

#[test]
fn effective_flm_source_accepts_qwen36_moe_model_variant() {
    validate_effective_flm_source_model(
        &cli_with_model_dir("/tmp/qwen36-35b-a3b.flm", &["--int4"]),
        &ModelVariant::Qwen3_6_35B_A3B,
    )
    .expect("qwen3.6-35b-a3b FLM should be accepted");
}
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
cargo test -q -p runner qwen36_moe_flm
```

Expected: FAIL because FLM policy currently accepts only `qwen3.6-27b`.

- [ ] **Step 3: Expand the accepted FLM model set**

Replace `flm_source_is_authoritative_for_model` with:

```rust
pub(crate) fn flm_source_is_authoritative_for_model(
    cli: &Cli,
    model_variant: &ModelVariant,
) -> bool {
    matches!(
        model_variant,
        ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_6_35B_A3B
    ) && effective_flm_source(cli).is_some()
}
```

Replace the accepted-model check in `validate_effective_flm_source_model` with:

```rust
    if matches!(
        model_variant,
        ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_6_35B_A3B
    ) {
        return Ok(());
    }
```

Update the error string in that function to say:

```rust
"FLM source from {source_flag} {} currently supports only --model qwen3.6-27b or qwen3.6-35b-a3b; got --model {}"
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
cargo test -q -p runner flm_source
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/bakes.rs
git commit -m "runner: accept Qwen36 MoE FLM source policy"
```

## Task 2: Convert FLM MoE Runtime Config

**Files:**
- Modify: `crates/qwen36_moe/Cargo.toml`
- Modify: `crates/qwen36_moe/src/config.rs`
- Modify: `crates/runner/src/flm_model_source.rs`

- [ ] **Step 1: Add the model-store dependency**

Add this dependency to `crates/qwen36_moe/Cargo.toml`:

```toml
model-store = { path = "../model-store" }
```

- [ ] **Step 2: Add failing conversion tests**

In `crates/qwen36_moe/src/config.rs`, add a test helper beside the existing
`real_qwen36_35b_a3b_config_json` tests:

```rust
fn flm_qwen36_moe_descriptor() -> model_store::FlmQwen36MoeConfig {
    model_store::FlmQwen36MoeConfig {
        vocab_size: 248320,
        hidden_size: 2048,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
        num_hidden_layers: 40,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 256,
        max_position_embeddings: 262144,
        linear_conv_kernel_dim: 4,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_num_key_heads: 16,
        linear_num_value_heads: 32,
        num_experts: 256,
        num_experts_per_tok: 8,
        mtp_num_hidden_layers: 0,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        activation_id: 1,
        tie_word_embeddings: false,
        attn_output_gate: true,
        mtp_use_dedicated_embeddings: false,
        mrope_interleaved: true,
        eos_token_ids: vec![248044],
        full_attention_layers: (3..40).step_by(4).collect(),
        mrope_section: [11, 11, 10],
    }
}

#[test]
fn builds_text_config_from_flm_qwen36_moe_descriptor() {
    let config = Config::try_from_flm_qwen36_moe(&flm_qwen36_moe_descriptor())
        .expect("valid FLM MoE descriptor")
        .normalized();

    let text = &config.text_config;
    assert_eq!(text.vocab_size, 248320);
    assert_eq!(text.hidden_size, 2048);
    assert_eq!(text.num_hidden_layers, 40);
    assert_eq!(text.num_full_attention_layers(), 10);
    assert_eq!(text.num_linear_attention_layers(), 30);
    assert_eq!(text.num_experts, 256);
    assert_eq!(text.num_experts_per_tok, 8);
    assert_eq!(text.linear_value_dim(), 4096);
    assert_eq!(text.rope_theta(), 10_000_000.0);
    assert_eq!(text.eos_token_ids(), vec![248044]);
    assert!(text.attn_output_gate);
    assert_eq!(
        text.rope_parameters.as_ref().unwrap().mrope_section,
        vec![11, 11, 10]
    );
}

#[test]
fn rejects_flm_qwen36_moe_descriptor_with_bad_expert_topk() {
    let mut flm = flm_qwen36_moe_descriptor();
    flm.num_experts_per_tok = 257;

    let err = Config::try_from_flm_qwen36_moe(&flm).unwrap_err();

    assert!(err.contains("num_experts_per_tok"), "{err}");
}

#[test]
fn rejects_flm_qwen36_moe_descriptor_with_bad_full_attention_layer() {
    let mut flm = flm_qwen36_moe_descriptor();
    flm.full_attention_layers.push(40);

    let err = Config::try_from_flm_qwen36_moe(&flm).unwrap_err();

    assert!(err.contains("full_attention_layers"), "{err}");
}

#[test]
fn rejects_flm_qwen36_moe_descriptor_with_unknown_activation() {
    let mut flm = flm_qwen36_moe_descriptor();
    flm.activation_id = 99;

    let err = Config::try_from_flm_qwen36_moe(&flm).unwrap_err();

    assert!(err.contains("activation_id"), "{err}");
}
```

- [ ] **Step 3: Run the focused failing tests**

Run:

```bash
cargo test -q -p qwen36_moe flm_qwen36_moe
```

Expected: FAIL because `Config::try_from_flm_qwen36_moe` does not exist.

- [ ] **Step 4: Implement the conversion**

Add helper functions near `load_config` in `crates/qwen36_moe/src/config.rs`:

```rust
fn validate_positive(field: &str, value: usize) -> Result<(), String> {
    if value == 0 {
        Err(format!("{field} must be non-zero"))
    } else {
        Ok(())
    }
}

fn validate_positive_finite(field: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        Err(format!("{field} must be finite"))
    } else if value <= 0.0 {
        Err(format!("{field} must be positive"))
    } else {
        Ok(())
    }
}

fn activation_from_flm_id(activation_id: u8) -> Result<Activation, String> {
    match activation_id {
        0 => Ok(Activation::Gelu),
        1 => Ok(Activation::Silu),
        2 => Ok(Activation::Swiglu),
        _ => Err(format!("unknown activation_id {activation_id}")),
    }
}
```

Add this method to `impl Config`:

```rust
pub fn try_from_flm_qwen36_moe(
    flm: &model_store::FlmQwen36MoeConfig,
) -> Result<Self, String> {
    validate_positive("vocab_size", flm.vocab_size)?;
    validate_positive("hidden_size", flm.hidden_size)?;
    validate_positive("moe_intermediate_size", flm.moe_intermediate_size)?;
    validate_positive(
        "shared_expert_intermediate_size",
        flm.shared_expert_intermediate_size,
    )?;
    validate_positive("num_hidden_layers", flm.num_hidden_layers)?;
    validate_positive("num_attention_heads", flm.num_attention_heads)?;
    validate_positive("num_key_value_heads", flm.num_key_value_heads)?;
    validate_positive("head_dim", flm.head_dim)?;
    validate_positive("max_position_embeddings", flm.max_position_embeddings)?;
    validate_positive("linear_conv_kernel_dim", flm.linear_conv_kernel_dim)?;
    validate_positive("linear_key_head_dim", flm.linear_key_head_dim)?;
    validate_positive("linear_value_head_dim", flm.linear_value_head_dim)?;
    validate_positive("linear_num_key_heads", flm.linear_num_key_heads)?;
    validate_positive("linear_num_value_heads", flm.linear_num_value_heads)?;
    validate_positive("num_experts", flm.num_experts)?;
    validate_positive("num_experts_per_tok", flm.num_experts_per_tok)?;
    validate_positive_finite("rms_norm_eps", flm.rms_norm_eps)?;
    validate_positive_finite("rope_theta", flm.rope_theta)?;
    validate_positive_finite("partial_rotary_factor", flm.partial_rotary_factor)?;
    if flm.partial_rotary_factor > 1.0 {
        return Err(format!(
            "partial_rotary_factor must be <= 1.0, got {}",
            flm.partial_rotary_factor
        ));
    }
    if flm.num_experts_per_tok > flm.num_experts {
        return Err(format!(
            "num_experts_per_tok {} must be <= num_experts {}",
            flm.num_experts_per_tok, flm.num_experts
        ));
    }

    let hidden_act = activation_from_flm_id(flm.activation_id)?;
    let mut layer_types = vec!["linear_attention".to_string(); flm.num_hidden_layers];
    for &idx in &flm.full_attention_layers {
        let slot = layer_types.get_mut(idx).ok_or_else(|| {
            format!(
                "full_attention_layers contains {idx}, but num_hidden_layers is {}",
                flm.num_hidden_layers
            )
        })?;
        *slot = "full_attention".to_string();
    }

    let config = Self {
        architectures: vec!["Qwen3_5MoeForConditionalGeneration".to_string()],
        model_type: Some("qwen3_5_moe".to_string()),
        text_config: TextConfig {
            vocab_size: flm.vocab_size,
            hidden_size: flm.hidden_size,
            num_hidden_layers: flm.num_hidden_layers,
            num_attention_heads: flm.num_attention_heads,
            num_key_value_heads: flm.num_key_value_heads,
            max_position_embeddings: flm.max_position_embeddings,
            rms_norm_eps: flm.rms_norm_eps,
            hidden_act,
            tie_word_embeddings: flm.tie_word_embeddings,
            eos_token_id: Some(serde_json::Value::Array(
                flm.eos_token_ids.iter().map(|&id| serde_json::json!(id)).collect(),
            )),
            bos_token_id: None,
            head_dim: flm.head_dim,
            full_attention_interval: 4,
            attn_output_gate: flm.attn_output_gate,
            linear_conv_kernel_dim: flm.linear_conv_kernel_dim,
            linear_key_head_dim: flm.linear_key_head_dim,
            linear_value_head_dim: flm.linear_value_head_dim,
            linear_num_key_heads: flm.linear_num_key_heads,
            linear_num_value_heads: flm.linear_num_value_heads,
            layer_types,
            rope_parameters: Some(RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: flm.rope_theta,
                partial_rotary_factor: flm.partial_rotary_factor,
                mrope_interleaved: flm.mrope_interleaved,
                mrope_section: flm.mrope_section.iter().map(|&v| v as usize).collect(),
            }),
            num_experts: flm.num_experts,
            num_experts_per_tok: flm.num_experts_per_tok,
            moe_intermediate_size: flm.moe_intermediate_size,
            shared_expert_intermediate_size: flm.shared_expert_intermediate_size,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        },
    }
    .normalized();

    validate(&config)?;
    Ok(config)
}
```

- [ ] **Step 5: Add the runner accessor**

Add this method to `impl FlmModelSource` in `crates/runner/src/flm_model_source.rs`:

```rust
pub fn qwen_moe_config(&self) -> anyhow::Result<qwen36_moe::config::Config> {
    let runtime = self.runtime()?;
    let cfg = runtime.qwen36_moe_config().ok_or_else(|| {
        anyhow::anyhow!("FLM {} is not Qwen3.6 MoE v1", self.path.display())
    })?;
    qwen36_moe::config::Config::try_from_flm_qwen36_moe(cfg).map_err(|e| {
        anyhow::anyhow!(
            "invalid FLM Qwen3.6 MoE config in {}: {e}",
            self.path.display()
        )
    })
}
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cargo test -q -p qwen36_moe flm_qwen36_moe
cargo test -q -p runner flm_model_source
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/qwen36_moe/Cargo.toml crates/qwen36_moe/src/config.rs crates/runner/src/flm_model_source.rs
git commit -m "qwen36_moe: convert FLM runtime config"
```

## Task 3: Add The MoE FLM Source Helper

**Files:**
- Create: `crates/runner/src/qwen36_moe/flm_source.rs`
- Modify: `crates/runner/src/qwen36_moe/mod.rs`

- [ ] **Step 1: Create failing source-helper tests**

Create `crates/runner/src/qwen36_moe/flm_source.rs` with the test module below
and no implementation above it:

```rust
#[cfg(test)]
mod tests {
    use model_store::manifest::QuantProfile;

    use super::*;

    #[test]
    fn maps_moe_mixed_lowbit_flm_to_int4_weight_mode() {
        let mode = qwen36_moe_flm_weight_mode(QuantProfile::Int4Gptq)
            .expect("INT4 profile is supported");

        assert_eq!(mode, Qwen36WeightMode::Int4);
    }

    #[test]
    fn rejects_q4km_flm_weight_mode_for_this_stage() {
        let err = qwen36_moe_flm_weight_mode(QuantProfile::Q4Km)
            .unwrap_err()
            .to_string();

        assert!(err.contains("Qwen3.6 MoE FLM"), "{err}");
        assert!(err.contains("INT4"), "{err}");
    }
}
```

- [ ] **Step 2: Export the module**

Add this line to `crates/runner/src/qwen36_moe/mod.rs`:

```rust
pub(crate) mod flm_source;
```

- [ ] **Step 3: Run the focused failing tests**

Run:

```bash
cargo test -q -p runner qwen36_moe_flm_weight_mode
```

Expected: FAIL because `qwen36_moe_flm_weight_mode` is not implemented.

- [ ] **Step 4: Implement the helper**

Add this implementation above the test module:

```rust
use anyhow::{anyhow, Result};
use model_store::manifest::QuantProfile;

use crate::bakes::{
    effective_flm_source, effective_quant_profile, flm_source_open_options,
    validate_effective_flm_source_model,
};
use crate::flm_model_source::FlmModelSource;
use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::registry::ModelVariant;
use crate::Cli;

pub(crate) struct Qwen36MoeFlmSource {
    pub(crate) source: FlmModelSource,
    pub(crate) config: qwen36_moe::config::Config,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) weight_mode: Qwen36WeightMode,
}

pub(crate) fn qwen36_moe_flm_weight_mode(profile: QuantProfile) -> Result<Qwen36WeightMode> {
    match profile {
        QuantProfile::Int4Gptq | QuantProfile::Int4Hqq | QuantProfile::Int4ModelOpt => {
            Ok(Qwen36WeightMode::Int4)
        }
        other => Err(anyhow!(
            "Qwen3.6 MoE FLM main path currently supports only INT4-compatible profiles; got {other}"
        )),
    }
}

pub(crate) fn open_qwen36_moe_flm_source(
    cli: &Cli,
) -> Result<Option<Qwen36MoeFlmSource>> {
    let Some(path) = effective_flm_source(cli) else {
        return Ok(None);
    };
    validate_effective_flm_source_model(cli, &ModelVariant::Qwen3_6_35B_A3B)?;
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
    let source = FlmModelSource::open_with_options(path, options)
        .map_err(|e| anyhow!("opening Qwen3.6 MoE FLM source {}: {e}", path.display()))?;
    eprintln!("[qwen36-moe] loading config from FLM runtime descriptor");
    let config = source.qwen_moe_config()?;
    eprintln!("[qwen36-moe] loading tokenizer from FLM assets");
    let tokenizer = source.qwen_tokenizer()?;
    let weight_mode = qwen36_moe_flm_weight_mode(effective_quant_profile(cli)?)?;
    Ok(Some(Qwen36MoeFlmSource {
        source,
        config,
        tokenizer,
        weight_mode,
    }))
}
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cargo test -q -p runner qwen36_moe_flm_weight_mode
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/qwen36_moe/flm_source.rs crates/runner/src/qwen36_moe/mod.rs
git commit -m "runner: add Qwen36 MoE FLM source helper"
```

## Task 4: Make Dry Run Consume FLM Config

**Files:**
- Modify: `crates/runner/src/qwen36_moe/dry_run.rs`

- [ ] **Step 1: Split config loading from dry-run accounting**

Change `run_qwen36_moe_dry_run` into a wrapper that loads JSON config and then
calls a new config-provided function:

```rust
pub fn run_qwen36_moe_dry_run(
    model_dir: &Path,
    entry: &RegistryEntry,
    total_vram: u64,
    context_size: usize,
    context_size_source: ContextSizeSource,
    batch_size: usize,
    kv_fp8: bool,
    no_bake: bool,
    ordinal: usize,
) -> Result<DryRunReport> {
    let config = qwen36_moe::config::load_config(model_dir)
        .map_err(|e| anyhow!("parse config.json: {e}"))?;
    run_qwen36_moe_dry_run_with_config(
        model_dir,
        None,
        config,
        entry,
        total_vram,
        context_size,
        context_size_source,
        batch_size,
        kv_fp8,
        no_bake,
        ordinal,
    )
}
```

- [ ] **Step 2: Add the config-provided dry-run function**

Move the body of the old function into:

```rust
pub fn run_qwen36_moe_dry_run_with_config(
    model_dir: &Path,
    flm_source_label: Option<&Path>,
    config: Config,
    entry: &RegistryEntry,
    total_vram: u64,
    context_size: usize,
    context_size_source: ContextSizeSource,
    batch_size: usize,
    kv_fp8: bool,
    no_bake: bool,
    ordinal: usize,
) -> Result<DryRunReport> {
    let kernel_params = match entry.params {
        FamilyParams::Qwen36Moe(p) => p,
        _ => return Err(anyhow!("registry entry is not Qwen36Moe family")),
    };

    sanity_check_kernel_params(&config.text_config, &kernel_params)?;

    let weight_prefix = kernel_params.weight_prefix;
    let specs = expected_tensor_specs(&config.text_config, weight_prefix);
    let checkpoint = CheckpointAccount::from_config(&config.text_config);
    let int4_projected_bytes = checkpoint.project_int4_total_bytes(&config.text_config, 128);
}
```

The implementation should reuse the existing body exactly after the old JSON
load line. The function must keep all existing fields in `DryRunReport`.

- [ ] **Step 3: Skip bake-directory accounting for FLM**

Change the existing bake-inspection line:

```rust
let bake = inspect_bake(model_dir, &config.text_config, weight_prefix, ordinal);
```

to:

```rust
let bake = if flm_source_label.is_some() {
    None
} else {
    inspect_bake(model_dir, &config.text_config, weight_prefix, ordinal)
};
```

Then set a clear dry-run note for FLM sources by changing the loader-warning
initialization:

```rust
let mut loader_warning = flm_source_label.map(|path| {
    format!(
        "FLM source {} supplies runtime weights directly; bake-directory accounting is skipped",
        path.display()
    )
});
```

Keep the existing safetensors accounting branch disabled for FLM sources by
wrapping the current branch that starts with:

```rust
if !no_bake_only_safetensors(model_dir, no_bake) {
```

Insert `if flm_source_label.is_none() {` immediately before that branch and add
the closing brace immediately after the branch's current final closing brace.
Do not duplicate the branch. The FLM path must not read safetensors or
`.supersonic` files during dry-run reporting.

- [ ] **Step 4: Run focused tests**

Run:

```bash
cargo test -q -p runner qwen36_moe::dry_run
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/qwen36_moe/dry_run.rs
git commit -m "runner: let Qwen36 MoE dry run use FLM config"
```

## Task 5: Let Prompt Setup Use The FLM Tokenizer

**Files:**
- Modify: `crates/runner/src/qwen36_moe/prompt.rs`

- [ ] **Step 1: Add an already-loaded tokenizer entry point**

Replace `prepare_prompt` with a wrapper plus a new helper:

```rust
pub(crate) fn prepare_prompt(
    model_dir: &Path,
    text_config: &TextConfig,
    prompt: &str,
) -> Result<Qwen36PromptSetup> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path).ok();
    prepare_prompt_with_tokenizer(tokenizer, text_config, prompt)
}

pub(crate) fn prepare_prompt_with_tokenizer(
    tokenizer: Option<tokenizers::Tokenizer>,
    text_config: &TextConfig,
    prompt: &str,
) -> Result<Qwen36PromptSetup> {
    let bos_id = text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let eos_id = text_config
        .eos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let prompt_ids = match (&tokenizer, prompt.is_empty()) {
        (Some(tok), false) => {
            let enc = tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            if ids.is_empty() {
                vec![bos_id]
            } else {
                ids
            }
        }
        _ => vec![bos_id],
    };

    Ok(Qwen36PromptSetup {
        tokenizer,
        prompt_ids,
        eos_id,
    })
}
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
cargo test -q -p runner qwen36_moe
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/src/qwen36_moe/prompt.rs
git commit -m "runner: support preloaded Qwen36 MoE tokenizer"
```

## Task 6: Thread The FLM Source Through MoE Decode

**Files:**
- Modify: `crates/runner/src/qwen36_moe/engine.rs`

- [ ] **Step 1: Import the FLM source helper**

Update imports in `crates/runner/src/qwen36_moe/engine.rs`:

```rust
use crate::qwen36_moe_cli::flm_source::{
    open_qwen36_moe_flm_source, Qwen36MoeFlmSource,
};
use crate::qwen36_moe_cli::prompt::{
    prepare_prompt, prepare_prompt_with_tokenizer, print_prompt_summary,
    validate_speculative_sampling,
};
```

- [ ] **Step 2: Open FLM before bake handling**

In `run_inner`, add:

```rust
let flm_source = open_qwen36_moe_flm_source(cli)?;
if flm_source.is_none() {
    ensure_qwen36_bake(cli, entry)?;
}
```

and remove the unconditional `ensure_qwen36_bake(cli, entry)?;`.

- [ ] **Step 3: Use FLM config for dry-run**

Replace the dry-run call with:

```rust
let report = if let Some(flm) = flm_source.as_ref() {
    run_qwen36_moe_dry_run_with_config(
        &cli.model_dir,
        Some(flm.source.path.as_path()),
        flm.config.clone(),
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
        cli.device,
    )?
} else {
    run_qwen36_moe_dry_run(
        &cli.model_dir,
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
        cli.device,
    )?
};
```

Add `run_qwen36_moe_dry_run_with_config` to the dry-run import list.

- [ ] **Step 4: Pass FLM source into decode**

Add this parameter to `decode_text`:

```rust
flm_source: Option<&Qwen36MoeFlmSource>,
```

Pass `flm_source.as_ref()` from `run_inner`.

- [ ] **Step 5: Use FLM tokenizer and store inside decode**

Replace prompt setup in `decode_text` with:

```rust
let prompt_setup = if let Some(flm) = flm_source {
    prepare_prompt_with_tokenizer(Some(flm.tokenizer.clone()), &report.config.text_config, prompt)?
} else {
    prepare_prompt(model_dir, &report.config.text_config, prompt)?
};
```

Add this helper enum inside `decode_text` before bake selection:

```rust
enum DecodeStore<'a> {
    Borrowed(&'a BakedStore),
    Owned(BakedStore),
}

impl<'a> DecodeStore<'a> {
    fn as_store(&self) -> &BakedStore {
        match self {
            DecodeStore::Borrowed(store) => store,
            DecodeStore::Owned(store) => store,
        }
    }
}
```

Replace bake selection/opening with:

```rust
let (decode_store, weight_mode, source_label) = if let Some(flm) = flm_source {
    eprintln!(
        "[qwen36-moe] loading weights from already-open FLM source at {} ({})",
        flm.source.path.display(),
        flm.weight_mode.display_name(),
    );
    (
        DecodeStore::Borrowed(&flm.source.store),
        flm.weight_mode,
        flm.source.path.display().to_string(),
    )
} else {
    let bake = select_decode_bake(model_dir, quant_profile, int4_runtime)?;
    if !bake.weight_mode.is_int4() {
        match backend {
            Backend::Cuda => anyhow::bail!(
                "Qwen3.6-35B-A3B CUDA v1 requires an INT4/q4km bake; selected {} from {}",
                bake.weight_mode.display_name(),
                bake.bake_dir.display(),
            ),
            Backend::Metal => anyhow::bail!(
                "Qwen3.6-35B-A3B Metal v1 requires an INT4-GPTQ bake; selected {} from {}",
                bake.weight_mode.display_name(),
                bake.bake_dir.display(),
            ),
            _ => {}
        }
    }
    println!(
        "  loading from bake: {} ({})",
        bake.bake_dir.display(),
        bake.weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake.bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake.bake_dir.display()))?;
    (
        DecodeStore::Owned(store),
        bake.weight_mode,
        bake.bake_dir.display().to_string(),
    )
};
let store = decode_store.as_store();
```

Then replace all `bake.weight_mode` references in `decode_text` with
`weight_mode`, and use `source_label` in progress output where the bake path was
previously printed.

- [ ] **Step 6: Run focused tests and check compile**

Run:

```bash
cargo test -q -p runner qwen36_moe
cargo check -q -p runner
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/runner/src/qwen36_moe/engine.rs
git commit -m "runner: load Qwen36 MoE decode from FLM source"
```

## Task 7: Add The Real MoE FLM Smoke

**Files:**
- Create: `crates/runner/tests/flm_moe_main_path.rs`
- Modify: `docs/testing.md`

- [ ] **Step 1: Add the env-gated integration test**

Create `crates/runner/tests/flm_moe_main_path.rs`:

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
fn occurrence_count(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

#[cfg(target_os = "linux")]
#[test]
fn qwen36_moe_flm_model_dir_runs_without_hf_snapshot() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        panic!(
            "SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM is set but the path does not exist: {}",
            path.display()
        );
    }

    let backend =
        std::env::var("SUPERSONIC_FLM_MOE_MAIN_PATH_BACKEND").unwrap_or_else(|_| "hip".to_string());
    let device =
        std::env::var("SUPERSONIC_FLM_MOE_MAIN_PATH_DEVICE").unwrap_or_else(|_| "0".to_string());

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args([
        "--backend",
        backend.as_str(),
        "--device",
        device.as_str(),
        "--model",
        "qwen3.6-35b-a3b",
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
        .unwrap_or_else(|e| panic!("run supersonic MoE FLM main-path smoke: {e}"));
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "MoE FLM main-path smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert_eq!(
        occurrence_count(&combined, "[flm] opening model source"),
        1,
        "MoE FLM main path should open the source exactly once:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading config from FLM runtime descriptor"),
        "MoE config was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading tokenizer from FLM assets"),
        "MoE tokenizer was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading weights from already-open FLM source"),
        "MoE weights were not loaded from the already-open FLM source:\n{combined}"
    );
    assert!(
        combined.contains("BLAKE3 hash verification enabled"),
        "--verify-flm-hashes was not threaded to the MoE FLM source open:\n{combined}"
    );
    assert!(
        combined.contains("[tokens] "),
        "decode did not emit generated token ids:\n{combined}"
    );
    assert!(
        combined.contains("[generated_json] "),
        "decode did not emit generated text JSON:\n{combined}"
    );

    for forbidden in [
        "[fetch]",
        "[bake]",
        "config.json",
        "tokenizer.json",
        "safetensors",
        ".supersonic",
    ] {
        assert!(
            !combined.contains(forbidden),
            "MoE FLM main path unexpectedly referenced {forbidden:?}:\n{combined}"
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn qwen36_moe_flm_model_dir_runs_without_hf_snapshot() {
    eprintln!("skipping: MoE FLM main-path smoke is Linux/HIP-only");
}
```

- [ ] **Step 2: Document the smoke command**

Add this section to `docs/testing.md` after the dense FLM smoke section:

````markdown
### FLM MoE Main-Path Smoke

Set `SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/path/to/qwen36-35b-a3b.flm`
to run the Qwen3.6 35B-A3B no-HF FLM smoke:

```bash
SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/path/to/qwen36-35b-a3b.flm \
  cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

The smoke runs one HIP decode token and asserts that config, tokenizer, and
weights are loaded from a single opened FLM source with BLAKE3 verification
enabled. It also asserts that the binary does not fetch, bake, or read HF
`config.json`, `tokenizer.json`, safetensors, or `.supersonic` bake files.
````

- [ ] **Step 3: Run tests without the env var**

Run:

```bash
cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

Expected: PASS with a skip message when `SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM`
is unset.

- [ ] **Step 4: Run the real-artifact smoke**

Run:

```bash
SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/path/to/qwen36-35b-a3b.flm \
  cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

Expected: PASS on the local ROCm host.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/tests/flm_moe_main_path.rs docs/testing.md
git commit -m "runner: assert Qwen36 MoE FLM main path"
```

## Final Verification

- [ ] **Step 1: Run unit and parser tests**

Run:

```bash
cargo test -q -p qwen36_moe flm_qwen36_moe
cargo test -q -p runner flm_source
cargo test -q -p runner qwen36_moe
cargo test -q -p model-store flm
```

Expected: all commands exit `0`.

- [ ] **Step 2: Run integration smoke without env vars**

Run:

```bash
cargo test -q -p runner --test flm_main_path -- --nocapture
cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

Expected: both commands exit `0`; each smoke prints its skip message when its
artifact env var is unset.

- [ ] **Step 3: Run the real MoE smoke**

Run:

```bash
SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/path/to/qwen36-35b-a3b.flm \
  cargo test -q -p runner --test flm_moe_main_path -- --nocapture
```

Expected: exits `0`, emits one generated token, opens the FLM once, and does not
touch HF JSON, safetensors, fetch, bake, or `.supersonic` paths.

- [ ] **Step 4: Commit final verification notes if docs changed**

Run:

```bash
git status -sb
```

Expected: clean working tree after the final implementation commit.

## Self-Review

- Spec coverage: tasks cover policy, config conversion, source opening, dry-run,
  prompt setup, decode store selection, real smoke, and docs.
- Placeholder scan: the plan contains no placeholder markers and every code-changing
  task names the file and exact target behavior.
- Type consistency: `Qwen36MoeFlmSource`, `qwen36_moe_flm_weight_mode`,
  `open_qwen36_moe_flm_source`, `prepare_prompt_with_tokenizer`, and
  `run_qwen36_moe_dry_run_with_config` are introduced before later tasks use
  them.
