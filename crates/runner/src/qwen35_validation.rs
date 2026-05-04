use anyhow::Result;
use std::env;
use std::path::{Path, PathBuf};

use crate::decode_engine::DecodeEngine;
use crate::model_files::model_dir_has_raw_safetensors;
use crate::qwen35_oracle_prefill_trace::trace_qwen35_oracle_prefill_layer;
use crate::qwen35_prefill_validation_report::report_qwen35_prefill_validation;
use crate::registry::{Backend, ModelVariant};
use crate::{oracle, resolve_oracle_device, Cli};

pub(crate) type Qwen35NativePrefillTrace = (
    Option<Vec<u8>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
);

pub(crate) struct Qwen35OracleContext {
    pub(crate) model_id: String,
    pub(crate) device: String,
    pub(crate) fp8_oracle_dir: Option<PathBuf>,
}

pub(crate) fn qwen35_oracle_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("oracle/run_oracle.py")
}

pub(crate) fn resolve_qwen_oracle_model_id(
    explicit_model_id: Option<&str>,
    model_dir: &Path,
    model_variant: &ModelVariant,
) -> String {
    if let Some(model_id) = explicit_model_id {
        return model_id.to_string();
    }
    if model_dir_has_raw_safetensors(model_dir) {
        return model_dir.to_string_lossy().into_owned();
    }
    model_variant.hf_model_id().to_string()
}

pub(crate) fn resolve_qwen35_oracle_context(
    cli: &Cli,
    backend: Backend,
    ordinal: usize,
    model_variant: &ModelVariant,
) -> Qwen35OracleContext {
    Qwen35OracleContext {
        model_id: resolve_qwen_oracle_model_id(
            cli.model_id.as_deref(),
            &cli.model_dir,
            model_variant,
        ),
        device: resolve_oracle_device(&cli.oracle_device, backend, ordinal),
        fp8_oracle_dir: cli.fp8_runtime.then(|| cli.model_dir.clone()),
    }
}

pub(crate) fn run_qwen35_oracle_validation(
    cli: &Cli,
    engine: &DecodeEngine,
    text_config: &qwen35::config::TextConfig,
    prompt_ids: &[u32],
    oracle_model_id: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    prefill_logits: &[f32],
    native_prefill_trace: Option<&Qwen35NativePrefillTrace>,
    next_token: u32,
) -> Result<Option<oracle::OracleOutput>> {
    if !cli.validate {
        return Ok(None);
    }

    let oracle_script = qwen35_oracle_script_path();
    let emit_state = cli.trace_prefill_layers || cli.trace_oracle_prefill_layer.is_some();
    let output = oracle::run_oracle(
        &oracle_script,
        oracle_model_id,
        prompt_ids,
        cli.max_new_tokens,
        &cli.oracle_dtype,
        oracle_device,
        emit_state,
        false,
        fp8_oracle_dir,
        cli.trace_oracle_prefill_layer
            .filter(|layer| text_config.is_full_attention(*layer)),
    )?;

    report_qwen35_prefill_validation(
        engine,
        text_config,
        prefill_logits,
        native_prefill_trace,
        next_token,
        &output,
        cli.trace_prefill_layers,
    )?;

    Ok(Some(output))
}

pub(crate) fn trace_qwen35_oracle_prefill_layer_if_requested(
    cli: &Cli,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    oracle_context: &Qwen35OracleContext,
    oracle_output: Option<&oracle::OracleOutput>,
) -> Result<()> {
    let (Some(trace_layer), Some(output)) = (cli.trace_oracle_prefill_layer, oracle_output) else {
        return Ok(());
    };

    let oracle_script = qwen35_oracle_script_path();
    trace_qwen35_oracle_prefill_layer(
        engine,
        trace_layer,
        prompt_ids,
        &oracle_script,
        &oracle_context.model_id,
        &cli.oracle_dtype,
        &oracle_context.device,
        oracle_context.fp8_oracle_dir.as_deref(),
        output,
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::resolve_qwen_oracle_model_id;
    use crate::registry::ModelVariant;

    #[test]
    fn qwen_oracle_uses_hf_id_without_local_safetensors() {
        let model_dir = unique_temp_dir("qwen-oracle-no-raw");
        fs::create_dir_all(&model_dir).unwrap();

        let resolved = resolve_qwen_oracle_model_id(None, &model_dir, &ModelVariant::Qwen3_5_0_8B);

        assert_eq!(resolved, "Qwen/Qwen3.5-0.8B");
        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn qwen_oracle_uses_local_dir_when_safetensors_present() {
        let model_dir = unique_temp_dir("qwen-oracle-raw");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.safetensors.index.json"), "{}").unwrap();

        let resolved = resolve_qwen_oracle_model_id(None, &model_dir, &ModelVariant::Qwen3_5_0_8B);

        assert_eq!(resolved, model_dir.to_string_lossy());
        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn qwen_oracle_explicit_model_id_wins() {
        let model_dir = unique_temp_dir("qwen-oracle-explicit");
        fs::create_dir_all(&model_dir).unwrap();

        let resolved = resolve_qwen_oracle_model_id(
            Some("local-or-remote/override"),
            &model_dir,
            &ModelVariant::Qwen3_5_0_8B,
        );

        assert_eq!(resolved, "local-or-remote/override");
        let _ = fs::remove_dir_all(model_dir);
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }
}
