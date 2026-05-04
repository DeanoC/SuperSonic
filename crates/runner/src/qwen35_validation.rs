use anyhow::Result;
use base64::Engine as _;
use std::env;
use std::path::{Path, PathBuf};

use crate::decode_engine::DecodeEngine;
use crate::model_files::model_dir_has_raw_safetensors;
use crate::qwen35_oracle_prefill_trace::trace_qwen35_oracle_prefill_layer;
use crate::registry::{Backend, ModelVariant};
use crate::tensor_bytes::{bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le};
use crate::{oracle, resolve_oracle_device, validate, Cli};

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

fn report_qwen35_prefill_validation(
    engine: &DecodeEngine,
    text_config: &qwen35::config::TextConfig,
    prefill_logits: &[f32],
    native_prefill_trace: Option<&Qwen35NativePrefillTrace>,
    next_token: u32,
    output: &oracle::OracleOutput,
    trace_prefill_layers: bool,
) -> Result<()> {
    let prefill_delta = validate::max_abs_delta(prefill_logits, &output.prefill_logits);
    eprintln!("[validate] prefill logit delta={prefill_delta:.4}");

    if let (Some((native_final_norm_trace, ..)), Some(oracle_final_norm_b64)) =
        (native_prefill_trace, output.prefill_hidden.as_ref())
    {
        if let Some(native_final_norm_trace) = native_final_norm_trace.as_ref() {
            let b64 = base64::engine::general_purpose::STANDARD;
            let oracle_final_norm_bytes = b64
                .decode(oracle_final_norm_b64)
                .map_err(|e| anyhow::anyhow!("decode oracle prefill_hidden: {e}"))?;
            let native_final_norm = decode_bf16_le(native_final_norm_trace);
            let oracle_final_norm = decode_bf16_le(&oracle_final_norm_bytes);
            let final_norm_delta = validate::max_abs_delta(&native_final_norm, &oracle_final_norm);
            eprintln!("[trace-prefill] final_norm_delta={final_norm_delta:.4}");
        }
    }

    let oracle_first = output.generated_token_ids[0];
    if oracle_first != next_token {
        eprintln!(
            "[validate] WARNING: prefill token mismatch! native={next_token} oracle={oracle_first}"
        );
    }

    if trace_prefill_layers {
        report_qwen35_prefill_layer_trace(engine, text_config, native_prefill_trace, output)?;
    }
    Ok(())
}

fn report_qwen35_prefill_layer_trace(
    engine: &DecodeEngine,
    text_config: &qwen35::config::TextConfig,
    native_prefill_trace: Option<&Qwen35NativePrefillTrace>,
    output: &oracle::OracleOutput,
) -> Result<()> {
    let (
        Some((
            _,
            native_attn_trace,
            native_post_norm_trace,
            native_mlp_out_trace,
            native_layer_trace,
        )),
        Some(oracle_attn_trace),
        Some(oracle_post_norm_trace),
        Some(oracle_mlp_out_trace),
        Some(oracle_layer_trace),
    ) = (
        native_prefill_trace,
        output.layer_attn_residual_states.as_ref(),
        output.layer_post_attn_norm_states.as_ref(),
        output.layer_mlp_outputs.as_ref(),
        output.layer_hidden_states.as_ref(),
    )
    else {
        eprintln!("[trace-prefill] missing native or oracle layer trace data");
        return Ok(());
    };

    let (
        Some(native_attn_trace),
        Some(native_post_norm_trace),
        Some(native_mlp_out_trace),
        Some(native_layer_trace),
    ) = (
        native_attn_trace.as_ref(),
        native_post_norm_trace.as_ref(),
        native_mlp_out_trace.as_ref(),
        native_layer_trace.as_ref(),
    )
    else {
        eprintln!("[trace-prefill] missing native attention, post-norm, mlp-out, or layer trace");
        return Ok(());
    };

    let b64 = base64::engine::general_purpose::STANDARD;
    let oracle_kv = output.kv_caches.as_ref();
    let oracle_conv = output.conv_states.as_ref();
    let oracle_recurrent = output.recurrent_states.as_ref();
    let mut first_bad = None;
    for layer in 0..native_layer_trace.len().min(oracle_layer_trace.len()) {
        let oracle_attn_bytes = b64.decode(&oracle_attn_trace[layer]).map_err(|e| {
            anyhow::anyhow!("decode oracle layer_attn_residual_states[{layer}]: {e}")
        })?;
        let oracle_post_norm_bytes = b64.decode(&oracle_post_norm_trace[layer]).map_err(|e| {
            anyhow::anyhow!("decode oracle layer_post_attn_norm_states[{layer}]: {e}")
        })?;
        let oracle_mlp_out_bytes = b64
            .decode(&oracle_mlp_out_trace[layer])
            .map_err(|e| anyhow::anyhow!("decode oracle layer_mlp_outputs[{layer}]: {e}"))?;
        let oracle_layer_bytes = b64
            .decode(&oracle_layer_trace[layer])
            .map_err(|e| anyhow::anyhow!("decode oracle layer_hidden_states[{layer}]: {e}"))?;
        let native_attn_f32 = decode_bf16_le(&native_attn_trace[layer]);
        let native_post_norm_f32 = decode_bf16_le(&native_post_norm_trace[layer]);
        let native_mlp_out_f32 = decode_bf16_le(&native_mlp_out_trace[layer]);
        let native_layer_f32 = decode_bf16_le(&native_layer_trace[layer]);
        let oracle_attn_f32 = decode_bf16_le(&oracle_attn_bytes);
        let oracle_post_norm_f32 = decode_bf16_le(&oracle_post_norm_bytes);
        let oracle_mlp_out_f32 = decode_bf16_le(&oracle_mlp_out_bytes);
        let oracle_layer_f32 = decode_bf16_le(&oracle_layer_bytes);
        let attn_delta = validate::max_abs_delta(&native_attn_f32, &oracle_attn_f32);
        let post_norm_delta = validate::max_abs_delta(&native_post_norm_f32, &oracle_post_norm_f32);
        let mlp_out_delta = validate::max_abs_delta(&native_mlp_out_f32, &oracle_mlp_out_f32);
        let layer_delta = validate::max_abs_delta(&native_layer_f32, &oracle_layer_f32);
        let state_delta = qwen35_prefill_state_delta(
            engine,
            text_config,
            layer,
            oracle_kv,
            oracle_conv,
            oracle_recurrent,
            &b64,
        )?;
        if first_bad.is_none() && layer_delta > 0.5 {
            first_bad = Some((
                layer,
                attn_delta,
                post_norm_delta,
                mlp_out_delta,
                layer_delta,
            ));
        }
        eprintln!(
            "[trace-prefill] layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}{state_delta}"
        );
    }
    if let Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta)) = first_bad {
        eprintln!(
            "[trace-prefill] first_bad_layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
        );
    } else {
        eprintln!("[trace-prefill] no layer exceeded delta threshold");
    }
    Ok(())
}

fn qwen35_prefill_state_delta(
    engine: &DecodeEngine,
    text_config: &qwen35::config::TextConfig,
    layer: usize,
    oracle_kv: Option<&Vec<oracle::KvCacheDump>>,
    oracle_conv: Option<&Vec<oracle::StateDump>>,
    oracle_recurrent: Option<&Vec<oracle::StateDump>>,
    b64: &base64::engine::general_purpose::GeneralPurpose,
) -> Result<String> {
    if text_config.is_full_attention(layer) {
        let native = engine.full_attention_prefix_cache_bf16_host(layer, 0);
        match (
            native,
            oracle_kv.and_then(|caches| caches.iter().find(|kv| kv.layer == layer)),
        ) {
            (Ok((native_k, native_v, _)), Some(oracle_kv)) => {
                let oracle_k = b64
                    .decode(&oracle_kv.k)
                    .map_err(|e| anyhow::anyhow!("decode oracle kv k[{layer}]: {e}"))?;
                let oracle_v = b64
                    .decode(&oracle_kv.v)
                    .map_err(|e| anyhow::anyhow!("decode oracle kv v[{layer}]: {e}"))?;
                Ok(format!(
                    " kv_k_delta={:.4} kv_v_delta={:.4}",
                    validate::max_abs_delta(&decode_bf16_le(&native_k), &decode_bf16_le(&oracle_k)),
                    validate::max_abs_delta(&decode_bf16_le(&native_v), &decode_bf16_le(&oracle_v)),
                ))
            }
            _ => Ok(String::new()),
        }
    } else {
        let native_layer = engine.state_for_batch(0).layers.get(layer);
        match (
            native_layer,
            oracle_conv.and_then(|states| states.iter().find(|state| state.layer == layer)),
            oracle_recurrent.and_then(|states| states.iter().find(|state| state.layer == layer)),
        ) {
            (Some(native_layer), Some(oracle_conv), Some(oracle_recurrent)) => {
                let native_conv = native_layer
                    .conv_state
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("native linear layer {layer} missing conv_state")
                    })?
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("native conv D2H layer {layer}: {e}"))?;
                let native_recurrent = native_layer
                    .recurrent_state
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("native linear layer {layer} missing recurrent_state")
                    })?
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("native recurrent D2H layer {layer}: {e}"))?;
                let oracle_conv = b64
                    .decode(&oracle_conv.data)
                    .map_err(|e| anyhow::anyhow!("decode oracle conv[{layer}]: {e}"))?;
                let oracle_recurrent = b64
                    .decode(&oracle_recurrent.data)
                    .map_err(|e| anyhow::anyhow!("decode oracle recurrent[{layer}]: {e}"))?;
                Ok(format!(
                    " conv_delta={:.4} recurrent_delta={:.4}",
                    validate::max_abs_delta(
                        &decode_bf16_le(&native_conv),
                        &decode_bf16_le(&oracle_conv)
                    ),
                    validate::max_abs_delta(
                        &decode_f32_le(&native_recurrent),
                        &decode_f32_le(&oracle_recurrent)
                    ),
                ))
            }
            _ => Ok(String::new()),
        }
    }
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
