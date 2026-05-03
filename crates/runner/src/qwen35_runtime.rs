use anyhow::Result;
use base64::Engine as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::bakes::load_qwen35_weights;
use crate::decode_engine::DecodeEngine;
use crate::prefill_engine::PrefillResult;
use crate::registry::{Backend, GpuArch, ModelVariant, Qwen35KernelParams, VramBudget};
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le, f32_to_bf16_bytes,
};
use crate::{model_dir_has_raw_safetensors, oracle, validate, Cli};

pub(crate) struct Qwen35Startup {
    pub(crate) text_config: qwen35::config::TextConfig,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
}

pub(crate) struct Qwen35EngineSetup {
    pub(crate) engine: DecodeEngine,
    pub(crate) use_4b_kernel: bool,
    pub(crate) cuda_08b_hero_enabled: bool,
    pub(crate) allow_host_lm_head_rescore: bool,
}

pub(crate) struct Qwen35Policy {
    pub(crate) trace_kv_cache_enabled: bool,
}

pub(crate) type Qwen35NativePrefillTrace = (
    Option<Vec<u8>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
);

pub(crate) struct Qwen35Prefill {
    pub(crate) logits: Vec<f32>,
    pub(crate) native_trace: Option<Qwen35NativePrefillTrace>,
    pub(crate) next_token: u32,
}

pub(crate) struct HostLmHeadRescorer {
    loader: qwen35::loader::WeightLoader,
    tensor_name: String,
}

impl HostLmHeadRescorer {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        if !model_dir_has_raw_safetensors(model_dir) {
            return Ok(None);
        }
        let loader = qwen35::loader::WeightLoader::from_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("open raw lm_head weights: {e}"))?;
        let tensor_name = if loader.contains("lm_head.weight") {
            "lm_head.weight".to_string()
        } else if loader.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight".to_string()
        } else {
            return Ok(None);
        };
        Ok(Some(Self {
            loader,
            tensor_name,
        }))
    }

    fn rescore(&self, normed: &[f32], candidate_ids: &[usize]) -> Result<u32> {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for &candidate in candidate_ids {
            let row = self
                .loader
                .load_bf16_row_f32(&self.tensor_name, candidate)
                .map_err(|e| anyhow::anyhow!("load lm_head row {candidate}: {e}"))?;
            anyhow::ensure!(
                row.len() == normed.len(),
                "lm_head row len {} != normed len {}",
                row.len(),
                normed.len()
            );
            let score = row
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            if score > best_val {
                best_val = score;
                best_idx = candidate;
            }
        }
        Ok(best_idx as u32)
    }
}

pub(crate) fn sample_qwen_logits_with_rescore(
    logits: &[f32],
    normed: Option<&[f32]>,
    rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let greedy = DecodeEngine::greedy_sample(logits);
    let Some(normed) = normed else {
        return Ok(greedy);
    };
    let Some(rescorer) = rescorer else {
        return Ok(greedy);
    };

    const RESCORE_MARGIN: f32 = 0.25;
    const RESCORE_TOP_K: usize = 4;

    let candidates = top_k_candidate_ids(logits, RESCORE_TOP_K);
    if candidates.len() < 2 {
        return Ok(greedy);
    }
    let top0 = logits[candidates[0]];
    let top1 = logits[candidates[1]];
    if top0 - top1 > RESCORE_MARGIN {
        return Ok(greedy);
    }

    let rescored = rescorer.rescore(normed, &candidates)?;
    if rescored != greedy {
        eprintln!(
            "[sample-rescore] token {} -> {} (top_margin={:.4})",
            greedy,
            rescored,
            top0 - top1
        );
    }
    Ok(rescored)
}

fn top_k_candidate_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(usize, f32)> = Vec::new();
    for (idx, &val) in logits.iter().enumerate() {
        let pos = best
            .iter()
            .position(|&(_, best_val)| val > best_val)
            .unwrap_or(best.len());
        if pos < k {
            best.insert(pos, (idx, val));
            if best.len() > k {
                best.pop();
            }
        }
    }
    best.into_iter().map(|(idx, _)| idx).collect()
}

pub(crate) fn run_qwen35_prefill(
    cli: &Cli,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    oracle_model_id: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
    allow_host_lm_head_rescore: bool,
) -> Result<Qwen35Prefill> {
    let prefill_start = Instant::now();
    if cli.oracle_prefill {
        let oracle_script = qwen35_oracle_script_path();
        let output = oracle::run_oracle(
            &oracle_script,
            oracle_model_id,
            prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            oracle_device,
            true,
            false,
            fp8_oracle_dir,
            None,
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids.first().copied().ok_or_else(|| {
            anyhow::anyhow!(
                "oracle prefill produced no generated tokens; --oracle-prefill requires --max-new-tokens > 0"
            )
        })?;
        eprintln!(
            "[prefill] oracle prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
        return Ok(Qwen35Prefill {
            logits: output.prefill_logits,
            native_trace: None,
            next_token: first,
        });
    }

    let prefill_result = if cli.trace_prefill_layers {
        engine.prefill_native_with_trace(prompt_ids)?
    } else {
        engine.prefill_native_with_final_norm(prompt_ids)?
    };
    let first = sample_qwen_prefill_token(
        &prefill_result,
        host_lm_head_rescorer.filter(|_| allow_host_lm_head_rescore),
    )?;
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

    Ok(Qwen35Prefill {
        logits: prefill_result.logits,
        native_trace: Some((
            prefill_result.final_norm_trace,
            prefill_result.layer_attn_trace,
            prefill_result.layer_post_attn_norm_trace,
            prefill_result.layer_mlp_out_trace,
            prefill_result.layer_hidden_trace,
        )),
        next_token: first,
    })
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

pub(crate) fn trace_qwen35_oracle_prefill_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    prompt_ids: &[u32],
    oracle_script: &Path,
    model_id: &str,
    oracle_dtype: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    oracle_full: &oracle::OracleOutput,
) -> Result<()> {
    anyhow::ensure!(
        trace_layer > 0,
        "--trace-oracle-prefill-layer currently requires layer > 0"
    );
    let row_bytes = engine.weights().config.hidden_size * 2;
    let prefix_ids = &prompt_ids[..prompt_ids.len() - 1];
    let prefix_oracle = if prefix_ids.is_empty() {
        None
    } else {
        Some(oracle::run_oracle(
            oracle_script,
            model_id,
            prefix_ids,
            1,
            oracle_dtype,
            oracle_device,
            true,
            false,
            fp8_oracle_dir,
            None,
        )?)
    };
    let mut native_prefix_k_delta = None;
    let mut native_prefix_v_delta = None;
    let mut native_prefix_conv_delta = None;
    let mut native_prefix_recurrent_delta = None;
    if engine.weights().config.is_full_attention(trace_layer) {
        if let Some(prefix_oracle) = prefix_oracle.as_ref() {
            engine.reset()?;
            let _ = engine.prefill_native(prefix_ids)?;
            let (native_prefix_k, native_prefix_v, native_prefix_len) =
                engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
            engine.reset()?;
            engine.load_prefill_state(prefix_oracle)?;
            let (oracle_prefix_k, oracle_prefix_v, oracle_prefix_len) =
                engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
            anyhow::ensure!(
                native_prefix_len == oracle_prefix_len,
                "trace layer {trace_layer} native prefix len {} != oracle prefix len {}",
                native_prefix_len,
                oracle_prefix_len,
            );
            native_prefix_k_delta = Some(validate::max_abs_delta(
                &decode_bf16_le(&native_prefix_k),
                &decode_bf16_le(&oracle_prefix_k),
            ));
            native_prefix_v_delta = Some(validate::max_abs_delta(
                &decode_bf16_le(&native_prefix_v),
                &decode_bf16_le(&oracle_prefix_v),
            ));
        }
    } else if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.reset()?;
        let _ = engine.prefill_native(prefix_ids)?;
        let native_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing native prefix layer {trace_layer}"))?;
        let native_conv = native_layer
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native prefix layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("native prefix conv D2H layer {trace_layer}: {e}"))?;
        let native_recurrent = native_layer
            .recurrent_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("native prefix layer {trace_layer} missing recurrent_state")
            })?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("native prefix recurrent D2H layer {trace_layer}: {e}"))?;

        engine.reset()?;
        engine.load_prefill_state(prefix_oracle)?;
        let oracle_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing oracle prefix layer {trace_layer}"))?;
        let oracle_conv = oracle_layer
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("oracle prefix layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("oracle prefix conv D2H layer {trace_layer}: {e}"))?;
        let oracle_recurrent = oracle_layer
            .recurrent_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("oracle prefix layer {trace_layer} missing recurrent_state")
            })?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("oracle prefix recurrent D2H layer {trace_layer}: {e}"))?;

        native_prefix_conv_delta = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_conv),
            &decode_bf16_le(&oracle_conv),
        ));
        native_prefix_recurrent_delta = Some(validate::max_abs_delta(
            &decode_f32_le(&native_recurrent),
            &decode_f32_le(&oracle_recurrent),
        ));
    }
    engine.reset()?;
    if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.load_prefill_state(prefix_oracle)?;
    }

    let b64 = base64::engine::general_purpose::STANDARD;
    let last_row = |bytes: Vec<u8>, label: &str| -> Result<Vec<u8>> {
        anyhow::ensure!(
            bytes.len() % row_bytes == 0,
            "{label} bytes {} not divisible by row_bytes {}",
            bytes.len(),
            row_bytes,
        );
        if bytes.len() == row_bytes {
            return Ok(bytes);
        }
        let start = bytes.len() - row_bytes;
        Ok(bytes[start..].to_vec())
    };
    let oracle_inputs = oracle_full
        .layer_hidden_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_hidden_states"))?;
    let oracle_attn = oracle_full
        .layer_attn_residual_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_attn_residual_states"))?;
    let oracle_post = oracle_full
        .layer_post_attn_norm_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_post_attn_norm_states"))?;
    let oracle_mlp = oracle_full
        .layer_mlp_outputs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_mlp_outputs"))?;

    let oracle_input_bytes = last_row(
        b64.decode(oracle_inputs.get(trace_layer - 1).ok_or_else(|| {
            anyhow::anyhow!(
                "oracle layer_hidden_states missing layer {}",
                trace_layer - 1
            )
        })?)
        .map_err(|e| anyhow::anyhow!("decode oracle input hidden for layer {trace_layer}: {e}"))?,
        "oracle input hidden",
    )?;
    let oracle_attn_bytes = last_row(
        b64.decode(oracle_attn.get(trace_layer).ok_or_else(|| {
            anyhow::anyhow!("oracle layer_attn_residual_states missing layer {trace_layer}")
        })?)
        .map_err(|e| anyhow::anyhow!("decode oracle attn for layer {trace_layer}: {e}"))?,
        "oracle attn",
    )?;
    let oracle_post_bytes = last_row(
        b64.decode(oracle_post.get(trace_layer).ok_or_else(|| {
            anyhow::anyhow!("oracle layer_post_attn_norm_states missing layer {trace_layer}")
        })?)
        .map_err(|e| anyhow::anyhow!("decode oracle post-norm for layer {trace_layer}: {e}"))?,
        "oracle post-norm",
    )?;
    let oracle_mlp_bytes = last_row(
        b64.decode(oracle_mlp.get(trace_layer).ok_or_else(|| {
            anyhow::anyhow!("oracle layer_mlp_outputs missing layer {trace_layer}")
        })?)
        .map_err(|e| anyhow::anyhow!("decode oracle mlp for layer {trace_layer}: {e}"))?,
        "oracle mlp",
    )?;
    let oracle_hidden_bytes = last_row(
        b64.decode(oracle_inputs.get(trace_layer).ok_or_else(|| {
            anyhow::anyhow!("oracle layer_hidden_states missing layer {trace_layer}")
        })?)
        .map_err(|e| anyhow::anyhow!("decode oracle hidden for layer {trace_layer}: {e}"))?,
        "oracle hidden",
    )?;

    engine.set_hidden_from_bytes(&oracle_input_bytes)?;
    let trace = engine.component_trace_full_layer_from_current_hidden_with_seqlen(
        trace_layer,
        prefix_ids.len(),
    )?;
    let attn_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.attn_hidden),
        &decode_bf16_le(&oracle_attn_bytes),
    );
    let post_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.post_attn_norm),
        &decode_bf16_le(&oracle_post_bytes),
    );
    let mlp_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.mlp_out),
        &decode_bf16_le(&oracle_mlp_bytes),
    );
    let hidden_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.layer_hidden),
        &decode_bf16_le(&oracle_hidden_bytes),
    );
    eprintln!(
        "[trace-oracle-prefill-layer] layer={trace_layer} attn_delta={attn_delta:.6} post_norm_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}"
    );
    if let (Some(k_delta), Some(v_delta)) = (native_prefix_k_delta, native_prefix_v_delta) {
        eprintln!(
            "[trace-oracle-prefix-kv] layer={trace_layer} k_delta={k_delta:.6} v_delta={v_delta:.6}"
        );
    }
    if let (Some(conv_delta), Some(recurrent_delta)) =
        (native_prefix_conv_delta, native_prefix_recurrent_delta)
    {
        eprintln!(
            "[trace-oracle-prefix-linear] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={recurrent_delta:.6}"
        );
    }

    if engine.weights().config.is_full_attention(trace_layer)
        && oracle_full.traced_full_attn_layer == Some(trace_layer)
    {
        let prefix_oracle = if let Some(prefix_oracle) = prefix_oracle.as_ref() {
            prefix_oracle
        } else {
            return Ok(());
        };
        engine.reset()?;
        engine.load_prefill_state(prefix_oracle)?;
        let decode_opt_bf16 = |field: &Option<String>, label: &str| -> Result<Vec<f32>> {
            let bytes = b64
                .decode(
                    field
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("oracle output missing {label}"))?,
                )
                .map_err(|e| anyhow::anyhow!("decode oracle {label}: {e}"))?;
            Ok(decode_bf16_le(&bytes))
        };
        let oracle_normed = decode_opt_bf16(
            &oracle_full.traced_full_attn_normed,
            "traced_full_attn_normed",
        )?;
        let oracle_q_proj = decode_opt_bf16(
            &oracle_full.traced_full_attn_q_proj,
            "traced_full_attn_q_proj",
        )?;
        let oracle_gate = decode_opt_bf16(
            &oracle_full.traced_full_attn_gate_proj,
            "traced_full_attn_gate_proj",
        )?;
        let oracle_k_proj = decode_opt_bf16(
            &oracle_full.traced_full_attn_k_proj,
            "traced_full_attn_k_proj",
        )?;
        let oracle_v_proj = decode_opt_bf16(
            &oracle_full.traced_full_attn_v_proj,
            "traced_full_attn_v_proj",
        )?;
        let oracle_q_rope = decode_opt_bf16(
            &oracle_full.traced_full_attn_q_rope,
            "traced_full_attn_q_rope",
        )?;
        let oracle_k_rope = decode_opt_bf16(
            &oracle_full.traced_full_attn_k_rope,
            "traced_full_attn_k_rope",
        )?;
        let oracle_pre_gate = decode_opt_bf16(
            &oracle_full.traced_full_attn_pre_gate,
            "traced_full_attn_pre_gate",
        )?;
        let oracle_gated = decode_opt_bf16(
            &oracle_full.traced_full_attn_gated,
            "traced_full_attn_gated",
        )?;
        let oracle_gated_actual = decode_opt_bf16(
            &oracle_full.traced_full_attn_gated_actual,
            "traced_full_attn_gated_actual",
        )?;
        let (prefix_k_bytes, prefix_v_bytes, prefix_len) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        let oracle_prefix_kv = prefix_oracle
            .kv_caches
            .as_ref()
            .and_then(|caches| caches.iter().find(|kv| kv.layer == trace_layer))
            .ok_or_else(|| {
                anyhow::anyhow!("prefix oracle missing kv cache for layer {trace_layer}")
            })?;
        let oracle_prefix_k = decode_bf16_le(&b64.decode(&oracle_prefix_kv.k).map_err(|e| {
            anyhow::anyhow!("decode prefix oracle K cache layer {trace_layer}: {e}")
        })?);
        let oracle_prefix_v = decode_bf16_le(&b64.decode(&oracle_prefix_kv.v).map_err(|e| {
            anyhow::anyhow!("decode prefix oracle V cache layer {trace_layer}: {e}")
        })?);

        let stage = engine.trace_full_attention_stages_from_hidden(
            trace_layer,
            &oracle_input_bytes,
            prefix_ids.len(),
        )?;
        let stage_out = engine.trace_full_attention_layer_output_from_hidden_current_state(
            trace_layer,
            0,
            &oracle_input_bytes,
            prefix_ids.len(),
        )?;
        let normed_delta = validate::max_abs_delta(&decode_bf16_le(&stage.normed), &oracle_normed);
        let q_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_proj), &oracle_q_proj);
        let gate_proj_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage.gate_proj), &oracle_gate);
        let k_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_proj), &oracle_k_proj);
        let v_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.v_proj), &oracle_v_proj);
        let q_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_rope), &oracle_q_rope);
        let k_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_rope), &oracle_k_rope);
        let pre_gate_stage_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.pre_gate), &oracle_pre_gate);
        let gated_stage_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.gated), &oracle_gated);
        let gated_actual_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.gated), &oracle_gated_actual);
        let gated_reconstruct_delta = validate::max_abs_delta(&oracle_gated, &oracle_gated_actual);
        let head_dim = engine.weights().config.head_dim;
        let num_heads = engine.weights().config.num_attention_heads;
        let num_kv_heads = engine.weights().config.num_key_value_heads;
        let kv_groups = num_heads / num_kv_heads;
        let pre_gate_host = decode_bf16_le(&stage_out.pre_gate);
        let q_rope_host = decode_bf16_le(&stage.q_rope);
        let k_rope_step = decode_bf16_le(&stage.k_rope);
        let v_step = decode_bf16_le(&stage.v_proj);
        let prefix_k = decode_bf16_le(&prefix_k_bytes);
        let prefix_v = decode_bf16_le(&prefix_v_bytes);
        let loaded_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing loaded layer {trace_layer}"))?;
        let loaded_raw_k = decode_bf16_le(
            &loaded_layer
                .kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("loaded layer {trace_layer} missing K cache"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("loaded layer {trace_layer} K cache D2H: {e}"))?,
        );
        let loaded_raw_v = decode_bf16_le(
            &loaded_layer
                .kv_cache_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("loaded layer {trace_layer} missing V cache"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("loaded layer {trace_layer} V cache D2H: {e}"))?,
        );
        anyhow::ensure!(
            prefix_len == prefix_ids.len(),
            "trace layer {trace_layer} prefix len {} != prompt prefix len {}",
            prefix_len,
            prefix_ids.len(),
        );
        let kv_len = prefix_len + 1;
        let mut full_k = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        let mut full_v = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        for kvh in 0..num_kv_heads {
            let prefix_base = kvh * prefix_len * head_dim;
            let full_base = kvh * kv_len * head_dim;
            let step_base = kvh * head_dim;
            full_k[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&prefix_k[prefix_base..prefix_base + prefix_len * head_dim]);
            full_v[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&prefix_v[prefix_base..prefix_base + prefix_len * head_dim]);
            full_k[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&k_rope_step[step_base..step_base + head_dim]);
            full_v[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&v_step[step_base..step_base + head_dim]);
        }
        let mut host_attn_pre_gate = vec![0.0f32; num_heads * head_dim];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for qh in 0..num_heads {
            let kvh = qh / kv_groups;
            let q_base = qh * head_dim;
            let mut scores = vec![0.0f32; kv_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = (kvh * kv_len + t) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_rope_host[q_base + d] * full_k[k_base + d];
                }
                *score = acc * scale;
            }
            let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; kv_len];
            for (idx, score) in scores.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            let out_base = qh * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let v_base = (kvh * kv_len + t) * head_dim;
                    acc += w * full_v[v_base + d];
                }
                host_attn_pre_gate[out_base + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let mut oracle_host_pre_gate = vec![0.0f32; num_heads * head_dim];
        let mut oracle_full_k = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        let mut oracle_full_v = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        for kvh in 0..num_kv_heads {
            let prefix_base = kvh * prefix_len * head_dim;
            let full_base = kvh * kv_len * head_dim;
            let step_base = kvh * head_dim;
            oracle_full_k[full_base..full_base + prefix_len * head_dim].copy_from_slice(
                &oracle_prefix_k[prefix_base..prefix_base + prefix_len * head_dim],
            );
            oracle_full_v[full_base..full_base + prefix_len * head_dim].copy_from_slice(
                &oracle_prefix_v[prefix_base..prefix_base + prefix_len * head_dim],
            );
            oracle_full_k[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&oracle_k_rope[step_base..step_base + head_dim]);
            oracle_full_v[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&oracle_v_proj[step_base..step_base + head_dim]);
        }
        for qh in 0..num_heads {
            let kvh = qh / kv_groups;
            let q_base = qh * head_dim;
            let mut scores = vec![0.0f32; kv_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = (kvh * kv_len + t) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += oracle_q_rope[q_base + d] * oracle_full_k[k_base + d];
                }
                *score = acc * scale;
            }
            let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; kv_len];
            for (idx, score) in scores.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            let out_base = qh * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let v_base = (kvh * kv_len + t) * head_dim;
                    acc += w * oracle_full_v[v_base + d];
                }
                oracle_host_pre_gate[out_base + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let host_pre_gate_vs_stage = validate::max_abs_delta(&host_attn_pre_gate, &pre_gate_host);
        let host_pre_gate_vs_oracle =
            validate::max_abs_delta(&host_attn_pre_gate, &oracle_pre_gate);
        let oracle_host_pre_gate_vs_oracle =
            validate::max_abs_delta(&oracle_host_pre_gate, &oracle_pre_gate);
        let kernel_pre_gate_direct = {
            let ordinal = engine.ordinal();
            let q_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_heads, 1, head_dim],
                &f32_to_bf16_bytes(&q_rope_host),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn q H2D: {e}"))?;
            let k_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &f32_to_bf16_bytes(&full_k),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn k H2D: {e}"))?;
            let v_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &f32_to_bf16_bytes(&full_v),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn v H2D: {e}"))?;
            let mut out_gpu = gpu_hal::GpuBuffer::zeros(
                ordinal,
                gpu_hal::ScalarType::F32,
                &[num_heads, 1, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn out alloc: {e}"))?;
            kernel_ffi::prefill_ffi::full_attention_prefill(
                ordinal,
                gpu_hal::ScalarType::BF16,
                1,
                num_heads,
                num_kv_heads,
                1,
                kv_len,
                head_dim,
                scale,
                prefix_len,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                &mut out_gpu,
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn kernel: {e}"))?;
            decode_f32_le(
                &out_gpu
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("trace direct attn out D2H: {e}"))?,
            )
        };
        let direct_kernel_vs_host =
            validate::max_abs_delta(&kernel_pre_gate_direct, &host_attn_pre_gate);
        let direct_kernel_vs_oracle =
            validate::max_abs_delta(&kernel_pre_gate_direct, &oracle_pre_gate);
        let loaded_prefix_k_vs_oracle = validate::max_abs_delta(&prefix_k, &oracle_prefix_k);
        let loaded_prefix_v_vs_oracle = validate::max_abs_delta(&prefix_v, &oracle_prefix_v);
        let loaded_raw_k_vs_oracle = validate::max_abs_delta(&loaded_raw_k, &oracle_prefix_k);
        let loaded_raw_v_vs_oracle = validate::max_abs_delta(&loaded_raw_v, &oracle_prefix_v);
        let mut head_deltas = Vec::with_capacity(num_heads);
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            head_deltas.push(validate::max_abs_delta(
                &pre_gate_host[start..end],
                &oracle_pre_gate[start..end],
            ));
        }
        eprintln!(
            "[trace-oracle-full-attn] layer={trace_layer} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} k_rope_delta={k_rope_delta:.6} pre_gate_delta={pre_gate_stage_delta:.6} host_pre_gate_vs_stage={host_pre_gate_vs_stage:.6} host_pre_gate_vs_oracle={host_pre_gate_vs_oracle:.6} oracle_host_pre_gate_vs_oracle={oracle_host_pre_gate_vs_oracle:.6} direct_kernel_vs_host={direct_kernel_vs_host:.6} direct_kernel_vs_oracle={direct_kernel_vs_oracle:.6} loaded_prefix_k_vs_oracle={loaded_prefix_k_vs_oracle:.6} loaded_prefix_v_vs_oracle={loaded_prefix_v_vs_oracle:.6} loaded_raw_k_vs_oracle={loaded_raw_k_vs_oracle:.6} loaded_raw_v_vs_oracle={loaded_raw_v_vs_oracle:.6} gated_delta={gated_stage_delta:.6} gated_actual_delta={gated_actual_delta:.6} gated_reconstruct_delta={gated_reconstruct_delta:.6} pre_gate_head_deltas={head_deltas:?}"
        );
    }
    Ok(())
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

pub(crate) fn report_qwen35_virtual_kv_after_prefill(engine: &mut DecodeEngine) -> Result<()> {
    let virtual_kv_stats = engine.virtual_kv_memory_stats();
    if virtual_kv_stats.layers > 0 {
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        let pct = if virtual_kv_stats.reserved_bytes > 0 {
            100.0 * virtual_kv_stats.resident_bytes as f64 / virtual_kv_stats.reserved_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "[vmm] virtual KV logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({pct:.1}%) mappings={} layers={}",
            mib(virtual_kv_stats.logical_bytes),
            mib(virtual_kv_stats.resident_bytes),
            mib(virtual_kv_stats.reserved_bytes),
            virtual_kv_stats.mappings,
            virtual_kv_stats.layers
        );
        if std::env::var_os("SUPERSONIC_VMM_KV_STATS").is_some() {
            for (layer_idx, stats) in engine.virtual_kv_memory_stats_by_layer() {
                let layer_pct = if stats.reserved_bytes > 0 {
                    100.0 * stats.resident_bytes as f64 / stats.reserved_bytes as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[vmm] layer={layer_idx} logical={:.2}MiB logical_resident={:.2}MiB backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({layer_pct:.1}%) mappings={}",
                    mib(stats.logical_bytes),
                    mib(stats.logical_resident_bytes),
                    mib(stats.logical_backup_bytes),
                    mib(stats.resident_bytes),
                    mib(stats.reserved_bytes),
                    stats.mappings
                );
            }
        }
    }

    if std::env::var_os("SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL").is_some() {
        let before = engine.virtual_kv_memory_stats();
        if before.layers == 0 {
            eprintln!("[vmm] SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL set but virtual KV is inactive");
        } else {
            verify_qwen35_virtual_kv_eviction(engine)?;
        }
    }
    Ok(())
}

fn verify_qwen35_virtual_kv_eviction(engine: &mut DecodeEngine) -> Result<()> {
    let verify_bytes = std::env::var_os("SUPERSONIC_VMM_KV_VERIFY_EVICT_BYTES").is_some();
    let kv_before = if verify_bytes {
        Some(engine.full_attention_prefix_cache_snapshots_bf16_host()?)
    } else {
        None
    };
    engine.evict_virtual_kv_to_host()?;
    let evicted = engine.virtual_kv_memory_stats();
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[vmm] evicted virtual KV to host logical_backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(evicted.logical_backup_bytes),
        mib(evicted.resident_bytes),
        mib(evicted.reserved_bytes),
        evicted.mappings
    );
    if std::env::var_os("SUPERSONIC_VMM_KV_RESTORE_TO_VMM").is_some() {
        engine.restore_virtual_kv_from_host_to_vmm()?;
    } else {
        engine.restore_virtual_kv_from_host()?;
    }
    if let Some(kv_before) = kv_before {
        let kv_after = engine.full_attention_prefix_cache_snapshots_bf16_host()?;
        if let Some(mismatch) = first_qwen35_virtual_kv_mismatch(&kv_before, &kv_after) {
            eprintln!("[vmm] warning: virtual KV eviction byte-restore mismatch: {mismatch}");
        }
    }
    let restored = engine.virtual_kv_memory_stats();
    eprintln!(
        "[vmm] restored virtual KV from host logical_resident={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(restored.logical_resident_bytes),
        mib(restored.resident_bytes),
        mib(restored.reserved_bytes),
        restored.mappings
    );
    Ok(())
}

type Qwen35VirtualKvSnapshot = (usize, Vec<u8>, Vec<u8>, usize);

fn first_qwen35_virtual_kv_mismatch(
    before: &[Qwen35VirtualKvSnapshot],
    after: &[Qwen35VirtualKvSnapshot],
) -> Option<String> {
    for (
        (before_layer, before_k, before_v, before_len),
        (after_layer, after_k, after_v, after_len),
    ) in before.iter().zip(after.iter())
    {
        if before_layer != after_layer || before_len != after_len {
            return Some(format!(
                "layer/id mismatch before={before_layer}:{before_len} after={after_layer}:{after_len}"
            ));
        }
        let k_diff = before_k
            .iter()
            .zip(after_k.iter())
            .position(|(a, b)| a != b);
        let v_diff = before_v
            .iter()
            .zip(after_v.iter())
            .position(|(a, b)| a != b);
        if before_k.len() != after_k.len() || before_v.len() != after_v.len() {
            return Some(format!(
                "layer={before_layer} len mismatch k {}->{} v {}->{}",
                before_k.len(),
                after_k.len(),
                before_v.len(),
                after_v.len()
            ));
        }
        if k_diff.is_some() || v_diff.is_some() {
            let sample = |before: &[u8], after: &[u8], diff: Option<usize>| {
                let start = diff.unwrap_or(0);
                let end = (start + 16).min(before.len()).min(after.len());
                format!(
                    "before={:?} after={:?}",
                    &before[start..end],
                    &after[start..end]
                )
            };
            return Some(format!(
                "layer={before_layer} first_k_diff={:?} first_v_diff={:?} k_sample={} v_sample={}",
                k_diff,
                v_diff,
                sample(before_k, after_k, k_diff),
                sample(before_v, after_v, v_diff)
            ));
        }
    }
    None
}

fn sample_qwen_prefill_token(
    prefill_result: &PrefillResult,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let prefill_normed = prefill_result
        .final_norm_trace
        .as_deref()
        .map(decode_bf16_le);
    sample_qwen_logits_with_rescore(
        &prefill_result.logits,
        prefill_normed.as_deref(),
        host_lm_head_rescorer,
    )
}

pub(crate) fn load_qwen35_startup(cli: &Cli) -> Result<Qwen35Startup> {
    let config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
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

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
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

pub(crate) fn validate_qwen35_startup(
    cli: &Cli,
    model_variant: &ModelVariant,
    params: &Qwen35KernelParams,
    backend: Backend,
    registry_arch: &GpuArch,
    q4km_like: bool,
) -> Result<Qwen35Policy> {
    if cli.trace_prefill_layers && !cli.validate {
        anyhow::bail!("--trace-prefill-layers requires --validate");
    }
    if cli.trace_oracle_prefill_layer.is_some() && !cli.validate {
        anyhow::bail!("--trace-oracle-prefill-layer requires --validate");
    }
    if cli.trace_kv_fp8_cache && !cli.kv_fp8 {
        anyhow::bail!("--trace-kv-fp8-cache requires --kv-fp8");
    }
    let trace_kv_cache_enabled = cli.trace_kv_cache || cli.trace_kv_fp8_cache;
    if cli.force_kernel_decode && cli.force_component_decode {
        anyhow::bail!("Choose at most one of --force-kernel-decode or --force-component-decode");
    }
    if cli.trace_component_input_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-input-layer requires --force-component-decode");
    }
    if cli.trace_component_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_state_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-state-layer requires --force-component-decode");
    }
    if cli.trace_persistent_input_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!("--trace-persistent-input-layer requires the real 4B persistent kernel path");
    }
    if cli.trace_persistent_linear_state_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!(
            "--trace-persistent-linear-state-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_full_attn_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode))
    {
        anyhow::bail!(
            "--trace-persistent-full-attn-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_linear_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode)
            && !cli.kv_fp8)
    {
        anyhow::bail!(
            "--trace-persistent-linear-layer requires the real 4B persistent BF16 kernel path"
        );
    }

    if backend == Backend::Cuda {
        let qwen35_sm86 = matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B
                | ModelVariant::Qwen3_5_2B
                | ModelVariant::Qwen3_5_4B
                | ModelVariant::Qwen3_5_9B
        ) && *registry_arch == GpuArch::Sm86;
        if cli.int4 && !qwen35_sm86 {
            anyhow::bail!("CUDA --int4 currently supports only Qwen3.5 on sm86");
        }
        if cli.fp8_runtime && !(qwen35_sm86 || *model_variant == ModelVariant::Qwen3_6_27B) {
            anyhow::bail!(
                "CUDA --fp8-runtime currently supports only Qwen3.5 and qwen3.6-27b on sm86"
            );
        }
        if cli.kv_fp8 {
            if !qwen35_sm86 {
                anyhow::bail!("CUDA --kv-fp8 currently supports only Qwen3.5 on sm86");
            }
            if std::env::var_os("SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR").is_none() {
                std::env::set_var("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW", "128");
                eprintln!(
                    "[cuda] KV-FP8 BF16 sidecar window capped to the most recent 128 tokens on CUDA; \
                     set SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR=1 \
                     to restore full-prefix debug sidecar coverage"
                );
            }
        }
    } else if backend == Backend::Metal {
        if !matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B | ModelVariant::Qwen3_5_2B
        ) {
            anyhow::bail!("Metal only supports --model qwen3.5-0.8b or qwen3.5-2b");
        }
        if q4km_like {
            anyhow::bail!("Metal does not support --q4km/--q4km-gptq on Qwen3.5 yet");
        }
        if cli.fp8_runtime {
            anyhow::bail!("Metal does not support --fp8-runtime on Qwen3.5 yet");
        }
        if cli.kv_fp8 {
            anyhow::bail!("Metal does not support --kv-fp8 on Qwen3.5 yet");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("Metal only supports --batch-size 1");
        }
        if cli.force_kernel_decode || cli.force_component_decode {
            anyhow::bail!(
                "Metal does not support --force-kernel-decode or --force-component-decode"
            );
        }
    }

    Ok(Qwen35Policy {
        trace_kv_cache_enabled,
    })
}

pub(crate) fn check_qwen35_vram(
    cli: &Cli,
    text_config: &qwen35::config::TextConfig,
    vram: &VramBudget,
    context_tokens: usize,
    total_vram: u64,
) -> Result<()> {
    let kv_dtype_bytes = if cli.kv_fp8 {
        1usize
    } else {
        gpu_hal::ScalarType::BF16.size_in_bytes()
    };
    let kv_per_token = text_config.kv_bytes_per_token(kv_dtype_bytes);
    let effective_fixed = effective_fixed_vram(
        vram.fixed_bytes,
        cli.q4km,
        cli.q4km_gptq,
        cli.int4,
        cli.fp8_runtime,
    );
    let kv_bytes = kv_per_token * context_tokens as u64;
    let estimated_vram = ((effective_fixed + kv_bytes) as f64 * vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(effective_fixed),
        gib(kv_bytes),
        context_tokens,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        anyhow::bail!(
            "Insufficient VRAM for {context_tokens}-token context: \
             need ~{:.2}GiB (weights {:.2}GiB + KV cache {:.2}GiB), \
             GPU has {:.1}GiB. Reduce --context-size or --max-new-tokens.",
            gib(estimated_vram),
            gib(effective_fixed),
            gib(kv_bytes),
            gib(total_vram),
        );
    }
    Ok(())
}

pub(crate) fn load_qwen35_engine(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    params: &Qwen35KernelParams,
    backend: Backend,
    gpu_arch: GpuArch,
    ordinal: usize,
    bootstrap_downloaded: bool,
    q4km_like: bool,
    context_tokens: usize,
) -> Result<Qwen35EngineSetup> {
    let t0 = std::time::Instant::now();
    let weights = load_qwen35_weights(
        cli,
        model_variant,
        text_config,
        ordinal,
        params.weight_prefix,
        bootstrap_downloaded,
        q4km_like,
    )?;
    if weights.is_fp8 {
        eprintln!(
            "[weights] FP8 runtime dequant active (block_size={})",
            weights.fp8_block_size
        );
    }
    if weights.is_int4 {
        eprintln!(
            "[weights] INT4 runtime dequant active (group_size={})",
            weights.int4_group_size
        );
    }
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let cuda_08b_hero_disabled = std::env::var_os("SUPERSONIC_DISABLE_CUDA_08B_HERO").is_some();
    let cuda_08b_hero_candidate = backend == Backend::Cuda
        && gpu_arch == GpuArch::Sm86
        && *model_variant == ModelVariant::Qwen3_5_0_8B
        && cli.batch_size == 1
        && !cli.validate
        && !(cli.gpu_validate && cli.batch_size == 1)
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && !weights.is_fp8
        && !weights.is_int4
        && !cuda_08b_hero_disabled;
    let use_4b_kernel = params.use_4b_kernel && !cuda_08b_hero_candidate;

    if cli.batch_size > 1 && !use_4b_kernel {
        anyhow::bail!("--batch-size > 1 requires 4B kernel (2B/4B/9B models)");
    }
    if cli.batch_size < 1 || cli.batch_size > kernel_ffi::MAX_BATCH_SIZE {
        anyhow::bail!("--batch-size must be 1..{}", kernel_ffi::MAX_BATCH_SIZE);
    }

    let required_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        text_config.num_attention_heads,
        text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );
    let attn_scratch_floats = params.attn_scratch_floats.max(required_attn_scratch);
    if attn_scratch_floats > params.attn_scratch_floats {
        eprintln!(
            "[scratch] context={} → attn_scratch_floats={} (registry floor {})",
            context_tokens, attn_scratch_floats, params.attn_scratch_floats
        );
    }

    let mut engine = DecodeEngine::new(
        weights,
        ordinal,
        params.proj_buf_floats,
        attn_scratch_floats,
        params.kv_chunk_size,
        use_4b_kernel,
        cli.prefill_chunk_size,
        cli.kv_fp8,
        cli.batch_size,
    )?;
    engine.set_decode_context_limit(context_tokens);
    let allow_host_lm_head_rescore =
        cli.no_bake && !engine.weights().is_fp8 && !engine.weights().is_int4;

    Ok(Qwen35EngineSetup {
        engine,
        use_4b_kernel,
        cuda_08b_hero_enabled: cuda_08b_hero_candidate,
        allow_host_lm_head_rescore,
    })
}

fn effective_fixed_vram(
    fixed_bytes: u64,
    q4km: bool,
    q4km_gptq: bool,
    int4: bool,
    fp8_runtime: bool,
) -> u64 {
    if q4km {
        (fixed_bytes as f64 * 0.30) as u64
    } else if q4km_gptq || int4 {
        // INT4: weights ~= fixed * 0.9, scratch ~= fixed * 0.1
        // INT4 weights = weights / 4 + ~5% scale/zero overhead
        // total ~= fixed * 0.9 * 0.3 + fixed * 0.1 = fixed * 0.37
        (fixed_bytes as f64 * 0.37) as u64
    } else if fp8_runtime {
        // FP8: weights / 2 plus scale/scratch overhead.
        (fixed_bytes as f64 * 0.55) as u64
    } else {
        fixed_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{effective_fixed_vram, resolve_qwen_oracle_model_id};
    use crate::registry::ModelVariant;

    #[test]
    fn q4km_gptq_uses_int4_vram_estimate() {
        assert_eq!(effective_fixed_vram(100, false, true, false, false), 37);
    }

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
