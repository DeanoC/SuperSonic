use anyhow::Result;
use base64::Engine as _;
use std::path::Path;

use crate::decode_engine::DecodeEngine;
use crate::oracle;
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le, f32_to_bf16_bytes,
};
use crate::validate;

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
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (full_k, full_v) = append_decode_kv_step(
            &prefix_k,
            &prefix_v,
            &k_rope_step,
            &v_step,
            num_kv_heads,
            prefix_len,
            head_dim,
        );
        let host_attn_pre_gate = reconstruct_attention_pre_gate(
            &q_rope_host,
            &full_k,
            &full_v,
            num_heads,
            num_kv_heads,
            kv_len,
            head_dim,
            scale,
        );
        let (oracle_full_k, oracle_full_v) = append_decode_kv_step(
            &oracle_prefix_k,
            &oracle_prefix_v,
            &oracle_k_rope,
            &oracle_v_proj,
            num_kv_heads,
            prefix_len,
            head_dim,
        );
        let oracle_host_pre_gate = reconstruct_attention_pre_gate(
            &oracle_q_rope,
            &oracle_full_k,
            &oracle_full_v,
            num_heads,
            num_kv_heads,
            kv_len,
            head_dim,
            scale,
        );
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

fn append_decode_kv_step(
    prefix_k: &[f32],
    prefix_v: &[f32],
    step_k: &[f32],
    step_v: &[f32],
    num_kv_heads: usize,
    prefix_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
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
            .copy_from_slice(&step_k[step_base..step_base + head_dim]);
        full_v[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
            .copy_from_slice(&step_v[step_base..step_base + head_dim]);
    }
    (full_k, full_v)
}

fn reconstruct_attention_pre_gate(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_groups = num_heads / num_kv_heads;
    let mut pre_gate = vec![0.0f32; num_heads * head_dim];
    for qh in 0..num_heads {
        let kvh = qh / kv_groups;
        let q_base = qh * head_dim;
        let mut scores = vec![0.0f32; kv_len];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_base = (kvh * kv_len + t) * head_dim;
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q[q_base + d] * k[k_base + d];
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
                acc += w * v[v_base + d];
            }
            pre_gate[out_base + d] = if denom > 0.0 { acc / denom } else { 0.0 };
        }
    }
    pre_gate
}
