use anyhow::Result;

use crate::decode_engine::DecodeStageTimings;

pub(crate) struct Qwen35DecodeReport<'a> {
    pub(crate) tokenizer: &'a tokenizers::Tokenizer,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) generated_ids: &'a [u32],
    pub(crate) emit_generated_json: bool,
    pub(crate) decode_ms: f64,
    pub(crate) max_delta: f32,
    pub(crate) gpu_max_delta: f32,
    pub(crate) batch_size: usize,
    pub(crate) emit_stage_timings: bool,
    pub(crate) native_decode_timings: &'a DecodeStageTimings,
    pub(crate) native_decode_timing_steps: usize,
}

pub(crate) fn emit_qwen35_decode_report(report: Qwen35DecodeReport<'_>) -> Result<()> {
    let all_ids: Vec<u32> = report
        .prompt_ids
        .iter()
        .copied()
        .chain(report.generated_ids.iter().copied())
        .collect();
    let text = report
        .tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    let generated_text = report
        .tokenizer
        .decode(report.generated_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize generated suffix: {e}"))?;

    println!("{text}");
    if report.emit_generated_json {
        println!(
            "[generated_json] {}",
            serde_json::to_string(&generated_text)?
        );
    }
    println!(
        "[tokens] {}",
        report
            .generated_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={:.0} ms_per_tok={:.0} decode_max_delta={:.4} gpu_oracle_max_delta={:.4} batch_size={}",
        report.prompt_ids.len(),
        report.generated_ids.len(),
        report.decode_ms,
        if report.generated_ids.is_empty() {
            0.0
        } else {
            report.decode_ms / report.generated_ids.len() as f64
        },
        report.max_delta,
        report.gpu_max_delta,
        report.batch_size,
    );
    if report.emit_stage_timings {
        emit_qwen35_stage_timings(
            report.native_decode_timings,
            report.native_decode_timing_steps,
        );
    }

    Ok(())
}

fn emit_qwen35_stage_timings(timings: &DecodeStageTimings, steps: usize) {
    if steps > 0 {
        eprintln!(
            "[stage-timings] steps={} persistent_ms={:.3} rms_norm_ms={:.3} lm_head_ms={:.3} logits_d2h_ms={:.3} host_sampling_ms={:.3} gpu_argmax_ms={:.3} token_d2h_ms={:.3} total_native_decode_ms={:.3} persistent_full_attn_ms={:.3} persistent_full_attn_proj_ms={:.3} persistent_full_attn_core_ms={:.3} persistent_full_attn_out_ms={:.3} persistent_linear_proj_ms={:.3} persistent_linear_core_ms={:.3} persistent_linear_core_conv_ms={:.3} persistent_linear_core_recurrent_ms={:.3} persistent_linear_core_post_ms={:.3} persistent_linear_out_ms={:.3} persistent_mlp_gate_up_ms={:.3} persistent_mlp_down_ms={:.3}",
            steps,
            timings.persistent_ms,
            timings.rms_norm_ms,
            timings.lm_head_ms,
            timings.logits_d2h_ms,
            timings.host_sampling_ms,
            timings.gpu_argmax_ms,
            timings.token_d2h_ms,
            timings.total_ms(),
            timings.persistent_full_attn_ms,
            timings.persistent_full_attn_proj_ms,
            timings.persistent_full_attn_core_ms,
            timings.persistent_full_attn_out_ms,
            timings.persistent_linear_proj_ms,
            timings.persistent_linear_core_ms,
            timings.persistent_linear_core_conv_ms,
            timings.persistent_linear_core_recurrent_ms,
            timings.persistent_linear_core_post_ms,
            timings.persistent_linear_out_ms,
            timings.persistent_mlp_gate_up_ms,
            timings.persistent_mlp_down_ms,
        );
    } else {
        eprintln!(
            "[stage-timings] steps=0 note=no native decode stage timings collected for this path"
        );
    }
}
