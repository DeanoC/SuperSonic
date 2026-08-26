use std::time::Instant;

use anyhow::Result;

use crate::decode_engine::DecodeStageTimings;
use crate::qwen38_decode_report::{emit_qwen38_decode_report, Qwen38DecodeReport};
use crate::qwen38_engine_setup::load_qwen38_engine;
use crate::qwen38_prefill::run_qwen38_prefill;
use crate::qwen38_startup::load_qwen38_startup;
use crate::profiling::DecodeProfileScope;
use crate::registry::{FamilyParams, RegistryEntry};
use crate::Cli;
use gpu_hal::Backend;

pub(crate) fn run_qwen38(cli: &Cli, entry: &RegistryEntry, ordinal: usize) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Qwen38(params) => params,
    };
    let startup = load_qwen38_startup(cli)?;
    let mut setup = load_qwen38_engine(
        cli,
        &startup.text_config,
        params,
        ordinal,
        startup.context_tokens,
    )?;
    let prefill = run_qwen38_prefill(cli, &mut setup.engine, &startup.prompt_ids)?;
    setup.engine.prepare_hip_gqh_decode()?;

    let decode_profile = DecodeProfileScope::new(
        cli.profile_decode,
        cli.profile_decode_json.as_deref(),
        "qwen3.8",
        &cli.model,
        backend_label(),
        cli.max_new_tokens,
    );

    let mut generated_ids = Vec::new();
    let mut timings = DecodeStageTimings::default();
    let mut timing_steps = 0usize;
    let decode_start = Instant::now();
    let mut next_token = prefill.next_token;
    let eos_ids = qwen_eos_ids(&startup.text_config, &startup.tokenizer, cli.chat);

    if setup.engine.mtp_spec_enabled() {
        let normed = prefill.final_norm.as_deref().ok_or_else(|| {
            anyhow::anyhow!("Qwen3.8 MTP requires the prefill final-norm hidden state")
        })?;
        setup.engine.seed_mtp_h_from_normed(normed)?;
        let mut seqlen_offset = startup.prompt_ids.len();
        while generated_ids.len() < cli.max_new_tokens {
            if !cli.ignore_eos && eos_ids.contains(&next_token) {
                break;
            }
            let remaining = cli.max_new_tokens - generated_ids.len();
            let round = setup
                .engine
                .run_mtp_spec_round(next_token, seqlen_offset, remaining)?;
            eprintln!(
                "[qwen38-mtp] pos={} drafted={} accepted={} emitted={}",
                seqlen_offset,
                round.n_drafted,
                round.n_accepted,
                round.emitted.len()
            );
            for token in round.emitted {
                generated_ids.push(token);
                seqlen_offset += 1;
                if generated_ids.len() >= cli.max_new_tokens
                    || (!cli.ignore_eos && eos_ids.contains(&token))
                {
                    break;
                }
            }
            next_token = round.next_token;
            if generated_ids
                .last()
                .is_some_and(|token| !cli.ignore_eos && eos_ids.contains(token))
            {
                break;
            }
        }
        if let Some((hits, total, rounds, emitted)) = setup.engine.mtp_spec_summary() {
            eprintln!(
                "[qwen38-mtp] accept {hits}/{total} ({:.0}%) over {rounds} rounds, emitted {emitted}",
                if total == 0 {
                    0.0
                } else {
                    100.0 * hits as f32 / total as f32
                }
            );
        }
    } else {
        for step in 0..cli.max_new_tokens {
            if !cli.ignore_eos && eos_ids.contains(&next_token) {
                break;
            }
            let (sampled, step_timings) = setup
                .engine
                .decode_step_greedy(next_token, startup.prompt_ids.len() + step)?;
            timings.add_assign(step_timings);
            timing_steps += 1;
            generated_ids.push(next_token);
            next_token = sampled;
        }
    }

    decode_profile.finish_with_steps(timing_steps)?;

    emit_qwen38_decode_report(Qwen38DecodeReport {
        tokenizer: &startup.tokenizer,
        prompt_ids: &startup.prompt_ids,
        generated_ids: &generated_ids,
        emit_generated_json: cli.emit_generated_json,
        decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
        emit_stage_timings: cli.emit_stage_timings,
        native_decode_timings: &timings,
        native_decode_timing_steps: timing_steps,
    })
}

fn backend_label() -> &'static str {
    match gpu_hal::current_backend() {
        Backend::Hip => "HIP",
        #[cfg(supersonic_backend_metal)]
        Backend::Metal => "Metal",
    }
}

fn qwen_eos_ids(
    text_config: &qwen38::config::TextConfig,
    tokenizer: &tokenizers::Tokenizer,
    chat: bool,
) -> Vec<u32> {
    let mut eos_ids = text_config.eos_token_ids();
    if chat {
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
    }
    eos_ids
}
