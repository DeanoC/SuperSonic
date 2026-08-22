use std::time::Instant;

use anyhow::Result;

use crate::decode_engine::DecodeStageTimings;
use crate::qwen35_decode_report::{emit_qwen35_decode_report, Qwen35DecodeReport};
use crate::qwen35_engine_setup::{install_qwen35_launch_preset, load_qwen35_engine};
use crate::qwen35_prefill::run_qwen35_prefill;
use crate::qwen35_startup::load_qwen35_startup;
use crate::registry::{FamilyParams, RegistryEntry};
use crate::Cli;

pub(crate) fn run_qwen35(cli: &Cli, entry: &RegistryEntry, ordinal: usize) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Qwen35(params) => params,
        _ => unreachable!("the direct startup entry must be Qwen3.8 Qwen35 parameters"),
    };
    install_qwen35_launch_preset(entry);

    let startup = load_qwen35_startup(cli)?;
    let mut setup = load_qwen35_engine(
        cli,
        &startup.text_config,
        params,
        ordinal,
        startup.context_tokens,
    )?;
    let prefill = run_qwen35_prefill(cli, &mut setup.engine, &startup.prompt_ids)?;
    setup.engine.prepare_hip_gqh_decode()?;

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
                .decode_step_hip_fast_greedy(next_token, startup.prompt_ids.len() + step)?;
            timings.add_assign(step_timings);
            timing_steps += 1;
            generated_ids.push(next_token);
            next_token = sampled;
        }
    }

    emit_qwen35_decode_report(Qwen35DecodeReport {
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

fn qwen_eos_ids(
    text_config: &qwen35::config::TextConfig,
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
