use std::time::Instant;

use anyhow::Result;

use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::profiling::DflashProfileScope;
use crate::qwen38_decode_report::{emit_qwen38_decode_report, Qwen38DecodeReport};
use crate::qwen38_engine_setup::load_qwen38_engine;
use crate::qwen38_prefill::run_qwen38_prefill;
use crate::qwen38_startup::load_qwen38_startup;
use crate::registry::{FamilyParams, RegistryEntry};
use crate::Cli;

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
    let mut generated_ids = Vec::new();
    let mut timings = DecodeStageTimings::default();
    let mut timing_steps = 0usize;
    let eos_ids = qwen_eos_ids(&startup.text_config, &startup.tokenizer, cli.chat);
    let decode_ms;

    if let Some(dflash) = setup.dflash.as_mut() {
        // DFlash2 handles its own prefill (with hidden-state capture).
        decode_ms = run_qwen38_dflash(
            cli,
            &mut setup.engine,
            dflash,
            &startup.prompt_ids,
            &eos_ids,
            &mut generated_ids,
        )?;
    } else {
        let prefill = run_qwen38_prefill(cli, &mut setup.engine, &startup.prompt_ids)?;
        setup.engine.prepare_hip_gqh_decode()?;
        let decode_start = Instant::now();
        let mut next_token = prefill.next_token;

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
                let round =
                    setup
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
        decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    }

    emit_qwen38_decode_report(Qwen38DecodeReport {
        tokenizer: &startup.tokenizer,
        prompt_ids: &startup.prompt_ids,
        generated_ids: &generated_ids,
        emit_generated_json: cli.emit_generated_json,
        decode_ms,
        emit_stage_timings: cli.emit_stage_timings,
        native_decode_timings: &timings,
        native_decode_timing_steps: timing_steps,
    })
}

fn run_qwen38_dflash(
    cli: &Cli,
    engine: &mut DecodeEngine,
    dflash: &mut supersonic_runtime::dflash_spec::DflashSpecDecoder,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    generated_ids: &mut Vec<u32>,
) -> Result<f64> {
    let prompt_len = prompt_ids.len();

    // Prefill with capture: record target hidden states at the 5 draft
    // target layers for all prompt positions.
    let capture_ptr = dflash.capture_ptr();
    let capture = unsafe { &mut *capture_ptr };
    let prefill_result = engine.prefill_with_dflash_capture(prompt_ids, capture)?;
    capture.committed = prompt_len;

    // First token from prefill logits.
    let first_token = DecodeEngine::greedy_sample(&prefill_result.logits);
    eprintln!("[prefill] dflash prefill done, first token: {first_token}");

    engine.prepare_hip_gqh_decode()?;

    let mut committed = prompt_len;
    let mut last_tok = first_token;
    let profile_scope =
        DflashProfileScope::new(cli.profile_prefill || cli.profile_prefill_json.is_some());
    let decode_start = Instant::now();

    while generated_ids.len() < cli.max_new_tokens {
        let remaining = cli.max_new_tokens - generated_ids.len();
        let round_start = Instant::now();
        let round = dflash.run_round(engine, last_tok, committed, remaining)?;
        let round_ms = round_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dflash] pos={} drafted={} accepted={} emitted={} round_ms={round_ms:.1}",
            committed,
            round.n_drafted,
            round.n_accepted,
            round.emitted.len()
        );
        let mut hit_eos = false;
        for token in &round.emitted {
            generated_ids.push(*token);
            committed += 1;
            if !cli.ignore_eos && eos_ids.contains(token) {
                hit_eos = true;
                break;
            }
            if generated_ids.len() >= cli.max_new_tokens {
                break;
            }
        }
        if hit_eos || generated_ids.len() >= cli.max_new_tokens {
            break;
        }
        last_tok = round.next_token;
    }

    let summary = dflash.summary();
    eprintln!(
        "[dflash] accept {}/{} ({:.0}%) over {} rounds",
        summary.n_accepted,
        summary.n_drafted,
        if summary.n_drafted == 0 {
            0.0
        } else {
            100.0 * summary.n_accepted as f32 / summary.n_drafted as f32
        },
        summary.n_rounds,
    );
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    profile_scope.finish();
    Ok(decode_ms)
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
