//! Rungs 11–13: component decode and a chat-templated Hello generate.

use std::path::PathBuf;
use std::sync::Mutex;

use gpu_hal::{GpuBuffer, ScalarType};
use qwen38::gguf_ingest::load_text_config;
use qwen38::scratch::required_attn_scratch_floats;
use qwen38::weights::Qwen38Weights;
use supersonic_runtime::chat_template::{ChatMessage, ChatTemplate};
use supersonic_runtime::decode_engine::DecodeEngine;
#[cfg(feature = "scalar-head-lab")]
use supersonic_runtime::decode_engine::ScalarHeadLabRoute;

static GPU: Mutex<()> = Mutex::new(());

fn gguf_path() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
        if require_gqh_artifacts() {
            panic!("SUPERSONIC_GQH_GGUF is required for Qwen3.8 GQH artifact tests");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else if require_gqh_artifacts() {
        panic!(
            "SUPERSONIC_GQH_GGUF points to a missing artifact: {}",
            path.display()
        );
    } else {
        None
    }
}

fn require_gqh_artifacts() -> bool {
    std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
}

fn qwen38_model_dir() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_QWEN38_MODEL_DIR") else {
        if require_gqh_artifacts() {
            panic!("SUPERSONIC_QWEN38_MODEL_DIR is required for Qwen3.8 GQH artifact tests");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if !path.is_dir() {
        if require_gqh_artifacts() {
            panic!(
                "SUPERSONIC_QWEN38_MODEL_DIR points to a missing model directory: {}",
                path.display()
            );
        }
        return None;
    }
    for required in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let child = path.join(required);
        if !child.is_file() {
            if require_gqh_artifacts() {
                panic!(
                    "SUPERSONIC_QWEN38_MODEL_DIR is missing required file: {}",
                    child.display()
                );
            }
            return None;
        }
    }
    Some(path)
}

fn greedy_token(logits: &[f32]) -> u32 {
    DecodeEngine::greedy_sample(logits)
}

fn hip_query_policy<T, E: std::fmt::Display>(
    query: Result<T, E>,
    require_artifacts: bool,
) -> Result<Option<T>, String> {
    match query {
        Ok(info) => Ok(Some(info)),
        Err(error) if require_artifacts => Err(format!(
            "HIP device 0 query failed while SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1: {error}; verify HIP_VISIBLE_DEVICES and ROCm device access"
        )),
        Err(error) => {
            eprintln!("skip: HIP device 0 query failed: {error}");
            Ok(None)
        }
    }
}

fn build_engine(max_context: usize) -> Option<(DecodeEngine, qwen38::config::TextConfig)> {
    let path = gguf_path()?;
    let model_dir = qwen38_model_dir()?;
    match hip_query_policy(kernel_ffi::query_gpu_info(0), require_gqh_artifacts()) {
        Ok(Some(_)) => {}
        Ok(None) => return None,
        Err(error) => panic!("{error}"),
    }
    let ordinal = 0usize;
    let config = load_text_config(&model_dir).expect("hf config");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let attn_scratch = required_attn_scratch_floats(
        config.num_attention_heads,
        config.head_dim,
        max_context,
        256,
    )
    .max(24_576);
    let engine = DecodeEngine::new(weights, ordinal, 16_480, attn_scratch, 256, true, 0)
        .expect("DecodeEngine");
    Some((engine, config))
}

#[test]
fn hip_query_policy_rejects_strict_configured_failure() {
    let error = hip_query_policy::<(), _>(Err("driver unavailable"), true)
        .expect_err("strict configured HIP query must fail");
    assert!(error.contains("HIP device 0 query failed"));
    assert!(error.contains("driver unavailable"));
    assert!(error.contains("SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1"));
}

#[test]
fn hip_query_policy_allows_unconfigured_skip() {
    let result = hip_query_policy::<(), _>(Err("driver unavailable"), false)
        .expect("unconfigured HIP query may skip");
    assert!(result.is_none());
}

#[cfg(feature = "scalar-head-lab")]
fn scalar_head_lab_prompt_ids(prompt_text: &str, expected_len: usize) -> Vec<u32> {
    let model_dir = qwen38_model_dir().expect("model dir");
    let template = ChatTemplate::try_load(&model_dir)
        .expect("load chat template")
        .expect("Qwen3.8 ships a chat template");
    let prompt = template
        .render(&[ChatMessage::text("user", prompt_text)], true)
        .expect("render scalar-head lab prompt");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt_ids = tokenizer
        .encode(prompt, false)
        .expect("encode scalar-head lab prompt")
        .get_ids()
        .to_vec();
    assert_eq!(
        prompt_ids.len(),
        expected_len,
        "unexpected scalar-head lab prompt length for {prompt_text:?}"
    );
    prompt_ids
}

#[cfg(feature = "scalar-head-lab")]
fn scalar_head_lab_generate(
    engine: &mut DecodeEngine,
    mut next_token: u32,
    prompt_len: usize,
    route: ScalarHeadLabRoute,
) -> (Vec<u32>, f64) {
    engine
        .set_scalar_head_lab_route(route)
        .expect("select fixed scalar-head lab route");
    let started = std::time::Instant::now();
    let mut generated = Vec::with_capacity(32);
    let mut pos = prompt_len;
    while generated.len() < 32 {
        generated.push(next_token);
        if generated.len() == 32 {
            break;
        }
        (next_token, _) = engine
            .decode_step_hip_fast_greedy(next_token, pos)
            .unwrap_or_else(|e| panic!("scalar-head lab decode pos={pos}: {e}"));
        pos += 1;
    }
    assert_eq!(generated.len(), 32, "lab route must generate 32 tokens");
    (generated, started.elapsed().as_secs_f64() * 1000.0)
}

#[cfg(feature = "scalar-head-lab")]
fn run_scalar_head_lab_repeatability_case(case: &str, prompt_text: &str, expected_len: usize) {
    let prompt_ids = scalar_head_lab_prompt_ids(prompt_text, expected_len);

    let Some((mut first_engine, _)) = build_engine(prompt_ids.len() + 32) else {
        return;
    };
    let first_prefill = first_engine
        .prefill_native(&prompt_ids)
        .expect("first scalar-head lab prefill");
    let first_token = greedy_token(&first_prefill);
    if case == "cold-chat" {
        assert_eq!(
            first_token, 32,
            "established cold-chat prefill trajectory changed"
        );
    }
    let prefix = first_engine
        .snapshot_prefix(first_prefill)
        .expect("snapshot scalar-head lab prefix");
    first_engine
        .prepare_hip_gqh_decode()
        .expect("prepare first scalar-head lab decode");
    let (first_scalar, first_scalar_ms) = scalar_head_lab_generate(
        &mut first_engine,
        first_token,
        prompt_ids.len(),
        ScalarHeadLabRoute::RawQ6Scalar,
    );

    let restored_logits = first_engine
        .restore_prefix(&prefix)
        .expect("restore production WMMA control prefix");
    let restored_token = greedy_token(&restored_logits);
    assert_eq!(restored_token, first_token, "restored prefix token changed");
    let (production_wmma, production_wmma_ms) = scalar_head_lab_generate(
        &mut first_engine,
        restored_token,
        prompt_ids.len(),
        ScalarHeadLabRoute::ProductionWmma,
    );
    drop(prefix);
    drop(first_engine);

    let Some((mut second_engine, _)) = build_engine(prompt_ids.len() + 32) else {
        return;
    };
    let second_prefill = second_engine
        .prefill_native(&prompt_ids)
        .expect("second scalar-head lab prefill");
    let second_token = greedy_token(&second_prefill);
    second_engine
        .prepare_hip_gqh_decode()
        .expect("prepare second scalar-head lab decode");
    let (second_scalar, second_scalar_ms) = scalar_head_lab_generate(
        &mut second_engine,
        second_token,
        prompt_ids.len(),
        ScalarHeadLabRoute::RawQ6Scalar,
    );

    assert_eq!(
        first_scalar, second_scalar,
        "raw-Q6 scalar tokens changed across fresh engines"
    );
    let first_wmma_divergence = first_scalar
        .iter()
        .zip(&production_wmma)
        .position(|(scalar, wmma)| scalar != wmma)
        .map(|index| (index, first_scalar[index], production_wmma[index]));
    if case == "cold-chat" {
        assert_eq!(
            first_wmma_divergence,
            Some((12, 8869, 9189)),
            "raw-Q6 scalar route must retain its established same-prefix WMMA divergence"
        );
    }
    println!(
        "scalar_head_lab_evidence case={case:?} prompt_tokens={} scalar_first={first_scalar:?} scalar_second={second_scalar:?} production_wmma={production_wmma:?} first_wmma_divergence={first_wmma_divergence:?} scalar_first_ms={first_scalar_ms:.3} scalar_second_ms={second_scalar_ms:.3} production_wmma_ms={production_wmma_ms:.3}",
        prompt_ids.len(),
    );
}

#[cfg(feature = "scalar-head-lab")]
fn run_scalar_head_lab_hello_measurement(route: ScalarHeadLabRoute) {
    const GENERATED_TOKENS: usize = 32;
    const TIMED_DECODE_STEPS: usize = GENERATED_TOKENS - 1;

    let prompt_ids = scalar_head_lab_prompt_ids("Hello", 13);
    let Some((mut engine, _)) = build_engine(prompt_ids.len() + GENERATED_TOKENS) else {
        return;
    };
    let prefill_logits = engine
        .prefill_native(&prompt_ids)
        .expect("scalar-head measurement prefill");
    let mut next_token = greedy_token(&prefill_logits);
    assert_eq!(next_token, 9419, "Hello prefill trajectory changed");
    engine
        .prepare_hip_gqh_decode()
        .expect("prepare scalar-head measurement decode");
    engine
        .set_scalar_head_lab_route(route)
        .expect("select fixed scalar-head measurement route");

    let raw_started = std::time::Instant::now();
    let mut tokens = vec![next_token];
    let mut lm_head_ms = 0.0f64;
    let mut timed_decode_steps = 0usize;
    let mut pos = prompt_ids.len();
    while tokens.len() < GENERATED_TOKENS {
        let (sampled_token, timings) = engine
            .decode_step_hip_fast_greedy(next_token, pos)
            .unwrap_or_else(|e| panic!("scalar-head measurement decode pos={pos}: {e}"));
        assert!(
            timings.lm_head_ms.is_finite() && timings.lm_head_ms > 0.0,
            "route {route:?} returned invalid lm_head_ms={} at pos={pos}",
            timings.lm_head_ms
        );
        lm_head_ms += timings.lm_head_ms;
        timed_decode_steps += 1;
        next_token = sampled_token;
        tokens.push(next_token);
        pos += 1;
    }
    let raw_total_ms = raw_started.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(tokens.len(), GENERATED_TOKENS);
    assert_eq!(timed_decode_steps, TIMED_DECODE_STEPS);
    assert!(lm_head_ms.is_finite() && lm_head_ms > 0.0);
    assert!(raw_total_ms.is_finite() && raw_total_ms > 0.0);
    println!(
        "scalar_head_lab_measurement route={route:?} prompt_tokens={} generated_tokens={} timed_decode_steps={timed_decode_steps} lm_head_ms={lm_head_ms:.6} raw_total_ms={raw_total_ms:.6} tokens={tokens:?}",
        prompt_ids.len(),
        tokens.len(),
    );
}

#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_hello_is_repeatable() {
    let _guard = GPU.lock().expect("gpu lock");
    run_scalar_head_lab_repeatability_case("hello", "Hello", 13);
}

#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_cold_chat_is_repeatable() {
    let _guard = GPU.lock().expect("gpu lock");
    run_scalar_head_lab_repeatability_case(
        "cold-chat",
        "Emit a single sentence describing cold-load benchmark startup.",
        23,
    );
}

#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_measure_hello_raw_q6_scalar() {
    let _guard = GPU.lock().expect("gpu lock");
    run_scalar_head_lab_hello_measurement(ScalarHeadLabRoute::RawQ6Scalar);
}

#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_measure_hello_production_wmma() {
    let _guard = GPU.lock().expect("gpu lock");
    run_scalar_head_lab_hello_measurement(ScalarHeadLabRoute::ProductionWmma);
}

#[test]
#[ignore = "requires R9700 artifact CI"]
fn rung11_one_token_component_decode() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, config)) = build_engine(16) else {
        return;
    };

    let token = 9419u32; // "Hello" in the Qwen3.8 tokenizer
    let started = std::time::Instant::now();
    let logits = engine.decode_step(token, 0).expect("decode_step");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits contain a non-finite value"
    );
    let energy: f32 = logits.iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "logits are all zeros");

    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let argmax = ranked[0].0 as u32;
    let top5: Vec<(u32, f32)> = ranked
        .iter()
        .take(5)
        .map(|(i, v)| (*i as u32, *v))
        .collect();

    let model_dir = qwen38_model_dir().expect("model dir");
    let decoded = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .ok()
        .and_then(|tok| tok.decode(&[argmax], false).ok());
    println!(
        "rung11: token={token} argmax={argmax} piece={decoded:?} top5={top5:?} energy={energy:.4} {elapsed_ms:.0}ms"
    );

    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).ok();
    let mut tokens = vec![token, argmax];
    for step in 1..8 {
        let started = std::time::Instant::now();
        let step_logits = engine
            .decode_step(tokens[step], step)
            .unwrap_or_else(|e| panic!("decode_step pos={step}: {e}"));
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert!(
            step_logits.iter().all(|v| v.is_finite()),
            "logits at pos={step} not finite"
        );
        let next = step_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .expect("argmax");
        tokens.push(next);
        let piece = tokenizer
            .as_ref()
            .and_then(|tok| tok.decode(&[next], false).ok());
        println!("rung12: pos={step} token={next} piece={piece:?} {elapsed_ms:.0}ms");
    }
    let text = tokenizer
        .as_ref()
        .and_then(|tok| tok.decode(&tokens, false).ok());
    println!("rung12: tokens={tokens:?} text={text:?}");
}

#[test]
#[ignore = "requires R9700 artifact CI"]
fn rung13_chat_hello_generate() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, config)) = build_engine(64) else {
        return;
    };
    let model_dir = qwen38_model_dir().expect("model dir");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer.json");
    let template = ChatTemplate::try_load(&model_dir)
        .expect("load chat template")
        .expect("Qwen3.8 ships a chat template");
    let prompt = template
        .render(&[ChatMessage::text("user", "Hello")], true)
        .expect("render chat");
    println!("rung13 prompt:\n{prompt}");
    let prompt_ids = tokenizer
        .encode(prompt.as_str(), false)
        .expect("encode prompt")
        .get_ids()
        .to_vec();
    assert!(
        !prompt_ids.is_empty() && prompt_ids.len() < 48,
        "unexpected prompt length {}",
        prompt_ids.len()
    );
    assert!(
        prompt_ids.len() > 8,
        "canonical chat prefill must exercise the >8-token GQH GEMM path; got {}",
        prompt_ids.len()
    );

    let eos = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

    let prefill_start = std::time::Instant::now();
    let mut logits = engine
        .prefill_native(&prompt_ids)
        .unwrap_or_else(|e| panic!("canonical >8-token prefill failed: {e}"));
    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "prefill logits are not finite"
    );
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "rung13: prefill {} tokens in {prefill_ms:.0}ms",
        prompt_ids.len()
    );

    let mut generated = Vec::new();
    let mut pos = prompt_ids.len();
    for step in 0..16 {
        let next = greedy_token(&logits);
        generated.push(next);
        let piece = tokenizer.decode(&[next], false).ok();
        println!("rung13: gen[{step}] token={next} piece={piece:?}");
        if eos == Some(next) {
            break;
        }
        let started = std::time::Instant::now();
        logits = engine
            .decode_step(next, pos)
            .unwrap_or_else(|e| panic!("generate pos={pos}: {e}"));
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "generate logits at pos={pos} not finite"
        );
        println!(
            "rung13: decode pos={pos} {:.0}ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        pos += 1;
    }

    let reply = tokenizer.decode(&generated, false).expect("decode reply");
    println!("rung13 reply: {reply:?}");
    assert!(!generated.is_empty(), "chat generate produced no tokens");
}

#[test]
#[ignore = "requires R9700 artifact CI"]
fn rung14_reset_refreshes_mutable_state_with_stable_descriptors() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, _config)) = build_engine(16) else {
        return;
    };

    let before: Vec<usize> = engine
        .state()
        .layers
        .iter()
        .filter_map(|layer| {
            layer
                .recurrent_state
                .as_ref()
                .map(|buf| buf.as_ptr() as usize)
        })
        .collect();
    assert!(!before.is_empty(), "expected linear recurrent state");
    let blockers: Vec<GpuBuffer> = (0..before.len())
        .map(|_| {
            GpuBuffer::zeros(0, ScalarType::F32, &[16, 128, 128])
                .expect("state-reallocation blocker")
        })
        .collect();

    engine.decode_step(9419, 0).expect("initial decode");
    engine.reset().expect("reset model state");
    let after: Vec<usize> = engine
        .state()
        .layers
        .iter()
        .filter_map(|layer| {
            layer
                .recurrent_state
                .as_ref()
                .map(|buf| buf.as_ptr() as usize)
        })
        .collect();
    assert_ne!(before, after, "reset should allocate fresh recurrent state");

    let logits = engine.decode_step(9419, 0).expect("decode after reset");
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(logits.iter().map(|value| value * value).sum::<f32>() > 0.0);
    drop(blockers);
}

#[test]
#[ignore = "requires R9700 artifact CI"]
fn rung15_hip_fast_greedy_matches_host_route_and_reports_device_transfer() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, _config)) = build_engine(64) else {
        return;
    };

    let model_dir = qwen38_model_dir().expect("model dir");
    let template = ChatTemplate::try_load(&model_dir)
        .expect("load chat template")
        .expect("Qwen3.8 ships a chat template");
    let prompt = template
        .render(&[ChatMessage::text("user", "Hello")], true)
        .expect("render prompt");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt_ids = tokenizer
        .encode(prompt, false)
        .expect("encode prompt")
        .get_ids()
        .to_vec();
    assert!(
        prompt_ids.len() > 8,
        "expected canonical multi-token prompt"
    );
    let prefill_logits = engine
        .prefill_native(&prompt_ids)
        .expect("canonical prefill");
    let token = greedy_token(&prefill_logits);
    let snapshot = engine
        .snapshot_prefix(prefill_logits.clone())
        .expect("snapshot canonical prefix");
    let (host_logits, host_timings) = engine
        .decode_step_4b_single_kernel_with_timings(token, prompt_ids.len())
        .expect("host-logit decode");
    let expected = greedy_token(&host_logits);
    assert!(
        host_timings.logits_d2h_ms > 0.0,
        "host-logit route must transfer the full BF16 vocabulary row"
    );

    let restored_logits = engine
        .restore_prefix(&snapshot)
        .expect("restore canonical prefix");
    assert_eq!(greedy_token(&restored_logits), token);
    let (fast_token, fast_timings) = engine
        .decode_step_hip_fast_greedy(token, prompt_ids.len())
        .expect("HIP fast greedy decode");
    println!(
        "rung15: expected={expected} fast={fast_token} host_logits_d2h_ms={:.3} fast_rms_norm_ms={:.3} fast_lm_head_ms={:.3} fast_gpu_argmax_ms={:.3} fast_token_d2h_ms={:.3}",
        host_timings.logits_d2h_ms,
        fast_timings.rms_norm_ms,
        fast_timings.lm_head_ms,
        fast_timings.gpu_argmax_ms,
        fast_timings.token_d2h_ms,
    );
    assert_eq!(
        fast_token, expected,
        "device argmax must match host sampling"
    );
    assert_eq!(fast_timings.logits_d2h_ms, 0.0);
    assert_eq!(fast_timings.host_sampling_ms, 0.0);
    assert!(
        fast_timings.gpu_argmax_ms > 0.0,
        "fast route must time the device argmax completion"
    );
    assert!(
        fast_timings.token_d2h_ms > 0.0,
        "fast route must transfer the U32 token"
    );
    assert!(
        fast_timings.lm_head_ms > fast_timings.token_d2h_ms,
        "fast route must charge lm-head completion to lm_head_ms, not token_d2h_ms"
    );
}
