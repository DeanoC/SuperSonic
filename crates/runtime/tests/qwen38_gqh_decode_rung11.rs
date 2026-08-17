//! Rungs 11–13: component decode and a chat-templated Hello generate.

use std::path::PathBuf;
use std::sync::Mutex;

use gpu_hal::{set_backend, Backend};
use qwen35::gguf_ingest::load_text_config;
use qwen35::scratch::required_attn_scratch_floats;
use qwen35::weights::Qwen35Weights;
use supersonic_runtime::chat_template::{ChatMessage, ChatTemplate};
use supersonic_runtime::decode_engine::DecodeEngine;

static GPU: Mutex<()> = Mutex::new(());

fn gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf");
    path.is_file().then_some(path)
}

fn hf_dir() -> PathBuf {
    PathBuf::from("/data/models/Qwen3.8-27B")
}

fn greedy_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .expect("logits")
}

fn build_engine(max_context: usize) -> Option<(DecodeEngine, qwen35::config::TextConfig)> {
    let path = gguf_path()?;
    if !hf_dir().join("config.json").is_file() {
        return None;
    }
    set_backend(Backend::Hip);
    if kernel_ffi::query_gpu_info(0).is_err() {
        eprintln!("skip: no HIP device 0");
        return None;
    }
    let ordinal = 0usize;
    let config = load_text_config(&hf_dir()).expect("hf config");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let attn_scratch = required_attn_scratch_floats(
        config.num_attention_heads,
        config.head_dim,
        max_context,
        256,
    )
    .max(24_576);
    let engine = DecodeEngine::new(
        weights,
        ordinal,
        16_480,
        attn_scratch,
        256,
        true,
        0,
        false,
        1,
    )
    .expect("DecodeEngine");
    Some((engine, config))
}

#[test]
fn rung11_one_token_component_decode() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, config)) = build_engine(16) else {
        return;
    };

    let token = 9419u32; // "Hello" in the Qwen3.8 tokenizer
    let started = std::time::Instant::now();
    let logits = engine
        .decode_step(token, 0)
        .expect("decode_step");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits contain a non-finite value"
    );
    let energy: f32 = logits.iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "logits are all zeros");

    let mut ranked: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let argmax = ranked[0].0 as u32;
    let top5: Vec<(u32, f32)> = ranked
        .iter()
        .take(5)
        .map(|(i, v)| (*i as u32, *v))
        .collect();

    let decoded = tokenizers::Tokenizer::from_file(hf_dir().join("tokenizer.json"))
        .ok()
        .and_then(|tok| tok.decode(&[argmax], false).ok());
    println!(
        "rung11: token={token} argmax={argmax} piece={decoded:?} top5={top5:?} energy={energy:.4} {elapsed_ms:.0}ms"
    );

    let tokenizer = tokenizers::Tokenizer::from_file(hf_dir().join("tokenizer.json")).ok();
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
fn rung13_chat_hello_generate() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some((mut engine, config)) = build_engine(64) else {
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(hf_dir().join("tokenizer.json"))
        .expect("tokenizer.json");
    let template = ChatTemplate::try_load(&hf_dir())
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

    let eos = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

    let prefill_start = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &id) in prompt_ids.iter().enumerate() {
        logits = engine
            .decode_step(id, pos)
            .unwrap_or_else(|e| panic!("prefill pos={pos} token={id}: {e}"));
        assert_eq!(logits.len(), config.vocab_size);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "prefill logits at pos={pos} not finite"
        );
    }
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
    assert!(
        !generated.is_empty(),
        "chat generate produced no tokens"
    );
}
