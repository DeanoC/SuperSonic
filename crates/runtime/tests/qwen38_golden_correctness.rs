//! Qwen3.8 greedy golden trajectories on the pinned Q3KXL artifact.
//!
//! Goldens were first captured on Metal (`91bc7e33…`, see `docs/benchmarks.md`) and
//! must match HIP production decode on gfx1100/gfx1201 when the same artifact is used.
//!
//! ```bash
//! export SUPERSONIC_GQH_GGUF=/path/to/Qwen3.8-27B-GQH-Q3KXL.gguf
//! export SUPERSONIC_QWEN38_MODEL_DIR=/path/to/Qwen3.8-27B
//! cargo test -p supersonic-runtime --test qwen38_golden_correctness -- --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Mutex;

use gpu_hal::Backend;
use qwen38::gguf_ingest::load_text_config;
use qwen38::scratch::required_attn_scratch_floats;
use qwen38::weights::Qwen38Weights;
use supersonic_runtime::chat_template::{ChatMessage, ChatTemplate};
use supersonic_runtime::decode_engine::DecodeEngine;

static GPU: Mutex<()> = Mutex::new(());

const PLAIN_HELLO_GENERATED: [u32; 2] = [11, 353];
const CHAT_HELLO_GENERATED: [u32; 8] = [9419, 0, 2500, 628, 353, 1438, 488, 3242];
const CHAT_HELLO_PREFILL_TOKEN: u32 = 9419;

fn gguf_path() -> Option<PathBuf> {
    let value = std::env::var_os("SUPERSONIC_GQH_GGUF")?;
    let path = PathBuf::from(value);
    path.is_file().then_some(path)
}

fn qwen38_model_dir() -> Option<PathBuf> {
    let value = std::env::var_os("SUPERSONIC_QWEN38_MODEL_DIR")?;
    let path = PathBuf::from(value);
    if !path.is_dir() {
        return None;
    }
    for required in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        if !path.join(required).is_file() {
            return None;
        }
    }
    Some(path)
}

fn build_engine(max_context: usize) -> Option<(DecodeEngine, qwen38::config::TextConfig)> {
    let path = gguf_path()?;
    let model_dir = qwen38_model_dir()?;
    gpu_hal::set_device(0).ok()?;
    let ordinal = 0usize;
    let config = load_text_config(&model_dir).ok()?;
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).ok()?;
    let attn_scratch = required_attn_scratch_floats(
        config.num_attention_heads,
        config.head_dim,
        max_context,
        256,
    )
    .max(24_576);
    let engine = DecodeEngine::new(weights, ordinal, 16_480, attn_scratch, 256, true, 0).ok()?;
    Some((engine, config))
}

fn prepare_decode_after_prefill(engine: &mut DecodeEngine) {
    if engine.backend() == Backend::Hip {
        engine
            .prepare_hip_gqh_decode()
            .expect("prepare_hip_gqh_decode");
    }
}

fn encode_plain(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .expect("encode plain prompt")
        .get_ids()
        .to_vec()
}

fn encode_chat(model_dir: &PathBuf, text: &str) -> Vec<u32> {
    let template = ChatTemplate::try_load(model_dir)
        .expect("load chat template")
        .expect("Qwen3.8 ships a chat template");
    let prompt = template
        .render(&[ChatMessage::text("user", text)], true)
        .expect("render chat");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    tokenizer
        .encode(prompt, false)
        .expect("encode chat prompt")
        .get_ids()
        .to_vec()
}

fn generate_greedy(engine: &mut DecodeEngine, prompt_ids: &[u32], count: usize) -> Vec<u32> {
    let logits = engine
        .prefill_native(prompt_ids)
        .expect("prefill_native");
    assert_eq!(logits.len(), engine.weights().config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "prefill logits not finite");
    let first = DecodeEngine::greedy_sample(&logits);
    prepare_decode_after_prefill(engine);
    continue_greedy(engine, prompt_ids.len(), first, count)
}

fn continue_greedy(
    engine: &mut DecodeEngine,
    prompt_len: usize,
    first_token: u32,
    count: usize,
) -> Vec<u32> {
    let mut next = first_token;
    let mut pos = prompt_len;
    let mut generated = Vec::with_capacity(count);
    for _ in 0..count {
        generated.push(next);
        let (sampled, _) = engine
            .decode_step_greedy(next, pos)
            .expect("decode_step_greedy");
        next = sampled;
        pos += 1;
    }
    generated
}

#[test]
fn qwen38_golden_correctness_suite() {
    let _guard = GPU.lock().expect("gpu lock");
    let Some(model_dir) = qwen38_model_dir() else {
        eprintln!("skip: SUPERSONIC_GQH_GGUF / SUPERSONIC_QWEN38_MODEL_DIR not configured");
        return;
    };
    let backend = gpu_hal::current_backend();
    eprintln!("qwen38_golden_correctness backend={backend:?}");

    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    let chat_prompt_ids = encode_chat(&model_dir, "Hello");
    assert_eq!(chat_prompt_ids.len(), 13, "chat Hello prompt length");
    let plain_prompt_ids = encode_plain(&tokenizer, "Hello");
    assert_eq!(plain_prompt_ids.len(), 1, "plain Hello prompt length");

    let max_context = chat_prompt_ids.len() + CHAT_HELLO_GENERATED.len();

    {
        let Some((mut engine, _)) = build_engine(max_context) else {
            panic!("failed to build engine for plain Hello");
        };
        let generated = generate_greedy(&mut engine, &plain_prompt_ids, PLAIN_HELLO_GENERATED.len());
        assert_eq!(
            generated.as_slice(),
            PLAIN_HELLO_GENERATED,
            "plain Hello greedy tokens"
        );
    }

    {
        let Some((mut engine, _)) = build_engine(max_context) else {
            panic!("failed to build engine for chat Hello");
        };
        let prefill_logits = engine
            .prefill_native(&chat_prompt_ids)
            .expect("chat prefill");
        let prefill_token = DecodeEngine::greedy_sample(&prefill_logits);
        assert_eq!(
            prefill_token, CHAT_HELLO_PREFILL_TOKEN,
            "chat Hello prefill token"
        );
        prepare_decode_after_prefill(&mut engine);
        let generated = continue_greedy(
            &mut engine,
            chat_prompt_ids.len(),
            prefill_token,
            CHAT_HELLO_GENERATED.len(),
        );
        assert_eq!(
            generated.as_slice(),
            CHAT_HELLO_GENERATED,
            "chat Hello greedy tokens"
        );
    }

    {
        let Some((mut engine, _)) = build_engine(max_context) else {
            panic!("failed to build engine for repeatability");
        };
        let first = generate_greedy(&mut engine, &chat_prompt_ids, CHAT_HELLO_GENERATED.len());
        assert_eq!(first.as_slice(), CHAT_HELLO_GENERATED);
        if backend == Backend::Hip {
            // HIP GQH graph capture survives `reset()` today; rebuilding the engine
            // for a second trajectory exceeds the VRAM budget of this suite.
            eprintln!("skip: HIP chat Hello repeatability on a reset engine");
        } else {
            engine.reset().expect("reset before repeatability rerun");
            let second = generate_greedy(&mut engine, &chat_prompt_ids, CHAT_HELLO_GENERATED.len());
            assert_eq!(first, second, "chat Hello not repeatable on {backend:?}");
        }
    }
}
