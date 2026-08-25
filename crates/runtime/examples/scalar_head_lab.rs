use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use qwen38::scratch::required_attn_scratch_floats;
use qwen38::weights::Qwen38Weights;
use supersonic_runtime::chat_template::{ChatMessage, ChatTemplate};
use supersonic_runtime::decode_engine::{DecodeEngine, ScalarHeadLabRoute};

const ENGINE_NAME: &str = "supersonic-scalar-lab";
const ENGINE_VERSION: &str = "scalar-head-lab-v1";

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    artifact: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    device: usize,
    mode: String,
    chat: bool,
    ignore_eos: bool,
}

fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    let config = qwen38::config::load_config(&args.model_dir)
        .map_err(|e| anyhow::anyhow!("loading scalar lab Qwen3.8 config: {e}"))?
        .text_config;
    anyhow::ensure!(
        config.hidden_size == 5_120,
        "scalar lab requires hidden size 5120"
    );
    anyhow::ensure!(
        config.vocab_size == 248_320,
        "scalar lab requires vocabulary 248320"
    );
    let tokenizer = tokenizers::Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading scalar lab tokenizer: {e}"))?;
    let rendered = if args.chat {
        ChatTemplate::try_load(&args.model_dir)?
            .ok_or_else(|| anyhow::anyhow!("--chat requires a chat template"))?
            .render(&[ChatMessage::text("user", &args.prompt)], true)?
    } else {
        args.prompt.clone()
    };
    let prompt_ids = tokenizer
        .encode(rendered, !args.chat)
        .map_err(|e| anyhow::anyhow!("tokenizing scalar lab prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(
        !prompt_ids.is_empty(),
        "scalar lab prompt tokenized to empty input"
    );
    let context = prompt_ids.len() + args.max_new_tokens;
    let weights = Qwen38Weights::load_gguf(&args.artifact, &config, args.device)
        .context("loading scalar lab GQH artifact")?;
    let attn_scratch =
        required_attn_scratch_floats(config.num_attention_heads, config.head_dim, context, 256)
            .max(24_576);
    let mut engine = DecodeEngine::new(weights, args.device, 16_480, attn_scratch, 256, true, 0)?;
    engine.set_decode_context_limit(context);
    engine.set_scalar_head_lab_route(ScalarHeadLabRoute::RawQ6Scalar)?;
    if args.mode == "mtp" {
        anyhow::ensure!(
            engine.weights().mtp.is_some(),
            "scalar lab MTP requires NextN weights in the artifact"
        );
        engine.set_mtp_spec(true);
    }

    let prefill = engine.prefill_native_with_final_norm(&prompt_ids)?;
    let final_norm = prefill
        .final_norm_trace
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("scalar lab prefill omitted final norm"))?;
    let mut next_token = engine.scalar_head_lab_greedy_from_normed(final_norm)?;
    if args.mode == "mtp" {
        engine.seed_mtp_h_from_normed(final_norm)?;
    }
    engine.prepare_hip_gqh_decode()?;

    let eos_ids = eos_ids(&config, &tokenizer, args.chat);
    let started = Instant::now();
    let mut generated = Vec::with_capacity(args.max_new_tokens);
    if args.mode == "mtp" {
        let mut pos = prompt_ids.len();
        while generated.len() < args.max_new_tokens {
            if !args.ignore_eos && eos_ids.contains(&next_token) {
                break;
            }
            let remaining = args.max_new_tokens - generated.len();
            let round = engine.run_mtp_spec_round(next_token, pos, remaining)?;
            for token in round.emitted {
                generated.push(token);
                pos += 1;
                if generated.len() == args.max_new_tokens
                    || (!args.ignore_eos && eos_ids.contains(&token))
                {
                    break;
                }
            }
            next_token = round.next_token;
            if generated
                .last()
                .is_some_and(|token| !args.ignore_eos && eos_ids.contains(token))
            {
                break;
            }
        }
    } else {
        for step in 0..args.max_new_tokens {
            if !args.ignore_eos && eos_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);
            if generated.len() == args.max_new_tokens {
                break;
            }
            (next_token, _) =
                engine.decode_step_hip_fast_greedy(next_token, prompt_ids.len() + step)?;
        }
    }
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(!generated.is_empty(), "scalar lab generated no tokens");
    let generated_text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decoding scalar lab output: {e}"))?;
    let ms_per_tok = decode_ms / generated.len() as f64;
    let tokens_per_second = 1000.0 / ms_per_tok;
    let payload = serde_json::json!({
        "decode_ms": decode_ms,
        "engine_name": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "generated_text": generated_text,
        "generated_tokens": generated.len(),
        "ms_per_tok": ms_per_tok,
        "prompt_tokens": prompt_ids.len(),
        "token_ids": generated,
        "tokens_per_second": tokens_per_second,
    });
    println!("[supersonic_json] {}", serde_json::to_string(&payload)?);
    Ok(())
}

fn parse_args(values: impl IntoIterator<Item = String>) -> Result<Args> {
    let mut fields = BTreeMap::<String, String>::new();
    let mut chat = false;
    let mut ignore_eos = false;
    let mut values = values.into_iter();
    while let Some(arg) = values.next() {
        match arg.as_str() {
            "--chat" => chat = true,
            "--ignore-eos" => ignore_eos = true,
            "--model-dir" | "--artifact" | "--prompt" | "--max-new-tokens" | "--device"
            | "--mode" => {
                let value = values
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} requires a value"))?;
                anyhow::ensure!(
                    fields.insert(arg.clone(), value).is_none(),
                    "duplicate {arg}"
                );
            }
            _ => anyhow::bail!("unknown scalar lab argument: {arg}"),
        }
    }
    let required = |name: &str| {
        fields
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing required {name}"))
    };
    let max_new_tokens = required("--max-new-tokens")?.parse::<usize>()?;
    anyhow::ensure!(max_new_tokens > 0, "--max-new-tokens must be positive");
    let device = required("--device")?.parse::<usize>()?;
    let mode = required("--mode")?;
    anyhow::ensure!(
        mode == "ordinary" || mode == "mtp",
        "--mode must be ordinary or mtp"
    );
    let prompt = required("--prompt")?;
    anyhow::ensure!(!prompt.is_empty(), "--prompt must be non-empty");
    Ok(Args {
        model_dir: PathBuf::from(required("--model-dir")?),
        artifact: PathBuf::from(required("--artifact")?),
        prompt,
        max_new_tokens,
        device,
        mode,
        chat,
        ignore_eos,
    })
}

fn eos_ids(
    config: &qwen38::config::TextConfig,
    tokenizer: &tokenizers::Tokenizer,
    chat: bool,
) -> Vec<u32> {
    let mut ids = config.eos_token_ids();
    if chat {
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}
