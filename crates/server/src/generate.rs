//! Core generation loop shared by `/v1/chat/completions` and
//! `/v1/completions`. Runs on a `spawn_blocking` thread (engines are
//! synchronous HIP calls), and publishes token-level events on an
//! unbounded channel so the HTTP layer can either collect them into one
//! response or stream them as SSE.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::sampling::{rng_from_seed, sample};
use crate::session::InferenceSession;
use crate::state::ServerState;

/// What a caller can tune per request. Missing fields fall back to
/// permissive defaults so clients that only supply `messages` still work.
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 256,
            stop: Vec::new(),
            seed: None,
        }
    }
}

pub enum GenEvent {
    /// A chunk of output text. May be empty if the decoded token produced
    /// no new characters (e.g. a BPE continuation piece that the tokenizer
    /// folds into the next step).
    Token(String),
    /// Terminal event: generation ended with this reason.
    Done { reason: FinishReason, prompt_tokens: u32, completion_tokens: u32 },
    /// Terminal error event: generation failed; no more events will arrive.
    Error(String),
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Start generation. Returns the receiver side of the event channel — the
/// caller drains it either eagerly (non-stream) or as an SSE stream.
pub fn spawn(
    state: Arc<ServerState>,
    prompt_text: String,
    add_special_tokens: bool,
    params: GenParams,
) -> UnboundedReceiver<GenEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run(state, prompt_text, add_special_tokens, params, tx.clone()) {
            let _ = tx.send(GenEvent::Error(e.to_string()));
        }
    });
    rx
}

fn run(
    state: Arc<ServerState>,
    prompt_text: String,
    add_special_tokens: bool,
    params: GenParams,
    tx: UnboundedSender<GenEvent>,
) -> Result<()> {
    let tokenizer = state.tokenizer.clone();
    let encoding = tokenizer
        .encode(prompt_text.as_str(), add_special_tokens)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(anyhow!("empty prompt after tokenization"));
    }
    let prompt_tokens = prompt_ids.len() as u32;

    let max_ctx = state.max_context;
    if prompt_ids.len() + params.max_tokens > max_ctx {
        return Err(anyhow!(
            "prompt ({} tokens) + max_tokens ({}) exceeds max_context ({})",
            prompt_ids.len(),
            params.max_tokens,
            max_ctx
        ));
    }

    let mut guard = state.session.blocking_lock();
    guard.reset()?;
    let prefill_logits = guard.prefill(&prompt_ids)?;

    let mut rng = rng_from_seed(params.seed);

    // Sample the first new token from prefill's final logits.
    let mut logits = prefill_logits;
    let mut next_token = sample(&mut logits, params.temperature, params.top_p, &mut rng);

    let mut emitted_ids: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let mut prev_decoded = String::new();
    let mut completion_tokens: u32 = 0;

    let finish = loop {
        if state.eos_ids.contains(&next_token) {
            break FinishReason::Stop;
        }

        emitted_ids.push(next_token);
        completion_tokens += 1;

        // Incremental detokenization: decode the full output then diff the
        // tail. Works around BPE tokens that only produce a character when
        // combined with following tokens.
        let decoded = detokenize(&tokenizer, &emitted_ids);
        let delta = decoded
            .strip_prefix(prev_decoded.as_str())
            .unwrap_or(&decoded[prev_decoded.len().min(decoded.len())..])
            .to_string();
        prev_decoded = decoded;

        if !delta.is_empty() && tx.send(GenEvent::Token(delta)).is_err() {
            // Receiver dropped — client disconnected. Bail out.
            break FinishReason::Stop;
        }

        if matches_stop(&prev_decoded, &params.stop) {
            break FinishReason::Stop;
        }

        if completion_tokens as usize >= params.max_tokens {
            break FinishReason::Length;
        }

        let pos = prompt_ids.len() + emitted_ids.len() - 1;
        let mut next_logits = guard.decode_step(next_token, pos)?;
        next_token = sample(
            &mut next_logits,
            params.temperature,
            params.top_p,
            &mut rng,
        );
    };

    let _ = tx.send(GenEvent::Done {
        reason: finish,
        prompt_tokens,
        completion_tokens,
    });
    Ok(())
}

fn detokenize(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    tokenizer.decode(ids, true).unwrap_or_default()
}

fn matches_stop(text: &str, stops: &[String]) -> bool {
    stops.iter().any(|s| !s.is_empty() && text.contains(s))
}

/// Drain the full event stream and return the concatenated text plus the
/// terminating event. Used by non-streaming responses.
pub async fn collect(mut rx: UnboundedReceiver<GenEvent>) -> Result<CollectedResult> {
    let mut text = String::new();
    let mut finish = FinishReason::Stop;
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(s) => text.push_str(&s),
            GenEvent::Done { reason, prompt_tokens: p, completion_tokens: c } => {
                finish = reason;
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            GenEvent::Error(msg) => return Err(anyhow!(msg)),
        }
    }
    Ok(CollectedResult {
        text,
        finish,
        prompt_tokens,
        completion_tokens,
    })
}

pub struct CollectedResult {
    pub text: String,
    pub finish: FinishReason,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Re-export kept so downstream route modules can name the session type.
pub type Session = InferenceSession;
