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
    Done {
        reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
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

/// Tokenize + bounds-check a request synchronously. The route handlers
/// call this before committing to an SSE response, so setup failures
/// (empty prompt, context overflow) surface as real HTTP errors instead
/// of in-band SSE error events under a misleading 200.
pub fn prepare(
    state: &ServerState,
    prompt_text: &str,
    add_special_tokens: bool,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    let encoding = state
        .tokenizer
        .encode(prompt_text, add_special_tokens)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(anyhow!("empty prompt after tokenization"));
    }
    // `saturating_add` so a pathological `max_tokens` near `usize::MAX` is
    // rejected here rather than overflowing and bypassing the bound.
    let total_ctx = prompt_ids.len().saturating_add(max_tokens);
    if total_ctx > state.max_context {
        return Err(anyhow!(
            "prompt ({} tokens) + max_tokens ({}) exceeds max_context ({})",
            prompt_ids.len(),
            max_tokens,
            state.max_context
        ));
    }
    Ok(prompt_ids)
}

/// Start generation from pre-validated token IDs. Returns the receiver
/// side of the event channel — the caller drains it either eagerly
/// (non-stream) or as an SSE stream. Call [`prepare`] first and bail out
/// of the handler with an HTTP error if it fails.
pub fn spawn(
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
) -> UnboundedReceiver<GenEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run(state, prompt_ids, params, tx.clone()) {
            let _ = tx.send(GenEvent::Error(e.to_string()));
        }
    });
    rx
}

fn run(
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    tx: UnboundedSender<GenEvent>,
) -> Result<()> {
    let tokenizer = state.tokenizer.clone();
    let prompt_tokens = prompt_ids.len() as u32;

    // Zero-token request: return an empty completion without touching the
    // engine. OpenAI semantics: `max_tokens=0` means no completion tokens.
    if params.max_tokens == 0 {
        let _ = tx.send(GenEvent::Done {
            reason: FinishReason::Length,
            prompt_tokens,
            completion_tokens: 0,
        });
        return Ok(());
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
        // Budget check first — prevents emitting a token when the caller
        // asked for `max_tokens == N` and we've already produced N.
        if completion_tokens as usize >= params.max_tokens {
            break FinishReason::Length;
        }

        if state.eos_ids.contains(&next_token) {
            break FinishReason::Stop;
        }

        emitted_ids.push(next_token);
        completion_tokens += 1;

        // Incremental detokenization: decode the full output then diff the
        // tail. Works around BPE tokens that only produce a character when
        // combined with following tokens.
        let decoded = detokenize(&tokenizer, &emitted_ids);

        // Stop-string detection must happen *before* we emit the delta —
        // otherwise the client sees the stop sequence (and any text that
        // followed it inside the same merged delta) even though we'll
        // report `finish_reason=stop`. Trim the delta at the first stop
        // occurrence in the cumulative output, emit the trimmed portion,
        // and break.
        if let Some(stop_at) = find_earliest_stop(&decoded, &params.stop) {
            let trimmed = &decoded[..stop_at];
            let delta = incremental_delta(&prev_decoded, trimmed);
            if !delta.is_empty() {
                let _ = tx.send(GenEvent::Token(delta));
            }
            break FinishReason::Stop;
        }

        let delta = incremental_delta(&prev_decoded, &decoded);
        prev_decoded = decoded;

        if !delta.is_empty() && tx.send(GenEvent::Token(delta)).is_err() {
            // Receiver dropped — client disconnected. Bail out.
            break FinishReason::Stop;
        }

        let mut next_logits = if guard.requires_replay_decode() {
            // Metal v1: re-run prefill over the full history each step.
            let history: Vec<u32> = prompt_ids
                .iter()
                .copied()
                .chain(emitted_ids.iter().copied())
                .chain(std::iter::once(next_token))
                .collect();
            guard.decode_step_replay(&history)?
        } else {
            let pos = prompt_ids.len() + emitted_ids.len() - 1;
            guard.decode_step(next_token, pos)?
        };
        next_token = sample(&mut next_logits, params.temperature, params.top_p, &mut rng);
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

/// Produce the new output text that should be emitted as a delta, given
/// the previously-emitted cumulative text `prev` and the latest
/// cumulative decode `now`. Always slices at UTF-8 char boundaries, even
/// when `prev` is not a strict byte prefix of `now` (which can happen if
/// the tokenizer renormalizes across steps — a multi-byte codepoint
/// composing across a token boundary, for instance).
fn incremental_delta(prev: &str, now: &str) -> String {
    if let Some(rest) = now.strip_prefix(prev) {
        return rest.to_string();
    }
    // Walk aligned codepoints until `prev` and `now` diverge; slice at
    // the last matching char boundary. Safe against non-prefix cases.
    let mut common_bytes = 0usize;
    let mut prev_chars = prev.chars();
    for (idx, ch) in now.char_indices() {
        match prev_chars.next() {
            Some(pc) if pc == ch => common_bytes = idx + ch.len_utf8(),
            _ => break,
        }
    }
    now[common_bytes..].to_string()
}

/// Return the lowest byte offset at which any non-empty stop string first
/// occurs in `text`, or `None` if none match. Note that BPE tokens can
/// straddle a stop string — e.g. stop="Hello" with tokens ["Hel","lo"]
/// produces a delta of "Hel" that cannot be retracted after the fact.
/// This function only detects stops that fall entirely inside the
/// cumulative decoded output; streaming callers will still see at most a
/// single-token overshoot in that straddling case.
fn find_earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// Drain the full event stream and return the concatenated text plus the
/// terminating event. Used by non-streaming responses.
///
/// A channel that closes without a terminal `Done` or `Error` is treated
/// as an error — otherwise a panic inside the `spawn_blocking` task would
/// silently produce a 200 response with empty content.
pub async fn collect(mut rx: UnboundedReceiver<GenEvent>) -> Result<CollectedResult> {
    let mut text = String::new();
    let mut finish: Option<FinishReason> = None;
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(s) => text.push_str(&s),
            GenEvent::Done {
                reason,
                prompt_tokens: p,
                completion_tokens: c,
            } => {
                finish = Some(reason);
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            GenEvent::Error(msg) => return Err(anyhow!(msg)),
        }
    }
    let finish = finish.ok_or_else(|| {
        anyhow!("generation task ended without a terminal event (likely panicked)")
    })?;
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

#[cfg(test)]
mod tests {
    use super::incremental_delta;

    #[test]
    fn prefix_case_returns_suffix() {
        assert_eq!(incremental_delta("Hello", "Hello, world"), ", world");
    }

    #[test]
    fn identical_returns_empty() {
        assert_eq!(incremental_delta("abc", "abc"), "");
    }

    #[test]
    fn prev_longer_returns_empty() {
        // Renormalization shortened the cumulative decode — safest delta is
        // empty (we can't retract already-emitted text).
        assert_eq!(incremental_delta("Hello!", "Hello"), "");
    }

    #[test]
    fn multibyte_divergence_slices_on_char_boundary() {
        // `prev` ends inside a multi-byte codepoint of `now`: naïve byte
        // slicing would panic. Must slice at the codepoint boundary.
        let prev = "caf";
        let now = "café world";
        let out = incremental_delta(prev, now);
        assert_eq!(out, "é world");
    }

    #[test]
    fn non_prefix_multibyte_walks_to_common_boundary() {
        // The tokenizer renormalized the last character. `prev` is not a
        // strict byte prefix. Fallback must still slice on a boundary.
        let prev = "naïve";
        let now = "naïvely";
        let out = incremental_delta(prev, now);
        assert_eq!(out, "ly");
    }

    #[test]
    fn fully_divergent_emits_full_now() {
        let prev = "foo";
        let now = "bar";
        assert_eq!(incremental_delta(prev, now), "bar");
    }
}
