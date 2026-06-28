//! Core generation loop shared by `/v1/chat/completions` and
//! `/v1/completions`. Runs on a `spawn_blocking` thread (engines are
//! synchronous HIP calls), and publishes token-level events on an
//! unbounded channel so the HTTP layer can either collect them into one
//! response or stream them as SSE.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::{OwnedSemaphorePermit, TryAcquireError};

use crate::prefix_cache::CacheRequest;
use crate::sampling::{rng_from_seed, sample};
use crate::session::InferenceSession;
use crate::state::ServerState;

const CACHE_ANCHOR_SUFFIX_TOKENS: usize = 16;

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
        cached_prompt_tokens: u32,
    },
    /// Terminal error event: generation failed; no more events will arrive.
    Error(String),
}

#[derive(Debug, Clone)]
pub struct SchedulerSnapshot {
    pub active: usize,
    pub queued: usize,
    pub max_queue: usize,
    pub queue_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct MockGeneration {
    pub chunks: Vec<String>,
    pub finish: FinishReason,
    pub delay_ms: u64,
}

impl MockGeneration {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            chunks: vec![text.into()],
            finish: FinishReason::Stop,
            delay_ms: 0,
        }
    }
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
    prepare_ids(state, prompt_ids, max_tokens)
}

/// Bounds-check pre-tokenized prompt IDs. This is used by compatibility
/// routes that accept OpenAI-style token-array prompts.
pub fn prepare_ids(
    state: &ServerState,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        return Err(anyhow!("empty prompt"));
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
    cache: Option<CacheRequest>,
) -> Result<UnboundedReceiver<GenEvent>> {
    let (tx, rx) = mpsc::unbounded_channel();

    let scheduler = state.scheduler.clone();
    match scheduler.permits.clone().try_acquire_owned() {
        Ok(permit) => {
            tokio::spawn(run_with_permit(
                scheduler, state, prompt_ids, params, cache, tx, permit,
            ));
            return Ok(rx);
        }
        Err(TryAcquireError::Closed) => {
            return Err(anyhow!("generation scheduler closed"));
        }
        Err(TryAcquireError::NoPermits) => {}
    }

    let queued = scheduler.queued.fetch_add(1, Ordering::SeqCst) + 1;
    if queued > scheduler.max_queue {
        scheduler.queued.fetch_sub(1, Ordering::SeqCst);
        return Err(anyhow!(
            "generation queue full (queued={}, max_queue={})",
            queued - 1,
            scheduler.max_queue
        ));
    }

    tokio::spawn(async move {
        let acquire = scheduler.permits.clone().acquire_owned();
        let timeout = tokio::time::sleep(Duration::from_millis(scheduler.queue_timeout_ms));
        tokio::pin!(timeout);
        let permit = tokio::select! {
            permit = acquire => match permit {
                Ok(permit) => permit,
                Err(_) => {
                    scheduler.queued.fetch_sub(1, Ordering::SeqCst);
                    let _ = tx.send(GenEvent::Error("generation scheduler closed".to_string()));
                    return;
                }
            },
            _ = &mut timeout => {
                scheduler.queued.fetch_sub(1, Ordering::SeqCst);
                let _ = tx.send(GenEvent::Error("generation queue timeout".to_string()));
                return;
            },
            _ = tx.closed() => {
                scheduler.queued.fetch_sub(1, Ordering::SeqCst);
                return;
            }
        };
        scheduler.queued.fetch_sub(1, Ordering::SeqCst);
        run_with_permit(scheduler, state, prompt_ids, params, cache, tx, permit).await;
    });
    Ok(rx)
}

async fn run_with_permit(
    scheduler: Arc<crate::state::GenerationScheduler>,
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    cache: Option<CacheRequest>,
    tx: UnboundedSender<GenEvent>,
    permit: OwnedSemaphorePermit,
) {
    if tx.is_closed() {
        drop(permit);
        return;
    }
    scheduler.active.fetch_add(1, Ordering::SeqCst);

    if let Some(mock) = state.mock_generation.clone() {
        run_mock(mock, &prompt_ids, &params, &tx).await;
        scheduler.active.fetch_sub(1, Ordering::SeqCst);
        drop(permit);
        return;
    }

    let scheduler_done = scheduler.clone();
    let join = tokio::task::spawn_blocking(move || {
        let result = run(state, prompt_ids, params, cache, tx.clone());
        if let Err(e) = result {
            let _ = tx.send(GenEvent::Error(e.to_string()));
        }
    })
    .await;
    if let Err(e) = join {
        tracing::error!("generation worker join error: {e}");
    }
    scheduler_done.active.fetch_sub(1, Ordering::SeqCst);
    drop(permit);
}

pub fn scheduler_snapshot(state: &ServerState) -> SchedulerSnapshot {
    SchedulerSnapshot {
        active: state.scheduler.active.load(Ordering::SeqCst),
        queued: state.scheduler.queued.load(Ordering::SeqCst),
        max_queue: state.scheduler.max_queue,
        queue_timeout_ms: state.scheduler.queue_timeout_ms,
    }
}

async fn run_mock(
    mock: MockGeneration,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &UnboundedSender<GenEvent>,
) {
    for chunk in &mock.chunks {
        if tx.is_closed() || tx.send(GenEvent::Token(chunk.clone())).is_err() {
            return;
        }
        if mock.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(mock.delay_ms)).await;
        }
        tokio::task::yield_now().await;
    }
    let completion_tokens = mock.chunks.iter().filter(|s| !s.is_empty()).count() as u32;
    let _ = tx.send(GenEvent::Done {
        reason: if params.max_tokens == 0 {
            FinishReason::Length
        } else {
            mock.finish
        },
        prompt_tokens: prompt_ids.len() as u32,
        completion_tokens: if params.max_tokens == 0 {
            0
        } else {
            completion_tokens
        },
        cached_prompt_tokens: 0,
    });
}

fn run(
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    cache: Option<CacheRequest>,
    tx: UnboundedSender<GenEvent>,
) -> Result<()> {
    let tokenizer = state.tokenizer.clone();
    let prompt_tokens = prompt_ids.len() as u32;
    let mut cached_prompt_tokens = 0u32;

    // Zero-token request: return an empty completion without touching the
    // engine. OpenAI semantics: `max_tokens=0` means no completion tokens.
    if params.max_tokens == 0 {
        let _ = tx.send(GenEvent::Done {
            reason: FinishReason::Length,
            prompt_tokens,
            completion_tokens: 0,
            cached_prompt_tokens: 0,
        });
        return Ok(());
    }

    let session = state
        .session
        .as_ref()
        .ok_or_else(|| anyhow!("no inference session configured"))?;
    let mut guard = session.blocking_lock();
    if guard.is_dflash() {
        if cache.is_some() {
            tracing::debug!("prefix cache skipped for DFlash session");
        }
        let output =
            guard.generate_dflash_greedy(&prompt_ids, params.max_tokens, &state.eos_ids)?;
        tracing::info!(
            rounds = output.rounds_run,
            accepted = output.accepted_total,
            generated = output.token_ids.len(),
            decode_ms = output.decode_ms,
            "DFlash generation complete"
        );
        emit_generated_ids(
            &tokenizer,
            &state.eos_ids,
            prompt_tokens,
            &output.token_ids,
            &params,
            &tx,
        );
        return Ok(());
    }

    let prefill_logits = if let Some(cache_req) = cache.as_ref() {
        if let Some(hit) = state.prefix_cache.lookup(cache_req, &prompt_ids) {
            match guard.restore_prefix(hit.snapshot) {
                Ok(mut logits) => {
                    cached_prompt_tokens = hit.cached_tokens as u32;
                    for (idx, token) in prompt_ids
                        .iter()
                        .copied()
                        .enumerate()
                        .skip(hit.cached_tokens)
                    {
                        logits = guard.decode_step(token, idx)?;
                    }
                    logits
                }
                Err(e) => {
                    tracing::warn!("prefix cache restore failed: {e}");
                    state.prefix_cache.record_restore_failure();
                    guard.reset()?;
                    guard.prefill(&prompt_ids)?
                }
            }
        } else if let Some(hit) = state.prefix_cache.lookup_disk_bytes(cache_req, &prompt_ids) {
            match guard.load_disk_prefix(&hit.bytes) {
                Ok(snapshot) => match guard.restore_prefix(snapshot) {
                    Ok(mut logits) => {
                        cached_prompt_tokens = hit.cached_tokens as u32;
                        for (idx, token) in prompt_ids
                            .iter()
                            .copied()
                            .enumerate()
                            .skip(hit.cached_tokens)
                        {
                            logits = guard.decode_step(token, idx)?;
                        }
                        logits
                    }
                    Err(e) => {
                        tracing::warn!("prefix cache disk restore failed: {e}");
                        state.prefix_cache.record_restore_failure();
                        guard.reset()?;
                        prefill_with_cache_anchor(&mut guard, &state, cache_req, &prompt_ids)?
                    }
                },
                Err(e) => {
                    tracing::warn!("prefix cache disk load failed: {e}");
                    state.prefix_cache.record_restore_failure();
                    guard.reset()?;
                    prefill_with_cache_anchor(&mut guard, &state, cache_req, &prompt_ids)?
                }
            }
        } else {
            guard.reset()?;
            prefill_with_cache_anchor(&mut guard, &state, cache_req, &prompt_ids)?
        }
    } else {
        guard.reset()?;
        guard.prefill(&prompt_ids)?
    };

    if let Some(cache_req) = cache.as_ref() {
        snapshot_prefix_if_admitted(&guard, &state, cache_req, &prompt_ids, &prefill_logits);
    }

    let mut rng = rng_from_seed(params.seed);

    // Sample the first new token from prefill's final logits. Keep an
    // unmodified logits row for prefix-cache snapshots; `sample` mutates
    // its input for non-greedy settings.
    let mut current_logits = prefill_logits;
    let mut sample_logits = current_logits.clone();
    let mut next_token = sample(
        &mut sample_logits,
        params.temperature,
        params.top_p,
        &mut rng,
    );

    let mut emitted_ids: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let mut state_token_ids = prompt_ids.clone();
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

        if finish_after_emitted_token(completion_tokens, params.max_tokens).is_some() {
            break FinishReason::Length;
        }

        let pos = prompt_ids.len() + emitted_ids.len() - 1;
        let raw_next_logits = guard.decode_step(next_token, pos)?;
        state_token_ids.push(next_token);
        current_logits = raw_next_logits;
        let mut sample_logits = current_logits.clone();
        next_token = sample(
            &mut sample_logits,
            params.temperature,
            params.top_p,
            &mut rng,
        );
    };

    if state_token_ids.len() > prompt_ids.len() {
        if let Some(cache_req) = cache.as_ref() {
            snapshot_prefix_if_admitted(
                &guard,
                &state,
                cache_req,
                &state_token_ids,
                &current_logits,
            );
        }
    }

    let _ = tx.send(GenEvent::Done {
        reason: finish,
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens,
    });
    Ok(())
}

fn emit_generated_ids(
    tokenizer: &Tokenizer,
    eos_ids: &[u32],
    prompt_tokens: u32,
    token_ids: &[u32],
    params: &GenParams,
    tx: &UnboundedSender<GenEvent>,
) {
    let mut emitted_ids: Vec<u32> = Vec::with_capacity(token_ids.len());
    let mut prev_decoded = String::new();
    let mut completion_tokens = 0u32;
    let mut finish = if token_ids.len() >= params.max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };

    for &token_id in token_ids {
        if eos_ids.contains(&token_id) {
            finish = FinishReason::Stop;
            break;
        }
        if completion_tokens as usize >= params.max_tokens {
            finish = FinishReason::Length;
            break;
        }
        emitted_ids.push(token_id);
        completion_tokens += 1;

        let decoded = detokenize(tokenizer, &emitted_ids);
        if let Some(stop_at) = find_earliest_stop(&decoded, &params.stop) {
            let trimmed = &decoded[..stop_at];
            let delta = incremental_delta(&prev_decoded, trimmed);
            if !delta.is_empty() {
                let _ = tx.send(GenEvent::Token(delta));
            }
            finish = FinishReason::Stop;
            break;
        }

        let delta = incremental_delta(&prev_decoded, &decoded);
        prev_decoded = decoded;
        if !delta.is_empty() && tx.send(GenEvent::Token(delta)).is_err() {
            finish = FinishReason::Stop;
            break;
        }
    }

    let _ = tx.send(GenEvent::Done {
        reason: finish,
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens: 0,
    });
}

fn prefill_with_cache_anchor(
    guard: &mut InferenceSession,
    state: &ServerState,
    cache_req: &CacheRequest,
    prompt_ids: &[u32],
) -> Result<Vec<f32>> {
    let min_tokens = state.prefix_cache.config().min_tokens;
    let Some(anchor_len) = prompt_ids.len().checked_sub(CACHE_ANCHOR_SUFFIX_TOKENS) else {
        return guard.prefill(prompt_ids);
    };
    if anchor_len < min_tokens || anchor_len == 0 {
        return guard.prefill(prompt_ids);
    }

    let mut logits = guard.prefill(&prompt_ids[..anchor_len])?;
    snapshot_prefix_if_admitted(guard, state, cache_req, &prompt_ids[..anchor_len], &logits);
    for (idx, token) in prompt_ids.iter().copied().enumerate().skip(anchor_len) {
        logits = guard.decode_step(token, idx)?;
    }
    Ok(logits)
}

fn snapshot_prefix_if_admitted(
    guard: &InferenceSession,
    state: &ServerState,
    cache_req: &CacheRequest,
    token_ids: &[u32],
    logits: &[f32],
) {
    let estimate = guard.prefix_snapshot_bytes(logits.len());
    if !state.prefix_cache.can_admit(token_ids.len(), estimate) {
        state.prefix_cache.record_admission_skip();
        return;
    }
    match guard.snapshot_prefix(logits.to_vec()) {
        Ok(snapshot) => {
            if let Err(e) = state.prefix_cache.insert(cache_req, token_ids, snapshot) {
                tracing::warn!("prefix cache insert failed: {e}");
            }
        }
        Err(e) => tracing::debug!("prefix cache snapshot skipped: {e}"),
    }
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

fn finish_after_emitted_token(completion_tokens: u32, max_tokens: usize) -> Option<FinishReason> {
    if completion_tokens as usize >= max_tokens {
        Some(FinishReason::Length)
    } else {
        None
    }
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
    let mut cached_prompt_tokens = 0;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(s) => text.push_str(&s),
            GenEvent::Done {
                reason,
                prompt_tokens: p,
                completion_tokens: c,
                cached_prompt_tokens: cached,
            } => {
                finish = Some(reason);
                prompt_tokens = p;
                completion_tokens = c;
                cached_prompt_tokens = cached;
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
        cached_prompt_tokens,
    })
}

pub struct CollectedResult {
    pub text: String,
    pub finish: FinishReason,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_prompt_tokens: u32,
}

/// Re-export kept so downstream route modules can name the session type.
pub type Session = InferenceSession;

#[cfg(test)]
mod tests {
    use super::{finish_after_emitted_token, incremental_delta, FinishReason};

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

    #[test]
    fn final_budgeted_token_finishes_before_next_decode_step() {
        assert!(matches!(
            finish_after_emitted_token(1, 1),
            Some(FinishReason::Length)
        ));
        assert!(finish_after_emitted_token(1, 2).is_none());
    }
}
