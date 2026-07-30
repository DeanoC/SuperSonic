//! Core generation loop shared by `/v1/chat/completions` and
//! `/v1/completions`. Runs on a `spawn_blocking` thread (engines are
//! synchronous HIP calls), and publishes token-level events on an
//! unbounded channel so the HTTP layer can either collect them into one
//! response or stream them as SSE.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::{OwnedSemaphorePermit, TryAcquireError};

use crate::prefix_cache::{supported_cache_request, CacheRequest};
use crate::sampling::{rng_from_seed, sample};
use crate::session::{
    classify_session_failure, is_request_local_cache_failure, is_session_cancellation,
    should_use_dflash_generation, InferenceSession, SessionFailureClass,
};
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

#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub dflash_rounds: Option<usize>,
    pub dflash_accepted_total: Option<usize>,
    pub decode_ms: Option<f64>,
}

impl GenerationStats {
    pub fn token_counts(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_prompt_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            ..Self::default()
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
        stats: GenerationStats,
    },
    /// Terminal error event: generation failed; no more events will arrive.
    Error(String),
}

trait GenerationEventSink {
    fn is_closed(&self) -> bool;
    fn send(&self, event: GenEvent) -> std::result::Result<(), GenEvent>;
}

impl GenerationEventSink for UnboundedSender<GenEvent> {
    fn is_closed(&self) -> bool {
        UnboundedSender::is_closed(self)
    }

    fn send(&self, event: GenEvent) -> std::result::Result<(), GenEvent> {
        UnboundedSender::send(self, event).map_err(|error| error.0)
    }
}

#[derive(Debug, Default)]
pub struct GenerationTelemetry {
    dflash_last_rounds: AtomicUsize,
    dflash_last_accepted_total: AtomicUsize,
    dflash_last_decode_ms_bits: AtomicU64,
}

impl GenerationTelemetry {
    pub fn record(&self, stats: &GenerationStats) {
        if let Some(rounds) = stats.dflash_rounds {
            self.dflash_last_rounds.store(rounds, Ordering::SeqCst);
        }
        if let Some(accepted_total) = stats.dflash_accepted_total {
            self.dflash_last_accepted_total
                .store(accepted_total, Ordering::SeqCst);
        }
        if let Some(decode_ms) = stats.decode_ms {
            self.dflash_last_decode_ms_bits
                .store(decode_ms.to_bits(), Ordering::SeqCst);
        }
    }

    pub fn snapshot(&self) -> DFlashTelemetrySnapshot {
        DFlashTelemetrySnapshot {
            last_rounds: self.dflash_last_rounds.load(Ordering::SeqCst),
            last_accepted_total: self.dflash_last_accepted_total.load(Ordering::SeqCst),
            last_decode_ms: f64::from_bits(self.dflash_last_decode_ms_bits.load(Ordering::SeqCst)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DFlashTelemetrySnapshot {
    pub last_rounds: usize,
    pub last_accepted_total: usize,
    pub last_decode_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationTelemetrySnapshot {
    pub active: usize,
    pub queued: usize,
    pub dflash: DFlashTelemetrySnapshot,
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
    if !state.is_ready() {
        anyhow::bail!("inference server is not ready after an integrity failure");
    }
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
        run_mock(mock, &prompt_ids, &params, &state.telemetry, &tx).await;
        scheduler.active.fetch_sub(1, Ordering::SeqCst);
        drop(permit);
        return;
    }

    let scheduler_done = scheduler.clone();
    let panic_state = state.clone();
    let panic_tx = tx.clone();
    let join = tokio::task::spawn_blocking(move || {
        let result = run(state, prompt_ids, params, cache, tx.clone());
        if let Err(e) = result {
            let _ = tx.send(GenEvent::Error(e.to_string()));
        }
    })
    .await;
    if let Err(e) = join {
        if e.is_panic() {
            panic_state.mark_integrity_lost();
            let _ = panic_tx.send(GenEvent::Error(
                "generation worker panicked; engine integrity is lost".to_string(),
            ));
        }
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

pub fn telemetry_snapshot(state: &ServerState) -> GenerationTelemetrySnapshot {
    let queue = scheduler_snapshot(state);
    GenerationTelemetrySnapshot {
        active: queue.active,
        queued: queue.queued,
        dflash: state.telemetry.snapshot(),
    }
}

async fn run_mock(
    mock: MockGeneration,
    prompt_ids: &[u32],
    params: &GenParams,
    telemetry: &GenerationTelemetry,
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
    let _ = send_done(
        telemetry,
        tx,
        if params.max_tokens == 0 {
            FinishReason::Length
        } else {
            mock.finish
        },
        GenerationStats::token_counts(
            prompt_ids.len() as u32,
            if params.max_tokens == 0 {
                0
            } else {
                completion_tokens
            },
            0,
        ),
    );
}

fn run<S: GenerationEventSink>(
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    cache: Option<CacheRequest>,
    tx: S,
) -> Result<()> {
    if !state.is_ready() {
        anyhow::bail!("inference server is not ready after an integrity failure");
    }
    let result = run_inner(state.clone(), prompt_ids, params, cache, &tx);
    if let Err(error) = &result {
        if classify_session_failure(error) == SessionFailureClass::IntegrityLost {
            state.mark_integrity_lost();
            tracing::error!(error = %error, "generation lost engine integrity");
        }
    }
    result
}

fn run_inner<S: GenerationEventSink>(
    state: Arc<ServerState>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    cache: Option<CacheRequest>,
    tx: &S,
) -> Result<()> {
    let tokenizer = state.tokenizer.clone();
    let prompt_tokens = prompt_ids.len() as u32;
    let mut cached_prompt_tokens = 0u32;

    // Zero-token request: return an empty completion without touching the
    // engine. OpenAI semantics: `max_tokens=0` means no completion tokens.
    if params.max_tokens == 0 {
        let _ = send_done(
            &state.telemetry,
            tx,
            FinishReason::Length,
            GenerationStats::token_counts(prompt_tokens, 0, 0),
        );
        return Ok(());
    }

    if tx.is_closed() {
        return Ok(());
    }
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| anyhow!("no inference session configured"))?;
    let mut guard = session.blocking_lock();
    if tx.is_closed() {
        return Ok(());
    }
    let features = guard.features();
    if should_use_dflash_generation(features) {
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
        if emit_generated_ids(
            &tokenizer,
            &state.eos_ids,
            &output.token_ids,
            &params,
            GenerationStats {
                prompt_tokens,
                dflash_rounds: Some(output.rounds_run),
                dflash_accepted_total: Some(output.accepted_total),
                decode_ms: Some(output.decode_ms),
                ..GenerationStats::default()
            },
            &state.telemetry,
            tx,
        )
        .is_err()
        {
            guard.reset()?;
        }
        return Ok(());
    }
    if !features.plain_prefill_decode {
        anyhow::bail!("loaded session does not expose a supported generation path");
    }
    let cache = supported_cache_request(features, cache.as_ref());
    let mut mutation_started = false;

    let prefill_result = (|| -> Result<Vec<f32>> {
        if let Some(cache_req) = cache {
            let memory_hit = match state.prefix_cache.lookup(cache_req, &prompt_ids) {
                Ok(hit) => hit,
                Err(error) => {
                    if classify_session_failure(&error) == SessionFailureClass::IntegrityLost
                        || !is_request_local_cache_failure(&error)
                    {
                        return Err(error);
                    }
                    tracing::warn!("prefix cache lookup failed: {error}");
                    state.prefix_cache.record_restore_failure();
                    None
                }
            };
            if let Some(hit) = memory_hit {
                check_cancelled(tx)?;
                mutation_started = true;
                match guard.restore_prefix(hit.snapshot) {
                    Ok(mut logits) => {
                        cached_prompt_tokens = hit.cached_tokens as u32;
                        for (idx, token) in prompt_ids
                            .iter()
                            .copied()
                            .enumerate()
                            .skip(hit.cached_tokens)
                        {
                            if tx.is_closed() {
                                return Err(crate::session::SessionCancelled.into());
                            }
                            logits = guard.decode_step(token, idx)?;
                        }
                        Ok(logits)
                    }
                    Err(e) => {
                        if classify_session_failure(&e) == SessionFailureClass::IntegrityLost
                            || !is_request_local_cache_failure(&e)
                        {
                            return Err(e);
                        }
                        tracing::warn!("prefix cache restore failed: {e}");
                        state.prefix_cache.record_restore_failure();
                        guard.reset()?;
                        mutation_started = false;
                        prefill_cancellable_tracking(
                            &mut guard,
                            &prompt_ids,
                            tx,
                            &mut mutation_started,
                        )
                    }
                }
            } else if let Some(hit) = state.prefix_cache.lookup_disk_bytes(cache_req, &prompt_ids) {
                match guard.load_disk_prefix(&hit.bytes) {
                    Ok(snapshot) => {
                        check_cancelled(tx)?;
                        mutation_started = true;
                        match guard.restore_prefix(snapshot) {
                            Ok(mut logits) => {
                                cached_prompt_tokens = hit.cached_tokens as u32;
                                for (idx, token) in prompt_ids
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .skip(hit.cached_tokens)
                                {
                                    if tx.is_closed() {
                                        return Err(crate::session::SessionCancelled.into());
                                    }
                                    logits = guard.decode_step(token, idx)?;
                                }
                                Ok(logits)
                            }
                            Err(e) => {
                                if classify_session_failure(&e)
                                    == SessionFailureClass::IntegrityLost
                                    || !is_request_local_cache_failure(&e)
                                {
                                    return Err(e);
                                }
                                tracing::warn!("prefix cache disk restore failed: {e}");
                                state.prefix_cache.record_restore_failure();
                                guard.reset()?;
                                mutation_started = false;
                                prefill_with_cache_anchor(
                                    &mut guard,
                                    &state,
                                    cache_req,
                                    &prompt_ids,
                                    tx,
                                    &mut mutation_started,
                                )
                            }
                        }
                    }
                    Err(e) => {
                        if classify_session_failure(&e) == SessionFailureClass::IntegrityLost
                            || !is_request_local_cache_failure(&e)
                        {
                            return Err(e);
                        }
                        tracing::warn!("prefix cache disk load failed: {e}");
                        state.prefix_cache.record_restore_failure();
                        check_cancelled(tx)?;
                        guard.reset()?;
                        mutation_started = false;
                        prefill_with_cache_anchor(
                            &mut guard,
                            &state,
                            cache_req,
                            &prompt_ids,
                            tx,
                            &mut mutation_started,
                        )
                    }
                }
            } else {
                check_cancelled(tx)?;
                guard.reset()?;
                mutation_started = false;
                prefill_with_cache_anchor(
                    &mut guard,
                    &state,
                    cache_req,
                    &prompt_ids,
                    tx,
                    &mut mutation_started,
                )
            }
        } else {
            check_cancelled(tx)?;
            guard.reset()?;
            mutation_started = false;
            prefill_cancellable_tracking(&mut guard, &prompt_ids, tx, &mut mutation_started)
        }
    })();
    let prefill_logits = match prefill_result {
        Err(error) if is_session_cancellation(&error) => {
            cleanup_cancelled_session(&mut guard, &mut mutation_started)?;
            return Ok(());
        }
        result => result?,
    };

    if let Some(cache_req) = cache {
        snapshot_prefix_if_admitted(&guard, &state, cache_req, &prompt_ids, &prefill_logits)?;
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
        if tx.is_closed() {
            cleanup_cancelled_session(&mut guard, &mut mutation_started)?;
            return Ok(());
        }

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
                if tx.send(GenEvent::Token(delta)).is_err() {
                    cleanup_cancelled_session(&mut guard, &mut mutation_started)?;
                    return Ok(());
                }
            }
            break FinishReason::Stop;
        }

        let delta = incremental_delta(&prev_decoded, &decoded);
        prev_decoded = decoded;

        if !delta.is_empty() && tx.send(GenEvent::Token(delta)).is_err() {
            cleanup_cancelled_session(&mut guard, &mut mutation_started)?;
            return Ok(());
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
        if let Some(cache_req) = cache {
            snapshot_prefix_if_admitted(
                &guard,
                &state,
                cache_req,
                &state_token_ids,
                &current_logits,
            )?;
        }
    }

    if send_done(
        &state.telemetry,
        tx,
        finish,
        GenerationStats::token_counts(prompt_tokens, completion_tokens, cached_prompt_tokens),
    )
    .is_err()
    {
        cleanup_cancelled_session(&mut guard, &mut mutation_started)?;
    }
    Ok(())
}

fn check_cancelled<S: GenerationEventSink>(tx: &S) -> Result<()> {
    if tx.is_closed() {
        Err(crate::session::SessionCancelled.into())
    } else {
        Ok(())
    }
}

fn cleanup_cancelled_session(
    guard: &mut InferenceSession,
    mutation_started: &mut bool,
) -> Result<()> {
    if std::mem::take(mutation_started) {
        guard.reset()?;
    }
    Ok(())
}

fn emit_generated_ids<S: GenerationEventSink>(
    tokenizer: &Tokenizer,
    eos_ids: &[u32],
    token_ids: &[u32],
    params: &GenParams,
    mut stats: GenerationStats,
    telemetry: &GenerationTelemetry,
    tx: &S,
) -> std::result::Result<(), GenEvent> {
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
                tx.send(GenEvent::Token(delta))?;
            }
            finish = FinishReason::Stop;
            break;
        }

        let delta = incremental_delta(&prev_decoded, &decoded);
        prev_decoded = decoded;
        if !delta.is_empty() {
            tx.send(GenEvent::Token(delta))?;
        }
    }

    stats.completion_tokens = completion_tokens;
    send_done(telemetry, tx, finish, stats)
}

fn send_done<S: GenerationEventSink>(
    telemetry: &GenerationTelemetry,
    tx: &S,
    reason: FinishReason,
    stats: GenerationStats,
) -> std::result::Result<(), GenEvent> {
    telemetry.record(&stats);
    tx.send(GenEvent::Done { reason, stats })
}

fn prefill_with_cache_anchor<S: GenerationEventSink>(
    guard: &mut InferenceSession,
    state: &ServerState,
    cache_req: &CacheRequest,
    prompt_ids: &[u32],
    tx: &S,
    mutation_started: &mut bool,
) -> Result<Vec<f32>> {
    let min_tokens = state.prefix_cache.config().min_tokens;
    let Some(anchor_len) = prompt_ids.len().checked_sub(CACHE_ANCHOR_SUFFIX_TOKENS) else {
        return prefill_cancellable_tracking(guard, prompt_ids, tx, mutation_started);
    };
    if anchor_len < min_tokens || anchor_len == 0 {
        return prefill_cancellable_tracking(guard, prompt_ids, tx, mutation_started);
    }

    let mut logits =
        prefill_cancellable_tracking(guard, &prompt_ids[..anchor_len], tx, mutation_started)?;
    snapshot_prefix_if_admitted(guard, state, cache_req, &prompt_ids[..anchor_len], &logits)?;
    for (idx, token) in prompt_ids.iter().copied().enumerate().skip(anchor_len) {
        if tx.is_closed() {
            return Err(crate::session::SessionCancelled.into());
        }
        logits = guard.decode_step(token, idx)?;
    }
    Ok(logits)
}

fn prefill_cancellable_tracking<S: GenerationEventSink>(
    guard: &mut InferenceSession,
    prompt_ids: &[u32],
    tx: &S,
    mutation_started: &mut bool,
) -> Result<Vec<f32>> {
    guard.prefill_cancellable_with_started(
        prompt_ids,
        || tx.is_closed(),
        || *mutation_started = true,
    )
}

fn snapshot_prefix_if_admitted(
    guard: &InferenceSession,
    state: &ServerState,
    cache_req: &CacheRequest,
    token_ids: &[u32],
    logits: &[f32],
) -> Result<()> {
    if !guard.features().prefix_snapshot {
        return Ok(());
    }
    let estimate = guard.prefix_snapshot_bytes(logits.len());
    if !state.prefix_cache.can_admit(token_ids.len(), estimate) {
        state.prefix_cache.record_admission_skip();
        return Ok(());
    }
    match guard.snapshot_prefix(logits.to_vec()) {
        Ok(snapshot) => {
            if let Err(e) = state.prefix_cache.insert(cache_req, token_ids, snapshot) {
                if classify_session_failure(&e) == SessionFailureClass::IntegrityLost
                    || !is_request_local_cache_failure(&e)
                {
                    return Err(e);
                }
                tracing::warn!("prefix cache insert failed: {e}");
            }
        }
        Err(e) => {
            if classify_session_failure(&e) == SessionFailureClass::IntegrityLost
                || !is_request_local_cache_failure(&e)
            {
                return Err(e);
            }
            tracing::debug!("prefix cache snapshot skipped: {e}");
        }
    }
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
    let mut stats = GenerationStats::default();
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(s) => text.push_str(&s),
            GenEvent::Done { reason, stats: s } => {
                finish = Some(reason);
                stats = s;
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
        stats,
    })
}

pub struct CollectedResult {
    pub text: String,
    pub finish: FinishReason,
    pub stats: GenerationStats,
}

/// Re-export kept so downstream route modules can name the session type.
pub type Session = InferenceSession;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use tokio::sync::{mpsc, Mutex};

    use super::{
        finish_after_emitted_token, incremental_delta, run, spawn, FinishReason, GenEvent,
        GenParams,
    };
    use crate::prefix_cache::{CacheRequest, CacheRetention, PrefixCache, PrefixCacheConfig};
    use crate::qwen36_moe::engine::Qwen36MoePrefillBoundary;
    use crate::session::{
        qwen36_moe_features, DeterministicSession, DeterministicSessionEvent, InferenceSession,
        SessionFeatures, SessionSnapshot,
    };
    use crate::state::{GenerationScheduler, ServerState};
    use supersonic_core::capabilities::capabilities_for_variant;
    use supersonic_core::registry::{ModelFamily, ModelVariant};

    #[derive(Clone, Default)]
    struct FailDoneSink {
        tokens: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl super::GenerationEventSink for FailDoneSink {
        fn is_closed(&self) -> bool {
            false
        }

        fn send(&self, event: GenEvent) -> std::result::Result<(), GenEvent> {
            match event {
                GenEvent::Done { .. } => Err(event),
                GenEvent::Token(token) => {
                    self.tokens.lock().unwrap().push(token);
                    Ok(())
                }
                GenEvent::Error(_) => Ok(()),
            }
        }
    }

    fn tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_bytes(
            r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#,
        )
        .expect("deterministic tokenizer")
    }

    fn params(max_tokens: usize) -> GenParams {
        GenParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop: Vec::new(),
            seed: Some(7),
        }
    }

    fn test_state(session: InferenceSession) -> Arc<ServerState> {
        Arc::new(ServerState {
            server_instance_id: 42,
            model_id: "qwen3.6-35b-a3b".to_string(),
            model_family: ModelFamily::Qwen36Moe,
            tokenizer: Arc::new(tokenizer()),
            chat_template: None,
            session: Some(Arc::new(Mutex::new(session))),
            qwen36_moe_engine: None,
            mock_generation: None,
            eos_ids: Vec::new(),
            max_context: 64,
            api_key: None,
            cors_allow_origin: None,
            response_store_max_entries: 16,
            scheduler: Arc::new(GenerationScheduler::new(4, 1_000)),
            telemetry: super::GenerationTelemetry::default(),
            capabilities: capabilities_for_variant(
                &ModelVariant::Qwen3_6_35B_A3B,
                gpu_hal::Backend::Hip,
                true,
                false,
                false,
            ),
            prefix_cache: Arc::new(PrefixCache::new(PrefixCacheConfig {
                enabled: true,
                dir: PathBuf::new(),
                min_tokens: 1,
                max_entries: 4,
                max_bytes: 1024 * 1024,
                memory_ttl_secs: 600,
                disk_ttl_secs: 86_400,
            })),
        })
    }

    fn cache_request() -> CacheRequest {
        CacheRequest {
            key: Some("shared-prefix".to_string()),
            retention: CacheRetention::InMemory,
            scope: "qwen36-test".to_string(),
        }
    }

    fn done_stats(mut rx: mpsc::UnboundedReceiver<GenEvent>) -> super::GenerationStats {
        loop {
            match rx.try_recv() {
                Ok(GenEvent::Done { stats, .. }) => return stats,
                Ok(GenEvent::Token(_)) => {}
                Ok(GenEvent::Error(error)) => panic!("unexpected generation error: {error}"),
                Err(error) => panic!("missing done event: {error}"),
            }
        }
    }

    async fn wait_for_idle(state: &ServerState) {
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.scheduler.active.load(Ordering::SeqCst) == 0
                    && state.scheduler.queued.load(Ordering::SeqCst) == 0
                    && state.scheduler.permits.available_permits() == 1
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("generation scheduler must become idle");
    }

    fn assert_admission_released(state: &ServerState) {
        assert_eq!(state.scheduler.active.load(Ordering::SeqCst), 0);
        assert_eq!(state.scheduler.queued.load(Ordering::SeqCst), 0);
        assert_eq!(state.scheduler.permits.available_permits(), 1);
    }

    fn snapshot_features() -> SessionFeatures {
        SessionFeatures {
            plain_prefill_decode: true,
            native_dflash_generate: false,
            prefix_snapshot: true,
            disk_prefix_snapshot: false,
        }
    }

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

    #[test]
    fn qwen36_generation_orders_session_calls_bounds_decode_and_bypasses_cache() {
        let (backend, events) = DeterministicSession::new(
            qwen36_moe_features(),
            vec![0.0, 5.0, 0.0],
            vec![vec![0.0, 0.0, 7.0]],
        );
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));
        let (tx, rx) = mpsc::unbounded_channel();

        run(
            state.clone(),
            vec![1, 2],
            params(2),
            Some(cache_request()),
            tx,
        )
        .unwrap();

        let stats = done_stats(rx);
        assert_eq!(stats.prompt_tokens, 2);
        assert_eq!(stats.completion_tokens, 2);
        assert_eq!(stats.cached_prompt_tokens, 0);
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1, 2]),
                DeterministicSessionEvent::Decode {
                    token_id: 1,
                    pos: 2,
                },
            ]
        );
        let cache = state.prefix_cache.stats();
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
        assert_eq!(cache.cached_tokens, 0);
        assert_eq!(cache.entries, 0);
        assert_eq!(cache.admission_skips, 0);
    }

    #[test]
    fn disconnect_after_prefill_resets_request_state_without_decoding() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0, 0.0], Vec::new());
        let backend = backend.after_prefill(move || drop(rx));
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        run(state.clone(), vec![1, 2], params(4), None, tx).unwrap();

        assert!(state.is_ready());
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1, 2]),
                DeterministicSessionEvent::Reset,
            ]
        );
    }

    #[test]
    fn integrity_failure_marks_server_unready_and_rejects_followup_generation() {
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0, 0.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_device_loss(),
        ));
        let (tx, _rx) = mpsc::unbounded_channel();

        let error = run(state.clone(), vec![1], params(1), None, tx).unwrap_err();
        assert!(error
            .downcast_ref::<gpu_hal::GpuError>()
            .is_some_and(gpu_hal::GpuError::is_device_lost));
        assert!(!state.is_ready());

        let (tx, _rx) = mpsc::unbounded_channel();
        let rejected = run(state.clone(), vec![1], params(1), None, tx).unwrap_err();
        assert!(rejected.to_string().contains("not ready"));
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1]),
            ]
        );
    }

    #[test]
    fn request_local_failure_preserves_readiness_for_next_request() {
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0, 0.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_failure("context limit exceeded"),
        ));
        let (tx, _rx) = mpsc::unbounded_channel();

        run(state.clone(), vec![1], params(1), None, tx).unwrap_err();
        assert!(state.is_ready());

        let (tx, rx) = mpsc::unbounded_channel();
        run(state.clone(), vec![1], params(1), None, tx).unwrap();
        assert_eq!(done_stats(rx).completion_tokens, 1);
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1]),
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1]),
            ]
        );
    }

    #[test]
    fn cuda_runtime_unavailable_status_46_preserves_readiness_for_followup() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_gpu_status(gpu_hal::Backend::Cuda, 46),
        ));
        let (tx, _rx) = mpsc::unbounded_channel();

        run(state.clone(), vec![1], params(1), None, tx).unwrap_err();
        assert!(state.is_ready());

        let (tx, rx) = mpsc::unbounded_channel();
        run(state.clone(), vec![1], params(1), None, tx).unwrap();
        assert_eq!(done_stats(rx).completion_tokens, 1);
    }

    #[test]
    fn fatal_cuda_runtime_statuses_poison_readiness_and_reject_followup() {
        for status in [700, 710] {
            let (backend, _) =
                DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
            let state = test_state(InferenceSession::test_qwen36_adapter(
                backend.with_prefill_gpu_status(gpu_hal::Backend::Cuda, status),
            ));
            let (tx, _rx) = mpsc::unbounded_channel();

            run(state.clone(), vec![1], params(1), None, tx).unwrap_err();
            assert!(!state.is_ready(), "CUDA runtime status {status}");

            let (tx, _rx) = mpsc::unbounded_channel();
            assert!(run(state, vec![1], params(1), None, tx).is_err());
        }
    }

    #[test]
    fn cancellation_cleanup_reset_failure_loses_integrity() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0, 0.0], Vec::new());
        let backend = backend
            .with_reset_failures(vec![None, Some("reset sync failed")])
            .after_prefill(move || drop(rx));
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        let error = run(state.clone(), vec![1], params(4), None, tx).unwrap_err();

        assert!(error.to_string().contains("reset sync failed"));
        assert!(!state.is_ready());
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
                DeterministicSessionEvent::Prefill(vec![1]),
                DeterministicSessionEvent::Reset,
            ]
        );
    }

    #[tokio::test]
    async fn spawn_rejects_unready_server_before_queue_or_stream_admission() {
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0, 0.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));
        state.mark_integrity_lost();

        let error = spawn(state.clone(), vec![1], params(1), None)
            .err()
            .expect("unready server must reject synchronously");

        assert!(error.to_string().contains("not ready"));
        assert_eq!(state.scheduler.active.load(Ordering::SeqCst), 0);
        assert_eq!(state.scheduler.queued.load(Ordering::SeqCst), 0);
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn spawn_success_releases_permit_and_counters() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        let result = super::collect(spawn(state.clone(), vec![1], params(1), None).unwrap())
            .await
            .unwrap();
        wait_for_idle(&state).await;

        assert_eq!(result.stats.completion_tokens, 1);
        assert!(state.is_ready());
        assert_admission_released(&state);
    }

    #[tokio::test]
    async fn spawn_cancellation_releases_permit_and_counters() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        let rx = spawn(state.clone(), vec![1], params(1), None).unwrap();
        drop(rx);
        wait_for_idle(&state).await;

        assert!(state.is_ready());
        assert_admission_released(&state);
    }

    #[tokio::test]
    async fn spawn_request_error_releases_permit_and_counters() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_failure("request context limit exceeded"),
        ));

        let error = super::collect(spawn(state.clone(), vec![1], params(1), None).unwrap())
            .await
            .err()
            .expect("request-local generation must fail");
        wait_for_idle(&state).await;

        assert!(error.to_string().contains("request context limit exceeded"));
        assert!(state.is_ready());
        assert_admission_released(&state);
    }

    #[tokio::test]
    async fn spawn_integrity_loss_releases_permit_and_counters() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_device_loss(),
        ));

        let error = super::collect(spawn(state.clone(), vec![1], params(1), None).unwrap())
            .await
            .err()
            .expect("integrity-lost generation must fail");
        wait_for_idle(&state).await;

        assert!(error.to_string().contains("status 709"));
        assert!(!state.is_ready());
        assert_admission_released(&state);
    }

    #[test]
    fn typed_device_loss_during_cache_restore_poison_rejects_followup() {
        let (backend, _) =
            DeterministicSession::new(snapshot_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::Deterministic(
            backend.with_restore_device_loss(),
        ));
        let cache = cache_request();
        state
            .prefix_cache
            .insert(
                &cache,
                &[1, 2],
                SessionSnapshot::Deterministic {
                    logits: vec![0.0, 5.0],
                },
            )
            .unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();

        let error = run(state.clone(), vec![1, 2], params(1), Some(cache), tx).unwrap_err();

        assert!(error
            .downcast_ref::<gpu_hal::GpuError>()
            .is_some_and(gpu_hal::GpuError::is_device_lost));
        assert!(!state.is_ready());
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(run(state, vec![1], params(1), None, tx).is_err());
    }

    #[test]
    fn typed_request_local_cache_restore_error_resets_and_retries() {
        let (backend, events) =
            DeterministicSession::new(snapshot_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::Deterministic(
            backend.with_restore_request_local_cache_failure(),
        ));
        let cache = cache_request();
        state
            .prefix_cache
            .insert(
                &cache,
                &[1, 2],
                SessionSnapshot::Deterministic {
                    logits: vec![0.0, 5.0],
                },
            )
            .unwrap();
        let (tx, rx) = mpsc::unbounded_channel();

        run(state.clone(), vec![1, 2], params(1), Some(cache), tx).unwrap();

        assert!(state.is_ready());
        assert_eq!(done_stats(rx).completion_tokens, 1);
        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::Prefill(vec![1, 2]),
            ]
        );
    }

    #[test]
    fn typed_device_loss_during_snapshot_capture_poison_rejects_followup() {
        let (backend, _) =
            DeterministicSession::new(snapshot_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::Deterministic(
            backend.with_snapshot_device_loss(),
        ));
        let (tx, _rx) = mpsc::unbounded_channel();

        let error = run(
            state.clone(),
            vec![1, 2],
            params(1),
            Some(cache_request()),
            tx,
        )
        .unwrap_err();

        assert!(error
            .downcast_ref::<gpu_hal::GpuError>()
            .is_some_and(gpu_hal::GpuError::is_device_lost));
        assert!(!state.is_ready());
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(run(state, vec![1], params(1), None, tx).is_err());
    }

    #[tokio::test]
    async fn blocking_worker_panic_poison_releases_admission_and_rejects_followup() {
        let (backend, _) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(
            backend.with_prefill_panic(),
        ));

        let mut rx = spawn(state.clone(), vec![1], params(1), None).unwrap();
        while rx.recv().await.is_some() {}
        wait_for_idle(&state).await;

        assert!(!state.is_ready());
        assert_admission_released(&state);
        assert!(spawn(state, vec![1], params(1), None).is_err());
    }

    #[test]
    fn closed_before_worker_session_access_performs_no_reset() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        run(state.clone(), vec![1], params(1), None, tx).unwrap();

        assert!(state.is_ready());
        assert!(events.lock().unwrap().is_empty());
    }

    #[test]
    fn cancellation_at_clean_normal_prefill_handoff_does_not_reset_twice() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let backend = backend
            .with_reset_failures(vec![None, Some("redundant reset sentinel")])
            .after_first_reset(move || drop(rx));
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        run(state.clone(), vec![1], params(1), None, tx).unwrap();

        assert!(state.is_ready());
        assert_eq!(*events.lock().unwrap(), [DeterministicSessionEvent::Reset]);
    }

    #[test]
    fn cancellation_at_clean_cache_recovery_prefill_handoff_does_not_reset_twice() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (backend, events) =
            DeterministicSession::new(snapshot_features(), vec![0.0, 5.0], Vec::new());
        let backend = backend
            .with_restore_request_local_cache_failure()
            .with_reset_failures(vec![None, Some("redundant reset sentinel")])
            .after_first_reset(move || drop(rx));
        let state = test_state(InferenceSession::Deterministic(backend));
        let cache = cache_request();
        state
            .prefix_cache
            .insert(
                &cache,
                &[1, 2],
                SessionSnapshot::Deterministic {
                    logits: vec![0.0, 5.0],
                },
            )
            .unwrap();

        run(state.clone(), vec![1, 2], params(1), Some(cache), tx).unwrap();

        assert!(state.is_ready());
        assert_eq!(*events.lock().unwrap(), [DeterministicSessionEvent::Reset]);
    }

    #[test]
    fn cancellation_at_each_qwen_prefill_boundary_cleans_up_exactly_once() {
        for boundary in [
            Qwen36MoePrefillBoundary::PrefixStarted,
            Qwen36MoePrefillBoundary::FinalProductionStarted,
        ] {
            let (tx, rx) = mpsc::unbounded_channel();
            let (backend, events) =
                DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
            let backend = backend.close_at_prefill_boundary(boundary, move || drop(rx));
            let state = test_state(InferenceSession::test_qwen36_adapter(backend));

            run(state.clone(), vec![1, 2], params(2), None, tx).unwrap();

            assert!(state.is_ready());
            let events = events.lock().unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, DeterministicSessionEvent::Reset))
                    .count(),
                2,
                "{boundary:?}: {events:?}"
            );
            assert_eq!(events.first(), Some(&DeterministicSessionEvent::Reset));
            assert_eq!(events.last(), Some(&DeterministicSessionEvent::Reset));
            assert!(events.contains(&DeterministicSessionEvent::PrefillBoundary(boundary)));
            assert!(!events
                .iter()
                .any(|event| matches!(event, DeterministicSessionEvent::Decode { .. })));
        }
    }

    #[test]
    fn cancellation_after_decode_cleans_up_exactly_once() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (backend, events) = DeterministicSession::new(
            qwen36_moe_features(),
            vec![0.0, 5.0, 0.0],
            vec![vec![0.0, 0.0, 7.0]],
        );
        let backend = backend.after_decode(move || drop(rx));
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));

        run(state.clone(), vec![1, 2], params(3), None, tx).unwrap();

        assert!(state.is_ready());
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, DeterministicSessionEvent::Reset))
                .count(),
            2,
            "{events:?}"
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, DeterministicSessionEvent::Decode { .. })));
        assert_eq!(events.last(), Some(&DeterministicSessionEvent::Reset));
    }

    #[test]
    fn final_token_before_done_drop_cleans_up_and_propagates_reset_integrity_failure() {
        let (backend, events) =
            DeterministicSession::new(qwen36_moe_features(), vec![0.0, 5.0], Vec::new());
        let backend =
            backend.with_reset_failures(vec![None, Some("terminal cleanup reset failed")]);
        let state = test_state(InferenceSession::test_qwen36_adapter(backend));
        let sink = FailDoneSink::default();

        let error = run(state.clone(), vec![1], params(1), None, sink.clone()).unwrap_err();

        assert!(error.to_string().contains("terminal cleanup reset failed"));
        assert!(!state.is_ready());
        assert_eq!(sink.tokens.lock().unwrap().as_slice(), ["hello"]);
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, DeterministicSessionEvent::Reset))
                .count(),
            2,
            "{events:?}"
        );
        assert_eq!(events.last(), Some(&DeterministicSessionEvent::Reset));
    }
}
