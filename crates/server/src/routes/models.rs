//! `GET /v1/models` — returns the single model loaded by this process.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{OriginalUri, Path, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::generate;
use crate::prefix_cache::PrefixCacheStats;
use crate::schemas::{FlmEvidence, ListModelsResponse, ModelObject};
use crate::state::ServerState;

pub async fn list(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<ListModelsResponse>, ApiError> {
    Ok(Json(ListModelsResponse {
        object: "list",
        data: vec![model_object(&state)],
    }))
}

pub async fn retrieve(
    State(state): State<Arc<ServerState>>,
    Path(model): Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    validate_model(Some(&model), &state.model_id)?;
    Ok(Json(model_object(&state)))
}

fn model_object(state: &ServerState) -> ModelObject {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    ModelObject {
        id: state.model_id.clone(),
        object: "model",
        created,
        owned_by: "supersonic",
    }
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub ready: bool,
    pub model: String,
    pub max_context: usize,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_queued_requests: usize,
    pub prefix_cache_entries: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flm: Option<FlmEvidence>,
}

pub async fn health(
    OriginalUri(uri): OriginalUri,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let queue = generate::scheduler_snapshot(&state);
    let cache = state.prefix_cache.stats();
    let ready = state.is_ready();
    let response = HealthResponse {
        status: if ready { "ok" } else { "degraded" },
        ready,
        model: state.model_id.clone(),
        max_context: state.max_context,
        active_requests: queue.active,
        queued_requests: queue.queued,
        max_queued_requests: queue.max_queue,
        prefix_cache_entries: cache.entries,
        flm: state.capabilities.flm.as_ref().map(FlmEvidence::from),
    };
    let status = if uri.path().ends_with("/ready") && !ready {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    (status, Json(response))
}

#[derive(Debug, Serialize)]
pub struct CapabilitiesResponse {
    pub model: String,
    pub family: String,
    pub backend: String,
    pub ready: bool,
    pub max_context: usize,
    pub endpoints: Vec<&'static str>,
    pub chat: bool,
    pub completions: bool,
    pub responses: bool,
    pub streaming: bool,
    pub stream_usage: bool,
    pub tools: bool,
    pub reasoning: bool,
    pub scheduler: SchedulerCapabilities,
    pub prefix_cache: PrefixCacheStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flm: Option<FlmEvidence>,
}

#[derive(Debug, Serialize)]
pub struct SchedulerCapabilities {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_queued_requests: usize,
    pub queue_timeout_ms: u64,
}

pub async fn capabilities(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<CapabilitiesResponse>, ApiError> {
    let queue = generate::scheduler_snapshot(&state);
    let cache = state.prefix_cache.stats();
    Ok(Json(CapabilitiesResponse {
        model: state.model_id.clone(),
        family: state.model_family.to_string(),
        backend: state.capabilities.backend.to_string(),
        ready: state.is_ready(),
        max_context: state.max_context,
        endpoints: vec![
            "/v1/models",
            "/v1/models/{model}",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/tokenize",
            "/v1/detokenize",
            "/v1/responses",
            "/health",
            "/v1/health",
            "/ready",
            "/v1/ready",
            "/v1/capabilities",
            "/metrics",
        ],
        chat: state.chat_template.is_some(),
        completions: true,
        responses: true,
        streaming: true,
        stream_usage: true,
        tools: state.chat_template.is_some(),
        reasoning: state.chat_template.is_some(),
        scheduler: SchedulerCapabilities {
            active_requests: queue.active,
            queued_requests: queue.queued,
            max_queued_requests: queue.max_queue,
            queue_timeout_ms: queue.queue_timeout_ms,
        },
        prefix_cache: cache,
        flm: state.capabilities.flm.as_ref().map(FlmEvidence::from),
    }))
}

pub async fn metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let queue = generate::scheduler_snapshot(&state);
    let generation = generate::telemetry_snapshot(&state);
    let cache = state.prefix_cache.stats();
    let mut body = format!(
        "# TYPE supersonic_ready gauge\n\
         supersonic_ready {}\n\
         # TYPE supersonic_active_requests gauge\n\
         supersonic_active_requests {}\n\
         # TYPE supersonic_queued_requests gauge\n\
         supersonic_queued_requests {}\n\
         # TYPE supersonic_generation_active gauge\n\
         supersonic_generation_active {}\n\
         # TYPE supersonic_generation_queued gauge\n\
         supersonic_generation_queued {}\n\
         # TYPE supersonic_max_queued_requests gauge\n\
         supersonic_max_queued_requests {}\n\
         # TYPE supersonic_queue_timeout_ms gauge\n\
         supersonic_queue_timeout_ms {}\n\
         # TYPE supersonic_max_context gauge\n\
         supersonic_max_context {}\n\
         # TYPE supersonic_prefix_cache_enabled gauge\n\
         supersonic_prefix_cache_enabled {}\n\
         # TYPE supersonic_prefix_cache_entries gauge\n\
         supersonic_prefix_cache_entries {}\n\
         # TYPE supersonic_prefix_cache_resident_bytes gauge\n\
         supersonic_prefix_cache_resident_bytes {}\n\
         # TYPE supersonic_prefix_cache_max_bytes gauge\n\
         supersonic_prefix_cache_max_bytes {}\n\
         # TYPE supersonic_prefix_cache_hits counter\n\
         supersonic_prefix_cache_hits {}\n\
         # TYPE supersonic_prefix_cache_misses counter\n\
         supersonic_prefix_cache_misses {}\n\
         # TYPE supersonic_prefix_cache_cached_tokens counter\n\
         supersonic_prefix_cache_cached_tokens {}\n\
         # TYPE supersonic_prefix_cache_evictions counter\n\
         supersonic_prefix_cache_evictions {}\n\
         # TYPE supersonic_prefix_cache_disk_writes counter\n\
         supersonic_prefix_cache_disk_writes {}\n\
         # TYPE supersonic_prefix_cache_disk_reads counter\n\
         supersonic_prefix_cache_disk_reads {}\n\
         # TYPE supersonic_prefix_cache_restore_failures counter\n\
         supersonic_prefix_cache_restore_failures {}\n\
         # TYPE supersonic_prefix_cache_admission_skips counter\n\
         supersonic_prefix_cache_admission_skips {}\n\
         # TYPE supersonic_dflash_last_rounds gauge\n\
         supersonic_dflash_last_rounds {}\n\
         # TYPE supersonic_dflash_last_accepted_total gauge\n\
         supersonic_dflash_last_accepted_total {}\n\
         # TYPE supersonic_dflash_last_decode_ms gauge\n\
         supersonic_dflash_last_decode_ms {}\n",
        usize::from(state.is_ready()),
        queue.active,
        queue.queued,
        generation.active,
        generation.queued,
        queue.max_queue,
        queue.queue_timeout_ms,
        state.max_context,
        usize::from(cache.enabled),
        cache.entries,
        cache.resident_bytes,
        cache.max_bytes,
        cache.hits,
        cache.misses,
        cache.cached_tokens,
        cache.evictions,
        cache.disk_writes,
        cache.disk_reads,
        cache.restore_failures,
        cache.admission_skips,
        generation.dflash.last_rounds,
        generation.dflash.last_accepted_total,
        generation.dflash.last_decode_ms,
    );
    if let Some(flm) = state.capabilities.flm.as_ref() {
        let _ = write!(
            body,
            "# TYPE supersonic_model_loads_total counter\n\
             supersonic_model_loads_total {}\n\
             # TYPE supersonic_flm_native_int4_direct_weights gauge\n\
             supersonic_flm_native_int4_direct_weights {}\n\
             # TYPE supersonic_flm_bf16_fallback_weights gauge\n\
             supersonic_flm_bf16_fallback_weights {}\n\
             # TYPE supersonic_flm_source_bytes gauge\n\
             supersonic_flm_source_bytes {}\n\
             # TYPE supersonic_flm_device_upload_bytes gauge\n\
             supersonic_flm_device_upload_bytes {}\n\
             # TYPE supersonic_flm_startup_seconds gauge\n\
             supersonic_flm_startup_seconds {}\n",
            flm.load_sequence,
            flm.direct_profile.native_int4_direct_weights,
            flm.direct_profile.bf16_fallback_weights,
            flm.source_bytes,
            flm.device_upload_bytes,
            flm.startup.total.as_secs_f64(),
        );
    }
    ([(CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}
