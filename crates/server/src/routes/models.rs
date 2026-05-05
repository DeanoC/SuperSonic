//! `GET /v1/models` — returns the single model loaded by this process.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::generate;
use crate::schemas::{ListModelsResponse, ModelObject};
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
    pub model: String,
    pub max_context: usize,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_queued_requests: usize,
}

pub async fn health(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<HealthResponse>, ApiError> {
    let queue = generate::scheduler_snapshot(&state);
    Ok(Json(HealthResponse {
        status: "ok",
        model: state.model_id.clone(),
        max_context: state.max_context,
        active_requests: queue.active,
        queued_requests: queue.queued,
        max_queued_requests: queue.max_queue,
    }))
}

#[derive(Debug, Serialize)]
pub struct CapabilitiesResponse {
    pub model: String,
    pub family: String,
    pub backend: String,
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
    Ok(Json(CapabilitiesResponse {
        model: state.model_id.clone(),
        family: state.model_family.to_string(),
        backend: state.capabilities.backend.to_string(),
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
    }))
}

pub async fn metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let queue = generate::scheduler_snapshot(&state);
    let body = format!(
        "# TYPE supersonic_active_requests gauge\n\
         supersonic_active_requests {}\n\
         # TYPE supersonic_queued_requests gauge\n\
         supersonic_queued_requests {}\n\
         # TYPE supersonic_max_queued_requests gauge\n\
         supersonic_max_queued_requests {}\n\
         # TYPE supersonic_queue_timeout_ms gauge\n\
         supersonic_queue_timeout_ms {}\n\
         # TYPE supersonic_max_context gauge\n\
         supersonic_max_context {}\n",
        queue.active, queue.queued, queue.max_queue, queue.queue_timeout_ms, state.max_context
    );
    ([(CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}
