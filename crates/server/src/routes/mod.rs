//! HTTP route registration and shared middleware.

use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use axum::extract::{Request, State};

use crate::errors::{ApiError, ApiErrorBody, ApiErrorEnvelope};
use crate::state::ServerState;

pub mod chat;
pub mod completions;
pub mod models;

pub fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/v1/models", get(models::list))
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/completions", post(completions::completions))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}

async fn auth_middleware(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Some(expected) = state.api_key.as_deref() else {
        return Ok(next.run(request).await);
    };
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").map(str::trim));
    match got {
        Some(k) if k == expected => Ok(next.run(request).await),
        _ => Err(ApiError::unauthorized("missing or invalid API key")),
    }
}

// Keep the envelope types referenced so unused-import lints stay quiet when
// a future route wants to hand-roll an error body.
#[allow(dead_code)]
fn _types_kept_live(_a: ApiErrorBody, _b: ApiErrorEnvelope, _s: StatusCode) {}
