//! HTTP route registration and shared middleware.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use crate::errors::{ApiError, ApiErrorBody, ApiErrorEnvelope};
use crate::state::ServerState;

pub mod chat;
pub mod completions;
pub mod models;
pub mod responses;
mod sse;
pub mod tokenizer;

pub fn router(state: Arc<ServerState>) -> Router {
    let app = Router::new()
        .route("/", get(models::health))
        .route("/v1", get(models::health))
        .route("/ready", get(models::health))
        .route("/v1/ready", get(models::health))
        .route("/v1/models", get(models::list))
        .route("/v1/models/:model", get(models::retrieve))
        .route("/health", get(models::health))
        .route("/v1/health", get(models::health))
        .route("/v1/capabilities", get(models::capabilities))
        .route("/metrics", get(models::metrics))
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/completions", post(completions::completions))
        .route("/tokenize", post(tokenizer::tokenize))
        .route("/v1/tokenize", post(tokenizer::tokenize))
        .route("/detokenize", post(tokenizer::detokenize))
        .route("/v1/detokenize", post(tokenizer::detokenize))
        .route("/v1/responses", post(responses::create))
        .route(
            "/v1/responses/:response_id",
            get(responses::get).delete(responses::delete),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state.clone());

    if let Some(origin) = state.cors_allow_origin.as_deref() {
        if origin == "*" {
            app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
        } else if let Ok(header) = origin.parse::<HeaderValue>() {
            app.layer(
                CorsLayer::new()
                    .allow_origin(header)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
        } else {
            tracing::warn!(origin, "ignoring invalid CORS allow-origin");
            app
        }
    } else {
        app
    }
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
    // HTTP auth schemes are case-insensitive per RFC 7235; accept both
    // `Bearer <token>` and `bearer <token>` (and any other casing).
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let mut parts = v.trim().splitn(2, ' ');
            let scheme = parts.next()?;
            let token = parts.next()?;
            if !scheme.eq_ignore_ascii_case("Bearer") {
                return None;
            }
            Some(token.trim().to_string())
        });
    match got {
        Some(k) if k == expected => Ok(next.run(request).await),
        _ => Err(ApiError::unauthorized("missing or invalid API key")),
    }
}

// Keep the envelope types referenced so unused-import lints stay quiet when
// a future route wants to hand-roll an error body.
#[allow(dead_code)]
fn _types_kept_live(_a: ApiErrorBody, _b: ApiErrorEnvelope, _s: StatusCode) {}
