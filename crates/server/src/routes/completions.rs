//! `POST /v1/completions` — raw text prompt, no chat template.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;

use super::sse;
use crate::errors::ApiError;
use crate::generate::{self, GenParams};
use crate::ids;
use crate::schemas::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamChunk, Usage,
};
use crate::state::ServerState;

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let prompt =
        req.prompt.clone().into_single().ok_or_else(|| {
            ApiError::bad_request("prompt must be a string or single-element array")
        })?;

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };
    // Raw-prompt path: let the tokenizer add its own BOS etc.
    let add_special_tokens = true;

    // Tokenize + bounds-check synchronously so setup failures become
    // structured HTTP errors (400), not in-band SSE error events.
    let prompt_ids = generate::prepare(&state, &prompt, add_special_tokens, params.max_tokens)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let id = ids::completion_id();
    let created = ids::epoch_secs();
    let model = state.model_id.clone();

    if req.stream {
        let rx = generate::spawn(state.clone(), prompt_ids, params);
        let stream = completion_sse_stream(rx, id, created, model);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let rx = generate::spawn(state.clone(), prompt_ids, params);
        let result = generate::collect(rx)
            .await
            .map_err(|e| ApiError::internal(format!("generation failed: {e}")))?;
        let resp = CompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: vec![CompletionChoice {
                text: result.text,
                index: 0,
                logprobs: None,
                finish_reason: result.finish.as_str(),
            }],
            usage: Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
            },
        };
        Ok(Json(resp).into_response())
    }
}

fn completion_sse_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<generate::GenEvent>,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = sse::SseEvent> {
    let token_id = id.clone();
    let token_model = model.clone();
    sse::generation_events(
        rx,
        move |text| CompletionStreamChunk {
            id: token_id.clone(),
            object: "text_completion",
            created,
            model: token_model.clone(),
            choices: vec![CompletionStreamChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: None,
            }],
        },
        move |reason| CompletionStreamChunk {
            id: id.clone(),
            object: "text_completion",
            created,
            model: model.clone(),
            choices: vec![CompletionStreamChoice {
                text: String::new(),
                index: 0,
                logprobs: None,
                finish_reason: Some(reason.as_str()),
            }],
        },
    )
}
