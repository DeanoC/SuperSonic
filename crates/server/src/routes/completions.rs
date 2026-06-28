//! `POST /v1/completions` — raw text prompt, no chat template.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;

use super::chat::{cache_request, generation_error, queue_error, usage};
use super::sse;
use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::generate::{self, GenParams};
use crate::ids;
use crate::schemas::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamChunk, SinglePrompt,
};
use crate::state::ServerState;

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    validate_model(req.model.as_deref(), &state.model_id)?;
    validate_unsupported(&req)?;
    let prompt = req.prompt.clone().into_single().ok_or_else(|| {
        ApiError::bad_request("prompt must be a string, token array, or single-element array")
    })?;

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };
    // Tokenize + bounds-check synchronously so setup failures become
    // structured HTTP errors (400), not in-band SSE error events.
    let prompt_ids = match prompt {
        SinglePrompt::Text(prompt) => {
            // Raw-prompt path: let the tokenizer add its own BOS etc.
            generate::prepare(&state, &prompt, true, params.max_tokens)
        }
        SinglePrompt::Tokens(prompt_ids) => {
            generate::prepare_ids(&state, prompt_ids, params.max_tokens)
        }
    }
    .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let id = ids::completion_id();
    let created = ids::epoch_secs();
    let model = state.model_id.clone();
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    if req.stream {
        let cache = cache_request(
            &state,
            req.user.as_deref(),
            req.metadata.as_ref(),
            req.prompt_cache_key.as_deref(),
            req.prompt_cache_retention.as_deref(),
        );
        let rx = generate::spawn(state.clone(), prompt_ids, params, cache).map_err(queue_error)?;
        let stream = completion_sse_stream(rx, id, created, model, include_usage);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let cache = cache_request(
            &state,
            req.user.as_deref(),
            req.metadata.as_ref(),
            req.prompt_cache_key.as_deref(),
            req.prompt_cache_retention.as_deref(),
        );
        let rx = generate::spawn(state.clone(), prompt_ids, params, cache).map_err(queue_error)?;
        let result = generate::collect(rx).await.map_err(generation_error)?;
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
            usage: usage(&result.stats),
        };
        Ok(Json(resp).into_response())
    }
}

fn completion_sse_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<generate::GenEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
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
            usage: None,
        },
        move |reason, usage| CompletionStreamChunk {
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
            usage: if include_usage { usage } else { None },
        },
    )
}

fn validate_unsupported(req: &CompletionRequest) -> Result<(), ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("n > 1 is not supported"));
    }
    if req.logprobs.is_some() {
        return Err(ApiError::bad_request("logprobs are not supported"));
    }
    if req.echo.unwrap_or(false) {
        return Err(ApiError::bad_request("echo=true is not supported"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn rejects_unsupported_completion_options() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hi",
            "n": 2
        }))
        .unwrap();
        assert!(validate_unsupported(&req).is_err());

        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hi",
            "logprobs": 1
        }))
        .unwrap();
        assert!(validate_unsupported(&req).is_err());

        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hi",
            "echo": true
        }))
        .unwrap();
        assert!(validate_unsupported(&req).is_err());
    }
}
