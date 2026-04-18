//! `POST /v1/completions` — raw text prompt, no chat template.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;

use crate::errors::ApiError;
use crate::generate::{self, GenEvent, GenParams};
use crate::schemas::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamChunk, Usage,
};
use crate::state::ServerState;

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let prompt = req
        .prompt
        .clone()
        .into_single()
        .ok_or_else(|| ApiError::bad_request("prompt must be a string or single-element array"))?;

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };
    // Raw-prompt path: let the tokenizer add its own BOS etc.
    let add_special_tokens = true;

    let id = make_id();
    let created = epoch_secs();
    let model = state.model_id.clone();

    if req.stream {
        let rx = generate::spawn(state.clone(), prompt, add_special_tokens, params);
        let stream = completion_sse_stream(rx, id, created, model);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let rx = generate::spawn(state.clone(), prompt, add_special_tokens, params);
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
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GenEvent>,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(text) => {
                    let chunk = CompletionStreamChunk {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: model.clone(),
                        choices: vec![CompletionStreamChoice {
                            text,
                            index: 0,
                            logprobs: None,
                            finish_reason: None,
                        }],
                    };
                    yield Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                }
                GenEvent::Done { reason, .. } => {
                    let chunk = CompletionStreamChunk {
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
                    };
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                GenEvent::Error(msg) => {
                    let payload = serde_json::json!({
                        "error": { "message": msg, "type": "internal_error" }
                    });
                    yield Ok(Event::default().data(payload.to_string()));
                    return;
                }
            }
        }
    }
}

fn make_id() -> String {
    let ts = epoch_secs();
    format!("cmpl-{ts:x}{:04x}", rand::random::<u16>())
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
