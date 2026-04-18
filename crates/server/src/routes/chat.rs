//! `POST /v1/chat/completions` — streaming and non-streaming.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream};
use futures::StreamExt;

use crate::chat_template::ChatMessage;
use crate::errors::ApiError;
use crate::generate::{self, GenEvent, GenParams};
use crate::schemas::{
    ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatStreamChoice, ChatStreamChunk, ChatStreamDelta, Usage,
};
use crate::state::ServerState;

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let template = state
        .chat_template
        .clone()
        .ok_or_else(|| ApiError::bad_request(
            "this model has no chat_template; use /v1/completions with a raw prompt instead",
        ))?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let messages: Vec<ChatMessage> = req.messages.into_iter().map(Into::into).collect();
    let prompt_text = template
        .render(&messages, true)
        .map_err(|e| ApiError::bad_request(format!("chat template render failed: {e}")))?;

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };

    // Chat templates typically emit their own BOS; avoid doubling it up.
    let add_special_tokens = false;

    let id = make_id();
    let created = epoch_secs();
    let model = state.model_id.clone();

    if req.stream {
        let rx = generate::spawn(state.clone(), prompt_text, add_special_tokens, params);
        let stream = chat_sse_stream(rx, id, created, model);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let rx = generate::spawn(state.clone(), prompt_text, add_special_tokens, params);
        let result = generate::collect(rx)
            .await
            .map_err(|e| ApiError::internal(format!("generation failed: {e}")))?;
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant",
                    content: result.text,
                },
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

fn chat_sse_stream(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GenEvent>,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let role_chunk = ChatStreamChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.clone(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let role_event = Event::default().data(serde_json::to_string(&role_chunk).unwrap());

    let body = async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(text) => {
                    let chunk = ChatStreamChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                }
                GenEvent::Done { reason, .. } => {
                    let chunk = ChatStreamChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta { role: None, content: None },
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
    };

    stream::once(async move { Ok(role_event) }).chain(body)
}

fn make_id() -> String {
    let ts = epoch_secs();
    format!("chatcmpl-{ts:x}{:04x}", rand::random::<u16>())
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
