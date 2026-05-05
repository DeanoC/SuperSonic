use std::convert::Infallible;

use axum::response::sse::Event;
use futures::Stream;
use serde::Serialize;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::generate::{FinishReason, GenEvent};
use crate::schemas::Usage;

pub type SseEvent = Result<Event, Infallible>;

pub(super) fn json_event<T: Serialize>(payload: &T) -> Event {
    Event::default().data(serde_json::to_string(payload).unwrap())
}

pub(super) fn generation_events<T, D, FT, FD>(
    mut rx: UnboundedReceiver<GenEvent>,
    mut token_chunk: FT,
    mut done_chunk: FD,
) -> impl Stream<Item = SseEvent>
where
    T: Serialize + 'static,
    D: Serialize + 'static,
    FT: FnMut(String) -> T + 'static,
    FD: FnMut(FinishReason, Option<Usage>) -> D + 'static,
{
    async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(text) => {
                    yield Ok::<_, Infallible>(json_event(&token_chunk(text)));
                }
                GenEvent::Done {
                    reason,
                    prompt_tokens,
                    completion_tokens,
                    cached_prompt_tokens,
                } => {
                    yield Ok(json_event(&done_chunk(reason, Some(Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        prompt_tokens_details: Some(crate::schemas::PromptTokensDetails {
                            cached_tokens: cached_prompt_tokens,
                        }),
                    }))));
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
