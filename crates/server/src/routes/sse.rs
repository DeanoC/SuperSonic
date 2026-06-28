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
                GenEvent::Done { reason, stats } => {
                    yield Ok(json_event(&done_chunk(
                        reason,
                        Some(Usage::from_generation_stats(&stats)),
                    )));
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
