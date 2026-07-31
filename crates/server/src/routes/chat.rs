//! `POST /v1/chat/completions` — streaming and non-streaming.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream};
use futures::StreamExt;
use serde_json::{json, Value};

use super::sse;
use crate::chat_template::{ChatMessage, IncomingChatMessage, RenderOptions};
use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::generate::{self, GenParams};
use crate::ids;
use crate::output::{
    parse_assistant_output_with_context, AssistantOutput, AssistantOutputParseContext,
};
use crate::prefix_cache::{self, CacheRequest, CacheRetention};
use crate::schemas::{
    ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatStreamChoice, ChatStreamChunk, ChatStreamDelta, Usage,
};
use crate::state::ServerState;

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let template = state.chat_template.clone().ok_or_else(|| {
        ApiError::bad_request(
            "this model has no chat_template; use /v1/completions with a raw prompt instead",
        )
    })?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    validate_model(req.model.as_deref(), &state.model_id)?;
    validate_unsupported(&req)?;

    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let max_tokens = req.max_tokens.unwrap_or(256);
    let enable_thinking = reasoning_enabled(req.reasoning_effort.as_deref());
    let mut messages = normalize_messages(req.messages)?;
    apply_response_format_hint(
        &mut messages,
        response_format_json_object(req.response_format.as_ref())?,
    );
    let prompt_text = template
        .render_with_options(
            &messages,
            RenderOptions {
                add_generation_prompt: true,
                tools: req.tools.clone(),
                enable_thinking,
            },
        )
        .map_err(|e| ApiError::bad_request(format!("chat template render failed: {e}")))?;
    let output_context = AssistantOutputParseContext::from_rendered_prompt(&prompt_text);

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens,
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };

    let prompt_ids = generate::prepare(&state, &prompt_text, false, params.max_tokens)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let id = ids::chat_completion_id();
    let created = ids::epoch_secs();
    let model = state.model_id.clone();

    if req.stream {
        let cache = cache_request(
            &state,
            req.user.as_deref(),
            req.metadata.as_ref(),
            req.prompt_cache_key.as_deref(),
            req.prompt_cache_retention.as_deref(),
        );
        let rx = generate::spawn(state.clone(), prompt_ids, params, cache).map_err(queue_error)?;
        let stream = chat_sse_stream(rx, id, created, model, include_usage, output_context);
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
        let parsed = parse_assistant_output_with_context(&result.text, output_context);
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                finish_reason: finish_reason(result.finish.as_str(), &parsed),
                message: ChatCompletionMessage {
                    role: "assistant",
                    content: message_content(&parsed),
                    reasoning_content: parsed.reasoning_content,
                    tool_calls: parsed.tool_calls,
                },
            }],
            usage: usage(&result.stats),
        };
        Ok(Json(resp).into_response())
    }
}

pub(crate) fn usage(stats: &generate::GenerationStats) -> Usage {
    Usage::from_generation_stats(stats)
}

pub(crate) fn cache_request(
    state: &ServerState,
    user: Option<&str>,
    metadata: Option<&Value>,
    prompt_cache_key: Option<&str>,
    retention: Option<&str>,
) -> Option<CacheRequest> {
    let retention = CacheRetention::from_openai(retention);
    if retention == CacheRetention::None || !state.prefix_cache.config().enabled {
        return None;
    }
    let thread = metadata
        .and_then(|m| {
            m.get("thread_id")
                .or_else(|| m.get("conversation_id"))
                .or_else(|| m.get("session_id"))
        })
        .and_then(Value::as_str);
    let scoped_user = user.or(thread);
    Some(CacheRequest {
        key: prompt_cache_key
            .map(ToOwned::to_owned)
            .or_else(|| thread.map(ToOwned::to_owned)),
        retention,
        scope: prefix_cache::scope_from_parts(
            &state.model_id,
            state.api_key.as_deref(),
            scoped_user,
        ),
    })
}

pub(crate) fn queue_error(err: anyhow::Error) -> ApiError {
    let msg = err.to_string();
    if msg.contains("generation queue full") {
        ApiError::too_many_requests(msg)
    } else {
        ApiError::internal(msg)
    }
}

pub(crate) fn generation_error(err: anyhow::Error) -> ApiError {
    let msg = err.to_string();
    if msg.contains("generation queue timeout") || msg.contains("generation scheduler closed") {
        ApiError::unavailable(format!("generation failed: {msg}"))
    } else {
        ApiError::internal(format!("generation failed: {msg}"))
    }
}

fn chat_sse_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<generate::GenEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    output_context: AssistantOutputParseContext,
) -> impl Stream<Item = sse::SseEvent> {
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
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
    };
    let role_event = sse::json_event(&role_chunk);
    let body = parsed_chat_events(
        rx,
        id.clone(),
        model.clone(),
        created,
        include_usage,
        output_context,
    );
    stream::once(async move { Ok(role_event) }).chain(body)
}

fn parsed_chat_events(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<generate::GenEvent>,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    output_context: AssistantOutputParseContext,
) -> impl Stream<Item = sse::SseEvent> {
    async_stream::stream! {
        let mut raw = String::new();
        let mut emitted_content = String::new();
        let mut emitted_reasoning = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                generate::GenEvent::Token(text) => {
                    raw.push_str(&text);
                    if has_incomplete_control_block(&raw, output_context) {
                        continue;
                    }
                    let parsed = parse_assistant_output_with_context(&raw, output_context);
                    for delta in output_deltas(
                        &parsed,
                        &mut emitted_reasoning,
                        &mut emitted_content,
                    ) {
                        yield Ok(sse::json_event(&chat_chunk(
                            &id,
                            &model,
                            created,
                            delta,
                            None,
                            None,
                        )));
                    }
                }
                generate::GenEvent::Done { reason, stats } => {
                    let parsed = parse_assistant_output_with_context(&raw, output_context);
                    for delta in output_deltas(
                        &parsed,
                        &mut emitted_reasoning,
                        &mut emitted_content,
                    ) {
                        yield Ok(sse::json_event(&chat_chunk(
                            &id,
                            &model,
                            created,
                            delta,
                            None,
                            None,
                        )));
                    }
                    let done_reason = finish_reason(reason.as_str(), &parsed);
                    yield Ok(sse::json_event(&chat_chunk(
                        &id,
                        &model,
                        created,
                        ChatStreamDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: parsed.tool_calls,
                        },
                        Some(done_reason),
                        None,
                    )));
                    if include_usage {
                        yield Ok(sse::json_event(&ChatStreamChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: Vec::new(),
                            usage: Some(usage(&stats)),
                        }));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                generate::GenEvent::Error(msg) => {
                    let payload = json!({ "error": { "message": msg, "type": "internal_error" } });
                    yield Ok(Event::default().data(payload.to_string()));
                    return;
                }
            }
        }
    }
}

fn output_deltas(
    parsed: &AssistantOutput,
    emitted_reasoning: &mut String,
    emitted_content: &mut String,
) -> Vec<ChatStreamDelta> {
    let mut deltas = Vec::new();
    if let Some(reasoning) = parsed.reasoning_content.as_ref() {
        if let Some(delta) = reasoning.strip_prefix(emitted_reasoning.as_str()) {
            if !delta.is_empty() {
                *emitted_reasoning = reasoning.clone();
                deltas.push(ChatStreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(delta.to_string()),
                    tool_calls: None,
                });
            }
        }
    }
    if let Some(delta) = parsed.content.strip_prefix(emitted_content.as_str()) {
        if !delta.is_empty() {
            *emitted_content = parsed.content.clone();
            deltas.push(ChatStreamDelta {
                role: None,
                content: Some(delta.to_string()),
                reasoning_content: None,
                tool_calls: None,
            });
        }
    }
    deltas
}

fn chat_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: ChatStreamDelta,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
) -> ChatStreamChunk {
    ChatStreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }
}

fn finish_reason(default: &'static str, parsed: &AssistantOutput) -> &'static str {
    if parsed.tool_calls.is_some() {
        "tool_calls"
    } else {
        default
    }
}

fn message_content(parsed: &AssistantOutput) -> Option<String> {
    if parsed.tool_calls.is_some() && parsed.content.trim().is_empty() {
        None
    } else {
        Some(parsed.content.clone())
    }
}

fn validate_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("n>1 is not supported"));
    }
    if req.logprobs.unwrap_or(false) {
        return Err(ApiError::bad_request("logprobs=true is not supported"));
    }
    if matches!(req.tool_choice.as_ref(), Some(Value::String(s)) if s == "required") {
        return Err(ApiError::bad_request(
            "tool_choice=required is not supported",
        ));
    }
    validate_tools(req.tools.as_ref())?;
    response_format_json_object(req.response_format.as_ref())?;
    Ok(())
}

pub(crate) fn validate_tools(tools: Option<&Value>) -> Result<(), ApiError> {
    let Some(tools) = tools else {
        return Ok(());
    };
    let Value::Array(items) = tools else {
        return Err(ApiError::bad_request("tools must be an array"));
    };
    for item in items {
        let typ = item.get("type").and_then(Value::as_str).unwrap_or("");
        if typ != "function" {
            return Err(ApiError::bad_request(format!(
                "unsupported tool type: {typ}"
            )));
        }
        if item.get("function").and_then(|f| f.get("name")).is_none() {
            return Err(ApiError::bad_request("function tool missing function.name"));
        }
    }
    Ok(())
}

pub(crate) fn reasoning_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(|s| s.to_ascii_lowercase()),
        Some(v) if !matches!(v.as_str(), "none" | "off" | "false" | "disabled")
    )
}

pub(crate) fn response_format_json_object(format: Option<&Value>) -> Result<bool, ApiError> {
    let Some(format) = format else {
        return Ok(false);
    };
    if format.is_null() {
        return Ok(false);
    }
    if let Some(s) = format.as_str() {
        return match s {
            "text" => Ok(false),
            "json_object" => Ok(true),
            "json_schema" => Err(ApiError::bad_request(
                "response_format=json_schema is not supported",
            )),
            other => Err(ApiError::bad_request(format!(
                "unsupported response_format: {other}"
            ))),
        };
    }
    let typ = format
        .get("type")
        .or_else(|| format.get("format").and_then(|f| f.get("type")))
        .and_then(Value::as_str)
        .unwrap_or("text");
    match typ {
        "text" => Ok(false),
        "json_object" => Ok(true),
        "json_schema" => Err(ApiError::bad_request(
            "response_format=json_schema is not supported",
        )),
        other => Err(ApiError::bad_request(format!(
            "unsupported response_format type: {other}"
        ))),
    }
}

pub(crate) fn apply_response_format_hint(messages: &mut Vec<ChatMessage>, json_object: bool) {
    if !json_object {
        return;
    }
    const HINT: &str = "Return a valid JSON object.";
    if let Some(first) = messages.first_mut() {
        if first.role == "system" {
            let content = content_to_text(&first.content).unwrap_or_default();
            first.content = Value::String(if content.trim().is_empty() {
                HINT.to_string()
            } else {
                format!("{content}\n\n{HINT}")
            });
            return;
        }
    }
    messages.insert(0, ChatMessage::text("system", HINT));
}

pub(crate) fn has_incomplete_control_block(
    raw: &str,
    output_context: AssistantOutputParseContext,
) -> bool {
    output_context.has_incomplete_think(raw)
        || (raw.contains("<tool_call>") && !raw.contains("</tool_call>"))
        || ["<tool_call>", "</tool_call>"]
            .iter()
            .any(|tag| (1..tag.len()).any(|len| raw.ends_with(&tag[..len])))
}

pub(crate) fn normalize_messages(
    messages: Vec<IncomingChatMessage>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut out = Vec::new();
    for msg in messages {
        let mut role = msg.role.to_ascii_lowercase();
        if role == "developer" {
            role = "system".to_string();
        } else if role == "function" {
            role = "tool".to_string();
        }
        let content = content_to_text(&msg.content)?;
        out.push(ChatMessage {
            role,
            content: Value::String(content),
            reasoning_content: msg.reasoning_content,
            tool_calls: normalize_tool_calls_for_template(msg.tool_calls),
            tool_call_id: msg.tool_call_id,
        });
    }
    Ok(out)
}

pub(crate) fn normalize_tool_calls_for_template(tool_calls: Option<Value>) -> Option<Value> {
    let mut calls = tool_calls?;
    let Value::Array(items) = &mut calls else {
        return Some(calls);
    };
    for item in items {
        let Some(args) = item
            .get_mut("function")
            .and_then(|f| f.get_mut("arguments"))
        else {
            continue;
        };
        if let Some(s) = args.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                *args = parsed;
            }
        }
    }
    Some(calls)
}

pub(crate) fn content_to_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                let typ = item.get("type").and_then(Value::as_str).unwrap_or("");
                match typ {
                    "text" | "input_text" | "output_text" => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            out.push_str(text);
                        }
                    }
                    "image" | "image_url" | "input_image" | "file" | "input_file" => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported_content_type: {typ}"
                        )));
                    }
                    _ if item.get("text").is_some() => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            out.push_str(text);
                        }
                    }
                    _ => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported_content_type: {typ}"
                        )));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(
            "message content must be a string or text content array",
        )),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn text_content_parts_are_joined() {
        let text = content_to_text(&json!([
            {"type": "text", "text": "hello "},
            {"type": "input_text", "text": "world"}
        ]))
        .unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn image_content_parts_are_rejected() {
        let err = content_to_text(&json!([
            {"type": "input_image", "image_url": "data:"}
        ]))
        .expect_err("expected unsupported image");
        assert!(err.body.message.contains("unsupported_content_type"));
    }

    #[test]
    fn developer_messages_are_normalized_to_system() {
        let messages = vec![
            IncomingChatMessage {
                role: "developer".to_string(),
                content: json!("follow policy"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            IncomingChatMessage {
                role: "user".to_string(),
                content: json!("hi"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let out = normalize_messages(messages).unwrap();
        assert_eq!(out[0].role, "system");
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn system_messages_preserve_original_order() {
        let messages = vec![
            IncomingChatMessage {
                role: "user".to_string(),
                content: json!("first"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            IncomingChatMessage {
                role: "developer".to_string(),
                content: json!("mid instruction"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            IncomingChatMessage {
                role: "assistant".to_string(),
                content: json!("second"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let out = normalize_messages(messages).unwrap();
        assert_eq!(out[0].role, "user");
        assert_eq!(out[1].role, "system");
        assert_eq!(out[2].role, "assistant");
        assert_eq!(content_to_text(&out[1].content).unwrap(), "mid instruction");
    }

    #[test]
    fn reasoning_defaults_off_and_can_be_enabled() {
        assert!(!reasoning_enabled(None));
        assert!(!reasoning_enabled(Some("none")));
        assert!(reasoning_enabled(Some("medium")));
    }

    #[test]
    fn response_format_json_object_adds_system_hint() {
        assert!(!response_format_json_object(Some(&json!({"type": "text"}))).unwrap());
        assert!(response_format_json_object(Some(&json!({"type": "json_object"}))).unwrap());
        assert!(response_format_json_object(Some(&json!({"type": "json_schema"}))).is_err());

        let mut messages = vec![ChatMessage::text("user", "hi")];
        apply_response_format_hint(&mut messages, true);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            content_to_text(&messages[0].content).unwrap(),
            "Return a valid JSON object."
        );
    }

    #[test]
    fn incomplete_think_and_tool_blocks_are_buffered() {
        let generated = AssistantOutputParseContext::default();
        let prefilled = AssistantOutputParseContext::from_rendered_prompt("assistant<think>\n");
        assert!(has_incomplete_control_block("<think>\npartial", generated));
        assert!(has_incomplete_control_block("partial", prefilled));
        assert!(has_incomplete_control_block(
            "x <tool_call>\npartial",
            generated
        ));
        assert!(!has_incomplete_control_block(
            "<think>x</think>\ny",
            generated
        ));
        assert!(!has_incomplete_control_block("x</think>\ny", prefilled));
        assert!(!has_incomplete_control_block(
            "<tool_call>x</tool_call>",
            generated
        ));
    }

    #[test]
    fn validates_function_tools_only() {
        validate_tools(Some(&json!([{
            "type": "function",
            "function": {"name": "lookup"}
        }])))
        .unwrap();
        let err = validate_tools(Some(&json!([{"type": "web_search"}])))
            .expect_err("expected unsupported built-in tool");
        assert!(err.body.message.contains("unsupported tool type"));
    }

    #[test]
    fn assistant_tool_call_arguments_are_template_objects() {
        let normalized = normalize_tool_calls_for_template(Some(json!([
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"query\":\"weather\"}"
                }
            }
        ])))
        .unwrap();
        assert_eq!(
            normalized[0]["function"]["arguments"]["query"]
                .as_str()
                .unwrap(),
            "weather"
        );
    }
}
