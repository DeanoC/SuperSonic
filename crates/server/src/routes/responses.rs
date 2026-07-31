//! Minimal OpenAI Responses API facade over SuperSonic's chat generation.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::chat_template::{ChatMessage, RenderOptions};
use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::generate::{self, GenParams};
use crate::output::{
    parse_assistant_output_with_context, AssistantOutput, AssistantOutputParseContext,
};
use crate::routes::chat::{
    apply_response_format_hint, cache_request, content_to_text, generation_error,
    has_incomplete_control_block, normalize_tool_calls_for_template, queue_error,
    reasoning_enabled, response_format_json_object, usage, validate_tools,
};
use crate::schemas::{OpenAiToolCall, StopParam, Usage};
use crate::state::ServerState;

static STORES: OnceLock<Mutex<HashMap<u64, ResponseStore>>> = OnceLock::new();

#[derive(Default)]
struct ResponseStore {
    entries: HashMap<String, StoredResponse>,
    order: VecDeque<String>,
    next_id: u64,
}

impl ResponseStore {
    fn next_response_id(&mut self) -> String {
        let seq = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        format!("resp-{:x}{seq:04x}", crate::ids::epoch_secs())
    }

    fn insert(&mut self, id: String, response: StoredResponse, max_entries: usize) {
        if max_entries == 0 {
            return;
        }
        if !self.entries.contains_key(&id) {
            self.order.push_back(id.clone());
        }
        self.entries.insert(id, response);
        while self.entries.len() > max_entries {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn get(&self, id: &str) -> Option<&StoredResponse> {
        self.entries.get(id)
    }

    fn remove(&mut self, id: &str) -> bool {
        let removed = self.entries.remove(id).is_some();
        if removed {
            self.order.retain(|k| k != id);
        }
        removed
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<ResponseOutputItem>,
    pub usage: Usage,
    #[serde(skip_serializing)]
    pub cache_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        content: Vec<ResponseContentPart>,
    },
    #[serde(rename = "reasoning")]
    Reasoning { id: String, summary: Vec<String> },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        annotations: Vec<Value>,
    },
}

#[derive(Debug, Deserialize)]
pub struct ResponseRequest {
    pub model: Option<String>,
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(
        default,
        alias = "max_tokens",
        alias = "max_completion_tokens",
        alias = "maxOutputTokens",
        alias = "maxCompletionTokens"
    )]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ResponseReasoning>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub prompt_cache_retention: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

pub async fn create(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ResponseRequest>,
) -> Result<Response, ApiError> {
    validate_model(req.model.as_deref(), &state.model_id)?;
    if matches!(req.tool_choice.as_ref(), Some(Value::String(s)) if s == "required") {
        return Err(ApiError::bad_request(
            "tool_choice=required is not supported",
        ));
    }
    validate_tools(req.tools.as_ref())?;
    let json_object = response_format_json_object(response_format_value(&req))?;

    let max_tokens = req.max_output_tokens.unwrap_or(256);
    let mut messages = response_messages(&req, state.server_instance_id)?;
    apply_response_format_hint(&mut messages, json_object);
    let prompt_text = if let Some(template) = state.chat_template.clone() {
        template
            .render_with_options(
                &messages,
                RenderOptions {
                    add_generation_prompt: true,
                    tools: req.tools.clone(),
                    enable_thinking: reasoning_enabled(
                        req.reasoning.as_ref().and_then(|r| r.effort.as_deref()),
                    ),
                },
            )
            .map_err(|e| ApiError::bad_request(format!("chat template render failed: {e}")))?
    } else if req.tools.is_some() {
        return Err(ApiError::bad_request(
            "tools require a model chat_template; use a chat-capable model",
        ));
    } else {
        messages
            .iter()
            .map(|m| content_to_text(&m.content))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
    };
    let output_context = if state.chat_template.is_some() {
        AssistantOutputParseContext::from_rendered_prompt(&prompt_text)
    } else {
        AssistantOutputParseContext::default()
    };

    let id = next_response_id(state.server_instance_id);
    let created_at = crate::ids::epoch_secs();
    let model = state.model_id.clone();
    let response_cache_key = response_cache_key(&req, state.server_instance_id, &id);

    let params = GenParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens,
        stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        seed: req.seed,
    };
    let prompt_ids = generate::prepare(
        &state,
        &prompt_text,
        state.chat_template.is_none(),
        params.max_tokens,
    )
    .map_err(|e| ApiError::bad_request(e.to_string()))?;

    if req.stream {
        let cache = cache_request(
            &state,
            req.user.as_deref(),
            req.metadata.as_ref(),
            response_cache_key.as_deref(),
            req.prompt_cache_retention.as_deref(),
        );
        let rx = generate::spawn(state.clone(), prompt_ids, params, cache).map_err(queue_error)?;
        let stream = response_sse_stream(
            rx,
            id,
            model,
            created_at,
            state.server_instance_id,
            state.response_store_max_entries,
            response_cache_key,
            output_context,
        );
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let cache = cache_request(
            &state,
            req.user.as_deref(),
            req.metadata.as_ref(),
            response_cache_key.as_deref(),
            req.prompt_cache_retention.as_deref(),
        );
        let rx = generate::spawn(state.clone(), prompt_ids, params, cache).map_err(queue_error)?;
        let result = generate::collect(rx).await.map_err(generation_error)?;
        let parsed = parse_assistant_output_with_context(&result.text, output_context);
        let stored = build_response(
            id,
            model,
            created_at,
            parsed,
            usage(&result.stats),
            response_cache_key,
        );
        insert_response(
            state.server_instance_id,
            stored.id.clone(),
            stored.clone(),
            state.response_store_max_entries,
        );
        Ok(Json(stored).into_response())
    }
}

pub async fn get(
    State(state): State<Arc<ServerState>>,
    Path(response_id): Path<String>,
) -> Result<Response, ApiError> {
    let Some(resp) = get_response(state.server_instance_id, &response_id) else {
        return Err(ApiError::bad_request("unknown response id"));
    };
    Ok(Json(resp).into_response())
}

pub async fn delete(
    State(state): State<Arc<ServerState>>,
    Path(response_id): Path<String>,
) -> Result<Response, ApiError> {
    let deleted = delete_response(state.server_instance_id, &response_id);
    Ok(Json(json!({
        "id": response_id,
        "object": "response.deleted",
        "deleted": deleted,
    }))
    .into_response())
}

fn response_sse_stream(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<generate::GenEvent>,
    id: String,
    model: String,
    created_at: u64,
    server_instance_id: u64,
    response_store_max_entries: usize,
    cache_key: Option<String>,
    output_context: AssistantOutputParseContext,
) -> impl Stream<Item = super::sse::SseEvent> {
    async_stream::stream! {
        yield Ok(Event::default()
            .event("response.created")
            .data(json!({ "type": "response.created", "response": {
                "id": id,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "in_progress",
            }}).to_string()));
        let mut raw = String::new();
        let mut emitted = String::new();
        let mut emitted_reasoning = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                generate::GenEvent::Token(text) => {
                    raw.push_str(&text);
                    if has_incomplete_control_block(&raw, output_context) {
                        continue;
                    }
                    let parsed = parse_assistant_output_with_context(&raw, output_context);
                    if let Some(reasoning) = parsed.reasoning_content.as_ref() {
                        if let Some(delta) = reasoning.strip_prefix(&emitted_reasoning) {
                            if !delta.is_empty() {
                                emitted_reasoning = reasoning.clone();
                                yield Ok(reasoning_delta_event(&id, delta));
                            }
                        }
                    }
                    if let Some(delta) = parsed.content.strip_prefix(&emitted) {
                        if !delta.is_empty() {
                            emitted = parsed.content.clone();
                            yield Ok(Event::default()
                                .event("response.output_text.delta")
                                .data(json!({
                                    "type": "response.output_text.delta",
                                    "response_id": id,
                                    "delta": delta,
                                }).to_string()));
                        }
                    }
                }
                generate::GenEvent::Done { stats, .. } => {
                    let parsed = parse_assistant_output_with_context(&raw, output_context);
                    if let Some(reasoning) = parsed.reasoning_content.as_ref() {
                        if let Some(delta) = reasoning.strip_prefix(&emitted_reasoning) {
                            if !delta.is_empty() {
                                emitted_reasoning = reasoning.clone();
                                yield Ok(reasoning_delta_event(&id, delta));
                            }
                        }
                    }
                    if let Some(delta) = parsed.content.strip_prefix(&emitted) {
                        if !delta.is_empty() {
                            emitted = parsed.content.clone();
                            yield Ok(Event::default()
                                .event("response.output_text.delta")
                                .data(json!({
                                    "type": "response.output_text.delta",
                                    "response_id": id,
                                    "delta": delta,
                                }).to_string()));
                        }
                    }
                    let stored = build_response(
                        id.clone(),
                        model.clone(),
                        created_at,
                        parsed,
                        usage(&stats),
                        cache_key.clone(),
                    );
                    insert_response(
                        server_instance_id,
                        stored.id.clone(),
                        stored.clone(),
                        response_store_max_entries,
                    );
                    if !emitted_reasoning.is_empty() {
                        yield Ok(Event::default()
                            .event("response.reasoning_summary_text.done")
                            .data(json!({
                                "type": "response.reasoning_summary_text.done",
                                "item_id": format!("{id}-rsn"),
                                "output_index": 0,
                                "summary_index": 0,
                                "text": emitted_reasoning,
                            }).to_string()));
                    }
                    yield Ok(Event::default()
                        .event("response.output_text.done")
                        .data(json!({
                            "type": "response.output_text.done",
                            "response_id": id,
                            "text": emitted,
                        }).to_string()));
                    for item in &stored.output {
                        yield Ok(Event::default()
                            .event("response.output_item.done")
                            .data(json!({
                                "type": "response.output_item.done",
                                "response_id": id,
                                "item": item,
                            }).to_string()));
                    }
                    yield Ok(Event::default()
                        .event("response.completed")
                        .data(json!({
                            "type": "response.completed",
                            "response": stored,
                        }).to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                generate::GenEvent::Error(msg) => {
                    yield Ok(Event::default()
                        .event("response.failed")
                        .data(json!({ "type": "response.failed", "error": { "message": msg } }).to_string()));
                    return;
                }
            }
        }
    }
}

fn reasoning_delta_event(id: &str, delta: &str) -> Event {
    Event::default()
        .event("response.reasoning_summary_text.delta")
        .data(
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": format!("{id}-rsn"),
                "output_index": 0,
                "summary_index": 0,
                "delta": delta,
            })
            .to_string(),
        )
}

fn response_messages(
    req: &ResponseRequest,
    server_instance_id: u64,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();
    if let Some(id) = req.previous_response_id.as_ref() {
        let prev = get_response(server_instance_id, id)
            .ok_or_else(|| ApiError::bad_request("unknown previous_response_id"))?;
        append_previous_response_messages(&mut messages, &prev);
    }
    if let Some(instructions) = req.instructions.as_ref() {
        messages.push(ChatMessage::text("system", instructions));
    }
    match &req.input {
        Value::String(s) => messages.push(ChatMessage::text("user", s)),
        Value::Array(items) => {
            for item in items {
                if item.get("type").and_then(Value::as_str) == Some("message") {
                    let role = normalize_response_role(
                        item.get("role").and_then(Value::as_str).unwrap_or("user"),
                    );
                    let content = item.get("content").unwrap_or(&Value::Null);
                    messages.push(ChatMessage::text(role, content_to_text(content)?));
                } else if item.get("type").and_then(Value::as_str) == Some("input_text") {
                    messages.push(ChatMessage::text(
                        "user",
                        item.get("text").and_then(Value::as_str).unwrap_or_default(),
                    ));
                } else if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
                    let output = item
                        .get("output")
                        .or_else(|| item.get("content"))
                        .unwrap_or(&Value::Null);
                    let output_text = output
                        .as_str()
                        .map(ToOwned::to_owned)
                        .map(Ok)
                        .unwrap_or_else(|| content_to_text(output))?;
                    let mut msg = ChatMessage::text("tool", output_text);
                    msg.tool_call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    messages.push(msg);
                } else if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    messages.push(function_call_input_message(item)?);
                } else if let Some(role) = item.get("role").and_then(Value::as_str) {
                    let role = normalize_response_role(role);
                    let content = item.get("content").unwrap_or(&Value::Null);
                    messages.push(ChatMessage::text(role, content_to_text(content)?));
                } else {
                    return Err(ApiError::bad_request("unsupported Responses input item"));
                }
            }
        }
        _ => {
            return Err(ApiError::bad_request(
                "input must be a string or message array",
            ))
        }
    }
    Ok(messages)
}

fn append_previous_response_messages(messages: &mut Vec<ChatMessage>, prev: &StoredResponse) {
    let text = output_text(prev).unwrap_or_default();
    let tool_calls = response_tool_calls(prev);
    if !text.is_empty() || tool_calls.as_ref().is_some_and(|v| !v.is_empty()) {
        let mut msg = ChatMessage::text("assistant", text);
        if let Some(calls) = tool_calls {
            msg.tool_calls = normalize_tool_calls_for_template(Some(Value::Array(calls)));
        }
        messages.push(msg);
    }
}

fn response_tool_calls(resp: &StoredResponse) -> Option<Vec<Value>> {
    let calls: Vec<Value> = resp
        .output
        .iter()
        .filter_map(|item| {
            if let ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } = item
            {
                Some(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }))
            } else {
                None
            }
        })
        .collect();
    (!calls.is_empty()).then_some(calls)
}

fn function_call_input_message(item: &Value) -> Result<ChatMessage, ApiError> {
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("call_0");
    let name = item
        .get("name")
        .or_else(|| item.get("function").and_then(|f| f.get("name")))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("function_call input missing name"))?;
    let arguments = item
        .get("arguments")
        .or_else(|| item.get("function").and_then(|f| f.get("arguments")))
        .cloned()
        .unwrap_or_else(|| json!("{}"));
    let arguments = if arguments.is_string() {
        arguments
    } else {
        Value::String(arguments.to_string())
    };
    let mut msg = ChatMessage::text("assistant", "");
    msg.tool_calls = normalize_tool_calls_for_template(Some(json!([{
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }])));
    Ok(msg)
}

fn normalize_response_role(role: &str) -> &str {
    match role {
        "developer" => "system",
        "function" => "tool",
        other => other,
    }
}

fn response_format_value(req: &ResponseRequest) -> Option<&Value> {
    req.response_format
        .as_ref()
        .or_else(|| req.text.as_ref().and_then(|text| text.get("format")))
}

fn response_cache_key(
    req: &ResponseRequest,
    server_instance_id: u64,
    current_response_id: &str,
) -> Option<String> {
    if let Some(key) = req.prompt_cache_key.as_ref() {
        return Some(key.clone());
    }
    if let Some(previous_response_id) = req.previous_response_id.as_ref() {
        return get_response(server_instance_id, previous_response_id)
            .and_then(|prev| prev.cache_key)
            .or_else(|| Some(previous_response_id.clone()));
    }
    Some(current_response_id.to_string())
}

fn build_response(
    id: String,
    model: String,
    created_at: u64,
    parsed: AssistantOutput,
    usage: Usage,
    cache_key: Option<String>,
) -> StoredResponse {
    let mut output = Vec::new();
    if let Some(reasoning) = parsed.reasoning_content {
        output.push(ResponseOutputItem::Reasoning {
            id: format!("{id}-rsn"),
            summary: vec![reasoning],
        });
    }
    if !parsed.content.trim().is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("{id}-msg"),
            role: "assistant",
            content: vec![ResponseContentPart::OutputText {
                text: parsed.content,
                annotations: Vec::new(),
            }],
        });
    }
    if let Some(calls) = parsed.tool_calls {
        append_tool_calls(&mut output, calls);
    }
    StoredResponse {
        id,
        object: "response",
        created_at,
        model,
        status: "completed",
        output,
        usage,
        cache_key,
    }
}

fn append_tool_calls(output: &mut Vec<ResponseOutputItem>, calls: Vec<OpenAiToolCall>) {
    for call in calls {
        output.push(ResponseOutputItem::FunctionCall {
            id: call.id.clone(),
            call_id: call.id,
            name: call.function.name,
            arguments: call.function.arguments,
            status: "completed",
        });
    }
}

fn output_text(resp: &StoredResponse) -> Option<String> {
    let mut text = String::new();
    for item in &resp.output {
        if let ResponseOutputItem::Message { content, .. } = item {
            for part in content {
                let ResponseContentPart::OutputText { text: t, .. } = part;
                text.push_str(t);
            }
        }
    }
    (!text.is_empty()).then_some(text)
}

fn stores() -> &'static Mutex<HashMap<u64, ResponseStore>> {
    STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_response_id(server_instance_id: u64) -> String {
    let mut stores = stores().lock().unwrap();
    stores
        .entry(server_instance_id)
        .or_default()
        .next_response_id()
}

fn insert_response(
    server_instance_id: u64,
    id: String,
    response: StoredResponse,
    max_entries: usize,
) {
    stores()
        .lock()
        .unwrap()
        .entry(server_instance_id)
        .or_default()
        .insert(id, response, max_entries);
}

fn get_response(server_instance_id: u64, id: &str) -> Option<StoredResponse> {
    stores()
        .lock()
        .unwrap()
        .get(&server_instance_id)
        .and_then(|store| store.get(id).cloned())
}

fn delete_response(server_instance_id: u64, id: &str) -> bool {
    stores()
        .lock()
        .unwrap()
        .get_mut(&server_instance_id)
        .is_some_and(|store| store.remove(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::parse_assistant_output;

    #[test]
    fn response_output_contains_reasoning_text_and_function_call() {
        let raw = "<think>plan</think>\nanswer\n<tool_call>\n<function=lookup>\n<parameter=q>\n42\n</parameter>\n</function>\n</tool_call>";
        let resp = build_response(
            "resp-test".to_string(),
            "model".to_string(),
            1,
            parse_assistant_output(raw),
            Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                prompt_tokens_details: None,
            },
            Some("thread-cache".to_string()),
        );
        assert!(matches!(
            resp.output[0],
            ResponseOutputItem::Reasoning { .. }
        ));
        assert!(matches!(resp.output[1], ResponseOutputItem::Message { .. }));
        assert!(matches!(
            resp.output[2],
            ResponseOutputItem::FunctionCall { .. }
        ));
    }

    #[test]
    fn response_store_evicts_oldest_entry() {
        let mut store = ResponseStore::default();
        store.insert("first".to_string(), empty_response("first"), 1);
        store.insert("second".to_string(), empty_response("second"), 1);

        assert!(store.get("first").is_none());
        assert!(store.get("second").is_some());
    }

    #[test]
    fn response_cache_key_inherits_previous_chain_key() {
        let server_instance_id = 900_001;
        let first_id = "resp-first".to_string();
        insert_response(
            server_instance_id,
            first_id.clone(),
            empty_response_with_cache_key(&first_id, Some("resp-first")),
            8,
        );
        let req = ResponseRequest {
            model: None,
            input: json!("next"),
            instructions: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: false,
            stop: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            text: None,
            reasoning: None,
            previous_response_id: Some(first_id),
            user: None,
            metadata: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
        };

        assert_eq!(
            response_cache_key(&req, server_instance_id, "resp-second").as_deref(),
            Some("resp-first")
        );
    }

    #[test]
    fn response_cache_key_prefers_explicit_request_key() {
        let req = ResponseRequest {
            model: None,
            input: json!("next"),
            instructions: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: false,
            stop: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            text: None,
            reasoning: None,
            previous_response_id: Some("resp-prev".to_string()),
            user: None,
            metadata: None,
            prompt_cache_key: Some("stable-thread".to_string()),
            prompt_cache_retention: None,
        };

        assert_eq!(
            response_cache_key(&req, 0, "resp-current").as_deref(),
            Some("stable-thread")
        );
    }

    #[test]
    fn response_messages_accept_developer_and_text_parts() {
        let req = ResponseRequest {
            model: None,
            input: json!([
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "be concise"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}]
                }
            ]),
            instructions: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: false,
            stop: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            text: None,
            reasoning: None,
            previous_response_id: None,
            user: None,
            metadata: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
        };
        let messages = response_messages(&req, 0).unwrap();
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn response_messages_accept_direct_text_and_function_outputs() {
        let req = ResponseRequest {
            model: None,
            input: json!([
                {"type": "input_text", "text": "hi"},
                {"type": "function_call_output", "call_id": "call_1", "output": "rainy"}
            ]),
            instructions: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: false,
            stop: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            text: None,
            reasoning: None,
            previous_response_id: None,
            user: None,
            metadata: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
        };
        let messages = response_messages(&req, 0).unwrap();
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "tool");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn previous_response_function_calls_become_assistant_tool_calls() {
        let mut prev = empty_response("resp-prev");
        prev.output.push(ResponseOutputItem::FunctionCall {
            id: "call_1".to_string(),
            call_id: "call_1".to_string(),
            name: "lookup".to_string(),
            arguments: "{\"query\":\"weather\"}".to_string(),
            status: "completed",
        });
        let mut messages = Vec::new();
        append_previous_response_messages(&mut messages, &prev);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(
            messages[0].tool_calls.as_ref().unwrap()[0]["function"]["arguments"]["query"],
            "weather"
        );
    }

    #[test]
    fn response_function_call_input_becomes_assistant_tool_call() {
        let msg = function_call_input_message(&json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {"query": "weather"}
        }))
        .unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(
            msg.tool_calls.as_ref().unwrap()[0]["function"]["arguments"]["query"],
            "weather"
        );
    }

    #[test]
    fn response_max_completion_tokens_alias_deserializes() {
        let req: ResponseRequest = serde_json::from_value(json!({
            "input": "hi",
            "max_completion_tokens": 9
        }))
        .unwrap();
        assert_eq!(req.max_output_tokens, Some(9));
    }

    fn empty_response(id: &str) -> StoredResponse {
        empty_response_with_cache_key(id, None)
    }

    fn empty_response_with_cache_key(id: &str, cache_key: Option<&str>) -> StoredResponse {
        StoredResponse {
            id: id.to_string(),
            object: "response",
            created_at: 1,
            model: "model".to_string(),
            status: "completed",
            output: Vec::new(),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: None,
            },
            cache_key: cache_key.map(ToOwned::to_owned),
        }
    }
}
