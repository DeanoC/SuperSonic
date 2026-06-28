use std::sync::Arc;

use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use server::generate::MockGeneration;
use server::prefix_cache::{PrefixCache, PrefixCacheConfig};
use server::state::{GenerationScheduler, ServerState};
use server::{capabilities, chat_template, registry};

fn test_tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#,
    )
    .expect("test tokenizer")
}

fn test_template() -> Arc<chat_template::ChatTemplate> {
    chat_template::ChatTemplate::from_template_source(
        r#"{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor -%}
{%- if tools -%}<tools>{{ tools | tojson }}</tools>{%- endif -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
{%- if enable_thinking -%}<think>
{%- endif -%}
{%- endif -%}"#,
    )
    .expect("test chat template")
}

fn test_state_with_mock(mock_generation: MockGeneration) -> Arc<ServerState> {
    test_state_with_scheduler(mock_generation, GenerationScheduler::new(32, 30_000))
}

fn test_state_with_scheduler(
    mock_generation: MockGeneration,
    scheduler: GenerationScheduler,
) -> Arc<ServerState> {
    Arc::new(ServerState {
        server_instance_id: next_test_server_instance_id(),
        model_id: "qwen3.5-0.8b".to_string(),
        model_family: registry::ModelFamily::Qwen35,
        tokenizer: Arc::new(test_tokenizer()),
        chat_template: Some(test_template()),
        session: None,
        mock_generation: Some(mock_generation),
        eos_ids: Vec::new(),
        max_context: 256,
        api_key: Some("secret".to_string()),
        cors_allow_origin: Some("*".to_string()),
        response_store_max_entries: 1024,
        scheduler: Arc::new(scheduler),
        telemetry: server::generate::GenerationTelemetry::default(),
        capabilities: capabilities::capabilities_for_variant(
            &registry::ModelVariant::Qwen3_5_0_8B,
            registry::Backend::Cuda,
            false,
            false,
            false,
        ),
        prefix_cache: Arc::new(PrefixCache::new(PrefixCacheConfig {
            enabled: true,
            dir: std::env::temp_dir().join("supersonic-test-prefix-cache"),
            min_tokens: 1,
            max_entries: 1,
            max_bytes: 64 * 1024 * 1024,
            memory_ttl_secs: 600,
            disk_ttl_secs: 86_400,
        })),
    })
}

fn next_test_server_instance_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(10_000);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

struct Harness {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn spawn(mock_text: &str) -> Harness {
    spawn_with_mock(MockGeneration::text(mock_text)).await
}

async fn spawn_chunks(chunks: &[&str]) -> Harness {
    spawn_with_mock(MockGeneration {
        chunks: chunks.iter().map(|s| (*s).to_string()).collect(),
        finish: server::generate::FinishReason::Stop,
        delay_ms: 0,
    })
    .await
}

async fn spawn_with_mock(mock_generation: MockGeneration) -> Harness {
    let state = test_state_with_mock(mock_generation);
    spawn_with_state(state).await
}

async fn spawn_with_state(state: Arc<ServerState>) -> Harness {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let _ = server::serve(state, listener).await;
    });
    Harness {
        base: format!("http://{addr}"),
        client: Client::new(),
        _task: task,
    }
}

async fn collect_sse(resp: reqwest::Response) -> Vec<Value> {
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut events = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("sse chunk");
        buf.push_str(std::str::from_utf8(&chunk).expect("sse utf8"));
        while let Some(idx) = buf.find("\n\n") {
            let raw = buf[..idx].to_string();
            buf = buf[idx + 2..].to_string();
            for line in raw.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data == "[DONE]" {
                    events.push(json!({"__done": true}));
                } else {
                    events.push(serde_json::from_str(data).expect("sse json"));
                }
            }
        }
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_chat_non_stream_parses_reasoning_and_tools() {
    let h = spawn(
        "<think>plan</think>\n<tool_call>\n<function=lookup>\n<parameter=query>\nweather\n</parameter>\n</function>\n</tool_call>",
    )
    .await;
    let http = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "model": "local",
            "messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "response_format": {"type": "json_object"},
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "max_completion_tokens": 8
        }))
        .send()
        .await
        .expect("send");
    let status = http.status();
    let resp: Value = http.json().await.expect("json");
    assert_eq!(status, reqwest::StatusCode::OK, "body={resp}");

    let choice = &resp["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["content"], Value::Null);
    assert_eq!(choice["message"]["reasoning_content"], "plan");
    assert_eq!(
        choice["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert!(resp["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(resp["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        resp["usage"]["total_tokens"].as_u64().unwrap(),
        resp["usage"]["prompt_tokens"].as_u64().unwrap()
            + resp["usage"]["completion_tokens"].as_u64().unwrap()
    );
    assert_eq!(resp["usage"]["prompt_tokens_details"]["cached_tokens"], 0);

    let unsupported = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
            "max_completion_tokens": 1
        }))
        .send()
        .await
        .expect("send unsupported");
    assert_eq!(unsupported.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_required_tool_choice_returns_400() {
    let h = spawn("hello").await;
    let tools = json!([{"type": "function", "function": {"name": "lookup"}}]);

    let chat = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": "required",
            "max_completion_tokens": 1
        }))
        .send()
        .await
        .expect("chat required");
    assert_eq!(chat.status(), reqwest::StatusCode::BAD_REQUEST);

    let responses = h
        .client
        .post(format!("{}/v1/responses", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "input": "hi",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "required",
            "max_output_tokens": 1
        }))
        .send()
        .await
        .expect("responses required");
    assert_eq!(responses.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_responses_get_delete_roundtrip() {
    let h = spawn("hello world").await;
    let created: Value = h
        .client
        .post(format!("{}/v1/responses", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "model": "qwen3.5-0.8b",
            "instructions": "brief",
            "input": [{"type": "input_text", "text": "hi"}],
            "text": {"format": {"type": "json_object"}},
            "maxOutputTokens": 4
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    let id = created["id"].as_str().unwrap();
    assert_eq!(created["output"][0]["type"], "message");

    let fetched: Value = h
        .client
        .get(format!("{}/v1/responses/{id}", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("get")
        .json()
        .await
        .expect("json");
    assert_eq!(fetched["id"], id);

    let deleted: Value = h
        .client
        .delete(format!("{}/v1/responses/{id}", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("delete")
        .json()
        .await
        .expect("json");
    assert_eq!(deleted["deleted"], true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_responses_previous_response_tool_loop_shape() {
    let h = spawn(
        "<tool_call>\n<function=lookup>\n<parameter=query>\nweather\n</parameter>\n</function>\n</tool_call>",
    )
    .await;
    let first: Value = h
        .client
        .post(format!("{}/v1/responses", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "input": "need weather",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "max_output_tokens": 8
        }))
        .send()
        .await
        .expect("first response")
        .json()
        .await
        .expect("first json");
    let first_id = first["id"].as_str().expect("first id");
    assert_eq!(first["output"][0]["type"], "function_call");
    assert_eq!(first["output"][0]["name"], "lookup");
    let call_id = first["output"][0]["call_id"]
        .as_str()
        .expect("call id")
        .to_string();

    let second_http = h
        .client
        .post(format!("{}/v1/responses", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "previous_response_id": first_id,
            "input": [{
                "type": "function_call_output",
                "call_id": call_id,
                "output": "{\"forecast\":\"sunny\"}"
            }],
            "max_output_tokens": 4
        }))
        .send()
        .await
        .expect("second response");
    let status = second_http.status();
    let second: Value = second_http.json().await.expect("second json");
    assert_eq!(status, reqwest::StatusCode::OK, "body={second}");
    assert_ne!(second["id"], first["id"]);
    assert_eq!(second["output"][0]["type"], "function_call");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_responses_store_is_scoped_per_server_instance() {
    let first = spawn("hello from first").await;
    let second = spawn("hello from second").await;
    let created: Value = first
        .client
        .post(format!("{}/v1/responses", first.base))
        .bearer_auth("secret")
        .json(&json!({
            "input": "hi",
            "max_output_tokens": 4
        }))
        .send()
        .await
        .expect("create")
        .json()
        .await
        .expect("json");
    let id = created["id"].as_str().unwrap();

    let cross_fetch = second
        .client
        .get(format!("{}/v1/responses/{id}", second.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("cross fetch");
    assert_eq!(cross_fetch.status(), reqwest::StatusCode::BAD_REQUEST);

    let cross_delete = second
        .client
        .delete(format!("{}/v1/responses/{id}", second.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("cross delete")
        .json::<Value>()
        .await
        .expect("delete json");
    assert_eq!(cross_delete["deleted"], false);

    let same_fetch = first
        .client
        .get(format!("{}/v1/responses/{id}", first.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("same fetch");
    assert_eq!(same_fetch.status(), reqwest::StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_chat_stream_buffers_reasoning_and_includes_usage() {
    let h = spawn_chunks(&["<think>plan", "</think>\nhello", " world"]).await;
    let resp = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "stream": true,
            "stream_options": {"include_usage": true},
            "prompt_cache_key": "protocol-mock",
            "prompt_cache_retention": "in_memory",
            "metadata": {"thread_id": "thread-a"},
            "max_tokens": 8
        }))
        .send()
        .await
        .expect("send");
    let events = collect_sse(resp).await;

    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut saw_usage = false;
    let mut saw_done = false;
    for event in &events {
        if event.get("__done").is_some() {
            saw_done = true;
            continue;
        }
        if let Some(s) = event["choices"][0]["delta"]["content"].as_str() {
            content.push_str(s);
        }
        if let Some(s) = event["choices"][0]["delta"]["reasoning_content"].as_str() {
            reasoning.push_str(s);
        }
        if event.get("usage").is_some_and(|u| !u.is_null()) {
            saw_usage = true;
            assert_eq!(event["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
        }
        let serialized = event.to_string();
        assert!(
            !serialized.contains("<think>") && !serialized.contains("</think>"),
            "stream leaked think tags: {serialized}"
        );
    }
    assert_eq!(reasoning, "plan");
    assert_eq!(content, "hello world");
    assert!(saw_usage);
    assert!(saw_done);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_responses_stream_emits_expected_events() {
    let h = spawn_chunks(&["hello", " world"]).await;
    let resp = h
        .client
        .post(format!("{}/v1/responses", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "input": "hi",
            "stream": true,
            "max_output_tokens": 8
        }))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body = resp.text().await.expect("sse text");
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains("event: response.output_text.done"));
    assert!(body.contains("event: response.output_item.done"));
    assert!(body.contains("event: response.completed"));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_auth_cors_and_model_mismatch() {
    let h = spawn("hello").await;
    let unauth = h
        .client
        .get(format!("{}/health", h.base))
        .send()
        .await
        .expect("unauth");
    assert_eq!(unauth.status(), reqwest::StatusCode::UNAUTHORIZED);

    let root = h
        .client
        .get(format!("{}/", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("root");
    assert_eq!(root.status(), reqwest::StatusCode::OK);

    let v1 = h
        .client
        .get(format!("{}/v1", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("v1");
    assert_eq!(v1.status(), reqwest::StatusCode::OK);

    let ready = h
        .client
        .get(format!("{}/ready", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("ready");
    assert_eq!(ready.status(), reqwest::StatusCode::OK);

    let v1_ready = h
        .client
        .get(format!("{}/v1/ready", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("v1 ready");
    assert_eq!(v1_ready.status(), reqwest::StatusCode::OK);

    let model: Value = h
        .client
        .get(format!("{}/v1/models/local", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("model")
        .json()
        .await
        .expect("model json");
    assert_eq!(model["id"], "qwen3.5-0.8b");

    let capabilities: Value = h
        .client
        .get(format!("{}/v1/capabilities", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("capabilities")
        .json()
        .await
        .expect("capabilities json");
    assert_eq!(capabilities["prefix_cache"]["enabled"], true);
    assert_eq!(capabilities["prefix_cache"]["min_tokens"], 1);
    assert_eq!(capabilities["prefix_cache"]["max_bytes"], 64 * 1024 * 1024);

    let missing_model = h
        .client
        .get(format!("{}/v1/models/gpt-4.1", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("missing model");
    assert_eq!(missing_model.status(), reqwest::StatusCode::BAD_REQUEST);

    let metrics = h
        .client
        .get(format!("{}/metrics", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text");
    assert!(metrics.contains("supersonic_active_requests"));
    assert!(metrics.contains("supersonic_queued_requests"));
    assert!(metrics.contains("supersonic_generation_active"));
    assert!(metrics.contains("supersonic_generation_queued"));
    assert!(metrics.contains("supersonic_prefix_cache_hits"));
    assert!(metrics.contains("supersonic_prefix_cache_cached_tokens"));
    assert!(metrics.contains("supersonic_prefix_cache_resident_bytes"));
    assert!(metrics.contains("supersonic_prefix_cache_admission_skips"));
    assert!(metrics.contains("supersonic_dflash_last_rounds"));
    assert!(metrics.contains("supersonic_dflash_last_accepted_total"));
    assert!(metrics.contains("supersonic_dflash_last_decode_ms"));

    let cors = h
        .client
        .request(
            reqwest::Method::OPTIONS,
            format!("{}/v1/chat/completions", h.base),
        )
        .header("origin", "http://localhost:3000")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .expect("cors");
    assert!(cors.status().is_success());

    let mismatch = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({"model": "gpt-4.1", "prompt": "hello", "max_tokens": 1}))
        .send()
        .await
        .expect("mismatch");
    assert_eq!(mismatch.status(), reqwest::StatusCode::BAD_REQUEST);

    let unsupported = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({"prompt": "hello", "n": 2, "max_tokens": 1}))
        .send()
        .await
        .expect("unsupported");
    assert_eq!(unsupported.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_generation_queue_full_returns_429() {
    let state = test_state_with_scheduler(
        MockGeneration {
            chunks: vec!["hello".to_string(), " world".to_string()],
            finish: server::generate::FinishReason::Stop,
            delay_ms: 100,
        },
        GenerationScheduler::new(0, 30_000),
    );
    let h = spawn_with_state(state).await;
    let first_client = h.client.clone();
    let first_url = format!("{}/v1/chat/completions", h.base);
    let first = tokio::spawn(async move {
        first_client
            .post(first_url)
            .bearer_auth("secret")
            .json(&json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8
            }))
            .send()
            .await
            .expect("first")
    });
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    let second = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }))
        .send()
        .await
        .expect("second");
    assert_eq!(second.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);

    let health: Value = h
        .client
        .get(format!("{}/health", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("health")
        .json()
        .await
        .expect("health json");
    assert_eq!(health["max_queued_requests"], 0);
    assert_eq!(health["queued_requests"], 0);

    let first = first.await.expect("first join");
    assert_eq!(first.status(), reqwest::StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_generation_queue_timeout_returns_503() {
    let state = test_state_with_scheduler(
        MockGeneration {
            chunks: vec!["hello".to_string(), " world".to_string()],
            finish: server::generate::FinishReason::Stop,
            delay_ms: 100,
        },
        GenerationScheduler::new(1, 20),
    );
    let h = spawn_with_state(state).await;
    let first_client = h.client.clone();
    let first_url = format!("{}/v1/completions", h.base);
    let first = tokio::spawn(async move {
        first_client
            .post(first_url)
            .bearer_auth("secret")
            .json(&json!({"prompt": "hello", "max_tokens": 8}))
            .send()
            .await
            .expect("first")
    });
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    let timed_out = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({"prompt": "hello", "max_tokens": 8}))
        .send()
        .await
        .expect("timed out");
    assert_eq!(timed_out.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

    let first = first.await.expect("first join");
    assert_eq!(first.status(), reqwest::StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_queued_stream_disconnect_releases_queue_slot() {
    let state = test_state_with_scheduler(
        MockGeneration {
            chunks: vec!["hello".to_string(), " world".to_string()],
            finish: server::generate::FinishReason::Stop,
            delay_ms: 100,
        },
        GenerationScheduler::new(1, 30_000),
    );
    let h = spawn_with_state(state).await;
    let first_client = h.client.clone();
    let first_url = format!("{}/v1/chat/completions", h.base);
    let first = tokio::spawn(async move {
        first_client
            .post(first_url)
            .bearer_auth("secret")
            .json(&json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8
            }))
            .send()
            .await
            .expect("first")
    });
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    let queued = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 8
        }))
        .send()
        .await
        .expect("queued stream");
    assert_eq!(queued.status(), reqwest::StatusCode::OK);

    let health: Value = h
        .client
        .get(format!("{}/health", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("health before drop")
        .json()
        .await
        .expect("health json");
    assert_eq!(health["queued_requests"], 1);

    drop(queued);
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    let health: Value = h
        .client
        .get(format!("{}/health", h.base))
        .bearer_auth("secret")
        .send()
        .await
        .expect("health after drop")
        .json()
        .await
        .expect("health json");
    assert_eq!(health["queued_requests"], 0);

    let first = first.await.expect("first join");
    assert_eq!(first.status(), reqwest::StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_tokenize_detokenize_roundtrip() {
    let h = spawn("hello").await;
    let tok: Value = h
        .client
        .post(format!("{}/v1/tokenize", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "model": "local",
            "input": "hello world",
            "add_special_tokens": false
        }))
        .send()
        .await
        .expect("tokenize")
        .json()
        .await
        .expect("json");
    assert_eq!(tok["object"], "tokenization");
    let tokens = tok["tokens"].as_array().expect("tokens");
    assert!(!tokens.is_empty());

    let detok: Value = h
        .client
        .post(format!("{}/v1/detokenize", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "model": "qwen3.5-0.8b",
            "tokens": [1, 2],
            "skip_special_tokens": true
        }))
        .send()
        .await
        .expect("detokenize")
        .json()
        .await
        .expect("json");
    assert_eq!(detok["object"], "detokenization");
    assert!(detok["text"].as_str().unwrap_or("").contains("hello"));

    let root_tok = h
        .client
        .post(format!("{}/tokenize", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "input": "hello",
            "add_special_tokens": false
        }))
        .send()
        .await
        .expect("root tokenize");
    assert_eq!(root_tok.status(), reqwest::StatusCode::OK);

    let root_detok = h
        .client
        .post(format!("{}/detokenize", h.base))
        .bearer_auth("secret")
        .json(&json!({"tokens": [1]}))
        .send()
        .await
        .expect("root detokenize");
    assert_eq!(root_detok.status(), reqwest::StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_completions_accept_token_prompt() {
    let h = spawn("hello").await;
    let resp: Value = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "model": "local",
            "prompt": [1, 2],
            "max_tokens": 4
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(resp["choices"][0]["text"], "hello");
    assert_eq!(resp["usage"]["prompt_tokens"], 2);
    assert_eq!(resp["usage"]["prompt_tokens_details"]["cached_tokens"], 0);

    let batched = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .bearer_auth("secret")
        .json(&json!({
            "prompt": [[1], [2]],
            "max_tokens": 4
        }))
        .send()
        .await
        .expect("batched");
    assert_eq!(batched.status(), reqwest::StatusCode::BAD_REQUEST);
}
