//! Comprehensive `supersonic-serve` integration test.
//!
//! **Gated on env vars.** When `SUPERSONIC_TEST_MODEL` or
//! `SUPERSONIC_TEST_MODEL_DIR` is unset, the suite prints a skip message
//! and returns cleanly — so `cargo test -p server` on a machine without a
//! model or GPU stays green.
//!
//! ## Running
//!
//! ```bash
//! # Qwen3.5 0.8B INT4
//! SUPERSONIC_TEST_MODEL=qwen3.5-0.8b \
//!   SUPERSONIC_TEST_MODEL_DIR=/path/to/Qwen3.5-0.8B \
//!   SUPERSONIC_TEST_INT4=1 \
//!   cargo test -p server --test chat_integration -- --nocapture
//!
//! # Gemma 4 E2B INT4 (same command, different model + dir)
//! SUPERSONIC_TEST_MODEL=gemma4-e2b \
//!   SUPERSONIC_TEST_MODEL_DIR=/path/to/gemma-4-E2B \
//!   SUPERSONIC_TEST_INT4=1 \
//!   cargo test -p server --test chat_integration -- --nocapture
//! ```
//!
//! The suite loads weights once, spawns the server on an ephemeral port,
//! and runs every scenario sequentially against that shared engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

use server::state::{self, LoaderConfig};

/* ---------- env-var config ---------- */

fn load_test_config() -> Option<LoaderConfig> {
    let model = std::env::var("SUPERSONIC_TEST_MODEL").ok()?;
    let dir = std::env::var("SUPERSONIC_TEST_MODEL_DIR").ok()?;
    let int4 = std::env::var("SUPERSONIC_TEST_INT4").is_ok();
    let fp8 = std::env::var("SUPERSONIC_TEST_FP8_RUNTIME").is_ok();
    let kv_fp8 = std::env::var("SUPERSONIC_TEST_KV_FP8").is_ok();
    let max_ctx = std::env::var("SUPERSONIC_TEST_MAX_CONTEXT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    Some(LoaderConfig {
        model,
        model_dir: PathBuf::from(dir),
        backend: std::env::var("SUPERSONIC_TEST_BACKEND").unwrap_or_else(|_| "auto".into()),
        device: 0,
        max_context: max_ctx,
        int4,
        fp8_runtime: fp8,
        kv_fp8,
        api_key: None,
        no_download: std::env::var("SUPERSONIC_TEST_NO_DOWNLOAD").is_ok(),
    })
}

/* ---------- harness ---------- */

struct Harness {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn spawn_harness() -> Option<Harness> {
    let cfg = load_test_config()?;
    let model_for_log = cfg.model.clone();
    let dir_for_log = cfg.model_dir.display().to_string();
    eprintln!(
        "[test] loading model={} dir={} int4={} fp8={} kv_fp8={}",
        model_for_log, dir_for_log, cfg.int4, cfg.fp8_runtime, cfg.kv_fp8
    );

    let state_built = tokio::task::spawn_blocking(move || state::build(cfg))
        .await
        .expect("spawn_blocking join")
        .expect("build server state");
    let state_arc = Arc::new(state_built);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    eprintln!("[test] server listening on http://{}", addr);

    let task = {
        let state = state_arc.clone();
        tokio::spawn(async move {
            let _ = server::serve(state, listener).await;
        })
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("build reqwest client");
    Some(Harness {
        base: format!("http://{}", addr),
        client,
        _task: task,
    })
}

/* ---------- SSE parser ---------- */

#[derive(Debug, Clone)]
enum SseEvent {
    Data(Value),
    Done,
    Error(String),
}

async fn collect_sse(resp: reqwest::Response) -> Vec<SseEvent> {
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "SSE response not 200"
    );
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
                let Some(payload) = line.strip_prefix("data:") else { continue; };
                let payload = payload.trim_start();
                if payload == "[DONE]" {
                    events.push(SseEvent::Done);
                } else if payload.contains("\"error\"") {
                    events.push(SseEvent::Error(payload.to_string()));
                } else {
                    let v: Value = serde_json::from_str(payload)
                        .unwrap_or_else(|e| panic!("parse SSE json: {e} — payload={payload}"));
                    events.push(SseEvent::Data(v));
                }
            }
        }
    }
    events
}

/* ---------- chat-template probe ---------- */

/// Probe the server once: if `/v1/chat/completions` returns a 400 with
/// "no chat_template" in the body, this model has no bundled chat template
/// and the chat-specific scenarios should be skipped.
async fn has_chat_template(h: &Harness) -> bool {
    let body = json!({
        "messages": [{"role": "user", "content": "probe"}],
        "max_tokens": 1
    });
    let resp = h.client.post(format!("{}/v1/chat/completions", h.base))
        .json(&body).send().await.expect("probe send");
    if resp.status().is_success() { return true; }
    let text = resp.text().await.unwrap_or_default();
    !text.contains("no chat_template")
}

/* ---------- scenarios ---------- */

async fn test_models(h: &Harness, expected_model: &str) {
    let resp: Value = h
        .client
        .get(format!("{}/v1/models", h.base))
        .send()
        .await
        .expect("GET /v1/models")
        .json()
        .await
        .expect("parse json");
    assert_eq!(resp["object"], "list", "unexpected top-level object field");
    let data = resp["data"].as_array().expect("data is array");
    assert_eq!(data.len(), 1, "expected exactly one model");
    assert_eq!(data[0]["id"], expected_model, "model id mismatch");
    assert_eq!(data[0]["object"], "model");
    eprintln!("[scenario] /v1/models → OK ({})", expected_model);
}

async fn test_chat_non_stream(h: &Harness) {
    let body = json!({
        "model": "any",
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 12,
        "temperature": 0
    });
    let resp: Value = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .json(&body)
        .send()
        .await
        .expect("POST chat")
        .json()
        .await
        .expect("parse json");
    assert_eq!(resp["object"], "chat.completion");
    let choice = &resp["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    let content = choice["message"]["content"].as_str().expect("content string");
    assert!(!content.is_empty(), "chat content must not be empty");
    let finish = choice["finish_reason"].as_str().expect("finish_reason");
    assert!(
        finish == "stop" || finish == "length",
        "unexpected finish_reason={finish}"
    );
    let usage = &resp["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(usage["completion_tokens"].as_u64().unwrap() > 0);
    eprintln!(
        "[scenario] chat non-stream → OK ({} toks, finish={})",
        usage["completion_tokens"], finish
    );
}

async fn test_chat_stream(h: &Harness) {
    let body = json!({
        "model": "any",
        "messages": [{"role": "user", "content": "Count one two"}],
        "max_tokens": 10,
        "temperature": 0,
        "stream": true
    });
    let resp = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .json(&body)
        .send()
        .await
        .expect("POST chat stream");
    let events = collect_sse(resp).await;
    assert!(!events.is_empty(), "stream produced no events");
    // First data chunk should establish role=assistant.
    let first = events.iter().find_map(|e| {
        if let SseEvent::Data(v) = e { Some(v) } else { None }
    }).expect("at least one data event");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant",
        "first chunk must open with role=assistant");

    // Should have at least one content delta, a final finish_reason chunk, and a [DONE].
    let mut content = String::new();
    let mut saw_finish = false;
    let mut saw_done = false;
    for e in &events {
        match e {
            SseEvent::Data(v) => {
                if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(c);
                }
                if v["choices"][0]["finish_reason"].is_string() {
                    saw_finish = true;
                }
            }
            SseEvent::Done => saw_done = true,
            SseEvent::Error(msg) => panic!("stream error event: {msg}"),
        }
    }
    assert!(!content.is_empty(), "streamed content empty");
    assert!(saw_finish, "no finish_reason chunk before [DONE]");
    assert!(saw_done, "never saw [DONE] sentinel");
    eprintln!("[scenario] chat stream → OK ({} chunks, content={:?})",
        events.len(), content);
}

async fn test_completions_non_stream(h: &Harness) {
    let body = json!({
        "model": "any",
        "prompt": "The capital of France is",
        "max_tokens": 6,
        "temperature": 0
    });
    let resp: Value = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .json(&body)
        .send()
        .await
        .expect("POST completions")
        .json()
        .await
        .expect("parse json");
    assert_eq!(resp["object"], "text_completion");
    let text = resp["choices"][0]["text"].as_str().expect("text");
    assert!(!text.is_empty(), "completion text empty");
    eprintln!("[scenario] completions non-stream → OK (text={:?})", text);
}

async fn test_completions_stream(h: &Harness) {
    let body = json!({
        "prompt": "ABC DEF",
        "max_tokens": 6,
        "temperature": 0,
        "stream": true
    });
    let resp = h
        .client
        .post(format!("{}/v1/completions", h.base))
        .json(&body)
        .send()
        .await
        .expect("POST completions stream");
    let events = collect_sse(resp).await;
    let mut text = String::new();
    let mut saw_done = false;
    let mut saw_finish = false;
    for e in &events {
        match e {
            SseEvent::Data(v) => {
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    text.push_str(t);
                }
                if v["choices"][0]["finish_reason"].is_string() {
                    saw_finish = true;
                }
            }
            SseEvent::Done => saw_done = true,
            SseEvent::Error(m) => panic!("stream error: {m}"),
        }
    }
    assert!(!text.is_empty(), "streamed completion text empty");
    assert!(saw_finish, "no finish_reason chunk");
    assert!(saw_done, "no [DONE]");
    eprintln!("[scenario] completions stream → OK ({} events)", events.len());
}

async fn test_seed_determinism(h: &Harness) {
    let mk = || json!({
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": 1337
    });
    let a: Value = h.client.post(format!("{}/v1/completions", h.base))
        .json(&mk()).send().await.expect("req a").json().await.expect("json a");
    let b: Value = h.client.post(format!("{}/v1/completions", h.base))
        .json(&mk()).send().await.expect("req b").json().await.expect("json b");
    let ta = a["choices"][0]["text"].as_str().expect("text a");
    let tb = b["choices"][0]["text"].as_str().expect("text b");
    assert_eq!(ta, tb, "same seed must produce same content: {ta:?} vs {tb:?}");
    eprintln!("[scenario] seed determinism → OK ({:?})", ta);
}

async fn test_multi_turn(h: &Harness) {
    // Conversation with explicit assistant history — exercises the chat
    // template's handling of assistant turns.
    let body = json!({
        "messages": [
            {"role": "user", "content": "What is 2 plus 2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "What is that number doubled?"}
        ],
        "max_tokens": 12,
        "temperature": 0
    });
    let resp: Value = h.client.post(format!("{}/v1/chat/completions", h.base))
        .json(&body).send().await.expect("POST multi-turn").json().await.expect("json");
    let content = resp["choices"][0]["message"]["content"].as_str().expect("content");
    assert!(!content.is_empty(), "multi-turn content empty");
    eprintln!("[scenario] multi-turn → OK ({:?})", content);
}

async fn test_system_prompt(h: &Harness) {
    let body = json!({
        "messages": [
            {"role": "system", "content": "You answer in one word."},
            {"role": "user", "content": "Capital of Japan?"}
        ],
        "max_tokens": 12,
        "temperature": 0
    });
    let resp: Value = h.client.post(format!("{}/v1/chat/completions", h.base))
        .json(&body).send().await.expect("POST system").json().await.expect("json");
    let content = resp["choices"][0]["message"]["content"].as_str().expect("content");
    assert!(!content.is_empty(), "system-prompt content empty");
    eprintln!("[scenario] system prompt → OK ({:?})", content);
}

async fn test_stop_sequence(h: &Harness) {
    // Give the model a runway where a stop string is likely to appear, then
    // verify the server trimmed generation on it (finish_reason=stop and
    // the stop string appears in the returned text at most once).
    let body = json!({
        "prompt": "List: A, B, C, D, E, F, G, H, I, J",
        "max_tokens": 32,
        "temperature": 0,
        "stop": ["E"]
    });
    let resp: Value = h.client.post(format!("{}/v1/completions", h.base))
        .json(&body).send().await.expect("POST stop").json().await.expect("json");
    let text = resp["choices"][0]["text"].as_str().expect("text");
    let finish = resp["choices"][0]["finish_reason"].as_str().expect("finish");
    // Accept either "stop" (string hit) or "length" (model never emitted it);
    // but if finish=="stop" the text must indeed contain the stop string.
    if finish == "stop" {
        assert!(text.contains('E'),
            "finish=stop but text did not contain the stop string 'E': {text:?}");
    }
    eprintln!("[scenario] stop sequence → OK (finish={}, text={:?})", finish, text);
}

async fn test_concurrent_requests(h: &Harness) {
    // Serial-Mutex contract: four concurrent requests must all complete
    // successfully. They'll queue behind each other, not corrupt state.
    let prompts = ["Red", "Green", "Blue", "Yellow"];
    let mut handles = Vec::new();
    for p in prompts {
        let client = h.client.clone();
        let url = format!("{}/v1/completions", h.base);
        handles.push(tokio::spawn(async move {
            let body = json!({"prompt": p, "max_tokens": 6, "temperature": 0});
            let v: Value = client.post(url).json(&body).send().await
                .expect("concurrent send").json().await.expect("concurrent json");
            v["choices"][0]["text"].as_str().unwrap_or("").to_string()
        }));
    }
    let mut outputs = Vec::new();
    for h in handles {
        outputs.push(h.await.expect("join task"));
    }
    assert_eq!(outputs.len(), 4);
    for o in &outputs {
        assert!(!o.is_empty(), "concurrent response had empty text");
    }
    eprintln!("[scenario] concurrent x4 → OK ({:?})", outputs);
}

async fn test_empty_messages_error(h: &Harness) {
    let body = json!({"messages": [], "max_tokens": 4});
    let resp = h.client.post(format!("{}/v1/chat/completions", h.base))
        .json(&body).send().await.expect("POST empty");
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let v: Value = resp.json().await.expect("parse error body");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("messages"), "error message should mention messages: {msg}");
    eprintln!("[scenario] empty messages → OK (400 invalid_request_error)");
}

async fn test_context_overflow_error(h: &Harness) {
    let body = json!({"prompt": "x", "max_tokens": 999_999});
    let resp = h.client.post(format!("{}/v1/completions", h.base))
        .json(&body).send().await.expect("POST overflow");
    let status = resp.status();
    let v: Value = resp.json().await.expect("parse body");
    assert!(status.is_client_error() || status.is_server_error(),
        "expected error status, got {status}");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("max_context") || msg.contains("exceeds"),
        "error should mention context overflow: {msg}");
    eprintln!("[scenario] context overflow → OK ({})", status);
}

/* ---------- the suite ---------- */

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn comprehensive_chat_suite() {
    let Some(h) = spawn_harness().await else {
        eprintln!(
            "[test] SKIPPED: set SUPERSONIC_TEST_MODEL and SUPERSONIC_TEST_MODEL_DIR to run this suite"
        );
        return;
    };
    let model_id = std::env::var("SUPERSONIC_TEST_MODEL").unwrap();

    test_models(&h, &model_id).await;

    // Gemma 4's HF bundle ships without a `chat_template` — the server
    // correctly refuses /v1/chat/completions in that case. Detect and
    // skip chat-specific scenarios so the rest of the suite still covers
    // /v1/completions, streaming, concurrency, and error paths.
    let chat_ok = has_chat_template(&h).await;
    if chat_ok {
        test_chat_non_stream(&h).await;
        test_chat_stream(&h).await;
        test_multi_turn(&h).await;
        test_system_prompt(&h).await;
        test_empty_messages_error(&h).await;
    } else {
        eprintln!("[scenario] chat scenarios skipped — model has no chat_template");
    }

    test_completions_non_stream(&h).await;
    test_completions_stream(&h).await;
    test_seed_determinism(&h).await;
    test_stop_sequence(&h).await;
    test_concurrent_requests(&h).await;
    test_context_overflow_error(&h).await;

    eprintln!(
        "[test] all scenarios passed for model={} (chat_template={})",
        model_id, chat_ok
    );
}
