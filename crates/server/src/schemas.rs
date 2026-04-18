//! OpenAI-compatible request/response types for `/v1/models`,
//! `/v1/chat/completions`, and `/v1/completions`.
//!
//! Only the fields SuperSonic actually honors are declared; unknown fields
//! on incoming requests are ignored (serde default).

use serde::{Deserialize, Serialize};

use crate::chat_template::IncomingChatMessage;

/* ---------- /v1/models ---------- */

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ListModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/* ---------- shared sampling params ---------- */

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl StopParam {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/* ---------- /v1/chat/completions ---------- */

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<IncomingChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

/* ---------- /v1/chat/completions streaming ---------- */

#[derive(Debug, Serialize)]
pub struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

/* ---------- /v1/completions ---------- */

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptParam {
    One(String),
    Many(Vec<String>),
}

impl PromptParam {
    pub fn into_single(self) -> Option<String> {
        match self {
            Self::One(s) => Some(s),
            Self::Many(mut v) if v.len() == 1 => v.pop(),
            Self::Many(_) => None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub prompt: PromptParam,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}
