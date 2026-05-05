//! OpenAI-compatible request/response types for `/v1/models`,
//! `/v1/chat/completions`, and `/v1/completions`.
//!
//! Only the fields SuperSonic actually honors are declared; unknown fields
//! on incoming requests are ignored (serde default).

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

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
    #[serde(
        default,
        alias = "max_completion_tokens",
        alias = "maxCompletionTokens",
        alias = "maxTokens"
    )]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<JsonValue>,
    #[serde(default)]
    pub tool_choice: Option<JsonValue>,
    #[serde(default)]
    pub response_format: Option<JsonValue>,
    #[serde(default, alias = "reasoningEffort")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub function: OpenAiFunctionCall,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/* ---------- /v1/completions ---------- */

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptParam {
    One(String),
    Many(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

impl PromptParam {
    pub fn into_single(self) -> Option<SinglePrompt> {
        match self {
            Self::One(s) => Some(SinglePrompt::Text(s)),
            Self::Many(mut v) if v.len() == 1 => v.pop().map(SinglePrompt::Text),
            Self::Many(_) => None,
            Self::Tokens(ids) => Some(SinglePrompt::Tokens(ids)),
            Self::TokenBatches(mut batches) if batches.len() == 1 => {
                batches.pop().map(SinglePrompt::Tokens)
            }
            Self::TokenBatches(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SinglePrompt {
    Text(String),
    Tokens(Vec<u32>),
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
    #[serde(
        default,
        alias = "max_completion_tokens",
        alias = "maxCompletionTokens",
        alias = "maxTokens"
    )]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<JsonValue>,
    #[serde(default)]
    pub echo: Option<bool>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/* ---------- /v1/tokenize + /v1/detokenize ---------- */

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: String,
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub object: &'static str,
    pub model: String,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub object: &'static str,
    pub model: String,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn chat_max_token_aliases_deserialize() {
        for key in [
            "max_tokens",
            "max_completion_tokens",
            "maxCompletionTokens",
            "maxTokens",
        ] {
            let value = json!({
                "messages": [{"role": "user", "content": "hi"}],
                key: 7
            });
            let req: ChatCompletionRequest = serde_json::from_value(value).unwrap();
            assert_eq!(req.max_tokens, Some(7));
        }
    }

    #[test]
    fn completion_stream_usage_deserializes() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hi",
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .unwrap();
        assert!(req.stream_options.unwrap().include_usage);
    }

    #[test]
    fn completion_prompt_accepts_token_arrays() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": [1, 2, 3]
        }))
        .unwrap();
        assert!(matches!(
            req.prompt.into_single(),
            Some(SinglePrompt::Tokens(ids)) if ids == vec![1, 2, 3]
        ));

        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt": [[1, 2, 3]]
        }))
        .unwrap();
        assert!(matches!(
            req.prompt.into_single(),
            Some(SinglePrompt::Tokens(ids)) if ids == vec![1, 2, 3]
        ));
    }
}
