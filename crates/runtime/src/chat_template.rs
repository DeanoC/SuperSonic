//! Load and render the chat template shipped with the model.
//!
//! Qwen3.8 bundles a Jinja chat template in `tokenizer_config.json`. The
//! template expects variables like `messages` and `add_generation_prompt`,
//! and commonly references `bos_token` / `eos_token`. We parse it once at
//! startup with `minijinja` and render per request.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use minijinja::{context, value::Value, Environment};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: JsonValue::String(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    pub tools: Option<JsonValue>,
    pub enable_thinking: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            tools: None,
            enable_thinking: false,
        }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    pub fn from_template_source(tpl_src: impl Into<String>) -> Result<Arc<Self>> {
        Self::compile(tpl_src.into(), None, None)
    }

    /// Load `{model_dir}/tokenizer_config.json` and compile its
    /// `chat_template` field. Returns `Ok(None)` if the file or field is
    /// missing — the direct non-chat prompt path can proceed in that case.
    pub fn try_load(model_dir: &Path) -> Result<Option<Arc<Self>>> {
        let path = model_dir.join("tokenizer_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let cfg: JsonValue =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;

        let tpl_src = match cfg.get("chat_template") {
            Some(JsonValue::String(s)) => s.clone(),
            Some(JsonValue::Array(arr)) => {
                // HF supports multiple named templates as an array; pick the
                // default (name=="default") or the first entry.
                arr.iter()
                    .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| arr.first())
                    .and_then(|e| e.get("template").and_then(|t| t.as_str()))
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("chat_template array has no usable entry"))?
            }
            _ => return Ok(None),
        };

        let bos_token = extract_token(&cfg, "bos_token");
        let eos_token = extract_token(&cfg, "eos_token");

        Self::compile(tpl_src, bos_token, eos_token).map(Some)
    }

    fn compile(
        tpl_src: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Arc<Self>> {
        let mut env = Environment::new();
        // HF chat templates routinely use Python string methods like
        // `.startswith` / `.endswith` / `.strip` that aren't part of the
        // Jinja2 core. `pycompat::unknown_method_callback` forwards those
        // to minijinja-contrib's Python-compatible implementations so the
        // templates render unchanged.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", tpl_src)
            .with_context(|| "compile chat_template")?;

        Ok(Arc::new(Self {
            env,
            bos_token,
            eos_token,
        }))
    }

    /// Render the template against a list of messages. Returns the prompt
    /// text to feed into the tokenizer.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with_options(
            messages,
            RenderOptions {
                add_generation_prompt,
                ..RenderOptions::default()
            },
        )
    }

    /// Render the template with optional OpenAI-compatible tool definitions
    /// and model-family thinking controls.
    pub fn render_with_options(
        &self,
        messages: &[ChatMessage],
        opts: RenderOptions,
    ) -> Result<String> {
        let tpl = self.env.get_template("chat")?;
        let msgs: Vec<Value> = messages.iter().map(Value::from_serialize).collect();
        let tools = opts.tools.unwrap_or(JsonValue::Null);
        let ctx = context! {
            messages => msgs,
            add_generation_prompt => opts.add_generation_prompt,
            tools => Value::from_serialize(&tools),
            enable_thinking => opts.enable_thinking,
            preserve_thinking => opts.enable_thinking,
            add_vision_id => false,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };
        tpl.render(ctx)
            .map_err(|e| anyhow!("render chat template: {e}"))
    }
}

fn extract_token(cfg: &JsonValue, key: &str) -> Option<String> {
    match cfg.get(key)? {
        JsonValue::String(s) => Some(s.clone()),
        JsonValue::Object(obj) => obj
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

/// Deserialization shape for chat messages supplied by callers.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: JsonValue,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<JsonValue>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl From<IncomingChatMessage> for ChatMessage {
    fn from(m: IncomingChatMessage) -> Self {
        Self {
            role: m.role,
            content: m.content,
            reasoning_content: m.reasoning_content,
            tool_calls: m.tool_calls,
            tool_call_id: m.tool_call_id,
        }
    }
}
