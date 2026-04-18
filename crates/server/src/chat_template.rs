//! Load and render the chat template shipped with the model.
//!
//! Both Qwen3.5 and Gemma 4 bundle a Jinja chat template in
//! `tokenizer_config.json`. The template expects variables like `messages`
//! and `add_generation_prompt`, and commonly references `bos_token` /
//! `eos_token`. We parse it once at startup with `minijinja` and render per
//! request.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use minijinja::{context, value::Value, Environment};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    /// Load `{model_dir}/tokenizer_config.json` and compile its
    /// `chat_template` field. Returns `Ok(None)` if the file or field is
    /// missing — the server can still serve `/v1/completions` in that case.
    pub fn try_load(model_dir: &Path) -> Result<Option<Arc<Self>>> {
        let path = model_dir.join("tokenizer_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let cfg: JsonValue = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))?;

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

        let mut env = Environment::new();
        // HF chat templates routinely use Python string methods like
        // `.startswith` / `.endswith` / `.strip` that aren't part of the
        // Jinja2 core. `pycompat::unknown_method_callback` forwards those
        // to minijinja-contrib's Python-compatible implementations so the
        // templates render unchanged.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", tpl_src)
            .with_context(|| "compile chat_template")?;

        let bos_token = extract_token(&cfg, "bos_token");
        let eos_token = extract_token(&cfg, "eos_token");

        Ok(Some(Arc::new(Self {
            env,
            bos_token,
            eos_token,
        })))
    }

    /// Render the template against a list of messages. Returns the prompt
    /// text to feed into the tokenizer.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tpl = self.env.get_template("chat")?;
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                Value::from_serialize(&serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                }))
            })
            .collect();
        let ctx = context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };
        tpl.render(ctx).map_err(|e| anyhow!("render chat template: {e}"))
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

/// Deserialization shape for incoming chat messages on the HTTP API.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

impl From<IncomingChatMessage> for ChatMessage {
    fn from(m: IncomingChatMessage) -> Self {
        Self {
            role: m.role,
            content: m.content,
        }
    }
}
