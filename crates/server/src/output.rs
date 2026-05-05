use serde_json::{Map, Value};

use crate::schemas::{OpenAiFunctionCall, OpenAiToolCall};

#[derive(Debug, Clone, Default)]
pub struct AssistantOutput {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

pub fn parse_assistant_output(raw: &str) -> AssistantOutput {
    let (without_reasoning, reasoning) = strip_think(raw);
    let (content, tool_calls) = extract_tool_calls(&without_reasoning);
    AssistantOutput {
        content: content.trim_start().to_string(),
        reasoning_content: reasoning.filter(|s| !s.trim().is_empty()),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

pub fn strip_think(raw: &str) -> (String, Option<String>) {
    let Some(start) = raw.find("<think>") else {
        return (raw.to_string(), None);
    };
    let body_start = start + "<think>".len();
    let Some(end_rel) = raw[body_start..].find("</think>") else {
        return (raw.to_string(), None);
    };
    let end = body_start + end_rel;
    let mut visible = String::new();
    visible.push_str(&raw[..start]);
    visible.push_str(&raw[end + "</think>".len()..]);
    (visible, Some(raw[body_start..end].trim().to_string()))
}

fn extract_tool_calls(raw: &str) -> (String, Vec<OpenAiToolCall>) {
    let mut rest = raw;
    let mut visible = String::new();
    let mut calls = Vec::new();
    while let Some(start) = rest.find("<tool_call>") {
        visible.push_str(&rest[..start]);
        let after_start = &rest[start + "<tool_call>".len()..];
        let Some(end) = after_start.find("</tool_call>") else {
            visible.push_str(&rest[start..]);
            return (visible, calls);
        };
        let block = &after_start[..end];
        if let Some(call) = parse_tool_call_block(block, calls.len()) {
            calls.push(call);
        } else {
            visible.push_str("<tool_call>");
            visible.push_str(block);
            visible.push_str("</tool_call>");
        }
        rest = &after_start[end + "</tool_call>".len()..];
    }
    visible.push_str(rest);
    (visible, calls)
}

fn parse_tool_call_block(block: &str, index: usize) -> Option<OpenAiToolCall> {
    let function_start = block.find("<function=")?;
    let name_start = function_start + "<function=".len();
    let name_end = block[name_start..].find('>')? + name_start;
    let name = block[name_start..name_end].trim().to_string();
    let close = "</function>";
    let body_start = name_end + 1;
    let body_end = block[body_start..].find(close)? + body_start;
    let body = &block[body_start..body_end];

    let mut args = Map::new();
    let mut rest = body;
    while let Some(param_start) = rest.find("<parameter=") {
        let key_start = param_start + "<parameter=".len();
        let key_end = rest[key_start..].find('>')? + key_start;
        let key = rest[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let val_end = rest[val_start..].find("</parameter>")? + val_start;
        let raw_val = rest[val_start..val_end].trim();
        let value =
            serde_json::from_str(raw_val).unwrap_or_else(|_| Value::String(raw_val.to_string()));
        args.insert(key, value);
        rest = &rest[val_end + "</parameter>".len()..];
    }

    Some(OpenAiToolCall {
        id: format!("call_{index:04x}"),
        type_: "function",
        function: OpenAiFunctionCall {
            name,
            arguments: Value::Object(args).to_string(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_reasoning_and_keeps_content() {
        let out = parse_assistant_output("<think>\nwork\n</think>\n\nanswer");
        assert_eq!(out.reasoning_content.as_deref(), Some("work"));
        assert_eq!(out.content.trim(), "answer");
    }

    #[test]
    fn parses_qwen_xml_tool_call() {
        let raw = "<tool_call>\n<function=lookup>\n<parameter=query>\nweather\n</parameter>\n</function>\n</tool_call>";
        let out = parse_assistant_output(raw);
        let calls = out.tool_calls.expect("tool calls");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, "{\"query\":\"weather\"}");
    }
}
