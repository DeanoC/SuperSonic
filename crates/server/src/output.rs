use serde_json::{Map, Value};

use crate::schemas::{OpenAiFunctionCall, OpenAiToolCall};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Clone, Default)]
pub struct AssistantOutput {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssistantOutputParseContext {
    prefilled_think: bool,
}

impl AssistantOutputParseContext {
    pub fn from_rendered_prompt(prompt: &str) -> Self {
        let Some(open) = prompt.rfind(THINK_OPEN) else {
            return Self::default();
        };
        let close = prompt.rfind(THINK_CLOSE);
        let unmatched = close.is_none_or(|close| open > close)
            && prompt[open + THINK_OPEN.len()..].trim().is_empty();
        Self {
            prefilled_think: unmatched,
        }
    }

    pub fn has_incomplete_think(self, raw: &str) -> bool {
        let mut in_reasoning = self.prefilled_think;
        let mut rest = raw;
        loop {
            match next_think_tag(rest) {
                Some((offset, ThinkTag::Open)) => {
                    in_reasoning = true;
                    rest = &rest[offset + THINK_OPEN.len()..];
                }
                Some((offset, ThinkTag::Close)) => {
                    in_reasoning = false;
                    rest = &rest[offset + THINK_CLOSE.len()..];
                }
                None => {
                    return in_reasoning
                        || ends_with_partial_tag(raw, THINK_OPEN)
                        || ends_with_partial_tag(raw, THINK_CLOSE)
                }
            }
        }
    }
}

pub fn parse_assistant_output(raw: &str) -> AssistantOutput {
    parse_assistant_output_with_context(raw, AssistantOutputParseContext::default())
}

pub fn parse_assistant_output_with_context(
    raw: &str,
    context: AssistantOutputParseContext,
) -> AssistantOutput {
    let (without_reasoning, reasoning) = strip_think_with_context(raw, context);
    let (content, tool_calls) = extract_tool_calls(&without_reasoning);
    AssistantOutput {
        content: content.trim_start().to_string(),
        reasoning_content: reasoning.filter(|s| !s.trim().is_empty()),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

pub fn strip_think(raw: &str) -> (String, Option<String>) {
    strip_think_with_context(raw, AssistantOutputParseContext::default())
}

fn strip_think_with_context(
    raw: &str,
    context: AssistantOutputParseContext,
) -> (String, Option<String>) {
    let mut visible = String::new();
    let mut reasoning = String::new();
    let mut in_reasoning = context.prefilled_think;
    let mut rest = raw;

    loop {
        let Some((offset, tag)) = next_think_tag(rest) else {
            if in_reasoning {
                reasoning.push_str(rest);
            } else {
                visible.push_str(rest);
            }
            break;
        };
        if in_reasoning {
            reasoning.push_str(&rest[..offset]);
        } else {
            visible.push_str(&rest[..offset]);
        }
        match tag {
            ThinkTag::Open => {
                in_reasoning = true;
                rest = &rest[offset + THINK_OPEN.len()..];
            }
            ThinkTag::Close => {
                in_reasoning = false;
                rest = &rest[offset + THINK_CLOSE.len()..];
            }
        }
    }

    let reasoning = reasoning.trim().to_string();
    (visible, (!reasoning.is_empty()).then_some(reasoning))
}

#[derive(Clone, Copy)]
enum ThinkTag {
    Open,
    Close,
}

fn next_think_tag(raw: &str) -> Option<(usize, ThinkTag)> {
    match (raw.find(THINK_OPEN), raw.find(THINK_CLOSE)) {
        (Some(open), Some(close)) if open <= close => Some((open, ThinkTag::Open)),
        (Some(_), Some(close)) => Some((close, ThinkTag::Close)),
        (Some(open), None) => Some((open, ThinkTag::Open)),
        (None, Some(close)) => Some((close, ThinkTag::Close)),
        (None, None) => None,
    }
}

fn ends_with_partial_tag(raw: &str, tag: &str) -> bool {
    (1..tag.len()).any(|prefix_len| raw.ends_with(&tag[..prefix_len]))
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
    fn parses_generated_think_block_without_prompt_context() {
        let out = parse_assistant_output("<think>plan</think>answer");
        assert_eq!(out.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(out.content, "answer");
    }

    #[test]
    fn parses_prefilled_open_think_from_rendered_prompt() {
        let context =
            AssistantOutputParseContext::from_rendered_prompt("<|im_start|>assistant\n<think>\n");
        let out = parse_assistant_output_with_context("plan</think>answer", context);
        assert_eq!(out.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(out.content, "answer");
    }

    #[test]
    fn incomplete_prefilled_think_is_reasoning_without_tag_leakage() {
        let context = AssistantOutputParseContext::from_rendered_prompt("assistant<think>");
        let out = parse_assistant_output_with_context("unfinished plan", context);
        assert_eq!(out.reasoning_content.as_deref(), Some("unfinished plan"));
        assert!(out.content.is_empty());
        assert!(!format!("{out:?}").contains("<think>"));
        assert!(context.has_incomplete_think("unfinished plan"));
    }

    #[test]
    fn incomplete_generated_think_is_reasoning_without_tag_leakage() {
        let out = parse_assistant_output("<think>unfinished plan");
        assert_eq!(out.reasoning_content.as_deref(), Some("unfinished plan"));
        assert!(out.content.is_empty());
        assert!(!format!("{out:?}").contains("<think>"));
    }

    #[test]
    fn content_after_prefilled_close_stays_visible() {
        let context = AssistantOutputParseContext::from_rendered_prompt("assistant<think>\n");
        let out = parse_assistant_output_with_context(
            "reasoning tokens</think>\n\nvisible answer",
            context,
        );
        assert_eq!(out.reasoning_content.as_deref(), Some("reasoning tokens"));
        assert_eq!(out.content, "visible answer");
        assert!(!out.content.contains("</think>"));
    }

    #[test]
    fn duplicate_generated_control_tags_never_leak_in_prefilled_mode() {
        let context = AssistantOutputParseContext::from_rendered_prompt("assistant<think>\n");
        let out = parse_assistant_output_with_context(
            "<think>plan</think>answer <think>more</think> done",
            context,
        );
        let serialized = format!("{out:?}");
        assert!(!serialized.contains("<think>"));
        assert!(!serialized.contains("</think>"));
    }

    #[test]
    fn ordinary_output_remains_non_reasoning() {
        let context = AssistantOutputParseContext::from_rendered_prompt("<|im_start|>assistant\n");
        let out = parse_assistant_output_with_context("ordinary answer", context);
        assert_eq!(out.reasoning_content, None);
        assert_eq!(out.content, "ordinary answer");
        assert!(!context.has_incomplete_think("ordinary answer"));
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
