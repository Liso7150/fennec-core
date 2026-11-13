//! Context economy for the agent loop - the Rust port of
//! `fennec_core/context.py`. Prefill dominates a turn on the Fennec engine, so
//! every character the loop re-sends each step costs real seconds. The UI keeps
//! the full transcript; only the payload sent to the model is compacted:
//! head/tail elision per result, duplicate results replaced by a reference, and
//! a total budget that shrinks the oldest tool results first.
//!
//! The rules, defaults and the exact strings are the Python module's, and the
//! golden tests in `tests/python_golden.rs` fail if this ever drifts.

use fennec_pyjson::char_len;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub const DEFAULT_BUDGET_CHARS: usize = 48_000;
pub const DEFAULT_RESULT_CHARS: usize = 6_000;
pub const MIN_RESULT_CHARS: usize = 600;
pub const MIN_BUDGET_CHARS: usize = 2_000;
const MARKER_HEAD: &str = "\n…[";
const MARKER_TAIL: &str = " chars elided; narrow the read or grep for the rest]\n";
/// head + tail + the marker without its digits: `elide` budgets 80 characters
/// of it, exactly as the Python module does.
const MARKER_COST: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub budget_chars: usize,
    pub result_chars: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub chars_before: usize,
    pub chars_after: usize,
    pub elided: usize,
    pub deduped: usize,
    pub budget_chars: usize,
    pub result_chars: usize,
}

fn positive_int(raw: Option<String>, default: usize, floor: usize) -> usize {
    let Some(text) = raw else { return default };
    match text.trim().parse::<i64>() {
        Ok(parsed) if parsed > 0 => floor.max(parsed as usize),
        _ => default,
    }
}

pub fn limits_in(env: &dyn Fn(&str) -> Option<String>) -> Limits {
    let result = positive_int(
        env("FENNEC_AGENT_RESULT_CHARS"),
        DEFAULT_RESULT_CHARS,
        MIN_RESULT_CHARS,
    );
    let budget = positive_int(
        env("FENNEC_AGENT_CONTEXT_BUDGET"),
        DEFAULT_BUDGET_CHARS,
        MIN_BUDGET_CHARS,
    );
    Limits { budget_chars: budget.max(result), result_chars: result }
}

pub fn limits() -> Limits {
    limits_in(&|key| std::env::var(key).ok())
}

/// Keep the head and the tail of `text`, marking what was removed.
pub fn elide(text: &str, limit: usize) -> String {
    let length = text.chars().count();
    if length <= limit {
        return text.to_string();
    }
    let head = ((limit * 2) / 3).max(1);
    let tail = limit.saturating_sub(head + MARKER_COST);
    let removed = length - head - tail;
    let mut out = String::with_capacity(limit + MARKER_COST);
    out.extend(text.chars().take(head));
    out.push_str(MARKER_HEAD);
    out.push_str(&removed.to_string());
    out.push_str(MARKER_TAIL);
    if tail > 0 {
        out.extend(text.chars().skip(length - tail));
    }
    out
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().map_or(true, |f| f != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

/// `len(content) + len(json.dumps(tool_calls, ensure_ascii=False))`
pub fn size(message: &Value) -> usize {
    let content = message.get("content").and_then(Value::as_str).map(str::chars).map(Iterator::count).unwrap_or(0);
    match message.get("tool_calls") {
        Some(calls) if py_truthy(calls) => content + char_len(calls),
        _ => content,
    }
}

pub fn compact(messages: &[Value], options: &Options, seen: &mut HashMap<String, String>) -> (Vec<Value>, Stats) {
    let before: usize = messages.iter().map(size).sum();
    let mut payload: Vec<Value> = Vec::with_capacity(messages.len());
    let mut elided = 0;
    let mut deduped = 0;

    for message in messages {
        let mut copy = message.clone();
        let content = copy.get("content").and_then(Value::as_str).map(str::to_string);
        let is_tool = copy.get("role").and_then(Value::as_str) == Some("tool");
        if let (true, Some(text)) = (is_tool, content) {
            if !text.is_empty() {
                let mut hasher = Sha256::new();
                hasher.update(text.as_bytes());
                let digest = format!("{:x}", hasher.finalize());
                let label = copy
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("an earlier call")
                    .to_string();
                if let Some(first) = seen.get(&digest) {
                    let replaced = format!(
                        "[identical to the result of {first}; {} chars not repeated]",
                        text.chars().count()
                    );
                    set_content(&mut copy, &replaced);
                    deduped += 1;
                } else {
                    seen.insert(digest, label);
                    let capped = elide(&text, options.result_chars);
                    if capped != text {
                        set_content(&mut copy, &capped);
                        elided += 1;
                    }
                }
            }
        }
        payload.push(copy);
    }

    let mut total: usize = payload.iter().map(size).sum();
    for message in &mut payload {
        if total <= options.budget_chars {
            break;
        }
        let content = match message.get("content").and_then(Value::as_str) {
            Some(text) if text.chars().count() > options.floor_chars => text.to_string(),
            _ => continue,
        };
        if message.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        let length = content.chars().count();
        let overshoot = total.saturating_sub(options.budget_chars);
        let target = options.floor_chars.max(length.saturating_sub(overshoot));
        if target >= length {
            continue;
        }
        let shrunk = elide(&content, target);
        let old_size = size(message);
        set_content(message, &shrunk);
        total = total - old_size + size(message);
    }

    (
        payload,
        Stats {
            chars_before: before,
            chars_after: total,
            elided,
            deduped,
            budget_chars: options.budget_chars,
            result_chars: options.result_chars,
        },
    )
}

fn set_content(message: &mut Value, text: &str) {
    if let Some(object) = message.as_object_mut() {
        object.insert("content".to_string(), Value::String(text.to_string()));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub budget_chars: usize,
    pub result_chars: usize,
    pub floor_chars: usize,
}

impl Default for Options {
    fn default() -> Self {
        let limits = limits();
        Options {
            budget_chars: limits.budget_chars,
            result_chars: limits.result_chars,
            floor_chars: MIN_RESULT_CHARS,
        }
    }
}

impl Options {
    pub fn sized(budget_chars: usize, result_chars: usize) -> Self {
        Options { budget_chars, result_chars, floor_chars: MIN_RESULT_CHARS }
    }
}
