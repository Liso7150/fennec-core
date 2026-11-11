//! The ACP pipe's pure parts, ported from `fennec_core/acp.py`: newline-delimited
//! JSON framing and the two bounds a streamed update passes through (`_clip`,
//! `_pieces`). Fennec's transport is one JSON object per line, so the codec is
//! the hot path of every token the agent emits.
//!
//! The agent itself - session store, approvals, MCP, the engine client - stays
//! in Python: it is orchestration over sockets and subprocesses, and a second
//! implementation of it would be a second thing to keep honest.

use fennec_pyjson::{dumps_compact, object, Value};

/// One outbound message: what `Agent._write` sends, `json.dumps` with the
/// compact separators it passes, plus the newline that lets the peer find its end.
pub fn encode_frame(message: &Value) -> String {
    let mut out = dumps_compact(message);
    out.push('\n');
    out
}

/// `json.loads` of one inbound line, or the parse error text ACP puts in the
/// JSON-RPC error object.
pub fn decode_frame(line: &str) -> Result<Value, String> {
    serde_json::from_str(line).map_err(|error| error.to_string())
}

/// Read a whole stream the way `Agent.run` does: one message per line, empty
/// lines skipped, a parse failure reported with the line that caused it.
pub fn decode_stream(raw: &str) -> (Vec<Value>, Vec<String>) {
    let mut frames = Vec::new();
    let mut errors = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match decode_frame(line) {
            Ok(value) => frames.push(value),
            Err(error) => errors.push(format!("{error}: {line}")),
        }
    }
    (frames, errors)
}

/// `_clip(value, limit)`: bound what a stored or streamed update can carry.
pub fn clip(value: &Value, limit: usize) -> Value {
    match value {
        Value::String(text) => {
            let length = text.chars().count();
            if length <= limit {
                return Value::String(text.clone());
            }
            let head: String = text.chars().take(limit).collect();
            Value::String(format!("{head}…[{} chars omitted]", length - limit))
        }
        Value::Object(map) => {
            let mut out = object();
            let inner = (200).max(limit / 2);
            for (key, item) in map {
                out.insert(key.clone(), clip(item, inner));
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            let inner = (200).max(limit / 2);
            Value::Array(items.iter().take(40).map(|item| clip(item, inner)).collect())
        }
        other => other.clone(),
    }
}

/// `_pieces(text, size)`: the chunks a streamed reply is cut into. Code points,
/// like Python's slicing.
pub fn pieces(text: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= size || size == 0 {
        return vec![text.to_string()];
    }
    chars.chunks(size).map(|chunk| chunk.iter().collect()).collect()
}
