//! Client for the Fennec inference engine (loopback 127.0.0.1:8184), the Rust
//! counterpart of `engine.py`. Chat is streamed as SSE:
//! `data: {json}\n\n`, terminated by `data: [DONE]`, and the socket closing
//! early is how the engine detects a cancelled client (`src/serve.rs`).
use crate::http;
use serde_json::{json, Value};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

#[derive(Clone)]
pub enum ChatDelta {
    Reasoning(String),
    Content(String),
    Done,
    Failed(String),
}

#[derive(Clone)]
pub struct Engine {
    pub base: String,
    pub key: String,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            base: std::env::var("FENNEC_ENGINE_URL").unwrap_or_else(|_| "http://127.0.0.1:8184".into()),
            key: std::env::var("FENNEC_API_KEY")
                .or_else(|_| std::env::var("COLI_API_KEY"))
                .unwrap_or_else(|_| "local".into()),
        }
    }

    fn auth_header(&self) -> String {
        http::bearer(&self.key)
    }

    pub fn health(&self) -> Option<Value> {
        let auth = self.auth_header();
        http::get_json(&self.base, "/health", &[("Authorization", auth.as_str())]).ok()
    }
    pub fn metrics(&self) -> Option<Value> {
        let auth = self.auth_header();
        http::get_json(&self.base, "/metrics", &[("Authorization", auth.as_str())]).ok()
    }
    pub fn model_id(&self) -> Option<(String, u64)> {
        let auth = self.auth_header();
        let v = http::get_json(&self.base, "/v1/models", &[("Authorization", auth.as_str())]).ok()?;
        let m = v["data"].as_array()?.first()?;
        Some((m["id"].as_str()?.to_string(), m["max_model_len"].as_u64().unwrap_or(0)))
    }

    /// Stream one chat turn. Runs on the caller's thread; emits deltas into
    /// `tx` and stops early when `cancel` flips — hanging up is the engine's
    /// documented cancellation signal.
    pub fn chat_stream(&self, messages: &Value, max_tokens: u32, tx: &Sender<ChatDelta>, cancel: &AtomicBool) {
        if let Err(e) = self.stream_inner(messages, max_tokens, tx, cancel) {
            let _ = tx.send(ChatDelta::Failed(e));
        }
    }

    fn stream_inner(&self, messages: &Value, max_tokens: u32, tx: &Sender<ChatDelta>, cancel: &AtomicBool) -> Result<(), String> {
        let body = json!({ "messages": messages, "stream": true, "max_tokens": max_tokens, "temperature": 0.0 });
        let text = body.to_string();
        let (host, _) = http::split_base(&self.base);
        let mut s = http::connect(&self.base, Duration::from_secs(120)).map_err(|e| e.to_string())?;
        let auth = http::bearer(&self.key);
        http::write_request(&mut s, &host, "POST", "/v1/chat/completions",
                            &[("Authorization", auth.as_str())], Some(&text))?;
        let mut reader = std::io::BufReader::new(s);
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("closed during headers".into());
            }
            if line.trim().is_empty() {
                break;
            }
        }
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Ok(()); // dropping the socket cancels the turn
            }
            line.clear();
            let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            let Some(data) = line.trim().strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data.trim()) else { continue };
            let delta = &v["choices"][0]["delta"];
            if let Some(r) = delta["reasoning_content"].as_str() {
                if !r.is_empty() {
                    let _ = tx.send(ChatDelta::Reasoning(r.to_string()));
                }
            }
            if let Some(c) = delta["content"].as_str() {
                if !c.is_empty() {
                    let _ = tx.send(ChatDelta::Content(c.to_string()));
                }
            }
        }
        let _ = tx.send(ChatDelta::Done);
        Ok(())
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
