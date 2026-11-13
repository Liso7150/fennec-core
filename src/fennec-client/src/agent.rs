//! Client for the fennec-core agent server (`fennec-agent serve`, loopback
//! 127.0.0.1:8787 by default). Mirrors `fennec_core/server.py`; every route
//! under `/api/*` except health requires the `X-Fennec-Client: 1` header.
use crate::http;
use serde_json::{json, Value};
use std::time::Duration;

/// `server.py:CLIENT_HEADER` — without it the service answers 403.
pub const CLIENT_HEADER: (&str, &str) = ("X-Fennec-Client", "1");

#[derive(Clone)]
pub struct Core {
    pub base: String,
}

impl Core {
    pub fn new() -> Self {
        Self { base: std::env::var("FENNEC_CORE_URL")
            .or_else(|_| std::env::var("FENNEC_SERVICE_URL"))
            .unwrap_or_else(|_| "http://127.0.0.1:8787".into()) }
    }

    fn get(&self, route: &str) -> Result<Value, String> {
        http::get_json(&self.base, route, &[CLIENT_HEADER])
    }
    fn post(&self, route: &str, body: &Value, timeout: Duration) -> Result<Value, String> {
        let mut s = http::connect(&self.base, timeout)?;
        let host = http::split_base(&self.base).0;
        http::write_request(&mut s, &host, "POST", route, &[CLIENT_HEADER], Some(&body.to_string()))?;
        let (status, text) = http::read_response(&mut s)?;
        if status != 200 {
            return Err(format!("http {status}: {text}"));
        }
        serde_json::from_str(&text).map_err(|e| format!("json: {e}"))
    }

    /// `{app, engine_running, workspace}` — the service also reports whether the
    /// engine it supervises is up, which is the desktop's engine indicator.
    pub fn health(&self) -> Option<Value> {
        self.get("/api/health").ok()
    }
    pub fn engine_status(&self) -> Option<Value> {
        self.get("/api/engine/status").ok()
    }
    pub fn engine_telemetry(&self) -> Option<Value> {
        self.get("/api/engine").ok()
    }
    pub fn outlets(&self) -> Option<Value> {
        self.get("/api/outlets").ok()
    }
    pub fn sessions(&self) -> Vec<Value> {
        self.get("/api/agent/sessions").ok()
            .and_then(|v| v["sessions"].as_array().cloned())
            .unwrap_or_default()
    }

    /// Start a supervised coding task. The `allow_*` flags are the loop's jail
    /// switches (`loop.py`), and `max_steps` is its own budget default of 14.
    pub fn start(&self, task: &str, workspace: &str, allow_write: bool, allow_shell: bool, allow_web: bool) -> Result<String, String> {
        let body = json!({ "task": task, "workspace": workspace, "allow_write": allow_write,
                           "allow_shell": allow_shell, "allow_web": allow_web, "max_steps": 14 });
        let v = self.post("/api/agent/start", &body, Duration::from_millis(4000))?;
        v["id"].as_str().map(str::to_string).ok_or_else(|| "no id in response".into())
    }

    /// Steps with index `>= after`, plus the session's task and status.
    pub fn poll(&self, id: &str, after: usize) -> Result<Value, String> {
        self.get(&format!("/api/agent/poll?id={id}&after={after}"))
    }

    pub fn cancel(&self, id: &str) -> Result<(), String> {
        self.post("/api/agent/cancel", &json!({ "id": id }), Duration::from_millis(2000)).map(|_| ())
    }

    /// Ask the service to start the engine it supervises. The engine lifecycle
    /// belongs to `fennec_core.engine`, not to this client — it is a long-running
    /// background thread server-side, so this returns as soon as it is queued.
    pub fn engine_start(&self) -> Result<Value, String> {
        self.post("/api/engine/start", &json!({}), Duration::from_millis(2000))
    }
}

impl Default for Core {
    fn default() -> Self {
        Self::new()
    }
}
