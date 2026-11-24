//! Client for the fennec-coding-agent Axum backend (loopback 127.0.0.1:8787).
//! Provides task CRUD, message sending, agent delegation, and SSE event streaming.
//! Both fennec-desktop and the GPUI GUI consume this instead of raw HTTP.
use crate::http;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub title: String,
    pub user_goal: Option<String>,
    pub repository_id: String,
    pub branch: Option<String>,
    pub working_directory: Option<String>,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Vec<TaskMessage>,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskMessage {
    pub id: String,
    pub task_id: String,
    pub role: String,
    pub content: MessageContent,
    pub status: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Rich(Value),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Rich(v) => v.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SseEvent {
    pub event_id: String,
    pub timestamp: String,
    pub event_type: String,
    pub task_id: Option<String>,
    pub status: String,
    pub progress: f64,
    pub payload: Option<Value>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: String,
    pub implementation: String,
    pub protocol_version: u32,
    pub version: String,
    pub features: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GithubAccount {
    pub login: String,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub html_url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GithubAccountStatus {
    pub authenticated: bool,
    pub account: Option<GithubAccount>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GithubRepository {
    pub name: String,
    pub full_name: String,
    pub private: bool,
    pub html_url: String,
    pub default_branch: Option<String>,
    pub owner_login: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GithubRepositoryStatus {
    pub authenticated: bool,
    pub repositories: Vec<GithubRepository>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ObsidianStatus {
    pub configured: bool,
    pub vault: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ObsidianNoteResponse {
    pub path: String,
    pub created: bool,
    pub bytes: usize,
}

impl BackendCapabilities {
    pub fn supports(&self, feature: &str) -> bool {
        self.features.iter().any(|candidate| candidate == feature)
    }
}

#[derive(Clone)]
pub struct CodingAgentClient {
    pub base: String,
}

impl CodingAgentClient {
    pub fn new() -> Self {
        Self {
            base: std::env::var("FENNEC_AGENT_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8787".into()),
        }
    }

    pub fn with_base(base: String) -> Self {
        Self { base }
    }

    fn get(&self, route: &str) -> Result<Value, String> {
        http::get_json(&self.base, route, &[("X-Fennec-Client", "1")])
    }

    /// A GET whose 4xx answer is expected to be a JSON refusal. The inspector's
    /// `work` routes put the reason for a refusal in the body, and the panes
    /// quote it, so `self.get` would throw away the only useful part.
    fn get_reason(&self, route: &str) -> Result<Value, String> {
        http::get_json_reason(&self.base, route, &[("X-Fennec-Client", "1")])
    }

    fn post(&self, route: &str, body: &Value) -> Result<Value, String> {
        http::post_json(&self.base, route, &[("X-Fennec-Client", "1")], body)
    }

    /// Check if the coding-agent backend is reachable.
    pub fn health(&self) -> Option<Value> {
        self.get("/health").ok()
    }

    /// Identify the Rust backend and its supported protocol features.
    pub fn capabilities(&self) -> Option<BackendCapabilities> {
        self.get("/api/capabilities")
            .ok()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    /// List all tasks.
    pub fn list_tasks(&self) -> Vec<AgentTask> {
        self.get("/api/tasks")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    /// Get one task by ID.
    pub fn get_task(&self, id: &str) -> Option<AgentTask> {
        self.get(&format!("/api/tasks/{id}"))
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
    }

    /// Create a new task.
    pub fn create_task(
        &self,
        title: &str,
        user_goal: Option<&str>,
        repository_id: &str,
        branch: Option<&str>,
    ) -> Result<AgentTask, String> {
        let body = json!({
            "title": title,
            "user_goal": user_goal,
            "repository_id": repository_id,
            "branch": branch,
        });
        let v = self.post("/api/tasks", &body)?;
        serde_json::from_value(v).map_err(|e| format!("json: {e}"))
    }

    /// Delete a task.
    pub fn delete_task(&self, id: &str) -> Result<(), String> {
        let (host, _) = http::split_base(&self.base);
        let mut s = http::connect(&self.base, Duration::from_secs(4))?;
        http::write_request(&mut s, &host, "DELETE", &format!("/api/tasks/{id}"), &[("X-Fennec-Client", "1")], None)?;
        let (status, _) = http::read_response(&mut s)?;
        if status == 204 || status == 200 {
            Ok(())
        } else {
            Err(format!("http {status}"))
        }
    }

    /// Start a task (set status to Running).
    pub fn start_task(&self, id: &str) -> Result<AgentTask, String> {
        let v = self.post(&format!("/api/tasks/{id}/start"), &json!({}))?;
        serde_json::from_value(v).map_err(|e| format!("json: {e}"))
    }

    /// Stop a task (set status to Cancelled).
    pub fn stop_task(&self, id: &str) -> Result<AgentTask, String> {
        let v = self.post(&format!("/api/tasks/{id}/stop"), &json!({}))?;
        serde_json::from_value(v).map_err(|e| format!("json: {e}"))
    }

    /// Send a message to a task. The backend processes it through the engine.
    pub fn send_message(&self, task_id: &str, content: &str) -> Result<TaskMessage, String> {
        let body = json!({ "content": content });
        let v = self.post(&format!("/api/tasks/{task_id}/messages"), &body)?;
        serde_json::from_value(v).map_err(|e| format!("json: {e}"))
    }

    /// List available agents.
    pub fn list_agents(&self) -> Vec<Value> {
        self.get("/api/agents")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    /// Read the local GitHub CLI authentication state.
    pub fn github_account(&self) -> GithubAccountStatus {
        self.get("/api/github/account")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_else(|| GithubAccountStatus {
                authenticated: false,
                account: None,
                error: Some("GitHub status unavailable".into()),
            })
    }

    /// Read the repositories visible to the local GitHub CLI account.
    pub fn github_repositories(&self) -> GithubRepositoryStatus {
        self.get("/api/github/repositories")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_else(|| GithubRepositoryStatus {
                authenticated: false,
                repositories: Vec::new(),
                error: Some("GitHub repositories unavailable".into()),
            })
    }

    /// Create a remote repository through the backend's local GitHub CLI.
    pub fn create_github_repository(&self, name: &str, private: bool) -> Result<GithubRepository, String> {
        let value = self.post("/api/github/repositories", &json!({
            "name": name,
            "private": private,
        }))?;
        serde_json::from_value(value).map_err(|error| format!("json: {error}"))
    }

    /// Read the configured local Obsidian vault without touching credentials.
    pub fn obsidian_status(&self) -> ObsidianStatus {
        self.get("/api/obsidian/status")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_else(|| ObsidianStatus {
                configured: false,
                vault: None,
                message: "Obsidian status unavailable".into(),
            })
    }

    /// Write a note through the backend's vault-jail validation.
    pub fn obsidian_write_note(
        &self,
        title: &str,
        content: &str,
        folder: Option<&str>,
        overwrite: bool,
    ) -> Result<ObsidianNoteResponse, String> {
        let value = self.post("/api/obsidian/notes", &json!({
            "title": title,
            "content": content,
            "folder": folder,
            "overwrite": overwrite,
        }))?;
        serde_json::from_value(value).map_err(|error| format!("json: {error}"))
    }

    /// Delegate work to another agent.
    pub fn delegate(
        &self,
        task_id: &str,
        target_agent: &str,
        scope: &str,
        instructions: &str,
    ) -> Result<Value, String> {
        let body = json!({
            "target_agent": target_agent,
            "scope": scope,
            "instructions": instructions,
        });
        self.post(&format!("/api/tasks/{task_id}/delegate"), &body)
    }

    /// List registered repositories.
    pub fn list_repositories(&self) -> Vec<Value> {
        self.get("/api/repositories")
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    /// Register a repository.
    pub fn register_repository(&self, name: &str, local_path: &str) -> Result<Value, String> {
        let body = json!({ "name": name, "local_path": local_path });
        self.post("/api/repositories", &body)
    }

    /// Get repository status (git info).
    pub fn repository_status(&self, id: &str) -> Option<Value> {
        self.get(&format!("/api/repositories/{id}/status")).ok()
    }

    /// Subscribe to the SSE event stream. Blocks until `cancel` flips or the
    /// socket closes. Calls `on_event` for each event received.
    pub fn subscribe_events(
        &self,
        cancel: &AtomicBool,
        mut on_event: impl FnMut(SseEvent),
    ) -> Result<(), String> {
        http::read_sse(&self.base, "/api/events", &[("X-Fennec-Client", "1")], cancel, |data| {
            if let Ok(ev) = serde_json::from_str::<SseEvent>(data) {
                on_event(ev);
            }
        })
    }

    /// Create a task and send the first message in one call.
    /// Convenience for the common "user types a prompt" flow.
    pub fn new_task_with_message(
        &self,
        title: &str,
        content: &str,
        repository_id: &str,
    ) -> Result<(AgentTask, TaskMessage), String> {
        let task = self.create_task(title, Some(content), repository_id, None)?;
        self.start_task(&task.id)?;
        let msg = self.send_message(&task.id, content)?;
        Ok((task, msg))
    }

    /// One directory listing for the inspector's Files tab. Empty `workspace`
    /// means the server's own root; see the backend's `work` module.
    pub fn work_files(&self, path: &str, workspace: &str) -> Result<Value, String> {
        self.get_reason(&format!(
            "/api/work/files?path={}&workspace={}",
            encode_query_value(path),
            encode_query_value(workspace),
        ))
    }

    /// One file preview for the Files tab — text, or an image as base64 under
    /// `mime`/`data`. Capped by the server, which refuses what it will not send.
    pub fn work_file(&self, path: &str, workspace: &str) -> Result<Value, String> {
        self.get_reason(&format!(
            "/api/work/file?path={}&workspace={}",
            encode_query_value(path),
            encode_query_value(workspace),
        ))
    }

    /// Read one web page as text for the inspector's Browser tab.
    pub fn browse(&self, url: &str) -> Result<Value, String> {
        self.post("/api/browse", &json!({ "url": url }))
    }
}

/// Percent-encode a query-component value. Path and workspace strings arrive
/// from the pane and may hold spaces, backslashes, `&`, `#`, or non-ASCII.
fn encode_query_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

impl Default for CodingAgentClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The process-wide environment can only be mutated under a lock; without
    // it these two tests race each other's FENNEC_AGENT_URL.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn default_base_is_loopback_8787() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let c = CodingAgentClient::new();
        assert_eq!(c.base, "http://127.0.0.1:8787");
    }

    #[test]
    fn env_var_overrides_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: held under ENV_LOCK, no other thread reads this env var concurrently.
        unsafe { std::env::set_var("FENNEC_AGENT_URL", "http://127.0.0.1:9999") };
        let c = CodingAgentClient::new();
        assert_eq!(c.base, "http://127.0.0.1:9999");
        // SAFETY: restore prior state.
        unsafe { std::env::remove_var("FENNEC_AGENT_URL") };
    }

    #[test]
    fn message_content_text_round_trips() {
        let mc = MessageContent::Text("hello".into());
        assert_eq!(mc.as_text(), "hello");
    }

    #[test]
    fn sse_event_deserializes() {
        let json = r#"{"event_id":"abc","timestamp":"2026-01-01T00:00:00Z","event_type":"TaskCreated","task_id":"t1","status":"created","progress":0.0}"#;
        let ev: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(ev.event_type, "TaskCreated");
        assert_eq!(ev.task_id, Some("t1".into()));
    }

    #[test]
    fn capabilities_report_supported_features() {
        let caps: BackendCapabilities = serde_json::from_value(serde_json::json!({
            "backend": "fennec-coding-agent",
            "implementation": "rust-axum",
            "protocol_version": 1,
            "version": "1.0.0",
            "features": ["tasks", "events-sse"]
        })).unwrap();
        assert!(caps.supports("tasks"));
        assert!(caps.supports("events-sse"));
        assert!(!caps.supports("events-websocket"));
    }

    #[test]
    fn query_values_escape_what_a_path_can_contain() {
        assert_eq!(encode_query_value("a b/c"), "a%20b%2Fc");
        assert_eq!(encode_query_value("a&b#c=d%e"), "a%26b%23c%3Dd%25e");
        assert_eq!(encode_query_value("保留"), "%E4%BF%9D%E7%95%99");
        assert_eq!(encode_query_value("ok-1.0~_"), "ok-1.0~_");
    }
}
