//! Client for the fennec-core agent server (`fennec-agent serve`, loopback
//! 127.0.0.1:8787 by default). Mirrors `fennec_core/server.py`; every route
//! under `/api/*` except health requires the `X-Fennec-Client: 1` header.
use crate::http;
use serde_json::{json, Value};
use std::time::Duration;

/// `server.py:CLIENT_HEADER` — without it the service answers 403.
pub const CLIENT_HEADER: (&str, &str) = ("X-Fennec-Client", "1");

/// `Outlets.available()` gives Jev 240 s and OpenBot 900 s, and the answer only
/// arrives when the run ends, so the read timeout has to outlast the longer one.
pub const OUTLET_RUN_TIMEOUT: Duration = Duration::from_secs(960);

/// One optional web-agent outlet, as `fennec_core/outlets.py` probes it.
///
/// `present` is the gate core itself applies — `/api/outlets/run` answers 409
/// for exactly this field — so a shell that enables a button on anything else
/// either offers a run core will refuse or refuses one it would serve. The
/// status payload has never carried an `available` key.
#[derive(Clone, Debug, PartialEq)]
pub struct Outlet {
    /// `"jev"` / `"openbot"` — the value `/api/outlets/run` takes as `outlet`.
    pub id: &'static str,
    pub label: &'static str,
    pub present: bool,
    /// How it is installed, or the reason it is not, built from core's probe.
    pub detail: String,
    /// OpenBot coworkers as `agent=` accepts them (name, else id). Jev takes a
    /// url instead of an agent, so its list stays empty.
    pub agents: Vec<String>,
}

/// Both outlets, in the order the environment panel lists them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outlets {
    pub jev: Option<Outlet>,
    pub openbot: Option<Outlet>,
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

impl Outlets {
    /// Reads the shape `outlets.py:status()` returns. A key core does not send
    /// stays `None`, which renders as that outlet being unknown rather than
    /// pretending it is absent.
    pub fn from_value(v: &Value) -> Self {
        Self { jev: Outlet::jev(v.get("jev")), openbot: Outlet::openbot(v.get("openbot")) }
    }

    pub fn list(&self) -> Vec<&Outlet> {
        self.jev.iter().chain(self.openbot.iter()).collect()
    }

    /// True only when core sent that outlet and it is installed.
    pub fn ready(&self, id: &str) -> bool {
        self.get(id).is_some_and(|o| o.present)
    }

    pub fn get(&self, id: &str) -> Option<&Outlet> {
        match id {
            "jev" => self.jev.as_ref(),
            "openbot" => self.openbot.as_ref(),
            _ => None,
        }
    }
}

impl Outlet {
    fn jev(v: Option<&Value>) -> Option<Self> {
        let v = v?;
        let present = flag(v, "present");
        let mut bits: Vec<String> = Vec::new();
        if v.get("root").and_then(Value::as_str).is_some() {
            bits.push("checkout".into());
        }
        if flag(v, "importable") {
            bits.push("importable".into());
        }
        match v.get("launcher").and_then(Value::as_str) {
            Some(l) => bits.push(format!("via {l}")),
            // `_jev_command` falls back to the interpreter that runs core, so a
            // missing uv is a slower start, not an unusable outlet.
            None => bits.push("via this interpreter".into()),
        }
        bits.push(if flag(v, "policy_key") { "policy key set".into() } else { "no policy key".into() });
        Some(Self {
            id: "jev",
            label: "Jev",
            present,
            detail: if present { format!("installed — {}", bits.join(" · ")) } else { "not installed".into() },
            agents: Vec::new(),
        })
    }

    fn openbot(v: Option<&Value>) -> Option<Self> {
        let v = v?;
        let present = flag(v, "present");
        let error = v.get("error").and_then(Value::as_str);
        let agents: Vec<String> = v.get("agents").and_then(Value::as_array).map(|list| {
            list.iter()
                .map(|a| a.get("name").and_then(Value::as_str)
                    .or_else(|| a.get("id").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string())
                .filter(|name| !name.is_empty())
                .collect()
        }).unwrap_or_default();
        let host = v.get("url").and_then(Value::as_str).unwrap_or("");
        // "not answering" is only true when nothing refused the probe; a probe
        // that reached a non-loopback or session-gated server has its own reason.
        let detail = match error {
            Some(e) => format!("unavailable — {e}"),
            None if present => format!("{} coworker{} at {host}", agents.len(), if agents.len() == 1 { "" } else { "s" }),
            None => format!("no agent list at {host}"),
        };
        Some(Self { id: "openbot", label: "OpenBot", present, detail, agents })
    }
}

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
    /// What the two optional web agents are on this machine. `None` means the
    /// service could not be reached, which is not the same as both outlets
    /// being uninstalled.
    pub fn outlets(&self) -> Option<Outlets> {
        self.get("/api/outlets").ok().map(|v| Outlets::from_value(&v))
    }
    /// Ask the service to re-probe instead of using its cache (5 min for Jev,
    /// 20 s for OpenBot), which is what a "check again" affordance needs.
    pub fn outlets_refresh(&self) -> Option<Outlets> {
        self.get("/api/outlets?refresh=true").ok().map(|v| Outlets::from_value(&v))
    }
    pub fn sessions(&self) -> Vec<Value> {
        self.get("/api/agent/sessions").ok()
            .and_then(|v| v["sessions"].as_array().cloned())
            .unwrap_or_default()
    }

    /// Start a supervised coding task. The `allow_*` flags are the loop's jail
    /// switches (`loop.py`), and `max_steps` is its own budget default of 14.
    /// `require_approval` is the client saying it answers `/api/agent/approve`:
    /// core then lifts the flags and asks before every gated call, so what admits
    /// a write is the user's answer rather than a box ticked when the run started.
    pub fn start(&self, task: &str, workspace: &str, allow_write: bool, allow_shell: bool,
                 allow_web: bool, require_approval: bool) -> Result<String, String> {
        let body = json!({ "task": task, "workspace": workspace, "allow_write": allow_write,
                           "allow_shell": allow_shell, "allow_web": allow_web,
                           "require_approval": require_approval, "max_steps": 14 });
        let v = self.post("/api/agent/start", &body, Duration::from_millis(4000))?;
        v["id"].as_str().map(str::to_string).ok_or_else(|| "no id in response".into())
    }

    /// Steps with index `>= after`, plus the session's task and status.
    pub fn poll(&self, id: &str, after: usize) -> Result<Value, String> {
        self.get(&format!("/api/agent/poll?id={id}&after={after}"))
    }

    /// Answer one gated call. `decision` is the loop's own vocabulary —
    /// `allow_once`, `allow_always`, `reject_once`, `reject_always` — and anything
    /// else it refuses. An approval that is no longer pending is a 404, which this
    /// reports as an error: the card has to say the answer did not land.
    pub fn approve(&self, id: &str, approval_id: &str, decision: &str) -> Result<(), String> {
        self.post("/api/agent/approve",
                  &json!({ "id": id, "approval_id": approval_id, "decision": decision }),
                  Duration::from_millis(2000)).map(|_| ())
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

    /// Hand one task to an installed outlet: `outlet` is `"jev"` (a browser end
    /// state at `url`) or `"openbot"` (a `goal` for the named coworker). Core
    /// answers 409 for an outlet its own probe says is absent, and 422 for any
    /// other id. This blocks for as long as the outlet runs — minutes, not
    /// milliseconds — so call it from a worker thread.
    pub fn outlet_run(&self, outlet: &str, goal: &str, url: &str, agent: &str, max_steps: u32) -> Result<Value, String> {
        let body = json!({ "outlet": outlet, "goal": goal, "url": url, "agent": agent, "max_steps": max_steps });
        self.post("/api/outlets/run", &body, OUTLET_RUN_TIMEOUT)
    }
}

impl Default for Core {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two probes verbatim from `outlets.py:_jev_probe` / `_openbot_probe`.
    fn jev(present: bool, root: Option<&str>, importable: bool, launcher: Option<&str>, key: bool) -> Value {
        json!({"present": present, "root": root, "importable": importable, "launcher": launcher,
               "policy_key": key, "paid_api": true, "note": "TypeSafe Jev answers CLICK/TYPE_TEXT/SELECT"})
    }

    #[test]
    fn a_jev_checkout_on_disk_is_installed_and_says_so() {
        let o = Outlets::from_value(&json!({"jev": jev(true, Some("/opt/jev-ultrafast"), false, Some("uv"), true)}));
        assert!(o.ready("jev"));
        assert_eq!(o.jev.as_ref().unwrap().detail, "installed — checkout · via uv · policy key set");
    }

    #[test]
    fn an_absent_jev_is_not_reported_as_a_failure_of_something_else() {
        let o = Outlets::from_value(&json!({"jev": jev(false, None, false, None, false)}));
        let jev = o.jev.unwrap();
        assert!(!jev.present);
        assert_eq!(jev.detail, "not installed");
    }

    #[test]
    fn openbot_reads_its_agents_and_its_refusal() {
        let up = json!({"openbot": {"present": true, "url": "http://127.0.0.1:3001", "loopback": true,
                                    "agents": [{"id": "a1", "name": "Review"}, {"id": "a2", "name": null}]}});
        let o = Outlets::from_value(&up);
        let bot = o.openbot.clone().unwrap();
        assert!(bot.present);
        assert_eq!(bot.detail, "2 coworkers at http://127.0.0.1:3001");
        // `agent=` accepts a name or an id, so an unnamed coworker lists by id.
        assert_eq!(bot.agents, vec!["Review".to_string(), "a2".to_string()]);

        let refused = Outlets::from_value(&json!({"openbot": {"present": false, "url": "https://bot.example",
            "loopback": false, "agents": [], "error": "OpenBot has no machine token for its run API"}}));
        let bot = refused.openbot.unwrap();
        assert!(!bot.present);
        assert_eq!(bot.detail, "unavailable — OpenBot has no machine token for its run API");
    }

    #[test]
    fn a_probe_that_reached_openbot_but_got_no_list_is_not_called_not_installed() {
        let o = Outlets::from_value(&json!({"openbot": {"present": false, "url": "http://127.0.0.1:3001",
                                                        "loopback": true, "agents": []}}));
        assert_eq!(o.openbot.unwrap().detail, "no agent list at http://127.0.0.1:3001");
    }

    /// The bug this replaced: the desktop keyed the rail on `available`, which
    /// core has never emitted, so every outlet read as "not installed" — and a
    /// backend that one day sends it would read as ready while `present` was false.
    #[test]
    fn an_available_key_core_never_sends_cannot_flip_the_gate() {
        let o = Outlets::from_value(&json!({"jev": {"available": true}, "openbot": {"available": true}}));
        assert!(!o.ready("jev"));
        assert!(!o.ready("openbot"));
        assert_eq!(o.jev.as_ref().unwrap().detail, "not installed");
    }

    #[test]
    fn an_outlet_the_service_never_mentioned_stays_unknown() {
        let o = Outlets::from_value(&json!({"jev": jev(true, Some("/opt/jev"), true, None, false)}));
        assert!(o.openbot.is_none());
        assert!(!o.ready("openbot"));
        assert_eq!(o.list().len(), 1);
        assert!(!o.ready("nothing"));
    }
}
