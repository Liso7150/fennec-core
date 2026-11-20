//! Client for the fennec-core agent server (`fennec-agent serve`, loopback
//! 127.0.0.1:8787 by default). Mirrors `server.py`; every route
//! under `/api/*` except health requires the `X-Fennec-Client: 1` header.
use crate::http;
use serde_json::{json, Value};
use std::time::Duration;

/// `server.py:CLIENT_HEADER` — without it the service answers 403.
pub const CLIENT_HEADER: (&str, &str) = ("X-Fennec-Client", "1");

/// Core gives the browser lane 240 s and OpenBot 900 s, each with a grace period
/// to die in. The answer only arrives when the run ends, so the read timeout has
/// to outlast the longest single route.
pub const OUTLET_RUN_TIMEOUT: Duration = Duration::from_secs(960);

/// One optional outlet, as `outlets.py` probes it.
///
/// `present` is the gate core itself applies — `/api/outlets/run` answers 409
/// for exactly this field — so a shell that enables a button on anything else
/// either offers a run core will refuse or refuses one it would serve. The
/// status payload has never carried an `available` key.
#[derive(Clone, Debug, PartialEq)]
pub struct Outlet {
    /// `"fennec"` / `"openbot"` — the value `/api/outlets/run` takes as `outlet`.
    pub id: &'static str,
    pub label: &'static str,
    pub present: bool,
    /// How it is installed, or the reason it is not, built from core's probe.
    pub detail: String,
    /// OpenBot coworkers as `agent=` accepts them (name, else id). The web lane
    /// takes a url instead of an agent, so its list stays empty.
    pub agents: Vec<String>,
}

/// The outlets the environment panel lists, in order: Fennec's own browser loop,
/// then the coworker workspace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outlets {
    pub fennec: Option<Outlet>,
    pub openbot: Option<Outlet>,
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Where the loop's browser transport came from, when that is not the obvious answer.
///
/// Core runs the loop on its own interpreter or on one the user pinned, and neither needs a
/// note. "the Fennec browser env" says a second venv this suite installed is doing the driving,
/// and anything else — "external" — says the transport was found in an interpreter Fennec does
/// not own, which is the row that would otherwise be lost by somebody else's uninstall.
fn transport_note(v: &Value) -> String {
    match v.get("transport_from").and_then(Value::as_str) {
        Some(s) if !s.is_empty() && s != "this interpreter" && s != "pinned" && s != "none" => {
            format!(" · browser transport {s}")
        }
        _ => String::new(),
    }
}

impl Outlets {
    /// Reads the shape `outlets.py:status()` returns. A key core does not send
    /// stays `None`, which renders as that outlet being unknown rather than
    /// pretending it is absent.
    pub fn from_value(v: &Value) -> Self {
        Self {
            fennec: Outlet::fennec(v.get("fennec")),
            openbot: Outlet::openbot(v.get("openbot")),
        }
    }

    pub fn list(&self) -> Vec<&Outlet> {
        self.fennec.iter().chain(self.openbot.iter()).collect()
    }

    /// True only when core sent that outlet and it is installed.
    pub fn ready(&self, id: &str) -> bool {
        self.get(id).is_some_and(|o| o.present)
    }

    pub fn get(&self, id: &str) -> Option<&Outlet> {
        match id {
            "fennec" => self.fennec.as_ref(),
            "openbot" => self.openbot.as_ref(),
            _ => None,
        }
    }
}

/// The environment rail: one `(installed, key, value)` row per outlet the agent
/// server reported, in the order core sent them. The three states are three
/// different claims — never probed, probed but empty, probed with data — so an
/// offline loop can never read as "not installed".
///
/// `known` is whether `/api/outlets` has answered once, which no field of
/// [`Outlets`] can say: a payload that named no outlet and no reply at all both
/// leave every lane `None`. Both native shells lay these rows out with their own
/// env-row widget; the PWA, which imports no Rust, mirrors them and a test holds
/// the two copies equal.
pub fn outlet_rows(known: bool, outlets: &[Outlet]) -> Vec<(bool, &'static str, String)> {
    if !known {
        return vec![(false, "Outlets", "unknown (agent loop offline)".into())];
    }
    if outlets.is_empty() {
        return vec![(false, "Outlets", "the agent server named none".into())];
    }
    outlets.iter().map(|o| (o.present, o.label, o.detail.clone())).collect()
}

impl Outlet {
    /// Fennec's own browser loop. Two independent halves make it runnable — an
    /// interpreter that can drive a page and an endpoint to ask what to drive (a
    /// `LIQUID_API_KEY` for the hosted d1.2, or a `LIQUID_ENDPOINT` for a free local
    /// model) — and `present` is their conjunction, so the row says which one is
    /// missing: one is an install, the other a `setx`. The key itself never crosses
    /// this boundary; only whether it is set does.
    fn fennec(v: Option<&Value>) -> Option<Self> {
        let v = v?;
        let present = flag(v, "present");
        let model = v.get("model").and_then(Value::as_str).unwrap_or("d1.2");
        let detail = if present {
            let from = v.get("interpreter").and_then(Value::as_str).unwrap_or("this interpreter");
            format!("installed — {from}{} · {model}", transport_note(v))
        } else if !flag(v, "transport") {
            "not installed — no python with the browser transport".into()
        } else {
            format!("not installed — {model} needs LIQUID_API_KEY or LIQUID_ENDPOINT")
        };
        Some(Self { id: "fennec", label: "Fennec", present, detail, agents: Vec::new() })
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
    /// What the two optional outlets are on this machine. `None` means the
    /// service could not be reached, which is not the same as both outlets
    /// being uninstalled.
    pub fn outlets(&self) -> Option<Outlets> {
        self.get("/api/outlets").ok().map(|v| Outlets::from_value(&v))
    }
    /// Ask the service to re-probe instead of using its cache (5 min for the
    /// browser lane, 20 s for OpenBot), which is what a "check again" affordance needs.
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
    /// `reasoning_effort` is the level the model dropdown shows; core owns the
    /// vocabulary and refuses anything outside it, so `auto` is always safe to send.
    pub fn start(&self, task: &str, workspace: &str, allow_write: bool, allow_shell: bool,
                 allow_web: bool, require_approval: bool,
                 reasoning_effort: &str) -> Result<String, String> {
        let body = json!({ "task": task, "workspace": workspace, "allow_write": allow_write,
                           "allow_shell": allow_shell, "allow_web": allow_web,
                           "require_approval": require_approval, "max_steps": 14,
                           "reasoning_effort": reasoning_effort });
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
    /// belongs to `engine.py`, not to this client — it is a long-running
    /// background thread server-side, so this returns as soon as it is queued.
    pub fn engine_start(&self) -> Result<Value, String> {
        self.post("/api/engine/start", &json!({}), Duration::from_millis(2000))
    }

    /// Hand one task to an installed outlet: `"fennec"` takes a browser end state
    /// at `url` and runs it through Fennec's own loop; `"openbot"` takes a `goal`
    /// for the named coworker. Core answers 409 for a web task with no browser
    /// agent installed at all, and 422 for any other id.
    /// This blocks for as long as the outlet runs — minutes, not milliseconds —
    /// so call it from a worker thread.
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

    /// The probe verbatim from `outlets.py:_fennec_probe`.
    fn fennec(present: bool, interpreter: Option<&str>, transport: bool, key: bool) -> Value {
        json!({"present": present, "interpreter": interpreter, "transport": transport,
               "transport_from": if transport { "this interpreter" } else { "none" },
               "decision_key": key, "model": "d1.2",
               "note": "Fennec's own loop: d1.2 names the operation and the node"})
    }

    #[test]
    fn the_fennec_lane_names_the_half_that_is_missing() {
        let o = Outlets::from_value(&json!({"fennec": fennec(true, Some("C:/py/python.exe"), true, true)}));
        assert!(o.ready("fennec"));
        assert_eq!(o.fennec.as_ref().unwrap().detail, "installed — C:/py/python.exe · d1.2");

        // A browser transport with nobody to ask is a setx away from working, and a
        // key with no transport is a pip install — telling them apart is the point.
        let no_key = Outlets::from_value(&json!({"fennec": fennec(false, None, true, false)}));
        assert_eq!(no_key.fennec.unwrap().detail, "not installed — d1.2 needs LIQUID_API_KEY or LIQUID_ENDPOINT");
        let no_browser = Outlets::from_value(&json!({"fennec": fennec(false, None, false, true)}));
        assert_eq!(no_browser.fennec.unwrap().detail, "not installed — no python with the browser transport");
    }

    /// The rail never collapses the lanes into one "outlets: no", and never lets a
    /// loop that has not answered claim any lane is missing.
    #[test]
    fn the_outlet_rail_keeps_not_probed_apart_from_probed_empty() {
        assert_eq!(outlet_rows(false, &[]), vec![(false, "Outlets", "unknown (agent loop offline)".to_string())]);
        assert_eq!(outlet_rows(true, &[]), vec![(false, "Outlets", "the agent server named none".to_string())]);
        let o = Outlets::from_value(&json!({
            "fennec": fennec(false, None, true, false),
            "openbot": {"present": true, "url": "http://127.0.0.1:3001",
                        "agents": [{"id": "r1", "name": "Review"}]},
        }));
        assert_eq!(outlet_rows(true, &o.list().into_iter().cloned().collect::<Vec<_>>()), vec![
            (false, "Fennec", "not installed — d1.2 needs LIQUID_API_KEY or LIQUID_ENDPOINT".to_string()),
            (true, "OpenBot", "1 coworker at http://127.0.0.1:3001".to_string()),
        ]);
    }

    /// Core runs the loop on whichever candidate can import the transport, so the row says
    /// which one that was. "the Fennec browser env" is a second venv this suite installed;
    /// anything else is an interpreter Fennec does not own, which is the difference between
    /// "my browser lane" and "one uninstall from gone".
    #[test]
    fn a_browser_transport_from_another_interpreter_is_said_out_loud() {
        let own_env = json!({"fennec": {"present": true, "transport": true, "decision_key": true,
            "interpreter": "C:/Users/me/.fennec/browser-tools/fennec/.venv/Scripts/python.exe",
            "transport_from": "the Fennec browser env", "model": "d1.2"}});
        assert_eq!(Outlets::from_value(&own_env).fennec.unwrap().detail,
            "installed — C:/Users/me/.fennec/browser-tools/fennec/.venv/Scripts/python.exe \
             · browser transport the Fennec browser env · d1.2");
        let borrowed = json!({"fennec": {"present": true, "transport": true, "decision_key": true,
            "interpreter": "C:/somewhere/else/.venv/Scripts/python.exe",
            "transport_from": "external", "model": "d1.2"}});
        assert_eq!(Outlets::from_value(&borrowed).fennec.unwrap().detail,
            "installed — C:/somewhere/else/.venv/Scripts/python.exe · browser transport external · d1.2");
        // A pin and core's own interpreter are both the expected arrangement: no note.
        for from in ["pinned", "this interpreter", "none", ""] {
            let v = json!({"fennec": {"present": true, "interpreter": "py", "transport_from": from,
                                      "model": "d1.2"}});
            assert_eq!(Outlets::from_value(&v).fennec.unwrap().detail, "installed — py · d1.2", "{from}");
        }
    }

    #[test]
    fn the_browser_lane_and_the_coworker_are_their_own_rows() {
        let o = Outlets::from_value(&json!({"fennec": fennec(true, Some("python"), true, true),
                                            "openbot": {"present": true, "url": "http://127.0.0.1:3001",
                                                        "agents": []}}));
        assert_eq!(o.list().iter().map(|x| x.id).collect::<Vec<_>>(), vec!["fennec", "openbot"]);
        assert!(o.ready("fennec"));
        assert!(o.ready("openbot"));
    }

    #[test]
    fn a_retired_outlet_key_in_the_payload_is_not_read_as_a_lane() {
        // `outlets.py` stopped probing Jev; a status file from before that says nothing
        // the rail should draw, and an unknown `outlet=` id is core's 422 to answer.
        let o = Outlets::from_value(&json!({"jev": {"present": true, "importable": true},
                                            "fennec": fennec(true, Some("python"), true, true)}));
        assert_eq!(o.list().len(), 1);
        assert!(!o.ready("jev"));
        assert!(o.get("jev").is_none());
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
        let o = Outlets::from_value(&json!({"fennec": {"available": true},
                                            "openbot": {"available": true}}));
        assert!(!o.ready("fennec"));
        assert!(!o.ready("openbot"));
        assert_eq!(o.fennec.as_ref().unwrap().detail, "not installed — no python with the browser transport");
    }

    #[test]
    fn an_outlet_the_service_never_mentioned_stays_unknown() {
        let o = Outlets::from_value(&json!({"fennec": fennec(true, Some("python"), true, true)}));
        assert!(o.openbot.is_none());
        assert!(!o.ready("openbot"));
        assert_eq!(o.list().len(), 1);
        assert!(!o.ready("nothing"));
    }
}
