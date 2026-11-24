//! Suite layout, read from the manifest the installer owns:
//! `fennec-environment-setup/repos.json`. The apps never hardcode a repo list —
//! `account`, `suite_root`, `engine_root`, `models_root` and the repos with
//! their branches all come from that file, so install/update assembles the
//! pieces and every run resolves them again. The CLI (`fennec-cli`) resolves the
//! same layout through these functions instead of carrying its own copy.
//!
//! Every rule here is a pure function of an injected `Environ` plus a home, and
//! the environment-reading functions are thin wrappers over them. That is the
//! contract the JetBrains sidecar consumes: it resolves the same suite from a
//! map its tests control instead of keeping a second copy of these rules.
use crate::git;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One checkout named by the manifest, resolved onto a real suite root.
#[derive(Clone)]
pub struct Repo {
    pub name: String,
    pub concept: String,
    pub branch: String,
    pub path: PathBuf,
    pub upstream: bool,
}

#[derive(Clone, Default)]
pub struct Suite {
    pub root: PathBuf,
    pub engine_root: PathBuf,
    pub models_root: PathBuf,
    pub manifest_found: bool,
    pub repos: Vec<Repo>,
}

/// An environment the lookup can be tested against. The shape is the sidecar's
/// own map, so a consumer passes what it has without converting.
pub type Environ = HashMap<String, String>;

/// What a non-absolute configured value resolves against. The Python it was
/// ported from disagreed per entry point - the launcher left the value alone,
/// `config.py` resolved it against the config file's directory - so the rule
/// belongs to the caller and the resolution lives here once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relative {
    /// Left as written: resolves against the process working directory.
    Keep,
    /// Resolved against the home the value was read from.
    Home,
}

/// The shim's truthiness: an unset or empty variable is absent.
pub fn env_value<'a>(environ: &'a Environ, key: &str) -> Option<&'a str> {
    environ.get(key).map(String::as_str).filter(|v| !v.is_empty())
}

/// SharedConfig's truthiness: a blank value is absent too.
pub fn env_nonblank<'a>(environ: &'a Environ, key: &str) -> Option<&'a str> {
    environ.get(key).map(String::as_str).filter(|v| !v.trim().is_empty())
}

/// The process environment in the injectable shape.
pub fn real_environ() -> Environ {
    std::env::vars().collect()
}

/// The home the suite and the shared config hang off.
pub fn home_in(environ: &Environ) -> PathBuf {
    #[cfg(windows)]
    {
        if let Some(profile) = env_value(environ, "USERPROFILE") {
            return PathBuf::from(profile);
        }
        if let (Some(drive), Some(path)) = (env_value(environ, "HOMEDRIVE"), env_value(environ, "HOMEPATH")) {
            let mut joined = PathBuf::from(drive);
            joined.push(path);
            return joined;
        }
        if let Some(home) = env_value(environ, "HOME") {
            return PathBuf::from(home);
        }
        PathBuf::from("C:\\")
    }
    #[cfg(not(windows))]
    {
        env_value(environ, "HOME")
            .or_else(|| env_value(environ, "USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"))
    }
}

pub fn home() -> PathBuf {
    home_in(&real_environ())
}

/// `~` and `~/x` expansion, the form the manifest uses, plus the caller's rule
/// for a value that is neither absolute nor home-prefixed.
pub fn expand_in(value: &str, home: &Path, relative: Relative) -> PathBuf {
    let expanded = if value == "~" {
        home.to_path_buf()
    } else if let Some(rest) = value.strip_prefix("~/").or_else(|| value.strip_prefix("~\\")) {
        home.join(rest)
    } else {
        PathBuf::from(value)
    };
    if relative == Relative::Home && !is_absolute_cross_platform(&expanded) {
        home.join(expanded)
    } else {
        expanded
    }
}

/// Check if a path is absolute on any platform, including Windows drive letters on Unix.
fn is_absolute_cross_platform(path: &Path) -> bool {
    if path.is_absolute() {
        return true;
    }
    // Check for Windows drive letter pattern (e.g., "C:/", "D:\")
    let s = path.to_string_lossy();
    s.len() >= 3
        && s.as_bytes()[0].is_ascii_alphabetic()
        && s.as_bytes()[1] == b':'
        && (s.as_bytes()[2] == b'/' || s.as_bytes()[2] == b'\\')
}

pub fn expand(p: &str) -> PathBuf {
    expand_in(p, &home(), Relative::Keep)
}

/// Where the shared config lives: `FENNEC_CONFIG`, else `~/.fennec/config.json`.
pub fn config_file_in(environ: &Environ, home: &Path, relative: Relative) -> PathBuf {
    match env_value(environ, "FENNEC_CONFIG") {
        Some(value) => expand_in(value, home, relative),
        None => home.join(".fennec").join("config.json"),
    }
}

/// What reading one configuration file produced, without the words each
/// consumer puts on it: the plugin's two contracts disagree about a file that
/// is not a JSON object.
#[derive(Debug)]
pub enum ConfigRead {
    Object(Map<String, Value>),
    Absent,
    NotAnObject,
    Unreadable(String),
}

pub fn read_config(location: &Path) -> ConfigRead {
    if !location.is_file() {
        return ConfigRead::Absent;
    }
    let text = match std::fs::read_to_string(location) {
        Ok(text) => text,
        Err(e) => return ConfigRead::Unreadable(e.to_string()),
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => ConfigRead::Object(map),
        Ok(_) => ConfigRead::NotAnObject,
        Err(e) => ConfigRead::Unreadable(e.to_string()),
    }
}

/// The shared config as a map, ignoring every failure the way suite discovery does.
fn config_map_in(environ: &Environ, home: &Path, relative: Relative) -> Map<String, Value> {
    match read_config(&config_file_in(environ, home, relative)) {
        ConfigRead::Object(map) => map,
        _ => Map::new(),
    }
}

/// The suite root the shared config alone defines, in the order the launcher
/// uses: `FENNEC_SUITE`, then the config's `suite_root`, then `~/Fennec-Suite`.
pub fn suite_root_in(
    environ: &Environ,
    config: &Map<String, Value>,
    home: &Path,
    relative: Relative,
) -> PathBuf {
    let configured = env_value(environ, "FENNEC_SUITE")
        .map(str::to_string)
        .or_else(|| config.get("suite_root").and_then(Value::as_str).map(str::to_string));
    match configured {
        Some(value) => expand_in(&value, home, relative),
        None => home.join("Fennec-Suite"),
    }
}

/// Candidate suite roots, in the priority the launcher uses. The
/// manifest gets its own step: the installer can point it at a different root
/// than the config does.
pub fn roots_in(environ: &Environ, home: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(s) = env_value(environ, "FENNEC_SUITE") {
        v.push(PathBuf::from(s));
    }
    if let Some(r) = config_map_in(environ, home, Relative::Keep)
        .get("suite_root")
        .and_then(Value::as_str)
    {
        v.push(expand_in(r, home, Relative::Keep));
    }
    if let Some(m) = manifest_at_any(&v) {
        if let Some(r) = m["suite_root"].as_str() {
            let p = expand_in(r, home, Relative::Keep);
            if !v.contains(&p) {
                v.push(p);
            }
        }
    }
    let def = home.join("Fennec-Suite");
    if !v.contains(&def) {
        v.push(def);
    }
    v
}

/// Candidate suite roots read from the process environment.
pub fn roots() -> Vec<PathBuf> {
    roots_in(&real_environ(), &home())
}

pub fn manifest_at_any(roots: &[PathBuf]) -> Option<Value> {
    roots.iter().find_map(|root| manifest_in(root))
}

/// The manifest one suite root owns, when it has one.
pub fn manifest_in(root: &Path) -> Option<Value> {
    let p = root.join("fennec-environment-setup").join("repos.json");
    let text = std::fs::read_to_string(&p).ok()?;
    serde_json::from_str::<Value>(&text).ok()
}

/// The checkouts a root offers: the manifest's `repos[]` that exist on disk, or
/// any `.git` directory under the root when the manifest lists none, so the
/// selector is never empty by accident.
pub fn repos_in(manifest: Option<&Value>, root: &Path) -> Vec<Repo> {
    let mut repos = Vec::new();
    if let Some(list) = manifest.and_then(|m| m["repos"].as_array()) {
        for r in list.iter().filter_map(|r| r.as_object()) {
            let Some(full) = r.get("repo").and_then(|v| v.as_str()) else { continue };
            let name = full.split('/').next_back().unwrap_or(full).to_string();
            let path = root.join(&name);
            if !path.is_dir() {
                continue;
            }
            repos.push(Repo {
                name,
                concept: r.get("concept").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                branch: r.get("branch").and_then(|v| v.as_str()).unwrap_or("main").to_string(),
                path,
                upstream: r.get("upstream").and_then(|v| v.as_bool()).unwrap_or(false),
            });
        }
    }
    if repos.is_empty() {
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() && p.join(".git").exists() {
                    repos.push(Repo {
                        name: e.file_name().to_string_lossy().into_owned(),
                        concept: String::new(),
                        branch: "main".into(),
                        path: p,
                        upstream: false,
                    });
                }
            }
        }
    }
    repos
}

/// The suite the process environment points at.
pub fn discover() -> Suite {
    discover_in(&real_environ(), &home())
}

/// `discover()` with the environment and home it resolves against injected.
pub fn discover_in(environ: &Environ, home: &Path) -> Suite {
    let mut out = Suite::default();
    let roots = roots_in(environ, home);
    let manifest = manifest_at_any(&roots);
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        out.root = root.clone();
        if let Some(m) = &manifest {
            out.manifest_found = true;
            out.engine_root = m["engine_root"].as_str()
                .map(|v| expand_in(v, home, Relative::Keep))
                .unwrap_or_else(|| home.join("Fennec"));
            out.models_root = m["models_root"].as_str()
                .map(|v| expand_in(v, home, Relative::Keep))
                .unwrap_or_else(|| home.join("fennec-models"));
        }
        out.repos = repos_in(manifest.as_ref(), &root);
        if !out.repos.is_empty() {
            break;
        }
    }
    out.repos.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The outlet gate report the Python core owns, as a child-process snippet.
/// The launcher hands this to a python child; the Rust agent captures it, so
/// both UI surfaces read one implementation without a Python server running.
pub const OUTLETS_CHECK: &str = "import importlib, json\n\
loaded = importlib.import_module('config').load_config()\n\
value = importlib.import_module('outlets').Outlets(loaded).status()\n\
text = json.dumps(value, indent=2, ensure_ascii=False)\n\
try: print(text)\n\
except UnicodeEncodeError: print(json.dumps(value, indent=2))\n";

impl Suite {
    /// Paths offered in the workspace selector: checkouts that are real git
    /// repositories (the manifest lists concept repos; only some are cloned).
    pub fn workspaces(&self) -> Vec<PathBuf> {
        self.repos.iter().map(|r| r.path.clone()).filter(|p| p.join(".git").exists()).collect()
    }

    /// Where the engine binary lives, per the manifest's `engine_root`. Fennec is
    /// the only engine, so only its native build layouts are probed.
    pub fn engine_binary(&self) -> PathBuf {
        let mut roots: Vec<PathBuf> = Vec::new();
        if !self.engine_root.as_os_str().is_empty() {
            roots.push(self.engine_root.clone());
        }
        roots.push(home().join("Fennec"));
        let names: &[&str] = if cfg!(windows) { &["fennec.exe", "fennec"] } else { &["fennec"] };
        for root in roots {
            for rel in ["target/release", "target/debug", "."] {
                for n in names {
                    let p = root.join(rel).join(n);
                    if p.is_file() {
                        return p;
                    }
                }
            }
        }
        PathBuf::from(if cfg!(windows) { "fennec.exe" } else { "fennec" })
    }

    /// The checkout that serves legacy compatibility routes, when present.
    pub fn core_dir(&self) -> Option<PathBuf> {
        self.repos.iter().find(|r| r.name == "fennec-core").map(|r| r.path.clone())
    }

    /// The checkout that owns the canonical Rust agent service, when present.
    pub fn agent_dir(&self) -> Option<PathBuf> {
        self.repos.iter().find(|r| r.name == "fennec-coding-agent").map(|r| r.path.clone())
    }

    pub fn python() -> Option<String> {
        for name in ["python", "python3"] {
            if git::which(name).is_some() {
                return Some(name.to_string());
            }
        }
        None
    }

    /// The core's outlet gates as JSON, or None when no core checkout or
    /// interpreter exists. Probes are the core's own; nothing is reimplemented.
    pub fn outlets_status(&self) -> Option<Value> {
        let core = self.core_dir()?;
        let py = Self::python()?;
        let out = std::process::Command::new(py)
            .arg("-c")
            .arg(OUTLETS_CHECK)
            .current_dir(core)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        serde_json::from_slice(&out.stdout).ok()
    }

    /// Launch the canonical Rust agent service from its checkout, then an
    /// installed Rust command. Python remains an explicit compatibility fallback
    /// only when no Rust service can be started.
    /// Best effort: false means "still unavailable", not a lie.
    pub fn serve(&self, port: u16) -> bool {
        let py = Suite::python();
        for attempt in self.serve_commands(py.as_deref(), port) {
            let mut cmd = std::process::Command::new(&attempt[0]);
            cmd.args(&attempt[1..]);
            cmd.env("FENNEC_AGENT_ADDR", format!("127.0.0.1:{port}"));
            if let Some(core) = self.core_dir() {
                cmd.current_dir(core);
            }
            let spawned = crate::git::quiet(&mut cmd)
                .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
                .spawn();
            if spawned.is_err() {
                continue;
            }
            for _ in 0..48 {
                std::thread::sleep(Duration::from_millis(250));
                let health = crate::coding_agent::CodingAgentClient::with_base(
                    format!("http://127.0.0.1:{port}"),
                ).health();
                if health.as_ref().is_some_and(|v| v["backend"] == "rust-axum") {
                    return true;
                }
            }
            return false;
        }
        false
    }

    fn serve_commands(&self, py: Option<&str>, port: u16) -> Vec<Vec<String>> {
        let mut out = Vec::new();
        if let Some(agent) = self.agent_dir() {
            let binary = if cfg!(windows) { "fennec-agent.exe" } else { "fennec-agent" };
            for rel in ["target/release", "target/debug"] {
                let path = agent.join(rel).join(binary);
                if path.is_file() {
                    out.push(vec![path.to_string_lossy().into_owned(), "serve".into(),
                                  "--host".into(), "127.0.0.1".into(), "--port".into(), port.to_string()]);
                }
            }
        }
        if git::which("fennec-agent").is_some() {
            out.push(vec!["fennec-agent".into(), "serve".into(), "--host".into(), "127.0.0.1".into(), "--port".into(), port.to_string()]);
        }
        if let Some(p) = py {
            if self.core_dir().map(|d| d.join("server.py").is_file()).unwrap_or(false) {
                out.push(vec![p.to_string(), "-m".into(), "server".into(), "serve".into(),
                              "--host".into(), "127.0.0.1".into(), "--port".into(), port.to_string()]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlets_status_is_none_without_a_core_checkout() {
        let home = std::env::temp_dir().join("fennec-suite-no-core-checkout");
        let suite = discover_in(&Environ::new(), &home);
        assert!(suite.core_dir().is_none());
        assert!(suite.outlets_status().is_none());
    }

    fn environ(pairs: &[(&str, &str)]) -> Environ {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    struct Temp {
        root: PathBuf,
        home: PathBuf,
    }

    impl Temp {
        fn new(tag: &str) -> Temp {
            let root = std::env::temp_dir().join(format!("fennec-client-suite-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let home = root.join("home");
            std::fs::create_dir_all(&home).unwrap();
            Temp { root, home }
        }

        fn write_config(&self, body: &str) -> PathBuf {
            let path = self.home.join(".fennec").join("config.json");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, body).unwrap();
            path
        }

        /// A suite root carrying a manifest that points somewhere else.
        fn manifest(&self, root: &Path, body: &str) -> PathBuf {
            let dir = root.join("fennec-environment-setup");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("repos.json"), body).unwrap();
            root.to_path_buf()
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn expand_handles_the_manifest_form() {
        assert_eq!(expand("~"), home());
        assert_eq!(expand("~/Fennec-Suite"), home().join("Fennec-Suite"));
        assert_eq!(expand("D:/elsewhere"), PathBuf::from("D:/elsewhere"));
    }

    #[test]
    fn discovery_never_panics_without_a_suite() {
        let s = Suite { root: PathBuf::from(if cfg!(windows) { "Z:/nope" } else { "/nope" }), ..Default::default() };
        assert!(s.workspaces().is_empty());
        // On Windows the binary has .exe extension; on Unix it doesn't
        if cfg!(windows) {
            assert!(s.engine_binary().extension().is_some());
        } else {
            assert!(s.engine_binary().extension().is_none());
        }
        let _ = discover();
    }

    #[test]
    fn serve_commands_always_pass_the_serve_subcommand() {
        let s = discover();
        for c in s.serve_commands(Some("python"), 8787) {
            assert!(c.iter().any(|a| a == "serve"), "server.py requires the serve subcommand: {c:?}");
            assert!(c.last().map(|p| p == "8787").unwrap_or(false));
        }
    }

    #[test]
    fn an_empty_variable_is_absent_and_whitespace_is_not() {
        let e = environ(&[("FENNEC_SUITE", ""), ("FENNEC_CONFIG", "   ")]);
        assert_eq!(env_value(&e, "FENNEC_SUITE"), None);
        assert_eq!(env_value(&e, "FENNEC_CONFIG"), Some("   "));
        assert_eq!(env_nonblank(&e, "FENNEC_CONFIG"), None);
        assert_eq!(env_value(&e, "FENNEC_CORE"), None);
    }

    #[test]
    fn home_in_treats_an_empty_value_as_absent() {
        let e = environ(&[("USERPROFILE", ""), ("HOME", "/the/home")]);
        assert_eq!(home_in(&e), PathBuf::from("/the/home"));
    }

    #[test]
    fn home_in_falls_back_to_the_platform_root() {
        let expected = if cfg!(windows) { "C:\\" } else { "/" };
        assert_eq!(home_in(&Environ::new()), PathBuf::from(expected));
    }

    #[test]
    fn home_in_prefers_the_platform_own_variable() {
        let e = environ(&[("USERPROFILE", "C:\\users\\f"), ("HOME", "/users/f")]);
        let expected = if cfg!(windows) { "C:\\users\\f" } else { "/users/f" };
        assert_eq!(home_in(&e), PathBuf::from(expected));
    }

    #[test]
    #[cfg(windows)]
    fn homedrive_and_homepath_come_before_home_on_windows() {
        let e = environ(&[("HOMEDRIVE", "D:"), ("HOMEPATH", "\\users\\f"), ("HOME", "/users/f")]);
        let mut expected = PathBuf::from("D:");
        expected.push("\\users\\f");
        assert_eq!(home_in(&e), expected);
    }

    #[test]
    fn the_two_relative_rules_differ_only_for_relative_values() {
        let home = PathBuf::from(if cfg!(windows) { "C:/fennec" } else { "/home/fennec" });
        assert_eq!(expand_in("~", &home, Relative::Keep), home);
        assert_eq!(expand_in("~/x", &home, Relative::Home), home.join("x"));
        assert_eq!(expand_in("D:/elsewhere", &home, Relative::Home), PathBuf::from("D:/elsewhere"));
        assert_eq!(expand_in("rel/path", &home, Relative::Keep), PathBuf::from("rel/path"));
        assert_eq!(expand_in("rel/path", &home, Relative::Home), home.join("rel/path"));
    }

    #[test]
    fn config_file_defaults_to_the_home_and_expands_tilde() {
        let home = PathBuf::from(if cfg!(windows) { "C:/fennec" } else { "/home/fennec" });
        assert_eq!(config_file_in(&Environ::new(), &home, Relative::Keep), home.join(".fennec").join("config.json"));
        let e = environ(&[("FENNEC_CONFIG", "")]);
        assert_eq!(config_file_in(&e, &home, Relative::Home), home.join(".fennec").join("config.json"));
        let e = environ(&[("FENNEC_CONFIG", "~/other.json")]);
        assert_eq!(config_file_in(&e, &home, Relative::Home), home.join("other.json"));
    }

    #[test]
    fn read_config_tells_absent_from_broken_from_not_an_object() {
        let t = Temp::new("read-config");
        assert!(matches!(read_config(&t.home.join("nope.json")), ConfigRead::Absent));
        t.write_config("[1, 2]");
        assert!(matches!(read_config(&t.home.join(".fennec").join("config.json")), ConfigRead::NotAnObject));
        t.write_config("{oops");
        match read_config(&t.home.join(".fennec").join("config.json")) {
            ConfigRead::Unreadable(_) => {}
            other => panic!("broken json must not read as {other:?}"),
        }
        t.write_config("{\"suite_root\": \"~/S\"}");
        match read_config(&t.home.join(".fennec").join("config.json")) {
            ConfigRead::Object(map) => assert_eq!(map["suite_root"].as_str(), Some("~/S")),
            other => panic!("object json must read as a map, got {other:?}"),
        }
    }

    #[test]
    fn suite_root_follows_environment_then_config_then_home() {
        let t = Temp::new("suite-root");
        assert_eq!(suite_root_in(&Environ::new(), &Map::new(), &t.home, Relative::Home), t.home.join("Fennec-Suite"));
        let mut config = Map::new();
        config.insert("suite_root".into(), Value::String("My/Suite".into()));
        assert_eq!(suite_root_in(&Environ::new(), &config, &t.home, Relative::Home), t.home.join("My").join("Suite"));
        assert_eq!(suite_root_in(&Environ::new(), &config, &t.home, Relative::Keep), PathBuf::from("My/Suite"));
        let absolute = if cfg!(windows) { "C:\\somewhere\\else" } else { "/somewhere/else" };
        let e = environ(&[("FENNEC_SUITE", absolute)]);
        assert_eq!(suite_root_in(&e, &config, &t.home, Relative::Home), PathBuf::from(absolute));
    }

    #[test]
    fn roots_chain_environment_config_manifest_then_default() {
        let t = Temp::new("roots");
        let configured = t.root.join("from-config");
        let other = t.root.join("from-manifest");
        let suite = t.manifest(&t.root.join("from-env"), &format!("{{\"suite_root\": {:?}}}", other.display().to_string()));
        t.write_config(&format!("{{\"suite_root\": {:?}}}", configured.display().to_string()));
        let e = environ(&[("FENNEC_SUITE", &suite.display().to_string())]);
        assert_eq!(
            roots_in(&e, &t.home),
            vec![suite, configured, other, t.home.join("Fennec-Suite")]
        );
    }

    #[test]
    fn roots_never_list_the_same_candidate_twice() {
        let t = Temp::new("roots-dedupe");
        let other = t.root.join("from-manifest");
        let suite = t.manifest(&t.root.join("suite"), &format!("{{\"suite_root\": {:?}}}", other.display().to_string()));
        t.write_config(&format!("{{\"suite_root\": {:?}}}", suite.display().to_string()));
        let roots = roots_in(&Environ::new(), &t.home);
        assert_eq!(roots, vec![suite.clone(), other, t.home.join("Fennec-Suite")]);
        assert_eq!(roots.iter().filter(|r| **r == suite).count(), 1);
    }

    #[test]
    fn an_empty_suite_root_environment_falls_through_to_the_config() {
        let t = Temp::new("roots-empty-env");
        let configured = t.root.join("from-config");
        t.write_config(&format!("{{\"suite_root\": {:?}}}", configured.display().to_string()));
        assert_eq!(
            roots_in(&environ(&[("FENNEC_SUITE", "")]), &t.home),
            vec![configured, t.home.join("Fennec-Suite")]
        );
    }

    #[test]
    fn repos_prefers_the_manifest_and_falls_back_to_git_checkouts() {
        let t = Temp::new("repos");
        let root = t.root.join("suite");
        std::fs::create_dir_all(root.join("fennec-core")).unwrap();
        std::fs::create_dir_all(root.join("bare").join(".git")).unwrap();
        let manifest: Value = serde_json::json!({
            "repos": [
                {"repo": "Liso7150/fennec-core", "concept": "core", "branch": "dev", "upstream": true},
                {"repo": "Liso7150/not-cloned"}
            ]
        });
        let repos = repos_in(Some(&manifest), &root);
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0].name, "fennec-core");
        assert_eq!(repos[0].branch, "dev");
        assert!(repos[0].upstream);
        let fallback = repos_in(None, &root);
        assert_eq!(fallback.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["bare"]);
        assert!(repos_in(Some(&manifest), &t.root.join("nowhere")).is_empty());
    }

    #[test]
    fn manifest_in_reads_only_the_root_it_is_given() {
        let t = Temp::new("manifest");
        let root = t.manifest(&t.root.join("suite"), "{\"suite_root\": \"x\"}");
        assert!(manifest_in(&root).is_some());
        assert!(manifest_in(&t.root.join("other")).is_none());
        assert!(manifest_at_any(&[t.root.join("other"), root.clone()]).is_some());
    }

    #[test]
    fn discover_in_resolves_the_manifest_it_found() {
        let t = Temp::new("discover");
        let engine = t.root.join("engine");
        let suite = t.manifest(
            &t.root.join("suite"),
            &format!(
                "{{\"engine_root\": {:?}, \"models_root\": \"~/models\", \"repos\": [{{\"repo\": \"Liso7150/fennec-core\"}}]}}",
                engine.display()
            ),
        );
        std::fs::create_dir_all(suite.join("fennec-core")).unwrap();
        std::fs::create_dir_all(suite.join("fennec-core").join(".git")).unwrap();
        let e = environ(&[("FENNEC_SUITE", &suite.display().to_string())]);
        let found = discover_in(&e, &t.home);
        assert_eq!(found.root, suite);
        assert!(found.manifest_found);
        assert_eq!(found.engine_root, engine);
        assert_eq!(found.models_root, t.home.join("models"));
        assert_eq!(found.workspaces(), vec![suite.join("fennec-core")]);
    }

    #[test]
    fn discover_without_a_manifest_leaves_the_engine_root_unset() {
        let t = Temp::new("discover-bare");
        let suite = t.root.join("suite");
        std::fs::create_dir_all(suite.join("only-repo").join(".git")).unwrap();
        let found = discover_in(&environ(&[("FENNEC_SUITE", &suite.display().to_string())]), &t.home);
        assert!(!found.manifest_found);
        assert!(found.engine_root.as_os_str().is_empty());
        assert_eq!(found.repos.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["only-repo"]);
    }
}
