//! Read-only port of `fennec-desktop/electron/config.js`: the suite's shared
//! configuration contract. Same defaults, env aliases, validation and engine
//! environment pinning, so the Rust shell and the old Electron main agree.
use std::path::{Path, PathBuf};

pub const INSTALLER_URL: &str = "https://github.com/Liso7150/fennec-installer";

pub fn is_loopback(host: &str) -> bool {
    if host == "localhost" || host == "[::1]" || host == "::1" {
        return true;
    }
    let octets: Vec<&str> = host.split('.').collect();
    if octets.len() != 4 || octets[0] != "127" {
        return false;
    }
    octets[1..].iter().all(|o| {
        !o.is_empty() && o.len() <= 3 && o.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Expand a leading `~` and resolve against `base`, like `path.resolve`.
fn resolve(value: &str, base: &Path, home: &Path) -> PathBuf {
    let expanded = if value == "~" || value.starts_with("~/") || value.starts_with("~\\") {
        home.join(value.trim_start_matches('~').trim_start_matches(['/', '\\']))
    } else {
        PathBuf::from(value)
    };
    if expanded.is_absolute() {
        normalize(&expanded)
    } else {
        normalize(&base.join(expanded))
    }
}

fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Config {
    pub engine_root: PathBuf,
    pub suite_root: PathBuf,
    pub models_root: PathBuf,
    pub cache_root: PathBuf,
    pub model: String,
    pub engine_url: String,
    pub backend: String,
    pub ctx: u64,
    pub policy: String,
    pub python: Option<PathBuf>,
    pub workspace: PathBuf,
    pub web_root: PathBuf,
    pub api_key: String,
    pub config_path: PathBuf,
    pub port: u16,
    pub web_url: String,
}

pub struct Origin {
    pub origin: String,
    pub hostname: String,
    pub port: u16,
}

/// Minimal `new URL()` for the engine origin: scheme, authority, optional port.
/// Rejects credentials, a non-root path, query, fragment and port 0, exactly as
/// the JS contract does.
pub fn parse_origin(url: &str) -> Result<Origin, String> {
    let (scheme, rest) = if let Some(r) = url.strip_prefix("http://") {
        ("http", r)
    } else if let Some(r) = url.strip_prefix("https://") {
        ("https", r)
    } else {
        return Err("engine_url must be an HTTP(S) origin".into());
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    let remainder = &rest[end..];
    if remainder.starts_with('?') || remainder.starts_with('#') || (remainder.starts_with('/') && remainder != "/") {
        return Err("engine_url must be an HTTP(S) origin without credentials, path or query".into());
    }
    if authority.contains('@') || authority.is_empty() {
        return Err("engine_url must be an HTTP(S) origin without credentials, path or query".into());
    }
    let (host, port_str) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') => (h, Some(p)),
        _ => (authority, None),
    };
    let port: u16 = match port_str {
        Some(p) => p.parse().map_err(|_| "engine_url must be an HTTP(S) origin".to_string())?,
        None => if scheme == "https" { 443 } else { 80 },
    };
    if port == 0 {
        return Err("engine_url must be an HTTP(S) origin without credentials, path or query".into());
    }
    let hostname = host.trim_start_matches('[').trim_end_matches(']').to_lowercase();
    Ok(Origin { origin: format!("{scheme}://{authority}"), hostname, port })
}

const SECRET_KEYS: [&str; 6] = ["api_key", "app_token", "token", "fennec_api_key", "fennec_app_token", "coli_api_key"];

/// `loadConfig` with the JS injectable dependencies, so tests can drive it
/// without touching the real filesystem or environment.
pub fn load_config(
    env: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    win: bool,
    exists: &dyn Fn(&Path) -> bool,
    read_file: &dyn Fn(&Path) -> Option<String>,
) -> Result<Config, String> {
    let sep = if win { '\\' } else { '/' };
    let join = |a: &Path, b: &str| -> PathBuf { a.join(b.replace('/', &sep.to_string())) };
    let default_config = join(home, ".fennec/config.json");
    let location = resolve(&env("FENNEC_CONFIG").unwrap_or_else(|| default_config.to_string_lossy().into_owned()), home, home);
    let mut data = serde_json::Map::new();
    if exists(&location) {
        let raw = read_file(&location).ok_or_else(|| format!("Cannot read configuration JSON: {}", location.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(|_| format!("Cannot read configuration JSON: {}", location.display()))?;
        match parsed {
            serde_json::Value::Object(m) => data = m,
            _ => return Err("Fennec configuration must be a JSON object".into()),
        }
    } else if env("FENNEC_CONFIG").is_some() {
        return Err(format!("Configured file does not exist: {}", location.display()));
    }
    if data.keys().any(|k| SECRET_KEYS.contains(&k.to_lowercase().as_str())) {
        return Err("Keep secrets in FENNEC_API_KEY, not configuration JSON".into());
    }
    let get = |key: &str, default: &str| -> String {
        data.get(key).and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| default.to_string())
    };
    let mut engine_root = get("engine_root", &join(home, "Colibri").to_string_lossy());
    let suite_root = get("suite_root", &join(home, "Fennec-Suite").to_string_lossy());
    let mut models_root = get("models_root", &join(home, "fennec-models").to_string_lossy());
    let mut cache_root = get("cache_root", &join(home, ".cache/fennec").to_string_lossy());
    let mut model = get("model", "");
    let mut engine_url = get("engine_url", "http://127.0.0.1:8184");
    let backend = get("backend", "auto");
    let policy = get("policy", "quality");
    let mut python = data.get("python").and_then(|v| v.as_str()).map(|s| s.to_string());
    let mut workspace = data.get("workspace").and_then(|v| v.as_str()).map(|s| s.to_string());
    let web_root = data.get("web_root").and_then(|v| v.as_str()).map(|s| s.to_string());
    let ctx: u64 = match data.get("ctx") {
        Some(serde_json::Value::Number(n)) => n.as_u64().ok_or("ctx must be an integer from 0 to 1048576")?,
        None => 0,
        _ => return Err("ctx must be an integer from 0 to 1048576".into()),
    };
    for (variable, key) in [
        ("COLI_PYTHON", "python"), ("COLI_MODEL", "model"), ("COLI_MODELS", "models_root"),
        ("FENNEC_HOME", "engine_root"), ("FENNEC_MODELS", "models_root"), ("FENNEC_CACHE", "cache_root"),
        ("FENNEC_PYTHON", "python"), ("FENNEC_MODEL", "model"), ("FENNEC_ENGINE_URL", "engine_url"),
        ("FENNEC_WORKSPACE", "workspace"),
    ] {
        if let Some(v) = env(variable) {
            match key {
                "python" => python = Some(v),
                "model" => model = v,
                "models_root" => models_root = v,
                "engine_root" => engine_root = v,
                "cache_root" => cache_root = v,
                "engine_url" => engine_url = v,
                "workspace" => workspace = Some(v),
                _ => {}
            }
        }
    }
    for (key, value) in [("engine_root", &engine_root), ("suite_root", &suite_root), ("models_root", &models_root),
        ("cache_root", &cache_root), ("engine_url", &engine_url), ("backend", &backend), ("policy", &policy)] {
        if value.trim().is_empty() {
            return Err(format!("Invalid configuration: {key}"));
        }
    }
    let base = location.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| home.to_path_buf());
    let engine_root = resolve(&engine_root, &base, home);
    let suite_root = resolve(&suite_root, &base, home);
    let models_root = resolve(&models_root, &base, home);
    let cache_root = resolve(&cache_root, &base, home);
    let workspace = resolve(workspace.as_deref().unwrap_or(&suite_root.to_string_lossy()), &base, home);
    let web_root = resolve(web_root.as_deref().unwrap_or(&join(&suite_root, "fennec-coding-agent/web").to_string_lossy()), &base, home);
    let model = if model.trim().is_empty() { String::new() } else { resolve(&model, &models_root, home).to_string_lossy().into_owned() };
    if !["auto", "cuda", "vulkan", "metal", "cpu"].contains(&backend.as_str()) {
        return Err("Invalid backend selection".into());
    }
    if !["quality", "balanced", "experimental-fast"].contains(&policy.as_str()) {
        return Err("Invalid resource policy".into());
    }
    if ctx > 1_048_576 {
        return Err("ctx must be an integer from 0 to 1048576".into());
    }
    let url = parse_origin(&engine_url)?;
    let mut api_key = env("FENNEC_API_KEY").or_else(|| env("COLI_API_KEY")).unwrap_or_default();
    if !is_loopback(&url.hostname) && api_key.is_empty() {
        return Err("A non-loopback engine requires FENNEC_API_KEY".into());
    }
    if api_key.is_empty() {
        api_key = "local".into();
    }
    let python = resolve_python(python.as_deref(), env, home, win, exists, &base);
    Ok(Config {
        engine_root, suite_root, models_root, cache_root, model,
        engine_url: url.origin, backend, ctx, policy, python, workspace, web_root,
        api_key, config_path: location, port: url.port, web_url: "http://127.0.0.1:8787".into(),
    })
}

fn resolve_python(python: Option<&str>, env: &dyn Fn(&str) -> Option<String>, home: &Path, win: bool,
                  exists: &dyn Fn(&Path) -> bool, base: &Path) -> Option<PathBuf> {
    let path_var = env("PATH").or_else(|| env("Path")).unwrap_or_default();
    let dirs: Vec<&str> = path_var.split(if win { ';' } else { ':' }).filter(|d| !d.is_empty()).collect();
    if let Some(p) = python {
        if !p.contains(['/', '\\']) {
            for dir in &dirs {
                let cand = Path::new(dir).join(p);
                if exists(&cand) { return Some(cand); }
                if win {
                    let exe = Path::new(dir).join(format!("{p}.exe"));
                    if exists(&exe) { return Some(exe); }
                }
            }
        }
        return Some(resolve(p, base, home));
    }
    let names: &[&str] = if win { &["python.exe", "python3.exe"] } else { &["python3", "python"] };
    for dir in &dirs {
        for name in names {
            let cand = Path::new(dir).join(name);
            if exists(&cand) { return Some(cand); }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub runtime: Option<PathBuf>,
    pub cli: PathBuf,
    pub server: Option<PathBuf>,
    pub kind: &'static str,
}

/// `runtimePaths`: a native Fennec binary wins; a Colibri checkout still works.
pub fn runtime_paths(config: &Config, win: bool, exists: &dyn Fn(&Path) -> bool) -> Runtime {
    let exe = if win { "fennec.exe" } else { "fennec" };
    for rel in [["target", "release"], ["target", "debug"], ["", ""]] {
        let mut cli = config.engine_root.clone();
        for part in rel { if !part.is_empty() { cli = cli.join(part); } }
        cli = cli.join(exe);
        if exists(&cli) {
            return Runtime { runtime: cli.parent().map(|p| p.to_path_buf()), cli, server: None, kind: "fennec" };
        }
    }
    let runtime = config.engine_root.join("c");
    let cli = runtime.join("coli");
    let server = runtime.join("openai_server.py");
    let present = exists(&cli) && exists(&server);
    Runtime { runtime: if present { Some(runtime.clone()) } else { None }, cli, server: Some(server), kind: "colibri" }
}

const VULKAN_LINUX: [&str; 3] = [
    "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib64/libvulkan.so.1", "/usr/lib/libvulkan.so.1",
];

pub fn has_vulkan_loader(platform: &str, env: &dyn Fn(&str) -> Option<String>, exists: &dyn Fn(&Path) -> bool) -> bool {
    if platform == "darwin" { return false; }
    if platform == "win32" {
        let root = env("SystemRoot").or_else(|| env("SYSTEMROOT")).unwrap_or_else(|| "C:\\Windows".into());
        return exists(&Path::new(&root).join("System32").join("vulkan-1.dll"));
    }
    VULKAN_LINUX.iter().any(|c| exists(Path::new(c)))
}

/// `engineEnvironment`: strip secrets/flags, pass the key via COLI_API_KEY (never
/// argv), and pin the backend the same way the Electron main did.
pub fn engine_environment(config: &Config, env: &dyn Fn(&str) -> Option<String>, platform: &str,
                          exists: &dyn Fn(&Path) -> bool) -> Vec<(String, String)> {
    // Overrides only; the caller merges these over the inherited process env,
    // matching the JS `{...env, ...}` spread without enumerating the closure.
    let mut next: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    next.insert("PYTHONIOENCODING".into(), "utf-8".into());
    next.insert("PYTHONUTF8".into(), "1".into());
    if !config.api_key.is_empty() {
        next.insert("COLI_API_KEY".into(), config.api_key.clone());
    }
    let kind = runtime_paths(config, platform == "win32", exists).kind;
    if config.backend == "auto" {
        if kind != "fennec" && has_vulkan_loader(platform, env, exists) {
            next.insert("COLI_VULKAN".into(), "1".into());
            if env("COLI_VK_TIER_HOST_GB").is_none() {
                next.insert("COLI_VK_TIER_HOST_GB".into(), "4".into());
            }
        }
    } else {
        next.insert("COLI_VULKAN".into(), "0".into());
        next.insert("COLI_METAL".into(), "0".into());
        next.insert("COLI_CUDA".into(), "0".into());
        if config.backend != "cpu" {
            next.insert(format!("COLI_{}", config.backend.to_uppercase()), "1".into());
        }
    }
    next.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn no_env(_: &str) -> Option<String> { None }
    fn no_exists(_: &Path) -> bool { false }
    fn no_read(_: &Path) -> Option<String> { None }
    fn home() -> PathBuf { PathBuf::from("/home/u") }

    #[test]
    fn loopback_forms() {
        for h in ["localhost", "[::1]", "::1", "127.0.0.1", "127.1.2.3"] { assert!(is_loopback(h), "{h}"); }
        for h in ["10.0.0.1", "127.0.0.1.evil", "example.com", "127.0.0"] { assert!(!is_loopback(h), "{h}"); }
    }

    #[test]
    fn defaults_match_js() {
        let c = load_config(&no_env, &home(), false, &no_exists, &no_read).unwrap();
        assert_eq!(c.engine_url, "http://127.0.0.1:8184");
        assert_eq!(c.port, 8184);
        assert_eq!(c.backend, "auto");
        assert_eq!(c.policy, "quality");
        assert_eq!(c.api_key, "local");
        assert_eq!(c.engine_root, PathBuf::from("/home/u/Colibri"));
        assert_eq!(c.web_url, "http://127.0.0.1:8787");
    }

    #[test]
    fn rejects_secret_in_config() {
        let read = |_: &Path| Some(r#"{"api_key":"x"}"#.to_string());
        let exists = |_: &Path| true;
        let err = load_config(&no_env, &home(), false, &exists, &read).unwrap_err();
        assert!(err.contains("Keep secrets"), "{err}");
    }

    #[test]
    fn non_loopback_requires_key() {
        let env = |k: &str| if k == "FENNEC_ENGINE_URL" { Some("http://10.0.0.1:8184".into()) } else { None };
        let err = load_config(&env, &home(), false, &no_exists, &no_read).unwrap_err();
        assert!(err.contains("FENNEC_API_KEY"), "{err}");
    }

    #[test]
    fn origin_validation() {
        assert!(parse_origin("http://127.0.0.1:8184").is_ok());
        assert!(parse_origin("http://127.0.0.1:8184/").is_ok());
        for bad in ["http://u:p@h:80", "http://h:0", "http://h/x", "http://h?q", "http://h#f", "ftp://h"] {
            assert!(parse_origin(bad).is_err(), "{bad}");
        }
        assert_eq!(parse_origin("https://h").unwrap().port, 443);
    }

    #[test]
    fn backend_and_policy_validated() {
        let read = |_: &Path| Some(r#"{"backend":"rocm"}"#.to_string());
        let exists = |_: &Path| true;
        assert!(load_config(&no_env, &home(), false, &exists, &read).unwrap_err().contains("backend"));
        let read2 = |_: &Path| Some(r#"{"policy":"turbo"}"#.to_string());
        assert!(load_config(&no_env, &home(), false, &exists, &read2).unwrap_err().contains("policy"));
    }

    #[test]
    fn engine_env_pins_vulkan_for_colibri_only() {
        let c = load_config(&no_env, &home(), false, &no_exists, &no_read).unwrap();
        let exists = |p: &Path| p.to_string_lossy().contains("vulkan-1.dll");
        let env = |k: &str| if k == "SystemRoot" { Some("C:\\Windows".into()) } else { None };
        let e = engine_environment(&c, &env, "win32", &exists);
        let get = |k: &str| e.iter().find(|(ek, _)| ek == k).map(|(_, v)| v.clone());
        assert_eq!(get("COLI_VULKAN").as_deref(), Some("1"));
        assert_eq!(get("COLI_VK_TIER_HOST_GB").as_deref(), Some("4"));
        assert_eq!(get("COLI_API_KEY").as_deref(), Some("local"));
    }
}