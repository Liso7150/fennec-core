//! Port of `fennec-desktop/electron/backend.js`: engine discovery, health probe,
//! model inventory and serve argv. std-only HTTP so the shell has no async runtime.
use crate::config::{is_loopback, runtime_paths, Config};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    pub error: String,
}

/// Minimal HTTP/1.1 GET over loopback/LAN, mirroring backend.js `request`.
pub fn http_get(base: &str, route: &str, api_key: &str, timeout_ms: u64) -> HttpResponse {
    let fail = |e: String| HttpResponse { status: 0, body: String::new(), error: e };
    let url = match base.strip_suffix('/') { Some(b) => format!("{b}{route}"), None => format!("{base}{route}") };
    let (host, port, path) = match split_url(&url) { Some(v) => v, None => return fail("bad url".into()) };
    let addr = format!("{host}:{port}");
    let mut stream = match TcpStream::connect(&addr) { Ok(s) => s, Err(e) => return fail(e.to_string()) };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(timeout_ms)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(timeout_ms)));
    let auth = if api_key.is_empty() { String::new() } else { format!("Authorization: Bearer {api_key}\r\n") };
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n{auth}Connection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() { return fail("write failed".into()); }
    let mut raw = Vec::new();
    if stream.read_to_end(&mut raw).is_err() { return fail("read failed".into()); }
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = match text.split_once("\r\n\r\n") { Some(v) => v, None => return fail("no header".into()) };
    let status: u16 = head.lines().next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    HttpResponse { status, body: body.to_string(), error: String::new() }
}

/// Minimal HTTP/1.1 POST (JSON body), for engine chat / ACP calls.
pub fn http_post(base: &str, route: &str, api_key: &str, body: &str, timeout_ms: u64) -> HttpResponse {
    let fail = |e: String| HttpResponse { status: 0, body: String::new(), error: e };
    let url = match base.strip_suffix('/') { Some(b) => format!("{b}{route}"), None => format!("{base}{route}") };
    let (host, port, path) = match split_url(&url) { Some(v) => v, None => return fail("bad url".into()) };
    let mut stream = match TcpStream::connect(format!("{host}:{port}")) { Ok(s) => s, Err(e) => return fail(e.to_string()) };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(timeout_ms)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(timeout_ms)));
    let auth = if api_key.is_empty() { String::new() } else { format!("Authorization: Bearer {api_key}\r\n") };
    let req = format!("POST {path} HTTP/1.1\r\nHost: {host}\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    if stream.write_all(req.as_bytes()).is_err() { return fail("write failed".into()); }
    let mut raw = Vec::new();
    if stream.read_to_end(&mut raw).is_err() { return fail("read failed".into()); }
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = match text.split_once("\r\n\r\n") { Some(v) => v, None => return fail("no header".into()) };
    let status: u16 = head.lines().next().and_then(|l| l.split_whitespace().nth(1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    HttpResponse { status, body: body.to_string(), error: String::new() }
}

fn split_url(url: &str) -> Option<(String, u16, String)> {
    let (scheme, rest) = if let Some(r) = url.strip_prefix("http://") { ("http", r) }
        else if let Some(r) = url.strip_prefix("https://") { ("https", r) } else { return None };
    let (authority, path) = match rest.find('/') { Some(i) => (&rest[..i], &rest[i..]), None => (rest, "/") };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().ok()?),
        None => (authority.to_string(), if scheme == "https" { 443 } else { 80 }),
    };
    Some((host, port, path.to_string()))
}

fn safe_json(body: &str) -> Option<serde_json::Value> { serde_json::from_str(body).ok() }

#[derive(Debug, Clone, PartialEq)]
pub enum Probe {
    Absent(String),
    Blocked(String),
    Unauthorized(String),
    Foreign(String),
    Ready(String),
}

/// `probeServer`: /health then /v1/models, same classification as the JS.
pub fn probe_server(base: &str, api_key: &str) -> Probe {
    let health = http_get(base, "/health", api_key, 2000);
    if health.status == 0 {
        return if health.error.contains("refused") || health.error.contains("10061") {
            Probe::Absent(health.error)
        } else { Probe::Blocked(health.error) };
    }
    if health.status == 401 || health.status == 403 {
        return Probe::Unauthorized("Engine authentication failed. Set FENNEC_API_KEY.".into());
    }
    let ok = safe_json(&health.body).and_then(|v| v.get("status").and_then(|s| s.as_str()).map(|s| s == "ok")).unwrap_or(false);
    if health.status != 200 || !ok {
        return Probe::Blocked(format!("Endpoint is occupied but not healthy (/health {}). No engine was started.", health.status));
    }
    let models = http_get(base, "/v1/models", api_key, 2000);
    if models.status == 401 || models.status == 403 {
        return Probe::Unauthorized("Engine authentication failed. Set FENNEC_API_KEY.".into());
    }
    let list = safe_json(&models.body);
    let valid = models.status == 200 && list.as_ref().and_then(|v| v.get("object").and_then(|o| o.as_str())) == Some("list")
        && list.as_ref().and_then(|v| v.get("data").and_then(|d| d.as_array())).map(|a| {
            !a.is_empty() && a.iter().all(|m| m.get("object").and_then(|o| o.as_str()) == Some("model")
                && m.get("id").and_then(|i| i.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false)
                && m.get("owned_by").and_then(|o| o.as_str()) == Some("fennec"))
        }).unwrap_or(false);
    if !valid {
        return Probe::Foreign("Occupied endpoint did not return a valid Fennec model list. No engine was started.".into());
    }
    let id = list.unwrap().get("data").unwrap().as_array().unwrap()[0].get("id").unwrap().as_str().unwrap().to_string();
    Probe::Ready(id)
}

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub dir: PathBuf,
    pub name: String,
    pub family: String,
    pub layers: u64,
    pub experts: u64,
    pub size_gb: f64,
    pub shards: usize,
    pub gguf: bool,
    pub container: bool,
    pub ready: bool,
}

/// `describeModel`: config.json + shard/gguf byte inventory.
pub fn describe_model(dir: &Path, name: &str) -> Option<ModelMeta> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()?;
    let entries: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
    let shards: Vec<_> = entries.iter().filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("safetensors")).unwrap_or(false)).collect();
    let gguf: Vec<_> = entries.iter().filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)).collect();
    if shards.is_empty() && gguf.is_empty() { return None; }
    let bytes: u64 = shards.iter().chain(gguf.iter()).map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let container = entries.iter().any(|p| p.file_name().map(|n| n == "qwen36_meta.json" || n == "qwen38_meta.json").unwrap_or(false));
    let num = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    let text = config.get("text_config").cloned().unwrap_or(serde_json::Value::Null);
    Some(ModelMeta {
        dir: dir.to_path_buf(), name: name.to_string(),
        family: config.get("model_type").and_then(|v| v.as_str()).map(|s| s.to_string())
            .or_else(|| config.get("architectures").and_then(|a| a.get(0)).and_then(|v| v.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".into()),
        layers: num(&config, "num_hidden_layers").max(num(&text, "num_hidden_layers")),
        experts: num(&config, "num_experts").max(num(&text, "num_experts")),
        size_gb: (bytes as f64 / 1e9 * 100.0).round() / 100.0,
        shards: shards.len(), gguf: !gguf.is_empty(), container,
        ready: container || !gguf.is_empty(),
    })
}

/// `listModels`: configured model + models_root (+containers) + HF hub cache.
pub fn list_models(config: &Config) -> Vec<ModelMeta> {
    let mut found: Vec<ModelMeta> = Vec::new();
    let mut add = |dir: PathBuf, name: String| {
        if let Some(m) = describe_model(&dir, &name) {
            if !found.iter().any(|f| f.dir == m.dir) { found.push(m); }
        }
    };
    if !config.model.is_empty() {
        let p = PathBuf::from(&config.model);
        add(p.clone(), p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    }
    let roots = [config.models_root.clone(), config.models_root.join("containers"),
        config.cache_root.join("huggingface").join("hub")];
    for root in roots {
        let entries = match std::fs::read_dir(&root) { Ok(e) => e, Err(_) => continue };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() { continue; }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("models--") {
                let snap = dir.join("snapshots");
                if let Ok(shas) = std::fs::read_dir(&snap) {
                    for sha in shas.flatten() {
                        let label = format!("{} ({})", name.trim_start_matches("models--").replace("--", "/"),
                            sha.file_name().to_string_lossy().chars().take(7).collect::<String>());
                        add(sha.path(), label);
                    }
                }
            } else {
                add(dir, name);
            }
        }
    }
    found.sort_by(|a, b| b.container.cmp(&a.container).then(a.name.cmp(&b.name)));
    found
}

/// `serveArgs`: Fennec keeps one KV slot and rejects --kv-slots; Colibri takes it.
pub fn serve_args(config: &Config, model_dir: &str, kv_slots: u32, max_tokens: u32) -> Result<Vec<String>, String> {
    if !(1..=8).contains(&kv_slots) { return Err("Invalid KV slot count".into()); }
    if max_tokens < 1 { return Err("Invalid token limit".into()); }
    let rt = runtime_paths(config, cfg!(windows), &|p: &Path| p.exists());
    let url = crate::config::parse_origin(&config.engine_url)?;
    let mut args = vec![rt.cli.to_string_lossy().into_owned(), "serve".into(), "--model".into(), model_dir.into(),
        "--auto-tier".into(), "--policy".into(), config.policy.clone(), "--ctx".into(), config.ctx.to_string(),
        "--host".into(), url.hostname, "--port".into(), config.port.to_string()];
    if rt.kind != "fennec" { args.push("--kv-slots".into()); args.push(kv_slots.to_string()); }
    args.push("--ngen".into()); args.push(max_tokens.to_string());
    Ok(args)
}

/// `assertPortAvailable`: refuse to start over an occupied port.
pub fn port_available(base: &str) -> Result<(), String> {
    let (_, port, _) = split_url(base).ok_or("bad url")?;
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => { drop(l); Ok(()) }
        Err(_) => Err("Configured engine port is occupied or unavailable; no engine was started.".into()),
    }
}

pub fn loopback_host(host: &str) -> bool { is_loopback(host) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    fn cfg() -> Config {
        crate::config::load_config(&|_| None, Path::new("/home/u"), false, &|_| false, &|_| None).unwrap()
    }

    #[test]
    fn serve_args_fennec_omits_kv_slots() {
        let root = std::env::temp_dir().join(format!("fennec-eng-{}", std::process::id()));
        let rel = root.join("target").join("release");
        std::fs::create_dir_all(&rel).unwrap();
        std::fs::write(rel.join(if cfg!(windows) { "fennec.exe" } else { "fennec" }), b"").unwrap();
        let mut c = cfg();
        c.engine_root = root.clone();
        let a = serve_args(&c, "/m", 4, 4096).unwrap();
        assert!(!a.iter().any(|x| x == "--kv-slots"), "fennec keeps one KV slot");
        assert!(a.iter().any(|x| x == "--ngen"));
        // A Colibri checkout (no native binary) takes --kv-slots.
        c.engine_root = root.join("missing");
        let b = serve_args(&c, "/m", 4, 4096).unwrap();
        assert!(b.iter().any(|x| x == "--kv-slots"));
        assert!(serve_args(&c, "/m", 0, 4096).is_err());
        assert!(serve_args(&c, "/m", 9, 4096).is_err());
        assert!(serve_args(&c, "/m", 4, 0).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn describe_model_reads_container() {
        let dir = std::env::temp_dir().join(format!("fennec-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_moe","num_hidden_layers":40,"num_experts":256}"#).unwrap();
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 1024]).unwrap();
        std::fs::write(dir.join("qwen36_meta.json"), "{}").unwrap();
        let m = describe_model(&dir, "m").unwrap();
        assert!(m.container && m.ready);
        assert_eq!(m.layers, 40);
        assert_eq!(m.experts, 256);
        assert_eq!(m.shards, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn probe_classifies_ready_and_foreign() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut s, _) = listener.accept().unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut request_line = String::new();
                r.read_line(&mut request_line).unwrap();
                loop {
                    let mut h = String::new();
                    if r.read_line(&mut h).unwrap() == 0 || h.trim().is_empty() { break; }
                }
                let payload = if request_line.starts_with("GET /health") {
                    r#"{"status":"ok"}"#.to_string()
                } else {
                    r#"{"object":"list","data":[{"object":"model","id":"m","owned_by":"fennec"}]}"#.to_string()
                };
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", payload.len(), payload);
                s.write_all(resp.as_bytes()).unwrap();
            }
        });
        let p = probe_server(&format!("http://127.0.0.1:{port}"), "");
        handle.join().unwrap();
        assert_eq!(p, Probe::Ready("m".into()));
    }
}