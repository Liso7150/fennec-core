//! The Rust desktop shell: replaces fennec-desktop/electron/*.js. A std-only
//! HTTP+SSE server that serves the web renderer and exposes the agent API
//! (config, models, engine lifecycle, context economy, outlets, ACP agent loop)
//! backed by the fennec-* Rust crates, then drives a bundled Chromium in app
//! mode. No Electron, no Node: the agent logic is Rust end to end.
use fennec_desktop_shell::{config, engine};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type Clients = Arc<Mutex<Vec<std::sync::mpsc::Sender<String>>>>;

const BENCH_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8"></head>
<body><div id="root">sse bench</div><script>
let n=0,t0=0,want=2000;
const es=new EventSource('/api/events');
es.addEventListener('bench-token',()=>{ if(n===0)t0=performance.now(); n++;
  if(n>=want){const ms=performance.now()-t0;
    fetch('/api/record',{method:'POST',body:JSON.stringify({name:'stream-delivered',elapsedMs:ms,count:n})});
    fetch('/api/record',{method:'POST',body:JSON.stringify({name:'stream-per-token',elapsedMs:ms/n,count:n})});
    document.getElementById('root').textContent='delivered '+n+' in '+ms.toFixed(1)+' ms'; es.close();}});
es.onopen=()=>fetch('/api/stream_emit',{method:'POST',body:JSON.stringify({count:want})});
</script></body></html>"#;

fn env(key: &str) -> Option<String> { std::env::var(key).ok() }

fn real_config() -> Result<config::Config, String> {
    let home = PathBuf::from(env("USERPROFILE").or_else(|| env("HOME")).unwrap_or_else(|| ".".into()));
    config::load_config(&env, &home, cfg!(windows), &|p| p.exists(), &|p| std::fs::read_to_string(p).ok())
}

fn send_json(stream: &mut TcpStream, status: &str, value: &serde_json::Value) {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let head = format!("HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", bytes.len());
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&bytes);
}

fn record_row(name: &str, ms: f64, count: usize) {
    let line = serde_json::json!({"shell":"chromium-sse","backend":name,"elapsed_ms":ms,"messages":count});
    let out = env("SHELL_BENCH_OUT").unwrap_or_default();
    if out.is_empty() { println!("{line}"); }
    else if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&out) { let _ = writeln!(f, "{line}"); }
}

fn content_type(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn handle_api(cmd: &str, args: &serde_json::Value, clients: &Clients) -> Result<serde_json::Value, String> {
    match cmd {
        "stream_emit" => {
            let n = args.get("count").and_then(|c| c.as_u64()).unwrap_or(2000) as usize;
            let started = std::time::Instant::now();
            let guard = clients.lock().unwrap();
            for i in 0..n {
                let event = format!("event: bench-token\ndata: {i}\n\n");
                for tx in guard.iter() { let _ = tx.send(event.clone()); }
            }
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            record_row("stream-emit", ms, n);
            Ok(serde_json::json!({ "emit_ms": ms }))
        }
        "record" => {
            let name = args.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let ms = args.get("elapsedMs").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let count = args.get("count").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            record_row(name, ms, count);
            Ok(serde_json::json!({}))
        }
        "config" => {
            let c = real_config()?;
            Ok(serde_json::json!({
                "engine_url": c.engine_url, "backend": c.backend, "policy": c.policy, "ctx": c.ctx,
                "models_root": c.models_root, "workspace": c.workspace, "web_url": c.web_url,
                "port": c.port, "engine_root": c.engine_root,
            }))
        }
        "models" => {
            let c = real_config()?;
            let list: Vec<serde_json::Value> = engine::list_models(&c).iter().map(|m| serde_json::json!({
                "dir": m.dir, "name": m.name, "family": m.family, "layers": m.layers, "experts": m.experts,
                "sizeGb": m.size_gb, "shards": m.shards, "gguf": m.gguf, "container": m.container, "ready": m.ready,
            })).collect();
            Ok(serde_json::Value::Array(list))
        }
        "engine_probe" => {
            let c = real_config()?;
            let probe = engine::probe_server(&c.engine_url, &c.api_key);
            Ok(serde_json::json!({ "probe": format!("{:?}", probe) }))
        }
        "engine_describe" => {
            let c = real_config()?;
            let rt = config::runtime_paths(&c, cfg!(windows), &|p| p.exists());
            Ok(serde_json::json!({ "kind": rt.kind, "cli": rt.cli, "runtime": rt.runtime }))
        }
        "context_compact" => {
            let messages = args.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
            let mut seen = std::collections::HashMap::new();
            let opts = fennec_economy::Options::sized(
                args.get("budgetChars").and_then(|v| v.as_u64()).unwrap_or(fennec_economy::DEFAULT_BUDGET_CHARS as u64) as usize,
                args.get("resultChars").and_then(|v| v.as_u64()).unwrap_or(fennec_economy::DEFAULT_RESULT_CHARS as u64) as usize);
            let (out, stats) = fennec_economy::compact(&messages, &opts, &mut seen);
            Ok(serde_json::json!({ "messages": out, "stats": {
                "chars_before": stats.chars_before, "chars_after": stats.chars_after,
                "elided": stats.elided, "deduped": stats.deduped } }))
        }
        "outlets_status" => {
            let origins = args.get("origins").and_then(|o| o.as_array()).cloned().unwrap_or_default();
            let rows: Vec<serde_json::Value> = origins.iter().filter_map(|o| o.as_str()).map(|o| serde_json::json!({
                "origin": o, "loopback": fennec_outlets::loopback(o), "blocked": fennec_outlets::blocked_host(o),
            })).collect();
            Ok(serde_json::Value::Array(rows))
        }
        "agent_send" => {
            let c = real_config()?;
            let messages = args.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
            let mut seen = std::collections::HashMap::new();
            let opts = fennec_economy::Options::sized(fennec_economy::DEFAULT_BUDGET_CHARS, fennec_economy::DEFAULT_RESULT_CHARS);
            let (compacted, stats) = fennec_economy::compact(&messages, &opts, &mut seen);
            let frame = fennec_acp::encode_frame(&serde_json::json!({ "jsonrpc": "2.0", "method": "session/update", "params": { "messages": compacted } }));
            let reply = engine::http_post(&c.engine_url, "/v1/chat/completions", &c.api_key,
                &serde_json::json!({ "model": c.model, "messages": compacted, "stream": false }).to_string(), 30000);
            Ok(serde_json::json!({
                "stats": { "chars_before": stats.chars_before, "chars_after": stats.chars_after },
                "acp_frame": frame, "engine_status": reply.status, "engine_body": reply.body,
            }))
        }
        _ => Err(format!("unknown command {cmd}")),
    }
}

fn handle_conn(mut stream: TcpStream, dist: PathBuf, clients: Clients) {
    let cloned = match stream.try_clone() { Ok(s) => s, Err(_) => return };
    let mut reader = BufReader::new(cloned);
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 { return; }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut content_len = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 { break; }
        let t = header.trim_end();
        if t.is_empty() { break; }
        if let Some((k, v)) = t.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") { content_len = v.trim().parse().unwrap_or(0); }
        }
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 && reader.read_exact(&mut body).is_err() { return; }
    let route = target.split('?').next().unwrap_or("/").to_string();

    if route == "/api/events" {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        clients.lock().unwrap().push(tx);
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        if stream.write_all(head.as_bytes()).is_err() { return; }
        let _ = stream.flush();
        for event in rx {
            if stream.write_all(event.as_bytes()).is_err() { break; }
            let _ = stream.flush();
        }
        return;
    }
    if route == "/bench" {
        let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", BENCH_HTML.len());
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(BENCH_HTML.as_bytes());
        return;
    }
    if let Some(cmd) = route.strip_prefix("/api/") {
        if method != "POST" { send_json(&mut stream, "405 Method Not Allowed", &serde_json::json!("POST required")); return; }
        let args: serde_json::Value = if body.is_empty() { serde_json::json!({}) } else { serde_json::from_slice(&body).unwrap_or(serde_json::json!({})) };
        match handle_api(cmd, &args, &clients) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(e) => send_json(&mut stream, "500 Internal Server Error", &serde_json::json!(e)),
        }
        return;
    }
    let rel = if route == "/" { "index.html".to_string() } else { route.trim_start_matches('/').to_string() };
    if rel.contains("..") { send_json(&mut stream, "403 Forbidden", &serde_json::json!("forbidden")); return; }
    let file = dist.join(&rel);
    match std::fs::read(&file) {
        Ok(bytes) => {
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", content_type(&file), bytes.len());
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&bytes);
        }
        Err(_) => send_json(&mut stream, "404 Not Found", &serde_json::json!(format!("missing {rel}"))),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let dist = PathBuf::from(flag("--dist").unwrap_or_else(|| "dist".into()));
    let no_browser = has("--no-browser");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().expect("addr").port();
    let url = format!("http://127.0.0.1:{port}/");
    eprintln!("fennec-desktop-shell listening on {url}");

    if !no_browser {
        let chrome = flag("--chrome").unwrap_or_else(|| "../chromium/chrome-win/chrome.exe".into());
        let profile = std::env::temp_dir().join("fennec-desktop-shell-profile");
        let spawned = std::process::Command::new(&chrome)
            .arg(format!("--app={url}"))
            .arg(format!("--user-data-dir={}", profile.display()))
            .arg("--no-first-run").arg("--no-default-browser-check")
            .arg("--disable-background-networking").arg("--disable-sync").arg("--disable-extensions")
            .arg("--disable-component-update").arg("--disable-default-apps").arg("--disable-popup-blocking")
            .arg("--noerrdialogs")
            .arg("--in-process-gpu").arg("--enable-features=NetworkServiceInProcess")
            .arg("--disable-crash-reporter").arg("--renderer-process-limit=1")
            .arg("--js-flags=--max-old-space-size=256").arg("--disk-cache-size=1")
            .arg("--disable-gpu-shader-disk-cache")
            .arg("--disable-features=Translate,MediaRouter,OptimizationHints,CalculateNativeWinOcclusion,BackForwardCache,SpareRendererForSitePerProcess,IntensiveWakeUpThrottling")
            .arg("--disable-hang-monitor").arg("--disable-client-side-phishing-detection")
            .arg("--disable-domain-reliability").arg("--disable-metrics").arg("--mute-audio")
            .arg("--window-size=1100,820").arg("--window-position=80,60")
            .spawn();
        if let Ok(mut child) = spawned {
            std::thread::spawn(move || { let _ = child.wait(); std::process::exit(0); });
        }
    }

    let dist = Arc::new(dist);
    let clients: Clients = Arc::new(Mutex::new(Vec::new()));
    for conn in listener.incoming() {
        if let Ok(stream) = conn {
            let dist = dist.clone();
            let clients = clients.clone();
            let _ = std::thread::Builder::new().stack_size(1024 * 1024)
                .spawn(move || handle_conn(stream, (*dist).clone(), clients));
        }
    }
}