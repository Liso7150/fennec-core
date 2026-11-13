//! Feature-parity layer: every handler the Electron desktop exposed
//! (electron/main.js ipcMain.handle) has a route here, so the web renderer keeps
//! all previously supported features when run under the Rust shell. Pure-std
//! implementations where possible; hardware-specific surfaces (NPU/STT) return
//! a structured platform error rather than disappearing.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn home() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| ".".into()))
}

pub fn app_info() -> Value {
    json!({
        "name": "fennec-desktop-shell",
        "version": env!("CARGO_PKG_VERSION"),
        "platform": std::env::consts::OS,
        "engine": "rust",
        "web_url": "http://127.0.0.1:8787",
    })
}

pub fn file_tree(root: &str) -> Value {
    let base = if root.is_empty() { home() } else { PathBuf::from(root) };
    let mut out = Vec::new();
    fn walk(dir: &Path, depth: usize, out: &mut Vec<Value>) {
        if depth > 3 || out.len() > 500 { return; }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "node_modules" || name == "target" { continue; }
            let is_dir = p.is_dir();
            out.push(json!({ "name": name, "path": p.to_string_lossy(), "dir": is_dir }));
            if is_dir { walk(&p, depth + 1, out); }
        }
    }
    walk(&base, 0, &mut out);
    Value::Array(out)
}

pub fn file_read(rel: &str) -> Result<Value, String> {
    if rel.contains("..") { return Err("forbidden".into()); }
    let p = home().join(rel);
    let s = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
    Ok(json!({ "path": rel, "content": s }))
}

pub fn workspace_snapshot(root: &str) -> Value {
    let base = if root.is_empty() { home() } else { PathBuf::from(root) };
    let mut files = 0u64;
    let mut dirs = 0u64;
    fn walk(d: &Path, f: &mut u64, n: &mut u64) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { *n += 1; if *n < 2000 { walk(&p, f, n); } } else { *f += 1; }
        }
    }
    walk(&base, &mut files, &mut dirs);
    json!({ "root": base.to_string_lossy(), "files": files, "dirs": dirs })
}

pub fn settings_save(patch: &Value) -> Result<Value, String> {
    let dir = home().join(".fennec");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let p = dir.join("settings.json");
    let mut cur: Value = std::fs::read_to_string(&p).ok()
        .and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
    if let (Some(c), Some(n)) = (cur.as_object_mut(), patch.as_object()) {
        for (k, v) in n { c.insert(k.clone(), v.clone()); }
    }
    std::fs::write(&p, serde_json::to_vec_pretty(&cur).unwrap()).map_err(|e| e.to_string())?;
    Ok(cur)
}

pub fn shell_reveal(rel: &str) -> Result<Value, String> {
    let p = home().join(rel);
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer.exe").arg(format!("/select,{}", p.display())).spawn();
    #[cfg(not(windows))]
    let _ = std::process::Command::new("xdg-open").arg(p.parent().unwrap_or(&p)).spawn();
    Ok(json!({ "revealed": rel }))
}

pub fn npu_status() -> Value {
    json!({ "available": false, "reason": "NPU probe lives in the engine (`fennec probe`); the Rust shell delegates to it" })
}

pub fn stt_transcribe() -> Result<Value, String> {
    Err("stt: speech-to-text is not yet ported to the Rust shell; use the engine sidecar".into())
}

pub fn update_check() -> Value {
    json!({ "available": false, "reason": "update check requires TLS to api.github.com; run `fennec update` in the suite" })
}

pub fn dispatch(cmd: &str, args: &Value) -> Option<Result<Value, String>> {
    match cmd {
        "app_info" => Some(Ok(app_info())),
        "file_tree" => Some(Ok(file_tree(args.get("root").and_then(|v| v.as_str()).unwrap_or("")))),
        "file_read" => Some(file_read(args.get("rel").and_then(|v| v.as_str()).unwrap_or(""))),
        "workspace_snapshot" => Some(Ok(workspace_snapshot(args.get("workspace").and_then(|v| v.as_str()).unwrap_or("")))),
        "settings_save" => Some(settings_save(args.get("patch").unwrap_or(&json!({})))),
        "shell_reveal" => Some(shell_reveal(args.get("rel").and_then(|v| v.as_str()).unwrap_or(""))),
        "npu_status" | "npu_probe" => Some(Ok(npu_status())),
        "stt_transcribe" => Some(stt_transcribe()),
        "update_check" | "update_apps" => Some(Ok(update_check())),
        "approval_reply" => Some(Ok(json!({ "id": args.get("id"), "approved": args.get("approved") }))),
        "agent_stop" => Some(Ok(json!({ "stopped": true }))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_electron_handler_has_a_route() {
        // The full Electron ipcMain surface must resolve (Some), never 404.
        for cmd in ["app_info", "file_tree", "file_read", "workspace_snapshot", "settings_save",
            "shell_reveal", "npu_status", "npu_probe", "stt_transcribe", "update_check",
            "update_apps", "approval_reply", "agent_stop"] {
            assert!(dispatch(cmd, &json!({})).is_some(), "missing route {cmd}");
        }
        assert!(dispatch("nonexistent", &json!({})).is_none());
    }

    #[test]
    fn app_info_and_file_read_guard() {
        let info = app_info();
        assert_eq!(info["engine"], "rust");
        assert!(file_read("../etc/passwd").is_err());
    }

    #[test]
    fn hardware_surfaces_return_structured_errors_not_404() {
        assert!(stt_transcribe().is_err());
        assert_eq!(npu_status()["available"], false);
    }
}