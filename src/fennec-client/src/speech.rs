//! The local speech sidecar: `fennec-core/stt.py`, faster-whisper on the CPU,
//! loopback only, holding the model on 8183.
//!
//! Shells do not start it themselves and a shell must not need the Python core
//! running to dictate: this is the one place that owns the lifecycle, so
//! `POST /api/stt` on the Rust agent and `fennec-core/server.py` both bring the
//! same process up. Going straight to 8183 — which is what a client does when
//! no route owns it — is what made dictation dead on arrival: nothing answers
//! the port, every request is refused, and an empty transcript is
//! indistinguishable from a silent microphone.
use crate::git;
use crate::http;
use crate::suite::{self, Suite};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const SIDECAR: &str = "http://127.0.0.1:8183";
/// The sidecar's own cap (`stt.py:MAX_BYTES`), so an over-long clip is refused
/// with our sentence instead of a bare 413 off a raw socket.
pub const MAX_AUDIO_BYTES: usize = 16 << 20;
/// `engine.py:transcribe` allows 180 s for a clip; the model runs int8 on CPU.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(180);
/// A cold start downloads the model, so the wait is the core's 300 s, not a probe.
const START_TIMEOUT: Duration = Duration::from_secs(300);

/// Held across a cold start so two dictations racing on an empty port spawn one
/// process, not two — the loser would die on an occupied 8183 and answer with
/// its own startup noise instead of a transcript.
fn start_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Where the sidecar writes what it says about itself, the same file the Python
/// core uses (`engine.py:129`), so one log describes the process either owner
/// started.
pub fn log_path() -> PathBuf {
    cache_root().join("fennec").join("stt.log")
}

/// `config.py:76` — `FENNEC_CACHE`, else `~/.cache/fennec`.
fn cache_root() -> PathBuf {
    std::env::var_os("FENNEC_CACHE").map(PathBuf::from).unwrap_or_else(|| suite::home().join(".cache").join("fennec"))
}

pub struct Speech {
    pub base: String,
}

impl Speech {
    pub fn new() -> Self {
        Self { base: std::env::var("FENNEC_STT_URL").unwrap_or_else(|_| SIDECAR.to_string()) }
    }

    /// `{"status":"ok","model":…,"loaded":…}` from the sidecar, or the reason it
    /// is not answering. `Ok` here means the port is held and ready.
    pub fn status(&self) -> Result<serde_json::Value, String> {
        http::get_json(&self.base, "/health", &[])
    }

    fn healthy(&self) -> bool {
        self.status().map(|v| v["status"] == "ok").unwrap_or(false)
    }

    /// Bring the sidecar up if the port is empty, then wait for it to answer.
    pub fn ensure(&self) -> Result<(), String> {
        self.ensure_in(&suite::discover())
    }

    /// `ensure()` with the suite layout injected, which is how the refusal
    /// paths are tested without starting an interpreter.
    pub fn ensure_in(&self, suite: &Suite) -> Result<(), String> {
        if self.healthy() {
            return Ok(());
        }
        let _guard = start_lock().lock().unwrap_or_else(|e| e.into_inner());
        if self.healthy() {
            return Ok(());
        }
        let script = sidecar_script(suite)?;
        let python = Suite::python().ok_or_else(|| {
            format!("{} needs python on PATH; run the Fennec environment setup", script.display())
        })?;
        let dir = script.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        let log = log_path();
        if let Some(parent) = log.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let file = std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut cmd = std::process::Command::new(&python);
        // The sidecar reads nothing relative — the model comes from `HF_HOME` and
        // the script from an absolute path — so it runs from its own directory.
        // `engine_root` is what the Python core passes, and it is only real when
        // an engine checkout was found: spawning there is what turned a missing
        // `~/Fennec` into `os error 267` and no transcript.
        cmd.arg(&script)
            .arg("--port")
            .arg(http::split_base(&self.base).1.to_string())
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(file.try_clone().map_err(|e| format!("{}: {e}", log.display()))?))
            .stderr(Stdio::from(file));
        // HF_HOME is what puts the downloaded model in the shared cache instead
        // of a second copy under the user's home; the keys are for services the
        // sidecar never calls (`engine.py:ensure_stt` scrubs the same names).
        cmd.env("HF_HOME", cache_root().join("huggingface"))
            .env_remove("FENNEC_API_KEY")
            .env_remove("COLI_API_KEY")
            .env_remove("FENNEC_APP_TOKEN");
        let mut child = git::quiet(&mut cmd)
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", script.display()))?;
        let began = Instant::now();
        while began.elapsed() < START_TIMEOUT {
            if self.healthy() {
                // Dropping the handle leaves the process running, which is the
                // point: the sidecar outlives the request that needed it.
                return Ok(());
            }
            if let Some(code) = child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!(
                    "the speech sidecar exited with {code} during startup; {}",
                    log_tail(&log)
                ));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        let _ = child.kill();
        Err(format!("the speech model did not become ready in {}s; {}", START_TIMEOUT.as_secs(), log_tail(&log)))
    }

    /// Raw audio bytes in, the sidecar's reply out (`text`, `language`,
    /// `duration_s`), so a server can answer with the same body the Python core
    /// does. `content_type` is whatever the browser or the shell recorded
    /// (`audio/webm`, `audio/wav`).
    pub fn transcribe(&self, audio: &[u8], content_type: &str) -> Result<serde_json::Value, String> {
        let ct = if content_type.trim().is_empty() { "application/octet-stream" } else { content_type.trim() };
        let reply = http::post_bytes(&self.base, "/transcribe", &[], ct, audio, TRANSCRIBE_TIMEOUT)?;
        if reply["text"].is_string() {
            Ok(reply)
        } else {
            Err(format!("the sidecar replied without a text field: {reply}"))
        }
    }
}

impl Default for Speech {
    fn default() -> Self {
        Self::new()
    }
}

/// The script inside the `fennec-core` checkout, named in the refusal when the
/// checkout is missing or predates the flattened package.
fn sidecar_script(suite: &Suite) -> Result<PathBuf, String> {
    match suite.core_dir() {
        Some(dir) if dir.join("stt.py").is_file() => Ok(dir.join("stt.py")),
        Some(dir) => Err(format!(
            "no stt.py in {} — update fennec-core, or set FENNEC_HOME to the checkout",
            dir.display()
        )),
        None => Err("no fennec-core checkout — install it or set FENNEC_HOME".to_string()),
    }
}

/// Last lines of the sidecar log, which is where "faster-whisper is not
/// installed" and a bad model name actually appear.
fn log_tail(log: &std::path::Path) -> String {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let tail: Vec<&str> = text.lines().rev().take(6).collect();
    if tail.is_empty() {
        format!("{} is empty", log.display())
    } else {
        format!("{} said: {}", log.display(), tail.join(" | "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::Repo;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    /// Answers one request and records what arrived, the way the sidecar does:
    /// a health reply for a GET, `transcript` for a POST, `refusal` when set.
    fn sidecar(transcript: Option<&str>, refusal: Option<(u16, &str)>) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let recorded = transcript.unwrap_or("nothing").to_string();
        let refused = refusal.map(|(code, reason)| (code, reason.to_string()));
        let sink = seen.clone();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            let mut get = false;
            let mut length = 0usize;
            let mut first = true;
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if first {
                    get = line.starts_with("GET");
                    first = false;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).ok();
            *sink.lock().unwrap() = format!("{} bytes: {:?}", body.len(), body);
            let (status, payload) = if get {
                (200, r#"{"status":"ok","model":"base.en","device":"cpu","loaded":true}"#.to_string())
            } else {
                match &refused {
                    Some((code, reason)) => (*code, format!(r#"{{"error":"{reason}"}}"#)),
                    None => (200, format!(r#"{{"text":"{recorded}","language":"en","duration_s":1.2}}"#)),
                }
            };
            let reply = format!(
                "HTTP/1.1 {status} Answer\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(reply.as_bytes()).ok();
        });
        (base, seen)
    }

    #[test]
    fn a_clip_is_posted_as_its_own_bytes_not_as_text() {
        // 0xFF then 0xC0 is invalid UTF-8: a body that traveled as a &str would
        // arrive as replacement characters and the sidecar could not decode it.
        let (base, seen) = sidecar(Some("hello there"), None);
        let speech = Speech { base };
        assert_eq!(speech.transcribe(&[0x52, 0xFF, 0xC0, 0x0A], "audio/wav").unwrap()["text"], "hello there");
        let recorded = seen.lock().unwrap().clone();
        assert!(recorded.contains("4 bytes"), "{recorded}");
        assert!(recorded.contains("[82, 255, 192, 10]"), "audio bytes were rewritten: {recorded}");
    }

    #[test]
    fn a_refusal_keeps_the_sidecar_sentence() {
        let (base, _) = sidecar(None, Some((413, "audio must be 1 byte to 16 MiB")));
        let speech = Speech { base };
        assert_eq!(
            speech.transcribe(b"clip", "").unwrap_err(),
            "http 413: audio must be 1 byte to 16 MiB"
        );
    }

    #[test]
    fn a_ready_port_needs_no_spawn() {
        let (base, _) = sidecar(None, None);
        // No fennec-core checkout at all, and it still succeeds: the script
        // lookup is only reached when nothing holds the port.
        assert!(Speech { base }.ensure_in(&Suite::default()).is_ok());
    }

    #[test]
    fn a_missing_sidecar_script_says_which_path_it_looked_at() {
        let dir = std::env::temp_dir().join(format!("fennec-speech-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut suite = Suite::default();
        suite.repos.push(Repo { name: "fennec-core".into(), concept: String::new(), branch: String::new(), path: dir.clone(), upstream: false });
        let speech = Speech { base: "http://127.0.0.1:1".into() };
        let err = speech.ensure_in(&suite).unwrap_err();
        assert!(err.contains("no stt.py") && err.contains(&dir.display().to_string()), "{err}");
        suite.repos.clear();
        assert_eq!(speech.ensure_in(&suite).unwrap_err(), "no fennec-core checkout — install it or set FENNEC_HOME");
        std::fs::remove_dir_all(&dir).ok();
    }
}
