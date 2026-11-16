//! One HTTP/1.1 request at a time, `Connection: close`, body read by
//! Content-Length. The services keep the socket open after answering (uvicorn
//! honours `Connection: close` but not before the keep-alive timeout), so
//! reading to end would wait out the whole timeout on every request.
use serde_json::Value;
use std::io::{BufRead, Read};
use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

/// `/api/health` proxies the engine liveness probe, which itself allows 2 s
/// (`engine.py:164`), so a GET has to survive a full 2 s of server-side waiting.
pub const GET_TIMEOUT: Duration = Duration::from_millis(3000);
pub const POST_TIMEOUT: Duration = Duration::from_millis(4000);

pub fn split_base(base: &str) -> (String, u16) {
    let rest = base.trim_start_matches("http://").trim_start_matches("https://");
    let hostport = rest.split('/').next().unwrap_or(rest);
    match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
        None => (hostport.to_string(), 80),
    }
}

pub fn connect(base: &str, timeout: Duration) -> Result<TcpStream, String> {
    let (host, port) = split_base(base);
    let s = TcpStream::connect(format!("{host}:{port}")).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(timeout)).ok();
    s.set_write_timeout(Some(timeout)).ok();
    Ok(s)
}

/// Write `{method} {route}` plus the given headers and an optional JSON body.
/// Used directly by the SSE path, which keeps the socket open afterwards.
pub fn write_request(s: &mut TcpStream, host: &str, method: &str, route: &str, headers: &[(&str, &str)], body: Option<&str>) -> Result<(), String> {
    let mut req = format!("{method} {route} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    match body {
        Some(b) => req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len())),
        None => req.push_str("\r\n"),
    }
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())
}

/// Returns `(status, body)`; body is everything after the header block.
pub fn read_response(s: &mut TcpStream) -> Result<(u16, String), String> {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        if raw.len() > 1 << 20 {
            return Err("header block too large".into());
        }
        let n = s.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("closed before the header block ended".into());
        }
        raw.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&raw[..head_end]).into_owned();
    let status = head.lines().next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|x| x.parse().ok()).unwrap_or(0);
    let length = head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.splitn(2, ':').nth(1))
        .and_then(|v| v.trim().parse::<usize>().ok());
    let mut body = raw[head_end..].to_vec();
    match length {
        Some(len) => {
            while body.len() < len {
                let n = s.read(&mut chunk).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..n]);
            }
            body.truncate(len);
        }
        // No framing length: the answer is whatever arrives before the close.
        None => {
            s.read_to_end(&mut body).map_err(|e| e.to_string())?;
        }
    }
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

pub fn get_json(base: &str, route: &str, headers: &[(&str, &str)]) -> Result<Value, String> {
    let (host, _) = split_base(base);
    let mut s = connect(base, GET_TIMEOUT)?;
    write_request(&mut s, &host, "GET", route, headers, None)?;
    let (status, body) = read_response(&mut s)?;
    if status != 200 {
        return Err(format!("http {status}"));
    }
    serde_json::from_str(&body).map_err(|e| format!("json: {e}"))
}

pub fn post_json(base: &str, route: &str, headers: &[(&str, &str)], body: &Value) -> Result<Value, String> {
    let (host, _) = split_base(base);
    let mut s = connect(base, POST_TIMEOUT)?;
    write_request(&mut s, &host, "POST", route, headers, Some(&body.to_string()))?;
    let (status, text) = read_response(&mut s)?;
    if status != 200 {
        return Err(format!("http {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("json: {e}"))
}

pub fn bearer(key: &str) -> String {
    format!("Bearer {key}")
}

/// Read an SSE stream from `route`, calling `on_event` for each `data:` line.
/// Blocks the caller's thread until the socket closes or `cancel` flips.
/// Used by the coding-agent's `/api/events` endpoint.
pub fn read_sse(
    base: &str,
    route: &str,
    headers: &[(&str, &str)],
    cancel: &std::sync::atomic::AtomicBool,
    mut on_event: impl FnMut(&str),
) -> Result<(), String> {
    let (host, _) = split_base(base);
    let mut s = connect(base, std::time::Duration::from_secs(300))?;
    write_request(&mut s, &host, "GET", route, headers, None)?;
    let mut reader = std::io::BufReader::new(s);
    let mut line = String::new();
    // Skip HTTP headers
    loop {
        line.clear();
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Err("closed before headers ended".into());
        }
        if line.trim().is_empty() {
            break;
        }
    }
    // Read SSE frames
    loop {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            break; // socket closed
        }
        if let Some(data) = line.trim().strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            on_event(data.trim());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::time::Instant;

    /// Answers one request, then holds the connection open for `hold` without closing it —
    /// which is what uvicorn does, keep-alive notwithstanding our `Connection: close`.
    fn serve(response: String, hold: Duration) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut line = String::new();
            let mut reader = BufReader::new(s.try_clone().unwrap());
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                    break;
                }
            }
            s.write_all(response.as_bytes()).ok();
            std::thread::sleep(hold);
        });
        format!("http://127.0.0.1:{port}")
    }

    fn ok(body: &str) -> String {
        format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len())
    }

    #[test]
    fn a_keep_alive_response_is_read_from_its_content_length_not_its_close() {
        let base = serve(ok(r#"{"app":"Fennec"}"#), Duration::from_millis(2000));
        let t = Instant::now();
        let v = get_json(&base, "/api/health", &[("X-Fennec-Client", "1")]).unwrap();
        assert_eq!(v["app"], "Fennec");
        assert!(t.elapsed() < Duration::from_millis(1500), "waited {:?} for a close that never came", t.elapsed());
    }

    #[test]
    fn a_body_split_across_reads_is_reassembled() {
        let body = r#"{"sessions":[{"id":"a"},{"id":"b"}]}"#;
        let head = format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n{}", body.len(), &body[..12]);
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let rest = body[12..].to_string();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            s.read(&mut buf).ok();
            s.write_all(head.as_bytes()).ok();
            s.flush().ok();
            std::thread::sleep(Duration::from_millis(120));
            s.write_all(rest.as_bytes()).ok();
            std::thread::sleep(Duration::from_millis(300));
        });
        let v = get_json(&format!("http://127.0.0.1:{port}"), "/api/agent/sessions", &[]).unwrap();
        assert_eq!(v["sessions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_non_200_is_reported_with_its_status_and_body() {
        let base = serve("HTTP/1.1 403 Forbidden\r\ncontent-length: 2\r\n\r\nno".to_string(), Duration::from_millis(200));
        assert_eq!(get_json(&base, "/api/outlets", &[]).unwrap_err(), "http 403");
    }

    #[test]
    fn split_base_reads_host_and_port() {
        assert_eq!(split_base("http://127.0.0.1:8787"), ("127.0.0.1".into(), 8787));
        assert_eq!(split_base("http://127.0.0.1:8184/v1/chat"), ("127.0.0.1".into(), 8184));
    }
}
