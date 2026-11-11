//! `urllib.parse.urlsplit` for the only part the outlet gate reads: the host.
//!
//! Ported from the stdlib, including the bracket validation that makes
//! `http://[127.0.0.1]/` and `http://[::1]noslash` raise rather than yield a
//! host, and the lowercasing of scheme and hostname. `Err(())` stands for the
//! ValueError, which `crate::gate` turns into a refusal instead of a crash.

use crate::ip::Ip;

const SCHEME_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

/// `(before, sep_present, after)` - Python's `str.partition`.
fn partition(text: &str, needle: char) -> (&str, bool, &str) {
    match text.find(needle) {
        Some(index) => (&text[..index], true, &text[index + needle.len_utf8()..]),
        None => (text, false, ""),
    }
}

/// Python's `str.rpartition`.
fn rpartition(text: &str, needle: char) -> (&str, bool, &str) {
    match text.rfind(needle) {
        Some(index) => (&text[..index], true, &text[index + needle.len_utf8()..]),
        None => ("", false, text),
    }
}

impl Split {
    /// `parts.hostname`: host with userinfo and port removed, brackets dropped,
    /// lowercased; `None` when empty.
    pub fn hostname(&self) -> Option<String> {
        let (_, _, hostinfo) = rpartition(&self.netloc, '@');
        let (head, bracketed) = match hostinfo.find('[') {
            Some(index) => {
                let inside = &hostinfo[index + 1..];
                (partition(inside, ']').0, true)
            }
            None => (partition(hostinfo, ':').0, false),
        };
        let _ = bracketed;
        if head.is_empty() {
            return None;
        }
        Some(head.to_lowercase())
    }

    /// `parts.username`: the userinfo before the first ':', `None` when empty.
    pub fn username(&self) -> Option<&str> {
        let (userinfo, present, _) = rpartition(&self.netloc, '@');
        if !present {
            return None;
        }
        let username = partition(userinfo, ':').0;
        if username.is_empty() { None } else { Some(username) }
    }

    /// `parts.password`: the userinfo after the first ':' - which may be empty,
    /// exactly as in Python, so `":@host"` has a password of `""`.
    pub fn password(&self) -> Option<&str> {
        let (userinfo, present, _) = rpartition(&self.netloc, '@');
        if !present {
            return None;
        }
        let (_, has_colon, password) = partition(userinfo, ':');
        if has_colon { Some(password) } else { None }
    }
}

/// `\Av[a-fA-F0-9]+\..+\Z` - the IPvFuture form urllib accepts inside brackets.
fn ipv_future_ok(text: &str) -> bool {
    let rest = &text[1..];
    match rest.find('.') {
        None => false,
        Some(index) => {
            index > 0
                && rest[..index].bytes().all(|byte| byte.is_ascii_hexdigit())
                && rest.len() > index + 1
        }
    }
}

fn check_bracketed_host(hostname: &str) -> Result<(), ()> {
    if hostname.starts_with('v') {
        return if ipv_future_ok(hostname) { Ok(()) } else { Err(()) };
    }
    match Ip::parse(hostname) {
        // "An IPv4 address cannot be in brackets".
        Some(Ip::V6(_)) => Ok(()),
        _ => Err(()),
    }
}

fn check_bracketed_netloc(netloc: &str) -> Result<(), ()> {
    let (_, _, hostname_and_port) = rpartition(netloc, '@');
    let (before_bracket, have_open_br, bracketed) = partition(hostname_and_port, '[');
    if have_open_br {
        if !before_bracket.is_empty() {
            return Err(());
        }
        let (hostname, _, port) = partition(bracketed, ']');
        if !port.is_empty() && !port.starts_with(':') {
            return Err(());
        }
        return check_bracketed_host(hostname);
    }
    let (hostname, _, _) = partition(bracketed, ':');
    check_bracketed_host(hostname)
}

pub fn urlsplit(url: &str) -> Result<Split, ()> {
    let stripped = url.trim_start_matches(|ch| ch <= '\u{1f}' || ch == ' ');
    let mut text: String = String::new();
    for ch in stripped.chars() {
        if ch != '\t' && ch != '\r' && ch != '\n' {
            text.push(ch);
        }
    }
    let mut scheme = String::new();
    if let Some(index) = text.find(':') {
        if index > 0 {
            let prefix = &text[..index];
            let first = prefix.as_bytes()[0];
            if first.is_ascii_alphabetic()
                && prefix.bytes().all(|byte| SCHEME_CHARS.contains(byte as char))
            {
                scheme = prefix.to_lowercase();
                text = text[index + 1..].to_string();
            }
        }
    }
    let mut netloc = String::new();
    if text.starts_with("//") {
        let rest = &text[2..];
        let end = ['/', '?', '#']
            .iter()
            .filter_map(|needle| rest.find(*needle))
            .min()
            .unwrap_or(rest.len());
        netloc = rest[..end].to_string();
        text = rest[end..].to_string();
        if netloc.contains('[') != netloc.contains(']') {
            return Err(());
        }
        if netloc.contains('[') && netloc.contains(']') {
            check_bracketed_netloc(&netloc)?;
        }
    }
    let (text, fragment) = match text.find('#') {
        Some(index) => (text[..index].to_string(), text[index + 1..].to_string()),
        None => (text, String::new()),
    };
    let (path, query) = match text.find('?') {
        Some(index) => (text[..index].to_string(), text[index + 1..].to_string()),
        None => (text, String::new()),
    };
    Ok(Split { scheme, netloc, path, query, fragment })
}
