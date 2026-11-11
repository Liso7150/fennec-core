//! Rust port of the pure half of `fennec_core/outlets.py`: transcript redaction
//! and the address gate that decides which URL a delegated browser agent may be
//! pointed at. The HTTP and process plumbing that drives Jev and OpenBot stays
//! in Python - those are I/O, not math, and the suite never starts either peer
//! from a test.
//!
//! `rust/goldens/` holds CPython's own output for 2 000 transcript lines, every
//! reserved-address sample and 2 000 URLs; the tests fail on any drift.

pub mod ip;
pub mod url;

use ip::Ip;
use url::urlsplit;

/// `SECRET_ENV_KEYS + ("TYPESAFE_API_KEY",)` in the order Python walks it.
pub const REDACT_ENV_KEYS: [&str; 4] = [
    "FENNEC_API_KEY",
    "COLI_API_KEY",
    "FENNEC_APP_TOKEN",
    "TYPESAFE_API_KEY",
];

const BEARER_MIN: usize = 8;

fn bearer_class(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')
}

/// `re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]{8,}", r"\1[redacted]", text)`.
///
/// Greedy `\s+` cannot backtrack into the character class (no whitespace is in
/// it), so a single left-to-right scan is equivalent to the regex.
fn redact_bearer(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        if !starts_bearer(&chars, index) {
            out.push(chars[index]);
            index += 1;
            continue;
        }
        let mut walk = index + 6;
        let space_start = walk;
        while walk < chars.len() && chars[walk].is_whitespace() {
            walk += 1;
        }
        let spaces = walk - space_start;
        let token_start = walk;
        while walk < chars.len() && bearer_class(chars[walk]) {
            walk += 1;
        }
        if spaces >= 1 && walk - token_start >= BEARER_MIN {
            out.extend(&chars[index..token_start]);
            out.push_str("[redacted]");
            index = walk;
        } else {
            out.push(chars[index]);
            index += 1;
        }
    }
    out
}

fn starts_bearer(chars: &[char], index: usize) -> bool {
    const WORD: [char; 6] = ['b', 'e', 'a', 'r', 'e', 'r'];
    if index + WORD.len() > chars.len() {
        return false;
    }
    chars[index..index + WORD.len()]
        .iter()
        .map(|ch| ch.to_ascii_lowercase())
        .eq(WORD.iter().copied())
}

/// Drop Fennec's own credentials from anything that may reach a transcript,
/// then any bearer token. `environ` is injectable so a caller can redact a
/// foreign environment.
pub fn redact_with(text: &str, environ: &dyn Fn(&str) -> Option<String>) -> String {
    let mut current = text.to_string();
    for key in REDACT_ENV_KEYS {
        if let Some(secret) = environ(key) {
            if !secret.is_empty() {
                current = current.replace(&secret, "[redacted]");
            }
        }
    }
    redact_bearer(&current)
}

pub fn redact(text: &str) -> String {
    redact_with(text, &|key| std::env::var(key).ok())
}

/// `_loopback(origin)`: only a plain `http://` origin on this machine, and never
/// one carrying credentials.
pub fn loopback(origin: &str) -> bool {
    let parts = match urlsplit(origin) {
        Ok(parts) => parts,
        Err(()) => return false,
    };
    if parts.scheme != "http" {
        return false;
    }
    let host = match parts.hostname() {
        Some(host) => host,
        None => return false,
    };
    if parts.username().is_some() || parts.password().map_or(false, |value| !value.is_empty()) {
        // "http://user:pw@127.0.0.1" reaches 127.0.0.1 but carries a secret to it.
        return false;
    }
    if host == "localhost" {
        return true;
    }
    match Ip::parse(&host) {
        Some(address) => address.is_loopback(),
        None => false,
    }
}

/// `_blocked_host(url)`: empty means "allowed", anything else is the refusal a
/// caller returns instead of launching a browser.
pub fn blocked_host(url: &str) -> String {
    let parts = match urlsplit(url) {
        Ok(parts) => parts,
        Err(()) => {
            return "web_task will not browse that url: its host is not a parseable \
                    address"
                .to_string()
        }
    };
    let host = parts.hostname().unwrap_or_default();
    let address = match Ip::parse(&host) {
        Some(address) => address,
        None => return String::new(), // a name is resolved by the browser, and may well be the user's own
    };
    if address.is_link_local()
        || address.is_multicast()
        || address.is_reserved()
        || address.is_unspecified()
    {
        return format!("web_task will not browse {host}: it is a link-local or reserved address");
    }
    String::new()
}
