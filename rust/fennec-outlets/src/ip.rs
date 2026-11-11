//! Address literals, exactly as CPython's `ipaddress` accepts and classifies
//! them. The outlet URL gate is a security boundary, so the accept/reject rules
//! and every predicate are ported from the stdlib source rather than from a
//! summary: `rust/goldens/ip-parse.jsonl` and `ip-classify.jsonl` are generated
//! by CPython itself and the test suite fails on any disagreement.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ip {
    V4(u32),
    V6(u128),
}

/// `IPv6Address._constants._reserved_networks` as (prefix length, prefix bits).
const RESERVED_V6: [(u32, u128); 15] = [
    (8, 0x00), (8, 0x01), (7, 0x02), (6, 0x04), (5, 0x10), (4, 0x01), (3, 0x02), (3, 0x03),
    (3, 0x04), (3, 0x05), (3, 0x06), (4, 0x0E), (5, 0x1E), (6, 0x3E), (9, 0x1FC),
];

fn prefix_matches(value: u128, bits: u128, len: u32) -> bool {
    value >> (128 - len) == bits
}

/// `IPv6Address.ipv4_mapped`: bits 32..48 must be `0xffff`, everything above zero.
fn mapped_v6(raw: u128) -> Option<u32> {
    if raw >> 32 != 0xFFFF {
        return None;
    }
    Some((raw & 0xFFFF_FFFF) as u32)
}

impl Ip {
    /// `ipaddress.ip_address(text)`: v4 first, then v6, `None` for a ValueError.
    pub fn parse(text: &str) -> Option<Ip> {
        parse_v4(text).map(Ip::V4).or_else(|| parse_v6(text).map(Ip::V6))
    }

    pub fn version(&self) -> u8 {
        match self {
            Ip::V4(_) => 4,
            Ip::V6(_) => 6,
        }
    }

    pub fn packed(&self) -> Vec<u8> {
        match self {
            Ip::V4(raw) => raw.to_be_bytes().to_vec(),
            Ip::V6(raw) => raw.to_be_bytes().to_vec(),
        }
    }

    /// `is_loopback`: `127.0.0.0/8` for v4, `::1` for v6, mapped v6 defers to v4.
    pub fn is_loopback(&self) -> bool {
        match self {
            Ip::V4(raw) => raw >> 24 == 127,
            Ip::V6(raw) => match mapped_v6(*raw) {
                Some(inner) => Ip::V4(inner).is_loopback(),
                None => *raw == 1,
            },
        }
    }

    pub fn is_link_local(&self) -> bool {
        match self {
            Ip::V4(raw) => raw >> 16 == 0xA9FE,
            Ip::V6(raw) => match mapped_v6(*raw) {
                Some(inner) => Ip::V4(inner).is_link_local(),
                None => prefix_matches(*raw, 0x3FA, 10),
            },
        }
    }

    pub fn is_multicast(&self) -> bool {
        match self {
            Ip::V4(raw) => raw >> 28 == 0xE,
            Ip::V6(raw) => match mapped_v6(*raw) {
                Some(inner) => Ip::V4(inner).is_multicast(),
                None => raw >> 120 == 0xFF,
            },
        }
    }

    pub fn is_unspecified(&self) -> bool {
        match self {
            Ip::V4(raw) => *raw == 0,
            Ip::V6(raw) => match mapped_v6(*raw) {
                Some(inner) => Ip::V4(inner).is_unspecified(),
                None => *raw == 0,
            },
        }
    }

    /// `is_reserved`: `240.0.0.0/4` for v4; for v6 the IETF reserved ranges, with
    /// an IPv4-mapped address deferring to the v4 rule.
    pub fn is_reserved(&self) -> bool {
        match self {
            Ip::V4(raw) => raw >> 28 == 0xF,
            Ip::V6(raw) => match mapped_v6(*raw) {
                Some(inner) => Ip::V4(inner).is_reserved(),
                None => RESERVED_V6.iter().any(|(len, bits)| prefix_matches(*raw, *bits, *len)),
            },
        }
    }
}

/// Four dot-separated decimal octets, no leading zeros, each 0..=255, and no
/// sign, space, hex prefix or prefix length.
fn parse_v4(text: &str) -> Option<u32> {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut raw = 0u32;
    for part in parts {
        if part.is_empty() || part.len() > 3 {
            return None;
        }
        if part.len() > 1 && part.starts_with('0') {
            return None;
        }
        if !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let value: u32 = part.parse().ok()?;
        if value > 255 {
            return None;
        }
        raw = (raw << 8) | value;
    }
    Some(raw)
}

fn hextet(text: &str) -> Option<u16> {
    if text.is_empty() || text.len() > 4 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u16::from_str_radix(text, 16).ok()
}

/// One side of a `::`: colon-separated hextets, the last of which may be a
/// dotted quad worth two of them. An empty side is zero components.
fn side(text: &str, allow_quad_at_end: bool) -> Option<Vec<u16>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    let parts: Vec<&str> = text.split(':').collect();
    let mut words: Vec<u16> = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            return None;
        }
        if part.contains('.') {
            if !allow_quad_at_end || index + 1 != parts.len() || words.len() + 2 > 8 {
                return None;
            }
            let quad = parse_v4(part)?;
            words.push((quad >> 16) as u16);
            words.push((quad & 0xFFFF) as u16);
            continue;
        }
        words.push(hextet(part)?);
    }
    Some(words)
}

/// A v6 literal, following `_BaseV6._split_scope_id` and the string parser: at
/// most one `::`, 1..=4 hex digits per hextet, an optional dotted quad only as
/// the final component, and an optional `%scope` that may not be empty or hold
/// a second `%`.
fn parse_v6(text: &str) -> Option<u128> {
    if text.contains('/') {
        return None;
    }
    let body = match text.split_once('%') {
        Some((addr, scope)) => {
            if scope.is_empty() || scope.contains('%') {
                return None;
            }
            addr
        }
        None => text,
    };
    if body.is_empty() || body.starts_with(':') && !body.starts_with("::")
        || body.ends_with(':') && !body.ends_with("::") {
        return None;
    }
    let (head_str, tail_str, compressed) = match body.find("::") {
        Some(index) => {
            let tail = &body[index + 2..];
            if tail.contains(':') && tail.starts_with(':') {
                return None;
            }
            (&body[..index], tail, true)
        }
        None => (body, "", false),
    };
    // The quad belongs to the address as a whole, so only the last side may end
    // with one.
    let head = side(head_str, !compressed && tail_str.is_empty())?;
    let tail = side(tail_str, true)?;
    let total = head.len() + tail.len();
    let words = if compressed {
        if total > 7 {
            return None;
        }
        let mut words = head;
        words.resize(8 - tail.len(), 0);
        words.extend(tail);
        words
    } else {
        if total != 8 {
            return None;
        }
        head
    };
    Some(words.into_iter().fold(0u128, |raw, word| (raw << 16) | word as u128))
}
