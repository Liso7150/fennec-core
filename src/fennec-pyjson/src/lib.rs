//! Byte-exact clone of `json.dumps(value, ensure_ascii=False)`: the separators
//! Python uses when not indenting (`", "` and `": "`), its escaping table and its
//! float repr. `serde_json` writes `{"a":1}` instead, and character counts are
//! part of the contract here - `fennec-economy` measures the payload with this,
//! so a port that guessed would silently move the elision budget.

use std::fmt::Write as _;

/// Re-exported so consumers write with the same `Value` and `Map` this does:
/// one serde_json in play across the workspace.
pub use serde_json::{Map, Value};

pub fn dumps(value: &Value) -> String {
    let mut out = String::with_capacity(64);
    write_value(value, &mut out, ", ", ": ");
    out
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))` - what
/// `acp.Agent._write` actually puts on the wire.
pub fn dumps_compact(value: &Value) -> String {
    let mut out = String::with_capacity(64);
    write_value(value, &mut out, ",", ":");
    out
}

/// `len(json.dumps(value, ensure_ascii=False))` in code points, like `len()`.
pub fn char_len(value: &Value) -> usize {
    dumps(value).chars().count()
}

pub fn object() -> Map<String, Value> {
    Map::new()
}

fn write_value(value: &Value, out: &mut String, item_sep: &str, key_sep: &str) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&dumps_number(number)),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_value(item, out, item_sep, key_sep);
            }
            out.push(']');
        }
        Value::Object(object) => {
            out.push('{');
            for (index, (key, item)) in object.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_string(key, out);
                out.push_str(key_sep);
                write_value(item, out, item_sep, key_sep);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn dumps_number(number: &serde_json::Number) -> String {
    if let Some(int) = number.as_i64() {
        return int.to_string();
    }
    if let Some(unsigned) = number.as_u64() {
        return unsigned.to_string();
    }
    float_repr(number.as_f64().unwrap_or(0.0))
}

/// `repr(float)`: shortest round-trip digits, exponent form once the decimal
/// point leaves `(-4, 17]`, always a fractional part in fixed form, at least two
/// exponent digits.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }

    let sign = if value < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("LowerExp carries an exponent");
    let exp: i32 = exponent.parse().expect("exponent digits");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let fractional = mantissa.split_once('.').map(|(_, rest)| rest).unwrap_or("");
    let decpt = exp + 1;

    if decpt <= -4 || decpt > 16 {
        let mut out = format!("{sign}{}", &digits[..1]);
        if !fractional.is_empty() {
            let _ = write!(out, ".{fractional}");
        }
        let exp_sign = if exp < 0 { '-' } else { '+' };
        let _ = write!(out, "e{exp_sign}{:02}", exp.abs());
        return out;
    }
    if decpt <= 0 {
        return format!("{sign}0.{}{digits}", "0".repeat(-decpt as usize));
    }
    if digits.len() <= decpt as usize {
        return format!("{sign}{digits}{}.0", "0".repeat(decpt as usize - digits.len()));
    }
    let split = decpt as usize;
    format!("{sign}{}.{}", &digits[..split], &digits[split..])
}
