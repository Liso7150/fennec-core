//! Node-API binding for the context economy, loadable by Bun and Electron's Node.
//!
//! There is no `napi-rs` in the offline crate cache and no `node.lib` or Node
//! headers on this box, so the ABI is declared by hand and every `napi_*` symbol
//! is resolved at load time out of the host process. Both runtimes export them,
//! which is what lets one `.node` binary serve both.
//!
//! Semantics are `fennec_core/context.py`'s, not the old `context.js`'s: sizes
//! count Unicode code points and `tool_calls` is measured with `fennec-pyjson`,
//! the byte-exact `json.dumps(ensure_ascii=False)` clone. The JS twin counted
//! UTF-16 units and compact `JSON.stringify`, which disagreed with the core by
//! 31% on astral-plane text.
//!
//! Messages cross as objects, not as JSON: the caller shallow-copies them, this
//! module reads them into `serde_json::Value`, runs the same `compact` the
//! Python golden tests cover, and writes back only the `content` fields that
//! changed. No pipe, no framing, no re-serialization of the payload.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

use fennec_economy::{self as economy, Options, Stats};
use serde_json::{Map, Value};

type Env = *mut c_void;
type Val = *mut c_void;
type Status = i32;
type CbInfo = *mut c_void;
type Callback = unsafe extern "system" fn(Env, CbInfo) -> Val;
type Finalize = unsafe extern "system" fn(Env, *mut c_void, *mut c_void);

const OK: Status = 0;
/// Tool results are external data; a hostile nesting depth must not become a
/// stack overflow inside the host process.
const MAX_DEPTH: usize = 64;

#[cfg(windows)]
unsafe fn host() -> Option<libloading::os::windows::Library> {
    libloading::os::windows::Library::this().ok()
}

#[cfg(not(windows))]
unsafe fn host() -> Option<libloading::os::unix::Library> {
    Some(libloading::os::unix::Library::this())
}

macro_rules! api {
    ($($name:ident : unsafe extern "system" fn($($arg:ty),*) -> $ret:ty),* $(,)?) => {
        struct Api { $($name: unsafe extern "system" fn($($arg),*) -> $ret),* }
        static API: OnceLock<Option<Api>> = OnceLock::new();
        fn api() -> Option<&'static Api> {
            API.get_or_init(|| unsafe {
                // The host executable owns the symbols, so this handle has to
                // outlive every call: it is deliberately never dropped.
                let lib = std::mem::ManuallyDrop::new(host()?);
                $(let $name = *lib.get(stringify!($name).as_bytes()).ok()?;)*
                Some(Api { $($name),* })
            }).as_ref()
        }
    };
}

api! {
    napi_get_undefined: unsafe extern "system" fn(Env, *mut Val) -> Status,
    napi_create_object: unsafe extern "system" fn(Env, *mut Val) -> Status,
    napi_create_string_utf8: unsafe extern "system" fn(Env, *const u8, usize, *mut Val) -> Status,
    napi_get_value_string_utf8: unsafe extern "system" fn(Env, Val, *mut u8, usize, *mut usize) -> Status,
    napi_create_double: unsafe extern "system" fn(Env, f64, *mut Val) -> Status,
    napi_get_value_double: unsafe extern "system" fn(Env, Val, *mut f64) -> Status,
    napi_get_value_bool: unsafe extern "system" fn(Env, Val, *mut bool) -> Status,
    napi_create_function: unsafe extern "system" fn(Env, *const u8, usize, Callback, *mut c_void, *mut Val) -> Status,
    napi_set_named_property: unsafe extern "system" fn(Env, Val, *const u8, Val) -> Status,
    napi_get_named_property: unsafe extern "system" fn(Env, Val, *const u8, *mut Val) -> Status,
    napi_get_property: unsafe extern "system" fn(Env, Val, Val, *mut Val) -> Status,
    napi_get_property_names: unsafe extern "system" fn(Env, Val, *mut Val) -> Status,
    napi_get_cb_info: unsafe extern "system" fn(Env, CbInfo, *mut usize, *mut Val, *mut Val, *mut *mut c_void) -> Status,
    napi_typeof: unsafe extern "system" fn(Env, Val, *mut i32) -> Status,
    napi_is_array: unsafe extern "system" fn(Env, Val, *mut bool) -> Status,
    napi_get_array_length: unsafe extern "system" fn(Env, Val, *mut u32) -> Status,
    napi_get_element: unsafe extern "system" fn(Env, Val, u32, *mut Val) -> Status,
    napi_create_external: unsafe extern "system" fn(Env, *mut c_void, Option<Finalize>, *mut c_void, *mut Val) -> Status,
    napi_get_value_external: unsafe extern "system" fn(Env, Val, *mut *mut c_void) -> Status,
    napi_throw_error: unsafe extern "system" fn(Env, *const u8, *const u8) -> Status,
}

/// Only reachable from a callback, and callbacks are only installed once every
/// symbol resolved, so the table is present by construction.
fn table() -> &'static Api {
    api().expect("napi symbols resolved at registration")
}

unsafe fn undefined(env: Env) -> Val {
    let mut out: Val = std::ptr::null_mut();
    (table().napi_get_undefined)(env, &mut out);
    out
}

unsafe fn string(env: Env, s: &str) -> Val {
    let mut out: Val = std::ptr::null_mut();
    (table().napi_create_string_utf8)(env, s.as_ptr(), s.len(), &mut out);
    out
}

unsafe fn number(env: Env, v: f64) -> Val {
    let mut out: Val = std::ptr::null_mut();
    (table().napi_create_double)(env, v, &mut out);
    out
}

unsafe fn object(env: Env) -> Val {
    let mut out: Val = std::ptr::null_mut();
    (table().napi_create_object)(env, &mut out);
    out
}

unsafe fn set(env: Env, target: Val, name: &str, value: Val) -> bool {
    let key = CString::new(name).expect("static property name");
    (table().napi_set_named_property)(env, target, key.as_ptr() as *const u8, value) == OK
}

unsafe fn get(env: Env, target: Val, name: &str) -> Option<Val> {
    let key = CString::new(name).expect("static property name");
    let mut out: Val = std::ptr::null_mut();
    if (table().napi_get_named_property)(env, target, key.as_ptr() as *const u8, &mut out) != OK {
        return None;
    }
    Some(out)
}

unsafe fn throw(env: Env, msg: &str) -> Val {
    let text = CString::new(msg).unwrap_or_else(|_| CString::new("invalid argument").unwrap());
    (table().napi_throw_error)(env, std::ptr::null(), text.as_ptr() as *const u8);
    undefined(env)
}

unsafe fn kind(env: Env, v: Val) -> i32 {
    let mut t = -1i32;
    (table().napi_typeof)(env, v, &mut t);
    t
}

unsafe fn is_array(env: Env, v: Val) -> bool {
    let mut flag = false;
    (table().napi_is_array)(env, v, &mut flag) == OK && flag
}

unsafe fn read_string(env: Env, v: Val) -> Option<String> {
    if kind(env, v) != 4 {
        return None;
    }
    let a = table();
    let mut len = 0usize;
    if (a.napi_get_value_string_utf8)(env, v, std::ptr::null_mut(), 0, &mut len) != OK {
        return None;
    }
    let mut buf = vec![0u8; len + 1];
    let mut written = 0usize;
    if (a.napi_get_value_string_utf8)(env, v, buf.as_mut_ptr(), buf.len(), &mut written) != OK {
        return None;
    }
    buf.truncate(written);
    String::from_utf8(buf).ok()
}

unsafe fn read_f64(env: Env, v: Val) -> Option<f64> {
    if kind(env, v) != 3 {
        return None;
    }
    let mut out = 0.0;
    if (table().napi_get_value_double)(env, v, &mut out) == OK {
        Some(out)
    } else {
        None
    }
}

unsafe fn read_usize(env: Env, v: Val) -> Option<usize> {
    let n = read_f64(env, v)?;
    if n.is_finite() && n >= 0.0 && n <= usize::MAX as f64 {
        Some(n as usize)
    } else {
        None
    }
}

/// JS value -> `serde_json::Value`. `undefined` becomes `None` so the key is
/// dropped, which is what `JSON.stringify` does with it; every other scalar maps
/// to its JSON counterpart.
unsafe fn to_value(env: Env, v: Val, depth: usize) -> Option<Value> {
    if depth > MAX_DEPTH {
        return Some(Value::Null);
    }
    let a = table();
    match kind(env, v) {
        1 => Some(Value::Null),
        2 => {
            let mut flag = false;
            if (a.napi_get_value_bool)(env, v, &mut flag) != OK {
                return None;
            }
            Some(Value::Bool(flag))
        }
        3 => {
            let n = read_f64(env, v)?;
            if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
                Some(Value::from(n as i64))
            } else {
                serde_json::Number::from_f64(n).map(Value::Number)
            }
        }
        4 => read_string(env, v).map(Value::String),
        6 => {
            if is_array(env, v) {
                let mut len = 0u32;
                if (a.napi_get_array_length)(env, v, &mut len) != OK {
                    return None;
                }
                let mut items = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let mut item: Val = std::ptr::null_mut();
                    if (a.napi_get_element)(env, v, i, &mut item) != OK {
                        return None;
                    }
                    items.push(to_value(env, item, depth + 1).unwrap_or(Value::Null));
                }
                Some(Value::Array(items))
            } else {
                let mut names: Val = std::ptr::null_mut();
                if (a.napi_get_property_names)(env, v, &mut names) != OK {
                    return None;
                }
                let mut len = 0u32;
                if (a.napi_get_array_length)(env, names, &mut len) != OK {
                    return None;
                }
                let mut map = Map::new();
                for i in 0..len {
                    let mut key: Val = std::ptr::null_mut();
                    if (a.napi_get_element)(env, names, i, &mut key) != OK {
                        continue;
                    }
                    let Some(name) = read_string(env, key) else { continue };
                    let mut field: Val = std::ptr::null_mut();
                    if (a.napi_get_property)(env, v, key, &mut field) != OK {
                        continue;
                    }
                    if let Some(value) = to_value(env, field, depth + 1) {
                        map.insert(name, value);
                    }
                }
                Some(Value::Object(map))
            }
        }
        _ => None,
    }
}

unsafe fn args(env: Env, info: CbInfo, want: usize) -> Option<Vec<Val>> {
    let mut argv = vec![std::ptr::null_mut::<c_void>(); want];
    let mut argc = want;
    if (table().napi_get_cb_info)(env, info, &mut argc, argv.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut()) != OK {
        return None;
    }
    if argc < want {
        return None;
    }
    Some(argv)
}

type Seen = Mutex<HashMap<String, String>>;

unsafe extern "system" fn drop_seen(_env: Env, data: *mut c_void, _hint: *mut c_void) {
    if !data.is_null() {
        drop(Box::from_raw(data as *mut Seen));
    }
}

unsafe fn seen_of(env: Env, v: Val) -> Option<&'static Seen> {
    let mut data: *mut c_void = std::ptr::null_mut();
    if (table().napi_get_value_external)(env, v, &mut data) != OK || data.is_null() {
        return None;
    }
    // The external owns the box for as long as the JS handle is reachable, and
    // the finalizer is the only thing that frees it.
    Some(&*(data as *const Seen))
}

unsafe extern "system" fn create_seen(env: Env, _info: CbInfo) -> Val {
    let boxed: *mut Seen = Box::into_raw(Box::new(Mutex::new(HashMap::new())));
    let mut out: Val = std::ptr::null_mut();
    if (table().napi_create_external)(env, boxed as *mut c_void, Some(drop_seen), std::ptr::null_mut(), &mut out) != OK {
        drop(Box::from_raw(boxed));
        return throw(env, "could not allocate the turn state");
    }
    out
}

unsafe extern "system" fn reset_seen(env: Env, info: CbInfo) -> Val {
    let Some(argv) = args(env, info, 1) else { return throw(env, "resetTurn needs the turn state") };
    let Some(seen) = seen_of(env, argv[0]) else { return throw(env, "resetTurn needs a state from createTurnCompactor()") };
    if let Ok(mut guard) = seen.lock() {
        guard.clear();
    }
    undefined(env)
}

unsafe fn stats_object(env: Env, s: Stats) -> Val {
    let out = object(env);
    for (name, value) in [
        ("charsBefore", s.chars_before),
        ("charsAfter", s.chars_after),
        ("elided", s.elided),
        ("deduped", s.deduped),
        ("budgetChars", s.budget_chars),
        ("resultChars", s.result_chars),
    ] {
        set(env, out, name, number(env, value as f64));
    }
    out
}

/// The economy reads exactly these four keys. Enumerating every property costs
/// two Node-API calls per field for values compaction never looks at, so only
/// these cross the boundary; the caller's other keys survive untouched because
/// only `content` is ever written back.
const MESSAGE_KEYS: [&str; 4] = ["role", "content", "tool_call_id", "tool_calls"];

unsafe fn message_value(env: Env, v: Val) -> Value {
    let mut out = serde_json::Map::new();
    for key in MESSAGE_KEYS {
        if let Some(value) = get(env, v, key).and_then(|field| to_value(env, field, 1)) {
            out.insert(key.to_string(), value);
        }
    }
    Value::Object(out)
}

/// `compact(messages, options, seen)` -> stats. `messages` must already be the
/// caller's shallow copies: only `content` is written, every other key and value
/// is left exactly as the caller built it.
unsafe extern "system" fn compact(env: Env, info: CbInfo) -> Val {
    let Some(argv) = args(env, info, 3) else {
        return throw(env, "compact(messages, options, seen) needs three arguments");
    };
    if !is_array(env, argv[0]) {
        return throw(env, "compact() needs an array of messages");
    }
    let Some(seen) = seen_of(env, argv[2]) else {
        return throw(env, "compact() needs a state from createTurnCompactor()");
    };

    let limits = economy::limits();
    let mut options = Options {
        budget_chars: limits.budget_chars,
        result_chars: limits.result_chars,
        floor_chars: economy::MIN_RESULT_CHARS,
    };
    if kind(env, argv[1]) == 6 {
        if let Some(v) = get(env, argv[1], "budgetChars").and_then(|v| read_usize(env, v)) {
            options.budget_chars = v;
        }
        if let Some(v) = get(env, argv[1], "resultChars").and_then(|v| read_usize(env, v)) {
            options.result_chars = v;
        }
        if let Some(v) = get(env, argv[1], "floorChars").and_then(|v| read_usize(env, v)) {
            options.floor_chars = v;
        }
    }

    let a = table();
    let mut len = 0u32;
    if (a.napi_get_array_length)(env, argv[0], &mut len) != OK {
        return throw(env, "could not read the message array");
    }
    let mut elements = Vec::with_capacity(len as usize);
    let mut messages = Vec::with_capacity(len as usize);
    for i in 0..len {
        let mut item: Val = std::ptr::null_mut();
        if (a.napi_get_element)(env, argv[0], i, &mut item) != OK {
            return throw(env, "could not read a message");
        }
        elements.push(item);
        messages.push(message_value(env, item));
    }

    let (payload, stats) = match seen.lock() {
        Ok(mut guard) => economy::compact(&messages, &options, &mut guard),
        Err(_) => return throw(env, "the turn state is poisoned"),
    };

    for (i, element) in elements.iter().enumerate() {
        let before = messages[i].get("content").and_then(Value::as_str);
        let after = payload[i].get("content").and_then(Value::as_str);
        if let (Some(after), true) = (after, before != after) {
            set(env, *element, "content", string(env, after));
        }
    }
    stats_object(env, stats)
}

unsafe extern "system" fn size_of(env: Env, info: CbInfo) -> Val {
    let Some(argv) = args(env, info, 1) else { return throw(env, "size(message) needs one argument") };
    let value = message_value(env, argv[0]);
    number(env, economy::size(&value) as f64)
}

unsafe extern "system" fn elide_fn(env: Env, info: CbInfo) -> Val {
    let Some(argv) = args(env, info, 2) else { return throw(env, "elide(text, limit) needs two arguments") };
    let Some(text) = read_string(env, argv[0]) else { return argv[0] };
    let Some(limit) = read_usize(env, argv[1]) else { return throw(env, "elide() needs a non-negative limit") };
    string(env, &economy::elide(&text, limit))
}

unsafe extern "system" fn limits_fn(env: Env, _info: CbInfo) -> Val {
    let limits = economy::limits();
    let out = object(env);
    set(env, out, "budgetChars", number(env, limits.budget_chars as f64));
    set(env, out, "resultChars", number(env, limits.result_chars as f64));
    out
}

/// `limitsIn(env)` reads the two FENNEC_AGENT_* keys out of a JS object, so a
/// caller can pass `process.env` or a fixture without touching the real one.
unsafe extern "system" fn limits_in_fn(env: Env, info: CbInfo) -> Val {
    let Some(argv) = args(env, info, 1) else { return throw(env, "limitsIn(env) needs one argument") };
    let source = argv[0];
    let read = |key: &str| unsafe { get(env, source, key).and_then(|v| read_string(env, v)) };
    let limits = economy::limits_in(&read);
    let out = object(env);
    set(env, out, "budgetChars", number(env, limits.budget_chars as f64));
    set(env, out, "resultChars", number(env, limits.result_chars as f64));
    out
}

unsafe fn export(env: Env, exports: Val, name: &str, cb: Callback) -> bool {
    let cname = CString::new(name).expect("static export name");
    let mut f: Val = std::ptr::null_mut();
    if (table().napi_create_function)(env, cname.as_ptr() as *const u8, name.len(), cb, std::ptr::null_mut(), &mut f) != OK {
        return false;
    }
    set(env, exports, name, f)
}

#[no_mangle]
pub extern "system" fn napi_register_module_v1(env: Env, exports: Val) -> Val {
    unsafe {
        if api().is_none() {
            return exports;
        }
        for (name, cb) in [
            ("createSeen", create_seen as Callback),
            ("resetSeen", reset_seen as Callback),
            ("compact", compact as Callback),
            ("size", size_of as Callback),
            ("elide", elide_fn as Callback),
            ("limits", limits_fn as Callback),
            ("limitsIn", limits_in_fn as Callback),
        ] {
            if !export(env, exports, name, cb) {
                return exports;
            }
        }
        set(env, exports, "DEFAULT_BUDGET_CHARS", number(env, economy::DEFAULT_BUDGET_CHARS as f64));
        set(env, exports, "DEFAULT_RESULT_CHARS", number(env, economy::DEFAULT_RESULT_CHARS as f64));
        set(env, exports, "MIN_RESULT_CHARS", number(env, economy::MIN_RESULT_CHARS as f64));
        exports
    }
}