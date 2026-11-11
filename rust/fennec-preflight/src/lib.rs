//! Device-aware model preflight, ported from `fennec_core/preflight.py`: the
//! weights on disk plus the KV cache a session really needs, against this
//! machine. Read-only and offline - no model is loaded and no request is made.
//!
//! The two device probes Python makes through third-party helpers (`psutil` for
//! free RAM, `nvidia-smi` for VRAM) stay injectable: `plan` takes them, so the
//! math is what gets tested, and `rust/goldens/preflight-matrix.jsonl` pins it.

use serde_json::{Map, Number, Value};
use std::path::{Path, PathBuf};

/// BF16/FP16 key and value tensors; the engine keeps the cache at the model dtype.
pub const BYTES_PER_ELEMENT: i64 = 2;

fn expanduser(raw: &str) -> PathBuf {
    let text = raw.trim();
    if text == "~" {
        return home();
    }
    if let Some(rest) = text.strip_prefix("~/") {
        return home().join(rest);
    }
    if let Some(rest) = text.strip_prefix("~\\") {
        return home().join(rest);
    }
    PathBuf::from(text)
}

fn home() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn dir_bytes(path: &Path) -> i64 {
    let mut total = 0i64;
    let stack = &mut vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let child = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(child);
            } else if meta.is_file() {
                total += meta.len() as i64;
            }
        }
    }
    total
}

/// `shutil.disk_usage(path).free` - Windows only; `None` elsewhere, which is
/// what "no reading" already means for the rest of the report.
pub fn disk_free(path: &Path) -> Option<i64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        extern "system" {
            fn GetDiskFreeSpaceExW(directory_name: *const u16,
                                   free_bytes_available_to_caller: *mut u64,
                                   total_number_of_bytes: *mut u64,
                                   total_number_of_free_bytes: *mut u64) -> i32;
        }
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut available = 0u64;
        let mut total = 0u64;
        let mut free = 0u64;
        let ok = unsafe {
            GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, &mut free)
        };
        if ok == 0 {
            return None;
        }
        Some(available as i64)
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        None
    }
}

/// Python truthiness: `None`, `false`, `0`, `""`, `[]` and `{}` are false.
fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().map_or(true, |float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

fn as_int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(flag) => Some(*flag as i64),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                return Some(int);
            }
            number.as_f64().map(|float| float.trunc() as i64)
        }
        Value::String(text) => {
            let cleaned: String = text.trim().chars().filter(|ch| *ch != '_').collect();
            cleaned.parse::<i64>().ok()
        }
        _ => None,
    }
}

fn text_config(data: &Map<String, Value>) -> &Map<String, Value> {
    match data.get("text_config") {
        Some(Value::Object(nested)) => nested,
        _ => data,
    }
}

/// KV cache growth per token, from the model's own config, or `None`.
pub fn kv_bytes_per_token(model_dir: &Path) -> Option<i64> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let data: Value = serde_json::from_str(&raw).ok()?;
    let Value::Object(outer) = &data else { return None };
    let cfg = text_config(outer);
    let layers = as_int(cfg.get("num_hidden_layers")?)?;
    // `cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")`: the
    // first key wins unless it is falsy, which includes 0 and "".
    let heads = cfg
        .get("num_key_value_heads")
        .filter(|value| py_truthy(value))
        .or(cfg.get("num_attention_heads"))?;
    let heads = as_int(heads)?;
    let mut head_dim = cfg.get("head_dim").and_then(as_int);
    if head_dim.unwrap_or(0) == 0 {
        let hidden = cfg.get("hidden_size").and_then(as_int);
        let attention = cfg.get("num_attention_heads").and_then(as_int);
        if let (Some(hidden), Some(attention)) = (hidden, attention) {
            if attention > 0 {
                head_dim = Some(hidden / attention);
            }
        }
    }
    let elements = 2 * layers * heads * head_dim?;
    if elements > 0 {
        Some(elements * BYTES_PER_ELEMENT)
    } else {
        None
    }
}

fn optional(value: Option<i64>) -> Value {    value.map_or(Value::Null, |number| Value::Number(Number::from(number)))
}

/// The two device reads `plan` falls back to when a caller leaves a value out.
/// `BLIND` is a machine with no GPU and no `psutil`, which is what makes the
/// golden matrix comparable on any host.
#[derive(Clone, Copy)]
pub struct Probes {
    pub nvidia_vram: fn() -> Option<i64>,
    pub ram_available: fn() -> Option<i64>,
}

fn no_probe() -> Option<i64> {
    None
}

impl Probes {
    pub const SYSTEM: Probes = Probes { nvidia_vram, ram_available };
    pub const BLIND: Probes = Probes { nvidia_vram: no_probe, ram_available: no_probe };
}

/// `psutil.virtual_memory().available` - `GlobalMemoryStatusEx` reads the same
/// counter psutil uses on Windows.
pub fn ram_available() -> Option<i64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut status = unsafe { std::mem::zeroed::<MemoryStatusEx>() };
        status.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok == 0 {
            return None;
        }
        Some(status.ull_avail_phys as i64)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// What this model needs for a `ctx`-token session, and whether it fits. The
/// report keeps the Python key order, so a byte-for-byte comparison is possible.
/// A `None` device argument means "go and look", exactly as in Python.
pub fn plan(model_dir: &str, ctx: i64, kv_bytes: Option<i64>, vram_bytes: Option<i64>,
            ram_available_bytes: Option<i64>) -> Value {
    plan_with(model_dir, ctx, kv_bytes, vram_bytes, ram_available_bytes, Probes::SYSTEM)
}

pub fn plan_with(model_dir: &str, ctx: i64, kv_bytes: Option<i64>, vram_bytes: Option<i64>,
                 ram_available_bytes: Option<i64>, probes: Probes) -> Value {
    let path = expanduser(model_dir);
    let is_dir = path.is_dir();
    let weights = if is_dir { dir_bytes(&path) } else { 0 };
    let per_token = match kv_bytes {
        Some(value) => Some(value),
        None => kv_bytes_per_token(&path),
    };
    let kv = match per_token {
        Some(token) if token != 0 => Some(token * ctx),
        _ => None,
    };
    let vram = vram_bytes.or_else(probes.nvidia_vram);
    let ram = ram_available_bytes.or_else(probes.ram_available);
    let mut notes: Vec<String> = Vec::new();
    if weights <= 0 {
        notes.push("model directory is missing or empty".to_string());
    }
    if per_token.is_none() {
        notes.push("KV size unknown: config.json has no usable layer/head geometry".to_string());
    }
    if vram.is_none() {
        notes.push("no NVIDIA GPU reported; treating VRAM as zero".to_string());
    }

    let budget = vram.unwrap_or(0) + ram.unwrap_or(0);
    let fit = if weights <= 0 {
        "unknown"
    } else if vram.unwrap_or(0) != 0 && weights <= vram.unwrap_or(0) {
        "vram"
    } else if budget != 0 && weights <= budget {
        "hybrid"
    } else {
        "disk"
    };
    let kv_fits = match kv {
        None => None,
        Some(value) => Some(budget != 0 && value <= budget),
    };
    if kv_fits == Some(false) {
        notes.push("KV cache alone exceeds VRAM plus available RAM at this context".to_string());
    }
    let free = disk_free(if is_dir { &path } else { path.parent().unwrap_or(&path) });

    let mut report = Map::new();
    report.insert("model".to_string(), Value::String(path.to_string_lossy().into_owned()));
    report.insert("ctx".to_string(), Value::Number(Number::from(ctx)));
    report.insert("weights_bytes".to_string(), Value::Number(Number::from(weights)));
    report.insert("kv_bytes_per_token".to_string(), optional(per_token));
    report.insert("kv_bytes".to_string(), optional(kv));
    report.insert("peak_bytes".to_string(), optional(kv.map(|value| weights + value)));
    report.insert("vram_bytes".to_string(), optional(vram));
    report.insert("ram_available_bytes".to_string(), optional(ram));
    report.insert("disk_free_bytes".to_string(), optional(free));
    report.insert("weights_fit".to_string(), Value::String(fit.to_string()));
    report.insert("kv_fits".to_string(), match kv_fits {
        None => Value::Null,
        Some(flag) => Value::Bool(flag),
    });
    report.insert("notes".to_string(), Value::Array(
        notes.into_iter().map(Value::String).collect()));
    Value::Object(report)
}

/// One line per fact, for a terminal - the same wording as `preflight.human`.
pub fn human(report: &Value) -> Vec<String> {
    fn gb(value: Option<i64>) -> String {
        match value {
            None => "unknown".to_string(),
            Some(bytes) => format!("{:.2} GiB", bytes as f64 / 1024f64.powi(3)),
        }
    }
    let field = |key: &str| report.get(key).and_then(Value::as_i64);
    let text = |key: &str| report.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    let mut lines = vec![
        format!("model        {}", text("model")),
        format!("weights      {}", gb(field("weights_bytes"))),
        format!("context      {} tokens", report.get("ctx").and_then(Value::as_i64).unwrap_or(0)),
        format!("KV per token {}", match field("kv_bytes_per_token") {
            Some(value) if value != 0 => gb(Some(value)),
            _ => "unknown".to_string(),
        }),
        format!("KV at ctx    {}", gb(field("kv_bytes"))),
        format!("peak         {}", gb(field("peak_bytes"))),
        format!("VRAM         {}", gb(field("vram_bytes"))),
        format!("RAM free     {}", gb(field("ram_available_bytes"))),
        format!("weights fit  {}", text("weights_fit")),
        format!("KV fits      {}", match report.get("kv_fits") {
            Some(Value::Bool(flag)) => flag.to_string(),
            _ => "None".to_string(),
        }),
    ];
    if let Some(Value::Array(notes)) = report.get("notes") {
        for note in notes {
            lines.push(format!("note         {}", note.as_str().unwrap_or("")));
        }
    }
    lines
}

/// `nvidia-smi --query-gpu=memory.total` in bytes, or `None` - the same probe
/// the Python module makes, without the psutil dependency.
pub fn nvidia_vram() -> Option<i64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim().lines().next()?;
    line.trim().parse::<i64>().ok().map(|mib| mib * 1024 * 1024)
}
