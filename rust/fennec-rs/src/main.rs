//! `fennec-rs` - the Rust port as a program, so the ported glue can be run,
//! measured and served to Python without a CPython interpreter in the loop.
//!
//! Two kinds of command live here. The product ones (`preflight`, `compact`,
//! `redact`, `urlgate`, `frames`, `acp`, `serve`, `mem`) are the ported logic
//! with its Python contract: same flags, same bytes where bytes are the contract.
//! The `bench` ones replay a workload file and answer with the same JSON shapes
//! the Python snippets in `benchmarks/bench.py` answer with, so `--impl python`
//! and `--impl rust` time the same work the same way.
//!
//! The name is deliberately not `fennec`: this must never shadow the Python CLI
//! on PATH. The CLI is the product; `fennec_core/rust_bridge.py` hands it the
//! work this binary measurably wins and nothing else.

mod mem;

use fennec_acp::{decode_stream, encode_frame, pieces};
use fennec_economy::{compact, elide, limits, Options, MIN_RESULT_CHARS};
use fennec_outlets::{blocked_host, redact};
use fennec_preflight::{kv_bytes_per_token, plan};
use fennec_pyjson::{dumps, Map, Value};
use serde_json::json;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "\
fennec-rs - the Rust port of the Fennec suite glue

  fennec-rs --version                 print the version
  fennec-rs preflight <model> [--ctx N] [--json]
  fennec-rs compact <transcript.json> [--stats]
  fennec-rs elide <limit>             stdin, head/tail elided to N characters
  fennec-rs redact                    stdin, Fennec's own redaction
  fennec-rs urlgate <url>...          allow/block, one line per url
  fennec-rs frames                    stdin NDJSON, decoded and re-encoded
  fennec-rs acp                       stdin ACP pipe: answers initialize/exit
  fennec-rs serve                     stdin NDJSON bridge for fennec_core
  fennec-rs mem                       this process' rss, peak rss and cpu time
  fennec-rs bench <target> [--reps N] target: session|outlets|frames|preflight
";

fn ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

/// `str.splitlines()` for the newline-terminated files these workloads are.
fn lines_of(raw: &str) -> Vec<&str> {
    let mut out: Vec<&str> = raw
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if out.last().is_some_and(|last| last.is_empty()) {
        out.pop();
    }
    out
}

struct Args {
    positional: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    fn has(&self, name: &str) -> bool {
        self.flags.contains_key(name)
    }

    fn number(&self, name: &str, default: i64) -> i64 {
        self.flag(name).and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    fn reps(&self) -> usize {
        self.number("reps", 1).max(0) as usize
    }
}

fn parse(args: &[String]) -> Args {
    let mut positional = Vec::new();
    let mut flags = HashMap::new();
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if let Some(name) = token.strip_prefix("--") {
            let next = args.get(index + 1);
            match next.filter(|value| !value.starts_with("--")) {
                Some(value) => {
                    flags.insert(name.to_string(), value.clone());
                    index += 2;
                    continue;
                }
                None => {
                    flags.insert(name.to_string(), "true".to_string());
                }
            }
        } else {
            positional.push(token.clone());
        }
        index += 1;
    }
    Args { positional, flags }
}

fn read_file(path: &str, command: &str) -> io::Result<String> {
    std::fs::read_to_string(path).map_err(|error| {
        eprintln!("fennec-rs {command}: {path}: {error}");
        error
    })
}

fn decode_array(raw: &str, command: &str) -> Option<Vec<Value>> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Array(items)) => Some(items),
        Ok(_) => {
            eprintln!("fennec-rs {command}: expected a JSON array");
            None
        }
        Err(error) => {
            eprintln!("fennec-rs {command}: {error}");
            None
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let command = argv.first().map(String::as_str).unwrap_or("");
    let rest: &[String] = if argv.is_empty() { &[] } else { &argv[1..] };
    std::process::exit(match command {
        "--version" | "-V" | "version" => {
            println!("fennec-rs {VERSION}");
            0
        }
        "" | "--help" | "-h" | "help" => {
            print!("{USAGE}");
            if command.is_empty() { 2 } else { 0 }
        }
        "mem" => {
            println!("{}", dumps(&mem::report()));
            0
        }
        "preflight" => cmd_preflight(rest),
        "compact" => cmd_compact(rest),
        "elide" => cmd_elide(rest),
        "redact" => cmd_redact(),
        "urlgate" => cmd_urlgate(rest),
        "frames" => cmd_frames(),
        "acp" => cmd_acp(),
        "serve" => cmd_serve(),
        "bench" => cmd_bench(rest),
        other => {
            eprintln!("fennec-rs: unknown command {other}\n\n{USAGE}");
            2
        }
    });
}

// ------------------------------------------------------------- product verbs

fn options(args: &Args) -> Options {
    let bounds = limits();
    Options {
        budget_chars: args.number("budget", bounds.budget_chars as i64).max(1) as usize,
        result_chars: args.number("result", bounds.result_chars as i64).max(1) as usize,
        floor_chars: args.number("floor", MIN_RESULT_CHARS as i64).max(1) as usize,
    }
}

fn stats_json(stats: &fennec_economy::Stats) -> Value {
    json!({
        "chars_before": stats.chars_before,
        "chars_after": stats.chars_after,
        "elided": stats.elided,
        "deduped": stats.deduped,
        "budget_chars": stats.budget_chars,
        "result_chars": stats.result_chars,
    })
}

fn cmd_compact(args_slice: &[String]) -> i32 {
    let args = parse(args_slice);
    let Some(path) = args.positional.first() else {
        eprintln!("fennec-rs compact needs a transcript path");
        return 2;
    };
    let Ok(raw) = read_file(path, "compact") else { return 1 };
    let Some(messages) = decode_array(&raw, "compact") else { return 1 };
    let (payload, stats) = compact(&messages, &options(&args), &mut HashMap::new());
    if args.has("stats") {
        println!("{}", dumps(&stats_json(&stats)));
    } else {
        println!("{}", dumps(&Value::Array(payload)));
    }
    0
}

fn cmd_elide(args_slice: &[String]) -> i32 {
    let args = parse(args_slice);
    let Some(limit) = args.positional.first().and_then(|value| value.parse::<usize>().ok()) else {
        eprintln!("fennec-rs elide needs a character limit");
        return 2;
    };
    let mut text = String::new();
    if io::stdin().read_to_string(&mut text).is_err() {
        return 1;
    }
    print!("{}", elide(&text, limit));
    0
}

fn cmd_redact() -> i32 {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in BufReader::new(stdin.lock()).lines() {
        let Ok(line) = line else { return 1 };
        if write!(out, "{}\n", redact(&line)).is_err() {
            return 1;
        }
    }
    if out.flush().is_err() { 1 } else { 0 }
}

fn cmd_urlgate(args_slice: &[String]) -> i32 {
    if args_slice.is_empty() {
        eprintln!("fennec-rs urlgate needs at least one url");
        return 2;
    }
    let mut blocked = 0;
    for url in args_slice {
        let reason = blocked_host(url);
        if reason.is_empty() {
            println!("allow\t{url}");
        } else {
            println!("block\t{url}\t{reason}");
            blocked += 1;
        }
    }
    i32::from(blocked > 0)
}

fn cmd_frames() -> i32 {
    let mut raw = String::new();
    if io::stdin().read_to_string(&mut raw).is_err() {
        return 1;
    }
    let (frames, errors) = decode_stream(&raw);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for frame in &frames {
        if out.write_all(encode_frame(frame).as_bytes()).is_err() {
            return 1;
        }
    }
    out.flush().ok();
    for error in &errors {
        eprintln!("fennec-rs frames: {error}");
    }
    if errors.is_empty() { 0 } else { 1 }
}

fn cmd_preflight(args_slice: &[String]) -> i32 {
    let args = parse(args_slice);
    let Some(path) = args.positional.first() else {
        eprintln!("fennec-rs preflight needs a model directory");
        return 2;
    };
    let ctx = args.number("ctx", 8192);
    let report = plan(path, ctx, None, None, None);
    if args.has("json") {
        println!("{}", dumps(&report));
        return 0;
    }
    for line in fennec_preflight::human(&report) {
        println!("{line}");
    }
    0
}

/// The transport half of ACP, without the engine: enough of the handshake for a
/// client to measure what a spawn plus one round trip costs. `session/prompt`
/// stays Python's job - this answers it the way an unimplemented method is
/// answered, and says so.
fn cmd_acp() -> i32 {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in BufReader::new(stdin.lock()).lines() {
        let Ok(line) = line else { return 1 };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            let frame = encode_frame(&json!({"jsonrpc": "2.0", "id": null, "error": {
                "code": -32700, "message": "invalid json"}}));
            out.write_all(frame.as_bytes()).ok();
            out.flush().ok();
            continue;
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        if method == "exit" {
            return 0;
        }
        let message = if method == "initialize" {
            json!({"jsonrpc": "2.0", "id": id, "result": {
                "protocolVersion": 1,
                "agentInfo": {"name": "fennec-rs", "title": "Fennec (Rust)", "version": VERSION},
                "authMethods": [],
                "agentCapabilities": {
                    "loadSession": false,
                    "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true},
                    "mcpCapabilities": {"http": false, "sse": false},
                    "sessionCapabilities": {}}}})
        } else {
            json!({"jsonrpc": "2.0", "id": id, "error": {
                "code": -32601, "message": format!("{method} is not served by fennec-rs")}})
        };
        if out.write_all(encode_frame(&message).as_bytes()).is_err() {
            return 1;
        }
        out.flush().ok();
    }
    0
}

// ------------------------------------------------------------------- bridge

fn seen_map(request: &Value) -> HashMap<String, String> {
    match request.get("seen") {
        Some(Value::Object(entries)) => entries
            .iter()
            .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_string())))
            .collect(),
        _ => HashMap::new(),
    }
}

fn serve_once(request: &Value) -> Value {
    let operation = request.get("op").and_then(Value::as_str).unwrap_or("");
    match operation {
        "ping" => json!({"ok": true, "impl": "fennec-rs", "version": VERSION}),
        "compact" => match request.get("messages") {
            Some(Value::Array(items)) => {
                let bounds = limits();
                let options = Options {
                    budget_chars: request
                        .get("budget_chars")
                        .and_then(Value::as_i64)
                        .map_or(bounds.budget_chars, |value| value as usize),
                    result_chars: request
                        .get("result_chars")
                        .and_then(Value::as_i64)
                        .map_or(bounds.result_chars, |value| value as usize),
                    floor_chars: request
                        .get("floor_chars")
                        .and_then(Value::as_i64)
                        .map_or(MIN_RESULT_CHARS, |value| value as usize),
                };
                let mut seen = seen_map(request);
                let (payload, stats) = compact(items, &options, &mut seen);
                let carried: Map<String, Value> = seen
                    .into_iter()
                    .map(|(digest, label)| (digest, Value::String(label)))
                    .collect();
                json!({"ok": true, "payload": payload, "stats": stats_json(&stats),
                       "seen": Value::Object(carried)})
            }
            _ => json!({"ok": false, "error": "compact needs a messages array"}),
        },
        "urlgate" => match request.get("url").and_then(Value::as_str) {
            Some(url) => json!({"ok": true, "blocked": blocked_host(url)}),
            None => json!({"ok": false, "error": "urlgate needs a url string"}),
        },
        "elide" => match (request.get("text").and_then(Value::as_str),
                          request.get("limit").and_then(Value::as_i64)) {
            (Some(text), Some(limit)) => {
                json!({"ok": true, "text": elide(text, limit.max(0) as usize)})
            }
            _ => json!({"ok": false, "error": "elide needs text and limit"}),
        },
        "pieces" => match (request.get("text").and_then(Value::as_str),
                           request.get("size").and_then(Value::as_i64)) {
            (Some(text), Some(size)) => json!({"ok": true, "pieces": pieces(text, size.max(0) as usize)}),
            _ => json!({"ok": false, "error": "pieces needs text and size"}),
        },
        "preflight" => match request.get("model").and_then(Value::as_str) {
            Some(model) => {
                let ctx = request.get("ctx").and_then(Value::as_i64).unwrap_or(8192);
                json!({"ok": true, "report": plan(model, ctx, None, None, None)})
            }
            None => json!({"ok": false, "error": "preflight needs a model path"}),
        },
        "exit" => json!({"ok": true, "bye": true}),
        other => json!({"ok": false, "error": format!("unknown op {other:?}")}),
    }
}

/// One request per line, one answer per line, in order. `fennec_core/rust_bridge.py`
/// owns the pipe and therefore the sequencing; keeping this side stateless means
/// a request is answerable without knowing what came before it.
fn cmd_serve() -> i32 {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in BufReader::new(stdin.lock()).lines() {
        let Ok(line) = line else { return 1 };
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<Value>(&line) {
            Ok(value) => value,
            Err(error) => json!({"ok": false, "error": format!("invalid json: {error}")}),
        };
        let leave = request.get("op").and_then(Value::as_str) == Some("exit");
        let frame = encode_frame(&serve_once(&request));
        if out.write_all(frame.as_bytes()).is_err() {
            return 1;
        }
        out.flush().ok();
        if leave {
            return 0;
        }
    }
    0
}

// -------------------------------------------------------------------- bench

/// What a turn costs, not what a pass costs: the transcript is replayed the way
/// the loop grows it - step 1, steps 1-2, steps 1-3 - with one `seen` map carried
/// across the whole run. A single full pass answers a different question, and only
/// this one lands in a response time. `benchmarks/bench.py` replays the same
/// boundaries through CPython, so the two rows are the same question twice.
fn bench_session(args: &Args) -> i32 {
    let Ok(raw) = read_file(&args.positional[0], "bench session") else { return 1 };
    let Some(messages) = decode_array(&raw, "bench session") else { return 1 };
    let Ok(raw) = read_file(&args.positional[1], "bench session") else { return 1 };
    let cuts: Vec<usize> = serde_json::from_str::<Vec<usize>>(&raw)
        .unwrap_or_default()
        .into_iter()
        .filter(|cut| *cut <= messages.len())
        .collect();
    if cuts.is_empty() {
        eprintln!("fennec-rs bench session: no step boundaries in {}", args.positional[1]);
        return 1;
    }
    let options = options(args);
    let reps = args.reps();
    let clock = Instant::now();
    let (_, first) = compact(&messages, &options, &mut HashMap::new());
    let cold = ms(clock.elapsed());
    let mut out = stats_json(&first);
    let mut step_samples: Vec<f64> = Vec::with_capacity(reps * cuts.len());
    let mut session_totals: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut seen = HashMap::new();
        let mut total = 0.0;
        for cut in &cuts {
            let clock = Instant::now();
            compact(&messages[..*cut], &options, &mut seen);
            let elapsed = ms(clock.elapsed());
            total += elapsed;
            step_samples.push(elapsed);
        }
        session_totals.push(total);
    }
    let fields = out.as_object_mut().expect("stats_json is an object");
    fields.insert("cold_ms".to_string(), Value::from(cold));
    fields.insert("step_ms".to_string(),
                  Value::Array(step_samples.iter().map(|v| Value::from(*v)).collect()));
    fields.insert("session_ms".to_string(),
                  Value::Array(session_totals.iter().map(|v| Value::from(*v)).collect()));
    fields.insert("steps".to_string(), Value::from(cuts.len() as u64));
    println!("{}", out);
    0
}

fn bench_outlets(args: &Args) -> i32 {
    let (secrets, urls) = (&args.positional[0], &args.positional[1]);
    let Ok(raw) = read_file(secrets, "bench outlets") else { return 1 };
    let lines: Vec<String> = lines_of(&raw).iter().map(|line| line.to_string()).collect();
    let Ok(raw) = read_file(urls, "bench outlets") else { return 1 };
    let targets: Vec<String> = lines_of(&raw).iter().map(|line| line.to_string()).collect();
    let clock = Instant::now();
    let redacted: Vec<String> = lines.iter().map(|line| redact(line)).collect();
    let redact_ms = ms(clock.elapsed());
    let hits = lines.iter().zip(&redacted).filter(|(a, b)| a != b).count();
    let clock = Instant::now();
    let gate: Vec<String> = targets.iter().map(|url| blocked_host(url)).collect();
    let block_ms = ms(clock.elapsed());
    let blocked = gate.iter().filter(|reason| !reason.is_empty()).count();
    let mut samples = Vec::with_capacity(args.reps());
    for _ in 0..args.reps() {
        let clock = Instant::now();
        for line in &lines {
            std::hint::black_box(redact(line));
        }
        samples.push(ms(clock.elapsed()));
    }
    println!(
        "{}",
        json!({"redact_ms": redact_ms, "samples": samples, "hits": hits,
               "block_ms": block_ms, "blocked": blocked, "n": lines.len()})
    );
    0
}

fn bench_frames(args: &Args) -> i32 {
    let path = &args.positional[0];
    let Ok(raw) = read_file(path, "bench frames") else { return 1 };
    let lines: Vec<&str> = lines_of(&raw);
    let clock = Instant::now();
    let decoded: Vec<Value> = lines
        .iter()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let decode_ms = ms(clock.elapsed());
    let clock = Instant::now();
    let blob = decoded
        .iter()
        .map(dumps)
        .collect::<Vec<String>>()
        .join("\n")
        + "\n";
    let encode_ms = ms(clock.elapsed());
    let mut samples = Vec::with_capacity(args.reps());
    for _ in 0..args.reps() {
        let clock = Instant::now();
        for line in &lines {
            let _ = std::hint::black_box(serde_json::from_str::<Value>(line));
        }
        samples.push(ms(clock.elapsed()));
    }
    println!(
        "{}",
        json!({"decode_ms": decode_ms, "encode_ms": encode_ms, "samples": samples,
               "n": lines.len(), "bytes": blob.len()})
    );
    0
}

fn bench_preflight(args: &Args) -> i32 {
    let path = &args.positional[0];
    let ctx = args.number("ctx", 16384);
    let clock = Instant::now();
    let report = plan(path, ctx, None, None, None);
    let first = ms(clock.elapsed());
    let mut samples = Vec::with_capacity(args.reps());
    for _ in 0..args.reps() {
        let clock = Instant::now();
        std::hint::black_box(kv_bytes_per_token(Path::new(path)));
        samples.push(ms(clock.elapsed()));
    }
    println!(
        "{}",
        json!({"first_ms": first, "samples": samples, "report": report})
    );
    0
}

fn cmd_bench(rest: &[String]) -> i32 {
    let target = rest.first().map(String::as_str).unwrap_or("");
    let args = parse(&rest[1.min(rest.len())..]);
    let needed: usize = match target {
        "frames" | "preflight" => 1,
        "outlets" | "session" => 2,
        other => {
            eprintln!("fennec-rs bench: unknown target {other}");
            return 2;
        }
    };
    if args.positional.len() < needed {
        eprintln!("fennec-rs bench {target} needs {needed} workload path(s)");
        return 2;
    }
    match target {
        "session" => bench_session(&args),
        "outlets" => bench_outlets(&args),
        "frames" => bench_frames(&args),
        "preflight" => bench_preflight(&args),
        _ => unreachable!(),
    }
}
