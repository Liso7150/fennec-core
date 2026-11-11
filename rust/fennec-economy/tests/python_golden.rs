//! Byte-for-byte agreement with `fennec_core/context.py`, against the goldens in
//! `rust/goldens/` (produced by `benchmarks/make_goldens.py`).

use fennec_economy::{compact, elide, size, Options, Stats};
use serde_json::Value;
use std::collections::HashMap;

fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is missing - run benchmarks/make_goldens.py: {error}", path.display()))
}

fn messages(raw: &str) -> Vec<Value> {
    match serde_json::from_str::<Value>(raw).unwrap() {
        Value::Array(items) => items,
        other => panic!("expected an array, got {other}"),
    }
}

#[test]
fn elide_matches_python() {
    let mut checked = 0;
    for line in golden("elide.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let text = case["text"].as_str().unwrap();
        let limit = case["limit"].as_u64().unwrap() as usize;
        let want = case["out"].as_str().unwrap();
        assert_eq!(elide(text, limit), want, "limit {limit} on {} chars", case["chars"]);
        checked += 1;
    }
    assert!(checked >= 30, "only {checked} elide cases");
}

#[test]
fn compact_payload_is_identical_to_python() {
    let input = messages(&golden("compact-input.json"));
    let (payload, _stats) = compact(&input, &Options::default(), &mut HashMap::new());
    let rendered = fennec_pyjson::dumps(&Value::Array(payload));
    assert_eq!(rendered, golden("compact-payload.json").trim_end());
}

#[test]
fn compact_stats_are_identical_to_python() {
    let input = messages(&golden("compact-input.json"));
    let (payload, stats) = compact(&input, &Options::default(), &mut HashMap::new());
    let want: Value = serde_json::from_str(&golden("compact-stats.json")).unwrap();
    assert_eq!(stats.chars_before, want["chars_before"].as_u64().unwrap() as usize);
    assert_eq!(stats.chars_after, want["chars_after"].as_u64().unwrap() as usize);
    assert_eq!(stats.elided, want["elided"].as_u64().unwrap() as usize);
    assert_eq!(stats.deduped, want["deduped"].as_u64().unwrap() as usize);
    let sizes: Vec<usize> = payload.iter().map(size).collect();
    let want_sizes: Vec<usize> = want["sizes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(sizes, want_sizes);
}

#[test]
fn a_second_pass_over_the_same_run_dedupes_across_steps() {
    let input = messages(&golden("compact-input.json"));
    let mut seen = HashMap::new();
    let (_, first) = compact(&input, &Options::default(), &mut seen);
    let (_, second) = compact(&input, &Options::default(), &mut seen);
    assert_eq!(first.deduped + first.elided, second.elided + second.deduped);
    assert!(second.deduped > first.deduped, "cross-step duplicates must grow");
}

#[test]
fn the_original_transcript_is_never_mutated() {
    let raw = golden("compact-input.json");
    let input = messages(&raw);
    let before = fennec_pyjson::dumps(&Value::Array(input.clone()));
    compact(&input, &Options::default(), &mut HashMap::new());
    assert_eq!(before, fennec_pyjson::dumps(&Value::Array(input)));
}

#[test]
fn env_knobs_move_the_budget_like_the_python_limits_helper() {
    let options = Options::sized(12_000, 2_000);
    assert_eq!(options.budget_chars, 12_000);
    let stats: Stats = Stats::default();
    assert_eq!(stats.budget_chars, 0);
}
