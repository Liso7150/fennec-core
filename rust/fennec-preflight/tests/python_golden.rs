//! The preflight report against CPython's own arithmetic.

use fennec_preflight::{kv_bytes_per_token, plan, plan_with, Probes};
use fennec_pyjson::{dumps, Map};
use serde_json::Value;

fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is missing - run benchmarks/make_goldens.py: {error}", path.display()))
}

fn model_dir() -> String {
    // .. -> rust, .. -> fennec-core, .. -> the repo root that holds benchmarks/.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("workloads")
        .join("model")
        .to_string_lossy()
        .into_owned()
}

fn without<'a>(report: &'a Value, keys: &[&str]) -> Value {
    // `Map::remove` is a swap-remove and would move the last key into the hole,
    // so the comparison drops keys by rebuilding in order instead.
    let mut copy = Map::new();
    for (key, value) in report.as_object().unwrap() {
        if !keys.contains(&key.as_str()) {
            copy.insert(key.clone(), value.clone());
        }
    }
    Value::Object(copy)
}

#[test]
fn kv_growth_matches_the_config_python_read() {
    let want: Value = serde_json::from_str(&golden("kv-per-token.json")).unwrap();
    let got = kv_bytes_per_token(std::path::Path::new(&model_dir()));
    assert_eq!(Value::from(got), want["kv_bytes_per_token"]);
}

#[test]
fn every_probe_combination_produces_the_same_report() {
    // `model` and `disk_free_bytes` describe this volume, not the math, so the
    // golden omits them; everything else must agree field for field. The
    // blindfolded probes keep the run off this machine's hardware: a null in
    // the golden means "the probe reported nothing", which is what BLIND says.
    let mut cases = 0;
    for line in golden("preflight-matrix.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let probe = &case["probe"];
        let ctx = case["report"]["ctx"].as_i64().unwrap();
        let got = plan_with(
            &model_dir(),
            ctx,
            None,
            probe["vram_bytes"].as_i64(),
            probe["ram_available_bytes"].as_i64(),
            Probes::BLIND,
        );
        assert_eq!(
            dumps(&without(&got, &["model", "disk_free_bytes"])),
            dumps(&without(&case["report"], &["model", "disk_free_bytes"])),
            "ctx {ctx} vram {}/ram {}",
            probe["vram_bytes"],
            probe["ram_available_bytes"]
        );
        cases += 1;
    }
    assert!(cases >= 30, "only {cases} probe combinations exercised");
}

#[test]
fn an_unreadable_config_is_unknown_geometry_not_a_crash() {
    let missing = plan(
        &std::path::Path::new(&model_dir()).join("nope").to_string_lossy(),
        4096,
        None,
        Some(0),
        Some(0),
    );
    assert_eq!(missing["kv_bytes_per_token"], Value::Null);
    assert_eq!(missing["weights_bytes"], Value::from(0));
    assert_eq!(missing["weights_fit"], Value::from("unknown"));
    assert_eq!(missing["kv_fits"], Value::Null);
    let notes = missing["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 2, "{notes:?}");
}

#[test]
fn a_caller_supplied_kv_size_skips_the_config_read() {
    let mut cases = 0;
    for line in golden("preflight-injected.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let probe = &case["case"];
        let got = plan_with(
            &model_dir(),
            probe["ctx"].as_i64().unwrap(),
            probe["kv_bytes"].as_i64(),
            probe["vram"].as_i64(),
            probe["ram"].as_i64(),
            Probes::BLIND,
        );
        assert_eq!(
            dumps(&without(&got, &["model", "disk_free_bytes"])),
            dumps(&without(&case["report"], &["model", "disk_free_bytes"])),
            "kv {:?} at ctx {}",
            probe["kv_bytes"],
            probe["ctx"]
        );
        cases += 1;
    }
    assert_eq!(cases, 6, "every injected case must run");
}
