//! Every `json.dumps(value, ensure_ascii=False)` shape Python produced, checked
//! string-for-string (`rust/goldens/pyjson.jsonl`, from
//! `benchmarks/make_goldens.py`).

use fennec_pyjson::{char_len, dumps, float_repr};
use serde_json::Value;

fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is missing - run benchmarks/make_goldens.py: {error}", path.display()))
}

#[test]
fn dumps_matches_python_string_for_string() {
    let mut checked = 0;
    for line in golden("pyjson.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let want = case["dumps"].as_str().unwrap();
        let got = dumps(&case["value"]);
        assert_eq!(got, want);
        assert_eq!(char_len(&case["value"]), case["length"].as_u64().unwrap() as usize);
        checked += 1;
    }
    assert!(checked >= 6, "only {checked} pyjson cases");
}

#[test]
fn object_key_order_survives_a_round_trip() {
    let raw = r#"{"z": 1, "a": 2, "m": 3}"#;
    let value: Value = serde_json::from_str(raw).unwrap();
    assert_eq!(dumps(&value), r#"{"z": 1, "a": 2, "m": 3}"#);
}

#[test]
fn floats_keep_the_python_repr_switch() {
    assert_eq!(float_repr(1.0), "1.0");
    assert_eq!(float_repr(0.1), "0.1");
    assert_eq!(float_repr(1e15), "1000000000000000.0");
    assert_eq!(float_repr(1e16), "1e+16");
    assert_eq!(float_repr(1e-4), "0.0001");
    assert_eq!(float_repr(1e-5), "1e-05");
    assert_eq!(float_repr(-0.0), "-0.0");
    assert_eq!(float_repr(f64::from_bits(1)), "5e-324");
    assert_eq!(float_repr(f64::MAX), "1.7976931348623157e+308");
}
