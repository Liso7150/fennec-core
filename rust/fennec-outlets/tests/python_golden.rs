//! The gate and the redactor against CPython's own answers.

use fennec_outlets::{blocked_host, ip::Ip, loopback, redact_with};
use serde_json::Value;

fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is missing - run benchmarks/make_goldens.py: {error}", path.display()))
}

fn lines(name: &str) -> Vec<Value> {
    golden(name)
        .lines()
        .map(|line| serde_json::from_str(line).expect("golden line is json"))
        .collect()
}

fn packed(value: &Value) -> Vec<u8> {
    value.as_array().unwrap().iter().map(|item| item.as_u64().unwrap() as u8).collect()
}

#[test]
fn address_literals_parse_as_python_does() {
    let mut accepted = 0;
    for case in lines("ip-parse.jsonl") {
        let raw = case["raw"].as_str().unwrap();
        match (case["ok"].as_bool().unwrap(), Ip::parse(raw)) {
            (true, Some(address)) => {
                assert_eq!(address.version() as u64, case["version"].as_u64().unwrap(), "{raw}");
                assert_eq!(address.packed(), packed(&case["packed"]), "{raw}");
                accepted += 1;
            }
            (false, None) => {}
            (want, got) => panic!("{raw}: python ok={want}, rust parsed={got:?}"),
        }
    }
    assert!(accepted >= 35, "only {accepted} accepted literals");
}

#[test]
fn every_predicate_matches_the_stdlib() {
    for case in lines("ip-classify.jsonl") {
        let raw = case["ip"].as_str().unwrap();
        let address = Ip::parse(raw).unwrap_or_else(|| panic!("{raw} should parse"));
        let flag = |key: &str| case[key].as_bool().unwrap();
        assert_eq!(address.is_loopback(), flag("loopback"), "{raw} loopback");
        assert_eq!(address.is_link_local(), flag("link_local"), "{raw} link_local");
        assert_eq!(address.is_multicast(), flag("multicast"), "{raw} multicast");
        assert_eq!(address.is_reserved(), flag("reserved"), "{raw} reserved");
        assert_eq!(address.is_unspecified(), flag("unspecified"), "{raw} unspecified");
    }
}

#[test]
fn the_gate_refuses_exactly_what_python_refuses() {
    let cases: Vec<Value> = serde_json::from_str(&golden("url-gate.json")).unwrap();
    let mut refused = 0;
    for case in cases.iter() {
        let url = case["url"].as_str().unwrap();
        assert_eq!(blocked_host(url), case["blocked"].as_str().unwrap(), "{url}");
        assert_eq!(loopback(url), case["loopback"].as_bool().unwrap(), "{url} loopback");
        if !case["blocked"].as_str().unwrap().is_empty() {
            refused += 1;
        }
    }
    assert!(cases.len() >= 2_000, "only {} gate cases", cases.len());
    assert!(refused > 0, "no refusal was exercised");
}

#[test]
fn redaction_matches_line_for_line() {
    for case in lines("redact.jsonl") {
        let text = case["in"].as_str().unwrap();
        let want = case["out"].as_str().unwrap();
        let out = match case.get("env") {
            // the plain lines are redacted with an empty environment, which is
            // what the golden was captured with; the env cases carry their own
            None => redact_with(text, &|_key| None),
            Some(env) => {
                let map = env.as_object().unwrap();
                redact_with(text, &|key| map.get(key).and_then(Value::as_str).map(String::from))
            }
        };
        assert_eq!(out, want, "redact {text:?}");
    }
}
