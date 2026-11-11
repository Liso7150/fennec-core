//! The framing and the update bounds against what Python's `json` and `_clip`
//! produce - including that a frame Python wrote is what this writes back.

use fennec_acp::{clip, decode_frame, decode_stream, encode_frame, pieces};
use fennec_pyjson::{dumps, Value};

fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is missing - run benchmarks/make_goldens.py: {error}", path.display()))
}

#[test]
fn clip_is_python_clip() {
    let mut checked = 0;
    for line in golden("clip.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let limit = case["limit"].as_u64().unwrap() as usize;
        let want = dumps(&case["out"]);
        let got = dumps(&clip(&case["value"], limit));
        assert_eq!(got, want, "limit {limit}");
        checked += 1;
    }
    assert!(checked >= 30, "only {checked} clip cases");
}

#[test]
fn pieces_cut_on_code_points_like_python_slicing() {
    for line in golden("pieces.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let size = case["size"].as_u64().unwrap() as usize;
        let want: Vec<String> = case["out"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_str().unwrap().to_string())
            .collect();
        assert_eq!(pieces(case["text"].as_str().unwrap(), size), want);
    }
}

#[test]
fn a_python_frame_survives_decode() {
    let raw = golden("frames.jsonl");
    let (frames, errors) = decode_stream(&raw);
    assert!(errors.is_empty(), "{:?}", errors.first());
    assert!(frames.len() >= 2_000, "only {} frames", frames.len());
}

#[test]
fn reencoding_writes_the_bytes_the_transport_writes() {
    let mut checked = 0;
    for line in golden("wire.jsonl").lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        assert_eq!(encode_frame(&case["frame"]), format!("{}\n", case["wire"].as_str().unwrap()));
        checked += 1;
    }
    assert!(checked >= 500, "only {checked} wire frames");
}

#[test]
fn a_broken_line_is_named_not_thrown() {
    let (frames, errors) = decode_stream("{\"jsonrpc\": \"2.0\"}\n\nnot json\n");
    assert_eq!(frames.len(), 1);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("not json"), "{}", errors[0]);
    assert_eq!(decode_frame("{}"), Ok(Value::Object(serde_json::Map::new())));
}
