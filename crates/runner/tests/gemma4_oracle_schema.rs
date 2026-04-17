// Round-trip the Gemma 4 oracle JSON schema through the same `OracleOutput`
// struct that the runner uses to consume PyTorch oracle output. Proves the
// minimal Gemma 4 oracle (oracle/gemma4_oracle.py) is wire-compatible with
// crates/runner/src/oracle.rs without wiring it into --validate yet.

#[path = "../src/oracle.rs"]
mod oracle;

use std::path::PathBuf;

#[test]
fn gemma4_oracle_json_deserializes_into_oracle_output() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gemma4_oracle_sample.json");
    let raw = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixture.display()));

    let parsed: oracle::OracleOutput =
        serde_json::from_str(&raw).expect("fixture should deserialize into OracleOutput");

    assert_eq!(parsed.prompt_tokens, 4);
    assert_eq!(parsed.generated_tokens, 4);
    assert_eq!(parsed.generated_token_ids.len(), parsed.generated_tokens);
    assert_eq!(parsed.decode_logits.len(), parsed.generated_tokens);
    assert!(!parsed.prefill_logits.is_empty(), "prefill_logits empty");

    let vocab = parsed.prefill_logits.len();
    for (i, step) in parsed.decode_logits.iter().enumerate() {
        assert_eq!(step.len(), vocab, "decode step {i} vocab mismatch");
    }

    // Minimal oracle does not emit state; those fields must be absent/None.
    assert!(parsed.prefill_hidden.is_none());
    assert!(parsed.kv_caches.is_none());
    assert!(parsed.conv_states.is_none());
    assert!(parsed.recurrent_states.is_none());
}
