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
    assert!(parsed.prefill_per_layer_hidden.is_none());
    assert!(parsed.prefill_per_layer_hidden_shape.is_none());
    assert!(parsed.prefill_per_layer_pre_ple.is_none());
}

#[test]
fn gemma4_oracle_state_json_deserializes_and_layout_matches_e2b() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gemma4_oracle_state_sample.json");
    let raw = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixture.display()));

    let parsed: oracle::OracleOutput =
        serde_json::from_str(&raw).expect("state fixture should deserialize into OracleOutput");

    // Prefill hidden is shape [1, 1, hidden_dim=1536] for E2B.
    let hidden_shape = parsed
        .prefill_hidden_shape
        .as_ref()
        .expect("prefill_hidden_shape must be present with --emit-state");
    assert_eq!(hidden_shape, &vec![1, 1, 1536]);
    assert!(parsed.prefill_hidden.is_some(), "prefill_hidden missing");

    // Gemma 4 E2B has 35 transformer layers but num_kv_shared_layers=20, so
    // HF's DynamicCache exposes exactly 15 entries.
    let kv_caches = parsed.kv_caches.as_ref().expect("kv_caches missing");
    assert_eq!(kv_caches.len(), 15, "E2B should have 15 unique KV slots");

    // Layer types alternate [SWA, SWA, SWA, SWA, FULL] across the 15 slots.
    // SWA: head_dim=256; FULL: head_dim=512. K and V shapes are identical
    // since K/V both have num_kv_heads=1.
    const SWA_HEAD_DIM: usize = 256;
    const FULL_HEAD_DIM: usize = 512;
    let full_attn_slots = [4_usize, 9, 14];
    for (i, kv) in kv_caches.iter().enumerate() {
        assert_eq!(kv.layer, i, "layer index should match slot order");
        assert_eq!(kv.k_shape, kv.v_shape, "K and V shapes must match");
        assert_eq!(kv.k_shape.len(), 4, "expected [batch, kv_heads, seq, head_dim]");
        let head_dim = kv.k_shape[3];
        let expected = if full_attn_slots.contains(&i) { FULL_HEAD_DIM } else { SWA_HEAD_DIM };
        assert_eq!(head_dim, expected, "slot {i} head_dim mismatch");
    }

    // Gemma 4 has no linear attention — conv/recurrent must remain absent.
    assert!(parsed.conv_states.is_none());
    assert!(parsed.recurrent_states.is_none());

    // Per-layer post-block hidden states: one per transformer layer.
    // E2B has 35 layers; shape per entry is [1, 1, hidden_dim=1536].
    let per_layer = parsed
        .prefill_per_layer_hidden
        .as_ref()
        .expect("prefill_per_layer_hidden missing from state dump");
    assert_eq!(per_layer.len(), 35, "E2B has 35 transformer layers");
    let per_layer_shape = parsed
        .prefill_per_layer_hidden_shape
        .as_ref()
        .expect("prefill_per_layer_hidden_shape missing");
    assert_eq!(per_layer_shape, &vec![1, 1, 1536]);

    // Pre-PLE snapshot: same count, same implicit shape. Same-length base64
    // stub assertion is sufficient schema-level coverage.
    let pre_ple = parsed
        .prefill_per_layer_pre_ple
        .as_ref()
        .expect("prefill_per_layer_pre_ple missing");
    assert_eq!(pre_ple.len(), 35);
}
