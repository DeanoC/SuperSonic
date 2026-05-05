//! Parity gate: assert that crates/bench's SUPPORTED_COMBOS table matches the
//! runner's REGISTRY + per-engine quant capabilities. If this test fails after
//! a change to runner/src/registry.rs or a feature-compatibility shift, update
//! crates/bench/src/matrix.rs::SUPPORTED_COMBOS to match.
//!
//! This test reads bench's static table at runtime by parsing
//! crates/bench/src/matrix.rs (text scan; cheap and avoids a dep cycle).

use std::path::PathBuf;

#[test]
fn bench_combo_table_mentions_every_runner_supported_pair() {
    let bench_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("src")
        .join("matrix.rs");
    let bench_text = std::fs::read_to_string(&bench_src)
        .unwrap_or_else(|e| panic!("read {}: {e}", bench_src.display()));

    // Use the same source-of-truth used by the runner engines. Adjust this list
    // when adding/removing a (model, quant) on gfx1100.
    let expected_pairs: &[(&str, &str)] = &[
        ("qwen3.5-0.8b", "bf16"), ("qwen3.5-0.8b", "int4"),
        ("qwen3.5-0.8b", "fp8r"), ("qwen3.5-0.8b", "kv-fp8"),
        ("qwen3.5-2b", "bf16"), ("qwen3.5-2b", "int4"),
        ("qwen3.5-2b", "fp8r"), ("qwen3.5-2b", "kv-fp8"),
        ("qwen3.5-4b", "bf16"), ("qwen3.5-4b", "int4"),
        ("qwen3.5-4b", "fp8r"), ("qwen3.5-4b", "kv-fp8"),
        ("qwen3.5-9b", "bf16"), ("qwen3.5-9b", "int4"),
        ("qwen3.5-9b", "fp8r"), ("qwen3.5-9b", "kv-fp8"),
        ("gemma4-e2b", "bf16"), ("gemma4-e2b", "int4"), ("gemma4-e2b", "kv-fp8"),
        ("gemma4-e4b", "bf16"), ("gemma4-e4b", "int4"), ("gemma4-e4b", "kv-fp8"),
        ("phi4-mini", "bf16"), ("phi4-mini", "int4"),
        ("phi4-mini", "fp8r"), ("phi4-mini", "kv-fp8"),
        ("qwen3.6-35b-a3b", "int4"), ("qwen3.6-35b-a3b", "kv-fp8"),
    ];

    for (model, quant) in expected_pairs {
        let needle = format!("model: \"{model}\", quant: \"{quant}\"");
        assert!(bench_text.contains(&needle),
                "bench/src/matrix.rs is missing combo: {model}/{quant}\n\
                 If runner support changed, add or remove the row to match.");
    }
}
