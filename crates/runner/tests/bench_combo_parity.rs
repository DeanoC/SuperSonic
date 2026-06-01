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
    let expected_rows: &[(&str, &str, &str)] = &[
        ("qwen3.5-0.8b", "bf16", "Gfx1100"),
        ("qwen3.5-0.8b", "int4", "Gfx1100"),
        ("qwen3.5-0.8b", "fp8r", "Gfx1100"),
        ("qwen3.5-0.8b", "kv-fp8", "Gfx1100"),
        ("qwen3.5-2b", "bf16", "Gfx1100"),
        ("qwen3.5-2b", "int4", "Gfx1100"),
        ("qwen3.5-2b", "fp8r", "Gfx1100"),
        ("qwen3.5-2b", "kv-fp8", "Gfx1100"),
        ("qwen3.5-4b", "bf16", "Gfx1100"),
        ("qwen3.5-4b", "int4", "Gfx1100"),
        ("qwen3.5-4b", "fp8r", "Gfx1100"),
        ("qwen3.5-4b", "kv-fp8", "Gfx1100"),
        ("qwen3.5-9b", "bf16", "Gfx1100"),
        ("qwen3.5-9b", "int4", "Gfx1100"),
        ("qwen3.5-9b", "fp8r", "Gfx1100"),
        ("qwen3.5-9b", "kv-fp8", "Gfx1100"),
        ("gemma4-e2b", "bf16", "Gfx1100"),
        ("gemma4-e2b", "int4", "Gfx1100"),
        ("gemma4-e2b", "fp8r", "Gfx1100"),
        ("gemma4-e2b", "kv-fp8", "Gfx1100"),
        ("gemma4-e4b", "bf16", "Gfx1100"),
        ("gemma4-e4b", "int4", "Gfx1100"),
        ("gemma4-e4b", "fp8r", "Gfx1100"),
        ("gemma4-e4b", "kv-fp8", "Gfx1100"),
        ("phi4-mini", "bf16", "Gfx1100"),
        ("phi4-mini", "int4", "Gfx1100"),
        ("phi4-mini", "fp8r", "Gfx1100"),
        ("phi4-mini", "kv-fp8", "Gfx1100"),
        ("qwen3.6-35b-a3b", "int4", "Gfx1100"),
        ("qwen3.6-35b-a3b", "kv-fp8", "Gfx1100"),
        ("qwen3.6-35b-a3b", "int4", "AppleM5Max"),
        ("qwen3.5-35b-a3b", "q4km", "AppleM5Max"),
        ("qwen3.6-35b-a3b", "int4", "Sm86"),
        ("qwen3.6-35b-a3b", "int4-spec025", "Sm86"),
        ("qwen3.6-35b-a3b", "int4-spec050", "Sm86"),
        ("qwen3.6-35b-a3b", "int4-spec075", "Sm86"),
    ];

    let combo_rows: Vec<&str> = bench_text
        .split("ComboDescriptor {")
        .skip(1)
        .filter_map(|rest| rest.split_once("},").map(|(row, _)| row))
        .collect();

    for (model, quant, arch) in expected_rows {
        let model_needle = format!("model: \"{model}\"");
        let quant_needle = format!("quant: \"{quant}\"");
        let arch_needle = format!("BenchArch::{arch}");
        assert!(
            combo_rows.iter().any(|row| row.contains(&model_needle)
                && row.contains(&quant_needle)
                && row.contains(&arch_needle)),
                "bench/src/matrix.rs is missing combo: {model}/{quant}/{arch}\n\
                 If runner support changed: add the row to crates/bench/src/matrix.rs::SUPPORTED_COMBOS, \
                 and if support was REMOVED upstream also remove the corresponding entry from \
                 the expected_rows list in this file."
        );
    }

    // Bidirectional gate: the bench combo table must not have unexpected extras.
    // If a new row was added to SUPPORTED_COMBOS without updating expected_pairs,
    // this assertion catches it. Counts entry rows by the `model: "` literal that
    // only appears in the static table (struct field declarations use a different
    // shape: `pub model: &'static str`).
    let actual_count = combo_rows.len();
    assert_eq!(
        actual_count,
        expected_rows.len(),
        "matrix.rs has {actual_count} combo rows but expected_rows has {} entries; \
         either remove the extra rows from SUPPORTED_COMBOS or add the corresponding (model, quant) \
         pairs to expected_rows in this file.",
        expected_rows.len()
    );
}
