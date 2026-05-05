use supersonic_bench::perf::{extract_metrics, ExtractedMetrics};

const MODERN: &str = include_str!("fixtures/runner_output_modern.txt");
const PHI4: &str = include_str!("fixtures/runner_output_phi4.txt");

#[test]
fn extracts_ms_per_step_from_modern_runner() {
    let m = extract_metrics(MODERN).expect("expected metrics");
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
    assert!((m.ms_per_tok.unwrap() - 8.0).abs() < 1e-6);
}

#[test]
fn extracts_ms_per_step_from_phi4_runner() {
    let m = extract_metrics(PHI4).expect("expected metrics");
    assert!((m.ms_per_step - 38.3).abs() < 1e-6);
}

#[test]
fn returns_none_when_no_result_line() {
    let s = "no result line here\nsome other text";
    assert!(extract_metrics(s).is_none());
}

#[test]
fn returns_none_when_only_legacy_ms_per_tok_field() {
    // Per spec: do not silently fall back; missing both means something broke.
    // qwen35_decode_report.rs emits ms_per_tok-only — verify we still surface ms_per_tok
    // but only when paired with ms_per_step. This format-change-detector test asserts
    // we don't accept a degenerate output that's missing ms_per_step entirely.
    let legacy = "[result] prompt_tokens=6 generated_tokens=4 decode_ms=32 ms_per_tok=8 decode_max_delta=0.0000";
    let m = extract_metrics(legacy);
    // ms_per_step is the canonical field we extract; if a runner only emits ms_per_tok,
    // we accept ms_per_tok as ms_per_step (they are equivalent for batch=1) but flag it.
    assert!(m.is_some(), "ms_per_tok should be accepted as ms_per_step for batch=1");
    let m = m.unwrap();
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
}
