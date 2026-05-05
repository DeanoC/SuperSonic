use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetrics {
    pub ms_per_step: f64,
    pub ms_per_tok: Option<f64>,
}

/// Parse the last `[result] ms_per_step=N ...` line from the combined
/// stdout+stderr output of a `supersonic` subprocess. Most runner engines
/// emit `[result]` via `eprintln!`, so callers MUST pass the merged stream.
/// Returns `None` if no `[result]` line is present.
/// Falls back to `ms_per_tok` for batch=1 paths that only emit that field
/// (qwen35_decode_report.rs).
pub fn extract_metrics(output: &str) -> Option<ExtractedMetrics> {
    let result_line = output.lines().rev().find(|l| l.starts_with("[result]"))?;
    let ms_per_step = parse_field(result_line, "ms_per_step");
    let ms_per_tok = parse_field(result_line, "ms_per_tok");
    match (ms_per_step, ms_per_tok) {
        (Some(s), t) => Some(ExtractedMetrics { ms_per_step: s, ms_per_tok: t.or(Some(s)) }),
        (None, Some(t)) => Some(ExtractedMetrics { ms_per_step: t, ms_per_tok: Some(t) }),
        (None, None) => None,
    }
}

fn parse_field(line: &str, key: &str) -> Option<f64> {
    // Assumes runner emits space-delimited key=value pairs (no trailing punctuation).
    let needle = format!("{key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[doc(hidden)]
pub fn run_one_combo(_model: &str, _quant: &str) -> Result<ExtractedMetrics> {
    Err(anyhow!("not implemented yet — Task 5"))
}
