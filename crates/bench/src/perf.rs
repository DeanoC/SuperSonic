use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetrics {
    pub ms_per_step: f64,
    pub ms_per_tok: Option<f64>,
}

/// Parse the `[result] ms_per_step=N ...` line from supersonic stdout.
/// Returns `None` if no `[result]` line is present.
/// Falls back to `ms_per_tok` for batch=1 paths that only emit that field
/// (qwen35_decode_report.rs).
pub fn extract_metrics(stdout: &str) -> Option<ExtractedMetrics> {
    let result_line = stdout.lines().rev().find(|l| l.starts_with("[result]"))?;
    let ms_per_step = parse_field(result_line, "ms_per_step");
    let ms_per_tok = parse_field(result_line, "ms_per_tok");
    match (ms_per_step, ms_per_tok) {
        (Some(s), t) => Some(ExtractedMetrics { ms_per_step: s, ms_per_tok: t.or(Some(s)) }),
        (None, Some(t)) => Some(ExtractedMetrics { ms_per_step: t, ms_per_tok: Some(t) }),
        (None, None) => None,
    }
}

fn parse_field(line: &str, key: &str) -> Option<f64> {
    let needle = format!("{key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

pub fn run_one_combo(_model: &str, _quant: &str) -> Result<ExtractedMetrics> {
    Err(anyhow!("not implemented yet — Task 5"))
}
