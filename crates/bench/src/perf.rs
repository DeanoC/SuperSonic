use crate::runs::{PerfCellJson, PerfStatus, SCHEMA_VERSION};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

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
    if let Some(result_line) = output.lines().rev().find(|l| l.starts_with("[result]")) {
        let ms_per_step = parse_field(result_line, "ms_per_step");
        let ms_per_tok = parse_field(result_line, "ms_per_tok");
        return match (ms_per_step, ms_per_tok) {
            (Some(s), t) => Some(ExtractedMetrics {
                ms_per_step: s,
                ms_per_tok: t.or(Some(s)),
            }),
            (None, Some(t)) => Some(ExtractedMetrics {
                ms_per_step: t,
                ms_per_tok: Some(t),
            }),
            (None, None) => None,
        };
    }

    let lifecycle_line = output
        .lines()
        .rev()
        .find(|l| l.starts_with("[qwen36-moe lifecycle-timings]"))?;
    parse_field(lifecycle_line, "prefill_total_ms").map(|s| ExtractedMetrics {
        ms_per_step: s,
        ms_per_tok: Some(s),
    })
}

fn parse_field(line: &str, key: &str) -> Option<f64> {
    // Assumes runner emits space-delimited key=value pairs (no trailing punctuation).
    let needle = format!("{key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[derive(Debug, Clone)]
pub struct ComboInvocation {
    pub binary: PathBuf,
    pub backend: Option<String>,
    pub model: String,
    pub model_dir: PathBuf,
    pub quant: String,
    pub specprefill_draft_dir: Option<PathBuf>,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub warmup_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct RunPolicy {
    pub measurement_runs: u32,
    pub cooldown_seconds: u32,
}

pub fn run_one_combo(invocation: &ComboInvocation, policy: &RunPolicy) -> Result<PerfCellJson> {
    if policy.cooldown_seconds > 0 {
        std::thread::sleep(Duration::from_secs(policy.cooldown_seconds as u64));
    }

    // Warmup pass — discard.
    let _ = invoke_supersonic(invocation, invocation.warmup_tokens);

    let mut samples = Vec::new();
    let mut last_err: Option<String> = None;
    for _ in 0..policy.measurement_runs {
        match invoke_supersonic(invocation, invocation.max_new_tokens) {
            Ok(m) => samples.push(m.ms_per_step),
            Err(e) => last_err = Some(e),
        }
    }

    let status = if samples.is_empty() {
        PerfStatus::Error {
            stderr_tail: last_err.unwrap_or_else(|| "no samples".into()),
        }
    } else {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        PerfStatus::Ok {
            ms_per_step: median,
            ms_per_tok: median,
            samples,
        }
    };

    Ok(PerfCellJson {
        schema_version: SCHEMA_VERSION,
        model: invocation.model.clone(),
        quant: invocation.quant.clone(),
        prompt: invocation.prompt.clone(),
        max_new_tokens: invocation.max_new_tokens,
        status,
        gpu_temp_c_end: None,
    })
}

fn invoke_supersonic(
    invocation: &ComboInvocation,
    max_new: u32,
) -> std::result::Result<ExtractedMetrics, String> {
    let mut cmd = Command::new(&invocation.binary);
    if let Some(backend) = &invocation.backend {
        cmd.arg("--backend").arg(backend);
    }
    cmd.arg("--model")
        .arg(&invocation.model)
        .arg("--model-dir")
        .arg(&invocation.model_dir)
        .arg("--prompt")
        .arg(&invocation.prompt)
        .arg("--max-new-tokens")
        .arg(max_new.to_string())
        .arg("--emit-stage-timings");
    apply_quant_flag(
        &mut cmd,
        &invocation.quant,
        invocation.specprefill_draft_dir.as_deref(),
    );
    let out = cmd.output().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");
    extract_metrics(&combined).ok_or_else(|| {
        let tail: String = combined
            .lines()
            .rev()
            .take(50)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        format!("no [result] line; tail:\n{tail}")
    })
}

fn apply_quant_flag(
    cmd: &mut Command,
    quant: &str,
    specprefill_draft_dir: Option<&std::path::Path>,
) {
    match quant {
        "bf16" => {} // default
        "int4" => {
            cmd.arg("--int4");
        }
        "fp8r" => {
            cmd.arg("--fp8-runtime");
        }
        "kv-fp8" => {
            cmd.arg("--kv-fp8");
        }
        "int8" => {
            cmd.arg("--int8");
        } // Llama CUDA path
        q if q.starts_with("int4-spec") => {
            cmd.arg("--int4");
            let keep = q.trim_start_matches("int4-spec");
            let keep: f32 = keep.parse::<f32>().unwrap_or(50.0) / 100.0;
            if let Some(draft_dir) = specprefill_draft_dir {
                cmd.arg("--specprefill-draft-dir")
                    .arg(draft_dir)
                    .arg("--specprefill-algorithm")
                    .arg("cosine")
                    .arg("--specprefill-keep-ratio")
                    .arg(format!("{keep:.2}"))
                    .arg("--specprefill-unload-draft");
            } else {
                eprintln!("warn: quant '{q}' requires a SpecPrefill draft dir; running dense INT4");
            }
        }
        other => {
            eprintln!("warn: unknown quant '{other}', running BF16");
        }
    }
}
