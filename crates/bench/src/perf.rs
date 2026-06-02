use crate::runs::{
    PerfCellJson, PerfStatus, ProfileEntryJson, ProfileJson, Qwen36ExpertResidencyPolicyJson,
    SCHEMA_VERSION,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetrics {
    pub ms_per_step: f64,
    pub ms_per_tok: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributionTimings {
    pub stage_timings: Option<BTreeMap<String, f64>>,
    pub chain_breakdown: Option<BTreeMap<String, f64>>,
    pub lifecycle_timings: Option<BTreeMap<String, f64>>,
    pub profile_stage_timings: Option<BTreeMap<String, f64>>,
    pub profile_chain_breakdown: Option<BTreeMap<String, f64>>,
    pub profile_lifecycle_timings: Option<BTreeMap<String, f64>>,
    pub mpp_pilot: Option<BTreeMap<String, f64>>,
    pub mps_expert_pilot: Option<BTreeMap<String, f64>>,
    pub qwen36_pack_cache: Option<BTreeMap<String, f64>>,
    pub qwen36_expert_residency: Option<BTreeMap<String, f64>>,
    pub qwen36_expert_residency_policies: Option<Vec<BTreeMap<String, f64>>>,
    pub qwen36_expert_residency_policy_rows: Option<Vec<Qwen36ExpertResidencyPolicyJson>>,
    pub metal_profile: Option<ProfileJson>,
    pub hal_profile: Option<ProfileJson>,
}

#[derive(Debug, Clone)]
struct ExtractedRun {
    metrics: ExtractedMetrics,
    attribution: AttributionTimings,
}

/// Parse the last `[result] ms_per_step=N ...` line from the combined
/// stdout+stderr output of a `supersonic` subprocess. Most runner engines
/// emit `[result]` via `eprintln!`, so callers MUST pass the merged stream.
/// Returns `None` if no recognized metric line is present.
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

    if let Some(lifecycle_line) = output
        .lines()
        .rev()
        .find(|l| l.starts_with("[qwen36-moe lifecycle-timings]"))
    {
        return parse_field(lifecycle_line, "prefill_total_ms").map(|s| ExtractedMetrics {
            ms_per_step: s,
            ms_per_tok: Some(s),
        });
    }

    let mut batched_prefill_total_ms = 0.0;
    let mut saw_batched_prefill = false;
    for line in output
        .lines()
        .filter(|l| l.starts_with("[qwen36-moe batched-prefill]"))
    {
        let embed_ms = parse_field(line, "embed_ms").unwrap_or(0.0);
        let chain_ms = parse_field(line, "chain_ms")?;
        batched_prefill_total_ms += embed_ms + chain_ms;
        saw_batched_prefill = true;
    }
    if saw_batched_prefill {
        return Some(ExtractedMetrics {
            ms_per_step: batched_prefill_total_ms,
            ms_per_tok: Some(batched_prefill_total_ms),
        });
    }

    None
}

pub fn extract_attribution_timings(output: &str) -> AttributionTimings {
    AttributionTimings {
        stage_timings: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-moe stage-timings]"))
            .map(parse_numeric_fields),
        chain_breakdown: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-moe chain-breakdown]"))
            .map(parse_numeric_fields),
        lifecycle_timings: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-moe lifecycle-timings]"))
            .map(parse_numeric_fields),
        profile_stage_timings: None,
        profile_chain_breakdown: None,
        profile_lifecycle_timings: None,
        mpp_pilot: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-moe mpp-pilot]"))
            .map(parse_numeric_fields),
        mps_expert_pilot: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-moe mps-expert-pilot]"))
            .map(parse_numeric_fields),
        qwen36_pack_cache: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-pack-cache]"))
            .map(parse_numeric_fields),
        qwen36_expert_residency: output
            .lines()
            .rev()
            .find(|l| l.starts_with("[qwen36-expert-residency]"))
            .map(parse_numeric_fields),
        qwen36_expert_residency_policies: {
            let policies: Vec<_> = output
                .lines()
                .filter(|l| l.starts_with("[qwen36-expert-residency-policy]"))
                .map(parse_numeric_fields)
                .collect();
            (!policies.is_empty()).then_some(policies)
        },
        qwen36_expert_residency_policy_rows: {
            let policies: Vec<_> = output
                .lines()
                .filter(|l| l.starts_with("[qwen36-expert-residency-policy]"))
                .map(parse_qwen36_expert_residency_policy)
                .collect();
            (!policies.is_empty()).then_some(policies)
        },
        metal_profile: extract_profile(output, "[metal-profile]", "[metal-profile-op]", true),
        hal_profile: extract_profile(output, "[hal-profile]", "[hal-profile-op]", false),
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

fn parse_numeric_fields(line: &str) -> BTreeMap<String, f64> {
    line.split_whitespace()
        .filter_map(|part| {
            let part = part.trim_matches(|c| c == '(' || c == ')');
            let (key, raw) = part.split_once('=')?;
            let value = raw
                .trim_end_matches(|c: char| c == ',' || c == ')')
                .parse::<f64>()
                .ok()?;
            Some((key.to_string(), value))
        })
        .collect()
}

fn parse_string_field(line: &str, key: &str) -> String {
    let needle = format!("{key}=");
    let Some(start) = line.find(&needle).map(|idx| idx + needle.len()) else {
        return String::new();
    };
    let rest = &line[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end]
        .trim_end_matches(|c: char| c == ',' || c == ')')
        .to_string()
}

fn parse_qwen36_expert_residency_policy(line: &str) -> Qwen36ExpertResidencyPolicyJson {
    let metrics = parse_numeric_fields(line);
    Qwen36ExpertResidencyPolicyJson {
        resident_format: parse_string_field(line, "resident_format"),
        scope: parse_string_field(line, "scope"),
        miss_policy: parse_string_field(line, "miss_policy"),
        capacity: metrics.get("capacity").copied().unwrap_or_default(),
        metrics,
    }
}

fn extract_profile(
    output: &str,
    summary_prefix: &str,
    entry_prefix: &str,
    has_path: bool,
) -> Option<ProfileJson> {
    let summary_line = output
        .lines()
        .rev()
        .find(|line| line.starts_with(summary_prefix))?;
    let summary = parse_numeric_fields(summary_line);
    let entries = output
        .lines()
        .filter(|line| line.starts_with(entry_prefix))
        .filter_map(|line| parse_profile_entry(line, has_path))
        .collect();
    Some(ProfileJson { summary, entries })
}

fn parse_profile_entry(line: &str, has_path: bool) -> Option<ProfileEntryJson> {
    let fields = parse_string_fields(line);
    Some(ProfileEntryJson {
        op: fields.get("op")?.to_string(),
        path: if has_path {
            fields.get("path").map(|value| value.to_string())
        } else {
            None
        },
        calls: fields.get("calls")?.parse().ok()?,
        mean_ms: fields.get("mean_ms")?.parse().ok()?,
        total_ms: fields.get("total_ms")?.parse().ok()?,
        max_ms: fields.get("max_ms")?.parse().ok()?,
        total_bytes: fields
            .get("total_bytes")
            .and_then(|value| value.parse().ok()),
    })
}

fn parse_string_fields(line: &str) -> BTreeMap<String, String> {
    line.split_whitespace()
        .filter_map(|part| {
            let part = part.trim_matches(|c| c == '(' || c == ')');
            let (key, raw) = part.split_once('=')?;
            let value = raw.trim_end_matches(|c: char| c == ',' || c == ')');
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct ComboInvocation {
    pub binary: PathBuf,
    pub backend: Option<String>,
    pub arch: String,
    pub model: String,
    pub model_dir: PathBuf,
    pub quant: String,
    pub specprefill_draft_dir: Option<PathBuf>,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub context_size: Option<u32>,
    pub warmup_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct RunPolicy {
    pub measurement_runs: u32,
    pub cooldown_seconds: u32,
    pub collect_attribution: bool,
}

pub fn run_one_combo(invocation: &ComboInvocation, policy: &RunPolicy) -> Result<PerfCellJson> {
    if policy.cooldown_seconds > 0 {
        std::thread::sleep(Duration::from_secs(policy.cooldown_seconds as u64));
    }

    // Warmup pass — discard.
    let _ = invoke_supersonic(invocation, invocation.warmup_tokens, false, false);

    let mut samples = Vec::new();
    let mut last_err: Option<String> = None;
    for _ in 0..policy.measurement_runs {
        match invoke_supersonic(invocation, invocation.max_new_tokens, false, false) {
            Ok(run) => samples.push(run.metrics.ms_per_step),
            Err(e) => last_err = Some(e),
        }
    }

    let (status, attribution) = if samples.is_empty() {
        (
            PerfStatus::Error {
                stderr_tail: last_err.unwrap_or_else(|| "no samples".into()),
            },
            AttributionTimings::default(),
        )
    } else {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let mut attribution = if policy.collect_attribution {
            invoke_supersonic(invocation, invocation.max_new_tokens, true, false)
                .map(|run| run.attribution)
                .unwrap_or_default()
        } else {
            AttributionTimings::default()
        };
        if policy.collect_attribution && should_collect_profile_attribution(invocation) {
            if let Ok(profile_run) =
                invoke_supersonic(invocation, invocation.max_new_tokens, true, true)
            {
                let profile = profile_run.attribution;
                attribution.profile_stage_timings = profile.stage_timings;
                attribution.profile_chain_breakdown = profile.chain_breakdown;
                attribution.profile_lifecycle_timings = profile.lifecycle_timings;
                attribution.metal_profile = profile.metal_profile;
                attribution.hal_profile = profile.hal_profile;
                if attribution.mpp_pilot.is_none() {
                    attribution.mpp_pilot = profile.mpp_pilot;
                }
                if attribution.mps_expert_pilot.is_none() {
                    attribution.mps_expert_pilot = profile.mps_expert_pilot;
                }
                if attribution.qwen36_pack_cache.is_none() {
                    attribution.qwen36_pack_cache = profile.qwen36_pack_cache;
                }
                if attribution.qwen36_expert_residency.is_none() {
                    attribution.qwen36_expert_residency = profile.qwen36_expert_residency;
                }
                if attribution.qwen36_expert_residency_policies.is_none() {
                    attribution.qwen36_expert_residency_policies =
                        profile.qwen36_expert_residency_policies;
                }
                if attribution.qwen36_expert_residency_policy_rows.is_none() {
                    attribution.qwen36_expert_residency_policy_rows =
                        profile.qwen36_expert_residency_policy_rows;
                }
            }
        }
        (
            PerfStatus::Ok {
                ms_per_step: median,
                ms_per_tok: median,
                samples,
            },
            attribution,
        )
    };

    Ok(PerfCellJson {
        schema_version: SCHEMA_VERSION,
        model: invocation.model.clone(),
        quant: invocation.quant.clone(),
        arch: invocation.arch.clone(),
        backend: invocation
            .backend
            .clone()
            .unwrap_or_else(|| "auto".to_string()),
        prompt: invocation.prompt.clone(),
        max_new_tokens: invocation.max_new_tokens,
        status,
        stage_timings: attribution.stage_timings,
        chain_breakdown: attribution.chain_breakdown,
        lifecycle_timings: attribution.lifecycle_timings,
        profile_stage_timings: attribution.profile_stage_timings,
        profile_chain_breakdown: attribution.profile_chain_breakdown,
        profile_lifecycle_timings: attribution.profile_lifecycle_timings,
        mpp_pilot: attribution.mpp_pilot,
        mps_expert_pilot: attribution.mps_expert_pilot,
        qwen36_pack_cache: attribution.qwen36_pack_cache,
        qwen36_expert_residency: attribution.qwen36_expert_residency,
        qwen36_expert_residency_policies: attribution.qwen36_expert_residency_policies,
        qwen36_expert_residency_policy_rows: attribution.qwen36_expert_residency_policy_rows,
        metal_profile: attribution.metal_profile,
        hal_profile: attribution.hal_profile,
        gpu_temp_c_end: None,
    })
}

fn invoke_supersonic(
    invocation: &ComboInvocation,
    max_new: u32,
    emit_stage_timings: bool,
    metal_profile: bool,
) -> std::result::Result<ExtractedRun, String> {
    let mut cmd = Command::new(&invocation.binary);
    if let Some(backend) = &invocation.backend {
        cmd.arg("--backend").arg(backend);
    }
    if invocation.arch == "apple-m5-max"
        && matches!(
            invocation.model.as_str(),
            "qwen3.5-35b-a3b" | "qwen3.6-35b-a3b"
        )
    {
        cmd.env("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL", "0")
            .env("SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP", "1");
        if emit_stage_timings {
            cmd.env("SUPERSONIC_METAL_QWEN36_MPP_PILOT", "1")
                .env("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT", "1");
        }
        if metal_profile {
            cmd.env("SUPERSONIC_METAL_PROFILE", "1");
        }
    }
    cmd.arg("--model")
        .arg(&invocation.model)
        .arg("--model-dir")
        .arg(&invocation.model_dir)
        .arg("--prompt")
        .arg(&invocation.prompt)
        .arg("--max-new-tokens")
        .arg(max_new.to_string());
    if let Some(context_size) = invocation.context_size {
        cmd.arg("--context-size").arg(context_size.to_string());
    }
    if emit_stage_timings {
        cmd.arg("--emit-stage-timings");
    }
    apply_quant_flag(
        &mut cmd,
        &invocation.quant,
        invocation.specprefill_draft_dir.as_deref(),
    );
    let out = cmd.output().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");
    let metrics = extract_metrics(&combined).ok_or_else(|| {
        let tail: String = combined
            .lines()
            .rev()
            .take(50)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        format!("no recognized metric line; tail:\n{tail}")
    })?;
    Ok(ExtractedRun {
        metrics,
        attribution: extract_attribution_timings(&combined),
    })
}

fn should_collect_profile_attribution(invocation: &ComboInvocation) -> bool {
    invocation.arch == "apple-m5-max"
        && matches!(
            invocation.model.as_str(),
            "qwen3.5-35b-a3b" | "qwen3.6-35b-a3b"
        )
        && matches!(invocation.quant.as_str(), "int4" | "q4km" | "q4km-gptq")
        && invocation.backend.as_deref().unwrap_or("metal") == "metal"
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
        "q4km" => {
            cmd.arg("--q4km");
        }
        "q4km-gptq" => {
            cmd.arg("--q4km-gptq");
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
