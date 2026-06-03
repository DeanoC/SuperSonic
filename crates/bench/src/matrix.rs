//! Bench-side mirror of the runner's (model, quant, arch) support matrix.
//!
//! INVARIANT: this table must match `crates/runner/src/registry.rs`'s REGISTRY
//! plus the per-family quant flags. A parity test in
//! `crates/runner/tests/bench_combo_parity.rs` enforces this — if you change
//! the runner registry, you MUST update this table or the test fails.

use crate::perf::{run_one_combo, ComboInvocation, RunPolicy};
use crate::runs::{MetaJson, PerfCellJson, PerfStatus, RunDir, SCHEMA_VERSION};
use anyhow::Result;
use chrono::Utc;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchArch {
    Gfx1100,
    Gfx1150,
    Sm86,
    AppleM4,
    AppleM5Max,
}

impl BenchArch {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "gfx1100" => Self::Gfx1100,
            "gfx1150" => Self::Gfx1150,
            "sm86" => Self::Sm86,
            "apple-m4" => Self::AppleM4,
            "apple-m5-max" => Self::AppleM5Max,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1150 => "gfx1150",
            Self::Sm86 => "sm86",
            Self::AppleM4 => "apple-m4",
            Self::AppleM5Max => "apple-m5-max",
        }
    }

    pub fn backend(&self) -> Option<&'static str> {
        match self {
            Self::Gfx1100 | Self::Gfx1150 => Some("hip"),
            Self::Sm86 => Some("cuda"),
            Self::AppleM4 | Self::AppleM5Max => Some("metal"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComboDescriptor {
    pub model: &'static str, // e.g. "qwen3.5-0.8b"
    pub quant: &'static str, // "bf16" | "int4" | "fp8r" | "kv-fp8" | "int8"
    pub arch: BenchArch,
    pub min_vram_gib: f64,
}

/// Mirrors docs/feature-compatibility.md + docs/performance.md as of 2026-05-05.
pub static SUPPORTED_COMBOS: &[ComboDescriptor] = &[
    // Qwen3.5 — full BF16/INT4/FP8r/KV-FP8 quad on gfx1100.
    ComboDescriptor {
        model: "qwen3.5-0.8b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 2.0,
    },
    ComboDescriptor {
        model: "qwen3.5-0.8b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 0.7,
    },
    ComboDescriptor {
        model: "qwen3.5-0.8b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 1.2,
    },
    ComboDescriptor {
        model: "qwen3.5-0.8b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 2.0,
    },
    ComboDescriptor {
        model: "qwen3.5-2b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 5.0,
    },
    ComboDescriptor {
        model: "qwen3.5-2b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 1.9,
    },
    ComboDescriptor {
        model: "qwen3.5-2b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 3.0,
    },
    ComboDescriptor {
        model: "qwen3.5-2b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 5.0,
    },
    ComboDescriptor {
        model: "qwen3.5-4b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 10.0,
    },
    ComboDescriptor {
        model: "qwen3.5-4b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 3.7,
    },
    ComboDescriptor {
        model: "qwen3.5-4b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 6.0,
    },
    ComboDescriptor {
        model: "qwen3.5-4b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 10.0,
    },
    ComboDescriptor {
        model: "qwen3.5-9b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 18.0,
    },
    ComboDescriptor {
        model: "qwen3.5-9b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 6.7,
    },
    ComboDescriptor {
        model: "qwen3.5-9b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 10.8,
    },
    ComboDescriptor {
        model: "qwen3.5-9b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 18.0,
    },
    // Gemma 4 — fp8r and kv-fp8 are wired into the single-batch persistent
    // decode kernel only (require --batch-size=1, cannot combine with --int4).
    // See docs/supported-matrix.md footnote 2.
    ComboDescriptor {
        model: "gemma4-e2b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 11.0,
    },
    ComboDescriptor {
        model: "gemma4-e2b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 4.1,
    },
    ComboDescriptor {
        model: "gemma4-e2b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 6.6,
    },
    ComboDescriptor {
        model: "gemma4-e2b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 11.0,
    },
    ComboDescriptor {
        model: "gemma4-e4b",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 10.0,
    },
    ComboDescriptor {
        model: "gemma4-e4b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 3.7,
    },
    ComboDescriptor {
        model: "gemma4-e4b",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 6.0,
    },
    ComboDescriptor {
        model: "gemma4-e4b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 10.0,
    },
    // Phi-4-mini — full quad
    ComboDescriptor {
        model: "phi4-mini",
        quant: "bf16",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 8.0,
    },
    ComboDescriptor {
        model: "phi4-mini",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 3.0,
    },
    ComboDescriptor {
        model: "phi4-mini",
        quant: "fp8r",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 4.8,
    },
    ComboDescriptor {
        model: "phi4-mini",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 8.0,
    },
    // Qwen3.6-MoE — INT4 + KV-FP8 only on gfx1100 (24 GiB cap).
    // KV-FP8 lane requires --int4 simultaneously (only quant lane shipped).
    // See docs/feature-compatibility.md footnote 4.
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 21.0,
    },
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "kv-fp8",
        arch: BenchArch::Gfx1100,
        min_vram_gib: 21.0,
    },
    // Apple M5 Max Metal v1: Qwen3.6 INT4 chained decode only. This is the
    // main-target harness lane; persistent decode, KV-FP8, speculative decode,
    // batching, and Metal VMM stay explicitly out of scope for this row.
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4",
        arch: BenchArch::AppleM5Max,
        min_vram_gib: 21.0,
    },
    // Public M5 Max comparison target: Qwen3.5-35B-A3B with Q4_K_M-sourced
    // GPTQ sidecars, because Metal dense projection kernels consume native
    // INT4 sidecars rather than raw GGML K-blocks.
    ComboDescriptor {
        model: "qwen3.5-35b-a3b",
        quant: "q4km-gptq",
        arch: BenchArch::AppleM5Max,
        min_vram_gib: 21.0,
    },
    // Raw GGUF Q4_K_M staged Metal lane. This is intentionally separate from
    // the Q4_K_M-sourced GPTQ control row above so external-equivalence work
    // does not blur into the faster native-sidecar path.
    ComboDescriptor {
        model: "qwen3.5-35b-a3b",
        quant: "q4km",
        arch: BenchArch::AppleM5Max,
        min_vram_gib: 21.0,
    },
    // CUDA sm86 Qwen3.6-MoE prefill lanes. `int4-specNNN` is INT4 plus
    // Qwen3.5-0.8B cross-family SpecPrefill cosine keep ratio NNN/100.
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4",
        arch: BenchArch::Sm86,
        min_vram_gib: 21.0,
    },
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4-spec025",
        arch: BenchArch::Sm86,
        min_vram_gib: 21.0,
    },
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4-spec050",
        arch: BenchArch::Sm86,
        min_vram_gib: 21.0,
    },
    ComboDescriptor {
        model: "qwen3.6-35b-a3b",
        quant: "int4-spec075",
        arch: BenchArch::Sm86,
        min_vram_gib: 21.0,
    },
];

pub fn combos_for_arch(arch: BenchArch) -> Vec<&'static ComboDescriptor> {
    SUPPORTED_COMBOS.iter().filter(|c| c.arch == arch).collect()
}

/// True iff `(model, quant, arch)` can be run by `bench-perf`. Most entries
/// must appear in `SUPPORTED_COMBOS`; sm86 Qwen3.6 also accepts ad hoc
/// `int4-specNNN` lanes so local sweeps can try intermediate keep ratios
/// without adding every exploratory value to the shipping matrix.
pub fn is_supported_combo(model: &str, quant: &str, arch: &BenchArch) -> bool {
    if model == "qwen3.6-35b-a3b"
        && *arch == BenchArch::Sm86
        && parse_specprefill_quant(quant).is_some()
    {
        return true;
    }

    SUPPORTED_COMBOS
        .iter()
        .any(|c| c.model == model && c.quant == quant && c.arch == *arch)
}

fn parse_specprefill_quant(quant: &str) -> Option<u32> {
    let suffix = quant.strip_prefix("int4-spec")?;
    if suffix.len() != 3 || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let keep_percent = suffix.parse::<u32>().ok()?;
    (5..=100).contains(&keep_percent).then_some(keep_percent)
}

pub struct MatrixConfig {
    pub arch: BenchArch,
    pub models: Vec<String>,
    pub quants: Vec<String>,
    pub binary: PathBuf,
    pub model_dir_resolver: Box<dyn Fn(&str) -> Result<PathBuf>>,
    pub specprefill_draft_dir_resolver: Box<dyn Fn(&str) -> Option<PathBuf>>,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub context_size: Option<u32>,
    pub warmup_tokens: u32,
    pub measurement_runs: u32,
    pub cooldown_seconds: u32,
    pub collect_attribution: bool,
    pub git_sha: String,
    pub git_dirty: bool,
    pub git_dirty_paths: Vec<String>,
    pub git_diff_hash: Option<String>,
    pub runner_version: String,
}

pub fn run_matrix(cfg: &MatrixConfig, rd: &RunDir) -> Result<()> {
    rd.create()?;

    let mut meta = MetaJson {
        schema_version: SCHEMA_VERSION,
        run_id: rd
            .root()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string(),
        timestamp_utc: Utc::now().to_rfc3339(),
        git_sha: cfg.git_sha.clone(),
        git_dirty: cfg.git_dirty,
        git_dirty_paths: cfg.git_dirty_paths.clone(),
        git_diff_hash: cfg.git_diff_hash.clone(),
        hostname: hostname_or_unknown(),
        arch: cfg.arch.as_str().to_string(),
        rocminfo: capture_cmd("rocminfo"),
        rocm_smi_u: capture_cmd_args("rocm-smi", &["-u"]),
        gpu_temp_c_pre: read_gpu_temp(),
        gpu_temp_c_post: None,
        runner_version: cfg.runner_version.clone(),
    };
    rd.write_meta(&meta)?;

    for model in &cfg.models {
        for quant in &cfg.quants {
            // Skip (model, quant, arch) triples that are not in the registry.
            // bench-perf's --models / --quants flags are independent sets, so
            // their Cartesian product can include unsupported pairs (e.g.
            // qwen3.6-35b-a3b + bf16). Record a Skipped cell rather than
            // letting the runner produce a useless Error cell.
            if !is_supported_combo(model, quant, &cfg.arch) {
                let cell = PerfCellJson {
                    schema_version: SCHEMA_VERSION,
                    model: model.clone(),
                    quant: quant.clone(),
                    arch: cfg.arch.as_str().to_string(),
                    backend: cfg.arch.backend().unwrap_or("auto").to_string(),
                    prompt: cfg.prompt.clone(),
                    max_new_tokens: cfg.max_new_tokens,
                    status: PerfStatus::Skipped {
                        reason: format!(
                            "unsupported combo for {}: ({}, {}) not in SUPPORTED_COMBOS",
                            cfg.arch.as_str(),
                            model,
                            quant
                        ),
                    },
                    stage_timings: None,
                    chain_breakdown: None,
                    lifecycle_timings: None,
                    profile_stage_timings: None,
                    profile_chain_breakdown: None,
                    profile_lifecycle_timings: None,
                    mpp_pilot: None,
                    mps_expert_pilot: None,
                    qwen36_pack_cache: None,
                    qwen36_expert_residency: None,
                    qwen36_expert_residency_policies: None,
                    qwen36_expert_residency_policy_rows: None,
                    metal_profile: None,
                    hal_profile: None,
                    gpu_temp_c_end: None,
                };
                rd.write_perf(&cell)?;
                continue;
            }

            let invocation = ComboInvocation {
                binary: cfg.binary.clone(),
                backend: cfg.arch.backend().map(str::to_string),
                arch: cfg.arch.as_str().to_string(),
                model: model.clone(),
                model_dir: (cfg.model_dir_resolver)(model)?,
                quant: quant.clone(),
                specprefill_draft_dir: (cfg.specprefill_draft_dir_resolver)(model),
                prompt: cfg.prompt.clone(),
                max_new_tokens: cfg.max_new_tokens,
                context_size: cfg.context_size,
                warmup_tokens: cfg.warmup_tokens,
            };
            let policy = RunPolicy {
                measurement_runs: cfg.measurement_runs,
                cooldown_seconds: cfg.cooldown_seconds,
                collect_attribution: cfg.collect_attribution,
            };
            let cell = run_one_combo(&invocation, &policy)?;
            rd.write_perf(&cell)?;
        }
    }

    meta.gpu_temp_c_post = read_gpu_temp();
    rd.write_meta(&meta)?;
    Ok(())
}

fn hostname_or_unknown() -> String {
    capture_cmd("hostname").trim().to_string()
}

fn capture_cmd(name: &str) -> String {
    capture_cmd_args(name, &[])
}

fn capture_cmd_args(name: &str, args: &[&str]) -> String {
    Command::new(name)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_else(|_| String::new())
}

fn read_gpu_temp() -> Option<f64> {
    let out = capture_cmd_args("rocm-smi", &["-t", "--json"]);
    // Best-effort: parse `"Temperature (Sensor edge) (C)": "XX.X"` or similar.
    // If parse fails, return None — the field is optional.
    serde_json::from_str::<serde_json::Value>(&out)
        .ok()
        .and_then(|v| {
            v.as_object()?
                .values()
                .next()?
                .as_object()?
                .iter()
                .find(|(k, _)| k.contains("Temperature"))
                .and_then(|(_, v)| v.as_str()?.parse().ok())
        })
}
