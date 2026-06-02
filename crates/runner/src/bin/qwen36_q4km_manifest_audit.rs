use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use model_store::manifest::Manifest;
use runner::qwen36_q4km_audit::{audit_qwen36_q4km_manifest, Qwen36Q4KmAuditSpec};

#[derive(Parser, Debug)]
#[command(about = "Audit a Qwen3.5/3.6 MoE raw Q4_K_M bake manifest for Metal runtime coverage")]
struct Cli {
    /// Model directory containing config.json and .supersonic/v*-q4km.
    #[arg(long)]
    model_dir: PathBuf,
    /// Override the bake directory. Defaults to the model's q4km bake dir.
    #[arg(long)]
    bake_dir: Option<PathBuf>,
    /// Weight prefix used by the Qwen36/MoE registry entry.
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,
    /// Emit the full report as JSON.
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = qwen36_moe::config::load_config(&cli.model_dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("load config from {}", cli.model_dir.display()))?;
    let bake_dir = cli
        .bake_dir
        .unwrap_or_else(|| model_store::bake_dir_q4km(&cli.model_dir));
    let manifest_path = model_store::manifest_path(&bake_dir);
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .with_context(|| format!("parse {}", manifest_path.display()))?;

    let spec = Qwen36Q4KmAuditSpec {
        weight_prefix: cli.weight_prefix,
        layer_is_full: (0..config.text_config.num_hidden_layers)
            .map(|idx| config.text_config.is_full_attention(idx))
            .collect(),
        tied_lm_head: config.text_config.tie_word_embeddings,
    };
    let report = audit_qwen36_q4km_manifest(&manifest, &spec);

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }
    if report.summary.current_metal_blockers > 0 {
        std::process::exit(2);
    }
    Ok(())
}

fn print_text_report(report: &runner::qwen36_q4km_audit::Qwen36Q4KmAuditReport) {
    println!(
        "[q4km-audit] layers={} full={} linear={} tied_lm_head={}",
        report.num_layers,
        report.full_attention_layers,
        report.linear_attention_layers,
        report.tied_lm_head
    );
    println!(
        "[q4km-audit] projections={} native_int4={} raw_ggml={} bf16_or_unquantized={} missing={} unsupported={} blockers={}",
        report.summary.projections,
        report.summary.native_int4_sidecars,
        report.summary.raw_ggml_k_blocks,
        report.summary.bf16_or_unquantized,
        report.summary.missing,
        report.summary.unsupported_layout,
        report.summary.current_metal_blockers
    );
    for projection in report
        .projections
        .iter()
        .filter(|projection| !projection.supported_by_current_metal)
    {
        println!(
            "[q4km-audit][blocker] {} {:?} layout={} reason={}",
            projection.name,
            projection.family,
            projection.layout.as_deref().unwrap_or("missing"),
            projection.blocker.as_deref().unwrap_or("unknown")
        );
    }
}
