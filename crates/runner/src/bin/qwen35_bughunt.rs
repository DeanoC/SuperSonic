use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use runner::bughunt::{BackendArg, BughuntArgs, BughuntLayerKind, BughuntMode};

#[derive(Debug, Parser)]
#[command(
    name = "qwen35_bughunt",
    about = "Qwen3.5 0.8B Metal parity gate and localization harness"
)]
struct Cli {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long, value_enum, default_value = "metal")]
    backend: BackendArg,

    #[arg(long, default_value = "cpu")]
    oracle_device: String,

    #[arg(long, value_enum)]
    mode: BughuntMode,

    #[arg(long)]
    prompt_manifest: PathBuf,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    report_json: Option<PathBuf>,

    #[arg(long)]
    position: Option<usize>,

    #[arg(long)]
    layer: Option<usize>,

    #[arg(long, value_enum)]
    layer_kind: Option<BughuntLayerKind>,

    #[arg(long, default_value_t = 0)]
    ordinal: usize,

    #[arg(long, default_value_t = 3)]
    iters: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report = runner::bughunt::run(BughuntArgs {
        mode: cli.mode,
        model_dir: cli.model_dir,
        backend: cli.backend,
        ordinal: cli.ordinal,
        oracle_device: cli.oracle_device,
        prompt_manifest: cli.prompt_manifest,
        prompt: cli.prompt,
        report_json: cli.report_json,
        position: cli.position,
        layer: cli.layer,
        layer_kind: cli.layer_kind,
        bench_iterations: cli.iters,
        bench_warmup: cli.warmup,
    })?;

    let exit_code = report.exit_code();
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}
