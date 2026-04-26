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

    #[arg(long, default_value_t = 0)]
    decode_tokens: usize,

    #[arg(long)]
    profile_ops: bool,

    #[arg(long)]
    profile_layers: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.backend == BackendArg::Metal
        && !matches!(cli.mode, BughuntMode::Bench)
        && std::env::var_os("SUPERSONIC_METAL_FORCE_F32_FINAL_NORM").is_none()
    {
        // The bug-hunt gate is a prefill-state parity harness. Keep its final
        // logit comparison on the F32 reference projection path so a known
        // low-precision Metal lm-head path cannot mask or fabricate oracle
        // failures.
        std::env::set_var("SUPERSONIC_METAL_FORCE_F32_FINAL_NORM", "1");
    }
    if cli.backend == BackendArg::Metal
        && matches!(cli.mode, BughuntMode::Bench)
        && cli.profile_layers
        && std::env::var_os("SUPERSONIC_METAL_PROFILE_FLUSH_LAYERS").is_none()
    {
        std::env::set_var("SUPERSONIC_METAL_PROFILE_FLUSH_LAYERS", "1");
    }
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
        bench_decode_tokens: cli.decode_tokens,
        bench_profile_ops: cli.profile_ops,
    })?;

    let exit_code = report.exit_code();
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}
