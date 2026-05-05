use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "bench-perf", about = "SuperSonic perf benchmark orchestrator")]
struct Cli {
    /// GPU arch filter (e.g. gfx1100). Auto-detect when not provided.
    #[arg(long)]
    arch: Option<String>,
    /// Comma-separated model list, or "all".
    #[arg(long, default_value = "all")]
    models: String,
    /// Comma-separated quant list, or "all".
    #[arg(long, default_value = "all")]
    quants: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("bench-perf invoked: arch={:?} models={} quants={}", cli.arch, cli.models, cli.quants);
    Ok(())
}
