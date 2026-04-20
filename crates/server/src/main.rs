//! `supersonic-serve` — long-lived OpenAI-compatible HTTP server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;

use server::routes;
use server::state::{self, LoaderConfig};

#[derive(Parser, Debug)]
#[command(
    name = "supersonic-serve",
    about = "SuperSonic — OpenAI-compatible inference server"
)]
struct Cli {
    /// Model variant (e.g. "qwen3.5-0.8b", "gemma4-e2b").
    #[arg(long)]
    model: String,

    /// Path to the HuggingFace model directory (config.json + safetensors
    /// or a pre-baked `.supersonic/` subdirectory).
    #[arg(long)]
    model_dir: PathBuf,

    /// Compute backend (`auto`, `hip`, `cuda`).
    #[arg(long, default_value = "auto")]
    backend: String,

    /// GPU device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Maximum context length (prompt + generated). Drives KV cache sizing
    /// and per-request bounds checks.
    #[arg(long, default_value_t = 4096)]
    max_context: usize,

    /// Use the INT4 GPTQ bake (requires a pre-existing bake).
    #[arg(long)]
    int4: bool,

    /// Keep FP8 weights on GPU and dequant at runtime (Qwen3.5 only).
    #[arg(long)]
    fp8_runtime: bool,

    /// Store KV cache in FP8 E4M3 with per-head scaling (Qwen3.5 only).
    #[arg(long)]
    kv_fp8: bool,

    /// Listen address.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Optional shared bearer token. When set, requests must send
    /// `Authorization: Bearer <token>`. Also read from `SUPERSONIC_API_KEY`.
    #[arg(long, env = "SUPERSONIC_API_KEY")]
    api_key: Option<String>,

    /// Disable automatic download of pre-baked weights from the GitHub
    /// `bakes-v{FORMAT_VERSION}` release. With this set, a missing INT4 bake
    /// produces a hard error instead of a fetch.
    #[arg(long)]
    no_download: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "supersonic_serve=info,server=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let loader = LoaderConfig {
        model: cli.model,
        model_dir: cli.model_dir,
        backend: cli.backend,
        device: cli.device,
        max_context: cli.max_context,
        int4: cli.int4,
        fp8_runtime: cli.fp8_runtime,
        kv_fp8: cli.kv_fp8,
        api_key: cli.api_key,
        no_download: cli.no_download,
    };

    let st = state::build(loader).context("build server state")?;
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .with_context(|| format!("invalid --host/--port: {}:{}", cli.host, cli.port))?;

    let app = routes::router(Arc::new(st));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind {addr}"))?;
        tracing::info!("supersonic-serve listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("axum serve")?;
        Ok::<_, anyhow::Error>(())
    })
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("ctrl-c received, shutting down");
}
