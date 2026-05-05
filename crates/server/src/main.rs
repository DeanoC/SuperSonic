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

    /// Compute backend (`auto`, `hip`, `cuda`, `metal`).
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

    /// Use a GGUF-like Q4KM bake in SuperSonic's native low-bit runtime layout.
    #[arg(long)]
    q4km: bool,

    /// Use a Q4KM-sourced GPTQ bake in SuperSonic's native INT4 runtime layout.
    #[arg(long)]
    q4km_gptq: bool,

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

    /// Optional CORS allow-origin value for browser clients, e.g.
    /// `http://localhost:3000` or `*`. Also read from
    /// `SUPERSONIC_CORS_ALLOW_ORIGIN`.
    #[arg(long, env = "SUPERSONIC_CORS_ALLOW_ORIGIN")]
    cors_allow_origin: Option<String>,

    /// Maximum number of `/v1/responses` objects retained in memory.
    #[arg(
        long,
        env = "SUPERSONIC_RESPONSE_STORE_MAX_ENTRIES",
        default_value_t = 1024
    )]
    response_store_max_entries: usize,

    /// Maximum number of requests allowed to wait for the single GPU
    /// generation slot before new requests receive 429.
    #[arg(long, env = "SUPERSONIC_MAX_QUEUED_REQUESTS", default_value_t = 32)]
    max_queued_requests: usize,

    /// Maximum time a request may wait in the generation queue before it
    /// fails. Also read from `SUPERSONIC_QUEUE_TIMEOUT_MS`.
    #[arg(long, env = "SUPERSONIC_QUEUE_TIMEOUT_MS", default_value_t = 30_000)]
    queue_timeout_ms: u64,

    /// Disable automatic download of pre-baked weights from the GitHub
    /// `bakes-v{FORMAT_VERSION}` release. With this set, a missing INT4 bake
    /// produces a hard error instead of a fetch.
    #[arg(long)]
    no_download: bool,

    /// Disable exact-prefix cache reuse for chat/agent loops.
    #[arg(long, env = "SUPERSONIC_PREFIX_CACHE_DISABLE")]
    prefix_cache_disable: bool,

    /// Prefix cache directory. Defaults to
    /// `{model-dir}/.supersonic/serve-cache/v1`.
    #[arg(long, env = "SUPERSONIC_PREFIX_CACHE_DIR")]
    prefix_cache_dir: Option<PathBuf>,

    /// Minimum prompt prefix length eligible for caching.
    #[arg(
        long,
        env = "SUPERSONIC_PREFIX_CACHE_MIN_TOKENS",
        default_value_t = 128
    )]
    prefix_cache_min_tokens: usize,

    /// Maximum number of resident prefix snapshots. Snapshots clone model
    /// state on GPU, so the default is intentionally conservative.
    #[arg(
        long,
        env = "SUPERSONIC_PREFIX_CACHE_MAX_ENTRIES",
        default_value_t = 1
    )]
    prefix_cache_max_entries: usize,

    /// Maximum resident prefix snapshot bytes. Defaults to an automatic
    /// conservative VRAM budget; set to 0 to disable the byte cap.
    #[arg(long, env = "SUPERSONIC_PREFIX_CACHE_MAX_BYTES")]
    prefix_cache_max_bytes: Option<usize>,

    /// In-memory prefix cache TTL in seconds.
    #[arg(
        long,
        env = "SUPERSONIC_PREFIX_CACHE_MEMORY_TTL_SECS",
        default_value_t = 600
    )]
    prefix_cache_memory_ttl_secs: u64,

    /// Disk-retained prefix cache metadata TTL in seconds.
    #[arg(
        long,
        env = "SUPERSONIC_PREFIX_CACHE_DISK_TTL_SECS",
        default_value_t = 86_400
    )]
    prefix_cache_disk_ttl_secs: u64,
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
        q4km: cli.q4km,
        q4km_gptq: cli.q4km_gptq,
        fp8_runtime: cli.fp8_runtime,
        kv_fp8: cli.kv_fp8,
        api_key: cli.api_key,
        cors_allow_origin: cli.cors_allow_origin,
        response_store_max_entries: cli.response_store_max_entries,
        max_queued_requests: cli.max_queued_requests,
        queue_timeout_ms: cli.queue_timeout_ms,
        no_download: cli.no_download,
        prefix_cache_enabled: !cli.prefix_cache_disable,
        prefix_cache_dir: cli.prefix_cache_dir,
        prefix_cache_min_tokens: cli.prefix_cache_min_tokens,
        prefix_cache_max_entries: cli.prefix_cache_max_entries,
        prefix_cache_max_bytes: cli.prefix_cache_max_bytes,
        prefix_cache_memory_ttl_secs: cli.prefix_cache_memory_ttl_secs,
        prefix_cache_disk_ttl_secs: cli.prefix_cache_disk_ttl_secs,
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
