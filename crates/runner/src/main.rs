#![recursion_limit = "512"]

mod backend_runtime;
mod bakes;
mod cli;
mod decode_engine;
mod model_files;
mod profiling;
mod qwen35_decode_report;
mod qwen35_engine_setup;
mod qwen35_prefill;
mod qwen35_runtime;
mod qwen35_startup;
mod registry;

use anyhow::Result;
use clap::Parser;

use backend_runtime::{install_arch_profile, lookup_registry_entry, query_gpu_info};
use cli::Cli;
use model_files::validate_input_contract;
use qwen35_runtime::run_qwen35;
use registry::Backend;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Keep all artifact/configuration errors on the host side. This must run
    // before HIP setup or any registry/engine path that can allocate buffers.
    validate_input_contract(&cli)?;

    // HIP is implicit in the product contract; there is no backend argument
    // or environment fallback in public startup.
    let gpu = query_gpu_info(cli.device)?;
    let entry = lookup_registry_entry(&gpu.gpu_arch)?;
    install_arch_profile(entry);

    run_qwen35(&cli, entry, cli.device)
}
