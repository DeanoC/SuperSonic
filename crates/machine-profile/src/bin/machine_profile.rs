use clap::{Parser, Subcommand};
use machine_profile::{measure, store};

#[derive(Parser)]
#[command(name = "machine-profile")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure the local machine and write to the cache.
    Run {
        /// Also publish a sanitized copy to <repo>/profiles/.
        #[arg(long)]
        publish: bool,
        /// Path to repo root (for --publish). Defaults to CWD.
        #[arg(long)]
        repo: Option<std::path::PathBuf>,
    },
    /// Print the cached profile (or the freshly measured one).
    Show {
        #[arg(long)]
        raw: bool,
    },
    /// Print the current machine fingerprint.
    Fingerprint,
    /// Delete the cache directory.
    ClearCache,
    /// Print the catalog (known device theoretical peaks).
    Catalog,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { publish, repo } => {
            let profile = measure();
            let dir = store::cache_dir();
            let path = store::save(&profile, &dir)?;
            println!("wrote {}", path.display());
            if publish {
                let repo = repo.unwrap_or_else(|| std::env::current_dir().unwrap());
                let p = store::publish_to(&profile, &repo)?;
                println!("published {}", p.display());
            }
        }
        Cmd::Show { raw } => {
            let profile = measure();
            if raw {
                println!("{}", serde_json::to_string_pretty(&profile)?);
            } else {
                pretty_print(&profile);
            }
        }
        Cmd::Fingerprint => {
            let profile = measure();
            println!("{}", profile.fingerprint);
        }
        Cmd::ClearCache => {
            let dir = store::cache_dir();
            if dir.exists() {
                std::fs::remove_dir_all(&dir)?;
            }
            println!("cleared {}", dir.display());
        }
        Cmd::Catalog => {
            println!("(catalog listing — see crates/machine-profile/src/catalog.rs)");
        }
    }
    Ok(())
}

fn pretty_print(p: &machine_profile::Profile) {
    println!("fingerprint: {}", p.fingerprint);
    println!("captured_at: {}", p.captured_at);
    if let Some(cpu) = &p.cpu {
        println!(
            "cpu:        {} {} ({} cores)",
            cpu.vendor, cpu.model, cpu.topology.cores_total
        );
        if let Some(fp32) = &cpu.vector_peak.fp32 {
            println!(
                "  fp32:     {:.1} GFLOPS aggregate",
                fp32.measured_aggregate
            );
        }
    }
    for g in &p.gpus {
        println!(
            "gpu[{}]:    {} {} ({} CUs, wave {})",
            g.device_index, g.backend, g.arch_name, g.cu_count, g.wave_size
        );
        if let Some(r) = g.vram_bw.read_gb_s {
            println!("  hbm read: {:.1} GB/s", r);
        }
        if let Some(m) = &g.mma_peak.bf16 {
            println!("  bf16 mma: {:.1} TFLOPS", m.measured_tflops);
        }
        if let Some(metal) = &g.metal {
            if let Some(family) = &metal.metal_family {
                println!("  metal:    {family}");
            }
            if let Some(supported) = metal.simdgroup_matrix_supported {
                println!("  simd mat: {supported}");
            }
            if let Some(supported) = metal.mpp_tensor_matmul_supported {
                println!("  mpp mat:  {supported}");
            }
            if let Some(status) = &metal.mpp_tensor_matmul_probe_status {
                println!("  mpp probe: {status}");
            }
            if let Some(value) = metal.mpp_tensor_write_probe_value {
                println!("  mpp tensor write: {value:.1}");
            }
            if let Some(value) = metal.mpp_tensor_matmul_probe_value {
                println!("  mpp matmul value: {value:.1}");
            }
        }
        for mk in &g.microkernels {
            match (mk.measured_gb_s, mk.measured_tflops) {
                (Some(gb_s), _) => println!("  {:<28} {:>8.1} GB/s", mk.name, gb_s),
                (_, Some(tflops)) => println!("  {:<28} {:>8.1} TFLOPS", mk.name, tflops),
                _ => {}
            }
        }
    }
    for w in &p.warnings {
        println!("warning [{}]: {}", w.component, w.reason);
    }
}
