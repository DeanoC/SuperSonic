//! Server startup: detect GPU, look up registry entry, load weights, build
//! the [`InferenceSession`]. Boiled-down version of the full `supersonic`
//! CLI flow in `crates/runner/src/main.rs` — the server skips the oracle,
//! tracing, and fallback paths the CLI exposes.

use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use supersonic_core::registry::{self, Backend, GpuArch, ModelFamily, ModelVariant};

use crate::backend_resolver::resolve_backend;
use crate::bakes::ensure_hf_metadata_present;
use crate::builders::{build_gemma4, build_qwen};
use crate::chat_template::ChatTemplate;
use crate::session::InferenceSession;
use supersonic_core::capabilities::{capabilities_for_variant, ModelCapabilities};

/// Per-process state shared across every HTTP request. Everything here is
/// built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub model_family: ModelFamily,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<Arc<ChatTemplate>>,
    pub session: Arc<Mutex<InferenceSession>>,
    pub eos_ids: Vec<u32>,
    pub max_context: usize,
    pub api_key: Option<String>,
    pub capabilities: ModelCapabilities,
}

/// Arguments captured from the CLI and forwarded into the loader.
pub struct LoaderConfig {
    pub model: String,
    pub model_dir: PathBuf,
    pub backend: String,
    pub device: usize,
    pub max_context: usize,
    pub int4: bool,
    pub q4km: bool,
    pub q4km_gptq: bool,
    pub fp8_runtime: bool,
    pub kv_fp8: bool,
    pub api_key: Option<String>,
    /// Disable automatic bake download from the GitHub release. Air-gapped
    /// or reproducibility-focused deploys should set this.
    pub no_download: bool,
}

/// Preferred runtime-facing name for loader configuration. `LoaderConfig`
/// remains as a compatibility alias for the existing server/tests.
pub type RuntimeConfig = LoaderConfig;

pub fn build(cfg: LoaderConfig) -> Result<ServerState> {
    validate_flag_exclusions(&cfg)?;
    /* ---- backend + GPU detection ---- */
    let backend = resolve_backend(&cfg.backend, cfg.device)?;
    gpu_hal::set_backend(backend);

    let variant = ModelVariant::from_cli_str(&cfg.model).ok_or_else(|| {
        anyhow!(
            "unknown --model '{}' (supported: {})",
            cfg.model,
            registry::supported_models_list().join(", ")
        )
    })?;
    validate_runtime_policy(&cfg, &variant, backend)?;

    let (arch_name, total_vram, warp_size) = match backend {
        Backend::Hip => {
            let (a, v) = kernel_ffi::query_gpu_info(cfg.device)
                .map_err(|e| anyhow!("GPU query failed for device {}: {}", cfg.device, e))?;
            (a, v, 32)
        }
        Backend::Cuda | Backend::Metal => {
            let info = gpu_hal::query_device_info(backend, cfg.device)
                .map_err(|e| anyhow!("GPU query failed for device {}: {}", cfg.device, e))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
    };
    let gpu_arch = GpuArch::from_backend_name(&backend, &arch_name);
    let capabilities =
        capabilities_for_variant(&variant, backend, cfg.int4, cfg.fp8_runtime, cfg.kv_fp8);
    tracing::info!(
        backend = %backend,
        device = cfg.device,
        arch = %arch_name,
        warp = warp_size,
        vram_gib = total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        "GPU detected"
    );

    let entry = registry::lookup(&variant, &backend, &gpu_arch).ok_or_else(|| {
        let archs = registry::supported_archs_for(&variant, &backend);
        anyhow!(
            "no registry entry for model={} backend={} arch={}; supported archs: [{}]",
            variant,
            backend,
            gpu_arch,
            archs.join(", ")
        )
    })?;

    /* ---- HF metadata preflight ---- */
    ensure_hf_metadata_present(&cfg)?;

    /* ---- tokenizer + chat template ---- */
    let tokenizer_path = cfg.model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!(
            "missing {} — cannot build tokenizer",
            tokenizer_path.display()
        );
    }
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let chat_template = ChatTemplate::try_load(&cfg.model_dir)?;
    if chat_template.is_none() {
        tracing::warn!(
            "no chat_template in tokenizer_config.json — /v1/chat/completions will reject \
             requests; /v1/completions still works"
        );
    }

    /* ---- engine construction ---- */
    let max_context = cfg.max_context.max(8);
    let (session, eos_ids) = match variant.family() {
        ModelFamily::Qwen35 => build_qwen(&cfg, entry, max_context)?,
        ModelFamily::Qwen36Moe => bail!("qwen3.6-35b-a3b MoE runtime is not implemented yet"),
        ModelFamily::Gemma4 => build_gemma4(&cfg, entry, max_context)?,
        ModelFamily::Phi4 => {
            bail!("Phi-4 engine is under development — not yet exposed via supersonic-serve");
        }
        ModelFamily::Llama31 => {
            bail!("Llama 3.1 is available through the supersonic CLI but is not wired into supersonic-serve yet");
        }
    };

    tracing::info!(
        model = %variant,
        family = %variant.family(),
        max_context,
        "server state ready"
    );

    Ok(ServerState {
        model_id: variant.to_string(),
        model_family: variant.family(),
        tokenizer: Arc::new(tokenizer),
        chat_template,
        session: Arc::new(Mutex::new(session)),
        eos_ids,
        max_context,
        api_key: cfg.api_key,
        capabilities,
    })
}

fn validate_flag_exclusions(cfg: &LoaderConfig) -> Result<()> {
    let q4km_like = cfg.q4km || cfg.q4km_gptq;
    if cfg.q4km && cfg.q4km_gptq {
        bail!("--q4km is mutually exclusive with --q4km-gptq");
    }
    if q4km_like && (cfg.int4 || cfg.fp8_runtime) {
        bail!("--q4km/--q4km-gptq are mutually exclusive with --int4 and --fp8-runtime");
    }
    Ok(())
}

fn validate_runtime_policy(
    cfg: &LoaderConfig,
    variant: &ModelVariant,
    backend: Backend,
) -> Result<()> {
    let q4km_like = cfg.q4km || cfg.q4km_gptq;
    if q4km_like
        && !matches!(
            variant.family(),
            ModelFamily::Qwen35 | ModelFamily::Qwen36Moe
        )
    {
        bail!("--q4km/--q4km-gptq are currently supported only for Qwen models");
    }
    if q4km_like && backend != Backend::Cuda {
        bail!("--q4km/--q4km-gptq are currently supported only on CUDA");
    }

    if backend == Backend::Metal {
        if *variant != ModelVariant::Qwen3_5_0_8B {
            bail!("Metal v1 only supports --model qwen3.5-0.8b");
        }
        if cfg.int4 {
            bail!("Metal does not support --int4 yet");
        }
        if cfg.fp8_runtime {
            bail!("Metal does not support --fp8-runtime yet");
        }
        if cfg.kv_fp8 {
            bail!("Metal does not support --kv-fp8 yet");
        }
    }

    match variant.family() {
        ModelFamily::Qwen35 | ModelFamily::Gemma4 => Ok(()),
        ModelFamily::Qwen36Moe => {
            bail!("qwen3.6-35b-a3b MoE runtime is not implemented yet")
        }
        ModelFamily::Phi4 => {
            bail!("Phi-4 engine is under development — not yet exposed via supersonic-serve");
        }
        ModelFamily::Llama31 => {
            bail!("Llama 3.1 is available through the supersonic CLI but is not wired into supersonic-serve yet");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LoaderConfig {
        LoaderConfig {
            model: "qwen3.5-0.8b".to_string(),
            model_dir: PathBuf::from("/tmp/model"),
            backend: "cuda".to_string(),
            device: 0,
            max_context: 1024,
            int4: false,
            q4km: false,
            q4km_gptq: false,
            fp8_runtime: false,
            kv_fp8: false,
            api_key: None,
            no_download: true,
        }
    }

    fn err_contains(result: Result<()>, needle: &str) {
        let err = result.expect_err("expected policy error").to_string();
        assert!(
            err.contains(needle),
            "expected error containing {needle:?}, got {err:?}"
        );
    }

    #[test]
    fn policy_rejects_q4km_flag_collisions() {
        let mut c = cfg();
        c.q4km = true;
        c.q4km_gptq = true;
        err_contains(validate_flag_exclusions(&c), "mutually exclusive");

        let mut c = cfg();
        c.q4km = true;
        c.int4 = true;
        err_contains(validate_flag_exclusions(&c), "--int4");

        let mut c = cfg();
        c.q4km_gptq = true;
        c.fp8_runtime = true;
        err_contains(validate_flag_exclusions(&c), "--fp8-runtime");
    }

    #[test]
    fn policy_rejects_q4km_for_non_qwen_models() {
        let mut c = cfg();
        c.q4km = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Gemma4_E2B, Backend::Cuda),
            "supported only for Qwen models",
        );
    }

    #[test]
    fn policy_rejects_q4km_outside_cuda() {
        let mut c = cfg();
        c.q4km_gptq = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Hip),
            "supported only on CUDA",
        );
    }

    #[test]
    fn policy_allows_supported_cuda_qwen_and_gemma_modes() {
        let mut c = cfg();
        c.q4km = true;
        validate_flag_exclusions(&c).unwrap();
        validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Cuda).unwrap();

        let mut c = cfg();
        c.int4 = true;
        validate_flag_exclusions(&c).unwrap();
        validate_runtime_policy(&c, &ModelVariant::Gemma4_E2B, Backend::Cuda).unwrap();
    }

    #[test]
    fn policy_rejects_metal_unsupported_models_and_modes() {
        let c = cfg();
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Gemma4_E2B, Backend::Metal),
            "Metal v1 only supports",
        );

        let mut c = cfg();
        c.int4 = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Metal),
            "Metal does not support --int4",
        );

        let mut c = cfg();
        c.fp8_runtime = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Metal),
            "Metal does not support --fp8-runtime",
        );

        let mut c = cfg();
        c.kv_fp8 = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Metal),
            "Metal does not support --kv-fp8",
        );
    }

    #[test]
    fn policy_rejects_cli_only_model_families_for_server() {
        let c = cfg();
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_6_35B_A3B, Backend::Cuda),
            "MoE runtime is not implemented",
        );
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Phi4_Mini, Backend::Cuda),
            "Phi-4 engine is under development",
        );
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Llama3_1_8B, Backend::Cuda),
            "Llama 3.1 is available through the supersonic CLI",
        );
    }
}
