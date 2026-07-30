//! Server startup: detect GPU, look up registry entry, load weights, build
//! the [`InferenceSession`]. Boiled-down version of the full `supersonic`
//! CLI flow in `crates/runner/src/main.rs` — the server skips the oracle,
//! tracing, and fallback paths the CLI exposes.

use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, Semaphore};

pub use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
use supersonic_core::registry::{self, Backend, GpuArch, ModelFamily, ModelVariant};

use crate::backend_resolver::resolve_backend;
use crate::bakes::ensure_hf_metadata_present;
use crate::builders::{build_gemma4, build_qwen};
use crate::chat_template::ChatTemplate;
use crate::generate::{GenerationTelemetry, MockGeneration};
use crate::prefix_cache::{PrefixCache, PrefixCacheConfig};
use crate::session::InferenceSession;
use supersonic_core::capabilities::{
    capabilities_for_variant, FlmDirectProfile, FlmLoadEvidence, FlmLoadWindowProfileDurations,
    FlmSourceOpenDurations, FlmStartupDurations, FlmTransferBackend, ModelCapabilities,
    ServingFeatures,
};

#[path = "model_source.rs"]
pub mod model_source;

static NEXT_SERVER_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Per-process state shared across every HTTP request. Everything here is
/// built once at startup.
pub struct ServerState {
    pub server_instance_id: u64,
    pub model_id: String,
    pub model_family: ModelFamily,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<Arc<ChatTemplate>>,
    pub session: Option<Arc<Mutex<InferenceSession>>>,
    pub qwen36_moe_engine: Option<Arc<Mutex<crate::qwen36_moe::engine::Qwen36MoeEngine>>>,
    pub mock_generation: Option<MockGeneration>,
    pub eos_ids: Vec<u32>,
    pub max_context: usize,
    pub api_key: Option<String>,
    pub cors_allow_origin: Option<String>,
    pub response_store_max_entries: usize,
    pub scheduler: Arc<GenerationScheduler>,
    pub telemetry: GenerationTelemetry,
    pub capabilities: ModelCapabilities,
    pub prefix_cache: Arc<PrefixCache>,
}

pub struct GenerationScheduler {
    pub permits: Arc<Semaphore>,
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
    pub max_queue: usize,
    pub queue_timeout_ms: u64,
    readiness: AtomicU8,
}

impl GenerationScheduler {
    const LOADING: u8 = 0;
    const READY: u8 = 1;
    const INTEGRITY_LOST: u8 = 2;

    pub fn new(max_queue: usize, queue_timeout_ms: u64) -> Self {
        Self::with_readiness(max_queue, queue_timeout_ms, Self::READY)
    }

    pub fn loading(max_queue: usize, queue_timeout_ms: u64) -> Self {
        Self::with_readiness(max_queue, queue_timeout_ms, Self::LOADING)
    }

    fn with_readiness(max_queue: usize, queue_timeout_ms: u64, readiness: u8) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(1)),
            active: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            max_queue,
            queue_timeout_ms,
            readiness: AtomicU8::new(readiness),
        }
    }
}

impl ServerState {
    pub fn is_ready(&self) -> bool {
        self.scheduler.readiness.load(Ordering::Acquire) == GenerationScheduler::READY
    }

    pub fn mark_loaded(&self) -> bool {
        self.scheduler
            .readiness
            .compare_exchange(
                GenerationScheduler::LOADING,
                GenerationScheduler::READY,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn mark_integrity_lost(&self) {
        self.scheduler
            .readiness
            .store(GenerationScheduler::INTEGRITY_LOST, Ordering::Release);
    }
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
    pub dflash: bool,
    pub dflash_draft_dir: Option<PathBuf>,
    pub dflash_block: Option<usize>,
    pub dflash_tap_layers: Option<String>,
    pub api_key: Option<String>,
    pub cors_allow_origin: Option<String>,
    pub response_store_max_entries: usize,
    pub max_queued_requests: usize,
    pub queue_timeout_ms: u64,
    /// Disable automatic bake download from the GitHub release. Air-gapped
    /// or reproducibility-focused deploys should set this.
    pub no_download: bool,
    pub prefix_cache_enabled: bool,
    pub prefix_cache_dir: Option<PathBuf>,
    pub prefix_cache_min_tokens: usize,
    pub prefix_cache_max_entries: usize,
    pub prefix_cache_max_bytes: Option<usize>,
    pub prefix_cache_memory_ttl_secs: u64,
    pub prefix_cache_disk_ttl_secs: u64,
}

/// Preferred runtime-facing name for loader configuration. `LoaderConfig`
/// remains as a compatibility alias for the existing server/tests.
pub type RuntimeConfig = LoaderConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLane {
    PlainDecode,
    NativeDFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePolicy {
    pub lane: RuntimeLane,
    pub low_bit_target_required: bool,
    pub prefix_cache_allowed: bool,
}

pub fn build(cfg: LoaderConfig) -> Result<ServerState> {
    let resolved =
        model_source::resolve_model_source(None, Some(cfg.model_dir.clone()), Some(&cfg.model))?;
    build_resolved(cfg, resolved)
}

pub fn build_resolved(
    cfg: LoaderConfig,
    resolved: model_source::ResolvedModelSource,
) -> Result<ServerState> {
    validate_flag_exclusions(&cfg)?;
    validate_model_source_options(&cfg, &resolved.source)?;

    /* ---- backend + GPU detection ---- */
    let backend = resolve_backend(&cfg.backend, cfg.device)?;
    gpu_hal::set_backend(backend);

    let variant = resolved.model;
    let runtime_policy =
        validate_resolved_runtime_policy(&cfg, &resolved.source, &variant, backend)?;

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
    let mut capabilities =
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

    /* ---- engine construction ---- */
    let max_context = cfg.max_context.max(8);
    let (tokenizer, chat_template, session, qwen36_moe_engine, eos_ids) = match &resolved.source {
        model_source::ModelSource::Directory(_) => {
            ensure_hf_metadata_present(&cfg)?;

            let tokenizer_path = cfg.model_dir.join("tokenizer.json");
            if !tokenizer_path.exists() {
                bail!(
                    "missing {} — cannot build tokenizer",
                    tokenizer_path.display()
                );
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("load tokenizer: {e}"))?;
            let chat_template = ChatTemplate::try_load(&cfg.model_dir)?;
            if chat_template.is_none() {
                tracing::warn!(
                    "no chat_template in tokenizer_config.json — /v1/chat/completions will reject \
                     requests; /v1/completions still works"
                );
            }

            let (session, eos_ids) = match variant.family() {
                ModelFamily::Qwen35 => build_qwen(&cfg, entry, max_context)?,
                ModelFamily::Qwen3Moe => {
                    bail!("qwen3-30b-a3b MoE runtime is not implemented yet")
                }
                ModelFamily::Qwen36Moe => {
                    bail!("directory startup is not implemented for qwen3.6-35b-a3b")
                }
                ModelFamily::Gemma4 => build_gemma4(&cfg, entry, max_context)?,
                ModelFamily::Phi4 => {
                    bail!(
                        "Phi-4 engine is under development — not yet exposed via supersonic-serve"
                    );
                }
                ModelFamily::Llama31 => {
                    bail!("Llama 3.1 is available through the supersonic CLI but is not wired into supersonic-serve yet");
                }
            };
            (
                Arc::new(tokenizer),
                chat_template,
                Some(Arc::new(Mutex::new(session))),
                None,
                eos_ids,
            )
        }
        model_source::ModelSource::Flm(flm_path) => {
            let engine = crate::qwen36_moe::engine::Qwen36MoeEngine::load(qwen36_moe_load_config(
                flm_path.clone(),
                backend,
                cfg.device,
                max_context,
            ))?;
            capabilities.flm = Some(project_qwen36_load_evidence(engine.load_evidence())?);
            let tokenizer = Arc::new(engine.tokenizer().clone());
            let chat_template = Some(ChatTemplate::from_template_source(
                engine.chat_template_source().to_owned(),
            )?);
            let eos_ids = engine.eos_ids().to_vec();
            (
                tokenizer,
                chat_template,
                Some(Arc::new(Mutex::new(InferenceSession::Qwen36Moe(engine)))),
                None,
                eos_ids,
            )
        }
    };

    let prefix_cache_config =
        prefix_cache_config(&cfg, &resolved.source, runtime_policy, total_vram);

    let state = ServerState {
        server_instance_id: NEXT_SERVER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        model_id: variant.to_string(),
        model_family: variant.family(),
        tokenizer,
        chat_template,
        session,
        qwen36_moe_engine,
        mock_generation: None,
        eos_ids,
        max_context,
        api_key: cfg.api_key,
        cors_allow_origin: cfg.cors_allow_origin,
        response_store_max_entries: cfg.response_store_max_entries,
        scheduler: Arc::new(GenerationScheduler::loading(
            cfg.max_queued_requests,
            cfg.queue_timeout_ms,
        )),
        telemetry: GenerationTelemetry::default(),
        capabilities,
        prefix_cache: Arc::new(PrefixCache::new(prefix_cache_config)),
    };
    if !state.mark_loaded() {
        bail!("server readiness was poisoned before startup completed");
    }
    tracing::info!(
        model = %variant,
        family = %variant.family(),
        max_context,
        "server state ready"
    );
    Ok(state)
}

fn project_qwen36_load_evidence(
    evidence: &crate::qwen36_moe::engine::Qwen36MoeLoadEvidence,
) -> Result<FlmLoadEvidence> {
    validate_startup_timing_hierarchy(evidence)?;
    let source_file = evidence
        .flm_path
        .file_name()
        .ok_or_else(|| anyhow!("Qwen3.6 FLM source path has no basename"))?
        .to_string_lossy()
        .into_owned();
    let transfer_backend = match evidence.transfer_backend {
        model_store::VirtualArenaTransferBackend::PageableH2d => FlmTransferBackend::PageableH2d,
        model_store::VirtualArenaTransferBackend::GpuDirectStorage => {
            FlmTransferBackend::GpuDirectStorage
        }
    };
    let features = crate::session::qwen36_moe_features();

    Ok(FlmLoadEvidence {
        source_file,
        architecture_id: evidence.architecture_id,
        model_id: evidence.model_id,
        storage_abi_ids: evidence.storage_abi_ids.clone(),
        direct_profile: FlmDirectProfile {
            required_weights: evidence.direct_profile.required_tensors,
            raw_dense_weights: evidence.direct_profile.raw_dense,
            native_int4_direct_weights: evidence.direct_profile.native_int4,
            bf16_fallback_weights: evidence.direct_profile.bf16_fallback,
        },
        transfer_backend,
        source_bytes: evidence.source_bytes,
        device_upload_bytes: evidence.device_upload_bytes,
        startup: FlmStartupDurations {
            total: evidence.total_duration,
            source_open: FlmSourceOpenDurations {
                total: evidence.source_open_duration,
                store_open: evidence.store_open_duration,
                config: evidence.config_duration,
                direct_plan: evidence.plan_duration,
            },
            tokenizer: evidence.tokenizer_duration,
            descriptor: evidence.descriptor_duration,
        },
        load_window_profile: FlmLoadWindowProfileDurations {
            allocation_api: evidence.allocation_duration,
            upload_api: evidence.upload_duration,
        },
        load_sequence: evidence.load_sequence,
        source_open_count: evidence.source_open_count,
        resident_allocation_count: evidence.resident_allocation_count,
        features: ServingFeatures {
            plain_prefill_decode: features.plain_prefill_decode,
            native_dflash_generate: features.native_dflash_generate,
            prefix_snapshot: features.prefix_snapshot,
            disk_prefix_snapshot: features.disk_prefix_snapshot,
        },
    })
}

fn validate_startup_timing_hierarchy(
    evidence: &crate::qwen36_moe::engine::Qwen36MoeLoadEvidence,
) -> Result<()> {
    let source_open_exclusive = evidence
        .store_open_duration
        .checked_add(evidence.config_duration)
        .and_then(|duration| duration.checked_add(evidence.plan_duration))
        .ok_or_else(|| anyhow!("Qwen3.6 FLM source-open exclusive phases overflow"))?;
    if source_open_exclusive > evidence.source_open_duration {
        bail!(
            "Qwen3.6 FLM source-open exclusive phases {:?} exceed aggregate {:?}",
            source_open_exclusive,
            evidence.source_open_duration
        );
    }

    let startup_exclusive = evidence
        .source_open_duration
        .checked_add(evidence.tokenizer_duration)
        .and_then(|duration| duration.checked_add(evidence.descriptor_duration))
        .ok_or_else(|| anyhow!("Qwen3.6 FLM startup exclusive components overflow"))?;
    if startup_exclusive > evidence.total_duration {
        bail!(
            "Qwen3.6 FLM startup exclusive components {:?} exceed total {:?}",
            startup_exclusive,
            evidence.total_duration
        );
    }
    Ok(())
}

fn prefix_cache_config(
    cfg: &LoaderConfig,
    source: &model_source::ModelSource,
    policy: RuntimePolicy,
    total_vram: u64,
) -> PrefixCacheConfig {
    let source_allows_cache = matches!(source, model_source::ModelSource::Directory(_));
    let dir = cfg
        .prefix_cache_dir
        .clone()
        .unwrap_or_else(|| match source {
            model_source::ModelSource::Directory(_) => {
                cfg.model_dir.join(".supersonic/serve-cache/v1")
            }
            model_source::ModelSource::Flm(_) => PathBuf::new(),
        });

    PrefixCacheConfig {
        enabled: source_allows_cache && effective_prefix_cache_enabled(cfg, policy),
        dir,
        min_tokens: cfg.prefix_cache_min_tokens,
        max_entries: cfg.prefix_cache_max_entries,
        max_bytes: cfg
            .prefix_cache_max_bytes
            .unwrap_or_else(|| default_prefix_cache_max_bytes(total_vram)),
        memory_ttl_secs: cfg.prefix_cache_memory_ttl_secs,
        disk_ttl_secs: cfg.prefix_cache_disk_ttl_secs,
    }
}

fn default_prefix_cache_max_bytes(total_vram: u64) -> usize {
    const MIN_BUDGET: u64 = 64 * 1024 * 1024;
    const MAX_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
    let budget = (total_vram / 20).clamp(MIN_BUDGET, MAX_BUDGET);
    budget.min(usize::MAX as u64) as usize
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

fn validate_model_source_options(
    cfg: &LoaderConfig,
    source: &model_source::ModelSource,
) -> Result<()> {
    if !matches!(source, model_source::ModelSource::Flm(_)) {
        return Ok(());
    }
    if cfg.int4 || cfg.q4km || cfg.q4km_gptq || cfg.fp8_runtime || cfg.kv_fp8 {
        bail!(
            "FLM sources derive weight and cache formats from native descriptors; \
             external quantization flags are not supported"
        );
    }
    if cfg.dflash
        || cfg.dflash_draft_dir.is_some()
        || cfg.dflash_block.is_some()
        || cfg.dflash_tap_layers.is_some()
    {
        bail!("DFlash options are not supported with FLM sources");
    }
    Ok(())
}

fn validate_resolved_runtime_policy(
    cfg: &LoaderConfig,
    source: &model_source::ModelSource,
    variant: &ModelVariant,
    backend: Backend,
) -> Result<RuntimePolicy> {
    validate_model_source_options(cfg, source)?;
    match source {
        model_source::ModelSource::Directory(_) => validate_runtime_policy(cfg, variant, backend),
        model_source::ModelSource::Flm(_) => {
            if *variant != ModelVariant::Qwen3_6_35B_A3B {
                bail!("FLM serving currently requires --model qwen3.6-35b-a3b (got {variant})");
            }
            if backend != Backend::Hip {
                bail!("Qwen3.6 MoE FLM serving currently requires the HIP backend");
            }
            let mut policy = resolve_runtime_policy(cfg, variant)?;
            policy.prefix_cache_allowed = false;
            Ok(policy)
        }
    }
}

fn qwen36_moe_load_config(
    flm_path: PathBuf,
    backend: Backend,
    device_ordinal: usize,
    max_context_len: usize,
) -> crate::qwen36_moe::engine::Qwen36MoeLoadConfig {
    crate::qwen36_moe::engine::Qwen36MoeLoadConfig {
        flm_path,
        backend,
        device_ordinal,
        max_context_len,
        policy: crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy {
            persistent_decode: true,
            kv_fp8: false,
            kv_vmm: crate::qwen36_moe_config::Qwen36KvVmmMode::Auto,
            moe: crate::qwen36_moe_config::Qwen36MoeRuntimeConfig::default(),
            virtual_transfer_backend: model_store::VirtualArenaTransferBackend::PageableH2d,
        },
        verify_block_hashes: false,
        execution_options: crate::qwen36_moe::decode::Qwen36ExecutionOptions::default(),
        accurate_stage_timings: false,
    }
}

pub fn resolve_runtime_policy(cfg: &LoaderConfig, variant: &ModelVariant) -> Result<RuntimePolicy> {
    let q4km_like = cfg.q4km || cfg.q4km_gptq;
    if cfg.dflash {
        if !matches!(
            variant,
            ModelVariant::Qwen3_5_9B | ModelVariant::Qwen3_6_27B
        ) {
            bail!("--dflash is supported for --model qwen3.5-9b and qwen3.6-27b (got {variant})");
        }
        if !(cfg.int4 || q4km_like) {
            bail!("--dflash requires a low-bit target bake (--int4, --q4km, or --q4km-gptq)");
        }
        if cfg.dflash_draft_dir.is_none() {
            bail!("--dflash requires --dflash-draft-dir");
        }
        if cfg.kv_fp8 {
            bail!("--dflash does not support --kv-fp8");
        }
        if let Some(block) = cfg.dflash_block {
            if block == 0 {
                bail!("--dflash-block must be greater than 0");
            }
        }
        return Ok(RuntimePolicy {
            lane: RuntimeLane::NativeDFlash,
            low_bit_target_required: true,
            prefix_cache_allowed: false,
        });
    }

    Ok(RuntimePolicy {
        lane: RuntimeLane::PlainDecode,
        low_bit_target_required: false,
        prefix_cache_allowed: true,
    })
}

fn effective_prefix_cache_enabled(cfg: &LoaderConfig, policy: RuntimePolicy) -> bool {
    cfg.prefix_cache_enabled && policy.prefix_cache_allowed
}

fn validate_runtime_policy(
    cfg: &LoaderConfig,
    variant: &ModelVariant,
    backend: Backend,
) -> Result<RuntimePolicy> {
    let runtime_policy = resolve_runtime_policy(cfg, variant)?;
    let q4km_like = cfg.q4km || cfg.q4km_gptq;
    if q4km_like
        && !matches!(
            variant.family(),
            ModelFamily::Qwen35 | ModelFamily::Qwen3Moe | ModelFamily::Qwen36Moe
        )
    {
        bail!("--q4km/--q4km-gptq are currently supported only for Qwen models");
    }
    if q4km_like
        && !(backend == Backend::Cuda
            || (backend == Backend::Hip
                && matches!(
                    variant.family(),
                    ModelFamily::Qwen35 | ModelFamily::Qwen36Moe
                )))
    {
        bail!(
            "--q4km/--q4km-gptq are currently supported only on CUDA Qwen paths and HIP Qwen3.5/3.6 paths"
        );
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
        ModelFamily::Qwen35 | ModelFamily::Gemma4 => Ok(runtime_policy),
        ModelFamily::Qwen3Moe => {
            bail!("qwen3-30b-a3b MoE runtime is not implemented yet")
        }
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

    fn qwen36_load_evidence(path: &str) -> crate::qwen36_moe::engine::Qwen36MoeLoadEvidence {
        crate::qwen36_moe::engine::Qwen36MoeLoadEvidence {
            flm_path: PathBuf::from(path),
            architecture_id: model_store::flm::ARCH_QWEN3_6_MOE,
            model_id: model_store::flm::MODEL_QWEN3_6_MOE_V1,
            storage_abi_ids: vec![8],
            direct_profile: crate::qwen36_moe::source::Qwen36MoeDirectProfile {
                required_tensors: 693,
                raw_dense: 363,
                native_int4: 330,
                bf16_fallback: 0,
            },
            transfer_backend: model_store::VirtualArenaTransferBackend::PageableH2d,
            source_bytes: 8_000_000_000,
            device_upload_bytes: 7_000_000_000,
            source_open_duration: std::time::Duration::from_millis(120),
            store_open_duration: std::time::Duration::from_millis(80),
            config_duration: std::time::Duration::from_millis(10),
            descriptor_duration: std::time::Duration::from_millis(40),
            tokenizer_duration: std::time::Duration::from_millis(50),
            plan_duration: std::time::Duration::from_millis(20),
            allocation_duration: std::time::Duration::from_millis(70),
            upload_duration: std::time::Duration::from_millis(80),
            total_duration: std::time::Duration::from_millis(1250),
            load_sequence: 1,
            source_open_count: 1,
            resident_allocation_count: 42,
            resident_allocation_pointers: vec![0x1234],
            mapped_virtual_ranges: Vec::new(),
            config: None,
            tokenizer_timings: crate::flm_tokenizer::QwenBpeTokenizerTimings::default(),
            hal_profile: gpu_hal::HalProfileSnapshot::default(),
        }
    }

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
            dflash: false,
            dflash_draft_dir: None,
            dflash_block: None,
            dflash_tap_layers: None,
            api_key: None,
            cors_allow_origin: None,
            response_store_max_entries: 1024,
            max_queued_requests: 32,
            queue_timeout_ms: 30_000,
            no_download: true,
            prefix_cache_enabled: true,
            prefix_cache_dir: None,
            prefix_cache_min_tokens: 128,
            prefix_cache_max_entries: 1,
            prefix_cache_max_bytes: None,
            prefix_cache_memory_ttl_secs: 600,
            prefix_cache_disk_ttl_secs: 86_400,
        }
    }

    fn err_contains<T: std::fmt::Debug>(result: Result<T>, needle: &str) {
        let err = result.expect_err("expected policy error").to_string();
        assert!(
            err.contains(needle),
            "expected error containing {needle:?}, got {err:?}"
        );
    }

    #[test]
    fn model_source_qwen36_engine_can_cross_server_worker_boundaries() {
        fn assert_send<T: Send>() {}

        assert_send::<crate::qwen36_moe::engine::Qwen36MoeEngine>();
    }

    #[test]
    fn flm_load_evidence_projection_is_basename_only_and_excludes_pointer_details() {
        let evidence = qwen36_load_evidence("/models/private/qwen36-native.flm");

        let projected = project_qwen36_load_evidence(&evidence).expect("FLM evidence projection");

        assert_eq!(projected.source_file, "qwen36-native.flm");
        assert_eq!(
            projected.architecture_id,
            model_store::flm::ARCH_QWEN3_6_MOE
        );
        assert_eq!(projected.model_id, model_store::flm::MODEL_QWEN3_6_MOE_V1);
        assert_eq!(projected.storage_abi_ids, vec![8]);
        assert_eq!(projected.direct_profile.required_weights, 693);
        assert_eq!(projected.direct_profile.raw_dense_weights, 363);
        assert_eq!(projected.direct_profile.native_int4_direct_weights, 330);
        assert_eq!(projected.direct_profile.bf16_fallback_weights, 0);
        assert_eq!(projected.source_bytes, 8_000_000_000);
        assert_eq!(projected.device_upload_bytes, 7_000_000_000);
        assert_eq!(
            projected.startup.total,
            std::time::Duration::from_millis(1250)
        );
        assert_eq!(
            projected.startup.source_open.total,
            std::time::Duration::from_millis(120)
        );
        assert_eq!(
            projected.load_window_profile.upload_api,
            std::time::Duration::from_millis(80)
        );
        assert_eq!(projected.load_sequence, 1);
        assert_eq!(projected.source_open_count, 1);
        assert_eq!(projected.resident_allocation_count, 42);
        assert!(projected.features.plain_prefill_decode);
        assert!(!projected.features.native_dflash_generate);
        assert!(!projected.features.prefix_snapshot);
        assert!(!projected.features.disk_prefix_snapshot);
    }

    #[test]
    fn flm_load_evidence_projection_rejects_impossible_startup_timing_hierarchies() {
        let mut source_children_exceed_aggregate =
            qwen36_load_evidence("/models/private/qwen36-native.flm");
        source_children_exceed_aggregate.source_open_duration =
            std::time::Duration::from_millis(100);
        err_contains(
            project_qwen36_load_evidence(&source_children_exceed_aggregate),
            "source-open exclusive phases",
        );

        let mut startup_components_exceed_total =
            qwen36_load_evidence("/models/private/qwen36-native.flm");
        startup_components_exceed_total.total_duration = std::time::Duration::from_millis(200);
        err_contains(
            project_qwen36_load_evidence(&startup_components_exceed_total),
            "startup exclusive components",
        );
    }

    #[test]
    fn model_source_flm_rejects_external_quantization_flags() {
        for set_flag in [
            |c: &mut LoaderConfig| c.int4 = true,
            |c: &mut LoaderConfig| c.q4km = true,
            |c: &mut LoaderConfig| c.q4km_gptq = true,
            |c: &mut LoaderConfig| c.fp8_runtime = true,
            |c: &mut LoaderConfig| c.kv_fp8 = true,
        ] {
            let mut c = cfg();
            set_flag(&mut c);
            err_contains(
                validate_model_source_options(
                    &c,
                    &model_source::ModelSource::Flm(PathBuf::from("/models/qwen36.flm")),
                ),
                "FLM sources",
            );
        }
    }

    #[test]
    fn model_source_flm_rejects_dflash_flags() {
        for set_flag in [
            |c: &mut LoaderConfig| c.dflash = true,
            |c: &mut LoaderConfig| c.dflash_draft_dir = Some(PathBuf::from("/models/draft")),
            |c: &mut LoaderConfig| c.dflash_block = Some(4),
            |c: &mut LoaderConfig| c.dflash_tap_layers = Some("1,2".to_string()),
        ] {
            let mut c = cfg();
            set_flag(&mut c);
            err_contains(
                validate_model_source_options(
                    &c,
                    &model_source::ModelSource::Flm(PathBuf::from("/models/qwen36.flm")),
                ),
                "DFlash",
            );
        }
    }

    #[test]
    fn model_source_flm_runtime_policy_is_hip_only_qwen36_moe() {
        let c = cfg();
        let flm = model_source::ModelSource::Flm(PathBuf::from("/models/qwen36.flm"));

        let policy = validate_resolved_runtime_policy(
            &c,
            &flm,
            &ModelVariant::Qwen3_6_35B_A3B,
            Backend::Hip,
        )
        .expect("Qwen3.6 MoE FLM on HIP");
        assert!(!policy.prefix_cache_allowed);

        err_contains(
            validate_resolved_runtime_policy(
                &c,
                &flm,
                &ModelVariant::Qwen3_6_35B_A3B,
                Backend::Cuda,
            ),
            "HIP",
        );
        err_contains(
            validate_resolved_runtime_policy(&c, &flm, &ModelVariant::Qwen3_5_0_8B, Backend::Hip),
            "qwen3.6-35b-a3b",
        );
    }

    #[test]
    fn model_source_flm_prefix_cache_is_disabled_without_a_file_child_path() {
        let mut c = cfg();
        let flm_path = PathBuf::from("/models/qwen36.flm");
        c.model_dir = flm_path.clone();
        let source = model_source::ModelSource::Flm(flm_path.clone());
        let policy = RuntimePolicy {
            lane: RuntimeLane::PlainDecode,
            low_bit_target_required: false,
            prefix_cache_allowed: true,
        };

        let cache = prefix_cache_config(&c, &source, policy, 16 * 1024 * 1024 * 1024);

        assert!(!cache.enabled);
        assert!(cache.dir.as_os_str().is_empty());
        assert!(!cache.dir.starts_with(flm_path));
    }

    #[test]
    fn model_source_flm_prefix_cache_keeps_explicit_dir_but_remains_disabled() {
        let mut c = cfg();
        c.model_dir = PathBuf::from("/models/qwen36.flm");
        c.prefix_cache_dir = Some(PathBuf::from("/var/cache/supersonic"));
        let source = model_source::ModelSource::Flm(PathBuf::from("/models/qwen36.flm"));
        let policy = RuntimePolicy {
            lane: RuntimeLane::PlainDecode,
            low_bit_target_required: false,
            prefix_cache_allowed: true,
        };

        let cache = prefix_cache_config(&c, &source, policy, 16 * 1024 * 1024 * 1024);

        assert!(!cache.enabled);
        assert_eq!(cache.dir, PathBuf::from("/var/cache/supersonic"));
    }

    #[test]
    fn model_source_directory_prefix_cache_preserves_legacy_default() {
        let c = cfg();
        let source =
            model_source::ModelSource::Directory(PathBuf::from("/resolved/model-directory"));
        let policy = RuntimePolicy {
            lane: RuntimeLane::PlainDecode,
            low_bit_target_required: false,
            prefix_cache_allowed: true,
        };

        let cache = prefix_cache_config(&c, &source, policy, 16 * 1024 * 1024 * 1024);

        assert!(cache.enabled);
        assert_eq!(
            cache.dir,
            PathBuf::from("/tmp/model/.supersonic/serve-cache/v1")
        );
    }

    #[test]
    fn model_source_flm_load_config_uses_resolved_server_policy() {
        let load =
            qwen36_moe_load_config(PathBuf::from("/models/qwen36.flm"), Backend::Hip, 3, 4096);

        assert_eq!(load.flm_path, PathBuf::from("/models/qwen36.flm"));
        assert_eq!(load.backend, Backend::Hip);
        assert_eq!(load.device_ordinal, 3);
        assert_eq!(load.max_context_len, 4096);
        assert!(load.policy.persistent_decode);
        assert!(!load.policy.kv_fp8);
        assert_eq!(
            load.policy.kv_vmm,
            crate::qwen36_moe_config::Qwen36KvVmmMode::Auto
        );
        assert_eq!(
            load.policy.moe,
            crate::qwen36_moe_config::Qwen36MoeRuntimeConfig::default()
        );
        assert_eq!(
            load.policy.virtual_transfer_backend,
            model_store::VirtualArenaTransferBackend::PageableH2d
        );
        assert!(!load.verify_block_hashes);
        assert!(!load.accurate_stage_timings);
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
    fn policy_allows_q4km_for_qwen35_family_on_hip() {
        let mut c = cfg();
        c.q4km_gptq = true;
        validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Hip).unwrap();

        let mut c = cfg();
        c.q4km_gptq = true;
        validate_runtime_policy(&c, &ModelVariant::Qwen3_6_27B, Backend::Hip).unwrap();
    }

    #[test]
    fn policy_rejects_q4km_on_unsupported_backend_family_pairs() {
        let mut c = cfg();
        c.q4km_gptq = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_30B_A3B, Backend::Hip),
            "CUDA Qwen paths and HIP Qwen3.5/3.6 paths",
        );
    }

    #[test]
    fn policy_rejects_dflash_without_lowbit_target_and_draft_dir() {
        let mut c = cfg();
        c.dflash = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_6_27B, Backend::Hip),
            "requires a low-bit target bake",
        );

        let mut c = cfg();
        c.dflash = true;
        c.q4km = true;
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_6_27B, Backend::Hip),
            "requires --dflash-draft-dir",
        );
    }

    #[test]
    fn policy_allows_dflash_for_supported_dense_qwen_lowbit_targets() {
        let mut c = cfg();
        c.dflash = true;
        c.q4km = true;
        c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));
        validate_flag_exclusions(&c).unwrap();
        validate_runtime_policy(&c, &ModelVariant::Qwen3_6_27B, Backend::Hip).unwrap();

        let mut c = cfg();
        c.dflash = true;
        c.int4 = true;
        c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));
        validate_runtime_policy(&c, &ModelVariant::Qwen3_5_9B, Backend::Cuda).unwrap();
    }

    #[test]
    fn policy_disables_prefix_cache_for_native_dflash() {
        let mut c = cfg();
        c.dflash = true;
        c.q4km_gptq = true;
        c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));

        let policy = resolve_runtime_policy(&c, &ModelVariant::Qwen3_6_27B).unwrap();

        assert_eq!(policy.lane, RuntimeLane::NativeDFlash);
        assert!(policy.low_bit_target_required);
        assert!(!policy.prefix_cache_allowed);
        assert!(!effective_prefix_cache_enabled(&c, policy));
    }

    #[test]
    fn policy_rejects_dflash_for_non_dflash_targets_and_kv_fp8() {
        let mut c = cfg();
        c.dflash = true;
        c.q4km = true;
        c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_5_0_8B, Backend::Hip),
            "supported for --model qwen3.5-9b and qwen3.6-27b",
        );

        let mut c = cfg();
        c.dflash = true;
        c.q4km = true;
        c.kv_fp8 = true;
        c.dflash_draft_dir = Some(PathBuf::from("/tmp/dflash"));
        err_contains(
            validate_runtime_policy(&c, &ModelVariant::Qwen3_6_27B, Backend::Hip),
            "does not support --kv-fp8",
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
