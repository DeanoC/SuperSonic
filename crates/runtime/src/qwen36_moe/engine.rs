use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{Backend, GpuBuffer, GpuError, HalProfileSnapshot, ScalarType, VirtualBuffer};
use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
use model_store::VirtualArenaTransferBackend;
use tokenizers::Tokenizer;

use crate::chat_template::ChatTemplate;
use crate::flm_model_source::FlmModelSourceOptions;
use crate::qwen36_moe::geometry::build_multi_layer_geom;
use crate::qwen36_moe::layer_loader::{
    load_qwen36_layers, Qwen36LayerLoadStrategy, Qwen36LoadOptions,
    Qwen36WeightMode as LayerWeightMode, SparseExpertLoadOptions,
};
use crate::qwen36_moe::layers::LoadedQwen36Layers;
use crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
use crate::qwen36_moe::persistent_decode::{
    build_int4_descs, build_kv_fp8_descs, build_layer_descs,
};
use crate::qwen36_moe::route_telemetry::{MoeRouteTelemetry, MoeTransitionPredictor};
use crate::qwen36_moe::source::{Qwen36MoeSource, Qwen36WeightMode};
use crate::qwen36_moe::types::{AttnLayerBuffers, LayerBuffers, MultiLayerGeom};
use crate::qwen36_moe::weights::{load_to_gpu, prepare_lm_head_bf16};
use crate::qwen36_moe_config::{
    should_try_moe_expert_vmm, should_use_qwen36_kv_vmm, MoeExpertVmmMode,
};

pub use crate::qwen36_moe::source::Qwen36MoeDirectProfile;

const QWEN36_35B_A3B_WEIGHT_PREFIX: &str = qwen36_moe::weights::DEFAULT_PREFIX;
const QWEN36_35B_A3B_VOCAB: usize = 248_320;
const QWEN36_35B_A3B_HIDDEN: usize = 2048;
const QWEN36_35B_A3B_LAYERS: usize = 40;
const QWEN36_35B_A3B_ATTN_HEADS: usize = 16;
const QWEN36_35B_A3B_KV_HEADS: usize = 2;
const QWEN36_35B_A3B_HEAD_DIM: usize = 256;
const QWEN36_35B_A3B_EXPERTS: usize = 256;
const QWEN36_35B_A3B_TOP_K: usize = 8;
const QWEN36_35B_A3B_MOE_INTERMEDIATE: usize = 512;
const QWEN36_35B_A3B_SHARED_INTERMEDIATE: usize = 512;

static ENGINE_LOAD_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static ENGINE_LOAD_LOCK: Mutex<()> = Mutex::new(());

pub struct Qwen36MoeLoadConfig {
    pub flm_path: PathBuf,
    pub backend: Backend,
    pub device_ordinal: usize,
    pub max_context_len: usize,
    pub policy: Qwen36MoeLoadPolicy,
    pub verify_block_hashes: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen36MoeLoadEvidence {
    pub flm_path: PathBuf,
    pub architecture_id: u32,
    pub model_id: u16,
    pub storage_abi_ids: Vec<u16>,
    pub direct_profile: Qwen36MoeDirectProfile,
    pub transfer_backend: VirtualArenaTransferBackend,
    pub source_bytes: u64,
    pub device_upload_bytes: u64,
    pub source_open_duration: Duration,
    pub descriptor_duration: Duration,
    pub tokenizer_duration: Duration,
    pub plan_duration: Duration,
    pub allocation_duration: Duration,
    pub upload_duration: Duration,
    pub total_duration: Duration,
    pub load_sequence: u64,
    pub source_open_count: u64,
    pub resident_allocation_count: u64,
    pub resident_weight_pointers: Vec<usize>,
    pub mapped_virtual_addresses: Vec<usize>,
}

struct LoadEvidenceInput {
    direct_profile: Qwen36MoeDirectProfile,
    source_bytes: u64,
    device_upload_bytes: u64,
    load_sequence: u64,
}

fn build_load_evidence(input: LoadEvidenceInput) -> Result<Qwen36MoeLoadEvidence> {
    if input.direct_profile.native_int4 == 0 {
        anyhow::bail!("Qwen3.6 load evidence requires positive native INT4 direct coverage");
    }
    if input.direct_profile.bf16_fallback != 0 {
        anyhow::bail!(
            "Qwen3.6 load evidence requires zero BF16 fallback direct weights, got {}",
            input.direct_profile.bf16_fallback
        );
    }
    if input.source_bytes == 0 {
        anyhow::bail!("Qwen3.6 load evidence requires positive FLM source bytes");
    }
    if input.device_upload_bytes == 0 {
        anyhow::bail!("Qwen3.6 load evidence requires positive device-upload bytes");
    }
    Ok(Qwen36MoeLoadEvidence {
        flm_path: PathBuf::new(),
        architecture_id: 0,
        model_id: 0,
        storage_abi_ids: Vec::new(),
        direct_profile: input.direct_profile,
        transfer_backend: VirtualArenaTransferBackend::PageableH2d,
        source_bytes: input.source_bytes,
        device_upload_bytes: input.device_upload_bytes,
        source_open_duration: Duration::ZERO,
        descriptor_duration: Duration::ZERO,
        tokenizer_duration: Duration::ZERO,
        plan_duration: Duration::ZERO,
        allocation_duration: Duration::ZERO,
        upload_duration: Duration::ZERO,
        total_duration: Duration::ZERO,
        load_sequence: input.load_sequence,
        source_open_count: 1,
        resident_allocation_count: 0,
        resident_weight_pointers: Vec::new(),
        mapped_virtual_addresses: Vec::new(),
    })
}

#[allow(dead_code)]
pub struct Qwen36MoeEngine {
    source: Qwen36MoeSource,
    tokenizer: Tokenizer,
    chat_template_source: String,
    chat_template: Arc<ChatTemplate>,
    eos_ids: Vec<u32>,
    geom: MultiLayerGeom,
    layers: LoadedQwen36Layers,
    embed_w: GpuBuffer,
    final_norm_w: GpuBuffer,
    lm_head_w: GpuBuffer,
    logits: GpuBuffer,
    counter: GpuBuffer,
    final_hidden: GpuBuffer,
    route_state: Qwen36MoeRouteState,
    next_position: Option<usize>,
    load_evidence: Qwen36MoeLoadEvidence,
    backend: Backend,
    device_ordinal: usize,
    max_context_len: usize,
}

struct Qwen36MoeRouteState {
    num_layers: usize,
    top_k: usize,
    sparse: bool,
    transition_min_observations: u32,
    previous_topk_by_layer: Vec<Vec<usize>>,
    telemetry: Option<MoeRouteTelemetry>,
    predictors: Option<Vec<MoeTransitionPredictor>>,
}

impl Qwen36MoeRouteState {
    fn new(
        top_k: usize,
        num_layers: usize,
        sparse: bool,
        transition_min_observations: u32,
    ) -> Self {
        Self {
            num_layers,
            top_k,
            sparse,
            transition_min_observations,
            previous_topk_by_layer: vec![Vec::new(); num_layers],
            telemetry: sparse.then(|| MoeRouteTelemetry::new(top_k)),
            predictors: (sparse && transition_min_observations > 0).then(|| {
                (0..num_layers)
                    .map(|_| MoeTransitionPredictor::new(top_k, transition_min_observations))
                    .collect()
            }),
        }
    }

    fn reset(&mut self) {
        for routes in &mut self.previous_topk_by_layer {
            routes.clear();
        }
        self.telemetry = self.sparse.then(|| MoeRouteTelemetry::new(self.top_k));
        self.predictors = (self.sparse && self.transition_min_observations > 0).then(|| {
            (0..self.num_layers)
                .map(|_| MoeTransitionPredictor::new(self.top_k, self.transition_min_observations))
                .collect()
        });
    }
}

struct ResidentGpuParts {
    layers: LoadedQwen36Layers,
    embed_w: GpuBuffer,
    final_norm_w: GpuBuffer,
    lm_head_w: GpuBuffer,
    logits: GpuBuffer,
    counter: GpuBuffer,
    final_hidden: GpuBuffer,
}

impl Qwen36MoeEngine {
    pub fn load(config: Qwen36MoeLoadConfig) -> Result<Self> {
        let _load_guard = ENGINE_LOAD_LOCK
            .lock()
            .map_err(|_| anyhow!("Qwen3.6 engine load lock poisoned"))?;
        let total_start = Instant::now();
        validate_pre_source_load_policy(&config)?;

        let mut source = Qwen36MoeSource::open(
            &config.flm_path,
            FlmModelSourceOptions {
                int4_runtime: true,
                verify_block_hashes: config.verify_block_hashes,
            },
        )?;
        let runtime = source
            .source
            .runtime()
            .context("read Qwen3.6 FLM runtime identity")?;
        let architecture_id = runtime.architecture_id;
        let model_id = runtime.model_descriptor().model_id;
        validate_load_contract(
            config.backend,
            config.max_context_len,
            source.config.text_config.max_position_embeddings,
            architecture_id,
            model_id,
            source.direct_profile,
        )?;
        validate_35b_a3b_config(&source.config.text_config)?;

        let chat_template_source = source
            .chat_template_source()
            .context("load Qwen3.6 FLM chat template source")?
            .to_owned();
        let chat_template = ChatTemplate::from_template_source(chat_template_source.clone())
            .context("compile Qwen3.6 FLM chat template")?;

        let tokenizer_start = Instant::now();
        let tokenizer_load = source
            .load_tokenizer_timed()
            .context("load Qwen3.6 FLM tokenizer")?;
        let tokenizer_duration = tokenizer_start.elapsed();
        source.timings.tokenizer = tokenizer_duration;
        source.timings.tokenizer_assets = tokenizer_load.timings.asset_lookup;
        source.timings.tokenizer_parse = tokenizer_load.timings.parse;
        source.timings.tokenizer_build = tokenizer_load.timings.build;
        let tokenizer = tokenizer_load.tokenizer;
        let eos_ids = source.config.text_config.eos_token_ids();
        if eos_ids.is_empty() {
            anyhow::bail!("Qwen3.6 FLM config must provide at least one EOS token id");
        }

        if !gpu_hal::is_backend_compiled(config.backend) {
            anyhow::bail!("Qwen3.6 serving requires a compiled HIP backend");
        }
        gpu_hal::set_backend(config.backend);
        if gpu_hal::current_backend() != Backend::Hip {
            anyhow::bail!(
                "Qwen3.6 backend initialization integrity failure: active backend is {:?}",
                gpu_hal::current_backend()
            );
        }
        gpu_hal::set_device(config.device_ordinal)
            .with_context(|| format!("initialize Qwen3.6 HIP device {}", config.device_ordinal))?;

        let geom = qwen36_geom(&source.config.text_config);
        let (gpu_result, hal_profile) =
            profile_gpu_load(|| load_resident_gpu_parts(&source, &config, &geom));
        let mut gpu = gpu_result?;
        validate_engine_pointer_ownership(&mut gpu.layers)?;

        let source_bytes = std::fs::metadata(&config.flm_path)
            .with_context(|| format!("stat Qwen3.6 FLM {}", config.flm_path.display()))?
            .len();
        let device_upload_bytes = device_upload_bytes(&hal_profile);
        let load_sequence = ENGINE_LOAD_SEQUENCE.fetch_add(1, Ordering::SeqCst) + 1;
        let mut load_evidence = build_load_evidence(LoadEvidenceInput {
            direct_profile: source.direct_profile,
            source_bytes,
            device_upload_bytes,
            load_sequence,
        })?;
        load_evidence.flm_path = config.flm_path.clone();
        load_evidence.architecture_id = architecture_id;
        load_evidence.model_id = model_id;
        load_evidence.storage_abi_ids = runtime
            .storage_abis()
            .iter()
            .map(|abi| abi.storage_abi_id)
            .collect();
        load_evidence.storage_abi_ids.sort_unstable();
        load_evidence.storage_abi_ids.dedup();
        load_evidence.transfer_backend = config.policy.virtual_transfer_backend;
        load_evidence.source_open_duration = source.timings.store_open;
        load_evidence.descriptor_duration = source.timings.config;
        load_evidence.tokenizer_duration = tokenizer_duration;
        load_evidence.plan_duration = source.timings.direct_plan;
        load_evidence.allocation_duration = profile_duration(&hal_profile, |op| {
            op == "alloc" || op.starts_with("vmm_reserve") || op.starts_with("vmm_map")
        });
        load_evidence.upload_duration = profile_duration(&hal_profile, |op| {
            op == "copy_h2d" || op == "copy_storage_to_device"
        });
        load_evidence.resident_allocation_count =
            hal_profile.alloc_calls + virtual_allocation_count(&gpu.layers) as u64;
        load_evidence.resident_weight_pointers =
            collect_resident_weight_pointers(&gpu.layers, &gpu);
        load_evidence.mapped_virtual_addresses = collect_virtual_addresses(&gpu.layers);
        load_evidence.total_duration = total_start.elapsed();

        let route_state = Qwen36MoeRouteState::new(
            geom.top_k as usize,
            geom.num_layers as usize,
            config.policy.moe.sparse_requested,
            config.policy.moe.transition_min_observations,
        );
        Ok(Self {
            source,
            tokenizer,
            chat_template_source,
            chat_template,
            eos_ids,
            geom,
            layers: gpu.layers,
            embed_w: gpu.embed_w,
            final_norm_w: gpu.final_norm_w,
            lm_head_w: gpu.lm_head_w,
            logits: gpu.logits,
            counter: gpu.counter,
            final_hidden: gpu.final_hidden,
            route_state,
            next_position: None,
            load_evidence,
            backend: config.backend,
            device_ordinal: config.device_ordinal,
            max_context_len: config.max_context_len,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn chat_template_source(&self) -> &str {
        &self.chat_template_source
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    pub fn load_evidence(&self) -> &Qwen36MoeLoadEvidence {
        &self.load_evidence
    }

    pub fn reset(&mut self) -> Result<()> {
        if gpu_hal::current_backend() != self.backend {
            return reset_phase(
                "backend",
                Err(anyhow!(
                    "active backend is {:?}, expected {:?}",
                    gpu_hal::current_backend(),
                    self.backend
                )),
            );
        }
        reset_phase(
            "device-sync-before",
            gpu_hal::sync(self.device_ordinal).map_err(anyhow::Error::from),
        )?;

        {
            let (layers, scratch, _) = self.layers.execution_parts();
            reset_phase(
                "layer-state",
                reset_layer_state(self.device_ordinal, layers),
            )?;
            if let Some(scratch) = scratch {
                reset_phase(
                    "persistent-scratch-hidden",
                    zero_gpu_buffer(
                        self.device_ordinal,
                        "persistent-scratch-hidden",
                        &mut scratch.hidden_ping,
                    ),
                )?;
                reset_phase(
                    "persistent-scratch-workspace",
                    zero_gpu_buffer(
                        self.device_ordinal,
                        "persistent-scratch-workspace",
                        &mut scratch.workspace,
                    ),
                )?;
            }
        }

        for (phase, buffer) in [
            ("logits", &mut self.logits),
            ("counter", &mut self.counter),
            ("final-hidden", &mut self.final_hidden),
        ] {
            reset_phase(phase, zero_gpu_buffer(self.device_ordinal, phase, buffer))?;
        }
        self.route_state.reset();
        self.next_position = None;

        reset_phase(
            "device-sync-after",
            gpu_hal::sync(self.device_ordinal).map_err(anyhow::Error::from),
        )?;
        reset_phase(
            "descriptor-ownership",
            validate_engine_pointer_ownership(&mut self.layers),
        )?;
        reset_phase("resident-identity", self.validate_resident_identity())?;
        Ok(())
    }

    fn validate_resident_identity(&self) -> Result<()> {
        let current_pointers = collect_engine_resident_pointers(
            &self.layers,
            &self.embed_w,
            &self.final_norm_w,
            &self.lm_head_w,
        );
        if current_pointers != self.load_evidence.resident_weight_pointers {
            anyhow::bail!("resident allocation pointers changed across reset");
        }
        let current_virtual_addresses = collect_virtual_addresses(&self.layers);
        if current_virtual_addresses != self.load_evidence.mapped_virtual_addresses {
            anyhow::bail!("mapped virtual addresses changed across reset");
        }
        Ok(())
    }
}

fn reset_phase<T>(phase: &str, result: Result<T>) -> Result<T> {
    result.map_err(|error| {
        anyhow!("Qwen3.6 engine reset integrity failure during {phase}: {error:#}")
    })
}

fn reset_layer_state(ordinal: usize, layers: &mut [LayerBuffers]) -> Result<()> {
    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        match &mut layer.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(cache),
                ..
            } => {
                for (state, buffer) in [("k", cache.k.as_mut()), ("v", cache.v.as_mut())] {
                    if let Some(buffer) = buffer {
                        zero_gpu_buffer(ordinal, &format!("layer-{layer_idx}-kv-{state}"), buffer)?;
                    }
                }
                for (state, buffer) in [
                    ("scale-k", cache.kv_scale_k.as_mut()),
                    ("scale-v", cache.kv_scale_v.as_mut()),
                    ("shadow-k", cache.kv_shadow_k.as_mut()),
                    ("shadow-v", cache.kv_shadow_v.as_mut()),
                ] {
                    if let Some(buffer) = buffer {
                        zero_gpu_buffer(ordinal, &format!("layer-{layer_idx}-kv-{state}"), buffer)?;
                    }
                }
                for (state, buffer) in [
                    ("k", cache.virtual_kv_cache_k.as_mut()),
                    ("v", cache.virtual_kv_cache_v.as_mut()),
                ] {
                    if let Some(buffer) = buffer {
                        zero_mapped_virtual_buffer(
                            ordinal,
                            &format!("layer-{layer_idx}-kv-vmm-{state}"),
                            buffer,
                        )?;
                    }
                }
                cache.kv_shadow_start = -1;
            }
            AttnLayerBuffers::Linear {
                conv_state,
                recurrent_state,
                ..
            } => {
                zero_gpu_buffer(
                    ordinal,
                    &format!("layer-{layer_idx}-linear-conv-state"),
                    conv_state,
                )?;
                zero_gpu_buffer(
                    ordinal,
                    &format!("layer-{layer_idx}-linear-recurrent-state"),
                    recurrent_state,
                )?;
            }
            AttnLayerBuffers::Full { kv_cache: None, .. } => {}
        }
    }
    Ok(())
}

fn zero_gpu_buffer(ordinal: usize, label: &str, buffer: &mut GpuBuffer) -> Result<()> {
    gpu_hal::memset_zeros(ordinal, buffer.as_mut_ptr(), buffer.len_bytes())
        .with_context(|| format!("zero {label}"))
}

fn zero_mapped_virtual_buffer(
    ordinal: usize,
    label: &str,
    buffer: &mut VirtualBuffer,
) -> Result<()> {
    let mapped_high_watermark = buffer.mapped_bytes();
    if mapped_high_watermark == 0 {
        return Ok(());
    }
    if buffer.resident_bytes() == mapped_high_watermark {
        return gpu_hal::memset_zeros(ordinal, buffer.as_mut_ptr(), mapped_high_watermark)
            .with_context(|| format!("zero mapped {label} prefix"));
    }

    let page = buffer.granularity();
    for offset in (0..mapped_high_watermark).step_by(page) {
        let len = page.min(buffer.reserved_bytes() - offset);
        match buffer.to_host_range_bytes(offset, len) {
            Ok(_) => {
                gpu_hal::memset_zeros(ordinal, buffer.offset_mut_ptr(offset), len)
                    .with_context(|| format!("zero mapped {label} range at byte {offset}"))?;
            }
            Err(GpuError::InvalidArg(message))
                if message.starts_with("virtual D2H range")
                    && message.ends_with("is not fully mapped") => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect mapped {label} range at byte {offset}"));
            }
        }
    }
    Ok(())
}

fn validate_pre_source_load_policy(config: &Qwen36MoeLoadConfig) -> Result<()> {
    if config.backend != Backend::Hip {
        anyhow::bail!(
            "Qwen3.6 first-class FLM serving is HIP-only, got {}",
            config.backend
        );
    }
    if config.max_context_len == 0 {
        anyhow::bail!("Qwen3.6 maximum context length must be greater than zero");
    }
    if config.max_context_len > i32::MAX as usize {
        anyhow::bail!(
            "Qwen3.6 maximum context length {} exceeds i32::MAX",
            config.max_context_len
        );
    }
    if !config.policy.persistent_decode {
        anyhow::bail!("Qwen3.6 first-class serving requires persistent decode");
    }
    Ok(())
}

fn validate_load_contract(
    backend: Backend,
    max_context_len: usize,
    model_max_context: usize,
    architecture_id: u32,
    model_id: u16,
    direct_profile: Qwen36MoeDirectProfile,
) -> Result<()> {
    if backend != Backend::Hip {
        anyhow::bail!("Qwen3.6 first-class FLM serving requires HIP, got {backend}");
    }
    if max_context_len == 0 {
        anyhow::bail!("Qwen3.6 maximum context length must be greater than zero");
    }
    if max_context_len > i32::MAX as usize {
        anyhow::bail!("Qwen3.6 maximum context length exceeds i32::MAX");
    }
    if max_context_len > model_max_context {
        anyhow::bail!(
            "Qwen3.6 maximum context length {max_context_len} exceeds model context {model_max_context}"
        );
    }
    if architecture_id != ARCH_QWEN3_6_MOE || model_id != MODEL_QWEN3_6_MOE_V1 {
        anyhow::bail!(
            "Qwen3.6 FLM model identity mismatch: architecture_id={architecture_id} \
             model_id={model_id}, expected architecture_id={ARCH_QWEN3_6_MOE} \
             model_id={MODEL_QWEN3_6_MOE_V1}"
        );
    }
    if direct_profile.bf16_fallback != 0 {
        anyhow::bail!(
            "Qwen3.6 production direct profile requires zero BF16 fallback weights, got {}",
            direct_profile.bf16_fallback
        );
    }
    if direct_profile.native_int4 == 0 {
        anyhow::bail!("Qwen3.6 production direct profile requires positive native INT4 coverage");
    }
    let covered = direct_profile
        .raw_dense
        .checked_add(direct_profile.native_int4)
        .and_then(|count| count.checked_add(direct_profile.bf16_fallback))
        .ok_or_else(|| anyhow!("Qwen3.6 direct profile coverage count overflow"))?;
    if covered != direct_profile.required_tensors {
        anyhow::bail!(
            "Qwen3.6 direct profile coverage mismatch: required={} covered={covered}",
            direct_profile.required_tensors
        );
    }
    Ok(())
}

fn validate_35b_a3b_config(text: &qwen36_moe::config::TextConfig) -> Result<()> {
    let actual = (
        text.vocab_size,
        text.hidden_size,
        text.num_hidden_layers,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        text.num_experts,
        text.num_experts_per_tok,
        text.moe_intermediate_size,
        text.shared_expert_intermediate_size,
    );
    let expected = (
        QWEN36_35B_A3B_VOCAB,
        QWEN36_35B_A3B_HIDDEN,
        QWEN36_35B_A3B_LAYERS,
        QWEN36_35B_A3B_ATTN_HEADS,
        QWEN36_35B_A3B_KV_HEADS,
        QWEN36_35B_A3B_HEAD_DIM,
        QWEN36_35B_A3B_EXPERTS,
        QWEN36_35B_A3B_TOP_K,
        QWEN36_35B_A3B_MOE_INTERMEDIATE,
        QWEN36_35B_A3B_SHARED_INTERMEDIATE,
    );
    if actual != expected {
        anyhow::bail!(
            "Qwen3.6 FLM model mismatch: got geometry {actual:?}, expected 35B-A3B {expected:?}"
        );
    }
    Ok(())
}

fn qwen36_geom(text: &qwen36_moe::config::TextConfig) -> MultiLayerGeom {
    let kernel_params = supersonic_core::registry::Qwen36MoeKernelParams {
        weight_prefix: QWEN36_35B_A3B_WEIGHT_PREFIX,
        kv_chunk_size: 256,
        proj_buf_floats: 16_480,
        attn_scratch_floats: 24_576,
        moe_scratch_floats: 4_096,
        num_experts: QWEN36_35B_A3B_EXPERTS as u32,
        top_k: QWEN36_35B_A3B_TOP_K as u32,
        moe_intermediate_size: QWEN36_35B_A3B_MOE_INTERMEDIATE as u32,
        shared_expert_intermediate_size: QWEN36_35B_A3B_SHARED_INTERMEDIATE as u32,
    };
    build_multi_layer_geom(text, &kernel_params)
}

fn profile_gpu_load<T>(load: impl FnOnce() -> Result<T>) -> (Result<T>, HalProfileSnapshot) {
    gpu_hal::hal_profile_set_enabled(true);
    gpu_hal::hal_profile_reset();
    let result = load();
    let profile = gpu_hal::hal_profile_snapshot();
    gpu_hal::hal_profile_set_enabled(false);
    (result, profile)
}

fn load_resident_gpu_parts(
    source: &Qwen36MoeSource,
    config: &Qwen36MoeLoadConfig,
    geom: &MultiLayerGeom,
) -> Result<ResidentGpuParts> {
    let layer_weight_mode = match source.weight_mode {
        Qwen36WeightMode::Int4 => LayerWeightMode::Int4,
        Qwen36WeightMode::Bf16 => {
            anyhow::bail!("Qwen3.6 first-class serving requires native INT4 layer weights")
        }
    };
    let kv_vmm =
        should_use_qwen36_kv_vmm(config.policy.kv_vmm, config.backend, config.device_ordinal)
            .context("resolve Qwen3.6 KV VMM policy")?;
    let strategy = layer_load_strategy(config, geom)?;
    let load_options = Qwen36LoadOptions::default().with_registered_mmap_upload(true);
    let mut layers = load_qwen36_layers(
        &source.source.store,
        config.device_ordinal,
        geom,
        &source.config.text_config,
        QWEN36_35B_A3B_WEIGHT_PREFIX,
        layer_weight_mode,
        config.max_context_len,
        config.policy.kv_fp8,
        kv_vmm,
        strategy,
        &load_options,
    )
    .context("load Qwen3.6 resident layers")?;

    if config.policy.persistent_decode {
        layers
            .enable_persistent(config.device_ordinal, geom)
            .context("allocate Qwen3.6 persistent descriptors and scratch")?;
    }

    let embed_name = format!("{QWEN36_35B_A3B_WEIGHT_PREFIX}.embed_tokens.weight");
    let embed_w = load_to_gpu(&source.source.store, config.device_ordinal, &embed_name)
        .with_context(|| format!("upload {embed_name}"))?;
    let prepared = prepare_lm_head_bf16(
        &source.source.store,
        &source.config.text_config,
        QWEN36_35B_A3B_WEIGHT_PREFIX,
        geom,
    )
    .context("prepare Qwen3.6 final norm and LM head")?;
    let final_norm_w = GpuBuffer::from_host_bytes(
        config.device_ordinal,
        ScalarType::BF16,
        &[geom.hidden as usize],
        &prepared.final_norm_bf16,
    )
    .context("upload Qwen3.6 final norm")?;
    let lm_head_w = GpuBuffer::from_host_bytes(
        config.device_ordinal,
        ScalarType::BF16,
        &[geom.vocab as usize, geom.hidden as usize],
        &prepared.lm_head_bf16,
    )
    .context("upload Qwen3.6 LM head")?;
    let logits = GpuBuffer::zeros(
        config.device_ordinal,
        ScalarType::BF16,
        &[geom.vocab as usize],
    )
    .context("allocate Qwen3.6 logits")?;
    let counter = GpuBuffer::zeros(config.device_ordinal, ScalarType::U32, &[1])
        .context("allocate Qwen3.6 LM-head counter")?;
    let final_hidden = GpuBuffer::zeros(
        config.device_ordinal,
        ScalarType::BF16,
        &[geom.hidden as usize],
    )
    .context("allocate Qwen3.6 final hidden")?;

    Ok(ResidentGpuParts {
        layers,
        embed_w,
        final_norm_w,
        lm_head_w,
        logits,
        counter,
        final_hidden,
    })
}

fn layer_load_strategy(
    config: &Qwen36MoeLoadConfig,
    geom: &MultiLayerGeom,
) -> Result<Qwen36LayerLoadStrategy> {
    let moe = &config.policy.moe;
    if moe.sparse_requested != moe.island_cap_experts.is_some() {
        anyhow::bail!(
            "Qwen3.6 sparse residency policy mismatch: sparse_requested={} cap={:?}",
            moe.sparse_requested,
            moe.island_cap_experts
        );
    }
    if let Some(cap_experts) = moe.island_cap_experts {
        if !should_try_moe_expert_vmm(
            MoeExpertVmmMode::Force,
            config.backend,
            true,
            "INT4 native FLM",
            config.device_ordinal,
        )? {
            unreachable!("forced sparse VMM validation must return true or fail");
        }
        if cap_experts < geom.top_k as usize || cap_experts > geom.num_experts as usize {
            anyhow::bail!(
                "Qwen3.6 sparse expert capacity {cap_experts} must be in [{}, {}]",
                geom.top_k,
                geom.num_experts
            );
        }
        return Ok(Qwen36LayerLoadStrategy::SparseExperts(
            SparseExpertLoadOptions {
                cap_experts,
                protected_experts: moe.protected_experts,
                fixed_hot_experts: moe.fixed_hot_experts,
                async_prefetch: moe.async_prefetch,
                async_staging_pages: moe.async_staging_pages,
                prefetch_evict: moe.prefetch_evict,
                transfer_backend: config.policy.virtual_transfer_backend,
            },
        ));
    }
    if should_try_moe_expert_vmm(
        moe.vmm_mode,
        config.backend,
        true,
        "INT4 native FLM",
        config.device_ordinal,
    )? {
        return Ok(Qwen36LayerLoadStrategy::VirtualExperts {
            transfer_backend: config.policy.virtual_transfer_backend,
        });
    }
    Ok(Qwen36LayerLoadStrategy::Dense)
}

fn device_upload_bytes(profile: &HalProfileSnapshot) -> u64 {
    profile
        .entries
        .iter()
        .filter(|entry| {
            entry.op == "copy_h2d"
                || entry.op == "copy_storage_to_device"
                || entry.op == "vmm_copy_h2d"
        })
        .map(|entry| entry.total_bytes)
        .sum()
}

fn profile_duration(profile: &HalProfileSnapshot, include: impl Fn(&str) -> bool) -> Duration {
    Duration::from_secs_f64(
        profile
            .entries
            .iter()
            .filter(|entry| include(&entry.op))
            .map(|entry| entry.total_ms)
            .sum::<f64>()
            / 1000.0,
    )
}

fn validate_engine_pointer_ownership(layers: &mut LoadedQwen36Layers) -> Result<()> {
    let owned = collect_owned_layer_pointers(layers.layers());
    let (live_layers, _, _) = layers.execution_parts();
    let descs = build_layer_descs(live_layers);
    let int4_descs = build_int4_descs(live_layers);
    let kv_fp8_descs = build_kv_fp8_descs(live_layers);
    let descriptor_pointers =
        descriptor_pointer_view(&descs, int4_descs.as_deref(), kv_fp8_descs.as_deref());
    validate_descriptor_pointer_ownership(&owned, &descriptor_pointers)
}

fn descriptor_pointer_view(
    descs: &[kernel_ffi::qwen36_moe::Qwen36MoeDecodeLayerDesc],
    int4_descs: Option<&[kernel_ffi::qwen36_moe::Qwen36MoeInt4ScaleDesc]>,
    kv_fp8_descs: Option<&[kernel_ffi::qwen36_moe::Qwen36MoeKVCacheFp8Desc]>,
) -> Vec<(&'static str, usize)> {
    let mut pointers = Vec::new();
    for desc in descs {
        pointers.extend([
            ("input_norm_w", desc.input_norm_w as usize),
            ("post_attn_norm_w", desc.post_attn_norm_w as usize),
            ("q_proj_w", desc.q_proj_w as usize),
            ("k_proj_w", desc.k_proj_w as usize),
            ("v_proj_w", desc.v_proj_w as usize),
            ("o_proj_w", desc.o_proj_w as usize),
            ("q_norm_w", desc.q_norm_w as usize),
            ("k_norm_w", desc.k_norm_w as usize),
            ("kv_cache_k", desc.kv_cache_k as usize),
            ("kv_cache_v", desc.kv_cache_v as usize),
            ("linear_in_proj_qkv_w", desc.linear_in_proj_qkv_w as usize),
            ("linear_in_proj_z_w", desc.linear_in_proj_z_w as usize),
            ("linear_in_proj_b_w", desc.linear_in_proj_b_w as usize),
            ("linear_in_proj_a_w", desc.linear_in_proj_a_w as usize),
            ("linear_out_proj_w", desc.linear_out_proj_w as usize),
            ("linear_conv1d_w", desc.linear_conv1d_w as usize),
            ("linear_dt_bias", desc.linear_dt_bias as usize),
            ("linear_a_log_exp", desc.linear_a_log_exp as usize),
            ("linear_norm_w", desc.linear_norm_w as usize),
            ("linear_conv_state", desc.linear_conv_state as usize),
            (
                "linear_recurrent_state",
                desc.linear_recurrent_state as usize,
            ),
            ("router_w", desc.router_w as usize),
            ("experts_gate_up_w", desc.experts_gate_up_w as usize),
            ("experts_down_w", desc.experts_down_w as usize),
            (
                "shared_expert_gate_proj_w",
                desc.shared_expert_gate_proj_w as usize,
            ),
            (
                "shared_expert_up_proj_w",
                desc.shared_expert_up_proj_w as usize,
            ),
            (
                "shared_expert_down_proj_w",
                desc.shared_expert_down_proj_w as usize,
            ),
            ("shared_expert_gate_w", desc.shared_expert_gate_w as usize),
            ("kv_shadow_k", desc.kv_shadow_k as usize),
            ("kv_shadow_v", desc.kv_shadow_v as usize),
        ]);
    }
    for desc in int4_descs.into_iter().flatten() {
        pointers.extend([
            ("q_proj_scale", desc.q_proj_scale as usize),
            ("q_proj_zero", desc.q_proj_zero as usize),
            ("k_proj_scale", desc.k_proj_scale as usize),
            ("k_proj_zero", desc.k_proj_zero as usize),
            ("v_proj_scale", desc.v_proj_scale as usize),
            ("v_proj_zero", desc.v_proj_zero as usize),
            ("o_proj_scale", desc.o_proj_scale as usize),
            ("o_proj_zero", desc.o_proj_zero as usize),
            (
                "linear_in_proj_qkv_scale",
                desc.linear_in_proj_qkv_scale as usize,
            ),
            (
                "linear_in_proj_qkv_zero",
                desc.linear_in_proj_qkv_zero as usize,
            ),
            (
                "linear_in_proj_z_scale",
                desc.linear_in_proj_z_scale as usize,
            ),
            ("linear_in_proj_z_zero", desc.linear_in_proj_z_zero as usize),
            ("linear_out_proj_scale", desc.linear_out_proj_scale as usize),
            ("linear_out_proj_zero", desc.linear_out_proj_zero as usize),
            ("experts_gate_up_scale", desc.experts_gate_up_scale as usize),
            ("experts_gate_up_zero", desc.experts_gate_up_zero as usize),
            ("experts_down_scale", desc.experts_down_scale as usize),
            ("experts_down_zero", desc.experts_down_zero as usize),
            (
                "shared_expert_gate_proj_scale",
                desc.shared_expert_gate_proj_scale as usize,
            ),
            (
                "shared_expert_gate_proj_zero",
                desc.shared_expert_gate_proj_zero as usize,
            ),
            (
                "shared_expert_up_proj_scale",
                desc.shared_expert_up_proj_scale as usize,
            ),
            (
                "shared_expert_up_proj_zero",
                desc.shared_expert_up_proj_zero as usize,
            ),
            (
                "shared_expert_down_proj_scale",
                desc.shared_expert_down_proj_scale as usize,
            ),
            (
                "shared_expert_down_proj_zero",
                desc.shared_expert_down_proj_zero as usize,
            ),
        ]);
    }
    for desc in kv_fp8_descs.into_iter().flatten() {
        pointers.extend([
            ("kv_scale_k", desc.kv_scale_k as usize),
            ("kv_scale_v", desc.kv_scale_v as usize),
        ]);
    }
    pointers
}

fn collect_owned_layer_pointers(layers: &[LayerBuffers]) -> HashSet<usize> {
    let mut pointers = HashSet::new();
    for layer in layers {
        collect_attn_owned_pointers(&layer.attn, &mut pointers);
        insert_buffer_ptr(&mut pointers, &layer.ffn.post_attn_norm_w);
        insert_buffer_ptr(&mut pointers, &layer.ffn.gate_w);
        pointers.insert(layer.ffn.gate_up_proj_w.as_ptr() as usize);
        pointers.insert(layer.ffn.down_proj_w.as_ptr() as usize);
        insert_buffer_ptr(&mut pointers, &layer.ffn.shared_gate_proj_w);
        insert_buffer_ptr(&mut pointers, &layer.ffn.shared_up_proj_w);
        insert_buffer_ptr(&mut pointers, &layer.ffn.shared_down_proj_w);
        insert_buffer_ptr(&mut pointers, &layer.ffn.shared_expert_gate_w);
        if let Some(sidecars) = &layer.ffn.int4 {
            for buffer in [
                &sidecars.gate_up_proj_scale,
                &sidecars.gate_up_proj_zero,
                &sidecars.down_proj_scale,
                &sidecars.down_proj_zero,
                &sidecars.shared_gate_proj_scale,
                &sidecars.shared_gate_proj_zero,
                &sidecars.shared_up_proj_scale,
                &sidecars.shared_up_proj_zero,
                &sidecars.shared_down_proj_scale,
                &sidecars.shared_down_proj_zero,
            ] {
                insert_buffer_ptr(&mut pointers, buffer);
            }
        }
    }
    pointers
}

fn collect_attn_owned_pointers(attn: &AttnLayerBuffers, pointers: &mut HashSet<usize>) {
    match attn {
        AttnLayerBuffers::Full {
            input_norm_w,
            q_proj_w,
            k_proj_w,
            v_proj_w,
            q_norm_w,
            k_norm_w,
            o_proj_w,
            int4,
            kv_cache,
        } => {
            for buffer in [
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
            ] {
                insert_buffer_ptr(pointers, buffer);
            }
            if let Some(sidecars) = int4 {
                for buffer in [
                    &sidecars.q_proj_scale,
                    &sidecars.q_proj_zero,
                    &sidecars.k_proj_scale,
                    &sidecars.k_proj_zero,
                    &sidecars.v_proj_scale,
                    &sidecars.v_proj_zero,
                    &sidecars.o_proj_scale,
                    &sidecars.o_proj_zero,
                ] {
                    insert_buffer_ptr(pointers, buffer);
                }
            }
            if let Some(cache) = kv_cache {
                for buffer in [
                    cache.k.as_ref(),
                    cache.v.as_ref(),
                    cache.kv_scale_k.as_ref(),
                    cache.kv_scale_v.as_ref(),
                    cache.kv_shadow_k.as_ref(),
                    cache.kv_shadow_v.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    insert_buffer_ptr(pointers, buffer);
                }
                if let Some(buffer) = &cache.virtual_kv_cache_k {
                    pointers.insert(buffer.as_ptr() as usize);
                }
                if let Some(buffer) = &cache.virtual_kv_cache_v {
                    pointers.insert(buffer.as_ptr() as usize);
                }
            }
        }
        AttnLayerBuffers::Linear {
            input_norm_w,
            in_proj_qkv_w,
            in_proj_z_w,
            in_proj_a_w,
            in_proj_b_w,
            conv1d_w,
            conv1d_bias,
            dt_bias,
            a_log,
            norm_w,
            out_proj_w,
            conv_state,
            recurrent_state,
            int4,
        } => {
            for buffer in [
                input_norm_w,
                in_proj_qkv_w,
                in_proj_z_w,
                in_proj_a_w,
                in_proj_b_w,
                conv1d_w,
                dt_bias,
                a_log,
                norm_w,
                out_proj_w,
                conv_state,
                recurrent_state,
            ] {
                insert_buffer_ptr(pointers, buffer);
            }
            if let Some(bias) = conv1d_bias {
                insert_buffer_ptr(pointers, bias);
            }
            if let Some(sidecars) = int4 {
                for buffer in [
                    &sidecars.in_proj_qkv_scale,
                    &sidecars.in_proj_qkv_zero,
                    &sidecars.in_proj_z_scale,
                    &sidecars.in_proj_z_zero,
                    &sidecars.out_proj_scale,
                    &sidecars.out_proj_zero,
                ] {
                    insert_buffer_ptr(pointers, buffer);
                }
            }
        }
    }
}

fn insert_buffer_ptr(pointers: &mut HashSet<usize>, buffer: &GpuBuffer) {
    pointers.insert(buffer.as_ptr() as usize);
}

fn validate_descriptor_pointer_ownership(
    owned: &HashSet<usize>,
    descriptor_pointers: &[(&str, usize)],
) -> Result<()> {
    for &(label, pointer) in descriptor_pointers {
        if pointer != 0 && !owned.contains(&pointer) {
            anyhow::bail!("Qwen3.6 descriptor pointer {label}={pointer:#x} is not engine-owned");
        }
    }
    Ok(())
}

fn collect_resident_weight_pointers(
    layers: &LoadedQwen36Layers,
    gpu: &ResidentGpuParts,
) -> Vec<usize> {
    collect_engine_resident_pointers(layers, &gpu.embed_w, &gpu.final_norm_w, &gpu.lm_head_w)
}

fn collect_engine_resident_pointers(
    layers: &LoadedQwen36Layers,
    embed_w: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
) -> Vec<usize> {
    let mut pointers = collect_owned_layer_pointers(layers.layers())
        .into_iter()
        .collect::<Vec<_>>();
    pointers.extend([
        embed_w.as_ptr() as usize,
        final_norm_w.as_ptr() as usize,
        lm_head_w.as_ptr() as usize,
    ]);
    pointers.sort_unstable();
    pointers.dedup();
    pointers
}

fn collect_virtual_addresses(layers: &LoadedQwen36Layers) -> Vec<usize> {
    let mut pointers = Vec::new();
    if let Some(arena) = layers.virtual_expert_arena() {
        pointers.extend(
            arena
                .allocations()
                .iter()
                .map(|allocation| allocation.buffer().as_ptr() as usize),
        );
    }
    if let Some(manager) = layers.sparse_expert_residency() {
        pointers.extend(
            manager
                .arena()
                .allocations()
                .iter()
                .map(|allocation| allocation.buffer().as_ptr() as usize),
        );
    }
    for layer in layers.layers() {
        let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &layer.attn
        else {
            continue;
        };
        pointers.extend(
            [
                cache.virtual_kv_cache_k.as_ref(),
                cache.virtual_kv_cache_v.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|buffer| buffer.as_ptr() as usize),
        );
    }
    pointers.sort_unstable();
    pointers.dedup();
    pointers
}

fn virtual_allocation_count(layers: &LoadedQwen36Layers) -> usize {
    let expert = layers
        .virtual_expert_arena()
        .map(|arena| arena.stats().allocations)
        .unwrap_or(0)
        + layers
            .sparse_expert_residency()
            .map(|manager| manager.arena().stats().allocations)
            .unwrap_or(0);
    let kv = layers
        .layers()
        .iter()
        .map(|layer| match &layer.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(cache),
                ..
            } => {
                usize::from(cache.virtual_kv_cache_k.is_some())
                    + usize::from(cache.virtual_kv_cache_v.is_some())
            }
            _ => 0,
        })
        .sum::<usize>();
    expert + kv
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::path::PathBuf;

    use gpu_hal::{Backend, GpuBuffer, ScalarType, VirtualBacking, VirtualBuffer};
    use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
    use model_store::VirtualArenaTransferBackend;

    use super::{
        build_load_evidence, reset_phase, validate_descriptor_pointer_ownership,
        validate_load_contract, zero_gpu_buffer, zero_mapped_virtual_buffer, LoadEvidenceInput,
        Qwen36MoeDirectProfile, Qwen36MoeEngine, Qwen36MoeLoadConfig, Qwen36MoeRouteState,
    };
    use crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
    use crate::qwen36_moe::types::ExpertRoute;
    use crate::qwen36_moe_config::{Qwen36KvVmmMode, Qwen36MoeRuntimeConfig};

    const MODEL_MAX_CONTEXT: usize = 262_144;

    #[test]
    fn load_contract_rejects_empty_or_out_of_model_context() {
        for max_context_len in [0, MODEL_MAX_CONTEXT + 1, i32::MAX as usize + 1] {
            let err = validate_load_contract(
                Backend::Hip,
                max_context_len,
                MODEL_MAX_CONTEXT,
                ARCH_QWEN3_6_MOE,
                MODEL_QWEN3_6_MOE_V1,
                Qwen36MoeDirectProfile {
                    required_tensors: 20,
                    raw_dense: 8,
                    native_int4: 12,
                    bf16_fallback: 0,
                },
            )
            .expect_err("invalid context must fail");

            assert!(err.to_string().contains("context"), "{err:#}");
        }
    }

    #[test]
    fn load_contract_is_hip_only() {
        for backend in [Backend::Cuda, Backend::Metal] {
            let err = validate_load_contract(
                backend,
                4096,
                MODEL_MAX_CONTEXT,
                ARCH_QWEN3_6_MOE,
                MODEL_QWEN3_6_MOE_V1,
                Qwen36MoeDirectProfile {
                    required_tensors: 20,
                    raw_dense: 8,
                    native_int4: 12,
                    bf16_fallback: 0,
                },
            )
            .expect_err("non-HIP backend must fail");

            assert!(err.to_string().contains("HIP"), "{err:#}");
        }
    }

    #[test]
    fn load_contract_rejects_fallback_or_missing_native_int4_profile() {
        for profile in [
            Qwen36MoeDirectProfile {
                required_tensors: 20,
                raw_dense: 8,
                native_int4: 11,
                bf16_fallback: 1,
            },
            Qwen36MoeDirectProfile {
                required_tensors: 8,
                raw_dense: 8,
                native_int4: 0,
                bf16_fallback: 0,
            },
        ] {
            let err = validate_load_contract(
                Backend::Hip,
                4096,
                MODEL_MAX_CONTEXT,
                ARCH_QWEN3_6_MOE,
                MODEL_QWEN3_6_MOE_V1,
                profile,
            )
            .expect_err("non-production direct profile must fail");

            assert!(
                err.to_string().contains("native INT4")
                    || err.to_string().contains("BF16 fallback"),
                "{err:#}"
            );
        }
    }

    #[test]
    fn load_contract_rejects_mismatched_model_identity() {
        for (architecture_id, model_id) in [
            (model_store::flm::ARCH_QWEN3_6_DENSE, MODEL_QWEN3_6_MOE_V1),
            (ARCH_QWEN3_6_MOE, model_store::flm::MODEL_QWEN3_6_DENSE_V1),
        ] {
            let err = validate_load_contract(
                Backend::Hip,
                4096,
                MODEL_MAX_CONTEXT,
                architecture_id,
                model_id,
                Qwen36MoeDirectProfile {
                    required_tensors: 20,
                    raw_dense: 8,
                    native_int4: 12,
                    bf16_fallback: 0,
                },
            )
            .expect_err("mismatched model identity must fail");

            assert!(err.to_string().contains("model identity"), "{err:#}");
        }
    }

    #[test]
    fn descriptor_pointers_must_resolve_to_engine_owned_allocations() {
        let owned = HashSet::from([0x1000usize, 0x2000, 0x3000]);
        validate_descriptor_pointer_ownership(
            &owned,
            &[("input_norm_w", 0x1000), ("kv_cache_k", 0x3000)],
        )
        .expect("owned pointers");

        let err = validate_descriptor_pointer_ownership(
            &owned,
            &[("input_norm_w", 0x1000), ("experts_gate_up_w", 0x4000)],
        )
        .expect_err("unowned descriptor pointer must fail");
        assert!(err.to_string().contains("experts_gate_up_w"), "{err:#}");
        assert!(err.to_string().contains("not engine-owned"), "{err:#}");
    }

    #[test]
    fn load_evidence_requires_positive_native_int4_and_zero_fallback() {
        let evidence = build_load_evidence(LoadEvidenceInput {
            direct_profile: Qwen36MoeDirectProfile {
                required_tensors: 20,
                raw_dense: 8,
                native_int4: 12,
                bf16_fallback: 0,
            },
            source_bytes: 4096,
            device_upload_bytes: 2048,
            load_sequence: 7,
        })
        .expect("production load evidence");

        assert_eq!(evidence.direct_profile.native_int4, 12);
        assert_eq!(evidence.direct_profile.bf16_fallback, 0);
        assert_eq!(evidence.source_bytes, 4096);
        assert_eq!(evidence.device_upload_bytes, 2048);
        assert_eq!(evidence.load_sequence, 7);
    }

    #[test]
    fn public_load_rejects_policy_before_source_open_or_backend_initialization() {
        let initial_backend = gpu_hal::current_backend();
        let config = Qwen36MoeLoadConfig {
            flm_path: PathBuf::from("/definitely/missing/qwen36.flm"),
            backend: Backend::Metal,
            device_ordinal: 0,
            max_context_len: 0,
            policy: Qwen36MoeLoadPolicy {
                persistent_decode: true,
                kv_fp8: false,
                kv_vmm: Qwen36KvVmmMode::Disabled,
                moe: Qwen36MoeRuntimeConfig::default(),
                virtual_transfer_backend: VirtualArenaTransferBackend::PageableH2d,
            },
            verify_block_hashes: false,
        };

        let err = match Qwen36MoeEngine::load(config) {
            Ok(_) => panic!("invalid load policy must fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("HIP"), "{err:#}");
        assert!(
            !err.to_string().contains("opening Qwen3.6 MoE FLM source"),
            "{err:#}"
        );
        assert_eq!(gpu_hal::current_backend(), initial_backend);
    }

    #[test]
    fn reset_api_is_part_of_the_engine_lifecycle() {
        let _: fn(&mut Qwen36MoeEngine) -> anyhow::Result<()> = Qwen36MoeEngine::reset;
    }

    #[test]
    fn route_reset_clears_history_and_residency_counters() {
        let mut state = Qwen36MoeRouteState::new(2, 3, true, 1);
        let routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 7,
                weight: 0.75,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 11,
                weight: 0.25,
            },
        ];
        state.previous_topk_by_layer[0] = vec![7, 13];
        state
            .telemetry
            .as_mut()
            .expect("sparse telemetry")
            .record_route_observation(&routes[0], &[7, 13]);
        state.predictors.as_mut().expect("transition predictors")[0].update(&routes, &[7, 13]);

        state.reset();

        assert!(state.previous_topk_by_layer.iter().all(Vec::is_empty));
        let telemetry = state.telemetry.as_ref().expect("reset telemetry");
        assert_eq!(telemetry.observations_by_rank, vec![0, 0]);
        assert_eq!(telemetry.resident_before_by_rank, vec![0, 0]);
        assert!(state.predictors.as_ref().expect("reset predictors")[0]
            .candidates(&[7, 13], 2)
            .is_empty());
    }

    #[test]
    fn gpu_reset_zeroes_linear_counter_and_persistent_scratch_without_reallocation() {
        let _backend_lock = crate::qwen36_moe::layer_loader::GPU_BACKEND_TEST_LOCK
            .lock()
            .expect("GPU backend test lock");
        if !gpu_hal::is_backend_compiled(Backend::Hip) {
            eprintln!("skip: HIP backend not compiled");
            return;
        }
        gpu_hal::set_backend(Backend::Hip);
        if gpu_hal::set_device(0).is_err() {
            eprintln!("skip: HIP device 0 unavailable");
            return;
        }
        let mut buffers = [
            ("linear-state", dirty_gpu_buffer()),
            ("counter", dirty_gpu_buffer()),
            ("persistent-scratch-hidden", dirty_gpu_buffer()),
            ("persistent-scratch-workspace", dirty_gpu_buffer()),
        ];
        let pointers = buffers
            .iter()
            .map(|(_, buffer)| buffer.as_ptr() as usize)
            .collect::<Vec<_>>();

        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        for (label, buffer) in &mut buffers {
            zero_gpu_buffer(0, label, buffer).expect("zero dirty state");
        }
        let profile = gpu_hal::hal_profile_snapshot();
        gpu_hal::hal_profile_set_enabled(false);

        for ((_, buffer), pointer) in buffers.iter().zip(pointers) {
            assert_eq!(
                buffer.to_host_bytes().expect("read reset state"),
                vec![0; 16]
            );
            assert_eq!(buffer.as_ptr() as usize, pointer);
        }
        assert_eq!(profile.alloc_calls, 0);
    }

    fn dirty_gpu_buffer() -> GpuBuffer {
        GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &[0xa5; 16])
            .expect("allocate dirty reset buffer")
    }

    #[test]
    fn vmm_reset_zeroes_only_mapped_ranges_and_preserves_mapping_identity() {
        let _backend_lock = crate::qwen36_moe::layer_loader::GPU_BACKEND_TEST_LOCK
            .lock()
            .expect("GPU backend test lock");
        if !gpu_hal::is_backend_compiled(Backend::Hip) {
            eprintln!("skip: HIP backend not compiled");
            return;
        }
        gpu_hal::set_backend(Backend::Hip);
        if gpu_hal::set_device(0).is_err() || !gpu_hal::vmm_is_supported(Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unavailable");
            return;
        }
        let mut buffer = VirtualBuffer::reserve(
            0,
            ScalarType::U8,
            &[8 * 1024 * 1024],
            VirtualBacking::Discard,
        )
        .expect("reserve reset VMM");
        let page = buffer.granularity();
        assert!(buffer.reserved_bytes() >= page * 3);
        buffer.map_range_bytes(0, 1).expect("map first page");
        buffer
            .map_range_bytes(page * 2, 1)
            .expect("map discontiguous page");
        let dirty = [0x5au8; 1];
        gpu_hal::copy_h2d(
            0,
            buffer.as_mut_ptr(),
            dirty.as_ptr() as *const c_void,
            dirty.len(),
        )
        .expect("dirty first VMM page");
        gpu_hal::copy_h2d(
            0,
            buffer.offset_mut_ptr(page * 2),
            dirty.as_ptr() as *const c_void,
            dirty.len(),
        )
        .expect("dirty second VMM page");
        let pointer = buffer.as_ptr() as usize;
        let stats = buffer.stats();

        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        zero_mapped_virtual_buffer(0, "kv-cache", &mut buffer).expect("zero mapped VMM pages");
        let profile = gpu_hal::hal_profile_snapshot();
        gpu_hal::hal_profile_set_enabled(false);

        let mut first = [1u8; 1];
        let mut third = [1u8; 1];
        gpu_hal::copy_d2h(
            0,
            first.as_mut_ptr() as *mut c_void,
            buffer.as_ptr(),
            first.len(),
        )
        .expect("read first VMM page");
        gpu_hal::copy_d2h(
            0,
            third.as_mut_ptr() as *mut c_void,
            buffer.offset_ptr(page * 2),
            third.len(),
        )
        .expect("read second VMM page");
        assert_eq!(first, [0]);
        assert_eq!(third, [0]);
        assert_eq!(buffer.as_ptr() as usize, pointer);
        assert_eq!(buffer.stats(), stats);
        assert_eq!(profile.alloc_calls, 0);
        assert!(
            profile
                .entries
                .iter()
                .all(|entry| !entry.op.starts_with("vmm_map")),
            "{:?}",
            profile.entries
        );
    }

    #[test]
    fn reset_failures_are_phase_labelled_integrity_errors() {
        let err = reset_phase::<()>("linear-state", Err(anyhow::anyhow!("device lost")))
            .expect_err("reset failure must propagate");

        assert!(err.to_string().contains("integrity failure"), "{err:#}");
        assert!(err.to_string().contains("linear-state"), "{err:#}");
        assert!(err.to_string().contains("device lost"), "{err:#}");
    }
}
