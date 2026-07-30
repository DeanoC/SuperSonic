use std::collections::HashSet;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{
    Backend, GpuBuffer, GpuError, HalProfileSnapshot, ScalarType, VirtualBuffer, VirtualBufferStats,
};
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
use crate::qwen36_moe::types::{
    AttnLayerBuffers, ExpertRoute, FullAttnKvCache, LayerBuffers, MultiLayerGeom,
};
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
    pub resident_allocation_pointers: Vec<usize>,
    pub mapped_virtual_ranges: Vec<Qwen36MoeMappedVirtualRangeEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen36MoeMappedVirtualRangeEvidence {
    pub address: usize,
    pub stats: VirtualBufferStats,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen36MoeResetTestSnapshot {
    pub source_open_count: u64,
    pub resident_allocation_pointers: Vec<usize>,
    pub mapped_virtual_ranges: Vec<Qwen36MoeMappedVirtualRangeEvidence>,
    pub persistent_descriptor_bytes: Vec<Vec<u8>>,
    pub mutable_nonzero_labels: Vec<String>,
    pub route_history_entries: usize,
    pub route_observations: u64,
    pub transition_candidates: usize,
    pub next_position: Option<usize>,
}

struct LoadEvidenceInput {
    direct_profile: Qwen36MoeDirectProfile,
    source_bytes: u64,
    device_upload_bytes: u64,
    load_sequence: u64,
    source_open_count: u64,
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
    if input.source_open_count == 0 {
        anyhow::bail!("Qwen3.6 load evidence requires an observed source open");
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
        source_open_count: input.source_open_count,
        resident_allocation_count: 0,
        resident_allocation_pointers: Vec::new(),
        mapped_virtual_ranges: Vec::new(),
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
    source_open_count: u64,
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

        let mut source_open_count = 0;
        let source_open_start = Instant::now();
        let mut source = observe_source_open(&mut source_open_count, || {
            Qwen36MoeSource::open(
                &config.flm_path,
                FlmModelSourceOptions {
                    int4_runtime: true,
                    verify_block_hashes: config.verify_block_hashes,
                },
            )
        })?;
        let source_open_duration = source_open_start.elapsed();
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
            source_open_count,
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
        load_evidence.source_open_duration = source_open_duration;
        load_evidence.descriptor_duration = persistent_descriptor_duration(&mut gpu.layers)?;
        load_evidence.tokenizer_duration = tokenizer_duration;
        load_evidence.plan_duration = source.timings.direct_plan;
        load_evidence.allocation_duration = profile_duration(&hal_profile, |op| {
            op == "alloc" || op.starts_with("vmm_reserve") || op.starts_with("vmm_map")
        });
        load_evidence.upload_duration = profile_duration(&hal_profile, |op| {
            op == "copy_h2d" || op == "copy_storage_to_device" || op == "vmm_copy_h2d"
        });
        load_evidence.resident_allocation_pointers = collect_resident_allocation_pointers(
            &mut gpu.layers,
            [
                &gpu.embed_w,
                &gpu.final_norm_w,
                &gpu.lm_head_w,
                &gpu.logits,
                &gpu.counter,
                &gpu.final_hidden,
            ],
        );
        load_evidence.resident_allocation_count =
            load_evidence.resident_allocation_pointers.len() as u64;
        load_evidence.mapped_virtual_ranges = collect_mapped_virtual_ranges(&gpu.layers);

        let route_state = Qwen36MoeRouteState::new(
            geom.top_k as usize,
            geom.num_layers as usize,
            config.policy.moe.sparse_requested,
            config.policy.moe.transition_min_observations,
        );
        let mut engine = Self {
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
            source_open_count,
            load_evidence,
            backend: config.backend,
            device_ordinal: config.device_ordinal,
            max_context_len: config.max_context_len,
        };
        engine.load_evidence.total_duration = total_start.elapsed();
        Ok(engine)
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

    #[doc(hidden)]
    pub fn test_only_dirty_reset_state(&mut self) -> Result<()> {
        if gpu_hal::current_backend() != self.backend {
            anyhow::bail!(
                "Qwen3.6 dirty-reset hook backend mismatch: active {:?}, expected {:?}",
                gpu_hal::current_backend(),
                self.backend
            );
        }
        gpu_hal::sync(self.device_ordinal).context("sync before dirty-reset hook")?;
        let mut made_discontiguous = false;
        {
            let (layers, scratch, _) = self.layers.execution_parts();
            for (layer_idx, layer) in layers.iter_mut().enumerate() {
                match &mut layer.attn {
                    AttnLayerBuffers::Linear {
                        conv_state,
                        recurrent_state,
                        ..
                    } => {
                        dirty_gpu_buffer_first_byte(self.device_ordinal, conv_state).with_context(
                            || format!("dirty layer-{layer_idx}-linear-conv-state"),
                        )?;
                        dirty_gpu_buffer_first_byte(self.device_ordinal, recurrent_state)
                            .with_context(|| {
                                format!("dirty layer-{layer_idx}-linear-recurrent-state")
                            })?;
                    }
                    AttnLayerBuffers::Full {
                        kv_cache: Some(cache),
                        ..
                    } => {
                        for (label, buffer) in [
                            ("k", cache.k.as_mut()),
                            ("v", cache.v.as_mut()),
                            ("scale-k", cache.kv_scale_k.as_mut()),
                            ("scale-v", cache.kv_scale_v.as_mut()),
                            ("shadow-k", cache.kv_shadow_k.as_mut()),
                            ("shadow-v", cache.kv_shadow_v.as_mut()),
                        ] {
                            if let Some(buffer) = buffer {
                                dirty_gpu_buffer_first_byte(self.device_ordinal, buffer)
                                    .with_context(|| {
                                        format!("dirty layer-{layer_idx}-kv-{label}")
                                    })?;
                            }
                        }
                        for (label, buffer) in [
                            ("k", cache.virtual_kv_cache_k.as_mut()),
                            ("v", cache.virtual_kv_cache_v.as_mut()),
                        ] {
                            if let Some(buffer) = buffer {
                                if !made_discontiguous
                                    && make_discontiguous_vmm_for_reset_test(buffer)?
                                {
                                    made_discontiguous = true;
                                }
                                dirty_mapped_virtual_pages(
                                    self.device_ordinal,
                                    &format!("layer-{layer_idx}-kv-vmm-{label}"),
                                    buffer,
                                )?;
                            }
                        }
                        cache.kv_shadow_start = 17;
                    }
                    AttnLayerBuffers::Full { kv_cache: None, .. } => {}
                }
            }
            scratch
                .ok_or_else(|| anyhow!("Qwen3.6 dirty-reset hook requires persistent scratch"))?
                .dirty_mutable_for_reset_test(self.device_ordinal)?;
        }
        if !made_discontiguous {
            anyhow::bail!("Qwen3.6 dirty-reset hook requires a three-page KV VMM reservation");
        }
        for (label, buffer) in [
            ("logits", &mut self.logits),
            ("counter", &mut self.counter),
            ("final-hidden", &mut self.final_hidden),
        ] {
            dirty_gpu_buffer_first_byte(self.device_ordinal, buffer)
                .with_context(|| format!("dirty engine {label}"))?;
        }

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
        self.route_state.previous_topk_by_layer[0] = vec![7, 11];
        if let Some(telemetry) = self.route_state.telemetry.as_mut() {
            for route in &routes {
                telemetry.record_route_observation(route, &[7, 11]);
                telemetry.record_resident_before(route.rank);
            }
        }
        if let Some(predictors) = self.route_state.predictors.as_mut() {
            let observations = self.route_state.transition_min_observations.max(1);
            for _ in 0..observations {
                predictors[0].update(&routes, &[7, 11]);
            }
        }
        self.next_position = Some(73);
        gpu_hal::sync(self.device_ordinal).context("sync after dirty-reset hook")
    }

    #[doc(hidden)]
    pub fn test_only_reset_snapshot(&mut self) -> Result<Qwen36MoeResetTestSnapshot> {
        gpu_hal::sync(self.device_ordinal).context("sync before reset test snapshot")?;
        let resident_allocation_pointers = collect_resident_allocation_pointers(
            &mut self.layers,
            [
                &self.embed_w,
                &self.final_norm_w,
                &self.lm_head_w,
                &self.logits,
                &self.counter,
                &self.final_hidden,
            ],
        );
        let mapped_virtual_ranges = collect_mapped_virtual_ranges(&self.layers);
        let (persistent_descriptor_bytes, mut mutable_nonzero_labels) = {
            let (layers, scratch, _) = self.layers.execution_parts();
            let mut labels = collect_layer_mutable_nonzero_labels(layers)?;
            let scratch = scratch
                .ok_or_else(|| anyhow!("Qwen3.6 reset snapshot requires persistent scratch"))?;
            labels.extend(scratch.mutable_nonzero_labels()?);
            (scratch.descriptor_bytes()?, labels)
        };
        for (label, buffer) in [
            ("logits", &self.logits),
            ("counter", &self.counter),
            ("final-hidden", &self.final_hidden),
        ] {
            if gpu_buffer_first_byte_nonzero(buffer)? {
                mutable_nonzero_labels.push(label.to_string());
            }
        }
        mutable_nonzero_labels.sort();
        let route_history_entries = self
            .route_state
            .previous_topk_by_layer
            .iter()
            .map(Vec::len)
            .sum();
        let route_observations = self
            .route_state
            .telemetry
            .as_ref()
            .map(|telemetry| telemetry.observations_by_rank.iter().sum())
            .unwrap_or(0);
        let transition_candidates = self
            .route_state
            .predictors
            .as_ref()
            .map(|predictors| {
                predictors
                    .iter()
                    .zip(&self.route_state.previous_topk_by_layer)
                    .map(|(predictor, previous)| {
                        predictor.candidates(previous, self.route_state.top_k).len()
                    })
                    .sum()
            })
            .unwrap_or(0);
        Ok(Qwen36MoeResetTestSnapshot {
            source_open_count: self.source_open_count,
            resident_allocation_pointers,
            mapped_virtual_ranges,
            persistent_descriptor_bytes,
            mutable_nonzero_labels,
            route_history_entries,
            route_observations,
            transition_candidates,
            next_position: self.next_position,
        })
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
        let ordinal = self.device_ordinal;
        run_reset_transaction(
            || gpu_hal::sync(ordinal).map_err(anyhow::Error::from),
            || self.clear_mutable_state(),
            || gpu_hal::sync(ordinal).map_err(anyhow::Error::from),
        )?;
        reset_phase(
            "descriptor-ownership",
            validate_engine_pointer_ownership(&mut self.layers),
        )?;
        reset_phase("resident-identity", self.validate_resident_identity())?;
        Ok(())
    }

    fn clear_mutable_state(&mut self) -> Result<()> {
        {
            let (layers, scratch, _) = self.layers.execution_parts();
            reset_phase(
                "layer-state",
                reset_layer_state(self.device_ordinal, layers),
            )?;
            if let Some(scratch) = scratch {
                reset_phase(
                    "persistent-scratch",
                    scratch.reset_mutable(self.device_ordinal),
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
        Ok(())
    }

    fn validate_resident_identity(&mut self) -> Result<()> {
        let current_pointers = collect_resident_allocation_pointers(
            &mut self.layers,
            [
                &self.embed_w,
                &self.final_norm_w,
                &self.lm_head_w,
                &self.logits,
                &self.counter,
                &self.final_hidden,
            ],
        );
        if current_pointers != self.load_evidence.resident_allocation_pointers {
            anyhow::bail!("resident allocation pointers changed across reset");
        }
        let current_virtual_addresses = collect_mapped_virtual_ranges(&self.layers)
            .into_iter()
            .map(|range| range.address)
            .collect::<Vec<_>>();
        let loaded_virtual_addresses = self
            .load_evidence
            .mapped_virtual_ranges
            .iter()
            .map(|range| range.address)
            .collect::<Vec<_>>();
        if current_virtual_addresses != loaded_virtual_addresses {
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

fn run_reset_transaction(
    sync_before: impl FnOnce() -> Result<()>,
    clear: impl FnOnce() -> Result<()>,
    sync_after: impl FnOnce() -> Result<()>,
) -> Result<()> {
    reset_phase("device-sync-before", sync_before())?;
    let clear_result = clear();
    let sync_after_result = reset_phase("device-sync-after", sync_after());
    match (clear_result, sync_after_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(first), Ok(())) => Err(first),
        (Ok(()), Err(sync_error)) => Err(sync_error),
        (Err(first), Err(sync_error)) => Err(anyhow!(
            "{first:#}; additionally, post-reset synchronization failed: {sync_error:#}"
        )),
    }
}

fn reset_layer_state(ordinal: usize, layers: &mut [LayerBuffers]) -> Result<()> {
    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        match &mut layer.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(cache),
                ..
            } => {
                reset_full_attention_cache(ordinal, layer_idx, cache)?;
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

fn reset_full_attention_cache(
    ordinal: usize,
    layer_idx: usize,
    cache: &mut FullAttnKvCache,
) -> Result<()> {
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

fn dirty_gpu_buffer_first_byte(ordinal: usize, buffer: &mut GpuBuffer) -> Result<()> {
    let dirty = [0xa5u8];
    gpu_hal::copy_h2d(
        ordinal,
        buffer.as_mut_ptr(),
        dirty.as_ptr() as *const c_void,
        dirty.len(),
    )
    .map_err(anyhow::Error::from)
}

fn gpu_buffer_first_byte_nonzero(buffer: &GpuBuffer) -> Result<bool> {
    let mut byte = [0u8];
    gpu_hal::copy_d2h(
        buffer.device_ordinal(),
        byte.as_mut_ptr() as *mut c_void,
        buffer.as_ptr(),
        byte.len(),
    )
    .map_err(anyhow::Error::from)?;
    Ok(byte[0] != 0)
}

fn make_discontiguous_vmm_for_reset_test(buffer: &mut VirtualBuffer) -> Result<bool> {
    let page = buffer.granularity();
    if buffer.reserved_bytes() < page * 3 {
        return Ok(false);
    }
    buffer
        .unmap_range_discard(0, buffer.reserved_bytes())
        .context("unmap KV VMM for discontiguous reset fixture")?;
    buffer
        .map_range_bytes(0, 1)
        .context("map first KV VMM reset-fixture page")?;
    buffer
        .map_range_bytes(page * 2, 1)
        .context("map third KV VMM reset-fixture page")?;
    Ok(true)
}

fn dirty_mapped_virtual_pages(
    ordinal: usize,
    label: &str,
    buffer: &mut VirtualBuffer,
) -> Result<()> {
    let dirty = [0xa5u8];
    let page = buffer.granularity();
    for offset in (0..buffer.mapped_bytes()).step_by(page) {
        match buffer.to_host_range_bytes(offset, 1) {
            Ok(_) => gpu_hal::copy_h2d(
                ordinal,
                buffer.offset_mut_ptr(offset),
                dirty.as_ptr() as *const c_void,
                dirty.len(),
            )
            .map_err(anyhow::Error::from)
            .with_context(|| format!("dirty mapped {label} page at byte {offset}"))?,
            Err(error) if is_unmapped_virtual_range_error(&error) => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect mapped {label} page at byte {offset}"));
            }
        }
    }
    Ok(())
}

fn collect_layer_mutable_nonzero_labels(layers: &[LayerBuffers]) -> Result<Vec<String>> {
    let mut labels = Vec::new();
    for (layer_idx, layer) in layers.iter().enumerate() {
        match &layer.attn {
            AttnLayerBuffers::Linear {
                conv_state,
                recurrent_state,
                ..
            } => {
                for (label, buffer) in [
                    ("linear-conv-state", conv_state),
                    ("linear-recurrent-state", recurrent_state),
                ] {
                    if gpu_buffer_first_byte_nonzero(buffer)? {
                        labels.push(format!("layer-{layer_idx}-{label}"));
                    }
                }
            }
            AttnLayerBuffers::Full {
                kv_cache: Some(cache),
                ..
            } => {
                for (label, buffer) in [
                    ("kv-dense-k", cache.k.as_ref()),
                    ("kv-dense-v", cache.v.as_ref()),
                    ("kv-scale-k", cache.kv_scale_k.as_ref()),
                    ("kv-scale-v", cache.kv_scale_v.as_ref()),
                    ("kv-shadow-k", cache.kv_shadow_k.as_ref()),
                    ("kv-shadow-v", cache.kv_shadow_v.as_ref()),
                ] {
                    if let Some(buffer) = buffer {
                        if gpu_buffer_first_byte_nonzero(buffer)? {
                            labels.push(format!("layer-{layer_idx}-{label}"));
                        }
                    }
                }
                if cache.kv_shadow_start != -1 {
                    labels.push(format!("layer-{layer_idx}-kv-shadow-start"));
                }
                for (label, buffer) in [
                    ("k", cache.virtual_kv_cache_k.as_ref()),
                    ("v", cache.virtual_kv_cache_v.as_ref()),
                ] {
                    if let Some(buffer) = buffer {
                        let page = buffer.granularity();
                        for offset in (0..buffer.mapped_bytes()).step_by(page) {
                            match buffer.to_host_range_bytes(offset, 1) {
                                Ok(bytes) if bytes[0] != 0 => labels.push(format!(
                                    "layer-{layer_idx}-kv-vmm-{label}-page-{offset}"
                                )),
                                Ok(_) => {}
                                Err(error) if is_unmapped_virtual_range_error(&error) => {}
                                Err(error) => {
                                    return Err(error).with_context(|| {
                                        format!(
                                            "inspect layer-{layer_idx}-kv-vmm-{label} page at byte {offset}"
                                        )
                                    });
                                }
                            }
                        }
                    }
                }
            }
            AttnLayerBuffers::Full { kv_cache: None, .. } => {}
        }
    }
    Ok(labels)
}

fn is_unmapped_virtual_range_error(error: &GpuError) -> bool {
    matches!(
        error,
        GpuError::InvalidArg(message)
            if message.starts_with("virtual D2H range")
                && message.ends_with("is not fully mapped")
    )
}

fn observe_source_open<T>(
    source_open_count: &mut u64,
    open: impl FnOnce() -> Result<T>,
) -> Result<T> {
    *source_open_count = source_open_count
        .checked_add(1)
        .ok_or_else(|| anyhow!("Qwen3.6 source-open count overflow"))?;
    open()
}

fn persistent_descriptor_duration(layers: &mut LoadedQwen36Layers) -> Result<Duration> {
    let (_, scratch, _) = layers.execution_parts();
    scratch
        .map(|scratch| scratch.descriptor_duration())
        .ok_or_else(|| anyhow!("Qwen3.6 persistent descriptor timing is unavailable"))
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
    macro_rules! require_canonical {
        ($field:expr, $expected:expr, $label:literal) => {
            if $field != $expected {
                anyhow::bail!(
                    "Qwen3.6 FLM 35B-A3B execution geometry mismatch for {}: got {:?}, expected {:?}",
                    $label,
                    $field,
                    $expected
                );
            }
        };
    }

    require_canonical!(text.vocab_size, QWEN36_35B_A3B_VOCAB, "vocab_size");
    require_canonical!(text.hidden_size, QWEN36_35B_A3B_HIDDEN, "hidden_size");
    require_canonical!(
        text.num_hidden_layers,
        QWEN36_35B_A3B_LAYERS,
        "num_hidden_layers"
    );
    require_canonical!(
        text.num_attention_heads,
        QWEN36_35B_A3B_ATTN_HEADS,
        "num_attention_heads"
    );
    require_canonical!(
        text.num_key_value_heads,
        QWEN36_35B_A3B_KV_HEADS,
        "num_key_value_heads"
    );
    require_canonical!(
        text.max_position_embeddings,
        262_144,
        "max_position_embeddings"
    );
    require_canonical!(text.rms_norm_eps, 1e-6, "rms_norm_eps");
    require_canonical!(
        text.hidden_act,
        qwen36_moe::config::Activation::Silu,
        "hidden_act"
    );
    require_canonical!(text.tie_word_embeddings, false, "tie_word_embeddings");
    require_canonical!(text.head_dim, QWEN36_35B_A3B_HEAD_DIM, "head_dim");
    require_canonical!(text.full_attention_interval, 4, "full_attention_interval");
    require_canonical!(text.attn_output_gate, true, "attn_output_gate");
    require_canonical!(text.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim");
    require_canonical!(text.linear_key_head_dim, 128, "linear_key_head_dim");
    require_canonical!(text.linear_value_head_dim, 128, "linear_value_head_dim");
    require_canonical!(text.linear_num_key_heads, 16, "linear_num_key_heads");
    require_canonical!(text.linear_num_value_heads, 32, "linear_num_value_heads");

    let expected_layer_types = (0..QWEN36_35B_A3B_LAYERS)
        .map(|index| {
            if (index + 1) % 4 == 0 {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            }
        })
        .collect::<Vec<_>>();
    require_canonical!(&text.layer_types, &expected_layer_types, "layer_types");
    let rope = text.rope_parameters.clone().unwrap_or_default();
    require_canonical!(rope.rope_type.as_str(), "default", "rope_type");
    require_canonical!(rope.rope_theta, 10_000_000.0, "rope_theta");
    require_canonical!(rope.partial_rotary_factor, 0.25, "partial_rotary_factor");
    require_canonical!(rope.mrope_interleaved, true, "mrope_interleaved");
    require_canonical!(
        rope.mrope_section.as_slice(),
        &[11, 11, 10],
        "mrope_section"
    );

    require_canonical!(text.num_experts, QWEN36_35B_A3B_EXPERTS, "num_experts");
    require_canonical!(
        text.num_experts_per_tok,
        QWEN36_35B_A3B_TOP_K,
        "num_experts_per_tok"
    );
    require_canonical!(
        text.moe_intermediate_size,
        QWEN36_35B_A3B_MOE_INTERMEDIATE,
        "moe_intermediate_size"
    );
    require_canonical!(
        text.shared_expert_intermediate_size,
        QWEN36_35B_A3B_SHARED_INTERMEDIATE,
        "shared_expert_intermediate_size"
    );
    require_canonical!(text.norm_topk_prob, true, "norm_topk_prob");
    require_canonical!(text.router_aux_loss_coef, 0.001, "router_aux_loss_coef");
    require_canonical!(
        text.mlp_only_layers.as_slice(),
        &[] as &[usize],
        "mlp_only_layers"
    );
    require_canonical!(
        text.decoder_sparse_step,
        None::<usize>,
        "decoder_sparse_step"
    );
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

fn collect_resident_allocation_pointers(
    layers: &mut LoadedQwen36Layers,
    engine_buffers: [&GpuBuffer; 6],
) -> Vec<usize> {
    let mut pointers = collect_owned_layer_pointers(layers.layers())
        .into_iter()
        .collect::<Vec<_>>();
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
    let (_, scratch, _) = layers.execution_parts();
    if let Some(scratch) = scratch {
        pointers.extend(scratch.allocation_pointers());
    }
    pointers.extend(engine_buffers.map(|buffer| buffer.as_ptr() as usize));
    pointers.sort_unstable();
    pointers.dedup();
    pointers
}

fn collect_mapped_virtual_ranges(
    layers: &LoadedQwen36Layers,
) -> Vec<Qwen36MoeMappedVirtualRangeEvidence> {
    let mut ranges = Vec::new();
    let mut push = |buffer: &VirtualBuffer| {
        ranges.push(Qwen36MoeMappedVirtualRangeEvidence {
            address: buffer.as_ptr() as usize,
            stats: buffer.stats(),
        });
    };
    if let Some(arena) = layers.virtual_expert_arena() {
        for allocation in arena.allocations() {
            push(allocation.buffer());
        }
    }
    if let Some(manager) = layers.sparse_expert_residency() {
        for allocation in manager.arena().allocations() {
            push(allocation.buffer());
        }
    }
    for layer in layers.layers() {
        let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &layer.attn
        else {
            continue;
        };
        for buffer in [
            cache.virtual_kv_cache_k.as_ref(),
            cache.virtual_kv_cache_v.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            push(buffer);
        }
    }
    ranges.sort_unstable_by_key(|range| range.address);
    ranges
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::path::PathBuf;

    use gpu_hal::{Backend, GpuBuffer, ScalarType, VirtualBacking, VirtualBuffer};
    use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
    use model_store::VirtualArenaTransferBackend;
    use qwen36_moe::config::{Activation, RopeParameters, TextConfig};

    use super::{
        build_load_evidence, observe_source_open, reset_full_attention_cache, reset_phase,
        run_reset_transaction, validate_35b_a3b_config, validate_descriptor_pointer_ownership,
        validate_load_contract, zero_gpu_buffer, zero_mapped_virtual_buffer, LoadEvidenceInput,
        Qwen36MoeDirectProfile, Qwen36MoeEngine, Qwen36MoeLoadConfig, Qwen36MoeRouteState,
    };
    use crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
    use crate::qwen36_moe::types::{ExpertRoute, FullAttnKvCache};
    use crate::qwen36_moe_config::{Qwen36KvVmmMode, Qwen36MoeRuntimeConfig};

    const MODEL_MAX_CONTEXT: usize = 262_144;

    fn canonical_35b_a3b_execution_config() -> TextConfig {
        TextConfig {
            vocab_size: 248_320,
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 256,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            layer_types: (0..40)
                .map(|index| {
                    if (index + 1) % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect(),
            rope_parameters: Some(RopeParameters {
                mrope_interleaved: true,
                mrope_section: vec![11, 11, 10],
                ..RopeParameters::default()
            }),
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
    }

    #[test]
    fn canonical_35b_a3b_validation_covers_complete_execution_geometry() {
        validate_35b_a3b_config(&canonical_35b_a3b_execution_config())
            .expect("canonical execution geometry");

        let mismatches: &[(&str, fn(&mut TextConfig))] = &[
            ("vocab_size", |text| text.vocab_size -= 1),
            ("hidden_size", |text| text.hidden_size -= 1),
            ("num_hidden_layers", |text| text.num_hidden_layers -= 1),
            ("num_attention_heads", |text| text.num_attention_heads -= 1),
            ("num_key_value_heads", |text| text.num_key_value_heads -= 1),
            ("max_position_embeddings", |text| {
                text.max_position_embeddings -= 1
            }),
            ("rms_norm_eps", |text| text.rms_norm_eps = 1e-5),
            ("hidden_act", |text| text.hidden_act = Activation::Gelu),
            ("tie_word_embeddings", |text| {
                text.tie_word_embeddings = true
            }),
            ("head_dim", |text| text.head_dim -= 1),
            ("full_attention_interval", |text| {
                text.full_attention_interval -= 1
            }),
            ("attn_output_gate", |text| text.attn_output_gate = false),
            ("linear_conv_kernel_dim", |text| {
                text.linear_conv_kernel_dim -= 1
            }),
            ("linear_key_head_dim", |text| text.linear_key_head_dim -= 1),
            ("linear_value_head_dim", |text| {
                text.linear_value_head_dim -= 1
            }),
            ("linear_num_key_heads", |text| {
                text.linear_num_key_heads -= 1
            }),
            ("linear_num_value_heads", |text| {
                text.linear_num_value_heads -= 1
            }),
            ("layer_types", |text| {
                text.layer_types.swap(0, 3);
            }),
            ("rope_type", |text| {
                text.rope_parameters.as_mut().unwrap().rope_type = "linear".to_string()
            }),
            ("rope_theta", |text| {
                text.rope_parameters.as_mut().unwrap().rope_theta = 10_000.0
            }),
            ("partial_rotary_factor", |text| {
                text.rope_parameters.as_mut().unwrap().partial_rotary_factor = 0.5
            }),
            ("mrope_interleaved", |text| {
                text.rope_parameters.as_mut().unwrap().mrope_interleaved = false
            }),
            ("mrope_section", |text| {
                text.rope_parameters.as_mut().unwrap().mrope_section.clear()
            }),
            ("num_experts", |text| text.num_experts -= 1),
            ("num_experts_per_tok", |text| text.num_experts_per_tok -= 1),
            ("moe_intermediate_size", |text| {
                text.moe_intermediate_size -= 1
            }),
            ("shared_expert_intermediate_size", |text| {
                text.shared_expert_intermediate_size -= 1
            }),
            ("norm_topk_prob", |text| text.norm_topk_prob = false),
            ("router_aux_loss_coef", |text| {
                text.router_aux_loss_coef = 0.0
            }),
            ("mlp_only_layers", |text| text.mlp_only_layers.push(0)),
            ("decoder_sparse_step", |text| {
                text.decoder_sparse_step = Some(1)
            }),
        ];
        for &(field, mutate) in mismatches {
            let mut text = canonical_35b_a3b_execution_config();
            mutate(&mut text);
            assert!(
                validate_35b_a3b_config(&text).is_err(),
                "{field} mismatch was accepted"
            );
        }
    }

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
            source_open_count: 1,
        })
        .expect("production load evidence");

        assert_eq!(evidence.direct_profile.native_int4, 12);
        assert_eq!(evidence.direct_profile.bf16_fallback, 0);
        assert_eq!(evidence.source_bytes, 4096);
        assert_eq!(evidence.device_upload_bytes, 2048);
        assert_eq!(evidence.load_sequence, 7);
        assert_eq!(evidence.source_open_count, 1);
    }

    #[test]
    fn source_open_evidence_counts_observed_attempts() {
        let mut count = 0;
        let opened = observe_source_open(&mut count, || Ok::<_, anyhow::Error>("source"))
            .expect("observed source open");
        assert_eq!(opened, "source");
        assert_eq!(count, 1);

        let err = observe_source_open(&mut count, || Err::<(), _>(anyhow::anyhow!("open failed")))
            .expect_err("failed open is still an observed attempt");
        assert!(err.to_string().contains("open failed"));
        assert_eq!(count, 2);
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

    #[test]
    fn dense_fp8_kv_reset_clears_cache_scales_shadows_and_host_position() {
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
        let mut cache = FullAttnKvCache {
            k: Some(dirty_gpu_buffer()),
            v: Some(dirty_gpu_buffer()),
            kv_max_t: 8,
            kv_scale_k: Some(dirty_gpu_buffer()),
            kv_scale_v: Some(dirty_gpu_buffer()),
            kv_shadow_k: Some(dirty_gpu_buffer()),
            kv_shadow_v: Some(dirty_gpu_buffer()),
            kv_shadow_start: 5,
            kv_shadow_window: 8,
            virtual_kv_cache_k: None,
            virtual_kv_cache_v: None,
            virtual_kv_max_t: None,
        };
        let pointers = [
            cache.k.as_ref(),
            cache.v.as_ref(),
            cache.kv_scale_k.as_ref(),
            cache.kv_scale_v.as_ref(),
            cache.kv_shadow_k.as_ref(),
            cache.kv_shadow_v.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|buffer| buffer.as_ptr() as usize)
        .collect::<Vec<_>>();

        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        reset_full_attention_cache(0, 0, &mut cache).expect("reset dense FP8 KV cache");
        reset_full_attention_cache(0, 0, &mut cache).expect("repeat dense FP8 KV reset");
        let profile = gpu_hal::hal_profile_snapshot();
        gpu_hal::hal_profile_set_enabled(false);

        let buffers = [
            cache.k.as_ref(),
            cache.v.as_ref(),
            cache.kv_scale_k.as_ref(),
            cache.kv_scale_v.as_ref(),
            cache.kv_shadow_k.as_ref(),
            cache.kv_shadow_v.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        assert!(buffers.iter().all(|buffer| {
            buffer
                .to_host_bytes()
                .expect("read reset dense KV state")
                .iter()
                .all(|byte| *byte == 0)
        }));
        assert_eq!(
            buffers
                .iter()
                .map(|buffer| buffer.as_ptr() as usize)
                .collect::<Vec<_>>(),
            pointers
        );
        assert_eq!(cache.kv_shadow_start, -1);
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

    #[test]
    fn reset_attempts_post_sync_after_clear_failure_and_preserves_first_error() {
        let phases = RefCell::new(Vec::new());
        let err = run_reset_transaction(
            || {
                phases.borrow_mut().push("pre-sync");
                Ok(())
            },
            || {
                phases.borrow_mut().push("clear");
                reset_phase("layer-state", Err(anyhow::anyhow!("clear failed")))
            },
            || {
                phases.borrow_mut().push("post-sync");
                Err(anyhow::anyhow!("sync failed"))
            },
        )
        .expect_err("clear failure must propagate");

        assert_eq!(*phases.borrow(), ["pre-sync", "clear", "post-sync"]);
        let message = format!("{err:#}");
        assert!(
            message.starts_with(
                "Qwen3.6 engine reset integrity failure during layer-state: clear failed"
            ),
            "{message}"
        );
        assert!(message.contains("device-sync-after"), "{message}");
        assert!(message.contains("sync failed"), "{message}");
    }

    #[test]
    fn reset_pre_sync_failure_does_not_start_clear_or_post_sync() {
        let phases = RefCell::new(Vec::new());
        let err = run_reset_transaction(
            || {
                phases.borrow_mut().push("pre-sync");
                Err(anyhow::anyhow!("pre-sync failed"))
            },
            || {
                phases.borrow_mut().push("clear");
                Ok(())
            },
            || {
                phases.borrow_mut().push("post-sync");
                Ok(())
            },
        )
        .expect_err("pre-sync failure must propagate");

        assert_eq!(*phases.borrow(), ["pre-sync"]);
        assert!(format!("{err:#}").contains("device-sync-before"));
    }
}
