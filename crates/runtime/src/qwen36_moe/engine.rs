use std::collections::{HashMap, HashSet};
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
use crate::qwen36_moe::chain::{run_chain_step, Qwen36ChainStep, Qwen36ChainStepOutput};
use crate::qwen36_moe::decode::Qwen36ExecutionOptions;
use crate::qwen36_moe::geometry::build_multi_layer_geom;
use crate::qwen36_moe::layer_loader::{
    load_qwen36_layers, Qwen36LayerLoadStrategy, Qwen36LoadOptions,
    Qwen36WeightMode as LayerWeightMode, SparseExpertLoadOptions,
};
use crate::qwen36_moe::layers::LoadedQwen36Layers;
use crate::qwen36_moe::lm_head::{
    bf16_bytes_to_f32, launch_lm_head_from_final_hidden_bytes, LmHeadBuffers,
};
use crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
use crate::qwen36_moe::persistent_decode::{
    build_int4_descs, build_kv_fp8_descs, build_layer_descs, LmHeadFold,
};
use crate::qwen36_moe::prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe::prefill::{
    lookup_embed_row, run_batched_prefill_with_workspace, PrefillTokenTimings,
    Qwen36PrefillWorkspace,
};
use crate::qwen36_moe::route_telemetry::{MoeRouteTelemetry, MoeTransitionPredictor};
use crate::qwen36_moe::source::{Qwen36MoeSource, Qwen36MoeSourceOpenObserver, Qwen36WeightMode};
use crate::qwen36_moe::types::{
    AttnLayerBuffers, ExpertRoute, FullAttnKvCache, LayerBuffers, MultiLayerGeom, PositionPair,
};
use crate::qwen36_moe::weights::{load_to_gpu, prepare_lm_head_bf16};
use crate::qwen36_moe_config::{
    should_try_moe_expert_vmm, should_use_qwen36_kv_vmm, MoeExpertVmmMode, Qwen36MoeRuntimeConfig,
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
    pub execution_options: Qwen36ExecutionOptions,
    pub accurate_stage_timings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoePrefillBoundary {
    PrefixStarted,
    FinalProductionStarted,
}

#[derive(Debug)]
pub struct Qwen36MoePrefillOutput {
    pub logits: Vec<f32>,
    pub prefix_token_count: usize,
    pub prefix_duration: Duration,
    pub final_production_duration: Duration,
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
    pub store_open_duration: Duration,
    pub config_duration: Duration,
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
    pub config: Option<qwen36_moe::config::Config>,
    pub tokenizer_timings: crate::flm_tokenizer::QwenBpeTokenizerTimings,
    pub hal_profile: HalProfileSnapshot,
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
    load_sequence: u64,
    source_open_count: u64,
}

fn build_load_evidence(
    input: LoadEvidenceInput,
    profile: &HalProfileSnapshot,
) -> Result<Qwen36MoeLoadEvidence> {
    let upload = load_upload_profile_evidence(profile);
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
    if upload.device_upload_bytes == 0 {
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
        device_upload_bytes: upload.device_upload_bytes,
        source_open_duration: Duration::ZERO,
        store_open_duration: Duration::ZERO,
        config_duration: Duration::ZERO,
        descriptor_duration: Duration::ZERO,
        tokenizer_duration: Duration::ZERO,
        plan_duration: Duration::ZERO,
        allocation_duration: Duration::ZERO,
        upload_duration: upload.upload_duration,
        total_duration: Duration::ZERO,
        load_sequence: input.load_sequence,
        source_open_count: input.source_open_count,
        resident_allocation_count: 0,
        resident_allocation_pointers: Vec::new(),
        mapped_virtual_ranges: Vec::new(),
        config: None,
        tokenizer_timings: crate::flm_tokenizer::QwenBpeTokenizerTimings::default(),
        hal_profile: profile.clone(),
    })
}

#[derive(Default)]
struct Qwen36SessionPosition {
    state: Qwen36SessionState,
}

#[derive(Default)]
enum Qwen36SessionState {
    #[default]
    Ready,
    Active {
        next: usize,
    },
    NeedsReset,
}

impl Qwen36SessionPosition {
    fn validate_prefill(&self, prompt_len: usize, max_context_len: usize) -> Result<()> {
        if prompt_len == 0 {
            anyhow::bail!("Qwen3.6 prefill rejects an empty prompt");
        }
        if prompt_len > max_context_len {
            anyhow::bail!(
                "Qwen3.6 prefill prompt length {prompt_len} exceeds context {max_context_len}"
            );
        }
        match self.state {
            Qwen36SessionState::Ready => {}
            Qwen36SessionState::Active { next } => {
                anyhow::bail!(
                    "Qwen3.6 prefill requires reset before starting a new request; \
                     current next absolute position is {next}"
                );
            }
            Qwen36SessionState::NeedsReset => {
                anyhow::bail!(
                    "Qwen3.6 prefill requires reset after failed or incomplete serving execution"
                );
            }
        }
        Ok(())
    }

    fn execution_started(&mut self) {
        self.state = Qwen36SessionState::NeedsReset;
    }

    fn prefill_succeeded(&mut self, prompt_len: usize) {
        self.state = Qwen36SessionState::Active { next: prompt_len };
    }

    fn validate_decode(&self, absolute_pos: usize, max_context_len: usize) -> Result<()> {
        let expected = match self.state {
            Qwen36SessionState::Ready => {
                return Err(anyhow!("Qwen3.6 decode_step called before prefill"));
            }
            Qwen36SessionState::Active { next } => next,
            Qwen36SessionState::NeedsReset => {
                anyhow::bail!(
                    "Qwen3.6 decode_step requires reset after failed or incomplete serving execution"
                );
            }
        };
        if absolute_pos != expected {
            anyhow::bail!(
                "Qwen3.6 decode_step expected absolute position {expected}, got {absolute_pos}"
            );
        }
        if absolute_pos >= max_context_len {
            anyhow::bail!(
                "Qwen3.6 decode_step absolute position {absolute_pos} exceeds context \
                 {max_context_len}"
            );
        }
        Ok(())
    }

    fn decode_succeeded(&mut self, absolute_pos: usize) -> Result<()> {
        let next = absolute_pos
            .checked_add(1)
            .ok_or_else(|| anyhow!("Qwen3.6 decode_step absolute position overflow"))?;
        self.state = Qwen36SessionState::Active { next };
        Ok(())
    }

    fn reset(&mut self) {
        self.state = Qwen36SessionState::Ready;
    }

    fn next(&self) -> Option<usize> {
        match self.state {
            Qwen36SessionState::Active { next } => Some(next),
            Qwen36SessionState::Ready | Qwen36SessionState::NeedsReset => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen36LmHeadSelection {
    DenseFolded,
    SparseStandalone,
}

#[allow(dead_code)]
enum Qwen36ServingEvent<'a> {
    Prefix {
        tokens: &'a [u32],
        positions: &'a [PositionPair],
    },
    ProductionToken {
        token_id: u32,
        absolute_pos: usize,
    },
    LmHead(Qwen36LmHeadSelection),
    OutputCompleted,
    PositionCommitted {
        next_position: usize,
    },
}

trait Qwen36ServingObserver {
    fn observe(&mut self, event: Qwen36ServingEvent<'_>);
}

struct IgnoreServingEvents;

impl Qwen36ServingObserver for IgnoreServingEvents {
    fn observe(&mut self, _event: Qwen36ServingEvent<'_>) {}
}

trait Qwen36ServingBackend {
    type PendingOutput;

    fn run_prefix(&mut self, tokens: &[u32], positions: &[PositionPair]) -> Result<()>;

    fn run_production_token(
        &mut self,
        token_id: u32,
        absolute_pos: usize,
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Self::PendingOutput>;

    fn complete_output(
        &mut self,
        pending: Self::PendingOutput,
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Vec<f32>>;
}

fn validate_serving_token_ids(token_ids: &[u32], vocab: usize, operation: &str) -> Result<()> {
    for (index, &token_id) in token_ids.iter().enumerate() {
        if token_id as usize >= vocab {
            anyhow::bail!(
                "Qwen3.6 {operation} token {token_id} at index {index} is outside vocabulary {vocab}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
fn run_prefill_orchestration<B: Qwen36ServingBackend>(
    session: &mut Qwen36SessionPosition,
    prompt_ids: &[u32],
    max_context_len: usize,
    vocab: usize,
    lm_head: Qwen36LmHeadSelection,
    backend: &mut B,
    observer: &mut impl Qwen36ServingObserver,
) -> Result<Vec<f32>> {
    let mut ignore_boundaries = |_| Ok(());
    Ok(run_prefill_orchestration_with_boundaries(
        session,
        prompt_ids,
        max_context_len,
        vocab,
        lm_head,
        backend,
        observer,
        &mut ignore_boundaries,
    )?
    .logits)
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_orchestration_with_boundaries<B: Qwen36ServingBackend>(
    session: &mut Qwen36SessionPosition,
    prompt_ids: &[u32],
    max_context_len: usize,
    vocab: usize,
    lm_head: Qwen36LmHeadSelection,
    backend: &mut B,
    observer: &mut impl Qwen36ServingObserver,
    boundary_observer: &mut impl FnMut(Qwen36MoePrefillBoundary) -> Result<()>,
) -> Result<Qwen36MoePrefillOutput> {
    session.validate_prefill(prompt_ids.len(), max_context_len)?;
    validate_serving_token_ids(prompt_ids, vocab, "prefill")?;

    let final_index = prompt_ids.len() - 1;
    let positions = (0..final_index)
        .map(|position| PositionPair::dense(position as i32))
        .collect::<Vec<_>>();
    session.execution_started();

    boundary_observer(Qwen36MoePrefillBoundary::PrefixStarted)?;
    observer.observe(Qwen36ServingEvent::Prefix {
        tokens: &prompt_ids[..final_index],
        positions: &positions,
    });
    let prefix_start = Instant::now();
    backend.run_prefix(&prompt_ids[..final_index], &positions)?;
    let prefix_duration = prefix_start.elapsed();

    let final_token = prompt_ids[final_index];
    boundary_observer(Qwen36MoePrefillBoundary::FinalProductionStarted)?;
    observer.observe(Qwen36ServingEvent::ProductionToken {
        token_id: final_token,
        absolute_pos: final_index,
    });
    observer.observe(Qwen36ServingEvent::LmHead(lm_head));
    let final_production_start = Instant::now();
    let pending = backend.run_production_token(final_token, final_index, lm_head)?;
    let logits = backend.complete_output(pending, lm_head)?;
    let final_production_duration = final_production_start.elapsed();
    observer.observe(Qwen36ServingEvent::OutputCompleted);

    session.prefill_succeeded(prompt_ids.len());
    observer.observe(Qwen36ServingEvent::PositionCommitted {
        next_position: prompt_ids.len(),
    });
    Ok(Qwen36MoePrefillOutput {
        logits,
        prefix_token_count: final_index,
        prefix_duration,
        final_production_duration,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_decode_orchestration<B: Qwen36ServingBackend>(
    session: &mut Qwen36SessionPosition,
    token_id: u32,
    absolute_pos: usize,
    max_context_len: usize,
    vocab: usize,
    lm_head: Qwen36LmHeadSelection,
    backend: &mut B,
    observer: &mut impl Qwen36ServingObserver,
) -> Result<Vec<f32>> {
    session.validate_decode(absolute_pos, max_context_len)?;
    validate_serving_token_ids(&[token_id], vocab, "decode_step")?;
    session.execution_started();

    observer.observe(Qwen36ServingEvent::ProductionToken {
        token_id,
        absolute_pos,
    });
    observer.observe(Qwen36ServingEvent::LmHead(lm_head));
    let pending = backend.run_production_token(token_id, absolute_pos, lm_head)?;
    let logits = backend.complete_output(pending, lm_head)?;
    observer.observe(Qwen36ServingEvent::OutputCompleted);

    session.decode_succeeded(absolute_pos)?;
    observer.observe(Qwen36ServingEvent::PositionCommitted {
        next_position: absolute_pos + 1,
    });
    Ok(logits)
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
    prefill_workspace: Option<Qwen36PrefillWorkspace>,
    route_state: Qwen36MoeRouteState,
    session_position: Qwen36SessionPosition,
    source_open_observer: Qwen36MoeSourceOpenObserver,
    load_evidence: Qwen36MoeLoadEvidence,
    backend: Backend,
    device_ordinal: usize,
    max_context_len: usize,
    execution_options: Qwen36ExecutionOptions,
    accurate_stage_timings: bool,
}

struct Qwen36MoeRouteState {
    num_layers: usize,
    top_k: usize,
    sparse: bool,
    policy: Qwen36MoeRuntimeConfig,
    previous_topk_by_layer: Vec<Vec<usize>>,
    telemetry: Option<MoeRouteTelemetry>,
    predictors: Option<Vec<MoeTransitionPredictor>>,
    hot_expert_counts: Option<Vec<HashMap<usize, u32>>>,
}

impl Qwen36MoeRouteState {
    #[cfg(test)]
    fn new(
        top_k: usize,
        num_layers: usize,
        sparse: bool,
        transition_min_observations: u32,
    ) -> Self {
        let mut policy = Qwen36MoeRuntimeConfig {
            sparse_requested: sparse,
            transition_min_observations,
            ..Qwen36MoeRuntimeConfig::default()
        };
        if !sparse {
            policy.transition_min_observations = 0;
        }
        Self::with_policy(top_k, num_layers, policy)
    }

    fn with_policy(top_k: usize, num_layers: usize, policy: Qwen36MoeRuntimeConfig) -> Self {
        let sparse = policy.sparse_requested;
        let transition_min_observations = policy.transition_min_observations;
        let track_hot_experts =
            policy.hot_protect_min_hits.is_some() || policy.fixed_hot_min_hits.is_some();
        Self {
            num_layers,
            top_k,
            sparse,
            policy,
            previous_topk_by_layer: vec![Vec::new(); num_layers],
            telemetry: sparse.then(|| MoeRouteTelemetry::new(top_k)),
            predictors: (sparse && transition_min_observations > 0).then(|| {
                (0..num_layers)
                    .map(|_| MoeTransitionPredictor::new(top_k, transition_min_observations))
                    .collect()
            }),
            hot_expert_counts: (sparse && track_hot_experts)
                .then(|| vec![HashMap::new(); num_layers]),
        }
    }

    fn reset(&mut self) {
        for routes in &mut self.previous_topk_by_layer {
            routes.clear();
        }
        self.telemetry = self.sparse.then(|| MoeRouteTelemetry::new(self.top_k));
        self.predictors = (self.sparse && self.policy.transition_min_observations > 0).then(|| {
            (0..self.num_layers)
                .map(|_| {
                    MoeTransitionPredictor::new(self.top_k, self.policy.transition_min_observations)
                })
                .collect()
        });
        if let Some(counts) = self.hot_expert_counts.as_mut() {
            for layer in counts {
                layer.clear();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &mut self,
        ordinal: usize,
        geom: &MultiLayerGeom,
        store: &model_store::BakedStore,
        loaded_layers: &mut LoadedQwen36Layers,
        initial_hidden: &[u8],
        position: PositionPair,
        step: usize,
        fold: Option<LmHeadFold<'_>>,
        download_final_hidden: bool,
        accurate_stage_timings: bool,
        execution: &Qwen36ExecutionOptions,
    ) -> Result<Qwen36ChainStepOutput> {
        let sparse = loaded_layers.has_sparse_expert_residency();
        if sparse != self.sparse {
            anyhow::bail!(
                "Qwen3.6 route-state residency mismatch: owner sparse={sparse}, \
                 policy sparse={}",
                self.sparse
            );
        }
        let mut next_topk_by_layer = vec![Vec::new(); self.num_layers];
        let output = {
            let policy = &self.policy;
            let previous_topk_by_layer = &self.previous_topk_by_layer;
            let telemetry = &mut self.telemetry;
            let predictors = &mut self.predictors;
            let hot_expert_counts = &mut self.hot_expert_counts;
            let mut prefetch =
                |manager: &mut crate::qwen36_moe::residency::MoeExpertResidencyManager,
                 phase,
                 layer_idx,
                 routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        store,
                        policy.prefetch_mode,
                        policy.prefetch_ranks,
                        policy.prefetch_evict_min_probability,
                        policy.protect_demand_routes,
                        policy.hot_protect_min_hits,
                        policy.fixed_hot_min_hits,
                        previous_topk_by_layer,
                        &mut next_topk_by_layer,
                        sparse,
                        telemetry.as_mut(),
                        predictors.as_deref_mut(),
                        hot_expert_counts.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
            let expert_prefetch = sparse.then_some(
                &mut prefetch as &mut crate::qwen36_moe::chain::ChainExpertPrefetchCallback<'_>,
            );
            run_chain_step(Qwen36ChainStep {
                ordinal,
                geom,
                loaded_layers,
                initial_hidden,
                position,
                step,
                accurate_stage_timings,
                execution,
                fold,
                download_final_hidden,
                expert_prefetch,
            })
        }?;
        if sparse {
            self.previous_topk_by_layer = next_topk_by_layer;
        }
        Ok(output)
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
    prefill_workspace: Option<Qwen36PrefillWorkspace>,
}

impl Qwen36MoeEngine {
    pub fn load(config: Qwen36MoeLoadConfig) -> Result<Self> {
        let _load_guard = ENGINE_LOAD_LOCK
            .lock()
            .map_err(|_| anyhow!("Qwen3.6 engine load lock poisoned"))?;
        let total_start = Instant::now();
        validate_pre_source_load_policy(&config)?;

        let source_open_observer = Qwen36MoeSourceOpenObserver::for_path(&config.flm_path);
        let source_open_start = Instant::now();
        let mut source = Qwen36MoeSource::open(
            &config.flm_path,
            FlmModelSourceOptions {
                int4_runtime: true,
                verify_block_hashes: config.verify_block_hashes,
            },
        )?;
        let source_open_duration = source_open_start.elapsed();
        let source_open_count = source_open_observer.observed_count();
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
        let tokenizer_timings = tokenizer_load.timings;
        source.timings.tokenizer = tokenizer_duration;
        source.timings.tokenizer_assets = tokenizer_timings.asset_lookup;
        source.timings.tokenizer_parse = tokenizer_timings.parse;
        source.timings.tokenizer_build = tokenizer_timings.build;
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
        let load_sequence = ENGINE_LOAD_SEQUENCE.fetch_add(1, Ordering::SeqCst) + 1;
        let mut load_evidence = build_load_evidence(
            LoadEvidenceInput {
                direct_profile: source.direct_profile,
                source_bytes,
                load_sequence,
                source_open_count,
            },
            &hal_profile,
        )?;
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
        load_evidence.store_open_duration = source.timings.store_open;
        load_evidence.config_duration = source.timings.config;
        load_evidence.descriptor_duration = persistent_descriptor_duration(&mut gpu.layers)?;
        load_evidence.tokenizer_duration = tokenizer_duration;
        load_evidence.plan_duration = source.timings.direct_plan;
        load_evidence.allocation_duration = profile_duration(&hal_profile, |op| {
            op == "alloc" || op.starts_with("vmm_reserve") || op.starts_with("vmm_map")
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
        if let Some(workspace) = gpu.prefill_workspace.as_ref() {
            load_evidence
                .resident_allocation_pointers
                .extend(workspace.allocation_pointers());
            load_evidence.resident_allocation_pointers.sort_unstable();
            load_evidence.resident_allocation_pointers.dedup();
        }
        load_evidence.resident_allocation_count =
            load_evidence.resident_allocation_pointers.len() as u64;
        load_evidence.mapped_virtual_ranges = collect_mapped_virtual_ranges(&gpu.layers);
        load_evidence.config = Some(source.config.clone());
        load_evidence.tokenizer_timings = tokenizer_timings;

        let route_state = Qwen36MoeRouteState::with_policy(
            geom.top_k as usize,
            geom.num_layers as usize,
            config.policy.moe.clone(),
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
            prefill_workspace: gpu.prefill_workspace,
            route_state,
            session_position: Qwen36SessionPosition::default(),
            source_open_observer,
            load_evidence,
            backend: config.backend,
            device_ordinal: config.device_ordinal,
            max_context_len: config.max_context_len,
            execution_options: config.execution_options,
            accurate_stage_timings: config.accurate_stage_timings,
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

    pub fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.prefill_with_boundaries(prompt_ids, |_| Ok(()))?.logits)
    }

    pub fn prefill_with_boundaries(
        &mut self,
        prompt_ids: &[u32],
        mut boundary_observer: impl FnMut(Qwen36MoePrefillBoundary) -> Result<()>,
    ) -> Result<Qwen36MoePrefillOutput> {
        let lm_head = self.serving_lm_head_selection();
        let max_context_len = self.max_context_len;
        let vocab = self.geom.vocab as usize;
        let mut session = std::mem::take(&mut self.session_position);
        let mut observer = IgnoreServingEvents;
        let result = run_prefill_orchestration_with_boundaries(
            &mut session,
            prompt_ids,
            max_context_len,
            vocab,
            lm_head,
            self,
            &mut observer,
            &mut boundary_observer,
        );
        self.session_position = session;
        result
    }

    pub fn decode_step(&mut self, token_id: u32, absolute_pos: usize) -> Result<Vec<f32>> {
        let lm_head = self.serving_lm_head_selection();
        let max_context_len = self.max_context_len;
        let vocab = self.geom.vocab as usize;
        let mut session = std::mem::take(&mut self.session_position);
        let mut observer = IgnoreServingEvents;
        let result = run_decode_orchestration(
            &mut session,
            token_id,
            absolute_pos,
            max_context_len,
            vocab,
            lm_head,
            self,
            &mut observer,
        );
        self.session_position = session;
        result
    }

    fn serving_lm_head_selection(&self) -> Qwen36LmHeadSelection {
        if self.layers.has_sparse_expert_residency() {
            Qwen36LmHeadSelection::SparseStandalone
        } else {
            Qwen36LmHeadSelection::DenseFolded
        }
    }

    #[doc(hidden)]
    pub fn test_only_observed_load_sequence() -> u64 {
        ENGINE_LOAD_SEQUENCE.load(Ordering::SeqCst)
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
            let observations = self.route_state.policy.transition_min_observations.max(1);
            for _ in 0..observations {
                predictors[0].update(&routes, &[7, 11]);
            }
        }
        self.session_position.prefill_succeeded(73);
        gpu_hal::sync(self.device_ordinal).context("sync after dirty-reset hook")
    }

    #[doc(hidden)]
    pub fn test_only_reset_snapshot(&mut self) -> Result<Qwen36MoeResetTestSnapshot> {
        gpu_hal::sync(self.device_ordinal).context("sync before reset test snapshot")?;
        let mut resident_allocation_pointers = collect_resident_allocation_pointers(
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
        if let Some(workspace) = self.prefill_workspace.as_ref() {
            resident_allocation_pointers.extend(workspace.allocation_pointers());
            resident_allocation_pointers.sort_unstable();
            resident_allocation_pointers.dedup();
        }
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
            source_open_count: self.source_open_observer.observed_count(),
            resident_allocation_pointers,
            mapped_virtual_ranges,
            persistent_descriptor_bytes,
            mutable_nonzero_labels,
            route_history_entries,
            route_observations,
            transition_candidates,
            next_position: self.session_position.next(),
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
        self.session_position.reset();
        Ok(())
    }

    fn validate_resident_identity(&mut self) -> Result<()> {
        let mut current_pointers = collect_resident_allocation_pointers(
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
        if let Some(workspace) = self.prefill_workspace.as_ref() {
            current_pointers.extend(workspace.allocation_pointers());
            current_pointers.sort_unstable();
            current_pointers.dedup();
        }
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

impl Qwen36ServingBackend for Qwen36MoeEngine {
    type PendingOutput = Qwen36ChainStepOutput;

    fn run_prefix(&mut self, tokens: &[u32], positions: &[PositionPair]) -> Result<()> {
        let execution = &self.execution_options;
        let accurate_stage_timings = self.accurate_stage_timings;
        let ordinal = self.device_ordinal;
        let geom = &self.geom;
        let store = &self.source.source.store;
        let route_state = &mut self.route_state;
        let prefill_workspace = self.prefill_workspace.as_mut();
        let mut fallback = |callback_layers: &mut LoadedQwen36Layers,
                            step: usize,
                            token: u32,
                            position: PositionPair|
         -> Result<PrefillTokenTimings> {
            let embed_start = Instant::now();
            let initial_hidden = lookup_embed_row(
                store,
                QWEN36_35B_A3B_WEIGHT_PREFIX,
                token as usize,
                geom.hidden as usize,
            )
            .with_context(|| {
                format!("Qwen3.6 serving prefill embedding token {token} at step {step}")
            })?;
            let embed = embed_start.elapsed();
            let chain_start = Instant::now();
            route_state.run_step(
                ordinal,
                geom,
                store,
                callback_layers,
                &initial_hidden,
                position,
                step,
                None,
                false,
                accurate_stage_timings,
                execution,
            )?;
            Ok(PrefillTokenTimings {
                embed,
                chain: chain_start.elapsed(),
            })
        };
        run_batched_prefill_with_workspace(
            ordinal,
            geom,
            store,
            QWEN36_35B_A3B_WEIGHT_PREFIX,
            &mut self.layers,
            tokens,
            positions,
            accurate_stage_timings,
            execution,
            Some(&mut fallback),
            None,
            prefill_workspace,
        )
        .context("Qwen3.6 serving batched prefill")?;
        Ok(())
    }

    fn run_production_token(
        &mut self,
        token_id: u32,
        absolute_pos: usize,
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Self::PendingOutput> {
        let expected = self.serving_lm_head_selection();
        if lm_head != expected {
            anyhow::bail!(
                "Qwen3.6 serving LM-head selection changed during execution: \
                 planned {lm_head:?}, current {expected:?}"
            );
        }
        let execution = &self.execution_options;
        let initial_hidden = lookup_embed_row(
            &self.source.source.store,
            QWEN36_35B_A3B_WEIGHT_PREFIX,
            token_id as usize,
            self.geom.hidden as usize,
        )
        .with_context(|| {
            format!(
                "Qwen3.6 serving embedding token {token_id} at absolute position {absolute_pos}"
            )
        })?;
        let fold = match lm_head {
            Qwen36LmHeadSelection::DenseFolded => Some(LmHeadFold {
                final_norm_w: &self.final_norm_w,
                lm_head_w: &self.lm_head_w,
                logits_out: Some(&mut self.logits),
                top1_out: None,
                vocab: self.geom.vocab,
            }),
            Qwen36LmHeadSelection::SparseStandalone => None,
        };
        self.route_state.run_step(
            self.device_ordinal,
            &self.geom,
            &self.source.source.store,
            &mut self.layers,
            &initial_hidden,
            PositionPair::dense(absolute_pos as i32),
            absolute_pos,
            fold,
            lm_head == Qwen36LmHeadSelection::SparseStandalone,
            self.accurate_stage_timings,
            execution,
        )
    }

    fn complete_output(
        &mut self,
        pending: Self::PendingOutput,
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Vec<f32>> {
        let expected_folded = lm_head == Qwen36LmHeadSelection::DenseFolded;
        if pending.lm_head_folded != expected_folded {
            anyhow::bail!(
                "Qwen3.6 serving LM-head completion mismatch: selection {lm_head:?}, \
                 chain folded={}",
                pending.lm_head_folded
            );
        }
        let logits_bytes = match lm_head {
            Qwen36LmHeadSelection::DenseFolded => self
                .logits
                .to_host_bytes()
                .context("download Qwen3.6 folded serving logits")?,
            Qwen36LmHeadSelection::SparseStandalone => launch_lm_head_from_final_hidden_bytes(
                self.device_ordinal,
                &self.geom,
                &pending.outputs.final_hidden_bytes,
                &self.execution_options.prefill_kernel,
                LmHeadBuffers {
                    final_norm_w: &self.final_norm_w,
                    lm_head_w: &self.lm_head_w,
                    final_hidden: &mut self.final_hidden,
                    logits: &mut self.logits,
                    counter: &mut self.counter,
                },
            )
            .context("run Qwen3.6 serving LM head")?,
        };
        Ok(bf16_bytes_to_f32(&logits_bytes))
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
    let outer_profile_active = gpu_hal::hal_profile_enabled();
    let before = gpu_hal::hal_profile_snapshot();
    if !outer_profile_active {
        gpu_hal::hal_profile_set_enabled(true);
    }
    let result = load();
    let after = gpu_hal::hal_profile_snapshot();
    if !outer_profile_active {
        gpu_hal::hal_profile_set_enabled(false);
    }
    (result, hal_profile_delta(&before, &after))
}

fn hal_profile_delta(
    before: &HalProfileSnapshot,
    after: &HalProfileSnapshot,
) -> HalProfileSnapshot {
    let mut delta = HalProfileSnapshot::default();
    for after_entry in &after.entries {
        let before_entry = before
            .entries
            .iter()
            .find(|entry| entry.op == after_entry.op);
        let calls = after_entry
            .calls
            .saturating_sub(before_entry.map_or(0, |entry| entry.calls));
        let total_ms =
            (after_entry.total_ms - before_entry.map_or(0.0, |entry| entry.total_ms)).max(0.0);
        let total_bytes = after_entry
            .total_bytes
            .saturating_sub(before_entry.map_or(0, |entry| entry.total_bytes));
        if calls == 0 && total_ms == 0.0 && total_bytes == 0 {
            continue;
        }
        let max_ms = match before_entry {
            None => after_entry.max_ms,
            Some(entry) if after_entry.max_ms > entry.max_ms => after_entry.max_ms,
            Some(_) => 0.0,
        };
        let entry = gpu_hal::HalProfileEntry {
            op: after_entry.op.clone(),
            calls,
            total_ms,
            max_ms,
            total_bytes,
        };
        delta.total_calls += entry.calls;
        delta.total_ms += entry.total_ms;
        match entry.op.as_str() {
            "alloc" => {
                delta.alloc_calls += entry.calls;
                delta.alloc_bytes += entry.total_bytes;
            }
            "free" => delta.free_calls += entry.calls,
            "copy_h2d" => delta.h2d_bytes += entry.total_bytes,
            "copy_d2h" => delta.d2h_bytes += entry.total_bytes,
            "copy_d2d" => delta.d2d_bytes += entry.total_bytes,
            "memset_zeros" => delta.memset_bytes += entry.total_bytes,
            "sync" => delta.sync_calls += entry.calls,
            _ => {}
        }
        delta.entries.push(entry);
    }
    delta.entries.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.op.cmp(&rhs.op))
    });
    delta
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
    let prefill_workspace = Qwen36PrefillWorkspace::allocate_for_engine(
        config.device_ordinal,
        geom,
        &layers,
        config.max_context_len,
    )
    .context("allocate Qwen3.6 engine prefill workspace")?;

    Ok(ResidentGpuParts {
        layers,
        embed_w,
        final_norm_w,
        lm_head_w,
        logits,
        counter,
        final_hidden,
        prefill_workspace,
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

struct LoadUploadProfileEvidence {
    device_upload_bytes: u64,
    upload_duration: Duration,
}

fn is_load_upload_operation(op: &str) -> bool {
    matches!(op, "copy_h2d" | "copy_h2d_async" | "copy_storage_to_device")
}

fn load_upload_profile_evidence(profile: &HalProfileSnapshot) -> LoadUploadProfileEvidence {
    let (device_upload_bytes, total_ms) = profile
        .entries
        .iter()
        .filter(|entry| is_load_upload_operation(&entry.op))
        .fold((0u64, 0.0f64), |(bytes, milliseconds), entry| {
            (
                bytes.saturating_add(entry.total_bytes),
                milliseconds + entry.total_ms,
            )
        });
    LoadUploadProfileEvidence {
        device_upload_bytes,
        upload_duration: Duration::from_secs_f64(total_ms / 1000.0),
    }
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
    use std::time::Duration;

    use anyhow::Result;
    use gpu_hal::{
        Backend, GpuBuffer, HalProfileEntry, HalProfileSnapshot, ScalarType, VirtualBacking,
        VirtualBuffer,
    };
    use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
    use model_store::VirtualArenaTransferBackend;
    use qwen36_moe::config::{Activation, RopeParameters, TextConfig};

    use super::{
        build_load_evidence, profile_gpu_load, reset_full_attention_cache, reset_phase,
        run_decode_orchestration, run_prefill_orchestration,
        run_prefill_orchestration_with_boundaries, run_reset_transaction, validate_35b_a3b_config,
        validate_descriptor_pointer_ownership, validate_load_contract, zero_gpu_buffer,
        zero_mapped_virtual_buffer, LoadEvidenceInput, Qwen36ExecutionOptions,
        Qwen36LmHeadSelection, Qwen36MoeDirectProfile, Qwen36MoeEngine, Qwen36MoeLoadConfig,
        Qwen36MoePrefillBoundary, Qwen36MoeRouteState, Qwen36ServingBackend, Qwen36ServingEvent,
        Qwen36ServingObserver, Qwen36SessionPosition,
    };
    use crate::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
    use crate::qwen36_moe::source::{Qwen36MoeSource, Qwen36MoeSourceOpenObserver};
    use crate::qwen36_moe::types::{ExpertRoute, FullAttnKvCache, PositionPair};
    use crate::qwen36_moe_config::{Qwen36KvVmmMode, Qwen36MoeRuntimeConfig};

    const MODEL_MAX_CONTEXT: usize = 262_144;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum RecordedServingEvent {
        Prefix {
            tokens: Vec<u32>,
            positions: Vec<crate::qwen36_moe::types::PositionPair>,
        },
        ProductionToken {
            token_id: u32,
            absolute_pos: usize,
        },
        LmHead(Qwen36LmHeadSelection),
        OutputCompleted,
        PositionCommitted(usize),
    }

    #[derive(Default)]
    struct RecordingServingObserver {
        events: Vec<RecordedServingEvent>,
    }

    impl Qwen36ServingObserver for RecordingServingObserver {
        fn observe(&mut self, event: Qwen36ServingEvent<'_>) {
            self.events.push(match event {
                Qwen36ServingEvent::Prefix { tokens, positions } => RecordedServingEvent::Prefix {
                    tokens: tokens.to_vec(),
                    positions: positions.to_vec(),
                },
                Qwen36ServingEvent::ProductionToken {
                    token_id,
                    absolute_pos,
                } => RecordedServingEvent::ProductionToken {
                    token_id,
                    absolute_pos,
                },
                Qwen36ServingEvent::LmHead(selection) => RecordedServingEvent::LmHead(selection),
                Qwen36ServingEvent::OutputCompleted => RecordedServingEvent::OutputCompleted,
                Qwen36ServingEvent::PositionCommitted { next_position } => {
                    RecordedServingEvent::PositionCommitted(next_position)
                }
            });
        }
    }

    #[derive(Default)]
    struct InjectedServingBackend {
        prefix_calls: Vec<(Vec<u32>, Vec<PositionPair>)>,
        production_calls: Vec<(u32, usize, Qwen36LmHeadSelection)>,
        fail_final_token: bool,
        fail_output: bool,
    }

    impl Qwen36ServingBackend for InjectedServingBackend {
        type PendingOutput = (u32, usize);

        fn run_prefix(&mut self, tokens: &[u32], positions: &[PositionPair]) -> Result<()> {
            self.prefix_calls
                .push((tokens.to_vec(), positions.to_vec()));
            Ok(())
        }

        fn run_production_token(
            &mut self,
            token_id: u32,
            absolute_pos: usize,
            lm_head: Qwen36LmHeadSelection,
        ) -> Result<Self::PendingOutput> {
            self.production_calls
                .push((token_id, absolute_pos, lm_head));
            if self.fail_final_token {
                anyhow::bail!("injected final-token failure");
            }
            Ok((token_id, absolute_pos))
        }

        fn complete_output(
            &mut self,
            pending: Self::PendingOutput,
            _lm_head: Qwen36LmHeadSelection,
        ) -> Result<Vec<f32>> {
            if self.fail_output {
                anyhow::bail!("injected output failure");
            }
            Ok(vec![pending.0 as f32, pending.1 as f32])
        }
    }

    fn run_injected_prefill(
        session: &mut Qwen36SessionPosition,
        backend: &mut InjectedServingBackend,
        observer: &mut RecordingServingObserver,
        prompt: &[u32],
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Vec<f32>> {
        run_prefill_orchestration(session, prompt, 16, 32, lm_head, backend, observer)
    }

    fn run_injected_decode(
        session: &mut Qwen36SessionPosition,
        backend: &mut InjectedServingBackend,
        observer: &mut RecordingServingObserver,
        token_id: u32,
        absolute_pos: usize,
        lm_head: Qwen36LmHeadSelection,
    ) -> Result<Vec<f32>> {
        run_decode_orchestration(
            session,
            token_id,
            absolute_pos,
            16,
            32,
            lm_head,
            backend,
            observer,
        )
    }

    #[test]
    fn prefill_prevalidates_every_token_before_execution_starts() {
        let mut session = Qwen36SessionPosition::default();
        let mut backend = InjectedServingBackend::default();
        let mut observer = RecordingServingObserver::default();

        let err = run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[1, 2, 32],
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect_err("out-of-vocabulary final token must fail before prefix execution");

        assert!(err.to_string().contains("token 32"), "{err:#}");
        assert!(err.to_string().contains("vocabulary 32"), "{err:#}");
        assert!(backend.prefix_calls.is_empty());
        assert!(backend.production_calls.is_empty());
        assert!(observer.events.is_empty());
        run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[1, 2, 3],
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect("prevalidation failure must not require reset");
    }

    #[test]
    fn failed_prefill_final_token_requires_public_reset_before_retry() {
        let mut session = Qwen36SessionPosition::default();
        let mut backend = InjectedServingBackend {
            fail_final_token: true,
            ..InjectedServingBackend::default()
        };
        let mut observer = RecordingServingObserver::default();

        let err = run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[4, 5],
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect_err("injected final-token failure");
        assert!(err.to_string().contains("final-token"), "{err:#}");
        assert_eq!(backend.prefix_calls.len(), 1);
        assert_eq!(backend.production_calls.len(), 1);

        backend.fail_final_token = false;
        let retry = run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[4, 5],
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect_err("retry after mutation must require reset");
        assert!(retry.to_string().contains("requires reset"), "{retry:#}");
        assert_eq!(backend.prefix_calls.len(), 1);
        assert_eq!(backend.production_calls.len(), 1);
        let decode_retry = run_injected_decode(
            &mut session,
            &mut backend,
            &mut observer,
            6,
            2,
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect_err("decode after failed prefill must require reset");
        assert!(
            decode_retry.to_string().contains("requires reset"),
            "{decode_retry:#}"
        );

        session.reset();
        run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[4, 5],
            Qwen36LmHeadSelection::DenseFolded,
        )
        .expect("public reset permits a new prefill");
    }

    #[test]
    fn failed_prefill_output_requires_public_reset_before_retry() {
        let mut session = Qwen36SessionPosition::default();
        let mut backend = InjectedServingBackend {
            fail_output: true,
            ..InjectedServingBackend::default()
        };
        let mut observer = RecordingServingObserver::default();

        run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[6, 7],
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("injected prefill output failure");
        let calls_after_failure = backend.production_calls.len();
        let retry = run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[6, 7],
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("prefill output failure must require reset");

        assert!(retry.to_string().contains("requires reset"), "{retry:#}");
        assert_eq!(backend.production_calls.len(), calls_after_failure);
        assert!(!observer
            .events
            .contains(&RecordedServingEvent::OutputCompleted));
        assert!(!observer
            .events
            .iter()
            .any(|event| matches!(event, RecordedServingEvent::PositionCommitted(_))));
    }

    #[test]
    fn failed_decode_output_requires_public_reset_before_retry() {
        let mut session = Qwen36SessionPosition::default();
        let mut backend = InjectedServingBackend::default();
        let mut observer = RecordingServingObserver::default();
        run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[8, 9],
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect("seed active session");
        backend.fail_output = true;

        run_injected_decode(
            &mut session,
            &mut backend,
            &mut observer,
            10,
            2,
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("injected decode output failure");
        let calls_after_failure = backend.production_calls.len();
        let retry = run_injected_decode(
            &mut session,
            &mut backend,
            &mut observer,
            10,
            2,
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("decode output retry must require reset");

        assert!(retry.to_string().contains("requires reset"), "{retry:#}");
        assert_eq!(backend.production_calls.len(), calls_after_failure);
        let prefill_retry = run_injected_prefill(
            &mut session,
            &mut backend,
            &mut observer,
            &[8, 9],
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("prefill after failed decode must require reset");
        assert!(
            prefill_retry.to_string().contains("requires reset"),
            "{prefill_retry:#}"
        );
        session.reset();
        let after_reset = run_injected_decode(
            &mut session,
            &mut backend,
            &mut observer,
            10,
            2,
            Qwen36LmHeadSelection::SparseStandalone,
        )
        .expect_err("reset requires a new prefill before decode");
        assert!(
            after_reset.to_string().contains("before prefill"),
            "{after_reset:#}"
        );
    }

    #[test]
    fn orchestration_records_exact_prompt_split_lm_head_output_and_commit_order() {
        for lm_head in [
            Qwen36LmHeadSelection::DenseFolded,
            Qwen36LmHeadSelection::SparseStandalone,
        ] {
            let mut session = Qwen36SessionPosition::default();
            let mut backend = InjectedServingBackend::default();
            let mut observer = RecordingServingObserver::default();

            let logits = run_injected_prefill(
                &mut session,
                &mut backend,
                &mut observer,
                &[7, 11, 13],
                lm_head,
            )
            .expect("orchestrated prefill");

            assert_eq!(logits, vec![13.0, 2.0]);
            assert_eq!(
                observer.events,
                vec![
                    RecordedServingEvent::Prefix {
                        tokens: vec![7, 11],
                        positions: vec![PositionPair::dense(0), PositionPair::dense(1)],
                    },
                    RecordedServingEvent::ProductionToken {
                        token_id: 13,
                        absolute_pos: 2,
                    },
                    RecordedServingEvent::LmHead(lm_head),
                    RecordedServingEvent::OutputCompleted,
                    RecordedServingEvent::PositionCommitted(3),
                ]
            );
            assert_eq!(
                backend.prefix_calls,
                vec![(
                    vec![7, 11],
                    vec![PositionPair::dense(0), PositionPair::dense(1)]
                )]
            );
            assert_eq!(backend.production_calls, vec![(13, 2, lm_head)]);

            observer.events.clear();
            run_injected_decode(&mut session, &mut backend, &mut observer, 17, 3, lm_head)
                .expect("orchestrated decode");
            assert_eq!(
                observer.events,
                vec![
                    RecordedServingEvent::ProductionToken {
                        token_id: 17,
                        absolute_pos: 3,
                    },
                    RecordedServingEvent::LmHead(lm_head),
                    RecordedServingEvent::OutputCompleted,
                    RecordedServingEvent::PositionCommitted(4),
                ]
            );
        }
    }

    #[test]
    fn multi_token_prefill_exposes_prefix_and_final_production_boundaries_once() {
        let mut session = Qwen36SessionPosition::default();
        let mut backend = InjectedServingBackend::default();
        let mut observer = RecordingServingObserver::default();
        let mut boundaries = Vec::new();

        let output = run_prefill_orchestration_with_boundaries(
            &mut session,
            &[7, 11, 13],
            16,
            32,
            Qwen36LmHeadSelection::DenseFolded,
            &mut backend,
            &mut observer,
            &mut |boundary| {
                boundaries.push(boundary);
                Ok(())
            },
        )
        .expect("timed orchestrated prefill");

        assert_eq!(
            boundaries,
            vec![
                Qwen36MoePrefillBoundary::PrefixStarted,
                Qwen36MoePrefillBoundary::FinalProductionStarted,
            ]
        );
        assert_eq!(output.logits, vec![13.0, 2.0]);
        assert_eq!(output.prefix_token_count, 2);
        assert_eq!(backend.prefix_calls.len(), 1);
        assert_eq!(backend.production_calls.len(), 1);
    }

    #[test]
    fn session_prefill_rejects_an_empty_prompt() {
        let session = Qwen36SessionPosition::default();

        let err = session
            .validate_prefill(0, 16)
            .expect_err("empty prompt must fail");

        assert!(err.to_string().contains("empty prompt"), "{err:#}");
    }

    #[test]
    fn session_prefill_rejects_prompt_context_overflow() {
        let session = Qwen36SessionPosition::default();

        let err = session
            .validate_prefill(17, 16)
            .expect_err("prompt beyond context must fail");

        assert!(err.to_string().contains("context"), "{err:#}");
    }

    #[test]
    fn session_prefill_requires_a_reset_after_a_completed_request() {
        let mut session = Qwen36SessionPosition::default();
        session.prefill_succeeded(4);

        let err = session
            .validate_prefill(2, 16)
            .expect_err("second prefill without reset must fail");

        assert!(err.to_string().contains("reset"), "{err:#}");
    }

    #[test]
    fn session_decode_rejects_decode_before_prefill() {
        let session = Qwen36SessionPosition::default();

        let err = session
            .validate_decode(0, 16)
            .expect_err("decode before prefill must fail");

        assert!(err.to_string().contains("before prefill"), "{err:#}");
    }

    #[test]
    fn session_decode_rejects_duplicate_and_skipped_absolute_positions() {
        let mut session = Qwen36SessionPosition::default();
        session.prefill_succeeded(4);

        for pos in [3, 5] {
            let err = session
                .validate_decode(pos, 16)
                .expect_err("non-next decode position must fail");
            assert!(
                err.to_string().contains("expected absolute position 4"),
                "{err:#}"
            );
        }
        session
            .validate_decode(4, 16)
            .expect("exact next position must pass");
    }

    #[test]
    fn session_decode_rejects_context_overflow() {
        let mut session = Qwen36SessionPosition::default();
        session.prefill_succeeded(16);

        let err = session
            .validate_decode(16, 16)
            .expect_err("decode beyond context must fail");

        assert!(err.to_string().contains("context"), "{err:#}");
    }

    #[test]
    fn session_reset_returns_to_prefill_ready_state() {
        let mut session = Qwen36SessionPosition::default();
        session.prefill_succeeded(4);
        session
            .validate_decode(4, 16)
            .expect("exact decode position");
        session.decode_succeeded(4).expect("advance session");

        session.reset();

        assert_eq!(session.next(), None);
        session
            .validate_prefill(2, 16)
            .expect("reset session accepts prefill");
        let err = session
            .validate_decode(0, 16)
            .expect_err("reset session rejects decode before prefill");
        assert!(err.to_string().contains("before prefill"), "{err:#}");
    }

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
        let evidence = build_load_evidence(
            LoadEvidenceInput {
                direct_profile: Qwen36MoeDirectProfile {
                    required_tensors: 20,
                    raw_dense: 8,
                    native_int4: 12,
                    bf16_fallback: 0,
                },
                source_bytes: 4096,
                load_sequence: 7,
                source_open_count: 1,
            },
            &HalProfileSnapshot {
                entries: vec![HalProfileEntry {
                    op: "copy_h2d".to_string(),
                    calls: 1,
                    total_ms: 2.0,
                    max_ms: 2.0,
                    total_bytes: 2048,
                }],
                ..HalProfileSnapshot::default()
            },
        )
        .expect("production load evidence");

        assert_eq!(evidence.direct_profile.native_int4, 12);
        assert_eq!(evidence.direct_profile.bf16_fallback, 0);
        assert_eq!(evidence.source_bytes, 4096);
        assert_eq!(evidence.device_upload_bytes, 2048);
        assert_eq!(evidence.upload_duration, Duration::from_millis(2));
        assert_eq!(evidence.load_sequence, 7);
        assert_eq!(evidence.source_open_count, 1);
    }

    #[test]
    fn async_upload_profile_contributes_to_public_bytes_and_duration_evidence() {
        let profile = HalProfileSnapshot {
            entries: vec![
                HalProfileEntry {
                    op: "copy_h2d".to_string(),
                    calls: 1,
                    total_ms: 1.0,
                    max_ms: 1.0,
                    total_bytes: 100,
                },
                HalProfileEntry {
                    op: "copy_h2d_async".to_string(),
                    calls: 1,
                    total_ms: 2.0,
                    max_ms: 2.0,
                    total_bytes: 200,
                },
                HalProfileEntry {
                    op: "copy_storage_to_device".to_string(),
                    calls: 1,
                    total_ms: 4.0,
                    max_ms: 4.0,
                    total_bytes: 300,
                },
                HalProfileEntry {
                    op: "vmm_copy_h2d".to_string(),
                    calls: 1,
                    total_ms: 8.0,
                    max_ms: 8.0,
                    total_bytes: 400,
                },
                HalProfileEntry {
                    op: "copy_d2h".to_string(),
                    calls: 1,
                    total_ms: 16.0,
                    max_ms: 16.0,
                    total_bytes: 500,
                },
            ],
            ..HalProfileSnapshot::default()
        };

        let evidence = build_load_evidence(
            LoadEvidenceInput {
                direct_profile: Qwen36MoeDirectProfile {
                    required_tensors: 2,
                    raw_dense: 1,
                    native_int4: 1,
                    bf16_fallback: 0,
                },
                source_bytes: 4096,
                load_sequence: 1,
                source_open_count: 1,
            },
            &profile,
        )
        .expect("profiled load evidence");

        assert_eq!(evidence.device_upload_bytes, 600);
        assert_eq!(evidence.upload_duration, Duration::from_millis(7));
    }

    #[test]
    fn source_open_evidence_observes_actual_boundary_attempts() {
        let path = std::env::temp_dir().join(format!(
            "supersonic-qwen36-source-observer-missing-{}.flm",
            std::process::id()
        ));
        let observer = Qwen36MoeSourceOpenObserver::for_path(&path);
        assert_eq!(observer.observed_count(), 0);

        for expected_count in [1, 2] {
            let err = match Qwen36MoeSource::open(
                &path,
                crate::flm_model_source::FlmModelSourceOptions {
                    int4_runtime: true,
                    verify_block_hashes: false,
                },
            ) {
                Ok(_) => panic!("missing source must fail after the boundary is observed"),
                Err(err) => err,
            };
            assert!(err.to_string().contains("opening Qwen3.6 MoE FLM source"));
            assert_eq!(observer.observed_count(), expected_count);
        }
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
            execution_options: Qwen36ExecutionOptions::default(),
            accurate_stage_timings: false,
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
    fn serving_apis_are_part_of_the_engine_lifecycle() {
        let _: fn(&mut Qwen36MoeEngine) -> anyhow::Result<()> = Qwen36MoeEngine::reset;
        let _: fn(&mut Qwen36MoeEngine, &[u32]) -> anyhow::Result<Vec<f32>> =
            Qwen36MoeEngine::prefill;
        let _: fn(&mut Qwen36MoeEngine, u32, usize) -> anyhow::Result<Vec<f32>> =
            Qwen36MoeEngine::decode_step;
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
    fn profiled_gpu_load_preserves_outer_profile_and_returns_load_only_evidence() {
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

        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        let _before_load = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &[0x11; 16])
            .expect("profile outer pre-load upload");
        let (load_result, load_profile) = profile_gpu_load(|| {
            Ok(GpuBuffer::from_host_bytes(
                0,
                ScalarType::U8,
                &[16],
                &[0x22; 16],
            )?)
        });
        let loaded = load_result.expect("profile nested load upload");

        assert_eq!(load_profile.alloc_calls, 1);
        assert_eq!(load_profile.h2d_bytes, 16);
        assert_eq!(load_profile.d2h_bytes, 0);

        assert_eq!(
            loaded
                .to_host_bytes()
                .expect("profile outer post-load download"),
            vec![0x22; 16]
        );
        let whole_run_profile = gpu_hal::hal_profile_snapshot();
        gpu_hal::hal_profile_set_enabled(false);

        assert_eq!(whole_run_profile.alloc_calls, 2);
        assert_eq!(whole_run_profile.h2d_bytes, 32);
        assert_eq!(whole_run_profile.d2h_bytes, 16);
        assert!(whole_run_profile.total_calls > load_profile.total_calls);
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
