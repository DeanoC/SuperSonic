//! Qwen3.6-MoE runtime engine.
//!
//! PR 3 stage: dry-run only. Loads `config.json`, enumerates the safetensors
//! checkpoint, computes the analytic weight + state footprint, and reports
//! it against the registry's VRAM budget. No GPU allocation, no kernel —
//! that lands in PR 4 (CUDA) and PR 6 (HIP).
//!
//! The reason for the enumerate-only dry-run is the BF16 35B-A3B checkpoint
//! is ~65 GiB and won't fit a 24 GiB GPU. Until the INT4/q4km bake exists,
//! the only meaningful runtime check is "did the safetensors index match
//! what we expect from the config" plus a budget comparison.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType, VirtualAllocationRole, VirtualArena};
use model_store::manifest::LayoutTag;
use model_store::BakedStore;
use qwen36_moe::config::{Config, TextConfig};
use qwen36_moe::loader::{ScalarKind, WeightLoader};
use qwen36_moe::state::{StateAccount, StateLayout};
use qwen36_moe::weights::{
    checkpoint_dtype_acceptable, expected_tensor_specs, CheckpointAccount, CheckpointDtype,
};

use crate::qwen36_moe_decode::{
    argmax_bf16_logits, host_final_norm_lm_head, run_chained_decode, run_chained_decode_fast,
    run_chained_decode_fast_with_expert_prefetch, sample_bf16_logits, ExpertPrefetchPhase,
    ExpertRoute, LayerBuffers, MtpLayerBuffers, MultiLayerGeom, XorshiftRng,
};
use crate::qwen36_moe_host::{host_load_bytes, load_lm_head_bf16, lookup_embed_row};
use crate::qwen36_moe_layers::{
    load_all_layer_buffers, load_layer_buffers, load_to_gpu, resolve_qwen36_store_name,
    store_contains_qwen36, store_layout_qwen36, Qwen36WeightMode,
};
use crate::qwen36_moe_prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe_residency::{MoeExpertResidencyConfig, MoeExpertResidencyManager};
use crate::qwen36_moe_speculative::{
    run_speculative_decode_step, run_speculative_decode_step_batched,
};
use crate::qwen36_moe_state::{refresh_linear_attn_state, restore_linear_attn_state};
use crate::qwen36_moe_telemetry::{
    MoeIslandPrefetchMode, MoeRouteTelemetry, MoeSparseTelemetry, MoeSparseTelemetrySnapshot,
    MoeTransitionPredictor,
};
use crate::qwen36_moe_vmm::{
    moe_island_cap_experts_from_env, moe_island_prefetch_ranks_from_env,
    moe_island_prefetch_transition_min_observations, should_try_moe_expert_vmm,
    should_use_qwen36_kv_vmm, virtual_kv_stats_for_layers, MoeExpertVmmMode,
};
use crate::registry::{FamilyParams, Qwen36MoeKernelParams, RegistryEntry};

const GIB: f64 = (1024 * 1024 * 1024) as f64;
const MIB: f64 = (1024 * 1024) as f64;
const QWEN36_NUM_SPECULATIVE_TOKENS: usize = 3;

pub struct DryRunReport {
    pub config: Config,
    pub kernel_params: Qwen36MoeKernelParams,
    pub checkpoint: CheckpointAccount,
    pub int4_projected_bytes: u64,
    pub state: StateAccount,
    pub on_disk_bytes: Option<u64>,
    pub on_disk_tensor_count: Option<usize>,
    pub missing_tensors: Vec<String>,
    pub dtype_mismatches: Vec<String>,
    /// Set when safetensors files are partially present but the loader
    /// couldn't open them (e.g. mid-download). The analytic accounting is
    /// still emitted so the user gets useful output.
    pub loader_warning: Option<String>,
    pub registry_budget_bytes: u64,
    pub gpu_total_vram_bytes: u64,
    /// Populated when an INT4 baked package is present and loadable.
    pub bake: Option<BakeAccount>,
    /// What the dry-run actually used for KV-cache sizing — and where it
    /// came from. The string flavor (`Explicit`, `EstimatedFromPrompt`,
    /// `MaxNewTokensOnly`) lets `print_report` flag the worst case so the
    /// user knows the `fit:YES` answer covered their realistic prompt.
    pub context_size_used: usize,
    pub context_size_source: ContextSizeSource,
}

/// How `context_size_used` was derived. Anything other than `Explicit`
/// means the dry-run is making an assumption the caller should verify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSizeSource {
    /// User passed `--context-size`; honoured verbatim.
    Explicit,
    /// User passed `--prompt` but no `--context-size`; estimated as
    /// (prompt char count) + max_new_tokens. Char count is a conservative
    /// upper bound on token count for English-ish text — true tokenisation
    /// would need the model's tokenizer, which the dry-run path doesn't
    /// load.
    EstimatedFromPrompt,
    /// Neither `--context-size` nor `--prompt` was given. Defaults to
    /// max_new_tokens — almost certainly an undercount for any real
    /// session. The report flags this as a worst-case caveat.
    MaxNewTokensOnly,
}

/// Summary of a baked-package's contents — the runtime-ready view the
/// kernel will see at decode time. Built by mmap'ing the bake at
/// `model-store::bake_dir_int4()` and walking its manifest.
#[derive(Debug, Clone)]
pub struct BakeAccount {
    pub bake_dir: PathBuf,
    pub manifest_format_version: u32,
    pub manifest_converter_version: u32,
    pub tensor_count: usize,
    pub weights_bin_bytes: u64,
    /// Per-LayoutTag tensor counts. Sorted alphabetically by tag name for
    /// deterministic output.
    pub by_layout: BTreeMap<String, usize>,
    /// Tensors expected per `expected_tensor_specs` that are absent from
    /// the bake's index. A non-empty list means the bake is incomplete.
    pub missing_specs: Vec<String>,
    /// INT4 expert tensors per layer the runtime relies on. Each entry is
    /// `(layer_idx, name)` for any of `gate_up_proj`, `down_proj` that's
    /// missing or has a wrong layout.
    pub bad_expert_tensors: Vec<(usize, String)>,
    /// Cached aggregate byte sizes per category. Useful for comparing the
    /// bake against the analytic INT4 projection.
    pub int4_quantized_bytes: u64,
    pub raw_bytes: u64,
    /// Optional live VMM probe populated when requested by environment.
    /// This loads real baked tensors into role-tagged virtual allocations
    /// without switching decode descriptors over to VMM yet.
    pub vmm_probe: Option<VirtualBakeProbe>,
}

#[derive(Debug, Clone)]
pub struct VirtualBakeProbe {
    pub allocations: usize,
    pub logical_bytes: usize,
    pub reserved_bytes: usize,
    pub resident_bytes: usize,
    pub logical_resident_bytes: usize,
    pub mapping_count: usize,
    pub loaded_tensors: Vec<String>,
    pub failed_tensors: Vec<String>,
}

pub fn run_qwen36_moe_dry_run(
    model_dir: &Path,
    entry: &RegistryEntry,
    total_vram: u64,
    context_size: usize,
    context_size_source: ContextSizeSource,
    batch_size: usize,
    kv_fp8: bool,
    no_bake: bool,
    ordinal: usize,
) -> Result<DryRunReport> {
    let kernel_params = match entry.params {
        FamilyParams::Qwen36Moe(p) => p,
        _ => return Err(anyhow!("registry entry is not Qwen36Moe family")),
    };

    let config = qwen36_moe::config::load_config(model_dir)
        .map_err(|e| anyhow!("parse config.json: {e}"))?;

    sanity_check_kernel_params(&config.text_config, &kernel_params)?;

    let weight_prefix = kernel_params.weight_prefix;
    let specs = expected_tensor_specs(&config.text_config, weight_prefix);
    let checkpoint = CheckpointAccount::from_config(&config.text_config);
    let int4_projected_bytes = checkpoint.project_int4_total_bytes(&config.text_config, 128);

    // The sidecar window is consulted only when --kv-fp8 is on AND the
    // env-var helper is enabled. The persistent kernel treats it as a
    // rolling recent-token window.
    let sidecar_window = if kv_fp8 && qwen35::state::kv_fp8_bf16_sidecar_enabled() {
        let window = qwen35::state::kv_fp8_bf16_sidecar_window_tokens()
            .unwrap_or(context_size)
            .min(context_size);
        (window > 0).then_some(window)
    } else {
        None
    };
    let layout = StateLayout::new(context_size, batch_size, kv_fp8, sidecar_window);
    let state = StateAccount::from_config(&config.text_config, layout);

    let bake = inspect_bake(model_dir, &config.text_config, weight_prefix, ordinal);

    let mut on_disk_bytes = None;
    let mut on_disk_tensor_count = None;
    let mut missing_tensors = Vec::new();
    let mut dtype_mismatches = Vec::new();
    let mut loader_warning = None;
    if !no_bake_only_safetensors(model_dir, no_bake) {
        // No safetensors expected — bake-only path. PR 3 doesn't yet open
        // the baked store; revisit once PR 6 lands the INT4 reader.
    } else if !has_safetensors(model_dir) {
        // Safetensors missing AND no bake support yet — caller probably
        // needs to download. Surface a helpful note rather than failing,
        // because the analytic numbers are still useful.
        loader_warning = Some(format!(
            "no safetensors in {} — analytic accounting only",
            model_dir.display()
        ));
    } else {
        match WeightLoader::from_dir(model_dir) {
            Ok(loader) => {
                on_disk_tensor_count = Some(loader.tensor_count());
                let mut total: u64 = 0;
                let mut io_failures: usize = 0;
                for spec in &specs {
                    match loader.meta(&spec.name) {
                        Ok(meta) => {
                            total += meta.byte_size();
                            let got_dtype = scalar_to_checkpoint_dtype(meta.dtype);
                            if let Some(got) = got_dtype {
                                if !checkpoint_dtype_acceptable(spec.role, got) {
                                    dtype_mismatches.push(format!(
                                        "{} got {:?} (not acceptable for {:?})",
                                        spec.name, got, spec.role
                                    ));
                                }
                            } else {
                                dtype_mismatches.push(format!(
                                    "{} got non-float dtype {:?}",
                                    spec.name, meta.dtype
                                ));
                            }
                        }
                        Err(qwen36_moe::loader::LoadError::NotFound(_)) => {
                            missing_tensors.push(spec.name.clone());
                        }
                        // Tensor is in the index but reading metadata failed
                        // (e.g. shard file not yet present mid-download).
                        // Don't kill the whole dry-run.
                        Err(_) => io_failures += 1,
                    }
                }
                on_disk_bytes = Some(total);
                if io_failures > 0 {
                    loader_warning = Some(format!(
                        "{io_failures} tensors could not be read (likely partial download); \
                         on-disk total is incomplete"
                    ));
                }
            }
            Err(e) => {
                loader_warning = Some(format!(
                    "could not open safetensors from {}: {e}",
                    model_dir.display()
                ));
            }
        }
    }

    let kv_bytes_per_token = config
        .text_config
        .kv_bytes_per_token(if kv_fp8 { 1 } else { 2 });
    let registry_budget_bytes = entry.vram.estimate_total(context_size, kv_bytes_per_token);

    Ok(DryRunReport {
        config,
        kernel_params,
        checkpoint,
        int4_projected_bytes,
        state,
        on_disk_bytes,
        on_disk_tensor_count,
        missing_tensors,
        dtype_mismatches,
        loader_warning,
        registry_budget_bytes,
        gpu_total_vram_bytes: total_vram,
        bake,
        context_size_used: context_size,
        context_size_source,
    })
}

/// Open the INT4 baked package (if present and valid) and summarise its
/// contents. Returns None when no bake is on disk or its manifest can't be
/// parsed; the dry-run continues either way.
fn inspect_bake(
    model_dir: &Path,
    text_config: &TextConfig,
    weight_prefix: &str,
    ordinal: usize,
) -> Option<BakeAccount> {
    let bake_dir = model_store::bake_dir_int4(model_dir);
    if !bake_dir.exists() {
        return None;
    }
    // Re-parse the manifest header even though BakedStore does too — we
    // expose format/converter versions in the report so a stale bake is
    // visible without having to dig into the file by hand.
    let manifest_text = std::fs::read_to_string(model_store::manifest_path(&bake_dir)).ok()?;
    let manifest: model_store::manifest::Manifest = serde_json::from_str(&manifest_text).ok()?;
    let store = BakedStore::open(&bake_dir).ok()?;
    let weights_bin_bytes = std::fs::metadata(model_store::weights_bin_path(&bake_dir))
        .ok()?
        .len();

    // Aggregate per-layout tensor counts and per-category byte totals.
    let mut by_layout: BTreeMap<String, usize> = BTreeMap::new();
    let mut int4_quantized_bytes: u64 = 0;
    let mut raw_bytes: u64 = 0;
    for t in &manifest.tensors {
        let key = format!("{:?}", t.layout);
        *by_layout.entry(key).or_default() += 1;
        match t.layout {
            LayoutTag::Int4Quantized => int4_quantized_bytes += t.byte_len,
            LayoutTag::Raw => raw_bytes += t.byte_len,
            _ => {}
        }
    }

    // Cross-check the bake against the runtime's expected tensor specs.
    // A missing entry here would crash the loader at decode time; surface
    // it during dry-run instead.
    let specs = expected_tensor_specs(text_config, weight_prefix);
    let mut missing_specs = Vec::new();
    for spec in &specs {
        if !store_contains_qwen36(&store, &spec.name) {
            missing_specs.push(spec.name.clone());
        }
    }

    // The fused MoE expert tensors are the bulk of the bake (~80% of the
    // INT4 byte total on 35B-A3B). Spot-check every layer; absence here is
    // fatal because the kernel can't run without a routed expert.
    let mut bad_expert_tensors = Vec::new();
    for li in 0..text_config.num_hidden_layers as usize {
        for kind in ["gate_up_proj", "down_proj"] {
            let name = format!("{weight_prefix}.layers.{li}.mlp.experts.{kind}");
            match store_layout_qwen36(&store, &name) {
                Some(LayoutTag::Int4Quantized) => {}
                Some(other) => bad_expert_tensors.push((li, format!("{name} (layout={other:?})"))),
                None => bad_expert_tensors.push((li, format!("{name} (missing)"))),
            }
        }
    }

    let vmm_probe = inspect_bake_vmm_probe(&store, text_config, weight_prefix, ordinal);

    Some(BakeAccount {
        bake_dir,
        manifest_format_version: manifest.format_version,
        manifest_converter_version: manifest.converter_version,
        tensor_count: manifest.tensors.len(),
        weights_bin_bytes,
        by_layout,
        missing_specs,
        bad_expert_tensors,
        int4_quantized_bytes,
        raw_bytes,
        vmm_probe,
    })
}

fn inspect_bake_vmm_probe(
    store: &BakedStore,
    text_config: &TextConfig,
    weight_prefix: &str,
    ordinal: usize,
) -> Option<VirtualBakeProbe> {
    let weight_probe = std::env::var_os("SUPERSONIC_VMM_WEIGHT_PROBE").is_some();
    let moe_probe_layers = std::env::var("SUPERSONIC_VMM_MOE_ISLAND_PROBE_LAYERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| std::env::var_os("SUPERSONIC_VMM_MOE_ISLAND_PROBE").map(|_| 1))
        .unwrap_or(0);
    if !weight_probe && moe_probe_layers == 0 {
        return None;
    }
    if !gpu_hal::vmm_is_supported(gpu_hal::current_backend(), ordinal) {
        return None;
    }

    let mut arena = BakedStore::virtual_weight_arena(ordinal);
    let mut loaded_tensors = Vec::new();
    let mut failed_tensors = Vec::new();
    if weight_probe {
        let name = resolve_qwen36_store_name(store, "lm_head.weight");
        match store.load_to_virtual_arena(&mut arena, name.as_ref(), VirtualAllocationRole::Weights)
        {
            Ok(_) => loaded_tensors.push(name.into_owned()),
            Err(err) => failed_tensors.push(format!("{name}: {err}")),
        }
    }

    for li in 0..moe_probe_layers.min(text_config.num_hidden_layers as usize) {
        for kind in ["gate_up_proj", "down_proj"] {
            let requested = format!("{weight_prefix}.layers.{li}.mlp.experts.{kind}");
            let name = resolve_qwen36_store_name(store, &requested);
            match store.load_to_virtual_arena(
                &mut arena,
                name.as_ref(),
                VirtualAllocationRole::MoeExpert,
            ) {
                Ok(_) => loaded_tensors.push(name.into_owned()),
                Err(err) => failed_tensors.push(format!("{name}: {err}")),
            }
        }
    }

    if loaded_tensors.is_empty() && failed_tensors.is_empty() {
        return None;
    }
    let stats = arena.stats();
    Some(VirtualBakeProbe {
        allocations: stats.allocations,
        logical_bytes: stats.logical_bytes,
        reserved_bytes: stats.reserved_bytes,
        resident_bytes: stats.resident_bytes,
        logical_resident_bytes: stats.logical_resident_bytes,
        mapping_count: stats.mapping_count,
        loaded_tensors,
        failed_tensors,
    })
}

fn no_bake_only_safetensors(_model_dir: &Path, no_bake: bool) -> bool {
    // PR 3 only supports safetensors. Future PRs will route to the baked
    // store when `no_bake` is false and a bake is on disk.
    let _ = no_bake;
    true
}

fn has_safetensors(model_dir: &Path) -> bool {
    model_dir.join("model.safetensors.index.json").exists()
        || model_dir.join("model.safetensors").exists()
}

fn scalar_to_checkpoint_dtype(s: ScalarKind) -> Option<CheckpointDtype> {
    match s {
        ScalarKind::Bf16 => Some(CheckpointDtype::Bf16),
        ScalarKind::F32 => Some(CheckpointDtype::F32),
        _ => None,
    }
}

fn sanity_check_kernel_params(config: &TextConfig, params: &Qwen36MoeKernelParams) -> Result<()> {
    if config.num_experts as u32 > params.num_experts {
        return Err(anyhow!(
            "config has num_experts={} but registry kernel scratch is sized for at most {}; \
             update Qwen36MoeKernelParams or use a smaller checkpoint",
            config.num_experts,
            params.num_experts,
        ));
    }
    if config.num_experts_per_tok as u32 > params.top_k {
        return Err(anyhow!(
            "config has num_experts_per_tok={} but registry kernel scratch is sized for top-{}; \
             update Qwen36MoeKernelParams or use a smaller checkpoint",
            config.num_experts_per_tok,
            params.top_k,
        ));
    }
    if config.moe_intermediate_size as u32 > params.moe_intermediate_size {
        return Err(anyhow!(
            "config moe_intermediate_size={} exceeds registry bound {}",
            config.moe_intermediate_size,
            params.moe_intermediate_size,
        ));
    }
    if config.shared_expert_intermediate_size as u32 > params.shared_expert_intermediate_size {
        return Err(anyhow!(
            "config shared_expert_intermediate_size={} exceeds registry bound {}",
            config.shared_expert_intermediate_size,
            params.shared_expert_intermediate_size,
        ));
    }
    Ok(())
}

pub fn print_report(report: &DryRunReport) {
    let cfg = &report.config.text_config;
    println!("[qwen3.6-moe] dry-run summary");
    println!(
        "  arch:           hidden={} layers={} (full={} linear={}) Q/KV heads={}/{} head_dim={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_full_attention_layers(),
        cfg.num_linear_attention_layers(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    println!(
        "  moe:            num_experts={} top_k={} moe_int={} shared_int={} norm_topk={} attn_output_gate={}",
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.shared_expert_intermediate_size,
        cfg.norm_topk_prob,
        cfg.attn_output_gate,
    );
    println!(
        "  vocab:          {}    rope_theta={}    max_pos={}    tie_embed={}",
        cfg.vocab_size,
        cfg.rope_theta(),
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings,
    );
    let kp = &report.kernel_params;
    println!(
        "  kernel params:  prefix={} kv_chunk={} proj_buf_f={} attn_scratch_f={} moe_scratch_f={}",
        kp.weight_prefix,
        kp.kv_chunk_size,
        kp.proj_buf_floats,
        kp.attn_scratch_floats,
        kp.moe_scratch_floats,
    );
    println!();
    println!("[checkpoint accounting (analytic, BF16 + F32 mix)]");
    println!(
        "  embed:          {:>8.2} GiB",
        report.checkpoint.embed_bytes as f64 / GIB
    );
    println!(
        "  lm_head:        {:>8.2} GiB",
        report.checkpoint.lm_head_bytes as f64 / GIB
    );
    println!(
        "  full attn:      {:>8.2} GiB ({} layers × {:.1} MiB)",
        (report.checkpoint.num_full_layers * report.checkpoint.full_attn_bytes_per_layer) as f64
            / GIB,
        report.checkpoint.num_full_layers,
        report.checkpoint.full_attn_bytes_per_layer as f64 / MIB,
    );
    println!(
        "  linear attn:    {:>8.2} GiB ({} layers × {:.1} MiB)",
        (report.checkpoint.num_linear_layers * report.checkpoint.linear_attn_bytes_per_layer)
            as f64
            / GIB,
        report.checkpoint.num_linear_layers,
        report.checkpoint.linear_attn_bytes_per_layer as f64 / MIB,
    );
    let n_hidden = cfg.num_hidden_layers as u64;
    println!(
        "  routers:        {:>8.2} GiB ({} layers × {:.1} MiB)",
        (n_hidden * report.checkpoint.router_bytes_per_layer) as f64 / GIB,
        n_hidden,
        report.checkpoint.router_bytes_per_layer as f64 / MIB,
    );
    println!(
        "  shared experts: {:>8.2} GiB ({} layers × {:.1} MiB)",
        (n_hidden * report.checkpoint.shared_expert_bytes_per_layer) as f64 / GIB,
        n_hidden,
        report.checkpoint.shared_expert_bytes_per_layer as f64 / MIB,
    );
    println!(
        "  routed experts: {:>8.2} GiB ({} layers × {:.1} GiB / {} experts)",
        (n_hidden * report.checkpoint.experts_bytes_per_layer) as f64 / GIB,
        n_hidden,
        report.checkpoint.experts_bytes_per_layer as f64 / GIB,
        cfg.num_experts,
    );
    println!(
        "  TOTAL (BF16):   {:>8.2} GiB",
        report.checkpoint.total_bytes as f64 / GIB
    );
    println!(
        "  TOTAL (INT4):   {:>8.2} GiB  (projected, gs=128)",
        report.int4_projected_bytes as f64 / GIB
    );
    println!();
    println!("[state accounting]");
    let max_pos = report.config.text_config.max_position_embeddings;
    let ctx_label = match report.context_size_source {
        ContextSizeSource::Explicit => "explicit --context-size".to_string(),
        ContextSizeSource::EstimatedFromPrompt => {
            format!("estimated from --prompt (chars + max_new_tokens)")
        }
        ContextSizeSource::MaxNewTokensOnly => "max_new_tokens only".to_string(),
    };
    println!(
        "  context tokens:     {:>8}     (source: {ctx_label}; model max_pos={max_pos})",
        report.context_size_used,
    );
    println!(
        "  KV cache (full attn): {:>8.2} GiB",
        report.state.full_kv_bytes as f64 / GIB,
    );
    println!(
        "  linear conv state:  {:>8.2} MiB",
        report.state.linear_conv_state_bytes as f64 / MIB
    );
    println!(
        "  linear recurrent:   {:>8.2} MiB",
        report.state.linear_recurrent_state_bytes as f64 / MIB
    );
    println!(
        "  moe scratch:        {:>8.2} KiB",
        report.state.moe_scratch_bytes as f64 / 1024.0
    );
    println!(
        "  activations:        {:>8.2} KiB",
        report.state.activation_bytes as f64 / 1024.0
    );
    println!(
        "  TOTAL state:        {:>8.2} GiB",
        report.state.total_bytes as f64 / GIB
    );
    println!();
    if let (Some(on_disk), Some(count)) = (report.on_disk_bytes, report.on_disk_tensor_count) {
        println!("[on-disk safetensors]");
        println!("  tensor count:    {count}");
        println!(
            "  total bytes:     {:.2} GiB  (analytic predicted {:.2} GiB; drift {:+.2} MiB)",
            on_disk as f64 / GIB,
            report.checkpoint.total_bytes as f64 / GIB,
            (on_disk as f64 - report.checkpoint.total_bytes as f64) / MIB,
        );
        if !report.missing_tensors.is_empty() {
            println!(
                "  MISSING TENSORS: {} (showing first 10)",
                report.missing_tensors.len()
            );
            for n in report.missing_tensors.iter().take(10) {
                println!("    - {n}");
            }
        }
        if !report.dtype_mismatches.is_empty() {
            println!(
                "  DTYPE MISMATCHES: {} (showing first 10)",
                report.dtype_mismatches.len()
            );
            for n in report.dtype_mismatches.iter().take(10) {
                println!("    - {n}");
            }
        }
    } else {
        println!("[on-disk safetensors] not present at model-dir; analytic numbers only");
    }
    if let Some(w) = &report.loader_warning {
        println!("  WARNING: {w}");
    }
    println!();
    if let Some(bake) = &report.bake {
        println!("[INT4 baked package]");
        println!("  bake_dir:        {}", bake.bake_dir.display());
        println!(
            "  manifest:        format_version={} converter_version={}",
            bake.manifest_format_version, bake.manifest_converter_version,
        );
        println!(
            "  weights.bin:     {:.2} GiB    ({} tensors indexed)",
            bake.weights_bin_bytes as f64 / GIB,
            bake.tensor_count,
        );
        println!(
            "  INT4 / Raw:      {:.2} GiB / {:.2} GiB",
            bake.int4_quantized_bytes as f64 / GIB,
            bake.raw_bytes as f64 / GIB,
        );
        let parts: Vec<String> = bake
            .by_layout
            .iter()
            .map(|(l, n)| format!("{l}={n}"))
            .collect();
        println!("  layouts:         {}", parts.join(", "));
        // Compare against the analytic INT4 projection — small drift is
        // expected (the analytic model rounds shapes; the bake is exact).
        let analytic_gib = report.int4_projected_bytes as f64 / GIB;
        let bake_gib = bake.weights_bin_bytes as f64 / GIB;
        println!(
            "  vs analytic:     bake={:.2} GiB analytic_int4={:.2} GiB drift={:+.2} GiB",
            bake_gib,
            analytic_gib,
            bake_gib - analytic_gib,
        );
        if !bake.missing_specs.is_empty() {
            println!(
                "  MISSING SPECS:   {} (showing first 10)",
                bake.missing_specs.len()
            );
            for n in bake.missing_specs.iter().take(10) {
                println!("    - {n}");
            }
        }
        if !bake.bad_expert_tensors.is_empty() {
            println!(
                "  BAD EXPERT TENSORS: {} (showing first 10)",
                bake.bad_expert_tensors.len()
            );
            for (li, n) in bake.bad_expert_tensors.iter().take(10) {
                println!("    - layer {li}: {n}");
            }
        }
        let bake_ok = bake.missing_specs.is_empty() && bake.bad_expert_tensors.is_empty();
        println!("  ready-for-decode: {}", if bake_ok { "YES" } else { "NO" });
        if let Some(vmm) = &bake.vmm_probe {
            println!();
            println!("  [VMM residency probe]");
            println!(
                "    allocations:      {}    mappings={}",
                vmm.allocations, vmm.mapping_count
            );
            println!(
                "    logical/resident: {:.2} MiB / {:.2} MiB",
                vmm.logical_bytes as f64 / MIB,
                vmm.resident_bytes as f64 / MIB,
            );
            println!(
                "    logical resident: {:.2} MiB    reserved={:.2} MiB",
                vmm.logical_resident_bytes as f64 / MIB,
                vmm.reserved_bytes as f64 / MIB,
            );
            for name in vmm.loaded_tensors.iter().take(8) {
                println!("    - {name}");
            }
            if vmm.loaded_tensors.len() > 8 {
                println!("    ... {} more", vmm.loaded_tensors.len() - 8);
            }
            if !vmm.failed_tensors.is_empty() {
                println!(
                    "    failed loads:    {} (showing first 8)",
                    vmm.failed_tensors.len()
                );
                for name in vmm.failed_tensors.iter().take(8) {
                    println!("    ! {name}");
                }
            }
        }
        println!();
    } else {
        println!(
            "[INT4 baked package] not present at .supersonic/v{}-int4-gptq/ \
             (run `oracle/bake_int4.py` to produce one)",
            model_store::manifest::FORMAT_VERSION
        );
        println!();
    }
    println!("[VRAM budget]");
    println!(
        "  registry estimate (INT4 weights + KV @ ctx, ×overhead): {:.2} GiB",
        report.registry_budget_bytes as f64 / GIB
    );
    println!(
        "  detected GPU VRAM:                                       {:.2} GiB",
        report.gpu_total_vram_bytes as f64 / GIB
    );
    let fits = report.registry_budget_bytes <= report.gpu_total_vram_bytes;
    println!(
        "  fit:                                                     {}",
        if fits { "YES" } else { "NO" }
    );
    if !fits {
        println!(
            "  (BF16 weights alone need ~{:.1} GiB; an INT4/q4km bake is required.)",
            report.checkpoint.total_bytes as f64 / GIB
        );
    }
    if report.context_size_source == ContextSizeSource::MaxNewTokensOnly {
        println!();
        println!(
            "  WARNING: context_size defaulted to max_new_tokens={}. KV cache for a real",
            report.context_size_used,
        );
        println!("  decode session will be larger; pass --context-size <N> (or --prompt) for an",);
        println!(
            "  honest fit number. Worst case at the model's max ({} tokens) needs",
            max_pos,
        );
        // Quick worst-case KV estimate at model max — gives the user one
        // concrete data point before they have to re-run with a flag.
        let worst_kv = report.config.text_config.kv_bytes_per_token(2) * max_pos as u64;
        println!(
            "  ~{:.2} GiB of KV cache alone (BF16).",
            worst_kv as f64 / GIB,
        );
    }
}

pub fn run(cli: &crate::Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    ensure_qwen36_bake(cli, entry)?;

    // Derive context_size + an honest source flag so the printed report can
    // tell the user which of three answers they got: explicit, prompt-derived
    // estimate, or worst-case defaults-only. The `--context-size` path is
    // verbatim; otherwise we fall back to (prompt char count) + max_new_tokens
    // when a prompt is given (chars are an upper bound on tokens for
    // English-ish text), or just max_new_tokens when the user gave neither
    // flag — that last case undercounts KV bytes for any realistic session,
    // and the report flags it.
    let max_new = cli.max_new_tokens.max(1);
    let (context_size, context_size_source) = if let Some(ctx) = cli.context_size {
        (ctx, ContextSizeSource::Explicit)
    } else if !cli.prompt.is_empty() {
        (
            cli.prompt.chars().count() + max_new,
            ContextSizeSource::EstimatedFromPrompt,
        )
    } else {
        (max_new, ContextSizeSource::MaxNewTokensOnly)
    };
    if cli.kv_fp8 && cli.no_persistent_decode {
        anyhow::bail!(
            "--kv-fp8 for Qwen3.6-35B-A3B requires the persistent megakernel; \
             remove --no-persistent-decode (persistent is on by default). The \
             back-compat step kernels stay BF16-KV."
        );
    }

    let report = run_qwen36_moe_dry_run(
        &cli.model_dir,
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
        cli.device,
    )?;
    print_report(&report);
    if cli.dry_run {
        return Ok(());
    }

    // The decode kernels (`kernels/qwen36_moe.hip`, the per-block step
    // launchers in `kernel-ffi`) are HIP-only. The registry currently has
    // both HIP and CUDA entries for `qwen3.6-35b-a3b` but the CUDA branches
    // of `attn_step_launch` / `linear_step_launch` / `ffn_step_launch` all
    // return `InvalidArg("CUDA backend not yet wired")`. Fail here with a
    // clear message instead of letting the engine commit to HIP buffers
    // (which would crash later inside the kernel-ffi wrappers when the
    // registry-selected backend disagrees).
    if entry.backend != Backend::Hip {
        anyhow::bail!(
            "qwen3.6-35b-a3b decode kernels are HIP-only at this stage; \
             registry-selected backend was {:?}. Re-run with --backend hip, \
             or use --dry-run for analytic accounting.",
            entry.backend,
        );
    }

    // Real decode path (PR 4c step 2). Uses the host-orchestrated chained
    // launches in `crate::qwen36_moe_decode::run_chained_decode` against
    // per-layer weight buffers loaded from the baked package. INT4 GPTQ
    // is the realistic path on 24 GiB VRAM; the BF16 fallback is wired
    // for completeness but won't fit the 65 GiB 35B model. The multi-layer
    // parity test in `crates/runner/tests/qwen36_moe_multilayer_parity.rs`
    // gates the decode core against the Python multi-layer oracle for both
    // BF16 (cos_sim 0.9999) and INT4 (cos_sim 0.9999) modes.
    //
    // Caveats for PR 4c step 2:
    //  - One token, fresh state. Conv + recurrent state start zeroed; the
    //    full-attn KV cache isn't allocated (single-block kernels run with
    //    `kv_len=1`). Multi-token generation needs prefill + state
    //    persistence which land later.
    //  - lm_head INT4 dequant runs host-side (~1 GiB BF16 buffer); the
    //    lm_head GEMV likewise. Lifting both to the GPU is PR 4d.
    //  - Tokenizer not wired — the produced token is printed as a raw vocab
    //    id so the "doesn't bail" criterion is verifiable end-to-end.
    println!();
    println!("=== Decode (PR 4c step 2: host-orchestrated chained launches) ===");
    let sampling = SamplingParams {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        seed: cli.sampling_seed,
    };
    decode_text(
        &cli.model_dir,
        &report,
        &cli.prompt,
        cli.max_new_tokens.max(1),
        sampling,
        cli.emit_stage_timings,
        cli.speculative_decode,
        cli.fp8_runtime,
        cli.batched_spec_verify,
        entry.backend,
        cli.device,
        // Phase 3e.4: persistent decode is now the default. The legacy
        // `--persistent-decode` flag is a hidden no-op (kept for harness
        // back-compat); `--no-persistent-decode` is the documented
        // opt-out for A/B comparison or bisecting megakernel regressions.
        !cli.no_persistent_decode,
        cli.kv_fp8,
        cli.dump_last_logits,
    )?;
    Ok(())
}

fn ensure_qwen36_bake(cli: &crate::Cli, entry: &RegistryEntry) -> Result<()> {
    // Auto-download the requested release bake if missing or stale. 35B-A3B
    // INT4 calibration OOMs on 24 GiB hosts, so release-hosted bakes are the
    // realistic default for decode and for dry-run residency probes. Run this
    // before dry-run reporting so `SUPERSONIC_VMM_*_PROBE` can inspect a
    // freshly populated bake.
    let variant = if cli.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Int4Gptq
    };
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    // `should_fetch_exact_bake` honors --download-bake (force) and refuses to
    // fetch when an up-to-date bake is already present.
    let force_download = cli.download_bake;
    if !cli.no_download
        && crate::should_fetch_exact_bake(force_download, model_store::version_ok(&bake_dir))
    {
        let canonical_model = entry.model.to_string();
        match crate::try_download_bake(cli, variant, &canonical_model, &bake_dir) {
            Ok(true) => eprintln!(
                "[fetch] installed qwen3.6-MoE {} bake at {}",
                if cli.fp8_runtime { "FP8" } else { "INT4" },
                bake_dir.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!(
                "[fetch] qwen3.6-MoE {} bake fetch failed: {e}",
                if cli.fp8_runtime { "FP8" } else { "INT4" }
            ),
        }
    }
    Ok(())
}

/// Bundles the sampling knobs for the multi-token decode loop. `temperature
/// <= 0` ⇔ greedy argmax (the deterministic default — bit-identical with
/// any seed). At temperature > 0, `top_k`/`top_p` filter the distribution
/// before sampling, then `seed` drives the xorshift RNG.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

/// Build the geometry the chained decoder needs from the parsed config +
/// the registry's per-family params. Mirrors what
/// `oracle/qwen36_moe_multilayer_oracle.py` puts in `config` and what
/// `MultiLayerGeom` consumes.
fn build_multi_layer_geom(
    text_config: &TextConfig,
    kernel_params: &Qwen36MoeKernelParams,
) -> MultiLayerGeom {
    MultiLayerGeom {
        hidden: text_config.hidden_size as i32,
        vocab: text_config.vocab_size as i32,
        num_layers: text_config.num_hidden_layers as i32,
        rms_norm_eps: text_config.rms_norm_eps as f32,

        num_attention_heads: text_config.num_attention_heads as i32,
        num_kv_heads: text_config.num_key_value_heads as i32,
        head_dim: text_config.head_dim as i32,
        rotary_dim: text_config.rotary_dim() as i32,
        rope_theta: text_config.rope_theta() as f32,

        num_k_heads: text_config.linear_num_key_heads as i32,
        num_v_heads: text_config.linear_num_value_heads as i32,
        head_k_dim: text_config.linear_key_head_dim as i32,
        head_v_dim: text_config.linear_value_head_dim as i32,
        conv_kernel_dim: text_config.linear_conv_kernel_dim as i32,

        num_experts: kernel_params.num_experts as i32,
        moe_intermediate: kernel_params.moe_intermediate_size as i32,
        shared_intermediate: kernel_params.shared_expert_intermediate_size as i32,
        top_k: kernel_params.top_k as i32,
    }
}

struct Qwen36DecodeLayers {
    layers: Vec<LayerBuffers>,
    moe_expert_arena: Option<VirtualArena>,
    moe_expert_residency: Option<MoeExpertResidencyManager>,
}

struct Qwen36DecodeSession {
    final_norm_w_buf: GpuBuffer,
    lm_head_w_buf: GpuBuffer,
    logits_buf: GpuBuffer,
    counter_buf: GpuBuffer,
    final_hidden_buf: GpuBuffer,
    mtp_buffers: Option<MtpLayerBuffers>,
    mtp_forward_scratch: Option<crate::qwen36_moe_mtp::MtpForwardScratch>,
    mtp_chain_scratch: Option<crate::qwen36_moe_mtp::MtpChainScratch>,
    embed_w_buf: Option<GpuBuffer>,
    linear_attn_snapshot: Option<crate::qwen36_moe_state::LinearAttnSnapshot>,
    persistent_scratch: Option<crate::qwen36_moe_persistent_decode::PersistentScratch>,
}

#[derive(Default)]
struct Qwen36StageTimingTotals {
    gen_steps: usize,
    embed: std::time::Duration,
    chain: std::time::Duration,
    lm_head: std::time::Duration,
    sample: std::time::Duration,
    detok: std::time::Duration,
    chain_full_attn_us: u64,
    chain_linear_attn_us: u64,
    chain_ffn_us: u64,
}

impl Qwen36StageTimingTotals {
    fn record_generation_step(
        &mut self,
        embed: std::time::Duration,
        chain: std::time::Duration,
        lm_head: std::time::Duration,
        sample: std::time::Duration,
        detok: std::time::Duration,
        outputs: &crate::qwen36_moe_decode::DecodeOutputs,
    ) {
        self.count_generation_step();
        self.embed += embed;
        self.chain += chain;
        self.lm_head += lm_head;
        self.sample += sample;
        self.detok += detok;
        self.record_chain_breakdown(outputs);
    }

    fn record_embed(&mut self, elapsed: std::time::Duration) {
        self.embed += elapsed;
    }

    fn record_chain(
        &mut self,
        elapsed: std::time::Duration,
        outputs: &crate::qwen36_moe_decode::DecodeOutputs,
    ) {
        self.chain += elapsed;
        self.record_chain_breakdown(outputs);
    }

    fn record_lm_head(&mut self, elapsed: std::time::Duration) {
        self.lm_head += elapsed;
    }

    fn count_generation_step(&mut self) {
        self.gen_steps += 1;
    }

    fn record_chain_breakdown(&mut self, outputs: &crate::qwen36_moe_decode::DecodeOutputs) {
        self.chain_full_attn_us += outputs.kernel_full_attn_us;
        self.chain_linear_attn_us += outputs.kernel_linear_attn_us;
        self.chain_ffn_us += outputs.kernel_ffn_us;
    }
}

#[allow(clippy::too_many_arguments)]
fn load_decode_layers_with_vmm_strategy(
    store: &BakedStore,
    ordinal: usize,
    backend: Backend,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    kv_max_t: usize,
    kv_fp8: bool,
    kv_vmm: bool,
    moe_vmm_mode: MoeExpertVmmMode,
    moe_island_cap_experts: Option<usize>,
    moe_prefetch_mode: MoeIslandPrefetchMode,
    moe_prefetch_ranks: usize,
    moe_transition_min_observations: u32,
    persistent_decode: bool,
) -> Result<Qwen36DecodeLayers> {
    if let Some(cap_experts) = moe_island_cap_experts {
        if cap_experts < geom.top_k as usize {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS={cap_experts} is smaller than model top_k={}; \
                 sparse decode needs at least one layer's routed experts resident",
                geom.top_k
            );
        }
        if !should_try_moe_expert_vmm(
            MoeExpertVmmMode::Force,
            backend,
            weight_mode.is_int4(),
            &format!("{weight_mode:?}"),
            ordinal,
        )? {
            unreachable!("forced VMM expert check should either return true or error");
        }
        let config = MoeExpertResidencyConfig::new(1)?;
        let mut manager = MoeExpertResidencyManager::new(ordinal, config);
        let layers = load_all_layer_buffers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            None,
            Some(&mut manager),
        )
        .context("reserve Qwen3.6-MoE routed experts for sparse VMM residency")?;
        let max_resident_pages = manager
            .page_budget_for_routed_experts(cap_experts)
            .context("derive sparse MoE page budget from routed expert tensor layout")?;
        manager
            .set_max_resident_pages(max_resident_pages)
            .context("apply sparse MoE page budget")?;
        let arena_stats = manager.arena().stats();
        let residency_stats = manager.stats();
        println!(
            "  [vmm] Qwen3.6-MoE sparse routed expert residency active on backend={} device {ordinal}: \
             tensors={} max_pages={} logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB",
            backend,
            residency_stats.registered_tensors,
            max_resident_pages,
            arena_stats.logical_bytes as f64 / MIB,
            arena_stats.resident_bytes as f64 / MIB,
            arena_stats.reserved_bytes as f64 / MIB,
        );
        if persistent_decode {
            println!(
                "  [vmm] sparse MoE residency will use segmented persistent decode \
                 (router prefetch + FFN resume per layer)"
            );
        }
        if moe_prefetch_mode.uses_previous_token_routes() {
            println!(
                "  [vmm] sparse MoE {} lookahead prefetch active \
                 (ranks={moe_prefetch_ranks}/{})",
                moe_prefetch_mode.as_str(),
                geom.top_k
            );
            if moe_prefetch_mode.transition_weighted() {
                println!(
                    "  [vmm] sparse MoE transition predictor warmup \
                     min_observations={moe_transition_min_observations}"
                );
            }
        }
        return Ok(Qwen36DecodeLayers {
            layers,
            moe_expert_arena: None,
            moe_expert_residency: Some(manager),
        });
    }

    if should_try_moe_expert_vmm(
        moe_vmm_mode,
        backend,
        weight_mode.is_int4(),
        &format!("{weight_mode:?}"),
        ordinal,
    )? {
        let mut arena = BakedStore::virtual_weight_arena(ordinal);
        match load_all_layer_buffers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            Some(&mut arena),
            None,
        ) {
            Ok(layers) => {
                let stats = arena.stats();
                println!(
                    "  [vmm] Qwen3.6-MoE routed expert slabs active on backend={} device {ordinal}: \
                     allocations={} mappings={} logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB",
                    backend,
                    stats.allocations,
                    stats.mapping_count,
                    stats.logical_bytes as f64 / MIB,
                    stats.resident_bytes as f64 / MIB,
                    stats.reserved_bytes as f64 / MIB,
                );
                return Ok(Qwen36DecodeLayers {
                    layers,
                    moe_expert_arena: Some(arena),
                    moe_expert_residency: None,
                });
            }
            Err(err) if moe_vmm_mode == MoeExpertVmmMode::Auto => {
                eprintln!(
                    "[vmm] Qwen3.6-MoE routed expert VMM load failed ({err:#}); falling back to dense expert buffers"
                );
                drop(arena);
            }
            Err(err) => return Err(err.context("load Qwen3.6-MoE routed experts into VMM")),
        }
    }

    let layers = load_all_layer_buffers(
        store,
        ordinal,
        geom,
        text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        None,
        None,
    )?;
    Ok(Qwen36DecodeLayers {
        layers,
        moe_expert_arena: None,
        moe_expert_residency: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_decode_session(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    kv_max_t: usize,
    speculative_decode: bool,
    batched_spec_verify: bool,
    persistent_decode: bool,
    layers: &mut Vec<LayerBuffers>,
) -> Result<Qwen36DecodeSession> {
    let mtp_buffers_opt = if speculative_decode {
        match crate::qwen36_moe_mtp_loader::load_mtp_buffers(store, ordinal, geom, kv_max_t)
            .context("load MTP head from bake")?
        {
            Some(mtp) => {
                println!(
                    "  MTP head: loaded 19 mtp.* tensors (~1.6 GiB BF16) — \
                     speculative draft + sequential-verify path active. \
                     NOTE: sequential verification has zero amortized speedup \
                     over plain greedy decode (Phase 6.4's batched verify \
                     kernel is what delivers throughput); this path is the \
                     correctness foundation."
                );
                Some(mtp)
            }
            None => {
                anyhow::bail!(
                    "--speculative-decode requested but the bake doesn't \
                     include mtp.* tensors. Re-bake against the post-#84 \
                     `oracle/bake_int4.py`, or pull the new release tarball \
                     once the producer workflow at GitHub issue #87 lands."
                );
            }
        }
    } else {
        None
    };

    let final_norm_bytes = host_load_bytes(store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    let lm_head_bf16_bytes = load_lm_head_bf16(store, text_config, weight_prefix, geom)
        .context("prepare lm_head BF16 buffer")?;
    println!(
        "  uploading lm_head BF16 ({:.1} MiB) and final norm ({:.1} KiB) to GPU…",
        lm_head_bf16_bytes.len() as f64 / MIB,
        final_norm_bytes.len() as f64 / 1024.0,
    );
    let final_norm_w_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[geom.hidden as usize],
        &final_norm_bytes,
    )
    .context("upload final_norm_w to GPU")?;
    let lm_head_w_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[geom.vocab as usize, geom.hidden as usize],
        &lm_head_bf16_bytes,
    )
    .context("upload lm_head BF16 to GPU")?;
    let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.vocab as usize])
        .context("alloc logits_buf on GPU")?;
    let counter_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .context("alloc lm_head counter_buf on GPU")?;
    let final_hidden_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.hidden as usize])
        .context("alloc final_hidden_buf on GPU")?;
    drop(lm_head_bf16_bytes);
    drop(final_norm_bytes);

    let mtp_buffers = mtp_buffers_opt;
    let mtp_forward_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_forward_scratch(ordinal, geom, kv_max_t)
                .context("alloc MTP forward scratch")?,
        )
    } else {
        None
    };
    let mtp_chain_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_chain_scratch(ordinal, geom)
                .context("alloc MTP chain scratch")?,
        )
    } else {
        None
    };
    let embed_w_buf = if mtp_buffers.is_some() {
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let embed = load_to_gpu(store, ordinal, &embed_name)
            .with_context(|| format!("upload {embed_name} to GPU"))?;
        println!(
            "  uploaded embed_tokens ({:.0} MiB BF16) and allocated MTP \
             scratches (K={} drafts/step)",
            (geom.vocab as f64 * geom.hidden as f64 * 2.0) / MIB,
            QWEN36_NUM_SPECULATIVE_TOKENS,
        );
        Some(embed)
    } else {
        None
    };

    let linear_attn_snapshot = if speculative_decode && batched_spec_verify {
        Some(
            crate::qwen36_moe_state::save_linear_attn_state(ordinal, layers)
                .context("alloc linear-attn state snapshot for batched spec verify")?,
        )
    } else {
        None
    };
    if linear_attn_snapshot.is_some() {
        println!(
            "  --batched-spec-verify: linear-attn state snapshot allocated \
             (K+1 chains run per spec iter; restore + replay accepted prefix \
             on partial accept)"
        );
    }

    let persistent_scratch = if persistent_decode {
        let scratch =
            crate::qwen36_moe_persistent_decode::PersistentScratch::new(ordinal, geom, layers)
                .context("alloc PersistentScratch for --persistent-decode")?;
        println!(
            "  --persistent-decode: megakernel scratch allocated \
             (descs={}KiB, workspace={}KiB, ping/pong={}KiB){}",
            scratch.layer_descs_dev.len_bytes() / 1024,
            scratch.workspace.len_bytes() / 1024,
            scratch.hidden_ping.len_bytes() / 1024,
            if speculative_decode {
                " — also routes spec-verify chains through persistent"
            } else {
                ""
            },
        );
        Some(scratch)
    } else {
        None
    };

    Ok(Qwen36DecodeSession {
        final_norm_w_buf,
        lm_head_w_buf,
        logits_buf,
        counter_buf,
        final_hidden_buf,
        mtp_buffers,
        mtp_forward_scratch,
        mtp_chain_scratch,
        embed_w_buf,
        linear_attn_snapshot,
        persistent_scratch,
    })
}

/// Tokenize the prompt and run the multi-token decode loop end-to-end:
/// prefill the prompt one token at a time, then generate `max_new`
/// tokens via greedy argmax against the (cached) host-side lm_head
/// GEMV. Streams decoded text to stdout as each token arrives.
///
/// State persistence across decode steps:
///  - Linear-attn `conv_state` + `recurrent_state` mutated in place by
///    the kernel.
///  - Full-attn KV cache (PR 4d): per-layer `[kv_max_t, Hkv*d]` BF16
///    buffers; the kernel writes the current step's K/V at slot
///    `position` and attends over `kv_len = position + 1` past tokens.
///    `kv_max_t` sized for `prompt_len + max_new` here.
///
/// Greedy decoding only — sampling (temperature/top-p) and GPU-side
/// lm_head GEMV (currently host F32) are next perf/quality steps.
fn decode_text(
    model_dir: &Path,
    report: &DryRunReport,
    prompt: &str,
    max_new: usize,
    sampling: SamplingParams,
    emit_stage_timings: bool,
    speculative_decode: bool,
    fp8_runtime: bool,
    batched_spec_verify: bool,
    backend: Backend,
    ordinal: usize,
    persistent_decode: bool,
    kv_fp8: bool,
    dump_last_logits: bool,
) -> Result<()> {
    use std::io::Write as _;

    // Greedy-only gate for speculative decode. The Phase 6.3 protocol
    // verifies MTP drafts via greedy argmax against the base model's
    // logits — extending it to non-greedy sampling needs rejection
    // sampling (Speculative Decoding §3 in vLLM's reference), which
    // hasn't been implemented. Reject up front rather than silently
    // mix `argmax` (verify) with `sample_bf16_logits` (regular sample),
    // which would emit a different distribution than plain decode and
    // break reproducibility.
    //
    // `sample_bf16_logits` falls back to argmax when `temperature <= 0`
    // (or `top_k == 1`), so any non-trivial sampling configuration —
    // any of `temperature > 0`, `top_k != 1`, `top_p < 1.0` — counts
    // as non-greedy and is rejected here.
    if speculative_decode {
        let is_greedy = sampling.temperature <= 0.0 || sampling.top_k == 1;
        if !is_greedy {
            anyhow::bail!(
                "--speculative-decode currently supports greedy sampling \
                 only (temperature ≤ 0 or top_k == 1). Got temperature={}, \
                 top_k={}, top_p={}. Phase 6.4 will add sampling-consistent \
                 verification (rejection sampling); until then, re-run with \
                 `--temperature 0` for speculative decode, or drop \
                 `--speculative-decode` for non-greedy sampling.",
                sampling.temperature,
                sampling.top_k,
                sampling.top_p
            );
        }
    }

    let weight_prefix = report.kernel_params.weight_prefix;

    // Tokenizer first — without it we can't tokenize the prompt or stream
    // decoded text. Falls back to BOS-only if the tokenizer can't load.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path).ok();

    let bos_id = report
        .config
        .text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let eos_id = report
        .config
        .text_config
        .eos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let prompt_ids: Vec<u32> = match (&tokenizer, prompt.is_empty()) {
        (Some(tok), false) => {
            let enc = tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            if ids.is_empty() {
                vec![bos_id]
            } else {
                ids
            }
        }
        _ => vec![bos_id],
    };
    println!(
        "  prompt: {prompt:?} → {} token{} ({:?}{}…)",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        &prompt_ids[..prompt_ids.len().min(8)],
        if prompt_ids.len() > 8 { ", " } else { "" },
    );

    // Pick the bake. INT4 remains the default small-VRAM path; explicit
    // --fp8-runtime selects the native FP8 bake.
    let fp8_dir = model_store::bake_dir_fp8(model_dir);
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if fp8_runtime {
        if !fp8_dir.exists() {
            return Err(anyhow!(
                "--fp8-runtime requested but no FP8-native bake exists at {}. \
                 Create one with `python3 oracle/bake_fp8.py --model-dir {}`.",
                fp8_dir.display(),
                model_dir.display()
            ));
        }
        (fp8_dir, Qwen36WeightMode::Fp8)
    } else if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if bf16_dir.exists() {
        (bf16_dir, Qwen36WeightMode::Bf16)
    } else {
        return Err(anyhow!(
            "decode requires a baked package — neither FP8-native ({}), \
             INT4-GPTQ ({}) nor BF16 ({}) exists. Create one with the standard bake pipeline \
             or re-run with --dry-run for analytic accounting only.",
            fp8_dir.display(),
            int4_dir.display(),
            bf16_dir.display()
        ));
    };
    println!(
        "  loading from bake: {} ({})",
        bake_dir.display(),
        weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(backend);

    // KV cache size: needs to fit prompt_len + max_new past tokens. Sized
    // generously here since per-layer KV is small (10 full-attn layers ×
    // [kv_max_t, Hkv*d=512] BF16 = 10 KiB per token of context).
    let kv_max_t = prompt_ids.len() + max_new;

    println!(
        "  loading {} layers ({} INT4 sidecar sets, KV cache cap = {} tokens)…",
        geom.num_layers,
        if weight_mode.is_int4() {
            geom.num_layers
        } else {
            0
        },
        kv_max_t,
    );

    let moe_vmm_mode = MoeExpertVmmMode::from_env()?;
    let moe_island_cap_experts = moe_island_cap_experts_from_env()?;
    if moe_island_cap_experts.is_some() && speculative_decode {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS sparse residency is not wired through speculative decode yet"
        );
    }
    if moe_island_cap_experts.is_some() && moe_vmm_mode == MoeExpertVmmMode::Disabled {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS requires VMM expert slabs; unset SUPERSONIC_VMM_MOE_ISLANDS=0"
        );
    }
    let sparse_moe_requested = moe_island_cap_experts.is_some();
    let moe_prefetch_mode = MoeIslandPrefetchMode::from_env()?;
    let moe_prefetch_ranks =
        moe_island_prefetch_ranks_from_env(moe_prefetch_mode, geom.top_k as usize)?;
    let moe_transition_min_observations =
        moe_island_prefetch_transition_min_observations(moe_prefetch_mode)?;
    if moe_prefetch_mode != MoeIslandPrefetchMode::Disabled && !sparse_moe_requested {
        anyhow::bail!("SUPERSONIC_MOE_ISLAND_PREFETCH requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS");
    }
    let mut moe_sparse_telemetry = MoeSparseTelemetry::from_env(
        sparse_moe_requested,
        persistent_decode,
        moe_prefetch_mode,
        moe_prefetch_ranks,
    )?;
    if let Some(path) = moe_sparse_telemetry
        .as_ref()
        .and_then(|telemetry| telemetry.dump_path.as_ref())
    {
        println!(
            "  [vmm] sparse MoE residency telemetry will be written to {}",
            path.display()
        );
    }
    let kv_vmm = should_use_qwen36_kv_vmm(backend, ordinal)?;
    let loaded_layers = load_decode_layers_with_vmm_strategy(
        &store,
        ordinal,
        backend,
        &geom,
        &report.config.text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        moe_vmm_mode,
        moe_island_cap_experts,
        moe_prefetch_mode,
        moe_prefetch_ranks,
        moe_transition_min_observations,
        persistent_decode,
    )?;
    let mut layers = loaded_layers.layers;
    let _moe_expert_arena = loaded_layers.moe_expert_arena;
    let mut _moe_expert_residency = loaded_layers.moe_expert_residency;
    let virtual_kv_stats = virtual_kv_stats_for_layers(&layers);
    if virtual_kv_stats.layers > 0 {
        println!(
            "  [vmm] Qwen3.6-MoE {} KV active on backend={} device {ordinal}: \
             layers={} mappings={} logical={:.2}MiB logical_resident={:.2}MiB \
             resident={:.2}MiB reserved={:.2}MiB",
            if kv_fp8 { "FP8" } else { "BF16" },
            backend,
            virtual_kv_stats.layers,
            virtual_kv_stats.mappings,
            virtual_kv_stats.logical_bytes as f64 / MIB,
            virtual_kv_stats.logical_resident_bytes as f64 / MIB,
            virtual_kv_stats.resident_bytes as f64 / MIB,
            virtual_kv_stats.reserved_bytes as f64 / MIB,
        );
    }
    let session = prepare_decode_session(
        &store,
        ordinal,
        &geom,
        &report.config.text_config,
        weight_prefix,
        kv_max_t,
        speculative_decode,
        batched_spec_verify,
        persistent_decode,
        &mut layers,
    )?;
    let Qwen36DecodeSession {
        final_norm_w_buf,
        lm_head_w_buf,
        mut logits_buf,
        mut counter_buf,
        mut final_hidden_buf,
        mut mtp_buffers,
        mut mtp_forward_scratch,
        mut mtp_chain_scratch,
        embed_w_buf,
        mut linear_attn_snapshot,
        mut persistent_scratch,
    } = session;

    println!(
        "  decoding {} prompt token{} + generating ≤{} new token{}…",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        max_new,
        if max_new == 1 { "" } else { "s" },
    );
    println!();
    print!("> ");
    if let Some(tok) = &tokenizer {
        if let Ok(prompt_text) = tok.decode(&prompt_ids, false) {
            print!("{prompt_text}");
        }
    }
    std::io::stdout().flush().ok();

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    // Track the BF16 logits bytes from the last decode step for --dump-last-logits.
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    let mut current_token: u32 = prompt_ids[0];
    let mut position: i32 = 0;
    // Standard prefill+generate shape: feed prompt[0..N-1] as prefill (logits
    // discarded), then prompt[N-1] is the first forward whose logits we
    // sample. Subsequent gen steps feed the just-sampled token. Total
    // forwards = (prompt_len - 1) prefill + max_new generation = prompt_len
    // + max_new - 1.
    let total_steps = prompt_ids.len() + max_new - 1;
    let mut rng = XorshiftRng::new(sampling.seed);
    println!(
        "  sampling: temp={} top_k={} top_p={} seed={}",
        sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed,
    );

    // Per-stage wall-clock accumulators. Aggregated across generation steps
    // only (prefill steps run the chain but skip the lm_head/sample stages,
    // so timing prefill mixed with gen would distort the per-token average).
    // `chain_ms` includes the GPU work + the D2H copy of `final_hidden_bytes`
    // — `run_chained_decode` syncs before returning, so the wall-clock here
    // is a real GPU+sync measurement. CPU-side stages (embed lookup, lm_head
    // GEMV, sampling, detokenize) are pure host work.
    let mut stage_timings = Qwen36StageTimingTotals::default();
    let mut previous_moe_topk_by_layer: Vec<Vec<usize>> =
        vec![Vec::new(); geom.num_layers as usize];
    let mut moe_route_telemetry =
        sparse_moe_requested.then(|| MoeRouteTelemetry::new(geom.top_k as usize));
    let mut moe_transition_predictors = moe_prefetch_mode.transition_weighted().then(|| {
        vec![
            MoeTransitionPredictor::new(geom.top_k as usize, moe_transition_min_observations,);
            geom.num_layers as usize
        ]
    });

    for step in 0..total_steps {
        // When speculative decode is on, each iteration can commit
        // multiple tokens (up to K+1), so the standard `total_steps =
        // prompt_len + max_new - 1` count over-shoots. Break here once
        // we've already committed `max_new` tokens — otherwise the
        // next regular chain call would request a cache slot beyond
        // `kv_max_t = prompt_len + max_new` (status 120). Plain decode
        // stays bit-identical because it always emits exactly one
        // token per iteration.
        if generated_ids.len() >= max_new {
            break;
        }
        // Embed lookup for the current token.
        let t0 = std::time::Instant::now();
        let initial_hidden = lookup_embed_row(
            &store,
            weight_prefix,
            current_token as usize,
            geom.hidden as usize,
        )
        .with_context(|| format!("embed lookup token {current_token} (step {step})"))?;
        let t_embed_step = t0.elapsed();

        // Run the chain. Linear-attn state mutates in `layers` in place.
        // `run_chained_decode_fast` skips the per-layer D2H sync chain
        // (~80 GPU syncs/token on 35B-A3B) — `decode_text` only consumes
        // `final_hidden_bytes`. The multilayer parity test still calls
        // the legacy `run_chained_decode` which captures per-layer.
        let t1 = std::time::Instant::now();
        // When `--emit-stage-timings` is set, sync after each step launch
        // so the per-stage `kernel_*_us` accumulators in `outputs` reflect
        // GPU compute time. Without it, PR #80's async dispatch path
        // would record host queue time instead — fast but useless for
        // stage-level perf attribution. The total `chain_ms` measured by
        // the wall-clock around this call stays correct either way
        // because `run_chained_decode_fast` ends with a D2H copy that
        // implicitly drains the queue.
        // Phase 3f: on generation steps when the persistent path is
        // active, fold final RMSnorm + lm_head GEMV into the
        // megakernel — saves the separate `lm_head_launch` (one launch
        // + ~30 µs) and the H2D round-trip that staged final_hidden
        // into final_hidden_buf. The host then D2Hs `logits_buf`
        // directly. On prefill steps logits aren't needed; on the
        // chained path the explicit lm_head_launch path stays.
        let is_gen_step = step + 1 >= prompt_ids.len();
        let fold = if is_gen_step {
            Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                final_norm_w: &final_norm_w_buf,
                lm_head_w: &lm_head_w_buf,
                logits_out: &mut logits_buf,
                vocab: geom.vocab,
            })
        } else {
            None
        };
        let lm_head_folded;
        let moe_telemetry_before = _moe_expert_residency
            .as_ref()
            .map(MoeSparseTelemetrySnapshot::capture);
        let track_moe_routes =
            moe_prefetch_mode.uses_previous_token_routes() || moe_route_telemetry.is_some();
        let mut next_moe_topk_by_layer = if track_moe_routes {
            previous_moe_topk_by_layer.clone()
        } else {
            Vec::new()
        };
        let outputs = if let Some(scratch) = persistent_scratch.as_mut() {
            if let Some(manager) = _moe_expert_residency.as_mut() {
                // Sparse VMM needs a host remap point after each layer's
                // router top-k is known. The segmented persistent path keeps
                // the persistent phase bodies but skips folded lm_head; the
                // standalone lm_head launch below consumes final_hidden_bytes.
                lm_head_folded = false;
                drop(fold);
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        &store,
                        moe_prefetch_mode,
                        moe_prefetch_ranks,
                        &previous_moe_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        moe_route_telemetry.as_mut(),
                        moe_transition_predictors.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                scratch
                    .run_sparse_with_expert_prefetch(
                        ordinal,
                        &initial_hidden,
                        position,
                        &mut prefetch,
                    )
                    .with_context(|| {
                        format!(
                            "segmented persistent sparse decode (step {step}, position {position})"
                        )
                    })?
            } else {
                lm_head_folded = fold.is_some();
                scratch
                    .run(ordinal, &initial_hidden, position, fold)
                    .with_context(|| {
                        format!("persistent decode (step {step}, position {position})")
                    })?
            }
        } else {
            // Chained path doesn't support the fold; lm_head still
            // launches separately below on gen steps.
            lm_head_folded = false;
            drop(fold);
            if let Some(manager) = _moe_expert_residency.as_mut() {
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        &store,
                        moe_prefetch_mode,
                        moe_prefetch_ranks,
                        &previous_moe_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        moe_route_telemetry.as_mut(),
                        moe_transition_predictors.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                run_chained_decode_fast_with_expert_prefetch(
                    ordinal,
                    &geom,
                    &mut layers,
                    &initial_hidden,
                    position,
                    emit_stage_timings,
                    &mut prefetch,
                )
            } else {
                run_chained_decode_fast(
                    ordinal,
                    &geom,
                    &mut layers,
                    &initial_hidden,
                    position,
                    emit_stage_timings,
                )
            }
            .with_context(|| format!("chained decode (step {step}, position {position})"))?
        };
        if track_moe_routes {
            previous_moe_topk_by_layer = next_moe_topk_by_layer;
        }
        if let (Some(telemetry), Some(before), Some(manager)) = (
            moe_sparse_telemetry.as_mut(),
            moe_telemetry_before,
            _moe_expert_residency.as_ref(),
        ) {
            let after = MoeSparseTelemetrySnapshot::capture(manager);
            telemetry.record_step(step, position, is_gen_step, before, after);
        }
        let t_chain_step = t1.elapsed();
        position += 1;

        // KV-FP8 sidecar descriptors stay fixed across decode. The
        // persistent kernel computes the rolling covered range from
        // `position` and `kv_shadow_window`, so no descriptor re-upload is
        // needed when old sidecar slots roll over.

        // Prefill steps: feed the next prompt token without computing logits.
        if step + 1 < prompt_ids.len() {
            current_token = prompt_ids[step + 1];
            continue;
        }

        // Optional dump for the host-side post-chain debug harness.
        if let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN") {
            std::fs::write(&dump_path, &outputs.final_hidden_bytes)
                .with_context(|| format!("write final_hidden dump to {dump_path}"))?;
            eprintln!(
                "[debug] dumped step={step} position={position} final_hidden ({} BF16 bytes) to {dump_path}",
                outputs.final_hidden_bytes.len()
            );
        }

        // Generation step: when the megakernel didn't fold lm_head
        // (chained path), launch the standalone final RMSnorm +
        // lm_head GEMV here. When folded (persistent + gen), skip the
        // launch — `logits_buf` is already populated by the
        // megakernel.
        let t2 = std::time::Instant::now();
        if !lm_head_folded {
            gpu_hal::copy_h2d(
                ordinal,
                final_hidden_buf.as_mut_ptr(),
                outputs.final_hidden_bytes.as_ptr() as *const _,
                outputs.final_hidden_bytes.len(),
            )
            .context("h2d final_hidden -> final_hidden_buf")?;
            kernel_ffi::qwen36_moe::lm_head_launch(
                ordinal,
                geom.hidden,
                geom.vocab,
                geom.rms_norm_eps,
                &final_hidden_buf,
                &final_norm_w_buf,
                &lm_head_w_buf,
                &mut logits_buf,
                None, // base decode doesn't capture h_post — that's MTP-only
                &mut counter_buf,
            )
            .context("gpu lm_head launch")?;
        }
        let logits = logits_buf
            .to_host_bytes()
            .context("d2h logits from GPU lm_head")?;
        if dump_last_logits {
            last_logits_bytes.clone_from(&logits);
        }
        let t_lm_head_step = t2.elapsed();
        if let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_LOGITS") {
            std::fs::write(&dump_path, &logits)
                .with_context(|| format!("write logits dump to {dump_path}"))?;
            eprintln!(
                "[debug] dumped step={step} logits ({} BF16 bytes) to {dump_path}",
                logits.len()
            );
        }
        let t3 = std::time::Instant::now();
        let next_token = sample_bf16_logits(
            &logits,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            &mut rng,
        );
        let t_sample_step = t3.elapsed();
        generated_ids.push(next_token);

        // Stream-decode and print.
        let t4 = std::time::Instant::now();
        if let Some(tok) = &tokenizer {
            if let Ok(text) = tok.decode(&[next_token], false) {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
        }
        let t_detok_step = t4.elapsed();

        stage_timings.record_generation_step(
            t_embed_step,
            t_chain_step,
            t_lm_head_step,
            t_sample_step,
            t_detok_step,
            &outputs,
        );

        if Some(next_token) == eos_id {
            break;
        }
        current_token = next_token;

        // Phase 6.3d: speculative extension. After the regular sample,
        // try to commit additional tokens via MTP draft chain +
        // sequential base verification. The closure wraps one base
        // decode step (embed → chain → lm_head → host argmax). Honors
        // `max_new` and EOS by truncating emitted tokens; the
        // outer-loop counter advances normally because each iteration
        // still runs at least one base step.
        //
        // Sequential verify gives no amortized speedup vs plain greedy
        // (each accepted draft costs one base step to produce the next
        // prediction). Phase 6.4's batched verification is what lifts
        // tok/s. This wiring is the correctness foundation.
        if let (Some(mtp), Some(fwd_scratch), Some(chain_scratch), Some(embed_w)) = (
            mtp_buffers.as_mut(),
            mtp_forward_scratch.as_mut(),
            mtp_chain_scratch.as_mut(),
            embed_w_buf.as_ref(),
        ) {
            if generated_ids.len() >= max_new {
                break;
            }
            // Cap K to the remaining max_new headroom so the verify
            // loop never writes cache slots beyond what we'll
            // actually commit to `generated_ids`. Spec emits up to
            // K+1 tokens (K accepted + 1 corrected/bonus), so the
            // available draft count is `headroom - 1`. If headroom <=
            // 1 we can still emit 1 token via the K=0 fallback; if
            // headroom == 0 we already broke out above.
            let headroom = max_new - generated_ids.len();
            let dynamic_k = QWEN36_NUM_SPECULATIVE_TOKENS.min(headroom.saturating_sub(1));
            let h_base = outputs.final_hidden_bytes.clone();
            // P2: thread spec-verify timings into the engine-level
            // accumulators so `--emit-stage-timings` reports honest
            // per-token costs under speculative decode. Without these
            // captures, every base step inside the verify loop would
            // be invisible to `gen_steps` / `t_chain` / `t_lm_head`,
            // making the speculative path look ~K+1× faster than it
            // really is on stage-timings dashboards.
            //
            // Phase 6.4c.2: route through the BATCHED driver when
            // `--batched-spec-verify` is set. The batched closure runs
            // K+1 chains sequentially (state mutates through them),
            // accumulates K+1 final_hidden bytes, runs ONE batched
            // lm_head over [K+1, hidden], does K+1 host argmaxes. On
            // partial-accept the engine restores linear-attn state
            // from the pre-spec snapshot then replays the accepted
            // prefix sequentially to advance state correctly.
            let result = if let Some(snapshot) = linear_attn_snapshot.as_mut() {
                refresh_linear_attn_state(ordinal, &layers, snapshot)
                    .context("refresh linear-attn snapshot before batched verify")?;

                let r = run_speculative_decode_step_batched(
                    ordinal,
                    &geom,
                    mtp,
                    fwd_scratch,
                    chain_scratch,
                    embed_w,
                    &lm_head_w_buf,
                    &h_base,
                    next_token,
                    position,
                    dynamic_k,
                    |inputs| -> anyhow::Result<Vec<(u32, Vec<u8>)>> {
                        let n = inputs.len();
                        if n == 0 {
                            return Ok(Vec::new());
                        }
                        let hidden = geom.hidden as usize;

                        // K+1 sequential chains, accumulate final_hiddens.
                        let mut final_hiddens: Vec<Vec<u8>> = Vec::with_capacity(n);
                        for &(pos, input) in inputs {
                            let t_embed_start = std::time::Instant::now();
                            let initial_hidden = lookup_embed_row(
                                &store,
                                weight_prefix,
                                input as usize,
                                geom.hidden as usize,
                            )?;
                            stage_timings.record_embed(t_embed_start.elapsed());

                            let t_chain_start = std::time::Instant::now();
                            let chain_outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                                scratch.run(ordinal, &initial_hidden, pos, None)?
                            } else {
                                run_chained_decode_fast(
                                    ordinal,
                                    &geom,
                                    &mut layers,
                                    &initial_hidden,
                                    pos,
                                    emit_stage_timings,
                                )?
                            };
                            stage_timings.record_chain(t_chain_start.elapsed(), &chain_outputs);
                            stage_timings.count_generation_step();
                            final_hiddens.push(chain_outputs.final_hidden_bytes);
                        }

                        // ONE batched lm_head over [n, hidden].
                        let t_lm_head_start = std::time::Instant::now();
                        let mut concat = Vec::with_capacity(n * hidden * 2);
                        for fh in &final_hiddens {
                            concat.extend_from_slice(fh);
                        }
                        let fh_buf = gpu_hal::GpuBuffer::from_host_bytes(
                            ordinal,
                            gpu_hal::ScalarType::BF16,
                            &[n, hidden],
                            &concat,
                        )?;
                        let mut logits_buf_b = gpu_hal::GpuBuffer::zeros(
                            ordinal,
                            gpu_hal::ScalarType::BF16,
                            &[n, geom.vocab as usize],
                        )?;
                        kernel_ffi::qwen36_moe::lm_head_batched_launch(
                            ordinal,
                            n as i32,
                            geom.hidden,
                            geom.vocab,
                            geom.rms_norm_eps,
                            &fh_buf,
                            &final_norm_w_buf,
                            &lm_head_w_buf,
                            &mut logits_buf_b,
                            None,
                        )?;
                        let logits_bytes =
                            logits_buf_b.to_host_bytes().context("d2h batched logits")?;
                        stage_timings.record_lm_head(t_lm_head_start.elapsed());

                        let row_bytes = geom.vocab as usize * 2;
                        let mut results: Vec<(u32, Vec<u8>)> = Vec::with_capacity(n);
                        for (i, fh) in final_hiddens.into_iter().enumerate() {
                            let row = &logits_bytes[i * row_bytes..(i + 1) * row_bytes];
                            results.push((argmax_bf16_logits(row), fh));
                        }
                        Ok(results)
                    },
                )
                .context("batched speculative decode step")?;

                // State mgmt: on partial-accept, restore + replay the
                // accepted prefix to advance linear-attn state correctly.
                // Full-accept (n_accepted == dynamic_k) needs no fixup —
                // K+1 chains advanced state through K+1 inputs which is
                // exactly what we committed (drafts + bonus's input was
                // drafts[K-1], one more chain's worth advances naturally
                // when next iter feeds the bonus token).
                if r.n_accepted < dynamic_k {
                    restore_linear_attn_state(ordinal, &mut layers, snapshot)
                        .context("restore linear-attn state after partial-accept")?;
                    // Replay (j+1) chains: first_token at `position`,
                    // then the j accepted drafts at `position+1..position+j`.
                    let mut replay: Vec<(i32, u32)> = Vec::with_capacity(r.n_accepted + 1);
                    replay.push((position, next_token));
                    for (i, &tok) in r.emitted_tokens.iter().take(r.n_accepted).enumerate() {
                        replay.push((position + 1 + i as i32, tok));
                    }
                    for &(pos, input) in &replay {
                        let t_embed_start = std::time::Instant::now();
                        let initial_hidden = lookup_embed_row(
                            &store,
                            weight_prefix,
                            input as usize,
                            geom.hidden as usize,
                        )?;
                        stage_timings.record_embed(t_embed_start.elapsed());
                        let t_chain_start = std::time::Instant::now();
                        let replay_outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                            scratch.run(ordinal, &initial_hidden, pos, None)?
                        } else {
                            run_chained_decode_fast(
                                ordinal,
                                &geom,
                                &mut layers,
                                &initial_hidden,
                                pos,
                                emit_stage_timings,
                            )?
                        };
                        stage_timings.record_chain(t_chain_start.elapsed(), &replay_outputs);
                        // Per-kernel-class breakdown for replay chains
                        // contributes to the same accumulators as the
                        // verify chains so `--emit-stage-timings` reports
                        // honest full-attn/linear-attn/ffn averages on
                        // partial-accept iters. Without this the reported
                        // chain breakdown undercounts actual work as the
                        // accept rate falls.
                        stage_timings.count_generation_step();
                    }
                }
                r
            } else {
                run_speculative_decode_step(
                    ordinal,
                    &geom,
                    mtp,
                    fwd_scratch,
                    chain_scratch,
                    embed_w,
                    &lm_head_w_buf,
                    &h_base,
                    next_token,
                    position,
                    dynamic_k,
                    |pos, input| -> anyhow::Result<(u32, Vec<u8>)> {
                        // Embed lookup is its own stage in the timing
                        // breakdown — bundling it into `t_chain` (as the
                        // first cut of this closure did) systematically
                        // inflates `chain_ms_avg` and deflates
                        // `embed_ms_avg` as the MTP accept rate rises.
                        let t_embed_start = std::time::Instant::now();
                        let initial_hidden = lookup_embed_row(
                            &store,
                            weight_prefix,
                            input as usize,
                            geom.hidden as usize,
                        )?;
                        stage_timings.record_embed(t_embed_start.elapsed());

                        let t_chain_start = std::time::Instant::now();
                        let outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                            scratch.run(ordinal, &initial_hidden, pos, None)?
                        } else {
                            run_chained_decode_fast(
                                ordinal,
                                &geom,
                                &mut layers,
                                &initial_hidden,
                                pos,
                                emit_stage_timings,
                            )?
                        };
                        stage_timings.record_chain(t_chain_start.elapsed(), &outputs);

                        let t_lm_head_start = std::time::Instant::now();
                        gpu_hal::copy_h2d(
                            ordinal,
                            final_hidden_buf.as_mut_ptr(),
                            outputs.final_hidden_bytes.as_ptr() as *const _,
                            outputs.final_hidden_bytes.len(),
                        )?;
                        kernel_ffi::qwen36_moe::lm_head_launch(
                            ordinal,
                            geom.hidden,
                            geom.vocab,
                            geom.rms_norm_eps,
                            &final_hidden_buf,
                            &final_norm_w_buf,
                            &lm_head_w_buf,
                            &mut logits_buf,
                            None,
                            &mut counter_buf,
                        )?;
                        let logits_bytes = logits_buf
                            .to_host_bytes()
                            .context("d2h logits from spec verify lm_head")?;
                        stage_timings.record_lm_head(t_lm_head_start.elapsed());
                        // Each verify base step counts as one decode step
                        // for the per-token average — emitted_tokens.len()
                        // tokens are committed per spec call, and
                        // closure-call-count == emitted_tokens.len() in
                        // both the partial-accept and full-accept (with
                        // bonus) cases. Bumping here is equivalent to
                        // "one closure call = one decode step worth of
                        // base work."
                        stage_timings.count_generation_step();
                        Ok((
                            argmax_bf16_logits(&logits_bytes),
                            outputs.final_hidden_bytes,
                        ))
                    },
                )
                .context("speculative decode step")?
            };

            // Append emitted tokens. Honour `max_new` and EOS by
            // breaking out cleanly mid-emission.
            let mut hit_stop = false;
            for tok in result.emitted_tokens.iter().copied() {
                if generated_ids.len() >= max_new {
                    hit_stop = true;
                    break;
                }
                generated_ids.push(tok);
                if let Some(t) = &tokenizer {
                    if let Ok(text) = t.decode(&[tok], false) {
                        print!("{text}");
                    }
                }
                if Some(tok) == eos_id {
                    hit_stop = true;
                    break;
                }
            }
            std::io::stdout().flush().ok();
            position += result.emitted_tokens.len() as i32;
            if hit_stop {
                break;
            }
            current_token = *result
                .emitted_tokens
                .last()
                .expect("speculative step must emit at least one token (K=0 fallback ensured)");
        }
    }

    // Emit last-step logits for integration parity tests (--dump-last-logits).
    // Printed BEFORE any other post-loop output so the test parser can grep for
    // the first line starting with "LAST_LOGITS: ".
    if dump_last_logits && !last_logits_bytes.is_empty() {
        use std::io::Write as _;
        let logits_f32 = crate::qwen36_moe_decode::bf16_bytes_to_f32(&last_logits_bytes);
        // Lead with `\n` so the marker lands at the start of its own line —
        // the streamed-token print path uses `print!` without a trailing
        // newline, so `LAST_LOGITS:` would otherwise concatenate onto the
        // last generated text and `lines().find(...starts_with...)` in the
        // parity/smoke tests wouldn't match.
        // Format with `{}` (Display) instead of `{:.6}` to preserve full
        // f32 precision — the VMM bit-exact smoke compares via
        // `a.to_bits() == b.to_bits()`, which fails on rounded values.
        print!("\nLAST_LOGITS: ");
        for (i, x) in logits_f32.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    println!();
    println!();
    println!(
        "Generated {} token{} ({} prompt + {} new). EOS: {}.",
        generated_ids.len(),
        if generated_ids.len() == 1 { "" } else { "s" },
        prompt_ids.len(),
        generated_ids.len(),
        if eos_id
            .map(|e| generated_ids.last() == Some(&e))
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no (max_new_tokens hit)"
        },
    );
    if !generated_ids.is_empty() {
        println!("  Generated ids: {generated_ids:?}");
    }
    if let Some(manager) = _moe_expert_residency.as_ref() {
        let residency = manager.stats();
        let arena = manager.arena().stats();
        let total_resident_bytes = arena.resident_bytes + virtual_kv_stats.resident_bytes;
        let total_reserved_bytes = arena.reserved_bytes + virtual_kv_stats.reserved_bytes;
        if let Some(telemetry) = moe_sparse_telemetry.as_ref() {
            println!(
                "  [vmm] MoE island residency: resident_slices={} peak_slices={} \
                 resident_pages={} peak_pages={} page_backed_slices={} \
                 hits={} misses={} page_hits={} page_misses={} evicted_slices={} evicted_pages={} \
                 prefetch_requests={} prefetch_hits={} prefetch_misses={} \
                 prefetch_page_hits={} prefetch_page_misses={} \
                 prefetch_skipped={} prefetch_skipped_pages={} \
                 uploaded={:.2}MiB unmapped={:.2}MiB \
                 resident={:.2}MiB peak_resident={:.2}MiB reserved={:.2}MiB \
                 kv_resident={:.2}MiB total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
                residency.resident_slices,
                telemetry.peak_resident_slices,
                residency.resident_pages,
                telemetry.peak_resident_pages,
                residency.page_backed_slices,
                residency.hits,
                residency.misses,
                residency.page_hits,
                residency.page_misses,
                residency.evicted_slices,
                residency.evicted_pages,
                residency.prefetch_requests,
                residency.prefetch_hits,
                residency.prefetch_misses,
                residency.prefetch_page_hits,
                residency.prefetch_page_misses,
                residency.prefetch_skipped,
                residency.prefetch_skipped_pages,
                residency.uploaded_bytes as f64 / MIB,
                residency.unmapped_bytes as f64 / MIB,
                arena.resident_bytes as f64 / MIB,
                telemetry.peak_resident_bytes as f64 / MIB,
                arena.reserved_bytes as f64 / MIB,
                virtual_kv_stats.resident_bytes as f64 / MIB,
                total_resident_bytes as f64 / MIB,
                total_reserved_bytes as f64 / MIB,
            );
            telemetry.write_json(
                manager,
                virtual_kv_stats,
                &generated_ids,
                moe_route_telemetry.as_ref(),
            )?;
        } else {
            println!(
                "  [vmm] MoE island residency: resident_slices={} resident_pages={} \
                 page_backed_slices={} hits={} misses={} page_hits={} page_misses={} \
                 evicted_slices={} evicted_pages={} prefetch_requests={} \
                 prefetch_hits={} prefetch_misses={} prefetch_page_hits={} \
                 prefetch_page_misses={} prefetch_skipped={} prefetch_skipped_pages={} \
                 uploaded={:.2}MiB unmapped={:.2}MiB \
                 resident={:.2}MiB reserved={:.2}MiB kv_resident={:.2}MiB \
                 total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
                residency.resident_slices,
                residency.resident_pages,
                residency.page_backed_slices,
                residency.hits,
                residency.misses,
                residency.page_hits,
                residency.page_misses,
                residency.evicted_slices,
                residency.evicted_pages,
                residency.prefetch_requests,
                residency.prefetch_hits,
                residency.prefetch_misses,
                residency.prefetch_page_hits,
                residency.prefetch_page_misses,
                residency.prefetch_skipped,
                residency.prefetch_skipped_pages,
                residency.uploaded_bytes as f64 / MIB,
                residency.unmapped_bytes as f64 / MIB,
                arena.resident_bytes as f64 / MIB,
                arena.reserved_bytes as f64 / MIB,
                virtual_kv_stats.resident_bytes as f64 / MIB,
                total_resident_bytes as f64 / MIB,
                total_reserved_bytes as f64 / MIB,
            );
        }
    }
    if emit_stage_timings && stage_timings.gen_steps > 0 {
        let to_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let chain_ms = to_ms(stage_timings.chain);
        let embed_ms = to_ms(stage_timings.embed);
        let lm_head_ms = to_ms(stage_timings.lm_head);
        let sample_ms = to_ms(stage_timings.sample);
        let detok_ms = to_ms(stage_timings.detok);
        let total_ms = chain_ms + embed_ms + lm_head_ms + sample_ms + detok_ms;
        let n = stage_timings.gen_steps as f64;
        let full_attn_ms = (stage_timings.chain_full_attn_us as f64) / 1000.0;
        let linear_attn_ms = (stage_timings.chain_linear_attn_us as f64) / 1000.0;
        let ffn_ms = (stage_timings.chain_ffn_us as f64) / 1000.0;
        eprintln!(
            "[qwen36-moe stage-timings] gen_steps={} \
             embed_ms_avg={:.3} chain_ms_avg={:.3} lm_head_ms_avg={:.3} \
             sample_ms_avg={:.3} detok_ms_avg={:.3} total_ms_avg={:.3} \
             (chain_total_ms={:.1} lm_head_total_ms={:.1})",
            stage_timings.gen_steps,
            embed_ms / n,
            chain_ms / n,
            lm_head_ms / n,
            sample_ms / n,
            detok_ms / n,
            total_ms / n,
            chain_ms,
            lm_head_ms,
        );
        eprintln!(
            "[qwen36-moe chain-breakdown] gen_steps={} \
             full_attn_ms_avg={:.3} linear_attn_ms_avg={:.3} ffn_ms_avg={:.3} \
             (full_attn_total_ms={:.1} linear_attn_total_ms={:.1} ffn_total_ms={:.1})",
            stage_timings.gen_steps,
            full_attn_ms / n,
            linear_attn_ms / n,
            ffn_ms / n,
            full_attn_ms,
            linear_attn_ms,
            ffn_ms,
        );
    }

    Ok(())
}

/// Load + dequantize lm_head into a BF16 byte buffer that the host-side
/// Legacy single-token entry point — keeps the original `decode_first_token`
/// callable so the path stays exercised. Currently unused but documents the
/// minimal one-step decode shape.
#[allow(dead_code)]
fn decode_first_token(model_dir: &Path, report: &DryRunReport, kv_fp8: bool) -> Result<u32> {
    let weight_prefix = report.kernel_params.weight_prefix;

    // Pick the bake. INT4 is the realistic path on 24 GiB VRAM.
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if bf16_dir.exists() {
        (bf16_dir, Qwen36WeightMode::Bf16)
    } else {
        return Err(anyhow!(
            "decode requires a baked package — neither INT4-GPTQ ({}) nor \
             BF16 ({}) exists. Create one with the standard bake pipeline \
             or re-run with --dry-run for analytic accounting only.",
            int4_dir.display(),
            bf16_dir.display()
        ));
    };
    println!(
        "  loading from bake: {} ({})",
        bake_dir.display(),
        if weight_mode == Qwen36WeightMode::Int4 {
            "INT4 GPTQ"
        } else {
            "BF16"
        }
    );
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    println!(
        "  loading {} layer{} ({} INT4 sidecar set{})…",
        geom.num_layers,
        if geom.num_layers == 1 { "" } else { "s" },
        if weight_mode == Qwen36WeightMode::Int4 {
            geom.num_layers
        } else {
            0
        },
        if geom.num_layers == 1 { "" } else { "s" },
    );
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            &store,
            ordinal,
            li,
            &geom,
            &report.config.text_config,
            weight_prefix,
            weight_mode,
            0, // legacy single-token path: no KV cache, kv_len=1 fast path.
            kv_fp8,
            false,
            None,
            None,
        )
        .with_context(|| format!("load layer {li} weights"))?;
        layers.push(layer);
    }

    // BOS token: if the config exposes one, prefer it; otherwise default to
    // 0. Either way the parity criterion is "doesn't bail and emits a token",
    // and the produced token id reflects whatever embedding row we picked.
    let bos = report
        .config
        .text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let initial_hidden = lookup_embed_row(&store, weight_prefix, bos, geom.hidden as usize)
        .with_context(|| format!("lookup embed row {bos}"))?;
    println!(
        "  embedding row {bos} loaded ({} BF16 bytes)",
        initial_hidden.len()
    );

    println!("  running chained decode…");
    let outputs = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, 0)
        .context("chained decode")?;
    println!(
        "  decode done; final hidden norm = {:.4}",
        crate::qwen36_moe_decode::bf16_bytes_to_f32(&outputs.final_hidden_bytes)
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    );

    let final_norm_bytes = host_load_bytes(&store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    let lm_head_bf16_bytes =
        load_lm_head_bf16(&store, &report.config.text_config, weight_prefix, &geom)
            .context("prepare lm_head BF16 buffer")?;

    println!("  computing host-side norm + lm_head GEMV…");
    let logits = host_final_norm_lm_head(
        &outputs.final_hidden_bytes,
        &final_norm_bytes,
        &lm_head_bf16_bytes,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );
    Ok(argmax_bf16_logits(&logits))
}

#[cfg(test)]
mod tests {
    use super::{ExpertRoute, MoeIslandPrefetchMode, MoeRouteTelemetry, MoeTransitionPredictor};

    #[test]
    fn moe_prefetch_mode_env_accepts_disabled_and_previous_token_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(None).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("disabled")).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("prev-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_resident_only_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token-resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous_token_resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("resident-previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert!(MoeIslandPrefetchMode::from_env_value(Some("resident")).is_err());
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_transition_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition-weighted")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(MoeIslandPrefetchMode::Transition.as_str(), "transition");
        assert!(MoeIslandPrefetchMode::Transition.uses_previous_token_routes());
        assert!(MoeIslandPrefetchMode::Transition.transition_weighted());
    }

    #[test]
    fn moe_transition_predictor_waits_for_warmup_and_scores_repeats() {
        let mut predictor = MoeTransitionPredictor::new(3, 2);
        let previous_routes = [10, 20, 30];
        let routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 20,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 99,
                weight: 0.25,
            },
        ];

        predictor.update(&routes, &previous_routes);
        assert!(predictor.candidates(&previous_routes, 2).is_empty());

        predictor.update(&routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20]);

        let later_routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 10,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 20,
                weight: 0.25,
            },
        ];
        predictor.update(&later_routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20, 10]);
    }

    #[test]
    fn moe_route_telemetry_records_previous_rank_transition_matrix() {
        let mut telemetry = MoeRouteTelemetry::new(3);
        let previous_routes = [7, 11, 13];
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 0,
                expert_idx: 11,
                weight: 0.5,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 1,
                expert_idx: 7,
                weight: 0.25,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 2,
                expert_idx: 99,
                weight: 0.125,
            },
            &previous_routes,
        );

        assert_eq!(telemetry.observations_by_rank, vec![1, 1, 1]);
        assert_eq!(telemetry.repeated_previous_by_rank, vec![1, 1, 0]);
        assert_eq!(
            telemetry.repeated_previous_rank_by_current_rank,
            vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]]
        );
        assert_eq!(
            telemetry
                .to_json()
                .get("repeated_previous_rank_by_current_rank")
                .unwrap(),
            &serde_json::json!([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        );
        let json = telemetry.to_json();
        assert_eq!(
            json.get("repeated_previous_probability_by_current_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("same_rank_repeat_probability_by_rank").unwrap(),
            &serde_json::json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            json.get("repeated_current_by_previous_rank").unwrap(),
            &serde_json::json!([1, 1, 0])
        );
        assert_eq!(
            json.get("repeated_current_probability_by_previous_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("best_previous_rank_by_current_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_current_rank_by_previous_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_transition").unwrap(),
            &serde_json::json!({
                "current_rank": 0,
                "previous_rank": 1,
                "count": 1,
                "probability_by_current_rank": 1.0,
            })
        );
    }
}
