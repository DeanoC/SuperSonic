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

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use model_store::manifest::LayoutTag;
use model_store::BakedStore;
use qwen36_moe::config::{Config, TextConfig};
use qwen36_moe::loader::{ScalarKind, WeightLoader};
use qwen36_moe::state::{StateAccount, StateLayout};
use qwen36_moe::weights::{
    checkpoint_dtype_acceptable, expected_tensor_specs, CheckpointAccount, CheckpointDtype,
};

use crate::qwen36_moe_decode::{
    argmax_bf16_logits, dequant_int4_to_bf16_bytes, host_final_norm_lm_head, run_chained_decode,
    run_chained_decode_fast, sample_bf16_logits, AttnLayerBuffers, FfnInt4Sidecars,
    FfnLayerBuffers, FullAttnInt4Sidecars, FullAttnKvCache, LayerBuffers,
    LinearAttnInt4Sidecars, MtpLayerBuffers, MultiLayerGeom, XorshiftRng,
};
use crate::qwen36_moe_speculative::run_speculative_decode_step;
use crate::registry::{FamilyParams, Qwen36MoeKernelParams, RegistryEntry};

const GIB: f64 = (1024 * 1024 * 1024) as f64;
const MIB: f64 = (1024 * 1024) as f64;

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

    let layout = StateLayout::new(context_size, batch_size, kv_fp8);
    let state = StateAccount::from_config(&config.text_config, layout);

    let bake = inspect_bake(model_dir, &config.text_config, weight_prefix);

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

    let kv_bytes_per_token =
        config.text_config.kv_bytes_per_token(if kv_fp8 { 1 } else { 2 });
    let registry_budget_bytes = entry
        .vram
        .estimate_total(context_size, kv_bytes_per_token);

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
) -> Option<BakeAccount> {
    let bake_dir = model_store::bake_dir_int4(model_dir);
    if !bake_dir.exists() {
        return None;
    }
    // Re-parse the manifest header even though BakedStore does too — we
    // expose format/converter versions in the report so a stale bake is
    // visible without having to dig into the file by hand.
    let manifest_text =
        std::fs::read_to_string(model_store::manifest_path(&bake_dir)).ok()?;
    let manifest: model_store::manifest::Manifest =
        serde_json::from_str(&manifest_text).ok()?;
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

fn sanity_check_kernel_params(
    config: &TextConfig,
    params: &Qwen36MoeKernelParams,
) -> Result<()> {
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
        let bake_ok = bake.missing_specs.is_empty()
            && bake.bad_expert_tensors.is_empty();
        println!(
            "  ready-for-decode: {}",
            if bake_ok { "YES" } else { "NO" }
        );
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
        println!(
            "  decode session will be larger; pass --context-size <N> (or --prompt) for an",
        );
        println!(
            "  honest fit number. Worst case at the model's max ({} tokens) needs",
            max_pos,
        );
        // Quick worst-case KV estimate at model max — gives the user one
        // concrete data point before they have to re-run with a flag.
        let worst_kv = report.config.text_config.kv_bytes_per_token(2)
            * max_pos as u64;
        println!(
            "  ~{:.2} GiB of KV cache alone (BF16).",
            worst_kv as f64 / GIB,
        );
    }
}

pub fn run(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    total_vram: u64,
) -> Result<()> {
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
        (cli.prompt.chars().count() + max_new, ContextSizeSource::EstimatedFromPrompt)
    } else {
        (max_new, ContextSizeSource::MaxNewTokensOnly)
    };
    let report = run_qwen36_moe_dry_run(
        &cli.model_dir,
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
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

    // Auto-download the INT4 bake from the GitHub release if missing or
    // stale. The qwen3.6-MoE engine had been missing this wiring — an
    // oversight visible during Phase 6 bring-up: 35B-A3B INT4
    // calibration OOMs on 24 GiB hosts, so release-hosted bakes are
    // the only realistic way to ship updates (e.g. the post-#84 bake
    // that includes mtp.* tensors for self-speculative decode). Mirrors
    // the Phi-4 / Llama-3.1 / Qwen3.5 paths.
    {
        let variant = model_store::fetch::BakeVariant::Int4Gptq;
        let bake_dir = variant.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
        // `should_fetch_exact_bake` honors --download-bake (force) and
        // refuses to fetch when an up-to-date bake is already present.
        let force_download = cli.download_bake;
        if !cli.no_download
            && crate::should_fetch_exact_bake(force_download, model_store::version_ok(&bake_dir))
        {
            let canonical_model = entry.model.to_string();
            match crate::try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => eprintln!(
                    "[fetch] installed qwen3.6-MoE INT4 bake at {}",
                    bake_dir.display()
                ),
                Ok(false) => {}
                Err(e) => eprintln!("[fetch] qwen3.6-MoE INT4 bake fetch failed: {e}"),
            }
        }
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
    )?;
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

/// Open a BakedStore from the bake dir, loading one tensor by name to a
/// fresh GpuBuffer. The wrapper exists to attach a useful context message
/// when a tensor is missing (the bake-validation in `inspect_bake` already
/// runs as part of the dry-run, so a missing tensor here is a real bug).
fn load_to_gpu(store: &BakedStore, ordinal: usize, name: &str) -> Result<GpuBuffer> {
    let resolved = resolve_qwen36_store_name(store, name);
    store
        .load_to_gpu(resolved.as_ref(), ordinal)
        .with_context(|| format!("BakedStore::load_to_gpu({name})"))
}

fn store_contains_qwen36(store: &BakedStore, name: &str) -> bool {
    store.contains(resolve_qwen36_store_name(store, name).as_ref())
}

fn store_layout_qwen36<'a>(store: &'a BakedStore, name: &str) -> Option<&'a LayoutTag> {
    store.layout(resolve_qwen36_store_name(store, name).as_ref())
}

fn resolve_qwen36_store_name<'a>(store: &BakedStore, name: &'a str) -> Cow<'a, str> {
    if store.contains(name) {
        return Cow::Borrowed(name);
    }
    if name.contains(".mlp.experts.") {
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if store.contains(&alt) {
                return Cow::Owned(alt);
            }
        }
    }
    Cow::Borrowed(name)
}

/// Pinned by `oracle/bake_int4.py` and the kernel — every INT4 tensor in
/// the bake quantizes at this group size. Detected per-tensor via a
/// `*_int4_scale` sidecar; if any quantizable tensor is present and
/// uses a different group_size we'd surface that as an error.
const QWEN36_MOE_INT4_GROUP_SIZE: i32 = 128;

/// Build one layer's worth of GPU-resident weight + state buffers from a
/// BakedStore. Decides full-attn vs linear-attn by consulting the config's
/// `layer_types` (every 4th layer is full per the standard hybrid pattern).
/// `int4_enabled` controls whether the weight tensors are loaded as packed
/// nibbles + sidecars or as BF16. The bake naming convention pairs
/// `<name>.weight` (packed) with `<name>.weight_int4_scale`/`_int4_zero`
/// for dense projections, and `<name>` (packed) with `<name>_int4_scale`/
/// `_int4_zero` for fused-expert tensors (no `.weight` suffix in the
/// HuggingFace checkpoint).
fn load_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    int4_enabled: bool,
    // When > 0, allocate a KV cache for full-attention layers sized for
    // `kv_max_t` past tokens. Linear-attn layers use `conv_state` +
    // `recurrent_state` instead (always allocated). 0 = no KV cache,
    // kernel falls back to kv_len=1 (back-compat for the parity test).
    kv_max_t: usize,
) -> Result<LayerBuffers> {
    let lp = format!("{weight_prefix}.layers.{layer_idx}");

    let attn = if text_config.is_full_attention(layer_idx) {
        let fa = format!("{lp}.self_attn");
        let int4 = if int4_enabled {
            Some(FullAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                q_proj_scale: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight_int4_scale"))?,
                q_proj_zero:  load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight_int4_zero"))?,
                k_proj_scale: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight_int4_scale"))?,
                k_proj_zero:  load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight_int4_zero"))?,
                v_proj_scale: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight_int4_scale"))?,
                v_proj_zero:  load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight_int4_zero"))?,
                o_proj_scale: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight_int4_scale"))?,
                o_proj_zero:  load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight_int4_zero"))?,
            })
        } else {
            None
        };
        // KV cache: allocate per-layer when multi-token decode is requested.
        // Layout: BF16 [kv_max_t, num_kv_heads * head_dim] for both K and V.
        let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
        let kv_cache = if kv_max_t > 0 {
            let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                .with_context(|| format!("alloc kv_cache_k (layer {layer_idx})"))?;
            let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                .with_context(|| format!("alloc kv_cache_v (layer {layer_idx})"))?;
            Some(FullAttnKvCache {
                k,
                v,
                kv_max_t: kv_max_t as i32,
            })
        } else {
            None
        };
        AttnLayerBuffers::Full {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            q_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight"))?,
            k_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight"))?,
            v_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight"))?,
            q_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.q_norm.weight"))?,
            k_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.k_norm.weight"))?,
            o_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight"))?,
            int4,
            kv_cache,
        }
    } else {
        let la = format!("{lp}.linear_attn");
        let kernel = geom.conv_kernel_dim as usize;
        let key_dim = (geom.num_k_heads as usize) * (geom.head_k_dim as usize);
        let val_dim = (geom.num_v_heads as usize) * (geom.head_v_dim as usize);
        let qkv_dim = 2 * key_dim + val_dim;
        let state_elems =
            (geom.num_v_heads as usize) * (geom.head_k_dim as usize) * (geom.head_v_dim as usize);

        // First-decode-token state: conv + recurrent both zeros. The kernel
        // mutates them in place; PR 4d will persist them across decode steps.
        let conv_state = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kernel - 1])
            .with_context(|| format!("alloc conv_state (layer {layer_idx})"))?;
        let recurrent_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[state_elems])
            .with_context(|| format!("alloc recurrent_state (layer {layer_idx})"))?;

        let int4 = if int4_enabled {
            Some(LinearAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                in_proj_qkv_scale: load_to_gpu(store, ordinal, &format!("{la}.in_proj_qkv.weight_int4_scale"))?,
                in_proj_qkv_zero:  load_to_gpu(store, ordinal, &format!("{la}.in_proj_qkv.weight_int4_zero"))?,
                in_proj_z_scale:   load_to_gpu(store, ordinal, &format!("{la}.in_proj_z.weight_int4_scale"))?,
                in_proj_z_zero:    load_to_gpu(store, ordinal, &format!("{la}.in_proj_z.weight_int4_zero"))?,
                out_proj_scale:    load_to_gpu(store, ordinal, &format!("{la}.out_proj.weight_int4_scale"))?,
                out_proj_zero:     load_to_gpu(store, ordinal, &format!("{la}.out_proj.weight_int4_zero"))?,
            })
        } else {
            None
        };

        AttnLayerBuffers::Linear {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            in_proj_qkv_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_qkv.weight"))?,
            in_proj_z_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_z.weight"))?,
            in_proj_a_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_a.weight"))?,
            in_proj_b_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_b.weight"))?,
            conv1d_w: load_to_gpu(store, ordinal, &format!("{la}.conv1d.weight"))?,
            // conv1d.bias may be absent — match the loader's behaviour.
            conv1d_bias: store
                .contains(&format!("{la}.conv1d.bias"))
                .then(|| load_to_gpu(store, ordinal, &format!("{la}.conv1d.bias")))
                .transpose()?,
            dt_bias: load_to_gpu(store, ordinal, &format!("{la}.dt_bias"))?,
            a_log: load_to_gpu(store, ordinal, &format!("{la}.A_log"))?,
            norm_w: load_to_gpu(store, ordinal, &format!("{la}.norm.weight"))?,
            out_proj_w: load_to_gpu(store, ordinal, &format!("{la}.out_proj.weight"))?,
            conv_state,
            recurrent_state,
            int4,
        }
    };

    let mp = format!("{lp}.mlp");
    // Fused-expert sidecars use `_int4_scale`/`_int4_zero` (no `.weight`).
    // Shared-expert MLPs use the dense `<name>.weight_int4_scale` form.
    let ffn_int4 = if int4_enabled {
        Some(FfnInt4Sidecars {
            group_size: QWEN36_MOE_INT4_GROUP_SIZE,
            gate_up_proj_scale: load_to_gpu(store, ordinal, &format!("{mp}.experts.gate_up_proj_int4_scale"))?,
            gate_up_proj_zero:  load_to_gpu(store, ordinal, &format!("{mp}.experts.gate_up_proj_int4_zero"))?,
            down_proj_scale:    load_to_gpu(store, ordinal, &format!("{mp}.experts.down_proj_int4_scale"))?,
            down_proj_zero:     load_to_gpu(store, ordinal, &format!("{mp}.experts.down_proj_int4_zero"))?,
            shared_gate_proj_scale: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.gate_proj.weight_int4_scale"))?,
            shared_gate_proj_zero:  load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.gate_proj.weight_int4_zero"))?,
            shared_up_proj_scale:   load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.up_proj.weight_int4_scale"))?,
            shared_up_proj_zero:    load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.up_proj.weight_int4_zero"))?,
            shared_down_proj_scale: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.down_proj.weight_int4_scale"))?,
            shared_down_proj_zero:  load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.down_proj.weight_int4_zero"))?,
        })
    } else {
        None
    };
    let ffn = FfnLayerBuffers {
        post_attn_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.post_attention_layernorm.weight"))?,
        gate_w: load_to_gpu(store, ordinal, &format!("{mp}.gate.weight"))?,
        // Note: experts.gate_up_proj / experts.down_proj have NO `.weight`
        // suffix in the published checkpoint — see expected_tensor_specs.
        gate_up_proj_w: load_to_gpu(store, ordinal, &format!("{mp}.experts.gate_up_proj"))?,
        down_proj_w: load_to_gpu(store, ordinal, &format!("{mp}.experts.down_proj"))?,
        shared_gate_proj_w: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.gate_proj.weight"))?,
        shared_up_proj_w: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.up_proj.weight"))?,
        shared_down_proj_w: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert.down_proj.weight"))?,
        shared_expert_gate_w: load_to_gpu(store, ordinal, &format!("{mp}.shared_expert_gate.weight"))?,
        int4: ffn_int4,
    };

    Ok(LayerBuffers { attn, ffn })
}

/// Load the Qwen3.6-MoE multi-token-prediction (MTP) head from the bake.
/// Used by Phase 6 self-speculative decode (`oracle/qwen36_moe_mtp_oracle.py`
/// is the bit-exact PyTorch reference).
///
/// Returns `Ok(Some(buffers))` when the bake has all 19 `mtp.*` tensors,
/// `Ok(None)` when the bake is pre-PR-#84 and lacks them (production
/// decode is unaffected; only the speculative-decode path is unavailable),
/// or `Err(...)` when the bake is partially-MTP (anomaly — some tensors
/// present, others missing — should never happen in the wild).
///
/// `kv_max_t > 0` allocates a per-layer KV cache for the MTP block —
/// separate from the base layers' KV caches per the vLLM reference.
fn load_mtp_buffers(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    kv_max_t: usize,
) -> Result<Option<MtpLayerBuffers>> {
    // The 19 names we expect. If `mtp.fc.weight` is absent we treat the
    // entire MTP block as missing; if it's present we require all 19.
    let probe = "mtp.fc.weight";
    if !store.contains(probe) {
        return Ok(None);
    }

    let required = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|n| !store.contains(n))
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "bake has `{probe}` but is missing {} of the other 18 mtp.* tensors \
             (e.g. {}); refusing to load a partial MTP block. Re-bake against \
             `oracle/bake_int4.py` (see GitHub issue #87 for the producer \
             workflow).",
            missing.len(),
            missing.first().copied().unwrap_or("<none>")
        ));
    }

    let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
    let kv_cache = if kv_max_t > 0 {
        let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
            .context("alloc mtp kv_cache_k")?;
        let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
            .context("alloc mtp kv_cache_v")?;
        Some(FullAttnKvCache {
            k,
            v,
            kv_max_t: kv_max_t as i32,
        })
    } else {
        None
    };

    Ok(Some(MtpLayerBuffers {
        pre_fc_norm_hidden_w: load_to_gpu(store, ordinal, "mtp.pre_fc_norm_hidden.weight")?,
        pre_fc_norm_embedding_w: load_to_gpu(store, ordinal, "mtp.pre_fc_norm_embedding.weight")?,
        fc_w: load_to_gpu(store, ordinal, "mtp.fc.weight")?,
        norm_w: load_to_gpu(store, ordinal, "mtp.norm.weight")?,
        input_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.input_layernorm.weight")?,
        post_attn_norm_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?,
        q_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.q_proj.weight")?,
        k_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.k_proj.weight")?,
        v_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.v_proj.weight")?,
        o_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.o_proj.weight")?,
        q_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.q_norm.weight")?,
        k_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.k_norm.weight")?,
        gate_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.gate.weight")?,
        gate_up_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.experts.gate_up_proj")?,
        down_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.experts.down_proj")?,
        shared_gate_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        )?,
        shared_up_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        )?,
        shared_down_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        )?,
        shared_expert_gate_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert_gate.weight",
        )?,
        kv_cache,
    }))
}

/// Look up one row of the embedding table on the host. For the
/// "first decode token" smoke path we use token 0 (or BOS if defined) so
/// the path runs end-to-end without a tokenizer. The full embed_tokens
/// tensor is BF16 `[vocab, hidden]`; we slice the first `hidden*2` bytes
/// out of its mmap-backed raw payload to avoid a full GPU upload of the
/// embedding table.
fn lookup_embed_row(
    store: &BakedStore,
    weight_prefix: &str,
    token_id: usize,
    hidden: usize,
) -> Result<Vec<u8>> {
    let name = format!("{weight_prefix}.embed_tokens.weight");
    let bytes = store
        .raw_bytes(&name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    let row_bytes = hidden * 2;
    let start = token_id * row_bytes;
    let end = start + row_bytes;
    if end > bytes.len() {
        return Err(anyhow!(
            "embed_tokens row {token_id} out of bounds (need {end} bytes, have {})",
            bytes.len()
        ));
    }
    Ok(bytes[start..end].to_vec())
}

/// Pull the host-side bytes for a small tensor (final norm, lm_head). For
/// lm_head with vocab=248K × hidden=2048 BF16 that's ~1 GiB — host RAM is
/// fine but a future revision should run lm_head on the GPU (PR 4d).
fn host_load_bytes(store: &BakedStore, name: &str) -> Result<Vec<u8>> {
    let raw = store
        .raw_bytes(name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    Ok(raw.to_vec())
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
        let is_greedy = sampling.temperature <= 0.0
            || sampling.top_k == 1;
        if !is_greedy {
            anyhow::bail!(
                "--speculative-decode currently supports greedy sampling \
                 only (temperature ≤ 0 or top_k == 1). Got temperature={}, \
                 top_k={}, top_p={}. Phase 6.4 will add sampling-consistent \
                 verification (rejection sampling); until then, re-run with \
                 `--temperature 0` for speculative decode, or drop \
                 `--speculative-decode` for non-greedy sampling.",
                sampling.temperature, sampling.top_k, sampling.top_p
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
            let enc = tok.encode(prompt, true)
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

    // Pick the bake. INT4 is the realistic path on 24 GiB VRAM.
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, int4_enabled) = if int4_dir.exists() {
        (int4_dir, true)
    } else if bf16_dir.exists() {
        (bf16_dir, false)
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
        if int4_enabled { "INT4 GPTQ" } else { "BF16" }
    );
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    // KV cache size: needs to fit prompt_len + max_new past tokens. Sized
    // generously here since per-layer KV is small (10 full-attn layers ×
    // [kv_max_t, Hkv*d=512] BF16 = 10 KiB per token of context).
    let kv_max_t = prompt_ids.len() + max_new;

    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    println!(
        "  loading {} layers ({} INT4 sidecar sets, KV cache cap = {} tokens)…",
        geom.num_layers,
        if int4_enabled { geom.num_layers } else { 0 },
        kv_max_t,
    );
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            &store,
            ordinal,
            li,
            &geom,
            &report.config.text_config,
            weight_prefix,
            int4_enabled,
            kv_max_t,
        )
        .with_context(|| format!("load layer {li} weights"))?;
        layers.push(layer);
    }

    // Phase 6 self-speculative decode: load the multi-token-prediction
    // (MTP) head from the bake when --speculative-decode is set.
    //
    // The buffers cost ~1.6 GiB BF16 + a per-MTP-layer KV cache, which
    // matters on memory-tight 24 GiB configs (the 17 GiB INT4 base bake
    // already fills most of VRAM; an extra 1.6 GiB unused buffer can
    // tip larger context_size / max_new runs into OOM). So we only load
    // when the user opts in via `--speculative-decode`. When the flag
    // isn't set, MTP weights are skipped entirely — production decode
    // gets the full VRAM headroom.
    //
    // The loader still returns Ok(None) gracefully if the bake is pre-
    // PR-#84 and lacks mtp.* tensors. Erroring out loudly when the user
    // explicitly asked for speculative decode but the bake can't support
    // it is the right move — silent fallback to non-speculative would
    // hide the bake-staleness from a downstream perf-sensitive user.
    //
    // Phase 6.2c+ wires the consumer side into the decode loop.
    let mtp_buffers_opt = if speculative_decode {
        match load_mtp_buffers(&store, ordinal, &geom, kv_max_t)
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

    // Upload final_norm + dequantized lm_head to the GPU once. The
    // GPU lm_head kernel (`qwen36_moe::lm_head_launch`) does
    // RMSNorm + GEMV in BF16 in one shot, replacing the previous
    // host-side F32 GEMV that dominated per-token wall-clock at
    // ~233 ms / 360 ms total on 35B-A3B greedy decode (PR #68
    // stage-timings).
    //
    // VRAM cost: ~970 MiB for lm_head BF16 + 4 KiB for final_norm +
    // 500 KiB for the logits buffer. Comfortably within the 24 GiB
    // budget alongside the 17 GiB INT4 weight bake.
    let final_norm_bytes = host_load_bytes(&store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    let lm_head_bf16_bytes = load_lm_head_bf16(&store, &report.config.text_config, weight_prefix, &geom)
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
    let mut logits_buf = GpuBuffer::zeros(
        ordinal,
        ScalarType::BF16,
        &[geom.vocab as usize],
    )
    .context("alloc logits_buf on GPU")?;
    let mut counter_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .context("alloc lm_head counter_buf on GPU")?;
    let mut final_hidden_buf =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.hidden as usize])
            .context("alloc final_hidden_buf on GPU")?;
    drop(lm_head_bf16_bytes);
    drop(final_norm_bytes);

    // Phase 6.3d speculative-decode setup. When `--speculative-decode` is
    // set, allocate the MTP forward + chain scratches and upload the
    // base model's `embed_tokens.weight` to GPU (the MTP draft chain
    // does GPU-side embed gathering via D2D copy from this buffer).
    //
    // The hardcoded num_speculative_tokens=3 mirrors the public Qwen3.6
    // MTP card's recommendation; making it CLI-tunable is a follow-up.
    // Memory: ~970 MiB embed_tokens BF16 + ~50 MiB MTP scratches.
    // Combined with the 17 GiB INT4 base bake + 970 MiB lm_head + 1.6
    // GiB MTP head, this lands at ~21 GiB GPU resident — within budget
    // on 24 GiB 7900 XTX with headroom for the per-token activation
    // buffers.
    const NUM_SPECULATIVE_TOKENS: usize = 3;
    let mut mtp_buffers = mtp_buffers_opt;
    let mut mtp_forward_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_forward_scratch(ordinal, &geom, kv_max_t)
                .context("alloc MTP forward scratch")?,
        )
    } else {
        None
    };
    let mut mtp_chain_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_chain_scratch(ordinal, &geom)
                .context("alloc MTP chain scratch")?,
        )
    } else {
        None
    };
    let embed_w_buf = if mtp_buffers.is_some() {
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let embed = load_to_gpu(&store, ordinal, &embed_name)
            .with_context(|| format!("upload {embed_name} to GPU"))?;
        println!(
            "  uploaded embed_tokens ({:.0} MiB BF16) and allocated MTP \
             scratches (K={NUM_SPECULATIVE_TOKENS} drafts/step)",
            (geom.vocab as f64 * geom.hidden as f64 * 2.0) / MIB
        );
        Some(embed)
    } else {
        None
    };

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
    let mut gen_steps: usize = 0;
    let mut t_embed = std::time::Duration::ZERO;
    let mut t_chain = std::time::Duration::ZERO;
    let mut t_lm_head = std::time::Duration::ZERO;
    let mut t_sample = std::time::Duration::ZERO;
    let mut t_detok = std::time::Duration::ZERO;
    // Within-chain breakdown by kernel class (microseconds).
    let mut t_chain_full_attn_us: u64 = 0;
    let mut t_chain_linear_attn_us: u64 = 0;
    let mut t_chain_ffn_us: u64 = 0;

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
        let outputs = run_chained_decode_fast(
            ordinal, &geom, &mut layers, &initial_hidden, position,
            emit_stage_timings,
        )
            .with_context(|| format!("chained decode (step {step}, position {position})"))?;
        let t_chain_step = t1.elapsed();
        position += 1;

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

        // Generation step: final_norm + lm_head GEMV on the GPU
        // (`kernel_ffi::qwen36_moe::lm_head_launch`). H2D the freshly
        // produced final_hidden into the persistent `final_hidden_buf`
        // — `run_chained_decode` already D2H'd it for diagnostics; the
        // re-upload is ~4 KB, microseconds at PCIe Gen4 speeds — then
        // launch the kernel + D2H the BF16 logits.
        let t2 = std::time::Instant::now();
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
        let logits = logits_buf
            .to_host_bytes()
            .context("d2h logits from GPU lm_head")?;
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

        gen_steps += 1;
        t_embed += t_embed_step;
        t_chain += t_chain_step;
        t_lm_head += t_lm_head_step;
        t_sample += t_sample_step;
        t_detok += t_detok_step;
        t_chain_full_attn_us += outputs.kernel_full_attn_us;
        t_chain_linear_attn_us += outputs.kernel_linear_attn_us;
        t_chain_ffn_us += outputs.kernel_ffn_us;

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
            let dynamic_k = NUM_SPECULATIVE_TOKENS.min(headroom.saturating_sub(1));
            let h_base = outputs.final_hidden_bytes.clone();
            // P2: thread spec-verify timings into the engine-level
            // accumulators so `--emit-stage-timings` reports honest
            // per-token costs under speculative decode. Without these
            // captures, every base step inside the verify loop would
            // be invisible to `gen_steps` / `t_chain` / `t_lm_head`,
            // making the speculative path look ~K+1× faster than it
            // really is on stage-timings dashboards.
            let result = run_speculative_decode_step(
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
                    let t_chain_start = std::time::Instant::now();
                    let initial_hidden = lookup_embed_row(
                        &store,
                        weight_prefix,
                        input as usize,
                        geom.hidden as usize,
                    )?;
                    let outputs = run_chained_decode_fast(
                        ordinal,
                        &geom,
                        &mut layers,
                        &initial_hidden,
                        pos,
                        emit_stage_timings,
                    )?;
                    t_chain += t_chain_start.elapsed();
                    t_chain_full_attn_us += outputs.kernel_full_attn_us;
                    t_chain_linear_attn_us += outputs.kernel_linear_attn_us;
                    t_chain_ffn_us += outputs.kernel_ffn_us;

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
                    t_lm_head += t_lm_head_start.elapsed();
                    // Each verify base step counts as one decode step
                    // for the per-token average — emitted_tokens.len()
                    // tokens are committed per spec call, and
                    // closure-call-count == emitted_tokens.len() in
                    // both the partial-accept and full-accept (with
                    // bonus) cases. Bumping here is equivalent to
                    // "one closure call = one decode step worth of
                    // base work."
                    gen_steps += 1;
                    Ok((
                        argmax_bf16_logits(&logits_bytes),
                        outputs.final_hidden_bytes,
                    ))
                },
            )
            .context("speculative decode step")?;

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

    println!();
    println!();
    println!(
        "Generated {} token{} ({} prompt + {} new). EOS: {}.",
        generated_ids.len(),
        if generated_ids.len() == 1 { "" } else { "s" },
        prompt_ids.len(),
        generated_ids.len(),
        if eos_id.map(|e| generated_ids.last() == Some(&e)).unwrap_or(false) {
            "yes"
        } else {
            "no (max_new_tokens hit)"
        },
    );
    if !generated_ids.is_empty() {
        println!("  Generated ids: {generated_ids:?}");
    }
    if emit_stage_timings && gen_steps > 0 {
        let to_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let chain_ms = to_ms(t_chain);
        let embed_ms = to_ms(t_embed);
        let lm_head_ms = to_ms(t_lm_head);
        let sample_ms = to_ms(t_sample);
        let detok_ms = to_ms(t_detok);
        let total_ms = chain_ms + embed_ms + lm_head_ms + sample_ms + detok_ms;
        let n = gen_steps as f64;
        let full_attn_ms = (t_chain_full_attn_us as f64) / 1000.0;
        let linear_attn_ms = (t_chain_linear_attn_us as f64) / 1000.0;
        let ffn_ms = (t_chain_ffn_us as f64) / 1000.0;
        eprintln!(
            "[qwen36-moe stage-timings] gen_steps={gen_steps} \
             embed_ms_avg={:.3} chain_ms_avg={:.3} lm_head_ms_avg={:.3} \
             sample_ms_avg={:.3} detok_ms_avg={:.3} total_ms_avg={:.3} \
             (chain_total_ms={:.1} lm_head_total_ms={:.1})",
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
            "[qwen36-moe chain-breakdown] gen_steps={gen_steps} \
             full_attn_ms_avg={:.3} linear_attn_ms_avg={:.3} ffn_ms_avg={:.3} \
             (full_attn_total_ms={:.1} linear_attn_total_ms={:.1} ffn_total_ms={:.1})",
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
/// GEMV in `host_final_norm_lm_head` consumes. Handles the three layouts:
/// tied embeddings (BF16 from embed_tokens), standalone BF16, or INT4
/// GPTQ (packed nibbles + scale/zero sidecars). The dequant runs once per
/// process; the result is reused across all decode steps.
fn load_lm_head_bf16(
    store: &BakedStore,
    text_config: &TextConfig,
    weight_prefix: &str,
    geom: &MultiLayerGeom,
) -> Result<Vec<u8>> {
    let (lm_name, lm_packed) = if text_config.tie_word_embeddings {
        let n = format!("{weight_prefix}.embed_tokens.weight");
        let b = host_load_bytes(store, &n).context("load tied lm_head from embed_tokens")?;
        (n, b)
    } else {
        let n = "lm_head.weight";
        let b = host_load_bytes(store, n).context("load lm_head")?;
        (n.to_string(), b)
    };
    let scale_name = format!("{lm_name}_int4_scale");
    if store.contains(&scale_name) {
        let zero_name = format!("{lm_name}_int4_zero");
        let scale = host_load_bytes(store, &scale_name).context("load lm_head INT4 scale")?;
        let zero = host_load_bytes(store, &zero_name).context("load lm_head INT4 zero")?;
        let vocab = geom.vocab as usize;
        let hidden = geom.hidden as usize;
        println!(
            "  dequantizing lm_head INT4 [{vocab}, {hidden}] (≈{:.1} MiB → {:.1} MiB BF16)…",
            lm_packed.len() as f64 / MIB,
            (vocab * hidden * 2) as f64 / MIB,
        );
        Ok(dequant_int4_to_bf16_bytes(
            &lm_packed,
            &scale,
            &zero,
            vocab,
            hidden,
            QWEN36_MOE_INT4_GROUP_SIZE as usize,
        ))
    } else {
        Ok(lm_packed)
    }
}

/// Legacy single-token entry point — keeps the original `decode_first_token`
/// callable so the path stays exercised. Currently unused but documents the
/// minimal one-step decode shape.
#[allow(dead_code)]
fn decode_first_token(model_dir: &Path, report: &DryRunReport) -> Result<u32> {
    let weight_prefix = report.kernel_params.weight_prefix;

    // Pick the bake. INT4 is the realistic path on 24 GiB VRAM.
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, int4_enabled) = if int4_dir.exists() {
        (int4_dir, true)
    } else if bf16_dir.exists() {
        (bf16_dir, false)
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
        if int4_enabled { "INT4 GPTQ" } else { "BF16" }
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
        if int4_enabled { geom.num_layers } else { 0 },
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
            int4_enabled,
            0, // legacy single-token path: no KV cache, kv_len=1 fast path.
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
    println!("  embedding row {bos} loaded ({} BF16 bytes)", initial_hidden.len());

    println!("  running chained decode…");
    let outputs =
        run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, 0).context("chained decode")?;
    println!(
        "  decode done; final hidden norm = {:.4}",
        crate::qwen36_moe_decode::bf16_bytes_to_f32(&outputs.final_hidden_bytes)
            .iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    let final_norm_bytes = host_load_bytes(&store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;

    // lm_head: may be tied to embed_tokens (BF16), standalone BF16, or
    // INT4-quantized in the bake. Dequantize to BF16 host-side when needed
    // (kernel-side INT4 lm_head is PR 4d).
    let (lm_name, lm_packed) = if report.config.text_config.tie_word_embeddings {
        // Tied: lm_head bytes == embed_tokens bytes (BF16 either way; the
        // bake doesn't INT4 the embed table). Use the same blob.
        let n = format!("{weight_prefix}.embed_tokens.weight");
        let b = host_load_bytes(&store, &n).context("load tied lm_head from embed_tokens")?;
        (n, b)
    } else {
        let n = "lm_head.weight";
        let b = host_load_bytes(&store, n).context("load lm_head")?;
        (n.to_string(), b)
    };
    // INT4 detection: if a `*_int4_scale` sidecar exists for this name, the
    // packed payload above is u8 nibbles (in_dim folded by 2).
    let scale_name = format!("{lm_name}_int4_scale");
    let lm_head_bf16_bytes = if store.contains(&scale_name) {
        let zero_name = format!("{lm_name}_int4_zero");
        let scale = host_load_bytes(&store, &scale_name).context("load lm_head INT4 scale")?;
        let zero = host_load_bytes(&store, &zero_name).context("load lm_head INT4 zero")?;
        let vocab = geom.vocab as usize;
        let hidden = geom.hidden as usize;
        println!(
            "  dequantizing lm_head INT4 [{vocab}, {hidden}] (≈{:.1} MiB → {:.1} MiB BF16)…",
            lm_packed.len() as f64 / MIB,
            (vocab * hidden * 2) as f64 / MIB,
        );
        dequant_int4_to_bf16_bytes(
            &lm_packed,
            &scale,
            &zero,
            vocab,
            hidden,
            QWEN36_MOE_INT4_GROUP_SIZE as usize,
        )
    } else {
        lm_packed
    };

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
