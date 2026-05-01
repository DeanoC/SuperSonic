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
use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use model_store::manifest::LayoutTag;
use model_store::BakedStore;
use qwen36_moe::config::{Config, TextConfig};
use qwen36_moe::loader::{ScalarKind, WeightLoader};
use qwen36_moe::state::{StateAccount, StateLayout};
use qwen36_moe::weights::{
    checkpoint_dtype_acceptable, expected_tensor_specs, CheckpointAccount, CheckpointDtype,
    DEFAULT_PREFIX,
};

use crate::qwen36_moe_decode::{
    argmax_bf16_logits, host_final_norm_lm_head, run_chained_decode, AttnLayerBuffers,
    FfnLayerBuffers, LayerBuffers, MultiLayerGeom,
};
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
        if !store.contains(&spec.name) {
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
            match store.layout(&name) {
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

    // Real decode path (PR 4c step 2). Uses the host-orchestrated chained
    // launches in `crate::qwen36_moe_decode::run_chained_decode` against
    // per-layer weight buffers loaded from the BF16 baked package. The
    // multi-layer parity test in `crates/runner/tests/qwen36_moe_multilayer_parity.rs`
    // gates the decode core against the Python multi-layer oracle (cos_sim
    // ≥ 0.999); this entry point is the same chain wired to the bake.
    //
    // Caveats for PR 4c step 2:
    //  - BF16 only. INT4 chained decode + KV-cache extension are PR 4d.
    //  - One token, fresh state. Conv + recurrent state start zeroed; the
    //    full-attn KV cache isn't allocated (single-block kernels run with
    //    `kv_len=1`). Multi-token generation needs prefill + state
    //    persistence which lands later.
    //  - Tokenizer not wired — the produced token is printed as a raw vocab
    //    id so the "doesn't bail" criterion is verifiable end-to-end.
    println!();
    println!("=== Decode (PR 4c step 2: host-orchestrated chained launches) ===");
    let token = decode_first_token(&cli.model_dir, &report)?;
    println!("First decoded token id: {token}");
    println!(
        "(Note: BF16 only, fresh state, raw vocab id. INT4 chained decode + \
         tokenizer + KV cache extension land in PR 4d.)"
    );
    Ok(())
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
    store
        .load_to_gpu(name, ordinal)
        .with_context(|| format!("BakedStore::load_to_gpu({name})"))
}

/// Build one layer's worth of GPU-resident weight + state buffers from a
/// BakedStore. Decides full-attn vs linear-attn by consulting the config's
/// `layer_types` (every 4th layer is full per the standard hybrid pattern).
fn load_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
) -> Result<LayerBuffers> {
    let lp = format!("{weight_prefix}.layers.{layer_idx}");

    let attn = if text_config.is_full_attention(layer_idx) {
        let fa = format!("{lp}.self_attn");
        AttnLayerBuffers::Full {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            q_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight"))?,
            k_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight"))?,
            v_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight"))?,
            q_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.q_norm.weight"))?,
            k_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.k_norm.weight"))?,
            o_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight"))?,
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
        }
    };

    let mp = format!("{lp}.mlp");
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
    };

    Ok(LayerBuffers { attn, ffn })
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

/// Run one decode step against a real bake. Returns the argmax token id
/// from the resulting logits.
///
/// **Untested on a real bake** — the 35B-A3B BF16 bake doesn't fit 24 GiB
/// VRAM, and we don't have a smaller-geometry production bake to validate
/// against. The decode core is parity-tested (cos_sim ≥ 0.999 vs Python
/// oracle) via `crates/runner/tests/qwen36_moe_multilayer_parity.rs`; if a
/// loadable bake materialises, this path uses the same chain wired to its
/// per-layer pointers, so a crash here is more likely a tensor-name typo
/// than an algorithmic bug.
fn decode_first_token(model_dir: &Path, report: &DryRunReport) -> Result<u32> {
    let weight_prefix = report.kernel_params.weight_prefix;
    let bake_dir = model_store::bake_dir(model_dir);
    if !bake_dir.exists() {
        return Err(anyhow!(
            "decode requires a BF16 baked package at {}; create one with the \
             standard bake pipeline or re-run with --dry-run for analytic accounting only. \
             (PR 4c step 2 wires BF16 only — INT4 chained decode is PR 4d.)",
            bake_dir.display()
        ));
    }
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            &store,
            ordinal,
            li,
            &geom,
            &report.config.text_config,
            weight_prefix,
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

    let outputs =
        run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, 0).context("chained decode")?;

    let final_norm_bytes = host_load_bytes(&store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    // `tie_word_embeddings`: lm_head shares bytes with embed_tokens. Either
    // way we read the [vocab, hidden] BF16 slab as a flat byte stream.
    let lm_head_bytes = if report.config.text_config.tie_word_embeddings {
        host_load_bytes(&store, &format!("{weight_prefix}.embed_tokens.weight"))
            .context("load lm_head (tied)")?
    } else {
        host_load_bytes(&store, "lm_head.weight").context("load lm_head")?
    };

    let logits = host_final_norm_lm_head(
        &outputs.final_hidden_bytes,
        &final_norm_bytes,
        &lm_head_bytes,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );
    Ok(argmax_bf16_logits(&logits))
}
