use std::path::Path;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{set_backend, Backend};
use model_store::{BakedStore, VirtualArenaTransferBackend};

use crate::qwen36_moe_cli::dry_run::DryRunReport;
use crate::qwen36_moe_cli::geom::build_multi_layer_geom;
use crate::qwen36_moe_cli::host::{host_load_bytes, load_lm_head_bf16, lookup_embed_row};
use crate::qwen36_moe_cli::layers::{load_layer_buffers, Qwen36WeightMode};
use crate::qwen36_moe_decode::run_chained_decode;
use crate::qwen36_moe_logits::{argmax_bf16_logits, host_final_norm_lm_head};

/// Legacy single-token entry point. Currently unused, but it documents and
/// preserves the minimal one-step decode shape.
#[allow(dead_code)]
pub(crate) fn decode_first_token(
    model_dir: &Path,
    report: &DryRunReport,
    kv_fp8: bool,
) -> Result<u32> {
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
            "decode requires a baked package - neither INT4-GPTQ ({}) nor \
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
        "  loading {} layer{} ({} INT4 sidecar set{})...",
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
            VirtualArenaTransferBackend::PageableH2d,
            &crate::qwen36_moe_cli::options::load_options_from_environment(),
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

    println!("  running chained decode...");
    let outputs = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, 0)
        .context("chained decode")?;
    println!(
        "  decode done; final hidden norm = {:.4}",
        crate::qwen36_moe_logits::bf16_bytes_to_f32(&outputs.final_hidden_bytes)
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

    println!("  computing host-side norm + lm_head GEMV...");
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
