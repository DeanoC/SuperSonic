use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use supersonic_core::registry::{self, Backend, FamilyParams};

use crate::bakes::{
    ensure_gemma4_int4_bake_available, ensure_qwen35_bake_available, selected_bake_variant,
};
use crate::decode_engine::DecodeEngine;
use crate::dflash::{DFlashOptions, DFlashSession};
use crate::gemma4_engine::Gemma4Engine;
use crate::gemma4_int4_engine::Gemma4Int4Engine;
use crate::session::InferenceSession;
use crate::state::LoaderConfig;

pub(crate) fn build_qwen(
    cfg: &LoaderConfig,
    entry: &'static registry::RegistryEntry,
    context_tokens: usize,
) -> Result<(InferenceSession, Vec<u32>)> {
    if entry.model.architecture_family() != registry::ArchitectureFamily::QwenHybridDense {
        bail!(
            "build_qwen requires Qwen hybrid dense architecture (got {})",
            entry.model
        );
    }
    let mut params = match &entry.params {
        FamilyParams::Qwen35(p) => *p,
        FamilyParams::Qwen3Moe(_) => unreachable!("caller filtered to Qwen hybrid dense"),
        FamilyParams::Qwen36Moe(_) => unreachable!("caller filtered to Qwen hybrid dense"),
        FamilyParams::Gemma4(_) => unreachable!("caller filtered to Qwen hybrid dense"),
        FamilyParams::Phi4(_) => unreachable!("caller filtered to Qwen hybrid dense"),
        FamilyParams::Llama31(_) => unreachable!("caller filtered to Qwen hybrid dense"),
    };

    // INT4 decode lives in the 4B kernel; force-route 0.8B through it.
    if cfg.int4 && !params.use_4b_kernel && matches!(entry.backend, Backend::Hip) {
        params.use_4b_kernel = true;
    }
    let params = &params;

    let config = qwen35::config::load_config(&cfg.model_dir)
        .map_err(|e| anyhow!("loading config.json: {e}"))?;
    let text_config = config.text_config;
    let eos_ids = text_config.eos_token_ids();

    let t0 = Instant::now();
    let variant_bake = selected_bake_variant(cfg);
    let bake_dir = variant_bake.bake_dir(&cfg.model_dir);
    ensure_qwen35_bake_available(
        cfg,
        variant_bake,
        &bake_dir,
        params.weight_prefix,
        &text_config,
    )?;

    let store =
        model_store::BakedStore::open(&bake_dir).map_err(|e| anyhow!("open baked store: {e}"))?;
    let weights = qwen35::weights::Qwen35Weights::load_baked(
        &store,
        &text_config,
        cfg.device,
        params.weight_prefix,
    )
    .map_err(|e| anyhow!("load baked weights: {e}"))?;
    tracing::info!("weights loaded in {:.0}ms", t0.elapsed().as_millis());
    if cfg.dflash && !weights.is_int4 {
        bail!("--dflash target loader did not produce low-bit weights");
    }

    let attn_scratch_floats =
        params
            .attn_scratch_floats
            .max(qwen35::scratch::required_attn_scratch_floats(
                text_config.num_attention_heads,
                text_config.head_dim,
                context_tokens,
                params.kv_chunk_size,
            ));

    let mut engine = DecodeEngine::new(
        weights,
        cfg.device,
        params.proj_buf_floats,
        attn_scratch_floats,
        params.kv_chunk_size,
        params.use_4b_kernel,
        0, // prefill_chunk_size — 0 = no chunking; server handles one prompt at a time
        cfg.kv_fp8,
        1, // batch_size — serial model for v1
    )
    .with_context(|| "build Qwen3.5 DecodeEngine")?;
    engine.set_decode_context_limit(context_tokens);

    if cfg.dflash {
        let draft_dir = cfg
            .dflash_draft_dir
            .clone()
            .ok_or_else(|| anyhow!("--dflash requires --dflash-draft-dir"))?;
        let dflash = DFlashSession::new(
            engine,
            DFlashOptions {
                draft_dir,
                block: cfg.dflash_block,
                tap_layers: cfg.dflash_tap_layers.clone(),
            },
            &entry.model,
            &text_config,
            context_tokens,
            cfg.device,
        )?;
        Ok((InferenceSession::QwenDFlash(dflash), eos_ids))
    } else {
        Ok((InferenceSession::Qwen(engine), eos_ids))
    }
}

pub(crate) fn build_gemma4(
    cfg: &LoaderConfig,
    entry: &'static registry::RegistryEntry,
    context_tokens: usize,
) -> Result<(InferenceSession, Vec<u32>)> {
    if cfg.fp8_runtime || cfg.kv_fp8 {
        bail!("Gemma 4 does not yet support --fp8-runtime / --kv-fp8");
    }
    let params = match &entry.params {
        FamilyParams::Gemma4(p) => p,
        FamilyParams::Qwen35(_) => unreachable!("caller filtered to Gemma 4"),
        FamilyParams::Qwen3Moe(_) => unreachable!("caller filtered to Gemma 4"),
        FamilyParams::Qwen36Moe(_) => unreachable!("caller filtered to Gemma 4"),
        FamilyParams::Phi4(_) => unreachable!("caller filtered to Gemma 4"),
        FamilyParams::Llama31(_) => unreachable!("caller filtered to Gemma 4"),
    };

    let g_cfg = gemma4::config::load_config(&cfg.model_dir)
        .map_err(|e| anyhow!("loading Gemma 4 config.json: {e}"))?;
    let eos_ids = g_cfg.text_config.eos_token_ids();

    let t0 = Instant::now();
    let session = if cfg.int4 {
        let target = model_store::fetch::BakeVariant::Int4Gptq.bake_dir(&cfg.model_dir);
        ensure_gemma4_int4_bake_available(cfg, &target)?;
        let engine = Gemma4Int4Engine::load_with_batch(
            &cfg.model_dir,
            params.weight_prefix,
            context_tokens,
            cfg.device,
            1, // batch_size — serial model for v1
        )?;
        InferenceSession::Gemma4Int4(engine)
    } else {
        let engine = Gemma4Engine::load_with_batch(
            &cfg.model_dir,
            params.weight_prefix,
            context_tokens,
            cfg.device,
            1,
        )?;
        InferenceSession::Gemma4Bf16(engine)
    };
    tracing::info!("weights loaded in {:.0}ms", t0.elapsed().as_millis());

    Ok((session, eos_ids))
}
