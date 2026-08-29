use anyhow::Result;

use crate::bakes::load_qwen38_weights;
use crate::decode_engine::DecodeEngine;
use crate::registry::Qwen38KernelParams;
use crate::Cli;

pub(crate) struct Qwen38EngineSetup {
    pub(crate) engine: DecodeEngine,
    pub(crate) dflash: Option<supersonic_runtime::dflash_spec::DflashSpecDecoder>,
}

pub(crate) fn load_qwen38_engine(
    cli: &Cli,
    text_config: &qwen38::config::TextConfig,
    params: &Qwen38KernelParams,
    ordinal: usize,
    context_tokens: usize,
) -> Result<Qwen38EngineSetup> {
    let t0 = std::time::Instant::now();
    let weights = load_qwen38_weights(cli, text_config, ordinal)?;
    eprintln!(
        "[weights] GQH megakernel dequant ({} headers, 4-plane d/ratio/lo/hi)",
        weights.gqh_headers.len()
    );
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let required_attn_scratch = qwen38::scratch::required_attn_scratch_floats(
        text_config.num_attention_heads,
        text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );
    let attn_scratch_floats = params.attn_scratch_floats.max(required_attn_scratch);
    if attn_scratch_floats > params.attn_scratch_floats {
        eprintln!(
            "[scratch] context={} → attn_scratch_floats={} (registry floor {})",
            context_tokens, attn_scratch_floats, params.attn_scratch_floats
        );
    }

    let mut engine = DecodeEngine::new(
        weights,
        ordinal,
        params.proj_buf_floats,
        attn_scratch_floats,
        params.kv_chunk_size,
        params.use_4b_kernel,
        cli.prefill_chunk_size,
    )?;
    engine.set_decode_context_limit(context_tokens);
    if cli.speculative_decode {
        if engine.weights().mtp.is_none() {
            anyhow::bail!("--speculative-decode on qwen3.8-27b needs NextN blk.64 in the GQH GGUF");
        }
        engine.set_mtp_spec(true);
        eprintln!("[qwen38-mtp] NextN spec on (K-draft + short-block verify; greedy-identical)");
    } else if engine.weights().mtp.is_some() {
        eprintln!("[qwen38-mtp] loaded NextN blk.64 (pass --speculative-decode to enable)");
    }

    // DFlash2 draft model: load + upload + create spec decoder.
    let dflash = if let Some(draft_path) = cli.draft_gguf_file.as_deref() {
        Some(load_dflash_decoder(
            draft_path,
            ordinal,
            context_tokens,
            text_config.hidden_size,
            text_config.num_hidden_layers,
            text_config,
        )?)
    } else {
        None
    };
    if dflash.is_some() {
        eprintln!("[dflash] draft model loaded; speculative decode enabled");
    }

    Ok(Qwen38EngineSetup { engine, dflash })
}

fn load_dflash_decoder(
    draft_path: &std::path::Path,
    ordinal: usize,
    max_ctx: usize,
    hidden: usize,
    target_layer_count: usize,
    target_config: &qwen38::config::TextConfig,
) -> anyhow::Result<supersonic_runtime::dflash_spec::DflashSpecDecoder> {
    let t0 = std::time::Instant::now();
    let draft_weights = model_store::dflash::load_draft(draft_path)
        .map_err(|e| anyhow::anyhow!("load draft GGUF {draft_path:?}: {e}"))?;
    eprintln!(
        "[dflash] draft config: hidden={} layers={} block={} mask_token={}",
        draft_weights.config.hidden,
        draft_weights.config.n_layers,
        draft_weights.config.block_size,
        draft_weights.config.mask_token_id,
    );
    let draft_gpu = draft_weights
        .upload(ordinal)
        .map_err(|e| anyhow::anyhow!("upload draft to GPU: {e}"))?;
    eprintln!(
        "[dflash] draft uploaded in {:.0}ms",
        t0.elapsed().as_millis()
    );
    let decoder = supersonic_runtime::dflash_spec::DflashSpecDecoder::new(
        draft_gpu,
        ordinal,
        max_ctx,
        hidden,
        target_layer_count,
        target_config,
    )
    .map_err(|e| anyhow::anyhow!("create dflash spec decoder: {e}"))?;
    eprintln!("[dflash] active verify block={}", decoder.block_size());
    Ok(decoder)
}
