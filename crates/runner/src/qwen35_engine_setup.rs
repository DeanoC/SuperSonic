use anyhow::Result;

use crate::bakes::load_qwen38_weights;
use crate::decode_engine::DecodeEngine;
use crate::registry::{self, Qwen35KernelParams, RegistryEntry};
use crate::Cli;

pub(crate) struct Qwen35EngineSetup {
    pub(crate) engine: DecodeEngine,
}

pub(crate) fn install_qwen35_launch_preset(entry: &RegistryEntry) {
    let preset = registry::qwen35_4b_launch_preset(&entry.arch, &entry.model);
    let (blocks, cooperative) = preset.unwrap_or((0, false));
    kernel_ffi::set_qwen35_4b_launch_preset(blocks, cooperative);
    if let Some((blocks, cooperative)) = preset {
        eprintln!("[preset] qwen3.8 4B launch: blocks={blocks} cooperative={cooperative}");
    }
}

pub(crate) fn load_qwen35_engine(
    cli: &Cli,
    text_config: &qwen35::config::TextConfig,
    params: &Qwen35KernelParams,
    ordinal: usize,
    context_tokens: usize,
) -> Result<Qwen35EngineSetup> {
    let t0 = std::time::Instant::now();
    let weights = load_qwen38_weights(cli, text_config, ordinal)?;
    eprintln!(
        "[weights] GQH megakernel dequant ({} headers, 4-plane d/ratio/lo/hi)",
        weights.gqh_headers.len()
    );
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let required_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
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
        false,
        1,
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

    Ok(Qwen35EngineSetup { engine })
}
