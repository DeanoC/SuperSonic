use anyhow::Result;

use crate::bakes::{load_qwen35_weights, load_qwen35_weights_from_flm_source};
use crate::decode_engine::DecodeEngine;
use crate::flm_model_source::FlmModelSource;
use crate::registry::{self, Backend, GpuArch, ModelVariant, Qwen35KernelParams, RegistryEntry};
use crate::Cli;

pub(crate) struct Qwen35EngineSetup {
    pub(crate) engine: DecodeEngine,
    pub(crate) use_4b_kernel: bool,
    pub(crate) cuda_08b_hero_enabled: bool,
    pub(crate) allow_host_lm_head_rescore: bool,
}

pub(crate) fn install_qwen35_launch_preset(entry: &RegistryEntry) {
    // Install the per-(arch, model) HIP launch preset (grid size +
    // cooperative flag) if one is registered. User env vars still override
    // inside the bridge. Always called: `(0, false)` clears any stale
    // preset from a prior run, so switching models doesn't inherit the
    // previous one's grid. No-op on CUDA builds.
    let preset = registry::qwen35_4b_launch_preset(&entry.arch, &entry.model);
    let (blocks, coop) = preset.unwrap_or((0, false));
    kernel_ffi::set_qwen35_4b_launch_preset(blocks, coop);
    if let Some((blocks, coop)) = preset {
        eprintln!("[preset] qwen35_4b launch: blocks={blocks} cooperative={coop}");
    }
}

pub(crate) fn load_qwen35_engine(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    flm_source: Option<&FlmModelSource>,
    params: &Qwen35KernelParams,
    backend: Backend,
    gpu_arch: GpuArch,
    ordinal: usize,
    bootstrap_downloaded: bool,
    q4km_like: bool,
    context_tokens: usize,
) -> Result<Qwen35EngineSetup> {
    let t0 = std::time::Instant::now();
    let weights = if let Some(source) = flm_source {
        load_qwen35_weights_from_flm_source(
            cli,
            model_variant,
            text_config,
            ordinal,
            params.weight_prefix,
            q4km_like,
            source,
        )?
    } else {
        load_qwen35_weights(
            cli,
            model_variant,
            text_config,
            ordinal,
            params.weight_prefix,
            bootstrap_downloaded,
            q4km_like,
        )?
    };
    if weights.is_fp8 {
        eprintln!(
            "[weights] FP8 runtime dequant active (block_size={})",
            weights.fp8_block_size
        );
    }
    if weights.is_int4 {
        eprintln!(
            "[weights] INT4 runtime dequant active (group_size={})",
            weights.int4_group_size
        );
    }
    if !weights.gqh_headers.is_empty() {
        eprintln!(
            "[weights] GQH megakernel dequant ({} headers)",
            weights.gqh_headers.len()
        );
    }
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let cuda_08b_hero_disabled = std::env::var_os("SUPERSONIC_DISABLE_CUDA_08B_HERO").is_some();
    let cuda_08b_hero_candidate = backend == Backend::Cuda
        && gpu_arch == GpuArch::Sm86
        && *model_variant == ModelVariant::Qwen3_5_0_8B
        && cli.batch_size == 1
        && !cli.validate
        && !(cli.gpu_validate && cli.batch_size == 1)
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && !weights.is_fp8
        && !weights.is_int4
        && !cuda_08b_hero_disabled;
    let use_4b_kernel = params.use_4b_kernel && !cuda_08b_hero_candidate;

    if cli.batch_size > 1 && !use_4b_kernel {
        anyhow::bail!("--batch-size > 1 requires 4B kernel (2B/4B/9B models)");
    }
    if cli.batch_size < 1 || cli.batch_size > kernel_ffi::MAX_BATCH_SIZE {
        anyhow::bail!("--batch-size must be 1..{}", kernel_ffi::MAX_BATCH_SIZE);
    }

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
        use_4b_kernel,
        cli.prefill_chunk_size,
        cli.kv_fp8,
        cli.batch_size,
    )?;
    engine.set_decode_context_limit(context_tokens);
    let allow_host_lm_head_rescore = cli.no_bake
        && !engine.weights().is_fp8
        && !engine.weights().is_int4
        && engine.weights().gqh_headers.is_empty();

    Ok(Qwen35EngineSetup {
        engine,
        use_4b_kernel,
        cuda_08b_hero_enabled: cuda_08b_hero_candidate,
        allow_host_lm_head_rescore,
    })
}
