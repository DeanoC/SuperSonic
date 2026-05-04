use anyhow::Result;
use std::env;
use std::path::Path;
use std::time::Instant;

use crate::bakes::{ensure_hf_metadata_present, load_qwen35_weights};
use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::model_files::model_dir_has_raw_safetensors;
use crate::prefill_engine::{self, PrefillResult};
use crate::qwen35_kv_trace::trace_kv_cache;
use crate::qwen35_trace::{
    trace_component_input_layer, trace_component_layer, trace_component_linear_layer,
    trace_component_linear_state_layer, trace_persistent_full_attn_layer,
    trace_persistent_input_layer, trace_persistent_linear_layer,
    trace_persistent_linear_state_layer,
};
use crate::qwen35_validation::{
    qwen35_oracle_script_path, resolve_qwen_oracle_model_id, run_qwen35_oracle_validation,
    trace_qwen35_oracle_prefill_layer, Qwen35NativePrefillTrace,
};
use crate::qwen35_vram::check_qwen35_vram;
use crate::registry::{
    self, Backend, FamilyParams, GpuArch, ModelVariant, Qwen35KernelParams, RegistryEntry,
};
use crate::tensor_bytes::bf16_bytes_to_f32 as decode_bf16_le;
use crate::{
    oracle, qwen35_dflash_engine, resolve_oracle_device, should_fetch_exact_bake,
    specprefill_engine, try_download_bake, validate, Cli,
};

pub(crate) struct Qwen35Startup {
    pub(crate) text_config: qwen35::config::TextConfig,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
}

pub(crate) struct Qwen35EngineSetup {
    pub(crate) engine: DecodeEngine,
    pub(crate) use_4b_kernel: bool,
    pub(crate) cuda_08b_hero_enabled: bool,
    pub(crate) allow_host_lm_head_rescore: bool,
}

pub(crate) struct Qwen35Policy {
    pub(crate) trace_kv_cache_enabled: bool,
}

pub(crate) struct Qwen35Prefill {
    pub(crate) logits: Vec<f32>,
    pub(crate) native_trace: Option<Qwen35NativePrefillTrace>,
    pub(crate) next_token: u32,
}

pub(crate) struct HostLmHeadRescorer {
    loader: qwen35::loader::WeightLoader,
    tensor_name: String,
}

impl HostLmHeadRescorer {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        if !model_dir_has_raw_safetensors(model_dir) {
            return Ok(None);
        }
        let loader = qwen35::loader::WeightLoader::from_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("open raw lm_head weights: {e}"))?;
        let tensor_name = if loader.contains("lm_head.weight") {
            "lm_head.weight".to_string()
        } else if loader.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight".to_string()
        } else {
            return Ok(None);
        };
        Ok(Some(Self {
            loader,
            tensor_name,
        }))
    }

    fn rescore(&self, normed: &[f32], candidate_ids: &[usize]) -> Result<u32> {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for &candidate in candidate_ids {
            let row = self
                .loader
                .load_bf16_row_f32(&self.tensor_name, candidate)
                .map_err(|e| anyhow::anyhow!("load lm_head row {candidate}: {e}"))?;
            anyhow::ensure!(
                row.len() == normed.len(),
                "lm_head row len {} != normed len {}",
                row.len(),
                normed.len()
            );
            let score = row
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            if score > best_val {
                best_val = score;
                best_idx = candidate;
            }
        }
        Ok(best_idx as u32)
    }
}

pub(crate) fn sample_qwen_logits_with_rescore(
    logits: &[f32],
    normed: Option<&[f32]>,
    rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let greedy = DecodeEngine::greedy_sample(logits);
    let Some(normed) = normed else {
        return Ok(greedy);
    };
    let Some(rescorer) = rescorer else {
        return Ok(greedy);
    };

    const RESCORE_MARGIN: f32 = 0.25;
    const RESCORE_TOP_K: usize = 4;

    let candidates = top_k_candidate_ids(logits, RESCORE_TOP_K);
    if candidates.len() < 2 {
        return Ok(greedy);
    }
    let top0 = logits[candidates[0]];
    let top1 = logits[candidates[1]];
    if top0 - top1 > RESCORE_MARGIN {
        return Ok(greedy);
    }

    let rescored = rescorer.rescore(normed, &candidates)?;
    if rescored != greedy {
        eprintln!(
            "[sample-rescore] token {} -> {} (top_margin={:.4})",
            greedy,
            rescored,
            top0 - top1
        );
    }
    Ok(rescored)
}

fn top_k_candidate_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(usize, f32)> = Vec::new();
    for (idx, &val) in logits.iter().enumerate() {
        let pos = best
            .iter()
            .position(|&(_, best_val)| val > best_val)
            .unwrap_or(best.len());
        if pos < k {
            best.insert(pos, (idx, val));
            if best.len() > k {
                best.pop();
            }
        }
    }
    best.into_iter().map(|(idx, _)| idx).collect()
}

pub(crate) fn run_qwen35_prefill(
    cli: &Cli,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    oracle_model_id: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
    allow_host_lm_head_rescore: bool,
) -> Result<Qwen35Prefill> {
    let prefill_start = Instant::now();
    if cli.oracle_prefill {
        let oracle_script = qwen35_oracle_script_path();
        let output = oracle::run_oracle(
            &oracle_script,
            oracle_model_id,
            prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            oracle_device,
            true,
            false,
            fp8_oracle_dir,
            None,
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids.first().copied().ok_or_else(|| {
            anyhow::anyhow!(
                "oracle prefill produced no generated tokens; --oracle-prefill requires --max-new-tokens > 0"
            )
        })?;
        eprintln!(
            "[prefill] oracle prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
        return Ok(Qwen35Prefill {
            logits: output.prefill_logits,
            native_trace: None,
            next_token: first,
        });
    }

    let prefill_result = if cli.trace_prefill_layers {
        engine.prefill_native_with_trace(prompt_ids)?
    } else {
        engine.prefill_native_with_final_norm(prompt_ids)?
    };
    let first = sample_qwen_prefill_token(
        &prefill_result,
        host_lm_head_rescorer.filter(|_| allow_host_lm_head_rescore),
    )?;
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

    Ok(Qwen35Prefill {
        logits: prefill_result.logits,
        native_trace: Some((
            prefill_result.final_norm_trace,
            prefill_result.layer_attn_trace,
            prefill_result.layer_post_attn_norm_trace,
            prefill_result.layer_mlp_out_trace,
            prefill_result.layer_hidden_trace,
        )),
        next_token: first,
    })
}

pub(crate) fn report_qwen35_virtual_kv_after_prefill(engine: &mut DecodeEngine) -> Result<()> {
    let virtual_kv_stats = engine.virtual_kv_memory_stats();
    if virtual_kv_stats.layers > 0 {
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        let pct = if virtual_kv_stats.reserved_bytes > 0 {
            100.0 * virtual_kv_stats.resident_bytes as f64 / virtual_kv_stats.reserved_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "[vmm] virtual KV logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({pct:.1}%) mappings={} layers={}",
            mib(virtual_kv_stats.logical_bytes),
            mib(virtual_kv_stats.resident_bytes),
            mib(virtual_kv_stats.reserved_bytes),
            virtual_kv_stats.mappings,
            virtual_kv_stats.layers
        );
        if std::env::var_os("SUPERSONIC_VMM_KV_STATS").is_some() {
            for (layer_idx, stats) in engine.virtual_kv_memory_stats_by_layer() {
                let layer_pct = if stats.reserved_bytes > 0 {
                    100.0 * stats.resident_bytes as f64 / stats.reserved_bytes as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[vmm] layer={layer_idx} logical={:.2}MiB logical_resident={:.2}MiB backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({layer_pct:.1}%) mappings={}",
                    mib(stats.logical_bytes),
                    mib(stats.logical_resident_bytes),
                    mib(stats.logical_backup_bytes),
                    mib(stats.resident_bytes),
                    mib(stats.reserved_bytes),
                    stats.mappings
                );
            }
        }
    }

    if std::env::var_os("SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL").is_some() {
        let before = engine.virtual_kv_memory_stats();
        if before.layers == 0 {
            eprintln!("[vmm] SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL set but virtual KV is inactive");
        } else {
            verify_qwen35_virtual_kv_eviction(engine)?;
        }
    }
    Ok(())
}

fn verify_qwen35_virtual_kv_eviction(engine: &mut DecodeEngine) -> Result<()> {
    let verify_bytes = std::env::var_os("SUPERSONIC_VMM_KV_VERIFY_EVICT_BYTES").is_some();
    let kv_before = if verify_bytes {
        Some(engine.full_attention_prefix_cache_snapshots_bf16_host()?)
    } else {
        None
    };
    engine.evict_virtual_kv_to_host()?;
    let evicted = engine.virtual_kv_memory_stats();
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[vmm] evicted virtual KV to host logical_backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(evicted.logical_backup_bytes),
        mib(evicted.resident_bytes),
        mib(evicted.reserved_bytes),
        evicted.mappings
    );
    if std::env::var_os("SUPERSONIC_VMM_KV_RESTORE_TO_VMM").is_some() {
        engine.restore_virtual_kv_from_host_to_vmm()?;
    } else {
        engine.restore_virtual_kv_from_host()?;
    }
    if let Some(kv_before) = kv_before {
        let kv_after = engine.full_attention_prefix_cache_snapshots_bf16_host()?;
        if let Some(mismatch) = first_qwen35_virtual_kv_mismatch(&kv_before, &kv_after) {
            eprintln!("[vmm] warning: virtual KV eviction byte-restore mismatch: {mismatch}");
        }
    }
    let restored = engine.virtual_kv_memory_stats();
    eprintln!(
        "[vmm] restored virtual KV from host logical_resident={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(restored.logical_resident_bytes),
        mib(restored.resident_bytes),
        mib(restored.reserved_bytes),
        restored.mappings
    );
    Ok(())
}

type Qwen35VirtualKvSnapshot = (usize, Vec<u8>, Vec<u8>, usize);

fn first_qwen35_virtual_kv_mismatch(
    before: &[Qwen35VirtualKvSnapshot],
    after: &[Qwen35VirtualKvSnapshot],
) -> Option<String> {
    for (
        (before_layer, before_k, before_v, before_len),
        (after_layer, after_k, after_v, after_len),
    ) in before.iter().zip(after.iter())
    {
        if before_layer != after_layer || before_len != after_len {
            return Some(format!(
                "layer/id mismatch before={before_layer}:{before_len} after={after_layer}:{after_len}"
            ));
        }
        let k_diff = before_k
            .iter()
            .zip(after_k.iter())
            .position(|(a, b)| a != b);
        let v_diff = before_v
            .iter()
            .zip(after_v.iter())
            .position(|(a, b)| a != b);
        if before_k.len() != after_k.len() || before_v.len() != after_v.len() {
            return Some(format!(
                "layer={before_layer} len mismatch k {}->{} v {}->{}",
                before_k.len(),
                after_k.len(),
                before_v.len(),
                after_v.len()
            ));
        }
        if k_diff.is_some() || v_diff.is_some() {
            let sample = |before: &[u8], after: &[u8], diff: Option<usize>| {
                let start = diff.unwrap_or(0);
                let end = (start + 16).min(before.len()).min(after.len());
                format!(
                    "before={:?} after={:?}",
                    &before[start..end],
                    &after[start..end]
                )
            };
            return Some(format!(
                "layer={before_layer} first_k_diff={:?} first_v_diff={:?} k_sample={} v_sample={}",
                k_diff,
                v_diff,
                sample(before_k, after_k, k_diff),
                sample(before_v, after_v, v_diff)
            ));
        }
    }
    None
}

fn sample_qwen_prefill_token(
    prefill_result: &PrefillResult,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let prefill_normed = prefill_result
        .final_norm_trace
        .as_deref()
        .map(decode_bf16_le);
    sample_qwen_logits_with_rescore(
        &prefill_result.logits,
        prefill_normed.as_deref(),
        host_lm_head_rescorer,
    )
}

pub(crate) fn load_qwen35_startup(cli: &Cli) -> Result<Qwen35Startup> {
    let config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
    let text_config = config.text_config;
    eprintln!(
        "[config] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);

    Ok(Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    })
}

pub(crate) fn validate_qwen35_startup(
    cli: &Cli,
    model_variant: &ModelVariant,
    params: &Qwen35KernelParams,
    backend: Backend,
    registry_arch: &GpuArch,
    q4km_like: bool,
) -> Result<Qwen35Policy> {
    if cli.trace_prefill_layers && !cli.validate {
        anyhow::bail!("--trace-prefill-layers requires --validate");
    }
    if cli.trace_oracle_prefill_layer.is_some() && !cli.validate {
        anyhow::bail!("--trace-oracle-prefill-layer requires --validate");
    }
    if cli.trace_kv_fp8_cache && !cli.kv_fp8 {
        anyhow::bail!("--trace-kv-fp8-cache requires --kv-fp8");
    }
    let trace_kv_cache_enabled = cli.trace_kv_cache || cli.trace_kv_fp8_cache;
    if cli.force_kernel_decode && cli.force_component_decode {
        anyhow::bail!("Choose at most one of --force-kernel-decode or --force-component-decode");
    }
    if cli.trace_component_input_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-input-layer requires --force-component-decode");
    }
    if cli.trace_component_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_state_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-state-layer requires --force-component-decode");
    }
    if cli.trace_persistent_input_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!("--trace-persistent-input-layer requires the real 4B persistent kernel path");
    }
    if cli.trace_persistent_linear_state_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!(
            "--trace-persistent-linear-state-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_full_attn_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode))
    {
        anyhow::bail!(
            "--trace-persistent-full-attn-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_linear_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode)
            && !cli.kv_fp8)
    {
        anyhow::bail!(
            "--trace-persistent-linear-layer requires the real 4B persistent BF16 kernel path"
        );
    }

    if backend == Backend::Cuda {
        let qwen35_sm86 = matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B
                | ModelVariant::Qwen3_5_2B
                | ModelVariant::Qwen3_5_4B
                | ModelVariant::Qwen3_5_9B
        ) && *registry_arch == GpuArch::Sm86;
        if cli.int4 && !qwen35_sm86 {
            anyhow::bail!("CUDA --int4 currently supports only Qwen3.5 on sm86");
        }
        if cli.fp8_runtime && !(qwen35_sm86 || *model_variant == ModelVariant::Qwen3_6_27B) {
            anyhow::bail!(
                "CUDA --fp8-runtime currently supports only Qwen3.5 and qwen3.6-27b on sm86"
            );
        }
        if cli.kv_fp8 {
            if !qwen35_sm86 {
                anyhow::bail!("CUDA --kv-fp8 currently supports only Qwen3.5 on sm86");
            }
            if std::env::var_os("SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR").is_none() {
                std::env::set_var("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW", "128");
                eprintln!(
                    "[cuda] KV-FP8 BF16 sidecar window capped to the most recent 128 tokens on CUDA; \
                     set SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR=1 \
                     to restore full-prefix debug sidecar coverage"
                );
            }
        }
    } else if backend == Backend::Metal {
        if !matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B | ModelVariant::Qwen3_5_2B
        ) {
            anyhow::bail!("Metal only supports --model qwen3.5-0.8b or qwen3.5-2b");
        }
        if q4km_like {
            anyhow::bail!("Metal does not support --q4km/--q4km-gptq on Qwen3.5 yet");
        }
        if cli.fp8_runtime {
            anyhow::bail!("Metal does not support --fp8-runtime on Qwen3.5 yet");
        }
        if cli.kv_fp8 {
            anyhow::bail!("Metal does not support --kv-fp8 on Qwen3.5 yet");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("Metal only supports --batch-size 1");
        }
        if cli.force_kernel_decode || cli.force_component_decode {
            anyhow::bail!(
                "Metal does not support --force-kernel-decode or --force-component-decode"
            );
        }
    }

    Ok(Qwen35Policy {
        trace_kv_cache_enabled,
    })
}

pub(crate) fn load_qwen35_engine(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    params: &Qwen35KernelParams,
    backend: Backend,
    gpu_arch: GpuArch,
    ordinal: usize,
    bootstrap_downloaded: bool,
    q4km_like: bool,
    context_tokens: usize,
) -> Result<Qwen35EngineSetup> {
    let t0 = std::time::Instant::now();
    let weights = load_qwen35_weights(
        cli,
        model_variant,
        text_config,
        ordinal,
        params.weight_prefix,
        bootstrap_downloaded,
        q4km_like,
    )?;
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
    let allow_host_lm_head_rescore =
        cli.no_bake && !engine.weights().is_fp8 && !engine.weights().is_int4;

    Ok(Qwen35EngineSetup {
        engine,
        use_4b_kernel,
        cuda_08b_hero_enabled: cuda_08b_hero_candidate,
        allow_host_lm_head_rescore,
    })
}

pub(crate) fn run_qwen35(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    backend: Backend,
    gpu_arch: GpuArch,
    ordinal: usize,
    total_vram: u64,
    q4km_like: bool,
) -> Result<()> {
    if cli.dflash {
        // DFlash needs the target's HF metadata (config.json + tokenizer.json)
        // and the INT4 bake. Reuse the same download hooks as the regular
        // Qwen35 path so the dflash dispatch is self-contained on a fresh
        // machine: ensure_hf_metadata_present fetches HF metadata from the
        // bake tarball if config.json is missing, then we verify or download
        // the INT4 bake itself.
        ensure_hf_metadata_present(&cli, &model_variant)?;
        if !cli.no_bake {
            let variant = model_store::fetch::BakeVariant::Int4Gptq;
            let bake_dir = variant.bake_dir(&cli.model_dir);
            let _lock = model_store::BakeLock::acquire(&cli.model_dir)
                .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
            if should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir)) {
                let canonical_model = model_variant.to_string();
                match try_download_bake(&cli, variant, &canonical_model, &bake_dir) {
                    Ok(true) => {
                        eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                    }
                    Ok(false) => {
                        anyhow::bail!(
                            "no INT4 bake at {} and --no-download set.\n\
                             Run:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                    Err(e) => {
                        anyhow::bail!(
                            "could not obtain INT4 bake for --dflash: {e}\n\n\
                             INT4 baking requires a GPTQ calibration pass in Python. \
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            cli.model_dir.display(),
                        );
                    }
                }
            }
        }
        return qwen35_dflash_engine::run_qwen35_dflash(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }
    // --dflash-* guard already ran before the family dispatch above.

    // --specprefill-* dispatch. Validation already ran in
    // validate_specprefill_flags; the presence of --specprefill-draft-dir
    // is the gate that switches to the SpecPrefill orchestrator.
    if cli.specprefill_draft_dir.is_some() {
        return specprefill_engine::run_specprefill(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Qwen36Moe(_) => unreachable!("qwen3.6-moe handled above"),
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
        FamilyParams::Phi4(_) => unreachable!("phi4 handled above"),
        FamilyParams::Llama31(_) => unreachable!("llama3.1 handled above"),
    };
    let host_lm_head_rescorer = HostLmHeadRescorer::from_model_dir(&cli.model_dir)?;

    // Install the per-(arch, model) HIP launch preset (grid size +
    // cooperative flag) if one is registered. User env vars still override
    // inside the bridge. Always called — `(0, false)` clears any stale
    // preset from a prior run, so switching models doesn't inherit the
    // previous one's grid. No-op on CUDA builds.
    {
        let preset = registry::qwen35_4b_launch_preset(&entry.arch, &entry.model);
        let (blocks, coop) = preset.unwrap_or((0, false));
        kernel_ffi::set_qwen35_4b_launch_preset(blocks, coop);
        if let Some((blocks, coop)) = preset {
            eprintln!("[preset] qwen35_4b launch: blocks={blocks} cooperative={coop}");
        }
    }

    let Qwen35Policy {
        trace_kv_cache_enabled,
    } = validate_qwen35_startup(
        &cli,
        &model_variant,
        params,
        backend,
        &entry.arch,
        q4km_like,
    )?;

    // If --model-dir is pristine (no config.json), fetch a bake first so the
    // downloader can populate HF metadata before we try to read it.
    let bootstrap_downloaded = ensure_hf_metadata_present(&cli, &model_variant)?;

    let Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    } = load_qwen35_startup(&cli)?;
    check_qwen35_vram(
        &cli,
        &text_config,
        &entry.vram,
        context_tokens,
        params.kv_chunk_size,
        total_vram,
    )?;

    let gpu_validate_enabled = cli.gpu_validate && cli.batch_size == 1;
    if cli.gpu_validate && cli.batch_size > 1 {
        eprintln!("[gpu-validate] GPU oracle disabled for batch_size > 1");
    }
    let Qwen35EngineSetup {
        mut engine,
        use_4b_kernel,
        cuda_08b_hero_enabled,
        allow_host_lm_head_rescore,
    } = load_qwen35_engine(
        &cli,
        &model_variant,
        &text_config,
        params,
        backend,
        gpu_arch,
        ordinal,
        bootstrap_downloaded,
        q4km_like,
        context_tokens,
    )?;

    // When using FP8 runtime weights, tell the oracle to use the same FP8 weights
    // (dequanted to BF16) so we compare apples-to-apples.
    let fp8_oracle_dir = if cli.fp8_runtime {
        Some(cli.model_dir.clone())
    } else {
        None
    };
    let oracle_device = resolve_oracle_device(&cli.oracle_device, backend, ordinal);

    // Run prefill (native GPU or oracle)
    let qwen_oracle_model_id =
        resolve_qwen_oracle_model_id(cli.model_id.as_deref(), &cli.model_dir, &model_variant);
    let Qwen35Prefill {
        logits: prefill_logits,
        native_trace: native_prefill_trace,
        mut next_token,
    } = run_qwen35_prefill(
        &cli,
        &mut engine,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        host_lm_head_rescorer.as_ref(),
        allow_host_lm_head_rescore,
    )?;

    if cli.dump_last_logits {
        use std::io::Write as _;
        print!("\nLAST_LOGITS: ");
        for (i, x) in prefill_logits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    report_qwen35_virtual_kv_after_prefill(&mut engine)?;

    // Optionally run oracle for validation
    let oracle_output = run_qwen35_oracle_validation(
        &cli,
        &engine,
        &text_config,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        &prefill_logits,
        native_prefill_trace.as_ref(),
        next_token,
    )?;

    if let (Some(trace_layer), Some(output)) =
        (cli.trace_oracle_prefill_layer, oracle_output.as_ref())
    {
        let oracle_script = qwen35_oracle_script_path();
        trace_qwen35_oracle_prefill_layer(
            &mut engine,
            trace_layer,
            &prompt_ids,
            &oracle_script,
            &qwen_oracle_model_id,
            &cli.oracle_dtype,
            &oracle_device,
            fp8_oracle_dir.as_deref(),
            output,
        )?;
    }

    // Replicate prefill state to batch items if batch_size > 1
    if cli.batch_size > 1 {
        eprintln!(
            "[batch] replicating prefill state to {} sequences",
            cli.batch_size
        );
        engine.replicate_state_to_batch()?;
    }

    if gpu_validate_enabled {
        eprintln!(
            "[gpu-validate] replaying full token history through GPU prefill for reference..."
        );
    }
    // Replay-prefill path used to be the default for 4B single-seq decode
    // (safety net for numerical-parity work during the CUDA sm86 bring-up)
    // but it scales O(N) per step with context length and was ~7x slower
    // than the persistent megakernel at 64-token generations. Default is now
    // the megakernel; --force-replay-decode re-enables the replay path for
    // the rare case where someone genuinely wants to reproduce the older
    // numeric semantics.
    let cuda_qwen2b_replay_default = backend == Backend::Cuda
        && *model_variant == ModelVariant::Qwen3_5_2B
        && cli.batch_size == 1
        && use_4b_kernel
        && !cli.kv_fp8
        && !cli.force_kernel_decode
        && !cli.force_component_decode;
    // Metal v2 wires per-op incremental decode through the standard
    // `engine.decode_step` path; only the legacy 4B / qwen3.5-2b CUDA replay
    // gates remain.
    let metal_v2_incremental = backend == Backend::Metal && cli.batch_size == 1;
    let replay_decode_enabled = cli.batch_size == 1
        && !cli.force_kernel_decode
        && !cli.force_component_decode
        && !cli.kv_fp8
        && use_4b_kernel
        && (cli.force_replay_decode || cuda_qwen2b_replay_default);
    let replay_kv_fp8_enabled =
        use_4b_kernel && cli.kv_fp8 && cli.batch_size == 1 && !cli.force_kernel_decode;
    let component_single_decode_enabled =
        cli.batch_size == 1 && use_4b_kernel && cli.force_component_decode;
    // Use the batched persistent megakernel path (decode_step_batch with b=1)
    // for 4B single-seq decode by default — measured ~300 ms/tok on gfx1150
    // vs ~500 ms/tok for decode_step() and ~2500 ms/tok for the legacy
    // replay path. Opt-out via --force-replay-decode (legacy parity) or
    // --force-component-decode (primitive-chain correctness).
    let kernel_single_decode_enabled = cli.batch_size == 1
        && use_4b_kernel
        && !cli.force_replay_decode
        && !cli.force_component_decode;
    let cuda_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_CUDA_FAST_GREEDY").is_some();
    let cuda_fast_greedy_enabled = backend == Backend::Cuda
        && !use_4b_kernel
        && cli.batch_size == 1
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && oracle_output.is_none()
        && !cuda_08b_hero_enabled
        && !cuda_fast_greedy_disabled;
    // Metal fast-greedy: same trigger conditions as CUDA's fast-greedy. Uses the
    // fused lm_head + argmax kernel and returns just the sampled token, skipping
    // the per-token 250k-element BF16 D2H + host argmax loop. Disable via env
    // var for bring-up/bisect.
    let metal_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_METAL_FAST_GREEDY").is_some();
    let metal_fast_greedy_enabled = backend == Backend::Metal
        && metal_v2_incremental
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && oracle_output.is_none()
        && !metal_fast_greedy_disabled;
    if metal_v2_incremental {
        if metal_fast_greedy_enabled {
            eprintln!("[decode] Metal v2 incremental decode (fast-greedy: fused argmax)");
        } else {
            eprintln!("[decode] Metal v2 incremental decode");
        }
    }
    if replay_decode_enabled {
        if cuda_qwen2b_replay_default {
            eprintln!(
                "[decode] single-sequence CUDA qwen3.5-2b uses replayed GPU prefill for correctness"
            );
        } else {
            eprintln!("[decode] single-sequence 4B uses replayed GPU prefill for correctness");
        }
    } else if replay_kv_fp8_enabled && cli.batch_size == 1 {
        eprintln!("[decode] single-sequence KV-FP8 uses replayed GPU prefill for correctness");
    } else if cli.batch_size > 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] batched KV-FP8 uses the persistent kernel path");
    } else if component_single_decode_enabled {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the component decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.force_kernel_decode {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the kernel decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] WARNING: single-sequence KV-FP8 uses the b=1 kernel path");
    } else if cuda_08b_hero_enabled {
        eprintln!("[decode] CUDA 0.8B sm86 hero path enabled");
    } else if cuda_fast_greedy_enabled {
        eprintln!("[decode] CUDA fast greedy sampling enabled for the non-4B native decode path");
    }

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;
    let mut gpu_max_delta = 0.0f32;
    let mut native_decode_timings = DecodeStageTimings::default();
    let mut native_decode_timing_steps = 0usize;
    let eos_ids = text_config.eos_token_ids();

    // For batched decode, track per-sequence tokens
    let mut batch_next_tokens: Vec<u32> = vec![next_token; cli.batch_size];

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        // Stop on EOS token (sequence 0 drives the output)
        if eos_ids.contains(&next_token) {
            break;
        }

        let seqlen_offset = seqlen_start + step;

        if cli.batch_size > 1 {
            if let Some(trace_layer) = cli.trace_persistent_linear_state_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                let _ = engine.decode_step_batch_trace_hidden_after_layers(
                    &trace_tokens,
                    seqlen_offset,
                    trace_layer + 1,
                    0,
                )?;
                trace_persistent_linear_state_layer(
                    &engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_input_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
                    &trace_tokens,
                    seqlen_offset,
                    trace_layer,
                    0,
                )?;
                trace_persistent_input_layer(
                    &engine,
                    &native_hidden,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_full_attn_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                trace_persistent_full_attn_layer(
                    &mut engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    trace_tokens.as_slice(),
                    seqlen_offset,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_linear_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                trace_persistent_linear_layer(
                    &mut engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    trace_tokens.as_slice(),
                    seqlen_offset,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            let (batch_logits, batch_timings) = if replay_kv_fp8_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let logits = engine.rebuild_prefill_state(&token_ids, true)?;
                (vec![logits; cli.batch_size], None)
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_batch_with_timings(&batch_next_tokens, seqlen_offset)?;
                (logits, Some(timings))
            } else {
                // Batched decode
                (
                    engine.decode_step_batch(&batch_next_tokens, seqlen_offset)?,
                    None,
                )
            };
            if let Some(timings) = batch_timings {
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
            }

            // Use sequence 0's logits for output and validation
            let logits = &batch_logits[0];

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(logits, oracle_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token} batch_size={}",
                        cli.batch_size
                    );
                }
            }

            // Sample next tokens for all sequences
            let sampling_start = Instant::now();
            for (bi, seq_logits) in batch_logits.iter().enumerate() {
                batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
            }
            if batch_timings.is_some() {
                native_decode_timings.host_sampling_ms +=
                    sampling_start.elapsed().as_secs_f64() * 1000.0;
            }

            generated_ids.push(next_token);
            if trace_kv_cache_enabled {
                let cache_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .collect();
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
            next_token = batch_next_tokens[0];
        } else {
            // Single-sequence decode (original path)
            let mut maybe_fast_token = None;
            let mut can_rescore_with_normed = false;
            let logits = if cuda_fast_greedy_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if metal_fast_greedy_enabled {
                let token = engine.decode_step_metal_fast_greedy(next_token, seqlen_offset)?;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if cuda_08b_hero_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_08b_hero(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if replay_decode_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?
            } else if replay_kv_fp8_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                engine.rebuild_prefill_state(&token_ids, false)?
            } else if component_single_decode_enabled {
                if let Some(trace_layer) = cli.trace_component_linear_state_layer {
                    trace_component_linear_state_layer(
                        &engine,
                        trace_layer,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                }
                if let Some(trace_layer) = cli.trace_component_input_layer {
                    let (logits, hidden_trace) = engine.component_decode_step_4b_traced(
                        next_token,
                        seqlen_offset,
                        trace_layer,
                    )?;
                    trace_component_input_layer(
                        &engine,
                        &hidden_trace,
                        trace_layer,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    logits
                } else if let Some(trace_layer) = cli.trace_component_layer {
                    let (logits, layer_trace) = engine.component_decode_step_4b_trace_layer(
                        next_token,
                        seqlen_offset,
                        trace_layer,
                    )?;
                    trace_component_layer(
                        &engine,
                        trace_layer,
                        &layer_trace,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    logits
                } else if let Some(trace_layer) = cli.trace_component_linear_layer {
                    let (logits, linear_trace) = engine
                        .component_decode_step_4b_trace_linear_layer(
                            next_token,
                            seqlen_offset,
                            trace_layer,
                        )?;
                    trace_component_linear_layer(
                        &engine,
                        trace_layer,
                        &linear_trace,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    logits
                } else {
                    engine.decode_step(next_token, seqlen_offset)?
                }
            } else if kernel_single_decode_enabled {
                if let Some(trace_layer) = cli.trace_persistent_linear_state_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    let _ = engine.decode_step_batch_trace_hidden_after_layers(
                        &[next_token],
                        seqlen_offset,
                        trace_layer + 1,
                        0,
                    )?;
                    trace_persistent_linear_state_layer(
                        &engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if let Some(trace_layer) = cli.trace_persistent_input_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
                        &[next_token],
                        seqlen_offset,
                        trace_layer,
                        0,
                    )?;
                    trace_persistent_input_layer(
                        &engine,
                        &native_hidden,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if let Some(trace_layer) = cli.trace_persistent_full_attn_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    trace_persistent_full_attn_layer(
                        &mut engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        &[next_token],
                        seqlen_offset,
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if let Some(trace_layer) = cli.trace_persistent_linear_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    trace_persistent_linear_layer(
                        &mut engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        &[next_token],
                        seqlen_offset,
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if cli.emit_stage_timings {
                    let (logits, timings) = engine
                        .decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    can_rescore_with_normed = true;
                    logits
                } else {
                    can_rescore_with_normed = true;
                    engine
                        .decode_step_batch(&[next_token], seqlen_offset)?
                        .remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                can_rescore_with_normed = true;
                logits
            } else {
                can_rescore_with_normed = true;
                engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token = if let Some(token) = maybe_fast_token {
                token
            } else {
                let normed = if can_rescore_with_normed && allow_host_lm_head_rescore {
                    Some(engine.last_normed_host_f32()?)
                } else {
                    None
                };
                sample_qwen_logits_with_rescore(
                    &logits,
                    normed.as_deref(),
                    host_lm_head_rescorer
                        .as_ref()
                        .filter(|_| allow_host_lm_head_rescore),
                )?
            };

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(&logits, oracle_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
                    );
                }
            }

            if gpu_validate_enabled {
                let gpu_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let gpu_logits = prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &gpu_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                let delta = validate::max_abs_delta(&logits, &gpu_logits);
                let gpu_token = DecodeEngine::greedy_sample(&gpu_logits);
                let token_match = if gpu_token == native_token {
                    ""
                } else {
                    " MISMATCH"
                };
                if delta > gpu_max_delta {
                    gpu_max_delta = delta;
                }
                eprintln!(
                    "[gpu-validate] step={step} seq_off={seqlen_offset} delta={delta:.4} native_token={native_token} gpu_token={gpu_token}{token_match}"
                );
            }

            generated_ids.push(next_token);
            next_token = native_token;

            if trace_kv_cache_enabled {
                let cache_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .collect();
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
        }
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    // Decode generated tokens to text
    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    let generated_text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize generated suffix: {e}"))?;

    println!("{text}");
    if cli.emit_generated_json {
        println!(
            "[generated_json] {}",
            serde_json::to_string(&generated_text)?
        );
    }
    println!(
        "[tokens] {}",
        generated_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_tok={:.0} decode_max_delta={max_delta:.4} gpu_oracle_max_delta={gpu_max_delta:.4} batch_size={}",
        prompt_ids.len(),
        generated_ids.len(),
        if generated_ids.is_empty() { 0.0 } else { decode_ms / generated_ids.len() as f64 },
        cli.batch_size,
    );
    if cli.emit_stage_timings {
        if native_decode_timing_steps > 0 {
            eprintln!(
                "[stage-timings] steps={} persistent_ms={:.3} rms_norm_ms={:.3} lm_head_ms={:.3} logits_d2h_ms={:.3} host_sampling_ms={:.3} gpu_argmax_ms={:.3} token_d2h_ms={:.3} total_native_decode_ms={:.3} persistent_full_attn_ms={:.3} persistent_full_attn_proj_ms={:.3} persistent_full_attn_core_ms={:.3} persistent_full_attn_out_ms={:.3} persistent_linear_proj_ms={:.3} persistent_linear_core_ms={:.3} persistent_linear_core_conv_ms={:.3} persistent_linear_core_recurrent_ms={:.3} persistent_linear_core_post_ms={:.3} persistent_linear_out_ms={:.3} persistent_mlp_gate_up_ms={:.3} persistent_mlp_down_ms={:.3}",
                native_decode_timing_steps,
                native_decode_timings.persistent_ms,
                native_decode_timings.rms_norm_ms,
                native_decode_timings.lm_head_ms,
                native_decode_timings.logits_d2h_ms,
                native_decode_timings.host_sampling_ms,
                native_decode_timings.gpu_argmax_ms,
                native_decode_timings.token_d2h_ms,
                native_decode_timings.total_ms(),
                native_decode_timings.persistent_full_attn_ms,
                native_decode_timings.persistent_full_attn_proj_ms,
                native_decode_timings.persistent_full_attn_core_ms,
                native_decode_timings.persistent_full_attn_out_ms,
                native_decode_timings.persistent_linear_proj_ms,
                native_decode_timings.persistent_linear_core_ms,
                native_decode_timings.persistent_linear_core_conv_ms,
                native_decode_timings.persistent_linear_core_recurrent_ms,
                native_decode_timings.persistent_linear_core_post_ms,
                native_decode_timings.persistent_linear_out_ms,
                native_decode_timings.persistent_mlp_gate_up_ms,
                native_decode_timings.persistent_mlp_down_ms,
            );
        } else {
            eprintln!("[stage-timings] steps=0 note=no native decode stage timings collected for this path");
        }
    }

    Ok(())
}
