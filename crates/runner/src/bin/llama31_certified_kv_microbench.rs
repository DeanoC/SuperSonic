use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::certified_kv;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Synthetic certified-KV CUDA microbench for Llama 3.1-style heads")]
struct Args {
    #[arg(long, default_value = "4096,8192,16384,32768")]
    contexts: String,
    #[arg(long, default_value_t = 20)]
    iters: usize,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long, default_value_t = 0)]
    ordinal: usize,
    #[arg(long, default_value_t = 32)]
    q_heads: usize,
    #[arg(long, default_value_t = 8)]
    kv_heads: usize,
    #[arg(long, default_value_t = 128)]
    head_dim: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 16)]
    value_group_size: usize,
    #[arg(long, default_value_t = 2)]
    k_min: usize,
    #[arg(long, default_value_t = 128)]
    k_max: usize,
    #[arg(long, default_value_t = 0.995)]
    tau_cov: f32,
    #[arg(long, default_value_t = 0.005)]
    rung1_threshold: f32,
    #[arg(long, default_value_t = 2.0)]
    rung1_multiplier: f32,
    #[arg(long, default_value_t = 3.0)]
    delta_guard_factor: f32,
    #[arg(long, default_value_t = 0.0)]
    score_exploration_rate: f32,
    #[arg(long, default_value_t = 0.05)]
    v_tol: f32,
    #[arg(long, default_value_t = 0.0)]
    eps_guard: f32,
    #[arg(long, default_value_t = true)]
    require_certified_tail_bound: bool,
}

#[derive(Default, Serialize)]
struct PhaseMs {
    score_blocks_int8: f64,
    select_blocks_device: f64,
    gather_promoted_bf16: f64,
    score_consistency: f64,
    selected_fp16_log_masses: f64,
    ranking_flags_device: f64,
    attend_mixed_key_int4: f64,
    block_masses_from_probs: f64,
    value_promotions: f64,
    gather_promoted_values_bf16: f64,
}

#[derive(Serialize)]
struct ContextReport {
    context_tokens: usize,
    num_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    value_group_size: usize,
    k_min: usize,
    k_max: usize,
    max_promoted_blocks: usize,
    max_promoted_value_blocks: usize,
    tau_cov: f32,
    v_tol: f32,
    iters: usize,
    warmup: usize,
    source_bf16_device_staging: bool,
    tier1_bytes: usize,
    bf16_staging_bytes: usize,
    scratch_bytes: usize,
    selected_blocks_mean: f64,
    value_promoted_blocks: u64,
    ranking_fallback_heads: u64,
    score_consistency_heads: u64,
    any_value_promoted: bool,
    phase_ms_mean: PhaseMs,
    total_profiled_ms_mean: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    gpu_hal::set_backend(Backend::Cuda);
    gpu_hal::set_device(args.ordinal).context("set CUDA device")?;

    if args.iters == 0 {
        bail!("--iters must be > 0");
    }
    if args.q_heads != args.kv_heads * (args.q_heads / args.kv_heads) {
        bail!("q_heads must be an integer multiple of kv_heads");
    }
    let contexts = parse_contexts(&args.contexts)?;
    let mut reports = Vec::with_capacity(contexts.len());
    for context_tokens in contexts {
        reports.push(run_context(&args, context_tokens)?);
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}

fn run_context(args: &Args, context_tokens: usize) -> Result<ContextReport> {
    if context_tokens == 0 || context_tokens % args.block_size != 0 {
        bail!(
            "context {context_tokens} must be non-zero and divisible by block_size {}",
            args.block_size
        );
    }
    let num_blocks = context_tokens / args.block_size;
    let gqa_group = args.q_heads / args.kv_heads;
    let max_promoted_blocks = (args.k_max * 2).min(num_blocks).max(args.k_min);
    let max_promoted_value_blocks = num_blocks;
    let q_scale = 1.0f32 / (args.head_dim as f32).sqrt();

    let query = GpuBuffer::from_host_bytes(
        args.ordinal,
        ScalarType::BF16,
        &[args.q_heads, args.head_dim],
        &make_bf16_bytes(args.q_heads * args.head_dim, 0xA5A5_1234, 0.35),
    )
    .context("query upload")?;
    let key_bf16 = GpuBuffer::from_host_bytes(
        args.ordinal,
        ScalarType::BF16,
        &[1, args.kv_heads, context_tokens, args.head_dim],
        &make_bf16_bytes(
            args.kv_heads * context_tokens * args.head_dim,
            0xBEEF_0001,
            0.20,
        ),
    )
    .context("Tier-2 key BF16 device staging upload")?;
    let value_bf16 = GpuBuffer::from_host_bytes(
        args.ordinal,
        ScalarType::BF16,
        &[1, args.kv_heads, context_tokens, args.head_dim],
        &make_bf16_bytes(
            args.kv_heads * context_tokens * args.head_dim,
            0xC0DE_0002,
            0.20,
        ),
    )
    .context("Tier-2 value BF16 device staging upload")?;

    let (key_i8_shape, key_scale_shape, value_i4_shape, value_meta_shape, value_error_shape) =
        certified_kv::quantized_shapes(
            args.kv_heads,
            context_tokens,
            args.head_dim,
            args.block_size,
            args.value_group_size,
        )
        .context("quantized shapes")?;

    let mut key_int8 = GpuBuffer::zeros(args.ordinal, ScalarType::U8, &key_i8_shape)?;
    let mut key_scale = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &key_scale_shape)?;
    let mut key_zero = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &key_scale_shape)?;
    let mut value_int4 = GpuBuffer::zeros(args.ordinal, ScalarType::U8, &value_i4_shape)?;
    let mut value_scale = GpuBuffer::zeros(args.ordinal, ScalarType::F16, &value_meta_shape)?;
    let mut value_zero = GpuBuffer::zeros(args.ordinal, ScalarType::F16, &value_meta_shape)?;
    let mut value_error = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &value_error_shape)?;
    let mut value_norm = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &value_error_shape)?;
    certified_kv::quantize_bf16_cache(
        args.ordinal,
        &key_bf16,
        &value_bf16,
        context_tokens,
        args.block_size,
        args.value_group_size,
        &mut key_int8,
        &mut key_scale,
        &mut key_zero,
        &mut value_int4,
        &mut value_scale,
        &mut value_zero,
        &mut value_error,
        &mut value_norm,
    )
    .context("quantize BF16 cache")?;

    let mut key_scale_norm =
        GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.kv_heads, num_blocks])?;
    certified_kv::key_scale_norms(args.ordinal, &key_scale, &mut key_scale_norm, num_blocks)
        .context("key scale norms")?;

    let mut block_max =
        GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads, num_blocks])?;
    let mut block_sum =
        GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads, num_blocks])?;
    let mut promote_index =
        GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.q_heads, num_blocks])?;
    let mut value_promote_index =
        GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.kv_heads, num_blocks])?;
    let mut selected_blocks = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::U32,
        &[args.q_heads, max_promoted_blocks],
    )?;
    let mut selected_counts = GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.q_heads])?;
    let mut fallback_flags = GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.q_heads])?;
    let mut score_consistency_flags =
        GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.q_heads])?;
    let mut delta_blocks =
        GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads, num_blocks])?;
    let mut e_key_by_head = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads])?;
    let mut delta_tail_by_head = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads])?;
    let mut vmax_by_head = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads])?;
    let mut true_tail_by_head = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads])?;
    let mut promoted_key_bf16 = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::BF16,
        &[
            args.q_heads,
            max_promoted_blocks,
            args.block_size,
            args.head_dim,
        ],
    )?;
    let mut promoted_value_bf16 = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::BF16,
        &[
            args.kv_heads,
            max_promoted_value_blocks,
            args.block_size,
            args.head_dim,
        ],
    )?;
    let mut selected_fp16_log_masses = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::F32,
        &[args.q_heads, max_promoted_blocks],
    )?;
    let mut score_scratch = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::F32,
        &[args.q_heads, context_tokens],
    )?;
    let mut output_bf16 = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::BF16,
        &[args.q_heads, args.head_dim],
    )?;
    let mut block_mass =
        GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads, num_blocks])?;
    let mut value_kv_counters = GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.kv_heads])?;
    let mut any_value_promoted = GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[1])?;
    let mut value_head_promoted_flags =
        GpuBuffer::zeros(args.ordinal, ScalarType::U32, &[args.q_heads])?;
    let mut e_val_by_head = GpuBuffer::zeros(args.ordinal, ScalarType::F32, &[args.q_heads])?;

    gpu_hal::sync(args.ordinal).context("initial sync")?;
    for _ in 0..args.warmup {
        run_once(
            args,
            context_tokens,
            gqa_group,
            q_scale,
            max_promoted_blocks,
            max_promoted_value_blocks,
            &query,
            &key_bf16,
            &value_bf16,
            &key_int8,
            &key_scale,
            &key_zero,
            &key_scale_norm,
            &value_int4,
            &value_scale,
            &value_zero,
            &value_error,
            &value_norm,
            &mut block_max,
            &mut block_sum,
            &mut promote_index,
            &mut value_promote_index,
            &mut selected_blocks,
            &mut selected_counts,
            &mut fallback_flags,
            &mut score_consistency_flags,
            &mut delta_blocks,
            &mut e_key_by_head,
            &mut delta_tail_by_head,
            &mut vmax_by_head,
            &mut true_tail_by_head,
            &mut promoted_key_bf16,
            &mut promoted_value_bf16,
            &mut selected_fp16_log_masses,
            &mut score_scratch,
            &mut output_bf16,
            &mut block_mass,
            &mut value_kv_counters,
            &mut any_value_promoted,
            &mut value_head_promoted_flags,
            &mut e_val_by_head,
            None,
        )?;
    }

    let mut phase = PhaseMs::default();
    for _ in 0..args.iters {
        run_once(
            args,
            context_tokens,
            gqa_group,
            q_scale,
            max_promoted_blocks,
            max_promoted_value_blocks,
            &query,
            &key_bf16,
            &value_bf16,
            &key_int8,
            &key_scale,
            &key_zero,
            &key_scale_norm,
            &value_int4,
            &value_scale,
            &value_zero,
            &value_error,
            &value_norm,
            &mut block_max,
            &mut block_sum,
            &mut promote_index,
            &mut value_promote_index,
            &mut selected_blocks,
            &mut selected_counts,
            &mut fallback_flags,
            &mut score_consistency_flags,
            &mut delta_blocks,
            &mut e_key_by_head,
            &mut delta_tail_by_head,
            &mut vmax_by_head,
            &mut true_tail_by_head,
            &mut promoted_key_bf16,
            &mut promoted_value_bf16,
            &mut selected_fp16_log_masses,
            &mut score_scratch,
            &mut output_bf16,
            &mut block_mass,
            &mut value_kv_counters,
            &mut any_value_promoted,
            &mut value_head_promoted_flags,
            &mut e_val_by_head,
            Some(&mut phase),
        )?;
    }
    scale_phase(&mut phase, 1.0 / args.iters as f64);

    let selected_counts_host = selected_counts.to_host_bytes()?;
    let selected_blocks_mean = decode_u32(&selected_counts_host)
        .into_iter()
        .map(|x| x as f64)
        .sum::<f64>()
        / args.q_heads as f64;
    let value_promoted_blocks = decode_u32(&value_kv_counters.to_host_bytes()?)
        .into_iter()
        .map(u64::from)
        .sum::<u64>();
    let ranking_fallback_heads = decode_u32(&fallback_flags.to_host_bytes()?)
        .into_iter()
        .filter(|&x| x != 0)
        .count() as u64;
    let score_consistency_heads = decode_u32(&score_consistency_flags.to_host_bytes()?)
        .into_iter()
        .filter(|&x| x != 0)
        .count() as u64;
    let any_value_promoted_flag = decode_u32(&any_value_promoted.to_host_bytes()?)
        .first()
        .copied()
        .unwrap_or(0)
        != 0;

    let tier1_bytes = key_int8.len_bytes()
        + key_scale.len_bytes()
        + key_zero.len_bytes()
        + value_int4.len_bytes()
        + value_scale.len_bytes()
        + value_zero.len_bytes()
        + value_error.len_bytes()
        + value_norm.len_bytes();
    let bf16_staging_bytes = key_bf16.len_bytes() + value_bf16.len_bytes();
    let scratch_bytes = block_max.len_bytes()
        + block_sum.len_bytes()
        + promote_index.len_bytes()
        + value_promote_index.len_bytes()
        + selected_blocks.len_bytes()
        + selected_counts.len_bytes()
        + fallback_flags.len_bytes()
        + score_consistency_flags.len_bytes()
        + delta_blocks.len_bytes()
        + e_key_by_head.len_bytes()
        + delta_tail_by_head.len_bytes()
        + vmax_by_head.len_bytes()
        + true_tail_by_head.len_bytes()
        + promoted_key_bf16.len_bytes()
        + promoted_value_bf16.len_bytes()
        + selected_fp16_log_masses.len_bytes()
        + score_scratch.len_bytes()
        + output_bf16.len_bytes()
        + block_mass.len_bytes()
        + value_kv_counters.len_bytes()
        + any_value_promoted.len_bytes()
        + value_head_promoted_flags.len_bytes()
        + e_val_by_head.len_bytes()
        + key_scale_norm.len_bytes();
    let total_profiled_ms_mean = phase.score_blocks_int8
        + phase.select_blocks_device
        + phase.gather_promoted_bf16
        + phase.score_consistency
        + phase.selected_fp16_log_masses
        + phase.ranking_flags_device
        + phase.attend_mixed_key_int4
        + phase.block_masses_from_probs
        + phase.value_promotions
        + phase.gather_promoted_values_bf16;

    Ok(ContextReport {
        context_tokens,
        num_blocks,
        q_heads: args.q_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        block_size: args.block_size,
        value_group_size: args.value_group_size,
        k_min: args.k_min,
        k_max: args.k_max,
        max_promoted_blocks,
        max_promoted_value_blocks,
        tau_cov: args.tau_cov,
        v_tol: args.v_tol,
        iters: args.iters,
        warmup: args.warmup,
        source_bf16_device_staging: true,
        tier1_bytes,
        bf16_staging_bytes,
        scratch_bytes,
        selected_blocks_mean,
        value_promoted_blocks,
        ranking_fallback_heads,
        score_consistency_heads,
        any_value_promoted: any_value_promoted_flag,
        phase_ms_mean: phase,
        total_profiled_ms_mean,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    args: &Args,
    context_tokens: usize,
    gqa_group: usize,
    q_scale: f32,
    max_promoted_blocks: usize,
    max_promoted_value_blocks: usize,
    query: &GpuBuffer,
    key_bf16: &GpuBuffer,
    value_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
    key_scale_norm: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    value_error: &GpuBuffer,
    value_norm: &GpuBuffer,
    block_max: &mut GpuBuffer,
    block_sum: &mut GpuBuffer,
    promote_index: &mut GpuBuffer,
    value_promote_index: &mut GpuBuffer,
    selected_blocks: &mut GpuBuffer,
    selected_counts: &mut GpuBuffer,
    fallback_flags: &mut GpuBuffer,
    score_consistency_flags: &mut GpuBuffer,
    delta_blocks: &mut GpuBuffer,
    e_key_by_head: &mut GpuBuffer,
    delta_tail_by_head: &mut GpuBuffer,
    vmax_by_head: &mut GpuBuffer,
    true_tail_by_head: &mut GpuBuffer,
    promoted_key_bf16: &mut GpuBuffer,
    promoted_value_bf16: &mut GpuBuffer,
    selected_fp16_log_masses: &mut GpuBuffer,
    score_scratch: &mut GpuBuffer,
    output_bf16: &mut GpuBuffer,
    block_mass: &mut GpuBuffer,
    value_kv_counters: &mut GpuBuffer,
    any_value_promoted: &mut GpuBuffer,
    value_head_promoted_flags: &mut GpuBuffer,
    e_val_by_head: &mut GpuBuffer,
    mut phase: Option<&mut PhaseMs>,
) -> Result<()> {
    let mut time_phase =
        |slot: fn(&mut PhaseMs) -> &mut f64, f: &mut dyn FnMut() -> Result<()>| -> Result<()> {
            let start = Instant::now();
            f()?;
            gpu_hal::sync(args.ordinal)?;
            if let Some(phase) = phase.as_deref_mut() {
                *slot(phase) += start.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(())
        };

    time_phase(|p| &mut p.score_blocks_int8, &mut || {
        certified_kv::score_blocks_int8(
            args.ordinal,
            query,
            key_int8,
            key_scale,
            key_zero,
            args.block_size,
            gqa_group,
            q_scale,
            block_max,
            block_sum,
        )
        .context("score blocks INT8")
    })?;
    time_phase(|p| &mut p.select_blocks_device, &mut || {
        certified_kv::select_blocks_device(
            args.ordinal,
            query,
            key_scale_norm,
            block_max,
            block_sum,
            value_norm,
            promote_index,
            value_promote_index,
            selected_blocks,
            selected_counts,
            fallback_flags,
            delta_blocks,
            e_key_by_head,
            delta_tail_by_head,
            vmax_by_head,
            true_tail_by_head,
            gqa_group,
            args.k_min,
            args.k_max,
            max_promoted_blocks,
            q_scale,
            args.tau_cov,
            args.rung1_threshold,
            args.rung1_multiplier,
            args.delta_guard_factor,
            args.score_exploration_rate,
            args.require_certified_tail_bound,
        )
        .context("device selector")
    })?;
    time_phase(|p| &mut p.gather_promoted_bf16, &mut || {
        certified_kv::gather_promoted_bf16_from_tier2(
            args.ordinal,
            key_bf16.as_ptr(),
            value_bf16.as_ptr(),
            promote_index,
            value_promote_index,
            promoted_key_bf16,
            promoted_value_bf16,
            args.block_size,
            context_tokens,
            max_promoted_blocks,
            max_promoted_value_blocks,
            gqa_group,
        )
        .context("promoted key/value gather")
    })?;
    time_phase(|p| &mut p.score_consistency, &mut || {
        certified_kv::score_consistency(
            args.ordinal,
            query,
            key_int8,
            key_scale,
            key_zero,
            promoted_key_bf16,
            promote_index,
            score_consistency_flags,
            args.block_size,
            max_promoted_blocks,
            gqa_group,
            q_scale,
            args.eps_guard,
        )
        .context("score consistency")
    })?;
    time_phase(|p| &mut p.selected_fp16_log_masses, &mut || {
        certified_kv::selected_fp16_log_masses(
            args.ordinal,
            query,
            promoted_key_bf16,
            promote_index,
            selected_fp16_log_masses,
            args.block_size,
            max_promoted_blocks,
            q_scale,
        )
        .context("selected FP16 log masses")
    })?;
    time_phase(|p| &mut p.ranking_flags_device, &mut || {
        certified_kv::ranking_flags_device(
            args.ordinal,
            block_max,
            block_sum,
            delta_blocks,
            selected_fp16_log_masses,
            promote_index,
            fallback_flags,
            max_promoted_blocks,
        )
        .context("ranking flags")
    })?;
    time_phase(|p| &mut p.attend_mixed_key_int4, &mut || {
        certified_kv::attend_mixed_key_int4_with_bf16_tail_strided(
            args.ordinal,
            query,
            key_int8,
            key_scale,
            key_zero,
            promoted_key_bf16,
            promote_index,
            promoted_value_bf16,
            value_promote_index,
            value_int4,
            value_scale,
            value_zero,
            None,
            None,
            context_tokens,
            args.block_size,
            args.value_group_size,
            gqa_group,
            q_scale,
            score_scratch,
            output_bf16,
        )
        .context("mixed-key INT4 attend")
    })?;
    time_phase(|p| &mut p.block_masses_from_probs, &mut || {
        certified_kv::block_masses_from_token_probs(
            args.ordinal,
            score_scratch,
            block_mass,
            args.block_size,
        )
        .context("block masses")
    })?;
    time_phase(|p| &mut p.value_promotions, &mut || {
        certified_kv::value_promotions_from_block_masses(
            args.ordinal,
            block_mass,
            value_error,
            Some(fallback_flags),
            value_promote_index,
            value_kv_counters,
            any_value_promoted,
            value_head_promoted_flags,
            e_val_by_head,
            gqa_group,
            args.v_tol,
        )
        .context("value promotions")
    })?;
    time_phase(|p| &mut p.gather_promoted_values_bf16, &mut || {
        certified_kv::gather_promoted_values_bf16_from_tier2(
            args.ordinal,
            value_bf16.as_ptr(),
            value_promote_index,
            promoted_value_bf16,
            args.block_size,
            context_tokens,
            max_promoted_value_blocks,
        )
        .context("promoted value gather")
    })?;

    Ok(())
}

fn parse_contexts(raw: &str) -> Result<Vec<usize>> {
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid context value {:?}", part))
        })
        .collect()
}

fn make_bf16_bytes(elems: usize, mut seed: u64, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(elems * 2);
    for i in 0..elems {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((seed >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
        let trend = ((i % 127) as f32 - 63.0) * 0.0005;
        let value = (unit * 2.0 - 1.0) * scale + trend;
        out.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    out
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn scale_phase(phase: &mut PhaseMs, scale: f64) {
    phase.score_blocks_int8 *= scale;
    phase.select_blocks_device *= scale;
    phase.gather_promoted_bf16 *= scale;
    phase.score_consistency *= scale;
    phase.selected_fp16_log_masses *= scale;
    phase.ranking_flags_device *= scale;
    phase.attend_mixed_key_int4 *= scale;
    phase.block_masses_from_probs *= scale;
    phase.value_promotions *= scale;
    phase.gather_promoted_values_bf16 *= scale;
}
