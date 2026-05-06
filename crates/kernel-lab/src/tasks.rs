use crate::registry::TaskDef;
use crate::run::{summarize_times_us, CaseResult, KernelLabConfig, TaskResult, SCHEMA_VERSION};
use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::{prefill_ffi, qwen36_moe};
use std::collections::BTreeMap;
use std::time::Instant;

const GROUP_SIZE: usize = 128;

#[derive(Clone, Copy)]
struct AttnShape {
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    seed: usize,
}

#[derive(Clone, Copy)]
struct RouterShape {
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    seed: u64,
}

#[derive(Clone, Copy)]
struct ExpertShape {
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    moe_intermediate: usize,
    seed: u64,
}

struct TimedSamples {
    source: &'static str,
    us: Vec<f64>,
}

pub fn qwen35_full_attention_prefill(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen35.full_attention_prefill").unwrap();
    let shapes = [
        AttnShape {
            q_heads: 4,
            kv_heads: 2,
            q_len: 16,
            kv_len: 16,
            head_dim: 256,
            seed: 0xCAFE,
        },
        AttnShape {
            q_heads: 4,
            kv_heads: 2,
            q_len: 17,
            kv_len: 17,
            head_dim: 256,
            seed: 0xCAFE,
        },
        AttnShape {
            q_heads: 4,
            kv_heads: 2,
            q_len: 16,
            kv_len: 256,
            head_dim: 256,
            seed: 0xCAFE,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_qwen35_attn_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_batched_prefill_attn_full(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.batched_prefill_attn_full").unwrap();
    let shapes = [
        AttnShape {
            q_heads: 16,
            kv_heads: 2,
            q_len: 16,
            kv_len: 16,
            head_dim: 256,
            seed: 0xC36E,
        },
        AttnShape {
            q_heads: 16,
            kv_heads: 2,
            q_len: 33,
            kv_len: 64,
            head_dim: 256,
            seed: 0xC36E,
        },
        AttnShape {
            q_heads: 16,
            kv_heads: 2,
            q_len: 64,
            kv_len: 64,
            head_dim: 256,
            seed: 0xC36E,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_qwen36_attn_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_batched_prefill_attn_full_stress(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.batched_prefill_attn_full.stress").unwrap();
    let shapes = [
        AttnShape {
            q_heads: 16,
            kv_heads: 2,
            q_len: 128,
            kv_len: 512,
            head_dim: 256,
            seed: 0x5A36,
        },
        AttnShape {
            q_heads: 16,
            kv_heads: 2,
            q_len: 256,
            kv_len: 1024,
            head_dim: 256,
            seed: 0x5A37,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_qwen36_attn_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_router_permute(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.router_permute").unwrap();
    let shapes = [
        RouterShape {
            n_tokens: 1,
            top_k: 8,
            num_experts: 256,
            seed: 0x1,
        },
        RouterShape {
            n_tokens: 64,
            top_k: 8,
            num_experts: 256,
            seed: 0x3,
        },
        RouterShape {
            n_tokens: 128,
            top_k: 8,
            num_experts: 256,
            seed: 0x4,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_router_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_router_permute_stress(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.router_permute.stress").unwrap();
    let shapes = [
        RouterShape {
            n_tokens: 512,
            top_k: 8,
            num_experts: 256,
            seed: 0x44,
        },
        RouterShape {
            n_tokens: 1024,
            top_k: 8,
            num_experts: 256,
            seed: 0x45,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_router_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_grouped_expert_int4(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.grouped_expert_int4").unwrap();
    let shapes = [
        ExpertShape {
            n_tokens: 4,
            top_k: 2,
            num_experts: 8,
            hidden: 128,
            moe_intermediate: 128,
            seed: 0x10,
        },
        ExpertShape {
            n_tokens: 16,
            top_k: 8,
            num_experts: 64,
            hidden: 512,
            moe_intermediate: 256,
            seed: 0x20,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_grouped_expert_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_grouped_expert_int4_stress(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.grouped_expert_int4.stress").unwrap();
    let shapes = [ExpertShape {
        n_tokens: 64,
        top_k: 8,
        num_experts: 256,
        hidden: 1024,
        moe_intermediate: 256,
        seed: 0x2036,
    }];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_grouped_expert_case(cfg, shape))
            .collect(),
    )
}

pub fn qwen36_unpermute_combine(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen36.unpermute_combine").unwrap();
    let shapes = [
        ExpertShape {
            n_tokens: 4,
            top_k: 2,
            num_experts: 8,
            hidden: 128,
            moe_intermediate: 128,
            seed: 0x42,
        },
        ExpertShape {
            n_tokens: 64,
            top_k: 8,
            num_experts: 256,
            hidden: 512,
            moe_intermediate: 128,
            seed: 0x43,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_unpermute_case(cfg, shape))
            .collect(),
    )
}

fn task_result(task: &TaskDef, cases: Result<Vec<CaseResult>>) -> Result<TaskResult> {
    let cases = cases?;
    let correct = cases.iter().all(|case| case.correct);
    Ok(TaskResult {
        schema_version: SCHEMA_VERSION,
        task_id: task.id.to_string(),
        family: task.family.to_string(),
        tags: task.tags.iter().map(|s| s.to_string()).collect(),
        required: task.required,
        correct,
        cases,
        error: None,
    })
}

fn ensure_hip(cfg: &KernelLabConfig) -> Result<()> {
    if cfg.backend != Backend::Hip {
        return Err(anyhow!(
            "kernel-lab v1 task requires HIP backend, got {}",
            cfg.backend
        ));
    }
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        return Err(anyhow!("HIP backend is not compiled"));
    }
    Ok(())
}

fn run_qwen35_attn_case(cfg: &KernelLabConfig, shape: AttnShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let batch = 1;
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let seqlen_offset = shape.kv_len.saturating_sub(shape.q_len);
    let (q_buf, k_buf, v_buf, q_f32, k_f32, v_f32) = make_attn_buffers(cfg.device, shape, batch)?;
    let mut out_buf = GpuBuffer::zeros(
        cfg.device,
        ScalarType::F32,
        &[batch, shape.q_heads, shape.q_len, shape.head_dim],
    )?;

    prefill_ffi::full_attention_prefill(
        cfg.device,
        ScalarType::BF16,
        batch,
        shape.q_heads,
        shape.kv_heads,
        shape.q_len,
        shape.kv_len,
        shape.head_dim,
        scale,
        seqlen_offset,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out_buf,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_f32(&out_buf)?;
    let want = cpu_attention_fp32(
        batch,
        shape.q_heads,
        shape.kv_heads,
        shape.q_len,
        shape.kv_len,
        shape.head_dim,
        scale,
        seqlen_offset,
        &q_f32,
        &k_f32,
        &v_f32,
    );
    let (max_abs, max_rel) = max_abs_rel(&got, &want, 1e-6);
    let samples = measure_us(cfg, || {
        prefill_ffi::full_attention_prefill(
            cfg.device,
            ScalarType::BF16,
            batch,
            shape.q_heads,
            shape.kv_heads,
            shape.q_len,
            shape.kv_len,
            shape.head_dim,
            scale,
            seqlen_offset,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
        )
    })?;
    Ok(case_result(
        "qwen35_attn",
        attn_shape_map(shape),
        max_abs < 2e-2 && max_rel < 1e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

fn run_qwen36_attn_case(cfg: &KernelLabConfig, shape: AttnShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let batch = 1;
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let seqlen_offset = shape.kv_len.saturating_sub(shape.q_len);
    let (q_buf, k_buf, v_buf, q_f32, k_f32, v_f32) = make_attn_buffers(cfg.device, shape, batch)?;
    let mut out_buf = GpuBuffer::zeros(
        cfg.device,
        ScalarType::F32,
        &[batch, shape.q_heads, shape.q_len, shape.head_dim],
    )?;
    qwen36_moe::batched_prefill_attn_full_launch(
        cfg.device,
        batch,
        shape.q_heads,
        shape.kv_heads,
        shape.q_len,
        shape.kv_len,
        shape.head_dim,
        scale,
        seqlen_offset,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out_buf,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_f32(&out_buf)?;
    let want = cpu_attention_fp32(
        batch,
        shape.q_heads,
        shape.kv_heads,
        shape.q_len,
        shape.kv_len,
        shape.head_dim,
        scale,
        seqlen_offset,
        &q_f32,
        &k_f32,
        &v_f32,
    );
    let (max_abs, max_rel) = max_abs_rel(&got, &want, 1e-3);
    let samples = measure_us(cfg, || {
        qwen36_moe::batched_prefill_attn_full_launch(
            cfg.device,
            batch,
            shape.q_heads,
            shape.kv_heads,
            shape.q_len,
            shape.kv_len,
            shape.head_dim,
            scale,
            seqlen_offset,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
        )
    })?;
    Ok(case_result(
        "qwen36_attn",
        attn_shape_map(shape),
        max_abs < 2e-2 && max_rel < 5e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

fn run_router_case(cfg: &KernelLabConfig, shape: RouterShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total = shape.n_tokens * shape.top_k;
    let topk_idx_host = make_topk_idx(shape.n_tokens, shape.top_k, shape.num_experts, shape.seed);
    let topk_w_host = make_topk_weight(shape.n_tokens, shape.top_k, shape.seed + 0xA5A5);
    let topk_idx = upload_i32(cfg.device, &topk_idx_host, &[shape.n_tokens, shape.top_k])?;
    let topk_weight = upload_bf16(cfg.device, &topk_w_host, &[shape.n_tokens, shape.top_k])?;
    let mut offsets = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[shape.num_experts + 1])?;
    let mut p_token = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[total])?;
    let mut p_kpos = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[total])?;
    let mut p_weight = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[total])?;

    qwen36_moe::batched_prefill_router_permute_launch(
        cfg.device,
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx,
        &topk_weight,
        &mut offsets,
        &mut p_token,
        &mut p_kpos,
        &mut p_weight,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got_offsets = download_i32(&offsets)?;
    let got_token = download_i32(&p_token)?;
    let got_kpos = download_i32(&p_kpos)?;
    let got_w = download_bf16(&p_weight)?;
    let (want_offsets, want_token, want_kpos, want_w) = cpu_router_reference(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx_host,
        &topk_w_host,
    );
    let exact = router_exact(
        shape,
        &got_offsets,
        &got_token,
        &got_kpos,
        &got_w,
        &want_offsets,
        &want_token,
        &want_kpos,
        &want_w,
    );
    let samples = measure_us(cfg, || {
        qwen36_moe::batched_prefill_router_permute_launch(
            cfg.device,
            shape.n_tokens,
            shape.top_k,
            shape.num_experts,
            &topk_idx,
            &topk_weight,
            &mut offsets,
            &mut p_token,
            &mut p_kpos,
            &mut p_weight,
        )
    })?;
    Ok(case_result(
        "router_permute",
        router_shape_map(shape),
        exact,
        None,
        None,
        None,
        Some(exact),
        cfg,
        &samples,
    ))
}

fn run_grouped_expert_case(cfg: &KernelLabConfig, shape: ExpertShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total_rows = shape.n_tokens * shape.top_k;
    let x_host = make_x_norm(shape.n_tokens, shape.hidden, shape.seed);
    let topk_idx = make_topk_idx(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.seed + 0x101,
    );
    let (offsets_host, perm_token_host) =
        cpu_router_permute_indices(shape.n_tokens, shape.top_k, shape.num_experts, &topk_idx);
    let (gu_w, gu_s, gu_z, dp_w, dp_s, dp_z) = make_expert_weights(shape);

    let x = upload_bf16(cfg.device, &x_host, &[shape.n_tokens, shape.hidden])?;
    let offsets = upload_i32(cfg.device, &offsets_host, &[shape.num_experts + 1])?;
    let perm_token = upload_i32(cfg.device, &perm_token_host, &[total_rows])?;
    let two_i = 2 * shape.moe_intermediate;
    let gu_w_buf = upload_u8(
        cfg.device,
        &gu_w,
        &[shape.num_experts, two_i, shape.hidden / 2],
    )?;
    let gu_s_buf = upload_bf16(
        cfg.device,
        &gu_s,
        &[
            shape.num_experts,
            two_i / GROUP_SIZE,
            shape.hidden / GROUP_SIZE,
        ],
    )?;
    let gu_z_buf = upload_bf16(
        cfg.device,
        &gu_z,
        &[
            shape.num_experts,
            two_i / GROUP_SIZE,
            shape.hidden / GROUP_SIZE,
        ],
    )?;
    let dp_w_buf = upload_u8(
        cfg.device,
        &dp_w,
        &[shape.num_experts, shape.hidden, shape.moe_intermediate / 2],
    )?;
    let dp_s_buf = upload_bf16(
        cfg.device,
        &dp_s,
        &[
            shape.num_experts,
            shape.hidden / GROUP_SIZE,
            shape.moe_intermediate / GROUP_SIZE,
        ],
    )?;
    let dp_z_buf = upload_bf16(
        cfg.device,
        &dp_z,
        &[
            shape.num_experts,
            shape.hidden / GROUP_SIZE,
            shape.moe_intermediate / GROUP_SIZE,
        ],
    )?;
    let mut out = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[total_rows, shape.hidden])?;
    let mut counters = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[1])?;
    qwen36_moe::batched_prefill_grouped_expert_launch(
        cfg.device,
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.moe_intermediate,
        GROUP_SIZE,
        &x,
        &offsets,
        &perm_token,
        &gu_w_buf,
        &gu_s_buf,
        &gu_z_buf,
        &dp_w_buf,
        &dp_s_buf,
        &dp_z_buf,
        &mut out,
        &mut counters,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&out)?;
    let want = cpu_grouped_expert(
        shape,
        &x_host,
        &perm_token_host,
        &offsets_host,
        &gu_w,
        &gu_s,
        &gu_z,
        &dp_w,
        &dp_s,
        &dp_z,
    );
    let (max_abs_norm, min_cos) = vector_abs_norm_and_min_cos(&got, &want, shape.hidden);
    let samples = measure_us(cfg, || {
        gpu_hal::memset_zeros(
            cfg.device,
            counters.as_mut_ptr(),
            counters.elem_count() * std::mem::size_of::<u32>(),
        )?;
        qwen36_moe::batched_prefill_grouped_expert_launch(
            cfg.device,
            shape.n_tokens,
            shape.top_k,
            shape.num_experts,
            shape.hidden,
            shape.moe_intermediate,
            GROUP_SIZE,
            &x,
            &offsets,
            &perm_token,
            &gu_w_buf,
            &gu_s_buf,
            &gu_z_buf,
            &dp_w_buf,
            &dp_s_buf,
            &dp_z_buf,
            &mut out,
            &mut counters,
        )
    })?;
    Ok(case_result(
        "grouped_expert_int4",
        expert_shape_map(shape),
        max_abs_norm < 2e-2 && min_cos >= 0.999,
        Some(max_abs_norm),
        None,
        Some(min_cos),
        None,
        cfg,
        &samples,
    ))
}

fn run_unpermute_case(cfg: &KernelLabConfig, shape: ExpertShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total = shape.n_tokens * shape.top_k;
    let topk_idx = make_topk_idx(shape.n_tokens, shape.top_k, shape.num_experts, shape.seed);
    let topk_w = make_topk_weight(shape.n_tokens, shape.top_k, shape.seed + 0x55);
    let (_offsets, perm_token, perm_kpos, perm_weight) = cpu_router_reference(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx,
        &topk_w,
    );
    let mut inverse = vec![0i32; total];
    for row in 0..total {
        inverse[perm_token[row] as usize * shape.top_k + perm_kpos[row] as usize] = row as i32;
    }
    let expert_out_host = make_bf16(total * shape.hidden, shape.seed + 0x777).0;
    let want = cpu_unpermute_combine(
        shape.n_tokens,
        shape.top_k,
        shape.hidden,
        &inverse,
        &perm_weight,
        &expert_out_host,
    );

    let inverse_buf = upload_i32(cfg.device, &inverse, &[total])?;
    let weight_buf = upload_bf16(cfg.device, &perm_weight, &[total])?;
    let expert_buf = upload_bf16(cfg.device, &expert_out_host, &[total, shape.hidden])?;
    let mut combined = GpuBuffer::zeros(
        cfg.device,
        ScalarType::BF16,
        &[shape.n_tokens, shape.hidden],
    )?;
    qwen36_moe::batched_prefill_unpermute_combine_launch(
        cfg.device,
        shape.n_tokens,
        shape.top_k,
        shape.hidden,
        &inverse_buf,
        &weight_buf,
        &expert_buf,
        &mut combined,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&combined)?;
    let (max_abs, max_rel) = max_abs_rel_bf16(&got, &want, 1e-3);
    let samples = measure_us(cfg, || {
        qwen36_moe::batched_prefill_unpermute_combine_launch(
            cfg.device,
            shape.n_tokens,
            shape.top_k,
            shape.hidden,
            &inverse_buf,
            &weight_buf,
            &expert_buf,
            &mut combined,
        )
    })?;
    let mut shape_map = expert_shape_map(shape);
    shape_map.remove("moe_intermediate");
    shape_map.remove("num_experts");
    Ok(case_result(
        "unpermute_combine",
        shape_map,
        max_abs < 2e-2 && max_rel < 5e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

fn measure_us<F>(cfg: &KernelLabConfig, mut f: F) -> Result<TimedSamples>
where
    F: FnMut() -> Result<(), gpu_hal::GpuError>,
{
    for _ in 0..cfg.warmup {
        f()?;
        gpu_hal::sync(cfg.device)?;
    }

    if let Ok(samples) = measure_gpu_event_us(cfg, &mut f) {
        return Ok(TimedSamples {
            source: "hip_event",
            us: samples,
        });
    }

    let mut us = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let start = Instant::now();
        f()?;
        gpu_hal::sync(cfg.device)?;
        us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    Ok(TimedSamples {
        source: "wall_sync",
        us,
    })
}

fn measure_gpu_event_us<F>(cfg: &KernelLabConfig, f: &mut F) -> Result<Vec<f64>>
where
    F: FnMut() -> Result<(), gpu_hal::GpuError>,
{
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let start = gpu_hal::GpuEvent::new(cfg.device)?;
        let end = gpu_hal::GpuEvent::new(cfg.device)?;
        start.record()?;
        f()?;
        end.record()?;
        end.synchronize()?;
        samples.push(gpu_hal::GpuEvent::elapsed_ms(&start, &end)? as f64 * 1000.0);
    }
    Ok(samples)
}

fn case_result(
    name: &str,
    shape: BTreeMap<String, usize>,
    correct: bool,
    max_abs: Option<f32>,
    max_rel: Option<f32>,
    min_cos: Option<f32>,
    exact: Option<bool>,
    cfg: &KernelLabConfig,
    samples: &TimedSamples,
) -> CaseResult {
    let (median_us, mean_us, min_us, p95_us) = summarize_times_us(&samples.us);
    CaseResult {
        name: name.to_string(),
        shape,
        correct,
        max_abs,
        max_rel,
        min_cos,
        exact,
        warmup: cfg.warmup,
        iters: cfg.iters,
        timing_source: samples.source.to_string(),
        median_us,
        mean_us,
        min_us,
        p95_us,
    }
}

fn make_attn_buffers(
    ordinal: usize,
    shape: AttnShape,
    batch: usize,
) -> Result<(
    GpuBuffer,
    GpuBuffer,
    GpuBuffer,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
)> {
    let q_elems = batch * shape.q_heads * shape.q_len * shape.head_dim;
    let k_elems = batch * shape.kv_heads * shape.kv_len * shape.head_dim;
    let v_elems = batch * shape.kv_heads * shape.kv_len * shape.head_dim;
    let (q_bf, q_f32) = make_bf16(q_elems, shape.seed as u64);
    let (k_bf, k_f32) = make_bf16(k_elems, shape.seed as u64 + 1000);
    let (v_bf, v_f32) = make_bf16(v_elems, shape.seed as u64 + 2000);
    Ok((
        upload_bf16(
            ordinal,
            &q_bf,
            &[batch, shape.q_heads, shape.q_len, shape.head_dim],
        )?,
        upload_bf16(
            ordinal,
            &k_bf,
            &[batch, shape.kv_heads, shape.kv_len, shape.head_dim],
        )?,
        upload_bf16(
            ordinal,
            &v_bf,
            &[batch, shape.kv_heads, shape.kv_len, shape.head_dim],
        )?,
        q_f32,
        k_f32,
        v_f32,
    ))
}

fn upload_bf16(ordinal: usize, host: &[half::bf16], shape: &[usize]) -> Result<GpuBuffer> {
    assert_eq!(host.len(), shape.iter().product::<usize>());
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, shape)?;
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )?;
    Ok(buf)
}

fn upload_i32(ordinal: usize, host: &[i32], shape: &[usize]) -> Result<GpuBuffer> {
    assert_eq!(host.len(), shape.iter().product::<usize>());
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::U32, shape)?;
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )?;
    Ok(buf)
}

fn upload_u8(ordinal: usize, host: &[u8], shape: &[usize]) -> Result<GpuBuffer> {
    Ok(GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::U8,
        shape,
        host,
    )?)
}

fn download_f32(buf: &GpuBuffer) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; buf.elem_count() * 4];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn download_i32(buf: &GpuBuffer) -> Result<Vec<i32>> {
    let mut bytes = vec![0u8; buf.elem_count() * 4];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn download_bf16(buf: &GpuBuffer) -> Result<Vec<half::bf16>> {
    let mut bytes = vec![0u8; buf.elem_count() * 2];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn make_bf16(n: usize, seed: u64) -> (Vec<half::bf16>, Vec<f32>) {
    let bf: Vec<_> = (0..n)
        .map(|i| {
            let v = ((i as u64 + seed) as f32 * 0.0017).sin() * 0.4 + 0.05;
            half::bf16::from_f32(v)
        })
        .collect();
    let f32_round = bf.iter().map(|x| x.to_f32()).collect();
    (bf, f32_round)
}

fn cpu_attention_fp32(
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let groups = q_heads / kv_heads;
    let mut out = vec![0.0f32; batch * q_heads * q_len * head_dim];
    for b in 0..batch {
        for hq in 0..q_heads {
            let hk = hq / groups;
            for qi in 0..q_len {
                let limit = (seqlen_offset + qi + 1).min(kv_len);
                let q_off = ((b * q_heads + hq) * q_len + qi) * head_dim;
                let k_head = (b * kv_heads + hk) * kv_len * head_dim;
                let v_head = (b * kv_heads + hk) * kv_len * head_dim;
                let mut scores = vec![0f32; limit];
                for kp in 0..limit {
                    let k_row = k_head + kp * head_dim;
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q[q_off + d] * k[k_row + d];
                    }
                    scores[kp] = s * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    denom += *s;
                }
                let out_off = ((b * q_heads + hq) * q_len + qi) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for kp in 0..limit {
                        acc += scores[kp] * v[v_head + kp * head_dim + d];
                    }
                    out[out_off + d] = acc / denom.max(1e-12);
                }
            }
        }
    }
    out
}

fn max_abs_rel(got: &[f32], want: &[f32], rel_floor: f32) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(rel_floor);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

fn max_abs_rel_bf16(got: &[half::bf16], want: &[half::bf16], rel_floor: f32) -> (f32, f32) {
    let got_f32: Vec<_> = got.iter().map(|x| x.to_f32()).collect();
    let want_f32: Vec<_> = want.iter().map(|x| x.to_f32()).collect();
    max_abs_rel(&got_f32, &want_f32, rel_floor)
}

fn lcg(state: u64) -> (u64, u64) {
    let next = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (next, next >> 32)
}

fn make_topk_idx(n_tokens: usize, top_k: usize, num_experts: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * top_k);
    for _ in 0..n_tokens {
        let mut chosen = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let (next, raw) = lcg(state);
            state = next;
            let cand = (raw as usize % num_experts) as i32;
            if !chosen.contains(&cand) {
                chosen.push(cand);
            }
        }
        out.extend_from_slice(&chosen);
    }
    out
}

fn make_topk_weight(n_tokens: usize, top_k: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * top_k);
    for _ in 0..n_tokens * top_k {
        let (next, raw) = lcg(state);
        state = next;
        let v = ((raw % 10_000) as f32) / 10_000.0;
        out.push(half::bf16::from_f32(v * 0.5 + 0.01));
    }
    out
}

fn cpu_router_reference(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    topk_idx: &[i32],
    topk_weight: &[half::bf16],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<half::bf16>) {
    let total = n_tokens * top_k;
    let mut counts = vec![0i32; num_experts];
    for &e in topk_idx {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut token = vec![0i32; total];
    let mut kpos = vec![0i32; total];
    let mut weight = vec![half::bf16::ZERO; total];
    let mut cursors = vec![0i32; num_experts];
    for entry in 0..total {
        let e = topk_idx[entry] as usize;
        let dst = (offsets[e] + cursors[e]) as usize;
        token[dst] = (entry / top_k) as i32;
        kpos[dst] = (entry % top_k) as i32;
        weight[dst] = topk_weight[entry];
        cursors[e] += 1;
    }
    (offsets, token, kpos, weight)
}

fn cpu_router_permute_indices(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    topk_idx: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let weights = vec![half::bf16::ZERO; n_tokens * top_k];
    let (offsets, token, _, _) =
        cpu_router_reference(n_tokens, top_k, num_experts, topk_idx, &weights);
    (offsets, token)
}

fn router_exact(
    shape: RouterShape,
    got_offsets: &[i32],
    got_token: &[i32],
    got_kpos: &[i32],
    got_w: &[half::bf16],
    want_offsets: &[i32],
    want_token: &[i32],
    want_kpos: &[i32],
    want_w: &[half::bf16],
) -> bool {
    if got_offsets != want_offsets
        || got_offsets[shape.num_experts] as usize != shape.n_tokens * shape.top_k
    {
        return false;
    }
    for e in 0..shape.num_experts {
        let lo = got_offsets[e] as usize;
        let hi = got_offsets[e + 1] as usize;
        let mut got: Vec<_> = (lo..hi)
            .map(|i| (got_token[i], got_kpos[i], got_w[i].to_bits()))
            .collect();
        let mut want: Vec<_> = (lo..hi)
            .map(|i| (want_token[i], want_kpos[i], want_w[i].to_bits()))
            .collect();
        got.sort();
        want.sort();
        if got != want {
            return false;
        }
    }
    true
}

fn make_x_norm(n_tokens: usize, hidden: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * hidden);
    for _ in 0..n_tokens * hidden {
        let (next, raw) = lcg(state);
        state = next;
        let v = ((raw % 10_000) as f32) / 10_000.0 - 0.5;
        out.push(half::bf16::from_f32(v * 0.4));
    }
    out
}

fn bf16_round_rne_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounding_bias = 0x7FFF_u32 + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(rounding_bias)) & 0xFFFF_0000)
}

fn int4_dequant_scalar(nibble: u32, scale: half::bf16, zero: half::bf16) -> f32 {
    bf16_round_rne_f32((nibble as f32) * scale.to_f32() - zero.to_f32() * scale.to_f32())
}

fn make_int4_slab(
    out_rows: usize,
    in_cols: usize,
    gs: usize,
    seed: u64,
) -> (Vec<u8>, Vec<half::bf16>, Vec<half::bf16>) {
    let scale_rows = out_rows / gs;
    let scale_cols = in_cols / gs;
    let mut state = seed;
    let mut packed = vec![0u8; out_rows * (in_cols / 2)];
    for byte in packed.iter_mut() {
        let (next, raw) = lcg(state);
        state = next;
        *byte = (raw & 0xFF) as u8;
    }
    let mut scales = vec![half::bf16::ZERO; scale_rows * scale_cols];
    let mut zeros = vec![half::bf16::ZERO; scale_rows * scale_cols];
    for i in 0..scales.len() {
        let (next, raw) = lcg(state);
        state = next;
        scales[i] = half::bf16::from_f32(0.001 + ((raw % 1000) as f32 / 1000.0) * 0.02);
        let (next, raw) = lcg(state);
        state = next;
        zeros[i] = half::bf16::from_f32((raw % 16) as f32);
    }
    (packed, scales, zeros)
}

type ExpertWeights = (
    Vec<u8>,
    Vec<half::bf16>,
    Vec<half::bf16>,
    Vec<u8>,
    Vec<half::bf16>,
    Vec<half::bf16>,
);

fn make_expert_weights(shape: ExpertShape) -> ExpertWeights {
    let two_i = 2 * shape.moe_intermediate;
    let mut gu_w = Vec::new();
    let mut gu_s = Vec::new();
    let mut gu_z = Vec::new();
    let mut dp_w = Vec::new();
    let mut dp_s = Vec::new();
    let mut dp_z = Vec::new();
    for e in 0..shape.num_experts {
        let (w, s, z) = make_int4_slab(
            two_i,
            shape.hidden,
            GROUP_SIZE,
            shape.seed + 0x200 + e as u64,
        );
        gu_w.extend(w);
        gu_s.extend(s);
        gu_z.extend(z);
        let (w, s, z) = make_int4_slab(
            shape.hidden,
            shape.moe_intermediate,
            GROUP_SIZE,
            shape.seed + 0x300 + e as u64,
        );
        dp_w.extend(w);
        dp_s.extend(s);
        dp_z.extend(z);
    }
    (gu_w, gu_s, gu_z, dp_w, dp_s, dp_z)
}

fn int4_matvec_row(
    packed: &[u8],
    scales: &[half::bf16],
    zeros: &[half::bf16],
    in_cols: usize,
    gs: usize,
    row: usize,
    x: &[f32],
) -> f32 {
    let byte_cols = in_cols / 2;
    let scale_cols = in_cols / gs;
    let scale_row = row / gs;
    let row_byte_off = row * byte_cols;
    let mut acc = 0.0f32;
    let mut col = 0usize;
    while col < in_cols {
        for i in 0..8 {
            let c = col + i;
            let byte = packed[row_byte_off + c / 2];
            let nibble = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let g = c / gs;
            acc += int4_dequant_scalar(
                nibble as u32,
                scales[scale_row * scale_cols + g],
                zeros[scale_row * scale_cols + g],
            ) * x[c];
        }
        col += 8;
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn cpu_grouped_expert(
    shape: ExpertShape,
    x_norm: &[half::bf16],
    permuted_token_idx: &[i32],
    expert_offsets: &[i32],
    gu_w_all: &[u8],
    gu_s_all: &[half::bf16],
    gu_z_all: &[half::bf16],
    dp_w_all: &[u8],
    dp_s_all: &[half::bf16],
    dp_z_all: &[half::bf16],
) -> Vec<half::bf16> {
    let total_rows = shape.n_tokens * shape.top_k;
    let i_dim = shape.moe_intermediate;
    let two_i = 2 * i_dim;
    let gu_per_expert_packed = two_i * (shape.hidden / 2);
    let gu_per_expert_scales = (two_i / GROUP_SIZE) * (shape.hidden / GROUP_SIZE);
    let dp_per_expert_packed = shape.hidden * (i_dim / 2);
    let dp_per_expert_scales = (shape.hidden / GROUP_SIZE) * (i_dim / GROUP_SIZE);
    let mut out = vec![half::bf16::ZERO; total_rows * shape.hidden];
    for e in 0..shape.num_experts {
        let lo = expert_offsets[e] as usize;
        let hi = expert_offsets[e + 1] as usize;
        let gu_w = &gu_w_all[e * gu_per_expert_packed..(e + 1) * gu_per_expert_packed];
        let gu_s = &gu_s_all[e * gu_per_expert_scales..(e + 1) * gu_per_expert_scales];
        let gu_z = &gu_z_all[e * gu_per_expert_scales..(e + 1) * gu_per_expert_scales];
        let dp_w = &dp_w_all[e * dp_per_expert_packed..(e + 1) * dp_per_expert_packed];
        let dp_s = &dp_s_all[e * dp_per_expert_scales..(e + 1) * dp_per_expert_scales];
        let dp_z = &dp_z_all[e * dp_per_expert_scales..(e + 1) * dp_per_expert_scales];
        for row in lo..hi {
            let token_idx = permuted_token_idx[row] as usize;
            let x: Vec<f32> = (0..shape.hidden)
                .map(|c| bf16_round_rne_f32(x_norm[token_idx * shape.hidden + c].to_f32()))
                .collect();
            let mut gu = vec![0.0f32; two_i];
            for r in 0..two_i {
                gu[r] = int4_matvec_row(gu_w, gu_s, gu_z, shape.hidden, GROUP_SIZE, r, &x);
            }
            let mut mid = vec![0.0f32; i_dim];
            for k in 0..i_dim {
                let gp = gu[k];
                mid[k] = bf16_round_rne_f32((gp / (1.0 + (-gp).exp())) * gu[i_dim + k]);
            }
            for r in 0..shape.hidden {
                let val = int4_matvec_row(dp_w, dp_s, dp_z, i_dim, GROUP_SIZE, r, &mid);
                out[row * shape.hidden + r] = half::bf16::from_f32(bf16_round_rne_f32(val));
            }
        }
    }
    out
}

fn vector_abs_norm_and_min_cos(
    got: &[half::bf16],
    want: &[half::bf16],
    row_width: usize,
) -> (f32, f32) {
    let rows = got.len() / row_width;
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    let mut min_cos = 1.0f32;
    for row in 0..rows {
        let mut dot = 0.0f64;
        let mut nrm_g = 0.0f64;
        let mut nrm_w = 0.0f64;
        for c in 0..row_width {
            let g = got[row * row_width + c].to_f32();
            let w = want[row * row_width + c].to_f32();
            max_abs = max_abs.max((g - w).abs());
            max_mag = max_mag.max(g.abs().max(w.abs()));
            dot += (g as f64) * (w as f64);
            nrm_g += (g as f64) * (g as f64);
            nrm_w += (w as f64) * (w as f64);
        }
        min_cos = min_cos.min((dot / (nrm_g.sqrt() * nrm_w.sqrt()).max(1e-12)) as f32);
    }
    (max_abs / max_mag.max(1e-3), min_cos)
}

fn cpu_unpermute_combine(
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
    inverse: &[i32],
    permuted_weight: &[half::bf16],
    expert_out: &[half::bf16],
) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; n_tokens * hidden];
    for t in 0..n_tokens {
        for c in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let row = inverse[t * top_k + k] as usize;
                acc += permuted_weight[row].to_f32() * expert_out[row * hidden + c].to_f32();
            }
            out[t * hidden + c] = half::bf16::from_f32(bf16_round_rne_f32(acc));
        }
    }
    out
}

fn attn_shape_map(shape: AttnShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("q_heads".into(), shape.q_heads),
        ("kv_heads".into(), shape.kv_heads),
        ("q_len".into(), shape.q_len),
        ("kv_len".into(), shape.kv_len),
        ("head_dim".into(), shape.head_dim),
    ])
}

fn router_shape_map(shape: RouterShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("n_tokens".into(), shape.n_tokens),
        ("top_k".into(), shape.top_k),
        ("num_experts".into(), shape.num_experts),
    ])
}

fn expert_shape_map(shape: ExpertShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("n_tokens".into(), shape.n_tokens),
        ("top_k".into(), shape.top_k),
        ("num_experts".into(), shape.num_experts),
        ("hidden".into(), shape.hidden),
        ("moe_intermediate".into(), shape.moe_intermediate),
    ])
}
