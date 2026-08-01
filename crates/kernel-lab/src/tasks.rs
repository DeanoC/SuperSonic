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

#[derive(Clone, Copy)]
struct Int4MatvecShape {
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    sparse_cols: usize,
    seed: u64,
}

#[derive(Clone, Copy)]
struct RmsNormShape {
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    seed: u64,
}

#[derive(Clone, Copy)]
struct RopeShape {
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos_offset: usize,
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

pub fn qwen35_int4_matvec(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen35.int4_matvec").unwrap();
    task_result(
        task,
        int4_matvec_shapes()
            .iter()
            .map(|&shape| run_int4_matvec_case(cfg, shape, Int4Sidecar::None))
            .collect(),
    )
}

pub fn qwen35_int4_awq_dense_matvec(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen35.int4_awq_dense_matvec").unwrap();
    task_result(
        task,
        int4_matvec_shapes()
            .iter()
            .map(|&shape| run_int4_matvec_case(cfg, shape, Int4Sidecar::DenseAwq))
            .collect(),
    )
}

pub fn qwen35_int4_awq_sparse_outlier_matvec(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("qwen35.int4_awq_sparse_outlier_matvec").unwrap();
    task_result(
        task,
        int4_matvec_shapes()
            .iter()
            .map(|&shape| run_int4_matvec_case(cfg, shape, Int4Sidecar::SparseOutlier))
            .collect(),
    )
}

pub fn functional_rmsnorm_bf16(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("functional.rmsnorm_bf16").unwrap();
    let shapes = [
        RmsNormShape {
            n_rows: 1,
            n_cols: 64,
            eps: 1e-6,
            seed: 0xF001,
        },
        RmsNormShape {
            n_rows: 3,
            n_cols: 256,
            eps: 1e-5,
            seed: 0xF002,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_rmsnorm_bf16_case(cfg, shape))
            .collect(),
    )
}

pub fn functional_rope_bf16(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("functional.rope_bf16").unwrap();
    let shapes = [
        RopeShape {
            seq_len: 4,
            num_heads: 2,
            head_dim: 16,
            rotary_dim: 16,
            pos_offset: 0,
            seed: 0xF101,
        },
        RopeShape {
            seq_len: 5,
            num_heads: 3,
            head_dim: 32,
            rotary_dim: 24,
            pos_offset: 7,
            seed: 0xF102,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_rope_bf16_case(cfg, shape))
            .collect(),
    )
}

pub fn functional_int4_dequant_matvec(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("functional.int4_dequant_matvec").unwrap();
    let shapes = [
        Int4MatvecShape {
            m: 1,
            n: 128,
            k: 128,
            group_size: GROUP_SIZE,
            sparse_cols: 8,
            seed: 0xF201,
        },
        Int4MatvecShape {
            m: 2,
            n: 256,
            k: 256,
            group_size: GROUP_SIZE,
            sparse_cols: 8,
            seed: 0xF202,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_int4_matvec_case(cfg, shape, Int4Sidecar::None))
            .collect(),
    )
}

pub fn functional_qwen36_moe_route_expert_combine(cfg: &KernelLabConfig) -> Result<TaskResult> {
    let task = crate::find_task("functional.qwen36_moe_route_expert_combine").unwrap();
    let shapes = [
        ExpertShape {
            n_tokens: 3,
            top_k: 2,
            num_experts: 4,
            hidden: 128,
            moe_intermediate: 128,
            seed: 0xF301,
        },
        ExpertShape {
            n_tokens: 5,
            top_k: 2,
            num_experts: 8,
            hidden: 128,
            moe_intermediate: 128,
            seed: 0xF302,
        },
    ];
    task_result(
        task,
        shapes
            .iter()
            .map(|&shape| run_qwen36_moe_pipeline_case(cfg, shape))
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
        description: task.description.to_string(),
        tags: task.tags.iter().map(|s| s.to_string()).collect(),
        backend_support: task
            .backend_support
            .iter()
            .map(|backend| backend.to_string())
            .collect(),
        correctness: task.correctness.to_string(),
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

#[derive(Clone, Copy)]
enum Int4Sidecar {
    None,
    DenseAwq,
    SparseOutlier,
}

fn int4_matvec_shapes() -> [Int4MatvecShape; 2] {
    [
        Int4MatvecShape {
            m: 1,
            n: 512,
            k: 512,
            group_size: GROUP_SIZE,
            sparse_cols: 32,
            seed: 0xA35A,
        },
        Int4MatvecShape {
            m: 1,
            n: 2048,
            k: 2048,
            group_size: GROUP_SIZE,
            sparse_cols: 64,
            seed: 0xA35B,
        },
    ]
}

fn run_rmsnorm_bf16_case(cfg: &KernelLabConfig, shape: RmsNormShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total = shape.n_rows * shape.n_cols;
    let (input_host, _) = make_bf16(total, shape.seed);
    let weight_host = make_rms_weight(shape.n_cols, shape.seed + 0x100);
    let input = upload_bf16(cfg.device, &input_host, &[shape.n_rows, shape.n_cols])?;
    let weight = upload_bf16(cfg.device, &weight_host, &[shape.n_cols])?;
    let mut out = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[shape.n_rows, shape.n_cols])?;

    prefill_ffi::rms_norm_rows_plain(
        cfg.device,
        ScalarType::BF16,
        shape.n_rows,
        shape.n_cols,
        shape.eps,
        &input,
        &weight,
        &mut out,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&out)?;
    let want = cpu_rms_norm_plain_bf16(
        shape.n_rows,
        shape.n_cols,
        shape.eps,
        &input_host,
        &weight_host,
    );
    let (max_abs, max_rel) = max_abs_rel_bf16(&got, &want, 1e-3);
    let samples = measure_us(cfg, || {
        prefill_ffi::rms_norm_rows_plain(
            cfg.device,
            ScalarType::BF16,
            shape.n_rows,
            shape.n_cols,
            shape.eps,
            &input,
            &weight,
            &mut out,
        )
    })?;
    Ok(case_result(
        "rmsnorm_bf16",
        rmsnorm_shape_map(shape),
        max_abs < 2e-2 && max_rel < 2e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

fn run_rope_bf16_case(cfg: &KernelLabConfig, shape: RopeShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total = shape.seq_len * shape.num_heads * shape.head_dim;
    let (data_host, _) = make_bf16(total, shape.seed);
    let (cos_host, sin_host) = make_rope_tables(
        shape.pos_offset + shape.seq_len,
        shape.rotary_dim / 2,
        shape.seed + 0x100,
    );
    let cos = upload_bf16(
        cfg.device,
        &cos_host,
        &[shape.pos_offset + shape.seq_len, shape.rotary_dim / 2],
    )?;
    let sin = upload_bf16(
        cfg.device,
        &sin_host,
        &[shape.pos_offset + shape.seq_len, shape.rotary_dim / 2],
    )?;
    let mut data = upload_bf16(
        cfg.device,
        &data_host,
        &[shape.seq_len, shape.num_heads, shape.head_dim],
    )?;

    prefill_ffi::apply_rope_prefill(
        cfg.device,
        ScalarType::BF16,
        shape.seq_len,
        shape.num_heads,
        shape.head_dim,
        shape.rotary_dim,
        &cos,
        &sin,
        shape.pos_offset,
        &mut data,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&data)?;
    let want = cpu_apply_rope_bf16(shape, &data_host, &cos_host, &sin_host);
    let (max_abs, max_rel) = max_abs_rel_bf16(&got, &want, 1e-3);
    let samples = measure_us(cfg, || {
        prefill_ffi::apply_rope_prefill(
            cfg.device,
            ScalarType::BF16,
            shape.seq_len,
            shape.num_heads,
            shape.head_dim,
            shape.rotary_dim,
            &cos,
            &sin,
            shape.pos_offset,
            &mut data,
        )
    })?;
    Ok(case_result(
        "rope_bf16",
        rope_shape_map(shape),
        max_abs < 2e-2 && max_rel < 2e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

fn run_int4_matvec_case(
    cfg: &KernelLabConfig,
    shape: Int4MatvecShape,
    sidecar: Int4Sidecar,
) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let batch = 1;
    let rows = batch * shape.m;
    let (lhs_host, lhs_f32) = make_bf16(rows * shape.k, shape.seed);
    let (rhs_int4, scales, zeros) =
        make_int4_slab(shape.n, shape.k, shape.group_size, shape.seed + 0x1000);
    let awq_host = match sidecar {
        Int4Sidecar::DenseAwq => Some(make_awq_inv_scale(shape.k, shape.seed + 0x2000)),
        _ => None,
    };
    let outlier_cols = make_outlier_cols(shape.k, shape.sparse_cols, shape.seed + 0x3000);
    let outlier_delta = make_sparse_delta(shape.n, shape.sparse_cols, shape.seed + 0x4000);

    let lhs = upload_bf16(cfg.device, &lhs_host, &[batch, shape.m, shape.k])?;
    let rhs = upload_u8(cfg.device, &rhs_int4, &[batch, shape.n, shape.k / 2])?;
    let scale = upload_bf16(
        cfg.device,
        &scales,
        &[shape.n / shape.group_size, shape.k / shape.group_size],
    )?;
    let zero = upload_bf16(
        cfg.device,
        &zeros,
        &[shape.n / shape.group_size, shape.k / shape.group_size],
    )?;
    let awq = awq_host
        .as_ref()
        .map(|host| upload_bf16(cfg.device, host, &[shape.k]))
        .transpose()?;
    let cols_buf = upload_u32(cfg.device, &outlier_cols, &[shape.sparse_cols])?;
    let delta_buf = upload_bf16(cfg.device, &outlier_delta, &[shape.n, shape.sparse_cols])?;
    let mut out = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[batch, shape.m, shape.n])?;

    prefill_ffi::matmul_rhs_transposed_int4(
        cfg.device,
        batch,
        shape.m,
        shape.n,
        shape.k,
        &lhs,
        &rhs,
        &scale,
        &zero,
        awq.as_ref(),
        shape.group_size,
        4,
        &mut out,
    )?;
    if matches!(sidecar, Int4Sidecar::SparseOutlier) {
        launch_int4_sparse_outlier_add(
            cfg.device,
            rows,
            shape.n,
            shape.k,
            shape.sparse_cols,
            &lhs,
            &cols_buf,
            &delta_buf,
            &mut out,
        )?;
    }
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&out)?;
    let mut want = cpu_int4_matmul(
        rows,
        shape.n,
        shape.k,
        shape.group_size,
        &lhs_f32,
        &rhs_int4,
        &scales,
        &zeros,
        awq_host.as_deref(),
    );
    if matches!(sidecar, Int4Sidecar::SparseOutlier) {
        cpu_sparse_outlier_add(
            rows,
            shape.n,
            shape.k,
            shape.sparse_cols,
            &lhs_f32,
            &outlier_cols,
            &outlier_delta,
            &mut want,
        );
    }
    let (max_abs, max_rel) = max_abs_rel_bf16(&got, &want, 1e-3);

    let samples = match sidecar {
        Int4Sidecar::SparseOutlier => measure_us(cfg, || {
            prefill_ffi::matmul_rhs_transposed_int4(
                cfg.device,
                batch,
                shape.m,
                shape.n,
                shape.k,
                &lhs,
                &rhs,
                &scale,
                &zero,
                None,
                shape.group_size,
                4,
                &mut out,
            )?;
            launch_int4_sparse_outlier_add(
                cfg.device,
                rows,
                shape.n,
                shape.k,
                shape.sparse_cols,
                &lhs,
                &cols_buf,
                &delta_buf,
                &mut out,
            )
        })?,
        _ => measure_us(cfg, || {
            prefill_ffi::matmul_rhs_transposed_int4(
                cfg.device,
                batch,
                shape.m,
                shape.n,
                shape.k,
                &lhs,
                &rhs,
                &scale,
                &zero,
                awq.as_ref(),
                shape.group_size,
                4,
                &mut out,
            )
        })?,
    };

    let name = match sidecar {
        Int4Sidecar::None => "int4_matvec",
        Int4Sidecar::DenseAwq => "int4_awq_dense_matvec",
        Int4Sidecar::SparseOutlier => "int4_awq_sparse_outlier_matvec",
    };
    Ok(case_result(
        name,
        int4_shape_map(shape),
        max_rel < 1e-2,
        Some(max_abs),
        Some(max_rel),
        None,
        None,
        cfg,
        &samples,
    ))
}

#[allow(clippy::too_many_arguments)]
#[cfg(kernel_lab_has_int4_sparse_outlier_add)]
fn launch_int4_sparse_outlier_add(
    ordinal: usize,
    rows: usize,
    n: usize,
    k: usize,
    sub_cols: usize,
    lhs: &GpuBuffer,
    outlier_cols: &GpuBuffer,
    outlier_delta: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), gpu_hal::GpuError> {
    prefill_ffi::int4_sparse_outlier_add(
        ordinal,
        rows,
        n,
        k,
        sub_cols,
        lhs,
        outlier_cols,
        outlier_delta,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(kernel_lab_has_int4_sparse_outlier_add))]
fn launch_int4_sparse_outlier_add(
    _ordinal: usize,
    _rows: usize,
    _n: usize,
    _k: usize,
    _sub_cols: usize,
    _lhs: &GpuBuffer,
    _outlier_cols: &GpuBuffer,
    _outlier_delta: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), gpu_hal::GpuError> {
    Err(gpu_hal::GpuError::InvalidArg(
        "kernel-ffi does not expose int4_sparse_outlier_add".to_string(),
    ))
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

fn run_qwen36_moe_pipeline_case(cfg: &KernelLabConfig, shape: ExpertShape) -> Result<CaseResult> {
    ensure_hip(cfg)?;
    let total = shape.n_tokens * shape.top_k;
    let x_host = make_x_norm(shape.n_tokens, shape.hidden, shape.seed);
    let topk_idx_host = make_topk_idx(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.seed + 0x101,
    );
    let topk_w_host = make_topk_weight(shape.n_tokens, shape.top_k, shape.seed + 0x55);
    let (want_offsets, want_token, want_kpos, want_weight) = cpu_router_reference(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx_host,
        &topk_w_host,
    );
    let (gu_w, gu_s, gu_z, dp_w, dp_s, dp_z) = make_expert_weights(shape);
    let want_expert = cpu_grouped_expert(
        shape,
        &x_host,
        &want_token,
        &want_offsets,
        &gu_w,
        &gu_s,
        &gu_z,
        &dp_w,
        &dp_s,
        &dp_z,
    );
    let want_inverse = inverse_permutation(shape.n_tokens, shape.top_k, &want_token, &want_kpos);
    let want = cpu_unpermute_combine(
        shape.n_tokens,
        shape.top_k,
        shape.hidden,
        &want_inverse,
        &want_weight,
        &want_expert,
    );

    let x = upload_bf16(cfg.device, &x_host, &[shape.n_tokens, shape.hidden])?;
    let topk_idx = upload_i32(cfg.device, &topk_idx_host, &[shape.n_tokens, shape.top_k])?;
    let topk_weight = upload_bf16(cfg.device, &topk_w_host, &[shape.n_tokens, shape.top_k])?;
    let mut offsets = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[shape.num_experts + 1])?;
    let mut perm_token = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[total])?;
    let mut perm_kpos = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[total])?;
    let mut perm_weight = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[total])?;
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
    let mut expert_out = GpuBuffer::zeros(cfg.device, ScalarType::BF16, &[total, shape.hidden])?;
    let mut counters = GpuBuffer::zeros(cfg.device, ScalarType::U32, &[1])?;
    let mut combined = GpuBuffer::zeros(
        cfg.device,
        ScalarType::BF16,
        &[shape.n_tokens, shape.hidden],
    )?;

    qwen36_moe::batched_prefill_router_permute_launch(
        cfg.device,
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx,
        &topk_weight,
        &mut offsets,
        &mut perm_token,
        &mut perm_kpos,
        &mut perm_weight,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got_offsets = download_i32(&offsets)?;
    let got_token = download_i32(&perm_token)?;
    let got_kpos = download_i32(&perm_kpos)?;
    let got_weight = download_bf16(&perm_weight)?;
    let router_ok = router_exact(
        RouterShape {
            n_tokens: shape.n_tokens,
            top_k: shape.top_k,
            num_experts: shape.num_experts,
            seed: shape.seed,
        },
        &got_offsets,
        &got_token,
        &got_kpos,
        &got_weight,
        &want_offsets,
        &want_token,
        &want_kpos,
        &want_weight,
    );
    let inverse = inverse_permutation(shape.n_tokens, shape.top_k, &got_token, &got_kpos);
    let inverse_buf = upload_i32(cfg.device, &inverse, &[total])?;

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
        &mut expert_out,
        &mut counters,
    )?;
    qwen36_moe::batched_prefill_unpermute_combine_launch(
        cfg.device,
        shape.n_tokens,
        shape.top_k,
        shape.hidden,
        &inverse_buf,
        &perm_weight,
        &expert_out,
        &mut combined,
    )?;
    gpu_hal::sync(cfg.device)?;
    let got = download_bf16(&combined)?;
    let (max_abs, max_rel) = max_abs_rel_bf16(&got, &want, 1e-3);
    let (_, min_cos) = vector_abs_norm_and_min_cos(&got, &want, shape.hidden);

    let samples = measure_us(cfg, || {
        qwen36_moe::batched_prefill_router_permute_launch(
            cfg.device,
            shape.n_tokens,
            shape.top_k,
            shape.num_experts,
            &topk_idx,
            &topk_weight,
            &mut offsets,
            &mut perm_token,
            &mut perm_kpos,
            &mut perm_weight,
        )?;
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
            &mut expert_out,
            &mut counters,
        )?;
        qwen36_moe::batched_prefill_unpermute_combine_launch(
            cfg.device,
            shape.n_tokens,
            shape.top_k,
            shape.hidden,
            &inverse_buf,
            &perm_weight,
            &expert_out,
            &mut combined,
        )
    })?;

    Ok(case_result(
        "qwen36_moe_route_expert_combine",
        expert_shape_map(shape),
        router_ok && max_abs < 2e-2 && max_rel < 5e-2 && min_cos >= 0.999,
        Some(max_abs),
        Some(max_rel),
        Some(min_cos),
        Some(router_ok),
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

fn upload_u32(ordinal: usize, host: &[u32], shape: &[usize]) -> Result<GpuBuffer> {
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

fn make_rms_weight(n_cols: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    (0..n_cols)
        .map(|col| {
            let (next, raw) = lcg(state);
            state = next;
            let base = 0.75 + ((raw % 2000) as f32 / 2000.0) * 0.5;
            let edge = if col % 29 == 0 { 0.125 } else { 1.0 };
            half::bf16::from_f32(base * edge)
        })
        .collect()
}

fn make_rope_tables(
    n_positions: usize,
    half_rot: usize,
    seed: u64,
) -> (Vec<half::bf16>, Vec<half::bf16>) {
    let seed_phase = (seed % 997) as f32 * 0.0001;
    let mut cos = Vec::with_capacity(n_positions * half_rot);
    let mut sin = Vec::with_capacity(n_positions * half_rot);
    for pos in 0..n_positions {
        for i in 0..half_rot {
            let theta = seed_phase + pos as f32 * 0.013 + i as f32 * 0.007;
            cos.push(half::bf16::from_f32(theta.cos()));
            sin.push(half::bf16::from_f32(theta.sin()));
        }
    }
    (cos, sin)
}

fn cpu_rms_norm_plain_bf16(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &[half::bf16],
    weight: &[half::bf16],
) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; n_rows * n_cols];
    for row in 0..n_rows {
        let base = row * n_cols;
        let mut mean_sq = 0.0f32;
        for col in 0..n_cols {
            let value = input[base + col].to_f32();
            mean_sq += value * value;
        }
        let inv_rms = 1.0 / ((mean_sq / n_cols as f32) + eps).sqrt();
        for col in 0..n_cols {
            let value = input[base + col].to_f32() * inv_rms * weight[col].to_f32();
            out[base + col] = half::bf16::from_f32(value);
        }
    }
    out
}

fn cpu_apply_rope_bf16(
    shape: RopeShape,
    data: &[half::bf16],
    cos: &[half::bf16],
    sin: &[half::bf16],
) -> Vec<half::bf16> {
    let mut out = data.to_vec();
    let half_rot = shape.rotary_dim / 2;
    for pos in 0..shape.seq_len {
        let table_base = (shape.pos_offset + pos) * half_rot;
        for head in 0..shape.num_heads {
            let base = (pos * shape.num_heads + head) * shape.head_dim;
            for i in 0..half_rot {
                let c = cos[table_base + i].to_f32();
                let s = sin[table_base + i].to_f32();
                let x0 = data[base + i].to_f32();
                let x1 = data[base + i + half_rot].to_f32();
                out[base + i] = half::bf16::from_f32(x0 * c - x1 * s);
                out[base + i + half_rot] = half::bf16::from_f32(x1 * c + x0 * s);
            }
        }
    }
    out
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

fn bf16_trunc_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xFFFF_0000)
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

fn make_awq_inv_scale(k: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    (0..k)
        .map(|_| {
            let (next, raw) = lcg(state);
            state = next;
            half::bf16::from_f32(0.75 + ((raw % 1000) as f32 / 1000.0) * 0.5)
        })
        .collect()
}

fn make_outlier_cols(k: usize, sub_cols: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut cols = std::collections::BTreeSet::new();
    while cols.len() < sub_cols {
        let (next, raw) = lcg(state);
        state = next;
        cols.insert((raw as usize % k) as u32);
    }
    cols.into_iter().collect()
}

fn make_sparse_delta(n: usize, sub_cols: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n * sub_cols);
    for _ in 0..n * sub_cols {
        let (next, raw) = lcg(state);
        state = next;
        let centered = (raw % 2001) as f32 / 1000.0 - 1.0;
        out.push(half::bf16::from_f32(centered * 0.01));
    }
    out
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
fn cpu_int4_matmul(
    rows: usize,
    n: usize,
    k: usize,
    gs: usize,
    lhs: &[f32],
    packed: &[u8],
    scales: &[half::bf16],
    zeros: &[half::bf16],
    awq_inv_scale: Option<&[half::bf16]>,
) -> Vec<half::bf16> {
    let byte_cols = k / 2;
    let scale_cols = k / gs;
    let mut out = vec![half::bf16::ZERO; rows * n];
    for row in 0..rows {
        let x = &lhs[row * k..(row + 1) * k];
        for out_col in 0..n {
            let scale_row = out_col / gs;
            let row_byte_off = out_col * byte_cols;
            let mut acc = 0.0f32;
            for c in 0..k {
                let byte = packed[row_byte_off + c / 2];
                let nibble = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let g = c / gs;
                let mut w = int4_dequant_scalar(
                    nibble as u32,
                    scales[scale_row * scale_cols + g],
                    zeros[scale_row * scale_cols + g],
                );
                if let Some(inv) = awq_inv_scale {
                    w = bf16_trunc_f32(w * inv[c].to_f32());
                }
                acc += x[c] * w;
            }
            out[row * n + out_col] = half::bf16::from_f32(bf16_round_rne_f32(acc));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_sparse_outlier_add(
    rows: usize,
    n: usize,
    k: usize,
    sub_cols: usize,
    lhs: &[f32],
    outlier_cols: &[u32],
    outlier_delta: &[half::bf16],
    out: &mut [half::bf16],
) {
    for row in 0..rows {
        for out_col in 0..n {
            let mut acc = out[row * n + out_col].to_f32();
            for j in 0..sub_cols {
                let k_col = outlier_cols[j] as usize;
                debug_assert!(k_col < k);
                acc += lhs[row * k + k_col] * outlier_delta[out_col * sub_cols + j].to_f32();
            }
            out[row * n + out_col] = half::bf16::from_f32(bf16_round_rne_f32(acc));
        }
    }
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

fn inverse_permutation(
    n_tokens: usize,
    top_k: usize,
    perm_token: &[i32],
    perm_kpos: &[i32],
) -> Vec<i32> {
    let total = n_tokens * top_k;
    let mut inverse = vec![0i32; total];
    for row in 0..total {
        inverse[perm_token[row] as usize * top_k + perm_kpos[row] as usize] = row as i32;
    }
    inverse
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

fn int4_shape_map(shape: Int4MatvecShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("m".into(), shape.m),
        ("n".into(), shape.n),
        ("k".into(), shape.k),
        ("group_size".into(), shape.group_size),
        ("sparse_cols".into(), shape.sparse_cols),
    ])
}

fn rmsnorm_shape_map(shape: RmsNormShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("n_rows".into(), shape.n_rows),
        ("n_cols".into(), shape.n_cols),
        (
            "eps_scaled_1e9".into(),
            (shape.eps * 1_000_000_000.0) as usize,
        ),
    ])
}

fn rope_shape_map(shape: RopeShape) -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("seq_len".into(), shape.seq_len),
        ("num_heads".into(), shape.num_heads),
        ("head_dim".into(), shape.head_dim),
        ("rotary_dim".into(), shape.rotary_dim),
        ("pos_offset".into(), shape.pos_offset),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_int4_matmul_matches_manual_reference() {
        let shape = Int4MatvecShape {
            m: 1,
            n: 128,
            k: 128,
            group_size: 128,
            sparse_cols: 8,
            seed: 0x51,
        };
        let (_lhs_bf16, lhs) = make_bf16(shape.m * shape.k, shape.seed);
        let (packed, scales, zeros) =
            make_int4_slab(shape.n, shape.k, shape.group_size, shape.seed + 1);
        let got = cpu_int4_matmul(
            shape.m,
            shape.n,
            shape.k,
            shape.group_size,
            &lhs,
            &packed,
            &scales,
            &zeros,
            None,
        );
        let want0 = half::bf16::from_f32(bf16_round_rne_f32(int4_matvec_row(
            &packed,
            &scales,
            &zeros,
            shape.k,
            shape.group_size,
            0,
            &lhs,
        )));
        assert_eq!(got[0], want0);
    }

    #[test]
    fn cpu_dense_awq_changes_reference_by_inverse_scale() {
        let k = 128;
        let n = 128;
        let (_lhs_bf16, lhs) = make_bf16(k, 0x61);
        let (packed, scales, zeros) = make_int4_slab(n, k, GROUP_SIZE, 0x62);
        let awq = vec![half::bf16::from_f32(0.5); k];
        let base = cpu_int4_matmul(1, n, k, GROUP_SIZE, &lhs, &packed, &scales, &zeros, None);
        let dense = cpu_int4_matmul(
            1,
            n,
            k,
            GROUP_SIZE,
            &lhs,
            &packed,
            &scales,
            &zeros,
            Some(&awq),
        );
        assert!(base
            .iter()
            .zip(dense.iter())
            .any(|(lhs, rhs)| lhs.to_bits() != rhs.to_bits()));
    }

    #[test]
    fn cpu_sparse_outlier_add_matches_manual_reference() {
        let rows = 1;
        let n = 4;
        let k = 8;
        let sub_cols = 3;
        let lhs: Vec<f32> = (0..k).map(|i| i as f32 * 0.25).collect();
        let cols = vec![1, 4, 6];
        let delta: Vec<_> = (0..n * sub_cols)
            .map(|i| half::bf16::from_f32((i as f32 - 3.0) * 0.125))
            .collect();
        let mut got = vec![half::bf16::ZERO; rows * n];
        cpu_sparse_outlier_add(rows, n, k, sub_cols, &lhs, &cols, &delta, &mut got);
        for out_col in 0..n {
            let mut want = 0.0f32;
            for j in 0..sub_cols {
                want += lhs[cols[j] as usize] * delta[out_col * sub_cols + j].to_f32();
            }
            assert_eq!(got[out_col], half::bf16::from_f32(bf16_round_rne_f32(want)));
        }
    }

    #[test]
    fn cpu_rms_norm_plain_matches_manual_reference() {
        let input = vec![half::bf16::from_f32(3.0), half::bf16::from_f32(4.0)];
        let weight = vec![half::bf16::from_f32(1.5), half::bf16::from_f32(0.5)];
        let got = cpu_rms_norm_plain_bf16(1, 2, 0.0, &input, &weight);
        let inv_rms = 1.0f32 / ((25.0f32 / 2.0).sqrt());
        let want = [
            half::bf16::from_f32(3.0 * inv_rms * 1.5),
            half::bf16::from_f32(4.0 * inv_rms * 0.5),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn cpu_rope_bf16_rotates_first_half_against_second_half() {
        let shape = RopeShape {
            seq_len: 1,
            num_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            pos_offset: 0,
            seed: 0,
        };
        let data = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(4.0),
        ];
        let cos = vec![half::bf16::from_f32(0.0), half::bf16::from_f32(1.0)];
        let sin = vec![half::bf16::from_f32(1.0), half::bf16::from_f32(0.0)];
        let got = cpu_apply_rope_bf16(shape, &data, &cos, &sin);
        assert_eq!(
            got,
            vec![
                half::bf16::from_f32(-3.0),
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(4.0),
            ]
        );
    }
}
