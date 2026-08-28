//! GPU draft forward engine vs CPU reference parity.
//!
//! Loads the canonical DFlash2 drafter, runs the same inputs through both the
//! CPU `draft_forward` oracle and the GPU `DraftEngine::forward`, and compares
//! the post-final-norm hidden states. The GPU path uses BF16 matmuls with F32
//! accumulation, so exact bit-equality is not expected; the test checks a
//! bounded relative error that is tight enough to catch orchestration bugs
//! (wrong RoPE style, wrong RMSNorm convention, wrong causal mask, wrong
//! matmul lhs dtype) while tolerating BF16 rounding.

use std::path::PathBuf;

use gpu_hal::{GpuBuffer, ScalarType};
use model_store::dflash::{load_draft, DraftWeights};
use model_store::dflash_ref::draft_forward;
use supersonic_runtime::draft_engine::DraftEngine;

fn require_artifacts() -> bool {
    std::env::var_os("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").is_some()
}

fn draft_path() -> Option<PathBuf> {
    let value = std::env::var_os("SUPERSONIC_DFLASH_DRAFT_GGUF")?;
    let path = PathBuf::from(value);
    if !path.is_file() {
        if require_artifacts() {
            panic!(
                "SUPERSONIC_DFLASH_DRAFT_GGUF points to a missing drafter: {}",
                path.display()
            );
        }
        return None;
    }
    Some(path)
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    (0..bytes.len() / 2)
        .map(|i| half::bf16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32())
        .collect()
}

/// Relative-L2 and max-abs comparison with diagnostics.
fn assert_close(cpu: &[f32], gpu: &[f32], rel_l2_tol: f64, max_abs_tol: f64, label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    let mut sq_err = 0.0f64;
    let mut sq_cpu = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_abs_idx = 0usize;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let d = (*g as f64) - (*c as f64);
        sq_err += d * d;
        sq_cpu += (*c as f64) * (*c as f64);
        if d.abs() > max_abs {
            max_abs = d.abs();
            max_abs_idx = i;
        }
        if c.abs() > 0.1 {
            max_rel = max_rel.max((d / *c as f64).abs());
        }
    }
    let rel_l2 = if sq_cpu > 0.0 {
        (sq_err / sq_cpu).sqrt()
    } else {
        0.0
    };
    eprintln!(
        "{label}: rel_l2={rel_l2:.5} max_abs={max_abs:.5} (idx {max_abs_idx}, cpu={} gpu={}) max_rel={max_rel:.4}",
        cpu.get(max_abs_idx).copied().unwrap_or(0.0),
        gpu.get(max_abs_idx).copied().unwrap_or(0.0),
    );
    assert!(
        rel_l2 < rel_l2_tol,
        "{label}: rel_l2 {rel_l2:.5} >= {rel_l2_tol}"
    );
    assert!(
        max_abs < max_abs_tol,
        "{label}: max_abs {max_abs:.5} >= {max_abs_tol}"
    );
}

/// Run one parity case with the given ctx_len and nq.
fn run_parity(weights: &DraftWeights, ordinal: usize, ctx_len: usize, nq: usize) {
    let cfg = &weights.config;
    let hidden = cfg.hidden;
    let ntl = cfg.n_target_layers;

    // Deterministic, bounded inputs (same pattern as the CPU forward test).
    let target_hidden: Vec<f32> = (0..ctx_len * ntl * hidden)
        .map(|i| (((i % 13) as f32) - 6.0) / 11.0)
        .collect();
    let noise_embed: Vec<f32> = (0..nq * hidden)
        .map(|i| (((i % 7) as f32) - 3.0) / 5.0)
        .collect();
    let positions_q: Vec<usize> = (0..nq).map(|i| ctx_len + i).collect();
    let positions_k: Vec<usize> = (0..ctx_len + nq).map(|i| i).collect();

    // CPU reference (F32 dequant, F64 accumulation).
    let cpu_out = draft_forward(
        weights,
        cfg,
        &target_hidden,
        &noise_embed,
        &positions_q,
        &positions_k,
    )
    .unwrap_or_else(|e| panic!("draft_forward: {e}"));
    assert!(
        cpu_out.iter().all(|v| v.is_finite()),
        "cpu output not finite"
    );

    // GPU engine (BF16 matmuls, F32 accumulation).
    let gpu_weights = weights
        .upload(ordinal)
        .unwrap_or_else(|e| panic!("upload: {e}"));
    let max_pos = ctx_len + nq + 16;
    let engine = DraftEngine::new(gpu_weights, ordinal, max_pos)
        .unwrap_or_else(|e| panic!("DraftEngine::new: {e}"));

    let th_bytes = f32_to_bf16_bytes(&target_hidden);
    let target_hidden_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[target_hidden.len()], &th_bytes)
            .unwrap_or_else(|e| panic!("upload target_hidden: {e}"));

    let ne_bytes = f32_to_bf16_bytes(&noise_embed);
    let noise_embed_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[noise_embed.len()], &ne_bytes)
            .unwrap_or_else(|e| panic!("upload noise_embed: {e}"));

    let gpu_out = engine
        .forward(
            &target_hidden_gpu,
            &noise_embed_gpu,
            &positions_q,
            &positions_k,
        )
        .unwrap_or_else(|e| panic!("DraftEngine::forward: {e}"));
    gpu_hal::sync(ordinal).unwrap_or_else(|e| panic!("sync: {e}"));

    let gpu_bytes = gpu_out
        .to_host_bytes()
        .unwrap_or_else(|e| panic!("download: {e}"));
    // The draft forward now outputs F32 directly (was BF16 before the
    // F32-precision conversion). Read as F32 bytes.
    let gpu_out_f32: Vec<f32> = gpu_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(gpu_out_f32.len(), nq * hidden, "gpu output length");
    assert!(
        gpu_out_f32.iter().all(|v| v.is_finite()),
        "gpu output not finite"
    );

    // Concise structural summary; the bounded `assert_close` below is the gate.
    let cpu_l2 = (cpu_out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
    let gpu_l2 = (gpu_out_f32.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
    eprintln!("diag: cpu_l2={cpu_l2:.2} gpu_l2={gpu_l2:.2}");
    let label = format!("dflash parity ctx={ctx_len} nq={nq}");
    // The draft forward runs the full F32 path (scalar Q8_0 matmul with F32
    // accumulation and F32 output, F32 SwiGLU), matching the upstream ggml F32
    // compute type. The residual error is F32-vs-F64 accumulation only.
    assert_close(&cpu_out, &gpu_out_f32, 0.15, 5.0, &label);
}

#[test]
fn dflash_draft_gpu_matches_cpu_reference() {
    let Some(path) = draft_path() else {
        eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
        return;
    };
    match kernel_ffi::query_gpu_info(0) {
        Ok(_) => {}
        Err(e) => {
            if require_artifacts() {
                panic!("HIP device 0 unavailable: {e}");
            }
            eprintln!("skip: HIP device 0 unavailable: {e}");
            return;
        }
    }
    let ordinal = 0usize;
    let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));

    run_parity(&weights, ordinal, 2, 1);
    run_parity(&weights, ordinal, 2, 2);
    run_parity(&weights, ordinal, 2, 8);
}

#[test]
fn dflash_fc_matmul_gpu_matches_cpu() {
    let Some(path) = draft_path() else {
        eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
        return;
    };
    match kernel_ffi::query_gpu_info(0) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("skip: HIP device 0 unavailable: {e}");
            return;
        }
    }
    let ordinal = 0usize;
    let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));
    let cfg = &weights.config;
    let hidden = cfg.hidden;
    let ntl = cfg.n_target_layers;
    let ctx_len = 2usize;
    let m = ctx_len;
    let n = hidden;
    let k = ntl * hidden;

    // Deterministic lhs (F32 -> BF16).
    let lhs_f32: Vec<f32> = (0..m * k)
        .map(|i| (((i % 13) as f32) - 6.0) / 11.0)
        .collect();
    let lhs_bytes = f32_to_bf16_bytes(&lhs_f32);
    let lhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m * k], &lhs_bytes)
        .unwrap_or_else(|e| panic!("lhs upload: {e}"));

    // Upload fc weight (Q8_0 packed U8).
    let gpu_weights = weights
        .upload(ordinal)
        .unwrap_or_else(|e| panic!("upload: {e}"));
    let rhs = &gpu_weights.fc;

    let mut out_gpu = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[m * n])
        .unwrap_or_else(|e| panic!("out alloc: {e}"));
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        rhs,
        rhs,
        rhs,
        None,
        0,
        qwen38::weights::LOWBIT_GGML_Q8_0,
        &mut out_gpu,
    )
    .unwrap_or_else(|e| panic!("fc matmul: {e}"));
    gpu_hal::sync(ordinal).unwrap_or_else(|e| panic!("sync: {e}"));
    let out_bytes = out_gpu
        .to_host_bytes()
        .unwrap_or_else(|e| panic!("download: {e}"));
    let gpu_out = bf16_bytes_to_f32(&out_bytes);

    // CPU reference: dequant fc, matvec per ctx row.
    use model_store::dflash_ref::{dequant_weight, matvec};
    // ne0=k=ntl*hidden (input, contiguous), ne1=n=hidden (output, rows).
    let fc = dequant_weight(&weights, "dflash.fc.weight", k, n)
        .unwrap_or_else(|e| panic!("dequant fc: {e}"));
    let mut cpu_out = vec![0.0f32; m * n];
    for ci in 0..m {
        matvec(
            &fc,
            &lhs_f32[ci * k..(ci + 1) * k],
            n,
            k,
            &mut cpu_out[ci * n..(ci + 1) * n],
        );
    }

    eprintln!(
        "fc matmul: gpu[0..4]={:?} cpu[0..4]={:?}",
        &gpu_out[0..4.min(gpu_out.len())],
        &cpu_out[0..4.min(cpu_out.len())]
    );
    assert_close(&cpu_out, &gpu_out, 0.15, 2.0, "fc matmul");
}

#[test]
fn dflash_output_proj_matmul_gpu_matches_cpu() {
    let Some(path) = draft_path() else {
        eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
        return;
    };
    match kernel_ffi::query_gpu_info(0) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("skip: HIP device 0 unavailable: {e}");
            return;
        }
    }
    let ordinal = 0usize;
    let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));
    let cfg = &weights.config;
    let hidden = cfg.hidden;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let m = 1usize;
    let n = hidden;
    let k = q_dim;

    // Small controlled lhs (O(1)).
    let lhs_f32: Vec<f32> = (0..m * k).map(|i| (((i % 7) as f32) - 3.0) / 5.0).collect();
    let lhs_bytes = f32_to_bf16_bytes(&lhs_f32);
    let lhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m * k], &lhs_bytes)
        .unwrap_or_else(|e| panic!("lhs upload: {e}"));

    let gpu_weights = weights
        .upload(ordinal)
        .unwrap_or_else(|e| panic!("upload: {e}"));
    let rhs = &gpu_weights.layers[0].output;

    let mut out_gpu = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[m * n])
        .unwrap_or_else(|e| panic!("out alloc: {e}"));
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        rhs,
        rhs,
        rhs,
        None,
        0,
        qwen38::weights::LOWBIT_GGML_Q8_0,
        &mut out_gpu,
    )
    .unwrap_or_else(|e| panic!("output proj matmul: {e}"));
    gpu_hal::sync(ordinal).unwrap_or_else(|e| panic!("sync: {e}"));
    let out_bytes = out_gpu
        .to_host_bytes()
        .unwrap_or_else(|e| panic!("download: {e}"));
    let gpu_out = bf16_bytes_to_f32(&out_bytes);

    use model_store::dflash_ref::{dequant_weight, matvec};
    // attn_output: ne0=q_dim (input), ne1=hidden (output).
    let wo = dequant_weight(&weights, "blk.0.attn_output.weight", k, n)
        .unwrap_or_else(|e| panic!("dequant wo: {e}"));
    let mut cpu_out = vec![0.0f32; m * n];
    for ci in 0..m {
        matvec(
            &wo,
            &lhs_f32[ci * k..(ci + 1) * k],
            n,
            k,
            &mut cpu_out[ci * n..(ci + 1) * n],
        );
    }
    eprintln!(
        "output proj: gpu[0..4]={:?} cpu[0..4]={:?}",
        &gpu_out[0..4.min(gpu_out.len())],
        &cpu_out[0..4.min(cpu_out.len())]
    );
    assert_close(&cpu_out, &gpu_out, 0.15, 2.0, "output proj");
}
