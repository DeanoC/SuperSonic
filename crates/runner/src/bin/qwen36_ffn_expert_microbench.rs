//! Focused Qwen3.6 MoE routed-expert FFN INT4 microbench.
//!
//! This exercises only the stage-5 routed expert gate/up projection shape:
//! top-k experts, rows `[2 * moe_intermediate, hidden]`, GPTQ INT4 sidecars,
//! and the same workspace offsets used by the decode fallback.

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe;
use std::time::Instant;

const HIDDEN: usize = 2048;
const NUM_EXPERTS: usize = 256;
const MOE_INTERMEDIATE: usize = 512;
const TOP_K: usize = 8;
const GROUP_SIZE: usize = 128;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 20)]
    iters: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
}

fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn nibble_at(expert: usize, row: usize, col: usize) -> u8 {
    let mut x = expert.wrapping_mul(0x9E37_79B1)
        ^ row.wrapping_mul(0x85EB_CA6B)
        ^ col.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    (x & 0x0f) as u8
}

fn scale_at(expert: usize, row_group: usize, col_group: usize) -> f32 {
    let x = (expert * 131 + row_group * 17 + col_group * 29) as f32;
    bf16_round(0.0035 + 0.009 * (0.5 + 0.5 * (x * 0.137).sin()))
}

fn zero_at(expert: usize, row_group: usize, col_group: usize) -> f32 {
    let x = (expert * 43 + row_group * 11 + col_group * 7) as f32;
    bf16_round(7.5 + 6.5 * (x * 0.071).cos())
}

fn activation_at(col: usize) -> f32 {
    bf16_round(((col as f32) * 0.013).sin() * 0.25 + ((col % 31) as f32 - 15.0) * 0.002)
}

fn build_packed(rows: usize, cols: usize) -> Vec<u8> {
    let byte_cols = cols / 2;
    let mut packed = vec![0u8; NUM_EXPERTS * rows * byte_cols];
    for expert in 0..NUM_EXPERTS {
        for row in 0..rows {
            let base = (expert * rows + row) * byte_cols;
            for byte_col in 0..byte_cols {
                let c0 = byte_col * 2;
                let lo = nibble_at(expert, row, c0);
                let hi = nibble_at(expert, row, c0 + 1);
                packed[base + byte_col] = lo | (hi << 4);
            }
        }
    }
    packed
}

fn build_sidecars(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let scale_rows = rows.div_ceil(GROUP_SIZE);
    let scale_cols = cols.div_ceil(GROUP_SIZE);
    let mut scales = vec![0.0f32; NUM_EXPERTS * scale_rows * scale_cols];
    let mut zeros = vec![0.0f32; NUM_EXPERTS * scale_rows * scale_cols];
    for expert in 0..NUM_EXPERTS {
        for rg in 0..scale_rows {
            for cg in 0..scale_cols {
                let idx = (expert * scale_rows + rg) * scale_cols + cg;
                scales[idx] = scale_at(expert, rg, cg);
                zeros[idx] = zero_at(expert, rg, cg);
            }
        }
    }
    (scales, zeros)
}

fn dequant(
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    expert: usize,
    row: usize,
    col: usize,
) -> f32 {
    let rows = 2 * MOE_INTERMEDIATE;
    let byte_cols = HIDDEN / 2;
    let scale_rows = rows.div_ceil(GROUP_SIZE);
    let scale_cols = HIDDEN.div_ceil(GROUP_SIZE);
    let byte = packed[(expert * rows + row) * byte_cols + col / 2];
    let nibble = if col & 1 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    let scale_idx = (expert * scale_rows + row / GROUP_SIZE) * scale_cols + col / GROUP_SIZE;
    let s = scales[scale_idx];
    let z = zeros[scale_idx];
    bf16_round(nibble as f32 * s - z * s)
}

fn reference(
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    h_norm: &[f32],
    experts: &[usize],
) -> Vec<f32> {
    let mut out = vec![0.0f32; experts.len() * MOE_INTERMEDIATE];
    for (group, &expert) in experts.iter().enumerate() {
        for row in 0..MOE_INTERMEDIATE {
            let mut gate = 0.0f32;
            let mut up = 0.0f32;
            for col in 0..HIDDEN {
                let x = h_norm[col];
                gate += dequant(packed, scales, zeros, expert, row, col) * x;
                up += dequant(packed, scales, zeros, expert, MOE_INTERMEDIATE + row, col) * x;
            }
            let silu = gate * (1.0f32 / (1.0f32 + (-gate).exp()));
            out[group * MOE_INTERMEDIATE + row] = silu * up;
        }
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !gpu_hal::is_backend_compiled(Backend::Metal) {
        bail!("Metal backend is not compiled");
    }
    gpu_hal::set_backend(Backend::Metal);

    let rows = 2 * MOE_INTERMEDIATE;
    let active_experts = [3usize, 17, 42, 87, 119, 140, 188, 251];
    let off_h_norm = 0usize;
    let off_topk_idx = off_h_norm + HIDDEN;
    let off_expert_mid = off_topk_idx + TOP_K;
    let workspace_len = off_expert_mid + TOP_K * MOE_INTERMEDIATE;

    eprintln!(
        "[qwen36-ffn-expert-microbench] building synthetic exact geometry: experts={NUM_EXPERTS} rows={rows} hidden={HIDDEN}"
    );
    let packed = build_packed(rows, HIDDEN);
    let (scales, zeros) = build_sidecars(rows, HIDDEN);
    let h_norm: Vec<f32> = (0..HIDDEN).map(activation_at).collect();
    let expected = reference(&packed, &scales, &zeros, &h_norm, &active_experts);

    let mut workspace_host = vec![0.0f32; workspace_len];
    workspace_host[off_h_norm..off_h_norm + HIDDEN].copy_from_slice(&h_norm);
    for (i, &expert) in active_experts.iter().enumerate() {
        workspace_host[off_topk_idx + i] = f32::from_bits(expert as u32);
    }

    let mut workspace = GpuBuffer::from_host_bytes(
        0,
        ScalarType::F32,
        &[workspace_len],
        &f32_bytes(&workspace_host),
    )?;
    let gate_up =
        GpuBuffer::from_host_bytes(0, ScalarType::U8, &[NUM_EXPERTS, rows, HIDDEN / 2], &packed)?;
    let gate_up_scale = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[NUM_EXPERTS, rows / GROUP_SIZE, HIDDEN / GROUP_SIZE],
        &bf16_bytes(&scales),
    )?;
    let gate_up_zero = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[NUM_EXPERTS, rows / GROUP_SIZE, HIDDEN / GROUP_SIZE],
        &bf16_bytes(&zeros),
    )?;

    for _ in 0..args.warmup {
        qwen36_moe::ffn_expert_gate_up_tiled_metal_launch(
            HIDDEN,
            MOE_INTERMEDIATE,
            TOP_K,
            GROUP_SIZE,
            &mut workspace,
            &gate_up,
            &gate_up_scale,
            &gate_up_zero,
            off_h_norm,
            off_topk_idx,
            off_expert_mid,
        )?;
    }

    let start = Instant::now();
    for _ in 0..args.iters {
        qwen36_moe::ffn_expert_gate_up_tiled_metal_launch(
            HIDDEN,
            MOE_INTERMEDIATE,
            TOP_K,
            GROUP_SIZE,
            &mut workspace,
            &gate_up,
            &gate_up_scale,
            &gate_up_zero,
            off_h_norm,
            off_topk_idx,
            off_expert_mid,
        )?;
    }
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / args.iters.max(1) as f64;

    let got_all = f32_from_bytes(&workspace.to_host_bytes()?);
    let got = &got_all[off_expert_mid..off_expert_mid + TOP_K * MOE_INTERMEDIATE];
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    let mut mismatches = 0usize;
    for i in 0..got.len() {
        let d = (got[i] - expected[i]).abs();
        let rel = d / expected[i].abs().max(1.0);
        if d > max_abs {
            max_abs = d;
            max_rel = rel;
            worst = i;
        }
        let tol = expected[i].abs().max(1.0) * 0.04 + 0.02;
        if d > tol {
            mismatches += 1;
        }
    }
    println!(
        "[qwen36-ffn-expert-gate-up-tiled] mean_ms={mean_ms:.4} iters={} warmup={} max_abs={max_abs:.6} max_rel={max_rel:.6} worst={} got={:.6} expected={:.6} mismatches={mismatches}",
        args.iters,
        args.warmup,
        worst,
        got[worst],
        expected[worst],
    );
    if mismatches > 0 {
        return Err(anyhow!("tiled expert gate/up mismatches: {mismatches}"));
    }
    Ok(())
}
