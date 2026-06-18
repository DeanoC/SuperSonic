use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};

const CONV_DIM: usize = 7;
const TREE: usize = 5;
const KERNEL: usize = 4;
const PAD: usize = KERNEL - 1;
const TOTAL: usize = PAD + TREE;
const BA_SEQ: usize = 16;
const BA_HIDDEN: usize = 64;
const BA_NV: usize = 4;
const FULL_S: usize = 5;
const FULL_H: usize = 3;
const FULL_D: usize = 7;
const ATTN_TREE: usize = 5;
const ATTN_PREFIX: usize = 6;
const ATTN_PREFIX_STRIDE: usize = 9;
const ATTN_Q_HEADS: usize = 4;
const ATTN_KV_HEADS: usize = 2;
const ATTN_HEAD_DIM: usize = 16;

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_conv_pack_chain_matches_prefill_conv_pack() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let mixed = patterned_bf16(CONV_DIM * TOTAL, 0.009, -0.11);
    let weights = patterned_bf16(CONV_DIM * KERNEL, 0.017, 0.03);
    let parent_ids = [-1, 0, 1, 2, 3];

    let mixed_gpu = upload_bf16(ordinal, &[CONV_DIM, TOTAL], &mixed)?;
    let weights_gpu = upload_bf16(ordinal, &[CONV_DIM, KERNEL], &weights)?;
    let parent_gpu = upload_i32_as_u32(ordinal, &parent_ids)?;
    let mut tree_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[TREE, CONV_DIM])?;
    let mut chain_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[TREE, CONV_DIM])?;

    kernel_ffi::prefill_ffi::linear_tree_conv_pack(
        ordinal,
        ScalarType::BF16,
        1,
        CONV_DIM,
        TOTAL,
        TREE,
        KERNEL,
        &mixed_gpu,
        &weights_gpu,
        &parent_gpu,
        &mut tree_out,
    )?;
    kernel_ffi::prefill_ffi::linear_prefill_conv_pack(
        ordinal,
        ScalarType::BF16,
        1,
        CONV_DIM,
        TOTAL,
        TREE,
        KERNEL,
        &mixed_gpu,
        &weights_gpu,
        &mut chain_out,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(download_bf16(&chain_out)?, download_bf16(&tree_out)?);
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_conv_pack_matches_cpu_branching_fixture() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let mixed = patterned_bf16(CONV_DIM * TOTAL, 0.009, -0.11);
    let weights = patterned_bf16(CONV_DIM * KERNEL, 0.017, 0.03);
    let parent_ids = [-1, 0, 0, 2, 2];

    let mixed_gpu = upload_bf16(ordinal, &[CONV_DIM, TOTAL], &mixed)?;
    let weights_gpu = upload_bf16(ordinal, &[CONV_DIM, KERNEL], &weights)?;
    let parent_gpu = upload_i32_as_u32(ordinal, &parent_ids)?;
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[TREE, CONV_DIM])?;

    kernel_ffi::prefill_ffi::linear_tree_conv_pack(
        ordinal,
        ScalarType::BF16,
        1,
        CONV_DIM,
        TOTAL,
        TREE,
        KERNEL,
        &mixed_gpu,
        &weights_gpu,
        &parent_gpu,
        &mut out,
    )?;
    gpu_hal::sync(ordinal)?;

    let want = cpu_tree_conv_pack(&mixed, &weights, &parent_ids);
    assert_close(
        "branch tree conv",
        &want,
        &download_bf16_as_f32(&out)?,
        1.0e-2,
    )
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_conv_pack_indexed_matches_parent_walk_fixture() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let mixed = patterned_bf16(CONV_DIM * TOTAL, 0.009, -0.11);
    let weights = patterned_bf16(CONV_DIM * KERNEL, 0.017, 0.03);
    let parent_ids = [-1, 0, 0, 2, 2];
    let source_cols = cpu_tree_conv_source_cols(&parent_ids);

    let mixed_gpu = upload_bf16(ordinal, &[CONV_DIM, TOTAL], &mixed)?;
    let weights_gpu = upload_bf16(ordinal, &[CONV_DIM, KERNEL], &weights)?;
    let parent_gpu = upload_i32_as_u32(ordinal, &parent_ids)?;
    let source_gpu = upload_u32(ordinal, &[TREE, KERNEL], &source_cols)?;
    let mut parent_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[TREE, CONV_DIM])?;
    let mut indexed_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[TREE, CONV_DIM])?;

    kernel_ffi::prefill_ffi::linear_tree_conv_pack(
        ordinal,
        ScalarType::BF16,
        1,
        CONV_DIM,
        TOTAL,
        TREE,
        KERNEL,
        &mixed_gpu,
        &weights_gpu,
        &parent_gpu,
        &mut parent_out,
    )?;
    kernel_ffi::prefill_ffi::linear_tree_conv_pack_indexed(
        ordinal,
        ScalarType::BF16,
        1,
        CONV_DIM,
        TOTAL,
        TREE,
        KERNEL,
        KERNEL,
        &mixed_gpu,
        &weights_gpu,
        &source_gpu,
        &mut indexed_out,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(download_bf16(&parent_out)?, download_bf16(&indexed_out)?);
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_conv_input_prepare_matches_transpose_plus_tail() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let src = patterned_bf16(TREE * CONV_DIM, 0.013, 0.05);
    let old_tail = patterned_bf16(CONV_DIM * PAD, -0.021, 0.14);

    let src_gpu = upload_bf16(ordinal, &[TREE, CONV_DIM], &src)?;
    let tail_gpu = upload_bf16(ordinal, &[CONV_DIM, PAD], &old_tail)?;
    let mut old_input = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[CONV_DIM, TOTAL])?;
    let mut fused_input = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[CONV_DIM, TOTAL])?;
    let mut fused_tail = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[CONV_DIM, PAD])?;

    kernel_ffi::prefill_ffi::transpose_pad_conv(
        ordinal,
        ScalarType::BF16,
        TREE,
        CONV_DIM,
        PAD,
        &src_gpu,
        &mut old_input,
    )?;
    kernel_ffi::prefill_ffi::fill_conv_tail(
        ordinal,
        ScalarType::BF16,
        CONV_DIM,
        PAD,
        TOTAL,
        &tail_gpu,
        &mut old_input,
    )?;
    kernel_ffi::prefill_ffi::prepare_conv_input_tail(
        ordinal,
        ScalarType::BF16,
        TREE,
        CONV_DIM,
        PAD,
        &src_gpu,
        &tail_gpu,
        &mut fused_input,
        &mut fused_tail,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(download_bf16(&old_input)?, download_bf16(&fused_input)?);
    assert_eq!(cpu_tail_from_src(&src), download_bf16(&fused_tail)?);
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_direct_ba_beta_g_matches_matmul_path() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let hidden = patterned_bf16(BA_SEQ * BA_HIDDEN, 0.003, -0.02);
    let ba_weight = patterned_bf16(2 * BA_NV * BA_HIDDEN, -0.002, 0.015);
    let dt_bias = patterned_bf16(BA_NV, 0.017, -0.03);
    let a_log_exp = patterned_bf16(BA_NV, -0.011, 0.12);

    let hidden_gpu = upload_bf16(ordinal, &[BA_SEQ, BA_HIDDEN], &hidden)?;
    let ba_weight_gpu = upload_bf16(ordinal, &[2 * BA_NV, BA_HIDDEN], &ba_weight)?;
    let dt_gpu = upload_bf16(ordinal, &[BA_NV], &dt_bias)?;
    let a_log_gpu = upload_bf16(ordinal, &[BA_NV], &a_log_exp)?;
    let mut ba = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[BA_SEQ, 2 * BA_NV])?;
    let mut beta_ref = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BA_NV, BA_SEQ])?;
    let mut g_ref = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BA_NV, BA_SEQ])?;
    let mut beta_direct = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BA_NV, BA_SEQ])?;
    let mut g_direct = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BA_NV, BA_SEQ])?;

    kernel_ffi::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        BA_SEQ,
        2 * BA_NV,
        BA_HIDDEN,
        &hidden_gpu,
        &ba_weight_gpu,
        &mut ba,
    )?;
    kernel_ffi::prefill_ffi::compute_beta_g_ba_bf16(
        ordinal,
        BA_SEQ,
        BA_NV,
        &ba,
        &dt_gpu,
        &a_log_gpu,
        &mut beta_ref,
        &mut g_ref,
    )?;
    kernel_ffi::prefill_ffi::project_ba_compute_beta_g_bf16(
        ordinal,
        BA_SEQ,
        BA_HIDDEN,
        BA_NV,
        &hidden_gpu,
        &ba_weight_gpu,
        &dt_gpu,
        &a_log_gpu,
        &mut beta_direct,
        &mut g_direct,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(
        download_f32_bits(&beta_ref)?,
        download_f32_bits(&beta_direct)?
    );
    assert_eq!(download_f32_bits(&g_ref)?, download_f32_bits(&g_direct)?);
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_full_kv_pair_transpose_matches_separate_calls() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let k = patterned_bf16(FULL_S * FULL_H * FULL_D, 0.011, -0.07);
    let v = patterned_bf16(FULL_S * FULL_H * FULL_D, -0.019, 0.13);
    let k_gpu = upload_bf16(ordinal, &[FULL_S, FULL_H, FULL_D], &k)?;
    let v_gpu = upload_bf16(ordinal, &[FULL_S, FULL_H, FULL_D], &v)?;
    let mut k_ref = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[FULL_H, FULL_S, FULL_D])?;
    let mut v_ref = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[FULL_H, FULL_S, FULL_D])?;
    let mut k_pair = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[FULL_H, FULL_S, FULL_D])?;
    let mut v_pair = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[FULL_H, FULL_S, FULL_D])?;

    kernel_ffi::prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        FULL_S,
        FULL_H,
        FULL_D,
        &k_gpu,
        &mut k_ref,
    )?;
    kernel_ffi::prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        FULL_S,
        FULL_H,
        FULL_D,
        &v_gpu,
        &mut v_ref,
    )?;
    kernel_ffi::prefill_ffi::transpose_shd_hsd_pair(
        ordinal,
        ScalarType::BF16,
        FULL_S,
        FULL_H,
        FULL_D,
        &k_gpu,
        &v_gpu,
        &mut k_pair,
        &mut v_pair,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(download_bf16(&k_ref)?, download_bf16(&k_pair)?);
    assert_eq!(download_bf16(&v_ref)?, download_bf16(&v_pair)?);
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_full_attention_strided_prefix_matches_contiguous_prefix() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let query = patterned_bf16(ATTN_Q_HEADS * ATTN_TREE * ATTN_HEAD_DIM, 0.007, -0.03);
    let prefix_k = patterned_bf16(ATTN_KV_HEADS * ATTN_PREFIX * ATTN_HEAD_DIM, -0.005, 0.04);
    let prefix_v = patterned_bf16(ATTN_KV_HEADS * ATTN_PREFIX * ATTN_HEAD_DIM, 0.006, -0.02);
    let tree_k = patterned_bf16(ATTN_KV_HEADS * ATTN_TREE * ATTN_HEAD_DIM, 0.009, 0.01);
    let tree_v = patterned_bf16(ATTN_KV_HEADS * ATTN_TREE * ATTN_HEAD_DIM, -0.008, 0.025);
    let mut prefix_k_strided =
        vec![half::bf16::from_f32(7.25); ATTN_KV_HEADS * ATTN_PREFIX_STRIDE * ATTN_HEAD_DIM];
    let mut prefix_v_strided =
        vec![half::bf16::from_f32(-3.5); ATTN_KV_HEADS * ATTN_PREFIX_STRIDE * ATTN_HEAD_DIM];
    for h in 0..ATTN_KV_HEADS {
        for p in 0..ATTN_PREFIX {
            for d in 0..ATTN_HEAD_DIM {
                let src = (h * ATTN_PREFIX + p) * ATTN_HEAD_DIM + d;
                let dst = (h * ATTN_PREFIX_STRIDE + p) * ATTN_HEAD_DIM + d;
                prefix_k_strided[dst] = prefix_k[src];
                prefix_v_strided[dst] = prefix_v[src];
            }
        }
    }
    let visibility = (0..ATTN_TREE)
        .flat_map(|q| {
            (0..ATTN_TREE).map(move |k| {
                if k <= q || (q == 1 && k == 3) {
                    1u8
                } else {
                    0u8
                }
            })
        })
        .collect::<Vec<_>>();

    let query_gpu = upload_bf16(ordinal, &[ATTN_Q_HEADS, ATTN_TREE, ATTN_HEAD_DIM], &query)?;
    let prefix_k_gpu = upload_bf16(
        ordinal,
        &[ATTN_KV_HEADS, ATTN_PREFIX, ATTN_HEAD_DIM],
        &prefix_k,
    )?;
    let prefix_v_gpu = upload_bf16(
        ordinal,
        &[ATTN_KV_HEADS, ATTN_PREFIX, ATTN_HEAD_DIM],
        &prefix_v,
    )?;
    let prefix_k_strided_gpu = upload_bf16(
        ordinal,
        &[ATTN_KV_HEADS, ATTN_PREFIX_STRIDE, ATTN_HEAD_DIM],
        &prefix_k_strided,
    )?;
    let prefix_v_strided_gpu = upload_bf16(
        ordinal,
        &[ATTN_KV_HEADS, ATTN_PREFIX_STRIDE, ATTN_HEAD_DIM],
        &prefix_v_strided,
    )?;
    let tree_k_gpu = upload_bf16(ordinal, &[ATTN_KV_HEADS, ATTN_TREE, ATTN_HEAD_DIM], &tree_k)?;
    let tree_v_gpu = upload_bf16(ordinal, &[ATTN_KV_HEADS, ATTN_TREE, ATTN_HEAD_DIM], &tree_v)?;
    let visibility_gpu = upload_u8(ordinal, &[ATTN_TREE, ATTN_TREE], &visibility)?;
    let mut out_ref = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[ATTN_Q_HEADS, ATTN_TREE, ATTN_HEAD_DIM],
    )?;
    let mut out_strided = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[ATTN_Q_HEADS, ATTN_TREE, ATTN_HEAD_DIM],
    )?;
    let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();

    kernel_ffi::prefill_ffi::full_attention_tree_prefill(
        ordinal,
        ScalarType::BF16,
        1,
        ATTN_Q_HEADS,
        ATTN_KV_HEADS,
        ATTN_TREE,
        ATTN_PREFIX,
        ATTN_HEAD_DIM,
        scale,
        &query_gpu,
        &prefix_k_gpu,
        &prefix_v_gpu,
        &tree_k_gpu,
        &tree_v_gpu,
        &visibility_gpu,
        &mut out_ref,
    )?;
    kernel_ffi::prefill_ffi::full_attention_tree_prefill_strided_raw(
        ordinal,
        ScalarType::BF16,
        1,
        ATTN_Q_HEADS,
        ATTN_KV_HEADS,
        ATTN_TREE,
        ATTN_PREFIX,
        ATTN_PREFIX_STRIDE,
        ATTN_HEAD_DIM,
        scale,
        &query_gpu,
        prefix_k_strided_gpu.as_ptr(),
        prefix_v_strided_gpu.as_ptr(),
        &tree_k_gpu,
        &tree_v_gpu,
        &visibility_gpu,
        &mut out_strided,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(
        download_f32_bits(&out_ref)?,
        download_f32_bits(&out_strided)?
    );
    Ok(())
}

fn cpu_tree_conv_pack(
    mixed: &[half::bf16],
    weights: &[half::bf16],
    parent_ids: &[i32; TREE],
) -> Vec<f32> {
    let mut out = vec![0.0f32; TREE * CONV_DIM];
    for t in 0..TREE {
        for c in 0..CONV_DIM {
            let mut acc = 0.0f32;
            for tap in 0..KERNEL {
                let steps = PAD - tap;
                let source_col = if steps == 0 {
                    PAD + t
                } else {
                    let mut node = t as i32;
                    let mut walked = 0usize;
                    while walked < steps {
                        let parent = parent_ids[node as usize];
                        if parent < 0 {
                            break;
                        }
                        node = parent;
                        walked += 1;
                    }
                    if walked == steps {
                        PAD + node as usize
                    } else {
                        tap + walked
                    }
                };
                let x = mixed[c * TOTAL + source_col].to_f32();
                let w = weights[c * KERNEL + tap].to_f32();
                acc += x * w;
            }
            let silu = acc / (1.0 + (-acc).exp());
            out[t * CONV_DIM + c] = half::bf16::from_f32(silu).to_f32();
        }
    }
    out
}

fn cpu_tree_conv_source_cols(parent_ids: &[i32; TREE]) -> Vec<u32> {
    let mut out = Vec::with_capacity(TREE * KERNEL);
    for t in 0..TREE {
        for tap in 0..KERNEL {
            let steps = PAD - tap;
            let source_col = if steps == 0 {
                PAD + t
            } else {
                let mut node = t as i32;
                let mut walked = 0usize;
                while walked < steps {
                    let parent = parent_ids[node as usize];
                    if parent < 0 {
                        break;
                    }
                    node = parent;
                    walked += 1;
                }
                if walked == steps {
                    PAD + node as usize
                } else {
                    tap + walked
                }
            };
            out.push(source_col as u32);
        }
    }
    out
}

fn cpu_tail_from_src(src: &[half::bf16]) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; CONV_DIM * PAD];
    for c in 0..CONV_DIM {
        for t in 0..PAD {
            out[c * PAD + t] = src[(TREE - PAD + t) * CONV_DIM + c];
        }
    }
    out
}

fn patterned_bf16(len: usize, scale: f32, offset: f32) -> Vec<half::bf16> {
    (0..len)
        .map(|i| half::bf16::from_f32(((i * 37 % 101) as f32 - 50.0) * scale + offset))
        .collect()
}

fn upload_bf16(ordinal: usize, shape: &[usize], values: &[half::bf16]) -> Result<GpuBuffer> {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2) };
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, bytes).map_err(Into::into)
}

fn upload_i32_as_u32(ordinal: usize, values: &[i32]) -> Result<GpuBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[values.len()], &bytes)
        .map_err(Into::into)
}

fn upload_u32(ordinal: usize, shape: &[usize], values: &[u32]) -> Result<GpuBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, shape, &bytes).map_err(Into::into)
}

fn upload_u8(ordinal: usize, shape: &[usize], values: &[u8]) -> Result<GpuBuffer> {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, shape, values).map_err(Into::into)
}

fn download_bf16(buffer: &GpuBuffer) -> Result<Vec<half::bf16>> {
    Ok(buffer
        .to_host_bytes()?
        .chunks_exact(2)
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

fn download_bf16_as_f32(buffer: &GpuBuffer) -> Result<Vec<f32>> {
    Ok(download_bf16(buffer)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect())
}

fn download_f32_bits(buffer: &GpuBuffer) -> Result<Vec<u32>> {
    Ok(buffer
        .to_host_bytes()?
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn assert_close(label: &str, want: &[f32], got: &[f32], tolerance: f32) -> Result<()> {
    if want.len() != got.len() {
        return Err(anyhow!(
            "{label}: length mismatch want={} got={}",
            want.len(),
            got.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        let err = (*a - *b).abs();
        if err > max_abs {
            max_abs = err;
            max_idx = i;
        }
    }
    if max_abs > tolerance {
        return Err(anyhow!(
            "{label}: max_abs={max_abs} at {max_idx}, want={} got={}",
            want[max_idx],
            got[max_idx]
        ));
    }
    Ok(())
}
