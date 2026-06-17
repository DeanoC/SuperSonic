use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};

const CONV_DIM: usize = 7;
const TREE: usize = 5;
const KERNEL: usize = 4;
const PAD: usize = KERNEL - 1;
const TOTAL: usize = PAD + TREE;

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
