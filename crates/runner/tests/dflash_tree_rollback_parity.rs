use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};

const QKV_DIM: usize = 5;
const PAD: usize = 3;
const TREE: usize = 5;
const NV: usize = 2;
const KHD: usize = 4;
const VHD: usize = 3;

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_rollback_gathers_branch_tail_and_terminal_state() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let accepted = [1u32, 3u32];
    let conv_input = patterned_bf16(QKV_DIM * (PAD + TREE), 0.013, -0.2);
    let recurrent_trace = patterned_f32(NV * TREE * KHD * VHD, 0.007, 0.04);
    let mut conv_state = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[QKV_DIM, PAD])?;
    let mut recurrent_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[NV, KHD, VHD])?;

    let conv_input_gpu = upload_bf16(ordinal, &[QKV_DIM, PAD + TREE], &conv_input)?;
    let accepted_gpu = upload_u32(ordinal, &accepted)?;
    let recurrent_trace_gpu = upload_f32(ordinal, &[NV, TREE, KHD, VHD], &recurrent_trace)?;

    kernel_ffi::prefill_ffi::dflash_apply_tree_rollback(
        ordinal,
        ScalarType::BF16,
        QKV_DIM,
        PAD,
        PAD + TREE,
        TREE,
        accepted.len(),
        NV,
        KHD,
        VHD,
        &conv_input_gpu,
        &accepted_gpu,
        &mut conv_state,
        &recurrent_trace_gpu,
        &mut recurrent_state,
    )?;
    gpu_hal::sync(ordinal)?;

    let (want_conv, want_rec) = cpu_tree_rollback(&conv_input, &recurrent_trace, &accepted);
    assert_eq!(
        want_conv,
        download_bf16(&conv_state)?,
        "conv tail should gather prior tail plus accepted branch nodes"
    );
    assert_close(
        "terminal recurrent state",
        &want_rec,
        &download_f32(&recurrent_state)?,
        0.0,
    )
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_rollback_chain_matches_contiguous_rollback() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let accepted = [0u32, 1u32, 2u32];
    let conv_input = patterned_bf16(QKV_DIM * (PAD + TREE), 0.011, -0.05);
    let recurrent_trace = patterned_f32(NV * TREE * KHD * VHD, 0.005, -0.02);
    let conv_input_gpu = upload_bf16(ordinal, &[QKV_DIM, PAD + TREE], &conv_input)?;
    let recurrent_trace_gpu = upload_f32(ordinal, &[NV, TREE, KHD, VHD], &recurrent_trace)?;

    let accepted_gpu = upload_u32(ordinal, &accepted)?;
    let mut tree_conv = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[QKV_DIM, PAD])?;
    let mut tree_rec = GpuBuffer::zeros(ordinal, ScalarType::F32, &[NV, KHD, VHD])?;
    kernel_ffi::prefill_ffi::dflash_apply_tree_rollback(
        ordinal,
        ScalarType::BF16,
        QKV_DIM,
        PAD,
        PAD + TREE,
        TREE,
        accepted.len(),
        NV,
        KHD,
        VHD,
        &conv_input_gpu,
        &accepted_gpu,
        &mut tree_conv,
        &recurrent_trace_gpu,
        &mut tree_rec,
    )?;

    let mut chain_conv = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[QKV_DIM, PAD])?;
    let mut chain_rec = GpuBuffer::zeros(ordinal, ScalarType::F32, &[NV, KHD, VHD])?;
    kernel_ffi::prefill_ffi::dflash_apply_rollback(
        ordinal,
        ScalarType::BF16,
        QKV_DIM,
        PAD,
        PAD + TREE,
        TREE,
        accepted.len(),
        NV,
        KHD,
        VHD,
        &conv_input_gpu,
        &mut chain_conv,
        &recurrent_trace_gpu,
        &mut chain_rec,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_eq!(download_bf16(&chain_conv)?, download_bf16(&tree_conv)?);
    assert_close(
        "chain recurrent",
        &download_f32(&chain_rec)?,
        &download_f32(&tree_rec)?,
        0.0,
    )
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_contiguous_rollback_bf16_trace_matches_quantized_fixture() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let accepted = [0u32, 1u32, 2u32];
    let conv_input = patterned_bf16(QKV_DIM * (PAD + TREE), 0.017, 0.03);
    let recurrent_trace_bf16 = patterned_bf16(NV * TREE * KHD * VHD, 0.009, -0.07);
    let recurrent_trace_f32: Vec<f32> = recurrent_trace_bf16
        .iter()
        .map(|value| value.to_f32())
        .collect();

    let conv_input_gpu = upload_bf16(ordinal, &[QKV_DIM, PAD + TREE], &conv_input)?;
    let recurrent_trace_gpu = upload_bf16(ordinal, &[NV, TREE, KHD, VHD], &recurrent_trace_bf16)?;
    let mut conv_state = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[QKV_DIM, PAD])?;
    let mut recurrent_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[NV, KHD, VHD])?;

    kernel_ffi::prefill_ffi::dflash_apply_rollback_bf16_trace(
        ordinal,
        ScalarType::BF16,
        QKV_DIM,
        PAD,
        PAD + TREE,
        TREE,
        accepted.len(),
        NV,
        KHD,
        VHD,
        &conv_input_gpu,
        &mut conv_state,
        &recurrent_trace_gpu,
        &mut recurrent_state,
    )?;
    gpu_hal::sync(ordinal)?;

    let (want_conv, want_rec) = cpu_tree_rollback(&conv_input, &recurrent_trace_f32, &accepted);
    assert_eq!(
        want_conv,
        download_bf16(&conv_state)?,
        "conv tail should match contiguous committed positions"
    );
    assert_close(
        "BF16 terminal recurrent state",
        &want_rec,
        &download_f32(&recurrent_state)?,
        0.0,
    )
}

fn cpu_tree_rollback(
    conv_input: &[half::bf16],
    recurrent_trace: &[f32],
    accepted: &[u32],
) -> (Vec<half::bf16>, Vec<f32>) {
    let commit_len = accepted.len();
    let mut conv = vec![half::bf16::ZERO; QKV_DIM * PAD];
    for ch in 0..QKV_DIM {
        for tap in 0..PAD {
            let logical = commit_len as isize + tap as isize - PAD as isize;
            let src_col = if logical < 0 {
                commit_len + tap
            } else {
                PAD + accepted[logical as usize] as usize
            };
            conv[ch * PAD + tap] = conv_input[ch * (PAD + TREE) + src_col];
        }
    }

    let terminal = *accepted.last().expect("non-empty accepted") as usize;
    let mut rec = vec![0.0f32; NV * KHD * VHD];
    let head_elems = KHD * VHD;
    for h in 0..NV {
        let src = (h * TREE + terminal) * head_elems;
        let dst = h * head_elems;
        rec[dst..dst + head_elems].copy_from_slice(&recurrent_trace[src..src + head_elems]);
    }
    (conv, rec)
}

fn patterned_bf16(len: usize, scale: f32, offset: f32) -> Vec<half::bf16> {
    (0..len)
        .map(|i| half::bf16::from_f32(((i * 37 % 101) as f32 - 50.0) * scale + offset))
        .collect()
}

fn patterned_f32(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 29 % 113) as f32 - 56.0) * scale + offset)
        .collect()
}

fn upload_bf16(ordinal: usize, shape: &[usize], values: &[half::bf16]) -> Result<GpuBuffer> {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2) };
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, bytes).map_err(Into::into)
}

fn upload_f32(ordinal: usize, shape: &[usize], values: &[f32]) -> Result<GpuBuffer> {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, shape, &f32_bytes(values))
        .map_err(Into::into)
}

fn upload_u32(ordinal: usize, values: &[u32]) -> Result<GpuBuffer> {
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

fn download_f32(buffer: &GpuBuffer) -> Result<Vec<f32>> {
    Ok(buffer
        .to_host_bytes()?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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
