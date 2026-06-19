use anyhow::{anyhow, Result};
use gpu_hal::{GpuBuffer, ScalarType};

const BH: usize = 2;
const SEQ: usize = 4;
const K: usize = 128;
const V: usize = 128;

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_delta_chain_matches_capture() -> Result<()> {
    let ordinal = 0usize;
    let inputs = Inputs::new();
    let mut chain_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ + K, V])?;
    let mut chain_trace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ, K, V])?;
    let mut tree_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ + K, V])?;
    let mut tree_trace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ, K, V])?;
    let parent_ids = upload_i32(ordinal, &[-1, 0, 1, 2])?;

    let gpu = inputs.upload(ordinal)?;
    kernel_ffi::prefill_ffi::delta_recurrent_prefill_capture(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &mut chain_out,
        &mut chain_trace,
    )?;
    kernel_ffi::prefill_ffi::delta_recurrent_tree_prefill_capture(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &parent_ids,
        &mut tree_out,
        &mut tree_trace,
    )?;

    assert_close(
        "chain/tree out",
        &download_f32(&chain_out)?,
        &download_f32(&tree_out)?,
        1.0e-5,
    )?;
    assert_close(
        "chain/tree trace",
        &download_f32(&chain_trace)?,
        &download_f32(&tree_trace)?,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_delta_matches_cpu_branching_fixture() -> Result<()> {
    let ordinal = 0usize;
    let inputs = Inputs::new();
    let parent_host = [-1, 0, 0, 2];
    let parent_ids = upload_i32(ordinal, &parent_host)?;
    let gpu = inputs.upload(ordinal)?;
    let mut tree_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ + K, V])?;
    let mut tree_trace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ, K, V])?;

    kernel_ffi::prefill_ffi::delta_recurrent_tree_prefill_capture(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &parent_ids,
        &mut tree_out,
        &mut tree_trace,
    )?;

    let (want_out, want_trace) = cpu_tree_reference(&inputs, &parent_host);
    assert_close("tree/cpu out", &want_out, &download_f32(&tree_out)?, 1.0e-5)?;
    assert_close(
        "tree/cpu trace",
        &want_trace,
        &download_f32(&tree_trace)?,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_delta_q8_direct_attention_matches_extract_path() -> Result<()> {
    let ordinal = 0usize;
    let inputs = Inputs::new();
    let parent_ids = upload_i32(ordinal, &[-1, 0, 0, 2])?;
    let gpu = inputs.upload(ordinal)?;

    let trace_bytes = q8_trace_bytes(BH, SEQ, K, V);
    let mut old_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ + K, V])?;
    let mut old_trace = GpuBuffer::zeros(ordinal, ScalarType::U8, &[trace_bytes])?;
    let mut old_attn = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[BH, SEQ, V])?;
    let mut old_dummy_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, K, V])?;
    let mut direct_attn = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[BH, SEQ, V])?;
    let mut direct_trace = GpuBuffer::zeros(ordinal, ScalarType::U8, &[trace_bytes])?;

    kernel_ffi::prefill_ffi::delta_recurrent_tree_prefill_capture_q8_trace(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &parent_ids,
        &mut old_out,
        &mut old_trace,
    )?;
    kernel_ffi::prefill_ffi::dflash_extract_recurrent_attn(
        ordinal,
        BH,
        SEQ,
        K,
        V,
        &old_out,
        &mut old_dummy_state,
        &mut old_attn,
    )?;

    kernel_ffi::prefill_ffi::delta_recurrent_tree_prefill_capture_q8_trace_attn(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &parent_ids,
        &mut direct_attn,
        &mut direct_trace,
    )?;

    assert_eq!(
        old_attn.to_host_bytes()?,
        direct_attn.to_host_bytes()?,
        "direct BF16 attention output must match extract path"
    );
    assert_eq!(
        old_trace.to_host_bytes()?,
        direct_trace.to_host_bytes()?,
        "direct path must preserve Q8 rollback trace bytes"
    );
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_append_delta_q8_direct_attention_matches_extract_path() -> Result<()> {
    let ordinal = 0usize;
    let inputs = Inputs::new();
    let gpu = inputs.upload(ordinal)?;

    let trace_bytes = q8_trace_bytes(BH, SEQ, K, V);
    let mut old_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, SEQ + K, V])?;
    let mut old_trace = GpuBuffer::zeros(ordinal, ScalarType::U8, &[trace_bytes])?;
    let mut old_attn = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[BH, SEQ, V])?;
    let mut old_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BH, K, V])?;
    let mut direct_state = upload_f32(ordinal, &[BH, K, V], &inputs.initial)?;
    let mut direct_attn = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[BH, SEQ, V])?;
    let mut direct_trace = GpuBuffer::zeros(ordinal, ScalarType::U8, &[trace_bytes])?;

    kernel_ffi::prefill_ffi::delta_recurrent_prefill_capture_q8_trace(
        ordinal,
        ScalarType::F32,
        BH,
        SEQ,
        K,
        V,
        &gpu.initial,
        &gpu.query,
        &gpu.key,
        &gpu.value,
        &gpu.beta,
        &gpu.g,
        &mut old_out,
        &mut old_trace,
    )?;
    kernel_ffi::prefill_ffi::dflash_extract_recurrent_attn(
        ordinal,
        BH,
        SEQ,
        K,
        V,
        &old_out,
        &mut old_state,
        &mut old_attn,
    )?;

    assert!(
        kernel_ffi::prefill_ffi::delta_recurrent_prefill_capture_q8_trace_attn(
            ordinal,
            ScalarType::F32,
            BH,
            SEQ,
            K,
            V,
            &mut direct_state,
            &gpu.query,
            &gpu.key,
            &gpu.value,
            &gpu.beta,
            &gpu.g,
            &mut direct_attn,
            &mut direct_trace,
        )?
    );

    assert_eq!(
        old_attn.to_host_bytes()?,
        direct_attn.to_host_bytes()?,
        "direct append BF16 attention output must match extract path"
    );
    assert_eq!(
        old_trace.to_host_bytes()?,
        direct_trace.to_host_bytes()?,
        "direct append path must preserve Q8 rollback trace bytes"
    );
    assert_close(
        "direct append recurrent state",
        &download_f32(&old_state)?,
        &download_f32(&direct_state)?,
        1.0e-5,
    )?;
    Ok(())
}

struct Inputs {
    initial: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    beta: Vec<f32>,
    g: Vec<f32>,
}

struct GpuInputs {
    initial: GpuBuffer,
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    beta: GpuBuffer,
    g: GpuBuffer,
}

impl Inputs {
    fn new() -> Self {
        Self {
            initial: patterned(BH * K * V, 0.0007, -0.01),
            query: patterned(BH * SEQ * K, 0.0011, 0.02),
            key: patterned(BH * SEQ * K, 0.0009, -0.015),
            value: patterned(BH * SEQ * V, 0.0013, 0.01),
            beta: patterned(BH * SEQ, 0.002, 0.2),
            g: patterned(BH * SEQ, 0.0005, -0.01),
        }
    }

    fn upload(&self, ordinal: usize) -> Result<GpuInputs> {
        Ok(GpuInputs {
            initial: upload_f32(ordinal, &[BH, K, V], &self.initial)?,
            query: upload_f32(ordinal, &[BH, SEQ, K], &self.query)?,
            key: upload_f32(ordinal, &[BH, SEQ, K], &self.key)?,
            value: upload_f32(ordinal, &[BH, SEQ, V], &self.value)?,
            beta: upload_f32(ordinal, &[BH, SEQ], &self.beta)?,
            g: upload_f32(ordinal, &[BH, SEQ], &self.g)?,
        })
    }
}

fn patterned(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * scale + offset)
        .collect()
}

fn cpu_tree_reference(inputs: &Inputs, parent_ids: &[i32; SEQ]) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; BH * (SEQ + K) * V];
    let mut trace = vec![0.0f32; BH * SEQ * K * V];
    for bh in 0..BH {
        for t in 0..SEQ {
            for v_idx in 0..V {
                let mut state = [0.0f32; K];
                if parent_ids[t] < 0 {
                    for k_idx in 0..K {
                        state[k_idx] = inputs.initial[(bh * K + k_idx) * V + v_idx];
                    }
                } else {
                    let parent = parent_ids[t] as usize;
                    for k_idx in 0..K {
                        state[k_idx] = trace[((bh * SEQ + parent) * K + k_idx) * V + v_idx];
                    }
                }

                let g_t = inputs.g[bh * SEQ + t].exp();
                let beta_t = inputs.beta[bh * SEQ + t];
                for value in state.iter_mut() {
                    *value *= g_t;
                }
                let mut kv_mem = 0.0f32;
                for k_idx in 0..K {
                    kv_mem += state[k_idx] * inputs.key[(bh * SEQ + t) * K + k_idx];
                }
                let delta = (inputs.value[(bh * SEQ + t) * V + v_idx] - kv_mem) * beta_t;
                let mut out_t = 0.0f32;
                for k_idx in 0..K {
                    state[k_idx] += inputs.key[(bh * SEQ + t) * K + k_idx] * delta;
                    trace[((bh * SEQ + t) * K + k_idx) * V + v_idx] = state[k_idx];
                    out_t += state[k_idx] * inputs.query[(bh * SEQ + t) * K + k_idx];
                }
                out[(bh * (SEQ + K) + t) * V + v_idx] = out_t;
                if t + 1 == SEQ {
                    for k_idx in 0..K {
                        out[(bh * (SEQ + K) + SEQ + k_idx) * V + v_idx] = state[k_idx];
                    }
                }
            }
        }
    }
    (out, trace)
}

fn upload_f32(ordinal: usize, shape: &[usize], values: &[f32]) -> Result<GpuBuffer> {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, shape, &f32_bytes(values))
        .map_err(Into::into)
}

fn upload_i32(ordinal: usize, values: &[i32]) -> Result<GpuBuffer> {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[values.len()], &bytes)
        .map_err(Into::into)
}

fn q8_trace_bytes(
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
) -> usize {
    const QK8_0: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 34;
    batch_heads * seq_len * v_head_dim * (k_head_dim / QK8_0) * Q8_0_BLOCK_BYTES
}

fn download_f32(buffer: &GpuBuffer) -> Result<Vec<f32>> {
    Ok(buffer
        .to_host_bytes()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn assert_close(label: &str, want: &[f32], got: &[f32], tol: f32) -> Result<()> {
    if want.len() != got.len() {
        return Err(anyhow!(
            "{label}: length mismatch want={} got={}",
            want.len(),
            got.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (idx, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
        let abs = (w - g).abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = idx;
        }
    }
    if max_abs > tol {
        return Err(anyhow!(
            "{label}: max_abs={max_abs:e} at {max_idx} want={:e} got={:e}",
            want[max_idx],
            got[max_idx]
        ));
    }
    Ok(())
}
