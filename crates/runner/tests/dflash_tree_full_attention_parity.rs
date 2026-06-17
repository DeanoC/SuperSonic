use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};

const BATCH: usize = 1;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const PREFIX: usize = 3;
const TREE: usize = 4;
const HD: usize = 16;

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_full_attention_chain_matches_causal_prefill() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let scale = 1.0 / (HD as f32).sqrt();
    let query = patterned(BATCH * Q_HEADS * TREE * HD, 0.0021, -0.04);
    let prefix_key = patterned(BATCH * KV_HEADS * PREFIX * HD, 0.0017, 0.02);
    let prefix_value = patterned(BATCH * KV_HEADS * PREFIX * HD, 0.0013, -0.01);
    let tree_key = patterned(BATCH * KV_HEADS * TREE * HD, 0.0019, 0.03);
    let tree_value = patterned(BATCH * KV_HEADS * TREE * HD, 0.0023, -0.02);
    let visibility = lower_tri_visibility(TREE);

    let q_buf = upload_f32(ordinal, &[BATCH, Q_HEADS, TREE, HD], &query)?;
    let pk_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, PREFIX, HD], &prefix_key)?;
    let pv_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, PREFIX, HD], &prefix_value)?;
    let tk_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, TREE, HD], &tree_key)?;
    let tv_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, TREE, HD], &tree_value)?;
    let visibility_buf = upload_u8(ordinal, &[TREE, TREE], &visibility)?;
    let mut tree_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BATCH, Q_HEADS, TREE, HD])?;

    kernel_ffi::prefill_ffi::full_attention_tree_prefill(
        ordinal,
        ScalarType::F32,
        BATCH,
        Q_HEADS,
        KV_HEADS,
        TREE,
        PREFIX,
        HD,
        scale,
        &q_buf,
        &pk_buf,
        &pv_buf,
        &tk_buf,
        &tv_buf,
        &visibility_buf,
        &mut tree_out,
    )?;

    let concat_key = concat_prefix_tree_kv(&prefix_key, &tree_key);
    let concat_value = concat_prefix_tree_kv(&prefix_value, &tree_value);
    let ck_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, PREFIX + TREE, HD], &concat_key)?;
    let cv_buf = upload_f32(
        ordinal,
        &[BATCH, KV_HEADS, PREFIX + TREE, HD],
        &concat_value,
    )?;
    let mut causal_out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BATCH, Q_HEADS, TREE, HD])?;
    kernel_ffi::prefill_ffi::full_attention_prefill(
        ordinal,
        ScalarType::F32,
        BATCH,
        Q_HEADS,
        KV_HEADS,
        TREE,
        PREFIX + TREE,
        HD,
        scale,
        PREFIX,
        &q_buf,
        &ck_buf,
        &cv_buf,
        &mut causal_out,
    )?;
    gpu_hal::sync(ordinal)?;

    assert_close(
        "tree chain vs causal",
        &download_f32(&tree_out)?,
        &download_f32(&causal_out)?,
        1.0e-5,
    )
}

#[test]
#[ignore = "requires a HIP GPU and /dev/kfd access"]
fn dflash_tree_full_attention_matches_cpu_branching_fixture() -> Result<()> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return Ok(());
    }

    let ordinal = 0usize;
    let scale = 1.0 / (HD as f32).sqrt();
    let query = patterned(BATCH * Q_HEADS * TREE * HD, 0.0021, -0.04);
    let prefix_key = patterned(BATCH * KV_HEADS * PREFIX * HD, 0.0017, 0.02);
    let prefix_value = patterned(BATCH * KV_HEADS * PREFIX * HD, 0.0013, -0.01);
    let tree_key = patterned(BATCH * KV_HEADS * TREE * HD, 0.0019, 0.03);
    let tree_value = patterned(BATCH * KV_HEADS * TREE * HD, 0.0023, -0.02);
    let visibility = [
        1u8, 0, 0, 0, //
        1, 1, 0, 0, //
        1, 0, 1, 0, //
        1, 0, 1, 1,
    ];

    let q_buf = upload_f32(ordinal, &[BATCH, Q_HEADS, TREE, HD], &query)?;
    let pk_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, PREFIX, HD], &prefix_key)?;
    let pv_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, PREFIX, HD], &prefix_value)?;
    let tk_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, TREE, HD], &tree_key)?;
    let tv_buf = upload_f32(ordinal, &[BATCH, KV_HEADS, TREE, HD], &tree_value)?;
    let visibility_buf = upload_u8(ordinal, &[TREE, TREE], &visibility)?;
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[BATCH, Q_HEADS, TREE, HD])?;

    kernel_ffi::prefill_ffi::full_attention_tree_prefill(
        ordinal,
        ScalarType::F32,
        BATCH,
        Q_HEADS,
        KV_HEADS,
        TREE,
        PREFIX,
        HD,
        scale,
        &q_buf,
        &pk_buf,
        &pv_buf,
        &tk_buf,
        &tv_buf,
        &visibility_buf,
        &mut out,
    )?;
    gpu_hal::sync(ordinal)?;

    let want = cpu_tree_attention(
        &query,
        &prefix_key,
        &prefix_value,
        &tree_key,
        &tree_value,
        &visibility,
        scale,
    );
    assert_close("tree branch vs cpu", &want, &download_f32(&out)?, 1.0e-5)
}

fn cpu_tree_attention(
    query: &[f32],
    prefix_key: &[f32],
    prefix_value: &[f32],
    tree_key: &[f32],
    tree_value: &[f32],
    visibility: &[u8],
    scale: f32,
) -> Vec<f32> {
    let groups = Q_HEADS / KV_HEADS;
    let mut out = vec![0.0f32; BATCH * Q_HEADS * TREE * HD];
    for batch in 0..BATCH {
        for q_head in 0..Q_HEADS {
            let kv_head = q_head / groups;
            for q_pos in 0..TREE {
                let q_off = ((batch * Q_HEADS + q_head) * TREE + q_pos) * HD;
                let mut scores = Vec::with_capacity(PREFIX + TREE);
                let mut values = Vec::with_capacity(PREFIX + TREE);
                for k_pos in 0..PREFIX {
                    let k_off = ((batch * KV_HEADS + kv_head) * PREFIX + k_pos) * HD;
                    let v_off = k_off;
                    scores.push(
                        dot(&query[q_off..q_off + HD], &prefix_key[k_off..k_off + HD]) * scale,
                    );
                    values.push(&prefix_value[v_off..v_off + HD]);
                }
                for k_pos in 0..TREE {
                    if visibility[q_pos * TREE + k_pos] == 0 {
                        continue;
                    }
                    let k_off = ((batch * KV_HEADS + kv_head) * TREE + k_pos) * HD;
                    let v_off = k_off;
                    scores
                        .push(dot(&query[q_off..q_off + HD], &tree_key[k_off..k_off + HD]) * scale);
                    values.push(&tree_value[v_off..v_off + HD]);
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut weights = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .collect::<Vec<_>>();
                let denom = weights.iter().sum::<f32>();
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let out_off = ((batch * Q_HEADS + q_head) * TREE + q_pos) * HD;
                for d in 0..HD {
                    let mut acc = 0.0f32;
                    for (weight, value) in weights.iter_mut().zip(values.iter()) {
                        acc += *weight * value[d];
                    }
                    out[out_off + d] = acc * inv;
                }
            }
        }
    }
    out
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn concat_prefix_tree_kv(prefix: &[f32], tree: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; BATCH * KV_HEADS * (PREFIX + TREE) * HD];
    for batch in 0..BATCH {
        for head in 0..KV_HEADS {
            for pos in 0..PREFIX {
                let src = ((batch * KV_HEADS + head) * PREFIX + pos) * HD;
                let dst = ((batch * KV_HEADS + head) * (PREFIX + TREE) + pos) * HD;
                out[dst..dst + HD].copy_from_slice(&prefix[src..src + HD]);
            }
            for pos in 0..TREE {
                let src = ((batch * KV_HEADS + head) * TREE + pos) * HD;
                let dst = ((batch * KV_HEADS + head) * (PREFIX + TREE) + PREFIX + pos) * HD;
                out[dst..dst + HD].copy_from_slice(&tree[src..src + HD]);
            }
        }
    }
    out
}

fn lower_tri_visibility(tree_len: usize) -> Vec<u8> {
    let mut visibility = vec![0u8; tree_len * tree_len];
    for row in 0..tree_len {
        for col in 0..=row {
            visibility[row * tree_len + col] = 1;
        }
    }
    visibility
}

fn patterned(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * scale + offset)
        .collect()
}

fn upload_f32(ordinal: usize, shape: &[usize], values: &[f32]) -> Result<GpuBuffer> {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, shape, &f32_bytes(values))
        .map_err(Into::into)
}

fn upload_u8(ordinal: usize, shape: &[usize], values: &[u8]) -> Result<GpuBuffer> {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, shape, values).map_err(Into::into)
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
