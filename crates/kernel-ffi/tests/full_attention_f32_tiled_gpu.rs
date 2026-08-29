use std::time::Instant;

use gpu_hal::{GpuBuffer, ScalarType};

const Q_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const Q_LEN: usize = 16;
const KV_LEN: usize = 160;
const HEAD_DIM: usize = 128;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn input_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let phase = ((index % 97) as f32 - 48.0) / 48.0;
            phase * (1.0 + (index as f32 / 512.0).sin())
        })
        .collect()
}

fn host_attention(query: &[f32], key: &[f32], value: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; Q_HEADS * Q_LEN * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for q_head in 0..Q_HEADS {
        let kv_head = q_head / (Q_HEADS / KV_HEADS);
        for q_row in 0..Q_LEN {
            let q_offset = ((q_head * Q_LEN + q_row) * HEAD_DIM) as usize;
            let mut scores = vec![0.0_f32; KV_LEN];
            for kv_row in 0..KV_LEN {
                let mut dot = 0.0_f32;
                for dim in 0..HEAD_DIM {
                    dot += query[q_offset + dim]
                        * key[((kv_head * KV_LEN + kv_row) * HEAD_DIM + dim) as usize];
                }
                scores[kv_row] = dot * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (score - max).exp()).sum();
            for dim in 0..HEAD_DIM {
                let mut accumulator = 0.0_f32;
                for kv_row in 0..KV_LEN {
                    let value_offset = ((kv_head * KV_LEN + kv_row) * HEAD_DIM + dim) as usize;
                    accumulator += (scores[kv_row] - max).exp() * value[value_offset] / denominator;
                }
                output[q_offset + dim] = accumulator;
            }
        }
    }
    output
}

fn call_attention(query: &GpuBuffer, key: &GpuBuffer, value: &GpuBuffer, output: &mut GpuBuffer) {
    kernel_ffi::prefill_ffi::full_attention_prefill(
        0,
        ScalarType::F32,
        1,
        Q_HEADS,
        KV_HEADS,
        Q_LEN,
        KV_LEN,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        KV_LEN - 1,
        query,
        key,
        value,
        output,
    )
    .expect("F32 full-attention prefill");
    gpu_hal::sync(0).expect("synchronize F32 full-attention prefill");
}

#[test]
fn hip_full_attention_f32_uses_fast_tiled_path_on_gfx1201() {
    let Ok((arch, _total_vram)) = kernel_ffi::query_gpu_info(0) else {
        eprintln!("skip: HIP device unavailable");
        return;
    };
    if arch != "gfx1201" {
        eprintln!("skip: F32 tiled-attention latency gate is gfx1201-only, got {arch}");
        return;
    }

    let query_values = input_values(Q_HEADS * Q_LEN * HEAD_DIM);
    let key_values = input_values(KV_HEADS * KV_LEN * HEAD_DIM);
    let value_values = input_values(KV_HEADS * KV_LEN * HEAD_DIM + 1);
    let value_values = &value_values[..KV_HEADS * KV_LEN * HEAD_DIM];
    let query = GpuBuffer::from_host_bytes(
        0,
        ScalarType::F32,
        &[Q_HEADS * Q_LEN * HEAD_DIM],
        &f32_bytes(&query_values),
    )
    .expect("upload F32 query");
    let key = GpuBuffer::from_host_bytes(
        0,
        ScalarType::F32,
        &[KV_HEADS * KV_LEN * HEAD_DIM],
        &f32_bytes(&key_values),
    )
    .expect("upload F32 key");
    let value = GpuBuffer::from_host_bytes(
        0,
        ScalarType::F32,
        &[KV_HEADS * KV_LEN * HEAD_DIM],
        &f32_bytes(value_values),
    )
    .expect("upload F32 value");
    let mut output = GpuBuffer::zeros(0, ScalarType::F32, &[Q_HEADS * Q_LEN * HEAD_DIM])
        .expect("allocate F32 attention output");

    call_attention(&query, &key, &value, &mut output);
    let gpu = output
        .to_host_bytes()
        .expect("download F32 attention output");
    let gpu: Vec<f32> = gpu
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 output chunk")))
        .collect();
    let expected = host_attention(&query_values, &key_values, value_values);
    let max_abs = gpu
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 2.0e-4,
        "F32 tiled attention differs from host oracle: max_abs={max_abs}"
    );

    for _ in 0..4 {
        call_attention(&query, &key, &value, &mut output);
    }
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        call_attention(&query, &key, &value, &mut output);
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    let median_ms = samples[samples.len() / 2];
    println!("gfx1201_f32_attention_median_ms={median_ms:.3}");
    assert!(
        median_ms < 2.0,
        "F32 draft attention did not use the tiled path: median={median_ms:.3}ms"
    );
}
