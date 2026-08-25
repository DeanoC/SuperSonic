use gpu_hal::{GpuBuffer, ScalarType};

fn bf16_bytes(rows: &[&[f32]]) -> Vec<u8> {
    rows.iter()
        .flat_map(|row| {
            row.iter()
                .flat_map(|&value| half::bf16::from_f32(value).to_bits().to_le_bytes())
        })
        .collect()
}

fn f32_bytes(rows: &[&[f32]]) -> Vec<u8> {
    rows.iter()
        .flat_map(|row| row.iter().flat_map(|&value| value.to_le_bytes()))
        .collect()
}

fn read_u32(buffer: &GpuBuffer, rows: usize) -> Vec<u32> {
    buffer
        .to_host_bytes()
        .expect("download argmax indices")
        .chunks_exact(4)
        .take(rows)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("U32 chunk")))
        .collect()
}

#[test]
fn hip_argmax_rows_matches_host_bf16_ordering_edges() {
    let Ok((_name, _clock)) = kernel_ffi::query_gpu_info(0) else {
        eprintln!("skip: HIP device unavailable");
        return;
    };

    let rows: &[&[f32]] = &[
        &[-1.5e30, -1.1e30, -1.2e30, -1.3e30],
        &[f32::NAN, -2.0, -3.0, -4.0],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, -1.0, -2.0],
        &[f32::NEG_INFINITY, f32::INFINITY, 0.0, f32::INFINITY],
        &[1.0, 1.001, 0.0, f32::NAN],
    ];
    let expected = [1, 1, 2, 1, 0];
    let rows_count = rows.len();
    let cols = rows[0].len();

    let bf16_logits =
        GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[rows_count, cols], &bf16_bytes(rows))
            .expect("upload BF16 logits");
    let mut bf16_indices = GpuBuffer::zeros(0, ScalarType::U32, &[rows_count]).expect("indices");
    kernel_ffi::prefill_ffi::argmax_bf16_rows(0, rows_count, cols, &bf16_logits, &mut bf16_indices)
        .expect("BF16 argmax");
    assert_eq!(read_u32(&bf16_indices, rows_count), expected);

    let f32_logits =
        GpuBuffer::from_host_bytes(0, ScalarType::F32, &[rows_count, cols], &f32_bytes(rows))
            .expect("upload F32 logits");
    let mut f32_indices = GpuBuffer::zeros(0, ScalarType::U32, &[rows_count]).expect("indices");
    kernel_ffi::prefill_ffi::argmax_f32_as_bf16_rows(
        0,
        rows_count,
        cols,
        &f32_logits,
        &mut f32_indices,
    )
    .expect("F32-as-BF16 argmax");
    assert_eq!(read_u32(&f32_indices, rows_count), expected);
}
