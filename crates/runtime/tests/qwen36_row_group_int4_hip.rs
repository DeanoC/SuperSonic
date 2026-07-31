use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe::{
    int4_descriptor_dequant_smoke_launch, int4_descriptor_wmma_parity_launch,
    Qwen36MoeInt4WeightDesc,
};

// Task 2's binding row-group fixture, copied verbatim rather than packed here.
const TASK2_PACKED: [u8; 16] = [
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f,
];

fn bf16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_values(buffer: &GpuBuffer) -> anyhow::Result<Vec<f32>> {
    Ok(buffer
        .to_host_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn row_group_desc(scale: &GpuBuffer, cols: usize) -> Qwen36MoeInt4WeightDesc {
    Qwen36MoeInt4WeightDesc {
        scale: scale.as_ptr(),
        zero: std::ptr::null(),
        packed_row_stride_bytes: (cols / 2) as u64,
        packed_expert_stride_bytes: 0,
        scale_row_stride_elements: (cols / 32) as u64,
        scale_expert_stride_elements: 0,
        input_group_size: 32,
        output_group_size: 1,
        implicit_zero_code: 8,
        encoding: 2,
    }
}

fn run_known_bytes(scale_bits: u16) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let rows = 1;
    let cols = 32;
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[scale_bits]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[rows * cols])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[rows * cols])?;
    let desc = row_group_desc(&scale, cols);
    int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &desc,
        1,
        rows as i32,
        cols as i32,
        &mut wide,
        &mut scalar,
    )?;
    Ok((f32_values(&wide)?, f32_values(&scalar)?))
}

#[test]
fn hip_scalar_and_8wide_reconstruct_task2_g32_known_bytes() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let (wide, scalar) = run_known_bytes(0x3f80)?;
    let nibbles = [
        1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 8,
    ];
    for (index, nibble) in nibbles.into_iter().enumerate() {
        let expected = f32::from(nibble) - 8.0;
        assert_eq!(wide[index], expected, "8-wide mismatch at col={index}");
        assert_eq!(scalar[index], expected, "scalar mismatch at col={index}");
    }

    // 0x3dcd is BF16 0.10009765625. Expectations use that represented
    // value, not the source decimal, to bind the descriptor's BF16 scale.
    let scale = f32::from(bf16::from_bits(0x3dcd));
    let (wide, scalar) = run_known_bytes(0x3dcd)?;
    for (index, nibble) in nibbles.into_iter().enumerate() {
        let expected = (f32::from(nibble) - 8.0) * scale;
        assert!(
            (wide[index] - expected).abs() <= 1e-6,
            "non-unit 8-wide mismatch at col={index}: got={} expected={expected}",
            wide[index]
        );
        assert!(
            (scalar[index] - expected).abs() <= 1e-6,
            "non-unit scalar mismatch at col={index}: got={} expected={expected}",
            scalar[index]
        );
    }
    Ok(())
}

#[test]
fn hip_scalar_and_8wide_follow_explicit_expert_strides() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let mut packed_bytes = vec![0u8; 36];
    packed_bytes[..16].copy_from_slice(&TASK2_PACKED);
    packed_bytes[20..].copy_from_slice(&TASK2_PACKED);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[36], &packed_bytes)?;
    let scale = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[3],
        &bf16_bytes(&[0x3f80, 0x0000, 0x3f00]),
    )?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[64])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[64])?;
    let mut desc = row_group_desc(&scale, 32);
    desc.packed_expert_stride_bytes = 20;
    desc.scale_expert_stride_elements = 2;

    int4_descriptor_dequant_smoke_launch(0, &packed, &desc, 2, 1, 32, &mut wide, &mut scalar)?;
    let wide = f32_values(&wide)?;
    let scalar = f32_values(&scalar)?;
    let nibbles = [
        1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 8,
    ];
    for expert in 0..2 {
        let scale = if expert == 0 { 1.0 } else { 0.5 };
        for (col, nibble) in nibbles.into_iter().enumerate() {
            let expected = (f32::from(nibble) - 8.0) * scale;
            let index = expert * 32 + col;
            assert_eq!(wide[index], expected, "8-wide e={expert} col={col}");
            assert_eq!(scalar[index], expected, "scalar e={expert} col={col}");
        }
    }
    Ok(())
}

#[test]
fn hip_wmma_matches_scalar_g32_matvec() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    const ROWS: usize = 32;
    const COLS: usize = 128;

    let mut packed_bytes = Vec::with_capacity(ROWS * COLS / 2);
    for row in 0..ROWS {
        for byte_col in 0..COLS / 2 {
            let low = 1 + ((row * 7 + byte_col * 3) % 15) as u8;
            let high = 1 + ((row * 11 + byte_col * 5 + 1) % 15) as u8;
            packed_bytes.push(low | (high << 4));
        }
    }
    let scale_pattern = [0x3f00u16, 0x3f40, 0x3f80, 0x3fc0];
    let scale_bits: Vec<u16> = (0..ROWS)
        .flat_map(|row| (0..COLS / 32).map(move |group| scale_pattern[(row + group) % 4]))
        .collect();
    let activation_bits: Vec<u16> = (0..COLS)
        .map(|col| bf16::from_f32(((col % 17) as f32 - 8.0) / 16.0).to_bits())
        .collect();

    let packed =
        GpuBuffer::from_host_bytes(0, ScalarType::U8, &[packed_bytes.len()], &packed_bytes)?;
    let scale = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[scale_bits.len()],
        &bf16_bytes(&scale_bits),
    )?;
    let activation =
        GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[COLS], &bf16_bytes(&activation_bits))?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS])?;
    let mut wmma = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS])?;
    let desc = row_group_desc(&scale, COLS);

    int4_descriptor_wmma_parity_launch(
        0,
        &packed,
        &desc,
        &activation,
        ROWS as i32,
        COLS as i32,
        &mut scalar,
        &mut wmma,
    )?;

    let scalar = f32_values(&scalar)?;
    let wmma = f32_values(&wmma)?;
    let mut dot = 0.0f64;
    let mut scalar_sq = 0.0f64;
    let mut wmma_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&reference, &actual) in scalar.iter().zip(&wmma) {
        dot += f64::from(reference) * f64::from(actual);
        scalar_sq += f64::from(reference) * f64::from(reference);
        wmma_sq += f64::from(actual) * f64::from(actual);
        max_abs = max_abs.max((reference - actual).abs());
    }
    let cosine = dot / (scalar_sq.sqrt() * wmma_sq.sqrt() + 1e-30);
    println!("G32 WMMA parity: cosine={cosine:.8} max_abs={max_abs:.8e}");
    assert!(cosine >= 0.99999, "WMMA cosine {cosine:.8} below 0.99999");
    assert!(max_abs <= 2e-2, "WMMA max abs {max_abs:.8e} exceeds 2e-2");
    Ok(())
}

#[test]
fn descriptor_surface_keeps_fp8_encoding_distinct() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut desc = row_group_desc(&scale, 32);
    desc.encoding = 3;
    let error =
        int4_descriptor_dequant_smoke_launch(0, &packed, &desc, 1, 1, 32, &mut wide, &mut scalar)
            .expect_err("FP8 encoding must not enter the INT4 descriptor primitive");
    assert!(
        error.to_string().contains("encoding 3"),
        "unexpected error: {error}"
    );
    Ok(())
}
