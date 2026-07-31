use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe::{
    int4_descriptor_dequant_smoke_launch, int4_descriptor_wmma_parity_launch,
    qwen36_moe_hip_int4_descriptor_wmma_parity_launch, Qwen36MoeInt4WeightDesc,
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

fn tile_v1_desc(scale: &GpuBuffer, zero: &GpuBuffer, cols: usize) -> Qwen36MoeInt4WeightDesc {
    Qwen36MoeInt4WeightDesc {
        scale: scale.as_ptr(),
        zero: zero.as_ptr(),
        packed_row_stride_bytes: (cols / 2) as u64,
        packed_expert_stride_bytes: 0,
        scale_row_stride_elements: (cols / 128) as u64,
        scale_expert_stride_elements: 0,
        input_group_size: 128,
        output_group_size: 128,
        implicit_zero_code: -1,
        encoding: 1,
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
        &scale,
        None,
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

    int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        None,
        &desc,
        2,
        1,
        32,
        &mut wide,
        &mut scalar,
    )?;
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
        let code = 9 + (row % 7) as u8;
        for _ in 0..COLS / 2 {
            packed_bytes.push(code | (code << 4));
        }
    }
    let scale_bits = vec![0x3dcdu16; ROWS * COLS / 32];
    let activation_bits = vec![bf16::ONE.to_bits(); COLS];
    let reconstructed_witness = 3.0 * f32::from(bf16::from_bits(0x3dcd));
    assert_ne!(
        reconstructed_witness,
        f32::from(bf16::from_f32(reconstructed_witness)),
        "code 11 with scale 0x3dcd must require BF16 RNE"
    );

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
        &scale,
        None,
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
fn hip_wmma_matches_scalar_for_varied_g32_operands() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    const ROWS: usize = 32;
    const COLS: usize = 128;
    const SCALE_PATTERNS: [[u16; 4]; 2] = [
        [0x3dcd, 0x3e80, 0x3f20, 0x3fc0],
        [0x3e40, 0x3ec0, 0x3f60, 0x3fe0],
    ];

    let mut packed_bytes = Vec::with_capacity(ROWS * COLS / 2);
    for row in 0..ROWS {
        let tile = row / 16;
        let tile_row = row % 16;
        for byte_col in 0..COLS / 2 {
            let low = ((tile_row * 3 + byte_col * 5 + byte_col / 3 + tile * (byte_col % 7 + 3) + 1)
                & 0x0f) as u8;
            let mut high = ((tile_row * 11
                + byte_col * 7
                + (byte_col % 5) * 3
                + tile * ((byte_col * 3) % 11 + 5)
                + 6)
                & 0x0f) as u8;
            if high == low {
                high = (high + 9) & 0x0f;
            }
            packed_bytes.push(low | (high << 4));
        }
    }
    let scale_bits: Vec<u16> = (0..ROWS)
        .flat_map(|row| {
            let tile = row / 16;
            let tile_row = row % 16;
            (0..COLS / 32).map(move |group| SCALE_PATTERNS[tile][(tile_row + group) % 4])
        })
        .collect();
    let activation_bits: Vec<u16> = (0..COLS)
        .map(|col| {
            let magnitude = (((col * 13 + 5) % 29) + 3) as f32 / 16.0;
            let sign = if col % 5 == 0 || col % 11 == 3 {
                -1.0
            } else {
                1.0
            };
            bf16::from_f32(sign * magnitude).to_bits()
        })
        .collect();

    let mut host_reference = vec![0.0f32; ROWS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let packed = packed_bytes[row * COLS / 2 + col / 2];
            let code = if col & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = f32::from(bf16::from_bits(scale_bits[row * 4 + col / 32]));
            let weight = f32::from(bf16::from_f32((f32::from(code) - 8.0) * scale));
            let activation = f32::from(bf16::from_bits(activation_bits[col]));
            host_reference[row] += weight * activation;
        }
    }
    let min_reference = host_reference.iter().copied().fold(f32::INFINITY, f32::min);
    let max_reference = host_reference
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_abs_reference = host_reference
        .iter()
        .map(|value| value.abs())
        .fold(f32::INFINITY, f32::min);
    assert!(
        max_reference - min_reference > 20.0 && min_abs_reference > 0.5,
        "varied fixture outputs must be nondegenerate: min={min_reference} max={max_reference} min_abs={min_abs_reference}"
    );
    let min_tile_pair_delta = (0..16)
        .map(|row| (host_reference[row] - host_reference[row + 16]).abs())
        .fold(f32::INFINITY, f32::min);
    assert!(
        min_tile_pair_delta > 1.0,
        "second-tile rows must differ materially from first-tile rows: min_delta={min_tile_pair_delta}"
    );
    let mut min_any_row_delta = f32::INFINITY;
    for lhs in 0..ROWS {
        for rhs in lhs + 1..ROWS {
            min_any_row_delta =
                min_any_row_delta.min((host_reference[lhs] - host_reference[rhs]).abs());
        }
    }
    assert!(
        min_any_row_delta > 5e-2,
        "all expected row outputs must be distinct beyond parity tolerance: min_delta={min_any_row_delta}"
    );

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
        &scale,
        None,
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
    let mut scalar_host_max_abs = 0.0f32;
    for ((&reference, &actual), &host) in scalar.iter().zip(&wmma).zip(&host_reference) {
        dot += f64::from(reference) * f64::from(actual);
        scalar_sq += f64::from(reference) * f64::from(reference);
        wmma_sq += f64::from(actual) * f64::from(actual);
        max_abs = max_abs.max((reference - actual).abs());
        scalar_host_max_abs = scalar_host_max_abs.max((reference - host).abs());
    }
    let cosine = dot / (scalar_sq.sqrt() * wmma_sq.sqrt() + 1e-30);
    println!(
        "varied G32 WMMA parity: cosine={cosine:.8} max_abs={max_abs:.8e} scalar_host_max_abs={scalar_host_max_abs:.8e} min_tile_pair_delta={min_tile_pair_delta:.8e} min_any_row_delta={min_any_row_delta:.8e}"
    );
    assert!(
        scalar_host_max_abs <= 2e-2,
        "scalar host-reference max abs {scalar_host_max_abs:.8e} exceeds 2e-2"
    );
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
    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        None,
        &desc,
        1,
        1,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("FP8 encoding must not enter the INT4 descriptor primitive");
    assert!(
        error.to_string().contains("encoding 3"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn safe_launch_rejects_descriptor_scale_pointer_mismatch_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let other_scale =
        GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f00]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut desc = row_group_desc(&other_scale, 32);
    desc.scale = other_scale.as_ptr();

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        None,
        &desc,
        1,
        1,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("descriptor scale pointer must identify the borrowed scale buffer");
    assert!(error.to_string().contains("scale pointer"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_short_scale_sidecar_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed_bytes = TASK2_PACKED.repeat(2);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[32], &packed_bytes)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[64])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[64])?;
    let desc = row_group_desc(&scale, 32);

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        None,
        &desc,
        1,
        2,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("two rows require two BF16 scale elements");
    assert!(error.to_string().contains("scale sidecar"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_non_bf16_scale_sidecar_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[2], &[0x80, 0x3f])?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let desc = row_group_desc(&scale, 32);

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        None,
        &desc,
        1,
        1,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("scale sidecar dtype must be validated before FFI");
    assert!(error.to_string().contains("BF16"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_zero_buffer_for_implicit_zero_encoding_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let zero = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x4100]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let desc = row_group_desc(&scale, 32);

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        Some(&zero),
        &desc,
        1,
        1,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("encoding 2 must not accept a zero sidecar buffer");
    assert!(error.to_string().contains("zero buffer"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_descriptor_zero_pointer_mismatch_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    const ROWS: usize = 128;
    const COLS: usize = 128;
    let packed = GpuBuffer::zeros(0, ScalarType::U8, &[ROWS * COLS / 2])?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let zero = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x4100]))?;
    let other_zero = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x40e0]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS * COLS])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS * COLS])?;
    let desc = tile_v1_desc(&scale, &other_zero, COLS);

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        Some(&zero),
        &desc,
        1,
        ROWS as i32,
        COLS as i32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("descriptor zero pointer must identify the borrowed zero buffer");
    assert!(error.to_string().contains("zero pointer"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_short_explicit_zero_sidecar_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    const ROWS: usize = 256;
    const COLS: usize = 128;
    let packed = GpuBuffer::zeros(0, ScalarType::U8, &[ROWS * COLS / 2])?;
    let scale =
        GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[2], &bf16_bytes(&[0x3f80, 0x3f80]))?;
    let zero = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x4100]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS * COLS])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS * COLS])?;
    let desc = tile_v1_desc(&scale, &zero, COLS);

    let error = int4_descriptor_dequant_smoke_launch(
        0,
        &packed,
        &scale,
        Some(&zero),
        &desc,
        1,
        ROWS as i32,
        COLS as i32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("two output groups require two BF16 zero elements");
    assert!(error.to_string().contains("zero sidecar"), "{error}");
    Ok(())
}

#[test]
fn safe_launch_rejects_buffer_launch_ordinal_mismatch_before_ffi() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    let packed = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[16], &TASK2_PACKED)?;
    let scale = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[1], &bf16_bytes(&[0x3f80]))?;
    let mut wide = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[32])?;
    let desc = row_group_desc(&scale, 32);

    let error = int4_descriptor_dequant_smoke_launch(
        usize::MAX,
        &packed,
        &scale,
        None,
        &desc,
        1,
        1,
        32,
        &mut wide,
        &mut scalar,
    )
    .expect_err("all borrowed buffers must match the launch ordinal before FFI");
    assert!(error.to_string().contains("device ordinal"), "{error}");
    Ok(())
}

#[test]
fn cpp_wmma_ffi_rejects_single_expert_scale_extent_overflow() -> anyhow::Result<()> {
    set_backend(Backend::Hip);
    const ROWS: usize = 32;
    const COLS: usize = 128;
    let packed = GpuBuffer::zeros(0, ScalarType::U8, &[ROWS * COLS / 2])?;
    let scale = GpuBuffer::zeros(0, ScalarType::BF16, &[ROWS * COLS / 32])?;
    let activation = GpuBuffer::zeros(0, ScalarType::BF16, &[COLS])?;
    let mut scalar = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS])?;
    let mut wmma = GpuBuffer::zeros(0, ScalarType::F32, &[ROWS])?;
    let mut desc = row_group_desc(&scale, COLS);
    desc.scale_row_stride_elements = u64::MAX;

    // The invalid ordinal keeps the pre-fix path from launching after it
    // incorrectly accepts the overflowing one-expert descriptor.
    let status = unsafe {
        qwen36_moe_hip_int4_descriptor_wmma_parity_launch(
            usize::MAX,
            packed.as_ptr() as *const u8,
            &desc,
            activation.as_ptr(),
            ROWS as i32,
            COLS as i32,
            scalar.as_mut_ptr() as *mut f32,
            wmma.as_mut_ptr() as *mut f32,
        )
    };
    let raw_status: u64 = unsafe { std::mem::transmute(status) };
    assert_eq!(
        raw_status, 179,
        "C++ descriptor validation must reject overflow before device probing"
    );
    Ok(())
}
