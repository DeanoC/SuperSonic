use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpu_hal::{GpuBuffer, ScalarType};
use model_store::gguf::GgufFile;
use model_store::q6_bound::{argmax_f32_as_bf16, f32_to_bf16_rne_finite, raw_q6_scalar_row_f32};

const HIDDEN: usize = 5_120;
const VOCAB: usize = 248_320;
const Q6_ROW_BYTES: usize = 20 * 210;
const TILE_ROWS: usize = 16;
const CPU_SCAN_CAP: Duration = Duration::from_secs(15 * 60);

fn require_gqh_artifacts() -> bool {
    std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
}

fn load_artifact() -> Option<GgufFile> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
        if require_gqh_artifacts() {
            panic!("SUPERSONIC_GQH_GGUF is required for the Q6 scalar-head artifact tests");
        }
        eprintln!("skip: SUPERSONIC_GQH_GGUF is not configured");
        return None;
    };
    let path = PathBuf::from(value);
    assert!(
        path.is_file(),
        "SUPERSONIC_GQH_GGUF points to a missing or unreadable artifact: {}",
        path.display()
    );
    let gguf = GgufFile::open(&path)
        .unwrap_or_else(|error| panic!("open configured artifact {}: {error}", path.display()));

    let (arch, total_vram) = kernel_ffi::query_gpu_info(0)
        .unwrap_or_else(|error| panic!("query logical HIP device 0: {error}"));
    assert_eq!(arch, "gfx1201", "logical HIP device 0 architecture");
    assert!(
        kernel_ffi::prefill_ffi::device_supports_q6_scalar_head(0)
            .expect("query Q6 scalar-head device support"),
        "logical HIP device 0 must support the Q6 scalar head"
    );
    println!(
        "gpu_logical_ordinal=0 hip_visible_devices={} arch={} total_vram={} artifact={}",
        std::env::var("HIP_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()),
        arch,
        total_vram,
        path.display()
    );
    Some(gguf)
}

fn output_weights(gguf: &GgufFile) -> &[u8] {
    let tensor = gguf
        .tensor("output.weight")
        .expect("configured artifact must contain output.weight");
    assert_eq!(tensor.tensor_type, 14, "output.weight qtype");
    assert_eq!(tensor.dims, [HIDDEN, VOCAB], "output.weight dimensions");
    let weights = gguf
        .tensor_bytes("output.weight")
        .expect("read output.weight bytes");
    assert_eq!(weights.len(), VOCAB * Q6_ROW_BYTES, "output.weight bytes");
    weights
}

fn activation_bits() -> Vec<u16> {
    let activation: Vec<u16> = (0..5120)
        .map(|i| half::bf16::from_f32(((i % 257) as f32 - 128.0) / 128.0).to_bits())
        .collect();
    activation
}

fn upload_inputs(activation: &[u16], weights: &[u8]) -> (GpuBuffer, GpuBuffer) {
    assert_eq!(activation.len(), HIDDEN, "activation length");
    let activation_bytes: Vec<u8> = activation
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let lhs = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[HIDDEN], &activation_bytes)
        .expect("upload activation");
    let rhs = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[weights.len()], weights)
        .expect("upload output.weight");
    (lhs, rhs)
}

fn run_rows(
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    row_start: usize,
    row_count: usize,
) -> (Vec<u8>, Duration) {
    let mut output = GpuBuffer::zeros(0, ScalarType::F32, &[VOCAB]).expect("fresh F32 output");
    let started = Instant::now();
    kernel_ffi::prefill_ffi::q6_k_scalar_head_f32(0, lhs, rhs, &mut output, row_start, row_count)
        .unwrap_or_else(|error| {
            panic!(
                "GPU scalar-head rows {row_start}..{}: {error}",
                row_start + row_count
            )
        });
    let elapsed = started.elapsed();
    let bytes = output.to_host_bytes().expect("download F32 output");
    (bytes, elapsed)
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len(), VOCAB * size_of::<f32>(), "F32 output bytes");
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 chunk")))
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("fnv1a64:{hash:016x}")
}

#[test]
#[ignore = "requires the configured Q3KXL GGUF and a gfx1201 HIP device"]
fn q6_scalar_head_full_row_matches_cpu_oracle() {
    let Some(gguf) = load_artifact() else {
        return;
    };
    let weights = output_weights(&gguf);
    let activation = activation_bits();
    let (lhs, rhs) = upload_inputs(&activation, weights);

    let (first_bytes, first_gpu_elapsed) = run_rows(&lhs, &rhs, 0, VOCAB);
    let (second_bytes, second_gpu_elapsed) = run_rows(&lhs, &rhs, 0, VOCAB);
    assert_eq!(second_bytes, first_bytes, "full-row GPU F32 bits changed");

    let gpu_logits = decode_f32(&first_bytes);
    let cpu_started = Instant::now();
    let mut cpu_logits = Vec::with_capacity(VOCAB);
    for (row_index, row) in weights.chunks_exact(Q6_ROW_BYTES).enumerate() {
        if row_index % 1_024 == 0 {
            assert!(
                cpu_started.elapsed() <= CPU_SCAN_CAP,
                "full CPU oracle scan exceeded the 15-minute cap at row {row_index}/{VOCAB}"
            );
        }
        let cpu = raw_q6_scalar_row_f32(row, &activation)
            .unwrap_or_else(|error| panic!("CPU oracle row {row_index}: {error}"));
        let gpu = gpu_logits[row_index];
        assert_eq!(
            gpu.to_bits(),
            cpu.to_bits(),
            "CPU/GPU F32 mismatch at row {row_index}: gpu={gpu:?} cpu={cpu:?}"
        );
        assert_eq!(
            f32_to_bf16_rne_finite(gpu).expect("finite GPU logit"),
            f32_to_bf16_rne_finite(cpu).expect("finite CPU logit"),
            "CPU/GPU BF16 mismatch at row {row_index}"
        );
        cpu_logits.push(cpu);
    }
    assert_eq!(cpu_logits.len(), VOCAB, "CPU oracle row count");
    let cpu_elapsed = cpu_started.elapsed();
    let gpu_winner = argmax_f32_as_bf16(&gpu_logits).expect("GPU BF16 winner");
    let cpu_winner = argmax_f32_as_bf16(&cpu_logits).expect("CPU BF16 winner");
    assert_eq!(gpu_winner, cpu_winner, "CPU/GPU BF16 winner");

    println!(
        "full_row_digest={} winner={} gpu_first_ms={} gpu_second_ms={} cpu_scan_ms={} rows={}",
        digest(&first_bytes),
        gpu_winner,
        first_gpu_elapsed.as_millis(),
        second_gpu_elapsed.as_millis(),
        cpu_elapsed.as_millis(),
        VOCAB
    );
}

#[test]
#[ignore = "requires the configured Q3KXL GGUF and a gfx1201 HIP device"]
fn q6_scalar_head_tiled_matches_full_row() {
    let Some(gguf) = load_artifact() else {
        return;
    };
    let weights = output_weights(&gguf);
    let activation = activation_bits();
    let (lhs, rhs) = upload_inputs(&activation, weights);

    let (full_bytes, full_elapsed) = run_rows(&lhs, &rhs, 0, VOCAB);
    let mut tiled = GpuBuffer::zeros(0, ScalarType::F32, &[VOCAB]).expect("tiled output");
    let tiled_started = Instant::now();
    for row_start in (0..VOCAB).step_by(TILE_ROWS) {
        kernel_ffi::prefill_ffi::q6_k_scalar_head_f32(
            0, &lhs, &rhs, &mut tiled, row_start, TILE_ROWS,
        )
        .unwrap_or_else(|error| {
            panic!(
                "tiled GPU rows {row_start}..{}: {error}",
                row_start + TILE_ROWS
            )
        });
    }
    let tiled_bytes = tiled.to_host_bytes().expect("download tiled output");
    let tiled_elapsed = tiled_started.elapsed();
    assert_eq!(tiled_bytes, full_bytes, "tiled/full GPU F32 bytes");
    let winner = argmax_f32_as_bf16(&decode_f32(&full_bytes)).expect("tiled/full BF16 winner");

    println!(
        "tiled_row_digest={} winner={} full_ms={} tiled_ms={} tiles={}",
        digest(&full_bytes),
        winner,
        full_elapsed.as_millis(),
        tiled_elapsed.as_millis(),
        VOCAB / TILE_ROWS
    );
}
