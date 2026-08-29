use gpu_hal::ScalarType;
use model_store::dflash::{load_draft, DraftConfig};
use model_store::dflash_ref::draft_forward;
use std::path::PathBuf;

fn read_bytes(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn traced_dflash_round_matches_cpu_reference() {
    let root = std::env::var("SUPERSONIC_DFLASH_TRACE_DIR").unwrap_or_else(|_| "/tmp".into());
    let root = PathBuf::from(root);
    let weights_path = PathBuf::from(std::env::var("SUPERSONIC_DFLASH_DRAFT_GGUF").unwrap());
    let weights = load_draft(&weights_path).unwrap();
    let cfg: DraftConfig = weights.config.clone();
    let hidden = cfg.hidden;
    let ntl = cfg.n_target_layers;
    let nq = 16;
    let ctx = 26usize;
    let target = bf16_to_f32(&read_bytes(
        root.join("supersonic-dflash-trace-target.bf16")
            .to_str()
            .unwrap(),
    ));
    let target = &target[..ctx * ntl * hidden];
    let noise = bf16_to_f32(&read_bytes(
        root.join("supersonic-dflash-trace-noise.bf16")
            .to_str()
            .unwrap(),
    ));
    let noise = &noise[..nq * hidden];
    let gpu = f32_bytes(&read_bytes(
        root.join("supersonic-dflash-trace-hidden.f32")
            .to_str()
            .unwrap(),
    ));
    let positions_q: Vec<usize> = (ctx..ctx + nq).collect();
    let positions_k: Vec<usize> = (0..ctx + nq).collect();
    let cpu = draft_forward(&weights, &cfg, target, noise, &positions_q, &positions_k).unwrap();
    let mut sq_err = 0.0f64;
    let mut sq_cpu = 0.0f64;
    let mut max_abs = 0.0f64;
    for (cpu_value, gpu_value) in cpu.iter().zip(gpu.iter()) {
        let error = *gpu_value as f64 - *cpu_value as f64;
        sq_err += error * error;
        sq_cpu += (*cpu_value as f64).powi(2);
        max_abs = max_abs.max(error.abs());
    }
    eprintln!(
        "trace parity: rel_l2={:.6} max_abs={:.6} _={:?}",
        (sq_err / sq_cpu).sqrt(),
        max_abs,
        ScalarType::F32
    );
    assert!((sq_err / sq_cpu).sqrt() < 0.15);
}
