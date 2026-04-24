use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use clap::Parser;
use gpu_hal::{GpuBuffer, ScalarType};
use half::{bf16, f16};
use model_store::{bake_dir_int8, BakedStore};
use runner::oracle as oracle_mod;
use runner::validate;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Diagnose Llama 3.1 INT8 down_proj parity against the live BnB oracle")]
struct Cli {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "")]
    prompt: String,
    #[arg(long, default_value_t = 1)]
    layer: usize,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    #[arg(long, default_value = "cuda:0")]
    oracle_device: String,
    #[arg(long, default_value = "bf16")]
    oracle_dtype: String,
}

fn decode_bf16_b64(b64: &str) -> Result<Vec<f32>> {
    let bytes = B64.decode(b64).context("base64 decode BF16 tensor")?;
    if bytes.len() % 2 != 0 {
        bail!("BF16 payload byte_len {} is not even", bytes.len());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect())
}

fn decode_i8_b64(b64: &str) -> Result<Vec<i8>> {
    let bytes = B64.decode(b64).context("base64 decode INT8 tensor")?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("F32 payload byte_len {} is not divisible by 4", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn post_acc_round(x: f32) -> f32 {
    bf16::from_f32(f16::from_f32(x).to_f32()).to_f32()
}

fn reference_from_exported_quant(
    ca: &[i8],
    sca: &[f32],
    cb: &[i8],
    scb: &[f32],
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if ca.len() != rows * in_dim {
        bail!("ca len {} != rows*in_dim {}", ca.len(), rows * in_dim);
    }
    if sca.len() != rows {
        bail!("sca len {} != rows {}", sca.len(), rows);
    }
    if cb.len() != out_dim * in_dim {
        bail!("cb len {} != out_dim*in_dim {}", cb.len(), out_dim * in_dim);
    }
    if scb.len() != out_dim {
        bail!("scb len {} != out_dim {}", scb.len(), out_dim);
    }
    let inv_127_sq = 1.0f32 / (127.0 * 127.0);
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0i32;
            let cb_row = &cb[o * in_dim..(o + 1) * in_dim];
            let ca_row = &ca[r * in_dim..(r + 1) * in_dim];
            for c in 0..in_dim {
                acc += (ca_row[c] as i32) * (cb_row[c] as i32);
            }
            out[r * out_dim + o] = post_acc_round((acc as f32) * sca[r] * scb[o] * inv_127_sq);
        }
    }
    Ok(out)
}

fn host_bf16_addmm(
    base: &[f32],
    suba: &[f32],
    suba_shape: &[usize],
    subb_t: &[f32],
    subb_t_shape: &[usize],
) -> Result<Vec<f32>> {
    if suba_shape.len() != 2 || subb_t_shape.len() != 2 {
        bail!(
            "host_bf16_addmm expects rank-2 shapes, got {:?} and {:?}",
            suba_shape,
            subb_t_shape
        );
    }
    let rows = suba_shape[0];
    let k = suba_shape[1];
    let out_dim = subb_t_shape[0];
    if subb_t_shape[1] != k {
        bail!("host_bf16_addmm k mismatch: {} vs {}", subb_t_shape[1], k);
    }
    if base.len() != rows * out_dim {
        bail!(
            "host_bf16_addmm base len {} != {}",
            base.len(),
            rows * out_dim
        );
    }
    let mut out = base.to_vec();
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += suba[r * k + kk] * subb_t[o * k + kk];
            }
            out[r * out_dim + o] = bf16::from_f32(out[r * out_dim + o] + acc).to_f32();
        }
    }
    Ok(out)
}

fn top_delta(label: &str, got: &[f32], expected: &[f32]) {
    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - e).abs();
        if d > worst {
            worst = d;
            worst_idx = idx;
        }
    }
    eprintln!(
        "[diag] {label} max_abs_delta={worst:.6} idx={worst_idx} got={:.6} expected={:.6}",
        got[worst_idx], expected[worst_idx],
    );
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to zero ids");
    }

    let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("could not derive repo root from CARGO_MANIFEST_DIR"))?
        .join("oracle/run_oracle.py");
    let model_id = cli.model_dir.to_string_lossy().into_owned();
    let oracle = oracle_mod::run_oracle(
        &oracle_script,
        &model_id,
        &prompt_ids,
        1,
        &cli.oracle_dtype,
        &cli.oracle_device,
        true,
        true,
        None,
        Some(cli.layer),
    )?;

    if let Some(ref oracle_ids) = oracle.prompt_token_ids {
        if oracle_ids != &prompt_ids {
            bail!("tokenizer mismatch: rust={prompt_ids:?} oracle={oracle_ids:?}");
        }
    }

    let swiglu = decode_bf16_b64(
        oracle
            .traced_mlp_swiglu
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_swiglu"))?,
    )?;
    let oracle_down = decode_bf16_b64(
        oracle
            .traced_mlp_down
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_down"))?,
    )?;
    let ca = decode_i8_b64(
        oracle
            .traced_mlp_down_ca
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_ca"))?,
    )?;
    let ca_dense = decode_i8_b64(
        oracle
            .traced_mlp_down_ca_dense
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_ca_dense"))?,
    )?;
    let ca_shape = oracle
        .traced_mlp_down_ca_shape
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_ca_shape"))?;
    let ca_dense_shape = oracle
        .traced_mlp_down_ca_dense_shape
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_ca_dense_shape"))?;
    let sca = oracle
        .traced_mlp_down_sca
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_sca"))?;
    let sca_dense = oracle
        .traced_mlp_down_sca_dense
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_sca_dense"))?;
    let outlier_cols = oracle
        .traced_mlp_down_outlier_cols
        .clone()
        .unwrap_or_default();
    let threshold = oracle.traced_mlp_down_outlier_threshold.unwrap_or(0.0);
    let suba = decode_bf16_b64(
        oracle
            .traced_mlp_down_suba
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_suba"))?,
    )?;
    let suba_shape = oracle
        .traced_mlp_down_suba_shape
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_suba_shape"))?;
    let subb_t = decode_bf16_b64(
        oracle
            .traced_mlp_down_subb_t
            .as_deref()
            .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_subb_t"))?,
    )?;
    let subb_t_shape = oracle
        .traced_mlp_down_subb_t_shape
        .clone()
        .ok_or_else(|| anyhow!("oracle missing traced_mlp_down_subb_t_shape"))?;

    if ca_shape.len() < 2 {
        bail!(
            "expected traced_mlp_down_ca_shape rank >= 2, got {:?}",
            ca_shape
        );
    }
    if ca_dense_shape != ca_shape {
        bail!(
            "dense CA shape {:?} != thresholded CA shape {:?}",
            ca_dense_shape,
            ca_shape
        );
    }
    let rows = ca_shape[..ca_shape.len() - 1].iter().product();
    let in_dim = ca_shape[ca_shape.len() - 1];
    if swiglu.len() != rows * in_dim {
        bail!(
            "swiglu len {} != rows*in_dim {}",
            swiglu.len(),
            rows * in_dim
        );
    }

    let bake_dir = bake_dir_int8(&cli.model_dir);
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open INT8 bake {}", bake_dir.display()))?;
    let weight_name = format!("model.layers.{}.mlp.down_proj.weight", cli.layer);
    let scb_name = format!("model.layers.{}.mlp.down_proj.SCB", cli.layer);
    let rhs_shape = store
        .shape(&weight_name)
        .ok_or_else(|| anyhow!("missing baked tensor {weight_name}"))?
        .to_vec();
    if rhs_shape.len() != 2 {
        bail!("expected {weight_name} rank 2, got {:?}", rhs_shape);
    }
    let out_dim = rhs_shape[0];
    if rhs_shape[1] != in_dim {
        bail!(
            "weight inner dim {} != traced input dim {}",
            rhs_shape[1],
            in_dim
        );
    }
    if oracle_down.len() != rows * out_dim {
        bail!(
            "oracle down len {} != rows*out_dim {}",
            oracle_down.len(),
            rows * out_dim
        );
    }

    let rhs_i8: Vec<i8> = store
        .raw_bytes(&weight_name)
        .ok_or_else(|| anyhow!("missing raw bytes for {weight_name}"))?
        .iter()
        .copied()
        .map(|b| b as i8)
        .collect();
    let scb = f32_bytes_to_vec(
        store
            .raw_bytes(&scb_name)
            .ok_or_else(|| anyhow!("missing raw bytes for {scb_name}"))?,
    )?;

    let ref_from_ca =
        reference_from_exported_quant(&ca, &sca, &rhs_i8, &scb, rows, out_dim, in_dim)?;
    let ref_from_ca_dense =
        reference_from_exported_quant(&ca_dense, &sca_dense, &rhs_i8, &scb, rows, out_dim, in_dim)?;

    gpu_hal::set_device(cli.device_ordinal)
        .map_err(|e| anyhow!("set_device({}): {e}", cli.device_ordinal))?;
    let lhs_gpu = GpuBuffer::from_host_bytes(
        cli.device_ordinal,
        ScalarType::BF16,
        &[rows, in_dim],
        &f32_to_bf16_bytes(&swiglu),
    )?;
    let rhs_gpu = store.load_to_gpu(&weight_name, cli.device_ordinal)?;
    let scb_gpu = store.load_to_gpu(&scb_name, cli.device_ordinal)?;
    let mut out_gpu = GpuBuffer::zeros(cli.device_ordinal, ScalarType::BF16, &[rows, out_dim])?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_int8(
        cli.device_ordinal,
        1,
        rows,
        out_dim,
        in_dim,
        &lhs_gpu,
        &rhs_gpu,
        &scb_gpu,
        &mut out_gpu,
    )?;
    let kernel_out = bf16_bytes_to_f32(&out_gpu.to_host_bytes()?);
    let mut corrected_out = kernel_out.clone();
    let mut mixed_from_thresholded = ref_from_ca.clone();
    let mut host_thresholded_plus_corr = ref_from_ca.clone();
    let mut zeroed_kernel_out = Vec::new();

    if !outlier_cols.is_empty() {
        let mut zeroed = swiglu.clone();
        for &col in &outlier_cols {
            let col = col as usize;
            for row in 0..rows {
                zeroed[row * in_dim + col] = 0.0;
            }
        }
        let zeroed_gpu = GpuBuffer::from_host_bytes(
            cli.device_ordinal,
            ScalarType::BF16,
            &[rows, in_dim],
            &f32_to_bf16_bytes(&zeroed),
        )?;
        let mut zeroed_out_gpu =
            GpuBuffer::zeros(cli.device_ordinal, ScalarType::BF16, &[rows, out_dim])?;
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int8(
            cli.device_ordinal,
            1,
            rows,
            out_dim,
            in_dim,
            &zeroed_gpu,
            &rhs_gpu,
            &scb_gpu,
            &mut zeroed_out_gpu,
        )?;
        zeroed_kernel_out = bf16_bytes_to_f32(&zeroed_out_gpu.to_host_bytes()?);
    }

    if suba_shape.len() != 2 || subb_t_shape.len() != 2 {
        bail!(
            "expected subA/subB_t rank 2, got {:?} and {:?}",
            suba_shape,
            subb_t_shape
        );
    }
    if suba_shape[0] != rows {
        bail!("subA rows {} != {}", suba_shape[0], rows);
    }
    if subb_t_shape[0] != out_dim {
        bail!("subB_t rows {} != out_dim {}", subb_t_shape[0], out_dim);
    }
    if suba_shape[1] != subb_t_shape[1] {
        bail!(
            "subA cols {} != subB_t cols {}",
            suba_shape[1],
            subb_t_shape[1]
        );
    }
    if suba.len() != suba_shape[0] * suba_shape[1] {
        bail!(
            "subA len {} does not match shape {:?}",
            suba.len(),
            suba_shape
        );
    }
    if subb_t.len() != subb_t_shape[0] * subb_t_shape[1] {
        bail!(
            "subB_t len {} does not match shape {:?}",
            subb_t.len(),
            subb_t_shape
        );
    }
    if suba_shape[1] > 0 {
        let suba_gpu = GpuBuffer::from_host_bytes(
            cli.device_ordinal,
            ScalarType::BF16,
            &[rows, suba_shape[1]],
            &f32_to_bf16_bytes(&suba),
        )?;
        let subb_t_gpu = GpuBuffer::from_host_bytes(
            cli.device_ordinal,
            ScalarType::BF16,
            &[out_dim, subb_t_shape[1]],
            &f32_to_bf16_bytes(&subb_t),
        )?;
        let mut corr_gpu =
            GpuBuffer::zeros(cli.device_ordinal, ScalarType::BF16, &[rows, out_dim])?;
        kernel_ffi::prefill_ffi::matmul_rhs_transposed(
            cli.device_ordinal,
            ScalarType::BF16,
            1,
            rows,
            out_dim,
            suba_shape[1],
            &suba_gpu,
            &subb_t_gpu,
            &mut corr_gpu,
        )?;
        kernel_ffi::prefill_ffi::element_add_inplace(
            cli.device_ordinal,
            ScalarType::BF16,
            rows * out_dim,
            &mut out_gpu,
            &corr_gpu,
        )?;
        corrected_out = bf16_bytes_to_f32(&out_gpu.to_host_bytes()?);

        let mut thresholded_gpu = GpuBuffer::from_host_bytes(
            cli.device_ordinal,
            ScalarType::BF16,
            &[rows, out_dim],
            &f32_to_bf16_bytes(&ref_from_ca),
        )?;
        kernel_ffi::prefill_ffi::element_add_inplace(
            cli.device_ordinal,
            ScalarType::BF16,
            rows * out_dim,
            &mut thresholded_gpu,
            &corr_gpu,
        )?;
        mixed_from_thresholded = bf16_bytes_to_f32(&thresholded_gpu.to_host_bytes()?);
        host_thresholded_plus_corr =
            host_bf16_addmm(&ref_from_ca, &suba, &suba_shape, &subb_t, &subb_t_shape)?;
    }

    eprintln!(
        "[diag] prompt_tokens={} layer={} rows={} in_dim={} out_dim={} outliers={} threshold={:.3}",
        prompt_ids.len(),
        cli.layer,
        rows,
        in_dim,
        out_dim,
        outlier_cols.len(),
        threshold,
    );
    eprintln!(
        "[diag] row_stats thresholded={:.6} dense={:.6}",
        sca[0], sca_dense[0],
    );
    if !outlier_cols.is_empty() {
        let preview: Vec<u32> = outlier_cols.iter().copied().take(8).collect();
        eprintln!("[diag] outlier_cols(first8)={preview:?}");
    }
    top_delta("kernel_vs_oracle", &kernel_out, &oracle_down);
    top_delta("corrected_vs_oracle", &corrected_out, &oracle_down);
    top_delta(
        "thresholded_plus_corr_vs_oracle",
        &mixed_from_thresholded,
        &oracle_down,
    );
    top_delta(
        "host_thresholded_plus_corr_vs_oracle",
        &host_thresholded_plus_corr,
        &oracle_down,
    );
    top_delta("exported_ca_ref_vs_oracle", &ref_from_ca, &oracle_down);
    top_delta("dense_ca_ref_vs_oracle", &ref_from_ca_dense, &oracle_down);
    if !zeroed_kernel_out.is_empty() {
        top_delta(
            "zeroed_kernel_vs_thresholded_ref",
            &zeroed_kernel_out,
            &ref_from_ca,
        );
        top_delta("zeroed_kernel_vs_oracle", &zeroed_kernel_out, &oracle_down);
    }
    top_delta("kernel_vs_exported_ca_ref", &kernel_out, &ref_from_ca);
    top_delta("kernel_vs_dense_ca_ref", &kernel_out, &ref_from_ca_dense);
    top_delta(
        "corrected_vs_dense_ca_ref",
        &corrected_out,
        &ref_from_ca_dense,
    );
    eprintln!(
        "[diag] summary kernel_vs_oracle={:.6} corrected_vs_oracle={:.6} thresholded_plus_corr_vs_oracle={:.6} host_thresholded_plus_corr_vs_oracle={:.6} ca_ref_vs_oracle={:.6} dense_ca_ref_vs_oracle={:.6} kernel_vs_ca_ref={:.6} kernel_vs_dense_ca_ref={:.6}",
        validate::max_abs_delta(&kernel_out, &oracle_down),
        validate::max_abs_delta(&corrected_out, &oracle_down),
        validate::max_abs_delta(&mixed_from_thresholded, &oracle_down),
        validate::max_abs_delta(&host_thresholded_plus_corr, &oracle_down),
        validate::max_abs_delta(&ref_from_ca, &oracle_down),
        validate::max_abs_delta(&ref_from_ca_dense, &oracle_down),
        validate::max_abs_delta(&kernel_out, &ref_from_ca),
        validate::max_abs_delta(&kernel_out, &ref_from_ca_dense),
    );

    Ok(())
}
